#!/usr/bin/env python3
"""Rebuild the CRC historical benchmark (2015-2021) on GBD 2021 data.

The v1/v2 benchmark used GBD 2023 downloads (camoufox exports); this v3 run
recomputes the same tables from the harmonized GBD 2021 inputs
(11_all_disease_extension Phase 1) and the Phase 3 historical PAF, which is
GBD-version independent (verified identical to v1 study_paf within 5e-12).

Outputs use the v1 table schemas so downstream scripts (S8 figure, Figure 3
donut prep, Figure 4 velocity prep) can swap paths without code changes.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "10_gbd_comparison" / "crc_historical_benchmark_2015_2021_v3_gbd2021"
TABLES = ROOT / "tables"
QA = ROOT / "qa"
SCRIPTS = ROOT / "scripts"

HARMONIZED = (
    PROJECT / "11_all_disease_extension" / "01_harmonize" / "harmonized_gbd_v1_20261005" / "data"
)
PAF_PATH = (
    PROJECT / "11_all_disease_extension" / "03_paf" / "paf_all_causes_v1_20261005"
    / "data" / "paf_historical_2015_2024_all_causes_v1.parquet"
)

CAUSE_ID = 441
YEARS = tuple(range(2015, 2022))
AGE_IDS = {
    "25-29": 10, "30-34": 11, "35-39": 12, "40-44": 13, "45-49": 14,
    "50-54": 15, "55-59": 16, "60-64": 17, "65-69": 18, "70-74": 19,
    "75-79": 20, "80-84": 30, "85-89": 31, "90-94": 32, "95+": 235,
}
SEX_IDS = {"Male": 1, "Female": 2}
MEASURE_IDS = {"Deaths": 1, "DALYs": 2}
GBD_AGE_TO_ANCHOR = {"85-89": "85+", "90-94": "85+", "95+": "85+"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def metrics_block(source: pd.DataFrame, measure: str) -> dict:
    frame = source[source["measure"].eq(measure)].copy()
    x = frame["gbd_official_attributable"].to_numpy(float)
    y = frame["study_derived_attributable"].to_numpy(float)
    positive = frame["positive_pair_for_log_plot"].to_numpy(bool)
    xp, yp = x[positive], y[positive]
    lx, ly = np.log10(xp), np.log10(yp)
    slope_log, intercept_log = np.polyfit(lx, ly, 1)
    fitted_log = intercept_log + slope_log * lx
    slope_raw, intercept_raw = np.polyfit(x, y, 1)
    fitted_raw = intercept_raw + slope_raw * x
    relative = np.where(x != 0, 100.0 * (y - x) / x, np.nan)
    return {
        "measure": measure,
        "country_year_points": int(len(frame)),
        "positive_country_year_points_log_plot": int(positive.sum()),
        "excluded_nonpositive_points_log_plot": int((~positive).sum()),
        "pearson_r_log10": float(np.corrcoef(lx, ly)[0, 1]),
        "R2_log10": float(1.0 - np.sum((ly - fitted_log) ** 2) / np.sum((ly - ly.mean()) ** 2)),
        "slope_log10": float(slope_log),
        "intercept_log10": float(intercept_log),
        "pearson_r_raw": float(np.corrcoef(x, y)[0, 1]),
        "R2_raw": float(1.0 - np.sum((y - fitted_raw) ** 2) / np.sum((y - y.mean()) ** 2)),
        "slope_raw": float(slope_raw),
        "intercept_raw": float(intercept_raw),
        "MAE_number": float(np.mean(np.abs(y - x))),
        "median_relative_error_pct": float(np.nanmedian(relative)),
        "median_absolute_relative_error_pct": float(np.nanmedian(np.abs(relative))),
        "sum_official_attributable": float(x.sum()),
        "sum_study_derived_attributable": float(y.sum()),
        "aggregate_study_to_official_ratio": float(y.sum() / x.sum()),
    }


def main() -> None:
    total = pd.read_parquet(HARMONIZED / "gbd_total_burden_harmonized_v1.parquet")
    total = total[(total["cause_id"] == CAUSE_ID) & total["year"].isin(YEARS) & (total["metric"] == "Number")].copy()
    official = pd.read_parquet(HARMONIZED / "gbd_high_bmi_attributable_harmonized_v1.parquet")
    official = official[(official["cause_id"] == CAUSE_ID) & official["year"].isin(YEARS) & (official["metric"] == "Number")].copy()

    paf = pd.read_parquet(PAF_PATH)
    paf = paf[(paf["cause_id"] == CAUSE_ID) & paf["year"].isin(YEARS)].copy()

    total["anchor_age_group"] = total["age_group"].replace(GBD_AGE_TO_ANCHOR)
    official["anchor_age_group"] = official["age_group"].replace(GBD_AGE_TO_ANCHOR)

    merged = total.merge(
        paf[["iso3", "year", "sex", "anchor_age_group", "paf"]],
        on=["iso3", "year", "sex", "anchor_age_group"],
        how="left",
        validate="many_to_one",
    )
    # Five small GBD locations have no NCD-RisC anchor (GUM/MCO/MNP/SMR/VIR);
    # v1 excluded them too (1393 country-year points = 199 locations x 7 years).
    excluded = sorted(merged.loc[merged["paf"].isna(), "iso3"].unique().tolist())
    merged = merged[merged["paf"].notna()].copy()
    merged = merged.merge(
        official[["location_id", "year", "sex", "age_group", "measure", "value"]].rename(
            columns={"value": "gbd_official_attributable"}
        ),
        on=["location_id", "year", "sex", "age_group", "measure"],
        how="left",
        validate="one_to_one",
    )
    if merged["gbd_official_attributable"].isna().any():
        raise RuntimeError("Unmatched official attributable rows in CRC v3 benchmark")

    merged["study_derived_attributable"] = merged["paf"] * merged["value"]
    merged["signed_error"] = merged["study_derived_attributable"] - merged["gbd_official_attributable"]
    merged["absolute_error"] = merged["signed_error"].abs()

    age_sex = pd.DataFrame({
        "location_id": merged["location_id"].astype(int),
        "year": merged["year"].astype(int),
        "age_id": merged["age_group"].map(AGE_IDS).astype(int),
        "sex_id": merged["sex"].map(SEX_IDS).astype(int),
        "measure_id": merged["measure"].map(MEASURE_IDS).astype(int),
        "iso3_gbd": merged["iso3"],
        "crosswalk_location_name_gbd": merged["crosswalk_location_name"],
        "age_group_gbd": merged["age_group"],
        "sex_gbd": merged["sex"],
        "measure": merged["measure"],
        "gbd_total_burden": merged["value"],
        "iso3": merged["iso3"],
        "crosswalk_location_name": merged["crosswalk_location_name"],
        "age_group": merged["age_group"],
        "sex": merged["sex"],
        "probability_source": "NCD_RisC_anchor",
        "historical_high_age_mapping": np.where(
            merged["age_group"].isin(GBD_AGE_TO_ANCHOR), "shared_historical_85plus", "exact_five_year_age"
        ),
        "weighted_rr": np.nan,
        "study_paf": merged["paf"],
        "gbd_official_attributable": merged["gbd_official_attributable"],
        "study_derived_attributable": merged["study_derived_attributable"],
        "signed_error": merged["signed_error"],
        "absolute_error": merged["absolute_error"],
    })
    age_sex_path = TABLES / "crc_historical_benchmark_age_sex_2015_2021_v3.csv.gz"
    age_sex.to_csv(age_sex_path, index=False, compression="gzip")

    geo = ["location_id", "iso3", "crosswalk_location_name"]
    country = merged.groupby(geo + ["year", "measure"], as_index=False, observed=True)[
        ["value", "gbd_official_attributable", "study_derived_attributable"]
    ].sum().rename(columns={"value": "gbd_total_burden"})
    country["measure_id"] = country["measure"].map(MEASURE_IDS).astype(int)
    country["signed_error"] = country["study_derived_attributable"] - country["gbd_official_attributable"]
    country["absolute_error"] = country["signed_error"].abs()
    country["relative_error_pct"] = np.where(
        country["gbd_official_attributable"] != 0,
        100.0 * country["signed_error"] / country["gbd_official_attributable"],
        np.nan,
    )
    country["positive_pair_for_log_plot"] = (
        (country["gbd_official_attributable"] > 0) & (country["study_derived_attributable"] > 0)
    )
    country = country[
        ["location_id", "iso3", "crosswalk_location_name", "year", "measure_id", "measure",
         "gbd_total_burden", "gbd_official_attributable", "study_derived_attributable",
         "signed_error", "absolute_error", "relative_error_pct", "positive_pair_for_log_plot"]
    ].sort_values(["location_id", "year", "measure_id"])
    country_path = TABLES / "crc_historical_benchmark_country_year_source_v3.csv"
    country.to_csv(country_path, index=False)

    metrics = pd.DataFrame([metrics_block(country, measure) for measure in ("Deaths", "DALYs")])
    metrics_path = QA / "benchmark_metrics_v3.csv"
    metrics.to_csv(metrics_path, index=False)

    summary = {
        "run": "crc_historical_benchmark_2015_2021_v3_gbd2021",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data_version": "GBD 2021 (harmonized_gbd_v1_20261005); supersedes v1/v2 built on GBD 2023",
        "excluded_locations_no_ncd_risc_anchor": excluded,
        "rows_age_sex": int(len(age_sex)),
        "rows_country_year": int(len(country)),
        "metrics": metrics.to_dict(orient="records"),
    }
    write_json(QA / "benchmark_QA_v3.json", summary)

    (SCRIPTS / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    outputs = [age_sex_path, country_path, metrics_path, QA / "benchmark_QA_v3.json", SCRIPTS / Path(__file__).name]
    pd.DataFrame(
        {"path": [str(p) for p in outputs], "sha256": [sha256(p) for p in outputs]}
    ).to_csv(ROOT / "crc_historical_benchmark_v3_formal_manifest.csv", index=False, encoding="utf-8-sig")
    (ROOT / "COMPLETE.flag").write_text(
        f"v3 benchmark completed {summary['completed_at_utc']}\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
