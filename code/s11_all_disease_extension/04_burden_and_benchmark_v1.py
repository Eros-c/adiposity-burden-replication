"""Attributable burden and GBD benchmark for 33 adiposity-associated causes."""

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
ROOT = PROJECT / "11_all_disease_extension" / "04_burden" / "burden_benchmark_v1_20261005"
DATA = ROOT / "data"
TABLES = ROOT / "tables"
QA = ROOT / "qa"
REPORTS = ROOT / "reports"
SCRIPTS = ROOT / "scripts"

HARMONIZED = (
    PROJECT / "11_all_disease_extension" / "01_harmonize" / "harmonized_gbd_v1_20261005" / "data"
)
PAF_DIR = PROJECT / "11_all_disease_extension" / "03_paf" / "paf_all_causes_v1_20261005" / "data"
CRC_FROZEN_OVERALL = (
    PROJECT
    / "09_colorectal_cancer_burden"
    / "crc_attributable_deaths_dalys_2025_2050_four_ssp_v1_20260910"
    / "tables"
    / "CRC_attributable_burden_country_overall_annual_four_SSP_v1.csv"
)
CRC_BENCHMARK_V2 = (
    PROJECT
    / "10_gbd_comparison"
    / "crc_historical_benchmark_2015_2021_v2"
    / "tables"
    / "crc_historical_benchmark_country_year_source_v2.csv"
)
DISEASE_LIST = (
    PROJECT / "11_all_disease_extension" / "01_harmonize" / "harmonized_gbd_v1_20261005"
    / "contracts" / "diseases_v1.csv"
)

SCENARIOS = ("ssp126", "ssp245", "ssp370", "ssp585")
FUTURE_YEARS = tuple(range(2025, 2051))
HISTORICAL_YEARS = tuple(range(2015, 2022))
KEY_YEARS = (2025, 2030, 2035, 2040, 2045, 2050)
SUBTYPE_CAUSES = {418, 419, 420, 495, 496, 845, 846, 847, 848, 943}
FEMALE_ONLY = {429, 435, 465}
GBD_AGE_TO_ANCHOR = {"85-89": "85+", "90-94": "85+", "95+": "85+"}
MEASURE_IDS = {"Deaths": 1, "DALYs": 2}
REGRESSION_RTOL = 1e-6

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def load_foresight() -> pd.DataFrame:
    frame = pd.read_parquet(HARMONIZED / "gbd_foresight_reference_harmonized_v1.parquet")
    return frame[frame["year"].isin(FUTURE_YEARS)].copy()

def load_total() -> pd.DataFrame:
    frame = pd.read_parquet(HARMONIZED / "gbd_total_burden_harmonized_v1.parquet")
    return frame[frame["year"].isin(HISTORICAL_YEARS) & (frame["metric"] == "Number")].copy()

def load_official() -> pd.DataFrame:
    frame = pd.read_parquet(HARMONIZED / "gbd_high_bmi_attributable_harmonized_v1.parquet")
    return frame[frame["year"].isin(HISTORICAL_YEARS) & (frame["metric"] == "Number")].copy()

def future_burden_one_scenario(foresight: pd.DataFrame, scenario: str) -> dict[str, pd.DataFrame]:
    paf = pd.read_parquet(PAF_DIR / f"paf_future_{scenario}_2025_2050_all_causes_v1.parquet")
    merged = foresight.merge(
        paf[["cause_id", "iso3", "year", "sex", "age_group",
             "paf", "paf_rr_lower", "paf_rr_upper"]],
        on=["cause_id", "iso3", "year", "sex", "age_group"],
        how="left",
        validate="many_to_one",
    )
    merged["paf"] = merged["paf"].fillna(0.0)
    merged["paf_rr_lower"] = merged["paf_rr_lower"].fillna(0.0)
    merged["paf_rr_upper"] = merged["paf_rr_upper"].fillna(0.0)
    merged["attributable_burden"] = merged["paf"] * merged["value"]
    merged["attrib_if_paf_rr_lower"] = merged["paf_rr_lower"] * merged["value"]
    merged["attrib_if_paf_rr_upper"] = merged["paf_rr_upper"] * merged["value"]
    merged["attrib_if_burden_lower"] = merged["paf"] * merged["lower"]
    merged["attrib_if_burden_upper"] = merged["paf"] * merged["upper"]

    sum_cols = [
        "value", "lower", "upper", "attributable_burden",
        "attrib_if_paf_rr_lower", "attrib_if_paf_rr_upper",
        "attrib_if_burden_lower", "attrib_if_burden_upper",
    ]
    geo = ["cause_id", "location_id", "iso3", "crosswalk_location_name"]

    country_sex = merged.groupby(geo + ["year", "sex", "measure"], as_index=False, observed=True)[sum_cols].sum()
    country_overall = merged.groupby(geo + ["year", "measure"], as_index=False, observed=True)[sum_cols].sum()
    global_year = merged.groupby(["cause_id", "year", "measure"], as_index=False, observed=True)[sum_cols].sum()
    global_year_sex = merged.groupby(["cause_id", "year", "sex", "measure"], as_index=False, observed=True)[sum_cols].sum()

    detail_2050 = merged[merged["year"] == 2050]
    age_sex_2050 = detail_2050.groupby(
        geo + ["sex", "age_group", "measure"], as_index=False, observed=True
    )[sum_cols].sum()

    for frame in (country_sex, country_overall, global_year, global_year_sex, age_sex_2050):
        frame["scenario_code"] = scenario

    return {
        "country_sex": country_sex,
        "country_overall": country_overall,
        "global_year": global_year,
        "global_year_sex": global_year_sex,
        "age_sex_2050": age_sex_2050,
    }

def crc_future_regression(country_overall_all: pd.DataFrame) -> dict:
    frozen = pd.read_csv(CRC_FROZEN_OVERALL, encoding="utf-8-sig")
    mine = country_overall_all[country_overall_all["cause_id"] == 441].copy()
    mine["measure_id"] = mine["measure"].map(MEASURE_IDS)
    pair = frozen.merge(
        mine,
        left_on=["scenario_code", "year", "location_id", "measure_id"],
        right_on=["scenario_code", "year", "location_id", "measure_id"],
        how="inner",
        suffixes=("_frozen", "_mine"),
        validate="one_to_one",
    )
    diff = (pair["attributable_burden_frozen"] - pair["attributable_burden_mine"]).abs()
    denom = pair["attributable_burden_frozen"].abs().clip(lower=1e-9)
    rel = (diff / denom).max()
    return {
        "rows_frozen": int(len(frozen)),
        "rows_compared": int(len(pair)),
        "max_abs_diff": float(diff.max()),
        "max_rel_diff": float(rel),
        "status": "PASS" if len(pair) == len(frozen) and rel <= REGRESSION_RTOL else "FAIL",
    }

def historical_benchmark(total: pd.DataFrame, official: pd.DataFrame) -> dict[str, pd.DataFrame]:
    paf = pd.read_parquet(PAF_DIR / "paf_historical_2015_2024_all_causes_v1.parquet")
    paf = paf[paf["year"].isin(HISTORICAL_YEARS)].copy()

    total = total.copy()
    total["anchor_age_group"] = total["age_group"].replace(GBD_AGE_TO_ANCHOR)
    merged = total.merge(
        paf[["cause_id", "iso3", "year", "sex", "anchor_age_group", "paf"]],
        on=["cause_id", "iso3", "year", "sex", "anchor_age_group"],
        how="left",
        validate="many_to_one",
    )
    merged["paf"] = merged["paf"].fillna(0.0)
    merged["study_attributable"] = merged["paf"] * merged["value"]

    geo = ["cause_id", "location_id", "iso3", "crosswalk_location_name"]
    keys = geo + ["year", "measure"]
    study = merged.groupby(keys, as_index=False, observed=True)[
        ["value", "study_attributable"]
    ].sum().rename(columns={"value": "gbd_total_burden"})

    off = official.groupby(keys, as_index=False, observed=True)[["value"]].sum().rename(
        columns={"value": "gbd_official_attributable"}
    )
    detail = study.merge(off, on=keys, how="outer", validate="one_to_one")
    for col in ("gbd_total_burden", "study_attributable", "gbd_official_attributable"):
        detail[col] = detail[col].fillna(0.0)
    detail["study_paf"] = np.where(
        detail["gbd_total_burden"] > 0, detail["study_attributable"] / detail["gbd_total_burden"], np.nan
    )
    detail["official_paf"] = np.where(
        detail["gbd_total_burden"] > 0, detail["gbd_official_attributable"] / detail["gbd_total_burden"], np.nan
    )
    detail["signed_relative_difference"] = np.where(
        detail["gbd_official_attributable"] > 0,
        (detail["study_attributable"] - detail["gbd_official_attributable"])
        / detail["gbd_official_attributable"],
        np.nan,
    )
    return {"detail": detail}

def benchmark_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cause_id, measure), group in detail.groupby(["cause_id", "measure"], observed=True):
        positive = group[(group["gbd_official_attributable"] > 0) & (group["study_attributable"] > 0)]
        if len(positive) >= 3:
            r = np.corrcoef(
                np.log10(positive["gbd_official_attributable"]),
                np.log10(positive["study_attributable"]),
            )[0, 1]
        else:
            r = np.nan
        rows.append(
            {
                "cause_id": cause_id,
                "measure": measure,
                "country_year_pairs": int(len(group)),
                "positive_pairs": int(len(positive)),
                "pearson_r_log10": r,
                "median_relative_difference": float(group["signed_relative_difference"].median()),
                "study_total": float(group["study_attributable"].sum()),
                "official_total": float(group["gbd_official_attributable"].sum()),
            }
        )
    return pd.DataFrame(rows)

def crc_historical_regression(detail: pd.DataFrame) -> dict:
    v2 = pd.read_csv(CRC_BENCHMARK_V2, encoding="utf-8-sig")
    mine = detail[detail["cause_id"] == 441]
    pair = v2.merge(
        mine,
        left_on=["location_id", "year", "measure"],
        right_on=["location_id", "year", "measure"],
        how="inner",
        suffixes=("_v2", "_mine"),
        validate="one_to_one",
    )
    out = {"rows_v2": int(len(v2)), "rows_compared": int(len(pair))}
    comparisons = (
        ("gbd_official_attributable_v2", "gbd_official_attributable_mine", "gbd_official_attributable"),
        ("study_derived_attributable", "study_attributable", "study_derived_attributable"),
    )
    for col_v2, col_mine, tag in comparisons:
        rel = (
            (pair[col_v2] - pair[col_mine]).abs()
            / pair[col_v2].abs().clip(lower=1e-9)
        )
        out[f"max_rel_diff_{tag}"] = float(rel.max())
        out[f"median_rel_diff_{tag}"] = float(rel.median())
    out["note"] = "soft gate; fresh GBD download round explains small differences"
    return out

def main() -> None:
    diseases = pd.read_csv(DISEASE_LIST, encoding="utf-8-sig")
    diseases = diseases[diseases["cause_id"] != 996][["cause_id", "cause_name_en", "cause_name_cn"]]
    diseases["cause_id"] = diseases["cause_id"].astype(int)
    diseases["analysis_tier"] = np.where(diseases["cause_id"].isin(SUBTYPE_CAUSES), "subtype", "headline")

    outputs: list[Path] = []
    foresight = load_foresight()

    scenario_frames: dict[str, list[pd.DataFrame]] = {
        k: [] for k in ("country_sex", "country_overall", "global_year", "global_year_sex", "age_sex_2050")
    }
    for scenario in SCENARIOS:
        result = future_burden_one_scenario(foresight, scenario)
        for key, frame in result.items():
            scenario_frames[key].append(frame)

    combined = {key: pd.concat(parts, ignore_index=True) for key, parts in scenario_frames.items()}
    for key, frame in combined.items():
        frame["attributable_fraction"] = np.where(
            frame["value"] > 0, frame["attributable_burden"] / frame["value"], np.nan
        )
        frame = frame.rename(columns={"value": "total_burden", "lower": "total_burden_lower", "upper": "total_burden_upper"})
        frame = frame.merge(diseases, on="cause_id", how="left", validate="many_to_one")
        combined[key] = frame

    scenario_names = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
    for key, frame in combined.items():
        frame["scenario"] = frame["scenario_code"].map(scenario_names)
        out_path = DATA / f"attributable_future_{key}_four_ssp_v1.parquet"
        frame.to_parquet(out_path, index=False)
        outputs.append(out_path)

    regression_future = crc_future_regression(
        combined["country_overall"].rename(
            columns={"total_burden": "value"}
        ).assign(value=lambda d: d["value"])
    )

    # Key-year global summary (mirrors the CRC key-year table).
    key = combined["global_year"]
    key = key[key["year"].isin(KEY_YEARS)].copy()
    key["measure_id"] = key["measure"].map(MEASURE_IDS)
    key_path = TABLES / "attributable_future_global_key_year_summary_v1.csv"
    key.sort_values(["cause_id", "scenario_code", "measure_id", "year"]).to_csv(
        key_path, index=False, encoding="utf-8-sig"
    )
    outputs.append(key_path)

    # Top-10 countries per cause in 2050 (SSP2-4.5, deaths).
    co = combined["country_overall"]
    top = (
        co[(co["year"] == 2050) & (co["scenario_code"] == "ssp245") & (co["measure"] == "Deaths")]
        .sort_values("attributable_burden", ascending=False)
        .groupby("cause_id", observed=True)
        .head(10)
    )
    top_path = TABLES / "attributable_future_top10_countries_2050_ssp245_deaths_v1.csv"
    top.to_csv(top_path, index=False, encoding="utf-8-sig")
    outputs.append(top_path)

    # Historical benchmark.
    total = load_total()
    official = load_official()
    benchmark = historical_benchmark(total, official)["detail"]
    benchmark = benchmark.merge(diseases, on="cause_id", how="left", validate="many_to_one")
    detail_path = DATA / "benchmark_historical_country_year_detail_v1.parquet"
    benchmark.to_parquet(detail_path, index=False)
    outputs.append(detail_path)

    metrics = benchmark_metrics(benchmark).merge(diseases, on="cause_id", how="left", validate="many_to_one")
    metrics_path = TABLES / "benchmark_metrics_per_cause_v1.csv"
    metrics.sort_values(["cause_id", "measure"]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    outputs.append(metrics_path)

    regression_historical = crc_historical_regression(benchmark)

    total_cont = total[["cause_id", "location_id", "iso3", "crosswalk_location_name", "year", "age_group", "sex", "measure", "value", "lower", "upper"]]
    fore_cont = foresight[["cause_id", "location_id", "iso3", "crosswalk_location_name", "year", "age_group", "sex", "measure", "value", "lower", "upper"]]
    continuous = pd.concat([total_cont, fore_cont], ignore_index=True)
    cont_path = DATA / "total_burden_continuous_2015_2050_reference_v1.parquet"
    continuous.to_parquet(cont_path, index=False)
    outputs.append(cont_path)

    problems = []
    if regression_future["status"] != "PASS":
        problems.append(f"future CRC regression failed: {regression_future}")
    if (combined["country_overall"]["attributable_burden"] < -1e-6).any():
        problems.append("negative attributable beyond float noise (check premenopausal breast)")

    status = "PASS" if not problems else "FAIL"
    summary = {
        "run": "burden_benchmark_v1_20261005",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "causes": int(diseases["cause_id"].nunique()),
        "future_rows_country_overall": int(len(combined["country_overall"])),
        "benchmark_country_year_rows": int(len(benchmark)),
        "continuous_total_burden_rows": int(len(continuous)),
        "crc_future_regression": regression_future,
        "crc_historical_regression_soft": regression_historical,
        "problems": problems,
        "status": status,
    }
    write_json(QA / "burden_benchmark_QA_v1.json", summary)

    lines = [
        "# Attributable burden and benchmark report (33 causes)",
        "",
        "## Future CRC regression (frozen CRC burden table)",
        "",
        f"- rows compared: {regression_future['rows_compared']} / {regression_future['rows_frozen']}",
        f"- max relative difference: {regression_future['max_rel_diff']:.3g}",
        f"- status: {regression_future['status']}",
        "",
        "## Historical CRC comparison with benchmark v2 (soft)",
        "",
        f"- rows compared: {regression_historical['rows_compared']} / {regression_historical['rows_v2']}",
        f"- official: median rel diff {regression_historical['median_rel_diff_gbd_official_attributable']:.4%}, "
        f"max {regression_historical['max_rel_diff_gbd_official_attributable']:.4%}",
        f"- study-derived: median rel diff {regression_historical['median_rel_diff_study_derived_attributable']:.4%}, "
        f"max {regression_historical['max_rel_diff_study_derived_attributable']:.4%}",
        f"- note: {regression_historical['note']}",
        "",
        "## Benchmark metrics (headline causes)",
        "",
        "| cause_id | cause | measure | r(log10) | median rel diff | n pairs |",
        "|---|---|---|---|---|---|",
    ]
    show = metrics[metrics["analysis_tier"] == "headline"].sort_values(["measure", "cause_id"], ascending=[False, True])
    for row in show.itertuples(index=False):
        lines.append(
            f"| {row.cause_id} | {row.cause_name_en} | {row.measure} "
            f"| {row.pearson_r_log10:.4f} | {row.median_relative_difference:.1%} | {row.positive_pairs} |"
        )
    lines += ["", f"QA status: {status}" + ("" if not problems else " — " + "; ".join(problems))]
    (REPORTS / "burden_benchmark_report_v1.md").write_text("\n".join(lines), encoding="utf-8")

    (SCRIPTS / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    outputs += [
        QA / "burden_benchmark_QA_v1.json",
        REPORTS / "burden_benchmark_report_v1.md",
        SCRIPTS / Path(__file__).name,
    ]
    pd.DataFrame(
        {"path": [str(p) for p in outputs], "sha256": [sha256(p) for p in outputs]}
    ).to_csv(ROOT / "burden_benchmark_formal_manifest_v1.csv", index=False, encoding="utf-8-sig")

    if status != "PASS":
        raise RuntimeError("Burden QA failed: " + "; ".join(problems))
    (ROOT / "COMPLETE.flag").write_text(
        f"burden_benchmark_v1_20261005 completed {summary['completed_at_utc']}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
