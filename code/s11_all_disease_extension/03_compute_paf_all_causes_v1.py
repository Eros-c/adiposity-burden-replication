#!/usr/bin/env python3
"""Compute high-BMI population attributable fractions (PAF) for 33 causes.

PAF = (sum_k p_k * RR_k - 1) / (sum_k p_k * RR_k) over the seven BMI categories,
with RR = 1 at or below the cause-specific reference BMI and log-linear growth
per +5 kg/m2 above it (see Phase-2 crosswalk).

Inputs:
  - Phase-2 RR crosswalk (cause x sex x age x BMI category)
  - Historical national anchors 2015-2024 (NCD-RisC anchored compositions)
  - Future national anchors 2025-2050 for four SSPs (Phase-6 anchor cache)

Outputs:
  - paf_historical_2015_2024_all_causes_v1.parquet  (age grain: 25-29 .. 85+)
  - paf_future_<scenario>_2025_2050_all_causes_v1.parquet  (age grain: 25-29 .. 95+)

Regression gate: the generalized engine must reproduce the frozen CRC national
PAF (08_colorectal_cancer_paf) for every scenario/year within float tolerance.
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
ROOT = PROJECT / "11_all_disease_extension" / "03_paf" / "paf_all_causes_v1_20261005"
DATA = ROOT / "data"
QA = ROOT / "qa"
REPORTS = ROOT / "reports"
SCRIPTS = ROOT / "scripts"

RR_EXPANDED = (
    PROJECT
    / "11_all_disease_extension"
    / "02_rr_crosswalk"
    / "rr_crosswalk_v1_20261005"
    / "data"
    / "RR_crosswalk_by_bmi_category_v1.csv"
)
HISTORICAL_ANCHOR = Path(_cfg.ROOT / "results" / "降尺度" / "01_parameters" / "ncd_risc_2015_2024_bmi7_age_sex.csv")
FUTURE_ANCHOR_ROOT = (
    PROJECT
    / "08_colorectal_cancer_paf"
    / "full_annual_four_ssp_wcrf_primary_v1_20260910"
    / "input_cache"
    / "national_anchors"
)
CRC_FROZEN_ROOT = (
    PROJECT
    / "08_colorectal_cancer_paf"
    / "full_annual_four_ssp_wcrf_primary_v1_20260910"
    / "data"
    / "national"
)

SCENARIOS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
FUTURE_YEARS = tuple(range(2025, 2051))
HISTORICAL_YEARS = tuple(range(2015, 2025))

# Anchor age "85+" is mapped to crosswalk age "85-89" for the RR lookup; all
# conditional RRs (498 bands, 429 menopause) are constant across 85-89/90-94/95+.
ANCHOR_85PLUS_LOOKUP = "85-89"

CATEGORY_HISTORICAL = {
    "bmi_lt18_5": "bmi_lt_18_5",
    "bmi_18_5_20": "bmi_18_5_20",
    "bmi_20_25": "bmi_20_25",
    "bmi_25_30": "bmi_25_30",
    "bmi_30_35": "bmi_30_35",
    "bmi_35_40": "bmi_35_40",
    "bmi_ge40": "bmi_ge_40",
}
CATEGORY_FUTURE = {
    "p_bmi_lt_18_5": "bmi_lt_18_5",
    "p_bmi_18_5_20": "bmi_18_5_20",
    "p_bmi_20_25": "bmi_20_25",
    "p_bmi_25_30": "bmi_25_30",
    "p_bmi_30_35": "bmi_30_35",
    "p_bmi_35_40": "bmi_35_40",
    "p_bmi_ge_40": "bmi_ge_40",
}
FUTURE_AGE_MAP = {f"{a.replace('-', ' to ')}": a for a in (
    "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94",
)} | {"95 plus": "95+"}

REGRESSION_TOLERANCE = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_rr() -> pd.DataFrame:
    rr = pd.read_csv(RR_EXPANDED, encoding="utf-8-sig")
    need = {"cause_id", "sex", "age_group", "bmi_category", "RR", "RR_lower", "RR_upper"}
    if not need.issubset(rr.columns):
        raise RuntimeError("RR crosswalk missing columns")
    return rr


def paf_from_long(long: pd.DataFrame, rr: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """long: one row per (group_keys, bmi_category) with column p."""
    out = []
    for cause_id, rr_cause in rr.groupby("cause_id", sort=True):
        merged = long.merge(
            rr_cause[["sex", "age_group", "bmi_category", "RR", "RR_lower", "RR_upper"]],
            on=["sex", "age_group", "bmi_category"],
            how="inner",
            validate="many_to_one",
        )
        for col, tag in (("RR", "point"), ("RR_lower", "lower"), ("RR_upper", "upper")):
            merged[f"weighted_{tag}"] = merged["p"] * merged[col]
        grouped = merged.groupby(group_keys, as_index=False, observed=True)[
            ["weighted_point", "weighted_lower", "weighted_upper"]
        ].sum()
        grouped["cause_id"] = int(cause_id)
        out.append(grouped)
    frame = pd.concat(out, ignore_index=True)
    frame["paf"] = (frame["weighted_point"] - 1.0) / frame["weighted_point"]
    frame["paf_rr_lower"] = (frame["weighted_lower"] - 1.0) / frame["weighted_lower"]
    frame["paf_rr_upper"] = (frame["weighted_upper"] - 1.0) / frame["weighted_upper"]
    return frame


def historical_long() -> pd.DataFrame:
    anchor = pd.read_csv(HISTORICAL_ANCHOR, encoding="utf-8-sig")
    anchor = anchor[anchor["age_group"] != "20-24"].copy()
    anchor["rr_age_group"] = anchor["age_group"].replace({"85+": ANCHOR_85PLUS_LOOKUP})
    long = anchor.melt(
        id_vars=["iso3", "year", "sex", "age_group", "rr_age_group"],
        value_vars=list(CATEGORY_HISTORICAL),
        var_name="bmi_category",
        value_name="p",
    )
    long["bmi_category"] = long["bmi_category"].map(CATEGORY_HISTORICAL)
    long = long.rename(columns={"rr_age_group": "age_group_lookup"})
    return long


def compute_historical(rr: pd.DataFrame) -> pd.DataFrame:
    long = historical_long()
    long = long.rename(columns={"age_group": "anchor_age_group", "age_group_lookup": "age_group"})
    frame = paf_from_long(
        long, rr, ["iso3", "year", "sex", "anchor_age_group", "age_group"]
    )
    frame = frame.rename(columns={"age_group": "rr_age_group"})
    frame = frame[
        ["cause_id", "iso3", "year", "sex", "anchor_age_group", "rr_age_group",
         "weighted_point", "weighted_lower", "weighted_upper",
         "paf", "paf_rr_lower", "paf_rr_upper"]
    ].sort_values(["cause_id", "iso3", "year", "sex", "anchor_age_group"])
    return frame.reset_index(drop=True)


def future_long(scenario: str) -> pd.DataFrame:
    parts = []
    for year in FUTURE_YEARS:
        path = FUTURE_ANCHOR_ROOT / scenario / f"{year}.parquet"
        part = pd.read_parquet(
            path,
            columns=["iso3", "sex", "age_group_output", "year"] + list(CATEGORY_FUTURE),
        )
        parts.append(part)
    anchor = pd.concat(parts, ignore_index=True)
    anchor["age_group"] = anchor["age_group_output"].map(FUTURE_AGE_MAP)
    if anchor["age_group"].isna().any():
        raise RuntimeError(f"Unmapped future age groups in {scenario}")
    long = anchor.melt(
        id_vars=["iso3", "year", "sex", "age_group"],
        value_vars=list(CATEGORY_FUTURE),
        var_name="bmi_category",
        value_name="p",
    )
    long["bmi_category"] = long["bmi_category"].map(CATEGORY_FUTURE)
    return long


def compute_future(rr: pd.DataFrame, scenario: str) -> pd.DataFrame:
    long = future_long(scenario)
    frame = paf_from_long(long, rr, ["iso3", "year", "sex", "age_group"])
    frame["scenario_code"] = scenario
    frame["scenario"] = SCENARIOS[scenario]
    frame = frame[
        ["cause_id", "scenario_code", "scenario", "iso3", "year", "sex", "age_group",
         "weighted_point", "weighted_lower", "weighted_upper",
         "paf", "paf_rr_lower", "paf_rr_upper"]
    ].sort_values(["cause_id", "iso3", "year", "sex", "age_group"])
    return frame.reset_index(drop=True)


def regression_check_crc(future: pd.DataFrame, scenario: str, year: int) -> dict:
    frozen_path = (
        CRC_FROZEN_ROOT / scenario / str(year)
        / f"CRC_PAF_{scenario}_{year}_national_age_sex_WCRF_primary_v1.parquet"
    )
    frozen = pd.read_parquet(frozen_path)
    mine = future[(future["cause_id"] == 441) & (future["year"] == year)]
    pair = frozen.merge(
        mine[["iso3", "sex", "age_group", "paf", "paf_rr_lower", "paf_rr_upper"]],
        on=["iso3", "sex", "age_group"],
        how="inner",
        suffixes=("_frozen", "_mine"),
        validate="one_to_one",
    )
    result = {
        "scenario": scenario,
        "year": year,
        "rows_compared": int(len(pair)),
        "rows_frozen": int(len(frozen)),
    }
    for col in ("paf", "paf_rr_lower", "paf_rr_upper"):
        result[f"max_abs_diff_{col}"] = float(
            (pair[f"{col}_frozen"] - pair[f"{col}_mine"]).abs().max()
        )
    result["status"] = (
        "PASS"
        if len(pair) == len(frozen)
        and all(result[f"max_abs_diff_{c}"] <= REGRESSION_TOLERANCE for c in ("paf", "paf_rr_lower", "paf_rr_upper"))
        else "FAIL"
    )
    return result


def main() -> None:
    rr = load_rr()

    historical = compute_historical(rr)
    hist_path = DATA / "paf_historical_2015_2024_all_causes_v1.parquet"
    historical.to_parquet(hist_path, index=False)

    outputs = [hist_path]
    regression = []
    futures = []
    for scenario in SCENARIOS:
        future = compute_future(rr, scenario)
        out_path = DATA / f"paf_future_{scenario}_2025_2050_all_causes_v1.parquet"
        future.to_parquet(out_path, index=False)
        outputs.append(out_path)
        futures.append((scenario, future))
        regression.append(regression_check_crc(future, scenario, 2050))
        regression.append(regression_check_crc(future, scenario, 2025))

    # QA gates.
    problems = []
    if historical["paf"].isna().any():
        problems.append("NaN historical PAF")
    if (historical["paf"] >= 1).any() or (historical["paf"] <= -1).any():
        problems.append("historical PAF outside (-1, 1)")
    for scenario, future in futures:
        if future["paf"].isna().any():
            problems.append(f"NaN future PAF in {scenario}")
        if (future["paf"] >= 1).any() or (future["paf"] <= -1).any():
            problems.append(f"future PAF outside (-1, 1) in {scenario}")
    reg_failed = [r for r in regression if r["status"] != "PASS"]
    if reg_failed:
        problems.append(f"CRC regression failures: {reg_failed}")

    negative_share = (
        historical.assign(neg=historical["paf"] < 0).groupby("cause_id")["neg"].mean().to_dict()
    )

    status = "PASS" if not problems else "FAIL"
    summary = {
        "run": "paf_all_causes_v1_20261005",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "causes": int(historical["cause_id"].nunique()),
        "historical_rows": int(len(historical)),
        "future_rows_per_scenario": {s: int(len(f)) for s, f in futures},
        "historical_years": [int(historical["year"].min()), int(historical["year"].max())],
        "future_years": FUTURE_YEARS[0:1] + FUTURE_YEARS[-1:],
        "anchor_85plus_rr_lookup": ANCHOR_85PLUS_LOOKUP,
        "negative_paf_share_by_cause_historical": {
            int(k): round(float(v), 6) for k, v in negative_share.items() if v > 0
        },
        "crc_regression": regression,
        "problems": problems,
        "status": status,
    }
    write_json(QA / "paf_all_causes_QA_v1.json", summary)

    lines = [
        "# PAF computation report (33 causes)",
        "",
        f"- historical rows: {len(historical):,} (2015-2024, anchor ages 25-29..85+)",
        "",
        "## CRC regression vs frozen CRC national PAF",
        "",
        "| scenario | year | rows | max|dpaf| | status |",
        "|---|---|---|---|---|---|",
    ]
    for r in regression:
        lines.append(
            f"| {r['scenario']} | {r['year']} | {r['rows_compared']} "
            f"| {r['max_abs_diff_paf']:.3g} | {r['status']} |"
        )
    lines += [
        "",
        "## Negative PAF shares (historical; expected for premenopausal breast cancer ages)",
        "",
    ]
    for k, v in summary["negative_paf_share_by_cause_historical"].items():
        lines.append(f"- cause {k}: {v:.4f}")
    lines += ["", f"QA status: {status}" + ("" if not problems else " — " + "; ".join(problems))]
    (REPORTS / "paf_all_causes_report_v1.md").write_text("\n".join(lines), encoding="utf-8")

    (SCRIPTS / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    outputs += [
        QA / "paf_all_causes_QA_v1.json",
        REPORTS / "paf_all_causes_report_v1.md",
        SCRIPTS / Path(__file__).name,
    ]
    pd.DataFrame(
        {"path": [str(p) for p in outputs], "sha256": [sha256(p) for p in outputs]}
    ).to_csv(ROOT / "paf_all_causes_formal_manifest_v1.csv", index=False, encoding="utf-8-sig")

    if status != "PASS":
        raise RuntimeError("PAF QA failed: " + "; ".join(problems))
    (ROOT / "COMPLETE.flag").write_text(
        f"paf_all_causes_v1_20261005 completed {summary['completed_at_utc']}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
