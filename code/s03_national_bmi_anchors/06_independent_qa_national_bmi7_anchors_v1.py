"""Independent QA for the 2025-2050 national BMI7 anchor product."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BMI = [
    "p_bmi_lt_18_5", "p_bmi_18_5_20", "p_bmi_20_25", "p_bmi_25_30",
    "p_bmi_30_35", "p_bmi_35_40", "p_bmi_ge_40",
]
KEY = ["anchor_scenario", "location_id", "sex", "age_group_id", "year"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", required=True)
    args = parser.parse_args()

    root = Path(args.anchor_dir)
    table_dir = root / "tables"
    anchor_path = table_dir / "national_bmi7_anchors_2025_2050_v1.csv.gz"
    bridge_path = table_dir / "ncd2024_to_gbd2025_absolute_anchor_bridge_diagnostics_v1.csv.gz"
    data = pd.read_csv(anchor_path)

    checks: list[dict] = []
    expected_per_scenario = 204 * 2 * 15 * 26
    for scenario, frame in data.groupby("anchor_scenario", sort=True):
        closure_error = (frame[BMI].sum(axis=1) - 1).abs().max()
        duplicate_rows = frame.duplicated(KEY).sum()
        checks.extend([
            {"scenario": scenario, "check": "row_count", "value": len(frame), "expected": expected_per_scenario, "status": "PASS" if len(frame) == expected_per_scenario else "FAIL"},
            {"scenario": scenario, "check": "duplicate_key_rows", "value": int(duplicate_rows), "expected": 0, "status": "PASS" if duplicate_rows == 0 else "FAIL"},
            {"scenario": scenario, "check": "max_closure_error", "value": float(closure_error), "expected": "<1e-12", "status": "PASS" if closure_error < 1e-12 else "FAIL"},
            {"scenario": scenario, "check": "minimum_probability", "value": float(frame[BMI].min().min()), "expected": ">=0", "status": "PASS" if frame[BMI].min().min() >= 0 else "FAIL"},
            {"scenario": scenario, "check": "maximum_probability", "value": float(frame[BMI].max().max()), "expected": "<=1", "status": "PASS" if frame[BMI].max().max() <= 1 else "FAIL"},
            {"scenario": scenario, "check": "country_count", "value": int(frame.location_id.nunique()), "expected": 204, "status": "PASS" if frame.location_id.nunique() == 204 else "FAIL"},
            {"scenario": scenario, "check": "age_group_count", "value": int(frame.age_group_id.nunique()), "expected": 15, "status": "PASS" if frame.age_group_id.nunique() == 15 else "FAIL"},
            {"scenario": scenario, "check": "year_range", "value": f"{frame.year.min()}-{frame.year.max()}", "expected": "2025-2050", "status": "PASS" if (frame.year.min(), frame.year.max()) == (2025, 2050) else "FAIL"},
        ])

    primary = data[data.anchor_scenario.eq("gbd_absolute_primary")].copy()
    p25_error = (primary.p_bmi_25_30 - primary.overweight_less_obesity_mean).abs().max()
    pob_error = (primary[BMI[4:]].sum(axis=1) - primary.obesity_mean).abs().max()
    fallback_rows = primary.subcomposition_source.eq("global_age_sex_median_fallback").sum()
    reused_85_rows = primary.age_mapping_method.eq("ncd_85plus_reused_for_gbd_85plus_subgroup").sum()
    matched_location_count = primary.loc[
        ~primary.subcomposition_source.eq("global_age_sex_median_fallback"), "location_id"
    ].nunique()
    expected_85_reuse_rows = matched_location_count * 2 * 3 * 26
    checks.extend([
        {"scenario": "gbd_absolute_primary", "check": "p25_30_anchor_max_abs_error", "value": float(p25_error), "expected": "<1e-12", "status": "PASS" if p25_error < 1e-12 else "FAIL"},
        {"scenario": "gbd_absolute_primary", "check": "obesity_anchor_max_abs_error", "value": float(pob_error), "expected": "<1e-12", "status": "PASS" if pob_error < 1e-12 else "FAIL"},
        {"scenario": "gbd_absolute_primary", "check": "global_fallback_rows", "value": int(fallback_rows), "expected": 5 * 2 * 15 * 26, "status": "PASS" if fallback_rows == 5 * 2 * 15 * 26 else "FAIL"},
        {"scenario": "gbd_absolute_primary", "check": "ncd_85plus_reuse_rows", "value": int(reused_85_rows), "expected": expected_85_reuse_rows, "status": "PASS" if reused_85_rows == expected_85_reuse_rows else "FAIL"},
    ])

    bridge = pd.read_csv(bridge_path)
    bridge_summary = []
    for col in ["p25_30_jump_pp", "obesity_jump_pp"]:
        values = bridge[col].dropna()
        abs_values = values.abs()
        bridge_summary.append({
            "quantity": col,
            "n": len(values),
            "mean_pp": float(values.mean()),
            "median_pp": float(values.median()),
            "p95_absolute_pp": float(abs_values.quantile(0.95)),
            "p99_absolute_pp": float(abs_values.quantile(0.99)),
            "max_absolute_pp": float(abs_values.max()),
        })

    wide = data[data.year.eq(2050)].pivot_table(
        index=["location_id", "location_name", "sex", "age_group_id", "age_group_name"],
        columns="anchor_scenario",
        values=BMI,
    )
    comparison = []
    for col in BMI:
        delta = 100 * (wide[(col, "ncd_2024_delta_sensitivity")] - wide[(col, "gbd_absolute_primary")])
        comparison.append({
            "bmi_category": col,
            "mean_delta_pp": float(delta.mean()),
            "median_delta_pp": float(delta.median()),
            "p95_absolute_delta_pp": float(delta.abs().quantile(0.95)),
            "max_absolute_delta_pp": float(delta.abs().max()),
        })

    qa = pd.DataFrame(checks)
    qa.to_csv(table_dir / "independent_anchor_qa_v1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bridge_summary).to_csv(table_dir / "bridge_summary_v1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison).to_csv(table_dir / "anchor_scenario_comparison_2050_v1.csv", index=False, encoding="utf-8-sig")
    result = {
        "status": "PASS" if not qa.status.eq("FAIL").any() else "FAIL",
        "anchor_rows": len(data),
        "scenarios": sorted(data.anchor_scenario.unique().tolist()),
        "fallback_locations": sorted(primary.loc[primary.subcomposition_source.eq("global_age_sex_median_fallback"), "location_name"].unique().tolist()),
        "bridge_summary": bridge_summary,
    }
    (root / "independent_anchor_qa_metadata_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
