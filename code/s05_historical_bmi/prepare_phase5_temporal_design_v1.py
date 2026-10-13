"""Freeze the Phase 5 temporal backtesting design before model fitting."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(_cfg.RESULTS / "05_historical_bmi_1990_2024" / "phase5_temporal_backtesting_v1")
TRAINING = Path(_cfg.ROOT / "results" / "bmi_downscaling_ascii" / "experiments" / "BMI7_MRP_STEPS_regional_augmentation_20260812" / "data" / "bmi7_enhanced_plus_steps_regional_training_v1.csv.gz")
ASSIGNMENTS = Path(_cfg.ROOT / "results" / "bmi_downscaling_ascii" / "experiments" / "BMI7_MRP_STEPS_regional_augmentation_20260812" / "validation_design" / "all_eligible20_v1" / "all_eligible20_validation_assignments_v1.csv.gz")
WINDOWS = [
    ("time_cut2014_test2015_2017", 2014, 2015, 2017),
    ("time_cut2017_test2018_2020", 2017, 2018, 2020),
    ("time_cut2020_test2021_2024", 2020, 2021, 2024),
]
MARGINS = {
    "maximum_weighted_r2_decline": 0.05,
    "maximum_weighted_w2_increase_kg_m2": 0.15,
    "maximum_obesity_mae_increase_pp": 0.30,
    "maximum_seven_category_mae_increase_pp": 0.30,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    contract_path = ROOT / "contracts" / "Phase5_temporal_backtest_contract_v1.json"
    tasks_path = ROOT / "design" / "Phase5_temporal_tasks_v1.csv"
    rows_path = ROOT / "design" / "Phase5_temporal_row_roles_v1.csv.gz"
    qa_path = ROOT / "design" / "Phase5_design_QA_v1.json"
    for path in (contract_path, tasks_path, rows_path, qa_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen design: {path}")

    data = pd.read_csv(
        TRAINING,
        usecols=["row_id_stage01", "year", "iso3", "source_family", "survey_id"],
        low_memory=False,
    )
    assignments = pd.read_csv(ASSIGNMENTS, low_memory=False)
    if data["row_id_stage01"].duplicated().any() or assignments["row_id_stage01"].duplicated().any():
        raise RuntimeError("Duplicate row identifiers")
    if set(data["row_id_stage01"]) != set(assignments["row_id_stage01"]):
        raise RuntimeError("Training and assignment row identifiers differ")
    eligible = set(assignments.loc[assignments["eligible_validation"].eq(True), "row_id_stage01"])

    task_rows = []
    role_frames = []
    all_checks_pass = True
    for task_id, cutoff, test_start, test_end in WINDOWS:
        train_mask = data["year"].le(cutoff)
        test_mask = data["year"].between(test_start, test_end) & data["row_id_stage01"].isin(eligible)
        roles = pd.DataFrame({
            "task_id": task_id,
            "row_id_stage01": data["row_id_stage01"],
            "role": "excluded",
        })
        roles.loc[train_mask.to_numpy(), "role"] = "train"
        roles.loc[test_mask.to_numpy(), "role"] = "test"
        role_frames.append(roles)
        train = data.loc[train_mask]
        test = data.loc[test_mask]
        leakage = bool(set(train["row_id_stage01"]) & set(test["row_id_stage01"]))
        year_ok = bool(train["year"].max() <= cutoff and test["year"].between(test_start, test_end).all())
        all_checks_pass &= (not leakage) and year_ok and len(test) > 0
        task_rows.append({
            "task_id": task_id,
            "train_cutoff_year": cutoff,
            "test_start_year": test_start,
            "test_end_year": test_end,
            "train_rows": len(train),
            "test_rows": len(test),
            "excluded_rows": len(data) - len(train) - len(test),
            "train_countries": train["iso3"].nunique(),
            "test_countries": test["iso3"].nunique(),
            "train_sources": train["source_family"].nunique(),
            "test_sources": test["source_family"].nunique(),
            "unseen_country_test_rows": int((~test["iso3"].isin(set(train["iso3"]))).sum()),
            "unseen_source_test_rows": int((~test["source_family"].isin(set(train["source_family"]))).sum()),
            "unseen_survey_test_rows": int((~test["survey_id"].isin(set(train["survey_id"]))).sum()),
            "train_test_row_overlap": int(leakage),
            "year_contract_pass": year_ok,
        })

    contract = {
        "phase": "Phase 5 temporal extrapolation backtesting",
        "version": "v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Evaluate temporal transport before any 2025-2050 BMI prediction.",
        "interpretation": "Conditional spatial backtest given the test-year NCD-RisC country-age-sex BMI composition anchor.",
        "primary_population": "All eligible measured subnational validation records; CHN and JPN remain training-only.",
        "training_rule": "All available records with year <= cutoff, including auxiliary training sources.",
        "test_rule": "Eligible validation records whose year is within the fixed future test interval.",
        "paired_models": [
            "future-capable MRP",
            "original full-covariate MRP",
            "zero spatial-offset NCD-RisC anchor baseline",
        ],
        "primary_aggregation": "survey x geography x year x sex x broad age (20-39, 40-59, 60+)",
        "primary_minimum_effective_n": 20,
        "noninferiority_comparator": "original full-covariate MRP",
        "noninferiority_scope": "All four margins must pass in each fixed window and in the pooled result.",
        "noninferiority_margins": MARGINS,
        "calibration_definition": "Weighted least-squares observed obesity prevalence = intercept + slope * predicted obesity prevalence.",
        "formal_uncertainty_from_vb_draws_authorized": False,
        "future_bmi_prediction_authorized_by_this_script": False,
        "disease_burden_analysis_authorized_by_this_script": False,
        "windows": [
            {"task_id": task, "train_through": cutoff, "test_start": start, "test_end": end}
            for task, cutoff, start, end in WINDOWS
        ],
        "inputs": {
            "training_path": str(TRAINING),
            "training_sha256": sha256(TRAINING),
            "assignment_path": str(ASSIGNMENTS),
            "assignment_sha256": sha256(ASSIGNMENTS),
        },
    }
    tasks = pd.DataFrame(task_rows)
    roles = pd.concat(role_frames, ignore_index=True)
    tasks.to_csv(tasks_path, index=False, encoding="utf-8-sig")
    roles.to_csv(rows_path, index=False, compression="gzip")
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    qa = {
        "status": "PASS" if all_checks_pass else "FAIL",
        "training_rows": len(data),
        "eligible_validation_rows": len(eligible),
        "tasks": task_rows,
        "candidate_and_baseline_row_contract_identical": True,
    }
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    if not all_checks_pass:
        raise RuntimeError("Phase 5 design QA failed")
    print(tasks.to_string(index=False))


if __name__ == "__main__":
    main()
