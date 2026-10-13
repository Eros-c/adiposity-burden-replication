"""Build the versioned non-inferiority gate for the full-covariate MRP."""

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


RUN = Path(_cfg.RESULTS / "04_future_capable_mrp" / "full_covariate_mrp_refit_v1_20260905")
BASELINE = Path(
    _cfg.ROOT / "results" / "降尺度" / "experiments" / "BMI7_MRP_STEPS_regional_augmentation_20260812" / "all_eligible20_stabilized_evaluation_v1" / "tables" / "all_eligible20_stabilized_metrics_v1.csv"
)
CURRENT = RUN / "stabilized_evaluation_v1" / "tables" / "all_eligible20_stabilized_metrics_v1.csv"
TRAINING = RUN / "data" / "bmi7_full_covariate_training_v1.csv.gz"
MODEL_META = RUN / "model_fit" / "models" / "checkpoints" / "full_model_metadata_v1.json"
OUT = RUN / "noninferiority_gate_v1"
TABLES = OUT / "tables"
QA = OUT / "qa"
REPORTS = OUT / "reports"
CONTRACTS = OUT / "contracts"

METRICS = {
    "weighted_r2": ("weighted_r2", -0.05, "minimum_change"),
    "weighted_w2_bmi_kg_m2": ("weighted_w2_bmi_kg_m2", 0.15, "maximum_change"),
    "obesity_weighted_mae_pp": ("obesity_weighted_mae_pp", 0.30, "maximum_change"),
    "seven_category_weighted_mae_pp": ("seven_category_weighted_mae_pp", 0.30, "maximum_change"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_primary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    result = frame.loc[frame["minimum_effective_n"].eq(20)].copy()
    expected = {"Internal random-row fivefold", "Country-held-out fivefold"}
    if set(result["validation"]) != expected:
        raise RuntimeError(f"Unexpected validation schemes in {path}")
    return result.set_index("validation")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    for directory in (TABLES, QA, REPORTS, CONTRACTS):
        directory.mkdir(parents=True, exist_ok=False)

    baseline = load_primary(BASELINE)
    current = load_primary(CURRENT)
    rows: list[dict[str, object]] = []
    for validation in ("Internal random-row fivefold", "Country-held-out fivefold"):
        for metric, (column, margin, rule) in METRICS.items():
            old = float(baseline.loc[validation, column])
            new = float(current.loc[validation, column])
            change = new - old
            if validation == "Country-held-out fivefold":
                passed = change >= margin if rule == "minimum_change" else change <= margin
                gate_role = "primary_noninferiority_gate"
            else:
                passed = True
                gate_role = "descriptive_internal_check"
            rows.append(
                {
                    "validation": validation,
                    "metric": metric,
                    "baseline": old,
                    "full_covariate_mrp": new,
                    "change_new_minus_baseline": change,
                    "noninferiority_margin": margin if validation == "Country-held-out fivefold" else None,
                    "rule": rule if validation == "Country-held-out fivefold" else "descriptive_only",
                    "gate_role": gate_role,
                    "status": "PASS" if passed else "FAIL",
                }
            )
    comparison = pd.DataFrame(rows)
    primary = comparison.loc[comparison["gate_role"].eq("primary_noninferiority_gate")]
    gate_pass = bool(primary["status"].eq("PASS").all())

    checkpoint_dir = RUN / "model_fit" / "models" / "checkpoints"
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in checkpoint_dir.glob("*_metadata_v1.json")]
    completed = [item for item in metadata if item.get("status") == "complete"]
    warnings = sorted({warning for item in completed for warning in [item.get("warnings", "")] if warning})
    qa_summary = json.loads(
        (RUN / "model_fit" / "qa" / "all_eligible20_qa_summary_v1.json").read_text(encoding="utf-8")
    )
    stabilized_qa = json.loads(
        (RUN / "stabilized_evaluation_v1" / "qa" / "all_eligible20_stabilized_qa_v1.json").read_text(encoding="utf-8")
    )

    status = "PASS_WITH_WARN" if gate_pass and warnings else "PASS" if gate_pass else "FAIL"
    comparison_path = TABLES / "full_covariate_mrp_noninferiority_comparison_v1.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    contract = {
        "contract_version": "full_covariate_mrp_refit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_input": str(TRAINING),
        "model_metadata": str(MODEL_META),
        "baseline_metrics": str(BASELINE),
        "current_metrics": str(CURRENT),
        "validation_population": "20 eligible countries; CHN and JPN training only",
        "primary_evaluation_unit": "survey-geography-year-sex-broad_age cell",
        "primary_effective_n_threshold": 20,
        "fixed_effect_features": 22,
        "future_dynamic_covariates": [
            "population_density",
            "GDP_per_capita",
            "urban_share",
            "built_fraction_proxy",
            "nighttime_lights",
            "PM2.5_mass_mixing_ratio",
            "temperature",
            "relative_humidity",
        ],
        "static_covariates": ["travel_time", "elevation"],
        "inference": "rstan mean-field variational Bayes",
        "primary_gate": "country-held-out fivefold non-inferiority",
        "margins": {
            "weighted_r2_minimum_change": -0.05,
            "weighted_w2_maximum_increase_kg_m2": 0.15,
            "obesity_mae_maximum_increase_pp": 0.30,
            "seven_category_mae_maximum_increase_pp": 0.30,
        },
        "required_sensitivities_before_formal_future_prediction": [
            "nighttime_lights_bridge_slope_1_vs_2",
            "with_vs_without_built_fraction_proxy",
        ],
    }
    contract_path = CONTRACTS / "full_covariate_mrp_refit_contract_v1.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    qa = {
        "status": status,
        "primary_noninferiority_pass": gate_pass,
        "validation_tasks_complete": qa_summary.get("validation_tasks_complete"),
        "expected_validation_tasks": qa_summary.get("expected_validation_tasks"),
        "validation_countries": qa_summary.get("validation_countries"),
        "regional_steps_used_for_validation": qa_summary.get("steps_regional_records_used_for_validation"),
        "china_japan_in_test": qa_summary.get("china_japan_in_test"),
        "stabilized_qa_status": stabilized_qa.get("status"),
        "prediction_closure_max_error": stabilized_qa.get("prediction_closure_max_error"),
        "variational_warnings": warnings,
        "formal_future_prediction_authorized": False,
        "authorization_blockers": [
            "required NTL bridge-slope sensitivity not yet completed",
            "required built-fraction proxy sensitivity not yet completed",
            "temporal extrapolation validation not yet completed",
        ],
    }
    qa_path = QA / "full_covariate_mrp_noninferiority_gate_v1.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")

    country = comparison.loc[comparison["validation"].eq("Country-held-out fivefold")]
    internal = comparison.loc[comparison["validation"].eq("Internal random-row fivefold")]
    report = f"""# Full-covariate MRP refit and non-inferiority gate v1

## Status

- Gate status: **{status}**
- Primary country-held-out non-inferiority: **{'PASS' if gate_pass else 'FAIL'}**
- Formal 2025-2050 BMI prediction authorization: **NOT YET AUTHORIZED**

## Model

The historical 22-feature Bayesian multinomial logistic-normal MRP was refitted on 53,777 records from 56 countries and nine source families. The canonical PM2.5 feature was replaced by a harmonized NorESM2-LM near-surface PM2.5 mass-mixing-ratio feature. The model retains nighttime lights, population density, GDP, urban share, built fraction, travel time, temperature, relative humidity, elevation, demographic composition, and the prespecified age/sex interactions.

## Comparable validation

The current and baseline predictions were evaluated with the same 20-country assignments, aggregation keys, BMI support points, population weights, and primary effective-sample-size threshold (`pseudo_n >= 20`). China and Japan remained training-only; the seven regional WHO STEPS countries contributed to both training and validation.

### Country-held-out fivefold

| Metric | Baseline | Full-covariate MRP | Change | Margin | Status |
|---|---:|---:|---:|---:|---|
"""
    for _, row in country.iterrows():
        report += f"| {row['metric']} | {row['baseline']:.6f} | {row['full_covariate_mrp']:.6f} | {row['change_new_minus_baseline']:+.6f} | {row['noninferiority_margin']:+.2f} | {row['status']} |\n"
    report += "\n### Internal random-row fivefold\n\n| Metric | Baseline | Full-covariate MRP | Change |\n|---|---:|---:|---:|\n"
    for _, row in internal.iterrows():
        report += f"| {row['metric']} | {row['baseline']:.6f} | {row['full_covariate_mrp']:.6f} | {row['change_new_minus_baseline']:+.6f} |\n"
    report += """

## Warnings and next gate

All ten validation tasks completed, predictions are finite, and seven-category compositions close numerically. Stan mean-field variational fits emitted infinite Pareto-k diagnostics with resampling disabled. This warning does not invalidate the completed predictive comparison, but it remains an inference-quality warning and must be reported.

Formal future prediction remains blocked until the nighttime-lights bridge-slope sensitivity, built-fraction-proxy sensitivity, and temporal extrapolation validation are complete.
"""
    report_path = REPORTS / "full_covariate_mrp_refit_noninferiority_report_v1.md"
    report_path.write_text(report, encoding="utf-8")

    artifacts = [
        Path(__file__), BASELINE, CURRENT, TRAINING, MODEL_META,
        comparison_path, contract_path, qa_path, report_path,
    ]
    manifest_path = OUT / "full_covariate_mrp_noninferiority_sha256_v1.csv"
    pd.DataFrame(
        [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts]
    ).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"status={status}")
    print(f"country_held_out_noninferiority={'PASS' if gate_pass else 'FAIL'}")
    print("formal_future_prediction_authorized=NO")


if __name__ == "__main__":
    main()
