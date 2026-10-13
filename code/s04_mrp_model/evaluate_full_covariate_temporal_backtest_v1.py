"""Paired temporal evaluation for the full-covariate MRP refit."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(_cfg.RESULTS / "04_future_capable_mrp" / "full_covariate_mrp_refit_v1_20260905" / "temporal_backtest_v1")
TRAINING = Path(_cfg.RESULTS / "04_future_capable_mrp" / "full_covariate_mrp_refit_v1_20260905" / "data" / "bmi7_full_covariate_training_v1.csv.gz")
CONTRACT = ROOT / "contracts" / "full_covariate_temporal_backtest_contract_v1.json"
TASKS = [
    ("time_cut2014_test2015_2017", 2014, 2015, 2017),
    ("time_cut2017_test2018_2020", 2017, 2018, 2020),
    ("time_cut2020_test2021_2024", 2020, 2021, 2024),
]
MODEL_ROOTS = {
    "full_covariate_mrp": ROOT,
    "original_full_mrp": Path(_cfg.RESULTS / "05_historical_bmi_1990_2024" / "phase5_temporal_backtesting_v1" / "baseline_full_mrp"),
}
BMI = ["bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30", "bmi_30_35", "bmi_35_40", "bmi_ge_40"]
SUPPORT = np.array([17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5])
PRIMARY_N = 20
MARGINS = {
    "weighted_r2": 0.05,
    "weighted_w2_bmi_kg_m2": 0.15,
    "obesity_weighted_mae_pp": 0.30,
    "seven_category_weighted_mae_pp": 0.30,
}
CANDIDATE_PREDICTORS = [
    "ntl_within", "pop_density_within", "gdp_within", "pm25_within_aug", "urban_share_within",
    "built_fraction_within", "travel_time_within", "temperature_c_within_aug",
    "relative_humidity_pct_within_aug", "elevation_m_within_aug", "age20_39_share_within_aug",
    "age60plus_share_within_aug", "female_adult_share_within_aug",
]
ORIGINAL_PREDICTORS = [
    "ntl_within", "pop_density_within", "gdp_within", "pm25_within_aug", "urban_share_within",
    "built_fraction_within", "travel_time_within", "temperature_c_within_aug",
    "relative_humidity_pct_within_aug", "elevation_m_within_aug", "age20_39_share_within_aug",
    "age60plus_share_within_aug", "female_adult_share_within_aug",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_to_markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    """Render a compact Markdown table without optional pandas dependencies."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append("NA" if not np.isfinite(value) else f"{value:.{digits}f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return math.nan if not keep.any() else float(np.average(x[keep], weights=w[keep]))


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[keep], y[keep], w[keep]
    if len(x) < 3:
        return math.nan
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    vx, vy = np.average((x - xm) ** 2, weights=w), np.average((y - ym) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return math.nan
    return float(np.average((x - xm) * (y - ym), weights=w) / np.sqrt(vx * vy))


def weighted_r2(observed: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    keep = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(weight) & (weight > 0)
    observed, predicted, weight = observed[keep], predicted[keep], weight[keep]
    if len(observed) < 2:
        return math.nan
    center = np.average(observed, weights=weight)
    denominator = np.sum(weight * (observed - center) ** 2)
    return math.nan if denominator <= 0 else float(1 - np.sum(weight * (observed - predicted) ** 2) / denominator)


def weighted_calibration(observed: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    keep = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(weight) & (weight > 0)
    observed, predicted, weight = observed[keep], predicted[keep], weight[keep]
    if len(observed) < 2 or np.var(predicted) <= 0:
        return math.nan, math.nan
    design = np.column_stack([np.ones(len(predicted)), predicted])
    solution = np.linalg.lstsq(design * np.sqrt(weight)[:, None], observed * np.sqrt(weight), rcond=None)[0]
    return float(solution[0]), float(solution[1])


def close_composition(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), 1e-12, None)
    return x / x.sum(axis=1, keepdims=True)


def row_w2(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed, predicted = close_composition(observed), close_composition(predicted)
    result = np.empty(len(observed), float)
    for row in range(len(observed)):
        boundaries = np.unique(np.concatenate(([0.0, 1.0], np.cumsum(observed[row]), np.cumsum(predicted[row]))))
        middle = (boundaries[:-1] + boundaries[1:]) / 2
        width = np.diff(boundaries)
        oi = np.clip(np.searchsorted(np.cumsum(observed[row]), middle, side="right"), 0, 6)
        pi = np.clip(np.searchsorted(np.cumsum(predicted[row]), middle, side="right"), 0, 6)
        result[row] = np.sqrt(np.sum(width * (SUPPORT[oi] - SUPPORT[pi]) ** 2))
    return result


def broad_age(age_group: pd.Series) -> pd.Series:
    lower = pd.to_numeric(age_group.astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
    upper = pd.to_numeric(age_group.astype(str).str.extract(r"-(\d+)")[0], errors="coerce")
    midpoint = ((lower + upper) / 2).where(upper.notna(), lower + 5)
    result = pd.cut(midpoint, [19.999, 39.999, 59.999, np.inf], labels=["20-39", "40-59", "60+"], right=True)
    if result.isna().any():
        raise RuntimeError("Unmapped age group")
    return result.astype(str)


def load_raw_predictions() -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    frames = []
    qa = {}
    for task_id, cutoff, test_start, test_end in TASKS:
        task_data = {}
        for model, model_root in MODEL_ROOTS.items():
            path = model_root / "validation" / "predictions" / f"{task_id}_predictions_v1.csv.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, low_memory=False)
            frame["model"] = model
            frame["window"] = f"{test_start}-{test_end}"
            if not frame["year"].between(test_start, test_end).all():
                raise RuntimeError(f"Test-year contract failed: {model}, {task_id}")
            if frame["row_id_stage01"].duplicated().any():
                raise RuntimeError(f"Duplicate raw prediction IDs: {model}, {task_id}")
            task_data[model] = frame
            frames.append(frame)
        left = task_data["full_covariate_mrp"].sort_values("row_id_stage01")
        right = task_data["original_full_mrp"].sort_values("row_id_stage01")
        paired = left["row_id_stage01"].tolist() == right["row_id_stage01"].tolist()
        observed_equal = np.allclose(
            left[[f"observed_{x}" for x in BMI]].to_numpy(),
            right[[f"observed_{x}" for x in BMI]].to_numpy(), atol=1e-12,
        )
        baseline_equal = np.allclose(
            left[[f"baseline_{x}" for x in BMI]].to_numpy(),
            right[[f"baseline_{x}" for x in BMI]].to_numpy(), atol=1e-12,
        )
        qa[task_id] = {
            "cutoff": cutoff, "test_start": test_start, "test_end": test_end,
            "raw_rows": len(left), "paired_row_ids": paired,
            "observed_values_equal": observed_equal, "anchor_values_equal": baseline_equal,
        }
        if not (paired and observed_equal and baseline_equal):
            raise RuntimeError(f"Paired raw prediction QA failed: {task_id}")
    return pd.concat(frames, ignore_index=True), qa


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["broad_age"] = broad_age(raw["age_group"])
    observed = [f"observed_{x}" for x in BMI]
    baseline = [f"baseline_{x}" for x in BMI]
    groups = ["model", "window", "source_family", "source_role", "survey_id", "iso3", "geography_id", "year", "sex", "broad_age", "development_stage"]
    value_frames = []
    for model, frame in raw.groupby("model", sort=False):
        predicted = [f"predicted_{x}" for x in BMI]
        work = frame[groups + ["pseudo_n", *observed, *predicted, *baseline]].copy()
        weight = work["pseudo_n"].to_numpy(float)
        for column in [*observed, *predicted, *baseline]:
            work[column] = pd.to_numeric(work[column], errors="raise") * weight
        work["underlying_rows"] = 1
        result = work.groupby(groups, observed=True, dropna=False, sort=False).sum(numeric_only=True).reset_index()
        for column in [*observed, *predicted, *baseline]:
            result[column] /= result["pseudo_n"]
        value_frames.append(result)
    cells = pd.concat(value_frames, ignore_index=True)
    for columns in ([f"observed_{x}" for x in BMI], [f"predicted_{x}" for x in BMI], [f"baseline_{x}" for x in BMI]):
        cells[columns] = close_composition(cells[columns].to_numpy(float))
    cells["validation_cell_id"] = cells[["window", "survey_id", "iso3", "geography_id", "year", "sex", "broad_age"]].astype(str).agg("|".join, axis=1)

    anchor = cells.loc[cells["model"].eq("full_covariate_mrp")].copy()
    anchor["model"] = "ncd_risc_anchor_only"
    for label in BMI:
        anchor[f"predicted_{label}"] = anchor[f"baseline_{label}"]
    cells = pd.concat([cells, anchor], ignore_index=True)
    return cells


def within_country_corr(frame: pd.DataFrame, observed: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    temporary = pd.DataFrame({"iso3": frame["iso3"].to_numpy(), "observed": observed, "predicted": predicted, "weight": weight})
    def demean(group: pd.DataFrame) -> pd.DataFrame:
        w = group["weight"].to_numpy(float)
        group = group.copy()
        group["observed"] -= np.average(group["observed"], weights=w)
        group["predicted"] -= np.average(group["predicted"], weights=w)
        return group
    centered = temporary.groupby("iso3", group_keys=False).apply(demean, include_groups=False)
    return weighted_corr(centered["observed"].to_numpy(), centered["predicted"].to_numpy(), centered["weight"].to_numpy())


def calculate_metrics(frame: pd.DataFrame, scope: str, model: str) -> dict[str, object]:
    observed = close_composition(frame[[f"observed_{x}" for x in BMI]].to_numpy(float))
    predicted = close_composition(frame[[f"predicted_{x}" for x in BMI]].to_numpy(float))
    baseline = close_composition(frame[[f"baseline_{x}" for x in BMI]].to_numpy(float))
    weight = frame["pseudo_n"].to_numpy(float)
    oo, po, bo = observed[:, 4:].sum(axis=1), predicted[:, 4:].sum(axis=1), baseline[:, 4:].sum(axis=1)
    intercept, slope = weighted_calibration(oo, po, weight)
    return {
        "scope": scope,
        "model": model,
        "validation_cells": len(frame),
        "underlying_rows": int(frame["underlying_rows"].sum()),
        "effective_n": float(weight.sum()),
        "countries": frame["iso3"].nunique(),
        "sources": frame["source_family"].nunique(),
        "weighted_r": weighted_corr(oo, po, weight),
        "weighted_r2": weighted_r2(oo, po, weight),
        "weighted_w2_bmi_kg_m2": weighted_mean(row_w2(observed, predicted), weight),
        "seven_category_weighted_mae_pp": 100 * weighted_mean(np.abs(observed - predicted).mean(axis=1), weight),
        "obesity_weighted_mae_pp": 100 * weighted_mean(np.abs(oo - po), weight),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "within_anchor_obesity_deviation_r": weighted_corr(oo - bo, po - bo, weight),
        "within_country_demeaned_obesity_r": within_country_corr(frame, oo, po, weight),
        "prediction_closure_max_error": float(np.abs(predicted.sum(axis=1) - 1).max()),
    }


def extrapolation_qa(raw_candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = sorted(set(CANDIDATE_PREDICTORS + ORIGINAL_PREDICTORS + ["row_id_stage01", "year", "iso3", "source_family", "survey_id", "pseudo_n"]))
    data = pd.read_csv(TRAINING, usecols=columns, low_memory=False)
    rows = []
    levels = []
    for task_id, cutoff, start, end in TASKS:
        test_ids = set(raw_candidate.loc[raw_candidate["window"].eq(f"{start}-{end}"), "row_id_stage01"])
        train = data.loc[data["year"].le(cutoff)]
        test = data.loc[data["row_id_stage01"].isin(test_ids)]
        for model, predictors in (("full_covariate_mrp", CANDIDATE_PREDICTORS), ("original_full_mrp", ORIGINAL_PREDICTORS)):
            any_out = np.zeros(len(test), dtype=bool)
            for variable in predictors:
                tr = pd.to_numeric(train[variable], errors="coerce")
                te = pd.to_numeric(test[variable], errors="coerce")
                lower, upper = float(tr.min()), float(tr.max())
                outside = te.lt(lower) | te.gt(upper) | te.isna()
                any_out |= outside.to_numpy()
                rows.append({
                    "window": f"{start}-{end}", "model": model, "variable": variable,
                    "train_min": lower, "train_max": upper, "test_min": float(te.min()), "test_max": float(te.max()),
                    "test_rows": len(test), "rows_outside_train_range": int(outside.sum()),
                    "weighted_fraction_outside_train_range": weighted_mean(outside.to_numpy(float), test["pseudo_n"].to_numpy(float)),
                })
            rows.append({
                "window": f"{start}-{end}", "model": model, "variable": "ANY_PREDICTOR",
                "train_min": math.nan, "train_max": math.nan, "test_min": math.nan, "test_max": math.nan,
                "test_rows": len(test), "rows_outside_train_range": int(any_out.sum()),
                "weighted_fraction_outside_train_range": weighted_mean(any_out.astype(float), test["pseudo_n"].to_numpy(float)),
            })
        for level_name in ("iso3", "source_family", "survey_id"):
            unseen = ~test[level_name].isin(set(train[level_name]))
            levels.append({
                "window": f"{start}-{end}", "level": level_name, "test_rows": len(test),
                "unseen_rows": int(unseen.sum()),
                "weighted_fraction_unseen": weighted_mean(unseen.to_numpy(float), test["pseudo_n"].to_numpy(float)),
                "unseen_values": "|".join(sorted(test.loc[unseen, level_name].astype(str).unique())),
            })
    return pd.DataFrame(rows), pd.DataFrame(levels)


def main() -> None:
    report_path = ROOT / "evaluation" / "Gate_D_report_v1.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    raw, pair_qa = load_raw_predictions()
    cells = aggregate(raw)
    primary = cells.loc[cells["pseudo_n"].ge(PRIMARY_N)].copy()

    metric_rows = []
    for window, frame in primary.groupby("window", sort=True):
        for model, model_frame in frame.groupby("model", sort=True):
            metric_rows.append(calculate_metrics(model_frame, window, model))
    for model, model_frame in primary.groupby("model", sort=True):
        metric_rows.append(calculate_metrics(model_frame, "pooled", model))
    metrics = pd.DataFrame(metric_rows)

    stratified_rows = []
    for dimension in ("iso3", "source_family", "broad_age", "sex", "development_stage"):
        for (window, model, level), frame in primary.groupby(["window", "model", dimension], observed=True, dropna=False):
            if len(frame) < 2:
                continue
            row = calculate_metrics(frame, window, model)
            row.update({"dimension": dimension, "level": level})
            stratified_rows.append(row)
    stratified = pd.DataFrame(stratified_rows)

    comparisons = []
    scopes = [f"{start}-{end}" for _, _, start, end in TASKS] + ["pooled"]
    for scope in scopes:
        scope_metrics = metrics.loc[metrics["scope"].eq(scope)].set_index("model")
        candidate, comparator = scope_metrics.loc["full_covariate_mrp"], scope_metrics.loc["original_full_mrp"]
        tests = [
            ("weighted_r2_decline", float(comparator["weighted_r2"] - candidate["weighted_r2"]), MARGINS["weighted_r2"]),
            ("weighted_w2_increase_kg_m2", float(candidate["weighted_w2_bmi_kg_m2"] - comparator["weighted_w2_bmi_kg_m2"]), MARGINS["weighted_w2_bmi_kg_m2"]),
            ("obesity_mae_increase_pp", float(candidate["obesity_weighted_mae_pp"] - comparator["obesity_weighted_mae_pp"]), MARGINS["obesity_weighted_mae_pp"]),
            ("seven_category_mae_increase_pp", float(candidate["seven_category_weighted_mae_pp"] - comparator["seven_category_weighted_mae_pp"]), MARGINS["seven_category_weighted_mae_pp"]),
        ]
        for check, adverse_delta, margin in tests:
            comparisons.append({"scope": scope, "check": check, "adverse_delta": adverse_delta, "margin": margin, "status": "PASS" if adverse_delta <= margin else "FAIL"})
    pass_fail = pd.DataFrame(comparisons)

    candidate_raw = raw.loc[raw["model"].eq("full_covariate_mrp")].copy()
    extrapolation, level_qa = extrapolation_qa(candidate_raw)
    pair_pass = all(all(v[k] for k in ("paired_row_ids", "observed_values_equal", "anchor_values_equal")) for v in pair_qa.values())
    closure_pass = bool(metrics["prediction_closure_max_error"].max() <= 1e-10)
    gate_pass = bool(pass_fail["status"].eq("PASS").all() and pair_pass and closure_pass)

    cells_path = ROOT / "evaluation" / "Gate_D_paired_aggregated_cells_v1.csv.gz"
    metrics_path = ROOT / "evaluation" / "Gate_D_metrics_v1.csv"
    stratified_path = ROOT / "evaluation" / "Gate_D_stratified_metrics_v1.csv.gz"
    pass_fail_path = ROOT / "evaluation" / "Gate_D_pass_fail_v1.csv"
    extrapolation_path = ROOT / "evaluation" / "Gate_D_extrapolation_QA_v1.csv"
    levels_path = ROOT / "evaluation" / "Gate_D_unseen_levels_QA_v1.csv"
    qa_path = ROOT / "evaluation" / "Gate_D_QA_v1.json"
    cells.to_csv(cells_path, index=False, compression="gzip")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    stratified.to_csv(stratified_path, index=False, compression="gzip")
    pass_fail.to_csv(pass_fail_path, index=False, encoding="utf-8-sig")
    extrapolation.to_csv(extrapolation_path, index=False, encoding="utf-8-sig")
    level_qa.to_csv(levels_path, index=False, encoding="utf-8-sig")

    warnings = []
    first = level_qa.loc[(level_qa["window"].eq("2015-2017")) & (level_qa["level"].eq("iso3"))]
    if not first.empty and first["weighted_fraction_unseen"].iat[0] == 1:
        warnings.append("The 2015-2017 window is simultaneously a temporal and unseen-country stress test because all test countries are absent before 2015.")
    high_extrapolation = extrapolation.loc[(extrapolation["variable"].eq("ANY_PREDICTOR")) & (extrapolation["weighted_fraction_outside_train_range"].gt(0.25))]
    for row in high_extrapolation.itertuples():
        warnings.append(f"{row.window} {row.model}: {row.weighted_fraction_outside_train_range:.1%} of effective test weight is outside at least one training predictor range.")
    warnings.append("Mean-field VB predictions are used for point-estimate backtesting only; approximate draws remain unauthorized for formal uncertainty intervals.")

    qa = {
        "status": "PASS" if gate_pass else "FAIL",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": sha256(CONTRACT),
        "primary_minimum_effective_n": PRIMARY_N,
        "paired_raw_QA": pair_qa,
        "paired_raw_QA_pass": pair_pass,
        "prediction_closure_pass": closure_pass,
        "all_noninferiority_checks_pass": bool(pass_fail["status"].eq("PASS").all()),
        "warnings": warnings,
        "future_bmi_prediction_authorized": gate_pass,
        "disease_burden_analysis_performed": False,
    }
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    primary_table = metrics.loc[metrics["model"].isin(["full_covariate_mrp", "original_full_mrp", "ncd_risc_anchor_only"]), [
        "scope", "model", "validation_cells", "countries", "weighted_r2", "weighted_w2_bmi_kg_m2",
        "obesity_weighted_mae_pp", "seven_category_weighted_mae_pp", "calibration_intercept", "calibration_slope",
        "within_anchor_obesity_deviation_r", "within_country_demeaned_obesity_r",
    ]]
    report = [
        "# Full-covariate MRP temporal extrapolation backtesting",
        "",
        f"**Gate D status: {'PASS' if gate_pass else 'FAIL'}**",
        "",
        "This is a conditional spatial backtest using the observed test-year NCD-RisC country-age-sex composition as the national anchor. It does not validate national BMI trend forecasting.",
        "",
        "## Frozen design",
        "",
        "- Train through 2014, test 2015-2017.",
        "- Train through 2017, test 2018-2020.",
        "- Train through 2020, test 2021-2024.",
        "- Primary analysis aggregates to survey x geography x year x sex x broad age and requires effective n >= 20.",
        "- The full-covariate and original full MRP are evaluated on exactly paired rows and cells; the NCD-RisC anchor-only result is descriptive.",
        "",
        "## Primary metrics",
        "",
        dataframe_to_markdown(primary_table),
        "",
        "## Noninferiority checks",
        "",
        dataframe_to_markdown(pass_fail),
        "",
        "## QA and interpretation",
        "",
        f"- Paired row QA: {'PASS' if pair_pass else 'FAIL'}.",
        f"- Composition closure QA: {'PASS' if closure_pass else 'FAIL'}.",
        f"- All {len(pass_fail)} fixed noninferiority checks: {'PASS' if pass_fail['status'].eq('PASS').all() else 'FAIL'}.",
    ]
    report.extend(f"- WARN: {warning}" for warning in warnings)
    report.extend([
        "",
        "## Decision",
        "",
        ("The temporal gate passes for the full-covariate MRP, subject to the remaining bridge sensitivities and a separate prediction contract." if gate_pass else "The temporal gate fails. Formal 2025-2050 BMI grid production is not authorized; inspect failed windows and revise the model or downgrade claims to anchor-constrained scenario allocation."),
        "",
        "No future BMI grids or disease-burden outputs were created in Phase 5.",
    ])
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest_path = ROOT / "evaluation" / "Gate_D_artifact_sha256_v1.csv"
    artifacts = sorted(p for p in ROOT.rglob("*") if p.is_file() and p != manifest_path)
    pd.DataFrame([{"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in artifacts]).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(primary_table.to_string(index=False))
    print(pass_fail.to_string(index=False))
    print(f"GATE_D_STATUS={'PASS' if gate_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
