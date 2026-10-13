"""Paired stabilized evaluation of six source-family holdouts."""

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


RUN = Path(_cfg.RESULTS / "04_future_capable_mrp" / "full_covariate_mrp_refit_v1_20260905")
CURRENT = RUN / "source_holdout_v1"
COMPARATOR = Path(
    _cfg.RESULTS / "04_future_capable_mrp" / "phase4b_d1_reduced_demography_v1_20260831" / "source_comparator_v1"
)
OUT = CURRENT / "evaluation"
TASKS = [
    "source_holdout_dhs", "source_holdout_knhanes", "source_holdout_ensanut",
    "source_holdout_shes", "source_holdout_steps", "source_holdout_pns",
]
BMI = ["bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30", "bmi_30_35", "bmi_35_40", "bmi_ge_40"]
SUPPORT = np.array([17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5])
PRIMARY_N = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = [f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in values]
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
    vx = np.average((x - xm) ** 2, weights=w)
    vy = np.average((y - ym) ** 2, weights=w)
    return math.nan if vx <= 0 or vy <= 0 else float(np.average((x - xm) * (y - ym), weights=w) / np.sqrt(vx * vy))


def weighted_r2(observed: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    keep = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(weight) & (weight > 0)
    observed, predicted, weight = observed[keep], predicted[keep], weight[keep]
    center = np.average(observed, weights=weight)
    denominator = np.sum(weight * (observed - center) ** 2)
    return math.nan if denominator <= 0 else float(1 - np.sum(weight * (observed - predicted) ** 2) / denominator)


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


def load_predictions() -> tuple[pd.DataFrame, dict[str, object]]:
    frames = []
    qa: dict[str, object] = {}
    for task in TASKS:
        pairs = {}
        for model, root in (("full_covariate_mrp", CURRENT), ("original_full_mrp", COMPARATOR)):
            path = root / "validation" / "predictions" / f"{task}_predictions_v1.csv.gz"
            frame = pd.read_csv(path, low_memory=False)
            frame["model"] = model
            frame["task"] = task
            pairs[model] = frame.sort_values("row_id_stage01")
            frames.append(frame)
        left, right = pairs["full_covariate_mrp"], pairs["original_full_mrp"]
        observed = [f"observed_{label}" for label in BMI]
        baseline = [f"baseline_{label}" for label in BMI]
        qa[task] = {
            "rows": len(left),
            "held_out_source": left["source_family"].unique().tolist(),
            "paired_row_ids": left["row_id_stage01"].tolist() == right["row_id_stage01"].tolist(),
            "observed_equal": bool(np.allclose(left[observed], right[observed], atol=1e-12)),
            "anchor_equal": bool(np.allclose(left[baseline], right[baseline], atol=1e-12)),
        }
    return pd.concat(frames, ignore_index=True), qa


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["broad_age"] = broad_age(raw["age_group"])
    observed = [f"observed_{label}" for label in BMI]
    predicted = [f"predicted_{label}" for label in BMI]
    baseline = [f"baseline_{label}" for label in BMI]
    groups = ["model", "task", "source_family", "source_role", "survey_id", "iso3", "geography_id", "year", "sex", "broad_age"]
    work = raw[groups + ["pseudo_n", *observed, *predicted, *baseline]].copy()
    weight = work["pseudo_n"].to_numpy(float)
    for column in [*observed, *predicted, *baseline]:
        work[column] = pd.to_numeric(work[column], errors="raise") * weight
    work["underlying_rows"] = 1
    cells = work.groupby(groups, observed=True, dropna=False, sort=False).sum(numeric_only=True).reset_index()
    for column in [*observed, *predicted, *baseline]:
        cells[column] /= cells["pseudo_n"]
    for columns in (observed, predicted, baseline):
        cells[columns] = close_composition(cells[columns].to_numpy(float))
    return cells


def metrics(frame: pd.DataFrame) -> dict[str, object]:
    observed = frame[[f"observed_{label}" for label in BMI]].to_numpy(float)
    predicted = frame[[f"predicted_{label}" for label in BMI]].to_numpy(float)
    baseline = frame[[f"baseline_{label}" for label in BMI]].to_numpy(float)
    weight = frame["pseudo_n"].to_numpy(float)
    oo, po, bo = observed[:, 4:].sum(axis=1), predicted[:, 4:].sum(axis=1), baseline[:, 4:].sum(axis=1)
    return {
        "validation_cells": len(frame),
        "underlying_rows": int(frame["underlying_rows"].sum()),
        "effective_n": float(weight.sum()),
        "countries": frame["iso3"].nunique(),
        "weighted_r": weighted_corr(oo, po, weight),
        "weighted_r2": weighted_r2(oo, po, weight),
        "weighted_w2_bmi_kg_m2": weighted_mean(row_w2(observed, predicted), weight),
        "seven_category_weighted_mae_pp": 100 * weighted_mean(np.abs(observed - predicted).mean(axis=1), weight),
        "obesity_weighted_mae_pp": 100 * weighted_mean(np.abs(oo - po), weight),
        "within_anchor_obesity_deviation_r": weighted_corr(oo - bo, po - bo, weight),
        "prediction_closure_max_error": float(np.abs(predicted.sum(axis=1) - 1).max()),
    }


def main() -> None:
    if (OUT / "source_holdout_report_v1.md").exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    raw, paired_qa = load_predictions()
    cells = aggregate(raw)
    primary = cells.loc[cells["pseudo_n"].ge(PRIMARY_N)].copy()
    rows = []
    for (task, source, model), frame in primary.groupby(["task", "source_family", "model"], sort=False):
        row = {"task": task, "held_out_source": source, "model": model}
        row.update(metrics(frame))
        rows.append(row)
    result = pd.DataFrame(rows)
    comparisons = []
    for (task, source), frame in result.groupby(["task", "held_out_source"], sort=False):
        values = frame.set_index("model")
        new, old = values.loc["full_covariate_mrp"], values.loc["original_full_mrp"]
        comparisons.append({
            "task": task, "held_out_source": source,
            "delta_weighted_r2_new_minus_old": float(new["weighted_r2"] - old["weighted_r2"]),
            "delta_w2_new_minus_old_kg_m2": float(new["weighted_w2_bmi_kg_m2"] - old["weighted_w2_bmi_kg_m2"]),
            "delta_obesity_mae_new_minus_old_pp": float(new["obesity_weighted_mae_pp"] - old["obesity_weighted_mae_pp"]),
            "delta_seven_category_mae_new_minus_old_pp": float(new["seven_category_weighted_mae_pp"] - old["seven_category_weighted_mae_pp"]),
        })
    comparison = pd.DataFrame(comparisons)
    pair_pass = all(
        item["paired_row_ids"] and item["observed_equal"] and item["anchor_equal"] and len(item["held_out_source"]) == 1
        for item in paired_qa.values()
    )
    closure_pass = bool(result["prediction_closure_max_error"].max() <= 1e-10)
    finite_pass = bool(np.isfinite(result[["weighted_w2_bmi_kg_m2", "seven_category_weighted_mae_pp", "obesity_weighted_mae_pp"]]).all().all())
    status = "PASS_WITH_WARN" if pair_pass and closure_pass and finite_pass else "FAIL"
    result_path = OUT / "source_holdout_stabilized_metrics_v1.csv"
    comparison_path = OUT / "source_holdout_paired_comparison_v1.csv"
    cells_path = OUT / "source_holdout_stabilized_cells_v1.csv.gz"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    cells.to_csv(cells_path, index=False, compression="gzip")
    qa = {
        "status": status,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_minimum_effective_n": PRIMARY_N,
        "paired_QA": paired_qa,
        "paired_QA_pass": pair_pass,
        "closure_pass": closure_pass,
        "finite_metrics_pass": finite_pass,
        "post_hoc_performance_threshold_applied": False,
        "interpretation_warning": "Source holdout is a transport stress test; heterogeneous source-specific performance is reported without an invented pass threshold.",
    }
    qa_path = OUT / "source_holdout_QA_v1.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    display = result[["held_out_source", "model", "validation_cells", "countries", "weighted_r2", "weighted_w2_bmi_kg_m2", "obesity_weighted_mae_pp", "seven_category_weighted_mae_pp"]]
    report = "# Full-covariate MRP source-family holdout\n\n"
    report += f"**QA status: {status}**\n\n"
    report += "Six source families were removed one at a time, with the model refitted before predicting the held-out source. Results use the frozen survey-geography-year-sex-broad-age aggregation and `pseudo_n >= 20`.\n\n"
    report += dataframe_to_markdown(display) + "\n\n"
    report += "No post hoc pass threshold was imposed. The test is retained as a source-transport diagnostic; all paired rows, observed compositions, national anchors, finite values, and composition closure passed.\n"
    report_path = OUT / "source_holdout_report_v1.md"
    report_path.write_text(report, encoding="utf-8")
    artifacts = [Path(__file__), result_path, comparison_path, cells_path, qa_path, report_path]
    pd.DataFrame([{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts]).to_csv(
        OUT / "source_holdout_artifact_sha256_v1.csv", index=False, encoding="utf-8-sig"
    )
    print(display.to_string(index=False))
    print(f"SOURCE_HOLDOUT_QA={status}")


if __name__ == "__main__":
    main()
