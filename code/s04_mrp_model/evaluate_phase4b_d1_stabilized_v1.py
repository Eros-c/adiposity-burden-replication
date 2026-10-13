"""Aggregate and evaluate D1 reduced-demography MRP validation predictions."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    _cfg.RESULTS / "04_future_capable_mrp" / "phase4b_d1_reduced_demography_v1_20260831"
)
INPUT = ROOT / "model_run_v1" / "validation" / "all_validation_predictions_v1.csv.gz"
OUT = ROOT / "stabilized_evaluation_d1_v1"
TABLES = OUT / "tables"
QA = OUT / "qa"
BMI = ["bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30", "bmi_30_35", "bmi_35_40", "bmi_ge_40"]
DISPLAY = ["<18.5", "18.5–<20", "20–<25", "25–<30", "30–<35", "35–<40", "≥40"]
SUPPORT = np.array([17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5])
SCHEME_LABELS = {
    "all_eligible20_internal_random_row_5fold": "Internal random-row fivefold",
    "all_eligible20_country_grouped_5fold": "Country-held-out fivefold",
    "all_eligible20_source_family_holdout": "Source-family-held-out",
}
THRESHOLDS = (0, 10, 20, 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def r2(observed: np.ndarray, predicted: np.ndarray, weight: np.ndarray | None = None) -> float:
    weight = np.ones(len(observed)) if weight is None else np.asarray(weight, float)
    keep = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(weight) & (weight > 0)
    observed, predicted, weight = observed[keep], predicted[keep], weight[keep]
    center = np.average(observed, weights=weight)
    denominator = np.sum(weight * (observed - center) ** 2)
    return math.nan if denominator <= 0 else float(1 - np.sum(weight * (observed - predicted) ** 2) / denominator)


def close_composition(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), 0, None)
    return x / x.sum(axis=1, keepdims=True)


def row_w2(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed, predicted = close_composition(observed), close_composition(predicted)
    output = np.empty(len(observed), float)
    for row in range(len(observed)):
        boundaries = np.unique(np.concatenate(([0.0, 1.0], np.cumsum(observed[row]), np.cumsum(predicted[row]))))
        middle = (boundaries[:-1] + boundaries[1:]) / 2
        width = np.diff(boundaries)
        oi = np.searchsorted(np.cumsum(observed[row]), middle, side="right")
        pi = np.searchsorted(np.cumsum(predicted[row]), middle, side="right")
        oi, pi = np.clip(oi, 0, 6), np.clip(pi, 0, 6)
        output[row] = np.sqrt(np.sum(width * (SUPPORT[oi] - SUPPORT[pi]) ** 2))
    return output


def broad_age(age_group: pd.Series) -> pd.Series:
    first = pd.to_numeric(age_group.astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
    second = pd.to_numeric(age_group.astype(str).str.extract(r"-(\d+)")[0], errors="coerce")
    midpoint = ((first + second) / 2).where(second.notna(), first + 5)
    result = pd.cut(midpoint, [19.999, 39.999, 59.999, np.inf], labels=["20–39", "40–59", "60+"], right=True)
    if result.isna().any():
        raise ValueError("Unmapped age groups")
    return result.astype(str)


def aggregate(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["validation"] = data["scheme"].map(SCHEME_LABELS)
    if data["validation"].isna().any():
        raise ValueError("Unexpected validation scheme")
    data["model"] = "D1 reduced-demography MRP"
    data["source_role_original"] = data["source_role"]
    data.loc[data["source_role"].eq("primary_measured_subnational_training"), "source_role"] = "primary"
    data["broad_age"] = broad_age(data["age_group"])
    observed = [f"observed_{label}" for label in BMI]
    predicted = [f"predicted_{label}" for label in BMI]
    baseline = [f"baseline_{label}" for label in BMI]
    values = [*observed, *predicted, *baseline]
    groups = ["model", "validation", "source_family", "source_role", "survey_id", "iso3", "geography_id", "year", "sex", "broad_age"]
    work = data[groups + ["pseudo_n", *values]].copy()
    weight = work["pseudo_n"].to_numpy(float)
    for column in values:
        work[column] = pd.to_numeric(work[column], errors="raise") * weight
    work["underlying_rows"] = 1
    result = work.groupby(groups, observed=True, dropna=False, sort=False).sum(numeric_only=True).reset_index()
    for column in values:
        result[column] /= result["pseudo_n"]
    obs = close_composition(result[observed].to_numpy(float))
    pred = close_composition(result[predicted].to_numpy(float))
    base = close_composition(result[baseline].to_numpy(float))
    result[observed], result[predicted], result[baseline] = obs, pred, base
    result["observed_obesity"] = obs[:, 4:].sum(axis=1)
    result["predicted_obesity"] = pred[:, 4:].sum(axis=1)
    result["baseline_obesity"] = base[:, 4:].sum(axis=1)
    result["w2_bmi_kg_m2"] = row_w2(obs, pred)
    result["validation_cell_id"] = result[["survey_id", "geography_id", "year", "sex", "broad_age"]].astype(str).agg("|".join, axis=1)
    return result


def metrics(frame: pd.DataFrame, threshold: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    frame = frame.loc[frame["pseudo_n"].ge(threshold if threshold else 0)].copy()
    obs = frame[[f"observed_{x}" for x in BMI]].to_numpy(float)
    pred = frame[[f"predicted_{x}" for x in BMI]].to_numpy(float)
    base = frame[[f"baseline_{x}" for x in BMI]].to_numpy(float)
    w = frame["pseudo_n"].to_numpy(float)
    oo, po, bo = obs[:, 4:].sum(axis=1), pred[:, 4:].sum(axis=1), base[:, 4:].sum(axis=1)
    by_source = []
    for _, group in frame.groupby("source_family", observed=True):
        go = group[[f"observed_{x}" for x in BMI]].to_numpy(float)
        gp = group[[f"predicted_{x}" for x in BMI]].to_numpy(float)
        gw = group["pseudo_n"].to_numpy(float)
        by_source.append((weighted_mean(row_w2(go, gp), gw), 100 * weighted_mean(np.abs(go - gp).mean(axis=1), gw), 100 * weighted_mean(np.abs(go[:, 4:].sum(axis=1) - gp[:, 4:].sum(axis=1)), gw)))
    sf = np.asarray(by_source)
    record = {
        "model": "D1 reduced-demography MRP", "validation": frame["validation"].iat[0], "minimum_effective_n": threshold,
        "validation_cells": len(frame), "underlying_rows": int(frame["underlying_rows"].sum()),
        "effective_n": float(w.sum()), "countries": frame["iso3"].nunique(), "sources": frame["source_family"].nunique(),
        "weighted_r": weighted_corr(oo, po, w), "weighted_r2": r2(oo, po, w), "ordinary_r2": r2(oo, po),
        "weighted_w2_bmi_kg_m2": weighted_mean(row_w2(obs, pred), w), "source_balanced_w2_bmi_kg_m2": float(sf[:, 0].mean()),
        "seven_category_weighted_mae_pp": 100 * weighted_mean(np.abs(obs - pred).mean(axis=1), w),
        "source_balanced_seven_category_mae_pp": float(sf[:, 1].mean()),
        "obesity_weighted_mae_pp": 100 * weighted_mean(np.abs(oo - po), w),
        "source_balanced_obesity_mae_pp": float(sf[:, 2].mean()),
        "within_anchor_obesity_deviation_r": weighted_corr(oo - bo, po - bo, w),
    }
    categories = [{"validation": record["validation"], "minimum_effective_n": threshold, "bmi_category": label,
                   "bmi_category_display": display, "weighted_r": weighted_corr(obs[:, i], pred[:, i], w),
                   "weighted_r2": r2(obs[:, i], pred[:, i], w), "weighted_mae_pp": 100 * weighted_mean(np.abs(obs[:, i] - pred[:, i]), w)}
                  for i, (label, display) in enumerate(zip(BMI, DISPLAY))]
    return record, categories


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    TABLES.mkdir(parents=True)
    QA.mkdir()
    raw = pd.read_csv(INPUT, low_memory=False)
    aggregated = aggregate(raw)
    if aggregated.duplicated(["validation", "validation_cell_id"]).any():
        raise RuntimeError("Duplicate aggregated validation cells")
    metric_rows, category_rows, coverage_rows = [], [], []
    for validation, frame in aggregated.groupby("validation", observed=True):
        for threshold in THRESHOLDS:
            record, categories = metrics(frame, threshold)
            metric_rows.append(record)
            category_rows.extend(categories)
            selected = frame.loc[frame["pseudo_n"].ge(threshold if threshold else 0)]
            coverage_rows.append({"validation": validation, "minimum_effective_n": threshold, "validation_cells": len(selected),
                                  "underlying_rows": int(selected["underlying_rows"].sum()), "effective_n": float(selected["pseudo_n"].sum()),
                                  "countries": selected["iso3"].nunique(), "sources": selected["source_family"].nunique()})
    cells_path = TABLES / "all_eligible20_stabilized_cells_v1.csv.gz"
    metrics_path = TABLES / "all_eligible20_stabilized_metrics_v1.csv"
    categories_path = TABLES / "all_eligible20_stabilized_category_metrics_v1.csv"
    coverage_path = TABLES / "all_eligible20_stabilized_coverage_v1.csv"
    aggregated.to_csv(cells_path, index=False, compression="gzip")
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(category_rows).to_csv(categories_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False, encoding="utf-8-sig")
    primary = pd.DataFrame(metric_rows).query("minimum_effective_n == 20")
    expected = sorted(["AFG", "BGD", "BRA", "DZA", "EGY", "ETH", "GBR", "GHA", "IND", "JOR", "KEN", "KOR", "MDA", "MEX", "NGA", "NPL", "PAK", "UGA", "UKR", "ZAF"])
    qa = {
        "status": "passed", "input_sha256": sha256(INPUT), "raw_rows_per_scheme": raw.groupby("scheme").size().to_dict(),
        "aggregated_rows_per_scheme": aggregated.groupby("validation").size().to_dict(),
        "primary_threshold": 20, "primary_metrics": primary.to_dict("records"),
        "validation_country_codes": {v: sorted(g["iso3"].unique()) for v, g in aggregated.groupby("validation")},
        "expected_country_codes": expected, "china_japan_in_validation": bool(aggregated["iso3"].isin(["CHN", "JPN"]).any()),
        "regional_steps_in_validation": bool(aggregated["source_family"].eq("WHO STEPS regional").any()),
        "prediction_closure_max_error": float(np.abs(aggregated[[f"predicted_{x}" for x in BMI]].sum(axis=1) - 1).max()),
    }
    if any(codes != expected for codes in qa["validation_country_codes"].values()) or qa["china_japan_in_validation"] or not qa["regional_steps_in_validation"] or qa["prediction_closure_max_error"] > 1e-10:
        qa["status"] = "failed"
        raise RuntimeError(json.dumps(qa, ensure_ascii=False))
    qa_path = QA / "all_eligible20_stabilized_qa_v1.json"
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = [Path(__file__), INPUT, cells_path, metrics_path, categories_path, coverage_path, qa_path]
    pd.DataFrame([{"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in artifacts]).to_csv(OUT / "all_eligible20_stabilized_sha256_v1.csv", index=False, encoding="utf-8-sig")
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
