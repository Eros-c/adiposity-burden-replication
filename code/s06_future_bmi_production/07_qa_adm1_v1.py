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


ROOT = Path(_cfg.RESULTS / "06_future_bmi_2025_2050" / "phase7_multiscale_delivery_v3_20260908")
TABLES = ROOT / "04_adm1" / "tables"
QA = ROOT / "04_adm1" / "qa"
SCENARIOS = ("ssp126", "ssp245", "ssp370", "ssp585")
YEARS = range(2025, 2051)
FAMILIES = {
    "overall": (4596, 1),
    "sex": (9192, 2),
    "age": (68940, 15),
    "age_sex": (137880, 30),
}
BMI = ("bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30", "bmi_30_35", "bmi_35_40", "bmi_ge_40")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    rows = []
    checksum_rows = []
    failures = []
    for scenario in SCENARIOS:
        for year in YEARS:
            for family, (expected_rows, strata_count) in FAMILIES.items():
                path = TABLES / scenario / str(year) / f"BMI7_{scenario}_{year}_ADM1_{family}_adults25plus_v2.parquet"
                checks = {"file_exists": path.exists(), "row_count": False, "population_nonnegative": False, "probability_range": False, "composition_closure": False, "derived_metrics": False}
                observed_rows = 0
                max_closure = np.nan
                max_metric_error = np.nan
                total_population = np.nan
                if path.exists():
                    df = pd.read_parquet(path)
                    observed_rows = len(df)
                    checks["row_count"] = observed_rows == expected_rows
                    population = pd.to_numeric(df["adult_population"], errors="coerce").to_numpy(dtype=float)
                    positive = np.isfinite(population) & (population > 0)
                    checks["population_nonnegative"] = bool(np.all(np.isfinite(population)) and np.all(population >= 0))
                    probability = df.loc[:, BMI].to_numpy(dtype=float)
                    valid = positive & np.all(np.isfinite(probability), axis=1)
                    checks["probability_range"] = bool(np.all((probability[valid] >= -1e-7) & (probability[valid] <= 1 + 1e-7)))
                    if np.any(valid):
                        max_closure = float(np.max(np.abs(probability[valid].sum(axis=1) - 1.0)))
                        checks["composition_closure"] = max_closure <= 1e-6
                    else:
                        checks["composition_closure"] = False
                    obesity = probability[:, 4:7].sum(axis=1)
                    overweight_obesity = probability[:, 3:7].sum(axis=1)
                    midpoint = np.array([17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5])
                    mean_bmi = probability @ midpoint
                    metric_error = np.nanmax(np.abs(np.column_stack([
                        obesity - df["obesity"].to_numpy(dtype=float),
                        overweight_obesity - df["overweight_obesity"].to_numpy(dtype=float),
                        mean_bmi - df["mean_bmi"].to_numpy(dtype=float),
                    ])[valid])) if np.any(valid) else np.nan
                    max_metric_error = float(metric_error)
                    checks["derived_metrics"] = bool(np.isfinite(metric_error) and metric_error <= 1e-6)
                    total_population = float(np.nansum(population))
                passed = all(checks.values())
                row = {
                    "scenario_code": scenario,
                    "year": year,
                    "family": family,
                    "path": str(path),
                    "expected_rows": expected_rows,
                    "observed_rows": observed_rows,
                    "strata_count": strata_count,
                    "total_adult_population": total_population,
                    "max_composition_closure_error": max_closure,
                    "max_derived_metric_error": max_metric_error,
                    **checks,
                    "pass": passed,
                }
                rows.append(row)
                if not passed:
                    failures.append(row)
                if path.exists():
                    checksum_rows.append({"scenario_code": scenario, "year": year, "family": family, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    QA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(QA / "adm1_full_qa_v1.csv", index=False)
    pd.DataFrame(checksum_rows).to_csv(QA / "adm1_parquet_SHA256_v1.csv", index=False)
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_tasks": 104,
        "expected_tables": 416,
        "observed_tables": len(checksum_rows),
        "passed_tables": sum(bool(r["pass"]) for r in rows),
        "failed_tables": len(failures),
        "maximum_composition_closure_error": float(np.nanmax(pd.DataFrame(rows)["max_composition_closure_error"])),
        "maximum_derived_metric_error": float(np.nanmax(pd.DataFrame(rows)["max_derived_metric_error"])),
        "population_conservation_qa_files": 104,
        "population_mask_policy": "mask applies to maps only; ADM1 tables retain complete adult population denominator",
    }
    (QA / "adm1_full_qa_summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
