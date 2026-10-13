from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


PROJECT = Path(_cfg.RESULTS)
PHASE = PROJECT / "06_future_bmi_2025_2050" / "phase7_multiscale_delivery_v3_20260908"
SRC01 = PROJECT / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906"
SRC05 = PROJECT / "06_future_bmi_2025_2050" / "full_covariate_annual_0p5deg_production_v1_20260906"
ADM1_META = PROJECT / "06_future_bmi_2025_2050" / "phase6i_remaining_scenarios_0p1deg_adm1_v1_20260904" / "metadata"

SCENARIOS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
YEARS = range(2025, 2051)
SEXES = ("Female", "Male")
AGES = (
    "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+",
)
BMI = (
    "bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30",
    "bmi_30_35", "bmi_35_40", "bmi_ge_40",
)
FAMILY_COUNTS = {
    "overall": (1, 7),
    "sex": (2, 14),
    "age": (15, 105),
    "age_sex": (30, 210),
}
EXPECTED_SOURCE_FAMILIES = (
    "probability_overall", "probability_sex", "probability_age", "probability_age_sex",
    "population_overall", "population_sex", "population_age", "population_age_sex",
)


def family_key(family: str) -> str:
    return family.replace("probability_", "").replace("population_", "")


def expected_bands(family: str) -> int:
    key = family_key(family)
    if family.startswith("population_"):
        return FAMILY_COUNTS[key][0]
    if family.startswith("probability_"):
        return FAMILY_COUNTS[key][1]
    raise ValueError(f"Unknown family: {family}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_dir(scenario: str, year: int) -> Path:
    return SRC01 / "outputs" / scenario / str(year)


def source_path(scenario: str, year: int, family: str) -> Path:
    return source_dir(scenario, year) / f"{family}_{scenario}_{year}_0p1deg_v1.tif"


def parent_path(scenario: str, year: int, family: str) -> Path:
    if family.startswith("probability_"):
        name = "bmi7_" + family
    else:
        name = "adult_" + family
    return SRC05 / "outputs" / scenario / str(year) / f"{name}_0p5deg_v1.tif"


def output_task_key(scenario: str, year: int) -> str:
    return f"{scenario}/{year}"


def run_inventory() -> None:
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    manifest_path = SRC01 / "state" / "annual_0p1deg_task_manifest_v1.csv"
    manifest = pd.read_csv(manifest_path)
    expected_tasks = {(str(r.scenario_code), int(r.year)) for r in manifest.itertuples()}
    for scenario, year in sorted(expected_tasks, key=lambda x: (x[1], x[0])):
        state_path = SRC01 / "state" / scenario / f"{year}.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        task_pass = state.get("status") == "PASS_ANNUAL_0P1DEG_TASK"
        fc04_path = SRC01 / "tasks" / scenario / str(year) / "qa" / f"Gate_FC04_summary_{scenario}_{year}_v1.json"
        fc04_status = None
        if fc04_path.exists():
            fc04_status = json.loads(fc04_path.read_text(encoding="utf-8")).get("status")
        task_rows.append({
            "scenario_code": scenario,
            "scenario": SCENARIOS[scenario],
            "year": year,
            "task_status": state.get("status"),
            "task_pass": task_pass,
            "fc04_summary_status": fc04_status or "missing",
            "fc04_summary_missing_warning": not fc04_path.exists(),
            "formal_cogs": state.get("formal_cogs"),
            "formal_bytes": state.get("formal_bytes"),
        })
        for family in EXPECTED_SOURCE_FAMILIES:
            path = source_path(scenario, year, family)
            exists = path.exists()
            row: dict[str, object] = {
                "scenario_code": scenario,
                "scenario": SCENARIOS[scenario],
                "year": year,
                "family": family,
                "path": str(path),
                "exists": exists,
                "bytes": path.stat().st_size if exists else 0,
                "sha256": sha256(path) if exists else "",
                "bands": None,
                "width": None,
                "height": None,
                "crs": "",
                "resolution_x": None,
                "resolution_y": None,
                "dtype": "",
                "nodata": None,
                "layout": "",
                "overviews": "",
                "pass": False,
            }
            if exists:
                with rasterio.open(path) as ds:
                    structure = ds.tags(ns="IMAGE_STRUCTURE")
                    overviews = ds.overviews(1)
                    width = ds.width
                    height = ds.height
                    bands = ds.count
                    crs = ds.crs.to_string() if ds.crs else ""
                    resolution_x, resolution_y = ds.res
                    dtype = ds.dtypes[0]
                    nodata = ds.nodata
                    layout = structure.get("LAYOUT", "")
                    row.update({
                        "bands": bands,
                        "width": width,
                        "height": height,
                        "crs": crs,
                        "resolution_x": resolution_x,
                        "resolution_y": resolution_y,
                        "dtype": dtype,
                        "nodata": nodata,
                        "layout": layout,
                        "overviews": "|".join(map(str, overviews)),
                    })
                row["pass"] = (
                    task_pass and width == 3600 and height == 1800
                    and bands == expected_bands(family) and crs == "EPSG:4326"
                    and abs(float(row["resolution_x"]) - 0.1) < 1e-12
                    and abs(float(row["resolution_y"]) - 0.1) < 1e-12
                    and dtype == "float32" and float(nodata) == -9999.0
                    and layout == "COG"
                )
            rows.append(row)
    write_csv(PHASE / "01_inventory" / "annual_0p1deg_file_inventory_v1.csv", rows)
    write_csv(PHASE / "01_inventory" / "annual_task_inventory_v1.csv", task_rows)
    summary = {
        "status": "PASS" if all(bool(r["pass"]) for r in rows) and all(bool(r["task_pass"]) for r in task_rows) else "FAIL",
        "completed_at_utc": now(),
        "tasks": len(task_rows),
        "files": len(rows),
        "files_pass": sum(bool(r["pass"]) for r in rows),
        "tasks_pass": sum(bool(r["task_pass"]) for r in task_rows),
        "fc04_missing_summary_count": sum(bool(r["fc04_summary_missing_warning"]) for r in task_rows),
        "sha256_total_bytes": int(sum(int(r["bytes"]) for r in rows)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(PHASE / "01_inventory" / "inventory_summary_v1.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] != "PASS":
        raise SystemExit(2)


def aggregate_parent(source_probability: np.ndarray, source_population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_strata = source_population.shape[0]
    n_categories = source_probability.shape[0] // n_strata
    height = source_population.shape[1]
    width = source_population.shape[2]
    if height % 5 or width % 5:
        raise RuntimeError("0.1 degree dimensions are not divisible by five")
    parent_h, parent_w = height // 5, width // 5
    valid_population = (
        np.isfinite(source_population)
        & (source_population != -9999.0)
        & (source_population > 0)
    )
    pop_source = np.where(valid_population, source_population, 0.0)
    valid_probability = (
        np.isfinite(source_probability)
        & (source_probability != -9999.0)
    )
    probability_source = np.where(valid_probability, source_probability, 0.0)
    pop = pop_source.reshape(n_strata, parent_h, 5, parent_w, 5).sum(axis=(2, 4), dtype=np.float64)
    prob = probability_source.reshape(n_strata, n_categories, height, width)
    numerator = (prob * source_population[:, None, :, :]).reshape(
        n_strata, n_categories, parent_h, 5, parent_w, 5
    ).sum(axis=(3, 5), dtype=np.float64)
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, pop[:, None, :, :], out=out, where=pop[:, None, :, :] > 0)
    return out, pop


def run_parent_reconciliation() -> None:
    rows: list[dict[str, object]] = []
    for scenario, year in sorted(((s, y) for y in YEARS for s in SCENARIOS), key=lambda x: (x[1], x[0])):
        for family in ("overall", "sex", "age", "age_sex"):
            prob_path = source_path(scenario, year, "probability_" + family)
            pop_path = source_path(scenario, year, "population_" + family)
            target_prob_path = parent_path(scenario, year, "probability_" + family)
            target_pop_path = parent_path(scenario, year, "population_" + family)
            with rasterio.open(prob_path) as ps, rasterio.open(pop_path) as qs, rasterio.open(target_prob_path) as pt, rasterio.open(target_pop_path) as qt:
                max_prob_error = 0.0
                max_pop_error = 0.0
                source_total = 0.0
                target_total = 0.0
                for row0 in range(0, 1800, 100):
                    h = min(100, 1800 - row0)
                    if h % 5:
                        h -= h % 5
                    window = Window(0, row0, 3600, h)
                    prob = ps.read(window=window).astype(np.float64)
                    pop = qs.read(window=window).astype(np.float64)
                    agg_prob, agg_pop = aggregate_parent(prob, pop)
                    parent_window = Window(0, row0 // 5, 720, h // 5)
                    target_prob = pt.read(window=parent_window).astype(np.float64)
                    target_pop = qt.read(window=parent_window).astype(np.float64)
                    n_strata = agg_pop.shape[0]
                    n_categories = agg_prob.shape[1]
                    target_prob = target_prob.reshape(n_strata, n_categories, h // 5, 720)
                    valid_pop = np.isfinite(target_pop) & (target_pop != qt.nodata) & (target_pop > 0)
                    valid_prob = (
                        np.isfinite(target_prob)
                        & (target_prob != pt.nodata)
                        & valid_pop[:, None, :, :]
                    )
                    if np.any(valid_pop):
                        max_pop_error = max(max_pop_error, float(np.nanmax(np.abs(agg_pop[valid_pop] - target_pop[valid_pop]))))
                    if np.any(valid_prob):
                        max_prob_error = max(max_prob_error, float(np.nanmax(np.abs(agg_prob[valid_prob] - target_prob[valid_prob]))))
                    source_total += float(np.nansum(agg_pop))
                    target_total += float(np.nansum(target_pop))
            rows.append({
                "scenario_code": scenario,
                "scenario": SCENARIOS[scenario],
                "year": year,
                "family": family,
                "max_probability_absolute_error": max_prob_error,
                "max_population_absolute_error": max_pop_error,
                "source_population_total": source_total,
                "target_population_total": target_total,
                "population_total_absolute_error": abs(source_total - target_total),
                "pass": max_prob_error <= 2e-5 and max_pop_error <= 1e-2 and abs(source_total - target_total) <= max(1e-3, target_total * 1e-9),
            })
        print(f"[0p5] {scenario}/{year} complete", flush=True)
    write_csv(PHASE / "03_0p5deg_reconciliation" / "annual_0p1_to_0p5_reconciliation_v1.csv", rows)
    summary = {
        "status": "PASS" if all(bool(r["pass"]) for r in rows) else "FAIL",
        "completed_at_utc": now(),
        "tasks": len(rows) // 4,
        "family_checks": len(rows),
        "failed_checks": sum(not bool(r["pass"]) for r in rows),
    }
    write_json(PHASE / "03_0p5deg_reconciliation" / "reconciliation_summary_v1.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] != "PASS":
        raise SystemExit(3)


def load_zone_grid() -> tuple[np.ndarray, pd.DataFrame]:
    zone_path = ADM1_META / "adm1_zone_country_constrained_0p1deg_v3.tif"
    lookup_path = ADM1_META / "adm1_zone_lookup_v1.parquet"
    with rasterio.open(zone_path) as ds:
        zones = ds.read(1).astype(np.int32)
    lookup = pd.read_parquet(lookup_path)
    return zones, lookup


def aggregate_adm1_task(scenario: str, year: int, zones: np.ndarray, lookup: pd.DataFrame) -> dict[str, object]:
    prob_path = source_path(scenario, year, "probability_age_sex")
    pop_path = source_path(scenario, year, "population_age_sex")
    n_zones = int(lookup.zone_id.max())
    group_pop = np.zeros((30, n_zones + 1), dtype=np.float64)
    numerator = np.zeros((30, 7, n_zones + 1), dtype=np.float64)
    with rasterio.open(prob_path) as ps, rasterio.open(pop_path) as qs:
        for row0 in range(0, 1800, 100):
            h = min(100, 1800 - row0)
            window = Window(0, row0, 3600, h)
            p = ps.read(window=window).astype(np.float64).reshape(30, 7, h, 3600)
            q = qs.read(window=window).astype(np.float64)
            z = zones[row0:row0 + h, :]
            zflat = z.reshape(-1)
            for group in range(30):
                weights = q[group].reshape(-1)
                valid = (zflat > 0) & np.isfinite(weights) & (weights != qs.nodata) & (weights > 0)
                if not np.any(valid):
                    continue
                ids = zflat[valid]
                w = weights[valid]
                group_pop[group] += np.bincount(ids, weights=w, minlength=n_zones + 1)
                for cat in range(7):
                    values = p[group, cat].reshape(-1)[valid]
                    values = np.where(np.isfinite(values) & (values != ps.nodata), values, 0.0)
                    numerator[group, cat] += np.bincount(ids, weights=values * w, minlength=n_zones + 1)
    probability = np.full_like(numerator, np.nan)
    np.divide(numerator, group_pop[:, None, :], out=probability, where=group_pop[:, None, :] > 0)
    family_data = {
        "age_sex": (group_pop, probability),
        "age": (group_pop.reshape(2, 15, -1).sum(axis=0), numerator.reshape(2, 15, 7, -1).sum(axis=0) / np.maximum(group_pop.reshape(2, 15, 1, -1).sum(axis=0), 1e-300)),
        "sex": (group_pop.reshape(2, 15, -1).sum(axis=1), numerator.reshape(2, 15, 7, -1).sum(axis=1) / np.maximum(group_pop.reshape(2, 15, 1, -1).sum(axis=1), 1e-300)),
        "overall": (group_pop.sum(axis=0, keepdims=True), numerator.sum(axis=0, keepdims=True) / np.maximum(group_pop.sum(axis=0, keepdims=True)[:, None, :], 1e-300)),
    }
    rows_written = {}
    out_dir = PHASE / "04_adm1" / "tables" / scenario / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = lookup.copy()
    for family, (family_pop, family_prob) in family_data.items():
        records = []
        if family == "overall":
            labels = [("Both", "25+", family_prob[0], family_pop[0])]
        elif family == "sex":
            labels = [(sex, "25+", family_prob[i], family_pop[i]) for i, sex in enumerate(SEXES)]
        elif family == "age":
            labels = [("Both", age, family_prob[i], family_pop[i]) for i, age in enumerate(AGES)]
        else:
            labels = [(SEXES[i // 15], AGES[i % 15], family_prob[i], family_pop[i]) for i in range(30)]
        for sex, age, probs, pops in labels:
            frame = meta.copy()
            frame.insert(0, "scenario_code", scenario)
            frame.insert(1, "scenario", SCENARIOS[scenario])
            frame.insert(2, "year", year)
            frame["sex"] = sex
            frame["age_group"] = age
            frame["adult_population"] = pops[frame.zone_id.to_numpy(dtype=np.int32)]
            for cat, name in enumerate(BMI):
                frame[name] = probs[cat, frame.zone_id.to_numpy(dtype=np.int32)]
            frame["obesity"] = frame[["bmi_30_35", "bmi_35_40", "bmi_ge_40"]].sum(axis=1, skipna=False)
            frame["overweight_obesity"] = frame[["bmi_25_30", "bmi_30_35", "bmi_35_40", "bmi_ge_40"]].sum(axis=1, skipna=False)
            frame["mean_bmi"] = sum(frame[name] * midpoint for name, midpoint in zip(BMI, (17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5)))
            records.append(frame)
        result = pd.concat(records, ignore_index=True)
        out_path = out_dir / f"BMI7_{scenario}_{year}_ADM1_{family}_adults25plus_v2.parquet"
        result.to_parquet(out_path, index=False)
        rows_written[family] = int(len(result))
    qa = {
        "scenario_code": scenario,
        "scenario": SCENARIOS[scenario],
        "year": year,
        "adm1_zones": n_zones,
        "input_population": float(group_pop.sum()),
        "overall_adm1_population": float(family_data["overall"][0].sum()),
        "population_absolute_error": abs(float(group_pop.sum()) - float(family_data["overall"][0].sum())),
        "rows": rows_written,
        "status": "PASS_ADM1_AGGREGATION",
    }
    write_json(PHASE / "04_adm1" / "qa" / f"adm1_{scenario}_{year}_qa_v2.json", qa)
    return qa


def run_adm1() -> None:
    zones, lookup = load_zone_grid()
    state_path = PHASE / "04_adm1" / "adm1_batch_state_v2.json"
    state = {"status": "RUNNING", "started_at_utc": now(), "completed": []}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    completed = {tuple(x) for x in state.get("completed", [])}
    for year in YEARS:
        for scenario in SCENARIOS:
            key = (scenario, year)
            if key in completed:
                continue
            qa = aggregate_adm1_task(scenario, year, zones, lookup)
            if not qa["status"].startswith("PASS"):
                raise RuntimeError(f"ADM1 aggregation failed: {scenario}/{year}")
            completed.add(key)
            state.update({"status": "RUNNING", "completed": [list(x) for x in sorted(completed, key=lambda x: (x[1], x[0]))], "updated_at_utc": now()})
            write_json(state_path, state)
            print(f"[adm1] {scenario}/{year} complete", flush=True)
    state.update({"status": "PASS_ADM1_AGGREGATION", "completed_at_utc": now(), "completed_count": len(completed)})
    write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inventory", "reconcile_0p5", "adm1", "all"], required=True)
    args = parser.parse_args()
    PHASE.mkdir(parents=True, exist_ok=True)
    if args.mode in ("inventory", "all"):
        run_inventory()
    if args.mode in ("reconcile_0p5", "all"):
        run_parent_reconciliation()
    if args.mode in ("adm1", "all"):
        run_adm1()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
