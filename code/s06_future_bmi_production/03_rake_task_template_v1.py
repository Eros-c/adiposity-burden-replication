"""Rake full-MRP 0.1-degree predictions to formal 0.5-degree targets."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy


ROOT = Path(_cfg.ROOT / ".codex_tmp" / "fullcov_0p1_pilot_v1")
PRODUCTION = Path(
    _cfg.RESULTS / "06_future_bmi_2025_2050" / "full_covariate_annual_0p5deg_production_v1_20260906"
)
SCENARIO = "SSP5-8.5"
SCENARIO_CODE = "ssp585"
YEAR = 2050
WIDTH = 3600
HEIGHT = 1800
NCELL = WIDTH * HEIGHT
NODATA = -9999.0
EPSILON = 1e-12
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
    "probability_overall": 7,
    "probability_sex": 14,
    "probability_age": 105,
    "probability_age_sex": 210,
    "population_overall": 1,
    "population_sex": 2,
    "population_age": 15,
    "population_age_sex": 30,
}
COG_NUM_THREADS = "ALL_CPUS"
AGE_SEX_PROBABILITY_FAMILY = "probability_age_sex"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv_new(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_binary(path: Path, value: np.ndarray, dtype: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    np.asarray(value, dtype=dtype).tofile(temporary)
    os.replace(temporary, path)


def group_paths(sex: str, age_index: int) -> tuple[Path, Path, Path]:
    base = ROOT / "work" / "binary_strict_density_v2" / SCENARIO_CODE / str(YEAR)
    code = sex.lower()
    return (
        base / f"probability_unraked_{code}_a{age_index:02d}_f32le_v1.bin",
        base / f"probability_raked_{code}_a{age_index:02d}_f32le_v1.bin",
        base / f"population_{code}_a{age_index:02d}_f64le_v1.bin",
    )


def load_state() -> dict:
    path = ROOT / "qa" / "Gate_FC02_summary_ssp585_2050_v2.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") != "PASS_GATE_FC02_UNRAKED_FULL_MRP":
        raise RuntimeError(f"Gate FC02 v2 is not PASS: {path}")
    return state


def load_geometry(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active_one = np.fromfile(state["active_cell_id_path"], dtype="<i4")
    parent_one = np.fromfile(state["active_parent_cell_id_path"], dtype="<i4")
    country_id = np.fromfile(state["active_country_id_path"], dtype="<i4")
    expected = int(state["active_child_cells"])
    if any(value.size != expected for value in (active_one, parent_one, country_id)):
        raise RuntimeError("Active-cell metadata length mismatch")
    if np.any(active_one < 1) or np.any(active_one > NCELL):
        raise RuntimeError("Invalid 0.1-degree cell identifiers")
    parent_levels = np.unique(parent_one)
    parent_index = np.searchsorted(parent_levels, parent_one)
    return active_one - 1, parent_levels, parent_index, country_id


def load_parent_targets(parent_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directory = PRODUCTION / "outputs" / SCENARIO_CODE / str(YEAR)
    probability_path = directory / "bmi7_probability_age_sex_0p5deg_v1.tif"
    population_path = directory / "adult_population_age_sex_0p5deg_v1.tif"
    parent_zero = parent_levels.astype(np.int64) - 1
    n_parent = len(parent_levels)
    probability = np.empty((n_parent, 30, 7), dtype=np.float64)
    population = np.empty((n_parent, 30), dtype=np.float64)
    with rasterio.open(population_path) as dataset:
        if dataset.count != 30 or dataset.width != 720 or dataset.height != 360:
            raise RuntimeError(f"Invalid parent population COG: {population_path}")
        for group in range(30):
            values = dataset.read(group + 1).reshape(-1)[parent_zero]
            population[:, group] = np.where(
                np.isfinite(values) & (values != dataset.nodata) & (values > 0),
                values,
                0.0,
            )
    with rasterio.open(probability_path) as dataset:
        if dataset.count != 210 or dataset.width != 720 or dataset.height != 360:
            raise RuntimeError(f"Invalid parent probability COG: {probability_path}")
        for group in range(30):
            for category in range(7):
                band = group * 7 + category + 1
                values = dataset.read(band).reshape(-1)[parent_zero]
                probability[:, group, category] = values
            positive = population[:, group] > 0
            valid = (
                np.all(np.isfinite(probability[:, group]), axis=1)
                & np.all(probability[:, group] >= 0, axis=1)
                & np.all(probability[:, group] <= 1, axis=1)
            )
            if np.any(positive & ~valid):
                raise RuntimeError(f"Positive-population parent lacks group {group} target")
            probability[~valid, group] = 1.0 / 7.0
            probability[:, group] = np.clip(probability[:, group], EPSILON, 1.0)
            probability[:, group] /= probability[:, group].sum(axis=1, keepdims=True)
    return probability, population


def rake_parent(
    probability: np.ndarray,
    target_probability: np.ndarray,
    child_population: np.ndarray,
    target_population: np.ndarray,
    parent_index: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 300,
) -> tuple[np.ndarray, int, float]:
    positive_parent = target_population >= 1e-8
    zero_child = ~positive_parent[parent_index]
    if np.any(zero_child):
        probability[zero_child] = target_probability[parent_index[zero_child]]
    current = target_probability.copy()
    error = np.inf
    for iteration in range(1, max_iterations + 1):
        numerator = np.column_stack(
            [
                np.bincount(
                    parent_index,
                    weights=probability[:, category] * child_population,
                    minlength=len(target_population),
                )
                for category in range(7)
            ]
        )
        current[positive_parent] = (
            numerator[positive_parent] / target_population[positive_parent, None]
        )
        ratio = target_probability / np.maximum(current, EPSILON)
        ratio[~positive_parent] = 1.0
        probability *= ratio[parent_index]
        probability /= probability.sum(axis=1, keepdims=True)
        numerator = np.column_stack(
            [
                np.bincount(
                    parent_index,
                    weights=probability[:, category] * child_population,
                    minlength=len(target_population),
                )
                for category in range(7)
            ]
        )
        current[positive_parent] = (
            numerator[positive_parent] / target_population[positive_parent, None]
        )
        error = float(
            np.max(np.abs(current[positive_parent] - target_probability[positive_parent]))
        )
        if error <= tolerance:
            return probability, iteration, error
    raise RuntimeError(f"Parent rake did not converge: {error:.3e}")


def run_raking(
    n_active: int,
    parent_index: np.ndarray,
    parent_probability: np.ndarray,
    parent_population: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sex_index, sex in enumerate(SEXES):
        for age_zero, age in enumerate(AGES):
            group = sex_index * 15 + age_zero
            unraked_path, raked_path, population_path = group_paths(sex, age_zero + 1)
            print(f"[rake] {sex} {age}", flush=True)
            if raked_path.exists():
                probability = np.fromfile(raked_path, dtype="<f4")
                if probability.size != 7 * n_active:
                    raise RuntimeError(f"Invalid raked checkpoint: {raked_path}")
                probability = probability.reshape(7, n_active).T.astype(np.float64)
                iterations = -1
                parent_error = np.nan
            else:
                probability = np.fromfile(unraked_path, dtype="<f4")
                if probability.size != 7 * n_active:
                    raise RuntimeError(f"Invalid unraked binary: {unraked_path}")
                probability = probability.reshape(7, n_active).T.astype(np.float64)
                probability = np.clip(probability, EPSILON, 1.0)
                probability /= probability.sum(axis=1, keepdims=True)
                child_population = np.fromfile(population_path, dtype="<f8")
                if child_population.size != n_active:
                    raise RuntimeError(f"Invalid population binary: {population_path}")
                probability, iterations, parent_error = rake_parent(
                    probability,
                    parent_probability[:, group],
                    child_population,
                    parent_population[:, group],
                    parent_index,
                )
                atomic_binary(raked_path, probability.T, "<f4")
            child_population = np.fromfile(population_path, dtype="<f8")
            probability32 = np.fromfile(raked_path, dtype="<f4").reshape(7, n_active).T
            positive = parent_population[:, group] >= 1e-8
            observed_population = np.bincount(
                parent_index, weights=child_population, minlength=len(parent_population)
            )
            maximum_float_error = 0.0
            for category in range(7):
                numerator = np.bincount(
                    parent_index,
                    weights=probability32[:, category].astype(np.float64) * child_population,
                    minlength=len(parent_population),
                )
                observed = parent_probability[:, group, category].copy()
                observed[positive] = numerator[positive] / parent_population[positive, group]
                maximum_float_error = max(
                    maximum_float_error,
                    float(np.max(np.abs(observed - parent_probability[:, group, category]))),
                )
            rows.append(
                {
                    "group_index": group + 1,
                    "sex": sex,
                    "age_group": age,
                    "active_cells": n_active,
                    "raking_iterations": iterations,
                    "maximum_parent_error_before_float32": parent_error,
                    "maximum_parent_error_after_float32": maximum_float_error,
                    "maximum_parent_population_error": float(
                        np.max(np.abs(observed_population - parent_population[:, group]))
                    ),
                    "maximum_row_closure_error_after_float32": float(
                        np.max(np.abs(probability32.sum(axis=1) - 1.0))
                    ),
                    "raked_probability_path": str(raked_path),
                    "population_path": str(population_path),
                }
            )
    return rows


def descriptions(family: str) -> list[str]:
    if family == "probability_overall":
        return list(BMI)
    if family == "probability_sex":
        return [f"{sex}__{bmi}" for sex in SEXES for bmi in BMI]
    if family == "probability_age":
        return [f"{age.replace('-', '_').replace('+', 'plus')}__{bmi}" for age in AGES for bmi in BMI]
    if family == "probability_age_sex":
        return [
            f"{sex}__{age.replace('-', '_').replace('+', 'plus')}__{bmi}"
            for sex in SEXES for age in AGES for bmi in BMI
        ]
    if family == "population_overall":
        return ["adult_population_age25plus"]
    if family == "population_sex":
        return [f"{sex}__adult_population_age25plus" for sex in SEXES]
    if family == "population_age":
        return [f"{age.replace('-', '_').replace('+', 'plus')}__adult_population" for age in AGES]
    if family == "population_age_sex":
        return [
            f"{sex}__{age.replace('-', '_').replace('+', 'plus')}__population"
            for sex in SEXES for age in AGES
        ]
    raise KeyError(family)


def create_staging(path: Path, family: str) -> rasterio.io.DatasetWriter:
    profile = {
        "driver": "GTiff",
        "width": WIDTH,
        "height": HEIGHT,
        "count": FAMILY_COUNTS[family],
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_origin(-180, 90, 0.1, 0.1),
        "nodata": NODATA,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "ZSTD",
        "zstd_level": 1,
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    dataset = rasterio.open(path, "w", **profile)
    for band, description in enumerate(descriptions(family), start=1):
        dataset.set_band_description(band, description)
    dataset.update_tags(
        scenario=SCENARIO,
        scenario_code=SCENARIO_CODE,
        year=str(YEAR),
        spatial_resolution="0.1 degree",
        population_mask="adult age 25+ density >= 0.1 persons per km2",
        parent_calibration="population-weighted to formal 0.5-degree age-sex target",
        model="validated 22-feature Bayesian multinomial logistic-normal MRP",
    )
    return dataset


def write_active_band(
    dataset: rasterio.io.DatasetWriter,
    band: int,
    values: np.ndarray,
    active_zero: np.ndarray,
    valid: np.ndarray | None = None,
) -> None:
    full = np.full(NCELL, NODATA, dtype=np.float32)
    if valid is None:
        full[active_zero] = np.asarray(values, dtype=np.float32)
    else:
        full[active_zero[valid]] = np.asarray(values[valid], dtype=np.float32)
    dataset.write(full.reshape(HEIGHT, WIDTH), band)


def make_cog(staging: Path, output: Path, *, build_overviews: bool) -> float:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    creation_options = {
        "compress": "ZSTD",
        "level": 1,
        "blocksize": 512,
        "nodata": NODATA,
        "BIGTIFF": "IF_SAFER",
        "NUM_THREADS": COG_NUM_THREADS,
        "OVERVIEWS": "AUTO" if build_overviews else "NONE",
    }
    if build_overviews:
        creation_options["overview_resampling"] = "nearest"
    started = time.perf_counter()
    with rasterio.Env(GDAL_NUM_THREADS=COG_NUM_THREADS):
        rio_copy(staging, output, driver="COG", **creation_options)
    return time.perf_counter() - started


def assemble_cogs(
    n_active: int, active_zero: np.ndarray
) -> tuple[list[dict[str, object]], float, dict[str, object]]:
    assembly_started = time.perf_counter()
    staging_dir = ROOT / "work" / "cog_staging" / SCENARIO_CODE / str(YEAR)
    output_dir = ROOT / "data" / "0p1deg" / "cog" / SCENARIO_CODE / str(YEAR)
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_paths = {family: staging_dir / f"{family}_staging_v1.tif" for family in FAMILY_COUNTS}
    if any(path.exists() for path in staging_paths.values()):
        raise RuntimeError("Existing staging file requires review")
    datasets = {family: create_staging(path, family) for family, path in staging_paths.items()}
    sex_numerator = {sex: np.zeros((7, n_active), dtype=np.float64) for sex in SEXES}
    sex_population = {sex: np.zeros(n_active, dtype=np.float64) for sex in SEXES}
    maximum_closure = 0.0
    try:
        for age_zero, age in enumerate(AGES):
            pair: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for sex_index, sex in enumerate(SEXES):
                _, raked_path, population_path = group_paths(sex, age_zero + 1)
                probability = np.fromfile(raked_path, dtype="<f4").reshape(7, n_active)
                population = np.fromfile(population_path, dtype="<f8")
                pair[sex] = (probability, population)
                sex_numerator[sex] += probability.astype(np.float64) * population
                sex_population[sex] += population
                group = sex_index * 15 + age_zero
                valid = population > 0
                for category in range(7):
                    write_active_band(
                        datasets["probability_age_sex"], group * 7 + category + 1,
                        probability[category], active_zero, valid,
                    )
                write_active_band(
                    datasets["population_age_sex"], group + 1,
                    population, active_zero,
                )

            female_probability, female_population = pair["Female"]
            male_probability, male_population = pair["Male"]
            age_population = female_population + male_population
            age_probability = (
                female_probability.astype(np.float64) * female_population
                + male_probability.astype(np.float64) * male_population
            ) / np.maximum(age_population, np.finfo(float).tiny)
            valid_age = age_population > 0
            maximum_closure = max(
                maximum_closure,
                float(np.max(np.abs(age_probability[:, valid_age].sum(axis=0) - 1.0))),
            )
            for category in range(7):
                write_active_band(
                    datasets["probability_age"], age_zero * 7 + category + 1,
                    age_probability[category], active_zero, valid_age,
                )
            write_active_band(
                datasets["population_age"], age_zero + 1,
                age_population, active_zero,
            )

        sex_probability: dict[str, np.ndarray] = {}
        for sex_index, sex in enumerate(SEXES):
            probability = sex_numerator[sex] / np.maximum(
                sex_population[sex], np.finfo(float).tiny
            )
            valid = sex_population[sex] > 0
            maximum_closure = max(
                maximum_closure,
                float(np.max(np.abs(probability[:, valid].sum(axis=0) - 1.0))),
            )
            sex_probability[sex] = probability
            for category in range(7):
                write_active_band(
                    datasets["probability_sex"], sex_index * 7 + category + 1,
                    probability[category], active_zero, valid,
                )
            write_active_band(
                datasets["population_sex"], sex_index + 1,
                sex_population[sex], active_zero,
            )

        adult_population = sex_population["Female"] + sex_population["Male"]
        overall_probability = (
            sex_numerator["Female"] + sex_numerator["Male"]
        ) / np.maximum(adult_population, np.finfo(float).tiny)
        valid_overall = adult_population > 0
        maximum_closure = max(
            maximum_closure,
            float(np.max(np.abs(overall_probability[:, valid_overall].sum(axis=0) - 1.0))),
        )
        for category in range(7):
            write_active_band(
                datasets["probability_overall"], category + 1,
                overall_probability[category], active_zero, valid_overall,
            )
        write_active_band(
            datasets["population_overall"], 1, adult_population, active_zero,
        )
    finally:
        for dataset in datasets.values():
            dataset.close()
    staging_assembly_seconds = time.perf_counter() - assembly_started

    rows: list[dict[str, object]] = []
    cog_write_seconds: dict[str, float] = {}
    inspection_and_hash_seconds: dict[str, float] = {}
    for family, staging in staging_paths.items():
        output = output_dir / f"{family}_{SCENARIO_CODE}_{YEAR}_0p1deg_v1.tif"
        build_overviews = family != AGE_SEX_PROBABILITY_FAMILY
        overview_policy = "AUTO" if build_overviews else "NONE"
        print(
            f"[COG] {family} NUM_THREADS={COG_NUM_THREADS} "
            f"OVERVIEWS={overview_policy}",
            flush=True,
        )
        cog_write_seconds[family] = make_cog(
            staging, output, build_overviews=build_overviews
        )
        print(
            f"[timing] {family}_cog_write_seconds="
            f"{cog_write_seconds[family]:.3f}",
            flush=True,
        )
        inspection_started = time.perf_counter()
        with rasterio.open(output) as dataset:
            structure = dataset.tags(ns="IMAGE_STRUCTURE")
            output_overviews = dataset.overviews(1)
            output_sha256 = sha256(output)
            rows.append(
                {
                    "family": family,
                    "path": str(output),
                    "bands": dataset.count,
                    "width": dataset.width,
                    "height": dataset.height,
                    "resolution_x": dataset.res[0],
                    "resolution_y": dataset.res[1],
                    "crs": dataset.crs.to_string(),
                    "nodata": dataset.nodata,
                    "layout": structure.get("LAYOUT", ""),
                    "compression": structure.get("COMPRESSION", ""),
                    "overviews": "|".join(map(str, output_overviews)),
                    "overview_policy": overview_policy,
                    "gdal_num_threads": COG_NUM_THREADS,
                    "cog_write_seconds": cog_write_seconds[family],
                    "bytes": output.stat().st_size,
                    "sha256": output_sha256,
                }
            )
        inspection_and_hash_seconds[family] = time.perf_counter() - inspection_started
        staging.unlink()
    timing = {
        "staging_assembly_seconds": staging_assembly_seconds,
        "probability_age_sex_cog_write_seconds": cog_write_seconds[
            AGE_SEX_PROBABILITY_FAMILY
        ],
        "probability_age_sex_overview_seconds": 0.0,
        "other_seven_cog_write_seconds": sum(
            seconds
            for family, seconds in cog_write_seconds.items()
            if family != AGE_SEX_PROBABILITY_FAMILY
        ),
        "inspection_and_sha256_seconds": sum(inspection_and_hash_seconds.values()),
        "gdal_cog_num_threads": COG_NUM_THREADS,
        "probability_age_sex_overviews_enabled": False,
    }
    return rows, maximum_closure, timing


def main() -> int:
    started = now_utc()
    total_started = time.perf_counter()
    state = load_state()
    active_zero, parent_levels, parent_index, country_id = load_geometry(state)
    n_active = len(active_zero)
    parent_probability, parent_population = load_parent_targets(parent_levels)
    raking_started = time.perf_counter()
    raking_rows = run_raking(
        n_active, parent_index, parent_probability, parent_population
    )
    raking_seconds = time.perf_counter() - raking_started
    print(f"[timing] raking_seconds={raking_seconds:.3f}", flush=True)
    raking_path = ROOT / "qa" / "Gate_FC03_age_sex_raking_ssp585_2050_v1.csv"
    write_csv_new(raking_path, raking_rows)

    cog_rows, derived_closure, timing = assemble_cogs(n_active, active_zero)
    timing["raking_seconds"] = raking_seconds
    timing["fc03_total_seconds"] = time.perf_counter() - total_started
    timing_rows = [
        {"phase": key, "value": value}
        for key, value in timing.items()
    ]
    timing_path = ROOT / "qa" / "Gate_FC03_timing_ssp585_2050_v1.csv"
    write_csv_new(timing_path, timing_rows)
    print(f"[timing] summary={json.dumps(timing, ensure_ascii=False)}", flush=True)
    inventory_path = ROOT / "qa" / "Gate_FC03_COG_inventory_ssp585_2050_v1.csv"
    write_csv_new(inventory_path, cog_rows)

    maximum_float_parent = max(
        float(row["maximum_parent_error_after_float32"]) for row in raking_rows
    )
    maximum_parent_population = max(
        float(row["maximum_parent_population_error"]) for row in raking_rows
    )
    maximum_group_closure = max(
        float(row["maximum_row_closure_error_after_float32"]) for row in raking_rows
    )
    expected_families = set(FAMILY_COUNTS)
    observed_families = {str(row["family"]) for row in cog_rows}
    geometry_pass = all(
        int(row["bands"]) == FAMILY_COUNTS[str(row["family"])]
        and int(row["width"]) == WIDTH
        and int(row["height"]) == HEIGHT
        and abs(float(row["resolution_x"]) - 0.1) < 1e-12
        and abs(float(row["resolution_y"]) - 0.1) < 1e-12
        and row["crs"] == "EPSG:4326"
        for row in cog_rows
    )
    checks = [
        {"check_id": "FC03_01", "requirement": "Gate FC02 v2 is PASS", "observed": state["status"], "pass": True},
        {"check_id": "FC03_02", "requirement": "All thirty age-sex groups were raked", "observed": len(raking_rows), "pass": len(raking_rows) == 30},
        {"check_id": "FC03_03", "requirement": "Float32 parent composition error <= 5e-6", "observed": maximum_float_parent, "pass": maximum_float_parent <= 5e-6},
        {"check_id": "FC03_04", "requirement": "Parent population error <= 1e-5 persons", "observed": maximum_parent_population, "pass": maximum_parent_population <= 1e-5},
        {"check_id": "FC03_05", "requirement": "Age-sex probability closure <= 1e-6", "observed": maximum_group_closure, "pass": maximum_group_closure <= 1e-6},
        {"check_id": "FC03_06", "requirement": "Derived probability closure <= 1e-6", "observed": derived_closure, "pass": derived_closure <= 1e-6},
        {"check_id": "FC03_07", "requirement": "All eight output families exist", "observed": len(cog_rows), "pass": observed_families == expected_families},
        {"check_id": "FC03_08", "requirement": "COG geometry and band counts match contract", "observed": str(geometry_pass), "pass": geometry_pass},
        {"check_id": "FC03_09", "requirement": "Every formal raster reports COG layout", "observed": f"{sum(row['layout'] == 'COG' for row in cog_rows)}/8", "pass": all(row["layout"] == "COG" for row in cog_rows)},
        {"check_id": "FC03_10", "requirement": "No model fitting or disease burden analysis occurred", "observed": "scope guard retained", "pass": True},
    ]
    checks_path = ROOT / "qa" / "Gate_FC03_checks_ssp585_2050_v1.csv"
    write_csv_new(checks_path, checks)
    status = "PASS_GATE_FC03_FULL_0P1DEG_PILOT" if all(row["pass"] for row in checks) else "FAIL_GATE_FC03_FULL_0P1DEG_PILOT"

    checksum_rows = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in cog_rows
    ]
    checksum_path = ROOT / "checksums" / "Gate_FC03_formal_COG_SHA256_ssp585_2050_v1.csv"
    write_csv_new(checksum_path, checksum_rows)

    summary = {
        "status": status,
        "started_at_utc": started,
        "completed_at_utc": now_utc(),
        "scenario": SCENARIO,
        "scenario_code": SCENARIO_CODE,
        "year": YEAR,
        "model": "validated 22-feature Bayesian multinomial logistic-normal MRP",
        "active_child_cells": n_active,
        "represented_parent_cells": len(parent_levels),
        "represented_countries": int(len(np.unique(country_id))),
        "formal_cogs": len(cog_rows),
        "formal_cog_bytes": int(sum(int(row["bytes"]) for row in cog_rows)),
        "maximum_float32_parent_composition_error": maximum_float_parent,
        "maximum_parent_population_absolute_error": maximum_parent_population,
        "maximum_age_sex_probability_closure_error": maximum_group_closure,
        "maximum_derived_probability_closure_error": derived_closure,
        "fc03_timing": timing,
        "outputs": [row["path"] for row in cog_rows],
    }
    write_json_new(ROOT / "qa" / "Gate_FC03_summary_ssp585_2050_v1.json", summary)
    write_json_new(
        ROOT / "contracts" / "full_covariate_0p1deg_nested_output_contract_v1.json",
        {
            "status": status,
            "scenario": SCENARIO,
            "year": YEAR,
            "grid": "global EPSG:4326 0.1 degree, 3600 x 1800",
            "output_families": FAMILY_COUNTS,
            "probability_unit": "fraction from 0 to 1",
            "population_unit": "persons",
            "nodata": NODATA,
            "cog_num_threads": COG_NUM_THREADS,
            "probability_age_sex_overviews": "disabled",
            "other_cog_overviews": "retained",
            "population_mask": "adult age 25+ density >= 0.1 persons per km2",
            "nesting": "each age-sex composition and population reproduces its retained formal 0.5-degree parent",
            "interpretation": "nested high-resolution allocation, not an independently validated 0.1-degree observation model",
        },
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
