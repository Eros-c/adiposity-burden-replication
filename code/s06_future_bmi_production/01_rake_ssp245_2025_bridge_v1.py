"""Build a non-destructive 2025 SSP2-4.5 national-anchor bridge pilot.

The pilot retains the official 0.1-degree population and MRP spatial gradient,
then rakes each country-by-sex-by-age composition to the Phase 3
``ncd_2024_delta_sensitivity`` target.  It is a continuity sensitivity output,
not a replacement for the frozen absolute-GBD production product.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(_cfg.ROOT / "future_scenario_ascii" / "06_future_bmi_2025_2050" / "bridge_ncd2024_gbdtrend_sensitivity_v1_20260909")
FORMAL = Path(_cfg.ROOT / "future_scenario_ascii" / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906" / "outputs" / "ssp245" / "2025")
PHASE3 = Path(_cfg.ROOT / "future_scenario_ascii" / "03_national_bmi_anchors" / "phase3_national_bmi7_anchors_v2_20260828_144604" / "tables" / "national_bmi7_anchors_2025_2050_v1.csv.gz")
ZONE_RASTER = Path(_cfg.ROOT / "future_scenario_ascii" / "06_future_bmi_2025_2050" / "phase6i_remaining_scenarios_0p1deg_adm1_v1_20260904" / "metadata" / "adm1_zone_country_constrained_0p1deg_v3.tif")
ZONE_LOOKUP = Path(_cfg.ROOT / "future_scenario_ascii" / "06_future_bmi_2025_2050" / "phase6i_remaining_scenarios_0p1deg_adm1_v1_20260904" / "metadata" / "adm1_zone_lookup_v1.csv")
SCENARIO = os.environ.get("BRIDGE_SCENARIO", "SSP2-4.5")
SCENARIO_CODE = os.environ.get("BRIDGE_SCENARIO_CODE", "ssp245")
YEAR = int(os.environ.get("BRIDGE_YEAR", "2025"))
WIDTH, HEIGHT = 3600, 1800
NODATA = -9999.0
EPS = 1e-12
BMI = ("bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30", "bmi_30_35", "bmi_35_40", "bmi_ge_40")
SEXES = ("Female", "Male")
AGES = ("25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+")
# The ADM1 boundary product and the Phase 3 GBD anchor file use different
# codes for these three represented locations. This mapping is exact and is
# retained in the QA artefacts rather than silently filling an anchor.
ANCHOR_TO_GRID_ISO3 = {"PSE": "PSX", "SSD": "SDS"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize_age(value: str) -> str:
    return str(value).replace(" to ", "-").replace(" plus", "+")


def source_paths() -> tuple[Path, Path]:
    return (
        FORMAL / "probability_age_sex_ssp245_2025_0p1deg_v1.tif",
        FORMAL / "population_age_sex_ssp245_2025_0p1deg_v1.tif",
    )


def descriptions(family: str) -> list[str]:
    if family == "probability_age_sex":
        return [f"{sex}__{age.replace('-', '_').replace('+', 'plus')}__{bmi}" for sex in SEXES for age in AGES for bmi in BMI]
    if family == "probability_age":
        return [f"{age.replace('-', '_').replace('+', 'plus')}__{bmi}" for age in AGES for bmi in BMI]
    if family == "probability_sex":
        return [f"{sex}__{bmi}" for sex in SEXES for bmi in BMI]
    if family == "probability_overall":
        return list(BMI)
    raise KeyError(family)


def write_dataset(path: Path, count: int, descriptions_: list[str], profile: dict) -> rasterio.io.DatasetWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".staging")
    if staging.exists() or path.exists():
        raise FileExistsError(f"Bridge pilot output already exists: {path}")
    output_profile = profile.copy()
    output_profile.update(driver="GTiff", count=count, dtype="float32", nodata=NODATA, tiled=True, blockxsize=256, blockysize=256, compress="ZSTD", zstd_level=1, predictor=3, BIGTIFF="IF_SAFER")
    dataset = rasterio.open(staging, "w", **output_profile)
    for index, description in enumerate(descriptions_, start=1):
        dataset.set_band_description(index, description)
    dataset.update_tags(
        scenario=SCENARIO,
        scenario_code=SCENARIO_CODE,
        year=str(YEAR),
        product_role="NCD-2024 anchored plus GBD-trend continuity sensitivity",
        population_source=str(source_paths()[1]),
        calibration="country-by-sex-by-age compositional rake",
        source_probability=str(source_paths()[0]),
    )
    return dataset


def finalize_dataset(staging: Path, output: Path) -> None:
    os.replace(staging, output)


def build_target_cube(country_codes: list[str], required_country_index: np.ndarray) -> tuple[np.ndarray, list[dict[str, str]]]:
    usecols = ["anchor_scenario", "location_name", "iso3", "sex", "age_group_name", "year", *[f"p_{x}" for x in BMI]]
    anchors = pd.read_csv(PHASE3, usecols=usecols)
    anchors = anchors.loc[(anchors.anchor_scenario == "ncd_2024_delta_sensitivity") & (anchors.year == YEAR)].copy()
    anchors["age_group"] = anchors.age_group_name.map(normalize_age)
    anchors["grid_iso3"] = anchors["iso3"].map(ANCHOR_TO_GRID_ISO3).fillna(anchors["iso3"])
    anchors.loc[anchors.location_name.eq("United States Virgin Islands"), "grid_iso3"] = "VIR"
    expected = len(required_country_index) * len(SEXES) * len(AGES)
    target = np.full((len(SEXES) * len(AGES), len(country_codes), 7), np.nan, dtype=np.float64)
    country_index = {code: index for index, code in enumerate(country_codes)}
    for _, row in anchors.iterrows():
        if row.grid_iso3 not in country_index or row.sex not in SEXES or row.age_group not in AGES:
            continue
        group = SEXES.index(row.sex) * len(AGES) + AGES.index(row.age_group)
        values = row[[f"p_{x}" for x in BMI]].to_numpy(dtype=float)
        if np.all(np.isfinite(values)) and np.all(values >= 0) and values.sum() > 0:
            target[group, country_index[row.grid_iso3]] = values / values.sum()
    present = int(np.isfinite(target[:, required_country_index]).all(axis=2).sum())
    if present != expected:
        raise RuntimeError(f"Bridge targets incomplete for represented countries: {present}/{expected}")
    mapping_audit = [
        {"grid_iso3": "PSX", "anchor_iso3": "PSE", "anchor_location_name": "Palestine", "rule": "boundary_to_GBD_code"},
        {"grid_iso3": "SDS", "anchor_iso3": "SSD", "anchor_location_name": "South Sudan", "rule": "boundary_to_GBD_code"},
        {"grid_iso3": "VIR", "anchor_iso3": "", "anchor_location_name": "United States Virgin Islands", "rule": "location_name_to_boundary_code"},
    ]
    return target, mapping_audit


def load_country_index() -> tuple[np.ndarray, list[str]]:
    lookup = pd.read_csv(ZONE_LOOKUP, usecols=["zone_id", "adm0_a3"])
    country_codes = sorted(lookup.adm0_a3.dropna().unique().tolist())
    code_index = {code: index for index, code in enumerate(country_codes)}
    zone_map = np.full(int(lookup.zone_id.max()) + 1, -1, dtype=np.int32)
    for row in lookup.itertuples(index=False):
        if row.adm0_a3 in code_index:
            zone_map[int(row.zone_id)] = code_index[row.adm0_a3]
    with rasterio.open(ZONE_RASTER) as dataset:
        zone = dataset.read(1)
        if zone.shape != (HEIGHT, WIDTH):
            raise RuntimeError("Unexpected ADM1 zone geometry")
        valid_zone = (zone > 0) & (zone < len(zone_map))
        country_grid = np.full(zone.shape, -1, dtype=np.int32)
        country_grid[valid_zone] = zone_map[zone[valid_zone]]
    return country_grid.reshape(-1), country_codes


def rake_country(probability: np.ndarray, population: np.ndarray, country: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int, float, int]:
    valid = np.isfinite(population) & (population > 0) & (country >= 0) & np.all(np.isfinite(probability), axis=0)
    unassigned = int(np.sum(np.isfinite(population) & (population > 0) & (country < 0)))
    if not np.any(valid):
        return probability, 0, 0.0, unassigned
    codes = country[valid]
    weights = population[valid].astype(np.float64, copy=False)
    values = np.clip(probability[:, valid].astype(np.float64, copy=False), EPS, 1.0)
    values /= values.sum(axis=0, keepdims=True)
    denominator = np.bincount(codes, weights=weights, minlength=target.shape[0])
    positive = denominator > 0
    error = np.inf
    for iteration in range(1, 301):
        numerator = np.vstack([np.bincount(codes, weights=values[k] * weights, minlength=target.shape[0]) for k in range(7)]).T
        current = numerator[positive] / denominator[positive, None]
        ratio = np.ones_like(target)
        ratio[positive] = target[positive] / np.maximum(current, EPS)
        values *= ratio[codes].T
        values /= values.sum(axis=0, keepdims=True)
        numerator = np.vstack([np.bincount(codes, weights=values[k] * weights, minlength=target.shape[0]) for k in range(7)]).T
        current = numerator[positive] / denominator[positive, None]
        error = float(np.max(np.abs(current - target[positive])))
        if error <= 1e-8:
            probability[:, valid] = values.astype(np.float32)
            return probability, iteration, error, unassigned
    raise RuntimeError(f"Country rake did not converge: {error:.3e}")


def main() -> int:
    started = utc()
    output_dir = ROOT / "outputs" / SCENARIO_CODE / str(YEAR)
    qa_dir = ROOT / "qa"
    source_probability, source_population = source_paths()
    country_grid, country_codes = load_country_index()
    with rasterio.open(source_paths()[1]) as support_dataset:
        support = support_dataset.read(1).reshape(-1)
    required_country_index = np.unique(country_grid[(country_grid >= 0) & np.isfinite(support) & (support > 0)])
    targets, mapping_audit = build_target_cube(country_codes, required_country_index)
    with rasterio.open(source_probability) as ps, rasterio.open(source_population) as qs:
        if ps.count != 210 or qs.count != 30 or ps.width != WIDTH or qs.width != WIDTH or ps.height != HEIGHT or qs.height != HEIGHT:
            raise RuntimeError("Source bridge pilot COG structure mismatch")
        profile = ps.profile
        output_paths = {
            "probability_age_sex": output_dir / f"probability_age_sex_{SCENARIO_CODE}_{YEAR}_0p1deg_bridge_v1.tif",
            "probability_age": output_dir / f"probability_age_{SCENARIO_CODE}_{YEAR}_0p1deg_bridge_v1.tif",
            "probability_sex": output_dir / f"probability_sex_{SCENARIO_CODE}_{YEAR}_0p1deg_bridge_v1.tif",
            "probability_overall": output_dir / f"probability_overall_{SCENARIO_CODE}_{YEAR}_0p1deg_bridge_v1.tif",
        }
        datasets = {
            "probability_age_sex": write_dataset(
                output_paths["probability_age_sex"],
                len(descriptions("probability_age_sex")),
                descriptions("probability_age_sex"),
                profile,
            )
        }
        rake_rows: list[dict[str, object]] = []
        try:
            for group in range(30):
                sex = SEXES[group // 15]
                age = AGES[group % 15]
                probability = ps.read(list(range(group * 7 + 1, group * 7 + 8))).reshape(7, -1)
                population = qs.read(group + 1).reshape(-1)
                valid_source = np.isfinite(population) & (population > 0) & np.all(np.isfinite(probability), axis=0) & np.all(probability != ps.nodata, axis=0)
                probability[:, ~valid_source] = np.nan
                probability, iterations, maximum_error, unassigned = rake_country(probability, population, country_grid, targets[group])
                write = np.full((7, HEIGHT * WIDTH), NODATA, dtype=np.float32)
                write[:, valid_source] = probability[:, valid_source]
                datasets["probability_age_sex"].write(write.reshape(7, HEIGHT, WIDTH), indexes=list(range(group * 7 + 1, group * 7 + 8)))
                rake_rows.append({"group": group + 1, "sex": sex, "age_group": age, "iterations": iterations, "maximum_country_target_error": maximum_error, "positive_population_without_adm1_country_assignment": unassigned, "active_cells": int(valid_source.sum())})
        finally:
            datasets["probability_age_sex"].close()

    # Derive age, sex and overall families from raked age-sex composition and the canonical population COG.
    with rasterio.open(output_paths["probability_age_sex"].with_suffix(".tif.staging")) as raked, rasterio.open(source_population) as population_source:
        profile = raked.profile
        for family in ("probability_age", "probability_sex", "probability_overall"):
            datasets[family] = write_dataset(output_paths[family], len(descriptions(family)), descriptions(family), profile)
        try:
            for row0 in range(0, HEIGHT, 40):
                height = min(40, HEIGHT - row0)
                window = rasterio.windows.Window(0, row0, WIDTH, height)
                p = raked.read(window=window).reshape(2, 15, 7, height, WIDTH).astype(np.float64)
                pop = population_source.read(window=window).reshape(2, 15, height, WIDTH).astype(np.float64)
                valid = (pop > 0) & np.isfinite(pop) & np.all(np.isfinite(p), axis=2) & np.all(p != NODATA, axis=2)
                pop = np.where(valid, pop, 0.0)
                p = np.where(valid[:, :, None], p, 0.0)
                age_population = pop.sum(axis=0)
                age_probability = (p * pop[:, :, None]).sum(axis=0) / np.maximum(age_population[:, None], EPS)
                sex_population = pop.sum(axis=1)
                sex_probability = (p * pop[:, :, None]).sum(axis=1) / np.maximum(sex_population[:, None], EPS)
                total_population = sex_population.sum(axis=0)
                total_probability = (sex_probability * sex_population[:, None]).sum(axis=0) / np.maximum(total_population[None], EPS)
                for age in range(15):
                    values = np.where(age_population[age, None] > 0, age_probability[age], NODATA).astype(np.float32)
                    datasets["probability_age"].write(values, indexes=list(range(age * 7 + 1, age * 7 + 8)), window=window)
                for sex in range(2):
                    values = np.where(sex_population[sex, None] > 0, sex_probability[sex], NODATA).astype(np.float32)
                    datasets["probability_sex"].write(values, indexes=list(range(sex * 7 + 1, sex * 7 + 8)), window=window)
                datasets["probability_overall"].write(np.where(total_population[None] > 0, total_probability, NODATA).astype(np.float32), window=window)
        finally:
            for family in ("probability_age", "probability_sex", "probability_overall"):
                datasets[family].close()

    inventory = []
    for family, output in output_paths.items():
        staging = output.with_suffix(".tif.staging")
        finalize_dataset(staging, output)
        with rasterio.open(output) as dataset:
            inventory.append({"family": family, "path": str(output), "bands": dataset.count, "width": dataset.width, "height": dataset.height, "nodata": dataset.nodata, "compression": dataset.compression.value if dataset.compression else "", "bytes": output.stat().st_size, "sha256": sha256(output)})
    qa_dir.mkdir(parents=True, exist_ok=True)
    with (qa_dir / f"{SCENARIO_CODE}_{YEAR}_bridge_raking_v1.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rake_rows[0])); writer.writeheader(); writer.writerows(rake_rows)
    with (qa_dir / f"{SCENARIO_CODE}_{YEAR}_bridge_inventory_v1.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0])); writer.writeheader(); writer.writerows(inventory)
    with (qa_dir / f"{SCENARIO_CODE}_{YEAR}_bridge_iso3_mapping_v1.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_audit[0])); writer.writeheader(); writer.writerows(mapping_audit)
    maximum_error = max(float(row["maximum_country_target_error"]) for row in rake_rows)
    write_json(qa_dir / f"{SCENARIO_CODE}_{YEAR}_bridge_summary_v1.json", {"status": "PASS_BRIDGE_PILOT" if maximum_error <= 1e-8 else "FAIL", "started_at_utc": started, "completed_at_utc": utc(), "scenario": SCENARIO, "year": YEAR, "maximum_country_target_error": maximum_error, "source_probability": str(source_probability), "canonical_population_source": str(source_population), "anchor_scenario": "ncd_2024_delta_sensitivity", "iso3_mapping_audit": mapping_audit, "outputs": inventory})
    print("BRIDGE_PILOT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
