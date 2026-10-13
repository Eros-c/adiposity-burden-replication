#!/usr/bin/env python3
"""One scenario-year task for full annual four-SSP CRC PAF production."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


PROJECT = Path(_cfg.RESULTS)
OUTPUT_ROOT = PROJECT / "08_colorectal_cancer_paf" / "full_annual_four_ssp_wcrf_primary_v1_20260910"
RR_PATH = PROJECT / "07_rr_cause_crosswalk" / "CRC_RR_by_BMI_category_WCRF_primary_v1.csv"
BMI_01 = PROJECT / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906" / "outputs"
BMI_05 = PROJECT / "06_future_bmi_2025_2050" / "full_covariate_annual_0p5deg_production_v1_20260906" / "outputs"
BMI_ADM1 = PROJECT / "06_future_bmi_2025_2050" / "phase7_multiscale_delivery_v3_20260908" / "04_adm1" / "tables"

SCENARIOS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
AGES = ("25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+")
SEXES = ("Female", "Male")
CATEGORIES = ("bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30", "bmi_30_35", "bmi_35_40", "bmi_ge_40")
ENDPOINTS = ("point", "lower", "upper")
NODATA = -9999.0
CLOSURE_TOLERANCE = 1e-5
SPATIAL_TOLERANCE = 1e-7
NATIONAL_TOLERANCE = 1e-3
ADM1_ID = ["scenario_code", "scenario", "year", "zone_id", "adm1_uid", "adm1_code", "iso_3166_2", "adm0_a3", "adm1_name", "name", "type_en", "region", "region_sub", "ne_id"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rr() -> dict[str, dict[str, np.ndarray]]:
    frame = pd.read_csv(RR_PATH)
    if len(frame) != 14 or set(frame["sex"]) != set(SEXES) or set(frame["BMI_category"]) != set(CATEGORIES):
        raise RuntimeError("Frozen RR crosswalk structure mismatch")
    result = {}
    for sex in SEXES:
        selected = frame[frame["sex"].eq(sex)].set_index("BMI_category").loc[list(CATEGORIES)]
        result[sex] = {"point": selected["RR"].to_numpy(float), "lower": selected["RR_lower"].to_numpy(float), "upper": selected["RR_upper"].to_numpy(float)}
    if min(x.min() for sex in result.values() for x in sex.values()) < 1:
        raise RuntimeError("Frozen RR contains value below one")
    return result


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, float)
    denominator = np.asarray(denominator, float)
    return np.divide(numerator, denominator, out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan), where=denominator > 0)


def add_risk_components(frame: pd.DataFrame, lookup) -> pd.DataFrame:
    frame = frame.copy()
    probabilities = frame[list(CATEGORIES)].to_numpy(float)
    frame["probability_sum"] = probabilities.sum(axis=1)
    for endpoint in ENDPOINTS:
        rr = np.vstack([lookup[sex][endpoint] for sex in frame["sex"]])
        weighted_rr = np.sum(probabilities * rr, axis=1)
        population = frame["adult_population"].to_numpy(float)
        frame[f"weighted_rr_{endpoint}"] = weighted_rr
        frame[f"risk_excess_numerator_{endpoint}"] = population * (weighted_rr - 1.0)
        frame[f"risk_denominator_{endpoint}"] = population * weighted_rr
        frame[f"paf_{'rr_' if endpoint != 'point' else ''}{endpoint if endpoint != 'point' else ''}".rstrip("_")] = safe_divide(frame[f"risk_excess_numerator_{endpoint}"], frame[f"risk_denominator_{endpoint}"])
    frame = frame.rename(columns={"paf": "paf", "paf_rr_lower": "paf_rr_lower", "paf_rr_upper": "paf_rr_upper"})
    return frame


def aggregate_risk(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    value_columns = ["adult_population"] + [f"risk_{part}_{endpoint}" for endpoint in ENDPOINTS for part in ("excess_numerator", "denominator")]
    result = frame.groupby(groups, observed=True, dropna=False, sort=False)[value_columns].sum().reset_index()
    for endpoint in ENDPOINTS:
        result[f"weighted_rr_{endpoint}"] = safe_divide(result[f"risk_denominator_{endpoint}"], result["adult_population"])
        name = "paf" if endpoint == "point" else f"paf_rr_{endpoint}"
        result[name] = safe_divide(result[f"risk_excess_numerator_{endpoint}"], result[f"risk_denominator_{endpoint}"])
    return result


def validate_frame(name: str, frame: pd.DataFrame, probability_column: bool = False) -> dict:
    selected = frame[frame["adult_population"].gt(0)].copy()
    paf_cols = ["paf", "paf_rr_lower", "paf_rr_upper"]
    rr_cols = [f"weighted_rr_{x}" for x in ENDPOINTS]
    finite = np.isfinite(selected[paf_cols + rr_cols].to_numpy(float)).all()
    closure = float(np.abs(selected["probability_sum"] - 1.0).max()) if probability_column and len(selected) else None
    result = {
        "rows": len(frame), "positive_population_rows": len(selected), "finite": bool(finite),
        "rr_min": float(selected[rr_cols].min().min()), "rr_max": float(selected[rr_cols].max().max()),
        "paf_min": float(selected[paf_cols].min().min()), "paf_max": float(selected[paf_cols].max().max()),
        "probability_closure_max_abs_error": closure,
    }
    if not finite or result["rr_min"] < 1 - 1e-12 or result["paf_min"] < -1e-12 or result["paf_max"] > 1 + 1e-12:
        raise RuntimeError(f"Frame QA failed: {name}")
    if closure is not None and closure > CLOSURE_TOLERANCE:
        raise RuntimeError(f"Probability closure failed: {name}")
    return result


def national_input(scenario: str, year: int) -> Path:
    return OUTPUT_ROOT / "input_cache" / "national_anchors" / scenario / f"{year}.parquet"


def process_national(scenario: str, year: int, lookup):
    path = national_input(scenario, year)
    raw = pd.read_parquet(path)
    raw = raw.rename(columns={"population_persons": "adult_population", **{f"p_{category}": category for category in CATEGORIES}})
    raw["scenario_code"] = scenario
    raw["age_group"] = raw["age_group_output"].str.replace(" to ", "-", regex=False).str.replace(" plus", "+", regex=False)
    raw = add_risk_components(raw, lookup)
    national_id = ["scenario_code", "scenario", "year", "iso3"]
    age_sex = raw[national_id + ["sex", "age_group", "adult_population", "probability_sum"] + [f"weighted_rr_{x}" for x in ENDPOINTS] + ["paf", "paf_rr_lower", "paf_rr_upper"] + [f"risk_{part}_{x}" for x in ENDPOINTS for part in ("excess_numerator", "denominator")]].copy()
    frames = {
        "age_sex": age_sex,
        "age": aggregate_risk(raw, national_id + ["age_group"]),
        "sex": aggregate_risk(raw, national_id + ["sex"]),
        "overall": aggregate_risk(raw, national_id),
        "global_age": aggregate_risk(raw.assign(global_scope="Global"), ["scenario_code", "scenario", "year", "global_scope", "age_group"]),
        "global_sex": aggregate_risk(raw.assign(global_scope="Global"), ["scenario_code", "scenario", "year", "global_scope", "sex"]),
        "global_overall": aggregate_risk(raw.assign(global_scope="Global"), ["scenario_code", "scenario", "year", "global_scope"]),
    }
    qa = validate_frame("national_age_sex", age_sex, probability_column=True)
    qa.update({"input": str(path), "input_sha256": sha256(path), "countries": raw["iso3"].nunique(), "ages": raw["age_group"].nunique(), "sexes": raw["sex"].nunique()})
    return frames, qa


def adm1_input(scenario: str, year: int) -> Path:
    return BMI_ADM1 / scenario / str(year) / f"BMI7_{scenario}_{year}_ADM1_age_sex_adults25plus_v2.parquet"


def process_adm1(scenario: str, year: int, lookup):
    path = adm1_input(scenario, year)
    raw = pd.read_parquet(path)
    raw = raw[raw["adult_population"].gt(0)].copy()
    raw = add_risk_components(raw, lookup)
    frames = {
        "age_sex": raw[ADM1_ID + ["sex", "age_group", "adult_population", "probability_sum"] + [f"weighted_rr_{x}" for x in ENDPOINTS] + ["paf", "paf_rr_lower", "paf_rr_upper"] + [f"risk_{part}_{x}" for x in ENDPOINTS for part in ("excess_numerator", "denominator")]].copy(),
        "age": aggregate_risk(raw, ADM1_ID + ["age_group"]),
        "sex": aggregate_risk(raw, ADM1_ID + ["sex"]),
        "overall": aggregate_risk(raw, ADM1_ID),
        "spatial_global": aggregate_risk(raw.assign(global_scope="Spatial support"), ["scenario_code", "scenario", "year", "global_scope"]),
    }
    spatial_national = aggregate_risk(raw, ["scenario_code", "scenario", "year", "adm0_a3"])
    qa = validate_frame("adm1_age_sex", frames["age_sex"], probability_column=True)
    qa.update({"input": str(path), "input_sha256": sha256(path), "adm1_zones": raw["adm1_uid"].nunique(), "spatial_countries": raw["adm0_a3"].nunique()})
    return frames, spatial_national, qa


def raster_paths(root: Path, scenario: str, year: int, resolution: str):
    if resolution == "0p5deg":
        base = root / scenario / str(year)
        return base / "bmi7_probability_age_sex_0p5deg_v1.tif", base / "adult_population_age_sex_0p5deg_v1.tif"
    base = root / scenario / str(year)
    return base / f"probability_age_sex_{scenario}_{year}_0p1deg_v1.tif", base / f"population_age_sex_{scenario}_{year}_0p1deg_v1.tif"


def expected_descriptions():
    population = [f"{sex}__{age.replace('-', '_').replace('+', 'plus')}" for sex in SEXES for age in AGES]
    probability = [f"{sex}__{age.replace('-', '_').replace('+', 'plus')}__{category}" for sex in SEXES for age in AGES for category in CATEGORIES]
    return population, probability


def make_profile(source, count: int, compression: str, bigtiff: str):
    profile = source.profile.copy()
    profile.update(driver="GTiff", count=count, dtype="float32", nodata=NODATA, tiled=True, blockxsize=512, blockysize=512, compress=compression, predictor=3, BIGTIFF=bigtiff, NUM_THREADS="ALL_CPUS")
    if compression == "ZSTD":
        profile["zstd_level"] = 1
    return profile


def write_full_raster(path: Path, source, arrays: list[np.ndarray], descriptions: list[str], compression="LZW"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **make_profile(source, len(arrays), compression, "IF_SAFER")) as dst:
        for index, (array, description) in enumerate(zip(arrays, descriptions), start=1):
            dst.write(np.where(np.isfinite(array), array, NODATA).astype(np.float32), index)
            dst.set_band_description(index, description)
        dst.update_tags(DISEASE="Colon and rectum cancer", RR_SOURCE="WCRF/AICR", UNCERTAINTY="RR-only for lower/upper; BMI uncertainty not propagated")


def process_0p5(scenario: str, year: int, lookup):
    probability_path, population_path = raster_paths(BMI_05, scenario, year, "0p5deg")
    output_dir = OUTPUT_ROOT / "data" / "0p5deg" / scenario / str(year)
    pop_desc, prob_desc = expected_descriptions()
    with rasterio.open(probability_path) as psrc, rasterio.open(population_path) as nsrc:
        if psrc.count != 210 or nsrc.count != 30 or list(psrc.descriptions) != prob_desc or list(nsrc.descriptions) != pop_desc:
            raise RuntimeError("0.5-degree raster band contract mismatch")
        probabilities = psrc.read(masked=True).astype(np.float64).reshape(30, 7, psrc.height, psrc.width)
        population = nsrc.read(masked=True).astype(np.float64)
        valid = (~np.ma.getmaskarray(population)) & (population.filled(0) > 0) & (~np.any(np.ma.getmaskarray(probabilities), axis=1))
        closure = np.sum(probabilities.filled(np.nan), axis=1)
        closure_error = float(np.nanmax(np.where(valid, np.abs(closure - 1.0), np.nan)))
        if closure_error > CLOSURE_TOLERANCE:
            raise RuntimeError("0.5-degree probability closure failed")
        rr_matrix = {endpoint: np.vstack([lookup[sex][endpoint] for sex in SEXES for _ in AGES]) for endpoint in ENDPOINTS}
        outputs = {}
        global_values = {}
        for endpoint in ENDPOINTS:
            weighted_rr = np.einsum("skhw,sk->shw", probabilities.filled(0), rr_matrix[endpoint], optimize=True)
            numerator = np.where(valid, population.filled(0) * (weighted_rr - 1.0), 0.0)
            denominator = np.where(valid, population.filled(0) * weighted_rr, 0.0)
            paf = safe_divide(numerator, denominator)
            overall = safe_divide(numerator.sum(axis=0), denominator.sum(axis=0))
            name = "point" if endpoint == "point" else f"rr_{endpoint}"
            age_sex_path = output_dir / f"CRC_PAF_{scenario}_{year}_0p5deg_age_sex_{name}_v1.tif"
            overall_path = output_dir / f"CRC_PAF_{scenario}_{year}_0p5deg_overall_{name}_v1.tif"
            write_full_raster(age_sex_path, psrc, list(paf), pop_desc)
            write_full_raster(overall_path, psrc, [overall], ["Both__25plus"])
            outputs[f"age_sex_{name}"] = age_sex_path
            outputs[f"overall_{name}"] = overall_path
            global_values[endpoint] = float(numerator.sum() / denominator.sum())
        finite_point = safe_divide(np.where(valid, population.filled(0) * (np.einsum("skhw,sk->shw", probabilities.filled(0), rr_matrix["point"], optimize=True) - 1), 0), np.where(valid, population.filled(0) * np.einsum("skhw,sk->shw", probabilities.filled(0), rr_matrix["point"], optimize=True), 0))
        values = finite_point[np.isfinite(finite_point)]
    qa = {"probability_input": str(probability_path), "population_input": str(population_path), "probability_closure_max_abs_error": closure_error, "valid_age_sex_cells": int(valid.sum()), "paf_point_min": float(values.min()), "paf_point_max": float(values.max()), "global_paf": global_values, "outputs": {k: str(v) for k, v in outputs.items()}}
    return outputs, qa


def process_0p1(scenario: str, year: int, lookup):
    probability_path, population_path = raster_paths(BMI_01, scenario, year, "0p1deg")
    output_dir = OUTPUT_ROOT / "data" / "0p1deg" / scenario / str(year)
    age_sex_path = output_dir / f"CRC_PAF_{scenario}_{year}_0p1deg_age_sex_point_v1.tif"
    overall_path = output_dir / f"CRC_PAF_{scenario}_{year}_0p1deg_overall_point_v1.tif"
    output_dir.mkdir(parents=True, exist_ok=True)
    pop_desc_base, prob_desc = expected_descriptions()
    pop_desc = [f"{description}__population" for description in pop_desc_base]
    rr_matrix = np.vstack([lookup[sex]["point"] for sex in SEXES for _ in AGES]).astype(np.float32)
    closure_max = 0.0
    paf_min = math.inf
    paf_max = -math.inf
    global_num = 0.0
    global_den = 0.0
    valid_count = 0
    with rasterio.open(probability_path) as psrc, rasterio.open(population_path) as nsrc:
        if psrc.count != 210 or nsrc.count != 30 or list(psrc.descriptions) != prob_desc or list(nsrc.descriptions) != pop_desc:
            raise RuntimeError("0.1-degree raster band contract mismatch")
        with rasterio.open(age_sex_path, "w", **make_profile(psrc, 30, "ZSTD", "YES")) as age_dst, rasterio.open(overall_path, "w", **make_profile(psrc, 1, "ZSTD", "IF_SAFER")) as overall_dst:
            for index, description in enumerate(pop_desc_base, start=1):
                age_dst.set_band_description(index, description)
            overall_dst.set_band_description(1, "Both__25plus")
            tags = {"DISEASE": "Colon and rectum cancer", "RR_SOURCE": "WCRF/AICR", "ESTIMATE": "point", "REFERENCE_BMI": "22.5 kg/m2 study-defined; not WCRF TMREL"}
            age_dst.update_tags(**tags); overall_dst.update_tags(**tags)
            for _, window in psrc.block_windows(1):
                probabilities = psrc.read(window=window, masked=False, out_dtype="float32").reshape(30, 7, int(window.height), int(window.width))
                population = nsrc.read(window=window, masked=False, out_dtype="float32")
                valid = (population != nsrc.nodata) & np.isfinite(population) & (population > 0) & np.all((probabilities != psrc.nodata) & np.isfinite(probabilities), axis=1)
                closure = probabilities.sum(axis=1, dtype=np.float64)
                if valid.any():
                    closure_max = max(closure_max, float(np.max(np.abs(closure[valid] - 1.0))))
                    valid_count += int(valid.sum())
                weighted_rr = np.einsum("skhw,sk->shw", probabilities, rr_matrix, optimize=True)
                numerator = np.where(valid, population * (weighted_rr - 1.0), 0.0)
                denominator = np.where(valid, population * weighted_rr, 0.0)
                paf = safe_divide(numerator, denominator)
                overall = safe_divide(numerator.sum(axis=0), denominator.sum(axis=0))
                finite = paf[np.isfinite(paf)]
                if finite.size:
                    paf_min = min(paf_min, float(finite.min())); paf_max = max(paf_max, float(finite.max()))
                global_num += float(numerator.sum(dtype=np.float64)); global_den += float(denominator.sum(dtype=np.float64))
                age_dst.write(np.where(np.isfinite(paf), paf, NODATA).astype(np.float32), window=window)
                overall_dst.write(np.where(np.isfinite(overall), overall, NODATA).astype(np.float32), 1, window=window)
    if closure_max > CLOSURE_TOLERANCE:
        raise RuntimeError("0.1-degree probability closure failed")
    outputs = {"age_sex_point": age_sex_path, "overall_point": overall_path}
    qa = {"probability_input": str(probability_path), "population_input": str(population_path), "probability_closure_max_abs_error": closure_max, "valid_age_sex_cells": valid_count, "paf_point_min": paf_min, "paf_point_max": paf_max, "global_paf_point": global_num / global_den, "outputs": {k: str(v) for k, v in outputs.items()}}
    return outputs, qa


def distribution(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return {"n": len(x), "min": float(x.min()), "median": float(np.median(x)), "mean": float(x.mean()), "p95": float(np.quantile(x, 0.95)), "max": float(x.max())}


def save_tables(scenario: str, year: int, national, adm1):
    outputs = []
    for geography, frames in (("national", national), ("adm1", adm1)):
        output_dir = OUTPUT_ROOT / "data" / geography / scenario / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)
        for level in ("age_sex", "age", "sex", "overall"):
            path = output_dir / f"CRC_PAF_{scenario}_{year}_{geography}_{level}_WCRF_primary_v1.parquet"
            frames[level].to_parquet(path, index=False)
            outputs.append(path)
    return outputs


def compare_national(national_overall: pd.DataFrame, spatial_national: pd.DataFrame):
    spatial_national = spatial_national.copy()
    spatial_national["adm0_a3_match"] = spatial_national["adm0_a3"].replace({"PSX": "PSE", "SDS": "SSD"})
    pair = national_overall.merge(spatial_national, left_on="iso3", right_on="adm0_a3_match", suffixes=("_anchor", "_spatial"), validate="one_to_one")
    differences = np.abs(pair["paf_anchor"] - pair["paf_spatial"])
    return {"matched_spatial_countries": len(pair), "max_abs_paf_difference": float(differences.max()), "mean_abs_paf_difference": float(differences.mean())}


def run_task(scenario: str, year: int):
    started = time.time()
    state_path = OUTPUT_ROOT / "state" / scenario / f"{year}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"status": "running", "scenario": scenario, "year": year}, indent=2) + "\n", encoding="utf-8")
    lookup = load_rr()
    national, national_qa = process_national(scenario, year, lookup)
    adm1, spatial_national, adm1_qa = process_adm1(scenario, year, lookup)
    outputs = save_tables(scenario, year, national, adm1)
    outputs05, qa05 = process_0p5(scenario, year, lookup); outputs.extend(outputs05.values())
    outputs01, qa01 = process_0p1(scenario, year, lookup); outputs.extend(outputs01.values())
    national_spatial = compare_national(national["overall"], spatial_national)
    global_row = national["global_overall"].iloc[0]
    sex = national["global_sex"].set_index("sex")
    summary = {
        "scenario_code": scenario, "scenario": SCENARIOS[scenario], "year": year,
        "global_paf": float(global_row["paf"]), "global_paf_rr_lower": float(global_row["paf_rr_lower"]), "global_paf_rr_upper": float(global_row["paf_rr_upper"]),
        "female_paf": float(sex.loc["Female", "paf"]), "male_paf": float(sex.loc["Male", "paf"]),
        "country": distribution(national["overall"]["paf"]), "adm1": distribution(adm1["overall"]["paf"]),
        "adm1_age_sex": distribution(adm1["age_sex"]["paf"]),
    }
    global_age_path = OUTPUT_ROOT / "tables" / "task_global_age" / scenario / f"{year}.csv"
    global_age_path.parent.mkdir(parents=True, exist_ok=True)
    national["global_age"].to_csv(global_age_path, index=False, encoding="utf-8-sig"); outputs.append(global_age_path)
    qa = {
        "status": "PASS", "scenario": scenario, "year": year, "RR_crosswalk_sha256": sha256(RR_PATH),
        "burden_calculated": False, "national": national_qa, "adm1": adm1_qa, "raster_0p5": qa05, "raster_0p1": qa01,
        "national_vs_adm1": national_spatial,
        "cross_scale_global_abs_error_0p1_vs_0p5": abs(qa01["global_paf_point"] - qa05["global_paf"]["point"]),
        "cross_scale_global_abs_error_0p5_vs_adm1": abs(qa05["global_paf"]["point"] - float(adm1["spatial_global"].iloc[0]["paf"])),
        "summary": summary,
    }
    if qa["cross_scale_global_abs_error_0p1_vs_0p5"] > SPATIAL_TOLERANCE or qa["cross_scale_global_abs_error_0p5_vs_adm1"] > SPATIAL_TOLERANCE or national_spatial["max_abs_paf_difference"] > NATIONAL_TOLERANCE:
        failure_path = OUTPUT_ROOT / "qa" / "tasks" / scenario / f"{year}_pre_failure_QA.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        state_path.write_text(json.dumps({"status": "FAIL_CROSS_SCALE", "scenario": scenario, "year": year, "qa": str(failure_path)}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"Cross-scale aggregation QA failed: {scenario}/{year}; "
            f"0p1_vs_0p5={qa['cross_scale_global_abs_error_0p1_vs_0p5']:.12g}; "
            f"0p5_vs_adm1={qa['cross_scale_global_abs_error_0p5_vs_adm1']:.12g}; "
            f"national_vs_adm1={national_spatial['max_abs_paf_difference']:.12g}"
        )
    qa_dir = OUTPUT_ROOT / "qa" / "tasks" / scenario
    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_path = qa_dir / f"{year}.json"
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); outputs.append(qa_path)
    summary_path = OUTPUT_ROOT / "tables" / "task_summary" / scenario / f"{year}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); outputs.append(summary_path)
    checksum_path = qa_dir / f"{year}_outputs_SHA256.csv"
    pd.DataFrame([{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in outputs]).to_csv(checksum_path, index=False, encoding="utf-8-sig")
    state = {"status": "PASS", "scenario": scenario, "year": year, "elapsed_seconds": round(time.time() - started, 3), "qa": str(qa_path), "checksum_manifest": str(checksum_path)}
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    complete = OUTPUT_ROOT / "state" / scenario / f"{year}.COMPLETE.flag"
    complete.write_text("CRC_PAF_TASK_PASS\n", encoding="ascii")
    print(json.dumps(state, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--year", type=int, choices=range(2025, 2051), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_task(args.scenario, args.year)
