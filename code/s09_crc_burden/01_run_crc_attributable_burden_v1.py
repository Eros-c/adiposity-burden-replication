#!/usr/bin/env python3
"""Audit Foresight CRC reference burden and compute national attributable deaths/DALYs."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(_cfg.RESULTS)
FORESIGHT = Path(_cfg.DATA / "GBD2021_foresight_exports" / "foresight_crc_reference_2024_2050.csv")
PAF_ROOT = PROJECT / "08_colorectal_cancer_paf" / "full_annual_four_ssp_wcrf_primary_v1_20260910"
LOCATION_SOURCE = PROJECT / "06_future_bmi_2025_2050" / "phase6_global_ssp245_contract_preflight_v3_20260830" / "tables" / "Phase6_anchor_location_iso3_crosswalk_v3.csv"
OUT = PROJECT / "09_colorectal_cancer_burden" / "crc_attributable_deaths_dalys_2025_2050_four_ssp_v1_20260910"

SCENARIOS = ("ssp126", "ssp245", "ssp370", "ssp585")
SCENARIO_LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
YEARS = tuple(range(2025, 2051))
KEY_YEARS = (2025, 2030, 2035, 2040, 2045, 2050)
AGES = ("25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+")
AGE_ID_MAP = {10: "25-29", 11: "30-34", 12: "35-39", 13: "40-44", 14: "45-49", 15: "50-54", 16: "55-59", 17: "60-64", 18: "65-69", 19: "70-74", 20: "75-79", 30: "80-84", 31: "85-89", 32: "90-94", 235: "95+"}
SEX_ID_MAP = {1: "Male", 2: "Female", 3: "Both"}
MEASURE_ID_MAP = {1: "Deaths", 2: "DALYs"}
EXPECTED_PAF_ROWS = 4 * 26 * 204 * 15 * 2
EXPECTED_BURDEN_ROWS = 26 * 204 * 15 * 2 * 2
EXPECTED_JOIN_ROWS = EXPECTED_PAF_ROWS * 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def audit_and_crosswalks():
    data = pd.read_csv(FORESIGHT, low_memory=False)
    required = {"cause_id", "cause_name", "age_id", "age_name", "sex_id", "sex_name", "year", "scenario_id", "scenario_name", "measure_id", "measure_name", "metric_id", "metric_name", "location_id", "location_name", "value", "lower", "upper"}
    if set(data.columns) != required:
        missing = required.difference(data.columns)
        extra = set(data.columns).difference(required)
        raise RuntimeError(f"Foresight schema mismatch; missing={missing}; extra={extra}")
    key = ["cause_id", "age_id", "sex_id", "year", "scenario_id", "measure_id", "metric_id", "location_id"]
    if data.duplicated(key).any() or data.isna().any().any():
        raise RuntimeError("Foresight duplicate key or missing value")
    if set(data["cause_id"]) != {441} or set(data["scenario_id"]) != {2} or set(data["measure_id"]) != {1, 2} or set(data["metric_id"]) != {1, 3}:
        raise RuntimeError("Foresight cause/scenario/measure/metric contract mismatch")
    if set(data["year"]) != set(range(2024, 2051)) or set(data["location_id"]) != set(data["location_id"].unique()) or data["location_id"].nunique() != 204:
        raise RuntimeError("Foresight year/location coverage mismatch")
    if set(data["age_id"]) != set(AGE_ID_MAP) or set(data["sex_id"]) != set(SEX_ID_MAP):
        raise RuntimeError("Foresight age/sex contract mismatch")
    if not np.isfinite(data[["value", "lower", "upper"]].to_numpy(float)).all() or (data[["value", "lower", "upper"]] < 0).any().any():
        raise RuntimeError("Foresight burden contains non-finite or negative values")
    if (data["lower"] > data["value"]).any() or (data["upper"] < data["value"]).any():
        raise RuntimeError("Foresight interval ordering failed")

    location = pd.read_csv(LOCATION_SOURCE)
    location = location[["location_id", "location_name", "iso3", "iso3_original", "crosswalk_remediated"]].copy()
    foresight_locations = data[["location_id", "location_name"]].drop_duplicates().rename(columns={"location_name": "foresight_location_name"})
    location = foresight_locations.merge(location.rename(columns={"location_name": "crosswalk_location_name"}), on="location_id", how="outer", validate="one_to_one", indicator=True)
    if len(location) != 204 or not location["_merge"].eq("both").all() or location["iso3"].nunique() != 204 or location["iso3"].duplicated().any():
        raise RuntimeError("Location crosswalk is not complete and one-to-one")
    location = location.drop(columns="_merge")

    age_source = data[["age_id", "age_name"]].drop_duplicates().sort_values("age_id")
    age = age_source.assign(age_group=age_source["age_id"].map(AGE_ID_MAP), age_order=age_source["age_id"].map({age_id: AGES.index(label) for age_id, label in AGE_ID_MAP.items()}))
    if age["age_group"].isna().any() or age["age_group"].nunique() != 15:
        raise RuntimeError("Age crosswalk failed")
    sex_source = data[["sex_id", "sex_name"]].drop_duplicates().sort_values("sex_id")
    sex = sex_source.assign(sex=sex_source["sex_id"].map(SEX_ID_MAP))
    if sex["sex"].isna().any():
        raise RuntimeError("Sex crosswalk failed")

    both = data[(data["metric_id"].eq(1)) & data["sex_id"].eq(3)].rename(columns={"value": "both_value"})
    mf = data[(data["metric_id"].eq(1)) & data["sex_id"].isin([1, 2])].groupby(["location_id", "age_id", "year", "measure_id"], as_index=False)["value"].sum().rename(columns={"value": "male_plus_female"})
    both_qa = both.merge(mf, on=["location_id", "age_id", "year", "measure_id"], validate="one_to_one")
    both_qa["absolute_difference"] = np.abs(both_qa["both_value"] - both_qa["male_plus_female"])
    both_qa["relative_difference"] = both_qa["absolute_difference"] / np.maximum(np.abs(both_qa["both_value"]), 1e-12)

    core = data[(data["year"].between(2025, 2050)) & data["sex_id"].isin([1, 2]) & data["metric_id"].eq(1)].copy()
    if len(core) != EXPECTED_BURDEN_ROWS or core.duplicated(["location_id", "age_id", "sex_id", "year", "measure_id"]).any():
        raise RuntimeError("Core Foresight Number burden is incomplete")

    audit = {
        "status": "PASS",
        "foresight_file": str(FORESIGHT), "foresight_sha256": sha256(FORESIGHT), "rows": len(data),
        "cause": {"id": 441, "name": sorted(data["cause_name"].unique().tolist())},
        "scenario": {"id": 2, "name": sorted(data["scenario_name"].unique().tolist())},
        "years": [int(data["year"].min()), int(data["year"].max())], "locations": data["location_id"].nunique(),
        "ages": age[["age_id", "age_name", "age_group"]].to_dict("records"),
        "sexes": sex.to_dict("records"), "measures": data[["measure_id", "measure_name"]].drop_duplicates().sort_values("measure_id").to_dict("records"),
        "metrics": data[["metric_id", "metric_name"]].drop_duplicates().sort_values("metric_id").to_dict("records"),
        "core_rows": len(core), "core_expected_rows": EXPECTED_BURDEN_ROWS,
        "both_sex_number_QA": {"rows": len(both_qa), "maximum_absolute_difference": float(both_qa["absolute_difference"].max()), "maximum_relative_difference": float(both_qa["relative_difference"].max())},
        "all_age_QA": "not applicable; no all-age records in the input",
        "location_crosswalk_rows": len(location), "location_crosswalk_one_to_one": True,
        "location_name_language_difference": "Foresight names are Chinese; existing crosswalk names are English. Matching uses exact location_id only.",
        "highest_age_mapping": "age_id 235 and source label >95岁 are normalized to study label 95+ without modifying the source file",
    }
    return data, core, location, age, sex, both_qa, audit


def load_paf():
    frames = []
    for scenario in SCENARIOS:
        for year in YEARS:
            path = PAF_ROOT / "data" / "national" / scenario / str(year) / f"CRC_PAF_{scenario}_{year}_national_age_sex_WCRF_primary_v1.parquet"
            columns = [
                "scenario_code", "scenario", "year", "iso3", "sex", "age_group", "adult_population",
                "weighted_rr_point", "weighted_rr_lower", "weighted_rr_upper",
                "paf", "paf_rr_lower", "paf_rr_upper",
            ]
            frame = pd.read_parquet(path, columns=columns)
            frame["paf_source_path"] = str(path)
            frames.append(frame)
    paf = pd.concat(frames, ignore_index=True)
    key = ["scenario_code", "year", "iso3", "age_group", "sex"]
    if len(paf) != EXPECTED_PAF_ROWS or paf.duplicated(key).any():
        raise RuntimeError("PAF input is incomplete or duplicated")
    recovery_rows = []
    any_recovered = np.zeros(len(paf), dtype=bool)
    recovery_map = {
        "paf": "weighted_rr_point",
        "paf_rr_lower": "weighted_rr_lower",
        "paf_rr_upper": "weighted_rr_upper",
    }
    for paf_column, rr_column in recovery_map.items():
        missing = paf[paf_column].isna()
        invalid_missing = missing & paf["adult_population"].gt(0)
        if invalid_missing.any():
            raise RuntimeError(f"PAF missing for positive-population cells: {paf_column}; rows={int(invalid_missing.sum())}")
        recoverable = missing & paf["adult_population"].eq(0) & np.isfinite(paf[rr_column]) & paf[rr_column].ge(1)
        if (missing & ~recoverable).any():
            raise RuntimeError(f"PAF missing values cannot be recovered from frozen weighted RR: {paf_column}")
        paf.loc[recoverable, paf_column] = (paf.loc[recoverable, rr_column] - 1.0) / paf.loc[recoverable, rr_column]
        any_recovered |= recoverable.to_numpy()
        recovery_rows.append({"field": paf_column, "recovered_rows": int(recoverable.sum())})
    values = paf[["paf", "paf_rr_lower", "paf_rr_upper"]].to_numpy(float)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError("PAF contains non-finite or out-of-range values")
    if not ((paf["paf_rr_lower"] <= paf["paf"]) & (paf["paf"] <= paf["paf_rr_upper"])).all():
        raise RuntimeError("PAF RR-only interval ordering failed")
    recovery = {
        "policy": "For zero-SSP-population cells only, missing PAF was recovered from the finite weighted RR stored in the same frozen PAF file using (weighted_RR-1)/weighted_RR. BMI was not reread and PAF production was not rerun.",
        "unique_cells_with_any_recovery": int(any_recovered.sum()),
        "field_counts": recovery_rows,
        "zero_population_rows": int(paf["adult_population"].eq(0).sum()),
    }
    return paf, recovery


def match_and_compute(core, location, age, sex, paf):
    burden = core.merge(location[["location_id", "iso3"]], on="location_id", validate="many_to_one")
    burden = burden.merge(age[["age_id", "age_group"]], on="age_id", validate="many_to_one")
    burden = burden.merge(sex[["sex_id", "sex"]], on="sex_id", validate="many_to_one")
    burden["measure"] = burden["measure_id"].map(MEASURE_ID_MAP)
    if burden[["iso3", "age_group", "sex", "measure"]].isna().any().any():
        raise RuntimeError("Crosswalk produced missing identifiers")
    paf_key = ["scenario_code", "year", "iso3", "age_group", "sex"]
    burden_key = ["year", "iso3", "age_group", "sex", "measure_id"]
    if paf.duplicated(paf_key).any():
        raise RuntimeError("PAF is not unique by scenario-location-age-sex-year")
    if burden.duplicated(burden_key).any():
        raise RuntimeError("Foresight burden is not unique by location-age-sex-year-measure")
    merged = paf.merge(
        burden,
        on=["year", "iso3", "age_group", "sex"],
        how="outer",
        validate="many_to_many",
        indicator=True,
        suffixes=("_paf", "_burden"),
    )
    counts = merged["_merge"].value_counts().to_dict()
    if len(merged) != EXPECTED_JOIN_ROWS or not merged["_merge"].eq("both").all():
        raise RuntimeError(f"PAF-Foresight match incomplete: {counts}; rows={len(merged)}")
    merged = merged.drop(columns="_merge")
    if not (merged["scenario"] == merged["scenario_code"].map(SCENARIO_LABELS)).all():
        raise RuntimeError("Scenario label mismatch")
    merged = merged.rename(columns={
        "value": "total_burden", "lower": "total_burden_lower", "upper": "total_burden_upper",
        "age_name": "foresight_age_name", "sex_name": "foresight_sex_name",
    })
    merged["attributable_burden"] = merged["paf"] * merged["total_burden"]
    merged["attrib_if_paf_rr_lower"] = merged["paf_rr_lower"] * merged["total_burden"]
    merged["attrib_if_paf_rr_upper"] = merged["paf_rr_upper"] * merged["total_burden"]
    merged["attrib_if_burden_lower"] = merged["paf"] * merged["total_burden_lower"]
    merged["attrib_if_burden_upper"] = merged["paf"] * merged["total_burden_upper"]
    merged["attributable_fraction"] = np.divide(merged["attributable_burden"], merged["total_burden"], out=np.zeros(len(merged)), where=merged["total_burden"].to_numpy(float) > 0)
    attr = merged[["attributable_burden", "attrib_if_paf_rr_lower", "attrib_if_paf_rr_upper", "attrib_if_burden_lower", "attrib_if_burden_upper"]].to_numpy(float)
    total = merged["total_burden"].to_numpy(float)
    if not np.isfinite(attr).all() or attr.min() < -1e-12 or (merged["attributable_burden"] > merged["total_burden"] + 1e-9).any():
        raise RuntimeError("Attributable burden range QA failed")
    keep = [
        "scenario_code", "scenario", "year", "location_id", "location_name", "iso3", "age_id", "foresight_age_name", "age_group",
        "sex_id", "foresight_sex_name", "sex", "measure_id", "measure_name", "measure", "metric_id", "metric_name",
        "total_burden", "total_burden_lower", "total_burden_upper", "paf", "paf_rr_lower", "paf_rr_upper", "attributable_fraction",
        "attributable_burden", "attrib_if_paf_rr_lower", "attrib_if_paf_rr_upper", "attrib_if_burden_lower", "attrib_if_burden_upper",
    ]
    return merged[keep].copy(), counts


SUM_COLUMNS = ["total_burden", "total_burden_lower", "total_burden_upper", "attributable_burden", "attrib_if_paf_rr_lower", "attrib_if_paf_rr_upper", "attrib_if_burden_lower", "attrib_if_burden_upper"]


def aggregate(frame, groups):
    result = frame.groupby(groups, observed=True, dropna=False, sort=False)[SUM_COLUMNS].sum().reset_index()
    result["attributable_fraction"] = np.divide(result["attributable_burden"], result["total_burden"], out=np.zeros(len(result)), where=result["total_burden"].to_numpy(float) > 0)
    return result


def temporal_flags(global_frame, country_frame):
    rows = []
    for level, data, ids in (("global", global_frame, ["scenario_code", "measure"]), ("country", country_frame, ["scenario_code", "iso3", "measure"])):
        work = data.sort_values(ids + ["year"]).copy()
        work["annual_relative_change"] = work.groupby(ids, observed=True)["attributable_burden"].pct_change(fill_method=None)
        for scenario, group in work.groupby("scenario_code"):
            absolute = group["annual_relative_change"].abs().replace([np.inf, -np.inf], np.nan).dropna()
            median = float(absolute.median()) if len(absolute) else 0.0
            mad = float(np.median(np.abs(absolute - median))) if len(absolute) else 0.0
            threshold = max(0.25, median + 10 * mad)
            flagged = group[group["annual_relative_change"].abs() > threshold]
            for item in flagged.itertuples():
                row = {"level": level, "scenario_code": scenario, "year": int(item.year), "measure": item.measure, "annual_relative_change": float(item.annual_relative_change), "threshold": threshold}
                if level == "country":
                    row["iso3"] = item.iso3
                rows.append(row)
    return pd.DataFrame(rows)


def checksum_manifest(paths):
    return pd.DataFrame([{"path": str(path), "relative_path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths])


def main():
    started = time.time()
    for name in ("data", "tables", "qa", "reports", "scripts", "contracts"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / "RUNNING.flag").write_text("CRC_ATTRIBUTABLE_BURDEN_RUNNING\n", encoding="ascii")

    source, core, location, age, sex, both_qa, audit = audit_and_crosswalks()
    location_path = OUT / "contracts" / "GBD_location_id_ISO3_PAF_crosswalk_v1.csv"
    age_path = OUT / "contracts" / "GBD_age_PAF_crosswalk_v1.csv"
    sex_path = OUT / "contracts" / "GBD_sex_PAF_crosswalk_v1.csv"
    location.to_csv(location_path, index=False, encoding="utf-8-sig")
    age.to_csv(age_path, index=False, encoding="utf-8-sig")
    sex.to_csv(sex_path, index=False, encoding="utf-8-sig")
    both_qa_path = OUT / "qa" / "Foresight_both_sex_number_QA_v1.parquet"
    both_qa.to_parquet(both_qa_path, index=False)
    audit_path = OUT / "qa" / "Foresight_CRC_input_matching_audit_v1.json"
    write_json(audit_path, audit)
    if audit["status"] != "PASS":
        raise RuntimeError("Foresight audit did not pass")

    paf, paf_recovery = load_paf()
    detailed, match_counts = match_and_compute(core, location, age, sex, paf)
    detailed_path = OUT / "data" / "CRC_attributable_burden_country_age_sex_annual_four_SSP_v1.parquet"
    detailed.to_parquet(detailed_path, index=False)

    base_country = ["scenario_code", "scenario", "year", "location_id", "location_name", "iso3", "measure_id", "measure_name", "measure"]
    country_sex = aggregate(detailed, base_country + ["sex_id", "sex"])
    country_age = aggregate(detailed, base_country + ["age_id", "age_group"])
    country_overall = aggregate(detailed, base_country)
    global_age_sex = aggregate(detailed.assign(global_scope="Global"), ["scenario_code", "scenario", "year", "global_scope", "age_id", "age_group", "sex_id", "sex", "measure_id", "measure_name", "measure"])
    global_sex = aggregate(detailed.assign(global_scope="Global"), ["scenario_code", "scenario", "year", "global_scope", "sex_id", "sex", "measure_id", "measure_name", "measure"])
    global_age = aggregate(detailed.assign(global_scope="Global"), ["scenario_code", "scenario", "year", "global_scope", "age_id", "age_group", "measure_id", "measure_name", "measure"])
    global_overall = aggregate(detailed.assign(global_scope="Global"), ["scenario_code", "scenario", "year", "global_scope", "measure_id", "measure_name", "measure"])

    output_tables = {
        "country_sex": country_sex,
        "country_age": country_age,
        "country_overall": country_overall,
        "global_age_sex": global_age_sex,
        "global_sex": global_sex,
        "global_age": global_age,
        "global_overall": global_overall,
    }
    paths = [location_path, age_path, sex_path, both_qa_path, audit_path, detailed_path]
    for name, frame in output_tables.items():
        suffix = "parquet" if name in {"country_sex", "country_age", "global_age_sex"} else "csv"
        path = OUT / "tables" / f"CRC_attributable_burden_{name}_annual_four_SSP_v1.{suffix}"
        if suffix == "parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)
    global_alias_path = OUT / "tables" / "CRC_attributable_burden_global_annual_four_SSP_v1.csv"
    global_overall.to_csv(global_alias_path, index=False, encoding="utf-8-sig")
    paths.append(global_alias_path)
    key_summary = global_overall[global_overall["year"].isin(KEY_YEARS)].copy()
    key_path = OUT / "tables" / "CRC_attributable_burden_key_year_summary_v1.csv"
    key_summary.to_csv(key_path, index=False, encoding="utf-8-sig"); paths.append(key_path)

    top_burden = country_overall[country_overall["year"].eq(2050)].sort_values(["scenario_code", "measure", "attributable_burden"], ascending=[True, True, False]).groupby(["scenario_code", "measure"], as_index=False, group_keys=False).head(10)
    top_fraction = country_overall[country_overall["year"].eq(2050)].sort_values(["scenario_code", "measure", "attributable_fraction"], ascending=[True, True, False]).groupby(["scenario_code", "measure"], as_index=False, group_keys=False).head(10)
    top_burden_path = OUT / "tables" / "CRC_attributable_burden_country_top10_2050_v1.csv"
    top_fraction_path = OUT / "tables" / "CRC_attributable_fraction_country_top10_2050_v1.csv"
    top_burden.to_csv(top_burden_path, index=False, encoding="utf-8-sig"); top_fraction.to_csv(top_fraction_path, index=False, encoding="utf-8-sig")
    paths += [top_burden_path, top_fraction_path]

    flags = temporal_flags(global_overall, country_overall)
    flags_path = OUT / "qa" / "CRC_attributable_burden_temporal_jump_flags_v1.csv"
    flags.to_csv(flags_path, index=False, encoding="utf-8-sig"); paths.append(flags_path)

    # Aggregation is deterministic direct summation; compare re-aggregation from the detailed table.
    check_global = detailed.groupby(["scenario_code", "year", "measure"], as_index=False)["attributable_burden"].sum().rename(columns={"attributable_burden": "detail_sum"})
    check_global = check_global.merge(global_overall[["scenario_code", "year", "measure", "attributable_burden"]], on=["scenario_code", "year", "measure"], validate="one_to_one")
    check_global["absolute_difference"] = np.abs(check_global["detail_sum"] - check_global["attributable_burden"])
    check_global["relative_difference"] = check_global["absolute_difference"] / np.maximum(np.abs(check_global["detail_sum"]), 1e-12)
    aggregation_path = OUT / "qa" / "CRC_attributable_burden_aggregation_QA_v1.csv"
    check_global.to_csv(aggregation_path, index=False, encoding="utf-8-sig"); paths.append(aggregation_path)

    qa = {
        "status": "PASS",
        "foresight_contract_pass": True,
        "location_match": {"expected": 204, "matched": 204, "unmatched": 0, "one_to_one": True},
        "age_sex_year_match": {"paf_rows": len(paf), "paf_expected": EXPECTED_PAF_ROWS, "foresight_core_rows": len(core), "foresight_expected": EXPECTED_BURDEN_ROWS, "joined_rows": len(detailed), "joined_expected": EXPECTED_JOIN_ROWS, "merge_counts": match_counts},
        "paf_range": [float(detailed["paf"].min()), float(detailed["paf"].max())],
        "zero_population_PAF_recovery": paf_recovery,
        "total_burden_range": [float(detailed["total_burden"].min()), float(detailed["total_burden"].max())],
        "attributable_burden_range": [float(detailed["attributable_burden"].min()), float(detailed["attributable_burden"].max())],
        "attributable_le_total": bool((detailed["attributable_burden"] <= detailed["total_burden"] + 1e-9).all()),
        "aggregation_max_absolute_difference": float(check_global["absolute_difference"].max()),
        "aggregation_max_relative_difference": float(check_global["relative_difference"].max()),
        "both_sex_QA": audit["both_sex_number_QA"],
        "temporal_jump_flags": len(flags),
        "uncertainty_fields": "component-wise only; not a joint 95% uncertainty interval",
        "baseline_burden_scenario": "GBD Foresight Reference shared by all four SSP PAF scenarios",
        "spatial_resolution": "national burden only; no ADM1 or grid burden downscaling",
        "PAF_manifest_sha256": sha256(PAF_ROOT / "CRC_PAF_full_formal_manifest_v1.csv"),
        "foresight_sha256": sha256(FORESIGHT),
    }
    qa_path = OUT / "qa" / "CRC_attributable_burden_full_QA_v1.json"
    write_json(qa_path, qa); paths.append(qa_path)
    if qa["status"] != "PASS" or len(detailed) != EXPECTED_JOIN_ROWS or qa["aggregation_max_relative_difference"] > 1e-12:
        raise RuntimeError("Final attributable burden QA failed")

    contract = {
        "status": "FROZEN",
        "date": "2026-09-10",
        "calculation": "country-age-sex PAF multiplied by matching GBD Foresight Reference Number burden",
        "measures": ["Deaths", "DALYs"],
        "year_range": [2025, 2050],
        "years_actual": list(YEARS),
        "SSP_policy": "all four SSPs share the same Foresight Reference baseline; differences arise only from SSP-specific PAF",
        "uncertainty": "component-wise fields only; no joint UI",
        "spatial_policy": "country-level attributable burden; spatial PAF remains separate",
        "prohibited": ["burden downscaling", "new forecasting model", "PAF recalculation", "RR sensitivity", "Figure 5 rendering"],
    }
    contract_path = OUT / "contracts" / "CRC_attributable_burden_contract_v1.json"
    write_json(contract_path, contract); paths.append(contract_path)

    archived_script = OUT / "scripts" / Path(__file__).name
    shutil.copy2(Path(__file__), archived_script)
    paths.append(archived_script)

    # Concise report source tables.
    global_key = global_overall[global_overall["year"].isin([2025, 2050])].copy()
    main = global_key[global_key["scenario_code"].eq("ssp245")]
    report_lines = ["# BMI-attributable CRC deaths and DALYs, 2025-2050", "", "Status: **PASS**", "", "All attributable burden remains at country level. Component-wise uncertainty fields are not joint 95% intervals.", "", "## SSP2-4.5 global results", ""]
    for measure in ("Deaths", "DALYs"):
        subset = main[main["measure"].eq(measure)].set_index("year")
        report_lines.append(f"- {measure}: total burden {subset.loc[2025, 'total_burden']:.3f} in 2025 and {subset.loc[2050, 'total_burden']:.3f} in 2050; attributable burden {subset.loc[2025, 'attributable_burden']:.3f} and {subset.loc[2050, 'attributable_burden']:.3f}, respectively.")
    report_lines += ["", "## QA", "", f"- 204/204 locations matched one-to-one by location_id.", f"- {len(detailed):,}/{EXPECTED_JOIN_ROWS:,} country-age-sex-scenario-measure rows matched.", f"- Male+Female versus Both-sex Number maximum relative difference: {audit['both_sex_number_QA']['maximum_relative_difference']:.3e}.", f"- Aggregation maximum relative difference: {qa['aggregation_max_relative_difference']:.3e}.", f"- Temporal jump flags: {len(flags)}.", "- No CRC burden was downscaled below national level."]
    report_path = OUT / "reports" / "CRC_attributable_burden_report_v1.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8"); paths.append(report_path)

    manifest_path = OUT / "CRC_attributable_burden_formal_manifest_v1.csv"
    manifest = checksum_manifest(paths)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    (OUT / "RUNNING.flag").unlink(missing_ok=True)
    (OUT / "COMPLETE.flag").write_text("CRC_ATTRIBUTABLE_DEATHS_DALYS_COMPLETE\n", encoding="ascii")
    print(json.dumps({"status": "PASS", "rows": len(detailed), "locations": 204, "temporal_flags": len(flags), "elapsed_seconds": round(time.time() - started, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
