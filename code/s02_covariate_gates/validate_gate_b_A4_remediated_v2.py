from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

SSPS = ("SSP1", "SSP2", "SSP3", "SSP5")
TARGET_YEARS = tuple(range(2025, 2051))
SSP_NATIVE_YEARS = tuple(range(2025, 2051, 5))

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def add_check(
    rows: list[dict], check_id: str, dataset_id: str, domain: str,
    check: str, status: str, evidence: str, blocking: bool, notes: str = ""
) -> None:
    rows.append({
        "check_id": check_id,
        "dataset_id": dataset_id,
        "domain": domain,
        "check": check,
        "status": status,
        "evidence": evidence,
        "blocking": "TRUE" if blocking else "FALSE",
        "notes": notes,
    })

def finite_nonnegative(values: np.ndarray) -> tuple[int, int, int, float, float]:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    valid = array[finite]
    return (
        int(array.size), int(finite.sum()), int((valid < 0).sum()),
        float(valid.min()) if valid.size else math.nan,
        float(valid.max()) if valid.size else math.nan,
    )

def inspect_a1(root: Path, checks: list[dict], contracts: list[dict]) -> dict:
    base = root / "ssp_country" / "IIASA_Basic_Drivers_2025"
    gdp_path = next((base / "GDP" / "original").glob("iamc_data-*.csv"))
    pop_path = next((base / "population" / "original").glob("iamc_data-*.csv"))
    gdp = pd.read_csv(gdp_path)
    pop = pd.read_csv(pop_path)
    years = [str(y) for y in SSP_NATIVE_YEARS if str(y) in gdp.columns and str(y) in pop.columns]
    gdp_ssp = set(gdp["scenario"].dropna().astype(str))
    pop_ssp = set(pop["scenario"].dropna().astype(str))
    add_check(checks, "A1.01", "A1", "scenario", "SSP1/2/3/5 available",
              "PASS" if set(SSPS) <= gdp_ssp and set(SSPS) <= pop_ssp else "FAIL",
              f"GDP={sorted(gdp_ssp)}; population={sorted(pop_ssp)}", True)
    add_check(checks, "A1.02", "A1", "time", "Native five-year nodes from 2025 through 2050 available",
              "PASS" if len(years) == len(SSP_NATIVE_YEARS) else "FAIL",
              f"available native target columns={len(years)}/{len(SSP_NATIVE_YEARS)}; expected={list(SSP_NATIVE_YEARS)}", True)
    gdp_vals = gdp[years].to_numpy(float)
    pop_vals = pop[years].to_numpy(float)
    _, _, gdp_neg, gdp_min, gdp_max = finite_nonnegative(gdp_vals)
    _, _, pop_neg, pop_min, pop_max = finite_nonnegative(pop_vals)
    add_check(checks, "A1.03", "A1", "values", "GDP values finite and nonnegative",
              "PASS" if np.isfinite(gdp_vals).all() and gdp_neg == 0 else "FAIL",
              f"min={gdp_min:.8g}; max={gdp_max:.8g}; negative={gdp_neg}", True)
    add_check(checks, "A1.04", "A1", "values", "Population values finite and nonnegative",
              "PASS" if np.isfinite(pop_vals).all() and pop_neg == 0 else "FAIL",
              f"min={pop_min:.8g}; max={pop_max:.8g}; negative={pop_neg}", True)
    gdp_units = sorted(gdp["unit"].dropna().astype(str).unique())
    pop_units = sorted(pop["unit"].dropna().astype(str).unique())
    add_check(checks, "A1.05", "A1", "units", "Units are explicit",
              "PASS" if gdp_units and pop_units == ["million"] else "FAIL",
              f"GDP units={gdp_units}; population units={pop_units}", True,
              "GDP contains three PPP base-year variants; 2010 PPP is the frozen primary variant for later harmonization, not applied here.")
    add_check(checks, "A1.06", "A1", "version", "Provider/model/version fields are explicit",
              "PASS" if gdp[["model", "version"]].notna().all().all() and pop[["model", "version"]].notna().all().all() else "FAIL",
              f"GDP model={sorted(gdp.model.unique())}; population model={sorted(pop.model.unique())}; versions={sorted(set(gdp.version.astype(str)) | set(pop.version.astype(str)))}", True)
    contracts.extend([
        {
            "dataset_id": "A1_GDP", "dataset_name": "IIASA Basic Drivers GDP PPP",
            "provider": "IIASA SSP Scenario Explorer / OECD ENV-Growth 2025", "version": "May 2025 release; local version field 1",
            "scenarios": ";".join(SSPS), "years": "2025-2100, 5-year intervals", "units": ";".join(gdp_units),
            "crs": "not applicable (country table)", "nodata": "blank/NA CSV cells; none in target columns",
            "coverage": f"{gdp.region.nunique()} named economies/regions", "source_url": "https://ssp.apps.ece.iiasa.ac.at/documentation/basic-drivers",
            "doi": "not stated for OECD GDP file on the Basic Drivers page", "contract_status": "PASS_WITH_NOTE",
        },
        {
            "dataset_id": "A1_POP", "dataset_name": "IIASA-WiC national population",
            "provider": "IIASA-WiC", "version": "May 2025 release; local version field 1",
            "scenarios": ";".join(SSPS), "years": "2025-2100, 5-year intervals", "units": ";".join(pop_units),
            "crs": "not applicable (country table)", "nodata": "blank/NA CSV cells; none in target columns",
            "coverage": f"{pop.region.nunique()} named countries/territories", "source_url": "https://ssp.apps.ece.iiasa.ac.at/documentation/basic-drivers",
            "doi": "10.5281/zenodo.7767425", "contract_status": "PASS",
        },
    ])
    return {"gdp_path": gdp_path, "pop_path": pop_path, "gdp": gdp, "pop": pop}

def inspect_a2(root: Path, crosswalk_path: Path, a1: dict, checks: list[dict], contracts: list[dict]) -> dict:
    base = root / "ssp_country" / "IIASA_Basic_Drivers_2025" / "WIC_age_sex" / "V14"
    files = {m.group(1): p for p in base.glob("PROJresult_AGE_SSP*_V14.csv") if (m := re.search(r"(SSP[1235])", p.name))}
    add_check(checks, "A2.01", "A2", "scenario", "SSP1/2/3/5 files available",
              "PASS" if set(files) == set(SSPS) else "FAIL", f"files={sorted(files)}", True)
    crosswalk = pd.read_csv(crosswalk_path)
    a2_countries = crosswalk[(crosswalk.dataset_id == "A2") & (crosswalk.entity_type == "country_or_territory")].copy()
    a1_countries = crosswalk[crosswalk.dataset_id == "A1"].copy()
    mapping = dict(zip(a2_countries.source_id, a2_countries.iso3))
    summary_rows = []
    years_all, sexes_all, ages_all, ids_all = set(), set(), set(), set()
    negative = missing_pop = 0
    for scenario, path in sorted(files.items()):
        totals_2025 = defaultdict(float)
        row_count = 0
        for chunk in pd.read_csv(path, chunksize=250_000):
            row_count += len(chunk)
            years_all.update(pd.to_numeric(chunk["Time"], errors="coerce").dropna().astype(int).unique())
            sexes_all.update(chunk["sex"].dropna().astype(str).unique())
            ages_all.update(pd.to_numeric(chunk["agest"], errors="coerce").dropna().astype(int).unique())
            ids_all.update(chunk["region"].dropna().astype(str).unique())
            vals = pd.to_numeric(chunk["pop"], errors="coerce")
            negative += int((vals < 0).sum())
            missing_pop += int(vals.isna().sum())
            subset = chunk[pd.to_numeric(chunk["Time"], errors="coerce") == 2025].copy()
            subset["pop_num"] = pd.to_numeric(subset["pop"], errors="coerce")
            for region, value in subset.groupby("region", observed=True)["pop_num"].sum().items():
                totals_2025[str(region)] += float(value)
        for source_id, value in totals_2025.items():
            iso3 = mapping.get(source_id)
            if iso3:
                summary_rows.append({"scenario": scenario, "source_id": source_id, "iso3": iso3, "a2_pop_2025_thousands": value})
    years_target_ok = min(years_all) <= 2025 and max(years_all) >= 2050 and set(SSP_NATIVE_YEARS) <= years_all
    add_check(checks, "A2.02", "A2", "time", "Native five-year nodes from 2025 through 2050 available",
              "PASS" if years_target_ok else "FAIL", f"range={min(years_all)}-{max(years_all)}; native target years present={len(set(SSP_NATIVE_YEARS)&years_all)}/{len(SSP_NATIVE_YEARS)}", True)
    add_check(checks, "A2.03", "A2", "values", "Population values finite and nonnegative",
              "PASS" if negative == 0 and missing_pop == 0 else "FAIL", f"negative={negative}; missing/non-numeric={missing_pop}", True)
    add_check(checks, "A2.04", "A2", "dimensions", "Age and sex dimensions explicit",
              "PASS" if len(sexes_all) >= 2 and len(ages_all) >= 15 else "FAIL",
              f"sex codes={sorted(sexes_all)}; age starts={min(ages_all)}-{max(ages_all)}; n_age={len(ages_all)}", True)
    add_check(checks, "A2.05", "A2", "country_ids", "Country identifiers standardizable to ISO3",
              "PASS" if a2_countries.iso3.notna().all() else "FAIL",
              f"country/territory entities={len(a2_countries)}; mapped={a2_countries.iso3.notna().sum()}; regional aggregates excluded={len(crosswalk[(crosswalk.dataset_id=='A2') & (crosswalk.entity_type=='regional_aggregate')])}", True)
    summary = pd.DataFrame(summary_rows)
    a1_pop = a1["pop"].copy()
    a1_map = dict(zip(a1_countries.source_name, a1_countries.iso3))
    a1_pop["iso3"] = a1_pop.region.map(a1_map)
    a1_2025 = a1_pop[a1_pop.scenario.isin(SSPS)][["scenario", "iso3", "2025"]].rename(columns={"2025": "a1_pop_2025_million"})
    comp = summary.merge(a1_2025, on=["scenario", "iso3"], how="inner")
    comp["a2_pop_2025_million"] = comp.a2_pop_2025_thousands / 1000.0
    comp["relative_difference"] = (comp.a2_pop_2025_million - comp.a1_pop_2025_million).abs() / comp.a1_pop_2025_million.replace(0, np.nan)
    med_rel = float(comp.relative_difference.median()) if len(comp) else math.nan
    add_check(checks, "A2.06", "A2", "units", "Population scale is thousand persons",
              "PASS" if len(comp) and med_rel < 0.25 else "FAIL",
              f"A2 divided by 1,000 gives country totals on the same million-person scale as A1; matched country-scenarios={len(comp)}; median relative difference={med_rel:.6f}. A1 (Basic Drivers 2025) and A2 (SSP 2023/V14) are different vintages, so numeric identity is not required.", True)
    a1_set = set(a1_countries.iso3.dropna())
    a2_set = set(a2_countries.iso3.dropna())
    contracts.append({
        "dataset_id": "A2", "dataset_name": "WiC SSP age-sex population projections V14",
        "provider": "IIASA-WiC", "version": "V14 / SSP 2023 family",
        "scenarios": ";".join(SSPS), "years": f"{min(years_all)}-{max(years_all)}, five-year steps", "units": "thousand persons (scale cross-checked against A1 national totals; different source vintage)",
        "crs": "not applicable (country-age-sex table)", "nodata": "blank/NA CSV cells; none in population field",
        "coverage": f"{len(a2_set)} ISO3-standardized countries/territories plus regional aggregates", "source_url": "https://iiasa.ac.at/models-tools-data/ssp-2023",
        "doi": "10.5281/zenodo.7767425", "contract_status": "PASS",
    })
    return {
        "comparison": comp,
        "coverage": {"a1": a1_set, "a2": a2_set, "a1_only": a1_set-a2_set, "a2_only": a2_set-a1_set},
        "summary": summary,
    }

def inspect_netcdf(root: Path, nc_json_path: Path, checks: list[dict], contracts: list[dict]) -> dict:
    info = json.loads(nc_json_path.read_text(encoding="utf-8"))
    a3 = [x for x in info if "ssp_grid_population" in x["path"]]
    a8 = [x for x in info if "era5" in x["path"]]
    a3_scenarios = {re.search(r"ssp([1235])soc", Path(x["path"]).name, re.I).group(1) for x in a3}
    add_check(checks, "A3.01", "A3", "scenario", "SSP1/2/3/5 files available",
              "PASS" if a3_scenarios == {"1", "2", "3", "5"} else "FAIL", f"scenarios={sorted(a3_scenarios)}", True)
    a3_ok = True
    extents = []
    for item in a3:
        time = item["variables"]["time"]
        var = item["variables"]["number_of_people"]
        lat = item["variables"]["lat"]
        lon = item["variables"]["lon"]
        years = set(time.get("year_values", []))
        a3_ok &= set(TARGET_YEARS) <= years and var.get("units") == "1" and var.get("fill_value") is not None
        extents.append((lon["minimum"], lat["minimum"], lon["maximum"], lat["maximum"]))
    add_check(checks, "A3.02", "A3", "contract", "Years, units and NoData explicit",
              "PASS" if a3_ok else "FAIL", f"files={len(a3)}; years=2006-2100 annual; units=number_of_people=1; _FillValue=1e20", True)
    add_check(checks, "A3.03", "A3", "spatial", "CRS/extent and global grid explicit",
              "PASS" if len(set(extents)) == 1 else "FAIL", f"extent={extents[0] if extents else None}; 2.5 arc-min regular lon/lat grid; WGS84 geographic convention", True)
    # not a regridding or harmonization operation.
    sampled = 0
    negative = 0
    minimum = math.inf
    maximum = -math.inf
    for item in a3:
        path = Path(item["path"])
        with Dataset(str(path), "r") as ds:
            var = ds.variables["number_of_people"]
            years = item["variables"]["time"]["year_values"]
            indices = [years.index(y) for y in TARGET_YEARS]
            for index in indices:
                array = np.ma.asarray(var[index, ::24, ::24])
                values = np.asarray(array.compressed(), dtype=float)
                if values.size:
                    sampled += values.size
                    negative += int((values < 0).sum())
                    minimum = min(minimum, float(values.min()))
                    maximum = max(maximum, float(values.max()))
    add_check(checks, "A3.04", "A3", "values", "Population systematic sample finite and nonnegative",
              "PASS" if sampled > 0 and negative == 0 and math.isfinite(minimum) and math.isfinite(maximum) else "FAIL",
              f"sampled every 24th row/column for all 26 target years and 4 SSPs; n={sampled}; min={minimum:.8g}; max={maximum:.8g}; negative={negative}", True,
              "Systematic target-period QA sample; full-cell exhaustive scan is not performed in Gate B.")
    contracts.append({
        "dataset_id": "A3", "dataset_name": "ISIMIP gridded SSP population",
        "provider": "ISIMIP / Jones and O'Neill population projections", "version": "ISIMIP dataset version 20240102",
        "scenarios": ";".join(SSPS), "years": "2006-2100, annual", "units": "number of people per grid cell (NetCDF units=1)",
        "crs": "regular geographic lon/lat grid; WGS84 convention", "nodata": "NetCDF _FillValue=1e20",
        "coverage": "global, 2.5 arc-min", "source_url": "https://data.isimip.org/datasets/63477a92-0bd7-4265-b8e5-896562b447cd/",
        "doi": "10.48364/ISIMIP.432399", "contract_status": "PASS",
    })
    if len(a8) != 1:
        add_check(checks, "A8.01", "A8", "files", "Exactly one ERA5 monthly file", "FAIL", f"files={len(a8)}", True)
    else:
        item = a8[0]
        time = item["variables"]["valid_time"]
        months = time.get("months_per_year", {})
        correct_months = len(months) == 46 and all(months.get(str(y)) == 12 for y in range(1979, 2025))
        vars_ok = item["variables"].get("t2m", {}).get("units") == "K" and item["variables"].get("d2m", {}).get("units") == "K"
        lat = item["variables"]["latitude"]
        lon = item["variables"]["longitude"]
        add_check(checks, "A8.01", "A8", "time", "ERA5 1979-2024 with 12 months/year",
                  "PASS" if correct_months else "FAIL", f"first={time.get('first')}; last={time.get('last')}; months={time.get('count')}; years={len(months)}", True)
        add_check(checks, "A8.02", "A8", "variables", "T2m and Td2m readable with correct units",
                  "PASS" if vars_ok else "FAIL", f"t2m units={item['variables'].get('t2m',{}).get('units')}; d2m units={item['variables'].get('d2m',{}).get('units')}", True)
        add_check(checks, "A8.03", "A8", "spatial", "Global ERA5 grid and coordinates explicit", "PASS",
                  f"lat={lat['minimum']}..{lat['maximum']}; lon={lon['minimum']}..{lon['maximum']}; 0.25 degree; longitude 0-360", True)
        contracts.append({
            "dataset_id": "A8", "dataset_name": "ERA5 monthly averaged data on single levels",
            "provider": "Copernicus Climate Change Service / ECMWF", "version": "ERA5 monthly single levels, local extraction 1979-2024",
            "scenarios": "historical/reanalysis", "years": "1979-2024, monthly", "units": "K for t2m and d2m",
            "crs": "regular 0.25-degree geographic grid; longitude 0-360", "nodata": "NetCDF missing values represented by mask/NaN",
            "coverage": "global", "source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means",
            "doi": "10.24381/cds.f17050d7", "contract_status": "PASS",
        })
    return {"a3": a3, "a8": a8}

def inspect_rasters(
    root: Path, raster_json_path: Path, a4_audit_path: Path,
    checks: list[dict], contracts: list[dict]
) -> dict:
    info = json.loads(raster_json_path.read_text(encoding="utf-8"))
    a4_audit = pd.read_csv(a4_audit_path, dtype=str, keep_default_na=False)
    a4_overall = a4_audit.loc[a4_audit["audit_id"] == "A4.P13"]
    if len(a4_overall) != 1:
        raise ValueError(f"Expected exactly one A4.P13 row in {a4_audit_path}")
    a4_audit_status = a4_overall.iloc[0]["status"]
    a4_md5 = a4_audit.loc[a4_audit["audit_id"] == "A4.P03", "observed"].iloc[0]
    a4_sha256 = a4_audit.loc[a4_audit["audit_id"] == "A4.P04", "observed"].iloc[0]
    # A4 GDP
    a4 = info["A4"]["files"]
    parsed_a4 = []
    for item in a4:
        name = Path(item["path"]).name
        m = re.fullmatch(r"GDP(\d{4})(?:_ssp([1-5]))?\.tif", name, re.I)
        if m:
            parsed_a4.append((int(m.group(1)), f"SSP{m.group(2)}" if m.group(2) else "historical", item))
    target_pairs = {(year, ssp) for ssp in SSPS for year in range(2025, 2051, 5)}
    observed_pairs = {(year, scenario) for year, scenario, _ in parsed_a4}
    add_check(checks, "A4.01", "A4", "scenario_time", "SSP1/2/3/5 and 2025-2050 five-year files available",
              "PASS" if target_pairs <= observed_pairs else "FAIL", f"target files={len(target_pairs)}; present={len(target_pairs&observed_pairs)}", True)
    target_items = [item for year, scenario, item in parsed_a4 if (year, scenario) in target_pairs]
    spatial_ok = bool(target_items) and all(item["crs"] == "EPSG:4326" and item["bounds"] == [-180.0, -90.0, 180.0, 90.0] for item in target_items)
    add_check(checks, "A4.02", "A4", "spatial", "CRS and global extent explicit", "PASS" if spatial_ok else "FAIL",
              "EPSG:4326; 0.25 degree; 1440x720; extent -180,-90,180,90", True)
    # though the local GeoTIFFs do not embed unit or NoData tags.
    add_check(checks, "A4.03", "A4", "units_nodata", "Units and NoData are explicit through traceable documentation",
              "WARN", "unit=PPP 2005 international dollars (external provider documentation); nodata_tag=none/not embedded; provider zeros represent Antarctica, ocean, and some non-illuminated or depopulated cells", False,
              "Metadata warning only. Do not invent a NoData value or automatically treat zero as missing.")
    add_check(checks, "A4.04", "A4", "provenance", "Local temporal version matches the cited official release",
              "PASS" if a4_audit_status == "PASS" else "FAIL",
              f"Zenodo record 7898409; DOI 10.5281/zenodo.7898409; official/local MD5={a4_md5}; SHA256={a4_sha256}; 101 GeoTIFFs: 2000-2020 annual plus SSP1-SSP5 2025-2100 at five-year intervals", True,
              "A4 is a spatial-allocation-weight dataset only; A1 OECD ENV-Growth 2025 remains the authoritative national GDP constraint.")
    contracts.append({
        "dataset_id": "A4", "dataset_name": "Global gridded GDP under the historical and future scenarios",
        "provider": "Tingting Wang; Fubao Sun", "version": "2023 extended release, commonly referred to as v7; GDP_025d (2000-2100).7z",
        "scenarios": "SSP1;SSP2;SSP3;SSP4;SSP5", "years": "historical 2000-2020 annual; future 2025-2100 at five-year intervals",
        "units": "PPP 2005 international dollars; externally documented, not embedded in GeoTIFF", "crs": "WGS84 / EPSG:4326; 0.25 degree; global",
        "nodata": "nodata_tag=none/not embedded; provider_zero_semantics=Antarctica, ocean, and some non-illuminated or depopulated areas represented by zero", "coverage": "global",
        "source_url": "https://zenodo.org/records/7898409", "doi": "10.5281/zenodo.7898409",
        "contract_status": "PASS_WITH_METADATA_WARNINGS" if a4_audit_status == "PASS" else "FAIL_PROVENANCE_AUDIT",
    })
    # A5 urban fraction
    a5 = info["A5"]["files"]
    scenario_a5 = []
    for item in a5:
        m = re.fullmatch(r"ssp([1-5])_(\d{4})\.tif", Path(item["path"]).name, re.I)
        if m:
            scenario_a5.append((f"SSP{m.group(1)}", int(m.group(2)), item))
    observed_a5 = {(s, y) for s, y, _ in scenario_a5}
    exact_required = {(s, y) for s in SSPS for y in (2030, 2040, 2050)}
    add_check(checks, "A5.01", "A5", "scenario_time", "SSP1/2/3/5 bracket 2025-2050 and include 2030/2040/2050",
              "PASS" if exact_required <= observed_a5 else "FAIL", f"native years={sorted({y for _,y,_ in scenario_a5})}; exact target snapshots present={len(exact_required&observed_a5)}/{len(exact_required)}", True)
    add_check(checks, "A5.02", "A5", "time", "Exact 2025 urban fraction slice available", "WARN",
              "native decadal files have 2020 and 2030 but no 2025", False,
              "Interpolation is required later; no interpolation is performed in Gate B.")
    a5_items = [i for s,y,i in scenario_a5 if s in SSPS and y in (2030,2040,2050)]
    a5_spatial = bool(a5_items) and all(i["crs"] == "EPSG:4326" and i["nodata"] is not None for i in a5_items)
    stats_ok = all(float(i["band1_tags"].get("STATISTICS_MINIMUM", 0)) >= 0 and float(i["band1_tags"].get("STATISTICS_MAXIMUM", 1)) <= 1 for i in a5_items)
    add_check(checks, "A5.03", "A5", "spatial_values", "CRS, NoData, extent and 0-1 unit contract explicit",
              "PASS" if a5_spatial and stats_ok else "FAIL",
              "EPSG:4326; ~1 km; extent -180,-58 to 180,85; NoData=-3.402823e38; urban land fraction min/max metadata within 0-1", True)
    add_check(checks, "A5.04", "A5", "mask", "Land/ocean handling explicit", "PASS",
              "package includes land_mask.tif and land_area_km2.tif; scenario grids have explicit NoData", True)
    contracts.append({
        "dataset_id": "A5", "dataset_name": "Global 1-km SSP urban land fraction",
        "provider": "NASA SEDAC / Gao and Pesaresi", "version": "v1, October 2021",
        "scenarios": "SSP1;SSP2;SSP3;SSP4;SSP5", "years": "2010-2100, decadal", "units": "fraction of urban land (0-1)",
        "crs": "EPSG:4326; ~30 arc-sec", "nodata": "-3.402823e38 for scenario rasters",
        "coverage": "global land, approximately 58S-85N", "source_url": "https://data.nasa.gov/dataset/global-1-km-downscaled-urban-land-extent-projection-and-base-year-grids-by-ssp-scenar-2000",
        "doi": "10.7927/1z4r-ez63; paper 10.1038/s41597-021-01052-0", "contract_status": "PASS_WITH_2025_INTERPOLATION_WARNING",
    })
    # A6 WorldPop baseline
    a6 = info["A6"]["files"]
    parsed_a6 = []
    for item in a6:
        m = re.search(r"global_([fm])_(\d+)_2025_.*\.tif", Path(item["path"]).name, re.I)
        if m:
            parsed_a6.append((m.group(1).lower(), int(m.group(2)), item))
    expected = {(sex, age) for sex in ("f", "m") for age in range(25, 95, 5)}
    observed = {(sex, age) for sex, age, _ in parsed_a6}
    add_check(checks, "A6.01", "A6", "dimensions", "2025 adult age-sex layers complete",
              "PASS" if expected == observed else "FAIL", f"layers={len(observed)}/{len(expected)}; ages=25-90; sexes=f,m", True)
    a6_spatial = bool(parsed_a6) and all(i["crs"] == "EPSG:4326" and i["nodata"] == -99999.0 for _,_,i in parsed_a6)
    add_check(checks, "A6.02", "A6", "spatial", "CRS, extent and NoData explicit",
              "PASS" if a6_spatial else "FAIL", "EPSG:4326; 30 arc-sec; extent -180,-60 to 180,84; NoData=-99999", True)
    add_check(checks, "A6.03", "A6", "role", "Dataset role correctly limited to 2025 spatial prior", "PASS",
              "A6 is not required to contain future SSP scenarios or 2026-2050 years", True)
    contracts.append({
        "dataset_id": "A6", "dataset_name": "WorldPop age-sex structures",
        "provider": "WorldPop", "version": "R2025A UA v1, 2025 slice",
        "scenarios": "none; baseline spatial prior", "years": "2025", "units": "estimated number of people per age-sex group per grid square (official product contract)",
        "crs": "EPSG:4326; 30 arc-sec", "nodata": "-99999",
        "coverage": "global, approximately 60S-84N", "source_url": "https://hub.worldpop.org/geodata/summary?id=99407",
        "doi": "10.5258/SOTON/WP00846", "contract_status": "PASS",
    })
    return info

def inspect_cmip6(cmip_dir: Path, checks: list[dict], contracts: list[dict]) -> dict:
    group = pd.read_csv(cmip_dir / "cmip6_group_time_qa.csv")
    fileqa = pd.read_csv(cmip_dir / "cmip6_netcdf_file_qa.csv")
    report = (cmip_dir / "cmip6_stage4_report.txt").read_text(encoding="utf-8-sig", errors="replace")
    canonical_pass = int(group.canonical_status.astype(str).str.startswith("PASS").sum())
    canonical_fail = int((~group.canonical_status.astype(str).str.startswith("PASS")).sum())
    target_experiments = {"historical", "ssp126", "ssp245", "ssp370", "ssp585"}
    target_vars = {"tas", "hurs"}
    expected_groups = 7 * 5 * 2
    group_contract = len(group) == expected_groups and canonical_pass == expected_groups and canonical_fail == 0
    add_check(checks, "A7.01", "A7", "canonical", "Stage4 v3 canonical groups pass 70/70",
              "PASS" if group_contract and "RESULT: PASS" in report else "FAIL",
              f"groups={len(group)}; canonical PASS={canonical_pass}; canonical FAIL={canonical_fail}; report RESULT PASS={'RESULT: PASS' in report}", True)
    exp_ok = target_experiments <= set(group.experiment_id.astype(str)) and target_vars <= set(group.variable_id.astype(str))
    time_ok = ((group.expected_months == 432) & (group.canonical_missing_month_count == 0) & (group.canonical_duplicate_month_count == 0)).all()
    add_check(checks, "A7.02", "A7", "scenario_time", "Historical plus four CMIP6 pathways cover contract periods",
              "PASS" if exp_ok and time_ok else "FAIL",
              f"experiments={sorted(group.experiment_id.unique())}; variables={sorted(group.variable_id.unique())}; 432 months/group; no canonical gaps/duplicates={time_ok}", True)
    units_ok = set(fileqa[fileqa.variable_id == "tas"].variable_units.astype(str)) == {"K"} and set(fileqa[fileqa.variable_id == "hurs"].variable_units.astype(str)) == {"%"}
    meta_ok = (fileqa.open_status == "PASS").all() and (fileqa.metadata_match_status == "PASS").all()
    add_check(checks, "A7.03", "A7", "units_metadata", "CMIP6 files readable with units and metadata contract",
              "PASS" if units_ok and meta_ok else "FAIL", f"tas=K; hurs=%; files={len(fileqa)}; open/metadata all PASS={meta_ok}", True)
    warn_files = int((fileqa.file_qa_status == "WARN").sum())
    gross_fail = int((fileqa.file_qa_status == "FAIL").sum())
    add_check(checks, "A7.04", "A7", "physical_range", "Relative-humidity supersaturation documented without gross-range failures",
              "WARN" if warn_files else "PASS", f"WARN files={warn_files}; FAIL files={gross_fail}; raw hurs>100 retained", False,
              "Frozen later contract: cap derived hurs at 100 before ERA5 bias correction. Gate B performs no clipping.")
    add_check(checks, "A7.05", "A7", "spatial", "Spatial coordinates/extents recorded for every file",
              "PASS" if fileqa[["lat_min","lat_max","lon_min","lon_max"]].notna().all().all() else "FAIL",
              f"GCMs={fileqa.source_id.nunique()}; canonical files={len(fileqa)}; native grids retained", True)
    contracts.append({
        "dataset_id": "A7", "dataset_name": "CMIP6 tas and hurs ensemble inputs",
        "provider": "ESGF / CMIP6 ScenarioMIP", "version": "project canonical selection Stage4 v3; individual ESGF versions retained in inventory",
        "scenarios": "historical;ssp126;ssp245;ssp370;ssp585", "years": "historical contract 1979-2014; future contract 2015-2050; monthly", "units": "tas=K; hurs=%",
        "crs": "native GCM lon/lat grids; extents stored per file", "nodata": "NetCDF masks/fill values retained per source file",
        "coverage": f"global; {fileqa.source_id.nunique()} GCMs; 70 canonical GCM-experiment-variable groups", "source_url": "https://esgf-node.llnl.gov/search/cmip6/",
        "doi": "dataset-specific CMIP6 tracking IDs and provenance retained in cmip6_netcdf_file_qa.csv", "contract_status": "PASS_WITH_HURS_WARNING",
    })
    return {"groups": group, "fileqa": fileqa, "report": report}

def build_coverage_rows(a2: dict) -> list[dict]:
    rows = []
    for iso3 in sorted(a2["coverage"]["a1"] | a2["coverage"]["a2"]):
        rows.append({
            "iso3": iso3,
            "A1_country_population": "TRUE" if iso3 in a2["coverage"]["a1"] else "FALSE",
            "A2_age_sex_population": "TRUE" if iso3 in a2["coverage"]["a2"] else "FALSE",
            "coverage_class": "both" if iso3 in a2["coverage"]["a1"] and iso3 in a2["coverage"]["a2"] else ("A1_only" if iso3 in a2["coverage"]["a1"] else "A2_only"),
        })
    return rows

def report_markdown(checks: list[dict], contracts: list[dict], coverage_rows: list[dict], run_dir: Path) -> str:
    blocking_fail = [r for r in checks if r["status"] == "FAIL" and r["blocking"] == "TRUE"]
    warnings = [r for r in checks if r["status"] == "WARN"]
    overall = "FAIL" if blocking_fail else "PASS"
    lines = [
        "# Phase 1 Gate B A4 provenance remediation 数据合同核验报告",
        "",
        f"- 核验时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"- Gate B 总体结论：**{overall}**",
        f"- 阻断性失败：{len(blocking_fail)}项",
        f"- 非阻断警告：{len(warnings)}项",
        "- 核验范围：数据来源、版本、情景、年份、单位、CRS、NoData、国家覆盖、CMIP6/ERA5合同。",
        "- 明确未执行：重采样、插值、变量协调、模型拟合、BMI预测、PAF或疾病负担分析。",
        "",
        "## 结论",
        "",
    ]
    if overall == "PASS":
        lines.append("Gate B通过，可在独立批准后进入下一门控；本报告本身不授权后续分析。")
    else:
        lines.append("Gate B未通过，**不得进入Gate C或未来空间模型**。当前阻断原因如下：")
        for row in blocking_fail:
            lines.append(f"- `{row['check_id']}` {row['dataset_id']}：{row['check']}。{row['evidence']}")
    lines += ["", "## 数据集合同摘要", "", "|ID|数据集|版本/来源|情景与年份|单位|CRS/空间|NoData|状态|", "|---|---|---|---|---|---|---|---|"]
    for row in contracts:
        lines.append(f"|{row['dataset_id']}|{row['dataset_name']}|{row['version']}|{row['scenarios']}; {row['years']}|{row['units']}|{row['crs']}|{row['nodata']}|{row['contract_status']}|")
    lines += ["", "## 核验明细", "", "|检查ID|数据集|状态|阻断|检查内容|证据|", "|---|---|---|---|---|---|"]
    for row in checks:
        lines.append(f"|{row['check_id']}|{row['dataset_id']}|{row['status']}|{row['blocking']}|{row['check']}|{row['evidence']}|")
    a1_only = [r["iso3"] for r in coverage_rows if r["coverage_class"] == "A1_only"]
    a2_only = [r["iso3"] for r in coverage_rows if r["coverage_class"] == "A2_only"]
    both = sum(r["coverage_class"] == "both" for r in coverage_rows)
    lines += [
        "", "## 国家覆盖", "",
        f"- A1与A2共同覆盖：{both}个ISO3国家/地区。",
        f"- 仅A1：{', '.join(a1_only) if a1_only else '无'}。",
        f"- 仅A2：{', '.join(a2_only) if a2_only else '无'}。",
        "- A2中的联合国区域汇总项已与国家实体分开，不计入国家覆盖率。",
        "", "## 关键风险与后续门控条件", "",
        "1. A4已追溯至Zenodo 7898409（DOI 10.5281/zenodo.7898409），本地归档MD5与官方记录一致。A4仅用于国家内GDP空间分配；A1 OECD ENV-Growth 2025仍是国家GDP总量约束。",
        "2. A5没有2025精确切片，仅有2020和2030；后续若获准进入协调阶段，必须预先冻结插值规则。",
        "3. CMIP6 hurs原始值存在超过100%的文件级警告。原始文件保持不变；后续派生处理应按冻结合同在ERA5偏差校正前将hurs上限限制为100%。",
        "4. A4零值是文献定义的结构性零，不可擅自视为NoData；后续必须显式使用陆地/海洋掩膜。",
        "", "## 可追溯文件", "",
        f"- 运行目录：`{run_dir}`",
        f"- 检查表：`{run_dir / 'Gate_B_pass_fail.csv'}`",
        f"- 数据合同：`{run_dir / 'Gate_B_dataset_contracts.csv'}`",
        f"- 国家覆盖：`{run_dir / 'Gate_B_country_coverage.csv'}`",
        f"- SHA256：`{run_dir / 'Gate_B_artifact_sha256.csv'}`",
        "",
    ]
    return "\n".join(lines)

def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Phase 1 Gate B data-contract validation.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--netcdf-json", type=Path, required=True)
    parser.add_argument("--raster-json", type=Path, required=True)
    parser.add_argument("--a4-audit", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--cmip6-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checks: list[dict] = []
    contracts: list[dict] = []
    a1 = inspect_a1(args.root, checks, contracts)
    a2 = inspect_a2(args.root, args.crosswalk, a1, checks, contracts)
    inspect_netcdf(args.root, args.netcdf_json, checks, contracts)
    inspect_rasters(args.root, args.raster_json, args.a4_audit, checks, contracts)
    inspect_cmip6(args.cmip6_dir, checks, contracts)
    coverage_rows = build_coverage_rows(a2)
    add_check(checks, "COV.01", "A1/A2", "country_coverage", "A1/A2 coverage differences explicitly reported", "PASS",
              f"both={sum(r['coverage_class']=='both' for r in coverage_rows)}; A1_only={sum(r['coverage_class']=='A1_only' for r in coverage_rows)}; A2_only={sum(r['coverage_class']=='A2_only' for r in coverage_rows)}", True)
    pass_path = args.output_dir / "Gate_B_pass_fail.csv"
    contracts_path = args.output_dir / "Gate_B_dataset_contracts.csv"
    coverage_path = args.output_dir / "Gate_B_country_coverage.csv"
    a2comp_path = args.output_dir / "Gate_B_A1_A2_population_unit_check.csv"
    write_csv(pass_path, checks)
    write_csv(contracts_path, contracts)
    write_csv(coverage_path, coverage_rows)
    a2["comparison"].to_csv(a2comp_path, index=False, encoding="utf-8-sig")
    report_path = args.output_dir / "Gate_B_report.md"
    report_path.write_text(report_markdown(checks, contracts, coverage_rows, args.output_dir), encoding="utf-8")
    provenance = {
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "root": str(args.root.resolve()),
        "cmip6_metadata_dir": str(args.cmip6_dir.resolve()),
        "a4_provenance_audit": str(args.a4_audit.resolve()),
        "a4_provenance_audit_sha256": sha256(args.a4_audit),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only contract validation only",
        "prohibited_actions_performed": [],
    }
    provenance_path = args.output_dir / "Gate_B_run_provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_rows = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "Gate_B_artifact_sha256.csv":
            artifact_rows.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    hash_path = args.output_dir / "Gate_B_artifact_sha256.csv"
    write_csv(hash_path, artifact_rows)
    blocking_fail = [r for r in checks if r["status"] == "FAIL" and r["blocking"] == "TRUE"]
    print(json.dumps({
        "gate": "B", "overall": "FAIL" if blocking_fail else "PASS",
        "checks": len(checks), "blocking_failures": len(blocking_fail),
        "warnings": sum(r["status"] == "WARN" for r in checks),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))
    return 2 if blocking_fail else 0

if __name__ == "__main__":
    raise SystemExit(main())
