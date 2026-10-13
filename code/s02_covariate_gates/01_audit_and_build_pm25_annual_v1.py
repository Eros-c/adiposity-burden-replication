#!/usr/bin/env python3
"""Audit NorESM2-LM PM2.5 mass mixing ratio and build annual surface fields.

The supplied variable is a dry-aerosol mass fraction (kg kg-1), not ambient
surface concentration (ug m-3).  We therefore retain its physical definition,
select the lowest hybrid model layer, scale it to ug kg-1 for numerical
stability, and refit the BMI model rather than reusing the old PM2.5
concentration coefficient.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


EXPECTED = {
    "historical": [(1990, 1999), (2000, 2009), (2010, 2014)],
    "ssp126": [(2015, 2020), (2021, 2030), (2031, 2040), (2041, 2050)],
    "ssp245": [(2015, 2020), (2021, 2030), (2031, 2040), (2041, 2050)],
    "ssp370": [(2015, 2020), (2021, 2030), (2031, 2040), (2041, 2050)],
    "ssp585": [(2015, 2020), (2021, 2030), (2031, 2040), (2041, 2050)],
}

FILE_RE = re.compile(
    r"mmrpm2p5_AERmon_NorESM2-LM_(historical|ssp\d+)_r1i1p1f1_gn_"
    r"(\d{4})\d{2}-(\d{4})\d{2}\.nc$"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def annual_surface(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path, decode_times=False)
    try:
        var = ds["mmrpm2p5"]
        if var.dims != ("time", "lev", "lat", "lon"):
            raise RuntimeError(f"Unexpected dimensions in {path.name}: {var.dims}")
        if var.attrs.get("units") != "kg kg-1":
            raise RuntimeError(f"Unexpected units in {path.name}: {var.attrs.get('units')}")
        if ds["lev"].attrs.get("positive") != "down":
            raise RuntimeError(f"Cannot identify surface layer in {path.name}")
        lev = np.asarray(ds["lev"].values)
        surface_i = int(np.argmax(lev))
        if surface_i != 0 or not np.isclose(lev[surface_i], 0.992556095123291):
            raise RuntimeError(f"Unexpected surface level in {path.name}: {lev[surface_i]}")

        # Monthly cell means are weighted by the exact no-leap interval length.
        bounds = np.asarray(ds["time_bnds"].values)
        weights = bounds[:, 1] - bounds[:, 0]
        m = FILE_RE.match(path.name)
        if not m:
            raise RuntimeError(f"Cannot parse year range: {path.name}")
        start_year = int(m.group(2))
        years = start_year + np.arange(var.sizes["time"]) // 12
        surface = var.isel(lev=surface_i).load()
        annual = []
        for year in np.unique(years):
            idx = np.where(years == year)[0]
            w = xr.DataArray(weights[idx], dims=("time",))
            a = surface.isel(time=idx).weighted(w).mean("time")
            annual.append(a.expand_dims(year=[int(year)]))
        out = xr.concat(annual, dim="year")
        out.name = "pm25_mmr_ug_per_kg"
        out = out * 1.0e9
        out.attrs = {
            "long_name": "Near-surface PM2.5 dry aerosol mass mixing ratio",
            "units": "microgram per kilogram of air",
            "source_variable": "mmrpm2p5",
            "source_units": "kg kg-1",
            "surface_layer_index_zero_based": surface_i,
            "surface_hybrid_coordinate": float(lev[surface_i]),
            "annual_aggregation": "time-bound weighted mean of monthly means",
        }
        return out
    finally:
        ds.close()


def normalize_grid(a: xr.DataArray) -> xr.DataArray:
    lon = ((a.lon + 180) % 360) - 180
    a = a.assign_coords(lon=lon).sortby("lon")
    return a.sortby("lat", ascending=False)


def write_dataset(a: xr.DataArray, path: Path, attrs: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    ds = a.to_dataset()
    ds.attrs.update(attrs)
    encoding = {a.name: {"zlib": True, "complevel": 5, "dtype": "float32"}}
    ds.to_netcdf(path, encoding=encoding)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(_cfg.ROOT / ".codex_tmp" / "full_covariate_raw_v1" / "CMIP6_AEROSOL" / "NorESM2-LM"),
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    run_dir = Path(args.run_dir)
    qa_dir = run_dir / "qa"
    data_dir = run_dir / "data" / "pm25_annual_native_trailing5y_v2"
    contracts_dir = run_dir / "contracts"
    logs_dir = run_dir / "logs"
    for d in (qa_dir, data_dir, contracts_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.nc"))
    inventory = []
    annual_by_exp: dict[str, list[xr.DataArray]] = {}
    for path in files:
        m = FILE_RE.match(path.name)
        if not m:
            raise RuntimeError(f"Unexpected PM2.5 filename: {path.name}")
        exp, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        a = annual_surface(path)
        annual_by_exp.setdefault(exp, []).append(a)
        vals = np.asarray(a.values)
        inventory.append(
            {
                "file": path.name,
                "experiment": exp,
                "start_year": start,
                "end_year": end,
                "year_count": int(a.sizes["year"]),
                "surface_lev": a.attrs["surface_hybrid_coordinate"],
                "units": a.attrs["units"],
                "min": float(np.nanmin(vals)),
                "median": float(np.nanmedian(vals)),
                "p99": float(np.nanquantile(vals, 0.99)),
                "max": float(np.nanmax(vals)),
                "negative_count": int(np.sum(vals < 0)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    actual = {
        exp: [(int(r["start_year"]), int(r["end_year"])) for r in inventory if r["experiment"] == exp]
        for exp in EXPECTED
    }
    if actual != EXPECTED:
        raise RuntimeError(f"Coverage mismatch. Expected {EXPECTED}; found {actual}")

    combined: dict[str, xr.DataArray] = {}
    for exp, parts in annual_by_exp.items():
        a = normalize_grid(xr.concat(parts, dim="year").sortby("year"))
        if len(np.unique(a.year.values)) != a.sizes["year"]:
            raise RuntimeError(f"Duplicate years for {exp}")
        combined[exp] = a

    scenario_exps = ["ssp126", "ssp245", "ssp370", "ssp585"]
    common_extension = xr.concat(
        [combined[e].sel(year=slice(2015, 2024)) for e in scenario_exps], dim="scenario"
    ).mean("scenario")
    common_extension.attrs = combined["ssp245"].attrs.copy()
    common_extension.attrs["historical_extension"] = (
        "2015-2024 arithmetic ensemble mean of SSP126, SSP245, SSP370 and SSP585 "
        "for a scenario-neutral common training covariate"
    )
    history_for_training = xr.concat([combined["historical"], common_extension], dim="year")
    history_for_training = history_for_training.sel(year=slice(1990, 2024))
    history_for_training = history_for_training.rolling(year=5, min_periods=1).mean()
    history_for_training.attrs = common_extension.attrs.copy()
    history_for_training.attrs["temporal_smoothing"] = (
        "five-year trailing mean; current and previous four years only"
    )

    write_dataset(
        history_for_training,
        data_dir / "pm25_mmr_ugkg_trailing5y_historical_common_1990_2024.nc",
        {"series_role": "historical and model-refit covariate"},
    )

    common_2024 = history_for_training.sel(year=2024)
    epsilon = 1.0e-6
    for exp in scenario_exps:
        raw = combined[exp].rolling(year=5, min_periods=1).mean()
        raw.attrs = combined[exp].attrs.copy()
        raw.attrs["temporal_smoothing"] = (
            "five-year trailing mean; current and previous four years only"
        )
        baseline = raw.sel(year=2024)
        future = raw.sel(year=slice(2025, 2050))
        adjusted = np.exp(np.log(future + epsilon) - np.log(baseline + epsilon) + np.log(common_2024 + epsilon)) - epsilon
        adjusted = adjusted.clip(min=0)
        adjusted.attrs = raw.attrs.copy()
        adjusted.attrs["continuity_adjustment"] = (
            "cellwise log-ratio anomaly relative to the scenario-specific 2024 field, "
            "added to the common 2024 field"
        )
        write_dataset(
            adjusted,
            data_dir / f"pm25_mmr_ugkg_trailing5y_{exp}_2025_2050.nc",
            {"series_role": "future scenario covariate", "scenario_experiment": exp},
        )

    pd.DataFrame(inventory).to_csv(qa_dir / "PM25_input_inventory_v2.csv", index=False, encoding="utf-8-sig")
    contract = {
        "dataset": "NorESM2-LM CMIP6 PM2.5 mass mixing ratio",
        "source_variable": "mmrpm2p5",
        "source_standard_name": "mass_fraction_of_pm2p5_dry_aerosol_particles_in_air",
        "source_units": "kg kg-1",
        "derived_variable": "pm25_mmr_ug_per_kg",
        "derived_units": "microgram per kilogram of air",
        "vertical_selection": "lowest hybrid sigma-pressure layer, zero-based index 0, lev=0.992556095123291",
        "not_equivalent_to": "surface PM2.5 concentration in microgram per cubic metre",
        "model_rule": "refit coefficient using the same derived definition in historical and future periods",
        "historical_splice": "historical through 2014; four-SSP mean for 2015-2024",
        "future_continuity": "scenario log-ratio anomalies anchored to common 2024 field",
        "temporal_smoothing": "five-year trailing mean with no future-year leakage",
        "coverage": {"historical": "1990-2024", "future": "2025-2050"},
        "experiments": scenario_exps,
        "input_file_count": len(files),
        "status": "PASS_WITH_CAVEAT_SINGLE_MODEL_AND_MIXING_RATIO_SEMANTICS",
    }
    (contracts_dir / "PM25_variable_contract_v2.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (logs_dir / "PM25_build_v2.log").write_text(
        f"PASS\ninput_files={len(files)}\nhistory_years=1990-2024\nfuture_years=2025-2050\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "input_files": len(files), "output_dir": str(data_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
