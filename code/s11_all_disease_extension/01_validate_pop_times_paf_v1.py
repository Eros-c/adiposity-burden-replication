#!/usr/bin/env python3
"""T2 validation: does population x PAF track official GBD attributable deaths?

Teacher's gate for spatial interpolation of disease burden:
  X = sum over grid cells of (adult population x PAF) per country
  Y = official GBD high-BMI-attributable deaths per country
If the two are highly correlated, spatializing national attributable burden
onto grids via pop x PAF weights is justified.

Test A (official validation): years 2015 and 2020 (historical 0.1-degree
products), 23 headline causes, country-level X vs official GBD deaths.
PAF is computed per sex-age slice exactly as in Phase 3 (RR crosswalk v1).

Test B (projection-period consistency): 2050, SSP2-4.5, for the four diseases
with 0.1-degree grid PAFs (CRC, IHD, T2DM, ALZ): grid X vs the study-derived
attributable deaths that the interpolation would allocate.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "11_all_disease_extension" / "05_spatial_interpolation_validation_v1"
SCRIPTS_DIR = PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909" / "scripts"

HIST_DIR = PROJECT / "05_historical_bmi_1990_2024" / "historical_25plus_compat_v1_20260909" / "data" / "0p1deg"
RR_EXPANDED = (PROJECT / "11_all_disease_extension" / "02_rr_crosswalk" / "rr_crosswalk_v1_20261005"
               / "data" / "RR_crosswalk_by_bmi_category_v1.csv")
OFFICIAL = (PROJECT / "11_all_disease_extension" / "01_harmonize" / "harmonized_gbd_v1_20261005"
            / "data" / "gbd_high_bmi_attributable_harmonized_v1.parquet")
FUTURE_PAF_GRID = (PROJECT / "11_all_disease_extension" / "03_paf" / "paf_grid_multidisease_v1_20261005" / "data")
FUTURE_BMI = (PROJECT / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906" / "outputs")
FUTURE_BURDEN = (PROJECT / "11_all_disease_extension" / "04_burden" / "burden_benchmark_v1_20261005"
                 / "data" / "attributable_future_country_overall_four_ssp_v1.parquet")
CRC_PAF_GRID = (PROJECT / "08_colorectal_cancer_paf" / "full_annual_four_ssp_wcrf_primary_v1_20260910"
                / "data" / "0p1deg" / "ssp245" / "2050" / "CRC_PAF_ssp245_2050_0p1deg_overall_point_v1.tif")

CATS = ["lt18p5", "18p5_to_lt20", "20_to_lt25", "25_to_lt30", "30_to_lt35", "35_to_lt40", "ge40"]
SLICES = [f"{sex}_age{age}" for sex in ("female", "male")
          for age in ("25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59",
                      "60_64", "65_69", "70_74", "75_79", "80_84", "85_")]
# RR crosswalk age/sex labels
RR_AGE = ["25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64",
          "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+"]
# raster slices: 13 ages x 2 sexes; the last age slice (85+) uses the 85-89 RR row
SLICE_SEX_AGE = [(s, a) for s in ("Female", "Male") for a in RR_AGE[:12] + ["85-89"]]
RR_CAT_KEYS = ["bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30",
               "bmi_30_35", "bmi_35_40", "bmi_ge_40"]

GRID_DISEASES_2050 = {441: "CRC", 493: "IHD", 976: "T2DM", 543: "ALZ"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_helper():
    path = SCRIPTS_DIR / "08_plot_figure1B_historical_template_test_v2.py"
    spec = importlib.util.spec_from_file_location("t2_helper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.historical


historical = import_helper()


def country_zones(shape, transform):
    countries = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")
    countries = countries[countries.geometry.notna()].copy()
    countries["iso3"] = countries["ISO_A3"].where(countries["ISO_A3"].ne("-99"), countries["ADM0_A3"])
    countries["iso3"] = countries["iso3"].astype(str).str.upper()
    keys = sorted(x for x in countries["iso3"].unique() if x not in {"-99", "NAN", "NONE"})
    ids = {iso: i for i, iso in enumerate(keys)}
    shapes = [(g, ids[i]) for g, i in zip(countries.geometry, countries["iso3"]) if i in ids]
    zones = rasterize(shapes, out_shape=shape, transform=transform, fill=-1, dtype="int32")
    return zones, keys, ids


def load_rr_lookup():
    rr = pd.read_csv(RR_EXPANDED, encoding="utf-8-sig")
    lookup = {}
    for (cid, sex, age), part in rr.groupby(["cause_id", "sex", "age_group"]):
        part = part.set_index("bmi_category").reindex(RR_CAT_KEYS)
        lookup[(int(cid), sex, age)] = part["RR"].to_numpy(float)
    return lookup


def slice_to_rr(sex: str, slice_age: str) -> tuple[str, str]:
    # SLICE_SEX_AGE already carries RR-format ages (85-89 row serves the 85+ slice)
    return sex, slice_age


def paf_slices_from_rasters(year: int):
    """Return shares (26, H, W) and population (26, H, W) for a historical year."""
    d = HIST_DIR / str(year)
    shares = []
    for cat in CATS:
        with rasterio.open(d / f"global_{year}_bmi_{cat}_regional_steps_mrp_sex_age_0p1deg_bp.tif") as ds:
            shares.append(ds.read(masked=True).astype(np.float64) / 10000.0)
    shares = np.stack(shares)  # (7, 26, H, W)
    with rasterio.open(d / f"global_{year}_adult_population_regional_steps_mrp_sex_age_0p1deg.tif") as ds:
        pop = ds.read(masked=True).astype(np.float64)
        transform, crs = ds.transform, ds.crs
    return shares, pop, transform, crs


def zone_sums_hist(year: int, rr_lookup, causes: list[int], zones: np.ndarray, keys: list[str]) -> pd.DataFrame:
    shares, pop, transform, crs = paf_slices_from_rasters(year)
    rows = []
    for cause in causes:
        pop_paf_total = None
        for si, (sex, age) in enumerate(SLICE_SEX_AGE):
            rsex, rage = slice_to_rr(sex, age)
            rr_vec = rr_lookup.get((cause, rsex, rage))
            if rr_vec is None:
                # female-only causes have no male rows; skip those slices
                continue
            p = shares[:, si]  # (7, H, W)
            w = pop[si]
            weighted_rr = np.nansum(np.ma.filled(p, 0.0) * rr_vec[:, None, None], axis=0)
            valid = (np.ma.filled(w, 0.0) > 0) & (weighted_rr > 0) & np.isfinite(np.ma.filled(weighted_rr, np.nan))
            paf = (weighted_rr - 1.0) / weighted_rr
            prod = np.where(valid, np.ma.filled(paf, 0.0) * np.ma.filled(w, 0.0), 0.0)
            pop_paf_total = prod if pop_paf_total is None else pop_paf_total + prod
        if pop_paf_total is None:
            continue
        flat = pop_paf_total.ravel()
        zflat = zones.ravel()
        valid = zflat >= 0
        sums = np.bincount(zflat[valid], weights=flat[valid])
        for iso, idx in zip(keys, range(len(keys))):
            if idx < len(sums):
                rows.append({"cause_id": cause, "year": year, "iso3": iso,
                             "pop_x_paf": float(sums[idx])})
    return pd.DataFrame(rows)


def zone_sums_grid(tag: str, scenario: str, year: int, zones: np.ndarray, keys: list[str]) -> pd.DataFrame:
    if tag == "CRC":
        path = CRC_PAF_GRID
        pop_path = FUTURE_BMI / scenario / str(year) / f"population_overall_{scenario}_{year}_0p1deg_v1.tif"
    else:
        path = FUTURE_PAF_GRID / tag / scenario / f"PAF_{tag}_{scenario}_{year}_0p1deg_overall_v1.tif"
        pop_path = FUTURE_BMI / scenario / str(year) / f"population_overall_{scenario}_{year}_0p1deg_v1.tif"
    with rasterio.open(path) as ds:
        paf = ds.read(1, masked=True).astype(np.float64)
        transform = ds.transform
        if tag == "CRC":
            paf = paf * 100.0  # CRC production stores percent
    with rasterio.open(pop_path) as ds:
        pop = ds.read(1, masked=True).astype(np.float64)
    valid = np.isfinite(paf.filled(np.nan)) & (np.ma.filled(pop, 0.0) > 0)
    prod = np.where(valid, np.ma.filled(paf, 0.0) * np.ma.filled(pop, 0.0), 0.0)
    flat = prod.ravel()
    zflat = zones.ravel()
    sel = zflat >= 0
    sums = np.bincount(zflat[sel], weights=flat[sel])
    return pd.DataFrame({"cause_id": {"CRC": 441, "IHD": 493, "T2DM": 976, "ALZ": 543}[tag],
                         "year": year, "iso3": keys[: len(sums)], "pop_x_paf": sums})


def corr_stats(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    out = {"n": int(len(x))}
    out["pearson_raw"] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan
    positive = (x > 0) & (y > 0)
    out["n_positive"] = int(positive.sum())
    if positive.sum() > 2:
        lx, ly = np.log10(x[positive]), np.log10(y[positive])
        out["pearson_log10"] = float(np.corrcoef(lx, ly)[0, 1])
        slope, intercept = np.polyfit(lx, ly, 1)
        fitted = intercept + slope * lx
        out["r2_log10"] = float(1 - np.sum((ly - fitted) ** 2) / np.sum((ly - ly.mean()) ** 2))
    else:
        out["pearson_log10"] = np.nan
        out["r2_log10"] = np.nan
    out["spearman"] = float(pd.Series(x).corr(pd.Series(y), method="spearman")) if len(x) > 2 else np.nan
    return out


def main() -> None:
    rr_lookup = load_rr_lookup()

    official = pd.read_parquet(OFFICIAL)
    official = official[(official["metric"] == "Number") & (official["measure"] == "Deaths")]
    official = official.groupby(["cause_id", "iso3", "year"], as_index=False)["value"].sum()
    official = official.rename(columns={"value": "official_attributable_deaths"})

    headline = pd.read_csv(
        PROJECT / "11_all_disease_extension" / "02_rr_crosswalk" / "rr_crosswalk_v1_20261005"
        / "qa" / "RR_crosswalk_decisions_v1.csv", encoding="utf-8-sig",
    )
    causes = sorted(headline.loc[headline["analysis_tier"] == "headline", "cause_id"].astype(int).unique())

    # zone raster from a historical raster's own grid
    shares0, pop0, transform, crs = paf_slices_from_rasters(2020)
    zones, keys, ids = country_zones(pop0.shape[1:], transform)

    rows_test_a = []
    for year in (2015, 2020):
        zx = zone_sums_hist(year, rr_lookup, causes, zones, keys)
        zx = zx.merge(official[official["year"] == year], on=["cause_id", "iso3"], how="inner")
        for cause, part in zx.groupby("cause_id"):
            stats = corr_stats(part["pop_x_paf"].to_numpy(), part["official_attributable_deaths"].to_numpy())
            rows_test_a.append({"cause_id": cause, "year": year, **stats})
    test_a = pd.DataFrame(rows_test_a)

    # Test B: 2050 grid products for the four diseases with grid PAFs.
    zones2050 = zones  # same grid family? grids are both 0.1-degree EPSG:4326; rebuild to be safe
    with rasterio.open(FUTURE_BMI / "ssp245" / "2050" / "population_overall_ssp245_2050_0p1deg_v1.tif") as ds:
        t2050 = ds.transform
        shape2050 = ds.shape
    zones2050, keys2050, _ = country_zones(shape2050, t2050)

    burden = pd.read_parquet(FUTURE_BURDEN)
    burden = burden[(burden["scenario_code"] == "ssp245") & (burden["year"] == 2050)
                    & (burden["measure"] == "Deaths")]
    burden = burden[["cause_id", "iso3", "attributable_burden"]].rename(
        columns={"attributable_burden": "study_attributable_deaths"})

    rows_test_b = []
    for cause, tag in GRID_DISEASES_2050.items():
        zx = zone_sums_grid(tag, "ssp245", 2050, zones2050, keys2050)
        zx = zx.merge(burden[burden["cause_id"] == cause], on="iso3", how="inner")
        stats = corr_stats(zx["pop_x_paf"].to_numpy(), zx["study_attributable_deaths"].to_numpy())
        rows_test_b.append({"cause_id": cause, "tag": tag, "year": 2050, **stats})
    test_b = pd.DataFrame(rows_test_b)

    ROOT_T = ROOT / "tables"
    ROOT_Q = ROOT / "qa"
    test_a_path = ROOT_T / "pop_times_paf_vs_gbd_attributable_2015_2020_v1.csv"
    test_b_path = ROOT_T / "pop_times_paf_vs_study_attributable_2050_v1.csv"
    test_a.to_csv(test_a_path, index=False, encoding="utf-8-sig")
    test_b.to_csv(test_b_path, index=False, encoding="utf-8-sig")

    summary = {
        "run": "spatial_interpolation_validation_v1",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "test_A": "country-level sum of grid pop x PAF (2015, 2020) vs official GBD high-BMI attributable deaths, 23 headline causes",
        "test_B": "country-level grid pop x PAF (2050, SSP2-4.5) vs study-derived attributable deaths for CRC/IHD/T2DM/ALZ",
        "test_A_min_pearson_log10": float(np.nanmin(test_a["pearson_log10"])),
        "test_A_median_pearson_log10": float(np.nanmedian(test_a["pearson_log10"])),
        "test_B": test_b.to_dict(orient="records"),
        "outputs": [str(test_a_path), str(test_b_path)],
    }
    write_json(ROOT_Q / "validation_QA_v1.json", summary)

    (ROOT / "scripts" / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(json.dumps({
        "A_median_r_log10": summary["test_A_median_pearson_log10"],
        "A_min_r_log10": summary["test_A_min_pearson_log10"],
        "B_r": [r["pearson_log10"] for r in summary["test_B"]],
    }, indent=2))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
