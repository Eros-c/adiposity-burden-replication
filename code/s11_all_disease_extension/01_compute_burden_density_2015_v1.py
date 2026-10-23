"""2015 0.1-degree high-BMI-attributable burden density (23 headline causes)."""

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

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "11_all_disease_extension" / "06_spatial_burden" / "burden_density_2015_v1"
DATA = ROOT / "data"
WEIGHTS_OUT = ROOT / "weights_2015"
QA = ROOT / "qa"

HIST2015 = (PROJECT / "05_historical_bmi_1990_2024" / "historical_25plus_compat_v1_20260909"
            / "data" / "0p1deg" / "2015")
RR_CSV = (PROJECT / "11_all_disease_extension" / "02_rr_crosswalk" / "rr_crosswalk_v1_20261005"
          / "data" / "RR_crosswalk_by_bmi_category_v1.csv")
BENCH = (PROJECT / "11_all_disease_extension" / "04_burden" / "burden_benchmark_v1_20261005"
         / "data" / "benchmark_historical_country_year_detail_v1.parquet")
DENSITY2050 = (PROJECT / "11_all_disease_extension" / "06_spatial_burden"
               / "burden_density_v1_20261007" / "data" / "2050")
HELPER_SCRIPT = (PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
                 / "scripts" / "08_plot_figure1B_historical_template_test_v2.py")

CAT_FILES = {
    "bmi_lt_18_5": "global_2015_bmi_lt18p5_regional_steps_mrp_sex_age_0p1deg_bp.tif",
    "bmi_18_5_20": "global_2015_bmi_18p5_to_lt20_regional_steps_mrp_sex_age_0p1deg_bp.tif",
    "bmi_20_25": "global_2015_bmi_20_to_lt25_regional_steps_mrp_sex_age_0p1deg_bp.tif",
    "bmi_25_30": "global_2015_bmi_25_to_lt30_regional_steps_mrp_sex_age_0p1deg_bp.tif",
    "bmi_30_35": "global_2015_bmi_30_to_lt35_regional_steps_mrp_sex_age_0p1deg_bp.tif",
    "bmi_35_40": "global_2015_bmi_35_to_lt40_regional_steps_mrp_sex_age_0p1deg_bp.tif",
    "bmi_ge_40": "global_2015_bmi_ge40_regional_steps_mrp_sex_age_0p1deg_bp.tif",
}
CAT_KEYS = list(CAT_FILES.keys())
POP_FILE = "global_2015_adult_population_regional_steps_mrp_sex_age_0p1deg.tif"
AGE_MAP = {
    "25_29": ["25-29"], "30_34": ["30-34"], "35_39": ["35-39"], "40_44": ["40-44"],
    "45_49": ["45-49"], "50_54": ["50-54"], "55_59": ["55-59"], "60_64": ["60-64"],
    "65_69": ["65-69"], "70_74": ["70-74"], "75_79": ["75-79"], "80_84": ["80-84"],
    "85_": ["85-89", "90-94", "95+"],
}
CAUSES = [417, 429, 435, 441, 453, 456, 465, 471, 480, 485, 486, 487,
          493, 494, 498, 500, 502, 515, 534, 543, 591, 976, 998]
NODATA = -9999.0
ISO_FIX = {"PSX": "PSE", "SDS": "SSD"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_helper():
    spec = importlib.util.spec_from_file_location("burden2015_helper", HELPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.historical


def load_rr() -> dict:
    rr = pd.read_csv(RR_CSV, encoding="utf-8-sig")
    lookup = {}
    for (cid, sex, age), part in rr.groupby(["cause_id", "sex", "age_group"]):
        lookup[(int(cid), sex, age)] = part.set_index("bmi_category") \
            .reindex(CAT_KEYS)["RR"].to_numpy(np.float64)
    return lookup


def parse_band(name: str):
    sex = "Female" if name.startswith("female") else "Male"
    age = name.split("age", 1)[1]
    return sex, age


def compute_weights(rr_lookup: dict) -> dict:
    pop_path = HIST2015 / POP_FILE
    pop_ds = rasterio.open(pop_path)
    cat_ds = {k: rasterio.open(HIST2015 / v) for k, v in CAT_FILES.items()}
    profile = pop_ds.profile.copy()
    profile.update(dtype="float32", count=1, compress="lzw", nodata=NODATA)
    shape = pop_ds.shape
    w = {c: np.zeros(shape, dtype=np.float64) for c in CAUSES}
    for band_i in range(1, pop_ds.count + 1):
        name = pop_ds.descriptions[band_i - 1]
        sex, age_key = parse_band(name)
        pop = pop_ds.read(band_i, masked=True)
        pop_f = np.ma.filled(pop.astype(np.float64), 0.0)
        if pop_f.sum() <= 0:
            continue
        rr_ages = AGE_MAP[age_key]
        shares = np.stack([
            np.ma.filled(cat_ds[k].read(band_i, masked=True).astype(np.float64), 0.0) / 10000.0
            for k in CAT_KEYS
        ])  # (7, H, W)
        denom_base = shares  # precompute per-cause below
        for cause in CAUSES:
            rr_parts = [rr_lookup.get((cause, sex, age)) for age in rr_ages]
            rr_parts = [r for r in rr_parts if r is not None]
            if not rr_parts:
                continue  # female-only causes have no male rows
            rr_cat = np.mean(rr_parts, axis=0)  # (7,)
            denom = np.tensordot(rr_cat, denom_base, axes=(0, 0))  # (H, W)
            with np.errstate(divide="ignore", invalid="ignore"):
                paf = np.where(denom > 0, 1.0 - 1.0 / denom, 0.0)
            paf = np.clip(paf, -1.0, 1.0)
            w[cause] += pop_f * paf
        print(f"band {band_i}/{pop_ds.count} {name} done", flush=True)
    pop_ds.close()
    for ds in cat_ds.values():
        ds.close()

    WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for cause in CAUSES:
        out = WEIGHTS_OUT / f"weight_pop_paf_c{cause}_ssp245_2015_0p1deg_v1.tif"
        data = w[cause].astype(np.float32)
        data[~np.isfinite(data)] = NODATA
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data, 1)
        out_paths.append(out)
        print("saved", out.name, flush=True)
    return w


def country_zones(shape, transform, historical):
    from rasterio.features import rasterize
    countries = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")
    countries = countries[countries.geometry.notna()].copy()
    countries["iso3"] = countries["ISO_A3"].where(countries["ISO_A3"].ne("-99"), countries["ADM0_A3"])
    countries["iso3"] = countries["iso3"].astype(str).str.upper()
    keys = sorted(x for x in countries["iso3"].unique() if x not in {"-99", "NAN", "NONE"})
    ids = {iso: i for i, iso in enumerate(keys)}
    shapes = [(g, ids[i]) for g, i in zip(countries.geometry, countries["iso3"]) if i in ids]
    zones = rasterize(shapes, out_shape=shape, transform=transform, fill=-1, dtype="int32")
    return zones, keys


def main():
    for sub in (DATA, WEIGHTS_OUT, QA):
        sub.mkdir(parents=True, exist_ok=True)
    rr_lookup = load_rr()
    w = compute_weights(rr_lookup)

    historical = import_helper()
    with rasterio.open(HIST2015 / POP_FILE) as ds:
        transform, shape = ds.transform, ds.shape
    zones, keys = country_zones(shape, transform, historical)
    zflat = zones.ravel()
    sel = zflat >= 0

    bench = pd.read_parquet(BENCH)
    bench = bench[(bench["analysis_tier"] == "headline") & (bench["year"] == 2015)]

    combined = {"Deaths": np.zeros(shape), "DALYs": np.zeros(shape)}
    closure = []
    for measure in ("Deaths", "DALYs"):
        for cause in CAUSES:
            nat = bench[(bench["cause_id"] == cause) & (bench["measure"] == measure)] \
                .set_index("iso3")["study_attributable"]
            wflat = np.nan_to_num(w[cause].ravel(), nan=0.0)
            wsum = np.bincount(zflat[sel], weights=wflat[sel], minlength=len(keys))
            nat_lookup = np.zeros(len(keys))
            for i, iso in enumerate(keys):
                nat_lookup[i] = float(nat.get(ISO_FIX.get(iso, iso), 0.0))
            cell_share = np.zeros(len(wflat))
            idx_z = zflat[sel]
            cell_share[sel] = np.divide(wflat[sel], wsum[idx_z],
                                        out=np.zeros(int(sel.sum())), where=wsum[idx_z] != 0)
            cell = cell_share * nat_lookup[zflat.clip(min=0)]
            cell = np.where(zflat >= 0, cell, np.nan).reshape(shape)
            combined[measure] += np.nan_to_num(cell, nan=0.0)
            alloc = np.bincount(idx_z, weights=np.nan_to_num(cell.ravel(), nan=0.0)[sel],
                                minlength=len(keys))
            for i, iso in enumerate(keys):
                if nat_lookup[i] > 0:
                    closure.append({"measure": measure, "cause_id": cause, "iso3": iso,
                                    "national": nat_lookup[i], "allocated": float(alloc[i])})
        out = DATA / f"burden_density_combined23_{measure.lower()}_ssp245_2015_0p1deg_v1.tif"
        arr = combined[measure].astype(np.float32)
        arr[~np.isfinite(arr)] = NODATA
        with rasterio.open(HIST2015 / POP_FILE) as src:
            profile = src.profile.copy()
        profile.update(dtype="float32", count=1, compress="lzw", nodata=NODATA)
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(arr, 1)
        print("saved", out.name, flush=True)

    # change rasters 2015 -> 2050
    for measure in ("Deaths", "DALYs"):
        with rasterio.open(DENSITY2050 / f"burden_density_combined23_{measure.lower()}_ssp245_2050_0p1deg_v1.tif") as src:
            d50 = src.read(1, masked=True).astype(np.float64)
            profile = src.profile.copy()
        d50 = np.ma.filled(d50, 0.0)
        d15 = combined[measure]
        abs_change = d50 - d15
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(d15 > 1e-6, abs_change / np.maximum(d15, 1e-6) * 100.0, np.nan)
        for stem, arr in ((f"absolute_change_{measure.lower()}_2015_2050_0p1deg_v1", abs_change),
                          (f"change_{measure.lower()}_2015_2050_0p1deg_v1", pct)):
            out = DATA / f"{stem}.tif"
            data = arr.astype(np.float32)
            data[~np.isfinite(data)] = NODATA
            profile.update(dtype="float32", count=1, compress="lzw", nodata=NODATA)
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(data, 1)
            print("saved", out.name, flush=True)

    cl = pd.DataFrame(closure)
    cl["rel_err"] = (cl["allocated"] - cl["national"]) / cl["national"]
    cl.to_csv(QA / "closure_2015_by_country_cause_v1.csv", index=False, encoding="utf-8-sig")
    qa = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "burden_cell = national_attributable x w(cell)/w(country); w = pop x PAF",
        "n_causes": len(CAUSES),
        "closure_max_abs_rel_err": float(cl["rel_err"].abs().max()),
        "closure_mean_abs_rel_err": float(cl["rel_err"].abs().mean()),
        "global_attributable_2015_deaths": float((bench[bench.measure == "Deaths"]
                                                  .groupby("cause_id")["study_attributable"].sum()).sum()),
        "notes": "RR for the merged 85+ historical band is the unweighted mean of RR ages 85-89/90-94/95+.",
    }
    (QA / "burden_density_2015_QA_v1.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [{"path": str(p), "sha256": sha256(p)} for p in sorted(DATA.glob("*.tif"))]
    rows += [{"path": str(p), "sha256": sha256(p)} for p in sorted(WEIGHTS_OUT.glob("*.tif"))]
    pd.DataFrame(rows).to_csv(ROOT / "burden_density_2015_manifest_v1.csv",
                              index=False, encoding="utf-8-sig")
    print("DONE", json.dumps(qa, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
