#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补齐 CRC (c441) 2015 年 0.1° 归因死亡栅格。
方法与 03_cause_level_2015_deaths_v1.py 完全一致：
  density(cell) = weight(cell) / sum(weight, country) x national study_attributable(2015, Deaths)
输出与既有产品同目录同名规则：burden_density_c441_deaths_ssp245_2015_0p1deg_v1.tif
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
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
ROOT = PROJECT / "11_all_disease_extension" / "06_spatial_burden" / "burden_density_2015_v1"
DATA = ROOT / "data"
WEIGHTS = ROOT / "weights_2015"
QA = ROOT / "qa"
BENCH = (PROJECT / "11_all_disease_extension" / "04_burden" / "burden_benchmark_v1_20261005"
         / "data" / "benchmark_historical_country_year_detail_v1.parquet")
HELPER_SCRIPT = (PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
                 / "scripts" / "08_plot_figure1B_historical_template_test_v2.py")
CID = 441
NODATA = -9999.0
ISO_FIX = {"PSX": "PSE", "SDS": "SSD"}


def import_helper():
    spec = importlib.util.spec_from_file_location("fig4_cause_helper", HELPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.historical


def main():
    historical = import_helper()
    w_path = WEIGHTS / f"weight_pop_paf_c{CID}_ssp245_2015_0p1deg_v1.tif"
    with rasterio.open(w_path) as ds:
        transform, shape, profile = ds.transform, ds.shape, ds.profile.copy()
        w = np.nan_to_num(np.asarray(ds.read(1, masked=True).astype(np.float64)), nan=0.0)
    profile.update(dtype="float32", count=1, compress="lzw", nodata=NODATA)

    countries = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")
    countries = countries[countries.geometry.notna()].copy()
    countries["iso3"] = countries["ISO_A3"].where(countries["ISO_A3"].ne("-99"), countries["ADM0_A3"])
    countries["iso3"] = countries["iso3"].astype(str).str.upper()
    keys = sorted(x for x in countries["iso3"].unique() if x not in {"-99", "NAN", "NONE"})
    ids = {iso: i for i, iso in enumerate(keys)}
    shapes = [(g, ids[i]) for g, i in zip(countries.geometry, countries["iso3"]) if i in ids]
    zones = rasterize(shapes, out_shape=shape, transform=transform, fill=-1, dtype="int32")
    zflat = zones.ravel()
    sel = zflat >= 0

    bench = pd.read_parquet(BENCH)
    bench = bench[(bench["analysis_tier"] == "headline") & (bench["year"] == 2015)
                  & (bench["measure"] == "Deaths") & (bench["cause_id"] == CID)]
    assert len(bench) > 0, "benchmark 中无 c441 2015 Deaths"
    nat = bench.set_index("iso3")["study_attributable"]

    nat_lookup = np.zeros(len(keys))
    for i, iso in enumerate(keys):
        nat_lookup[i] = float(nat.get(ISO_FIX.get(iso, iso), 0.0))
    wflat = w.ravel()
    wsum = np.bincount(zflat[sel], weights=wflat[sel], minlength=len(keys))
    share = np.zeros(len(wflat))
    idx_z = zflat[sel]
    share[sel] = np.divide(wflat[sel], wsum[idx_z], out=np.zeros(int(sel.sum())),
                           where=wsum[idx_z] != 0)
    cell_flat = share * nat_lookup[zflat.clip(min=0)]
    cell = np.where(zflat >= 0, cell_flat, np.nan).reshape(shape)
    arr = cell.astype(np.float32)
    arr[~np.isfinite(arr)] = NODATA
    out = DATA / f"burden_density_c{CID}_deaths_ssp245_2015_0p1deg_v1.tif"
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(arr, 1)
    stats = {"national_2015_deaths": float(nat_lookup.sum()),
             "grid_sum": float(np.nansum(cell)),
             "ratio": float(np.nansum(cell) / nat_lookup.sum())}
    qaf = QA / "cause_level_2015_deaths_QA_c441_v1.json"
    qaf.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
                               "cause": CID, **stats}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("saved", out.name, json.dumps(stats, indent=2), flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
