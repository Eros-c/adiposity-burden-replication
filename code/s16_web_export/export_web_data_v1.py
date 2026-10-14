#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网页展示数据导出（v1）
================================================================
为静态交互网页（GitHub Pages）导出轻量数据：
  data/meta.json                     病种/年份/情景/色阶/国名 元数据
  data/countries.geojson             简化国界（hover/点击联动）
  data/trends_bmi_national.json      国家 BMI 趋势（历史 2015/2020/2024 + 四 SSP 2025-2050）
  data/trends_burden_<code>.json     各病种国家归因负担趋势（GBD 2015-2021 + 四 SSP 2025-2050）
  maps/bmi_<metric>_<ssp>_<year>.webp      BMI 地图叠加层（0.25°，透明底）
  maps/burden_<code>_<measure>_<ssp>_<year>.webp  归因负担地图叠加层

数据口径与论文图一致（midpoints、PowerNorm、色系相同）。
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features as rfeatures
import rasterio.warp as rwarp
from affine import Affine
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from PIL import Image

PROJECT = Path(r"D:\try\results\未来情景预测")
ROOT = PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
OUT = ROOT / "web_v1"
(OUT / "data").mkdir(parents=True, exist_ok=True)
(OUT / "maps").mkdir(parents=True, exist_ok=True)

E = PROJECT / "11_all_disease_extension"
BENCH = E / "04_burden" / "burden_benchmark_v1_20261005" / "data"
COUNTRY_OVERALL = BENCH / "attributable_future_country_overall_four_ssp_v1.parquet"
HIST_BENCH = BENCH / "benchmark_historical_country_year_detail_v1.parquet"
GRID_DIR = E / "06_spatial_burden" / "burden_density_v1_20261007" / "data"
GRID15_DIR = E / "06_spatial_burden" / "burden_density_2015_v1" / "data"
ADM1_DIR = (PROJECT / "06_future_bmi_2025_2050" / "phase7_multiscale_delivery_v3_20260908"
            / "04_adm1" / "tables")
FUT_GRID = (PROJECT / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906"
            / "outputs")
HIST_GRID = (PROJECT / "05_historical_bmi_1990_2024" / "historical_25plus_compat_v1_20260909"
             / "data" / "0p1deg")
COUNTRIES_PATH = r"D:\try\data\地图数据\ne_10m_admin_0_countries.zip"

DISEASES = [(441, "CRC", "Colorectal cancer"), (493, "IHD", "Ischemic heart disease"),
            (498, "HHD", "Hypertensive heart disease"), (976, "T2DM", "Type 2 diabetes mellitus")]
MIDPOINTS = np.asarray([17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5], dtype=np.float32)
CATS = ["lt18p5", "18p5_to_lt20", "20_to_lt25", "25_to_lt30", "30_to_lt35", "35_to_lt40", "ge40"]
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]
SSP_LABEL = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
BMI_MAP_YEARS = {"ssp245": [2015, 2020, 2024, 2025, 2035, 2050],
                 "ssp126": [2050], "ssp370": [2050], "ssp585": [2050]}
BURDEN_MAP_YEARS = {"ssp245": [2015, 2025, 2050], "ssp126": [2050], "ssp370": [2050], "ssp585": [2050]}

SEQ = LinearSegmentedColormap.from_list(
    "web_seq", ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"], N=256)

# 0.25° 全球网格（-180..180, -90..90）
OUT_RES = 0.25
DST_W, DST_H = int(360 / OUT_RES), int(180 / OUT_RES)
DST_TRANSFORM = Affine(OUT_RES, 0, -180, 0, -OUT_RES, 90)


def load_countries():
    c = gpd.read_file(COUNTRIES_PATH)
    c = c[c.geometry.notna()].copy()
    c["iso3"] = c["ISO_A3"].where(c["ISO_A3"].ne("-99"), c["ADM0_A3"]).astype(str).str.upper()
    c["iso3"] = c["iso3"].replace({"PSX": "PSE", "SDS": "SSD"})
    return c


def export_countries_geojson(countries):
    g = countries[["iso3", "NAME", "geometry"]].copy()
    g["name"] = g["NAME"]
    g.geometry = g.geometry.simplify(0.1, preserve_topology=True)
    g[["iso3", "name", "geometry"]].to_file(OUT / "data" / "countries.geojson", driver="GeoJSON")
    print("countries.geojson", (OUT / "data" / "countries.geojson").stat().st_size // 1024, "KB")


# ------------------------------------------------------------------ 栅格工具 ----
def read_grid(path):
    with rasterio.open(path) as ds:
        arr = np.ma.filled(ds.read(1, masked=True).astype(np.float32), np.nan)
        return arr, ds.bounds


def read_grid_band(path, band):
    with rasterio.open(path) as ds:
        arr = np.ma.filled(ds.read(band, masked=True).astype(np.float32), np.nan)
        return arr, ds.bounds


def downsample(arr, resampling):
    """0.1° (-90..90) → 0.25° 全球网格。"""
    if np.isnan(arr).all():
        return np.full((DST_H, DST_W), np.nan, dtype=np.float32)
    src_t = Affine(0.1, 0, -180, 0, -0.1, 90)
    filled = np.nan_to_num(arr, nan=0.0)
    dst = np.zeros((DST_H, DST_W), dtype=np.float32)
    rwarp.reproject(filled, dst, src_transform=src_t, src_crs="EPSG:4326",
                    dst_transform=DST_TRANSFORM, dst_crs="EPSG:4326",
                    resampling=resampling, src_nodata=0.0, dst_nodata=0.0)
    dst[dst == 0] = np.nan
    return dst


def save_overlay(arr, norm, path, alpha=235):
    """按色带渲染 RGBA 并保存 WebP；NaN 全透明。"""
    rgba = (SEQ(norm(arr)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(arr), alpha, 0).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path, "WEBP", quality=88, method=4)


# ------------------------------------------------------------------ BMI 趋势 ----
def bmi_trends_national():
    """未来：ADM1 overall 表按国家人口加权聚合；历史：0.1° 栅格聚合。"""
    rows = []
    for ssp in SSPS:
        for year_dir in sorted((ADM1_DIR / ssp).iterdir()):
            if not year_dir.is_dir():
                continue
            f = year_dir / f"BMI7_{ssp}_{year_dir.name}_ADM1_overall_adults25plus_v2.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f, columns=["adm0_a3", "year", "adult_population",
                                             "mean_bmi", "obesity", "overweight_obesity"])
            df = df[df["adult_population"] > 0]
            df["mean_bmi"] *= df["adult_population"]
            df["obesity"] *= df["adult_population"]
            df["overweight_obesity"] *= df["adult_population"]
            g = df.groupby("adm0_a3")[["mean_bmi", "obesity", "overweight_obesity",
                                       "adult_population"]].sum()
            g["mean_bmi"] /= g["adult_population"]
            g["obesity"] /= g["adult_population"]
            g["overweight_obesity"] /= g["adult_population"]
            for iso, r in g.iterrows():
                rows.append((ssp, iso, int(year_dir.name), r["mean_bmi"],
                             r["overweight_obesity"] * 100, r["obesity"] * 100))
        print("bmi future agg done:", ssp, flush=True)
    out = {}
    for ssp, iso, yr, mb, ow, ob in rows:
        out.setdefault(iso, {}).setdefault(ssp, {})[str(yr)] = [round(mb, 2), round(ow, 1), round(ob, 1)]
    return out


def bmi_trends_historical(countries, out):
    """2015/2020/2024 历史栅格 → 国家人口加权。"""
    for year in (2015, 2020, 2024):
        d = HIST_GRID / str(year)
        shares = []
        bounds = None
        for cat in CATS:
            f = d / f"global_{year}_bmi_{cat}_regional_steps_mrp_0p1deg_bp.tif"
            arr, bounds = read_grid(f)
            shares.append(arr / 10000.0)  # uint16 basis points (0-10000) → 比例
        shares = np.stack(shares)  # 7 x H x W
        pop, _ = read_grid(d / f"global_{year}_adult_population_regional_steps_mrp_0p1deg.tif")
        mean_bmi = np.tensordot(MIDPOINTS, shares, axes=(0, 0))
        ge25 = shares[3:].sum(axis=0) * 100
        ge30 = shares[4:].sum(axis=0) * 100
        # 国家分区
        with rasterio.open(d / f"global_{year}_adult_population_regional_steps_mrp_0p1deg.tif") as ds:
            transform, shape = ds.transform, ds.shape
        keys = sorted(countries["iso3"].unique())
        ids = {k: i for i, k in enumerate(keys)}
        zones = rfeatures.rasterize(
            [(g, ids[i]) for g, i in zip(countries.geometry, countries["iso3"])],
            out_shape=shape, transform=transform, fill=-1, dtype="int32")
        sel = (zones.ravel() >= 0) & np.isfinite(pop.ravel()) & (pop.ravel() > 0)
        z = zones.ravel()[sel]
        w = pop.ravel()[sel].astype(np.float64)
        for metric, arr in (("mean_bmi", mean_bmi), ("ge25", ge25), ("ge30", ge30)):
            v = arr.ravel()[sel]
            s = np.bincount(z, weights=v * w, minlength=len(keys))
            p = np.bincount(z, weights=w, minlength=len(keys))
            agg = np.divide(s, p, out=np.full(len(keys), np.nan), where=p > 0)
            for i, iso in enumerate(keys):
                if np.isfinite(agg[i]):
                    rec = out.setdefault(iso, {}).setdefault("historical", {}).setdefault(str(year), [None, None, None])
                    idx = {"mean_bmi": 0, "ge25": 1, "ge30": 2}[metric]
                    rec[idx] = round(float(agg[i]), 2 if metric == "mean_bmi" else 1)
        print("bmi historical agg done:", year, flush=True)
    return out


# ------------------------------------------------------------------ BMI 地图 ----
def bmi_grids(ssp, year):
    """返回 (mean_bmi_grid, obesity_pct_grid)（0.1°）。"""
    if year <= 2024:
        d = HIST_GRID / str(year)
        shares = []
        for cat in CATS:
            arr, b = read_grid(d / f"global_{year}_bmi_{cat}_regional_steps_mrp_0p1deg_bp.tif")
            shares.append(arr / 10000.0)  # uint16 basis points (0-10000) → 比例
        shares = np.stack(shares)
        mean_bmi = np.tensordot(MIDPOINTS, shares, axes=(0, 0)).astype(np.float32)
        ge30 = (shares[4:].sum(axis=0) * 100).astype(np.float32)
        # 无数据区掩膜：人口为 0 或无权重
        mean_bmi[shares.sum(axis=0) <= 0] = np.nan
        ge30[shares.sum(axis=0) <= 0] = np.nan
        return mean_bmi, ge30
    f = FUT_GRID / ssp / str(year) / f"probability_overall_{ssp}_{year}_0p1deg_v1.tif"
    with rasterio.open(f) as ds:
        prob = np.ma.filled(ds.read(masked=True).astype(np.float32), np.nan)  # 7 x H x W
    mean_bmi = np.tensordot(MIDPOINTS, prob, axes=(0, 0)).astype(np.float32)
    ge30 = (prob[4:].sum(axis=0) * 100).astype(np.float32)
    s = np.nansum(prob, axis=0)
    mean_bmi[s <= 0] = np.nan
    ge30[s <= 0] = np.nan
    return mean_bmi, ge30


def export_bmi_maps():
    from matplotlib.colors import Normalize
    norms = {"meanbmi": Normalize(20, 34), "obesity": Normalize(0, 60)}  # 线性即可读
    for ssp, years in BMI_MAP_YEARS.items():
        for year in years:
            mb, ob = bmi_grids(ssp, year)
            for name, arr in (("meanbmi", mb), ("obesity", ob)):
                d = downsample(arr, rwarp.Resampling.average)
                save_overlay(d, norms[name], OUT / "maps" / f"bmi_{name}_{ssp}_{year}.webp")
            print("bmi map:", ssp, year, flush=True)


# ------------------------------------------------------------------ 负担趋势 ----
def export_burden_trends():
    fut = pd.read_parquet(COUNTRY_OVERALL)
    fut = fut[fut["analysis_tier"] == "headline"]
    hist = pd.read_parquet(HIST_BENCH)
    hist = hist[hist["analysis_tier"] == "headline"]
    for cid, code, _ in DISEASES:
        f = fut[fut["cause_id"] == cid]
        h = hist[hist["cause_id"] == cid]
        out = {}
        for ssp in SSPS:
            sub = f[f["scenario_code"] == ssp]
            piv_d = sub[sub["measure"] == "Deaths"].pivot_table(
                index="iso3", columns="year", values="attributable_burden", aggfunc="sum")
            piv_l = sub[sub["measure"] == "DALYs"].pivot_table(
                index="iso3", columns="year", values="attributable_burden", aggfunc="sum")
            for iso in piv_d.index:
                rec = out.setdefault(ssp, {}).setdefault(iso, {})
                for yr in piv_d.columns:
                    d_, l_ = piv_d.loc[iso, yr], piv_l.loc[iso, yr]
                    rec[str(int(yr))] = [int(round(d_)), int(round(l_))]
            # 历史段并入 ssp245（参考情景口径，与论文图一致）
            if ssp == "ssp245":
                hd = h[h["measure"] == "Deaths"].pivot_table(
                    index="iso3", columns="year", values="study_attributable", aggfunc="sum")
                hl = h[h["measure"] == "DALYs"].pivot_table(
                    index="iso3", columns="year", values="study_attributable", aggfunc="sum")
                for iso in hd.index:
                    rec = out["ssp245"].setdefault(iso, {})
                    for yr in hd.columns:
                        rec[str(int(yr))] = [int(round(hd.loc[iso, yr])),
                                             int(round(hl.loc[iso, yr]))]
        # 全球汇总
        glob = {}
        for ssp in SSPS:
            g = {}
            for iso, rec in out[ssp].items():
                for yr, v in rec.items():
                    g[yr] = [a + b for a, b in zip(g.get(yr, [0, 0]), v)]
            glob[ssp] = g
        out["_global"] = glob
        (OUT / "data" / f"trends_burden_{code}.json").write_text(
            json.dumps(out, separators=(",", ":")), encoding="utf-8")
        print("burden trend:", code, flush=True)


# ------------------------------------------------------------------ 负担地图 ----
def export_burden_maps():
    # 两遍扫描：先按 病种×指标 求 ssp245 全年份 99.5 分位统一色阶，再逐张写出（不驻留内存）
    for cid, code, _ in DISEASES:
        for measure in ("deaths", "dalys"):
            vals = []
            for year in BURDEN_MAP_YEARS["ssp245"]:
                if year == 2015:
                    f = GRID15_DIR / f"burden_density_c{cid}_{measure}_ssp245_2015_0p1deg_v1.tif"
                else:
                    f = GRID_DIR / str(year) / f"burden_density_c{cid}_{measure}_ssp245_{year}_0p1deg_v1.tif"
                arr, _ = read_grid(f)
                vals.append(arr[np.isfinite(arr)])
                del arr
            vmax = float(np.percentile(np.concatenate(vals), 99.5))
            del vals
            norm = PowerNorm(0.5, 0, vmax)
            for ssp, years in BURDEN_MAP_YEARS.items():
                for year in years:
                    if year == 2015:
                        f = GRID15_DIR / f"burden_density_c{cid}_{measure}_ssp245_2015_0p1deg_v1.tif"
                    else:
                        f = GRID_DIR / str(year) / f"burden_density_c{cid}_{measure}_ssp245_{year}_0p1deg_v1.tif"
                    arr, _ = read_grid(f)
                    d = downsample(arr, rwarp.Resampling.sum)
                    save_overlay(d, norm, OUT / "maps" / f"burden_{code}_{measure}_{ssp}_{year}.webp")
                    del arr, d
            print("burden map:", code, measure, f"vmax={vmax:.3g}", flush=True)
            BURDEN_NORMS[f"{code}_{measure}"] = vmax


BURDEN_NORMS: dict = {}


# ------------------------------------------------------------------ meta ----
def export_meta(countries):
    names = dict(zip(countries["iso3"], countries["NAME"]))
    meta = {
        "diseases": [{"cid": cid, "code": code, "name": name} for cid, code, name in DISEASES],
        "scenarios": [{"code": s, "label": SSP_LABEL[s]} for s in SSPS],
        "bmi_metrics": [{"code": "meanbmi", "label": "Mean BMI", "unit": "kg/m²"},
                        {"code": "obesity", "label": "Obesity (BMI ≥ 30)", "unit": "%"}],
        "bmi_map_years": BMI_MAP_YEARS,
        "burden_map_years": BURDEN_MAP_YEARS,
        "burden_measures": [{"code": "deaths", "label": "Deaths"},
                            {"code": "dalys", "label": "DALYs"}],
        "country_names": names,
        "colorbar": {
            "meanbmi": {"label": "Mean BMI (kg/m²)", "vmin": 20, "vmax": 34},
            "obesity": {"label": "Obesity prevalence (%)", "vmin": 0, "vmax": 60},
        },
        "burden_norms": {k: round(v, 6) for k, v in BURDEN_NORMS.items()},
    }
    (OUT / "data" / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print("meta.json")


def export_bmi_global_trend():
    """全球 BMI 趋势（Figure 2 聚合源表，Both sexes 人口加权）。"""
    f = ROOT / "figure2_bmi_global_2015_2050_v1" / "tables" / "Figure2_sex_trend_source_v1.csv"
    df = pd.read_csv(f)
    df["w"] = df["population"]
    g = df.groupby(["year", "period"]).apply(
        lambda d: pd.Series({
            "mean_bmi": float((d["mean_bmi"] * d["w"]).sum() / d["w"].sum()),
            "ge25": float((d["bmi_ge25"] * d["w"]).sum() / d["w"].sum()),
            "ge30": float((d["bmi_ge30"] * d["w"]).sum() / d["w"].sum()),
        }), include_groups=False).reset_index()
    out = {}
    for _, r in g.iterrows():
        key = "historical" if r["period"] == "Historical" else "ssp245"
        out.setdefault(key, {})[str(int(r["year"]))] = [round(r["mean_bmi"], 2),
                                                         round(r["ge25"], 1), round(r["ge30"], 1)]
    (OUT / "data" / "trends_bmi_global.json").write_text(
        json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("trends_bmi_global.json")


def wrap_json_as_js():
    """把 data/*.json 与 countries.geojson 包装为 .js（window.X = ...），
    使网页在 file:// 下直接双击可开（fetch 在 file:// 被 CORS 拦截）。"""
    wraps = {"meta.json": "META", "trends_bmi_national.json": "BMI_TRENDS",
             "trends_bmi_global.json": "BMI_GLOBAL", "countries.geojson": "COUNTRIES"}
    for _, code, _ in DISEASES:
        wraps[f"trends_burden_{code}.json"] = f"BURDEN_{code}"
    for fname, var in wraps.items():
        p = OUT / "data" / fname
        obj = json.loads(p.read_text(encoding="utf-8"))
        (p.with_suffix(".js")).write_text(
            f"window.{var} = " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + ";\n",
            encoding="utf-8")
    print("js wrappers written")


def main():
    countries = load_countries()
    export_countries_geojson(countries)
    bmi = bmi_trends_national()
    bmi = bmi_trends_historical(countries, bmi)
    (OUT / "data" / "trends_bmi_national.json").write_text(
        json.dumps(bmi, separators=(",", ":")), encoding="utf-8")
    print("trends_bmi_national.json",
          (OUT / "data" / "trends_bmi_national.json").stat().st_size // 1024, "KB", flush=True)
    export_bmi_global_trend()
    export_burden_trends()
    export_bmi_maps()
    export_burden_maps()
    export_meta(countries)
    wrap_json_as_js()
    print("DONE")


if __name__ == "__main__":
    main()
