#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S15 系列 v2：单病种综合图（0.1° 栅格地图，无重叠布局）
================================================================
相对 v1 (78_plot_figureS15_composites_v1.py) 的修改：
  1. 所有地图面板改为 0.1° 栅格（burden_density tif），不再用国家面填色；
  2. 全面重排版面，消除标题/色条/面板之间的所有重叠（标题+副题用实际渲染宽度测量）；
  3. (h) Top20 使用国家名（crosswalk_location_name），不再用 ISO3；
  4. (a) 不再单独放色条，与 (b)(c) 共享量纲；(d) 副题去掉 "pp"（数值为 deaths）。

数据：
  0.1° 栅格：06_spatial_burden/burden_density_v1_20261007/data/{2025,2050}/
             burden_density_c{cid}_{deaths,dalys}_ssp245_{year}_0p1deg_v1.tif
  国家表  ：04_burden/burden_benchmark_v1_20261005/data/attributable_future_country_*_four_ssp_v1.parquet
  性别地图：grid x 各国男性占比（0.1° 重采样 share）
  情景敏感：grid2050 x 各国 (ssp585-ssp126)/ssp245 比值
输出：figureS15_disease_composites_v1/panels/FigureS15_composite_{code}_v2.{pdf,tif}（1800dpi LZW）
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

import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features as rfeatures
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
OUT = ROOT / "figureS15_disease_composites_v1"
PANELS = OUT / "panels"
QA = OUT / "qa"

E = PROJECT / "11_all_disease_extension"
BENCH = E / "04_burden" / "burden_benchmark_v1_20261005" / "data"
COUNTRY_OVERALL = BENCH / "attributable_future_country_overall_four_ssp_v1.parquet"
COUNTRY_SEX = BENCH / "attributable_future_country_sex_four_ssp_v1.parquet"
HIST_COUNTRY = BENCH / "benchmark_historical_country_year_detail_v1.parquet"
GRID_DIR = E / "06_spatial_burden" / "burden_density_v1_20261007" / "data"
POP50 = (PROJECT / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906"
         / "outputs" / "ssp245" / "2050" / "population_age_sex_ssp245_2050_0p1deg_v1.tif")
COUNTRIES_PATH = str(_cfg.DATA / "地图数据" / "ne_10m_admin_0_countries.zip")
HELPER_SCRIPT = (ROOT / "scripts" / "08_plot_figure1B_historical_template_test_v2.py")

DISEASES = [
    (441, "CRC", "Colon and rectum cancer"),
    (493, "IHD", "Ischemic heart disease"),
    (498, "HHD", "Hypertensive heart disease"),
    (976, "T2DM", "Diabetes mellitus type 2"),
]
TIFF_DPI = 1800
BLACK = "#17191A"
GREY = "#747C80"
BLUE_LINE = "#2676A8"
C_MALE, C_FEMALE = "#2676A8", "#F2C14E"

PALETTE = LinearSegmentedColormap.from_list(
    "s15_seq", ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"], N=256)
DIVERGING = LinearSegmentedColormap.from_list(
    "s15_div", ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"],
    N=256)
NODATA = (0.965, 0.965, 0.965, 1.0)
PALETTE.set_bad(NODATA)
DIVERGING.set_bad(NODATA)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def import_helper():
    spec = importlib.util.spec_from_file_location("s15_helper", HELPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    import_helper()  # 与 v1 相同：字体/环境副作用
except Exception as exc:
    print("helper import skipped:", exc)


def load_countries():
    try:
        c = gpd.read_file(COUNTRIES_PATH)
    except Exception:
        c = gpd.read_file(f"zip://{COUNTRIES_PATH}")
    c = c[c.geometry.notna()].copy()
    c["iso3"] = c["ISO_A3"].where(c["ISO_A3"].ne("-99"), c["ADM0_A3"]).astype(str).str.upper()
    c["iso3"] = c["iso3"].replace({"PSX": "PSE", "SDS": "SSD"})
    c.geometry = c.geometry.simplify(0.15, preserve_topology=True)
    return c


def to_np(series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").astype(float).to_numpy()


def fmt(v, digits=1):
    if abs(v) >= 1e6:
        return f"{v/1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:.0f}k"
    return f"{v:.0f}"


# ------------------------------------------------------------------ 数据 ----
def load_grid(cid: int, year: int, measure: str = "deaths"):
    """读 0.1° 归因负担栅格，返回 (grid[nan 填充], bounds, transform)。
    2015 年历史栅格在 burden_density_2015_v1 目录（无年份子目录）。"""
    if year == 2015:
        p = (GRID_DIR.parent.parent / "burden_density_2015_v1" / "data"
             / f"burden_density_c{cid}_{measure}_ssp245_2015_0p1deg_v1.tif")
    else:
        p = GRID_DIR / str(year) / f"burden_density_c{cid}_{measure}_ssp245_{year}_0p1deg_v1.tif"
    with rasterio.open(p) as ds:
        grid = np.ma.filled(ds.read(1, masked=True).astype(np.float32), np.nan)
        b = ds.bounds
        transform = ds.transform
    return grid, (b.left, b.right, b.bottom, b.top), transform


def disease_data(cid: int) -> dict:
    """国家表：轨迹 / 2050 国家值 / 性别 / 情景差（与 v1 同口径）。"""
    ov = pd.read_parquet(COUNTRY_OVERALL)
    ov = ov[(ov["analysis_tier"] == "headline") & (ov["cause_id"] == cid)]
    names = ov.drop_duplicates("iso3").set_index("iso3")["crosswalk_location_name"]
    sx = pd.read_parquet(COUNTRY_SEX)
    sx = sx[(sx["analysis_tier"] == "headline") & (sx["cause_id"] == cid)
            & (sx["scenario_code"] == "ssp245") & (sx["year"] == 2050)
            & (sx["measure"] == "Deaths")]
    out = {"names": names}
    hh = pd.read_parquet(HIST_COUNTRY)
    hh = hh[(hh["analysis_tier"] == "headline") & (hh["cause_id"] == cid)]
    for meas, key in (("Deaths", "deaths"), ("DALYs", "dalys")):
        sub = ov[(ov["measure"] == meas) & (ov["scenario_code"] == "ssp245")]
        traj = sub.groupby("year")[["total_burden", "attributable_burden"]].sum().sort_index()
        out[f"traj_{key}"] = traj
        out[f"nat2050_{key}"] = sub[sub["year"] == 2050].groupby("iso3")["attributable_burden"].sum()
        hsub = hh[hh["measure"] == meas]
        out[f"hist_{key}"] = (hsub.groupby("year")[["gbd_total_burden", "study_attributable"]]
                              .sum().sort_index())
    sex = sx.pivot_table(index="iso3", columns="sex", values="attributable_burden").fillna(0.0)
    out["male"] = sex.get("Male", pd.Series(dtype=float))
    out["female"] = sex.get("Female", pd.Series(dtype=float))
    s_all = ov[(ov["year"] == 2050) & (ov["measure"] == "Deaths")
               & (ov["scenario_code"].isin(["ssp126", "ssp585"]))]
    sens = s_all.pivot_table(index="iso3", columns="scenario_code",
                             values="attributable_burden")
    out["sens"] = sens.get("ssp585", pd.Series(dtype=float)) - sens.get("ssp126", pd.Series(dtype=float))
    both = sex[(sex.sum(axis=1) > 0)]
    rel = ((both["Male"] - both["Female"]) / ((both["Male"] + both["Female"]) / 2.0) * 100.0)
    out["sex_gap_median"] = float(np.median(np.abs(rel))) if len(rel) else 0.0
    out["keep_sex_panel"] = out["sex_gap_median"] >= 15.0
    return out


def rasterize_by_country(countries, values: pd.Series, shape, transform, fill=np.nan):
    """把 iso3->value 重采样到 0.1° 栅格。"""
    cmap = pd.to_numeric(countries["iso3"].map(values), errors="coerce").astype(float).to_numpy()
    shapes = [(g, float(v)) for g, v in zip(countries.geometry, cmap)
              if g is not None and np.isfinite(v)]
    arr = rfeatures.rasterize(shapes, out_shape=shape, transform=transform,
                              fill=np.nan, dtype="float32")
    return arr


def build_grids(cid: int, countries, data: dict):
    """组装全部 0.1° 栅格面板数据。变化图 = 2050 - 2015。
    注意：2015 栅格范围为 lat -60..84（1440 行），2050 为 -90..90（1800 行），
    需按纬度对齐（行偏移 = (top_2050 - top_2015)/0.1）。"""
    g50, bounds, transform = load_grid(cid, 2050, "deaths")
    g15, b15, _ = load_grid(cid, 2015, "deaths")
    if g15.shape != g50.shape:
        res = (bounds[3] - bounds[2]) / g50.shape[0]
        r0 = int(round((bounds[3] - b15[3]) / res))  # 2015 顶行在 2050 网格中的行号
        aligned = np.full(g50.shape, np.nan, dtype=np.float32)
        r1 = min(r0 + g15.shape[0], g50.shape[0])
        aligned[r0:r1, :] = g15[:r1 - r0, :g50.shape[1]]
        g15 = aligned
    shape = g50.shape
    male, female = data["male"], data["female"]
    tot = (male.fillna(0.0) + female.fillna(0.0))
    share = (male / tot.replace(0.0, np.nan)).clip(0, 1)
    share_r = rasterize_by_country(countries, share, shape, transform)
    grid_male = g50 * share_r
    grid_female = g50 * (1.0 - share_r)
    grid_diff = g50 * (2.0 * share_r - 1.0)
    nat = data["nat2050_deaths"]
    ratio = (data["sens"] / nat.replace(0.0, np.nan))
    ratio_r = rasterize_by_country(countries, ratio, shape, transform)
    grid_sens = g50 * ratio_r
    grid_change = g50 - g15
    return {
        "bounds": bounds, "transform": transform, "shape": shape,
        "overall": g50, "male": grid_male, "female": grid_female,
        "diff": grid_diff, "change": grid_change, "sens": grid_sens,
    }


def grid_income(cid: int, countries, shape, transform):
    """0.1° 栅格 + 收入组分区 + 人口（小提琴图）。"""
    grid, _, _ = load_grid(cid, 2050, "deaths")
    grid = grid[:shape[0], :shape[1]]
    with rasterio.open(POP50) as ds:
        pop = np.ma.filled(ds.read(1, masked=True).astype(np.float32), np.nan)
        pop = pop[:shape[0], :shape[1]]
    inc_map = {"1. High income: OECD": "High income", "2. High income: nonOECD": "High income",
               "3. Upper middle income": "Upper-middle income",
               "4. Lower middle income": "Lower-middle income", "5. Low income": "Low income"}
    countries = countries.copy()
    countries["income"] = countries["INCOME_GRP"].map(inc_map)
    keys = ["Low income", "Lower-middle income", "Upper-middle income", "High income"]
    ids = {k: i for i, k in enumerate(keys)}
    shapes = [(g, ids[v]) for g, v in zip(countries.geometry, countries["income"]) if v in ids]
    zones = rfeatures.rasterize(shapes, out_shape=shape, transform=transform,
                                fill=-1, dtype="int32")
    return grid, pop, zones, keys


# ------------------------------------------------------------------ 绘图 ----
def _wfrac(fig, s, fontsize, bold=False) -> float:
    """文本在图中的宽度（figure fraction），用真实渲染器测量。"""
    t = fig.text(0, 0, s, fontsize=fontsize, fontweight="bold" if bold else "normal")
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    t.remove()
    return bb.width / (fig.get_size_inches()[0] * fig.dpi)


def panel_header(fig, rect, title, sub=None, fontsize=8.5, sub_fontsize=6.5):
    """面板标题（左，加粗）+ 副题（右对齐，灰色）；按实测宽度防重叠：
    标题超宽时自动缩字号；副题缩到 5pt 仍放不下则自动省略。"""
    y = rect[1] + rect[3] + 0.008
    fs_t = fontsize
    while _wfrac(fig, title, fs_t, True) > (0.995 - rect[0]) and fs_t > 6.5:
        fs_t -= 0.25
    fig.text(rect[0], y, title, ha="left", va="bottom", fontsize=fs_t,
             fontweight="bold", color=BLACK)
    if not sub:
        return
    avail = rect[2] - _wfrac(fig, title, fs_t, True) - 0.012
    fs = sub_fontsize
    while _wfrac(fig, sub, fs) > avail and fs > 5.0:
        fs -= 0.3
    if _wfrac(fig, sub, fs) <= avail:
        fig.text(rect[0] + rect[2], y, sub, ha="right", va="bottom", fontsize=fs, color=GREY)


def draw_raster_map(ax, grid, bounds, cmap, norm, countries=None, region=None,
                    border=False):
    """0.1° 栅格地图；region=(lon0,lon1,lat0,lat1) 时裁剪放大。"""
    if region is None:
        sub = grid
        ext = bounds  # imshow 数据范围保持完整
        view = (bounds[0], bounds[1], max(bounds[2], -60.0), min(bounds[3], 84.0))
    else:
        lon0, lon1, lat0, lat1 = region
        res = (bounds[1] - bounds[0]) / grid.shape[1]
        c0 = max(int((lon0 - bounds[0]) / res), 0)
        c1 = min(int((lon1 - bounds[0]) / res) + 1, grid.shape[1])
        r0 = max(int((bounds[3] - lat1) / res), 0)
        r1 = min(int((bounds[3] - lat0) / res) + 1, grid.shape[0])
        sub = grid[r0:r1, c0:c1]
        ext = (lon0, lon1, lat0, lat1)
        view = ext
    ax.imshow(np.ma.masked_invalid(sub), extent=ext, origin="upper",
              cmap=cmap, norm=norm, interpolation="nearest", aspect="auto",
              rasterized=True, zorder=1)
    if countries is not None:
        if region is None:
            countries.boundary.plot(ax=ax, linewidth=0.10, color="#5A6165", zorder=2)
        else:
            countries.boundary.plot(ax=ax, linewidth=0.35, color="#5A6165", zorder=2)
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(border)
        if border:
            sp.set_edgecolor(BLACK)
            sp.set_linewidth(0.9)


def draw_colorbar(fig, rect, cmap, norm, label, ticks=None):
    """水平色条：标签在条上方居中（省垂直空间、避免与下行标题重叠）。"""
    cax = fig.add_axes(rect)
    if ticks is None:
        ticks = np.linspace(norm.vmin, norm.vmax, 4)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, ticks=ticks,
                                   orientation="horizontal",
                                   format=FuncFormatter(lambda v, p: f"{v:.2g}"))
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.2)
    cb.outline.set_visible(False)
    fig.text(rect[0] + rect[2] / 2, rect[1] + rect[3] + 0.004, label,
             ha="center", va="bottom", fontsize=6.5, color=BLACK)
    return cb


def top_charts(fig, data, code):
    """顶部 (a)：Total vs BMI-attributable，Deaths | DALYs；2015–2021 历史 + 2025–2050 预测，
    两段直接相连，2021.5 处竖虚线分隔。"""
    fig.text(0.035, 0.986, f"(a) Global {code} burden attributable to high BMI",
             ha="left", va="top", fontsize=9.5, fontweight="bold", color=BLACK)
    for i, (key, meas) in enumerate((("deaths", "Deaths"), ("dalys", "DALYs"))):
        ax = fig.add_axes([0.075 + i * 0.475, 0.815, 0.375, 0.110])
        hist, traj = data[f"hist_{key}"], data[f"traj_{key}"]
        x = np.concatenate([hist.index.to_numpy(), traj.index.to_numpy()])
        tot = np.concatenate([to_np(hist["gbd_total_burden"]) / 1e6,
                              to_np(traj["total_burden"]) / 1e6])
        att = np.concatenate([to_np(hist["study_attributable"]) / 1e6,
                              to_np(traj["attributable_burden"]) / 1e6])
        ax.plot(x, tot, color=GREY, linewidth=1.2, zorder=2)
        ax.plot(x, att, color=BLUE_LINE, linewidth=1.5, zorder=3)
        ax.axvline(2021.5, color="#B0B7BC", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
        ax.scatter([2050], [att[-1]], s=12, facecolor="white", edgecolor=BLUE_LINE,
                   linewidth=0.8, zorder=4)
        t50, a50 = float(tot[-1]), float(att[-1])
        share = a50 / t50 * 100.0 if t50 > 0 else float("nan")
        ax.annotate(f"{fmt(t50 * 1e6)} total", (2050, t50), xytext=(-3, 5),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=6.5, color=GREY)
        ax.annotate(f"{fmt(a50 * 1e6)} attributable\n{share:.2f}% of total",
                    (2050, a50), xytext=(-3, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.5, color=BLUE_LINE, linespacing=1.1)
        ax.set_xlim(2015, 2050.5)
        ax.set_ylim(0, float(np.nanmax(tot)) * 1.16)
        ax.set_xticks([2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}M"))
        ax.set_title(meas, loc="left", fontsize=8.5, fontweight="bold", pad=3, color=BLACK)
        ax.set_xlabel("Year", fontsize=7.5, labelpad=1.5)
        ax.set_ylabel("Burden", fontsize=7.5, labelpad=1.5)
        ax.tick_params(labelsize=6.8, length=2.2, width=0.5, pad=1.2)
        ax.grid(axis="y", color="#E9ECEE", linewidth=0.4)
        ax.set_axisbelow(True)
        if i == 0:
            if code in ("HHD", "T2DM"):
                # 两线间距窄，图例置于图上侧（左上空白区）
                ax.legend(handles=[Line2D([], [], color=GREY, lw=1.2, label="Total burden"),
                                   Line2D([], [], color=BLUE_LINE, lw=1.5, label="BMI-attributable burden")],
                          loc="upper left", bbox_to_anchor=(0.01, 0.98),
                          fontsize=6.2, handlelength=1.6, labelspacing=0.35)
            else:
                # 图例放在两条曲线之间的最大空隙（按 2018 年处两线中点定位），避免压线
                i2018 = int(np.argmin(np.abs(x - 2018)))
                y_mid = 0.5 * (tot[i2018] + att[i2018]) / ax.get_ylim()[1]
                ax.legend(handles=[Line2D([], [], color=GREY, lw=1.2, label="Total burden"),
                                   Line2D([], [], color=BLUE_LINE, lw=1.5, label="BMI-attributable burden")],
                          loc="center left", bbox_to_anchor=(0.01, y_mid),
                          fontsize=6.2, handlelength=1.6, labelspacing=0.35)


def overall_map(fig, rect, countries, grids, norm_shared):
    """(b) Overall burden 0.1° 大地图；三个放大插图置于地图下方独立条带（不遮挡主图）；
    插图下方放共享色条（b/c/d 三个地图面板共用此色阶）。"""
    ax = fig.add_axes(rect)
    draw_raster_map(ax, grids["overall"], grids["bounds"], PALETTE, norm_shared,
                    countries=countries)
    panel_header(fig, rect, "(b) Overall burden", "SSP2-4.5 | 2050")
    # 三个黑框放大插图（地图下方，不遮挡主图）
    regions = [("Gulf + MENA", (25, 68, 10, 42), 0.040),
               ("South Asia", (62, 95, 5, 37), 0.255),
               ("Maritime SE Asia", (95, 142, -12, 22), 0.470)]
    for name, region, x in regions:
        iax = fig.add_axes([x, 0.462, 0.150, 0.073])
        draw_raster_map(iax, grids["overall"], grids["bounds"], PALETTE, norm_shared,
                        countries=countries, region=region, border=True)
        fig.text(x + 0.075, 0.454, name, ha="center", va="top",
                 fontsize=6.5, fontweight="bold", color=BLACK)
    # (b)(c)(d) 共享色条
    vm = norm_shared.vmax
    seq_ticks = [0.0, vm * 0.0625, vm * 0.25, vm]  # PowerNorm(0.5) 下等距
    draw_colorbar(fig, [0.183, 0.422, 0.300, 0.009], PALETTE, norm_shared,
                  "Attributable deaths per 0.1\u00b0 cell", ticks=seq_ticks)
    return ax


def sex_maps(fig, countries, grids, keep_d, norm_shared):
    """(c) Male-only / (d) Female-only / (e) Male-female，全部 0.1° 栅格。
    (c)(d) 与 (b) 共享色阶（色条在 (b) 下方），仅 (e) 带独立发散色条。"""
    panels = [("(c) Male-only", grids["male"], PALETTE, norm_shared, False),
              ("(d) Female-only", grids["female"], PALETTE, norm_shared, False)]
    if keep_d:
        d = grids["diff"]
        lim = float(np.nanpercentile(np.abs(d[np.isfinite(d)]), 99.5)) if np.isfinite(d).any() else 1.0
        nd = Normalize(-max(lim, 1e-9), max(lim, 1e-9))
        panels.append(("(e) Male \u2013 female", d, DIVERGING, nd, True))
    w, xs = 0.285, [0.035, 0.365, 0.695]
    for i, (title, grid, cmap, norm, with_cb) in enumerate(panels):
        rect = [xs[i], 0.285, w, 0.100]
        ax = fig.add_axes(rect)
        draw_raster_map(ax, grid, grids["bounds"], cmap, norm, countries=countries)
        panel_header(fig, rect, title, "SSP2-4.5 | 2050" if i < 2 else "2050")
        if with_cb:
            draw_colorbar(fig, [rect[0] + 0.030, 0.262, rect[2] - 0.060, 0.009],
                          cmap, norm, "Difference (deaths per 0.1\u00b0 cell)",
                          ticks=[norm.vmin, 0, norm.vmax])


def change_sensitivity_maps(fig, countries, grids):
    """(f) 变化图 2050-2015；(g) 情景敏感性，全部 0.1° 栅格。"""
    ch, se = grids["change"], grids["sens"]
    lim_c = float(np.nanpercentile(np.abs(ch[np.isfinite(ch)]), 99.5)) if np.isfinite(ch).any() else 1.0
    lim_s = float(np.nanpercentile(np.abs(se[np.isfinite(se)]), 99.5)) if np.isfinite(se).any() else 1.0
    for i, (title, sub, fs_t, grid, lim) in enumerate((
            ("(f) Change", "2050 \u2013 2015 | SSP2-4.5", 8.5, ch, max(lim_c, 1e-9)),
            ("(g) Scenario gap", "SSP5-8.5 \u2013 SSP1-2.6", 7.0, se, max(lim_s, 1e-9)))):
        rect = [0.035 + i * 0.310, 0.105, 0.270, 0.113]
        ax = fig.add_axes(rect)
        norm = Normalize(-lim, lim)
        draw_raster_map(ax, grid, grids["bounds"], DIVERGING, norm, countries=countries)
        panel_header(fig, rect, title, sub, fontsize=fs_t, sub_fontsize=5.5)
        cb_label = ("Difference (deaths per 0.1\u00b0 cell)" if i == 0
                    else "Difference (deaths per 0.1\u00b0 cell, 2050)")
        draw_colorbar(fig, [rect[0] + 0.030, 0.082, rect[2] - 0.060, 0.009],
                      DIVERGING, norm, cb_label,
                      ticks=[norm.vmin, 0, norm.vmax])


def income_violin(fig, rect, grid, pop, zones, keys):
    """(g) 收入组小提琴+箱线：0.1° 归因死亡分布（log 刻度，数据重尾），
    n=格点数，橙菱形=人口加权均值（真实值，不裁剪）。"""
    ax = fig.add_axes(rect)
    FLOOR = 1e-4
    groups, pops = [], []
    for k in range(len(keys)):
        mask = (zones == k) & np.isfinite(grid)
        groups.append(grid[mask])
        pops.append(pop[mask] if pop.shape == grid.shape else None)
    lg = [np.log10(np.clip(g, FLOOR, None)) if len(g) else np.array([np.log10(FLOOR)])
          for g in groups]
    vp = ax.violinplot(lg, showmeans=False, showmedians=False, widths=0.8)
    for body in vp["bodies"]:
        body.set_facecolor("#C9D2D8")
        body.set_edgecolor("#8A949B")
        body.set_linewidth(0.5)
        body.set_alpha(0.9)
    for part in ("cmaxes", "cmins", "cbars"):
        if part in vp:
            vp[part].set_edgecolor("#8A949B")
            vp[part].set_linewidth(0.5)
    for i, g in enumerate(lg, start=1):
        q1, med, q3 = np.percentile(g, [25, 50, 75]) if len(g) else (0, 0, 0)
        ax.plot([i - 0.14, i + 0.14], [q1, q1], color="white", lw=1.4, zorder=3)
        ax.plot([i - 0.14, i + 0.14], [q3, q3], color="white", lw=1.4, zorder=3)
        ax.plot([i - 0.14, i + 0.14], [med, med], color=BLACK, lw=1.2, zorder=4)
        pm = pops[i - 1]
        if pm is not None and np.nansum(pm) > 0:
            wm = float(np.nansum(groups[i - 1] * pm) / np.nansum(pm))
            ax.plot([i], [np.log10(max(wm, FLOOR))], marker="D", ms=4.2, color="#E8871E",
                    markeredgecolor=BLACK, markeredgewidth=0.4, zorder=5)
    gmax = max(float(np.nanmax(g)) if len(g) else FLOOR for g in groups)
    ytop = np.log10(gmax) + 0.62
    ax.set_ylim(np.log10(FLOOR) - 0.18, ytop)

    def fmt_n(n):
        return f"n={n/1e6:.1f}M" if n >= 1e6 else f"n={n/1e3:.0f}k"

    for i in range(1, len(keys) + 1):
        ax.annotate(fmt_n(len(groups[i - 1])), (i, ytop - 0.16), ha="center",
                    va="top", fontsize=5.8, color=GREY)
    yticks = [t for t in (-4, -3, -2, -1, 0, 1, 2) if np.log10(FLOOR) - 0.1 <= t <= ytop - 0.5]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"$10^{{{t}}}$" for t in yticks])
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([k.replace(" income", "\nincome").replace("-", "-\n") for k in keys],
                       fontsize=6.5)
    ax.set_ylabel("Deaths per 0.1\u00b0 cell (log scale)", fontsize=7.5, labelpad=1.5)
    ax.tick_params(labelsize=6.8, length=2.2, width=0.5, pad=1.2)
    ax.grid(axis="y", color="#E9ECEE", linewidth=0.4)
    ax.set_axisbelow(True)
    panel_header(fig, rect, "(h) Burden by income group", "2050 | overall")
    ax.scatter([], [], marker="D", s=16, color="#E8871E", edgecolor=BLACK,
               linewidth=0.4, label="Population-weighted mean")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=5.8,
              frameon=False, handletextpad=0.4, borderpad=0.0, labelspacing=0.25)


NAME_FIX = {"United States of America": "United States", "Russian Federation": "Russia",
            "Republic of Korea": "South Korea", "Iran (Islamic Republic of)": "Iran",
            "Venezuela (Bolivarian Republic of)": "Venezuela",
            "Syrian Arab Republic": "Syria", "Lao People's Democratic Republic": "Laos",
            "Viet Nam": "Vietnam", "United Republic of Tanzania": "Tanzania",
            "Democratic Republic of the Congo": "DR Congo",
            "Taiwan (Province of China)": "Taiwan"}


def top20_bars(fig, rect, data):
    """(i) Top20 国家条形：蓝=Male，黄=Female；分性别数值标签（水平方向自动防重叠）。"""
    ax = fig.add_axes(rect)
    male, female = data["male"], data["female"]
    overall = (male.fillna(0.0) + female.fillna(0.0))
    top = overall.sort_values(ascending=False).head(20)
    names = data["names"]
    labels = [NAME_FIX.get(str(names.get(i, i)), str(names.get(i, i))) for i in top.index]
    top = top.iloc[::-1]
    labels = labels[::-1]
    ypos = np.arange(len(top))
    mv = np.array([male.get(i, 0.0) for i in top.index])
    fv = np.array([female.get(i, 0.0) for i in top.index])
    ax.barh(ypos - 0.19, mv, height=0.36, color=C_MALE, edgecolor="none", label="Male")
    ax.barh(ypos + 0.19, fv, height=0.36, color=C_FEMALE, edgecolor="none", label="Female")
    xmax = float(max(mv.max() if len(mv) else 1, fv.max() if len(fv) else 1))
    xlim1 = xmax * 1.30
    off = xmax * 0.015

    def wdata(s, fs):
        return _wfrac(fig, s, fs) / rect[2] * xlim1

    for y, a, b in zip(ypos, mv, fv):
        xa = a + off
        ax.text(xa, y - 0.19, fmt(a), va="center", fontsize=5.0, color=C_MALE)
        xb = b + off
        need = xa + wdata(fmt(a), 5.0) + xmax * 0.010
        if xb < need:  # 与男性标签水平距离不足时右移，避免压字
            xb = need
        ax.text(xb, y + 0.19, fmt(b), va="center", fontsize=5.0, color="#B8860B")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_xlim(0, xlim1)
    ax.set_xlabel("National burden (attributable deaths)", fontsize=7.5, labelpad=1.5)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt(v)))
    ax.tick_params(axis="x", labelsize=6.8, length=2.2, width=0.5, pad=1.2)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.grid(axis="x", color="#E9ECEE", linewidth=0.4)
    ax.set_axisbelow(True)
    panel_header(fig, rect, "(i) Top 20 countries", "SSP2-4.5 | 2050")
    ax.legend(loc="lower right", fontsize=6.5, frameon=False, handlelength=1.1,
              labelspacing=0.35, borderaxespad=0.2)


# ====================================================================
def build_figure(cid: int, code: str, full_name: str, countries) -> dict:
    data = disease_data(cid)
    grids = build_grids(cid, countries, data)
    grid, pop, zones, keys = grid_income(cid, countries, grids["shape"], grids["transform"])

    ov = grids["overall"]
    vmax_shared = float(np.nanpercentile(ov[np.isfinite(ov)], 99.5)) if np.isfinite(ov).any() else 1.0
    norm_shared = PowerNorm(0.5, vmin=0, vmax=max(vmax_shared, 1e-9))  # 拉伸低值区，提高对比度

    fig = plt.figure(figsize=(183 / 25.4, 252 / 25.4), constrained_layout=False)
    fig.canvas.get_renderer()  # 初始化渲染器，供文本测量
    top_charts(fig, data, code)
    overall_map(fig, [0.025, 0.545, 0.615, 0.207], countries, grids, norm_shared)
    income_violin(fig, [0.730, 0.535, 0.240, 0.217], grid, pop, zones, keys)
    keep_d = bool(data["keep_sex_panel"])
    sex_maps(fig, countries, grids, keep_d, norm_shared)
    change_sensitivity_maps(fig, countries, grids)
    top20_bars(fig, [0.730, 0.045, 0.240, 0.173], data)

    PANELS.mkdir(parents=True, exist_ok=True)
    stem = PANELS / f"FigureS15_composite_{code}_v3"
    fig.savefig(f"{stem}.pdf",
                metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    fig.savefig(f"{stem}.tif", dpi=TIFF_DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved", f"{stem.name}.tif", flush=True)
    return {"disease": full_name, "sex_gap_median_pct": round(data["sex_gap_median"], 1),
            "panel_d_kept": keep_d,
            "deaths_2050_attributable": float(data["traj_deaths"]["attributable_burden"].iloc[-1]),
            "dalys_2050_attributable": float(data["traj_dalys"]["attributable_burden"].iloc[-1]),
            "grid_deaths_2050_total": float(np.nansum(grids["overall"]))}


def main():
    QA.mkdir(parents=True, exist_ok=True)
    countries = load_countries()
    only = sys.argv[1].upper() if len(sys.argv) > 1 else None
    qa = {"generated_at": datetime.now(timezone.utc).isoformat(),
          "layout": "per 汇报文件/FigureS15.png, v2 no-overlap",
          "content": "Deaths and DALYs (no PAF); all maps 0.1 deg grids",
          "note": "maps = 0.1deg burden_density grids; sex maps = grid x national male share; "
                  "scenario gap = grid2050 x (ssp585-ssp126)/ssp245 by country"}
    for cid, code, full_name in DISEASES:
        if only and code != only:
            continue
        qa[code] = build_figure(cid, code, full_name, countries)
    (QA / "figureS15_composites_QA_v2.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
