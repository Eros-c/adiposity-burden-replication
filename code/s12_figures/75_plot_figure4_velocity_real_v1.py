#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 —— 全球高 BMI 归因疾病负担的"变化速度（velocity）"主图（真实数据版 v3）
========================================================================

与 Figure 3 分工：Figure 3 = 负担有多大（绝对量/构成/累计变化率）；
                Figure 4 = 负担变得有多快（速度及其空间格局）。

面板（独立文件，1800dpi LZW TIF + PDF；图内不含字母，字母仅在文件名）：
  A  全球速度时间趋势（Deaths + DALYs；历史/投影分区、±1SD 带、原始同比点）
  B  速度空间分布核心地图：年增量（deaths per 0.1° cell per year），
     分类着色（6 档，实际值区间图例），与 Figure 3 地图同风格
  C  区域速度对比：Deaths + DALYs 分组条形；区域名标于条形上方（无 Y 轴）
  D  Top4 疾病速度小地图 2x2（illustrative），与 B 同色系分类着色

真实数据：
  归因序列 = 研究 PAF x GBD/Foresight 总量（2015-2021 实测 + 2022-2024 Foresight
             + 2025-2050 未来产品）；栅格 = pop x PAF 分配的 0.1° 负担密度。
  速度口径 = 年增量 (2050-2015)/35；仅掩膜真零/无效格。
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

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
OUT = ROOT / "figure4_velocity_v1"
PANELS = OUT / "panels"
QA = OUT / "qa"

E = PROJECT / "11_all_disease_extension"
BENCH = E / "04_burden" / "burden_benchmark_v1_20261005" / "data"
BENCH_FILE = BENCH / "benchmark_historical_country_year_detail_v1.parquet"
FORESIGHT = (E / "01_harmonize" / "harmonized_gbd_v1_20261005" / "data"
             / "gbd_foresight_reference_harmonized_v1.parquet")
PAF_HIST = (E / "03_paf" / "paf_all_causes_v1_20261005" / "data"
            / "paf_historical_2015_2024_all_causes_v1.parquet")
FUTURE_COUNTRY = BENCH / "attributable_future_country_overall_four_ssp_v1.parquet"
D15 = E / "06_spatial_burden" / "burden_density_2015_v1" / "data"
D50 = E / "06_spatial_burden" / "burden_density_v1_20261007" / "data" / "2050"
STACK_CSV = (ROOT / "figure3_country_burden_v1" / "tables"
             / "fig3a_disease_stacked_attributable_burden_2015_2050_v1.csv")
HELPER_SCRIPT = (ROOT / "scripts" / "08_plot_figure1B_historical_template_test_v2.py")

HEADLINE = {417, 429, 435, 441, 453, 456, 465, 471, 480, 485, 486, 487,
            493, 494, 498, 500, 502, 515, 534, 543, 591, 976, 998}
GBD_AGE_TO_ANCHOR = {"85-89": "85+", "90-94": "85+", "95+": "85+"}
TOP4 = [(493, "Ischemic heart disease"), (976, "Diabetes mellitus type 2"),
        (498, "Hypertensive heart disease"), (494, "Stroke")]
ISO_FIX = {"PSX": "PSE", "SDS": "SSD"}

TIFF_DPI = 1800
# 分类色带（发散，蓝=下降/低速，红=高速；类标签为实际值区间）
CLASS_COLORS = ["#2166AC", "#67A9CF", "#D1E5F0", "#FDE3D2", "#EF8A62", "#B2182B"]
B_EDGES = [-1e9, 0.0, 0.05, 0.2, 1.0, 5.0, 1e9]          # 例/格/年（合并23病）
B_LABELS = ["Declining", "0-0.05", "0.05-0.2", "0.2-1", "1-5", "\u22655"]
D_EDGES = [-1e9, 0.0, 0.01, 0.05, 0.2, 1.0, 1e9]         # 分病因（约 4-6 倍更小）
D_LABELS = ["Declining", "0-0.01", "0.01-0.05", "0.05-0.2", "0.2-1", "\u22651"]

BLACK = "#17191A"
GREY = "#747C80"
C_DEATHS, C_DALYS = "#1F3864", "#B2182B"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.linewidth": 0.7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def import_helper():
    spec = importlib.util.spec_from_file_location("fig4_map_helper", HELPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.historical


historical = import_helper()


def save_panel(fig, stem: str):
    PANELS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PANELS / f"{stem}.pdf",
                metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    fig.savefig(PANELS / f"{stem}.tif", dpi=TIFF_DPI,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved", f"{stem}.tif", flush=True)


# ====================================================================
# 真实数据
# ====================================================================
def global_series(measure: str) -> pd.Series:
    bench = pd.read_parquet(BENCH_FILE)
    bench = bench[(bench["analysis_tier"] == "headline") & (bench["measure"] == measure)]
    s = bench.groupby("year")["study_attributable"].sum()
    paf = pd.read_parquet(PAF_HIST)
    paf = paf[paf["year"].isin([2022, 2023, 2024])][
        ["cause_id", "iso3", "year", "sex", "anchor_age_group", "paf"]]
    fores = pd.read_parquet(FORESIGHT,
                            columns=["cause_id", "iso3", "year", "age_group", "sex",
                                     "measure", "metric", "value"])
    fores = fores[(fores["cause_id"].isin(HEADLINE)) & (fores["measure"] == measure)
                  & (fores["metric"] == "Number") & (fores["year"] <= 2024)]
    fores["anchor_age_group"] = fores["age_group"].replace(GBD_AGE_TO_ANCHOR)
    m = fores.merge(paf, on=["cause_id", "iso3", "year", "sex", "anchor_age_group"],
                    how="left", validate="many_to_one")
    m["paf"] = m["paf"].fillna(0.0)
    s = pd.concat([s, m.assign(a=m["paf"] * m["value"]).groupby("year")["a"].sum()])
    fut = pd.read_csv(STACK_CSV)
    fut = fut[(fut["measure"] == measure) & (fut["year"] >= 2025)] \
        .groupby("year")["attributable_burden"].sum()
    return pd.concat([s, fut]).sort_index()


def velocity_stats(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回 (5 年滚动均值, 窗口内 SD, 原始同比)：均值为主曲线，SD 为真实波动带。"""
    yoy = (series.pct_change() * 100.0).dropna()
    roll = yoy.rolling(5, center=True, min_periods=2)
    return roll.mean(), roll.std().fillna(0.0), yoy


def region_velocity(measure: str) -> dict:
    b = pd.read_parquet(BENCH_FILE)
    b = b[(b["analysis_tier"] == "headline") & (b["year"] == 2015) & (b["measure"] == measure)]
    v15 = b.groupby("iso3")["study_attributable"].sum()
    fut = pd.read_parquet(FUTURE_COUNTRY,
                          columns=["iso3", "year", "measure", "attributable_burden",
                                   "analysis_tier", "scenario_code"])
    fut = fut[(fut["analysis_tier"] == "headline") & (fut["scenario_code"] == "ssp245")
              & (fut["year"] == 2050) & (fut["measure"] == measure)]
    v50 = fut.groupby("iso3")["attributable_burden"].sum()
    ne = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")
    ne = ne[ne.geometry.notna()].copy()
    ne["iso3"] = ne["ISO_A3"].where(ne["ISO_A3"].ne("-99"), ne["ADM0_A3"]) \
        .astype(str).str.upper().replace(ISO_FIX)
    cont = ne[["iso3", "CONTINENT"]].rename(columns={"CONTINENT": "continent"}).drop_duplicates("iso3")
    cont = cont[~cont["continent"].isin(["Seven seas (open ocean)", "Antarctica"])]
    df = pd.concat([v15.rename("v15"), v50.rename("v50")], axis=1).dropna().reset_index()
    df = df.merge(cont, on="iso3", how="left").dropna(subset=["continent"])
    df = df[df["v15"] > 0]
    reg = df.groupby("continent")[["v15", "v50"]].sum()
    return ((reg["v50"] / reg["v15"] - 1.0) / 35.0 * 100.0).to_dict()


def read_windowed(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = np.ma.filled(ds.read(1, masked=True).astype(float), np.nan)
        t = ds.transform
    ro = int(round((t.f - 84.0) / 0.1)); co = int(round((-180.0 - t.c) / 0.1))
    out = np.full((1440, 3600), np.nan)
    r0, c0 = max(ro, 0), max(co, 0)
    r1, c1 = min(ro + 1440, arr.shape[0]), min(co + 3600, arr.shape[1])
    out[r0 - ro:r1 - ro, c0 - co:c1 - co] = arr[r0:r1, c0:c1]
    return out


def increment_field(c15: Path, c50: Path) -> np.ndarray:
    """年增量（例/格/年）；仅掩膜无效与真零格（增量不需基线阈值）。"""
    a, b = read_windowed(c15), read_windowed(c50)
    ok = np.isfinite(a) & np.isfinite(b) & ~((a <= 0) & (b <= 0))
    return np.where(ok, (b - a) / 35.0, np.nan)


# ====================================================================
# 地图（Figure-3 同款：等经纬网格 + 海岸线 + 白海洋 + 分类色带）
# ====================================================================
def draw_map_fig3style(ax, values, cmap, norm):
    show = np.where(np.isfinite(values), values, np.nan)
    im = ax.imshow(show, extent=(-180, 180, -60, 84), origin="upper",
                   cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
    countries = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")
    countries = countries[countries.geometry.notna()].copy()
    countries.geometry = countries.geometry.simplify(0.035, preserve_topology=True)
    segments = historical.geometry_segments(countries)
    historical.add_boundaries(ax, segments, None)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 84)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return im


def class_cmap_norm(edges):
    cmap = ListedColormap(CLASS_COLORS)
    return cmap, BoundaryNorm(edges, len(CLASS_COLORS))


def draw_class_colorbar(fig, rect, disp_edges, labels, unit_label):
    """分类色条：有限显示边界 + 类中心刻度 + 实际值区间标签（6 类 6 刻度）。"""
    centers = [(disp_edges[i] + disp_edges[i + 1]) / 2.0 for i in range(len(labels))]
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(disp_edges, len(CLASS_COLORS))
    cax = fig.add_axes(rect)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                   boundaries=disp_edges, ticks=centers,
                                   orientation="horizontal")
    cb.set_ticklabels(labels)
    cb.ax.tick_params(labelsize=7.5, length=2, width=0.5, pad=1.5)
    cb.outline.set_visible(False)
    cb.set_label(unit_label, fontsize=8.5, labelpad=2)


def panel_a(vd, sd, yd, vy, sy, yy):
    """A：全球速度趋势（Deaths/DALYs）——历史/投影分区 + ±1SD 带 + 原始点 + 差值填充。"""
    fig = plt.figure(figsize=(110 / 25.4, 76 / 25.4))
    ax = fig.add_axes([0.135, 0.175, 0.755, 0.71])
    # 历史 / 投影 分区底色与标签
    ax.axvspan(2015, 2024.5, color="#F2F4F5", zorder=0)
    ax.axvline(2024.5, color="#C9D2D8", linewidth=0.8, zorder=1)
    ax.text(2019.7, ax.get_ylim()[1] * 0.97 if False else 0.975, "Historical",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8, color=GREY)
    ax.text(2037.2, 0.975, "Projection", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color=GREY)
    # 两条曲线之间的差值带
    # 原始同比点（浅色）+ ±1SD 波动带 + 主曲线
    for m, s, y, color, name in ((vd, sd, yd, C_DEATHS, "Deaths"),
                                 (vy, sy, yy, C_DALYS, "DALYs")):
        ax.plot(y.index, y.values, ".", ms=1.6, color=color, alpha=0.30, zorder=3)
        ax.fill_between(m.index, (m - s).values, (m + s).values, color=color,
                        alpha=0.16, linewidth=0, zorder=3)
        ax.plot(m.index, m.values, color=color, linewidth=1.9, zorder=5)
        ax.annotate(name, (m.index[-1], m.values[-1]), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, color=color, fontweight="bold")
        for yr in (2025, 2035, 2050):
            v = float(m.loc[yr])
            ax.plot([yr], [v], "o", ms=3.2, color=color, zorder=6)
            ax.annotate(f"{v:.1f}", (yr, v), xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=color)
    for yr in (2025, 2035, 2050):
        ax.axvline(yr, color="#C9D2D8", linewidth=0.6, zorder=1)
    ax.set_xlim(2015, 2053)
    band_top = float(max((vd + sd).max(), (vy + sy).max()))
    ax.set_ylim(0, band_top * 1.14)
    ax.set_xticks([2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050])
    ax.set_xlabel("Year", fontsize=8.5, labelpad=2)
    ax.set_ylabel("Annual change in attributable burden (% per year)",
                  fontsize=8.5, labelpad=2)
    ax.tick_params(labelsize=7.5, length=2.5, width=0.55, pad=1.5)
    ax.grid(axis="y", color="#E9ECEE", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_title("Global velocity of high-BMI attributable burden",
                 loc="left", fontsize=10, fontweight="bold", pad=5, color=BLACK)
    save_panel(fig, "Figure4_A_global_velocity_trend_v1")


def panel_b(field: np.ndarray):
    """B：速度空间分布，分类着色（6 档实际值区间），Figure-3 同款地图。"""
    cmap, norm = class_cmap_norm(B_EDGES)
    fig = plt.figure(figsize=(175 / 25.4, 108 / 25.4))
    ax = fig.add_axes([0.01, 0.175, 0.98, 0.735])
    draw_map_fig3style(ax, field, cmap, norm)
    ax.set_title("Spatial velocity of high-BMI attributable deaths, 2015\u20132050 "
                 "(SSP2-4.5)", loc="left", fontsize=10, fontweight="bold", pad=4,
                 color=BLACK)
    draw_class_colorbar(fig, [0.14, 0.085, 0.72, 0.018], [-2.0, 0.0, 0.05, 0.2, 1.0, 5.0, 20.0], B_LABELS,
                        "Change in attributable deaths per 0.1\u00b0 cell (deaths per year)")
    save_panel(fig, "Figure4_B_spatial_velocity_map_v1")


def panel_c(reg_d: dict, reg_y: dict):
    """C：区域速度 Deaths/DALYs 分组条形；区域名在条形上方（无 Y 轴）。"""
    names = sorted(reg_d, key=lambda k: reg_d[k])
    fig = plt.figure(figsize=(115 / 25.4, 88 / 25.4))
    ax = fig.add_axes([0.055, 0.155, 0.915, 0.745])
    h = 0.30
    xmax = max(max(reg_d.values()), max(reg_y.values()))
    for i, name in enumerate(names):
        ax.barh(i - h / 2, reg_d[name], height=h, color=C_DEATHS, edgecolor="none", zorder=3)
        ax.barh(i + h / 2, reg_y[name], height=h, color=C_DALYS, edgecolor="none", zorder=3)
        ax.text(0, i + 0.34, name, ha="left", va="bottom", fontsize=8.5,
                fontweight="bold", color=BLACK)
        ax.text(reg_d[name] + xmax * 0.015, i - h / 2, f"{reg_d[name]:.1f}",
                va="center", fontsize=7.5, color=C_DEATHS)
        ax.text(reg_y[name] + xmax * 0.015, i + h / 2, f"{reg_y[name]:.1f}",
                va="center", fontsize=7.5, color=C_DALYS)
    ax.axvline(0, color="#3F464A", linewidth=0.8, zorder=4)
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(names) + 0.2)
    ax.set_xlim(0, xmax * 1.16)
    ax.set_xlabel("Mean annual change (% per year)", fontsize=8.5, labelpad=2)
    ax.tick_params(axis="x", labelsize=7.5, length=2.5, width=0.55, pad=1.5)
    ax.grid(axis="x", color="#E9ECEE", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("Mean velocity by region", loc="left", fontsize=10,
                 fontweight="bold", pad=5, color=BLACK)
    ax.legend(handles=[Patch(facecolor=C_DEATHS, edgecolor="none", label="Deaths"),
                       Patch(facecolor=C_DALYS, edgecolor="none", label="DALYs")],
              loc="lower right", fontsize=8, handlelength=1.2, labelspacing=0.4)
    save_panel(fig, "Figure4_C_region_velocity_v1")


def panel_d(fields: list, names: list):
    """D：Top4 疾病速度小地图 2x2（Figure-3 同款地图，分类着色）。"""
    fig = plt.figure(figsize=(165 / 25.4, 118 / 25.4))
    fig.text(0.03, 0.965, "Velocity of top 4 diseases (illustrative)",
             ha="left", va="top", fontsize=10, fontweight="bold", color=BLACK)
    fig.text(0.03, 0.925,
             "selected high-burden diseases; change in attributable deaths "
             "per 0.1\u00b0 cell (deaths per year)",
             ha="left", va="top", fontsize=8, color=GREY)
    cmap, norm = class_cmap_norm(D_EDGES)
    xs = [0.035, 0.525]
    ys = [0.615, 0.245]
    for i, (name, field) in enumerate(zip(names, fields)):
        ax = fig.add_axes([xs[i % 2], ys[i // 2], 0.44, 0.275])
        draw_map_fig3style(ax, field, cmap, norm)
        ax.set_title(name, loc="left", fontsize=8.5, fontweight="bold", pad=3,
                     color=BLACK)
    draw_class_colorbar(fig, [0.16, 0.15, 0.68, 0.016], [-0.5, 0.0, 0.01, 0.05, 0.2, 1.0, 5.0], D_LABELS,
                        "Change in attributable deaths per 0.1\u00b0 cell (deaths per year)")
    save_panel(fig, "Figure4_D_top4_disease_velocity_maps_v1")


# ====================================================================
def main():
    QA.mkdir(parents=True, exist_ok=True)
    sd_series = {}
    stats = {}
    for meas in ("Deaths", "DALYs"):
        m, sd, y = velocity_stats(global_series(meas))
        sd_series[meas] = (m, sd, y)
    combined = increment_field(
        D15 / "burden_density_combined23_deaths_ssp245_2015_0p1deg_v1.tif",
        D50 / "burden_density_combined23_deaths_ssp245_2050_0p1deg_v1.tif")
    fields, names = [], []
    for cid, name in TOP4:
        fields.append(increment_field(
            D15 / f"burden_density_c{cid}_deaths_ssp245_2015_0p1deg_v1.tif",
            D50 / f"burden_density_c{cid}_deaths_ssp245_2050_0p1deg_v1.tif"))
        names.append(name)

    reg_d = region_velocity("Deaths")
    reg_y = region_velocity("DALYs")

    qa = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": "REAL (study PAF x GBD/Foresight totals; pop x PAF grid allocation)",
        "units": {"A": "% per year (5-yr rolling mean, band = window SD)",
                  "B_D": "deaths per 0.1deg cell per year (6 classes)",
                  "C": "% per year"},
        "maps": "Figure-3 style (equirectangular, coastlines, white ocean)",
        "region_velocity_deaths": {k: round(v, 2) for k, v in reg_d.items()},
        "region_velocity_dalys": {k: round(v, 2) for k, v in reg_y.items()},
    }
    panel_a(sd_series["Deaths"][0], sd_series["Deaths"][1], sd_series["Deaths"][2],
            sd_series["DALYs"][0], sd_series["DALYs"][1], sd_series["DALYs"][2])
    panel_b(combined)
    panel_c(reg_d, reg_y)
    panel_d(fields, names)

    (QA / "figure4_velocity_real_QA_v3.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
