#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4 DALYs 地图（与面板 B / D 风格一致）：
  B-DALYs：23 病因合计 DALYs 年增量 0.1° 速度分布图（分类着色）
  D-DALYs：Top4 疾病 DALYs 年增量 2x2 小地图（illustrative）
地图风格 = Figure-3 同款（等经纬网格 + 海岸线 + 白海洋），6 档发散分类色带。
输出：1800dpi LZW TIF + PDF，独立文件。
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
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
OUT = ROOT / "figure4_velocity_v1"
PANELS = OUT / "panels"
QA = OUT / "qa"

E = PROJECT / "11_all_disease_extension"
D15 = E / "06_spatial_burden" / "burden_density_2015_v1" / "data"
D50 = E / "06_spatial_burden" / "burden_density_v1_20261007" / "data" / "2050"
HELPER_SCRIPT = (ROOT / "scripts" / "08_plot_figure1B_historical_template_test_v2.py")

# Top4 按 2050 DALYs 排序：T2DM > IHD > HHD > Stroke
TOP4 = [(976, "Diabetes mellitus type 2"), (493, "Ischemic heart disease"),
        (498, "Hypertensive heart disease"), (494, "Stroke")]

TIFF_DPI = 1800
CLASS_COLORS = ["#2166AC", "#67A9CF", "#D1E5F0", "#FDE3D2", "#EF8A62", "#B2182B"]
B_EDGES = [-1e9, 0.0, 1.0, 5.0, 20.0, 100.0, 1e9]          # DALYs/格/年（合计）
B_LABELS = ["Declining", "0-1", "1-5", "5-20", "20-100", "\u2265100"]
D_EDGES = [-1e9, 0.0, 0.1, 0.5, 2.0, 10.0, 1e9]            # 分病因
D_LABELS = ["Declining", "0-0.1", "0.1-0.5", "0.5-2", "2-10", "\u226510"]
B_DISP = [-5.0, 0.0, 1.0, 5.0, 20.0, 100.0, 400.0]
D_DISP = [-0.05, 0.0, 0.1, 0.5, 2.0, 10.0, 40.0]
BLACK = "#17191A"
GREY = "#747C80"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def import_helper():
    spec = importlib.util.spec_from_file_location("fig4y_map_helper", HELPER_SCRIPT)
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
    a, b = read_windowed(c15), read_windowed(c50)
    ok = np.isfinite(a) & np.isfinite(b) & ~((a <= 0) & (b <= 0))
    return np.where(ok, (b - a) / 35.0, np.nan)


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


def draw_class_colorbar(fig, rect, disp_edges, labels, unit_label):
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


def panel_b_dalys(field: np.ndarray):
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(B_DISP, len(CLASS_COLORS))
    fig = plt.figure(figsize=(175 / 25.4, 108 / 25.4))
    ax = fig.add_axes([0.01, 0.175, 0.98, 0.735])
    draw_map_fig3style(ax, field, cmap, norm)
    ax.set_title("Spatial velocity of high-BMI attributable DALYs, 2015\u20132050 "
                 "(SSP2-4.5)", loc="left", fontsize=10, fontweight="bold", pad=4,
                 color=BLACK)
    draw_class_colorbar(fig, [0.14, 0.085, 0.72, 0.018], B_DISP, B_LABELS,
                        "Change in attributable DALYs per 0.1\u00b0 cell (DALYs per year)")
    save_panel(fig, "Figure4_B_dalys_spatial_velocity_map_v1")


def panel_d_dalys(fields, names):
    fig = plt.figure(figsize=(165 / 25.4, 118 / 25.4))
    fig.text(0.03, 0.965, "Velocity of top 4 diseases (illustrative)",
             ha="left", va="top", fontsize=10, fontweight="bold", color=BLACK)
    fig.text(0.03, 0.925,
             "selected high-burden diseases; change in attributable DALYs "
             "per 0.1\u00b0 cell (DALYs per year)",
             ha="left", va="top", fontsize=8, color=GREY)
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(D_DISP, len(CLASS_COLORS))
    xs = [0.035, 0.525]
    ys = [0.615, 0.245]
    for i, (name, field) in enumerate(zip(names, fields)):
        ax = fig.add_axes([xs[i % 2], ys[i // 2], 0.44, 0.275])
        draw_map_fig3style(ax, field, cmap, norm)
        ax.set_title(name, loc="left", fontsize=8.5, fontweight="bold", pad=3,
                     color=BLACK)
    draw_class_colorbar(fig, [0.16, 0.15, 0.68, 0.016], D_DISP, D_LABELS,
                        "Change in attributable DALYs per 0.1\u00b0 cell (DALYs per year)")
    save_panel(fig, "Figure4_D_dalys_top4_disease_velocity_maps_v1")


def main():
    QA.mkdir(parents=True, exist_ok=True)
    combined = increment_field(
        D15 / "burden_density_combined23_dalys_ssp245_2015_0p1deg_v1.tif",
        D50 / "burden_density_combined23_dalys_ssp245_2050_0p1deg_v1.tif")
    fields, names = [], []
    for cid, name in TOP4:
        fields.append(increment_field(
            D15 / f"burden_density_c{cid}_dalys_ssp245_2015_0p1deg_v1.tif",
            D50 / f"burden_density_c{cid}_dalys_ssp245_2050_0p1deg_v1.tif"))
        names.append(name)

    panel_b_dalys(combined)
    panel_d_dalys(fields, names)
    (QA / "figure4_dalys_maps_QA_v1.json").write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
                    "data": "REAL (pop x PAF grid allocation, DALYs)",
                    "unit": "DALYs per 0.1deg cell per year",
                    "classes_B": B_LABELS, "classes_D": D_LABELS,
                    "maps": "Figure-3 style (equirectangular, coastlines, white ocean)"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
