#!/usr/bin/env python3
"""Figure 3 panels B and C: 0.1-degree gridded high-BMI-attributable burden.

  Figure3_B: Change in attributable deaths, 2015-2050  +  High BMI attributable deaths, 2050
  Figure3_C: Change in attributable DALYs, 2015-2050  +  High BMI attributable DALYs, 2050

Each panel stacks two global 0.1-degree maps (each with its own title and
colourbar). Level maps: linear 0-99.9th percentile, PALETTE colormap (same
convention as the production density-map figures). Change maps: absolute
change per cell (2050 minus 2015) on the same linear convention; declining
cells clipped to the lowest colour (reported in QA).

Output: PDF + 1800-dpi LZW TIF in figure3_country_burden_v1/panels/.
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
from matplotlib.colors import (BoundaryNorm, LinearSegmentedColormap,
                             ListedColormap, LogNorm, Normalize)

HERE = Path(__file__).resolve().parent
HELPER_SCRIPT = HERE / "08_plot_figure1B_historical_template_test_v2.py"
PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
OUT = ROOT / "figure3_country_burden_v1"
PANELS = OUT / "panels"
QA = OUT / "qa"

D2015 = (PROJECT / "11_all_disease_extension" / "06_spatial_burden"
         / "burden_density_2015_v1" / "data")
D2050 = (PROJECT / "11_all_disease_extension" / "06_spatial_burden"
         / "burden_density_v1_20261007" / "data" / "2050")

PALETTE = ("#FFFFFF", "#c6dbef", "#6baed6", "#3182bd", "#08306b")  # single-hue Blues
CHANGE_EDGES = [-1e9, 0.0, 100.0, 250.0, 500.0, 1000.0, 1e9]
CHANGE_COLORS = ["#2166AC", "#FEE0D2", "#FC9272", "#EF3B2C", "#A50F15", "#67001F"]
CHANGE_LABELS = ["Declining", "0-100", "100-250", "250-500", "500-1000", "≥1000"]
BLACK = "#17191A"
TIFF_DPI = 1800
VQ = 0.999

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def import_helper():
    spec = importlib.util.spec_from_file_location("fig3bc_map_helper", HELPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.historical


historical = import_helper()


def read_raster(path: Path) -> np.ndarray:
    """Read a 0.1-degree raster and window it onto the display grid
    (84N-60S, 180W-180E = 1440 x 3600 cells). The 2050 production rasters
    cover the full 1800-row globe and must be sliced, not stretched."""
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True).astype(np.float64)
        transform = ds.transform
    arr = np.ma.filled(arr, np.nan)
    row_off = int(round((transform.f - 84.0) / 0.1))
    col_off = int(round((-180.0 - transform.c) / 0.1))
    out = np.full((1440, 3600), np.nan)
    r0, c0 = max(row_off, 0), max(col_off, 0)
    r1 = min(row_off + 1440, arr.shape[0])
    c1 = min(col_off + 3600, arr.shape[1])
    if r1 > r0 and c1 > c0:
        out[r0 - row_off:r1 - row_off, c0 - col_off:c1 - col_off] = arr[r0:r1, c0:c1]
    return out


def map_cmap():
    cmap = LinearSegmentedColormap.from_list("fig3bc_density", PALETTE, N=256)
    cmap.set_bad("white")
    return cmap


def draw_map(ax, values, cmap, norm):
    left, right, bottom, top = -180, 180, -60, 84
    show = np.clip(values, norm.vmin, norm.vmax)
    if isinstance(norm, LogNorm):
        show = np.where(values > 0, show, np.nan)   # log level map: non-positive -> white
    else:
        show = np.where(np.isfinite(values), show, np.nan)  # change map keeps negatives (blue)
    ax.imshow(show, extent=(left, right, bottom, top), origin="upper",
              cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
    countries = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")
    countries = countries[countries.geometry.notna()].copy()
    countries.geometry = countries.geometry.simplify(0.035, preserve_topology=True)
    segments = historical.geometry_segments(countries)
    historical.add_boundaries(ax, segments, None)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def draw_panel(measure: str, out_stem: str, change_title: str, level_title: str):
    change = read_raster(D2015 / f"change_{measure.lower()}_2015_2050_0p1deg_v1.tif")
    level = read_raster(D2050 / f"burden_density_combined23_{measure.lower()}_ssp245_2050_0p1deg_v1.tif")

    fig = plt.figure(figsize=(140 / 25.4, 165 / 25.4), constrained_layout=False)
    stats = {}
    # explicit layout: two maps stacked; each map has its own colourbar below
    slots = [dict(map_rect=[0.01, 0.615, 0.98, 0.335], cb_rect=[0.12, 0.573, 0.76, 0.014]),
             dict(map_rect=[0.01, 0.135, 0.98, 0.335], cb_rect=[0.12, 0.093, 0.76, 0.014])]
    for idx, (values, title, tag) in enumerate([
            (change, change_title, "change"),
            (level, level_title, "level")]):
        slot = slots[idx]
        ax = fig.add_axes(slot["map_rect"])
        valid = values[np.isfinite(values)]
        positive = valid[valid > 0]
        cmap = map_cmap()
        if tag == "change":
            # percent change with classified legend (actual percentage ranges)
            norm = BoundaryNorm(CHANGE_EDGES, len(CHANGE_COLORS))
            cmap = ListedColormap(CHANGE_COLORS)
            vmax = CHANGE_EDGES[-2]
        else:
            vmax = float(np.quantile(valid, VQ))
            vmin = float(np.quantile(positive, 0.02)) if positive.size else 1e-3
            vmin = max(min(vmin, vmax * 1e-3), 1e-4)
            norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
        draw_map(ax, values, cmap, norm)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=6, color=BLACK)
        fig.text(0.985, slot["map_rect"][1] + slot["map_rect"][3] + 0.030, "SSP2-4.5",
                 ha="right", va="top", fontsize=9, color="#596064", fontweight="bold")
        cax = fig.add_axes(slot["cb_rect"])
        if tag == "change":
            disp_edges = [-50.0, 0.0, 100.0, 250.0, 500.0, 1000.0, 3000.0]
            centers = [(disp_edges[i] + disp_edges[i + 1]) / 2.0 for i in range(6)]
            cb = mpl.colorbar.ColorbarBase(
                cax, cmap=cmap, norm=BoundaryNorm(disp_edges, len(CHANGE_COLORS)),
                boundaries=disp_edges, ticks=centers, orientation="horizontal")
            cb.set_ticklabels(CHANGE_LABELS)
            norm = Normalize(0, 1)  # stats placeholder
        else:
            e0 = int(np.floor(np.log10(norm.vmin)))
            e1 = int(np.ceil(np.log10(norm.vmax)))
            tick_vals = [10.0 ** e for e in range(e0, e1 + 1)]
            tick_vals = [t for t in tick_vals if norm.vmin <= t <= norm.vmax] or [norm.vmin, norm.vmax]
            cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                           ticks=tick_vals, orientation="horizontal")
            cb.set_ticklabels([f"{t:g}" for t in tick_vals])
        cb.ax.tick_params(labelsize=9, length=2, width=0.5, pad=1.5)
        cb.outline.set_visible(False)
        measure_disp = "deaths" if measure == "deaths" else "DALYs"
        label_txt = "Change (%)" if tag == "change" else "%s per 0.1° cell (log scale)" % measure_disp
        cb.set_label(label_txt, labelpad=2, fontsize=10)
        stats[tag] = {"vmax_q999": vmax,
                      "min": float(np.nanmin(values)), "median": float(np.nanmedian(values)),
                      "max": float(np.nanmax(values)),
                      "n_declining": int(np.nansum(values < 0)) if tag == "change" else 0}

    stem = PANELS / out_stem
    fig.savefig(f"{stem}.pdf",
                metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    fig.savefig(f"{stem}.tif", dpi=TIFF_DPI, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(QA / f"{out_stem}_preview.png", dpi=300)
    plt.close(fig)
    print("saved", f"{stem}.tif", flush=True)
    return stats


def main():
    for sub in (PANELS, QA):
        sub.mkdir(parents=True, exist_ok=True)
    qa = {"generated_at": datetime.now(timezone.utc).isoformat(),
          "resolution": "0.1 degree", "scenario": "SSP2-4.5",
          "change_metric": "absolute change per cell (2050 minus 2015); declines clipped at 0 for display",
          "B": draw_panel("deaths", "Figure3_B_grid_change_and_attributable_deaths_2015_2050_v1",
                          "Change in attributable deaths, 2015\u20132050",
                          "High BMI attributable deaths, 2050"),
          "C": draw_panel("dalys", "Figure3_C_grid_change_and_attributable_dalys_2015_2050_v1",
                          "Change in attributable DALYs, 2015\u20132050",
                          "High BMI attributable DALYs, 2050")}
    (QA / "figure3BC_grid_QA_v1.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print("QA written", flush=True)


if __name__ == "__main__":
    main()
