#!/usr/bin/env python3
"""Figure 3 small panel: top-2 diseases x 2050 deaths/DALYs at 0.1 degree.

2x2 grid of gridded high-BMI-attributable burden density maps (SSP2-4.5):
  row 1: Ischemic heart disease    | Deaths | DALYs
  row 2: Hypertensive heart disease | Deaths | DALYs
Top-2 diseases by global attributable deaths in 2050. Source: production
per-cause 0.1-degree burden-density rasters (pop x PAF allocation), windowed
onto the 84N-60S display grid.

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
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).resolve().parent
HELPER_SCRIPT = HERE / "08_plot_figure1B_historical_template_test_v2.py"
PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
OUT = ROOT / "figure3_country_burden_v1"
PANELS = OUT / "panels"
QA = OUT / "qa"

D2050 = (PROJECT / "11_all_disease_extension" / "06_spatial_burden"
         / "burden_density_v1_20261007" / "data" / "2050")

DISEASES = [
    ("Ischemic heart disease", 493),
    ("Hypertensive heart disease", 498),
]
MEASURES = [("Deaths", "deaths"), ("DALYs", "dalys")]

PALETTE = ("#FFFFFF", "#B7C7E6", "#6A85C2", "#3B4992", "#1F2A63")  # Science/AAAS blues
BLACK = "#17191A"
TIFF_DPI = 1800

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
    spec = importlib.util.spec_from_file_location("fig3h_map_helper", HELPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.historical


historical = import_helper()


def read_raster(path: Path) -> np.ndarray:
    """Window a 0.1-degree raster onto the 84N-60S display grid (1440 x 3600)."""
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
    cmap = LinearSegmentedColormap.from_list("fig3h_density", PALETTE, N=256)
    cmap.set_bad("white")
    return cmap


def draw_map(ax, values, norm):
    cmap = map_cmap()
    left, right, bottom, top = -180, 180, -60, 84
    show = np.clip(values, norm.vmin, norm.vmax)
    show = np.where(values > 0, show, np.nan)
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
    return cmap


def main():
    for sub in (PANELS, QA):
        sub.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(140 / 25.4, 130 / 25.4), constrained_layout=False)
    stats = {}
    col_x = [0.015, 0.515]
    map_w, map_h = 0.465, 0.20
    row_y = [0.645, 0.305]          # bottom of each map row (row 1 upper)
    row_label_y = [0.885, 0.545]
    for r, (disease, cid) in enumerate(DISEASES):
        fig.text(0.015, row_label_y[r], disease,
                 ha="left", va="top", fontsize=10, fontweight="bold", color=BLACK)
        for c, (measure, mkey) in enumerate(MEASURES):
            values = read_raster(
                D2050 / f"burden_density_c{cid}_{mkey}_ssp245_2050_0p1deg_v1.tif")
            valid = values[np.isfinite(values)]
            positive = valid[valid > 0]
            vmax = float(np.quantile(valid, 0.999))
            vmin = float(np.quantile(positive, 0.02)) if positive.size else 1e-3
            vmin = max(min(vmin, vmax * 1e-3), 1e-4)
            norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))

            ax = fig.add_axes([col_x[c], row_y[r], map_w, map_h])
            cmap = draw_map(ax, values, norm)
            if r == 0:
                fig.text(col_x[c] + 0.01, 0.930, f"{measure}, 2050",
                         ha="left", va="top", fontsize=10, fontweight="bold", color=BLACK)

            cax = fig.add_axes([col_x[c] + 0.05, row_y[r] - 0.047, map_w - 0.10, 0.013])
            e0 = int(np.floor(np.log10(norm.vmin)))
            e1 = int(np.ceil(np.log10(norm.vmax)))
            ticks = [10.0 ** e for e in range(e0, e1 + 1)]
            ticks = [t for t in ticks if norm.vmin <= t <= norm.vmax] or [norm.vmin, norm.vmax]
            cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                           ticks=ticks, orientation="horizontal")
            cb.set_ticklabels([f"{t:g}" for t in ticks])
            cb.ax.tick_params(labelsize=8, length=2, width=0.5, pad=1.5)
            cb.outline.set_visible(False)
            cb.set_label(f"{measure} per 0.1\u00b0 cell (log scale)",
                         labelpad=2, fontsize=8.5)
            stats[f"{cid}_{mkey}"] = {"vmax_q999": vmax, "vmin": vmin,
                                      "max": float(np.nanmax(values))}

    fig.text(0.985, 0.975, "SSP2-4.5", ha="right", va="top",
             fontsize=9, color="#596064", fontweight="bold")
    fig.text(0.015, 0.975, "Top 2 diseases: attributable burden density, 2050",
             ha="left", va="top", fontsize=11, fontweight="bold", color=BLACK)

    stem = PANELS / "Figure3_H_top2_diseases_deaths_dalys_2050_0p1deg_maps_v1"
    fig.savefig(f"{stem}.pdf",
                metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    fig.savefig(f"{stem}.tif", dpi=TIFF_DPI, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(QA / "Figure3_H_top2_maps_preview_v1.png", dpi=300)
    plt.close(fig)
    (QA / "figure3H_top2_maps_QA_v1.json").write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
                    "scenario": "SSP2-4.5", "resolution": "0.1 degree",
                    "selection": "top-2 diseases by attributable deaths 2050",
                    **stats}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", f"{stem}.tif")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
