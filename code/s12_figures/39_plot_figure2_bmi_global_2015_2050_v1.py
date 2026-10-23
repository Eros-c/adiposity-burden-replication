"""Figure 2 (redraw v1): Global BMI trajectories and spatial patterns, 2015-2050."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import csv
import os
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 600,
})

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from audit_panel_alignment import require_matplotlib_panel_alignment

HERE = Path(__file__).resolve().parent
HELPER_SCRIPT = HERE / "08_plot_figure1B_historical_template_test_v2.py"
ROOT = Path(_cfg.RESULTS / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909")
TREND_CSV = ROOT / "test_figure1_AB_v2" / "TEST_Fig1A_trend_source_four_SSP_v2.csv"
OUT = ROOT / "figure2_bmi_global_2015_2050_v1"

WIDTH_MM = 180.0
HEIGHT_MM = 200.0
DPI_TIF = 1600
raster_export_dpi = 1600
DPI_PREVIEW = 400
YEARS_MAP = (2015, 2025, 2035, 2050)
STEM = "Figure2_BMI_global_trajectories_and_spatial_2015_2050_v1"

MIDPOINTS = np.asarray((17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5), dtype=np.float32)
MAP_COLORS = ("#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c")

HISTORICAL_COLOR = "#151718"
REFERENCE_COLOR = "#009E73"
STRIP_COLORS = {"ssp126": "#0072B2", "ssp370": "#E69F00", "ssp585": "#D55E00"}
STRIP_LABELS = {"ssp126": "SSP1-2.6", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
SCENARIO_NAMES = {"Historical": "Historical", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
                  "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}

BLACK = "#151718"
GREY = "#596064"

@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    vmin: float
    vmax: float
    tmin: float
    tmax: float

METRICS = (
    Metric("mean_bmi", "Mean BMI (kg/m\u00b2)", "kg/m\u00b2", 18.0, 32.0, 24.0, 28.5),
    Metric("bmi_ge25", "BMI \u226525 (%)", "%", 0.0, 90.0, 40.0, 65.0),
    Metric("bmi_ge30", "BMI \u226530 (%)", "%", 0.0, 60.0, 12.0, 32.0),
)

def import_helper():
    spec = importlib.util.spec_from_file_location("figure2_redraw_helper", HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper: {HELPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

helper = import_helper()
historical = helper.historical

def read_categories(year: int) -> np.ma.MaskedArray:
    layers = []
    for category_index in range(7):
        if year == 2015:
            layer = helper.read_historical_percent(year, category_index)
        else:
            layer = helper.read_future_percent(year, category_index)
        layers.append(layer.astype(np.float32))
    data = np.ma.stack(layers, axis=0)
    common_mask = np.ma.getmaskarray(data).any(axis=0)
    return np.ma.array(data, mask=np.broadcast_to(common_mask, data.shape))

def derive_metric(categories: np.ma.MaskedArray, metric: Metric) -> np.ma.MaskedArray:
    if metric.key == "mean_bmi":
        return np.ma.sum((categories / 100.0) * MIDPOINTS[:, None, None], axis=0)
    if metric.key == "bmi_ge25":
        return np.ma.sum(categories[3:7], axis=0)
    if metric.key == "bmi_ge30":
        return np.ma.sum(categories[4:7], axis=0)
    raise ValueError(metric.key)

def metric_cmap(name: str):
    cmap = LinearSegmentedColormap.from_list(name, MAP_COLORS, N=256)
    cmap.set_bad("white")
    return cmap

def build_boundaries():
    countries = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")
    countries = countries[countries.geometry.notna()].copy()
    countries.geometry = countries.geometry.simplify(0.035, preserve_topology=True)
    country_segments = historical.geometry_segments(countries)
    adm1 = historical.gpd.read_file(historical.ADM1_PATH, engine="pyogrio")
    adm1 = adm1[adm1.geometry.notna()].copy()
    adm1.geometry = adm1.geometry.simplify(0.035, preserve_topology=True)
    adm1_segments = historical.geometry_segments(adm1)
    return country_segments, adm1_segments

def draw_map(axis, data, metric: Metric, country_segments, adm1_segments) -> None:
    axis.imshow(
        data,
        extent=(-180, 180, -60, 84),
        origin="upper",
        cmap=metric_cmap(f"fig2r_{metric.key}"),
        norm=Normalize(vmin=metric.vmin, vmax=metric.vmax),
        interpolation="nearest",
        rasterized=True,
        zorder=1,
    )
    historical.add_boundaries(axis, country_segments, adm1_segments)
    axis.set_xlim(-180, 180)
    axis.set_ylim(-60, 84)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

def add_colorbar(axis, metric: Metric) -> None:
    colorbar = mpl.colorbar.ColorbarBase(
        axis,
        cmap=metric_cmap(f"fig2r_cb_{metric.key}"),
        norm=Normalize(vmin=metric.vmin, vmax=metric.vmax),
        orientation="horizontal",
    )
    ticks = np.linspace(metric.vmin, metric.vmax, 4)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:.0f}" for value in ticks])
    colorbar.ax.tick_params(labelsize=5.0, length=1.4, width=0.4, pad=0.6)
    colorbar.outline.set_visible(False)

def load_trend() -> pd.DataFrame:
    frame = pd.read_csv(TREND_CSV)
    return frame

def value_series(frame: pd.DataFrame, metric: Metric, scenario: str) -> pd.Series:
    subset = frame[frame["scenario"] == SCENARIO_NAMES[scenario]].sort_values("year")
    column = {"mean_bmi": "mean_bmi", "bmi_ge25": "overweight_obesity", "bmi_ge30": "obesity"}[metric.key]
    values = subset[column].astype(float)
    if metric.key != "mean_bmi":
        values = values * 100.0
    return pd.Series(values.values, index=subset["year"].values)

def plot() -> list[Path]:
    figure = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), constrained_layout=False)
    outer = figure.add_gridspec(
        2, 1, left=0.100, right=0.995, bottom=0.030, top=0.985,
        height_ratios=(42, 58), hspace=0.10,
    )
    trend = load_trend()

    traj = outer[0].subgridspec(
        6, 1, height_ratios=(2.9, 1.0, 2.9, 1.0, 2.9, 1.0), hspace=0.55,
    )
    letters = ("a", "b", "c")
    axes_main: list = []
    axes_strip: list = []
    for row, metric in enumerate(METRICS):
        main_axis = figure.add_subplot(traj[row * 2, 0])
        strip_axis = figure.add_subplot(traj[row * 2 + 1, 0], sharex=main_axis)
        axes_main.append(main_axis)
        axes_strip.append(strip_axis)

        historical_series = value_series(trend, metric, "Historical")
        reference_series = value_series(trend, metric, "ssp245")
        main_axis.plot(
            historical_series.index, historical_series.values,
            color=HISTORICAL_COLOR, linewidth=0.9, zorder=3,
            marker="o", markersize=1.8, markerfacecolor=HISTORICAL_COLOR,
            markeredgecolor=HISTORICAL_COLOR, label="Historical",
        )
        main_axis.plot(
            reference_series.index, reference_series.values,
            color=REFERENCE_COLOR, linewidth=1.0, zorder=3,
        )
        main_axis.axvline(2024.5, color="#8a9094", linewidth=0.5, linestyle=(0, (1.5, 1.5)), zorder=2)
        main_axis.set_ylim(metric.tmin, metric.tmax)
        main_axis.set_xlim(2015, 2050)
        main_axis.set_yticks(np.linspace(metric.tmin, metric.tmax, 3))
        main_axis.tick_params(axis="y", labelsize=5.2, length=2.0, width=0.5, pad=1.5)
        main_axis.tick_params(axis="x", labelbottom=False)
        for spine in ("top", "right"):
            main_axis.spines[spine].set_visible(False)
        main_axis.spines["left"].set_linewidth(0.5)
        main_axis.spines["bottom"].set_linewidth(0.5)
        ylabel = "Mean BMI (kg/m\u00b2)" if metric.key == "mean_bmi" else "Prevalence (%)"
        main_axis.set_ylabel(ylabel, fontsize=5.6, labelpad=2.0)
        main_axis.text(
            0.005, 0.955, f"({letters[row]})", transform=main_axis.transAxes,
            ha="left", va="top", fontsize=7.4, fontweight="bold", color=BLACK,
        )
        if row == 0:
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([], [], color=HISTORICAL_COLOR, linewidth=0.9,
                       marker="o", markersize=1.8, label="Historical"),
                Line2D([], [], color=REFERENCE_COLOR, linewidth=1.0, label="SSP2-4.5"),
                Line2D([], [], color=STRIP_COLORS["ssp126"], linewidth=0.7, label="SSP1-2.6"),
                Line2D([], [], color=STRIP_COLORS["ssp370"], linewidth=0.7, label="SSP3-7.0"),
                Line2D([], [], color=STRIP_COLORS["ssp585"], linewidth=0.7, label="SSP5-8.5"),
            ]
            main_axis.legend(
                handles=legend_handles, loc="lower right", bbox_to_anchor=(0.995, 0.05),
                fontsize=5.0, ncol=3, handlelength=1.3, handletextpad=0.5,
                columnspacing=0.8, borderpad=0.2, labelspacing=0.3,
            )

        reference_values = reference_series.reindex(range(2024, 2051)).interpolate()
        for scenario, color in STRIP_COLORS.items():
            series = value_series(trend, metric, scenario)
            difference = series.reindex(range(2024, 2051)).interpolate() - reference_values
            strip_axis.plot(
                difference.index, difference.values,
                color=color, linewidth=0.7, zorder=3,
                label=STRIP_LABELS[scenario],
            )
        strip_axis.axhline(0.0, color="#9aa0a4", linewidth=0.45, zorder=2)
        spread = max(abs(value_series(trend, metric, s).reindex(range(2024, 2051)).subtract(reference_values).abs().max()) for s in STRIP_COLORS)
        limit = max(np.ceil(spread * 1.35 * 20) / 20, 0.05)
        strip_axis.set_ylim(-limit, limit)
        strip_axis.set_yticks([-limit, 0.0, limit])
        strip_axis.tick_params(axis="y", labelsize=5.0, length=1.6, width=0.45, pad=1.2)
        strip_axis.tick_params(axis="x", labelbottom=(row == 2))
        for spine in ("top", "right"):
            strip_axis.spines[spine].set_visible(False)
        strip_axis.spines["left"].set_linewidth(0.45)
        strip_axis.spines["bottom"].set_linewidth(0.45)
        if row == 2:
            strip_axis.set_xticks([2015, 2025, 2035, 2050])
            strip_axis.set_xticklabels(["2015", "2025", "2035", "2050"], fontsize=5.0)
            strip_axis.set_xlabel("Year", fontsize=5.2, labelpad=1.5)

    maps = outer[1].subgridspec(3, 4, wspace=0.014, hspace=0.62)
    category_cache: dict[int, np.ma.MaskedArray] = {}
    map_letters = ("d", "e", "f")
    country_segments, adm1_segments = build_boundaries()
    first_map_axes: list = []
    for metric_index, metric in enumerate(METRICS):
        for year_index, year in enumerate(YEARS_MAP):
            if year not in category_cache:
                category_cache[year] = read_categories(year)
            axis = figure.add_subplot(maps[metric_index, year_index])
            draw_map(axis, derive_metric(category_cache[year], metric), metric, country_segments, adm1_segments)
            if metric_index == 0:
                axis.set_title(str(year), fontsize=6.4, fontweight="bold", pad=2.0, color=BLACK)
            if year_index == 0:
                first_map_axes.append(axis)
                axis.text(
                    0.0, 1.045, f"({map_letters[metric_index]})", transform=axis.transAxes,
                    ha="left", va="bottom", fontsize=7.4, fontweight="bold", color=BLACK,
                )

    require_matplotlib_panel_alignment(
        figure,
        json_out=str(OUT / f"{STEM}.alignment.json"),
        overlay_svg=str(OUT / f"{STEM}.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )

    figure.canvas.draw()
    for metric_index, metric in enumerate(METRICS):
        first_map = first_map_axes[metric_index]
        bbox = first_map.get_position()
        color_axis = figure.add_axes([bbox.x0, bbox.y0 - 0.022, bbox.width * 0.62, 0.006])
        add_colorbar(color_axis, metric)
        row_center_y = bbox.y0 + bbox.height * 0.5
        figure.text(
            0.092, row_center_y + 0.018, metric.label, ha="right", va="center",
            fontsize=6.6, fontweight="bold", color=BLACK,
        )
        figure.text(
            0.092, row_center_y - 0.014, "SSP2-4.5", ha="right", va="center",
            fontsize=5.0, color=GREY,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{STEM}.pdf"
    tif = OUT / f"{STEM}.tif"
    preview = OUT / f"{STEM}_preview.png"

    if os.environ.get("FIG2_DEBUG"):
        figure.canvas.draw()
        for index, axis in enumerate(figure.axes):
            label = axis.get_label() or f"axes_{index}"
            print("AXES_DEBUG", index, label, [round(v, 4) for v in axis.get_position().bounds])
        figure.savefig(preview, dpi=150)
        plt.close(figure)
        return [preview]

    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(tif, dpi=DPI_TIF, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
    figure.savefig(preview, dpi=DPI_PREVIEW, bbox_inches="tight")
    plt.close(figure)
    return [pdf, tif, preview]

def write_qa(paths: list[Path]) -> None:
    rows = [
        {"path": str(path), "format": path.suffix.lstrip("."), "bytes": path.stat().st_size,
         "pass": path.stat().st_size > 10_000}
        for path in paths
    ]
    with (OUT / "figure2_redraw_export_QA_v1.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    qa = {
        "status": "PASS_EXPORT_PENDING_VISUAL_REVIEW",
        "backend": "Python/matplotlib (nature-figure skill)",
        "archetype": "quantitative grid: trajectory rows + map grid",
        "core_conclusion": "Global adult BMI keeps rising through 2050 under SSP2-4.5 across mean BMI, overweight and obesity, with pronounced spatial heterogeneity; alternative scenarios differ modestly (strips).",
        "panels": {"a": "mean BMI trajectory + scenario strip", "b": "BMI>=25 trajectory + strip",
                    "c": "BMI>=30 trajectory + strip", "d": "mean BMI maps 2015/2025/2035/2050",
                    "e": "BMI>=25 maps", "f": "BMI>=30 maps"},
        "reference_scenario": "SSP2-4.5 (strips show SSP1-2.6/3-7.0/5-8.5 minus SSP2-4.5)",
        "scenario_divergence_panel": "removed from main figure per advisor; remains in SM",
        "years_maps": list(YEARS_MAP),
        "fixed_map_scales": {m.key: [m.vmin, m.vmax, m.unit] for m in METRICS},
        "trajectory_y_limits": {m.key: [m.tmin, m.tmax] for m in METRICS},
        "canvas_mm": [WIDTH_MM, HEIGHT_MM],
        "tiff_dpi": DPI_TIF,
        "tiff_compression": "LZW",
        "population_mask": "adult population >=100 persons per 0.1-degree cell (helper-inherited)",
        "in_figure_title": False,
        "formal_products_modified": False,
    }
    (OUT / "figure2_redraw_parameter_QA_v1.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not all(row["pass"] for row in rows):
        raise RuntimeError("Figure 2 redraw export QA failed")

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "worker_status.json"
    status.write_text(json.dumps({"status": "running"}) + "\n", encoding="utf-8")
    paths = plot()
    write_qa(paths)
    status.write_text(json.dumps({"status": "complete", "outputs": [str(path) for path in paths]}, indent=2) + "\n", encoding="utf-8")
    print("FIGURE2_REDRAW_PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
