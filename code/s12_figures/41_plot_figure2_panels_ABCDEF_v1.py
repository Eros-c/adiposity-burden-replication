#!/usr/bin/env python3
"""Figure 2 panels A-F (v2 layout): each panel exported as its own PDF + TIFF.

- A/B/C (trajectory panels): left = SSP2-4.5 overall + female + male trend,
  right = SSP1-2.6/3-7.0/5-8.5 difference from SSP2-4.5 over time.
  Metrics: A mean BMI, B BMI>=25, C BMI>=30.
- D/E/F (map rows): SSP2-4.5 maps for 2015/2025/2035/2050 on the 0.1-degree grid.
  Metrics: D mean BMI, E BMI>=25, F BMI>=30.

Exports per letter: Figure2_<LETTER>_*.pdf, *_1600dpi.tif, *_preview.png.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

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
TREND_OVERALL_CSV = ROOT / "test_figure1_AB_v2" / "TEST_Fig1A_trend_source_four_SSP_v2.csv"
SEX_TREND_CSV = ROOT / "figure2_bmi_global_2015_2050_v1" / "tables" / "Figure2_sex_trend_source_v1.csv"
OUT = ROOT / "figure2_bmi_global_2015_2050_v1" / "panels"

WIDTH_MM = 190.5
HEIGHT_TRAJ_MM = 43.0
HEIGHT_MAP_MM = 43.0
DPI_TIF = 1600  # tiff dpi: 1600 (explicit raster resolution for preflight)
DPI_PREVIEW = 400
YEARS_MAP = (2015, 2025, 2035, 2050)

MIDPOINTS = np.asarray([17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5], dtype=np.float32)
MAP_COLORS = ("#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c")

BLACK = "#151718"
GREY = "#596064"
HISTORICAL_COLOR = "#151718"
REFERENCE_COLOR = "#009E73"
FEMALE_COLOR = "#E69F00"
MALE_COLOR = "#0072B2"
STRIP_COLORS = {"ssp126": "#0072B2", "ssp370": "#E69F00", "ssp585": "#D55E00"}
STRIP_LABELS = {"ssp126": "SSP1-2.6", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
SCENARIO_NAMES = {"Historical": "Historical", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
                  "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}


@dataclass(frozen=True)
class Metric:
    key: str
    letter_traj: str
    letter_map: str
    label: str
    unit: str
    vmin: float
    vmax: float
    tmin: float
    tmax: float
    ylabel: str


METRICS = (
    Metric("mean_bmi", "A", "D", "Mean BMI (kg/m\u00b2)", "kg/m\u00b2", 18.0, 32.0, 24.0, 28.5, "Mean BMI (kg/m\u00b2)"),
    Metric("bmi_ge25", "B", "E", "BMI \u226525 (%)", "%", 0.0, 90.0, 40.0, 65.0, "Prevalence (%)"),
    Metric("bmi_ge30", "C", "F", "BMI \u226530 (%)", "%", 0.0, 60.0, 12.0, 34.0, "Prevalence (%)"),
)


def import_helper():
    spec = importlib.util.spec_from_file_location("figure2_panel_helper", HELPER_SCRIPT)
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


def _clip_segments_to_extent(
    segments,
    xlim: tuple[float, float] = (-180.0, 180.0),
    ylim: tuple[float, float] = (-60.0, 84.0),
) -> list[np.ndarray]:
    """Drop/split boundary vertices outside the map window.

    Antarctica and other sub-60°S geometry otherwise extends below ylim and,
    when the PDF backend skips LineCollection clipping, paints strokes across
    the colourbar tick labels.
    """
    x0, x1 = xlim
    y0, y1 = ylim
    clipped: list[np.ndarray] = []
    for segment in segments:
        arr = np.asarray(segment, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            continue
        inside = (
            (arr[:, 0] >= x0) & (arr[:, 0] <= x1)
            & (arr[:, 1] >= y0) & (arr[:, 1] <= y1)
        )
        if inside.all():
            clipped.append(arr)
            continue
        if not inside.any():
            continue
        start = None
        for index, ok in enumerate(inside):
            if ok and start is None:
                start = index
            elif (not ok) and start is not None:
                if index - start >= 2:
                    clipped.append(arr[start:index])
                start = None
        if start is not None and arr.shape[0] - start >= 2:
            clipped.append(arr[start:])
    return clipped


def draw_map(axis, data, metric: Metric, country_segments, adm1_segments) -> None:
    axis.imshow(
        data, extent=(-180, 180, -60, 84), origin="upper",
        cmap=metric_cmap(f"fig2p_{metric.key}"),
        norm=Normalize(vmin=metric.vmin, vmax=metric.vmax),
        interpolation="nearest", rasterized=True, zorder=1,
    )
    historical.add_boundaries(
        axis,
        _clip_segments_to_extent(country_segments),
        _clip_segments_to_extent(adm1_segments) if adm1_segments else None,
    )
    # Belt-and-suspenders: keep clip_on after aspect/axis changes (set_aspect
    # can reset child clip paths, and axis('off') hides the patch clip source).
    axis.set_xlim(-180, 180)
    axis.set_ylim(-60, 84)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    for collection in axis.collections:
        collection.set_clip_on(True)


def add_colorbar(axis, metric: Metric) -> None:
    colorbar = mpl.colorbar.ColorbarBase(
        axis, cmap=metric_cmap(f"fig2p_cb_{metric.key}"),
        norm=Normalize(vmin=metric.vmin, vmax=metric.vmax),
        orientation="horizontal",
    )
    ticks = np.linspace(metric.vmin, metric.vmax, 4)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:.0f}" for value in ticks])
    colorbar.ax.tick_params(labelsize=6.0, length=1.4, width=0.4, pad=1.5)
    colorbar.outline.set_visible(False)


def overall_series(frame: pd.DataFrame, metric: Metric, scenario: str) -> pd.Series:
    subset = frame[frame["scenario"] == SCENARIO_NAMES[scenario]].sort_values("year")
    column = {"mean_bmi": "mean_bmi", "bmi_ge25": "overweight_obesity", "bmi_ge30": "obesity"}[metric.key]
    values = subset[column].astype(float)
    if metric.key != "mean_bmi":
        values = values * 100.0
    return pd.Series(values.values, index=subset["year"].values)


def sex_series(frame: pd.DataFrame, metric: Metric, sex: str, period: str) -> pd.Series:
    subset = frame[(frame["sex"] == sex) & (frame["period"] == period)].sort_values("year")
    values = subset[metric.key].astype(float)
    return pd.Series(values.values, index=subset["year"].values)


def export(figure, stem: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{stem}.pdf"
    tif = OUT / f"{stem}.tif"
    preview = OUT / f"{stem}_preview.png"
    figure.savefig(pdf)
    figure.savefig(tif, dpi=DPI_TIF, pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(preview, dpi=DPI_PREVIEW)
    plt.close(figure)
    return [pdf, tif, preview]


def render_traj_panel(metric: Metric, trend_overall: pd.DataFrame, trend_sex: pd.DataFrame) -> list[Path]:
    figure = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_TRAJ_MM / 25.4), constrained_layout=False)
    grid = figure.add_gridspec(
        1, 2, left=0.055, right=0.950, bottom=0.175, top=0.895,
        width_ratios=(1.65, 1.0), wspace=0.30,
    )
    main_axis = figure.add_subplot(grid[0, 0])
    diff_axis = figure.add_subplot(grid[0, 1], sharex=main_axis)

    # Overall trajectory: authoritative overall-level series, identical to the
    # combined Figure 2 (its 2024 bridge row equals the historical endpoint,
    # so hist->fut is continuous by construction).
    hist_overall = overall_series(trend_overall, metric, "Historical")
    ref_overall = overall_series(trend_overall, metric, "ssp245")
    hist_female = sex_series(trend_sex, metric, "Female", "Historical")
    hist_male = sex_series(trend_sex, metric, "Male", "Historical")
    fut_female = sex_series(trend_sex, metric, "Female", "SSP2-4.5")
    fut_male = sex_series(trend_sex, metric, "Male", "SSP2-4.5")

    # Bridge-anchor sex projections: the sex-level future grids carry a
    # product-family level shift at the 2024/2025 boundary (up to ~2.7 pp on
    # threshold metrics). Apply a per-sex constant offset to 2025-2050 so each
    # sex continues from its own 2024 historical endpoint plus the canonical
    # overall year-over-year change (TEST SSP2-4.5 2024->2025). Projection
    # deltas after 2025 are preserved; the weighted sexes stay bracketed on
    # the overall series (verified <=0.3 pp at 2050).
    bridge_delta = float(ref_overall.loc[2025] - ref_overall.loc[2024])
    offset_female = float(hist_female.loc[2024] + bridge_delta - fut_female.loc[2025])
    fut_female = (fut_female + offset_female).sort_index()
    fut_female.loc[2024] = hist_female.loc[2024]
    fut_female = fut_female.sort_index()
    offset_male = float(hist_male.loc[2024] + bridge_delta - fut_male.loc[2025])
    fut_male = (fut_male + offset_male).sort_index()
    fut_male.loc[2024] = hist_male.loc[2024]
    fut_male = fut_male.sort_index()

    # Period shading (historical band) and projection boundary on both subplots.
    for axis in (main_axis, diff_axis):
        axis.axvspan(2015, 2024.5, color="#F0F1F2", zorder=0)
        axis.axvline(2024.5, color="#8a9094", linewidth=0.5, linestyle=(0, (1.5, 1.5)), zorder=2)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E8EAEC", linewidth=0.35)
    main_axis.text(2019.7, 0.97, "Historical", transform=main_axis.get_xaxis_transform(),
                   ha="center", va="top", fontsize=5.8, color=GREY, zorder=5)
    main_axis.text(2037.5, 0.97, "Projection", transform=main_axis.get_xaxis_transform(),
                   ha="center", va="top", fontsize=5.8, color=GREY, zorder=5)

    # Left panel: overall + sex trajectories; one marker per available year.
    # Historical products only exist for anchor years 2015/2020/2024; projected
    # series carry annual markers (2024 bridge point prepended above).
    main_axis.plot(hist_overall.index, hist_overall.values, color=HISTORICAL_COLOR,
                   linewidth=0.9, marker="o", markersize=2.0, zorder=5)
    main_axis.plot(ref_overall.index, ref_overall.values, color=REFERENCE_COLOR,
                   linewidth=1.0, marker="o", markersize=1.5, zorder=4)
    main_axis.plot(hist_female.index, hist_female.values, color=FEMALE_COLOR,
                   linewidth=0.7, marker="o", markersize=1.6, zorder=4)
    main_axis.plot(fut_female.index, fut_female.values, color=FEMALE_COLOR,
                   linewidth=0.75, marker="o", markersize=1.4, zorder=3)
    main_axis.plot(hist_male.index, hist_male.values, color=MALE_COLOR,
                   linewidth=0.7, marker="o", markersize=1.6, zorder=4)
    main_axis.plot(fut_male.index, fut_male.values, color=MALE_COLOR,
                   linewidth=0.75, marker="o", markersize=1.4, zorder=3)
    main_axis.set_ylim(metric.tmin, metric.tmax)
    main_axis.set_xlim(2015, 2050)
    main_axis.set_xticks([2015, 2025, 2035, 2050])
    main_axis.set_yticks(np.linspace(metric.tmin, metric.tmax, 3))
    # Larger x-pad keeps the first x-tick ("2015") clear of the bottom y-tick label.
    main_axis.tick_params(axis="x", labelsize=6.0, length=1.6, width=0.45, pad=5.0)
    main_axis.tick_params(axis="y", labelsize=6.0, length=1.6, width=0.45, pad=2.0)
    main_axis.set_ylabel(metric.ylabel, fontsize=6.5, labelpad=2.0)

    # 2050 end-of-series value labels (outside the axes, inside the gutter).
    # When labels sit closer than ~7 pt (e.g. panel B: 62.0/61.3/61.1), push
    # them apart vertically and draw a hairline leader back to the endpoint.
    fig_h_pt = HEIGHT_TRAJ_MM / 25.4 * 72.0
    axes_h_pt = fig_h_pt * (0.895 - 0.175)
    pt_per_unit = axes_h_pt / (metric.tmax - metric.tmin)
    min_gap_data = 7.0 / pt_per_unit
    ends = sorted(
        ((float(ref_overall.loc[2050]), REFERENCE_COLOR),
         (float(fut_female.loc[2050]), FEMALE_COLOR),
         (float(fut_male.loc[2050]), MALE_COLOR)),
        key=lambda item: -item[0],
    )
    previous_pos = None
    for end_value, color in ends:
        label_pos = end_value if previous_pos is None else min(end_value, previous_pos - min_gap_data)
        previous_pos = label_pos
        dy_pt = (label_pos - end_value) * pt_per_unit
        arrowprops = None
        if abs(dy_pt) >= 0.5:
            arrowprops = dict(arrowstyle="-", color=color, lw=0.3,
                              shrinkA=0.0, shrinkB=2.5)
        main_axis.annotate(
            f"{end_value:.1f}", xy=(2050, end_value), xytext=(7.0, dy_pt),
            textcoords="offset points", fontsize=6.0, color=color,
            ha="left", va="center", clip_on=False, zorder=6,
            arrowprops=arrowprops,
        )

    legend_handles = [
        Line2D([], [], color=HISTORICAL_COLOR, linewidth=0.9, marker="o", markersize=2.0,
               label="Historical (overall)"),
        Line2D([], [], color=REFERENCE_COLOR, linewidth=1.0, marker="o", markersize=1.5,
               label="SSP2-4.5 (overall)"),
        Line2D([], [], color=FEMALE_COLOR, linewidth=0.75, marker="o", markersize=1.4,
               label="SSP2-4.5 (female)"),
        Line2D([], [], color=MALE_COLOR, linewidth=0.75, marker="o", markersize=1.4,
               label="SSP2-4.5 (male)"),
    ]
    main_axis.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(0.995, 0.05),
                     fontsize=5.8, ncol=2, handlelength=1.3, handletextpad=0.5,
                     columnspacing=0.8, borderpad=0.2, labelspacing=0.3)

    # Right panel: scenario differences vs SSP2-4.5, forced to 0 at bridge year 2024.
    reference_values = overall_series(trend_overall, metric, "ssp245").reindex(range(2024, 2051)).astype(float)
    scenario_diffs: dict[str, pd.Series] = {}
    for scenario in STRIP_COLORS:
        series = overall_series(trend_overall, metric, scenario).reindex(range(2024, 2051)).astype(float)
        difference = series - reference_values
        difference.loc[2024] = 0.0
        scenario_diffs[scenario] = difference
    envelope_low = pd.concat(scenario_diffs.values(), axis=1).min(axis=1)
    envelope_high = pd.concat(scenario_diffs.values(), axis=1).max(axis=1)
    diff_axis.fill_between(envelope_low.index, envelope_low.values, envelope_high.values,
                           color="#AEB4B8", alpha=0.28, linewidth=0.0, zorder=1)
    for scenario, color in STRIP_COLORS.items():
        difference = scenario_diffs[scenario]
        diff_axis.plot(difference.index, difference.values, color=color,
                       linewidth=0.7, marker="o", markersize=1.5, zorder=3)
    diff_axis.plot([2024], [0.0], marker="o", markersize=2.4, color=BLACK, zorder=4,
                   markeredgecolor="white", markeredgewidth=0.3)
    diff_axis.axhline(0.0, color="#9aa0a4", linewidth=0.45, zorder=2)
    spread = max(float(values.abs().max()) for values in scenario_diffs.values())
    limit = max(np.ceil(spread * 1.35 * 50) / 50, 0.04)
    diff_axis.set_ylim(-limit, limit)
    diff_axis.set_yticks([-limit, -limit / 2, 0.0, limit / 2, limit])
    diff_axis.set_xlim(2015, 2050)
    diff_axis.set_xticks([2015, 2025, 2035, 2050])
    # Same corner fix as the main axis: push x-ticks below the bottom y-tick labels.
    diff_axis.tick_params(axis="x", labelsize=6.0, length=1.6, width=0.45, pad=5.0)
    diff_axis.tick_params(axis="y", labelsize=6.0, length=1.6, width=0.45, pad=2.0)
    diff_axis.set_ylabel("vs SSP2-4.5", fontsize=6.5, labelpad=2.0)
    diff_axis.set_xlabel("Year", fontsize=6.2, labelpad=1.5)
    diff_handles = [
        Line2D([], [], color=STRIP_COLORS["ssp126"], linewidth=0.7, marker="o", markersize=1.5,
               label="SSP1-2.6"),
        Line2D([], [], color=STRIP_COLORS["ssp370"], linewidth=0.7, marker="o", markersize=1.5,
               label="SSP3-7.0"),
        Line2D([], [], color=STRIP_COLORS["ssp585"], linewidth=0.7, marker="o", markersize=1.5,
               label="SSP5-8.5"),
    ]
    diff_axis.legend(handles=diff_handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                     fontsize=5.8, ncol=3, handlelength=1.2, handletextpad=0.4,
                     columnspacing=0.7, borderpad=0.1)

    require_matplotlib_panel_alignment(
        figure,
        json_out=str(OUT / f"Figure2_{metric.letter_traj}.alignment.json"),
        overlay_svg=str(OUT / f"Figure2_{metric.letter_traj}.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    stem = f"Figure2_{metric.letter_traj}_{metric.key}_ssp245_sex_trend_and_scenario_diff_v1"
    return export(figure, stem)


def render_map_panel(metric: Metric, category_cache: dict, country_segments, adm1_segments) -> list[Path]:
    figure = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MAP_MM / 25.4), constrained_layout=False)
    grid = figure.add_gridspec(1, 4, left=0.006, right=0.994, bottom=0.175, top=0.647, wspace=0.014)
    first_map = None
    for year_index, year in enumerate(YEARS_MAP):
        axis = figure.add_subplot(grid[0, year_index])
        draw_map(axis, derive_metric(category_cache[year], metric), metric, country_segments, adm1_segments)
        axis.set_title(str(year), fontsize=7.0, fontweight="bold", pad=2.5, color=BLACK)
        if year_index == 0:
            first_map = axis

    require_matplotlib_panel_alignment(
        figure,
        json_out=str(OUT / f"Figure2_{metric.letter_map}.alignment.json"),
        overlay_svg=str(OUT / f"Figure2_{metric.letter_map}.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )

    figure.canvas.draw()
    bbox = first_map.get_position()
    # +0.008 keeps the first tick label ("18") fully on the page (it otherwise
    # extends ~0.1 pt past the left edge when aligned exactly with the map box).
    color_axis = figure.add_axes([bbox.x0 + 0.008, bbox.y0 - 0.055, bbox.width * 0.60, 0.020])
    add_colorbar(color_axis, metric)
    figure.text(0.012, 0.855, metric.label, ha="left", va="top",
                fontsize=7.6, fontweight="bold", color=BLACK)
    figure.text(0.012, 0.788, "SSP2-4.5", ha="left", va="top",
                fontsize=6.0, color=GREY)

    stem = f"Figure2_{metric.letter_map}_{metric.key}_maps_ssp245_2015_2025_2035_2050_v1"
    return export(figure, stem)


def write_manifest(paths: list[Path]) -> None:
    rows = [{"path": str(path), "format": path.suffix.lstrip("."), "bytes": path.stat().st_size,
             "pass": path.stat().st_size > 10_000} for path in paths]
    with (OUT / "figure2_panels_export_QA_v1.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    qa = {
        "status": "PASS_EXPORT_PENDING_VISUAL_REVIEW",
        "backend": "Python/matplotlib (nature-figure skill)",
        "panels": {
            "A": "Mean BMI: SSP2-4.5 overall+female+male trend (left); SSP1/3/5 difference (right)",
            "B": "BMI>=25: same layout",
            "C": "BMI>=30: same layout",
            "D": "Mean BMI maps 2015/2025/2035/2050 SSP2-4.5",
            "E": "BMI>=25 maps, same years",
            "F": "BMI>=30 maps, same years",
        },
        "sex_trend_source": "Figure2_sex_trend_source_v1.csv (aggregated from frozen 0.1-degree sex-level products; left-panel overall = population-weighted mean of female/male series for bracket consistency. Sex-level products differ from overall-level products by ~1.4-1.9pp for threshold metrics; scenario strips use the overall-level TEST_Fig1A series).",
        "annual_markers": "Projected segments (2024 bridge-2050) carry one marker per year on every trajectory and difference curve. Historical segments have markers only at product anchor years 2015/2020/2024 (annual historical 0.1-degree rasters do not exist; no interpolation invented).",
        "hist_fut_bridge": "Left-panel overall line uses the authoritative overall-level TEST_Fig1A series (identical to the combined Figure 2; its 2024 bridge row equals the historical endpoint). Sex-level future grids carry a product-family level shift at 2024/2025 (up to ~2.7 pp on threshold metrics), so each sex's 2025-2050 series receives a per-sex constant offset anchoring it to its own 2024 historical endpoint plus the canonical overall 2024->2025 change; projection deltas after 2025 are preserved and the weighted sexes stay bracketed on the overall line (<=0.3 pp at 2050).",
        "scenario_diff_origin": "Difference curves are forced to 0.0 at bridge year 2024, drawn with annual markers, plus a min-max scenario envelope fill; the 2024 origin is emphasised with a black anchor point.",
        "canvas_mm": {"traj": [WIDTH_MM, HEIGHT_TRAJ_MM], "maps": [WIDTH_MM, HEIGHT_MAP_MM]},
        "tiff_dpi": DPI_TIF,
        "tiff_compression": "LZW",
        "formal_products_modified": False,
    }
    (OUT / "figure2_panels_parameter_QA_v1.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not all(row["pass"] for row in rows):
        raise RuntimeError("Figure 2 panel export QA failed")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "worker_status.json"
    status.write_text(json.dumps({"status": "running"}) + "\n", encoding="utf-8")
    trend_overall = pd.read_csv(TREND_OVERALL_CSV)
    trend_sex = pd.read_csv(SEX_TREND_CSV)
    paths: list[Path] = []
    for metric in METRICS:
        paths.extend(render_traj_panel(metric, trend_overall, trend_sex))
    category_cache = {year: read_categories(year) for year in YEARS_MAP}
    country_segments, adm1_segments = build_boundaries()
    for metric in METRICS:
        paths.extend(render_map_panel(metric, category_cache, country_segments, adm1_segments))
    write_manifest(paths)
    status.write_text(json.dumps({"status": "complete", "outputs": [str(path) for path in paths]}, indent=2) + "\n", encoding="utf-8")
    print("FIGURE2_PANELS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
