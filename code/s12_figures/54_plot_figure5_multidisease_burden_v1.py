#!/usr/bin/env python3
"""Figure 5 (multi-disease): high-BMI-attributable burden across 23 diseases, 2025-2050.

Panels (A4 width, 4 rows x 2 columns):
  (a) Global burden trajectories: total 23-disease burden (grey) vs attributable
      burden stacked by disease group (Deaths | DALYs), SSP2-4.5.
  (b) National combined attributable fraction in 2050 (Deaths | DALYs): the share
      of a country's combined 23-disease burden attributable to high BMI.
  (c) Age- and sex-specific attributable burden in 2050, summed over the 23
      diseases (Deaths | DALYs).
  (d) Ten countries with the largest attributable burden in 2050, stacked by
      disease group (Deaths | DALYs).

Data: 11_all_disease_extension Phase 4 (GBD 2021 harmonized, scenario-specific
study PAF x common GBD Foresight reference burden).
"""

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

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
HELPER_SCRIPT = HERE / "08_plot_figure1B_historical_template_test_v2.py"
PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
OUT = ROOT / "figure5_multidisease_burden_v1"
PANELS = OUT / "panels"
TABLES = OUT / "tables"
QA = OUT / "qa"

BURDEN = PROJECT / "11_all_disease_extension" / "04_burden" / "burden_benchmark_v1_20261005" / "data"
GLOBAL_YEAR = BURDEN / "attributable_future_global_year_four_ssp_v1.parquet"
COUNTRY_OVERALL = BURDEN / "attributable_future_country_overall_four_ssp_v1.parquet"
AGE_SEX_2050 = BURDEN / "attributable_future_age_sex_2050_four_ssp_v1.parquet"

SCENARIO_CODE = "ssp245"
AGES = ("25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64",
        "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+")
MEASURES = ("Deaths", "DALYs")
DPI = 600
TIFF_DPI = 1800

BLUE = "#2676A8"
ORANGE = "#D55E00"
GREY = "#A7AEB2"
BLACK = "#17191A"
PALETTE = ("#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c")
MAP_MAX = 55.0

GROUP_COLORS = {
    "Cardiovascular": "#2676A8",
    "Metabolic/renal": "#4E9A51",
    "Cancer": "#B8562F",
    "Other": "#8A7CA8",
}
GROUP_ORDER = ["Cardiovascular", "Metabolic/renal", "Cancer", "Other"]
CAUSE_GROUPS = {
    441: "Cancer", 417: "Cancer", 453: "Cancer", 456: "Cancer", 429: "Cancer",
    435: "Cancer", 465: "Cancer", 471: "Cancer", 480: "Cancer", 485: "Cancer",
    486: "Cancer", 487: "Cancer",
    493: "Cardiovascular", 494: "Cardiovascular", 498: "Cardiovascular",
    500: "Cardiovascular", 502: "Cardiovascular",
    976: "Metabolic/renal", 998: "Metabolic/renal", 591: "Metabolic/renal",
    515: "Other", 534: "Other", 543: "Other",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.65,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_helper():
    spec = importlib.util.spec_from_file_location("figure5_md_map_helper", HELPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


historical = import_helper().historical


def format_axis_number(value, _position=None):
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def load_sources():
    globe = pd.read_parquet(GLOBAL_YEAR)
    globe = globe[(globe["analysis_tier"] == "headline") & (globe["scenario_code"] == SCENARIO_CODE)].copy()
    country = pd.read_parquet(COUNTRY_OVERALL)
    country = country[(country["analysis_tier"] == "headline") & (country["scenario_code"] == SCENARIO_CODE)].copy()
    age_sex = pd.read_parquet(AGE_SEX_2050)
    age_sex = age_sex[(age_sex["analysis_tier"] == "headline") & (age_sex["scenario_code"] == SCENARIO_CODE)].copy()
    for frame in (globe, country, age_sex):
        frame["cause_group"] = frame["cause_id"].map(CAUSE_GROUPS)
    return globe, country, age_sex


def draw_trend_axes(axes, globe):
    summary = {}
    for axis, measure in zip(axes, MEASURES):
        data = globe[globe["measure"] == measure]
        yearly = data.groupby("year", as_index=False)[["total_burden", "attributable_burden"]].sum()
        by_group = (
            data.groupby(["year", "cause_group"], as_index=False)["attributable_burden"].sum()
            .pivot(index="year", columns="cause_group", values="attributable_burden")
            .fillna(0.0)
        )
        by_group = by_group[GROUP_ORDER]
        years = yearly["year"].to_numpy()
        axis.plot(years, yearly["total_burden"], color="#747C80", linewidth=1.25, zorder=4)
        axis.stackplot(
            years, *[by_group[g].to_numpy() for g in GROUP_ORDER],
            colors=[GROUP_COLORS[g] for g in GROUP_ORDER], alpha=0.88, zorder=2,
        )
        axis.plot(years, yearly["attributable_burden"], color=BLACK, linewidth=0.9, zorder=3)
        last = yearly.iloc[-1]
        af_pct = 100.0 * last["attributable_burden"] / last["total_burden"]
        axis.annotate(
            f"{format_axis_number(last['total_burden'])} total",
            (2050, last["total_burden"]), xytext=(-3, 5), textcoords="offset points",
            ha="right", va="bottom", fontsize=5.2, color="#747C80",
        )
        axis.annotate(
            f"{format_axis_number(last['attributable_burden'])} attributable\n{af_pct:.1f}% of total",
            (2050, last["attributable_burden"]), xytext=(-3, 3), textcoords="offset points",
            ha="right", va="bottom", fontsize=5.2, color=BLACK, linespacing=1.12,
        )
        axis.set_xlim(2024.5, 2051.0)
        axis.set_ylim(0, last["total_burden"] * 1.12)
        axis.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
        axis.yaxis.set_major_formatter(FuncFormatter(format_axis_number))
        axis.grid(axis="y", color="#E9ECEE", linewidth=0.4)
        axis.set_title(measure, loc="left", fontsize=7.2, fontweight="bold", pad=2.5, color=BLACK)
        axis.set_xlabel("Year", fontsize=5.8, labelpad=2)
        axis.set_ylabel("Burden", fontsize=5.8, labelpad=2)
        axis.tick_params(axis="both", labelsize=5.4, length=2.5, width=0.5, pad=1.5)
        summary[measure] = {
            "total_2025": float(yearly.iloc[0]["total_burden"]),
            "total_2050": float(last["total_burden"]),
            "attributable_2025": float(yearly.iloc[0]["attributable_burden"]),
            "attributable_2050": float(last["attributable_burden"]),
            "af_pct_2025": float(100.0 * yearly.iloc[0]["attributable_burden"] / yearly.iloc[0]["total_burden"]),
            "af_pct_2050": float(af_pct),
        }
    handles = [Line2D([], [], color="#747C80", linewidth=1.25,
                      label="Total burden, 23 diseases")]
    handles += [Patch(facecolor=GROUP_COLORS[g], edgecolor="none", label=g) for g in GROUP_ORDER]
    axes[0].legend(handles=handles, loc="upper left", fontsize=5.0, ncol=1,
                   handlelength=1.6, labelspacing=0.42)
    return summary


def build_map_data():
    countries = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")
    countries = countries[countries.geometry.notna()].copy()
    countries["plot_iso"] = countries["ISO_A3_EH"].where(
        ~countries["ISO_A3_EH"].isin(["-99", ""]), countries["ADM0_A3"]
    )
    countries["plot_iso"] = countries["plot_iso"].replace({"PSX": "PSE", "SDS": "SSD"})
    countries.geometry = countries.geometry.simplify(0.035, preserve_topology=True)
    country_segments = historical.geometry_segments(countries)
    return countries, country_segments


def map_cmap():
    cmap = LinearSegmentedColormap.from_list("fig5_md_country_fraction", PALETTE, N=256)
    cmap.set_bad("white")
    return cmap


def draw_country_map(axis, countries, segments, values, title):
    mapping = values.set_index("iso3")["combined_af_pct"]
    plot = countries.copy()
    plot["value"] = plot["plot_iso"].map(mapping)
    norm = Normalize(0, MAP_MAX)
    colors = np.asarray(
        [map_cmap()(norm(value)) if np.isfinite(value) else mpl.colors.to_rgba("white")
         for value in plot["value"].to_numpy(float)],
        dtype=float,
    )
    plot.plot(ax=axis, color=colors, edgecolor="none", linewidth=0, zorder=1)
    historical.add_boundaries(axis, segments, None)
    axis.set_xlim(-180, 180)
    axis.set_ylim(-60, 84)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    axis.set_title(title, fontsize=6.8, fontweight="bold", pad=2.0, color=BLACK)
    return int(plot["value"].notna().sum())


def add_map_colorbar(figure, rect):
    axis = figure.add_axes(rect)
    cb = mpl.colorbar.ColorbarBase(axis, cmap=map_cmap(), norm=Normalize(0, MAP_MAX),
                                   orientation="horizontal")
    cb.set_ticks([0, 10, 20, 30, 40, 50])
    cb.ax.tick_params(labelsize=5.0, length=1.5, width=0.4, pad=0.8)
    cb.outline.set_visible(False)
    cb.ax.text(0, 1.65, "%", transform=cb.ax.transAxes, ha="left", va="bottom",
               fontsize=5.0, color="#4a5053")


def draw_age_sex_axes(axes, age_sex):
    colors = {"Male": BLUE, "Female": ORANGE}
    markers = {"Male": "o", "Female": "s"}
    age_order = {age: i for i, age in enumerate(AGES)}
    for axis, measure in zip(axes, MEASURES):
        data = (
            age_sex[age_sex["measure"] == measure]
            .groupby(["sex", "age_group"], as_index=False)["attributable_burden"].sum()
        )
        data["age_order"] = data["age_group"].map(age_order)
        for sex in ("Male", "Female"):
            selected = data[data["sex"] == sex].sort_values("age_order")
            axis.plot(selected["age_order"], selected["attributable_burden"],
                      color=colors[sex], linewidth=1.25, marker=markers[sex],
                      markersize=2.8, label=sex)
        peak = data.loc[data["attributable_burden"].idxmax()]
        axis.annotate(f"Peak: {peak.age_group}", (peak.age_order, peak.attributable_burden),
                      xytext=(3, 4), textcoords="offset points", fontsize=5.0,
                      color=colors[peak.sex])
        axis.set_xlim(-0.5, len(AGES) - 0.5)
        axis.set_xticks(range(len(AGES)), AGES, rotation=45, ha="right")
        axis.yaxis.set_major_formatter(FuncFormatter(format_axis_number))
        axis.grid(axis="y", color="#E9ECEE", linewidth=0.4)
        axis.set_title(measure, loc="left", fontsize=7.0, fontweight="bold", pad=2.3, color=BLACK)
        axis.set_xlabel("Age group (years)", fontsize=5.7, labelpad=2)
        axis.set_ylabel("Attributable burden", fontsize=5.7, labelpad=2)
        axis.tick_params(axis="both", labelsize=5.0, length=2.3, width=0.45, pad=1.2)
    axes[0].legend(loc="upper left", fontsize=5.2, ncol=2, handlelength=1.5, columnspacing=0.9)


def draw_top10_axes(axes, country):
    short_names = {
        "China": "China", "India": "India", "United States of America": "United States",
        "Egypt": "Egypt", "Brazil": "Brazil", "Russian Federation": "Russia",
        "Mexico": "Mexico", "Pakistan": "Pakistan", "Indonesia": "Indonesia",
        "Iran (Islamic Republic of)": "Iran", "Nigeria": "Nigeria", "Bangladesh": "Bangladesh",
        "Japan": "Japan", "Turkiye": "Turkiye", "Germany": "Germany",
    }
    qa = {}
    for axis, measure in zip(axes, MEASURES):
        data = country[country["measure"] == measure]
        totals = data.groupby(["iso3", "crosswalk_location_name"], as_index=False)["attributable_burden"].sum()
        top_iso = totals.sort_values("attributable_burden", ascending=False).head(10)["iso3"].tolist()
        sub = data[data["iso3"].isin(top_iso)]
        stack = sub.pivot_table(index="iso3", columns="cause_group",
                                values="attributable_burden", aggfunc="sum", fill_value=0.0)
        stack = stack.reindex(index=top_iso, columns=GROUP_ORDER, fill_value=0.0)
        names = totals.set_index("iso3")["crosswalk_location_name"]
        labels = [short_names.get(names[i], names[i]) for i in stack.index]
        y = np.arange(len(stack))[::-1]
        left = np.zeros(len(stack))
        for group in GROUP_ORDER:
            vals = stack[group].to_numpy()
            axis.barh(y, vals, left=left, color=GROUP_COLORS[group], height=0.66,
                      edgecolor="none", zorder=2)
            left += vals
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=5.2)
        axis.xaxis.set_major_formatter(FuncFormatter(format_axis_number))
        axis.grid(axis="x", color="#E9ECEE", linewidth=0.4)
        axis.set_axisbelow(True)
        axis.set_title(f"Attributable {'deaths' if measure == 'Deaths' else 'DALYs'}",
                       loc="left", fontsize=7.0, fontweight="bold", pad=2.3, color=BLACK)
        axis.set_xlabel("Attributable burden", fontsize=5.7, labelpad=2)
        axis.tick_params(axis="x", labelsize=5.1, length=2.3, width=0.45, pad=1.2)
        axis.tick_params(axis="y", labelsize=5.2, length=0, pad=2)
        qa[measure] = [names[i] for i in stack.index]
    axes[1].legend(
        handles=[Patch(facecolor=GROUP_COLORS[g], edgecolor="none", label=g) for g in GROUP_ORDER],
        loc="lower right", fontsize=5.0, handlelength=1.2, borderaxespad=0.5,
    )
    return qa


def main() -> None:
    globe, country, age_sex = load_sources()

    # Panel (b) combined attributable fraction.
    cb = (
        country[country["year"] == 2050]
        .groupby(["iso3", "measure"], as_index=False)[["total_burden", "attributable_burden"]].sum()
    )
    cb["combined_af_pct"] = 100.0 * cb["attributable_burden"] / cb["total_burden"]

    fig = plt.figure(figsize=(190.5 / 25.4, 240 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        4, 2, left=0.130, right=0.985, top=0.950, bottom=0.050,
        hspace=0.62, wspace=0.28, height_ratios=[0.9, 1.05, 1.0, 1.15],
    )
    axes_a = [fig.add_subplot(grid[0, i]) for i in range(2)]
    axes_b = [fig.add_subplot(grid[1, i]) for i in range(2)]
    axes_c = [fig.add_subplot(grid[2, i]) for i in range(2)]
    axes_d = [fig.add_subplot(grid[3, i]) for i in range(2)]

    trend_summary = draw_trend_axes(axes_a, globe)

    countries, segments = build_map_data()
    for axis, measure in zip(axes_b, MEASURES):
        values = cb[cb["measure"] == measure]
        n_mapped = draw_country_map(
            axis, countries, segments, values,
            f"Combined attributable fraction of {'deaths' if measure == 'Deaths' else 'DALYs'} (%)",
        )
        if n_mapped < 190:
            raise RuntimeError(f"map coverage too low for {measure}: {n_mapped}")

    draw_age_sex_axes(axes_c, age_sex)
    top10 = draw_top10_axes(axes_d, country[country["year"] == 2050])

    # Row-band titles anchored just above each row's left subplot (no collision).
    fig.canvas.draw()
    band_texts = {
        0: "(a) Global burden attributable to high BMI, 23 diseases",
        1: "(b) National combined attributable fractions in 2050",
        2: "(c) Age- and sex-specific attributable burden in 2050",
        3: "(d) Countries with the largest attributable burden in 2050",
    }
    for row_idx, text in band_texts.items():
        pos = (axes_a, axes_b, axes_c, axes_d)[row_idx][0].get_position()
        fig.text(0.130, pos.y1 + 0.012, text, ha="left", va="bottom",
                 fontsize=7.6, fontweight="bold", color=BLACK)

    # Map colorbar centred below the map row.
    map_pos = axes_b[0].get_position()
    map_pos_r = axes_b[1].get_position()
    cb_width = 0.36
    cb_left = (map_pos.x0 + map_pos_r.x1) / 2 - cb_width / 2
    add_map_colorbar(fig, [cb_left, map_pos.y0 - 0.030, cb_width, 0.008])

    PANELS.mkdir(parents=True, exist_ok=True)
    stem = "Figure5_multidisease_attributable_burden_2025_2050_v1"
    pdf = PANELS / f"{stem}.pdf"
    tif = PANELS / f"{stem}.tif"
    prev = PANELS / f"{stem}_preview.png"
    fig.savefig(pdf, metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    fig.savefig(tif, dpi=TIFF_DPI, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(prev, dpi=DPI)
    plt.close(fig)

    # Source tables.
    TABLES.mkdir(parents=True, exist_ok=True)
    t_a = globe.groupby(["year", "measure"], as_index=False)[["total_burden", "attributable_burden"]].sum()
    t_a["af_pct"] = 100.0 * t_a["attributable_burden"] / t_a["total_burden"]
    t_a_path = TABLES / "Figure5A_global_trend_source_v1.csv"
    t_a.to_csv(t_a_path, index=False, encoding="utf-8-sig")
    t_b_path = TABLES / "Figure5B_country_combined_fraction_2050_source_v1.csv"
    cb.to_csv(t_b_path, index=False, encoding="utf-8-sig")
    t_c = (
        age_sex.groupby(["measure", "sex", "age_group"], as_index=False)["attributable_burden"].sum()
    )
    t_c_path = TABLES / "Figure5C_age_sex_2050_source_v1.csv"
    t_c.to_csv(t_c_path, index=False, encoding="utf-8-sig")
    t_d = (
        country[country["year"] == 2050]
        .groupby(["iso3", "crosswalk_location_name", "cause_group", "measure"], as_index=False)["attributable_burden"].sum()
    )
    t_d_path = TABLES / "Figure5D_country_top10_by_group_2050_source_v1.csv"
    t_d.to_csv(t_d_path, index=False, encoding="utf-8-sig")

    summary = {
        "run": "figure5_multidisease_burden_v1",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "scenario": "SSP2-4.5",
        "headline_causes": int(globe["cause_id"].nunique()),
        "trend_summary": trend_summary,
        "map_max_pct": MAP_MAX,
        "top10": top10,
        "outputs": [str(pdf), str(tif), str(prev)],
    }
    write_json(QA / "figure5_md_parameter_QA_v1.json", summary)
    (OUT / "scripts" / Path(__file__).name).parent.mkdir(parents=True, exist_ok=True)
    (OUT / "scripts" / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    outputs = [pdf, tif, prev, t_a_path, t_b_path, t_c_path, t_d_path,
               QA / "figure5_md_parameter_QA_v1.json", OUT / "scripts" / Path(__file__).name]
    pd.DataFrame({"path": [str(p) for p in outputs], "sha256": [sha256(p) for p in outputs]}).to_csv(
        OUT / "figure5_md_formal_manifest_v1.csv", index=False, encoding="utf-8-sig"
    )
    print("FIGURE5_MD_PASS")
    print(json.dumps(trend_summary, indent=2))


if __name__ == "__main__":
    main()
