#!/usr/bin/env python3
"""Supplementary figure: study-derived vs GBD high-BMI attributable benchmark, 23 causes.

Layout v5 (per user review):
  - Top block: 4 rows (disease groups) x 3 columns — (PAF scatter) /
    (attributable deaths, log-log) / (attributable DALYs, log-log). Each cell is
    one disease group, with per-cell fit lines and compact statistics.
  - Bottom block: (m) per-cause ranking agreement and (n) per-cause median
    relative difference; causes grouped and ordered as in the top block.
  - Log-log axes use data-driven bounds with a sensible floor (1e-2 deaths,
    1 DALY) so near-zero floating-point artefacts no longer create large empty
    regions.

Data: 11_all_disease_extension Phase 4 benchmark tables (GBD 2021).
Output: PDF + 1800-dpi TIF + preview.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = Path(_cfg.RESULTS / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909")
OUT = ROOT / "supplementary_multidisease_benchmark_v1"
PANELS = OUT / "panels"
TABLES = OUT / "tables"
QA = OUT / "qa"

BURDEN = Path(
    _cfg.RESULTS / "11_all_disease_extension" / "04_burden" / "burden_benchmark_v1_20261005"
)
DETAIL = BURDEN / "data" / "benchmark_historical_country_year_detail_v1.parquet"
METRICS = BURDEN / "tables" / "benchmark_metrics_per_cause_v1.csv"

CAUSE_GROUPS = {
    441: "Cancer", 417: "Cancer", 453: "Cancer", 456: "Cancer", 429: "Cancer",
    435: "Cancer", 465: "Cancer", 471: "Cancer", 480: "Cancer", 485: "Cancer",
    486: "Cancer", 487: "Cancer",
    493: "Cardiovascular", 494: "Cardiovascular", 498: "Cardiovascular",
    500: "Cardiovascular", 502: "Cardiovascular",
    976: "Metabolic/renal", 998: "Metabolic/renal", 591: "Metabolic/renal",
    515: "Other", 534: "Other", 543: "Other",
}
GROUP_COLORS = {
    "Cancer": "#B8562F",
    "Cardiovascular": "#2676A8",
    "Metabolic/renal": "#4E9A51",
    "Other": "#8A7CA8",
}
GROUP_ORDER = ["Cancer", "Cardiovascular", "Metabolic/renal", "Other"]
SHORT_NAMES = {
    "Ischemic heart disease": "Ischemic heart disease",
    "Hypertensive heart disease": "Hypertensive heart disease",
    "Diabetes mellitus type 2": "Type 2 diabetes",
    "Stroke": "Stroke",
    "Chronic kidney disease due to hypertension": "CKD due to hypertension",
    "Alzheimer's disease and other dementias": "Alzheimer's & other dementias",
    "Chronic kidney disease due to diabetes mellitus type 2": "CKD due to type 2 diabetes",
    "Atrial fibrillation and flutter": "Atrial fibrillation & flutter",
    "Liver cancer": "Liver cancer",
    "Breast cancer": "Breast cancer",
    "Colon and rectum cancer": "Colorectal cancer",
    "Gallbladder and biliary diseases": "Gallbladder & biliary diseases",
    "Pancreatic cancer": "Pancreatic cancer",
    "Kidney cancer": "Kidney cancer",
    "Uterine cancer": "Uterine cancer",
    "Asthma": "Asthma",
    "Gallbladder and biliary tract cancer": "Gallbladder & biliary tract cancer",
    "Peripheral artery disease": "Peripheral artery disease",
    "Leukemia": "Leukemia",
    "Multiple myeloma": "Multiple myeloma",
    "Non-Hodgkin lymphoma": "Non-Hodgkin lymphoma",
    "Ovarian cancer": "Ovarian cancer",
    "Thyroid cancer": "Thyroid cancer",
}
BLUE = "#2676A8"
ORANGE = "#D55E00"
GREY = "#777F83"
BLACK = "#17191A"
MEASURE_STYLES = {"Deaths": {"color": BLUE, "marker": "o"},
                  "DALYs": {"color": ORANGE, "marker": "^"}}
DEATHS_FLOOR = 1e-2   # attributable deaths below 0.01 are floating-point artefacts
DALYS_FLOOR = 1.0     # attributable DALYs below 1 are floating-point artefacts

DPI_TIF = 1800
DPI_PREVIEW = 300

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


def style_axis(ax: plt.Axes, y_row_grid: bool = False) -> None:
    ax.grid(which="major", axis="x", color="#E9ECEE", linewidth=0.45, zorder=0)
    ax.grid(which="major", axis="y",
            color="#F1F4F5" if y_row_grid else "#E9ECEE",
            linewidth=0.35 if y_row_grid else 0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(which="major", direction="out", labelsize=5.0, width=0.55, length=2.6, pad=1.2)
    ax.tick_params(which="minor", direction="out", width=0.40, length=1.6)


def log_bounds(x: np.ndarray, y: np.ndarray, floor: float) -> tuple[float, float, int]:
    vals = np.concatenate([x, y])
    n_below = int((vals <= floor).sum())
    vals = vals[vals > floor]
    lo = 10.0 ** math.floor(math.log10(vals.min()))
    hi = 10.0 ** math.ceil(math.log10(vals.max()))
    return lo, hi, n_below


def draw_paf_cell(ax: plt.Axes, sub: pd.DataFrame) -> dict:
    upper = 60.0
    ax.plot([0, upper], [0, upper], color=GREY, lw=0.8, ls=(0, (3, 2.5)), zorder=1)
    line_x = np.linspace(0, upper, 100)
    stats = {}
    for measure in ("Deaths", "DALYs"):
        part = sub[sub["measure"] == measure]
        x = (100.0 * part["official_paf"]).to_numpy(float)
        y = (100.0 * part["study_paf"]).to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        style = MEASURE_STYLES[measure]
        ax.scatter(x, y, s=3.2, marker=style["marker"], color=style["color"],
                   alpha=0.14, linewidths=0, rasterized=True, zorder=2)
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        ax.plot(line_x, intercept + slope * line_x, color=style["color"], lw=1.1, zorder=3)
        stats[measure] = {"N": int(len(x)), "r": r, "slope": float(slope)}
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_box_aspect(1)
    ticks = np.arange(0, upper + 0.1, 20.0)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    percent = FuncFormatter(lambda v, _: f"{v:.0f}%")
    ax.xaxis.set_major_formatter(percent)
    ax.yaxis.set_major_formatter(percent)
    style_axis(ax)
    ax.text(0.045, 0.955,
            f"Deaths r = {stats['Deaths']['r']:.3f}\nDALYs r = {stats['DALYs']['r']:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=4.5, color=BLACK,
            linespacing=1.2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.4), zorder=5)
    return stats


def draw_burden_cell(ax: plt.Axes, sub: pd.DataFrame, measure: str) -> dict:
    x = sub["gbd_official_attributable"].to_numpy(float)
    y = sub["study_attributable"].to_numpy(float)
    floor = DEATHS_FLOOR if measure == "Deaths" else DALYS_FLOOR
    keep = (x > floor) & (y > floor)
    x, y = x[keep], y[keep]
    n_below = int((~keep).sum())
    color = MEASURE_STYLES[measure]["color"]
    lo, hi, _ = log_bounds(x, y, floor)
    ax.scatter(x, y, s=3.2, color=color, alpha=0.16, linewidths=0, rasterized=True, zorder=2)
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls=(0, (3, 2.5)), zorder=1)
    lx, ly = np.log10(x), np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    r = float(np.corrcoef(lx, ly)[0, 1])
    fitted_x = np.logspace(math.log10(lo), math.log10(hi), 100)
    ax.plot(fitted_x, (10.0 ** intercept) * np.power(fitted_x, slope),
            color=color, lw=1.1, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis(ax)
    ax.text(0.045, 0.955, f"r = {r:.3f}\nN = {len(x):,}",
            transform=ax.transAxes, va="top", ha="left", fontsize=4.5, color=BLACK,
            linespacing=1.2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.4), zorder=5)
    return {"N": int(len(x)), "r_log10": r, "slope_log10": float(slope), "n_below_floor": n_below}


def main() -> None:
    detail = pd.read_parquet(DETAIL)
    detail = detail[detail["analysis_tier"] == "headline"].copy()
    detail["cause_group"] = detail["cause_id"].map(CAUSE_GROUPS)
    detail["cause_short"] = detail["cause_name_en"].map(SHORT_NAMES)

    metrics = pd.read_csv(METRICS)
    metrics = metrics[metrics["analysis_tier"] == "headline"].copy()
    metrics["cause_group"] = metrics["cause_id"].map(CAUSE_GROUPS)
    metrics["cause_short"] = metrics["cause_name_en"].map(SHORT_NAMES)

    # Grouped cause order (Cancer first; within group by deaths r, best on top).
    deaths_r = metrics[metrics["measure"] == "Deaths"].set_index("cause_short")["pearson_r_log10"]
    order: list[str] = []
    group_blocks: list[tuple[str, int, int]] = []
    for group in GROUP_ORDER:
        members = sorted(
            metrics.loc[metrics["cause_group"] == group, "cause_short"].unique(),
            key=lambda c: deaths_r[c], reverse=True,
        )
        start = len(order)
        order.extend(members)
        group_blocks.append((group, start, len(order) - 1))

    gap = 0.9
    positions: list[float] = []
    ycur = 0.0
    for idx in range(len(order)):
        if idx > 0 and any(idx - 1 == end for _, _, end in group_blocks[:-1]):
            ycur += gap
        positions.append(ycur)
        ycur += 1.0
    ymax = positions[-1]
    ymap = {cause: ymax - pos for cause, pos in zip(order, positions)}
    ylo, yhi = -0.8, ymax + 0.8

    cause_group_by_short = metrics.drop_duplicates("cause_short").set_index("cause_short")["cause_group"]
    mb = metrics.pivot_table(index="cause_short", columns="measure",
                             values="pearson_r_log10", observed=True).loc[order]
    mc = metrics.pivot_table(index="cause_short", columns="measure",
                             values="median_relative_difference", observed=True).loc[order]

    fig = plt.figure(figsize=(190.5 / 25.4, 250 / 25.4), constrained_layout=False)
    fig.suptitle(
        "Study-derived high-BMI-attributable burden compared with GBD estimates for 23 adiposity-associated diseases, 2015\u20132021",
        x=0.040, y=0.986, ha="left", va="top", fontsize=8.0, fontweight="bold", color=BLACK,
    )

    # ---- Top block: 4 groups x 3 columns ----
    cell_w, cell_h = 0.260, 0.118
    col_x = [0.085, 0.395, 0.705]
    gap_y = 0.032
    block_top = 0.935
    letters = iter("abcdefghijkl")
    col_titles = ["PAF: study-derived vs GBD", "Attributable deaths", "Attributable DALYs"]
    cell_stats: dict[str, dict] = {}
    for gi, group in enumerate(GROUP_ORDER):
        sub = detail[detail["cause_group"] == group]
        y0 = block_top - cell_h - gi * (cell_h + gap_y)
        for ci in range(3):
            ax = fig.add_axes([col_x[ci], y0, cell_w, cell_h])
            letter = next(letters)
            if ci == 0:
                cell_stats[f"{group}|PAF"] = draw_paf_cell(ax, sub)
                ax.set_ylabel("Study-derived PAF", fontsize=5.6, labelpad=3)
                if gi == 3:
                    ax.set_xlabel("GBD-derived PAF", fontsize=5.6, labelpad=3)
            else:
                measure = "Deaths" if ci == 1 else "DALYs"
                cell_stats[f"{group}|{measure}"] = draw_burden_cell(ax, sub, measure)
                if gi == 3:
                    ax.set_xlabel(
                        f"GBD attributable {'deaths' if measure == 'Deaths' else 'DALYs'} (number)",
                        fontsize=5.6, labelpad=3,
                    )
                ax.set_ylabel(
                    f"Study-derived {'deaths' if measure == 'Deaths' else 'DALYs'} (number)",
                    fontsize=5.6, labelpad=3,
                )
            if gi < 3:
                ax.set_xticklabels([])
            if gi == 0:
                ax.set_title(f"({letter}) {col_titles[ci]}", loc="left", fontsize=7.0,
                             fontweight="bold", pad=5, color=BLACK)
            else:
                ax.set_title(f"({letter})", loc="left", fontsize=7.0, fontweight="bold",
                             pad=2.5, color=BLACK)
        fig.text(0.018, y0 + cell_h / 2, f"{group} ({len(sub['cause_id'].unique())})",
                 rotation=90, ha="center", va="center", fontsize=5.6, fontweight="bold",
                 color=GROUP_COLORS[group])

    # shared legend for the top block
    handles = [
        Line2D([], [], marker="o", ms=4.0, color=BLUE, lw=1.1, label="Deaths"),
        Line2D([], [], marker="^", ms=4.0, color=ORANGE, lw=1.1, label="DALYs"),
        Line2D([], [], color=GREY, lw=0.8, ls=(0, (3, 2.5)), label="Identity"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.965),
               fontsize=5.0, handletextpad=0.4, borderaxespad=0.4, ncol=3,
               columnspacing=1.2)

    # ---- Bottom block: (m) ranking agreement, (n) level difference ----
    ax_d = fig.add_axes([0.300, 0.055, 0.200, 0.235])
    ax_e = fig.add_axes([0.545, 0.055, 0.440, 0.235])

    def dumbbell(ax: plt.Axes, frame: pd.DataFrame, scale: float) -> None:
        d = frame["Deaths"].to_numpy() * scale
        l = frame["DALYs"].to_numpy() * scale
        yy = np.array([ymap[c] for c in frame.index])
        for yi, vd, vl in zip(yy, d, l):
            ax.plot([vd, vl], [yi, yi], color="#C9CDD1", lw=0.8, zorder=2)
        ax.scatter(d, yy, s=13, marker="o", color=BLUE, zorder=3, label="Deaths")
        ax.scatter(l, yy, s=14, marker="^", color=ORANGE, zorder=3, label="DALYs")

    def group_bands(ax: plt.Axes) -> None:
        for gi, (group, start, end) in enumerate(group_blocks):
            if gi % 2 == 0:
                ax.axhspan(ymap[order[end]] - 0.55, ymap[order[start]] + 0.55,
                           color=GROUP_COLORS[group], alpha=0.055, zorder=0)

    def group_dividers(ax: plt.Axes) -> None:
        for _, start, end in group_blocks[:-1]:
            ysep = (ymap[order[end]] + ymap[order[end + 1]]) / 2.0
            ax.axhline(ysep, color="#D7DCDE", lw=0.5, zorder=0)

    dumbbell(ax_d, mb, 1.0)
    ax_d.axvline(0.95, color="#9AA3A7", lw=0.6, ls=(0, (3, 2.5)), zorder=1)
    group_bands(ax_d)
    group_dividers(ax_d)
    ax_d.set_yticks([ymap[c] for c in order])
    ax_d.set_yticklabels(order, fontsize=4.4)
    for cause in order:
        ax_d.scatter(0.8465, ymap[cause], marker="s", s=11, clip_on=False,
                     color=GROUP_COLORS[cause_group_by_short[cause]], zorder=4)
    ax_d.set_xlim(0.845, 1.002)
    ax_d.set_ylim(ylo, yhi)
    ax_d.set_xlabel("Pearson r (log10 scale)", fontsize=5.6, labelpad=3)
    style_axis(ax_d, y_row_grid=True)
    ax_d.set_title("(m) Ranking agreement by cause", loc="left", fontsize=7.0,
                   fontweight="bold", pad=5)
    ax_d.legend(loc="lower left", fontsize=4.8, handletextpad=0.3, borderaxespad=0.5)
    ax_d.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

    dumbbell(ax_e, mc, 100.0)
    ax_e.axvline(0.0, color="#3B4144", lw=0.7, zorder=1)
    group_bands(ax_e)
    group_dividers(ax_e)
    ax_e.set_yticks([ymap[c] for c in order])
    ax_e.set_yticklabels([])
    ax_e.set_ylim(ylo, yhi)
    ax_e.set_xlabel("Median relative difference, study vs GBD (%)", fontsize=5.6, labelpad=3)
    style_axis(ax_e, y_row_grid=True)
    ax_e.set_title("(n) Level difference by cause", loc="left", fontsize=7.0,
                   fontweight="bold", pad=5)
    ax_e.legend(loc="upper right", fontsize=4.8, handletextpad=0.3, borderaxespad=0.5)
    ax_e.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}"))

    PANELS.mkdir(parents=True, exist_ok=True)
    stem = "Supp_Fig_multidisease_high_BMI_benchmark_2015_2021_v5"
    pdf = PANELS / f"{stem}.pdf"
    tif = PANELS / f"{stem}.tif"
    prev = PANELS / f"{stem}_preview.png"
    fig.savefig(pdf, metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    fig.savefig(tif, dpi=DPI_TIF, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(prev, dpi=DPI_PREVIEW)
    plt.close(fig)

    metrics_out = metrics[["cause_id", "cause_short", "cause_group", "measure",
                           "positive_pairs", "pearson_r_log10", "median_relative_difference",
                           "study_total", "official_total"]]
    t_path = TABLES / "Supp_multidisease_benchmark_metrics_v1.csv"
    metrics_out.to_csv(t_path, index=False, encoding="utf-8-sig")

    summary = {
        "run": "supplementary_multidisease_benchmark_v5",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "layout": "top: 4 disease groups x (PAF, deaths, DALYs); bottom: (m) r + (n) level difference",
        "log_floor": {"deaths": DEATHS_FLOOR, "dalys": DALYS_FLOOR},
        "cell_stats": cell_stats,
        "outputs": [str(pdf), str(tif), str(prev), str(t_path)],
    }
    write_json(QA / "supp_benchmark_parameter_QA_v5.json", summary)
    (OUT / "scripts" / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    outputs = [pdf, tif, prev, t_path, QA / "supp_benchmark_parameter_QA_v5.json",
               OUT / "scripts" / Path(__file__).name]
    pd.DataFrame({"path": [str(p) for p in outputs], "sha256": [sha256(p) for p in outputs]}).to_csv(
        OUT / "supp_benchmark_formal_manifest_v5.csv", index=False, encoding="utf-8-sig"
    )
    n_below_total = sum(v.get("n_below_floor", 0) for v in cell_stats.values())
    print("SUPP_BENCHMARK_V5_PASS", f"points below log floor: {n_below_total}")


if __name__ == "__main__":
    main()
