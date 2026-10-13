#!/usr/bin/env python3
"""Figure 3 panel A (all-disease): high-BMI-attributable share of the total
23-disease burden, 2025-2050, SSP2-4.5.

Same visual language as the original CRC panel (Total vs BMI-attributable
trend lines for Deaths and DALYs), extended to the 23 headline diseases. The
attributable share (%) is labelled at both the 2025 start and the 2050 end of
the blue line so the share trend reads directly (e.g. 16.8% -> 23.2%).

Output: PDF + 1800-dpi LZW TIF in figure3_country_burden_v1/panels/.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
OUT = ROOT / "figure3_country_burden_v1"
PANELS = OUT / "panels"
TABLES = OUT / "tables"
QA = OUT / "qa"

BURDEN = PROJECT / "11_all_disease_extension" / "04_burden" / "burden_benchmark_v1_20261005" / "data"
GLOBAL_YEAR = BURDEN / "attributable_future_global_year_four_ssp_v1.parquet"

SCENARIO_CODE = "ssp245"
MEASURES = ("Deaths", "DALYs")
TIFF_DPI = 1800

BLUE = "#2676A8"
GREY = "#A7AEB2"
BLACK = "#17191A"

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


def format_axis_number(value, _position=None):
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def load_global_trends() -> pd.DataFrame:
    g = pd.read_parquet(GLOBAL_YEAR)
    g = g[(g["analysis_tier"] == "headline") & (g["scenario_code"] == SCENARIO_CODE)].copy()
    yearly = (g.groupby(["year", "measure"], as_index=False)[["total_burden", "attributable_burden"]]
              .sum())
    yearly["attributable_fraction"] = yearly["attributable_burden"] / yearly["total_burden"]
    return yearly.sort_values(["measure", "year"]).reset_index(drop=True)


def draw_panel(trend: pd.DataFrame):
    figure, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 70 / 25.4),
        gridspec_kw={"left": 0.085, "right": 0.985, "bottom": 0.18,
                     "top": 0.75, "wspace": 0.27})
    figure.text(0.035, 0.975, "(a) Global burden of 23 diseases attributable to high BMI",
                ha="left", va="top", fontsize=9.2, fontweight="bold", color=BLACK)
    figure.text(0.035, 0.895, "SSP2-4.5; adults aged 25 years and older",
                ha="left", va="top", fontsize=5.9, color="#555D61")

    summary = {}
    for axis, measure in zip(axes, MEASURES):
        data = trend[trend["measure"] == measure].sort_values("year")
        axis.plot(data["year"], data["total_burden"], color=GREY, linewidth=1.25, zorder=2)
        axis.plot(data["year"], data["attributable_burden"], color=BLUE, linewidth=1.6, zorder=3)
        axis.scatter([2050], [data.iloc[-1]["attributable_burden"]], s=18,
                     facecolor="white", edgecolor=BLUE, linewidth=0.9, zorder=4)

        total_last = float(data.iloc[-1]["total_burden"])
        attrib_last = float(data.iloc[-1]["attributable_burden"])
        share_first = float(data.iloc[0]["attributable_fraction"]) * 100.0
        share_last = float(data.iloc[-1]["attributable_fraction"]) * 100.0

        axis.annotate(f"{format_axis_number(total_last)} total",
                      (2050, total_last), xytext=(-3, 5), textcoords="offset points",
                      ha="right", va="bottom", fontsize=5.2, color="#747C80")
        axis.annotate(f"{format_axis_number(attrib_last)} attributable\n{share_last:.2f}% of total",
                      (2050, attrib_last), xytext=(-3, 3), textcoords="offset points",
                      ha="right", va="bottom", fontsize=5.2, color=BLUE, linespacing=1.12)
        axis.annotate(f"{share_first:.2f}%", (2025, float(data.iloc[0]["attributable_burden"])),
                      xytext=(4, 5), textcoords="offset points",
                      ha="left", va="bottom", fontsize=5.2, color=BLUE)

        axis.set_xlim(2024.5, 2051.0)
        axis.set_ylim(0, total_last * 1.12)
        axis.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
        axis.yaxis.set_major_formatter(FuncFormatter(format_axis_number))
        axis.grid(axis="y", color="#E9ECEE", linewidth=0.4)
        axis.set_title(measure, loc="left", fontsize=7.2, fontweight="bold", pad=2.5, color=BLACK)
        axis.set_xlabel("Year", fontsize=5.8, labelpad=2)
        axis.set_ylabel("Burden", fontsize=5.8, labelpad=2)
        axis.tick_params(axis="both", labelsize=5.4, length=2.5, width=0.5, pad=1.5)
        summary[measure] = {
            "total_2025": float(data.iloc[0]["total_burden"]),
            "total_2050": total_last,
            "attributable_2025": float(data.iloc[0]["attributable_burden"]),
            "attributable_2050": attrib_last,
            "share_2025_pct": share_first,
            "share_2050_pct": share_last,
        }

    handles = [
        mpl.lines.Line2D([], [], color=GREY, linewidth=1.25, label="Total burden, 23 diseases"),
        mpl.lines.Line2D([], [], color=BLUE, linewidth=1.6, label="BMI-attributable burden"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=5.2, ncol=1, handlelength=1.8)
    return figure, summary


def main():
    for sub in (PANELS, TABLES, QA):
        sub.mkdir(parents=True, exist_ok=True)
    trend = load_global_trends()
    trend.to_csv(TABLES / "fig3a_global_23disease_burden_trends_v1.csv",
                 index=False, encoding="utf-8-sig")
    figure, summary = draw_panel(trend)
    stem = PANELS / "Figure3_A_global_burden_23diseases_attributable_to_high_bmi_v1"
    figure.savefig(f"{stem}.pdf",
                   metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    figure.savefig(f"{stem}.tif", dpi=TIFF_DPI,
                   pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(QA / "Figure3_A_preview_v1.png", dpi=300)
    plt.close(figure)
    with open(QA / "figure3A_QA_v1.json", "w", encoding="utf-8") as fh:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "scenario": "SSP2-4.5", "n_headline_causes": 23, **summary},
                  fh, ensure_ascii=False, indent=2)
    print("saved", f"{stem}.tif")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
