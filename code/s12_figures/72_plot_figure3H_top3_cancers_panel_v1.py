#!/usr/bin/env python3
"""Figure 3 small panel: top-3 diseases (all causes, by attributable deaths).

Teacher's decision (2026-10-07 meeting): all-disease burden stays in the main
figure; only the top-3 typical cancers are shown as a small panel / supplement.
Top-3 diseases by global attributable deaths in 2050: ischemic heart disease,
hypertensive heart disease, diabetes mellitus type 2.

Panel layout: three mini trend axes side by side, global high-BMI-attributable
deaths 2015-2050 (SSP2-4.5), each with the 2050 value and the 2025-2050
fold-change annotated (level + speed in one glance).

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
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
OUT = ROOT / "figure3_country_burden_v1"
PANELS = OUT / "panels"
TABLES = OUT / "tables"
QA = OUT / "qa"

SRC = TABLES / "fig3a_disease_stacked_attributable_burden_2015_2050_v1.csv"

TOP3 = ["Ischemic heart disease", "Hypertensive heart disease", "Diabetes mellitus type 2"]
SHORT = {"Ischemic heart disease": "Ischemic\nheart disease",
         "Hypertensive heart disease": "Hypertensive\nheart disease",
         "Diabetes mellitus type 2": "Diabetes mellitus\ntype 2"}
TIFF_DPI = 1800
BLUE = "#2676A8"
FILL = "#B7D3E6"
BLACK = "#17191A"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def fmt_count(value, _position=None):
    a = abs(value)
    if a >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if a >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def main():
    for sub in (PANELS, TABLES, QA):
        sub.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SRC)
    frame = frame[(frame["measure"] == "Deaths") & (frame["cause_name_en"].isin(TOP3))]

    fig = plt.figure(figsize=(132 / 25.4, 52 / 25.4), constrained_layout=False)
    fig.text(0.02, 0.965, "Top 3 diseases by attributable deaths",
             ha="left", va="top", fontsize=11, fontweight="bold", color=BLACK)
    fig.text(0.985, 0.965, "SSP2-4.5", ha="right", va="top",
             fontsize=9, color="#596064", fontweight="bold")
    fig.text(0.02, 0.865, "High-BMI-attributable deaths, 2015\u20132050",
             ha="left", va="top", fontsize=9, color="#555D61")

    summary = {}
    for i, disease in enumerate(TOP3):
        ax = fig.add_axes([0.105 + i * 0.305, 0.22, 0.255, 0.47])
        part = frame[frame["cause_name_en"] == disease].sort_values("year")
        x = part["year"].to_numpy()
        y = part["attributable_burden"].to_numpy(float)
        ax.fill_between(x, 0, y, color=FILL, alpha=0.85, linewidth=0)
        ax.plot(x, y, color=BLUE, linewidth=1.4, zorder=3)
        ax.scatter([2050], [y[-1]], s=14, facecolor="white", edgecolor=BLUE,
                   linewidth=0.9, zorder=4)

        v25 = float(part[part["year"] == 2025]["attributable_burden"].iloc[0])
        v50 = float(y[-1])
        fold = v50 / v25 if v25 > 0 else float("nan")
        ax.annotate(f"{fmt_count(v50)}", (2050, v50), xytext=(-2, 6),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=8.5, color=BLUE, fontweight="bold")
        ax.annotate(f"\u00d7{fold:.1f}", (2050, v50 * 0.35), xytext=(-2, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=8, color="#3F464A")

        ax.set_xlim(2014.5, 2051.5)
        ax.set_ylim(0, float(y.max()) * 1.30)
        ax.set_xticks([2015, 2030, 2050])
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_count))
        ax.set_title(SHORT[disease], loc="left", fontsize=9,
                     fontweight="bold", pad=4, color=BLACK, linespacing=1.05)
        ax.tick_params(axis="both", labelsize=8, length=2.5, width=0.55, pad=1.5)
        ax.grid(axis="y", color="#E9ECEE", linewidth=0.4)
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_ylabel("Attributable deaths", fontsize=9, labelpad=2)
        summary[disease] = {"deaths_2015": float(part[part.year == 2015]["attributable_burden"].iloc[0]),
                            "deaths_2025": v25, "deaths_2050": v50, "fold_2025_2050": fold}

    stem = PANELS / "Figure3_H_top3_diseases_attributable_deaths_2015_2050_v1"
    fig.savefig(f"{stem}.pdf",
                metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    fig.savefig(f"{stem}.tif", dpi=TIFF_DPI, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(QA / "Figure3_H_top3_diseases_preview_v1.png", dpi=300)
    plt.close(fig)
    (QA / "figure3H_top3_cancers_QA_v1.json").write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
                    "scenario": "SSP2-4.5", "metric": "attributable deaths",
                    "selection": "top-3 diseases (all causes) by attributable deaths 2050",
                    **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", f"{stem}.tif")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
