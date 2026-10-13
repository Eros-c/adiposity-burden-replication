#!/usr/bin/env python3
"""Figure 3 panel A (per-disease composition): each disease's share of the
total high-BMI-attributable burden, 2025-2050, SSP2-4.5.

Two sub-panels (Deaths | DALYs). Each of the 23 headline diseases is a thin
line coloured by disease group; the top contributors at 2050 carry end labels.
This replaces the combined-share panel with the per-disease breakdown of the
attributable burden over time.

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

BLACK = "#17191A"
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
SHORT_NAME = {
    "Ischemic heart disease": "IHD",
    "Hypertensive heart disease": "HHD",
    "Diabetes mellitus type 2": "T2DM",
    "Stroke": "Stroke",
    "Chronic kidney disease due to hypertension": "CKD-HTN",
    "Chronic kidney disease due to diabetes mellitus type 2": "CKD-DM",
    "Alzheimer's disease and other dementias": "Alzheimer's",
    "Atrial fibrillation and flutter": "AF",
    "Liver cancer": "Liver cancer",
    "Gallbladder and biliary diseases": "Gallbladder diseases",
    "Breast cancer": "Breast cancer",
}
N_LABELS = 6

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


def load_composition() -> pd.DataFrame:
    g = pd.read_parquet(GLOBAL_YEAR)
    g = g[(g["analysis_tier"] == "headline") & (g["scenario_code"] == SCENARIO_CODE)].copy()
    g["cause_group"] = g["cause_id"].map(CAUSE_GROUPS)
    g["share_pct"] = g["attributable_burden"] / g.groupby(["measure", "year"])["attributable_burden"].transform("sum") * 100.0
    return g.sort_values(["measure", "cause_name_en", "year"]).reset_index(drop=True)


def draw_panel(comp: pd.DataFrame):
    figure, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 76 / 25.4),
        gridspec_kw={"left": 0.075, "right": 0.985, "bottom": 0.16,
                     "top": 0.76, "wspace": 0.24})
    figure.text(0.035, 0.975,
                "(a) Contribution of each disease to the high-BMI-attributable burden, 2025\u20132050",
                ha="left", va="top", fontsize=9.2, fontweight="bold", color=BLACK)
    figure.text(0.035, 0.905,
                "SSP2-4.5; adults aged 25 years and older; share of total attributable "
                "burden across the 23 diseases",
                ha="left", va="top", fontsize=5.9, color="#555D61")

    for axis, measure in zip(axes, MEASURES):
        data = comp[comp["measure"] == measure]
        # draw smaller contributors first, top ones last (on top)
        last = (data[data["year"] == 2050]
                .groupby("cause_name_en")["share_pct"].sum()
                .sort_values())
        for name in last.index:
            part = data[data["cause_name_en"] == name].sort_values("year")
            group = part["cause_group"].iloc[0]
            axis.plot(part["year"], part["share_pct"], color=GROUP_COLORS[group],
                      linewidth=0.9, alpha=0.9, zorder=3)

        # end labels for the top N_LABELS diseases at 2050, spread vertically
        top = last.tail(N_LABELS).iloc[::-1]
        values = top.to_dict()
        order = sorted(values, key=lambda c: -values[c])
        offsets = {order[i]: off for i, off in enumerate([8, 2, -4, -10, -16, -22])}
        for name in order:
            axis.annotate(f"{SHORT_NAME.get(name, name)} {values[name]:.1f}%",
                          (2050, values[name]),
                          xytext=(5, offsets[name]), textcoords="offset points",
                          fontsize=5.2, color=GROUP_COLORS[data[data["cause_name_en"] == name]["cause_group"].iloc[0]],
                          va="center", ha="left", zorder=4)

        axis.set_xlim(2024.5, 2056.5)
        axis.set_ylim(0, 36)
        axis.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
        axis.set_ylabel("Share of attributable burden (%)", fontsize=5.8, labelpad=2)
        axis.set_xlabel("Year", fontsize=5.8, labelpad=2)
        axis.grid(axis="y", color="#E9ECEE", linewidth=0.4)
        axis.set_axisbelow(True)
        axis.set_title(measure, loc="left", fontsize=7.2, fontweight="bold", pad=2.5, color=BLACK)
        axis.tick_params(axis="both", labelsize=5.4, length=2.5, width=0.5, pad=1.5)

    handles = [mpl.lines.Line2D([], [], color=GROUP_COLORS[g], linewidth=1.6, label=g)
               for g in GROUP_ORDER]
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.868),
                  fontsize=5.6, ncol=4, handlelength=1.5, columnspacing=1.1,
                  borderaxespad=0.0)
    return figure


def main():
    for sub in (PANELS, TABLES, QA):
        sub.mkdir(parents=True, exist_ok=True)
    comp = load_composition()
    comp.to_csv(TABLES / "fig3a_disease_composition_trends_v1.csv",
                index=False, encoding="utf-8-sig")
    figure = draw_panel(comp)
    stem = PANELS / "Figure3_A_disease_composition_attributable_burden_2025_2050_v1"
    figure.savefig(f"{stem}.pdf",
                   metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    figure.savefig(f"{stem}.tif", dpi=TIFF_DPI,
                   pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(QA / "Figure3_A_composition_preview_v1.png", dpi=300)
    plt.close(figure)

    qa = {"generated_at": datetime.now(timezone.utc).isoformat(),
          "scenario": "SSP2-4.5", "n_headline_causes": 23,
          "top6_2050": {}}
    for m in MEASURES:
        d = comp[(comp["measure"] == m) & (comp["year"] == 2050)]
        tot = d["attributable_burden"].sum()
        top = d.nlargest(6, "share_pct")[["cause_name_en", "share_pct"]]
        qa["top6_2050"][m] = [(r["cause_name_en"], round(float(r["share_pct"]), 2)) for _, r in top.iterrows()]
    with open(QA / "figure3A_composition_QA_v1.json", "w", encoding="utf-8") as fh:
        json.dump(qa, fh, ensure_ascii=False, indent=2)
    print("saved", f"{stem}.tif")
    print(json.dumps(qa["top6_2050"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
