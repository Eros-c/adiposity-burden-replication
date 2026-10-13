#!/usr/bin/env python3
"""Figure 3 panel A (stacked, absolute): high-BMI-attributable Deaths and
DALYs stacked by disease, 2015-2050, SSP2-4.5.

Historical segment 2015-2024: study PAF x GBD reference burden (same join as
the Phase-4 benchmark: burden age groups mapped to anchor ages 85+ then merged
with the historical PAF). Future segment 2025-2050: Phase-4 future product.
QA compares the recomputed 2015-2021 historical totals against the benchmark.

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

EXT = PROJECT / "11_all_disease_extension"
GLOBAL_YEAR = EXT / "04_burden" / "burden_benchmark_v1_20261005" / "data" / "attributable_future_global_year_four_ssp_v1.parquet"
TOTAL_REF = EXT / "04_burden" / "burden_benchmark_v1_20261005" / "data" / "total_burden_continuous_2015_2050_reference_v1.parquet"
BENCH = EXT / "04_burden" / "burden_benchmark_v1_20261005" / "data" / "benchmark_historical_country_year_detail_v1.parquet"
PAF_HIST = EXT / "03_paf" / "paf_all_causes_v1_20261005" / "data" / "paf_historical_2015_2024_all_causes_v1.parquet"

SCENARIO_CODE = "ssp245"
MEASURES = ("Deaths", "DALYs")
TIFF_DPI = 1800
GBD_AGE_TO_ANCHOR = {"85-89": "85+", "90-94": "85+", "95+": "85+"}
CAUSE_GROUPS = {
    441: "Cancer", 417: "Cancer", 453: "Cancer", 456: "Cancer", 429: "Cancer",
    435: "Cancer", 465: "Cancer", 471: "Cancer", 480: "Cancer", 485: "Cancer",
    486: "Cancer", 487: "Cancer",
    493: "Cardiovascular", 494: "Cardiovascular", 498: "Cardiovascular",
    500: "Cardiovascular", 502: "Cardiovascular",
    976: "Metabolic/renal", 998: "Metabolic/renal", 591: "Metabolic/renal",
    515: "Other", 534: "Other", 543: "Other",
}
HEADLINE = set(CAUSE_GROUPS.keys())

BLACK = "#17191A"
GROUP_ORDER = ["Cardiovascular", "Metabolic/renal", "Cancer", "Other"]
GROUP_HUES = {
    "Cardiovascular": ["#0E3A5D", "#1F628F", "#3D8FC2", "#77B5D9", "#B3D6EA"],
    "Metabolic/renal": ["#1E5B34", "#2E8B57", "#63B98A"],
    "Cancer": ["#641E16", "#78281F", "#922B21", "#A93226", "#B03A2E", "#C0392B",
               "#CD6155", "#D35400", "#DC7633", "#E67E22", "#EB984E", "#F0B27A"],
    "Other": ["#4A235A", "#7D3C98", "#B085C9"],
}
SHORT_NAME = {
    "Ischemic heart disease": "IHD",
    "Hypertensive heart disease": "HHD",
    "Diabetes mellitus type 2": "T2DM",
    "Stroke": "Stroke",
    "Chronic kidney disease due to hypertension": "CKD-HTN",
    "Chronic kidney disease due to diabetes mellitus type 2": "CKD-DM",
    "Alzheimer's disease and other dementias": "Alzheimer's",
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


def format_axis_number(value, _position=None):
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def load_data() -> tuple[pd.DataFrame, dict]:
    future = pd.read_parquet(GLOBAL_YEAR)
    future = future[(future["analysis_tier"] == "headline")
                    & (future["scenario_code"] == SCENARIO_CODE)].copy()
    name_of = future.drop_duplicates("cause_id").set_index("cause_id")["cause_name_en"].to_dict()

    total = pd.read_parquet(TOTAL_REF)
    total = total[total["year"] <= 2024 & total["cause_id"].isin(HEADLINE)] \
        if False else total[(total["year"] <= 2024) & (total["cause_id"].isin(HEADLINE))].copy()
    total["anchor_age_group"] = total["age_group"].replace(GBD_AGE_TO_ANCHOR)
    paf = pd.read_parquet(PAF_HIST)
    paf = paf[paf["year"] <= 2024][["cause_id", "iso3", "year", "sex", "anchor_age_group", "paf"]]
    merged = total.merge(paf, on=["cause_id", "iso3", "year", "sex", "anchor_age_group"],
                         how="left", validate="many_to_one")
    merged["paf"] = merged["paf"].fillna(0.0)
    merged["attributable_burden"] = merged["paf"] * merged["value"]
    hist = (merged.groupby(["year", "measure", "cause_id"], as_index=False)
            [["value", "attributable_burden"]].sum()
            .rename(columns={"value": "total_burden"}))
    hist["cause_name_en"] = hist["cause_id"].map(name_of)
    hist["cause_group"] = hist["cause_id"].map(CAUSE_GROUPS)

    # QA vs benchmark 2015-2021
    bench = pd.read_parquet(BENCH)
    bench = bench[bench["analysis_tier"] == "headline"]
    bsum = bench.groupby(["year", "measure"])["study_attributable"].sum()
    hsum = hist.groupby(["year", "measure"])["attributable_burden"].sum()
    qa = {}
    for key in bsum.index:
        qa[f"{key[0]}_{key[1]}"] = {
            "benchmark": float(bsum[key]), "recomputed": float(hsum.get(key, np.nan)),
            "rel_diff": float(abs(hsum.get(key, np.nan) - bsum[key]) / max(bsum[key], 1e-9)),
        }

    fut = future[["year", "measure", "cause_id", "cause_name_en", "total_burden",
                  "attributable_burden"]].copy()
    fut["cause_group"] = fut["cause_id"].map(CAUSE_GROUPS)
    cols = ["year", "measure", "cause_id", "cause_name_en", "cause_group",
            "total_burden", "attributable_burden"]
    frame = pd.concat([hist[cols], fut[cols]], ignore_index=True)
    frame = frame.sort_values(["measure", "cause_name_en", "year"]).reset_index(drop=True)
    return frame, qa


def assign_colors(causes_2050: pd.Series, name_group: dict) -> dict:
    out = {}
    for group in GROUP_ORDER:
        members = causes_2050[[name_group[c] == group for c in causes_2050.index]] \
            .sort_values(ascending=False)
        hues = GROUP_HUES[group]
        for i, cause in enumerate(members.index):
            out[cause] = hues[min(i, len(hues) - 1)]
    return out


def draw_panel(frame: pd.DataFrame):
    figure, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 76 / 25.4),
        gridspec_kw={"left": 0.075, "right": 0.985, "bottom": 0.16,
                     "top": 0.76, "wspace": 0.24})
    figure.text(0.035, 0.975,
                "(a) High-BMI-attributable burden by disease, 2015\u20132050",
                ha="left", va="top", fontsize=11, fontweight="bold", color=BLACK)
    figure.text(0.035, 0.905,
                "SSP2-4.5; adults aged 25 years and older; stacked by disease, "
                "coloured by disease group",
                ha="left", va="top", fontsize=9, color="#555D61")

    qa_summary = {}
    for axis, measure in zip(axes, MEASURES):
        data = frame[frame["measure"] == measure]
        order = (data[data["year"] == 2050].groupby("cause_name_en")["attributable_burden"]
                 .sum().sort_values(ascending=False))
        name_group = (data.drop_duplicates("cause_name_en")
                      .set_index("cause_name_en")["cause_group"].to_dict())
        colors = assign_colors(order, name_group)
        causes = list(order.index)
        years = np.sort(data["year"].unique())
        stack = np.vstack([data[data["cause_name_en"] == c]
                           .set_index("year").loc[years, "attributable_burden"].to_numpy(float)
                           for c in causes])
        axis.stackplot(years, stack,
                       colors=[colors[c] for c in causes],
                       edgecolor="white", linewidth=0.25, alpha=0.95)

        cumulative = np.cumsum(stack, axis=0)
        totals = cumulative[-1]
        v2050 = {c: stack[i, -1] for i, c in enumerate(causes)}
        centre = {c: cumulative[i, -1] - v2050[c] / 2.0 for i, c in enumerate(causes)}
        top = causes[:N_LABELS]
        for c in top:
            axis.annotate(f"{SHORT_NAME.get(c, c)} {format_axis_number(v2050[c])}",
                          (2050, centre[c]), xytext=(5, 0), textcoords="offset points",
                          fontsize=7, color=colors[c], va="center", ha="left", zorder=5)
        axis.annotate(f"{format_axis_number(totals[-1])} attributable",
                      (2050, totals[-1]), xytext=(5, 4), textcoords="offset points",
                      fontsize=7.5, color="#3F464A", va="bottom", ha="left", zorder=5,
                      fontweight="bold")

        axis.set_xlim(2014.5, 2061.5)
        axis.set_ylim(0, float(totals[-1]) * 1.14)
        axis.set_xticks([2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050])
        axis.yaxis.set_major_formatter(FuncFormatter(format_axis_number))
        axis.set_ylabel("Attributable burden", fontsize=10, labelpad=2)
        axis.set_xlabel("Year", fontsize=10, labelpad=2)
        axis.grid(axis="y", color="#E9ECEE", linewidth=0.4)
        axis.set_axisbelow(True)
        axis.set_title(measure, loc="left", fontsize=10, fontweight="bold", pad=2.5, color=BLACK)
        axis.tick_params(axis="both", labelsize=5.4, length=2.5, width=0.5, pad=1.5)
        qa_summary[measure] = {"total_2050": float(totals[-1]),
                               "total_2015": float(totals[0]),
                               "top6": {SHORT_NAME.get(c, c): float(v2050[c]) for c in top}}

    handles = [mpl.patches.Patch(facecolor=GROUP_HUES[g][1], edgecolor="none", label=g)
               for g in GROUP_ORDER]
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.868),
                  fontsize=9, ncol=4, handlelength=1.1, columnspacing=1.1,
                  borderaxespad=0.0)
    return figure, qa_summary


def main():
    for sub in (PANELS, TABLES, QA):
        sub.mkdir(parents=True, exist_ok=True)
    frame, bench_qa = load_data()
    frame.to_csv(TABLES / "fig3a_disease_stacked_attributable_burden_2015_2050_v1.csv",
                 index=False, encoding="utf-8-sig")
    figure, summary = draw_panel(frame)
    stem = PANELS / "Figure3_A_stacked_attributable_burden_by_disease_2015_2050_v1"
    figure.savefig(f"{stem}.pdf",
                   metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    figure.savefig(f"{stem}.tif", dpi=TIFF_DPI,
                   pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(QA / "Figure3_A_stacked_preview_v1.png", dpi=300)
    plt.close(figure)
    with open(QA / "figure3A_stacked_QA_v1.json", "w", encoding="utf-8") as fh:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "scenario": "SSP2-4.5", "n_headline_causes": 23,
                   "benchmark_check_2015_2021": bench_qa, **summary},
                  fh, ensure_ascii=False, indent=2)
    worst = max(v["rel_diff"] for v in bench_qa.values())
    print("benchmark max rel diff:", f"{worst:.2e}")
    print("saved", f"{stem}.tif")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
