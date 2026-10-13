#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证图 7行 x 3列 重排
=====================
行 = 7 类 BMI 分类（<18.5 / 18.5–<20 / 20–<25 / 25–<30 / 30–<35 / 35–<40 / ≥40）；
列 = 三种验证：随机行五折、国家留出五折、前向时间外推（2015–2017/2018–2020/2021–2024 合并）。

每个子图：横轴=观测患病率(%), 纵轴=预测患病率(%)；
灰点=聚合单元（国家-地区 x 年份 x 性别，按样本量加权聚合），白点=按观测值分箱的加权均值，
蓝线=加权线性拟合，虚线=1:1。左上标注 加权 r / 加权 MAE(pp) / 拟合斜率。

数据：
  列(a)(b)：04_future_capable_mrp/full_covariate_mrp_refit_v1_20260905/
            model_fit/validation/all_validation_predictions_v1.csv.gz
  列(c)  ：05_historical_bmi_1990_2024/phase5_temporal_backtesting_v1/evaluation/
            Gate_D_paired_aggregated_cells_v1.csv.gz（model=future_capable_mrp）
输出：PDF + 1800 dpi LZW TIF。
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

ROOT = Path(_cfg.RESULTS / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909")
OUT = ROOT / "validation_bmi7_grid_v1"
PANELS = OUT / "panels"
QA = OUT / "qa"

SRC_5FOLD = (str(_cfg.RESULTS / "04_future_capable_mrp" / "full_covariate_mrp_refit_v1_20260905" / "model_fit" / "validation" / "all_validation_predictions_v1.csv.gz"))
SRC_FORWARD = (str(_cfg.RESULTS / "05_historical_bmi_1990_2024" / "phase5_temporal_backtesting_v1" / "evaluation" / "Gate_D_paired_aggregated_cells_v1.csv.gz"))

CATS = [("bmi_lt_18_5", "BMI <18.5"), ("bmi_18_5_20", "18.5\u2013<20"),
        ("bmi_20_25", "20\u2013<25"), ("bmi_25_30", "25\u2013<30"),
        ("bmi_30_35", "30\u2013<35"), ("bmi_35_40", "35\u2013<40"),
        ("bmi_ge_40", "BMI \u226540")]
COLS = [("internal_random_row", "(a) Random-row\nfivefold CV"),
        ("country_grouped", "(b) Country-held-out\nfivefold CV"),
        ("forward", "(c) Forward temporal\nholdout")]
TIFF_DPI = 1800
GREY, BLUE, BLACK = "#8C8C8C", "#1F4E8C", "#17191A"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def wcorr(x, y, w):
    m = w.sum(); mx = (x * w).sum() / m; my = (y * w).sum() / m
    return float((w * (x - mx) * (y - my)).sum() / m
                 / np.sqrt((w * (x - mx) ** 2).sum() / m * (w * (y - my) ** 2).sum() / m))


def wslope(x, y, w):
    m = w.sum(); mx = (x * w).sum() / m; my = (y * w).sum() / m
    return float((w * (x - mx) * (y - my)).sum() / (w * (x - mx) ** 2).sum())


def build_frames():
    """返回 {列键: DataFrame[obs_*, pred_*, w]}，统一为加权聚合单元。"""
    five = pd.read_csv(SRC_5FOLD)
    frames = {}
    for key, pattern in [("internal_random_row", "internal_random_row"),
                         ("country_grouped", "country_grouped")]:
        g = five[five["scheme"].str.contains(pattern)].copy()
        out = {}
        for cat, _ in CATS:
            tmp = g[["geography_id", "year", "sex", "pseudo_n",
                     f"observed_{cat}", f"predicted_{cat}"]].copy()
            agg = (tmp.groupby(["geography_id", "year", "sex"])
                   .apply(lambda d: pd.Series({
                       "obs": np.average(d[f"observed_{cat}"], weights=d["pseudo_n"]),
                       "pred": np.average(d[f"predicted_{cat}"], weights=d["pseudo_n"]),
                       "w": d["pseudo_n"].sum()}), include_groups=False)
                   .reset_index())
            out[cat] = agg
        frames[key] = out

    fwd = pd.read_csv(SRC_FORWARD)
    fwd = fwd[fwd["model"] == "future_capable_mrp"].copy()
    out = {}
    for cat, _ in CATS:
        tmp = fwd[["geography_id", "year", "sex", "pseudo_n",
                   f"observed_{cat}", f"predicted_{cat}"]].copy()
        agg = (tmp.groupby(["geography_id", "year", "sex"])
               .apply(lambda d: pd.Series({
                   "obs": np.average(d[f"observed_{cat}"], weights=d["pseudo_n"]),
                   "pred": np.average(d[f"predicted_{cat}"], weights=d["pseudo_n"]),
                   "w": d["pseudo_n"].sum()}), include_groups=False)
               .reset_index())
        out[cat] = agg
    frames["forward"] = out
    return frames


def draw_cell(ax, agg):
    """灰点 + 分箱均值白点 + 加权拟合线 + 1:1 虚线；返回统计量。"""
    x = agg["obs"].to_numpy(float) * 100.0
    y = agg["pred"].to_numpy(float) * 100.0
    w = agg["w"].to_numpy(float)
    ax.scatter(x, y, s=4, color=GREY, alpha=0.35, linewidths=0, zorder=2)
    # 分箱均值（按观测值 12 分位箱）
    edges = np.unique(np.quantile(x, np.linspace(0, 1, 13)))
    if len(edges) > 2:
        idx = np.digitize(x, edges[1:-1])
        bx, by = [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() > 0:
                bx.append(float(np.average(x[m], weights=w[m])))
                by.append(float(np.average(y[m], weights=w[m])))
        ax.scatter(bx, by, s=16, facecolor="white", edgecolor="#3F464A",
                   linewidth=0.6, zorder=4)
    lim = float(max(np.quantile(x, 0.995), np.quantile(y, 0.995))) * 1.12
    ax.plot([0, lim], [0, lim], ls="--", lw=0.8, color="#9AA3A7", zorder=3)
    sl = wslope(x, y, w); mx = np.average(x, weights=w); my = np.average(y, weights=w)
    xs = np.array([0.0, lim])
    ax.plot(xs, my + sl * (xs - mx), color=BLUE, lw=1.4, zorder=5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    r = wcorr(x, y, w)
    mae = float(np.average(np.abs(y - x), weights=w))
    ax.text(0.03, 0.97, f"r = {r:.2f}\nMAE = {mae:.1f} pp\nslope = {sl:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color=BLACK, linespacing=1.25)
    ax.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.2)
    return {"r": round(r, 3), "mae_pp": round(mae, 2), "slope": round(sl, 3),
            "cells": int(len(agg)), "n_eff": float(w.sum())}


def main():
    PANELS.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    frames = build_frames()

    fig = plt.figure(figsize=(180 / 25.4, 210 / 25.4))
    grid = fig.add_gridspec(7, 3, left=0.115, right=0.985, top=0.905,
                            bottom=0.075, wspace=0.16, hspace=0.30)
    stats = {}
    for j, (ckey, clabel) in enumerate(COLS):
        ax_top = fig.add_subplot(grid[0, j])
        ax_top.set_title(clabel, fontsize=9, fontweight="bold", color=BLACK, pad=6)
        for i, (cat, rlabel) in enumerate(CATS):
            ax = ax_top if i == 0 else fig.add_subplot(grid[i, j])
            st = draw_cell(ax, frames[ckey][cat])
            stats[f"{ckey}|{cat}"] = st
            if j == 0:
                ax.set_ylabel(rlabel, fontsize=8.5, fontweight="bold",
                              color=BLACK, labelpad=2)
            if j > 0:
                ax.set_yticklabels([])
    fig.text(0.028, 0.52, "Predicted BMI-category prevalence (%)",
             rotation=90, ha="center", va="center", fontsize=8.5, color=BLACK)
    fig.text(0.55, 0.018, "Observed BMI-category prevalence (%)",
             ha="center", va="center", fontsize=8.5, color=BLACK)

    stem = PANELS / "Validation_bmi7_categories_by_scheme_7x3_v1"
    fig.savefig(f"{stem}.pdf",
                metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    fig.savefig(f"{stem}.tif", dpi=TIFF_DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    (QA / "validation_7x3_stats_v1.json").write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
                    "unit": "geo x year x sex aggregated cells, pseudo_n weighted",
                    "stats": stats}, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [{"scheme_cat": k, **v} for k, v in stats.items()]
    pd.DataFrame(rows).to_csv(QA / "validation_7x3_stats_v1.csv", index=False,
                              encoding="utf-8-sig")
    print("saved", f"{stem}.tif / .pdf")


if __name__ == "__main__":
    main()
