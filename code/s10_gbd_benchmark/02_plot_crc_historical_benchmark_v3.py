#!/usr/bin/env python3
"""Create the v3 three-panel CRC historical benchmark from GBD 2021 v3 data."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


ROOT = Path(_cfg.RESULTS / "10_gbd_comparison")
V3 = ROOT / "crc_historical_benchmark_2015_2021_v3_gbd2021"
OUT = V3
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
QA = OUT / "qa"

SOURCE_V1 = V3 / "tables" / "crc_historical_benchmark_country_year_source_v3.csv"
METRICS_V1 = V3 / "qa" / "benchmark_metrics_v3.csv"

BLUE = "#2676A8"
ORANGE = "#D55E00"
GREY = "#777F83"
BLACK = "#17191A"
GRID = "#E9ECEE"
DPI = 600

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


def linear_metrics(frame: pd.DataFrame, measure: str) -> dict:
    subset = frame[frame["measure"].eq(measure)].copy()
    valid = np.isfinite(subset["gbd_derived_paf_pct"]) & np.isfinite(subset["study_derived_paf_pct"])
    subset = subset.loc[valid].copy()
    x = subset["gbd_derived_paf_pct"].to_numpy(float)
    y = subset["study_derived_paf_pct"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    r = float(np.corrcoef(x, y)[0, 1])
    r2 = float(1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))
    nonzero = x != 0
    relative_difference = 100.0 * (y[nonzero] - x[nonzero]) / x[nonzero]
    return {
        "panel": "a",
        "benchmark": "PAF-vs-PAF",
        "measure": measure,
        "scale": "linear percentage points",
        "N": int(len(subset)),
        "pearson_r": r,
        "R2": r2,
        "slope": float(slope),
        "intercept_percentage_points": float(intercept),
        "median_relative_difference_pct": float(np.median(relative_difference)),
        "median_absolute_difference_pp": float(np.median(np.abs(y - x))),
        "gbd_paf_min_pct": float(x.min()),
        "gbd_paf_max_pct": float(x.max()),
        "study_paf_min_pct": float(y.min()),
        "study_paf_max_pct": float(y.max()),
        "zero_gbd_paf_points": int((x == 0).sum()),
    }


def compact_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}k"
    return f"{value:.2f}"


def prepare_source() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE_V1)
    required = {
        "location_id", "iso3", "crosswalk_location_name", "year", "measure_id", "measure",
        "gbd_total_burden", "gbd_official_attributable", "study_derived_attributable",
        "positive_pair_for_log_plot",
    }
    if not required.issubset(source.columns):
        raise RuntimeError(f"v1 source is missing required columns: {sorted(required - set(source.columns))}")
    if source.duplicated(["location_id", "year", "measure_id"]).any():
        raise RuntimeError("Duplicate country-year-measure rows in frozen v1 source")
    if (source["gbd_total_burden"] <= 0).any():
        raise RuntimeError("Nonpositive GBD total burden prevents PAF derivation")

    source["gbd_derived_paf"] = source["gbd_official_attributable"] / source["gbd_total_burden"]
    source["study_derived_paf"] = source["study_derived_attributable"] / source["gbd_total_burden"]
    source["gbd_derived_paf_pct"] = 100.0 * source["gbd_derived_paf"]
    source["study_derived_paf_pct"] = 100.0 * source["study_derived_paf"]
    source["paf_difference_pp"] = source["study_derived_paf_pct"] - source["gbd_derived_paf_pct"]
    source["paf_relative_difference_pct"] = np.where(
        source["gbd_derived_paf_pct"].ne(0),
        100.0 * source["paf_difference_pp"] / source["gbd_derived_paf_pct"],
        np.nan,
    )
    source["positive_pair_for_log_plot"] = source["positive_pair_for_log_plot"].astype(str).str.lower().eq("true")

    paf_metrics = pd.DataFrame([linear_metrics(source, measure) for measure in ("Deaths", "DALYs")])
    burden_v1 = pd.read_csv(METRICS_V1)
    burden_metrics = pd.DataFrame({
        "panel": np.where(burden_v1["measure"].eq("Deaths"), "b", "c"),
        "benchmark": "Attributable burden",
        "measure": burden_v1["measure"],
        "scale": "log10 number",
        "N": burden_v1["positive_country_year_points_log_plot"].astype(int),
        "pearson_r": burden_v1["pearson_r_log10"],
        "R2": burden_v1["R2_log10"],
        "slope": burden_v1["slope_log10"],
        "intercept_log10": burden_v1["intercept_log10"],
        "MAE_number": burden_v1["MAE_number"],
        "median_relative_difference_pct": burden_v1["median_relative_error_pct"],
        "excluded_nonpositive_points": burden_v1["excluded_nonpositive_points_log_plot"].astype(int),
    })
    return source, paf_metrics, burden_metrics


def style_axis(axis: plt.Axes) -> None:
    axis.grid(which="major", color=GRID, linewidth=0.45, zorder=0)
    axis.tick_params(which="major", direction="out", labelsize=5.4, width=0.55, length=3, pad=1.4)
    axis.tick_params(which="minor", direction="out", width=0.40, length=1.8)


def draw_paf_panel(axis: plt.Axes, source: pd.DataFrame, metrics: pd.DataFrame) -> None:
    styles = {
        "Deaths": {"color": BLUE, "marker": "o"},
        "DALYs": {"color": ORANGE, "marker": "^"},
    }
    max_value = float(np.nanmax(source[["gbd_derived_paf_pct", "study_derived_paf_pct"]].to_numpy()))
    upper = max(25.0, math.ceil(max_value / 5.0) * 5.0)
    line_x = np.linspace(0, upper, 250)
    axis.plot([0, upper], [0, upper], color=GREY, lw=0.9, ls=(0, (3, 2.5)), zorder=1)

    for measure in ("Deaths", "DALYs"):
        panel = source[source["measure"].eq(measure)]
        row = metrics[metrics["measure"].eq(measure)].iloc[0]
        style = styles[measure]
        axis.scatter(
            panel["gbd_derived_paf_pct"], panel["study_derived_paf_pct"],
            s=8.5, marker=style["marker"], color=style["color"], alpha=0.20,
            linewidths=0, rasterized=True, zorder=2,
        )
        axis.plot(line_x, row["intercept_percentage_points"] + row["slope"] * line_x,
                  color=style["color"], lw=1.25, zorder=3)

    axis.set_xlim(0, upper)
    axis.set_ylim(0, upper)
    axis.set_box_aspect(1)
    ticks = np.arange(0, upper + 0.1, 5.0)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    percent = FuncFormatter(lambda value, _: f"{value:.0f}%")
    axis.xaxis.set_major_formatter(percent)
    axis.yaxis.set_major_formatter(percent)
    style_axis(axis)
    axis.set_xlabel("GBD-derived PAF", fontsize=5.8, labelpad=3)
    axis.set_ylabel("Study-derived PAF", fontsize=5.8, labelpad=3)
    axis.set_title("(a) PAF: study-derived vs GBD", loc="left", fontsize=7.2, fontweight="bold", pad=5, color=BLACK)

    handles = [
        Line2D([], [], marker="o", ms=4.2, color=BLUE, lw=1.2, label="Deaths"),
        Line2D([], [], marker="^", ms=4.2, color=ORANGE, lw=1.2, label="DALYs"),
        Line2D([], [], color=GREY, lw=0.9, ls=(0, (3, 2.5)), label="Identity"),
    ]
    axis.legend(handles=handles, loc="lower right", fontsize=4.8, handlelength=1.8, borderaxespad=0.6)

    death = metrics[metrics["measure"].eq("Deaths")].iloc[0]
    daly = metrics[metrics["measure"].eq("DALYs")].iloc[0]
    death_text = "\n".join([
        "Deaths",
        f"N = {int(death['N']):,}  |  r = {death['pearson_r']:.3f}  |  R² = {death['R2']:.3f}",
        f"Slope = {death['slope']:.3f}  |  intercept = {death['intercept_percentage_points']:.3f} pp",
        f"Median rel. difference = {death['median_relative_difference_pct']:.1f}%",
    ])
    daly_text = "\n".join([
        "DALYs",
        f"N = {int(daly['N']):,}  |  r = {daly['pearson_r']:.3f}  |  R² = {daly['R2']:.3f}",
        f"Slope = {daly['slope']:.3f}  |  intercept = {daly['intercept_percentage_points']:.3f} pp",
        f"Median rel. difference = {daly['median_relative_difference_pct']:.1f}%",
    ])
    axis.text(0.035, 0.955, death_text, transform=axis.transAxes, va="top", ha="left",
              fontsize=4.15, color=BLUE, linespacing=1.12)
    axis.text(0.035, 0.735, daly_text, transform=axis.transAxes, va="top", ha="left",
              fontsize=4.15, color=ORANGE, linespacing=1.12)


def draw_burden_panel(axis: plt.Axes, source: pd.DataFrame, metric: pd.Series, measure: str, letter: str) -> None:
    panel = source[source["measure"].eq(measure) & source["positive_pair_for_log_plot"]].copy()
    x = panel["gbd_official_attributable"].to_numpy(float)
    y = panel["study_derived_attributable"].to_numpy(float)
    color = BLUE if measure == "Deaths" else ORANGE
    lower_power = math.floor(math.log10(min(x.min(), y.min())))
    upper_power = math.ceil(math.log10(max(x.max(), y.max())))
    lower, upper = 10.0 ** lower_power, 10.0 ** upper_power
    axis.scatter(x, y, s=8.5, color=color, alpha=0.25, linewidths=0, rasterized=True, zorder=2)
    axis.plot([lower, upper], [lower, upper], color=GREY, lw=0.9, ls=(0, (3, 2.5)), zorder=1)
    fitted_x = np.logspace(lower_power, upper_power, 200)
    fitted_y = (10.0 ** float(metric["intercept_log10"])) * np.power(fitted_x, float(metric["slope"]))
    axis.plot(fitted_x, fitted_y, color=color, lw=1.25, zorder=3)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_box_aspect(1)
    axis.xaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_minor_formatter(NullFormatter())
    style_axis(axis)
    outcome_label = "deaths" if measure == "Deaths" else "DALYs"
    axis.set_xlabel(f"GBD attributable {outcome_label} (number)", fontsize=5.8, labelpad=3)
    axis.set_ylabel(f"Study-derived attributable {outcome_label} (number)", fontsize=5.8, labelpad=3)
    axis.set_title(f"({letter}) Attributable {outcome_label}: study-derived vs GBD", loc="left",
                   fontsize=7.2, fontweight="bold", pad=5, color=BLACK)
    annotation = "\n".join([
        f"N = {int(metric['N']):,}",
        f"Pearson r (log10) = {metric['pearson_r']:.3f}",
        f"R² (log10) = {metric['R2']:.3f}",
        f"Slope = {metric['slope']:.3f}",
        f"Intercept = {metric['intercept_log10']:.3f}",
        f"MAE = {compact_number(float(metric['MAE_number']))}",
    ])
    axis.text(0.035, 0.965, annotation, transform=axis.transAxes, va="top", ha="left",
              fontsize=4.75, color=BLACK, linespacing=1.18)


def render(source: pd.DataFrame, paf_metrics: pd.DataFrame, burden_metrics: pd.DataFrame) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(270 / 25.4, 90 / 25.4), constrained_layout=False)
    figure.subplots_adjust(left=0.050, right=0.990, bottom=0.145, top=0.755, wspace=0.30)
    figure.suptitle(
        "Study-derived high-BMI-attributable colorectal cancer burden compared with GBD estimates, 2015–2021",
        x=0.040, y=0.965, ha="left", va="top", fontsize=8.3, fontweight="bold", color=BLACK,
    )
    figure.text(
        0.040, 0.885,
        "Panel (a) directly compares attributable-fraction frameworks; panels (b,c) additionally reflect their shared GBD total CRC burden.",
        ha="left", va="top", fontsize=5.5, color="#555D61",
    )
    draw_paf_panel(axes[0], source, paf_metrics)
    for axis, measure, letter in zip(axes[1:], ("Deaths", "DALYs"), ("b", "c")):
        metric = burden_metrics[burden_metrics["measure"].eq(measure)].iloc[0]
        draw_burden_panel(axis, source, metric, measure, letter)

    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = "Supp_Fig_crc_high_BMI_PAF_and_burden_GBD_benchmark_2015_2021_v3"
    outputs = [FIGURES / f"{stem}.pdf", FIGURES / f"{stem}.png"]
    figure.savefig(outputs[0], metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    figure.savefig(outputs[1], dpi=DPI)
    plt.close(figure)
    return outputs


def render_only() -> int:
    """Redraw the existing v3 source and statistics without recomputing them."""
    source = pd.read_csv(TABLES / "crc_historical_benchmark_country_year_source_v3_plot.csv")
    source["positive_pair_for_log_plot"] = source["positive_pair_for_log_plot"].astype(str).str.lower().eq("true")
    metrics = pd.read_csv(QA / "benchmark_plot_metrics_v3.csv")
    paf_metrics = metrics[metrics["benchmark"].eq("PAF-vs-PAF")].copy()
    burden_metrics = metrics[metrics["benchmark"].eq("Attributable burden")].copy()
    outputs = render(source, paf_metrics, burden_metrics)
    print("RENDER_ONLY_PASS")
    for output in outputs:
        print(output)
    return 0


def main() -> int:
    for directory in (OUT, FIGURES, TABLES, QA):
        directory.mkdir(parents=True, exist_ok=True)
    status_path = QA / "status_v3.json"
    status_path.write_text(json.dumps({"status": "running"}, indent=2) + "\n", encoding="utf-8")

    source, paf_metrics, burden_metrics = prepare_source()
    source_path = TABLES / "crc_historical_benchmark_country_year_source_v3_plot.csv"
    metrics_path = QA / "benchmark_plot_metrics_v3.csv"
    qa_path = QA / "benchmark_QA_summary_v3.json"
    source.to_csv(source_path, index=False, encoding="utf-8-sig")
    combined_metrics = pd.concat([paf_metrics, burden_metrics], ignore_index=True, sort=False)
    combined_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig", na_rep="NA")
    outputs = render(source, paf_metrics, burden_metrics)

    original_metrics = pd.read_csv(METRICS_V1).sort_values("measure").reset_index(drop=True)
    copied = burden_metrics.sort_values("measure").reset_index(drop=True)
    preserved = (
        np.allclose(copied["pearson_r"], original_metrics["pearson_r_log10"], rtol=0, atol=0)
        and np.allclose(copied["R2"], original_metrics["R2_log10"], rtol=0, atol=0)
        and np.allclose(copied["slope"], original_metrics["slope_log10"], rtol=0, atol=0)
        and np.allclose(copied["intercept_log10"], original_metrics["intercept_log10"], rtol=0, atol=0)
        and np.allclose(copied["MAE_number"], original_metrics["MAE_number"], rtol=0, atol=0)
    )
    qa = {
        "status": "PASS" if preserved else "FAIL",
        "backend": "Python/matplotlib",
        "data_version": "GBD 2021 (supersedes v1/v2 built on GBD 2023)",
        "input_source_v3": str(SOURCE_V1),
        "input_source_v3_sha256": sha256(SOURCE_V1),
        "input_metrics_v3": str(METRICS_V1),
        "input_metrics_v3_sha256": sha256(METRICS_V1),
        "v3_burden_panel_statistics_preserved_exactly": bool(preserved),
        "production_results_recomputed_or_modified": False,
        "country_year_rows": int(len(source)),
        "country_years_per_measure": source.groupby("measure").size().astype(int).to_dict(),
        "paf_definition": {
            "GBD-derived": "GBD official high-BMI attributable burden / GBD total CRC burden",
            "Study-derived": "Study-derived attributable burden / the same GBD total CRC burden",
        },
        "paf_metrics": paf_metrics.to_dict(orient="records"),
        "burden_metrics_inherited_from_v3": burden_metrics.to_dict(orient="records"),
        "nonpositive_log_plot_handling": "Omitted only from burden log10 panels; retained in source CSV and PAF panel",
        "png_dpi": DPI,
    }
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not preserved:
        raise RuntimeError("v3 burden benchmark statistics were not preserved exactly")

    report_path = OUT / "benchmark_v3_report.md"
    report_path.write_text(
        "# CRC historical benchmark v3 (GBD 2021)\n\n"
        "## Revision\n\n"
        "Identical analysis to v2 but rebuilt on GBD 2021 downloads "
        "(11_all_disease_extension Phase 1 harmonized inputs). The v1/v2 runs used GBD 2023 "
        "exports and are superseded. Study-derived PAF is unchanged (NCD-RisC anchored; "
        "verified identical to v1 within 5e-12); only the GBD burden inputs changed.\n\n"
        "## Interpretation\n\n"
        "The PAF comparison is the more direct comparison of the two attributable-risk frameworks. "
        "The high correlations in the attributable-burden panels partly arise because both estimates share the same GBD total CRC burden. "
        "Accordingly, panels (b) and (c) represent benchmark or consistency comparisons rather than strong independent external validation.\n\n"
        "Nonpositive observations were omitted only from the log10 burden panels, following v1/v2, but remain in the source CSV and linear PAF panel.\n",
        encoding="utf-8",
    )

    output_rows = []
    for path in [*outputs, source_path, metrics_path, qa_path, report_path]:
        output_rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "nonempty": path.stat().st_size > 0})
    output_qa_path = QA / "output_file_QA_v3.csv"
    pd.DataFrame(output_rows).to_csv(output_qa_path, index=False, encoding="utf-8-sig")
    status = {
        "status": "complete",
        "paf_metrics": paf_metrics.to_dict(orient="records"),
        "v3_burden_results_preserved": True,
        "outputs": [str(path) for path in [*outputs, source_path, metrics_path, qa_path, report_path, output_qa_path]],
    }
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(render_only() if "--render-only" in sys.argv else main())
