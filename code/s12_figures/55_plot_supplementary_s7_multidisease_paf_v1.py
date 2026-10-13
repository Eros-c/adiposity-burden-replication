#!/usr/bin/env python3
"""Supplementary Figure S7 (multi-disease): spatial PAF for four representative diseases.

Layout (mirrors the CRC-only S7 page geometry, 183 x 170 mm, 3 x 6 grid):
  (a) CRC 0.1-degree PAF, 2050 (with Gulf+MENA / South Asia / Maritime SE Asia insets)
  (b) Ischemic heart disease PAF, 2050
  (c) Type 2 diabetes PAF, 2050
  (d) Alzheimer's disease and other dementias PAF, 2050
  (e) CRC male - female PAF difference, 2050
  (f) IHD scenario sensitivity (SSP5-8.5 minus SSP1-2.6), 2050
  (g) grid-cell PAF distribution by income group, four diseases
  (h) top-20 countries by combined 23-disease attributable fraction (Deaths, 2050),
      with per-disease national PAF markers

CRC rasters come from the frozen CRC PAF production; the other three diseases use
the 0.1-degree PAF grids computed in 11_all_disease_extension/03_paf
(paf_grid_multidisease_v1_20261005). National values come from Phase-4 burden
tables (GBD 2021). Each disease map uses its own colour scale; see caption.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import csv
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

PROJECT = Path(_cfg.RESULTS)
FIGURE_ROOT = PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
S7_MODULE_PATH = FIGURE_ROOT / "scripts" / "38_plot_supplementary_s7_crc_paf_multipanel_v1.py"
GRID_PAF_ROOT = PROJECT / "11_all_disease_extension" / "03_paf" / "paf_grid_multidisease_v1_20261005" / "data"
COUNTRY_OVERALL = (
    PROJECT / "11_all_disease_extension" / "04_burden" / "burden_benchmark_v1_20261005"
    / "data" / "attributable_future_country_overall_four_ssp_v1.parquet"
)

OUT = FIGURE_ROOT / "supplementary_s7_multidisease_paf_v1"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
QA = OUT / "qa"

DISEASES = {
    "CRC": {"label": "Colorectal cancer", "color": "#B8562F"},
    "IHD": {"label": "Ischaemic heart disease", "color": "#2676A8"},
    "T2DM": {"label": "Type 2 diabetes", "color": "#4E9A51"},
    "ALZ": {"label": "Alzheimer's & other dementias", "color": "#8A7CA8"},
}
DISEASE_ORDER = ["CRC", "IHD", "T2DM", "ALZ"]
SHORT_LEGEND = {"CRC": "CRC", "IHD": "IHD", "T2DM": "T2DM", "ALZ": "Alzheimer's"}
DPI = 1800

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_s7():
    spec = importlib.util.spec_from_file_location("s7_crc_module", S7_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(S7_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


s7 = import_s7()
historical = s7.historical


def read_disease_grid(tag: str, code: str, year: int):
    """Read a new-disease 0.1-degree PAF raster (fraction) -> percent masked array."""
    path = GRID_PAF_ROOT / tag / code / f"PAF_{tag}_{code}_{year}_0p1deg_overall_v1.tif"
    values, meta = s7.read_window(path, 1)
    pop, _ = s7.read_window(s7.population_path(code, year, "overall"), 1)
    paf = values * 100.0
    mask = (
        ~np.isfinite(paf.filled(np.nan))
        | np.ma.getmaskarray(pop)
        | (pop.filled(0) < s7.POPULATION_DISPLAY_THRESHOLD)
    )
    result = np.ma.array(paf, mask=mask)
    values_valid = result.compressed()
    if len(values_valid) == 0 or not np.isfinite(values_valid).all():
        raise RuntimeError(f"Invalid grid PAF: {tag} {code} {year}")
    return result, pop.filled(0).astype(np.float32), meta


def disease_norm(tag: str) -> tuple[Normalize, list[float]]:
    limits = {"CRC": 20.0, "IHD": 55.0, "T2DM": 85.0, "ALZ": 40.0}
    vmax = limits[tag]
    ticks = [0, vmax / 4, vmax / 2, 3 * vmax / 4, vmax]
    return Normalize(0, vmax), ticks


def plot_grouped_violins(axis, per_disease: dict, summaries: dict) -> None:
    income_groups = ["Low income", "Lower-middle income", "Upper-middle income", "High income"]
    centers = np.arange(1, len(income_groups) + 1)
    offsets = np.linspace(-0.27, 0.27, len(DISEASE_ORDER))
    ymax = 0.0
    for disease, offset in zip(DISEASE_ORDER, offsets):
        distributions = per_disease[disease][0]
        summary = summaries[disease]
        data = [distributions[g] for g in income_groups if g in distributions]
        positions = centers[: len(data)] + offset
        violin = axis.violinplot(data, positions=positions, widths=0.17,
                                 showmeans=False, showmedians=False, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(DISEASES[disease]["color"])
            body.set_edgecolor("none")
            body.set_alpha(0.55)
        groups_present = [g for g in income_groups if g in distributions]
        means = summary.set_index("income_group").loc[groups_present, "population_weighted_mean_paf_percent"].to_numpy(float)
        axis.scatter(positions, means, marker="D", s=8, color="white",
                     edgecolor=DISEASES[disease]["color"], linewidth=0.7, zorder=5)
        ymax = max(ymax, float(np.max([np.max(d) for d in data])))
    axis.set_xticks(centers)
    axis.set_xticklabels([g.replace("-", "-\n") for g in income_groups], fontsize=4.4)
    axis.set_ylim(0, max(95.0, ymax * 1.05))
    axis.set_ylabel("Grid-cell PAF (%)", fontsize=5.2, labelpad=1.5)
    axis.tick_params(direction="out", length=2.0, width=0.5, labelsize=4.6, pad=1.0)
    axis.grid(axis="y", color="#E5E9EA", linewidth=0.45)
    axis.set_axisbelow(True)
    handles = [Line2D([], [], marker="s", ms=4.4, color=DISEASES[d]["color"], lw=0,
                      label=SHORT_LEGEND[d]) for d in DISEASE_ORDER]
    # legend sits in the empty strip above the violins, with a white backing box
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.985),
                ncol=4, fontsize=4.4, handletextpad=0.35, columnspacing=1.0,
                borderaxespad=0.3, frameon=True, framealpha=0.95,
                edgecolor="none", facecolor="white")


def plot_top20(axis, top: pd.DataFrame, disease_paf: pd.DataFrame) -> None:
    ranked = top.sort_values("combined_af_pct", ascending=True).reset_index(drop=True)
    y = np.arange(len(ranked))
    axis.barh(y, ranked["combined_af_pct"], height=0.62, color="#C9CDD1",
              edgecolor="none", zorder=2, label="Combined AF")
    marker_styles = {"CRC": "D", "IHD": "o", "T2DM": "s", "ALZ": "^"}
    lookup = disease_paf.set_index("iso3")
    for disease in DISEASE_ORDER:
        vals = lookup.reindex(ranked["iso3"])[disease].to_numpy(float)
        axis.scatter(vals, y, marker=marker_styles[disease], s=13,
                     facecolor="white", edgecolor=DISEASES[disease]["color"],
                     linewidth=0.8, zorder=4, label=SHORT_LEGEND[disease])
    short_names = {
        "United Arab Emirates": "UAE", "Federated States of Micronesia": "Micronesia",
        "Cook Islands": "Cook Is.", "Solomon Islands": "Solomon Is.",
        "American Samoa": "Am. Samoa", "Saudi Arabia": "Saudi Arabia",
    }
    axis.set_yticks(y)
    axis.set_yticklabels(ranked["NAME_EN"].fillna(ranked["iso3"]).replace(short_names),
                         fontsize=4.4)
    axis.set_xlabel("National attributable fraction, Deaths, 2050 (%)", fontsize=5.2, labelpad=1.5)
    axis.tick_params(direction="out", length=2.0, width=0.5, labelsize=4.6, pad=1.0)
    axis.grid(axis="x", color="#E5E9EA", linewidth=0.45)
    axis.set_axisbelow(True)
    axis.set_xlim(0, 95.0)
    # legend in the empty band left of the T2DM marker column (~x>74% free above bars)
    axis.legend(loc="center left", bbox_to_anchor=(0.80, 0.42), fontsize=4.3,
                handletextpad=0.3, borderaxespad=0.4, labelspacing=0.42)


def main() -> int:
    # CRC rasters via the frozen production.
    crc_overall = {}
    crc_pop = {}
    for code, year in (("ssp245", 2025), ("ssp245", 2050)):
        value, pop, _ = s7.read_overall(code, year)
        crc_overall[(code, year)] = value
        crc_pop[(code, year)] = pop
    crc_sex = {}
    for sex_name in s7.SEXES:
        value, _, _ = s7.read_sex("ssp245", 2050, sex_name)
        crc_sex[sex_name] = value

    # New-disease rasters (2050, ssp245; IHD also both extreme scenarios).
    grids = {}
    pops = {}
    for tag in ("IHD", "T2DM", "ALZ"):
        grids[(tag, "ssp245", 2050)], pops[(tag, "ssp245", 2050)], _ = read_disease_grid(tag, "ssp245", 2050)
    for code in ("ssp126", "ssp585"):
        grids[("IHD", code, 2050)], _, _ = read_disease_grid("IHD", code, 2050)

    first = crc_overall[("ssp245", 2050)]
    segments, country_raster, lookup = s7.build_boundaries_and_country_raster(
        first.shape, s7.Affine(*s7.read_window(
            s7.paf_path("ssp245", 2050, "overall"), 1)[1]["window_transform"])
    )

    # Income-group distributions per disease (2050, ssp245).
    per_disease_dist = {}
    per_disease_summary = {}
    dist_inputs = {"CRC": (crc_overall[("ssp245", 2050)], crc_pop[("ssp245", 2050)])}
    for tag in ("IHD", "T2DM", "ALZ"):
        dist_inputs[tag] = (grids[(tag, "ssp245", 2050)], pops[(tag, "ssp245", 2050)])
    for disease, (grid, pop) in dist_inputs.items():
        values, summary = s7.income_group_distribution(grid, pop, country_raster, lookup)
        per_disease_dist[disease] = (values,)
        per_disease_summary[disease] = summary

    # National combined AF and per-disease PAF (Deaths, 2050, ssp245).
    country = pd.read_parquet(COUNTRY_OVERALL)
    country = country[(country["scenario_code"] == "ssp245") & (country["year"] == 2050)
                      & (country["measure"] == "Deaths") & (country["analysis_tier"] == "headline")]
    combined = country.groupby(["iso3", "crosswalk_location_name"], as_index=False)[
        ["total_burden", "attributable_burden"]].sum()
    combined["combined_af_pct"] = 100.0 * combined["attributable_burden"] / combined["total_burden"]
    names = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")[["ISO_A3", "ADM0_A3", "NAME_EN"]].copy()
    names["iso3"] = names["ISO_A3"].where(names["ISO_A3"].ne("-99"), names["ADM0_A3"]).astype(str).str.upper()
    names = names.drop_duplicates("iso3")
    combined = combined.merge(names[["iso3", "NAME_EN"]], on="iso3", how="left", validate="many_to_one")
    top = combined.sort_values("combined_af_pct", ascending=False).head(20).copy()

    cause_to_tag = {"Colon and rectum cancer": "CRC", "Ischemic heart disease": "IHD",
                    "Diabetes mellitus type 2": "T2DM",
                    "Alzheimer's disease and other dementias": "ALZ"}
    disease_paf = (
        country[country["cause_name_en"].isin(cause_to_tag)]
        .assign(tag=lambda d: d["cause_name_en"].map(cause_to_tag))
        .pivot_table(index="iso3", columns="tag", values="attributable_fraction", observed=True)
        .mul(100.0)
        .reset_index()
    )

    figure = plt.figure(figsize=(s7.WIDTH_MM / 25.4, s7.HEIGHT_MM / 25.4), constrained_layout=False)
    grid = figure.add_gridspec(3, 6, left=0.035, right=0.985, bottom=0.040, top=0.965,
                               height_ratios=(1.90, 0.85, 1.55), wspace=0.36, hspace=0.20)
    cmap_level = s7.map_cmap("level")
    cmap_diff = s7.map_cmap("difference")

    # (a) CRC with insets.
    norm_crc, ticks_crc = disease_norm("CRC")
    axis_a = figure.add_subplot(grid[0, :4])
    s7.draw_map(axis_a, crc_overall[("ssp245", 2050)], cmap_level, norm_crc, segments, anchor="N")
    s7.add_panel_header(axis_a, "a", "Colorectal cancer PAF", "SSP2-4.5 | 2050")
    s7.attach_insets(axis_a, crc_overall[("ssp245", 2050)], segments, cmap_level, norm_crc)

    # (g) income-group violins.
    axis_g = figure.add_subplot(grid[0, 4:])
    plot_grouped_violins(axis_g, per_disease_dist, per_disease_summary)
    s7.add_panel_header(axis_g, "g", "PAF distribution by income group", "2050 | overall")

    # (b)-(d): IHD, T2DM, ALZ maps.
    row2_axes = [figure.add_subplot(grid[1, 0:2]), figure.add_subplot(grid[1, 2:4]),
                 figure.add_subplot(grid[1, 4:])]
    norms = {}
    for axis, tag, letter in zip(row2_axes, ("IHD", "T2DM", "ALZ"), ("b", "c", "d")):
        norm, ticks = disease_norm(tag)
        norms[tag] = (norm, ticks)
        s7.draw_map(axis, grids[(tag, "ssp245", 2050)], cmap_level, norm, segments, anchor="N")
        s7.add_panel_header(axis, letter, f"{SHORT_LEGEND[tag]} PAF", "SSP2-4.5 | 2050")

    # (e) CRC male - female; (f) IHD scenario sensitivity. Centered vertically so
    # the map+colorbar block sits balanced in the cell instead of leaving a
    # bottom blank band.
    axis_e = figure.add_subplot(grid[2, 0:2])
    male_female = s7.diff_map(crc_sex["Male"], crc_sex["Female"])
    limit_e = s7.robust_diff_limit(male_female)
    s7.draw_map(axis_e, male_female, cmap_diff,
                TwoSlopeNorm(vmin=-limit_e, vcenter=0, vmax=limit_e), segments, anchor="C")
    s7.add_panel_header(axis_e, "e", "CRC male − female PAF", "pp | 2050")

    axis_f = figure.add_subplot(grid[2, 2:4])
    sensitivity = s7.diff_map(grids[("IHD", "ssp585", 2050)], grids[("IHD", "ssp126", 2050)])
    limit_f = s7.robust_diff_limit(sensitivity)
    s7.draw_map(axis_f, sensitivity, cmap_diff,
                TwoSlopeNorm(vmin=-limit_f, vcenter=0, vmax=limit_f), segments, anchor="C")
    s7.add_panel_header(axis_f, "f", "IHD scenario sensitivity", "SSP5-8.5 − SSP1-2.6 | 2050")

    axis_h = figure.add_subplot(grid[2, 4:])
    plot_top20(axis_h, top, disease_paf)
    s7.add_panel_header(axis_h, "h", "Top 20 countries by combined AF", "Deaths | 2050")

    # Colorbars.
    s7.add_panel_colorbar(figure, axis_a, cmap_level, norm_crc, "PAF (%)", ticks_crc)
    for axis, tag in zip(row2_axes, ("IHD", "T2DM", "ALZ")):
        norm, ticks = norms[tag]
        s7.add_panel_colorbar(figure, axis, cmap_level, norm, "PAF (%)", ticks)
    s7.add_panel_colorbar(figure, axis_e, cmap_diff,
                          TwoSlopeNorm(vmin=-limit_e, vcenter=0, vmax=limit_e),
                          "Difference (pp)", [-limit_e, 0, limit_e])
    s7.add_panel_colorbar(figure, axis_f, cmap_diff,
                          TwoSlopeNorm(vmin=-limit_f, vcenter=0, vmax=limit_f),
                          "Difference (pp)", [-limit_f, 0, limit_f])

    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    stem = "Supplementary_Figure_S7_multidisease_PAF_multipanel_v1"
    pdf = FIGURES / f"{stem}.pdf"
    tif = FIGURES / f"{stem}.tif"
    tmp_tif = FIGURES / f"{stem}.rebuild.tif"
    preview = QA / "S7_multidisease_preview_v1.png"
    figure.savefig(pdf, metadata={"Creator": "Python/matplotlib Nature-figure workflow"})
    figure.savefig(preview, dpi=400)
    if tmp_tif.exists():
        tmp_tif.unlink()
    figure.savefig(tmp_tif, dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    os.replace(tmp_tif, tif)

    TABLES.mkdir(parents=True, exist_ok=True)
    top_path = TABLES / "S7md_top20_country_combined_af_v1.csv"
    top.to_csv(top_path, index=False, encoding="utf-8-sig")
    disease_paf_path = TABLES / "S7md_national_disease_paf_2050_v1.csv"
    disease_paf.to_csv(disease_paf_path, index=False, encoding="utf-8-sig")
    summary_all = pd.concat(
        [summary.assign(disease=disease) for disease, summary in per_disease_summary.items()],
        ignore_index=True,
    )
    dist_path = TABLES / "S7md_income_group_distribution_v1.csv"
    summary_all.to_csv(dist_path, index=False, encoding="utf-8-sig")

    outputs = [pdf, tif, top_path, disease_paf_path, dist_path]
    rows = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in outputs]
    with (QA / "S7md_export_QA_v1.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(QA / "S7md_parameter_QA_v1.json", {
        "run": "supplementary_s7_multidisease_paf_v1",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "diseases": {k: v["label"] for k, v in DISEASES.items()},
        "crc_source": "frozen CRC 0.1-degree PAF production",
        "new_disease_source": "11_all_disease_extension/03_paf/paf_grid_multidisease_v1_20261005",
        "national_source": "Phase-4 country_overall (GBD 2021)",
        "per_disease_scales_pct": {t: disease_norm(t)[0].vmax for t in DISEASE_ORDER},
        "diff_limits_pp": {"crc_male_minus_female": limit_e, "ihd_ssp585_minus_ssp126": limit_f},
        "top20_countries": top["NAME_EN"].tolist(),
    })
    print("S7_MD_PASS", f"limits: CRC-FM {limit_e:.2f} pp, IHD scenario {limit_f:.2f} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
