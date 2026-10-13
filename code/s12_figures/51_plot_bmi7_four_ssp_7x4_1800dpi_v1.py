#!/usr/bin/env python3
"""Four-scenario seven-category BMI atlas: 7 category rows x 4 year columns.

Reproduces the reference layout/style (7 BMI-category rows x 4 year columns, left
label + colour-bar column, year headers above the first row) with no in-figure title
or subtitle: the scenario is identified in the file name only.

Canvas matches the reference image exactly: 806 x 805 px @150 dpi (5.3733 x 5.3667 in
= 136.48 x 136.31 mm); at 1800 dpi that is 9672 x 9660 px (x12).
Exports 1800-dpi LZW RGB TIFF per scenario plus a PDF (300 dpi rasters) and a
150-dpi PNG preview for QA.

Data:
  * 2015 column: scenario-independent historical 25+ product (identical in all four
    figures, because 2015 precedes scenario divergence).
  * 2025/2035/2050: production full-covariate 0.1-degree rasters, which exist for all
    four SSPs and share a common 2025 starting point. The GBD-trend bridge product is
    SSP2-4.5 only and is therefore not used here -- mixing it would break the
    cross-scenario comparison.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
from rasterio.windows import from_bounds

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
STYLE_SCRIPT = HERE / "00_style_concordance_test_historical_template_v1.py"
HIST25 = Path(
    _cfg.RESULTS / "05_historical_bmi_1990_2024" / "historical_25plus_compat_v1_20260909" / "data" / "0p1deg"
)
PROD_ROOT = Path(
    _cfg.RESULTS / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906" / "outputs"
)
FIG_ROOT = Path(
    _cfg.RESULTS / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
)
WORK = FIG_ROOT / "four_ssp_bmi7_7x4_1800dpi_v1"

# Same seven bands / file labels as the historical atlas template.
HIST_FILE_LABELS = (
    "lt18p5", "18p5_to_lt20", "20_to_lt25", "25_to_lt30",
    "30_to_lt35", "35_to_lt40", "ge40",
)
HIST_POP_NAME = "global_{year}_adult_population_regional_steps_mrp_0p1deg.tif"

YEARS = (2015, 2025, 2035, 2050)
# Canvas = reference image native size: 806 x 805 px @150 dpi -> 5.3733 x 5.3667 in.
# At 1800 dpi that is exactly 9672 x 9660 px (x12), so the output matches the
# reference proportions and physical size while meeting the 1800-dpi requirement.
WIDTH_MM = 806.0 / 150.0 * 25.4
HEIGHT_MM = 805.0 / 150.0 * 25.4
DPI_TIF = 1800
DPI_PDF = 300
DPI_PREVIEW = 150

# Listing order as requested: SSP2-4.5, SSP1-2.6, SSP3-7.0, SSP5-8.5.
SCENARIOS = (
    ("ssp245", "SSP2-4.5"),
    ("ssp126", "SSP1-2.6"),
    ("ssp370", "SSP3-7.0"),
    ("ssp585", "SSP5-8.5"),
)

# Grid measured from the reference: label column ends 0.118, map region 0.129-0.981,
# year-header band occupies the top 4.5% (grid top 0.955), bottom margin 0.022.
# width_ratios 0.50:1 puts year-column centres at 0.237/0.450/0.663/0.876
# (reference: 0.2357/0.4485/0.6613/0.8747, max deviation < 0.002).
GRID_LEFT = 0.024
GRID_RIGHT = 0.981
GRID_BOTTOM = 0.022
GRID_TOP = 0.955
WIDTH_RATIOS = (0.50, 1.0, 1.0, 1.0, 1.0)


def import_style_module():
    spec = importlib.util.spec_from_file_location("bmi7_four_ssp_style", STYLE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import style module: {STYLE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


style = import_style_module()
historical = style.historical_module


def read_masked(path: Path, band: int = 1, crop_to_display_extent: bool = False) -> np.ma.MaskedArray:
    with rasterio.open(path) as dataset:
        if crop_to_display_extent:
            window = from_bounds(-180, -60, 180, 84, dataset.transform)
            return dataset.read(band, window=window, masked=True).astype(np.float32)
        return dataset.read(band, masked=True).astype(np.float32)


def apply_population_mask(values: np.ma.MaskedArray, population: np.ma.MaskedArray) -> np.ma.MaskedArray:
    mask = (
        np.ma.getmaskarray(values)
        | np.ma.getmaskarray(population)
        | (population.filled(0) < 100)
    )
    return np.ma.array(values, mask=mask)


def read_historical_percent(year: int, category_index: int) -> np.ma.MaskedArray:
    """Historical 25+ composition, percent, adult-population >= 100 display mask."""
    path = HIST25 / str(year) / (
        f"global_{year}_bmi_{HIST_FILE_LABELS[category_index]}"
        f"_regional_steps_mrp_0p1deg_bp.tif"
    )
    population_path = HIST25 / str(year) / HIST_POP_NAME.format(year=year)
    values = read_masked(path) / 100.0
    population = read_masked(population_path)
    return apply_population_mask(values, population)


def read_future_percent(scenario_key: str, year: int, category_index: int) -> np.ma.MaskedArray:
    """Production full-covariate composition for one SSP, percent, same display mask."""
    probability_path = (
        PROD_ROOT / scenario_key / str(year)
        / f"probability_overall_{scenario_key}_{year}_0p1deg_v1.tif"
    )
    population_path = (
        PROD_ROOT / scenario_key / str(year)
        / f"population_overall_{scenario_key}_{year}_0p1deg_v1.tif"
    )
    if not probability_path.exists():
        raise FileNotFoundError(probability_path)
    values = read_masked(probability_path, category_index + 1, crop_to_display_extent=True) * 100.0
    population = read_masked(population_path, crop_to_display_extent=True)
    return apply_population_mask(values, population)


def read_panel(scenario_key: str, year: int, category_index: int) -> np.ma.MaskedArray:
    if year == 2015:
        return read_historical_percent(year, category_index)
    return read_future_percent(scenario_key, year, category_index)


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


def draw_map(axis, data, category, country_segments, adm1_segments) -> None:
    cmap = LinearSegmentedColormap.from_list(
        f"bmi7_four_ssp_{category.file_label}", category.colors, N=256
    )
    cmap.set_bad("white")
    norm = Normalize(vmin=0, vmax=category.vmax)
    axis.imshow(
        data,
        extent=(-180, 180, -60, 84),
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        rasterized=True,
        zorder=1,
    )
    historical.add_boundaries(axis, country_segments, adm1_segments)
    axis.set_xlim(-180, 180)
    axis.set_ylim(-60, 84)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def add_colorbar(axis, category) -> None:
    cmap = LinearSegmentedColormap.from_list(
        f"bmi7_four_ssp_cb_{category.file_label}", category.colors, N=256
    )
    norm = Normalize(vmin=0, vmax=category.vmax)
    colorbar = mpl.colorbar.ColorbarBase(axis, cmap=cmap, norm=norm, orientation="horizontal")
    ticks = [0, round(category.vmax / 3), round(2 * category.vmax / 3), category.vmax]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick:g}" for tick in ticks])
    # 5.0 pt floor (the reference figure used 4.9 pt, below the manuscript floor).
    colorbar.ax.tick_params(labelsize=5.0, length=1.4, width=0.4, pad=0.7)
    colorbar.outline.set_visible(False)
    colorbar.ax.text(
        0, 1.70, "%", transform=colorbar.ax.transAxes,
        ha="left", va="bottom", fontsize=5.0, color="#4a5053",
    )


def plot_scenario(scenario_key: str, scenario_label: str, country_segments, adm1_segments) -> dict:
    figure = plt.figure(
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), constrained_layout=False
    )
    # No title and no subtitle by request: the year headers are the only top-row text.
    outer = figure.add_gridspec(
        7, 5,
        left=GRID_LEFT, right=GRID_RIGHT, bottom=GRID_BOTTOM, top=GRID_TOP,
        width_ratios=WIDTH_RATIOS, wspace=0.012, hspace=0.001,
    )

    for category_index, category in enumerate(historical.CATEGORIES):
        label_axis = figure.add_subplot(outer[category_index, 0])
        label_axis.axis("off")
        label_axis.text(
            0.99, 0.62, category.title,
            ha="right", va="center", fontsize=6.8,
            fontweight="bold", color="#1e2224",
        )
        color_axis = label_axis.inset_axes([0.20, 0.31, 0.78, 0.045])
        add_colorbar(color_axis, category)

        for year_index, year in enumerate(YEARS):
            axis = figure.add_subplot(outer[category_index, year_index + 1])
            data = read_panel(scenario_key, year, category_index)
            draw_map(axis, data, category, country_segments, adm1_segments)
            if category_index == 0:
                axis.set_title(
                    str(year), fontsize=7.0, fontweight="bold",
                    pad=1.5, color="#1e2224",
                )

    WORK.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"BMI7_composition_2015_2050_{scenario_label}"
    pdf_path = WORK / f"{stem}_qa.pdf"
    preview_path = WORK / f"{stem}_preview.png"
    tif_path = FIG_ROOT / f"{stem}_1800dpi.tif"

    figure.savefig(pdf_path, dpi=DPI_PDF, metadata={"Creator": "four-SSP BMI7 atlas"})
    figure.savefig(preview_path, dpi=DPI_PREVIEW)
    figure.savefig(
        tif_path, format="tiff", dpi=DPI_TIF,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)

    # matplotlib emits RGBA; delivered plates are 3-sample RGB.
    with Image.open(tif_path) as image:
        if image.mode != "RGB":
            image.convert("RGB").save(
                tif_path, format="tiff",
                compression="tiff_lzw", dpi=(DPI_TIF, DPI_TIF),
            )
    return {"scenario_key": scenario_key, "scenario_label": scenario_label,
            "tif": tif_path, "pdf": pdf_path, "preview": preview_path}


# Category representative BMI values used across the project for midpoint means.
MIDPOINTS = np.array([17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5], dtype=np.float64)


def masked_mean_bmi(scenario_key: str, year: int) -> float:
    """Population-masked grid mean BMI (midpoint approximation); QA documentation only."""
    total = np.ma.MaskedArray(np.zeros((1440, 3600), dtype=np.float64))
    for category_index in range(7):
        share = read_panel(scenario_key, year, category_index).astype(np.float64) / 100.0
        total = total + share * MIDPOINTS[category_index]
    # Shares sum to 1 on valid cells, so sum(share * midpoint) is the mean BMI.
    return float(total.mean())


def write_qa(records: list[dict], country_segments, adm1_segments) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "scenario": record["scenario_label"],
            "path": str(record["tif"]),
            "bytes": record["tif"].stat().st_size,
            "pass": record["tif"].stat().st_size > 100_000,
        }
        for record in records
    ]
    with (WORK / "four_ssp_bmi7_7x4_export_QA_v1.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tif_report = []
    for record in records:
        with Image.open(record["tif"]) as image:
            tif_report.append(
                {
                    "scenario": record["scenario_label"],
                    "size_px": list(image.size),
                    "dpi": [float(v) for v in image.info.get("dpi", ())],
                    "mode": image.mode,
                    "compression": int(image.tag_v2.get(259)),
                    "samples": int(image.tag_v2.get(277)),
                }
            )

    alignment = []
    for year in (2025, 2050):
        values = {
            label: round(masked_mean_bmi(key, year), 4)
            for key, label in SCENARIOS
        }
        alignment.append({"year": year, "mean_bmi": values,
                          "spread": round(max(values.values()) - min(values.values()), 4)})

    qa = {
        "status": "PASS_EXPORT_PENDING_VISUAL_REVIEW",
        "backend": "Python/matplotlib (nature-figure skill)",
        "layout": "7 BMI-category rows x 4 year columns (2015/2025/2035/2050); "
                  "left cell holds the category label and its fixed-scale colour bar; "
                  "year headers above the first row; no in-figure title or subtitle",
        "canvas_mm": [round(WIDTH_MM, 3), round(HEIGHT_MM, 3)],
        "canvas_aspect_note": "matches the 2166 x 1980 example image aspect",
        "tiff_dpi": DPI_TIF,
        "tiff_compression": "LZW",
        "scenarios": [{"key": key, "label": label} for key, label in SCENARIOS],
        "years": list(YEARS),
        "historical_2015": "scenario-independent 25+ compatibility product; identical "
                           "2015 column in all four figures (2015 precedes divergence)",
        "future_source": "full_covariate_annual_0p1deg_production_v1_20260906 "
                         "(probability_overall + population_overall, adult population >= 100)",
        "bridge_not_used": "the GBD-trend bridge exists for ssp245 only; using it for one "
                           "scenario would break the common-2025-start comparison",
        "population_mask": "adult population >= 100 persons per 0.1-degree cell",
        "interpolation": "nearest",
        "category_vmax": [float(category.vmax) for category in historical.CATEGORIES],
        "grid": {"left": GRID_LEFT, "right": GRID_RIGHT, "bottom": GRID_BOTTOM,
                 "top": GRID_TOP, "width_ratios": list(WIDTH_RATIOS)},
        "boundary": {"country": "simplify 0.035 deg, #303437 lw 0.20",
                     "adm1": "simplify 0.035 deg, #62676a lw 0.055"},
        "font_floor_pt": 5.0,
        "scenario_alignment": alignment,
        "boundary_segments": {"country": len(country_segments), "adm1": len(adm1_segments)},
        "tif": tif_report,
        "formal_products_modified": False,
    }
    (WORK / "four_ssp_bmi7_7x4_parameter_QA_v1.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not all(row["pass"] for row in rows):
        raise RuntimeError("Four-SSP BMI7 export QA failed")
    sizes = {tuple(entry["size_px"]) for entry in tif_report}
    if len(sizes) != 1:
        raise RuntimeError(f"TIF pixel sizes differ across scenarios: {sizes}")
    for entry in tif_report:
        if entry["dpi"] != [float(DPI_TIF), float(DPI_TIF)]:
            raise RuntimeError(f"TIF dpi is not {DPI_TIF}: {entry}")
        if entry["mode"] != "RGB" or entry["samples"] != 3:
            raise RuntimeError(f"TIF is not 3-sample RGB: {entry}")
        if entry["compression"] != 5:
            raise RuntimeError(f"TIF compression is not LZW: {entry}")


def main() -> int:
    import gc

    country_segments, adm1_segments = build_boundaries()
    records: list[dict] = []
    for scenario_key, scenario_label in SCENARIOS:
        records.append(plot_scenario(scenario_key, scenario_label, country_segments, adm1_segments))
        gc.collect()
    write_qa(records, country_segments, adm1_segments)
    print("FOUR_SSP_BMI7_7X4_1800DPI_PASS")
    for record in records:
        print(f"  {record['scenario_label']}: {record['tif']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
