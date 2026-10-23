"""Test renderer for Figure 1B using the accepted historical map template."""

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
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent
STYLE_SCRIPT = HERE / "00_style_concordance_test_historical_template_v1.py"
HISTORICAL_25 = Path(_cfg.RESULTS / "05_historical_bmi_1990_2024" / "historical_25plus_compat_v1_20260909" / "data" / "0p1deg")
BRIDGE_ROOT = Path(_cfg.RESULTS / "06_future_bmi_2025_2050" / "bridge_ncd2024_gbdtrend_sensitivity_v1_20260909" / "outputs" / "ssp245")
FUTURE_POPULATION_ROOT = Path(_cfg.RESULTS / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906" / "outputs" / "ssp245")
OUT = Path(_cfg.RESULTS / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909" / "test_figure1_AB_v2")
WORK = Path(_cfg.ROOT / "phase7_r_inputs" / "test_figure1_AB_v2")
YEARS = (2015, 2025, 2035, 2050)
HIST_FILE_LABELS = ("lt18p5", "18p5_to_lt20", "20_to_lt25", "25_to_lt30", "30_to_lt35", "35_to_lt40", "ge40")
POPULATION_NAME = "global_{year}_adult_population_regional_steps_mrp_0p1deg.tif"
DPI = 600

def import_style():
    spec = importlib.util.spec_from_file_location("figure1b_accepted_historical_style", STYLE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(STYLE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

style = import_style()
historical = style.historical_module

def read_masked(path: Path, band: int = 1, crop_to_display_extent: bool = False) -> np.ma.MaskedArray:
    with rasterio.open(path) as dataset:
        if crop_to_display_extent:
            window = from_bounds(-180, -60, 180, 84, dataset.transform)
            return dataset.read(band, window=window, masked=True).astype(np.float32)
        return dataset.read(band, masked=True).astype(np.float32)

def apply_population_mask(probability: np.ma.MaskedArray, population_path: Path) -> np.ma.MaskedArray:
    if not population_path.exists():
        raise FileNotFoundError(population_path)
    population = read_masked(population_path)
    mask = np.ma.getmaskarray(probability) | np.ma.getmaskarray(population) | (population.filled(0) < 100)
    return np.ma.array(probability, mask=mask)

def read_historical_percent(year: int, category_index: int) -> np.ma.MaskedArray:
    path = HISTORICAL_25 / str(year) / f"global_{year}_bmi_{HIST_FILE_LABELS[category_index]}_regional_steps_mrp_0p1deg_bp.tif"
    data = read_masked(path) / 100.0
    population_path = HISTORICAL_25 / str(year) / POPULATION_NAME.format(year=year)
    return apply_population_mask(data, population_path)

def read_future_percent(year: int, category_index: int) -> np.ma.MaskedArray:
    probability_path = BRIDGE_ROOT / str(year) / f"probability_overall_ssp245_{year}_0p1deg_bridge_v2.tif"
    population_path = FUTURE_POPULATION_ROOT / str(year) / f"population_overall_ssp245_{year}_0p1deg_v1.tif"
    data = read_masked(probability_path, category_index + 1, crop_to_display_extent=True) * 100.0
    population = read_masked(population_path, crop_to_display_extent=True)
    mask = np.ma.getmaskarray(data) | np.ma.getmaskarray(population) | (population.filled(0) < 100)
    return np.ma.array(data, mask=mask)

def build_country_segments() -> list[np.ndarray]:
    countries = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")
    countries = countries[countries.geometry.notna()].copy()
    countries.geometry = countries.geometry.simplify(0.035, preserve_topology=True)
    return historical.geometry_segments(countries)

def draw_map(axis: plt.Axes, data: np.ma.MaskedArray, category, country_segments: list[np.ndarray], interpolation: str) -> None:
    cmap = LinearSegmentedColormap.from_list(f"fig1b_{category.file_label}", category.colors, N=256)
    cmap.set_bad("white")
    norm = Normalize(vmin=0, vmax=category.vmax)
    axis.imshow(data, extent=(-180, 180, -60, 84), origin="upper", cmap=cmap, norm=norm, interpolation=interpolation, rasterized=True, zorder=1)
    historical.add_boundaries(axis, country_segments, None)
    axis.set_xlim(-180, 180)
    axis.set_ylim(-60, 84)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

def add_colorbar(axis: plt.Axes, category) -> None:
    cmap = LinearSegmentedColormap.from_list(f"fig1b_cb_{category.file_label}", category.colors, N=256)
    cmap.set_bad("white")
    norm = Normalize(vmin=0, vmax=category.vmax)
    colorbar = mpl.colorbar.ColorbarBase(axis, cmap=cmap, norm=norm, orientation="horizontal")
    ticks = np.asarray([0, round(category.vmax / 3), round(2 * category.vmax / 3), category.vmax], dtype=float)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick:g}" for tick in ticks])
    colorbar.ax.tick_params(labelsize=5.0, length=1.5, width=0.4, pad=0.8)
    colorbar.outline.set_visible(False)
    colorbar.ax.text(0, 1.72, "%", transform=colorbar.ax.transAxes, ha="left", va="bottom", fontsize=5.0, color="#4a5053")

def plot(country_segments: list[np.ndarray], interpolation: str, stem: str) -> list[Path]:
    figure = plt.figure(figsize=(183 / 25.4, 78 / 25.4), constrained_layout=False)
    figure.text(0.065, 0.982, "(b) Global seven-category BMI composition", ha="left", va="top", fontsize=10.4, fontweight="bold", color="#151718")
    figure.text(0.065, 0.925, "Adults aged 25 years and older; 0.1 degree grid", ha="left", va="top", fontsize=6.7, color="#50575b")
    outer = figure.add_gridspec(5, 7, left=0.065, right=0.998, bottom=0.085, top=0.820, height_ratios=(1.0, 1.0, 1.0, 1.0, 0.11), wspace=0.055, hspace=0.065)

    for category_index, category in enumerate(historical.CATEGORIES):
        for row_index, year in enumerate(YEARS):
            axis = figure.add_subplot(outer[row_index, category_index])
            data = read_historical_percent(year, category_index) if year == 2015 else read_future_percent(year, category_index)
            draw_map(axis, data, category, country_segments, interpolation)
            if row_index == 0:
                axis.set_title(category.title, fontsize=6.7, fontweight="bold", pad=1.7, color="#1e2224")
            if category_index == 0:
                axis.text(-0.055, 0.50, str(year), transform=axis.transAxes, ha="right", va="center", fontsize=6.2, color="#3f4548", clip_on=False)
        add_colorbar(figure.add_subplot(outer[4, category_index]), category)

    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("pdf", "png"):
        work = WORK / f"{stem}.{suffix}"
        figure.savefig(work, dpi=DPI if suffix == "png" else None, bbox_inches="tight", pad_inches=0.035, metadata={"Creator": "accepted historical BMI map template"} if suffix == "pdf" else None)
        target = OUT / work.name
        target.write_bytes(work.read_bytes())
        paths.append(target)
    plt.close(figure)
    return paths

def write_qa(paths: list[Path], interpolation: str, qa_stem: str) -> None:
    rows = [{"path": str(path), "format": path.suffix.lstrip("."), "bytes": path.stat().st_size, "pass": path.stat().st_size > 10_000} for path in paths]
    with (OUT / f"{qa_stem}_export_QA.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    qa = {
        "status": "PASS_EXPORT_PENDING_VISUAL_REVIEW",
        "raster_interpolation": interpolation,
        "backend": "Python/matplotlib",
        "archetype": "quantitative map grid",
        "years": list(YEARS),
        "columns": [category.title for category in historical.CATEGORIES],
        "layout": "4 year rows x 7 BMI-category columns",
        "historical_template_reused": True,
        "historical_style_source": str(STYLE_SCRIPT),
        "category_vmax": [category.vmax for category in historical.CATEGORIES],
        "country_boundary": {"color": "#303437", "linewidth": 0.20, "alpha": 0.72, "simplification_tolerance_degrees": 0.035, "drawn_after_raster": True},
        "adm1_shown": False,
        "population_mask": "adult population >=100 persons per 0.1-degree cell",
        "formal_bmi_products_modified": False,
        "png_dpi": DPI,
        "category_letters_before_titles": False,
    }
    (OUT / f"{qa_stem}_parameter_QA.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not all(row["pass"] for row in rows):
        raise RuntimeError("Figure 1B export QA failed")

def main() -> int:
    segments = build_country_segments()
    main_paths = plot(segments, "bilinear", "TEST_Fig1B_BMI7_spatial_2015_2025_2035_2050_historical_template_bilinear_v3")
    write_qa(main_paths, "bilinear", "TEST_Fig1B_bilinear_v3")
    supplementary_paths = plot(segments, "nearest", "TEST_Fig1B_BMI7_spatial_2015_2025_2035_2050_historical_template_nearest_v3")
    write_qa(supplementary_paths, "nearest", "TEST_Fig1B_nearest_v3")
    print("TEST_FIGURE1B_BILINEAR_AND_NEAREST_EXPORT_PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
