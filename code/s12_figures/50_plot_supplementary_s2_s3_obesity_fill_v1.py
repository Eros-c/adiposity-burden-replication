#!/usr/bin/env python3
"""Supplementary Figures 2 and 3 with an obesity panel filling the empty cell.

Both delivered plates use a 2 x 4 gridspec in which the seven BMI categories occupy
seven cells, leaving row-1/column-3 empty. This script reproduces the delivered layout
exactly (same gridspec, margins, fonts, colour ramps and boundary renderer) and draws an
eighth panel, obesity prevalence (BMI >= 30, the sum of the three upper categories), in
the previously empty cell. Reads BMI products only; exports 1800-dpi LZW TIFF plus
PDF/PNG for QA. Formal delivered figures are not modified.
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

Image.MAX_IMAGE_PIXELS = None


HERE = Path(__file__).resolve().parent
TEST_SCRIPT = HERE / "00_style_concordance_test_historical_template_v1.py"
HIST25 = Path(
    _cfg.RESULTS / "05_historical_bmi_1990_2024" / "historical_25plus_compat_v1_20260909" / "data" / "0p1deg"
)
FIG_ROOT = Path(
    _cfg.RESULTS / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
)
WORK = FIG_ROOT / "supplementary_s2_s3_obesity_fill_v1"
HIST_FILE_LABELS = (
    "lt18p5", "18p5_to_lt20", "20_to_lt25", "25_to_lt30",
    "30_to_lt35", "35_to_lt40", "ge40",
)
# Obesity = BMI >= 30. Fixed scale matches the main-text Figure 2 BMI >= 30 maps so the
# same quantity carries the same colour mapping across the manuscript.
OBESITY_VMAX = 60.0
OBESITY_TITLE = "Obesity (BMI \u226530)"
OBESITY_SOURCE_CATEGORIES = (4, 5, 6)  # 30-<35 + 35-<40 + >=40
PLATES = (
    ((2015, 2020, 2025, 2030), "2015-2030", "Supplementary Figure 2_new"),
    ((2035, 2040, 2045, 2050), "2035-2050", "Supplementary Figure 3_new"),
)
TIFF_DPI = 1800
SUBTITLE = "Adults aged 25 years and older; 0.1 degree grid; fixed category-specific scales"


def import_test_module():
    spec = importlib.util.spec_from_file_location("style_test_module", TEST_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(TEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


style = import_test_module()
historical = style.historical_module


def read_historical25_percent(year: int, category_index: int) -> np.ma.MaskedArray:
    label = HIST_FILE_LABELS[category_index]
    path = HIST25 / str(year) / (
        f"global_{year}_bmi_{label}_regional_steps_mrp_0p1deg_bp.tif"
    )
    with rasterio.open(path) as dataset:
        return dataset.read(1, masked=True).astype(np.float32) / 100.0


def read_category_percent(year: int, category_index: int) -> np.ma.MaskedArray:
    if year <= 2020:
        return read_historical25_percent(year, category_index)
    return style.read_future_percent(category_index, year)


def read_obesity_percent(year: int) -> np.ma.MaskedArray:
    parts = [read_category_percent(year, index) for index in OBESITY_SOURCE_CATEGORIES]
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total


def read_panel(year: int, panel: dict) -> np.ma.MaskedArray:
    if panel["kind"] == "obesity":
        return read_obesity_percent(year)
    return read_category_percent(year, panel["cat_index"])


def build_panels() -> list[dict]:
    panels: list[dict] = []
    for index, category in enumerate(historical.CATEGORIES):
        panels.append(
            {
                "title": category.title,
                "vmax": float(category.vmax),
                "colors": category.colors,
                "kind": "category",
                "cat_index": index,
            }
        )
    panels.append(
        {
            "title": OBESITY_TITLE,
            "vmax": OBESITY_VMAX,
            "colors": historical.CATEGORIES[0].colors,
            "kind": "obesity",
            "cat_index": None,
        }
    )
    return panels


def plot_plate(years: tuple[int, ...], label: str, stem: str, panels: list[dict],
               country_segments) -> None:
    """Draw one plate; the eighth panel fills row-1/column-3 of the existing gridspec."""
    width, height = 13.0, 11.2
    figure = plt.figure(figsize=(width, height), constrained_layout=False)
    title_y = 1.0 - 0.06 / height
    subtitle_y = 1.0 - 0.36 / height
    content_top = 1.0 - 0.66 / height
    outer = figure.add_gridspec(
        2, 4, left=0.035, right=0.995, bottom=0.030, top=content_top,
        wspace=0.10, hspace=0.12,
    )
    figure.text(
        0.035, title_y,
        f"Global seven-category BMI composition, {label}",
        ha="left", va="top", fontsize=15, fontweight="bold", color="#151718",
    )
    figure.text(
        0.035, subtitle_y, SUBTITLE,
        ha="left", va="top", fontsize=9.4, color="#50575b",
    )

    for panel_index, panel in enumerate(panels):
        row = 0 if panel_index < 4 else 1
        column = panel_index if panel_index < 4 else panel_index - 4
        cell = outer[row, column].subgridspec(
            len(years) + 1, 1,
            height_ratios=[1.0] * len(years) + [0.075],
            hspace=0.015,
        )
        cmap = LinearSegmentedColormap.from_list(
            f"bmi_fill_{stem}_{panel_index}", panel["colors"], N=256
        )
        cmap.set_bad("white")
        norm = Normalize(vmin=0, vmax=panel["vmax"])
        letter = chr(ord("a") + panel_index)

        for map_index, year in enumerate(years):
            axis = figure.add_subplot(cell[map_index, 0])
            data = read_panel(year, panel)
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
            historical.add_boundaries(axis, country_segments, None)
            axis.set_xlim(-180, 180)
            axis.set_ylim(-60, 84)
            axis.set_aspect("equal", adjustable="box")
            axis.axis("off")
            if map_index == 0:
                axis.set_title(
                    panel["title"], fontsize=9.4, fontweight="bold",
                    pad=2.5, color="#1e2224",
                )
                axis.text(
                    0.0, 1.035, letter, transform=axis.transAxes,
                    ha="left", va="bottom", fontsize=9.4, fontweight="bold",
                    color="#151718", clip_on=False,
                )
            if column == 0:
                axis.text(
                    -0.025, 0.50, str(year), transform=axis.transAxes,
                    ha="right", va="center", fontsize=7.3,
                    color="#3f4548", clip_on=False,
                )

        color_axis = figure.add_subplot(cell[-1, 0])
        colorbar = mpl.colorbar.ColorbarBase(
            color_axis, cmap=cmap, norm=norm, orientation="horizontal",
        )
        ticks = np.asarray(
            [0, round(panel["vmax"] / 3), round(2 * panel["vmax"] / 3), panel["vmax"]],
            dtype=float,
        )
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{tick:g}" for tick in ticks])
        colorbar.ax.tick_params(labelsize=6.5, length=2, width=0.45, pad=1)
        colorbar.outline.set_visible(False)
        colorbar.ax.text(
            0, 1.8, "%", transform=colorbar.ax.transAxes,
            ha="left", va="bottom", fontsize=6.5, color="#4a5053",
        )

    WORK.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        WORK / f"{stem}.pdf", dpi=200,
        bbox_inches="tight", pad_inches=0.04,
        metadata={"Creator": "supplementary S2/S3 obesity fill"},
    )
    figure.savefig(
        WORK / f"{stem}_preview.png", dpi=100,
        bbox_inches="tight", pad_inches=0.04,
    )
    tif_path = FIG_ROOT / f"{stem}.tif"
    figure.savefig(
        tif_path,
        format="tiff", dpi=TIFF_DPI,
        bbox_inches="tight", pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)

    # Match the delivered plates: 3-sample RGB, not matplotlib's 4-sample RGBA.
    with Image.open(tif_path) as image:
        if image.mode != "RGB":
            rgb = image.convert("RGB")
            rgb.save(
                tif_path, format="tiff",
                compression="tiff_lzw",
                dpi=(TIFF_DPI, TIFF_DPI),
            )


def write_qa(records: list[dict], panels: list[dict]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "path": str(path),
            "format": path.suffix.lstrip("."),
            "bytes": path.stat().st_size,
            "pass": path.stat().st_size > 100_000,
        }
        for record in records
        for path in record["paths"]
    ]
    with (WORK / "supplementary_s2_s3_obesity_fill_export_QA_v1.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    tif_dims = []
    for record in records:
        tif = FIG_ROOT / f"{record['stem']}.tif"
        with Image.open(tif) as image:
            tif_dims.append(
                {
                    "stem": record["stem"],
                    "size_px": list(image.size),
                    "dpi": [float(value) for value in image.info.get("dpi", ())],
                    "mode": image.mode,
                    "compression": int(image.tag_v2.get(259)),
                    "samples": int(image.tag_v2.get(277)),
                }
            )

    qa = {
        "status": "PASS_EXPORT_PENDING_VISUAL_REVIEW",
        "role": "Supplementary Figure 2/3 regeneration with obesity panel",
        "supersedes": [
            str(FIG_ROOT / "Supplementary Figure 2.tif"),
            str(FIG_ROOT / "Supplementary Figure 3.tif"),
        ],
        "layout": "2 x 4 gridspec; seven BMI categories in cells a-g; obesity panel h "
                  "fills the previously empty row-1/column-3 cell",
        "panels": [
            {
                "letter": chr(ord("a") + index),
                "title": panel["title"],
                "kind": panel["kind"],
                "vmax": panel["vmax"],
            }
            for index, panel in enumerate(panels)
        ],
        "obesity_definition": "sum of the 30-<35, 35-<40 and >=40 kg/m2 category shares "
                              "(adults aged 25 years and older), historical 2015/2020 from "
                              "the 25+ compatibility product and 2025+ from the SSP2-4.5 "
                              "bridge with the adult-population >=100 display mask",
        "obesity_scale": "0-60% (ticks 0/20/40/60), identical to the main-text Figure 2 "
                         "BMI >= 30 maps",
        "historical_template_imported": str(TEST_SCRIPT),
        "plates": [
            {"stem": record["stem"], "years": list(record["years"]), "title_label": record["label"]}
            for record in records
        ],
        "canvas_inch": [13.0, 11.2],
        "tiff_dpi": TIFF_DPI,
        "tif": tif_dims,
        "formal_delivered_figures_modified": False,
    }
    (WORK / "supplementary_s2_s3_obesity_fill_parameter_QA_v1.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not all(row["pass"] for row in rows):
        raise RuntimeError("Supplementary S2/S3 obesity-fill export QA failed")
    for entry in tif_dims:
        if entry["dpi"] != [float(TIFF_DPI), float(TIFF_DPI)]:
            raise RuntimeError(f"TIFF dpi is not {TIFF_DPI}: {entry}")


def main() -> int:
    countries = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")
    countries = countries[countries.geometry.notna()].copy()
    countries.geometry = countries.geometry.simplify(0.035, preserve_topology=True)
    country_segments = historical.geometry_segments(countries)

    panels = build_panels()
    records: list[dict] = []
    for years, label, stem in PLATES:
        plot_plate(years, label, stem, panels, country_segments)
        records.append(
            {
                "stem": stem,
                "years": list(years),
                "label": label,
                "paths": [
                    FIG_ROOT / f"{stem}.tif",
                    WORK / f"{stem}.pdf",
                    WORK / f"{stem}_preview.png",
                ],
            }
        )
    write_qa(records, panels)
    print("SUPPLEMENTARY_S2_S3_OBESITY_FILL_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
