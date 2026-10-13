#!/usr/bin/env python3
"""Aggregate sex-specific global population-weighted BMI metrics (2015-2050).

Builds the annual sex-specific trend series needed by the revised Figure 2
panels A-C (SSP2-4.5 overall + female + male), from the frozen products:

- Historical key years 2015/2020/2024: harmonized 25+ sex-age 0.1-degree
  products (seven-category proportions per sex-age cell + sex-age population).
- SSP2-4.5 2025-2050: probability_sex and population_sex 0.1-degree products.

Output: tables/Figure2_sex_trend_source_v1.csv (year, period, sex, population,
mean_bmi, bmi_ge25, bmi_ge30) and a consistency check against the existing
overall trend source (TEST_Fig1A_trend_source_four_SSP_v2.csv).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HIST = Path(_cfg.RESULTS / "05_historical_bmi_1990_2024" / "historical_25plus_compat_v1_20260909" / "data" / "0p1deg")
FUT = Path(_cfg.RESULTS / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906" / "outputs" / "ssp245")
OUT = Path(_cfg.RESULTS / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909" / "figure2_bmi_global_2015_2050_v1")
TREND_OVERALL = Path(_cfg.RESULTS / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909" / "test_figure1_AB_v2" / "TEST_Fig1A_trend_source_four_SSP_v2.csv")

MIDPOINTS = np.asarray([17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5], dtype=np.float32)
CAT_STEMS = ["bmi_lt18p5", "bmi_18p5_to_lt20", "bmi_20_to_lt25", "bmi_25_to_lt30", "bmi_30_to_lt35", "bmi_35_to_lt40", "bmi_ge40"]
SEXES = ("Female", "Male")


def masked_window(dataset, band, row0, row1):
    window = rasterio.windows.Window(0, row0, dataset.width, row1 - row0)
    return dataset.read(band, window=window, masked=True)


def aggregate_historical_year(year: int) -> dict:
    rows = {}
    pop_path = HIST / str(year) / f"global_{year}_adult_population_regional_steps_mrp_sex_age_0p1deg.tif"
    cat_paths = [
        HIST / str(year) / f"global_{year}_{stem}_regional_steps_mrp_sex_age_0p1deg_bp.tif"
        for stem in CAT_STEMS
    ]
    with rasterio.open(pop_path) as pop_ds:
        cats = [rasterio.open(p) for p in cat_paths]
        try:
            with rasterio.open(pop_path) as ds:
                total_rows = ds.height
            for row0 in range(0, total_rows, 720):
                row1 = min(row0 + 720, total_rows)
                pop = masked_window(pop_ds, list(range(1, 27)), row0, row1).astype(np.float64)
                female_pop = pop[:13].sum(axis=0)
                male_pop = pop[13:].sum(axis=0)
                for cat_index, cat_ds in enumerate(cats):
                    cat = masked_window(cat_ds, list(range(1, 27)), row0, row1).astype(np.float64)
                    # historical sex-age category files are uint16 basis points (0-10000)
                    cat = cat / 10000.0
                    for sex_name, sex_slice, sex_pop in (
                        ("Female", slice(0, 13), female_pop),
                        ("Male", slice(13, 26), male_pop),
                    ):
                        rows.setdefault((sex_name, cat_index), [0.0, 0.0])
                        weighted = (cat[sex_slice] * pop[sex_slice]).sum(axis=0)
                        valid = (sex_pop > 0) & np.isfinite(weighted.filled(np.nan))
                        num = np.ma.sum(np.where(valid, weighted, 0.0))
                        den = np.ma.sum(np.where(valid, sex_pop, 0.0))
                        rows[(sex_name, cat_index)][0] += float(num)
                        rows[(sex_name, cat_index)][1] += float(den)
        finally:
            for cat_ds in cats:
                cat_ds.close()
    result = {}
    for sex_name in SEXES:
        shares = np.array([rows[(sex_name, k)][0] / rows[(sex_name, k)][1] for k in range(7)])
        pop_total = rows[(sex_name, 0)][1]
        result[sex_name] = {
            "population": pop_total,
            "mean_bmi": float(np.dot(shares, MIDPOINTS)),
            "bmi_ge25": float(shares[3:].sum() * 100.0),
            "bmi_ge30": float(shares[4:].sum() * 100.0),
        }
    return result


def aggregate_future_year(year: int) -> dict:
    prob_path = FUT / str(year) / f"probability_sex_ssp245_{year}_0p1deg_v1.tif"
    pop_path = FUT / str(year) / f"population_sex_ssp245_{year}_0p1deg_v1.tif"
    rows = {}
    with rasterio.open(prob_path) as prob_ds, rasterio.open(pop_path) as pop_ds:
        total_rows = prob_ds.height
        for row0 in range(0, total_rows, 720):
            row1 = min(row0 + 720, total_rows)
            prob = masked_window(prob_ds, list(range(1, 15)), row0, row1).astype(np.float64)
            pop = masked_window(pop_ds, [1, 2], row0, row1).astype(np.float64)
            for sex_index, sex_name in enumerate(SEXES):
                sex_prob = prob[sex_index * 7:(sex_index + 1) * 7]
                sex_pop = pop[sex_index]
                valid = (sex_pop > 0) & np.isfinite(sex_prob.filled(np.nan)).all(axis=0)
                for cat_index in range(7):
                    num = np.ma.sum(np.where(valid, sex_prob[cat_index] * sex_pop, 0.0))
                    den = np.ma.sum(np.where(valid, sex_pop, 0.0))
                    rows.setdefault((sex_name, cat_index), [0.0, 0.0])
                    rows[(sex_name, cat_index)][0] += float(num)
                    rows[(sex_name, cat_index)][1] += float(den)
    result = {}
    for sex_name in SEXES:
        shares = np.array([rows[(sex_name, k)][0] / rows[(sex_name, k)][1] for k in range(7)])
        pop_total = rows[(sex_name, 0)][1]
        result[sex_name] = {
            "population": pop_total,
            "mean_bmi": float(np.dot(shares, MIDPOINTS)),
            "bmi_ge25": float(shares[3:].sum() * 100.0),
            "bmi_ge30": float(shares[4:].sum() * 100.0),
        }
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for year in (2015, 2020, 2024):
        result = aggregate_historical_year(year)
        for sex_name, values in result.items():
            records.append({"year": year, "period": "Historical", "sex": sex_name, **values})
        print("HIST", year, json.dumps(result, ensure_ascii=False)[:300])
    for year in range(2025, 2051):
        result = aggregate_future_year(year)
        for sex_name, values in result.items():
            records.append({"year": year, "period": "SSP2-4.5", "sex": sex_name, **values})
    frame = pd.DataFrame(records)
    overall = frame.groupby(["year", "period"], as_index=False).apply(
        lambda g: pd.Series({
            "sex": "Overall",
            "population": g["population"].sum(),
            "mean_bmi": np.average(g["mean_bmi"], weights=g["population"]),
            "bmi_ge25": np.average(g["bmi_ge25"], weights=g["population"]),
            "bmi_ge30": np.average(g["bmi_ge30"], weights=g["population"]),
        }), include_groups=False
    )
    frame = pd.concat([frame, overall], ignore_index=True).sort_values(["sex", "year"])
    csv_path = OUT / "tables" / "Figure2_sex_trend_source_v1.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    reference = pd.read_csv(TREND_OVERALL)
    checks = []
    merged = overall.merge(reference[reference["scenario"] == "SSP2-4.5"], on="year", suffixes=("", "_ref"))
    checks.append(float(np.abs(merged["mean_bmi"].astype(float) - merged["mean_bmi_ref"].astype(float)).max()))
    checks.append(float(np.abs(merged["bmi_ge25"].astype(float) - merged["overweight_obesity"].astype(float) * 100.0).max()))
    checks.append(float(np.abs(merged["bmi_ge30"].astype(float) - merged["obesity"].astype(float) * 100.0).max()))
    reference_hist = reference[reference["scenario"] == "Historical"]
    hist_new = overall[overall["period"] == "Historical"].merge(reference_hist, on="year", suffixes=("", "_ref"))
    checks.append(float(np.abs(hist_new["mean_bmi"].astype(float) - hist_new["mean_bmi_ref"].astype(float)).max()))
    checks.append(float(np.abs(hist_new["bmi_ge25"].astype(float) - hist_new["overweight_obesity"].astype(float) * 100.0).max()))
    checks.append(float(np.abs(hist_new["bmi_ge30"].astype(float) - hist_new["obesity"].astype(float) * 100.0).max()))
    qa = {
        "status": "PASS" if max(checks) < 0.15 else "CHECK",
        "max_abs_deviation_vs_existing_overall_trend": max(checks),
        "historical_years": [2015, 2020, 2024],
        "future_years": [2025, 2050],
        "sex_levels": ["Overall", "Female", "Male"],
    }
    (OUT / "tables" / "Figure2_sex_trend_source_QA_v1.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("QA", json.dumps(qa, ensure_ascii=False))
    print("WROTE", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
