#!/usr/bin/env python3
"""Audit frozen GBD/NCD-RisC inputs and build 2025-2050 BMI7 national anchors.

This stage builds national-age-sex BMI composition anchors only. It does not fit
the future spatial MRP, download SSP/CMIP6 data, or create disease burden maps.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "05_audit_and_build_national_bmi7_anchors_v1_1"
EPS = 1e-10

BMI_COLUMNS = [
    "p_bmi_lt_18_5",
    "p_bmi_18_5_20",
    "p_bmi_20_25",
    "p_bmi_25_30",
    "p_bmi_30_35",
    "p_bmi_35_40",
    "p_bmi_ge_40",
]

NCD_COLUMN_MAP = {
    "Prevalence of BMI<18.5 kg/m² (underweight)": "p_bmi_lt_18_5",
    "Prevalence of BMI 18.5 kg/m² to <20 kg/m²": "p_bmi_18_5_20",
    "Prevalence of BMI 20 kg/m² to <25 kg/m²": "p_bmi_20_25",
    "Prevalence of BMI 25 kg/m² to <30 kg/m²": "p_bmi_25_30",
    "Prevalence of BMI 30 kg/m² to <35 kg/m²": "p_bmi_30_35",
    "Prevalence of BMI 35 kg/m² to <40 kg/m²": "p_bmi_35_40",
    "Prevalence of BMI >=40 kg/m² (morbid obesity)": "p_bmi_ge_40",
}

GBD_TO_NCD_ALIASES = {
    "Democratic People's Republic of Korea": "North Korea",
    "Taiwan (Province of China)": "Taiwan",
    "Lao People's Democratic Republic": "Lao PDR",
    "Micronesia (Federated States of)": "Federated States of Micronesia",
    "Republic of Moldova": "Moldova",
    "Republic of Korea": "South Korea",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Iran (Islamic Republic of)": "Iran",
    "Democratic Republic of the Congo": "DR Congo",
    "United Republic of Tanzania": "Tanzania",
}

NCD_AGE_TO_GBD = {
    "25-29": ["25 to 29"],
    "30-34": ["30 to 34"],
    "35-39": ["35 to 39"],
    "40-44": ["40 to 44"],
    "45-49": ["45 to 49"],
    "50-54": ["50 to 54"],
    "55-59": ["55 to 59"],
    "60-64": ["60 to 64"],
    "65-69": ["65 to 69"],
    "70-74": ["70 to 74"],
    "75-79": ["75 to 79"],
    "80-84": ["80 to 84"],
    "85plus": ["85 to 89", "90 to 94", "95 plus"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    text = text.replace("&", "and")
    return re.sub(r"[^a-z0-9]", "", text)


def parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def softmax_from_two_alrs(alr1: np.ndarray, alr2: np.ndarray) -> np.ndarray:
    values = np.column_stack([alr1, alr2, np.zeros_like(alr1)])
    values -= values.max(axis=1, keepdims=True)
    expv = np.exp(values)
    return expv / expv.sum(axis=1, keepdims=True)


def weighted_slope(year: np.ndarray, value: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(year) & np.isfinite(value)
    x = year[mask].astype(float)
    y = value[mask].astype(float)
    if len(x) < 5 or np.ptp(x) <= 0:
        return 0.0, math.inf
    xc = x - x.mean()
    sxx = float(np.sum(xc * xc))
    slope = float(np.sum(xc * (y - y.mean())) / sxx)
    resid = y - (y.mean() + slope * xc)
    sigma2 = float(np.sum(resid * resid) / max(len(x) - 2, 1))
    se2 = sigma2 / sxx if sxx > 0 else math.inf
    return slope, se2


def load_ncd(ncd_root: Path) -> pd.DataFrame:
    frames = []
    for label, file_name in [
        ("Female", "NCD_RisC_Nature_2026_BMI_female_age_specific_country.zip"),
        ("Male", "NCD_RisC_Nature_2026_BMI_male_age_specific_country.zip"),
    ]:
        path = ncd_root / file_name
        with zipfile.ZipFile(path) as archive:
            entry = archive.namelist()[0]
            with archive.open(entry) as stream:
                usecols = ["Year", "Sex", "Country/Region/World", "ISO", "Age group", *NCD_COLUMN_MAP]
                frame = pd.read_csv(stream, usecols=usecols)
        frame = frame.rename(columns={
            "Year": "year",
            "Country/Region/World": "ncd_country_name",
            "ISO": "iso3",
            "Age group": "ncd_age_group",
            **NCD_COLUMN_MAP,
        })
        frame["sex"] = label
        frame = frame[frame["year"].between(1990, 2024)].copy()
        frame = frame[frame["ncd_age_group"].isin(NCD_AGE_TO_GBD)].copy()
        frames.append(frame[["year", "sex", "ncd_country_name", "iso3", "ncd_age_group", *BMI_COLUMNS]])
    ncd = pd.concat(frames, ignore_index=True)
    ncd["closure"] = ncd[BMI_COLUMNS].sum(axis=1)
    return ncd


def load_gbd_forecast(bmi_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    age_dir = bmi_root / "IHME_GLOBAL_OVERWEIGHT_OBESITY_PREVALENCE_1990_2050_AGES_25_125"
    files = {
        "obesity": next(age_dir.glob("*_OB_*.CSV")),
        "overweight": next(age_dir.glob("*_OW_OB_*.CSV")),
        "overweight_less_obesity": next(age_dir.glob("*_OW_ONLY_*.CSV")),
    }
    frames = []
    id_cols = [
        "location_level", "location_name", "location_id", "Sex", "sex_id",
        "age_group_name", "age_group_id", "year_id",
    ]
    for expected_metric, path in files.items():
        frame = pd.read_csv(path)
        observed = set(frame["metric"].dropna().unique())
        if observed != {expected_metric}:
            raise RuntimeError(f"Unexpected metric in {path}: {observed}")
        frame = frame[(frame["location_level"] == "Country") & frame["sex_id"].isin([1, 2])].copy()
        frame = frame[frame["year_id"].between(1990, 2050)].copy()
        frame = frame[id_cols + ["mean_prev", "lower_prev", "upper_prev"]]
        frame = frame.rename(columns={
            "Sex": "sex",
            "year_id": "year",
            "mean_prev": f"{expected_metric}_mean",
            "lower_prev": f"{expected_metric}_lower",
            "upper_prev": f"{expected_metric}_upper",
        })
        frames.append(frame)
    merge_keys = [
        "location_level", "location_name", "location_id", "sex", "sex_id",
        "age_group_name", "age_group_id", "year",
    ]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=merge_keys, how="outer", validate="one_to_one")
    return merged, list(files.values())


def audit_crc(path: Path, high_bmi: bool) -> tuple[dict, pd.DataFrame]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    expected = ["测量", "指标", "原因", "地理位置", "年龄", "性别", "年份", "数值", "上", "降低"]
    if high_bmi:
        expected.insert(3, "风险")
    missing = sorted(set(expected) - set(header))
    if missing:
        raise RuntimeError(f"Missing CRC columns in {path}: {missing}")

    key_cols = ["测量", "指标", "原因", "地理位置", "年龄", "性别", "年份"]
    if high_bmi:
        key_cols.insert(3, "风险")
    usecols = key_cols + ["数值", "上", "降低"]
    row_count = 0
    nonfinite = 0
    order_fail = 0
    negative = 0
    duplicate_count = 0
    seen_hashes: set[int] = set()
    unique_values = {c: set() for c in key_cols}
    # Freeze all raw query-table columns as strings before numeric parsing.
    # This avoids pandas mixed-type inference failures without changing values.
    raw_dtypes = {column: "string" for column in usecols}
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=200_000,
        dtype=raw_dtypes,
        low_memory=False,
    ):
        row_count += len(chunk)
        for col in ["数值", "上", "降低"]:
            chunk[col] = parse_numeric(chunk[col])
        numeric = chunk[["数值", "上", "降低"]].to_numpy(float)
        nonfinite += int((~np.isfinite(numeric)).sum())
        order_fail += int(((chunk["降低"] > chunk["数值"] + 1e-12) | (chunk["数值"] > chunk["上"] + 1e-12)).sum())
        negative += int((numeric < -1e-12).sum())
        hashes = pd.util.hash_pandas_object(chunk[key_cols], index=False).astype("uint64")
        duplicate_count += int(hashes.duplicated().sum())
        for item in hashes.tolist():
            if item in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes.add(item)
        for col in key_cols:
            unique_values[col].update(chunk[col].dropna().astype(str).unique().tolist())

    dimensions = []
    for col, values in unique_values.items():
        dimensions.append({"dataset": "crc_high_bmi" if high_bmi else "crc_total", "dimension": col, "n_unique": len(values), "values": " | ".join(sorted(values))})
    summary = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": row_count,
        "unique_keys": len(seen_hashes),
        "duplicate_key_rows": duplicate_count,
        "nonfinite_numeric_cells": nonfinite,
        "interval_order_fail_rows": order_fail,
        "negative_numeric_cells": negative,
    }
    return summary, pd.DataFrame(dimensions)


def build_crosswalk(gbd: pd.DataFrame, ncd: pd.DataFrame) -> pd.DataFrame:
    g_locations = gbd[["location_id", "location_name"]].drop_duplicates().sort_values("location_id")
    n_locations = ncd[["iso3", "ncd_country_name"]].drop_duplicates()
    by_norm = {normalize_name(row.ncd_country_name): (row.iso3, row.ncd_country_name) for row in n_locations.itertuples()}
    by_name = {row.ncd_country_name: (row.iso3, row.ncd_country_name) for row in n_locations.itertuples()}
    records = []
    for row in g_locations.itertuples():
        exact = by_norm.get(normalize_name(row.location_name))
        method = "normalized_exact"
        if exact is None and row.location_name in GBD_TO_NCD_ALIASES:
            exact = by_name.get(GBD_TO_NCD_ALIASES[row.location_name])
            method = "manual_alias"
        if exact is None:
            records.append({
                "location_id": row.location_id,
                "location_name": row.location_name,
                "iso3": "",
                "ncd_country_name": "",
                "crosswalk_method": "no_ncd_country_match_global_age_sex_fallback",
            })
        else:
            records.append({
                "location_id": row.location_id,
                "location_name": row.location_name,
                "iso3": exact[0],
                "ncd_country_name": exact[1],
                "crosswalk_method": method,
            })
    return pd.DataFrame(records)


def forecast_subcomposition(ncd: pd.DataFrame, future_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = ncd.copy()
    low_sum = data[BMI_COLUMNS[:3]].sum(axis=1).clip(lower=EPS)
    high_sum = data[BMI_COLUMNS[4:]].sum(axis=1).clip(lower=EPS)
    data["low_alr_1"] = np.log(data[BMI_COLUMNS[0]].clip(lower=EPS) / data[BMI_COLUMNS[2]].clip(lower=EPS))
    data["low_alr_2"] = np.log(data[BMI_COLUMNS[1]].clip(lower=EPS) / data[BMI_COLUMNS[2]].clip(lower=EPS))
    data["high_alr_1"] = np.log(data[BMI_COLUMNS[5]].clip(lower=EPS) / data[BMI_COLUMNS[4]].clip(lower=EPS))
    data["high_alr_2"] = np.log(data[BMI_COLUMNS[6]].clip(lower=EPS) / data[BMI_COLUMNS[4]].clip(lower=EPS))
    alr_cols = ["low_alr_1", "low_alr_2", "high_alr_1", "high_alr_2"]

    recent = data[data["year"].between(2005, 2024)].copy()
    slope_records = []
    group_cols = ["iso3", "ncd_country_name", "sex", "ncd_age_group"]
    for key, group in recent.groupby(group_cols, sort=False):
        record = dict(zip(group_cols, key))
        for col in alr_cols:
            slope, se2 = weighted_slope(group["year"].to_numpy(), group[col].to_numpy())
            record[f"{col}_slope_raw"] = slope
            record[f"{col}_se2"] = se2
        slope_records.append(record)
    slopes = pd.DataFrame(slope_records)

    for col in alr_cols:
        raw = f"{col}_slope_raw"
        se2col = f"{col}_se2"
        slopes[f"{col}_global_slope"] = slopes.groupby(["sex", "ncd_age_group"])[raw].transform("median")
        slopes[f"{col}_tau2"] = slopes.groupby(["sex", "ncd_age_group"])[raw].transform("var").fillna(0.0).clip(lower=1e-10)
        weight = slopes[f"{col}_tau2"] / (slopes[f"{col}_tau2"] + slopes[se2col].replace([np.inf, -np.inf], np.nan).fillna(1e6))
        slopes[f"{col}_slope_eb"] = weight * slopes[raw] + (1.0 - weight) * slopes[f"{col}_global_slope"]
        lower = slopes.groupby(["sex", "ncd_age_group"])[raw].transform(lambda s: s.quantile(0.01))
        upper = slopes.groupby(["sex", "ncd_age_group"])[raw].transform(lambda s: s.quantile(0.99))
        slopes[f"{col}_slope_eb"] = slopes[f"{col}_slope_eb"].clip(lower=lower, upper=upper)

    base = data[data["year"] == 2024][group_cols + alr_cols].merge(slopes, on=group_cols, how="left", validate="one_to_one")
    bounds = {}
    for col in alr_cols:
        bounds[col] = data.groupby(["sex", "ncd_age_group"])[col].quantile([0.005, 0.995]).unstack().rename(columns={0.005: f"{col}_q005", 0.995: f"{col}_q995"})
        base = base.merge(bounds[col].reset_index(), on=["sex", "ncd_age_group"], how="left", validate="many_to_one")

    pred_records = []
    for year in future_years:
        current = base[group_cols].copy()
        current["year"] = year
        horizon = year - 2024
        for col in alr_cols:
            raw = base[col] + horizon * base[f"{col}_slope_eb"]
            current[col] = raw.clip(lower=base[f"{col}_q005"], upper=base[f"{col}_q995"])
        low = softmax_from_two_alrs(current["low_alr_1"].to_numpy(), current["low_alr_2"].to_numpy())
        high = softmax_from_two_alrs(current["high_alr_1"].to_numpy(), current["high_alr_2"].to_numpy())
        current["low_share_lt_18_5"] = low[:, 0]
        current["low_share_18_5_20"] = low[:, 1]
        current["low_share_20_25"] = low[:, 2]
        current["obesity_share_30_35"] = high[:, 2]
        current["obesity_share_35_40"] = high[:, 0]
        current["obesity_share_ge_40"] = high[:, 1]
        pred_records.append(current)
    predictions = pd.concat(pred_records, ignore_index=True)
    return predictions, slopes


def expand_ncd_ages(subcomp: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ncd_age, gbd_ages in NCD_AGE_TO_GBD.items():
        subset = subcomp[subcomp["ncd_age_group"] == ncd_age].copy()
        for gbd_age in gbd_ages:
            current = subset.copy()
            current["age_group_name"] = gbd_age
            current["age_mapping_method"] = "exact_5y" if ncd_age != "85plus" else "ncd_85plus_reused_for_gbd_85plus_subgroup"
            frames.append(current)
    return pd.concat(frames, ignore_index=True)


def make_anchor_scenario(macro: pd.DataFrame, subcomp: pd.DataFrame, scenario: str, ncd: pd.DataFrame) -> pd.DataFrame:
    merged = macro.merge(subcomp, on=["iso3", "sex", "age_group_name", "year"], how="left", validate="many_to_one")
    share_cols = [
        "low_share_lt_18_5", "low_share_18_5_20", "low_share_20_25",
        "obesity_share_30_35", "obesity_share_35_40", "obesity_share_ge_40",
    ]
    fallback = subcomp.groupby(["sex", "age_group_name", "year"])[share_cols].median().reset_index()
    missing = merged[share_cols[0]].isna()
    if missing.any():
        fill = merged.loc[missing, ["sex", "age_group_name", "year"]].merge(fallback, on=["sex", "age_group_name", "year"], how="left", validate="many_to_one")
        merged.loc[missing, share_cols] = fill[share_cols].to_numpy()
    merged["subcomposition_source"] = np.where(missing, "global_age_sex_median_fallback", "ncd_country_age_sex_eb_trend")

    # Component-wise medians are not themselves guaranteed to be a closed
    # composition. Renormalize each block after any fallback assignment.
    low_share_cols = ["low_share_lt_18_5", "low_share_18_5_20", "low_share_20_25"]
    obesity_share_cols = ["obesity_share_30_35", "obesity_share_35_40", "obesity_share_ge_40"]
    low_total = merged[low_share_cols].sum(axis=1)
    obesity_total = merged[obesity_share_cols].sum(axis=1)
    if (low_total <= 0).any() or (obesity_total <= 0).any():
        raise RuntimeError("Non-positive BMI subcomposition block encountered after fallback assignment")
    merged[low_share_cols] = merged[low_share_cols].div(low_total, axis=0)
    merged[obesity_share_cols] = merged[obesity_share_cols].div(obesity_total, axis=0)

    if scenario == "gbd_absolute_primary":
        p_mid = merged["overweight_less_obesity_mean"].to_numpy()
        p_ob = merged["obesity_mean"].to_numpy()
    elif scenario == "ncd_2024_delta_sensitivity":
        ncd_2024 = ncd[ncd["year"] == 2024].copy()
        ncd_2024["p_mid_ncd_2024"] = ncd_2024["p_bmi_25_30"]
        ncd_2024["p_ob_ncd_2024"] = ncd_2024[BMI_COLUMNS[4:]].sum(axis=1)
        ncd_macro = ncd_2024[["iso3", "sex", "ncd_age_group", "p_mid_ncd_2024", "p_ob_ncd_2024"]]
        ncd_macro = expand_ncd_ages(ncd_macro)
        ncd_macro = ncd_macro[["iso3", "sex", "age_group_name", "p_mid_ncd_2024", "p_ob_ncd_2024"]]
        gbd_2024 = macro[macro["year"] == 2024][["location_id", "sex", "age_group_name", "overweight_less_obesity_mean", "obesity_mean"]].rename(columns={
            "overweight_less_obesity_mean": "p_mid_gbd_2024",
            "obesity_mean": "p_ob_gbd_2024",
        })
        merged = merged.merge(ncd_macro, on=["iso3", "sex", "age_group_name"], how="left", validate="many_to_one")
        merged = merged.merge(gbd_2024, on=["location_id", "sex", "age_group_name"], how="left", validate="many_to_one")
        ncd_age_median = ncd_macro.groupby(["sex", "age_group_name"])[["p_mid_ncd_2024", "p_ob_ncd_2024"]].median().reset_index()
        need = merged["p_mid_ncd_2024"].isna()
        if need.any():
            fill = merged.loc[need, ["sex", "age_group_name"]].merge(ncd_age_median, on=["sex", "age_group_name"], how="left", validate="many_to_one")
            merged.loc[need, ["p_mid_ncd_2024", "p_ob_ncd_2024"]] = fill[["p_mid_ncd_2024", "p_ob_ncd_2024"]].to_numpy()
        def macro_alr(mid: np.ndarray, ob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            low = np.clip(1.0 - mid - ob, EPS, 1.0)
            return np.log(np.clip(mid, EPS, 1.0) / low), np.log(np.clip(ob, EPS, 1.0) / low)
        n1, n2 = macro_alr(merged["p_mid_ncd_2024"].to_numpy(), merged["p_ob_ncd_2024"].to_numpy())
        y1, y2 = macro_alr(merged["overweight_less_obesity_mean"].to_numpy(), merged["obesity_mean"].to_numpy())
        b1, b2 = macro_alr(merged["p_mid_gbd_2024"].to_numpy(), merged["p_ob_gbd_2024"].to_numpy())
        composition = softmax_from_two_alrs(n1 + (y1 - b1), n2 + (y2 - b2))
        p_mid = composition[:, 0]
        p_ob = composition[:, 1]
    else:
        raise ValueError(scenario)

    p_low = np.clip(1.0 - p_mid - p_ob, 0.0, 1.0)
    merged["p_bmi_lt_18_5"] = p_low * merged["low_share_lt_18_5"]
    merged["p_bmi_18_5_20"] = p_low * merged["low_share_18_5_20"]
    merged["p_bmi_20_25"] = p_low * merged["low_share_20_25"]
    merged["p_bmi_25_30"] = p_mid
    merged["p_bmi_30_35"] = p_ob * merged["obesity_share_30_35"]
    merged["p_bmi_35_40"] = p_ob * merged["obesity_share_35_40"]
    merged["p_bmi_ge_40"] = p_ob * merged["obesity_share_ge_40"]
    merged["anchor_scenario"] = scenario
    merged["p_overweight_bmi_ge_25"] = merged[BMI_COLUMNS[3:]].sum(axis=1)
    merged["p_obesity_bmi_ge_30"] = merged[BMI_COLUMNS[4:]].sum(axis=1)
    merged["closure"] = merged[BMI_COLUMNS].sum(axis=1)
    keep = [
        "anchor_scenario", "location_id", "location_name", "iso3", "sex", "sex_id",
        "age_group_name", "age_group_id", "year", "crosswalk_method", "age_mapping_method",
        "subcomposition_source", *BMI_COLUMNS, "p_overweight_bmi_ge_25", "p_obesity_bmi_ge_30",
        "overweight_mean", "overweight_less_obesity_mean", "obesity_mean", "closure",
    ]
    return merged[keep]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(_cfg.RESULTS))
    parser.add_argument("--bmi-root", default=str(_cfg.DATA / "Adult_BMI_forecast_1990_2050" / "Adult_BMI_forecast_1990_2050"))
    parser.add_argument("--crc-root", default=str(_cfg.DATA / "GBD2021_query_table_exports_camoufox" / "merged"))
    parser.add_argument("--ncd-root", default=str(_cfg.DATA / "肥胖" / "全球" / "NCD_RisC" / "国家-年份-年龄-性别肥胖"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    tables.mkdir()

    bmi_root = Path(args.bmi_root)
    crc_root = Path(args.crc_root)
    ncd_root = Path(args.ncd_root)
    input_paths = [
        crc_root / "C013_total_burden.csv",
        crc_root / "C013_high_bmi_attributable.csv",
        ncd_root / "NCD_RisC_Nature_2026_BMI_female_age_specific_country.zip",
        ncd_root / "NCD_RisC_Nature_2026_BMI_male_age_specific_country.zip",
    ]

    ncd = load_ncd(ncd_root)
    gbd, forecast_files = load_gbd_forecast(bmi_root)
    input_paths.extend(forecast_files)
    input_paths.extend(sorted(bmi_root.glob("*CB_*.csv")))

    total_summary, total_dims = audit_crc(input_paths[0], high_bmi=False)
    attributable_summary, attributable_dims = audit_crc(input_paths[1], high_bmi=True)

    gbd["ow_identity_error"] = gbd["overweight_mean"] - gbd["overweight_less_obesity_mean"] - gbd["obesity_mean"]
    gbd["intervals_ordered"] = True
    for metric in ["overweight", "overweight_less_obesity", "obesity"]:
        gbd["intervals_ordered"] &= (
            (gbd[f"{metric}_lower"] <= gbd[f"{metric}_mean"] + 1e-12)
            & (gbd[f"{metric}_mean"] <= gbd[f"{metric}_upper"] + 1e-12)
        )

    crosswalk = build_crosswalk(gbd, ncd)
    crosswalk.to_csv(tables / "gbd_bmi_to_ncd_country_crosswalk_v1.csv", index=False, encoding="utf-8-sig")
    gbd = gbd.merge(crosswalk, on=["location_id", "location_name"], how="left", validate="many_to_one")

    gbd_future = gbd[gbd["year"].between(2025, 2050)].copy()
    normalized_cols = [
        "location_id", "location_name", "iso3", "crosswalk_method", "sex", "sex_id",
        "age_group_name", "age_group_id", "year",
        "overweight_mean", "overweight_lower", "overweight_upper",
        "overweight_less_obesity_mean", "overweight_less_obesity_lower", "overweight_less_obesity_upper",
        "obesity_mean", "obesity_lower", "obesity_upper", "ow_identity_error",
    ]
    gbd_future[normalized_cols].to_csv(
        tables / "gbd_adult_bmi_forecast_country_age_sex_2025_2050_normalized_v1.csv.gz",
        index=False,
        compression="gzip",
    )

    subcomp, slope_table = forecast_subcomposition(ncd, list(range(2024, 2051)))
    subcomp = expand_ncd_ages(subcomp)
    subcomp.to_csv(tables / "ncd_bmi7_internal_subcomposition_2024_2050_v1.csv.gz", index=False, compression="gzip")
    slope_table.to_csv(tables / "ncd_bmi7_empirical_bayes_slopes_v1.csv.gz", index=False, compression="gzip")

    # Keep 2024 in memory while constructing both scenarios because the
    # delta-based sensitivity anchor explicitly needs the GBD 2024 baseline.
    macro = gbd[gbd["year"].between(2024, 2050)].copy()
    primary = make_anchor_scenario(macro, subcomp, "gbd_absolute_primary", ncd)
    sensitivity = make_anchor_scenario(macro, subcomp, "ncd_2024_delta_sensitivity", ncd)
    primary = primary[primary["year"].between(2025, 2050)].copy()
    sensitivity = sensitivity[sensitivity["year"].between(2025, 2050)].copy()
    anchors = pd.concat([primary, sensitivity], ignore_index=True)
    anchors.to_csv(tables / "national_bmi7_anchors_2025_2050_v1.csv.gz", index=False, compression="gzip")

    primary_rows = anchors[anchors["anchor_scenario"] == "gbd_absolute_primary"]
    qas = [
        {"check": "ncd_rows", "value": len(ncd), "status": "PASS" if len(ncd) == 200 * 2 * 13 * 35 else "FAIL"},
        {"check": "ncd_country_count", "value": ncd["iso3"].nunique(), "status": "PASS" if ncd["iso3"].nunique() == 200 else "FAIL"},
        {"check": "ncd_max_closure_error", "value": float((ncd["closure"] - 1).abs().max()), "status": "PASS" if (ncd["closure"] - 1).abs().max() < 1e-8 else "FAIL"},
        {"check": "gbd_country_count", "value": int(gbd.loc[gbd["location_level"] == "Country", "location_id"].nunique()), "status": "PASS" if gbd["location_id"].nunique() == 204 else "FAIL"},
        {"check": "gbd_forecast_rows_2025_2050", "value": len(gbd_future), "status": "PASS" if len(gbd_future) == 204 * 2 * 15 * 26 else "FAIL"},
        {"check": "gbd_overweight_identity_max_abs_error", "value": float(gbd["ow_identity_error"].abs().max()), "status": "PASS" if gbd["ow_identity_error"].abs().max() < 1e-10 else "FAIL"},
        {"check": "gbd_interval_order_fail_rows", "value": int((~gbd["intervals_ordered"]).sum()), "status": "PASS" if gbd["intervals_ordered"].all() else "FAIL"},
        {"check": "gbd_to_ncd_crosswalk_complete", "value": int(len(crosswalk)), "status": "PASS" if len(crosswalk) == 204 else "FAIL"},
        {"check": "gbd_to_ncd_matched_locations", "value": int(crosswalk["iso3"].ne("").sum()), "status": "PASS" if crosswalk["iso3"].ne("").sum() == 199 else "FAIL"},
        {"check": "gbd_to_ncd_fallback_locations", "value": int(crosswalk["iso3"].eq("").sum()), "status": "WARN" if crosswalk["iso3"].eq("").sum() > 0 else "PASS"},
        {"check": "primary_anchor_rows", "value": len(primary_rows), "status": "PASS" if len(primary_rows) == 204 * 2 * 15 * 26 else "FAIL"},
        {"check": "primary_anchor_max_closure_error", "value": float((primary_rows["closure"] - 1).abs().max()), "status": "PASS" if (primary_rows["closure"] - 1).abs().max() < 1e-12 else "FAIL"},
        {"check": "primary_anchor_min_probability", "value": float(primary_rows[BMI_COLUMNS].min().min()), "status": "PASS" if primary_rows[BMI_COLUMNS].min().min() >= -1e-12 else "FAIL"},
        {"check": "primary_p25_30_gbd_max_abs_error", "value": float((primary_rows["p_bmi_25_30"] - primary_rows["overweight_less_obesity_mean"]).abs().max()), "status": "PASS" if (primary_rows["p_bmi_25_30"] - primary_rows["overweight_less_obesity_mean"]).abs().max() < 1e-12 else "FAIL"},
        {"check": "primary_obesity_gbd_max_abs_error", "value": float((primary_rows["p_obesity_bmi_ge_30"] - primary_rows["obesity_mean"]).abs().max()), "status": "PASS" if (primary_rows["p_obesity_bmi_ge_30"] - primary_rows["obesity_mean"]).abs().max() < 1e-12 else "FAIL"},
        {"check": "crc_total_rows", "value": total_summary["rows"], "status": "PASS" if total_summary["rows"] == 204 * 15 * 2 * 32 * 3 * 2 else "FAIL"},
        {"check": "crc_high_bmi_rows", "value": attributable_summary["rows"], "status": "PASS" if attributable_summary["rows"] == 204 * 15 * 2 * 32 * 2 * 2 else "FAIL"},
        {"check": "crc_total_key_duplicates", "value": total_summary["duplicate_key_rows"], "status": "PASS" if total_summary["duplicate_key_rows"] == 0 else "FAIL"},
        {"check": "crc_high_bmi_key_duplicates", "value": attributable_summary["duplicate_key_rows"], "status": "PASS" if attributable_summary["duplicate_key_rows"] == 0 else "FAIL"},
    ]
    qa = pd.DataFrame(qas)
    qa.to_csv(tables / "content_audit_checks_v1.csv", index=False, encoding="utf-8-sig")
    pd.concat([total_dims, attributable_dims], ignore_index=True).to_csv(tables / "crc_dimension_inventory_v1.csv", index=False, encoding="utf-8-sig")

    # NCD-to-GBD 2024/2025 bridge diagnostics for the primary absolute anchor.
    ncd_2024 = ncd[ncd["year"] == 2024].copy()
    ncd_2024["p_obesity_ncd_2024"] = ncd_2024[BMI_COLUMNS[4:]].sum(axis=1)
    ncd_2024 = expand_ncd_ages(ncd_2024[["iso3", "sex", "ncd_age_group", "p_bmi_25_30", "p_obesity_ncd_2024"]])
    bridge = primary_rows[primary_rows["year"] == 2025].merge(
        ncd_2024[["iso3", "sex", "age_group_name", "p_bmi_25_30", "p_obesity_ncd_2024"]],
        on=["iso3", "sex", "age_group_name"], how="left", suffixes=("_2025", "_ncd_2024"), validate="many_to_one",
    )
    bridge["p25_30_jump_pp"] = 100 * (bridge["p_bmi_25_30_2025"] - bridge["p_bmi_25_30_ncd_2024"])
    bridge["obesity_jump_pp"] = 100 * (bridge["p_obesity_bmi_ge_30"] - bridge["p_obesity_ncd_2024"])
    bridge.to_csv(tables / "ncd2024_to_gbd2025_absolute_anchor_bridge_diagnostics_v1.csv.gz", index=False, compression="gzip")

    audit = {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "crc_total": total_summary,
        "crc_high_bmi": attributable_summary,
        "gbd_bmi": {
            "rows_all_years_country_male_female": len(gbd),
            "rows_2025_2050": len(gbd_future),
            "countries": int(gbd["location_id"].nunique()),
            "years": [int(gbd["year"].min()), int(gbd["year"].max())],
            "ages": int(gbd["age_group_id"].nunique()),
            "sexes": sorted(gbd["sex"].unique().tolist()),
            "overweight_definition": "BMI >=25 kg/m2",
            "overweight_less_obesity_definition": "25 <= BMI <30 kg/m2",
            "obesity_definition": "BMI >=30 kg/m2",
            "ui_note": "Published lower/upper fields are marginal uncertainty intervals; they are retained but not treated as joint compositional draws in this stage.",
        },
        "ncd_risc": {
            "rows": len(ncd),
            "countries": int(ncd["iso3"].nunique()),
            "years": [int(ncd["year"].min()), int(ncd["year"].max())],
            "age_groups_used": sorted(ncd["ncd_age_group"].unique().tolist()),
            "sexes": sorted(ncd["sex"].unique().tolist()),
        },
        "anchor_method": {
            "primary": "GBD absolute 25-<30 and >=30 prevalence; within-block BMI7 shares extrapolated in ALR space using empirical-Bayes-shrunk 2005-2024 NCD-RisC slopes.",
            "sensitivity": "NCD-RisC 2024 macro composition plus GBD 2024-to-future ALR change.",
            "slope_extrapolation_caps": "Age-sex-specific 0.5th and 99.5th percentiles of observed 1990-2024 ALRs; slopes clipped to age-sex 1st-99th percentile of country slopes.",
            "gbd_only_locations": crosswalk.loc[crosswalk["iso3"].eq(""), "location_name"].tolist(),
            "age_85plus_rule": "NCD-RisC 85+ internal shares reused for GBD 85-89, 90-94 and 95+.",
        },
        "qa_status": "PASS" if not (qa["status"] == "FAIL").any() else "FAIL",
    }
    atomic_json(output / "content_audit_and_anchor_metadata_v1.json", audit)

    report = f"""# GBD/NCD-RisC content audit and national BMI7 anchor build v1

## Scope

This stage audited the frozen GBD adult BMI forecast and colorectal-cancer tables, then built national-age-sex BMI7 anchors for 2025-2050. It did not fit the future spatial MRP or create disease-burden maps.

## Input conclusions

- GBD adult BMI forecast: {gbd['location_id'].nunique()} countries/territories, 15 five-year age groups from 25-29 to 95+, male and female, 1990-2050.
- GBD definitions were confirmed from the machine-readable metric labels: `overweight` = BMI >=25; `overweight_less_obesity` = BMI 25-<30; `obesity` = BMI >=30.
- CRC total burden: {total_summary['rows']:,} rows, with incidence, deaths and DALYs, number and rate, 1990-2021.
- High-BMI-attributable CRC: {attributable_summary['rows']:,} rows, with deaths and DALYs, number and rate, 1990-2021.
- NCD-RisC: {ncd['iso3'].nunique()} countries, 13 adult age groups from 25-29 to 85+, male and female, 1990-2024, seven mutually exclusive BMI categories.

## Anchor construction

The primary anchor exactly preserves the GBD mean prevalence of BMI 25-<30 and BMI >=30. The remaining probability is distributed across the three BMI <25 categories using NCD-RisC within-block shares; BMI >=30 is divided into 30-<35, 35-<40 and >=40 using the corresponding NCD-RisC shares. Within-block shares are extrapolated in ALR space using empirical-Bayes-shrunk country trends estimated over 2005-2024 and bounded by observed age-sex distributions.

A continuity sensitivity anchor is also provided. It starts from the NCD-RisC 2024 macro composition and applies the GBD 2024-to-future change in coherent three-part ALR space. This sensitivity series is not the GBD absolute anchor.

## Explicit limitations and flags

- NCD-RisC has one 85+ group. Its internal BMI shares are reused for GBD 85-89, 90-94 and 95+, with an explicit row-level flag.
- {len(audit['anchor_method']['gbd_only_locations'])} GBD locations have no NCD-RisC country match: {', '.join(audit['anchor_method']['gbd_only_locations'])}. Their within-block shares use the global age-sex median, are renormalized within each BMI block and are flagged.
- The high-BMI-attributable CRC source contains {attributable_summary['negative_numeric_cells']:,} negative numeric cells, all in rounded lower uncertainty bounds; point estimates are non-negative. Source values are retained unchanged and must be truncated at zero only during uncertainty sampling.
- GBD lower/upper uncertainty intervals are marginal intervals. They are retained in the normalized input but are not treated as joint compositional draws.
- The primary absolute GBD anchor can show a 2024-to-2025 level discontinuity relative to NCD-RisC. The bridge diagnostic table quantifies this; the continuity sensitivity series is provided for comparison.

## QA

- Overall status: **{audit['qa_status']}**
- Detailed checks: `tables/content_audit_checks_v1.csv`
- Country crosswalk: `tables/gbd_bmi_to_ncd_country_crosswalk_v1.csv`
- Bridge diagnostics: `tables/ncd2024_to_gbd2025_absolute_anchor_bridge_diagnostics_v1.csv.gz`
"""
    (output / "REPORT_CN_v1.md").write_text(report, encoding="utf-8")

    artifact_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            artifact_rows.append({"relative_path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(artifact_rows).to_csv(output / "artifact_sha256_v1.csv", index=False, encoding="utf-8-sig")
    # Recompute the checksum file once so it also records the final report/tables; it intentionally does not hash itself.
    artifact_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256_v1.csv":
            artifact_rows.append({"relative_path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(artifact_rows).to_csv(output / "artifact_sha256_v1.csv", index=False, encoding="utf-8-sig")

    if audit["qa_status"] != "PASS":
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": "PASS", "output": str(output), "anchor_rows": len(anchors)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
