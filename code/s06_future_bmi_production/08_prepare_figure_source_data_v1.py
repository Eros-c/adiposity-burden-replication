from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(_cfg.RESULTS / "06_future_bmi_2025_2050" / "phase7_multiscale_delivery_v3_20260908")
TABLES = ROOT / "04_adm1" / "tables"
OUT = ROOT / "05_figures" / "source_data"
SCENARIOS = ("ssp126", "ssp245", "ssp370", "ssp585")
YEARS = (2025, 2030, 2040, 2050)
BMI = ("bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30", "bmi_30_35", "bmi_35_40", "bmi_ge_40")
MIDPOINT = np.array([17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5])
AGE_BINS = {
    "25-39": ("25-29", "30-34", "35-39"),
    "40-49": ("40-44", "45-49"),
    "50-59": ("50-54", "55-59"),
    "60-69": ("60-64", "65-69"),
    "70-79": ("70-74", "75-79"),
    "80+": ("80-84", "85-89", "90-94", "95+"),
}


def read_table(scenario: str, year: int, family: str) -> pd.DataFrame:
    path = TABLES / scenario / str(year) / f"BMI7_{scenario}_{year}_ADM1_{family}_adults25plus_v2.parquet"
    return pd.read_parquet(path)


def add_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["obesity_pct"] = frame["obesity"] * 100.0
    frame["overweight_obesity_pct"] = frame["overweight_obesity"] * 100.0
    frame["bmi_ge40_pct"] = frame["bmi_ge_40"] * 100.0
    return frame


def aggregate_age(frame: pd.DataFrame, include_sex: bool) -> pd.DataFrame:
    keys = ["zone_id", "adm1_uid", "adm1_code", "iso_3166_2", "adm0_a3", "adm1_name", "name", "type_en", "region", "region_sub", "ne_id", "scenario_code", "scenario", "year"]
    if include_sex:
        keys.append("sex")
    rows = []
    for age_label, age_values in AGE_BINS.items():
        x = frame[frame["age_group"].isin(age_values)].copy()
        x["age_group"] = age_label
        rows.append(x)
    x = pd.concat(rows, ignore_index=True)
    group_keys = keys + ["age_group"]
    numerators = {c: (x[c] * x["adult_population"]).fillna(0.0) for c in BMI}
    grouped = x.groupby(group_keys, dropna=False, sort=False)["adult_population"].sum().reset_index()
    for c in BMI:
        grouped[c] = x.assign(_num=numerators[c]).groupby(group_keys, dropna=False, sort=False)["_num"].sum().to_numpy() / grouped["adult_population"].replace(0, np.nan).to_numpy()
    grouped["obesity"] = grouped[["bmi_30_35", "bmi_35_40", "bmi_ge_40"]].sum(axis=1, min_count=3)
    grouped["overweight_obesity"] = grouped[["bmi_25_30", "bmi_30_35", "bmi_35_40", "bmi_ge_40"]].sum(axis=1, min_count=4)
    grouped["mean_bmi"] = grouped[list(BMI)].to_numpy(dtype=float) @ MIDPOINT
    return add_metrics(grouped)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    overall_rows = []
    sex_rows = []
    age_rows = []
    age_sex_rows = []
    for scenario in SCENARIOS:
        for year in YEARS:
            overall_rows.append(add_metrics(read_table(scenario, year, "overall")))
            sex_rows.append(add_metrics(read_table(scenario, year, "sex")))
            age_rows.append(aggregate_age(read_table(scenario, year, "age"), include_sex=False))
            age_sex_rows.append(aggregate_age(read_table(scenario, year, "age_sex"), include_sex=True))
            print(f"prepared {scenario}/{year}", flush=True)
    overall = pd.concat(overall_rows, ignore_index=True)
    sex = pd.concat(sex_rows, ignore_index=True)
    age = pd.concat(age_rows, ignore_index=True)
    age_sex = pd.concat(age_sex_rows, ignore_index=True)
    overall.to_csv(OUT / "adm1_overall_keyyears_v1.csv.gz", index=False)
    sex.to_csv(OUT / "adm1_sex_keyyears_v1.csv.gz", index=False)
    age.to_csv(OUT / "adm1_age_keyyears_broad6_v1.csv.gz", index=False)
    age_sex.to_csv(OUT / "adm1_age_sex_keyyears_broad6_v1.csv.gz", index=False)
    metadata = {
        "scenarios": list(SCENARIOS),
        "years": list(YEARS),
        "age_visualization_bins": {k: list(v) for k, v in AGE_BINS.items()},
        "source_tables": "Phase 7 ADM1 Parquet tables; complete population denominator retained",
        "map_mask_policy": "ADM1 polygons with adult_population <= 0 are shown as no-data; this affects display only",
        "bmi_midpoints": MIDPOINT.tolist(),
    }
    (OUT / "figure_source_data_contract_v1.json").write_text(__import__("json").dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SOURCE_DATA_PREP=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
