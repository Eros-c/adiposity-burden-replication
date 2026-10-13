#!/usr/bin/env python3
"""Harmonize GBD 2021 total burden, high-BMI attributable and Foresight reference exports for 33 adiposity-associated causes.

Inputs (per cause, one CSV each):
  - data/GBD2021_downloads/results_total_burden  (Chinese columns, 2015-2021, Number)
  - data/GBD2021_downloads/results_high_bmi      (Chinese columns, 2015-2021, Number + Percent)
  - data/GBD2021_downloads/foresight             (English columns, 2022-2050, reference scenario, Number)

Outputs: three unified long parquet tables plus QA artifacts.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "11_all_disease_extension" / "01_harmonize" / "harmonized_gbd_v1_20261005"
CONTRACTS = ROOT / "contracts"
DATA = ROOT / "data"
QA = ROOT / "qa"
REPORTS = ROOT / "reports"
SCRIPTS = ROOT / "scripts"

SOURCE = Path(_cfg.DATA / "GBD2021_downloads")
LOCATION_CROSSWALK = (
    PROJECT
    / "09_colorectal_cancer_burden"
    / "crc_attributable_deaths_dalys_2025_2050_four_ssp_v1_20260910"
    / "contracts"
    / "GBD_location_id_ISO3_PAF_crosswalk_v1.csv"
)
DISEASE_LIST = CONTRACTS / "diseases_v1.csv"

AGES = (
    "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+",
)
AGE_MAP = {f"{age}岁": age for age in AGES[:-1]} | {">95岁": "95+"}
SEX_MAP = {"男性": "Male", "女性": "Female"}
MEASURE_MAP = {"死亡": "Deaths", "伤残调整生命年": "DALYs"}
METRIC_MAP = {"数量": "Number", "百分比": "Percent"}
EXCLUDED_CAUSE = 996  # Liver cancer due to NASH: no official high-BMI attributable data

DATASETS = {
    "total_burden": SOURCE / "results_total_burden",
    "high_bmi_attributable": SOURCE / "results_high_bmi",
    "foresight_reference": SOURCE / "foresight",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False), errors="coerce"
    )


def load_disease_list() -> pd.DataFrame:
    frame = pd.read_csv(DISEASE_LIST, encoding="utf-8-sig")
    frame = frame[frame["cause_id"] != EXCLUDED_CAUSE].copy()
    frame["cause_id"] = frame["cause_id"].astype(int)
    if frame["cause_id"].duplicated().any():
        raise RuntimeError("Duplicate cause_id in disease list")
    return frame


def load_location_crosswalk() -> pd.DataFrame:
    frame = pd.read_csv(LOCATION_CROSSWALK, encoding="utf-8-sig")
    required = {"location_id", "foresight_location_name", "iso3", "crosswalk_location_name"}
    if not required.issubset(frame.columns) or frame["location_id"].duplicated().any():
        raise RuntimeError("Invalid frozen GBD location crosswalk")
    return frame


def find_cause_file(directory: Path, cause_id: int) -> Path:
    matches = sorted(directory.glob(f"C{cause_id}_*.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one file for cause {cause_id} in {directory}, found {len(matches)}")
    return matches[0]


def standardize_chinese(frame: pd.DataFrame, cause_id: int, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Standardize results_total_burden / results_high_bmi (Chinese columns)."""
    expected = {"测量", "指标", "原因", "地理位置", "年龄", "性别", "年份", "数值", "上", "降低"}
    missing = expected - set(frame.columns)
    if missing:
        raise RuntimeError(f"Cause {cause_id}: missing Chinese columns {missing}")
    frame = frame[
        frame["测量"].isin(MEASURE_MAP)
        & frame["指标"].isin(METRIC_MAP)
        & frame["年龄"].isin(AGE_MAP)
        & frame["性别"].isin(SEX_MAP)
        & (frame["地理位置"] != "全球")
    ].copy()
    frame["cause_id"] = cause_id
    frame["year"] = frame["年份"].astype(int)
    frame["measure"] = frame["测量"].map(MEASURE_MAP)
    frame["metric"] = frame["指标"].map(METRIC_MAP)
    frame["age_group"] = frame["年龄"].map(AGE_MAP)
    frame["sex"] = frame["性别"].map(SEX_MAP)
    is_percent = frame["metric"] == "Percent"
    frame.loc[~is_percent, "value"] = parse_number(frame.loc[~is_percent, "数值"])
    frame.loc[is_percent, "value"] = parse_number(
        frame.loc[is_percent, "数值"].str.rstrip("%")
    )
    frame["lower"] = np.where(
        is_percent,
        parse_number(frame["降低"].str.rstrip("%")),
        parse_number(frame["降低"]),
    )
    frame["upper"] = np.where(
        is_percent,
        parse_number(frame["上"].str.rstrip("%")),
        parse_number(frame["上"]),
    )
    frame = frame.merge(
        crosswalk[["location_id", "foresight_location_name", "iso3", "crosswalk_location_name"]],
        left_on="地理位置",
        right_on="foresight_location_name",
        how="left",
        validate="many_to_one",
    )
    return frame[
        ["cause_id", "location_id", "iso3", "crosswalk_location_name", "year",
         "age_group", "sex", "measure", "metric", "value", "lower", "upper"]
    ]


def standardize_foresight(frame: pd.DataFrame, cause_id: int, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Standardize foresight exports (English columns, Chinese values)."""
    frame = frame[
        (frame["scenario_name"] == "参考")
        & frame["measure_name"].isin(MEASURE_MAP)
        & (frame["metric_name"] == "数量")
        & frame["age_name"].isin(AGE_MAP)
        & frame["sex_name"].isin(SEX_MAP)
        & (frame["location_name"] != "全球")
    ].copy()
    frame["cause_id"] = cause_id
    frame["year"] = frame["year"].astype(int)
    frame["measure"] = frame["measure_name"].map(MEASURE_MAP)
    frame["metric"] = "Number"
    frame["age_group"] = frame["age_name"].map(AGE_MAP)
    frame["sex"] = frame["sex_name"].map(SEX_MAP)
    frame["value"] = parse_number(frame["value"])
    frame["lower"] = parse_number(frame["lower"])
    frame["upper"] = parse_number(frame["upper"])
    source_location_id = frame["location_id"].astype(int).to_numpy()
    frame = frame.drop(columns=["location_id"]).merge(
        crosswalk[["location_id", "foresight_location_name", "iso3", "crosswalk_location_name"]],
        left_on="location_name",
        right_on="foresight_location_name",
        how="left",
        validate="many_to_one",
    )
    mismatched = frame["location_id"].notna().to_numpy() & (
        frame["location_id"].to_numpy() != source_location_id
    )
    if mismatched.any():
        raise RuntimeError(
            f"Cause {cause_id}: foresight location_id disagrees with frozen crosswalk for "
            f"{frame.loc[mismatched, 'location_name'].unique().tolist()}"
        )
    return frame[
        ["cause_id", "location_id", "iso3", "crosswalk_location_name", "year",
         "age_group", "sex", "measure", "metric", "value", "lower", "upper"]
    ]


def qa_frame(name: str, frame: pd.DataFrame, cause_id: int) -> dict:
    keys = ["cause_id", "location_id", "year", "age_group", "sex", "measure", "metric"]
    record = {
        "dataset": name,
        "cause_id": cause_id,
        "rows": int(len(frame)),
        "locations": int(frame["location_id"].nunique()),
        "years": f"{int(frame['year'].min())}-{int(frame['year'].max())}",
        "age_groups": int(frame["age_group"].nunique()),
        "measures": sorted(frame["measure"].unique().tolist()),
        "metrics": sorted(frame["metric"].unique().tolist()),
        "duplicate_key_rows": int(frame.duplicated(subset=keys).sum()),
        "null_value_rows": int(frame["value"].isna().sum()),
        "unmapped_location_rows": int(frame["location_id"].isna().sum()),
    }
    if record["duplicate_key_rows"] or record["null_value_rows"] or record["unmapped_location_rows"]:
        record["status"] = "FAIL"
    else:
        record["status"] = "PASS"
    return record


def main() -> None:
    diseases = load_disease_list()
    crosswalk = load_location_crosswalk()
    outputs: list[Path] = []
    qa_records: list[dict] = []
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in DATASETS}

    for _, row in diseases.iterrows():
        cause_id = int(row["cause_id"])
        for name, directory in DATASETS.items():
            path = find_cause_file(directory, cause_id)
            raw = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
            if name == "foresight_reference":
                frame = standardize_foresight(raw, cause_id, crosswalk)
            else:
                frame = standardize_chinese(raw, cause_id, crosswalk)
            frames[name].append(frame)
            qa_records.append(qa_frame(name, frame, cause_id))

    written = []
    for name, parts in frames.items():
        combined = pd.concat(parts, ignore_index=True)
        combined["location_id"] = combined["location_id"].astype("Int64")
        combined["year"] = combined["year"].astype(int)
        out_path = DATA / f"gbd_{name}_harmonized_v1.parquet"
        combined.to_parquet(out_path, index=False)
        outputs.append(out_path)
        written.append(
            {
                "dataset": name,
                "path": str(out_path),
                "rows": int(len(combined)),
                "causes": int(combined["cause_id"].nunique()),
                "sha256": sha256(out_path),
            }
        )

    qa_frame_out = pd.DataFrame(qa_records)
    qa_csv = QA / "harmonize_per_cause_QA_v1.csv"
    qa_frame_out.to_csv(qa_csv, index=False, encoding="utf-8-sig")
    outputs.append(qa_csv)

    failures = qa_frame_out[qa_frame_out["status"] != "PASS"]
    summary = {
        "run": "harmonized_gbd_v1_20261005",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "diseases_expected": int(len(diseases)),
        "excluded_causes": [EXCLUDED_CAUSE],
        "datasets": written,
        "qa_rows": int(len(qa_frame_out)),
        "qa_failures": int(len(failures)),
        "location_crosswalk_sha256": sha256(LOCATION_CROSSWALK),
        "status": "PASS" if len(failures) == 0 else "FAIL",
    }
    qa_json = QA / "harmonize_QA_v1.json"
    write_json(qa_json, summary)
    outputs.append(qa_json)

    report = REPORTS / "harmonize_report_v1.md"
    lines = [
        "# GBD harmonization report (33 causes)",
        "",
        f"- diseases: {len(diseases)} (excluded cause {EXCLUDED_CAUSE}: liver cancer due to NASH)",
        f"- location crosswalk: {LOCATION_CROSSWALK.name} (sha256 {summary['location_crosswalk_sha256'][:16]}…)",
        "",
        "## Datasets",
    ]
    for item in written:
        lines.append(
            f"- `{Path(item['path']).name}`: {item['rows']:,} rows, "
            f"{item['causes']} causes, sha256 {item['sha256'][:16]}…"
        )
    lines += [
        "",
        "## QA",
        f"- per-cause QA rows: {len(qa_frame_out)}; failures: {len(failures)}",
        f"- overall status: {summary['status']}",
    ]
    if len(failures):
        lines.append("\n## Failures\n")
        lines.append(failures.to_string(index=False))
    report.write_text("\n".join(lines), encoding="utf-8")
    outputs.append(report)

    archived = SCRIPTS / Path(__file__).name
    archived.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    outputs.append(archived)

    manifest = ROOT / "harmonized_gbd_formal_manifest_v1.csv"
    pd.DataFrame(
        {"path": [str(p) for p in outputs], "sha256": [sha256(p) for p in outputs]}
    ).to_csv(manifest, index=False, encoding="utf-8-sig")

    if summary["status"] != "PASS":
        raise RuntimeError(f"Harmonization QA failed; see {qa_csv}")
    (ROOT / "COMPLETE.flag").write_text(
        f"harmonized_gbd_v1_20261005 completed {summary['completed_at_utc']}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
