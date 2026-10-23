"""Read-only audit of CRC dose-response, GBD burden, and BMI products."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(_cfg.RESULTS)
RR_ROOT = Path(_cfg.DATA / "肥胖-疾病剂量反应关系")
GBD_ROOT = Path(_cfg.DATA / "GBD2021_query_table_exports_camoufox" / "merged")
OUT = PROJECT / "01_input_inventory" / "crc_paf_input_audit_v1_20260910"
TOTAL = GBD_ROOT / "C013_total_burden.csv"
ATTRIB = GBD_ROOT / "C013_high_bmi_attributable.csv"
HIST_ORIGINAL = Path(_cfg.ROOT / "results" / "降尺度" / "experiments" / "BMI7_MRP_STEPS_regional_augmentation_20260812" / "regional_steps_mapping_v1" / "data")
HIST_25 = PROJECT / "05_historical_bmi_1990_2024" / "historical_25plus_compat_v1_20260909"
FUT_01 = PROJECT / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906" / "outputs"
FUT_05 = PROJECT / "06_future_bmi_2025_2050" / "full_covariate_annual_0p5deg_production_v1_20260906" / "outputs"
FUT_ADM1 = PROJECT / "06_future_bmi_2025_2050" / "phase7_multiscale_delivery_v3_20260908" / "04_adm1" / "tables"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def audit_gbd(path: Path, attributable: bool) -> dict:
    unique = defaultdict(set)
    dimension_counts = Counter()
    row_count = 0
    numeric_missing = Counter()
    numeric_min = {x: math.inf for x in ("数值", "上", "降低")}
    numeric_max = {x: -math.inf for x in ("数值", "上", "降低")}
    interval_qa = Counter()
    negative_lower_examples = []
    seen_hashes: set[int] = set()
    duplicate_keys = 0
    columns = None
    key_columns = ["Population Group", "测量", "指标"] + (["风险"] if attributable else []) + ["原因", "地理位置", "年龄", "性别", "年份"]
    dimension_columns = key_columns
    for chunk in pd.read_csv(path, dtype=str, chunksize=100_000, low_memory=False):
        if columns is None:
            columns = chunk.columns.tolist()
        row_count += len(chunk)
        for column in dimension_columns:
            unique[column].update(chunk[column].dropna().astype(str).str.strip().tolist())
        grouped = chunk.groupby(["测量", "指标"], dropna=False).size()
        for key, count in grouped.items():
            dimension_counts[tuple(str(x) for x in key)] += int(count)
        for column in ("数值", "上", "降低"):
            numeric = parse_numeric(chunk[column])
            numeric_missing[column] += int(numeric.isna().sum())
            if numeric.notna().any():
                numeric_min[column] = min(numeric_min[column], float(numeric.min()))
                numeric_max[column] = max(numeric_max[column], float(numeric.max()))
        value = parse_numeric(chunk["数值"])
        upper = parse_numeric(chunk["上"])
        lower = parse_numeric(chunk["降低"])
        interval_qa["lower_negative"] += int(lower.lt(0).sum())
        interval_qa["lower_gt_value"] += int(lower.gt(value).sum())
        interval_qa["upper_lt_value"] += int(upper.lt(value).sum())
        interval_qa["upper_lt_lower"] += int(upper.lt(lower).sum())
        if len(negative_lower_examples) < 20:
            selected = chunk.loc[lower.lt(0), key_columns + ["数值", "上", "降低"]].head(20 - len(negative_lower_examples))
            negative_lower_examples.extend(selected.fillna("").to_dict("records"))
        hashes = pd.util.hash_pandas_object(chunk[key_columns].fillna("<NA>").astype(str), index=False).to_numpy(np.uint64)
        for value in hashes:
            integer = int(value)
            if integer in seen_hashes:
                duplicate_keys += 1
            else:
                seen_hashes.add(integer)

    years = sorted(int(x) for x in unique["年份"])
    expected_rows = 1
    for column in dimension_columns:
        expected_rows *= len(unique[column])
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "columns": columns,
        "row_count": row_count,
        "unique_key_count": len(seen_hashes),
        "duplicate_key_rows": duplicate_keys,
        "cartesian_expected_rows": expected_rows,
        "complete_cartesian_product": row_count == expected_rows and duplicate_keys == 0,
        "population_groups": sorted(unique["Population Group"]),
        "measures": sorted(unique["测量"]),
        "metrics": sorted(unique["指标"]),
        "measure_metric_row_counts": [{"measure": k[0], "metric": k[1], "rows": v} for k, v in sorted(dimension_counts.items())],
        "risks": sorted(unique["风险"]) if attributable else [],
        "causes": sorted(unique["原因"]),
        "location_count": len(unique["地理位置"]),
        "locations": sorted(unique["地理位置"]),
        "ages": sorted(unique["年龄"]),
        "sexes": sorted(unique["性别"]),
        "years": years,
        "year_min": min(years),
        "year_max": max(years),
        "years_contiguous": years == list(range(min(years), max(years) + 1)),
        "numeric_missing": dict(numeric_missing),
        "numeric_min": numeric_min,
        "numeric_max": numeric_max,
        "interval_qa": dict(interval_qa),
        "negative_lower_examples": negative_lower_examples,
        "geography_fields": ["地理位置"],
        "location_id_present": False,
    }


def dose_candidates() -> pd.DataFrame:
    records = []
    summary = pd.read_csv(RR_ROOT / "obesity_disease_rr_summary.csv", encoding="utf-8-sig", dtype=str)
    row = summary[summary["matched_ihme_outcome"].eq("Colon and rectum cancer")].iloc[0]
    records.append({
        "candidate": "IHME GBD 2019 summary median",
        "disease_outcome": "Colon and rectum cancer",
        "exposure_definition": "Higher BMI",
        "effect_measure": "RR",
        "effect_unit": row["unit"],
        "sex": row["sex_scope"],
        "age_specific": row["age_specific"],
        "outcome_scope": row["measure_scope"],
        "point": float(row["rr_per_5_bmi"]),
        "lower": float(row["rr_lci"]),
        "upper": float(row["rr_uci"]),
        "dose_response": "Per-5 kg/m2 increment; local structured table does not encode curve shape",
        "nonlinear": "not specified",
        "reference_bmi_tmrel": "not present",
        "source": row["source_name"],
        "source_file": row["source_file_or_url"],
        "note": "Median across the two sex-specific values; not preferred when sex-specific BMI is available.",
    })

    long = pd.read_csv(RR_ROOT / "obesity_disease_rr_long.csv", encoding="utf-8-sig", dtype=str)
    long = long[long["ihme_outcome"].eq("Colon and rectum cancer")].copy()
    for values, group in long.groupby(["sex", "rr_per_5_bmi", "lci", "uci", "source_name", "source_file", "source_url"], dropna=False):
        sex, rr, lower, upper, source, source_file, source_url = values
        records.append({
            "candidate": f"IHME GBD 2019 sex-specific ({sex})",
            "disease_outcome": "Colon and rectum cancer",
            "exposure_definition": "Higher BMI",
            "effect_measure": "RR",
            "effect_unit": "per 5 kg/m2 higher BMI",
            "sex": sex,
            "age_specific": "False (age field blank)",
            "outcome_scope": str(group["measure"].iloc[0]),
            "point": float(rr),
            "lower": float(lower),
            "upper": float(upper),
            "dose_response": "Per-5 kg/m2 increment; local structured table does not encode curve shape",
            "nonlinear": "not specified",
            "reference_bmi_tmrel": "not present",
            "source": source,
            "source_file": source_file,
            "note": f"Exact duplicate rows in long table: {len(group)}; deduplicate before crosswalk.",
        })

    non_gbd = pd.read_excel(RR_ROOT / "肥胖_疾病剂量反应_非GBD权威来源_v2.xlsx", sheet_name="剂量反应主表", dtype=str)
    row = non_gbd[non_gbd["疾病英文名"].str.contains("Colorectal", case=False, na=False)].iloc[0]
    records.append({
        "candidate": "WCRF/AICR CUP overall CRC",
        "disease_outcome": row["疾病英文名"],
        "exposure_definition": row["暴露/剂量尺度"],
        "effect_measure": row["效应指标"],
        "effect_unit": "per 5 kg/m2 higher BMI",
        "sex": "Both sexes",
        "age_specific": "No",
        "outcome_scope": "CRC incidence/risk",
        "point": float(row["效应值"]),
        "lower": float(row["95%CI下限"]),
        "upper": float(row["95%CI上限"]),
        "dose_response": "Continuous per-5 BMI RR; report also documents a nonlinear association",
        "nonlinear": "Yes; p<=0.01, with a greater increase above BMI 27 kg/m2",
        "reference_bmi_tmrel": "not specified in structured row; nonlinear curve coefficients are not machine-readable",
        "source": row["作者/机构和年份"] + "; " + row["来源题名"],
        "source_file": str(RR_ROOT / "WCRF_Body-fatness-and-weight-gain_2024.pdf") + " (report pages 43-45)",
        "note": "Report gives overall RR 1.05; stratified RR 1.08 for men and 1.05 for women per 5 kg/m2.",
    })

    association = pd.read_excel(RR_ROOT / "BMI_疾病关联.xlsx", sheet_name="GBD完整主表", dtype=str)
    row = association[association["Disease"].eq("Colon and rectum cancer")].iloc[0]
    records.append({
        "candidate": "Watts et al. meta-analysis",
        "disease_outcome": row["Disease"],
        "exposure_definition": row["效应单位"],
        "effect_measure": "RR/HR as locally summarized",
        "effect_unit": "per 5 kg/m2 higher BMI",
        "sex": "not specified in local row",
        "age_specific": "not specified",
        "outcome_scope": row["结局类型"],
        "point": float(row["RR_HR"]),
        "lower": float(row["95CI下限"]),
        "upper": float(row["95CI上限"]),
        "dose_response": "Per-5 kg/m2 increment",
        "nonlinear": "not specified in local row",
        "reference_bmi_tmrel": "not present",
        "source": row["来源"],
        "source_file": str(RR_ROOT / "BMI_疾病关联.xlsx"),
        "note": "Citation and exact outcome definition require verification before parameter freezing.",
    })

    legacy = pd.read_csv(RR_ROOT / "obesity_disease_doseresponse.csv", encoding="gb18030", dtype=str)
    for _, row in legacy[legacy["Disease"].str.contains("Colorectal cancer", case=False, na=False)].iterrows():
        records.append({
            "candidate": "Renehan et al. 2008 colon-specific",
            "disease_outcome": row["Disease"],
            "exposure_definition": "BMI increment",
            "effect_measure": "RR",
            "effect_unit": "per 5 kg/m2 higher BMI",
            "sex": "Women" if "women" in row["Disease"].lower() else "Men",
            "age_specific": "not specified",
            "outcome_scope": "Colon cancer; not identical to combined colon and rectum cancer",
            "point": float(row["RR_per_5_BMI_units"]),
            "lower": float(row["CI_lower"]),
            "upper": float(row["CI_upper"]),
            "dose_response": "Per-5 kg/m2 increment",
            "nonlinear": "not specified",
            "reference_bmi_tmrel": "not present",
            "source": row["Source"],
            "source_file": str(RR_ROOT / "obesity_disease_doseresponse.csv"),
            "note": row["Notes"],
        })
    return pd.DataFrame(records)


def years_under(root: Path) -> dict[str, list[int]]:
    result = {}
    for scenario in sorted([p for p in root.iterdir() if p.is_dir()]):
        result[scenario.name] = sorted(int(p.name) for p in scenario.iterdir() if p.is_dir() and p.name.isdigit())
    return result


def bmi_readiness() -> tuple[pd.DataFrame, dict]:
    rows = [
        {"period": "historical original", "years": "2015-2024 annual", "age_scope": "20+", "age_groups": "20-29,30-39,40-49,50-59,60-69,70+", "sex": "Male,Female", "bmi7": True, "population": True, "resolution": "0.1°,0.5°,ADM1", "status": "available but not aligned to the 25+ GBD five-year burden interface"},
        {"period": "historical 25+ compatibility", "years": "2015,2020,2024", "age_scope": "25+", "age_groups": "25-29 through 80-84 plus pooled 85+", "sex": "Female,Male", "bmi7": True, "population": True, "resolution": "0.1°", "status": "usable for key years; 85-89/90-94/95+ require shared 85+ exposure; no annual 1990-2021 series"},
        {"period": "future formal", "years": "2025-2050 annual", "age_scope": "25+", "age_groups": "25-29 through 95+ (15 groups)", "sex": "Female,Male", "bmi7": True, "population": True, "resolution": "0.1°,0.5°,ADM1", "status": "ready for CRC point-estimate pilot and four-SSP analysis"},
    ]
    detail = {"future_0p1_years": years_under(FUT_01), "future_0p5_years": years_under(FUT_05), "future_adm1_years": years_under(FUT_ADM1), "historical_25plus_years": sorted(int(p.name) for p in (HIST_25 / "data" / "0p1deg").iterdir() if p.is_dir() and p.name.isdigit())}
    return pd.DataFrame(rows), detail


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = dose_candidates()
    candidates.to_csv(OUT / "CRC_dose_response_candidates_audit_v1.csv", index=False, encoding="utf-8-sig")

    total = audit_gbd(TOTAL, attributable=False)
    attributable = audit_gbd(ATTRIB, attributable=True)
    (OUT / "GBD_CRC_total_structure_v1.json").write_text(json.dumps(total, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "GBD_CRC_high_BMI_attributable_structure_v1.json").write_text(json.dumps(attributable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame({"location": total["locations"]}).to_csv(OUT / "GBD_CRC_total_locations_v1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"location": attributable["locations"]}).to_csv(OUT / "GBD_CRC_high_BMI_locations_v1.csv", index=False, encoding="utf-8-sig")

    bmi, bmi_detail = bmi_readiness()
    bmi.to_csv(OUT / "BMI_products_CRC_readiness_v1.csv", index=False, encoding="utf-8-sig")
    (OUT / "BMI_products_CRC_readiness_detail_v1.json").write_text(json.dumps(bmi_detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    files = [TOTAL, ATTRIB, RR_ROOT / "obesity_disease_rr_long.csv", RR_ROOT / "obesity_disease_rr_summary.csv", RR_ROOT / "肥胖_疾病剂量反应_非GBD权威来源_v2.xlsx", RR_ROOT / "BMI_疾病关联.xlsx", RR_ROOT / "WCRF_Body-fatness-and-weight-gain_2024.pdf"]
    pd.DataFrame([{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]).to_csv(OUT / "CRC_input_SHA256_v1.csv", index=False, encoding="utf-8-sig")

    issues = [
        "No candidate dose-response table contains an explicit reference BMI or TMREL.",
        "WCRF reports nonlinearity and a steeper association above BMI 27, but no machine-readable spline coefficients are present locally.",
        "Candidate RRs conflict by source, sex, disease granularity, and outcome definition; a parameter decision is required before crosswalk construction.",
        "GBD burden files contain location names but no location_id/ISO3, so a frozen location crosswalk is required.",
        "GBD total burden is national/territory level; attributable burden must initially remain at that level.",
        "Historical GBD burden spans 1990-2021, but the model's original spatial BMI output begins in 2015; the harmonized 25+ output currently contains only 2015, 2020, and 2024.",
        "Historical 85+ BMI exposure must be replicated to 85-89, 90-94, and 95+ only with explicit QA and age-specific population/burden retained.",
        "The total burden file includes incidence, deaths, and DALYs, but high-BMI attributable data include only deaths and DALYs; attributable incidence must be calculated from the study PAF and total incidence.",
        "An incidence RR must not automatically be applied to death or DALY burden without a frozen causal-effect assumption or an outcome-compatible RR.",
    ]
    audit = {
        "status": "INPUT_AUDIT_COMPLETE_PAF_NOT_CALCULATED",
        "dose_response_candidate_count": len(candidates),
        "gbd_total": {k: total[k] for k in ["row_count", "measures", "metrics", "location_count", "ages", "sexes", "year_min", "year_max", "complete_cartesian_product", "duplicate_key_rows"]},
        "gbd_high_bmi_attributable": {k: attributable[k] for k in ["row_count", "measures", "metrics", "risks", "location_count", "ages", "sexes", "year_min", "year_max", "complete_cartesian_product", "duplicate_key_rows"]},
        "locations_identical_between_gbd_files": total["locations"] == attributable["locations"],
        "bmi_products": bmi_detail,
        "issues": issues,
        "usage_contract": {"total_burden": "baseline burden for study PAF-derived attributable burden", "high_bmi_attributable": "external comparison only; never PAF input or calibration"},
        "next_step_recommendation": "Freeze a sex-aware CRC RR crosswalk after choosing between non-GBD WCRF incidence RRs and IHME sex-specific RRs, documenting reference/TMREL and whether the same RR is permitted for incidence, deaths, and DALYs.",
    }
    (OUT / "CRC_input_audit_QA_v1.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    measure_total = ", ".join(total["measures"])
    measure_attr = ", ".join(attributable["measures"])
    report = f"""# CRC PAF input audit

Status: **INPUT AUDIT COMPLETE; NO PAF CALCULATED**

## Dose-response evidence

The local archive contains conflicting candidate parameters. Structured IHME GBD 2019 records give sex-specific per-5-kg/m2 RRs (male 1.177, 95% CI 1.145-1.208; female 1.059, 1.031-1.083), with no age specificity and no explicit reference BMI/TMREL. WCRF/AICR reports an overall CRC RR of 1.05 (1.03-1.07) per 5 kg/m2, male 1.08 (1.04-1.11), female 1.05 (1.02-1.08), and a nonlinear association that becomes steeper above BMI 27 kg/m2; local files do not contain machine-readable spline coefficients. A later locally summarized meta-analysis reports 1.10 (1.08-1.12). These must not be averaged without a prespecified evidence rule.

## GBD CRC total burden

- File: `{TOTAL}`
- Rows: {total['row_count']:,}; unique locations: {total['location_count']}
- Measures: {measure_total}
- Metrics: {', '.join(total['metrics'])}
- Years: {total['year_min']}-{total['year_max']} (contiguous: {total['years_contiguous']})
- Ages: {', '.join(total['ages'])}
- Sexes: {', '.join(total['sexes'])}
- Location identifier: name only; no location_id/ISO3.

## GBD high-BMI-attributable CRC burden

- File: `{ATTRIB}`
- Rows: {attributable['row_count']:,}; unique locations: {attributable['location_count']}
- Measures: {measure_attr}
- Metrics: {', '.join(attributable['metrics'])}
- Risk: {', '.join(attributable['risks'])}
- Years: {attributable['year_min']}-{attributable['year_max']}
- Contract: external benchmark only, not an input to the study PAF.

## BMI compatibility

Formal future BMI products are complete for all four SSPs, annually from 2025 to 2050, at 0.1 degree, 0.5 degree, and ADM1, with 15 age groups and both sexes. They are ready for a 2025/2050 pilot. The original historical 2015-2024 products use a 20+ broad-age contract; the harmonized 25+ re-export currently covers 2015, 2020, and 2024 at 0.1 degree and retains a pooled 85+ exposure.

## Blocking issues before the RR crosswalk

""" + "\n".join(f"- {item}" for item in issues) + "\n"
    (OUT / "CRC_input_audit_report_v1.md").write_text(report, encoding="utf-8")
    (OUT / "COMPLETE.flag").write_text("CRC_INPUT_AUDIT_COMPLETE_NO_PAF\n", encoding="ascii")
    print(json.dumps({"status": audit["status"], "total_rows": total["row_count"], "attributable_rows": attributable["row_count"], "locations": total["location_count"], "dose_candidates": len(candidates)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
