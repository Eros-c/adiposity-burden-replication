#!/usr/bin/env python3
"""Finalize the 104-task CRC PAF production after all tasks pass."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "08_colorectal_cancer_paf" / "full_annual_four_ssp_wcrf_primary_v1_20260910"
SCENARIOS = ("ssp126", "ssp245", "ssp370", "ssp585")
SCENARIO_LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
YEARS = tuple(range(2025, 2051))
KEY_YEARS = (2025, 2030, 2035, 2040, 2045, 2050)
AGES = ("25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tasks():
    summaries = []
    ages = []
    qa_rows = []
    national = []
    adm1 = []
    for scenario in SCENARIOS:
        for year in YEARS:
            state = json.loads((ROOT / "state" / scenario / f"{year}.json").read_text(encoding="utf-8"))
            if state.get("status") != "PASS":
                raise RuntimeError(f"Task not PASS: {scenario}/{year}")
            summary = json.loads((ROOT / "tables" / "task_summary" / scenario / f"{year}.json").read_text(encoding="utf-8"))
            summaries.append({
                "scenario_code": scenario, "scenario": summary["scenario"], "year": year,
                "global_paf": summary["global_paf"], "global_paf_rr_lower": summary["global_paf_rr_lower"], "global_paf_rr_upper": summary["global_paf_rr_upper"],
                "female_paf": summary["female_paf"], "male_paf": summary["male_paf"],
                "country_median": summary["country"]["median"], "country_p95": summary["country"]["p95"], "country_max": summary["country"]["max"],
                "adm1_median": summary["adm1"]["median"], "adm1_p95": summary["adm1"]["p95"], "adm1_max": summary["adm1"]["max"],
            })
            age = pd.read_csv(ROOT / "tables" / "task_global_age" / scenario / f"{year}.csv")
            ages.append(age)
            qa = json.loads((ROOT / "qa" / "tasks" / scenario / f"{year}.json").read_text(encoding="utf-8"))
            qa_rows.append({
                "scenario_code": scenario, "year": year,
                "closure_national": qa["national"]["probability_closure_max_abs_error"],
                "closure_adm1": qa["adm1"]["probability_closure_max_abs_error"],
                "closure_0p5": qa["raster_0p5"]["probability_closure_max_abs_error"],
                "closure_0p1": qa["raster_0p1"]["probability_closure_max_abs_error"],
                "error_0p1_vs_0p5": qa["cross_scale_global_abs_error_0p1_vs_0p5"],
                "error_0p5_vs_adm1": qa["cross_scale_global_abs_error_0p5_vs_adm1"],
                "national_vs_adm1_max": qa["national_vs_adm1"]["max_abs_paf_difference"],
                "matched_spatial_countries": qa["national_vs_adm1"]["matched_spatial_countries"],
            })
            national.append(pd.read_parquet(ROOT / "data" / "national" / scenario / str(year) / f"CRC_PAF_{scenario}_{year}_national_overall_WCRF_primary_v1.parquet", columns=["scenario_code", "scenario", "year", "iso3", "adult_population", "paf"]))
            adm1.append(pd.read_parquet(ROOT / "data" / "adm1" / scenario / str(year) / f"CRC_PAF_{scenario}_{year}_adm1_overall_WCRF_primary_v1.parquet", columns=["scenario_code", "scenario", "year", "adm1_uid", "adm0_a3", "adult_population", "paf"]))
    return pd.DataFrame(summaries), pd.concat(ages, ignore_index=True), pd.DataFrame(qa_rows), pd.concat(national, ignore_index=True), pd.concat(adm1, ignore_index=True)


def anomaly_table(frame: pd.DataFrame, id_columns: list[str], level: str) -> pd.DataFrame:
    data = frame.sort_values(id_columns + ["year"]).copy()
    data["annual_change"] = data.groupby(id_columns, observed=True)["paf"].diff()
    rows = []
    for scenario, group in data.groupby("scenario_code"):
        absolute = group["annual_change"].abs().dropna()
        median = float(absolute.median())
        mad = float(np.median(np.abs(absolute - median)))
        threshold = max(0.02, median + 10 * mad)
        flagged = group[group["annual_change"].abs() > threshold]
        for row in flagged.itertuples():
            item = {"level": level, "scenario_code": scenario, "year": int(row.year), "annual_change": float(row.annual_change), "threshold": threshold}
            for column in id_columns:
                item[column] = getattr(row, column)
            rows.append(item)
    return pd.DataFrame(rows)


def build_manifest():
    manifest_rows = []
    include_roots = [ROOT / "data", ROOT / "tables", ROOT / "qa", ROOT / "reports", ROOT / "state", ROOT / "contracts"]
    for base in include_roots:
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            if path.name == "CRC_PAF_full_formal_manifest_v1.csv":
                continue
            manifest_rows.append({"path": str(path), "relative_path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return pd.DataFrame(manifest_rows)


def main():
    summaries, ages, qa, national, adm1 = load_tasks()
    tables = ROOT / "tables"
    reports = ROOT / "reports"
    qa_dir = ROOT / "qa"
    tables.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True); qa_dir.mkdir(parents=True, exist_ok=True)

    summaries = summaries.sort_values(["scenario_code", "year"])
    summaries.to_csv(tables / "CRC_PAF_global_annual_four_SSP_WCRF_primary_v1.csv", index=False, encoding="utf-8-sig")
    summaries[summaries["year"].isin(KEY_YEARS)].to_csv(tables / "CRC_PAF_key_year_summary_WCRF_primary_v1.csv", index=False, encoding="utf-8-sig")
    summaries[summaries["year"].eq(2050)].to_csv(tables / "CRC_PAF_four_SSP_2050_summary_WCRF_primary_v1.csv", index=False, encoding="utf-8-sig")
    ages["age_order"] = ages["age_group"].map({age: index for index, age in enumerate(AGES)})
    ages.sort_values(["scenario_code", "year", "age_order"]).drop(columns="age_order").to_csv(tables / "CRC_PAF_global_age_annual_four_SSP_WCRF_primary_v1.csv", index=False, encoding="utf-8-sig")
    ages[ages["year"].isin(KEY_YEARS)].sort_values(["scenario_code", "year", "age_order"]).drop(columns="age_order").to_csv(tables / "CRC_PAF_global_age_key_years_WCRF_primary_v1.csv", index=False, encoding="utf-8-sig")
    qa.to_csv(qa_dir / "CRC_PAF_all_task_QA_summary_v1.csv", index=False, encoding="utf-8-sig")

    global_changes = summaries[["scenario_code", "scenario", "year", "global_paf"]].rename(columns={"global_paf": "paf"})
    global_anomalies = anomaly_table(global_changes, ["scenario_code"], "global")
    national_anomalies = anomaly_table(national, ["scenario_code", "iso3"], "national")
    adm1_anomalies = anomaly_table(adm1, ["scenario_code", "adm1_uid"], "ADM1")
    anomalies = pd.concat([global_anomalies, national_anomalies, adm1_anomalies], ignore_index=True, sort=False)
    anomalies.to_csv(qa_dir / "CRC_PAF_temporal_jump_flags_v1.csv", index=False, encoding="utf-8-sig")

    max_closure = float(qa[["closure_national", "closure_adm1", "closure_0p5", "closure_0p1"]].max().max())
    max_cross_scale = float(qa[["error_0p1_vs_0p5", "error_0p5_vs_adm1"]].max().max())
    max_national = float(qa["national_vs_adm1_max"].max())
    final_qa = {
        "status": "PASS" if len(qa) == 104 and max_closure <= 1e-5 and max_cross_scale <= 1e-7 and max_national <= 1e-3 else "FAIL",
        "tasks_passed": len(qa), "tasks_expected": 104,
        "maximum_probability_closure_error": max_closure,
        "maximum_cross_scale_aggregation_error": max_cross_scale,
        "maximum_national_anchor_vs_adm1_error": max_national,
        "national_anchor_vs_adm1_tolerance": 1e-3,
        "minimum_matched_spatial_countries": int(qa["matched_spatial_countries"].min()),
        "temporal_jump_flags": len(anomalies),
        "temporal_jump_rule": "absolute year-to-year change greater than max(0.02 PAF, median absolute change + 10 MAD) within each scenario and spatial level",
        "monotonicity_required": False,
        "burden_calculated": False,
        "RR_uncertainty_scope": "RR-only; BMI prediction uncertainty not propagated",
    }
    (qa_dir / "CRC_PAF_full_production_QA_v1.json").write_text(json.dumps(final_qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if final_qa["status"] != "PASS":
        raise RuntimeError("Full production QA failed")

    row2025 = summaries[summaries["year"].eq(2025)].set_index("scenario_code")
    row2050 = summaries[summaries["year"].eq(2050)].set_index("scenario_code")
    lines = [
        "# CRC PAF full production report", "", "Status: **PASS**", "",
        "This production contains PAF only. No CRC incidence, deaths, DALYs, or GBD high-BMI calibration were calculated.", "",
        "## Global PAF in 2050", "",
    ]
    for scenario in SCENARIOS:
        lines.append(f"- {SCENARIO_LABELS[scenario]}: {row2050.loc[scenario, 'global_paf']*100:.3f}% (RR-only range {row2050.loc[scenario, 'global_paf_rr_lower']*100:.3f}%-{row2050.loc[scenario, 'global_paf_rr_upper']*100:.3f}%).")
    lines += ["", "## Change from 2025 to 2050", ""]
    for scenario in SCENARIOS:
        lines.append(f"- {SCENARIO_LABELS[scenario]}: {(row2050.loc[scenario, 'global_paf']-row2025.loc[scenario, 'global_paf'])*100:.3f} percentage points.")
    lines += ["", "## QA", "", f"- 104/104 scenario-year tasks passed.", f"- Maximum probability closure error: {max_closure:.3e}.", f"- Maximum cross-scale aggregation error: {max_cross_scale:.3e}.", f"- Maximum national-anchor versus ADM1 error: {max_national:.3e}.", f"- Temporal jump flags: {len(anomalies)}.", "- National PAF covers 204 countries and territories. Spatial outputs retain only represented 0.1-degree/0.5-degree/ADM1 support; national-only locations are not spatially fabricated."]
    (reports / "CRC_PAF_full_production_report_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = build_manifest()
    manifest_path = ROOT / "CRC_PAF_full_formal_manifest_v1.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    (ROOT / "COMPLETE.flag").write_text("CRC_PAF_FULL_PRODUCTION_COMPLETE_NO_BURDEN\n", encoding="ascii")
    print(json.dumps({"status": "PASS", "tasks": 104, "max_closure": max_closure, "max_cross_scale": max_cross_scale, "temporal_flags": len(anomalies), "manifest_files": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
