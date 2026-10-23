"""Resumable local controller for 104 CRC PAF scenario-year tasks."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import csv
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKER = HERE / "02_crc_paf_full_task_v1.py"
FINALIZER = HERE / "03_finalize_crc_paf_full_v1.py"
PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "08_colorectal_cancer_paf" / "full_annual_four_ssp_wcrf_primary_v1_20260910"
ANCHORS = PROJECT / "06_future_bmi_2025_2050" / "full_covariate_global_preflight_v2_20260906" / "inputs" / "national_bmi7_scenario_anchors_full204_v2.csv.gz"
PILOT = PROJECT / "08_colorectal_cancer_paf" / "pilot_ssp245_2025_2050_wcrf_primary_v1"
RR = PROJECT / "07_rr_cause_crosswalk" / "CRC_RR_by_BMI_category_WCRF_primary_v1.csv"
SCENARIOS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
YEARS = tuple(range(2025, 2051))
GATES = (("ssp245", 2025), ("ssp245", 2050))
PILOT_TOLERANCE = 1e-10


def log(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with (ROOT / "logs" / "controller.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def task_complete(scenario: str, year: int) -> bool:
    state = ROOT / "state" / scenario / f"{year}.json"
    flag = ROOT / "state" / scenario / f"{year}.COMPLETE.flag"
    if not state.exists() or not flag.exists():
        return False
    try:
        return json.loads(state.read_text(encoding="utf-8")).get("status") == "PASS"
    except Exception:
        return False


def update_manifest():
    rows = []
    for scenario, label in SCENARIOS.items():
        for year in YEARS:
            state_path = ROOT / "state" / scenario / f"{year}.json"
            status = "pending"
            elapsed = None
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    status = state.get("status", "unknown")
                    elapsed = state.get("elapsed_seconds")
                except Exception:
                    status = "invalid_state"
            rows.append({"scenario_code": scenario, "scenario": label, "year": year, "priority": "pilot_gate" if (scenario, year) in GATES else "production", "status": status, "elapsed_seconds": elapsed})
    path = ROOT / "state" / "CRC_PAF_full_task_manifest_v1.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def prepare_contract_and_cache():
    for name in ("data/national", "data/adm1", "data/0p5deg", "data/0p1deg", "tables", "qa", "reports", "logs", "state", "contracts", "input_cache/national_anchors"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    contract = {
        "status": "FROZEN",
        "created_date": "2026-09-10",
        "scope": "CRC PAF only; four SSPs; annual 2025-2050; adults 25+",
        "tasks": 104,
        "RR_crosswalk": str(RR),
        "RR_source": "WCRF/AICR primary and only",
        "reference_BMI": "22.5 kg/m2 study-defined; not WCRF/AICR TMREL",
        "aggregation": "sum population*(weighted_RR-1) divided by sum population*weighted_RR",
        "national_outputs": ["age_sex", "age", "sex", "overall"],
        "ADM1_outputs": ["age_sex", "age", "sex", "overall"],
        "0p5deg_outputs": ["age_sex", "overall", "point", "RR_lower", "RR_upper"],
        "0p1deg_outputs": ["age_sex point", "overall point"],
        "uncertainty": "RR-only lower/upper; BMI prediction uncertainty not propagated",
        "national_only_policy": "retain national PAF; do not fabricate spatial PAF",
        "prohibited": ["CRC cases", "CRC deaths", "CRC DALYs", "GBD burden extrapolation", "GBD high-BMI calibration", "RR sensitivity", "Monte Carlo"],
    }
    write_json(ROOT / "contracts" / "CRC_PAF_full_production_contract_v1.json", contract)

    cache_root = ROOT / "input_cache" / "national_anchors"
    missing = [(scenario, year) for scenario in SCENARIOS for year in YEARS if not (cache_root / scenario / f"{year}.parquet").exists()]
    if missing:
        log(f"Preparing {len(missing)} national anchor cache files")
        data = pd.read_csv(ANCHORS, low_memory=False)
        for scenario, year in missing:
            selected = data[(data["scenario"].eq(SCENARIOS[scenario])) & data["year"].eq(year)].copy()
            if len(selected) != 6120 or selected["iso3"].nunique() != 204:
                raise RuntimeError(f"National cache coverage failed: {scenario}/{year}")
            path = cache_root / scenario / f"{year}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            selected.to_parquet(path, index=False)
        del data
    update_manifest()


def run_task(scenario: str, year: int, force=False):
    if not force and task_complete(scenario, year):
        log(f"SKIP PASS {scenario}/{year}")
        return True
    task_logs = ROOT / "logs" / "tasks" / scenario
    task_logs.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        stdout_path = task_logs / f"{year}_attempt_{attempt}.stdout.log"
        stderr_path = task_logs / f"{year}_attempt_{attempt}.stderr.log"
        log(f"START {scenario}/{year} attempt {attempt}")
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run([sys.executable, str(WORKER), "--scenario", scenario, "--year", str(year)], cwd=str(HERE), stdout=stdout, stderr=stderr, text=True, check=False)
        if result.returncode == 0 and task_complete(scenario, year):
            state = json.loads((ROOT / "state" / scenario / f"{year}.json").read_text(encoding="utf-8"))
            log(f"PASS {scenario}/{year} elapsed={state.get('elapsed_seconds')}s")
            update_manifest()
            return True
        log(f"FAIL {scenario}/{year} attempt {attempt} returncode={result.returncode}")
    update_manifest()
    return False


def pilot_reproduction():
    pilot_global = pd.read_csv(PILOT / "data" / "tables" / "CRC_PAF_ssp245_2025_2050_global_overall_WCRF_primary_v1.csv")
    pilot_sex = pd.read_csv(PILOT / "data" / "tables" / "CRC_PAF_ssp245_2025_2050_global_sex_WCRF_primary_v1.csv")
    pilot_national = pd.read_csv(PILOT / "data" / "tables" / "CRC_PAF_ssp245_2025_2050_national_overall_WCRF_primary_v1.csv")
    pilot_adm1 = pd.read_parquet(PILOT / "data" / "tables" / "CRC_PAF_ssp245_2025_2050_adm1_age_sex_WCRF_primary_v1.parquet")
    rows = []
    for _, year in GATES:
        summary = json.loads((ROOT / "tables" / "task_summary" / "ssp245" / f"{year}.json").read_text(encoding="utf-8"))
        comparisons = {
            "global_paf": (summary["global_paf"], pilot_global.loc[pilot_global["year"].eq(year), "PAF_point"].iloc[0]),
            "female_paf": (summary["female_paf"], pilot_sex.loc[pilot_sex["year"].eq(year) & pilot_sex["sex"].eq("Female"), "PAF_point"].iloc[0]),
            "male_paf": (summary["male_paf"], pilot_sex.loc[pilot_sex["year"].eq(year) & pilot_sex["sex"].eq("Male"), "PAF_point"].iloc[0]),
            "country_median": (summary["country"]["median"], pilot_national.loc[pilot_national["year"].eq(year), "PAF_point"].median()),
            "country_p95": (summary["country"]["p95"], pilot_national.loc[pilot_national["year"].eq(year), "PAF_point"].quantile(0.95)),
            "country_max": (summary["country"]["max"], pilot_national.loc[pilot_national["year"].eq(year), "PAF_point"].max()),
            "adm1_median": (summary["adm1_age_sex"]["median"], pilot_adm1.loc[pilot_adm1["year"].eq(year), "PAF_point"].median()),
            "adm1_p95": (summary["adm1_age_sex"]["p95"], pilot_adm1.loc[pilot_adm1["year"].eq(year), "PAF_point"].quantile(0.95)),
            "adm1_max": (summary["adm1_age_sex"]["max"], pilot_adm1.loc[pilot_adm1["year"].eq(year), "PAF_point"].max()),
        }
        for metric, (production, reference) in comparisons.items():
            difference = float(production - reference)
            rows.append({"scenario_code": "ssp245", "year": year, "metric": metric, "production": float(production), "pilot": float(reference), "difference": difference, "absolute_difference": abs(difference), "tolerance": PILOT_TOLERANCE, "status": "PASS" if abs(difference) <= PILOT_TOLERANCE else "FAIL"})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "qa" / "CRC_PAF_pilot_reproduction_QA_v1.csv", index=False, encoding="utf-8-sig")
    result = {"status": "PASS" if frame["status"].eq("PASS").all() else "FAIL", "maximum_absolute_error": float(frame["absolute_difference"].max()), "tolerance": PILOT_TOLERANCE, "comparisons": len(frame)}
    write_json(ROOT / "qa" / "CRC_PAF_pilot_reproduction_summary_v1.json", result)
    return result


def run_finalizer():
    for attempt in (1, 2):
        stdout_path = ROOT / "logs" / f"finalizer_attempt_{attempt}.stdout.log"
        stderr_path = ROOT / "logs" / f"finalizer_attempt_{attempt}.stderr.log"
        log(f"START finalizer attempt {attempt}")
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run([sys.executable, str(FINALIZER)], cwd=str(HERE), stdout=stdout, stderr=stderr, text=True, check=False)
        if result.returncode == 0 and (ROOT / "COMPLETE.flag").exists():
            log("PASS finalizer")
            return True
        log(f"FAIL finalizer attempt {attempt} returncode={result.returncode}")
    return False


def main():
    started = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "FAILED.flag").unlink(missing_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    status_path = ROOT / "state" / "controller_status.json"
    try:
        prepare_contract_and_cache()
        write_json(status_path, {"status": "pilot_gate", "started_epoch": started})
        for scenario, year in GATES:
            if not run_task(scenario, year, force=True):
                raise RuntimeError(f"Pilot gate task failed: {scenario}/{year}")
        reproduction = pilot_reproduction()
        log(f"Pilot reproduction {reproduction['status']} max_error={reproduction['maximum_absolute_error']:.3e}")
        if reproduction["status"] != "PASS":
            raise RuntimeError("Pilot reproduction failed; production stopped")

        write_json(status_path, {"status": "production", "started_epoch": started, "pilot_reproduction": reproduction})
        for scenario in SCENARIOS:
            for year in YEARS:
                if (scenario, year) in GATES:
                    continue
                if not run_task(scenario, year):
                    raise RuntimeError(f"Production task failed after retry: {scenario}/{year}")

        if not run_finalizer():
            raise RuntimeError("Finalizer failed after retry")
        elapsed = round(time.time() - started, 3)
        result = {"status": "complete", "tasks": 104, "elapsed_seconds": elapsed, "pilot_reproduction": reproduction, "output_root": str(ROOT)}
        write_json(status_path, result)
        write_json(ROOT / "state" / "controller_final_summary.json", result)
        log(f"COMPLETE 104 tasks elapsed={elapsed}s")
        return 0
    except Exception:
        error = traceback.format_exc()
        write_json(status_path, {"status": "failed", "traceback": error, "elapsed_seconds": round(time.time() - started, 3)})
        write_json(ROOT / "state" / "controller_final_summary.json", {"status": "failed", "traceback": error})
        (ROOT / "FAILED.flag").write_text("CRC_PAF_FULL_PRODUCTION_FAILED\n", encoding="ascii")
        log("FATAL controller failure; see state/controller_status.json")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
