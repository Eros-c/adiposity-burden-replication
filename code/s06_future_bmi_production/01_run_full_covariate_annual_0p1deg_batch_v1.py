#!/usr/bin/env python3
"""Checkpointed annual 0.1-degree full-covariate BMI production controller."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(_cfg.ROOT / ".codex_tmp" / "fullcov_annual_0p1_v1")
PILOT = Path(_cfg.ROOT / ".codex_tmp" / "fullcov_0p1_pilot_v1")
PYTHON = Path(r"python")
RSCRIPT = Path(r"Rscript")
TEMPLATES = ROOT / "templates"

SCENARIOS = {
    "ssp126": {"display": "SSP1-2.6", "ssp": "ssp1", "upper": "SSP1"},
    "ssp245": {"display": "SSP2-4.5", "ssp": "ssp2", "upper": "SSP2"},
    "ssp370": {"display": "SSP3-7.0", "ssp": "ssp3", "upper": "SSP3"},
    "ssp585": {"display": "SSP5-8.5", "ssp": "ssp5", "upper": "SSP5"},
}
FAMILIES = (
    "probability_overall",
    "probability_sex",
    "probability_age",
    "probability_age_sex",
    "population_overall",
    "population_sex",
    "population_age",
    "population_age_sex",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Template replacement missing: {label}")
    return text.replace(old, new)


def transform_common_r(text: str, task_root: Path, scenario_code: str, year: int) -> str:
    scenario = SCENARIOS[scenario_code]
    text = replace_required(
        text,
        str(_cfg.ROOT / ".codex_tmp" / "fullcov_0p1_pilot_v1"),
        task_root.as_posix(),
        "R task root",
    )
    text = text.replace('scenario <- "SSP5-8.5"', f'scenario <- "{scenario["display"]}"')
    text = text.replace('scenario_code <- "ssp585"', f'scenario_code <- "{scenario_code}"')
    text = text.replace('ssp <- "ssp5"', f'ssp <- "{scenario["ssp"]}"')
    text = text.replace("year <- 2050L", f"year <- {year}L")
    text = text.replace("ssp585_2050", f"{scenario_code}_{year}")
    text = text.replace("ssp5_2050", f"{scenario['ssp']}_{year}")
    text = text.replace("SSP5_2050", f"{scenario['upper']}_{year}")
    text = text.replace("SSP5-8.5 2050", f"{scenario['display']} {year}")
    text = text.replace("SSP5", scenario["upper"])
    text = text.replace("ssp5", scenario["ssp"])
    text = text.replace("missing 2050", f"missing {year}")
    return text


def transform_harmonization(task_root: Path, scenario_code: str, year: int) -> str:
    text = (TEMPLATES / "01_harmonize_task_template_v1.R").read_text(encoding="utf-8")
    old_gdp = '''gdp_source <- file.path(
  raw_root, "ssp_grid_gdp", "WangSun_0p25deg", "extracted", "025d",
  "GDP2050_ssp5.tif"
)
gdp_raw <- rast(gdp_source)
gdp_raw[gdp_raw < 0] <- NA
'''
    new_gdp = '''gdp_nodes <- seq(2025L, 2050L, by = 5L)
gdp_y0 <- max(gdp_nodes[gdp_nodes <= year])
gdp_y1 <- min(gdp_nodes[gdp_nodes >= year])
gdp_directory <- file.path(
  raw_root, "ssp_grid_gdp", "WangSun_0p25deg", "extracted", "025d"
)
gdp_raw0 <- rast(file.path(gdp_directory, sprintf("GDP%d_%s.tif", gdp_y0, ssp)))
gdp_raw0[gdp_raw0 < 0] <- NA
if (gdp_y0 == gdp_y1) {
  gdp_raw <- gdp_raw0
} else {
  gdp_raw1 <- rast(file.path(gdp_directory, sprintf("GDP%d_%s.tif", gdp_y1, ssp)))
  gdp_raw1[gdp_raw1 < 0] <- NA
  gdp_fraction <- (year - gdp_y0) / (gdp_y1 - gdp_y0)
  gdp_raw <- exp(
    (1 - gdp_fraction) * log(1 + gdp_raw0) +
      gdp_fraction * log(1 + gdp_raw1)
  ) - 1
}
'''
    old_ntl = '''ntl_source <- file.path(raw_root, "FUT_NTL", "extracted", "ssp5", "NTL_SSP5_2050.tif")
ntl_raw <- rast(ntl_source)
ntl_raw[ntl_raw < 0] <- NA
ntl_log_raw <- log1p(ntl_raw)
'''
    new_ntl = '''ntl_nodes <- seq(2025L, 2050L, by = 5L)
ntl_y0 <- max(ntl_nodes[ntl_nodes <= year])
ntl_y1 <- min(ntl_nodes[ntl_nodes >= year])
ntl_directory <- file.path(raw_root, "FUT_NTL", "extracted", ssp)
ntl_raw0 <- rast(file.path(
  ntl_directory, sprintf("NTL_%s_%d.tif", toupper(ssp), ntl_y0)
))
ntl_raw0[ntl_raw0 < 0] <- NA
ntl_log_raw <- log1p(ntl_raw0)
if (ntl_y0 != ntl_y1) {
  ntl_raw1 <- rast(file.path(
    ntl_directory, sprintf("NTL_%s_%d.tif", toupper(ssp), ntl_y1)
  ))
  ntl_raw1[ntl_raw1 < 0] <- NA
  ntl_fraction <- (year - ntl_y0) / (ntl_y1 - ntl_y0)
  ntl_log_raw <- (1 - ntl_fraction) * ntl_log_raw +
    ntl_fraction * log1p(ntl_raw1)
}
'''
    old_urban = '''urban_source <- file.path(
  raw_root, "ssp_urban", "Gao_Pesaresi_SEDAC_1km", "extracted",
  "ssp5-geotiff", "ssp5_2050.tif"
)
urban_raw <- rast(urban_source)
urban_raw[urban_raw < 0] <- NA
'''
    new_urban = '''urban_nodes <- seq(2020L, 2050L, by = 10L)
urban_y0 <- max(urban_nodes[urban_nodes <= year])
urban_y1 <- min(urban_nodes[urban_nodes >= year])
urban_directory <- file.path(
  raw_root, "ssp_urban", "Gao_Pesaresi_SEDAC_1km", "extracted",
  paste0(ssp, "-geotiff")
)
urban_raw0 <- rast(file.path(urban_directory, sprintf("%s_%d.tif", ssp, urban_y0)))
urban_raw0[urban_raw0 < 0] <- NA
if (urban_y0 == urban_y1) {
  urban_raw <- urban_raw0
} else {
  urban_raw1 <- rast(file.path(urban_directory, sprintf("%s_%d.tif", ssp, urban_y1)))
  urban_raw1[urban_raw1 < 0] <- NA
  urban_fraction <- (year - urban_y0) / (urban_y1 - urban_y0)
  urban_raw <- (1 - urban_fraction) * urban_raw0 + urban_fraction * urban_raw1
}
'''
    text = replace_required(text, old_gdp, new_gdp, "GDP node interpolation")
    text = replace_required(text, old_ntl, new_ntl, "NTL node interpolation")
    text = replace_required(text, old_urban, new_urban, "urban node interpolation")
    text = transform_common_r(text, task_root, scenario_code, year)
    text = text.replace(
        "Wang-Sun SSP node weights, exact 0.5-degree sum calibration",
        "Wang-Sun SSP node weights, log1p temporal interpolation and exact 0.5-degree sum calibration",
    )
    text = text.replace(
        "native 1-km projected NTL texture; additive within-parent contrast",
        "temporally interpolated native 1-km projected NTL texture; additive within-parent contrast",
    )
    text = text.replace(
        "native 1-km urban-land texture with frozen bridge slopes",
        "linearly interpolated native 1-km urban-land texture with frozen bridge slopes",
    )
    return text


def transform_model(task_root: Path, scenario_code: str, year: int) -> str:
    text = (TEMPLATES / "02_model_task_template_v1.R").read_text(encoding="utf-8")
    return transform_common_r(text, task_root, scenario_code, year)


def transform_python(template_name: str, task_root: Path, scenario_code: str, year: int) -> str:
    scenario = SCENARIOS[scenario_code]
    text = (TEMPLATES / template_name).read_text(encoding="utf-8")
    text = replace_required(
        text,
        str(_cfg.ROOT / ".codex_tmp" / "fullcov_0p1_pilot_v1"),
        str(task_root),
        "Python task root",
    )
    text = text.replace(r"\outputs\ssp585\2050", rf"\outputs\{scenario_code}\{year}")
    text = text.replace(
        '"ssp585" / "2050"',
        f'"{scenario_code}" / "{year}"',
    )
    text = text.replace('SCENARIO = "SSP5-8.5"', f'SCENARIO = "{scenario["display"]}"')
    text = text.replace('SCENARIO_CODE = "ssp585"', f'SCENARIO_CODE = "{scenario_code}"')
    text = text.replace("YEAR = 2050", f"YEAR = {year}")
    text = text.replace("ssp585_2050", f"{scenario_code}_{year}")
    text = text.replace('"scenario": "SSP5-8.5"', f'"scenario": "{scenario["display"]}"')
    text = text.replace('"year": 2050', f'"year": {year}')
    return text


def write_task_scripts(task_root: Path, scenario_code: str, year: int) -> list[Path]:
    scripts = task_root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    generated = {
        "01_harmonize_task_v1.R": transform_harmonization(task_root, scenario_code, year),
        "02_model_task_v1.R": transform_model(task_root, scenario_code, year),
        "03_rake_task_v1.py": transform_python("03_rake_task_template_v1.py", task_root, scenario_code, year),
        "04_qa_task_v1.py": transform_python("04_qa_task_template_v1.py", task_root, scenario_code, year),
    }
    paths: list[Path] = []
    for name, content in generated.items():
        path = scripts / name
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing != content:
                # Allow only the audited ZSTD level migration between retries.
                migrated = (
                    existing.replace("ZSTD_LEVEL=9", "ZSTD_LEVEL=1")
                    .replace('"zstd_level": 9', '"zstd_level": 1')
                    .replace("level=9", "level=1")
                )
                if migrated == content:
                    path.write_text(content, encoding="utf-8")
                    print(f"[{now_utc()}] migrated compression settings in {path.name}", flush=True)
                    paths.append(path)
                    continue
                raise RuntimeError(f"Generated task script changed: {path}")
        else:
            path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def run_stage(command: list[str], log_path: Path, label: str) -> None:
    print(f"[{now_utc()}] {label} START", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=Path(command[1]).parent, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}; see {log_path}")
    print(f"[{now_utc()}] {label} PASS", flush=True)


def task_state_path(scenario_code: str, year: int) -> Path:
    return ROOT / "state" / scenario_code / f"{year}.json"


def published_paths(scenario_code: str, year: int) -> list[Path]:
    directory = ROOT / "outputs" / scenario_code / str(year)
    return [directory / f"{family}_{scenario_code}_{year}_0p1deg_v1.tif" for family in FAMILIES]


def task_passed(scenario_code: str, year: int) -> bool:
    path = task_state_path(scenario_code, year)
    if not path.exists():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    return state.get("status") == "PASS_ANNUAL_0P1DEG_TASK" and all(path.is_file() for path in published_paths(scenario_code, year))


def publish_task(task_root: Path, scenario_code: str, year: int) -> list[dict[str, object]]:
    source_dir = task_root / "data" / "0p1deg" / "cog" / scenario_code / str(year)
    output_dir = ROOT / "outputs" / scenario_code / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        name = f"{family}_{scenario_code}_{year}_0p1deg_v1.tif"
        source = source_dir / name
        destination = output_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = sha256(source)
        if destination.exists():
            if sha256(destination) != source_hash:
                raise RuntimeError(f"Published output differs: {destination}")
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        rows.append({"family": family, "path": str(destination), "bytes": destination.stat().st_size, "sha256": source_hash})
    return rows


def safe_remove_task_subdirectory(path: Path, task_root: Path) -> int:
    if not path.exists():
        return 0
    resolved = path.resolve()
    allowed = task_root.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise RuntimeError(f"Cleanup target escaped task root: {resolved}")
    bytes_removed = sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file())
    shutil.rmtree(resolved)
    return bytes_removed


def archive_stale_attempt_artifacts(task_root: Path) -> str | None:
    artifact_names = ("qa", "contracts", "logs")
    existing = [
        task_root / name
        for name in artifact_names
        if (task_root / name).is_dir() and any((task_root / name).iterdir())
    ]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt_root = task_root / "attempts" / f"attempt_{stamp}"
    suffix = 1
    while attempt_root.exists():
        attempt_root = task_root / "attempts" / f"attempt_{stamp}_{suffix}"
        suffix += 1
    attempt_root.mkdir(parents=True, exist_ok=False)
    for source in existing:
        shutil.move(str(source), str(attempt_root / source.name))
    return str(attempt_root)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def acquire_controller_lock() -> Path:
    """Prevent two local annual controllers from producing the same outputs."""
    lock_dir = ROOT / "state" / "annual_controller_v2.lock"
    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        owner_path = lock_dir / "owner.json"
        owner: dict[str, object] = {}
        if owner_path.is_file():
            try:
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                owner = {}
        owner_pid = owner.get("pid")
        alive = False
        if isinstance(owner_pid, int):
            try:
                os.kill(owner_pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False
        if alive:
            raise RuntimeError(
                f"Another annual controller is running (pid={owner_pid}); "
                f"lock={lock_dir}"
            )
        # The owner process is gone; its directory is a recoverable stale lock.
        shutil.rmtree(lock_dir)
        lock_dir.mkdir(parents=True, exist_ok=False)
    (lock_dir / "owner.json").write_text(
        json.dumps({"pid": os.getpid(), "started_at_utc": now_utc()}, indent=2),
        encoding="utf-8",
    )
    return lock_dir


def release_controller_lock(lock_dir: Path) -> None:
    if not lock_dir.exists():
        return
    resolved = lock_dir.resolve()
    expected_parent = (ROOT / "state").resolve()
    if resolved.parent != expected_parent or resolved.name != "annual_controller_v2.lock":
        raise RuntimeError(f"Refusing to remove unexpected controller lock: {resolved}")
    shutil.rmtree(resolved)


def run_task(scenario_code: str, year: int, attempt_number: int = 1) -> None:
    if task_passed(scenario_code, year):
        print(f"[{now_utc()}] {scenario_code}/{year} already PASS", flush=True)
        return
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise RuntimeError("Free disk below 30 GiB")
    task_root = ROOT / "tasks" / scenario_code / str(year)
    task_root.mkdir(parents=True, exist_ok=True)
    archive_stale_attempt_artifacts(task_root)
    safe_remove_task_subdirectory(task_root / "work", task_root)
    safe_remove_task_subdirectory(task_root / "data", task_root)
    state_path = task_state_path(scenario_code, year)
    state = {
        "status": "RUNNING_ANNUAL_0P1DEG_TASK",
        "scenario_code": scenario_code,
        "scenario": SCENARIOS[scenario_code]["display"],
        "year": year,
        "attempt_number": attempt_number,
        "started_at_utc": now_utc(),
        "current_stage": "generate_scripts",
    }
    atomic_json(state_path, state)
    try:
        scripts = write_task_scripts(task_root, scenario_code, year)
        stages = (
            ("FC01", [str(RSCRIPT), str(scripts[0])]),
            ("FC02", [str(RSCRIPT), str(scripts[1])]),
            ("FC03", [str(PYTHON), str(scripts[2])]),
            ("FC04", [str(PYTHON), str(scripts[3])]),
        )
        for stage, command in stages:
            state["current_stage"] = stage
            atomic_json(state_path, state)
            run_stage(command, task_root / "logs" / f"controller_{stage}_v1.log", f"{scenario_code}/{year} {stage}")
        fc04_path = task_root / "qa" / f"Gate_FC04_summary_{scenario_code}_{year}_v1.json"
        fc04 = json.loads(fc04_path.read_text(encoding="utf-8"))
        if fc04.get("status") != "PASS_GATE_FC04_INDEPENDENT_SPATIAL_QA":
            raise RuntimeError(f"Independent spatial QA did not pass: {fc04_path}")
        state["current_stage"] = "publish"
        atomic_json(state_path, state)
        outputs = publish_task(task_root, scenario_code, year)
        write_manifest(ROOT / "checksums" / scenario_code / f"task_{year}_formal_COG_SHA256_v1.csv", outputs)
        state.update({
            "status": "PASS_ANNUAL_0P1DEG_TASK",
            "completed_at_utc": now_utc(),
            "current_stage": None,
            "formal_cogs": len(outputs),
            "formal_bytes": sum(int(row["bytes"]) for row in outputs),
            "outputs": outputs,
            "interpolation": {
                "gdp": "log1p between five-year nodes",
                "ntl": "log1p between five-year nodes",
                "urban_land": "linear between ten-year nodes",
                "final_parent_calibration": "exact formal 0.5-degree scenario-year target",
            },
        })
        atomic_json(state_path, state)
        removed = safe_remove_task_subdirectory(task_root / "work", task_root)
        removed += safe_remove_task_subdirectory(task_root / "data", task_root)
        state["validated_transient_bytes_removed"] = removed
        state["cleanup_completed_at_utc"] = now_utc()
        atomic_json(state_path, state)
        print(f"[{now_utc()}] {scenario_code}/{year} PASS; cleaned {removed / 1024**3:.2f} GiB", flush=True)
    except Exception as error:
        state["status"] = "FAIL_ANNUAL_0P1DEG_TASK"
        state["failed_at_utc"] = now_utc()
        state["failed_stage"] = state.get("current_stage")
        state["error"] = str(error)
        atomic_json(state_path, state)
        raise


def run_task_with_retries(
    scenario_code: str,
    year: int,
    retry_count: int,
    retry_delay_seconds: int,
) -> None:
    """Run one task locally, retrying only the failed task with archived artifacts."""
    errors: list[str] = []
    for attempt_number in range(1, retry_count + 2):
        try:
            run_task(scenario_code, year, attempt_number=attempt_number)
            return
        except Exception as error:
            errors.append(f"attempt {attempt_number}: {error}")
            if attempt_number > retry_count:
                raise RuntimeError(
                    f"{scenario_code}/{year} failed after {attempt_number} attempt(s): "
                    + " | ".join(errors)
                ) from error
            print(
                f"[{now_utc()}] {scenario_code}/{year} FAIL; "
                f"retry {attempt_number}/{retry_count} after {retry_delay_seconds}s",
                flush=True,
            )
            time.sleep(retry_delay_seconds)


def import_pilot() -> None:
    scenario_code = "ssp585"
    year = 2050
    if task_passed(scenario_code, year):
        return
    source_dir = PILOT / "data" / "0p1deg" / "cog" / scenario_code / str(year)
    output_dir = ROOT / "outputs" / scenario_code / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in FAMILIES:
        name = f"{family}_{scenario_code}_{year}_0p1deg_v1.tif"
        source = source_dir / name
        destination = output_dir / name
        digest = sha256(source)
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        if sha256(destination) != digest:
            raise RuntimeError(f"Imported pilot output differs: {destination}")
        rows.append({"family": family, "path": str(destination), "bytes": destination.stat().st_size, "sha256": digest})
    write_manifest(ROOT / "checksums" / scenario_code / f"task_{year}_formal_COG_SHA256_v1.csv", rows)
    atomic_json(task_state_path(scenario_code, year), {
        "status": "PASS_ANNUAL_0P1DEG_TASK",
        "scenario_code": scenario_code,
        "scenario": "SSP5-8.5",
        "year": year,
        "completed_at_utc": now_utc(),
        "source": str(PILOT),
        "source_gate": "PASS_WITH_WARN_ANNUAL_0P1DEG_PRODUCTION_AUTHORIZED",
        "formal_cogs": 8,
        "formal_bytes": sum(int(row["bytes"]) for row in rows),
        "outputs": rows,
        "import_method": "same-volume hard link where supported; content verified by SHA256",
    })
    print(f"[{now_utc()}] imported validated pilot ssp585/2050", flush=True)


def refresh_global_manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_code in SCENARIOS:
        for year in range(2025, 2051):
            state_path = task_state_path(scenario_code, year)
            status = "PENDING"
            formal_cogs = 0
            formal_bytes = 0
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                status = state.get("status", "UNKNOWN")
                formal_cogs = int(state.get("formal_cogs", 0))
                formal_bytes = int(state.get("formal_bytes", 0))
            rows.append({
                "scenario_code": scenario_code,
                "scenario": SCENARIOS[scenario_code]["display"],
                "year": year,
                "status": status,
                "formal_cogs": formal_cogs,
                "formal_bytes": formal_bytes,
                "state_path": str(state_path),
            })
    write_manifest(ROOT / "state" / "annual_0p1deg_task_manifest_v1.csv", rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--end-year", type=int, default=2050)
    parser.add_argument("--max-tasks", type=int, default=104)
    parser.add_argument("--parallel-tasks", type=int, default=1)
    parser.add_argument("--retry-count", type=int, default=1)
    parser.add_argument("--retry-delay-seconds", type=int, default=60)
    parser.add_argument("--confirm-run", choices=["NO", "YES"], default="NO")
    parser.add_argument("--import-pilot", action="store_true")
    args = parser.parse_args()
    if args.start_year < 2025 or args.end_year > 2050 or args.start_year > args.end_year:
        raise ValueError("Require 2025 <= start-year <= end-year <= 2050")
    if args.max_tasks < 1:
        raise ValueError("max-tasks must be positive")
    if args.parallel_tasks < 1 or args.parallel_tasks > 4:
        raise ValueError("parallel-tasks must be between 1 and 4")
    if args.retry_count < 0:
        raise ValueError("retry-count must be non-negative")
    if args.retry_delay_seconds < 0:
        raise ValueError("retry-delay-seconds must be non-negative")

    lock_dir = acquire_controller_lock()
    try:
        if args.import_pilot:
            import_pilot()
        refresh_global_manifest()
        selected_scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        # Year-major order: finish all four scenarios for one year before advancing.
        selected = [
            (scenario_code, year)
            for year in range(args.start_year, args.end_year + 1)
            for scenario_code in selected_scenarios
            if not task_passed(scenario_code, year)
        ][: args.max_tasks]
        if args.confirm_run != "YES":
            print(json.dumps({"status": "DRY_RUN", "selected_tasks": selected}, ensure_ascii=False))
            return 0
        batch_path = ROOT / "state" / "active_batch_v1.json"
        batch = {
            "status": "RUNNING_ANNUAL_0P1DEG_BATCH",
            "started_at_utc": now_utc(),
            "selected_tasks": selected,
            "completed_tasks": [],
            "current_task": None,
            "current_tasks": [],
            "parallel_tasks": args.parallel_tasks,
            "retry_count": args.retry_count,
            "retry_delay_seconds": args.retry_delay_seconds,
            "order": "year_major_then_scenario",
        }
        atomic_json(batch_path, batch)
        try:
            if args.parallel_tasks == 1:
                for scenario_code, year in selected:
                    batch["current_task"] = [scenario_code, year]
                    batch["current_tasks"] = [[scenario_code, year]]
                    atomic_json(batch_path, batch)
                    run_task_with_retries(
                        scenario_code,
                        year,
                        retry_count=args.retry_count,
                        retry_delay_seconds=args.retry_delay_seconds,
                    )
                    batch["completed_tasks"].append([scenario_code, year])
                    refresh_global_manifest()
            else:
                required_free = (30 + 6 * args.parallel_tasks) * 1024**3
                if shutil.disk_usage(ROOT).free < required_free:
                    raise RuntimeError(
                        f"Free disk below parallel safety threshold: {required_free / 1024**3:.0f} GiB"
                    )
                batch["current_task"] = None
                batch["current_tasks"] = [[scenario_code, year] for scenario_code, year in selected]
                atomic_json(batch_path, batch)
                failures: list[str] = []
                with ThreadPoolExecutor(max_workers=args.parallel_tasks) as executor:
                    futures = {
                        executor.submit(
                            run_task_with_retries,
                            scenario_code,
                            year,
                            args.retry_count,
                            args.retry_delay_seconds,
                        ): (scenario_code, year)
                        for scenario_code, year in selected
                    }
                    for future in as_completed(futures):
                        scenario_code, year = futures[future]
                        try:
                            future.result()
                            batch["completed_tasks"].append([scenario_code, year])
                        except Exception as error:
                            failures.append(f"{scenario_code}/{year}: {error}")
                        atomic_json(batch_path, batch)
                        refresh_global_manifest()
                if failures:
                    raise RuntimeError("Parallel task failures: " + " | ".join(failures))
            batch["status"] = "PASS_REQUESTED_ANNUAL_0P1DEG_BATCH"
            batch["completed_at_utc"] = now_utc()
            batch["current_task"] = None
            batch["current_tasks"] = []
            atomic_json(batch_path, batch)
        except Exception as error:
            batch["status"] = "FAIL_REQUESTED_ANNUAL_0P1DEG_BATCH"
            batch["failed_at_utc"] = now_utc()
            batch["error"] = str(error)
            atomic_json(batch_path, batch)
            refresh_global_manifest()
            raise
        print(json.dumps(batch, ensure_ascii=False))
        return 0
    finally:
        release_controller_lock(lock_dir)


if __name__ == "__main__":
    raise SystemExit(main())
