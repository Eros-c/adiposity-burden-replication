"""Run the reference-path bridge for main-text SSP2-4.5 key years.

The controller is local, serial and restartable. Existing outputs are skipped
only when their per-year bridge summary is PASS.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(_cfg.ROOT / "future_scenario_ascii" / "06_future_bmi_2025_2050" / "bridge_ncd2024_gbdtrend_sensitivity_v1_20260909")
SCRIPT = ROOT / "scripts" / "01_rake_ssp245_2025_bridge_v1.py"
YEARS = (2025, 2030, 2035, 2040, 2045, 2050)

def main() -> int:
    lock = ROOT / "ssp245_keyyears_controller.lock"
    try:
        lock.write_text(str(os.getpid()), encoding="ascii", errors="ignore")
        state_path = ROOT / "qa" / "ssp245_keyyears_controller_state_v1.json"
        for year in YEARS:
            summary = ROOT / "qa" / f"ssp245_{year}_bridge_summary_v1.json"
            if summary.exists() and json.loads(summary.read_text(encoding="utf-8")).get("status") == "PASS_BRIDGE_PILOT":
                continue
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({"status":"RUNNING", "year":year, "updated":time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
            env = os.environ.copy(); env.update(BRIDGE_YEAR=str(year), BRIDGE_SCENARIO="SSP2-4.5", BRIDGE_SCENARIO_CODE="ssp245")
            result = subprocess.run([sys.executable, str(SCRIPT)], env=env)
            if result.returncode != 0:
                state_path.write_text(json.dumps({"status":"FAILED", "year":year, "updated":time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
                return result.returncode
        state_path.write_text(json.dumps({"status":"PASS", "years":YEARS, "updated":time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
        return 0
    finally:
        lock.unlink(missing_ok=True)

if __name__ == "__main__":
    raise SystemExit(main())
