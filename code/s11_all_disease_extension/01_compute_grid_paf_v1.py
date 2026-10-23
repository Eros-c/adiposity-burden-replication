"""Compute 0.1-degree PAF rasters for three representative diseases."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "11_all_disease_extension" / "03_paf" / "paf_grid_multidisease_v1_20261005"
DATA = ROOT / "data"
QA = ROOT / "qa"
REPORTS = ROOT / "reports"
SCRIPTS = ROOT / "scripts"

BMI_ROOT = (
    PROJECT / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906" / "outputs"
)
NATIONAL_PAF = (
    PROJECT / "11_all_disease_extension" / "03_paf" / "paf_all_causes_v1_20261005"
    / "data" / "paf_future_ssp245_2025_2050_all_causes_v1.parquet"
)

DISEASES = {
    493: ("IHD", 1.27, "Ischemic heart disease"),
    976: ("T2DM", 1.72, "Diabetes mellitus type 2"),
    543: ("ALZ", 1.16, "Alzheimer's disease and other dementias"),
}
COMBOS = [("ssp245", 2025), ("ssp245", 2050), ("ssp126", 2050), ("ssp585", 2050)]

CATEGORY_REP = np.array([17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5], dtype=np.float64)
REFERENCE = 22.5
NODATA = -9999.0
COUNTRY_RASTER_LOOKUP = None  # set in main

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def category_rr(rr_per_5: float) -> np.ndarray:
    exponent = np.maximum(CATEGORY_REP - REFERENCE, 0.0) / 5.0
    return np.power(rr_per_5, exponent)

def compute_paf(scenario: str, year: int, rr_per_5: float):
    prob_path = BMI_ROOT / scenario / str(year) / f"probability_sex_{scenario}_{year}_0p1deg_v1.tif"
    pop_path = BMI_ROOT / scenario / str(year) / f"population_sex_{scenario}_{year}_0p1deg_v1.tif"
    rr = category_rr(rr_per_5)
    with rasterio.open(prob_path) as ds:
        prob = ds.read().astype(np.float64)  # (14, H, W)
        profile = ds.profile.copy()
    with rasterio.open(pop_path) as ds:
        pop = ds.read().astype(np.float64)  # (2, H, W)
    female_p = prob[0:7]
    male_p = prob[7:14]
    female_pop = pop[0]
    male_pop = pop[1]
    valid = (
        (female_pop > 0) & (male_pop > -9000)
        & np.all(female_p > -9000, axis=0) & np.all(male_p > -9000, axis=0)
    )
    total_pop = np.where(pop > 0, pop, 0.0).sum(axis=0)
    valid = (total_pop > 0) & np.all(prob > -9000, axis=0)

    def paf_sex(p: np.ndarray) -> np.ndarray:
        weighted = np.tensordot(rr, p, axes=(0, 0))
        return (weighted - 1.0) / weighted

    paf_f = paf_sex(female_p)
    paf_m = paf_sex(male_p)
    w_f = np.where(female_pop > 0, female_pop, 0.0)
    w_m = np.where(male_pop > 0, male_pop, 0.0)
    overall = np.where(
        valid,
        (paf_f * w_f + paf_m * w_m) / np.maximum(w_f + w_m, 1e-12),
        np.nan,
    )
    closure_f = female_p.sum(axis=0)
    closure_m = male_p.sum(axis=0)
    closure_err = float(
        max(
            np.abs(closure_f[valid] - 1.0).max() if valid.any() else 0.0,
            np.abs(closure_m[valid] - 1.0).max() if valid.any() else 0.0,
        )
    )
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    return overall.astype(np.float32), profile, closure_err

def main() -> None:
    outputs: list[Path] = []
    qa_rows: list[dict] = []
    for cause_id, (tag, rr, name) in DISEASES.items():
        for scenario, year in COMBOS:
            paf, profile, closure_err = compute_paf(scenario, year, rr)
            out_dir = DATA / tag / scenario
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"PAF_{tag}_{scenario}_{year}_0p1deg_overall_v1.tif"
            with rasterio.open(out_path, "w", **profile) as ds:
                ds.write(paf, 1)
                ds.set_band_description(1, f"PAF {name} {scenario} {year} overall")
            outputs.append(out_path)
            valid = np.isfinite(paf)
            qa_rows.append({
                "cause_id": cause_id, "disease": name, "scenario": scenario, "year": year,
                "closure_max_abs_error": closure_err,
                "valid_cells": int(valid.sum()),
                "paf_min": float(np.nanmin(paf)), "paf_max": float(np.nanmax(paf)),
                "paf_pop_weighted_note": "population-weighted mean of sex-specific PAFs",
            })
            print(f"done {tag} {scenario} {year}: paf range {qa_rows[-1]['paf_min']:.4f}..{qa_rows[-1]['paf_max']:.4f}")

    qa = pd.DataFrame(qa_rows)
    qa_path = QA / "grid_paf_QA_v1.csv"
    qa.to_csv(qa_path, index=False, encoding="utf-8-sig")
    outputs.append(qa_path)

    problems = []
    if (qa["closure_max_abs_error"] > 1e-4).any():
        problems.append("probability closure violated")
    if (qa["paf_max"] >= 1).any() or (qa["paf_min"] < 0).any():
        problems.append("PAF outside [0, 1)")
    status = "PASS" if not problems else "FAIL"
    summary = {
        "run": "paf_grid_multidisease_v1_20261005",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "diseases": {str(k): v[2] for k, v in DISEASES.items()},
        "combos": [f"{s}/{y}" for s, y in COMBOS],
        "reference_bmi": REFERENCE,
        "problems": problems,
        "status": status,
    }
    write_json(QA / "grid_paf_QA_v1.json", summary)
    (REPORTS / "grid_paf_report_v1.md").write_text(
        "# Grid PAF (3 representative diseases)\n\n"
        + qa.to_string(index=False)
        + f"\n\nstatus: {status}\n",
        encoding="utf-8",
    )
    (SCRIPTS / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    outputs += [QA / "grid_paf_QA_v1.json", REPORTS / "grid_paf_report_v1.md",
                SCRIPTS / Path(__file__).name]
    pd.DataFrame({"path": [str(p) for p in outputs], "sha256": [sha256(p) for p in outputs]}).to_csv(
        ROOT / "paf_grid_multidisease_formal_manifest_v1.csv", index=False, encoding="utf-8-sig"
    )
    if status != "PASS":
        raise RuntimeError("grid PAF QA failed: " + "; ".join(problems))
    (ROOT / "COMPLETE.flag").write_text(
        f"paf_grid_multidisease_v1_20261005 completed {summary['completed_at_utc']}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
