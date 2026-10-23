"""Grid PAF and pop-x-PAF weight rasters for all 23 headline causes."""

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
ROOT = PROJECT / "11_all_disease_extension" / "06_spatial_burden" / "paf_weights_all_causes_v1_20261007"
DATA = ROOT / "data"
QA = ROOT / "qa"
REPORTS = ROOT / "reports"
SCRIPTS = ROOT / "scripts"

FUT_ROOT = (PROJECT / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906"
            / "outputs" / "ssp245")
RR_EXPANDED = (PROJECT / "11_all_disease_extension" / "02_rr_crosswalk" / "rr_crosswalk_v1_20261005"
               / "data" / "RR_crosswalk_by_bmi_category_v1.csv")
CAUSE_SET = (PROJECT / "11_all_disease_extension" / "02_rr_crosswalk" / "rr_crosswalk_v1_20261005"
             / "qa" / "RR_crosswalk_decisions_v1.csv")

YEARS = (2025, 2050)
SEXES = ("Female", "Male")
AGE_KEYS = ["25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59",
            "60_64", "65_69", "70_74", "75_79", "80_84", "85_89", "90_94", "95plus"]
RR_AGE = ["25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64",
          "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+"]
CAT_KEYS = ["bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30",
            "bmi_30_35", "bmi_35_40", "bmi_ge_40"]
NODATA = -9999.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_rr() -> dict:
    rr = pd.read_csv(RR_EXPANDED, encoding="utf-8-sig")
    lookup = {}
    for (cid, sex, age), part in rr.groupby(["cause_id", "sex", "age_group"]):
        part = part.set_index("bmi_category").reindex(CAT_KEYS)
        lookup[(int(cid), sex, age)] = part["RR"].to_numpy(np.float64)
    return lookup


def process_year(year: int, rr_lookup: dict, causes: list[int]) -> dict:
    prob_path = FUT_ROOT / str(year) / f"probability_age_sex_ssp245_{year}_0p1deg_v1.tif"
    pop_path = FUT_ROOT / str(year) / f"population_age_sex_ssp245_{year}_0p1deg_v1.tif"
    prob_ds = rasterio.open(prob_path)
    pop_ds = rasterio.open(pop_path)
    profile = prob_ds.profile.copy()
    profile.update(dtype="float32", count=1, compress="lzw")

    weight = {c: None for c in causes}   # sum_sa pop x PAF
    pop_total = None
    for sex_i, sex in enumerate(SEXES):
        for age_i, age_key in enumerate(AGE_KEYS):
            band_pop = sex_i * 15 + age_i + 1
            pop = pop_ds.read(band_pop, masked=True).astype(np.float64)
            pop_f = np.ma.filled(pop, 0.0)
            pop_total = pop_f if pop_total is None else pop_total + pop_f
            shares = []
            for cat_i in range(7):
                band = (sex_i * 15 + age_i) * 7 + cat_i + 1
                share = prob_ds.read(band, masked=True).astype(np.float64)
                shares.append(np.ma.filled(share, 0.0))
            shares = np.stack(shares)  # (7, H, W)
            valid = (pop_f > 0) & (shares.sum(axis=0) > 0)
            for cause in causes:
                rr_vec = rr_lookup.get((cause, sex, RR_AGE[age_i]))
                if rr_vec is None:
                    continue  # female-only causes have no male rows
                wr = np.tensordot(rr_vec, shares, axes=(0, 0))
                paf = np.where(valid & (wr > 0), (wr - 1.0) / wr, 0.0)
                contrib = np.where(valid, paf * pop_f, 0.0)
                weight[cause] = contrib if weight[cause] is None else weight[cause] + contrib
    prob_ds.close()
    pop_ds.close()

    out_rows = []
    out_dir = DATA / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cause in causes:
        w = weight[cause]
        if w is None:
            raise RuntimeError(f"cause {cause} has no slices")
        paf_display = np.divide(w, pop_total, out=np.zeros_like(w), where=pop_total > 0)
        w_path = out_dir / f"weight_pop_paf_c{cause}_ssp245_{year}_0p1deg_v1.tif"
        with rasterio.open(w_path, "w", **profile) as ds:
            ds.write(np.where(pop_total > 0, w, np.nan).astype(np.float32), 1)
            ds.set_band_description(1, f"weight pop x PAF cause {cause}")
        p_path = out_dir / f"paf_display_c{cause}_ssp245_{year}_0p1deg_v1.tif"
        with rasterio.open(p_path, "w", **profile) as ds:
            ds.write(np.where(pop_total > 0, paf_display * 100.0, np.nan).astype(np.float32), 1)
            ds.set_band_description(1, f"PAF percent cause {cause}")
        out_rows.append({
            "cause_id": cause, "year": year,
            "weight_path": str(w_path), "paf_path": str(p_path),
            "paf_pct_min": float(np.nanmin(paf_display * 100.0)),
            "paf_pct_max": float(np.nanmax(paf_display * 100.0)),
        })
    return {"year": year, "rows": out_rows, "pop_total": pop_total}


def main() -> None:
    rr_lookup = load_rr()
    cause_set = pd.read_csv(CAUSE_SET, encoding="utf-8-sig")
    causes = sorted(cause_set.loc[cause_set["analysis_tier"] == "headline", "cause_id"].astype(int).tolist())
    print("causes:", causes, flush=True)

    all_rows = []
    outputs = []
    for year in YEARS:
        result = process_year(year, rr_lookup, causes)
        all_rows += result["rows"]
        for row in result["rows"]:
            outputs += [Path(row["weight_path"]), Path(row["paf_path"])]
        print(f"year {year} done", flush=True)

    inventory = pd.DataFrame(all_rows)
    inv_path = QA / "paf_weights_inventory_v1.csv"
    inventory.to_csv(inv_path, index=False, encoding="utf-8-sig")
    outputs.append(inv_path)

    problems = []
    if (inventory["paf_pct_min"] < 0).any() or (inventory["paf_pct_max"] >= 100).any():
        problems.append("PAF outside [0, 100) percent")
    status = "PASS" if not problems else "FAIL"
    summary = {
        "run": "paf_weights_all_causes_v1_20261007",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "causes": causes,
        "years": list(YEARS),
        "method": "w(cell) = sum_sex,age pop_sa x PAF_sa; display PAF = w / pop_total",
        "problems": problems,
        "status": status,
    }
    write_json(QA / "paf_weights_QA_v1.json", summary)
    (SCRIPTS / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    outputs += [QA / "paf_weights_QA_v1.json", SCRIPTS / Path(__file__).name]
    pd.DataFrame({"path": [str(p) for p in outputs], "sha256": [sha256(p) for p in outputs]}).to_csv(
        ROOT / "paf_weights_formal_manifest_v1.csv", index=False, encoding="utf-8-sig"
    )
    if status != "PASS":
        raise RuntimeError("QA failed: " + "; ".join(problems))
    (ROOT / "COMPLETE.flag").write_text(
        f"paf_weights_all_causes_v1_20261007 completed {summary['completed_at_utc']}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
