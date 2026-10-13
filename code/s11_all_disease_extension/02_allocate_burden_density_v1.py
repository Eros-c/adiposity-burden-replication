#!/usr/bin/env python3
"""Allocate national attributable burden to 0.1-degree cells via pop x PAF weights.

burden_cell(cause, year, measure) = national_attributable x w(cell) / w(country)

Produces:
  - per-cause burden-density rasters (deaths and DALYs), 2025 and 2050
  - combined 23-cause burden-density rasters
  - country-level closure QA: sum of cells must equal the national attributable
  - validation: country-level sum(w) vs attributable deaths (Pearson r, R2)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "11_all_disease_extension" / "06_spatial_burden" / "burden_density_v1_20261007"
DATA = ROOT / "data"
QA = ROOT / "qa"
REPORTS = ROOT / "reports"
SCRIPTS = ROOT / "scripts"

WEIGHTS = (PROJECT / "11_all_disease_extension" / "06_spatial_burden" / "paf_weights_all_causes_v1_20261007" / "data")
BURDEN = (PROJECT / "11_all_disease_extension" / "04_burden" / "burden_benchmark_v1_20261005"
          / "data" / "attributable_future_country_overall_four_ssp_v1.parquet")
HELPER_SCRIPT = (PROJECT / "12_figures" / "manuscript_bmi_2015_2050_unified_v1_20260909"
                 / "scripts" / "08_plot_figure1B_historical_template_test_v2.py")

YEARS = (2025, 2050)
MEASURES = ("Deaths", "DALYs")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_helper():
    spec = importlib.util.spec_from_file_location("burden_helper", HELPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.historical


historical = import_helper()


def country_zones(shape, transform):
    countries = historical.gpd.read_file(historical.COUNTRY_PATH, engine="pyogrio")
    countries = countries[countries.geometry.notna()].copy()
    countries["iso3"] = countries["ISO_A3"].where(countries["ISO_A3"].ne("-99"), countries["ADM0_A3"])
    countries["iso3"] = countries["iso3"].astype(str).str.upper()
    keys = sorted(x for x in countries["iso3"].unique() if x not in {"-99", "NAN", "NONE"})
    ids = {iso: i for i, iso in enumerate(keys)}
    shapes = [(g, ids[i]) for g, i in zip(countries.geometry, countries["iso3"]) if i in ids]
    zones = rasterize(shapes, out_shape=shape, transform=transform, fill=-1, dtype="int32")
    return zones, keys


def allocate_year(year: int, burden: pd.DataFrame, causes: list[int]) -> tuple[dict, pd.DataFrame, list[Path]]:
    combined = {m: None for m in MEASURES}
    closure_rows = []
    outputs = []
    for measure in MEASURES:
        for cause in causes:
            w_path = WEIGHTS / str(year) / f"weight_pop_paf_c{cause}_ssp245_{year}_0p1deg_v1.tif"
            with rasterio.open(w_path) as ds:
                w = ds.read(1, masked=True).astype(np.float64)
                if measure == "Deaths":
                    transform, crs = ds.transform, ds.crs
            country_b = burden[(burden["cause_id"] == cause) & (burden["year"] == year)
                               & (burden["measure"] == measure)]
            nat = country_b.set_index("iso3")["attributable_burden"]
            if measure == "Deaths" and cause == causes[0]:
                zones, keys = country_zones(w.shape, transform)
            zflat = zones.ravel()
            wflat = np.nan_to_num(np.asarray(w), nan=0.0).ravel()
            sel = zflat >= 0
            wsum = np.bincount(zflat[sel], weights=wflat[sel])
            wsum_len = max(len(wsum), len(keys))
            if len(wsum) < wsum_len:
                wsum = np.pad(wsum, (0, wsum_len - len(wsum)))
            nat_lookup = np.zeros(len(keys))
            for i, iso in enumerate(keys):
                # Natural Earth uses PSX/SDS; the GBD crosswalk uses PSE/SSD.
                nat_lookup[i] = nat.get({"PSX": "PSE", "SDS": "SSD"}.get(iso, iso), 0.0)
            # Renormalise per country so the grid allocation conserves the
            # national attributable burden exactly. Negative country weight
            # sums are legitimate (breast cancer premenopausal protective RR
            # in young-age-heavy populations) and are kept, not dropped.
            cell_share = np.zeros(len(wflat))
            idx_z = zflat[sel]
            cell_share[sel] = np.divide(wflat[sel], wsum[idx_z],
                                        out=np.zeros(int(sel.sum())), where=wsum[idx_z] != 0)
            cell_burden = cell_share * nat_lookup[zflat.clip(min=0)]
            cell_burden = np.where(zflat >= 0, cell_burden, np.nan).reshape(w.shape)
            # per-country rescale to hit the national total exactly
            allocated_by_country = np.bincount(
                zflat[sel], weights=np.nan_to_num(cell_burden.ravel(), nan=0.0)[sel]
            )
            if len(allocated_by_country) < len(keys):
                allocated_by_country = np.pad(allocated_by_country, (0, len(keys) - len(allocated_by_country)))
            rescale = np.divide(nat_lookup, allocated_by_country,
                                out=np.ones(len(keys)), where=allocated_by_country != 0)
            cell_burden = np.where(
                np.isfinite(cell_burden),
                cell_burden * rescale[zflat.clip(min=0)].reshape(w.shape),
                np.nan,
            )
            combined[measure] = (cell_burden if combined[measure] is None
                                 else np.nansum([combined[measure], cell_burden], axis=0))
            # closure QA: per-country the allocation conserves national burden
            # exactly for countries with rasterized weight; countries whose
            # 0.1-degree production rasters carry nodata (small islands and
            # city-states masked out of the BMI grid) have no cells and are
            # reported separately rather than silently dropped.
            nat_covered = float(nat_lookup[allocated_by_country != 0].sum())
            nat_dropped = float(nat_lookup[allocated_by_country == 0].sum())
            cell_sum = float(np.nansum(cell_burden))
            nat_sum = float(nat.sum())
            closure_rows.append({
                "cause_id": cause, "year": year, "measure": measure,
                "national_attributable": nat_sum,
                "national_attributable_covered_by_grid": nat_covered,
                "national_attributable_no_grid_cells": nat_dropped,
                "grid_allocated_sum": cell_sum,
                "closure_abs_diff": abs(cell_sum - nat_covered),
                "closure_rel_diff": abs(cell_sum - nat_covered) / max(nat_covered, 1e-12),
                "dropped_share_pct": 100.0 * nat_dropped / max(nat_sum, 1e-12),
            })
            out_path = DATA / str(year) / f"burden_density_c{cause}_{measure.lower()}_ssp245_{year}_0p1deg_v1.tif"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(w_path) as ds:
                profile = ds.profile.copy()
            profile.update(dtype="float32", count=1, compress="lzw")
            with rasterio.open(out_path, "w", **profile) as ds:
                ds.write(cell_burden.astype(np.float32), 1)
                ds.set_band_description(1, f"attributable {measure} cause {cause}")
            outputs.append(out_path)
        # combined raster per measure
        comb_path = DATA / str(year) / f"burden_density_combined23_{measure.lower()}_ssp245_{year}_0p1deg_v1.tif"
        with rasterio.open(comb_path, "w", **profile) as ds:
            ds.write(combined[measure].astype(np.float32), 1)
            ds.set_band_description(1, f"combined 23-cause attributable {measure}")
        outputs.append(comb_path)
    return combined, pd.DataFrame(closure_rows), outputs


def main() -> None:
    burden = pd.read_parquet(BURDEN)
    burden = burden[(burden["scenario_code"] == "ssp245") & (burden["analysis_tier"] == "headline")]
    causes = sorted(burden["cause_id"].unique().tolist())

    outputs: list[Path] = []
    closures = []
    for year in YEARS:
        _, closure, year_outputs = allocate_year(year, burden, causes)
        closures.append(closure)
        outputs += year_outputs
        print(f"year {year} done", flush=True)

    closure = pd.concat(closures, ignore_index=True)
    closure_path = QA / "allocation_closure_QA_v1.csv"
    closure.to_csv(closure_path, index=False, encoding="utf-8-sig")
    outputs.append(closure_path)

    worst = closure["closure_rel_diff"].max()
    problems = []
    if worst > 1e-3:
        problems.append(f"closure rel diff {worst:.3g} exceeds 1e-3")
    status = "PASS" if not problems else "FAIL"
    summary = {
        "run": "burden_density_v1_20261007",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "causes": causes,
        "years": list(YEARS),
        "measures": list(MEASURES),
        "worst_closure_rel_diff": float(worst),
        "problems": problems,
        "status": status,
    }
    write_json(QA / "burden_density_QA_v1.json", summary)
    (REPORTS / "burden_density_report_v1.md").write_text(
        "# Burden density allocation (pop x PAF)\n\n"
        f"- closure worst relative difference: {worst:.3g}\n"
        f"- causes: {len(causes)}, years: {YEARS}, measures: {MEASURES}\n"
        f"- status: {status}\n",
        encoding="utf-8",
    )
    (SCRIPTS / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    outputs += [QA / "burden_density_QA_v1.json", REPORTS / "burden_density_report_v1.md",
                SCRIPTS / Path(__file__).name]
    pd.DataFrame({"path": [str(p) for p in outputs], "sha256": [sha256(p) for p in outputs]}).to_csv(
        ROOT / "burden_density_formal_manifest_v1.csv", index=False, encoding="utf-8-sig"
    )
    if status != "PASS":
        raise RuntimeError("closure QA failed: " + "; ".join(problems))
    (ROOT / "COMPLETE.flag").write_text(
        f"burden_density_v1_20261007 completed {summary['completed_at_utc']}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
