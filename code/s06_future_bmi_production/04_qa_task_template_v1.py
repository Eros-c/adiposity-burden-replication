#!/usr/bin/env python3
"""Independent, file-level QA for the full-covariate 0.1-degree pilot."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(_cfg.ROOT / ".codex_tmp" / "fullcov_0p1_pilot_v1")
COG_DIR = ROOT / "data" / "0p1deg" / "cog" / "ssp585" / "2050"
PARENT_DIR = Path(
    _cfg.RESULTS / "06_future_bmi_2025_2050" / "full_covariate_annual_0p5deg_production_v1_20260906" / "outputs" / "ssp585" / "2050"
)
NODATA = -9999.0
WIDTH = 3600
HEIGHT = 1800
PARENT_WIDTH = 720
PARENT_HEIGHT = 360
BMI_MIDPOINTS = np.array([17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5])

FAMILIES = {
    "probability_overall": 7,
    "probability_sex": 14,
    "probability_age": 105,
    "probability_age_sex": 210,
    "population_overall": 1,
    "population_sex": 2,
    "population_age": 15,
    "population_age_sex": 30,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_path(family: str) -> Path:
    return COG_DIR / f"{family}_ssp585_2050_0p1deg_v1.tif"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def valid(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values != NODATA)


def parent_sum(values: np.ndarray) -> np.ndarray:
    return values.reshape(PARENT_HEIGHT, 5, PARENT_WIDTH, 5).sum(axis=(1, 3))


def check_inventory() -> tuple[list[dict[str, object]], bool]:
    expected_hashes = {
        Path(row["path"]).name: row["sha256"]
        for row in csv.DictReader(
            (ROOT / "checksums" / "Gate_FC03_formal_COG_SHA256_ssp585_2050_v1.csv")
            .open("r", encoding="utf-8-sig")
        )
    }
    rows: list[dict[str, object]] = []
    all_pass = True
    for family, expected_bands in FAMILIES.items():
        path = file_path(family)
        digest = sha256(path)
        with rasterio.open(path) as dataset:
            structure = dataset.tags(ns="IMAGE_STRUCTURE")
            passed = (
                dataset.count == expected_bands
                and dataset.width == WIDTH
                and dataset.height == HEIGHT
                and dataset.crs.to_string() == "EPSG:4326"
                and abs(dataset.res[0] - 0.1) < 1e-12
                and abs(dataset.res[1] - 0.1) < 1e-12
                and dataset.nodata == NODATA
                and structure.get("LAYOUT") == "COG"
                and all(bool(description) for description in dataset.descriptions)
                and digest == expected_hashes.get(path.name)
            )
            rows.append(
                {
                    "family": family,
                    "bands": dataset.count,
                    "width": dataset.width,
                    "height": dataset.height,
                    "crs": dataset.crs.to_string(),
                    "resolution": dataset.res[0],
                    "nodata": dataset.nodata,
                    "layout": structure.get("LAYOUT", ""),
                    "overviews": "|".join(map(str, dataset.overviews(1))),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                    "matches_fc03_sha256": digest == expected_hashes.get(path.name),
                    "pass": passed,
                }
            )
            all_pass &= passed
    return rows, all_pass


def probability_group_qa(
    probability_family: str,
    population_family: str,
    groups: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with rasterio.open(file_path(probability_family)) as probability_dataset, rasterio.open(
        file_path(population_family)
    ) as population_dataset:
        for group in groups:
            population = population_dataset.read(group).astype(np.float64)
            population_valid = valid(population) & (population > 0)
            bands = list(range((group - 1) * 7 + 1, group * 7 + 1))
            probability = probability_dataset.read(bands).astype(np.float64)
            probability_valid = np.all(valid(probability), axis=0)
            closure = probability.sum(axis=0)
            mask_mismatch = int(np.count_nonzero(probability_valid != population_valid))
            values = probability[:, probability_valid]
            rows.append(
                {
                    "probability_family": probability_family,
                    "population_family": population_family,
                    "group": group,
                    "positive_population_cells": int(np.count_nonzero(population_valid)),
                    "mask_mismatch_cells": mask_mismatch,
                    "minimum_probability": float(values.min()),
                    "maximum_probability": float(values.max()),
                    "maximum_closure_error": float(
                        np.max(np.abs(closure[probability_valid] - 1.0))
                    ),
                    "pass": bool(
                        mask_mismatch == 0
                        and values.min() >= 0
                        and values.max() <= 1
                        and np.max(np.abs(closure[probability_valid] - 1.0)) <= 1e-6
                    ),
                }
            )
    return rows


def population_conservation() -> tuple[list[dict[str, object]], np.ndarray]:
    with rasterio.open(file_path("population_overall")) as dataset:
        overall = dataset.read(1).astype(np.float64)
    overall_valid = valid(overall)
    overall_values = np.where(overall_valid, overall, 0.0)

    sums: dict[str, np.ndarray] = {}
    for family in ("population_sex", "population_age", "population_age_sex"):
        accumulated = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
        with rasterio.open(file_path(family)) as dataset:
            for band in range(1, dataset.count + 1):
                values = dataset.read(band).astype(np.float64)
                accumulated += np.where(valid(values), values, 0.0)
        sums[family] = accumulated

    rows: list[dict[str, object]] = []
    for family, accumulated in sums.items():
        error = np.abs(accumulated[overall_valid] - overall_values[overall_valid])
        rows.append(
            {
                "comparison": f"{family}_sum_vs_population_overall",
                "valid_overall_cells": int(np.count_nonzero(overall_valid)),
                "maximum_absolute_error_persons": float(error.max()),
                "maximum_relative_error": float(
                    np.max(error / np.maximum(overall_values[overall_valid], np.finfo(float).tiny))
                ),
                "pass": bool(
                    np.max(error / np.maximum(overall_values[overall_valid], 1.0)) <= 2e-6
                ),
            }
        )
    return rows, overall_values


def parent_and_heterogeneity_qa(
    child_population: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    with rasterio.open(file_path("probability_overall")) as dataset:
        probability = dataset.read().astype(np.float64)
    child_mask = child_population > 0
    weighted = probability.copy()
    weighted[:, ~child_mask] = 0.0
    child_population_2d = child_population.reshape(HEIGHT, WIDTH)
    child_parent_population = parent_sum(child_population_2d)

    with rasterio.open(PARENT_DIR / "adult_population_overall_0p5deg_v1.tif") as dataset:
        target_population = dataset.read(1).astype(np.float64)
        target_population = np.where(valid(target_population), target_population, 0.0)
    with rasterio.open(PARENT_DIR / "bmi7_probability_overall_0p5deg_v1.tif") as dataset:
        target_probability = dataset.read().astype(np.float64)

    positive_parent = target_population > 0
    rows: list[dict[str, object]] = []
    population_error = np.abs(child_parent_population[positive_parent] - target_population[positive_parent])
    rows.append(
        {
            "comparison": "0p1_population_to_formal_0p5_population",
            "maximum_absolute_error": float(population_error.max()),
            "maximum_relative_error": float(
                np.max(population_error / target_population[positive_parent])
            ),
            "pass": bool(np.max(population_error / target_population[positive_parent]) <= 1e-6),
        }
    )

    for category in range(7):
        numerator = parent_sum(
            weighted[category].reshape(HEIGHT, WIDTH) * child_population_2d
        )
        observed = np.zeros_like(numerator)
        observed[positive_parent] = numerator[positive_parent] / child_parent_population[positive_parent]
        difference = np.abs(observed[positive_parent] - target_probability[category][positive_parent])
        rows.append(
            {
                "comparison": f"0p1_probability_category_{category + 1}_to_formal_0p5",
                "maximum_absolute_error": float(difference.max()),
                "maximum_relative_error": "not_applicable",
                "pass": bool(difference.max() <= 5e-6),
            }
        )

    obesity = probability[4:7].sum(axis=0)
    mean_bmi = np.tensordot(BMI_MIDPOINTS, probability, axes=(0, 0))
    obesity[~child_mask] = 0.0
    mean_bmi[~child_mask] = 0.0
    populated_children = parent_sum(child_mask.reshape(HEIGHT, WIDTH).astype(np.float64))

    obesity_mean = parent_sum(obesity.reshape(HEIGHT, WIDTH) * child_population_2d)
    obesity_mean[positive_parent] /= child_parent_population[positive_parent]
    obesity_variance = parent_sum(
        ((obesity.reshape(HEIGHT, WIDTH) - np.repeat(np.repeat(obesity_mean, 5, axis=0), 5, axis=1)) ** 2)
        * child_population_2d
    )
    obesity_variance[positive_parent] /= child_parent_population[positive_parent]
    obesity_sd_pp = 100.0 * np.sqrt(np.maximum(obesity_variance, 0.0))

    mean_bmi_parent = parent_sum(mean_bmi.reshape(HEIGHT, WIDTH) * child_population_2d)
    mean_bmi_parent[positive_parent] /= child_parent_population[positive_parent]
    mean_bmi_variance = parent_sum(
        ((mean_bmi.reshape(HEIGHT, WIDTH) - np.repeat(np.repeat(mean_bmi_parent, 5, axis=0), 5, axis=1)) ** 2)
        * child_population_2d
    )
    mean_bmi_variance[positive_parent] /= child_parent_population[positive_parent]
    mean_bmi_sd = np.sqrt(np.maximum(mean_bmi_variance, 0.0))

    eligible = positive_parent & (populated_children >= 2)
    heterogeneity = {
        "eligible_parent_cells_with_at_least_two_populated_children": int(np.count_nonzero(eligible)),
        "obesity_within_parent_sd_pp_median": float(np.median(obesity_sd_pp[eligible])),
        "obesity_within_parent_sd_pp_p95": float(np.quantile(obesity_sd_pp[eligible], 0.95)),
        "obesity_within_parent_sd_pp_maximum": float(np.max(obesity_sd_pp[eligible])),
        "parents_with_obesity_sd_above_0p1pp_fraction": float(np.mean(obesity_sd_pp[eligible] > 0.1)),
        "mean_bmi_within_parent_sd_median": float(np.median(mean_bmi_sd[eligible])),
        "mean_bmi_within_parent_sd_p95": float(np.quantile(mean_bmi_sd[eligible], 0.95)),
        "mean_bmi_within_parent_sd_maximum": float(np.max(mean_bmi_sd[eligible])),
        "pass": bool(np.quantile(obesity_sd_pp[eligible], 0.95) > 0.01),
    }
    return rows, heterogeneity


def main() -> int:
    started = now_utc()
    inventory, inventory_pass = check_inventory()
    inventory_path = ROOT / "qa" / "Gate_FC04_independent_COG_inventory_ssp585_2050_v1.csv"
    write_csv(inventory_path, inventory)

    group_rows: list[dict[str, object]] = []
    group_rows.extend(probability_group_qa("probability_overall", "population_overall", [1]))
    group_rows.extend(probability_group_qa("probability_sex", "population_sex", [1, 2]))
    group_rows.extend(probability_group_qa("probability_age", "population_age", [1, 8, 15]))
    group_rows.extend(
        probability_group_qa("probability_age_sex", "population_age_sex", [1, 8, 15, 16, 23, 30])
    )
    group_path = ROOT / "qa" / "Gate_FC04_probability_mask_closure_ssp585_2050_v1.csv"
    write_csv(group_path, group_rows)

    population_rows, child_population = population_conservation()
    population_path = ROOT / "qa" / "Gate_FC04_population_conservation_ssp585_2050_v1.csv"
    write_csv(population_path, population_rows)

    parent_rows, heterogeneity = parent_and_heterogeneity_qa(child_population)
    parent_path = ROOT / "qa" / "Gate_FC04_parent_closure_ssp585_2050_v1.csv"
    write_csv(parent_path, parent_rows)
    heterogeneity_path = ROOT / "qa" / "Gate_FC04_within_parent_heterogeneity_ssp585_2050_v1.json"
    write_json(heterogeneity_path, heterogeneity)

    partial_files = [
        str(path) for path in COG_DIR.iterdir()
        if path.is_file() and (".tmp" in path.name.lower() or ".partial" in path.name.lower())
    ]
    checks = [
        {
            "check_id": "FC04_01",
            "requirement": "All eight COGs independently match geometry, metadata, and FC03 SHA256",
            "observed": f"{sum(bool(row['pass']) for row in inventory)}/8",
            "pass": inventory_pass,
        },
        {
            "check_id": "FC04_02",
            "requirement": "Overall and representative derived strata have exact population masks and close to one",
            "observed": f"{sum(bool(row['pass']) for row in group_rows)}/{len(group_rows)}",
            "pass": all(bool(row["pass"]) for row in group_rows),
        },
        {
            "check_id": "FC04_03",
            "requirement": "Sex, age, and age-sex population sums reproduce overall population",
            "observed": f"{sum(bool(row['pass']) for row in population_rows)}/3",
            "pass": all(bool(row["pass"]) for row in population_rows),
        },
        {
            "check_id": "FC04_04",
            "requirement": "0.1-degree overall population and composition reproduce formal 0.5-degree parents",
            "observed": f"{sum(bool(row['pass']) for row in parent_rows)}/{len(parent_rows)}",
            "pass": all(bool(row["pass"]) for row in parent_rows),
        },
        {
            "check_id": "FC04_05",
            "requirement": "The nested 0.1-degree surface contains nonzero within-parent BMI heterogeneity",
            "observed": heterogeneity["obesity_within_parent_sd_pp_p95"],
            "pass": bool(heterogeneity["pass"]),
        },
        {
            "check_id": "FC04_06",
            "requirement": "No temporary or partial files remain in the formal COG directory",
            "observed": len(partial_files),
            "pass": len(partial_files) == 0,
        },
    ]
    checks_path = ROOT / "qa" / "Gate_FC04_checks_ssp585_2050_v1.csv"
    write_csv(checks_path, checks)
    status = (
        "PASS_GATE_FC04_INDEPENDENT_SPATIAL_QA"
        if all(bool(row["pass"]) for row in checks)
        else "FAIL_GATE_FC04_INDEPENDENT_SPATIAL_QA"
    )
    summary = {
        "status": status,
        "started_at_utc": started,
        "completed_at_utc": now_utc(),
        "scenario": "SSP5-8.5",
        "year": 2050,
        "formal_cogs": 8,
        "active_overall_cells": int(group_rows[0]["positive_population_cells"]),
        "inventory_pass": inventory_pass,
        "representative_probability_groups_pass": all(bool(row["pass"]) for row in group_rows),
        "population_conservation_pass": all(bool(row["pass"]) for row in population_rows),
        "parent_closure_pass": all(bool(row["pass"]) for row in parent_rows),
        "heterogeneity": heterogeneity,
        "partial_files": partial_files,
    }
    write_json(ROOT / "qa" / "Gate_FC04_summary_ssp585_2050_v1.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
