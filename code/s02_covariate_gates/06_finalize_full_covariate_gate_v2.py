"""Freeze the full-covariate extension and write auditable gate artifacts."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


RUN = Path(_cfg.ROOT / ".codex_tmp" / "future_results_v1" / "02_harmonized_covariates" / "full_covariate_extension_v1_20260905")
RAW = Path(_cfg.ROOT / ".codex_tmp" / "full_covariate_raw_v1")
HIST = Path(_cfg.ROOT / ".codex_tmp" / "historic_results_v1")
GATE_C = Path(_cfg.ROOT / ".codex_tmp" / "future_results_v1" / "02_harmonized_covariates" / "gate_c_v1_country_coverage_remediated_20260828_122619")

CONTRACTS = RUN / "contracts"
QA = RUN / "qa"
REPORTS = RUN / "reports"
for directory in (CONTRACTS, QA, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)


def canonical(path: Path) -> str:
    return str(path.resolve())


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite versioned artifact: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_rows(paths: list[Path], root_label: str) -> list[dict]:
    rows: list[dict] = []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        print(f"HASH {root_label} {index}/{total}: {path.name}", flush=True)
        stat = path.stat()
        rows.append(
            {
                "collection": root_label,
                "path": canonical(path),
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256(path),
            }
        )
    return rows


def read_status_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


expected = read_status_csv(QA / "Full_covariate_expected_file_count_QA_v2.csv")
geometry = read_status_csv(QA / "Full_covariate_geometry_QA_v2.csv")
transition = read_status_csv(QA / "PM25_trailing5y_2024_2025_transition_QA_v2.csv")
require(all(row["status"] == "PASS" for row in expected), "Final file-count QA is not PASS")
require(len(geometry) == 435 and all(row["status"] == "PASS" for row in geometry), "Final geometry QA is not 435/435 PASS")
require(max(float(row["population_weighted_mean_abs_change"]) for row in transition) < 1.0, "PM2.5 transition exceeds the frozen threshold")

contracts = [
    {
        "dataset_id": "PM25_MMR",
        "role": "historical and future fixed effect; coefficient must be refitted",
        "source_path": canonical(RAW / "CMIP6_AEROSOL" / "NorESM2-LM"),
        "source_definition": "NorESM2-LM near-surface dry-aerosol PM2.5 mass mixing ratio (mmrpm2p5; lowest hybrid level)",
        "historical_coverage": "1990-2024 monthly source; final model rasters 2006-2024",
        "future_coverage": "2025-2050 annual; SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5",
        "unit": "microgram PM2.5 per kilogram dry air (source kg kg-1 multiplied by 1e9)",
        "transform_interpolation": "annual mean, trailing five-year mean, scenario log-ratio anomaly anchored to common 2024",
        "final_path": canonical(RUN / "data" / "0p5deg" / "pm25_mmr_trailing5y_v2"),
        "status": "PASS_WITH_WARN",
        "required_sensitivity": "single-GCM structural uncertainty; do not interpret as microgram per cubic metre; never reuse old concentration coefficient",
    },
    {
        "dataset_id": "FUTURE_NTL",
        "role": "future NTL fixed effect and interactions",
        "source_path": canonical(RAW / "FUT_NTL" / "extracted"),
        "source_definition": "scenario-specific projected night-time light intensity",
        "historical_coverage": "historical NTL retained from existing 2015-2024 covariate stack",
        "future_coverage": "2025-2050 annual; interpolated from 2020/2025/.../2050 nodes for SSP1/2/3/5",
        "unit": "provider intensity, analysed on log1p scale",
        "transform_interpolation": "log1p before area averaging; linear interpolation; 2024 delta anchoring; overlap slope capped at 2.0",
        "final_path": canonical(RUN / "data" / "0p5deg" / "ntl_log"),
        "status": "PASS_WITH_WARN",
        "required_sensitivity": "compare bridge slopes 1.0 and 2.0 because fitted slope reached the prespecified upper bound",
    },
    {
        "dataset_id": "SSP_URBAN_SHARE",
        "role": "future urbanisation fixed effect",
        "source_path": canonical(RAW / "ssp_urban" / "Gao_Pesaresi_SEDAC_1km" / "extracted"),
        "source_definition": "Gao-Pesaresi SSP urban land extent fraction",
        "historical_coverage": "historical GHS-SMOD urban-area share retained through 2024",
        "future_coverage": "2025-2050 annual; logit interpolation from decadal SSP nodes",
        "unit": "cell fraction, 0-1",
        "transform_interpolation": "area averaging, logit interpolation, change anchored to observed 2020/2024 GHS-SMOD level",
        "final_path": canonical(RUN / "data" / "0p5deg" / "urban_share"),
        "status": "PASS_PROXY",
        "required_sensitivity": "document the historical-to-future product bridge",
    },
    {
        "dataset_id": "SSP_BUILT_FRACTION_PROXY",
        "role": "future trajectory proxy for built-fraction fixed effect",
        "source_path": canonical(RAW / "ssp_urban" / "Gao_Pesaresi_SEDAC_1km" / "extracted"),
        "source_definition": "SSP urban land extent change applied to observed GHS built-surface fraction",
        "historical_coverage": "historical GHS built-surface fraction retained through 2024",
        "future_coverage": "2025-2050 annual; logit interpolation from decadal SSP nodes",
        "unit": "cell fraction, 0-1",
        "transform_interpolation": "area averaging, logit interpolation, change anchored to observed GHS built fraction",
        "final_path": canonical(RUN / "data" / "0p5deg" / "built_fraction_proxy"),
        "status": "WARN_PROXY_SHARED_SOURCE",
        "required_sensitivity": "primary full model versus model excluding built-fraction proxy; urban-share and built proxy share one future source",
    },
]
contract_fields = list(contracts[0])
contract_path = CONTRACTS / "Full_covariate_dataset_contracts_v2.csv"
write_csv(contract_path, contracts, contract_fields)

gate_rows = [
    {"check_id": "FC01", "check": "final file counts", "status": "PASS", "evidence": "5 groups match expected counts"},
    {"check_id": "FC02", "check": "final 0.5-degree geometry", "status": "PASS", "evidence": "435/435 rasters are EPSG:4326 and grid-consistent"},
    {"check_id": "FC03", "check": "PM2.5 variable identity", "status": "PASS_WITH_WARN", "evidence": "mass mixing ratio, not surface concentration; refit required"},
    {"check_id": "FC04", "check": "PM2.5 2024-2025 continuity", "status": "PASS", "evidence": "population-weighted mean absolute change 0.40-0.62 microgram/kg"},
    {"check_id": "FC05", "check": "NTL future bridge", "status": "PASS_WITH_WARN", "evidence": "weighted r=0.733; fitted slope 2.004 capped at 2.0"},
    {"check_id": "FC06", "check": "urban-share bridge", "status": "PASS_PROXY", "evidence": "weighted r=0.675; logit change anchored to historical GHS-SMOD"},
    {"check_id": "FC07", "check": "built-fraction future definition", "status": "WARN_PROXY_SHARED_SOURCE", "evidence": "SSP urban-land change is a proxy, not direct future built surface"},
    {"check_id": "FC08", "check": "2050 scenario signal", "status": "PASS", "evidence": "all four variables retain non-zero cross-scenario spread"},
    {"check_id": "FC09", "check": "training/prediction definition consistency", "status": "PENDING_REFIT_INPUT_RECONSTRUCTION", "evidence": "new PM2.5 definition must be extracted for every training record before refit"},
    {"check_id": "FC10", "check": "formal BMI prediction authorization", "status": "NOT_AUTHORIZED", "evidence": "requires refit, internal/country/source/time validation and non-inferiority gate"},
]
gate_path = QA / "Full_covariate_gate_status_v2.csv"
write_csv(gate_path, gate_rows, list(gate_rows[0]))

model_contract = {
    "version": "v2",
    "frozen_at": datetime.now(timezone.utc).isoformat(),
    "historical_anchor_1990_2024": "NCD-RisC country-year-age-sex seven-category BMI composition on the ALR scale",
    "future_primary_anchor_ssp2_2025_2050": "GBD adult overweight and obesity forecasts constrain broad national age-sex totals; the national composition model allocates them to seven BMI categories",
    "spatial_model_role": "MRP predicts within-country ALR spatial deviations using covariates; national anchors and spatial deviations are separate components",
    "alternative_ssp_rule": "start from the common 2025 anchor, then apply scenario-specific national-composition and within-country covariate contrasts; do not force all scenarios to identical post-2025 GBD totals",
    "population_rule": "use scenario-specific population for scenario-specific post-stratification; additionally publish a common-population decomposition to isolate BMI-composition effects",
    "prediction_authorized": False,
    "next_gate": "reconstruct training features under the new definitions, refit the complete 22-feature MRP, then pass non-inferiority and extrapolation validation",
}
model_contract_path = CONTRACTS / "BMI_anchor_and_spatial_model_contract_v2.json"
if model_contract_path.exists():
    raise FileExistsError(f"Refusing to overwrite versioned artifact: {model_contract_path}")
model_contract_path.write_text(json.dumps(model_contract, ensure_ascii=False, indent=2), encoding="utf-8")

manifest = {
    "run_id": RUN.name,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS_FOR_MODEL_REFIT_WITH_REQUIRED_SENSITIVITY",
    "formal_bmi_prediction_authorized": False,
    "final_pm25_version": "pm25_mmr_trailing5y_v2",
    "final_raster_count": 435,
    "final_geometry_pass_count": 435,
    "input_collections": {"pm25_netcdf": 19, "future_ntl_tiff": 28, "ssp_urban_tiff_including_ancillary": 42},
    "warnings": [
        "PM2.5 is a single-GCM near-surface mass mixing ratio, not surface concentration.",
        "The NTL bridge slope reached the prespecified cap of 2.0.",
        "Future built fraction is a proxy based on the same SSP urban-land source as urban share.",
    ],
    "cleanup_audit": canonical(Path(_cfg.ROOT / ".codex_tmp" / "future_results_v1" / "15_code_logs_checksums" / "cleanup_full_covariate_reset_v1_20260905")),
}
manifest_path = CONTRACTS / "Full_covariate_execution_manifest_v2.json"
if manifest_path.exists():
    raise FileExistsError(f"Refusing to overwrite versioned artifact: {manifest_path}")
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

report = f"""# Full-covariate extension gate report v2

## Decision

**PASS_FOR_MODEL_REFIT_WITH_REQUIRED_SENSITIVITY.** The harmonised covariates may be used to reconstruct training features and refit the complete MRP. They do not yet authorise formal 2025-2050 BMI prediction.

## Final data products

- PM2.5: 19 historical/current 0.5-degree rasters (2006-2024) and 104 future rasters (4 SSPs x 2025-2050), using the trailing-five-year mass-mixing-ratio definition.
- Night-time light: 104 annual future rasters, log1p transformed and anchored to observed 2024 NTL.
- Urban share: 104 annual future rasters, using SSP urban-land change as a bridge to historical GHS-SMOD urban share.
- Built fraction: 104 annual future proxy rasters, anchoring SSP urban-land change to historical GHS built surface.
- Geometry: 435/435 final rasters passed EPSG:4326, 0.5-degree grid checks.

## Required cautions

1. The PM2.5 source is NorESM2-LM `mmrpm2p5` at the lowest hybrid level. It is a mass mixing ratio in microgram per kilogram of dry air, not a concentration in microgram per cubic metre. The old PM2.5 coefficient cannot be reused.
2. The NTL overlap bridge reached its prespecified slope cap (2.0). Refit and forecast sensitivity analyses must compare bridge slopes 1.0 and 2.0.
3. SSP urban land is a defensible proxy for future urban-share change, but not a direct projection of built surface. The complete model must be compared with a model excluding the built-fraction proxy.
4. A single aerosol model does not represent full CMIP6 structural uncertainty and must be disclosed.

## Anchor logic

- Through 2024, NCD-RisC country-year-age-sex seven-category BMI composition provides the national ALR anchor.
- From 2025, the SSP2 primary analysis uses GBD country-year-age-sex overweight and obesity forecasts as broad constraints; a national composition model allocates the remaining detail across all seven BMI categories.
- MRP estimates within-country ALR spatial deviations. National anchoring and spatial allocation are separate operations.
- Alternative SSPs must not be forced to the same post-2025 GBD trajectory. They start from a common 2025 anchor and then receive scenario-specific national and spatial changes.

## Next gate

Rebuild PM2.5, NTL, urban-share, and built-fraction values for every eligible training record under these frozen definitions; refit the 22-feature multinomial logistic-normal MRP; then run internal, country-held-out, source-held-out, and temporal validation. Formal BMI prediction remains blocked until that model passes the non-inferiority gate.
"""
report_path = REPORTS / "Full_covariate_extension_report_v2.md"
if report_path.exists():
    raise FileExistsError(f"Refusing to overwrite versioned artifact: {report_path}")
report_path.write_text(report, encoding="utf-8")

input_paths = sorted((RAW / "CMIP6_AEROSOL" / "NorESM2-LM").glob("*.nc"))
input_paths += sorted((RAW / "FUT_NTL" / "extracted").rglob("*.tif"))
input_paths += sorted((RAW / "ssp_urban" / "Gao_Pesaresi_SEDAC_1km" / "extracted").rglob("*.tif"))
input_paths += [
    HIST / "cache" / "auxiliary_mrp_0p5deg" / "covariates" / "global_covariates_2020_0p5deg.tif",
    HIST / "cache" / "auxiliary_mrp_0p5deg" / "covariates" / "global_covariates_2024_0p5deg.tif",
    GATE_C / "data" / "0p5deg" / "country_id_0p5deg.tif",
    GATE_C / "data" / "0p5deg" / "population" / "population_ssp2_2025_0p5deg.tif",
]
require(all(path.is_file() for path in input_paths), "One or more frozen inputs are missing")
input_hash_path = CONTRACTS / "Full_covariate_input_sha256_v2.csv"
write_csv(input_hash_path, hash_rows(input_paths, "frozen_input"), ["collection", "path", "size_bytes", "modified_utc", "sha256"])

excluded = {"Full_covariate_input_sha256_v2.csv", "Full_covariate_artifact_sha256_v2.csv"}
artifact_paths = sorted(
    path for path in RUN.rglob("*")
    if path.is_file() and path.name not in excluded and "work" not in path.parts
)
artifact_hash_path = CONTRACTS / "Full_covariate_artifact_sha256_v2.csv"
write_csv(artifact_hash_path, hash_rows(artifact_paths, "run_artifact"), ["collection", "path", "size_bytes", "modified_utc", "sha256"])

print("FULL_COVARIATE_STATUS=PASS_FOR_MODEL_REFIT_WITH_REQUIRED_SENSITIVITY")
print("FORMAL_BMI_PREDICTION_AUTHORIZED=NO")
print(f"INPUT_HASHES={len(input_paths)}")
print(f"ARTIFACT_HASHES={len(artifact_paths)}")
