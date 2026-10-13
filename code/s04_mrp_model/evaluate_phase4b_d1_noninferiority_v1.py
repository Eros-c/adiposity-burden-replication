"""Evaluate frozen D1 noninferiority margins for cross-sectional validation."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(
    _cfg.RESULTS / "04_future_capable_mrp" / "phase4b_d1_reduced_demography_v1_20260831"
)
OLD_ROOT = Path(
    _cfg.RESULTS / "04_future_capable_mrp" / "phase4_future_capable_mrp_v1_20260828"
)
OUT = ROOT / "noninferiority_cross_sectional_v1"
D1_METRICS = (
    ROOT / "stabilized_evaluation_d1_v1" / "tables"
    / "all_eligible20_stabilized_metrics_v1.csv"
)
OLD_METRICS = (
    OLD_ROOT / "stabilized_evaluation_v1" / "tables"
    / "all_eligible20_stabilized_metrics_v1.csv"
)
OLD_SOURCE_METRICS = (
    ROOT / "stabilized_evaluation_source_comparator_v1" / "tables"
    / "all_eligible20_stabilized_metrics_v1.csv"
)
CONTRACT = ROOT / "contracts" / "Phase4b_D1_validation_contract_frozen_v1.json"

METRICS = {
    "weighted_r2": ("decline", 0.05),
    "weighted_w2_bmi_kg_m2": ("increase", 0.15),
    "obesity_weighted_mae_pp": ("increase", 0.30),
    "seven_category_weighted_mae_pp": ("increase", 0.30),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_primary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["minimum_effective_n"].eq(20)].copy()
    if frame["validation"].duplicated().any():
        raise RuntimeError(f"Duplicate primary validation rows in {path}")
    return frame.set_index("validation")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    d1 = load_primary(D1_METRICS)
    old = load_primary(OLD_METRICS)
    old_source = load_primary(OLD_SOURCE_METRICS)
    old = pd.concat([old, old_source], axis=0)

    expected = [
        "Internal random-row fivefold",
        "Country-held-out fivefold",
        "Source-family-held-out",
    ]
    if sorted(d1.index) != sorted(expected) or sorted(old.index) != sorted(expected):
        raise RuntimeError("Validation scopes do not match the frozen contract")

    rows: list[dict[str, object]] = []
    for scope in expected:
        for metric, (direction, margin) in METRICS.items():
            candidate = float(d1.loc[scope, metric])
            comparator = float(old.loc[scope, metric])
            delta = comparator - candidate if direction == "decline" else candidate - comparator
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "candidate_d1": candidate,
                    "comparator_19_feature": comparator,
                    "adverse_delta": delta,
                    "direction": direction,
                    "frozen_margin": margin,
                    "status": "PASS" if delta <= margin else "FAIL",
                }
            )
    checks = pd.DataFrame(rows)
    overall = "PASS" if checks["status"].eq("PASS").all() else "FAIL"

    checks_path = OUT / "D1_cross_sectional_noninferiority_checks_v1.csv"
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")
    summary = {
        "status": overall,
        "candidate": "D1 reduced-demography MRP with fold-specific soft support",
        "comparator": "Original 19-feature future-capable MRP",
        "minimum_effective_n": 20,
        "scopes": expected,
        "checks": len(checks),
        "checks_passed": int(checks["status"].eq("PASS").sum()),
        "checks_failed": int(checks["status"].eq("FAIL").sum()),
        "future_prediction_authorized": False,
        "authorization_note": "Temporal noninferiority and D1 2025 tail robustness remain required.",
    }
    summary_path = OUT / "D1_cross_sectional_noninferiority_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# D1 cross-sectional noninferiority report v1",
        "",
        f"Overall status: **{overall}**",
        "",
        "The candidate is evaluated at the frozen effective-n threshold of 20.",
        "An adverse delta is comparator minus candidate for R2, and candidate minus comparator for error metrics.",
        "",
        "| Scope | Metric | D1 | 19-feature | Adverse delta | Margin | Status |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['scope']} | {row['metric']} | {row['candidate_d1']:.6f} | "
            f"{row['comparator_19_feature']:.6f} | {row['adverse_delta']:.6f} | "
            f"{row['frozen_margin']:.2f} | {row['status']} |"
        )
    report_lines.extend(
        [
            "",
            "This report does not authorize future BMI production. Temporal backtesting and the D1 H1-versus-S1 2025 tail gate remain mandatory.",
        ]
    )
    report_path = OUT / "D1_cross_sectional_noninferiority_report_v1.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    artifacts = [
        Path(__file__), CONTRACT, D1_METRICS, OLD_METRICS,
        OLD_SOURCE_METRICS, checks_path, summary_path, report_path,
    ]
    pd.DataFrame(
        [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifacts
        ]
    ).to_csv(OUT / "D1_cross_sectional_noninferiority_sha256_v1.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
