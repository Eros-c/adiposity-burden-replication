"""Build the dose-response relative-risk crosswalk for 33 adiposity-associated causes."""

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

PROJECT = Path(_cfg.RESULTS)
ROOT = PROJECT / "11_all_disease_extension" / "02_rr_crosswalk" / "rr_crosswalk_v1_20261005"
CONTRACTS = ROOT / "contracts"
DATA = ROOT / "data"
QA = ROOT / "qa"
REPORTS = ROOT / "reports"
SCRIPTS = ROOT / "scripts"

XLSX = (
    PROJECT
    / "11_all_disease_extension"
    / "01_harmonize"
    / "harmonized_gbd_v1_20261005"
    / "contracts"
    / "BMI_disease_RR_source_v1.xlsx"
)
DISEASE_LIST = (
    PROJECT
    / "11_all_disease_extension"
    / "01_harmonize"
    / "harmonized_gbd_v1_20261005"
    / "contracts"
    / "diseases_v1.csv"
)

AGES = (
    "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95+",
)
BMI_CATEGORIES = (
    ("bmi_lt_18_5", "<18.5", 17.0),
    ("bmi_18_5_20", "18.5-<20", 19.25),
    ("bmi_20_25", "20-<25", 22.5),
    ("bmi_25_30", "25-<30", 27.5),
    ("bmi_30_35", "30-<35", 32.5),
    ("bmi_35_40", "35-<40", 37.5),
    ("bmi_ge_40", ">=40", 42.5),
)
DEFAULT_REFERENCE = 22.5

FEMALE_ONLY = {429, 435, 465}
BMI_GE25_REFERENCE = {418, 419, 420}
SUBTYPE_CAUSES = {418, 419, 420, 495, 496, 845, 846, 847, 848, 943}
CAUSE_GROUPS = {
    441: "Cancer", 417: "Cancer", 418: "Cancer", 419: "Cancer", 420: "Cancer",
    453: "Cancer", 456: "Cancer", 429: "Cancer", 435: "Cancer", 465: "Cancer",
    471: "Cancer", 480: "Cancer", 485: "Cancer", 486: "Cancer", 487: "Cancer",
    845: "Cancer", 846: "Cancer", 847: "Cancer", 848: "Cancer", 943: "Cancer",
    493: "Cardiovascular", 494: "Cardiovascular", 495: "Cardiovascular",
    496: "Cardiovascular", 498: "Cardiovascular", 500: "Cardiovascular",
    502: "Cardiovascular",
    976: "Metabolic/renal", 998: "Metabolic/renal", 591: "Metabolic/renal",
    515: "Other", 534: "Other", 543: "Other",
}

SPECIAL = {
    441: {
        "kind": "sex_specific",
        "Male": (1.08, 1.04, 1.11),
        "Female": (1.05, 1.02, 1.08),
        "flag": "direct_sex_specific",
        "source": "WCRF/AICR, Body fatness and weight gain and the risk of cancer (frozen CRC crosswalk values)",
        "note": "Identical to the frozen CRC crosswalk; reproduces published CRC results.",
    },
    453: {
        "kind": "pooled",
        "value": (1.27, 1.23, 1.32),
        "flag": "conditional_subtype_aggregate",
        "source": "Watts et al., 2026, Nature Metabolism (gallbladder cancer value)",
        "note": "Source reports gallbladder 1.27 (1.23-1.32), intrahepatic 1.32 (1.21-1.45), "
        "extrahepatic 1.13 (1.03-1.23) separately; the gallbladder value is applied to the "
        "GBD aggregate cause because GBD has no subtype split.",
    },
    429: {
        "kind": "menopause_split",
        "pre": (0.95, 0.94, 0.97),
        "post": (1.14, 1.12, 1.17),
        "pre_ages": ("25-29", "30-34", "35-39", "40-44", "45-49"),
        "flag": "conditional_menopause",
        "source": "Watts et al., 2026, Nature Metabolism; Amadou et al., 2013, Obesity Reviews",
        "note": "Premenopausal RR 0.95 (0.94-0.97) applied to 25-49y; postmenopausal RR 1.14 "
        "(1.12-1.17) applied to 50y and older; female only. The premenopausal inverse "
        "association is uncertain in Asian women (see source note).",
    },
    498: {
        "kind": "age_specific",
        "bands": {
            "35-44": (2.15, 0.80, 5.78),
            "45-54": (2.02, 0.97, 4.21),
            "55-64": (1.90, 1.17, 3.07),
            "65-74": (1.81, 1.45, 2.26),
            "75-84": (1.63, 1.53, 1.74),
            "85+": (1.45, 1.05, 2.01),
        },
        "band_map": {
            "25-29": "35-44", "30-34": "35-44", "35-39": "35-44", "40-44": "35-44",
            "45-49": "45-54", "50-54": "45-54", "55-59": "55-64", "60-64": "55-64",
            "65-69": "65-74", "70-74": "65-74", "75-79": "75-84", "80-84": "75-84",
            "85-89": "85+", "90-94": "85+", "95+": "85+",
        },
        "flag": "conditional_age_specific",
        "source": "Singh et al., 2013, PLOS ONE (PSC/APCSC pooled analysis)",
        "note": "Six decade-banded RRs; 25-34y carries the 35-44y estimate as nearest-band "
        "extension. CIs for 35-44y and 45-54y cross 1 in the source.",
    },
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def numeric(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def expand_categories(rr: float, lower: float, upper: float, reference: float) -> list[tuple]:
    rows = []
    for key, label, rep in BMI_CATEGORIES:
        exponent = max(rep - reference, 0.0) / 5.0
        rows.append((key, label, rep, rr**exponent, lower**exponent, upper**exponent))
    return rows

def main() -> None:
    diseases = pd.read_csv(DISEASE_LIST, encoding="utf-8-sig")
    diseases = diseases[diseases["cause_id"] != 996].copy()
    diseases["cause_id"] = diseases["cause_id"].astype(int)
    book = pd.ExcelFile(XLSX).parse("GBD完整主表")
    book = book.set_index("GBD_Cause_ID")

    records: list[dict] = []
    qa_rows: list[dict] = []
    for _, disease in diseases.iterrows():
        cause_id = int(disease["cause_id"])
        if cause_id not in book.index:
            raise RuntimeError(f"Cause {cause_id} missing from RR workbook")
        entry = book.loc[cause_id]
        sexes = ("Female",) if cause_id in FEMALE_ONLY else ("Female", "Male")
        reference = 25.0 if cause_id in BMI_GE25_REFERENCE else DEFAULT_REFERENCE

        if cause_id in SPECIAL:
            spec = SPECIAL[cause_id]
            kind = spec["kind"]
            if kind == "sex_specific":
                for sex in sexes:
                    rr, lo, hi = spec[sex]
                    for age in AGES:
                        records.append(
                            dict(cause_id=cause_id, sex=sex, age_group=age,
                                 rr_per_5=rr, rr_per_5_lower=lo, rr_per_5_upper=hi,
                                 reference_bmi=DEFAULT_REFERENCE)
                        )
                rr_desc = "; ".join(f"{s} {spec[s][0]} ({spec[s][1]}-{spec[s][2]})" for s in sexes)
            elif kind == "pooled":
                rr, lo, hi = spec["value"]
                for sex in sexes:
                    for age in AGES:
                        records.append(
                            dict(cause_id=cause_id, sex=sex, age_group=age,
                                 rr_per_5=rr, rr_per_5_lower=lo, rr_per_5_upper=hi,
                                 reference_bmi=reference)
                        )
                rr_desc = f"{rr} ({lo}-{hi})"
            elif kind == "menopause_split":
                for age in AGES:
                    rr, lo, hi = spec["pre"] if age in spec["pre_ages"] else spec["post"]
                    records.append(
                        dict(cause_id=cause_id, sex="Female", age_group=age,
                             rr_per_5=rr, rr_per_5_lower=lo, rr_per_5_upper=hi,
                             reference_bmi=reference)
                    )
                rr_desc = (f"pre(25-49y) {spec['pre'][0]} ({spec['pre'][1]}-{spec['pre'][2]}); "
                           f"post(50+y) {spec['post'][0]} ({spec['post'][1]}-{spec['post'][2]})")
            elif kind == "age_specific":
                for age in AGES:
                    band = spec["band_map"][age]
                    rr, lo, hi = spec["bands"][band]
                    for sex in sexes:
                        records.append(
                            dict(cause_id=cause_id, sex=sex, age_group=age,
                                 rr_per_5=rr, rr_per_5_lower=lo, rr_per_5_upper=hi,
                                 reference_bmi=reference)
                        )
                rr_desc = "; ".join(
                    f"{band} {v[0]} ({v[1]}-{v[2]})" for band, v in spec["bands"].items()
                )
            else:
                raise RuntimeError(f"Unknown special kind {kind}")
            flag = spec["flag"]
            source = spec["source"]
            note = spec["note"]
        else:
            rr = numeric(entry["RR_HR"])
            lo = numeric(entry["95CI下限"])
            hi = numeric(entry["95CI上限"])
            if rr is None or lo is None or hi is None:
                raise RuntimeError(f"Cause {cause_id}: unparseable single RR in workbook")
            for sex in sexes:
                for age in AGES:
                    records.append(
                        dict(cause_id=cause_id, sex=sex, age_group=age,
                             rr_per_5=rr, rr_per_5_lower=lo, rr_per_5_upper=hi,
                             reference_bmi=reference)
                    )
            rr_desc = f"{rr} ({lo}-{hi})"
            flag = "conditional_bmi_ge25" if cause_id in BMI_GE25_REFERENCE else "direct"
            source = str(entry["来源"])
            note = "" if flag == "direct" else "RR reported for BMI >= 25 range only; reference set to 25.0 kg/m2."

        qa_rows.append(
            dict(
                cause_id=cause_id,
                cause_name_en=disease["cause_name_en"],
                cause_group=CAUSE_GROUPS[cause_id],
                analysis_tier="subtype" if cause_id in SUBTYPE_CAUSES else "headline",
                sex_scope="+".join(sexes),
                reference_bmi=reference,
                rr_per_5=rr_desc,
                condition_flag=flag,
                evidence_status=str(entry["证据状态"]),
                source=source,
                note=note,
            )
        )

    crosswalk = pd.DataFrame(records)
    crosswalk["cause_name_en"] = crosswalk["cause_id"].map(
        diseases.set_index("cause_id")["cause_name_en"]
    )
    crosswalk = crosswalk[
        ["cause_id", "cause_name_en", "sex", "age_group",
         "rr_per_5", "rr_per_5_lower", "rr_per_5_upper", "reference_bmi"]
    ]
    out_main = DATA / "RR_crosswalk_all_causes_v1.csv"
    crosswalk.to_csv(out_main, index=False, encoding="utf-8-sig")

    expanded = []
    for row in crosswalk.itertuples(index=False):
        for key, label, rep, rr_k, lo_k, hi_k in expand_categories(
            row.rr_per_5, row.rr_per_5_lower, row.rr_per_5_upper, row.reference_bmi
        ):
            expanded.append(
                dict(
                    cause_id=row.cause_id, cause_name_en=row.cause_name_en, sex=row.sex,
                    age_group=row.age_group, bmi_category=key, bmi_category_label=label,
                    representative_bmi=rep, RR=rr_k, RR_lower=lo_k, RR_upper=hi_k,
                    reference_bmi=row.reference_bmi,
                )
            )
    expanded = pd.DataFrame(expanded)
    out_expanded = DATA / "RR_crosswalk_by_bmi_category_v1.csv"
    expanded.to_csv(out_expanded, index=False, encoding="utf-8-sig")

    qa_table = pd.DataFrame(qa_rows)
    qa_csv = QA / "RR_crosswalk_decisions_v1.csv"
    qa_table.to_csv(qa_csv, index=False, encoding="utf-8-sig")

    # Integrity gates.
    problems = []
    if (crosswalk["rr_per_5"] <= 0).any():
        problems.append("non-positive RR found")
    if (crosswalk["rr_per_5_lower"] > crosswalk["rr_per_5_upper"]).any():
        problems.append("CI lower > upper")
    expected = int(
        sum(len(AGES) * (1 if cid in FEMALE_ONLY else 2) for cid in diseases["cause_id"])
    )
    if len(crosswalk) != expected:
        problems.append(f"row count {len(crosswalk)} != expected {expected}")
    if (expanded["RR"] < 0).any() or (expanded["RR"] == 0).any():
        problems.append("expanded RR <= 0")
    key = ["cause_id", "sex", "age_group"]
    if crosswalk.duplicated(subset=key).any():
        problems.append("duplicate cause-sex-age rows")

    crc_check = expanded[
        (expanded["cause_id"] == 441)
        & (expanded["sex"] == "Male")
        & (expanded["bmi_category"] == "bmi_25_30")
    ]["RR"].unique()
    if len(crc_check) != 1 or abs(crc_check[0] - 1.08) > 1e-12:
        problems.append(f"CRC frozen value mismatch: {crc_check}")

    status = "PASS" if not problems else "FAIL"
    summary = {
        "run": "rr_crosswalk_v1_20261005",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "causes": int(crosswalk["cause_id"].nunique()),
        "rows_cause_sex_age": int(len(crosswalk)),
        "rows_expanded": int(len(expanded)),
        "default_reference_bmi": DEFAULT_REFERENCE,
        "bmi_ge25_reference_causes": sorted(BMI_GE25_REFERENCE),
        "female_only_causes": sorted(FEMALE_ONLY),
        "subtype_tier_causes": sorted(SUBTYPE_CAUSES),
        "problems": problems,
        "status": status,
    }
    write_json(QA / "RR_crosswalk_QA_v1.json", summary)

    lines = [
        "# RR crosswalk decisions (33 causes)",
        "",
        "Rule: PAF reference 22.5 kg/m2 (liver subtypes 25.0); RR = 1 at or below the reference;",
        "log-linear per +5 kg/m2 above it; seven-category representatives 17.0/19.25/22.5/27.5/32.5/37.5/42.5.",
        "",
        "| cause_id | cause | group | tier | sex | ref | RR per +5 kg/m2 | flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in qa_table.itertuples(index=False):
        lines.append(
            f"| {row.cause_id} | {row.cause_name_en} | {row.cause_group} | {row.analysis_tier} "
            f"| {row.sex_scope} | {row.reference_bmi} | {row.rr_per_5} | {row.condition_flag} |"
        )
    lines += [
        "",
        "## Special-case notes",
        "",
    ]
    for row in qa_table.itertuples(index=False):
        if row.note:
            lines.append(f"- **{row.cause_id} {row.cause_name_en}**: {row.note}")
    lines += ["", f"QA status: {status}" + ("" if not problems else " — " + "; ".join(problems))]
    (REPORTS / "RR_crosswalk_report_v1.md").write_text("\n".join(lines), encoding="utf-8")

    (SCRIPTS / Path(__file__).name).write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    outputs = [out_main, out_expanded, qa_csv, QA / "RR_crosswalk_QA_v1.json",
               REPORTS / "RR_crosswalk_report_v1.md", SCRIPTS / Path(__file__).name]
    pd.DataFrame(
        {"path": [str(p) for p in outputs], "sha256": [sha256(p) for p in outputs]}
    ).to_csv(ROOT / "RR_crosswalk_formal_manifest_v1.csv", index=False, encoding="utf-8-sig")

    if status != "PASS":
        raise RuntimeError("RR crosswalk QA failed: " + "; ".join(problems))
    (ROOT / "COMPLETE.flag").write_text(
        f"rr_crosswalk_v1_20261005 completed {summary['completed_at_utc']}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
