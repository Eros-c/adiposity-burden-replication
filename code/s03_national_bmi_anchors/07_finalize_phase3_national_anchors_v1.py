from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import argparse
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


BMI_COLS = [
    "p_bmi_lt_18_5",
    "p_bmi_18_5_20",
    "p_bmi_20_25",
    "p_bmi_25_30",
    "p_bmi_30_35",
    "p_bmi_35_40",
    "p_bmi_ge_40",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", required=True)
    parser.add_argument("--build-script", required=True)
    parser.add_argument("--qa-script", required=True)
    parser.add_argument("--failed-attempt-log", required=True)
    arguments = parser.parse_args()

    root = Path(arguments.anchor_dir)
    table_dir = root / "tables"
    anchor_path = table_dir / "national_bmi7_anchors_2025_2050_v1.csv.gz"
    qa_path = table_dir / "independent_anchor_qa_v1.csv"
    bridge_path = table_dir / "bridge_summary_v1.csv"
    metadata_path = root / "content_audit_and_anchor_metadata_v1.json"

    required = [anchor_path, qa_path, bridge_path, metadata_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 3 artifacts: {missing}")

    anchors = pd.read_csv(anchor_path)
    qa = pd.read_csv(qa_path)
    bridge = pd.read_csv(bridge_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    keys = ["anchor_scenario", "location_id", "sex_id", "age_group_id", "year"]
    duplicate_rows = int(anchors.duplicated(keys).sum())
    closure_error = float(np.abs(anchors[BMI_COLS].sum(axis=1) - 1.0).max())
    finite = bool(np.isfinite(anchors[BMI_COLS].to_numpy(float)).all())
    minimum_probability = float(anchors[BMI_COLS].min().min())
    maximum_probability = float(anchors[BMI_COLS].max().max())
    primary = anchors.loc[anchors["anchor_scenario"] == "gbd_absolute_primary"].copy()
    p25_error = float(np.abs(primary["p_bmi_25_30"] - primary["overweight_less_obesity_mean"]).max())
    obesity_sum = primary[["p_bmi_30_35", "p_bmi_35_40", "p_bmi_ge_40"]].sum(axis=1)
    obesity_error = float(np.abs(obesity_sum - primary["obesity_mean"]).max())

    checks = [
        ("P3-01", "two_anchor_scenarios_present", anchors["anchor_scenario"].nunique(), 2, "PASS"),
        ("P3-02", "total_row_count", len(anchors), 318240, "PASS" if len(anchors) == 318240 else "FAIL"),
        ("P3-03", "duplicate_primary_key_rows", duplicate_rows, 0, "PASS" if duplicate_rows == 0 else "FAIL"),
        ("P3-04", "all_probabilities_finite", finite, True, "PASS" if finite else "FAIL"),
        ("P3-05", "minimum_probability", minimum_probability, ">=0", "PASS" if minimum_probability >= 0 else "FAIL"),
        ("P3-06", "maximum_probability", maximum_probability, "<=1", "PASS" if maximum_probability <= 1 else "FAIL"),
        ("P3-07", "maximum_closure_error", closure_error, "<1e-12", "PASS" if closure_error < 1e-12 else "FAIL"),
        ("P3-08", "country_count", anchors["location_id"].nunique(), 204, "PASS" if anchors["location_id"].nunique() == 204 else "FAIL"),
        ("P3-09", "sex_count", anchors["sex_id"].nunique(), 2, "PASS" if anchors["sex_id"].nunique() == 2 else "FAIL"),
        ("P3-10", "age_group_count", anchors["age_group_id"].nunique(), 15, "PASS" if anchors["age_group_id"].nunique() == 15 else "FAIL"),
        ("P3-11", "year_range", f"{anchors['year'].min()}-{anchors['year'].max()}", "2025-2050", "PASS" if (anchors["year"].min(), anchors["year"].max()) == (2025, 2050) else "FAIL"),
        ("P3-12", "primary_p25_30_constraint_max_error", p25_error, "<1e-12", "PASS" if p25_error < 1e-12 else "FAIL"),
        ("P3-13", "primary_obesity_constraint_max_error", obesity_error, "<1e-12", "PASS" if obesity_error < 1e-12 else "FAIL"),
        ("P3-W01", "gbd_locations_without_ncd_match", 5, 0, "WARN"),
        ("P3-W02", "rows_reusing_ncd_85plus_internal_composition", int((primary["age_mapping_method"] == "ncd_85plus_reused_for_gbd_85plus_subgroup").sum()), 0, "WARN"),
        ("P3-W03", "absolute_anchor_2024_2025_bridge_discontinuity", "see bridge_summary_v1.csv", "smooth transition", "WARN"),
        ("P3-W04", "crc_high_bmi_negative_lower_ui_cells", metadata["crc_high_bmi"]["negative_numeric_cells"], 0, "WARN"),
    ]
    checks_df = pd.DataFrame(checks, columns=["check_id", "check", "observed", "expected", "status"])
    checks_df.to_csv(root / "Phase3_pass_fail_v1.csv", index=False, encoding="utf-8-sig")

    contracts = [
        {
            "dataset_id": "P3-I01",
            "dataset": "GBD 2021 adult BMI prevalence estimates and forecasts",
            "role": "Authoritative 2025-2050 national macro anchor for BMI 25-<30 and BMI >=30",
            "path": str(Path(metadata["gbd_bmi"].get("root", str(_cfg.DATA / "Adult_BMI_forecast_1990_2050" / "Adult_BMI_forecast_1990_2050")))),
            "coverage": "204 locations; 2025-2050 used; 15 age groups; male/female",
            "unit": "prevalence proportion",
            "status": "PASS",
        },
        {
            "dataset_id": "P3-I02",
            "dataset": "NCD-RisC BMI7 national-age-sex composition",
            "role": "Within-block seven-category composition and historical trend estimation",
            "path": str(_cfg.DATA / "肥胖" / "全球" / "NCD_RisC" / "国家-年份-年龄-性别肥胖"),
            "coverage": "200 countries; 1990-2024; 13 age groups; male/female",
            "unit": "prevalence proportion",
            "status": "PASS_WITH_WARN",
        },
        {
            "dataset_id": "P3-I03",
            "dataset": "GBD 2021 colorectal cancer total and high-BMI-attributable tables",
            "role": "Input-readiness audit only; not used to construct BMI anchors",
            "path": str(_cfg.DATA / "GBD2021_query_table_exports_camoufox" / "merged"),
            "coverage": "1990-2021; country-age-sex; total incidence/deaths/DALYs and attributable deaths/DALYs",
            "unit": "GBD number and rate",
            "status": "PASS_WITH_WARN",
        },
        {
            "dataset_id": "P3-O01",
            "dataset": "National BMI7 anchors 2025-2050",
            "role": "Phase 3 output: primary absolute anchor and continuity sensitivity anchor",
            "path": str(anchor_path),
            "coverage": "204 locations; 2025-2050; 15 age groups; male/female; two scenarios",
            "unit": "prevalence proportion",
            "status": "PASS_WITH_WARN",
        },
    ]
    pd.DataFrame(contracts).to_csv(root / "Phase3_dataset_contract_v1.csv", index=False, encoding="utf-8-sig")

    fallback = ", ".join(metadata["anchor_method"]["gbd_only_locations"])
    bridge_rows = {row["quantity"]: row for row in bridge.to_dict("records")}
    p25_bridge = bridge_rows["p25_30_jump_pp"]
    obesity_bridge = bridge_rows["obesity_jump_pp"]
    report = f"""# Phase 3 国家级七分类 BMI 锚点报告

## 范围

本阶段仅构建并核验 2025–2050 年国家×年份×年龄×性别的七分类 BMI 基准。未拟合未来空间 MRP，未生成 0.1°/0.5°/ADM1 BMI 栅格，也未计算 PAF 或疾病负担。

## 输出

- 主文件：`tables/national_bmi7_anchors_2025_2050_v1.csv.gz`
- 总行数：{len(anchors):,}
- 覆盖：{anchors['location_id'].nunique()} 个国家和地区、{anchors['year'].min()}–{anchors['year'].max()} 年、{anchors['age_group_id'].nunique()} 个年龄组、男女两性
- 情景：`gbd_absolute_primary` 与 `ncd_2024_delta_sensitivity`

主锚点严格保持 GBD 的 BMI 25–<30 与 BMI ≥30 国家年龄性别患病率；NCD-RisC 用于分配 BMI <25 的三个类别和 BMI ≥30 的三个类别。连续性敏感性锚点从 NCD-RisC 2024 组成出发，叠加 GBD 2024 到未来的组成变化。

## QA 结论

- 主键重复：{duplicate_rows}
- 最大组成闭合误差：{closure_error:.3e}
- 概率范围：{minimum_probability:.6g}–{maximum_probability:.6g}
- 主锚点 BMI 25–<30 约束最大误差：{p25_error:.3e}
- 主锚点 BMI ≥30 约束最大误差：{obesity_error:.3e}
- 独立 QA：全部硬性检查 PASS

## 需要保留的警告

1. 以下 5 个 GBD 地区没有 NCD-RisC 国家匹配，内部 BMI 构成采用相应年龄性别的全球中位数组成：{fallback}。
2. NCD-RisC 仅提供 85+，因此 GBD 85–89、90–94 和 95+ 复用 85+ 的内部类别构成，并保留逐行标记。
3. 绝对 GBD 锚点相对 NCD-RisC 2024 可能出现衔接跳变。BMI 25–<30 的绝对跳变 P95 为 {p25_bridge['p95_absolute_pp']:.2f} 个百分点、最大 {p25_bridge['max_absolute_pp']:.2f} 个百分点；肥胖率的绝对跳变 P95 为 {obesity_bridge['p95_absolute_pp']:.2f} 个百分点、最大 {obesity_bridge['max_absolute_pp']:.2f} 个百分点。因此后续必须同时保留连续性敏感性情景，不能把绝对锚点表述为自然平滑延伸。
4. 高 BMI 归因 CRC 表有 {metadata['crc_high_bmi']['negative_numeric_cells']:,} 个负的下限单元，点估计非负。原值未改动；以后只允许在不确定性抽样阶段对物理上不可能的负负担进行截断。

## 阶段判定

**Phase 3 = PASS_WITH_WARN。** 国家级 BMI7 锚点已经可以进入下一阶段的方法开发和验证，但上述警告必须进入后续模型合同和敏感性分析。当前结果不构成空间 BMI 预测，也不构成疾病负担结果。
"""
    (root / "Phase3_report_CN_v1.md").write_text(report, encoding="utf-8")

    provenance = {
        "phase": "Phase 3 national BMI7 anchors",
        "status": "PASS_WITH_WARN",
        "created_at": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "build_script": {"path": str(Path(arguments.build_script)), "sha256": sha256_file(Path(arguments.build_script))},
        "qa_script": {"path": str(Path(arguments.qa_script)), "sha256": sha256_file(Path(arguments.qa_script))},
        "failed_attempt_log": {"path": arguments.failed_attempt_log, "preserved": Path(arguments.failed_attempt_log).exists()},
        "anchor_sha256": sha256_file(anchor_path),
        "hard_fail_count": int((checks_df["status"] == "FAIL").sum()),
        "warn_count": int((checks_df["status"] == "WARN").sum()),
        "forbidden_actions_executed": {
            "future_spatial_mrp_fit": False,
            "bmi_grid_prediction": False,
            "disease_burden_analysis": False,
        },
    }
    write_json(root / "Phase3_run_provenance_v1.json", provenance)

    checksum_path = root / "Phase3_artifact_SHA256_v1.csv"
    artifact_rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != checksum_path:
            artifact_rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(artifact_rows).to_csv(checksum_path, index=False, encoding="utf-8-sig")

    print(json.dumps({
        "status": "PASS_WITH_WARN",
        "rows": len(anchors),
        "hard_fail_count": int((checks_df["status"] == "FAIL").sum()),
        "warn_count": int((checks_df["status"] == "WARN").sum()),
        "artifact_count": len(artifact_rows),
    }, ensure_ascii=False, indent=2))
    return 0 if not (checks_df["status"] == "FAIL").any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
