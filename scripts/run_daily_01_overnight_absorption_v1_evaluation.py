"""一次性运行DAILY_01历史预测检验；失败后禁止策略回测。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.daily_01_overnight_absorption_v1 import (
    audit_and_load_inputs,
    build_absorption_features,
    load_config,
    sha256_file,
)
from research.daily_01_overnight_absorption_v1_evaluation import (
    build_forward_labels,
    evaluate_factor,
    load_evaluation_config,
    verify_parent_protocol,
)


IMPLEMENTATION_MANIFEST = ROOT / "config" / "daily_01_overnight_absorption_v1_evaluation_manifest.json"


def _verify_implementation_manifest() -> dict:
    manifest = json.loads(IMPLEMENTATION_MANIFEST.read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in manifest["frozen_files"].items():
        path = ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f"预测实现冻结文件偏离：{mismatches}")
    if manifest.get("historical_run_completed") is not False:
        raise RuntimeError("冻结清单显示历史运行已完成，禁止重复运行")
    return manifest


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _markdown(report: dict) -> str:
    gates = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in report["gate_checks"].items()
    )
    regression = report["incremental_hac_regression"]
    coefficient = regression["coefficients"][2]
    standard_error = regression["standard_errors"][2]
    t_value = regression["t_values"][2]
    return f"""# DAILY_01隔夜冲击—日内吸收预测检验

- 状态：`{report['status']}`
- 数据截止：`{report['data_cutoff']}`
- 证据：`HISTORICAL_DISCOVERY_ONLY`
- Alpha通过：`false`
- 策略回测授权：`false`

## 样本

- 有效候选日：{report['sample']['candidate_rows']}
- 成熟ENTER事件：{report['sample']['mature_enter_events']}
- D20有利组/不利组：{report['d20']['favorable_observations']} / {report['d20']['unfavorable_observations']}

## 冻结结果

- D20有利组减不利组均值：{report['d20']['top_minus_bottom_mean']:.4%}
- D5有利组减不利组均值：{report['d5']['top_minus_bottom_mean']:.4%}
- D20区块Bootstrap 95%区间：[{report['block_bootstrap_d20_top_minus_bottom']['lower_95']:.4%}, {report['block_bootstrap_d20_top_minus_bottom']['upper_95']:.4%}]
- 控制H001后的DAILY_01系数：{coefficient:.6f}；HAC标准误：{standard_error:.6f}；t值：{t_value:.3f}

## 门槛

{gates}

任一门槛失败即停止，不运行策略资产曲线，不修改窗口、阈值、方向或目标期限。本报告不生成当前观点、仓位、订单或券商动作。
"""


def main() -> int:
    evaluation_config = load_evaluation_config()
    parent_config = load_config()
    verify_parent_protocol(ROOT, evaluation_config)
    implementation_manifest = _verify_implementation_manifest()
    market, dividends, audit = audit_and_load_inputs(ROOT, parent_config)
    if audit["status"] != "PASS":
        raise RuntimeError("运行前数据闸门失败")

    features = build_absorption_features(market, dividends, parent_config)
    labeled = build_forward_labels(features, dividends, parent_config, evaluation_config)
    report = evaluate_factor(labeled, evaluation_config)
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["project_id"] = parent_config["protocol"]["project_id"]
    report["data_cutoff"] = parent_config["protocol"]["historical_evaluation_cutoff"]
    report["parent_protocol_manifest_sha256"] = evaluation_config["evaluation_protocol"]["parent_protocol_manifest_sha256"]
    report["implementation_manifest_sha256_before_run"] = sha256_file(IMPLEMENTATION_MANIFEST)

    artifacts = evaluation_config["artifacts"]
    feature_columns = [
        "date",
        "total_return",
        "overnight_contribution",
        "intraday_contribution",
        "overnight_contribution_5d",
        "intraday_contribution_5d",
        "realized_volatility_20d",
        "absorption_raw",
        "absorption_percentile",
        "entry_sign_eligible",
        "past_total_return_5d",
        "research_state_target",
        "signal_action",
        "holding_signal_days",
    ]
    feature_path = ROOT / artifacts["feature_output"]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_temp = feature_path.with_suffix(".parquet.tmp")
    features[feature_columns].to_parquet(feature_temp, index=False)
    feature_temp.replace(feature_path)

    candidates = labeled.loc[
        labeled["entry_sign_eligible"] & labeled["absorption_percentile"].notna(),
        [
            "date",
            "absorption_raw",
            "absorption_percentile",
            "past_total_return_5d",
            "signal_action",
            "target_d5_net_excess",
            "target_d20_net_excess",
        ],
    ].copy()
    candidate_path = ROOT / artifacts["candidate_rows"]
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_temp = candidate_path.with_suffix(".csv.tmp")
    candidates.to_csv(candidate_temp, index=False, encoding="utf-8-sig")
    candidate_temp.replace(candidate_path)

    report["artifacts"] = {
        "feature_output": artifacts["feature_output"],
        "feature_sha256": sha256_file(feature_path),
        "candidate_rows": artifacts["candidate_rows"],
        "candidate_rows_sha256": sha256_file(candidate_path),
    }
    _atomic_json(report, ROOT / artifacts["report_json"])
    markdown_path = ROOT / artifacts["report_markdown"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_temp = markdown_path.with_suffix(".md.tmp")
    markdown_temp.write_text(_markdown(report), encoding="utf-8")
    markdown_temp.replace(markdown_path)

    implementation_manifest["historical_run_completed"] = True
    implementation_manifest["historical_run_completed_at"] = report["generated_at"]
    implementation_manifest["historical_result_status"] = report["status"]
    implementation_manifest["report_file"] = artifacts["report_json"]
    implementation_manifest["report_sha256"] = sha256_file(ROOT / artifacts["report_json"])
    _atomic_json(implementation_manifest, IMPLEMENTATION_MANIFEST)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
