"""按冻结规则构建 V2 G1 事件预测准入账本，不拟合模型。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    canonical_sha256,
    file_evidence,
    normalize_project_relative_path,
)
from research.stress_transmission_hazard_v2_g1_event_prediction_admission_v1 import (
    G1AdmissionArtifacts,
    G1EventPredictionAdmissionError,
    build_g1_admission_artifacts,
)
from scripts.build_510300_stress_transmission_hazard_v2_four_state_ledger_v1 import (
    _write_parquet_new,
    _write_text_new,
)
from scripts.build_510300_stress_transmission_hazard_v2_mft_features_v1 import (
    frame_semantic_sha256,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
)


TABLE_OUTPUTS = {
    "b1_feature_panel": "b1_feature_panel",
    "sample_eligibility_ledger": "sample_eligibility_ledger",
    "event_eligibility_ledger": "event_eligibility_ledger",
}


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        name: _project_path(str(relative))
        for name, relative in config["outputs"].items()
        if name != "curated_root"
    }


def load_and_build(config: Mapping[str, Any]) -> G1AdmissionArtifacts:
    """读取冻结输入；BAD10 仅按配置 usecols 解析，不读未来路径幅度。"""

    inputs = config["inputs"]
    etf_contract = inputs["etf_unadjusted_daily"]
    etf = pd.read_parquet(
        _project_path(str(etf_contract["path"])),
        columns=[
            str(etf_contract["date_column"]),
            "symbol",
            str(etf_contract["close_column"]),
        ],
    )
    dividend_contract = inputs["etf_cash_dividends"]
    dividends = pd.read_csv(
        _project_path(str(dividend_contract["path"])),
        usecols=[
            "symbol",
            "record_date",
            "ex_date",
            "payment_date",
            "cash_dividend_per_share",
        ],
    )
    mft = pd.read_parquet(
        _project_path(str(inputs["mft_feature_panel"]["path"])),
        columns=["date", "F", "M", "T", "B2_feature_state", "B3_feature_state"],
    )
    print("冻结 manifest 已验证；现在首次读取 BAD10 最小必要标签列。", flush=True)
    events = pd.read_csv(
        _project_path(str(inputs["bad10_events"]["path"])),
        usecols=[str(column) for column in inputs["bad10_events"]["read_columns"]],
    )
    samples = pd.read_csv(
        _project_path(str(inputs["bad10_samples"]["path"])),
        usecols=[str(column) for column in inputs["bad10_samples"]["read_columns"]],
    )
    return build_g1_admission_artifacts(
        etf_daily=etf,
        dividends=dividends,
        mft=mft,
        samples=samples,
        events=events,
        config=config,
    )


def _build_report(artifacts: G1AdmissionArtifacts) -> str:
    metrics = artifacts.metrics
    gate = artifacts.gate_result
    return "\n".join(
        [
            "# 510300 压力传导危险率 V2：G1 事件预测准入 V1",
            "",
            "## 共同样本计数",
            "",
            f"- 冻结 BAD10 独立事件总数：{metrics['total_independent_event_count']:,}",
            f"- 冻结 BAD10 正样本原点总数：{metrics['total_positive_origin_count']:,}",
            f"- 冻结非事件风险日总数：{metrics['total_non_event_risk_day_count']:,}",
            f"- B1 可见样本日：{metrics['b1_eligible_sample_day_count']:,}",
            f"- B2/B1 共同样本日：{metrics['b2_common_sample_day_count']:,}",
            f"- B2 可识别独立事件：{metrics['b2_identifiable_event_count']:,}",
            f"- B2 可见正样本原点：{metrics['b2_eligible_positive_origin_count']:,}",
            f"- B2 可见非事件风险日：{metrics['b2_eligible_non_event_risk_day_count']:,}",
            f"- B2 可见日期：{metrics['b2_first_eligible_origin_date']} 至 {metrics['b2_last_eligible_origin_date']}",
            f"- B3/B2 共同样本日：{metrics['b3_common_sample_day_count']:,}",
            f"- B3 可识别独立事件：{metrics['b3_identifiable_event_count']:,}",
            f"- B3 可见正样本原点：{metrics['b3_eligible_positive_origin_count']:,}",
            f"- B3 可见非事件风险日：{metrics['b3_eligible_non_event_risk_day_count']:,}",
            f"- B3 可见日期：{metrics['b3_first_eligible_origin_date']} 至 {metrics['b3_last_eligible_origin_date']}",
            "",
            "## G1 裁决",
            "",
            f"- `G1_DATA_AND_EVENTS`：`{gate['G1_DATA_AND_EVENTS']}`",
            f"- B2 机制发现计数门：{'PASS' if gate['mechanism_discovery_prerequisite_passed'] else 'FAIL'}",
            f"- B3 完整三系数模型计数门：{'PASS' if gate['full_three_coefficient_model_prerequisite_passed'] else 'FAIL'}",
            f"- 下一允许步骤：`{gate['next_allowed_step']}`",
            "",
            "## 权重与读取边界",
            "",
            "- 每个可识别事件内的可见正样本权重重新归一为 1；每个可见非事件风险日权重为 1。",
            "- B2/B1 与 B3/B2 分别使用严格共同可见样本；`NO_VIEW` 行保留且模型权重为空。",
            "- 本执行只读取 BAD10 最小必要标签列，没有读取最小路径收益、策略收益或组合表现。",
            "- 本执行没有拟合 B0/B1/B2/B3，没有生成概率、阈值、警报或任何预测评价指标。",
            "- 组合、Paper/Shadow、经纪商、仓位和订单权限继续关闭，`POSITION_IMPACT=0`。",
            "",
        ]
    )


def _validate_parent_aggregate_counts(
    artifacts: G1AdmissionArtifacts, config: Mapping[str, Any]
) -> None:
    expected = config["parent_chain"]["bad10_ledger_receipt"]
    metrics = artifacts.metrics
    checks = {
        "独立事件": (
            metrics["total_independent_event_count"],
            expected["expected_independent_events"],
        ),
        "正样本原点": (
            metrics["total_positive_origin_count"],
            expected["expected_positive_origins"],
        ),
        "非事件风险日": (
            metrics["total_non_event_risk_day_count"],
            expected["expected_non_event_risk_days"],
        ),
    }
    for label, (actual, frozen) in checks.items():
        if int(actual) != int(frozen):
            raise EvidenceContractError(
                f"BAD10 {label}与父收据不一致：actual={actual}, frozen={frozen}"
            )


def build(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    verify_frozen_manifest(config_path)
    config = load_config(config_path)
    outputs = _output_paths(config)
    owned = [
        *TABLE_OUTPUTS.keys(),
        "report",
        "build_receipt",
        "clean_replay_receipt",
        "clean_replay_failure_receipt",
        "status",
    ]
    existing = [str(outputs[name]) for name in owned if outputs[name].exists()]
    if existing:
        raise EvidenceContractError(f"G1 准入输出已存在，禁止覆盖：{existing}")

    artifacts = load_and_build(config)
    _validate_parent_aggregate_counts(artifacts, config)
    print("G1 共同样本与事件可识别性断言通过，正在原子写入新输出。", flush=True)
    table_evidence: dict[str, dict[str, Any]] = {}
    for output_name, attribute_name in TABLE_OUTPUTS.items():
        frame = getattr(artifacts, attribute_name)
        _write_parquet_new(frame, outputs[output_name])
        persisted = pd.read_parquet(outputs[output_name])
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            persisted.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
            check_categorical=False,
        )
        table_evidence[output_name] = {
            **file_evidence(outputs[output_name], project_root=ROOT),
            "row_count": int(len(persisted)),
            "persisted_semantic_sha256": frame_semantic_sha256(persisted),
        }

    report = _build_report(artifacts)
    _write_text_new(report, outputs["report"])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_EVENT_PREDICTION_ADMISSION_BUILD_V1",
        "created_at": now,
        "status": "PASS_G1_COMMON_SAMPLE_AND_IDENTIFIABILITY_COUNTS_PENDING_CLEAN_REPLAY",
        "frozen_manifest": file_evidence(
            _project_path(config["freeze_contract"]["manifest_output"]),
            project_root=ROOT,
        ),
        "inputs": {
            name: file_evidence(_project_path(contract["path"]), project_root=ROOT)
            for name, contract in config["inputs"].items()
        },
        "outputs": {
            **table_evidence,
            "report": file_evidence(outputs["report"], project_root=ROOT),
        },
        "metrics": artifacts.metrics,
        "gate_result": artifacts.gate_result,
        "actual_label_artifact_read": True,
        "actual_label_columns_read": list(config["inputs"]["bad10_samples"]["read_columns"]),
        "actual_future_path_value_read": False,
        "minimum_path_return_read": False,
        "model_trained": False,
        "probability_generated": False,
        "prediction_metric_generated": False,
        "portfolio_or_performance_artifact_read": False,
        "portfolio_generated": False,
        "paper_or_shadow_authorized": False,
        "broker_connection_authorized": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(outputs["build_receipt"], receipt)
    print(
        "G1 准入构建完成；"
        f"B2_events={artifacts.metrics['b2_identifiable_event_count']}，"
        f"B3_events={artifacts.metrics['b3_identifiable_event_count']}，"
        f"state={artifacts.gate_result['G1_DATA_AND_EVENTS']}；"
        "未拟合模型或读取绩效。"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        build(args.config)
        return 0
    except (
        EvidenceContractError,
        G1EventPredictionAdmissionError,
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"G1 事件预测准入构建失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
