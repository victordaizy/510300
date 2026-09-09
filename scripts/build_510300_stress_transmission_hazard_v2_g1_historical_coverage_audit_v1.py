"""按冻结协议构建 V2 G1 历史覆盖审计，不重算 G1。"""

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
from research.stress_transmission_hazard_v2_g1_historical_coverage_audit_v1 import (
    HistoricalCoverageAuditArtifacts,
    HistoricalCoverageAuditError,
    build_historical_coverage_audit,
)
from scripts.build_510300_stress_transmission_hazard_v2_four_state_ledger_v1 import (
    _write_parquet_new,
    _write_text_new,
)
from scripts.build_510300_stress_transmission_hazard_v2_mft_features_v1 import (
    frame_semantic_sha256,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_historical_coverage_audit_v1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
    verify_registered_inputs,
)


TABLE_OUTPUTS = {
    "prehistory_coverage_comparison": "prehistory_coverage_comparison",
    "event_gap_ledger": "event_gap_ledger",
    "member_window_gap_ledger": "member_window_gap_ledger",
}


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        name: _project_path(str(path)) for name, path in config["outputs"].items()
    }


def load_and_build(config: Mapping[str, Any]) -> HistoricalCoverageAuditArtifacts:
    """只按配置列清单读取历史截止内父账本。"""

    verify_registered_inputs(config)
    inputs = config["inputs"]

    def read(name: str) -> pd.DataFrame:
        contract = inputs[name]
        return pd.read_parquet(
            _project_path(str(contract["path"])),
            columns=[str(column) for column in contract["read_columns"]],
        )

    audit = config["historical_audit_contract"]
    return build_historical_coverage_audit(
        sample_eligibility=read("g1_sample_eligibility"),
        event_eligibility=read("g1_event_eligibility"),
        mft_coverage=read("mft_coverage"),
        classified_returns=read("classified_constituent_returns"),
        membership=read("point_in_time_membership"),
        historical_cutoff=config["program"]["historical_cutoff"],
        lookback_days=int(audit["individual_return_lookback_market_days"]),
        tail_days=int(audit["tail_volatility_lookback_market_days"]),
        tail_return_days=int(audit["tail_return_lookback_market_days"]),
        member_coverage_minimum=float(audit["member_coverage_minimum"]),
        comovement_member_ratio_minimum=float(
            audit["comovement_member_ratio_minimum"]
        ),
        expected_members_per_day=int(audit["expected_members_per_day"]),
        deficit_caps=[int(value) for value in audit["non_admissible_tail_deficit_caps"]],
        mechanism_event_minimum=int(audit["mechanism_minimum_independent_events"]),
        full_model_event_minimum=int(audit["full_model_minimum_independent_events"]),
    )


def _validate_parent_counts(
    artifacts: HistoricalCoverageAuditArtifacts, config: Mapping[str, Any]
) -> None:
    metrics = artifacts.metrics
    expected = config["expected_parent_state"]
    checks = {
        "独立事件": (
            metrics["total_independent_event_count"],
            expected["total_independent_event_count"],
        ),
        "正样本原点": (
            metrics["total_positive_origin_count"],
            expected["total_positive_origin_count"],
        ),
        "非事件风险日": (
            metrics["total_non_event_risk_day_count"],
            expected["total_non_event_risk_day_count"],
        ),
        "B2 可识别事件": (
            metrics["current_b2_identifiable_event_count"],
            expected["b2_identifiable_event_count"],
        ),
        "B3 可识别事件": (
            metrics["current_b3_identifiable_event_count"],
            expected["b3_identifiable_event_count"],
        ),
    }
    for label, (actual, frozen) in checks.items():
        if int(actual) != int(frozen):
            raise EvidenceContractError(
                f"父 G1 {label}不一致：actual={actual}, frozen={frozen}"
            )
    if artifacts.audit_state["G1_DATA_AND_EVENTS"] != expected["G1_DATA_AND_EVENTS"]:
        raise EvidenceContractError("历史审计意外改变父 G1 状态")


def _build_report(artifacts: HistoricalCoverageAuditArtifacts) -> str:
    metrics = artifacts.metrics
    upper = metrics["tail_deficit_cap_event_count_upper_bounds"]
    reasons = metrics["member_window_gap_primary_reason_counts"]
    state_counts = metrics["member_day_state_counts"]
    event_rows = artifacts.event_gap_ledger.loc[
        ~artifacts.event_gap_ledger["current_b2_identifiable_event"]
    ]
    event_lines = [
        "| 事件 | 代表原点 | 尾部缺口 | 20日缺口 | 共同运动缺口 | 入指前尾部增量 | 当前原因 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in event_rows.itertuples(index=False):
        event_lines.append(
            "| "
            f"{row.event_id} | {pd.Timestamp(row.deterministic_best_origin_date).date()} | "
            f"{int(row.tail_member_deficit)} | {int(row.return20_member_deficit)} | "
            f"{int(row.comovement_member_deficit)} | {int(row.prehistory_tail_member_delta)} | "
            f"{row.b2_vs_b1_no_view_reason} |"
        )
    upper_lines = [
        f"- 尾部缺口不超过 {cap} 个成员：事件数上界 {count}"
        for cap, count in sorted(upper.items(), key=lambda item: int(item[0]))
    ]
    reason_lines = [f"- `{key}`：{value}" for key, value in sorted(reasons.items())]
    state_lines = [f"- `{key}`：{value}" for key, value in state_counts.items()]
    return "\n".join(
        [
            "# 510300 压力传导危险率 V2：G1 历史覆盖审计 V1",
            "",
            "## 裁决",
            "",
            f"- 审计状态：`{artifacts.audit_state['audit_status']}`",
            f"- 父 G1：`{artifacts.audit_state['G1_DATA_AND_EVENTS']}`（保持不变）",
            f"- 历史截止日：{metrics['historical_cutoff']}；未使用更晚观察。",
            f"- 当前 B2/B3 可识别事件：{metrics['current_b2_identifiable_event_count']}/{metrics['current_b3_identifiable_event_count']}；总事件 {metrics['total_independent_event_count']}。",
            f"- 实现—协议不一致：{'已确认' if metrics['implementation_protocol_mismatch_confirmed'] else '未发现'}。",
            f"- 入指前历史使 20 日计数增加的日期：{metrics['prehistory_return20_positive_delta_date_count']}；最大增加 {metrics['prehistory_return20_max_member_delta']} 个成员。",
            f"- 入指前历史使 60 日尾部计数增加的日期：{metrics['prehistory_tail_positive_delta_date_count']}；最大增加 {metrics['prehistory_tail_max_member_delta']} 个成员；新增跨过 98% 门的日期 {metrics['prehistory_tail_newly_passing_date_count']}。",
            "",
            "当前成员日历口径已逐日复现冻结的 20 日与 60 日覆盖计数。因此差异来自是否保留已经准入的观察开始日前公开价格历史，而不是阈值、标签或样本选择变化。",
            "",
            "## 四态历史缺口",
            "",
            *state_lines,
            f"- `source_observed=false` 的点时成员日：{metrics['explicit_source_observed_false_member_day_count']}，涉及 {metrics['explicit_source_observed_false_symbol_count']} 个证券。",
            "",
            "代表原点中，当前 60 日窗口不完整的成员—事件行按主因统计：",
            "",
            *reason_lines,
            "",
            "## 非准入诊断上界",
            "",
            "以下数字均为 `NON_ADMISSIBLE_DIAGNOSTIC_ONLY`。它们只说明如果对应尾部缺口全被合法历史证据修复时的单维事件数上界；没有重算 F/T、共同运动、模型或 G1。",
            "",
            *upper_lines,
            f"- 在已观察缺口档位中，首次达到 30 个事件的尾部缺口上限：{metrics['first_observed_tail_deficit_cap_reaching_30']}。",
            f"- 在已观察缺口档位中，首次达到 40 个事件的尾部缺口上限：{metrics['first_observed_tail_deficit_cap_reaching_40']}。",
            "",
            "## 逐事件代表原点",
            "",
            *event_lines,
            "",
            "## 权限与下一步",
            "",
            "- 本阶段只读父账本中的最小 BAD10 列及既有准入状态；未读取未来路径值、收益或组合表现。",
            "- 没有训练模型、生成概率或预测指标，没有创建 Paper/Shadow、经纪商、仓位或订单动作，`POSITION_IMPACT=0`。",
            "- 供应商缺行或停牌标志没有被改写为官方停牌，也没有填 0。",
            f"- 唯一允许的下一步：`{artifacts.audit_state['next_allowed_step']}`。",
            "- 在新阶段完整冻结、构建并全新进程重放以前，G1 继续为 `NO_VIEW`。",
            "",
        ]
    )


def build(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    verify_frozen_manifest(config_path)
    config = load_config(config_path)
    outputs = _output_paths(config)
    owned = [*TABLE_OUTPUTS.keys(), "report", "build_receipt"]
    existing = [str(outputs[name]) for name in owned if outputs[name].exists()]
    if existing:
        raise EvidenceContractError(f"历史覆盖审计构建输出已存在，禁止覆盖：{existing}")

    artifacts = load_and_build(config)
    _validate_parent_counts(artifacts, config)
    print("历史覆盖审计断言通过，正在原子写入三张诊断表。", flush=True)
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
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_HISTORICAL_COVERAGE_AUDIT_BUILD_V1",
        "created_at": now,
        "status": "PASS_HISTORICAL_COVERAGE_AUDIT_PENDING_CLEAN_REPLAY_G1_UNCHANGED",
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
        "audit_state": artifacts.audit_state,
        "actual_label_artifact_read": True,
        "actual_label_columns_read": ["origin_date", "bad10", "event_id"],
        "actual_future_path_value_read": False,
        "observation_after_historical_cutoff_read": False,
        "g1_gate_promoted": False,
        "source_state_reclassified": False,
        "missing_return_zero_filled": False,
        "model_trained": False,
        "probability_generated": False,
        "prediction_metric_generated": False,
        "portfolio_or_performance_artifact_read": False,
        "paper_or_shadow_authorized": False,
        "broker_connection_authorized": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(outputs["build_receipt"], receipt)
    print(
        "历史覆盖审计构建完成；"
        f"mismatch={artifacts.metrics['implementation_protocol_mismatch_confirmed']}，"
        f"B2_events={artifacts.metrics['current_b2_identifiable_event_count']}；"
        "G1 未提升、未训练模型。"
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
        HistoricalCoverageAuditError,
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"历史覆盖审计构建失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
