"""在 V2 协议冻结后重建 BAD10 事件、样本权重与状态账本。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    atomic_write_text_new,
    canonical_sha256,
    file_evidence,
    sha256_file,
)
from research.stress_transmission_hazard_v2 import (
    PROGRAM_ID,
    StressTransmissionContractError,
    build_bad10_event_ledger,
)
from scripts.freeze_510300_stress_transmission_hazard_v2 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
)


def _project_path(value: str) -> Path:
    path = ROOT / Path(value.replace("/", "\\"))
    return path


def _csv_text(frame: pd.DataFrame) -> str:
    return frame.to_csv(
        index=False,
        date_format="%Y-%m-%d",
        lineterminator="\n",
        na_rep="",
    )


def _build_human_report(
    *,
    event_count: int,
    positive_origin_count: int,
    non_event_risk_day_count: int,
    mechanism_event_prerequisite: bool,
    full_model_event_prerequisite: bool,
    non_event_prerequisite: bool,
) -> str:
    return "\n".join(
        [
            "# 510300 压力传导危险率 V2 BAD10 事件账本",
            "",
            "本文件只报告标签事件结构与样本权重，不包含特征、概率、AUC、收益、",
            "夏普、净值、仓位或订单。",
            "",
            "## 固定构造",
            "",
            "- 正样本按 `[下一开盘, 十日路径末日]` 闭区间的传递重叠合并；",
            "- 每个独立事件内所有正样本训练权重之和为 1；",
            "- 每个非事件风险日训练权重为 1；",
            "- 每个事件保存唯一 split group、事件 Bootstrap block 与日历年 block；",
            "- 当前 split 一律为 `UNASSIGNED_PRE_MODEL_FREEZE`。",
            "",
            "## 计数前提",
            "",
            f"- 独立 BAD10 事件：{event_count}",
            f"- BAD10 正样本原点：{positive_origin_count}",
            f"- 非事件风险日：{non_event_risk_day_count}",
            f"- 机制发现事件数前提（至少 30）：{'PASS' if mechanism_event_prerequisite else 'FAIL'}",
            f"- 完整三系数模型/组合事件数前提（至少 40）：{'PASS' if full_model_event_prerequisite else 'FAIL'}",
            f"- 非事件风险日前提（至少 750）：{'PASS' if non_event_prerequisite else 'FAIL'}",
            "",
            "这些计数只裁决 G1 的事件数量前提。历史成分收益四态尚未完成来源重建，",
            "因此 G0 和完整 G1 均未通过，G2—G7 未运行。",
            "",
            "```text",
            "RESEARCH_STATE = DISCOVERY_ONLY",
            "RETURN_EVALUATION = NOT_ALLOWED",
            "MODEL_TRAINING = NOT_ALLOWED_AT_CURRENT_STAGE",
            "PORTFOLIO_EVALUATION = NOT_ALLOWED",
            "POSITION_IMPACT = 0",
            "```",
            "",
        ]
    )


def build(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    manifest = verify_frozen_manifest(config_path)
    if config["program"]["program_id"] != PROGRAM_ID:
        raise EvidenceContractError("V2 program_id 不一致")
    if config["program"]["return_evaluation"] != "NOT_ALLOWED":
        raise EvidenceContractError("当前阶段必须禁止收益读取")
    if manifest.get("feature_values_constructed") is not False:
        raise EvidenceContractError("冻结 manifest 不再处于特征值构造前状态")

    freeze_contract = config["freeze_contract"]
    outputs = {
        "events": _project_path(str(freeze_contract["event_ledger_output"])),
        "samples": _project_path(str(freeze_contract["sample_ledger_output"])),
        "report": _project_path(str(freeze_contract["event_ledger_report_output"])),
        "status": _project_path(str(freeze_contract["status_output"])),
        "receipt": _project_path(str(freeze_contract["event_ledger_receipt_output"])),
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise EvidenceContractError(f"BAD10 V2 输出已存在，禁止覆盖：{existing}")

    origin_contract = config["input_identity_registry"]["bad10_origin_panel"]
    origin_path = _project_path(str(origin_contract["path"]))
    if sha256_file(origin_path) != str(origin_contract["sha256"]):
        raise EvidenceContractError("BAD10 原点面板在协议冻结后漂移")
    origin_panel = pd.read_parquet(origin_path)
    events, samples = build_bad10_event_ledger(origin_panel)

    event_count = int(len(events))
    positive_origin_count = int(samples["bad10"].eq(1).sum())
    non_event_risk_day_count = int(samples["bad10"].eq(0).sum())
    g1 = config["gates"]["G1_DATA_AND_EVENTS"]
    mechanism_event_prerequisite = event_count >= int(
        g1["minimum_independent_events_for_mechanism_discovery"]
    )
    full_model_event_prerequisite = event_count >= int(
        g1["minimum_independent_events_for_full_three_coefficient_model"]
    )
    non_event_prerequisite = non_event_risk_day_count >= int(
        g1["minimum_non_event_risk_days"]
    )

    atomic_write_text_new(outputs["events"], _csv_text(events))
    atomic_write_text_new(outputs["samples"], _csv_text(samples))
    report_text = _build_human_report(
        event_count=event_count,
        positive_origin_count=positive_origin_count,
        non_event_risk_day_count=non_event_risk_day_count,
        mechanism_event_prerequisite=mechanism_event_prerequisite,
        full_model_event_prerequisite=full_model_event_prerequisite,
        non_event_prerequisite=non_event_prerequisite,
    )
    atomic_write_text_new(outputs["report"], report_text)

    created_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    status: dict[str, Any] = {
        "status_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_STATUS_20260903",
        "created_at": created_at,
        "program_id": PROGRAM_ID,
        "program_version": config["program"]["version"],
        "current_stage": "PROTOCOL_FROZEN_AND_BAD10_EVENT_LEDGER_REBUILT",
        "protocol_freeze": "PASS_HASH_REGISTERED_G0_CLEAN_REPLAY_PENDING",
        "v1_original_output": "PRESERVE_IMMUTABLE",
        "v1_g2_authoritative_result": "INVALIDATED",
        "economic_hypothesis": "NOT_CLEANLY_ADJUDICATED",
        "bad10_event_ledger": "PASS_DETERMINISTIC_EVENT_AND_WEIGHT_CONTRACT",
        "event_count_prerequisites": {
            "independent_event_count": event_count,
            "mechanism_discovery_minimum": int(
                g1["minimum_independent_events_for_mechanism_discovery"]
            ),
            "mechanism_discovery_count_passed": mechanism_event_prerequisite,
            "full_model_and_portfolio_minimum": int(
                g1["minimum_independent_events_for_full_three_coefficient_model"]
            ),
            "full_model_and_portfolio_count_passed": full_model_event_prerequisite,
            "positive_origin_count": positive_origin_count,
            "non_event_risk_day_count": non_event_risk_day_count,
            "non_event_risk_day_minimum": int(g1["minimum_non_event_risk_days"]),
            "non_event_risk_day_count_passed": non_event_prerequisite,
        },
        "historical_constituent_four_state_ledger": (
            "NOT_BUILT_LEGACY_INPUTS_QUARANTINED_PENDING_OFFICIAL_SUSPENSION_"
            "AND_CORPORATE_ACTION_PROVENANCE"
        ),
        "industry_predictive_use": "NOT_ALLOWED",
        "gates": {
            "G0": "NOT_PASSED_PENDING_COMMITTED_CLEAN_REPLAY_AND_HISTORICAL_FOUR_STATE_LEDGER",
            "G1": "NOT_ADJUDICATED_EVENT_COUNT_PREREQUISITES_ONLY",
            "G2": "NOT_RUN",
            "G3": "NOT_RUN",
            "G4": "NOT_RUN",
            "G5": "NOT_RUN",
            "G6": "NOT_RUN",
            "G7": "NOT_RUN",
        },
        "current_validated_high_sharpe_strategy": "NONE",
        "research_state": "DISCOVERY_ONLY",
        "return_evaluation": "NOT_ALLOWED",
        "model_training": "NOT_ALLOWED_AT_CURRENT_STAGE",
        "probability_threshold": None,
        "portfolio_evaluation": "NOT_ALLOWED",
        "model_position_target": "UNSET",
        "order_authorization": "NOT_AUTHORIZED",
        "paper_signal_authorized": False,
        "shadow_signal_authorized": False,
        "broker_connection_authorized": False,
        "live_trading_authorized": False,
        "position_impact": 0,
        "auc_read": False,
        "portfolio_return_read": False,
        "sharpe_read": False,
    }
    status["status_payload_sha256"] = canonical_sha256(status)
    atomic_write_json_new(outputs["status"], status)

    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_BAD10_LEDGER_20260903",
        "created_at": created_at,
        "program_id": PROGRAM_ID,
        "status": "PASS_BAD10_EVENT_AND_SAMPLE_WEIGHT_LEDGER_NO_PERFORMANCE_READ",
        "frozen_manifest": file_evidence(
            _project_path(str(freeze_contract["manifest_output"])), project_root=ROOT
        ),
        "source_origin_panel": file_evidence(origin_path, project_root=ROOT),
        "outputs": {
            key: file_evidence(path, project_root=ROOT)
            for key, path in outputs.items()
            if key != "receipt"
        },
        "event_count": event_count,
        "positive_origin_count": positive_origin_count,
        "non_event_risk_day_count": non_event_risk_day_count,
        "event_weight_sum_invariant": "EACH_POSITIVE_EVENT_EQUALS_ONE",
        "split_integrity": "WHOLE_EVENT_ONLY_CURRENTLY_UNASSIGNED",
        "performance_values_read": False,
        "model_trained": False,
        "portfolio_generated": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(outputs["receipt"], receipt)
    print(
        "BAD10 V2 事件账本已重建；"
        f"events={event_count}，positive_origins={positive_origin_count}，"
        f"non_event_days={non_event_risk_day_count}；未读取任何性能结果。"
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="重建 510300 压力传导危险率 V2 BAD10 事件账本"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        build(args.config)
        return 0
    except (
        EvidenceContractError,
        StressTransmissionContractError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"BAD10 V2 事件账本构建失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
