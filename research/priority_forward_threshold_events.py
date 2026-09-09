"""优先前瞻研究门槛的首次跨越事件账本。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


class ThresholdEventError(ValueError):
    """门槛事件账本不满足追加式或安全约束。"""


SAFETY = {
    "position_mapping_enabled": False,
    "order_generation_enabled": False,
    "broker_connection_enabled": False,
    "live_trading_enabled": False,
}


def _candidate(
    event_id: str,
    direction: str,
    gate: str,
    required: Mapping[str, int],
    observed: Mapping[str, int],
    action: str,
    triggered_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "event_id": event_id,
        "event_type": "THRESHOLD_FIRST_CROSSING",
        "triggered_at": triggered_at,
        "direction": direction,
        "gate": gate,
        "required": dict(required),
        "observed": dict(observed),
        "action": action,
        "automatic_trading_authorized": False,
        "safety": dict(SAFETY),
    }


def derive_threshold_candidates(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """仅为已经满足的门槛生成稳定事件 ID，不把门槛视为交易授权。"""

    triggered_at = str(report["generated_at"])
    directions = report["directions"]
    primary = directions["primary_market_pcf_iopv"]
    full_days = int(primary["full_coverage_days"])
    candidates: list[dict[str, Any]] = []
    primary_definitions = [
        (
            "quality_audit",
            "PRIMARY_MARKET_QUALITY_AUDIT_20",
            "QUALITY_AUDIT_ONLY",
        ),
        (
            "feature_freeze",
            "PRIMARY_MARKET_FEATURE_FREEZE_40",
            "FREEZE_FEATURE_DEFINITION_ONLY",
        ),
        (
            "first_unseen_evaluation",
            "PRIMARY_MARKET_FIRST_UNSEEN_EVALUATION_80",
            "RUN_PREREGISTERED_RESEARCH_EVALUATION_ONLY",
        ),
        (
            "replication",
            "PRIMARY_MARKET_REPLICATION_EVALUATION_120",
            "RUN_PREREGISTERED_REPLICATION_ONLY",
        ),
    ]
    for gate, event_id, action in primary_definitions:
        gate_state = primary["gates"][gate]
        if gate_state["eligible"] is True:
            candidates.append(
                _candidate(
                    event_id,
                    "PRIMARY_MARKET_PCF_IOPV",
                    gate.upper(),
                    {"full_coverage_days": int(gate_state["required"])},
                    {"full_coverage_days": full_days},
                    action,
                    triggered_at,
                )
            )

    industry = directions["industry_expectation_gap"]
    industry_definitions = [
        (
            "calibration",
            "INDUSTRY_EXPECTATION_GAP_CALIBRATION_20_ORIGINS_4_BLOCKS",
            "CALIBRATION_AUDIT_ONLY",
        ),
        (
            "model_comparison",
            "INDUSTRY_EXPECTATION_GAP_MODEL_COMPARISON_40_ORIGINS_8_BLOCKS",
            "RUN_FROZEN_MODEL_COMPARISON_ONLY",
        ),
    ]
    for gate, event_id, action in industry_definitions:
        gate_state = industry[gate]
        if gate_state["eligible"] is True:
            candidates.append(
                _candidate(
                    event_id,
                    "INDUSTRY_EXPECTATION_GAP",
                    gate.upper(),
                    {
                        "mature_origin_clusters": int(gate_state["minimum_origin_clusters"]),
                        "non_overlapping_60d_blocks": int(
                            gate_state["minimum_non_overlapping_60d_blocks"]
                        ),
                    },
                    {
                        "mature_origin_clusters": int(
                            industry["mature_origin_cluster_count"]
                        ),
                        "non_overlapping_60d_blocks": int(
                            industry["non_overlapping_60d_block_count"]
                        ),
                    },
                    action,
                    triggered_at,
                )
            )

    low_vol = directions["orthogonal_low_vol_replication"]
    if low_vol["eligible_to_start"] is True:
        candidates.append(
            _candidate(
                "ORTHOGONAL_LOW_VOL_GOVERNANCE_GATE_PASS",
                "ORTHOGONAL_LOW_VOL_REPLICATION",
                "GOVERNANCE_PASS",
                {"governance_pass": 1},
                {"governance_pass": 1},
                "REQUIRES_SEPARATE_PREREGISTERED_PROTOCOL_NO_AUTOMATIC_START",
                triggered_at,
            )
        )
    return candidates


def load_threshold_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ThresholdEventError(f"门槛事件账本第{line_number}行非法") from exc
        if not isinstance(event, dict) or not isinstance(event.get("event_id"), str):
            raise ThresholdEventError(f"门槛事件账本第{line_number}行缺少event_id")
        if event["event_id"] in seen:
            raise ThresholdEventError(f"门槛事件ID重复：{event['event_id']}")
        if event.get("automatic_trading_authorized") is not False:
            raise ThresholdEventError(f"门槛事件非法授权交易：{event['event_id']}")
        if event.get("safety") != SAFETY:
            raise ThresholdEventError(f"门槛事件安全字段不完整：{event['event_id']}")
        seen.add(event["event_id"])
        events.append(event)
    return events


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def append_new_threshold_events(
    path: Path,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """只追加首次跨越事件；相同事件再次出现不重复写入。"""

    existing = load_threshold_events(path)
    by_id = {event["event_id"]: event for event in existing}
    new_events: list[dict[str, Any]] = []
    for candidate_value in candidates:
        candidate = dict(candidate_value)
        event_id = str(candidate["event_id"])
        prior = by_id.get(event_id)
        if prior is not None:
            immutable_fields = (
                "event_type",
                "direction",
                "gate",
                "required",
                "action",
                "automatic_trading_authorized",
                "safety",
            )
            if any(prior.get(field) != candidate.get(field) for field in immutable_fields):
                raise ThresholdEventError(f"既有门槛事件定义发生变化：{event_id}")
            continue
        by_id[event_id] = candidate
        new_events.append(candidate)
    if new_events:
        combined = [*existing, *new_events]
        content = "".join(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            for event in combined
        )
        _atomic_write(path, content)
    return {
        "ledger_path": path.as_posix(),
        "total_event_count": len(existing) + len(new_events),
        "new_event_count": len(new_events),
        "new_event_ids": [event["event_id"] for event in new_events],
        "all_event_ids": [event["event_id"] for event in existing + new_events],
    }


def next_thresholds(report: Mapping[str, Any]) -> dict[str, Any]:
    directions = report["directions"]
    primary = directions["primary_market_pcf_iopv"]
    full_days = int(primary["full_coverage_days"])
    primary_remaining = {
        gate: max(0, int(value["required"]) - full_days)
        for gate, value in primary["gates"].items()
        if value["eligible"] is not True
    }
    industry = directions["industry_expectation_gap"]
    calibration = industry["calibration"]
    comparison = industry["model_comparison"]
    return {
        "primary_market_remaining_full_coverage_days": primary_remaining,
        "industry_calibration_remaining": {
            "mature_origin_clusters": max(
                0,
                int(calibration["minimum_origin_clusters"])
                - int(industry["mature_origin_cluster_count"]),
            ),
            "non_overlapping_60d_blocks": max(
                0,
                int(calibration["minimum_non_overlapping_60d_blocks"])
                - int(industry["non_overlapping_60d_block_count"]),
            ),
        },
        "industry_model_comparison_remaining": {
            "mature_origin_clusters": max(
                0,
                int(comparison["minimum_origin_clusters"])
                - int(industry["mature_origin_cluster_count"]),
            ),
            "non_overlapping_60d_blocks": max(
                0,
                int(comparison["minimum_non_overlapping_60d_blocks"])
                - int(industry["non_overlapping_60d_block_count"]),
            ),
        },
        "orthogonal_low_vol_governance_gate_passed": bool(
            directions["orthogonal_low_vol_replication"]["eligible_to_start"]
        ),
    }


__all__ = [
    "ThresholdEventError",
    "append_new_threshold_events",
    "derive_threshold_candidates",
    "load_threshold_events",
    "next_thresholds",
]
