from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.priority_forward_threshold_events import (
    ThresholdEventError,
    append_new_threshold_events,
    derive_threshold_candidates,
    load_threshold_events,
    next_thresholds,
)


def _report() -> dict:
    return {
        "generated_at": "2026-08-19T17:15:00+08:00",
        "directions": {
            "primary_market_pcf_iopv": {
                "full_coverage_days": 2,
                "gates": {
                    "quality_audit": {"required": 20, "eligible": False},
                    "feature_freeze": {"required": 40, "eligible": False},
                    "first_unseen_evaluation": {"required": 80, "eligible": False},
                    "replication": {"required": 120, "eligible": False},
                },
            },
            "industry_expectation_gap": {
                "mature_origin_cluster_count": 0,
                "non_overlapping_60d_block_count": 0,
                "calibration": {
                    "minimum_origin_clusters": 20,
                    "minimum_non_overlapping_60d_blocks": 4,
                    "eligible": False,
                },
                "model_comparison": {
                    "minimum_origin_clusters": 40,
                    "minimum_non_overlapping_60d_blocks": 8,
                    "eligible": False,
                },
            },
            "orthogonal_low_vol_replication": {"eligible_to_start": False},
        },
    }


def test_current_counts_produce_no_threshold_event() -> None:
    report = _report()
    assert derive_threshold_candidates(report) == []
    remaining = next_thresholds(report)
    assert remaining["primary_market_remaining_full_coverage_days"]["quality_audit"] == 18
    assert remaining["industry_calibration_remaining"] == {
        "mature_origin_clusters": 20,
        "non_overlapping_60d_blocks": 4,
    }
    assert remaining["orthogonal_low_vol_governance_gate_passed"] is False


def test_first_crossing_is_append_only_and_never_authorizes_trading(tmp_path: Path) -> None:
    report = _report()
    report["directions"]["primary_market_pcf_iopv"]["full_coverage_days"] = 20
    report["directions"]["primary_market_pcf_iopv"]["gates"]["quality_audit"][
        "eligible"
    ] = True
    candidates = derive_threshold_candidates(report)
    assert [item["event_id"] for item in candidates] == [
        "PRIMARY_MARKET_QUALITY_AUDIT_20"
    ]
    assert candidates[0]["action"] == "QUALITY_AUDIT_ONLY"
    assert candidates[0]["automatic_trading_authorized"] is False
    assert all(value is False for value in candidates[0]["safety"].values())

    ledger = tmp_path / "threshold_events.jsonl"
    first = append_new_threshold_events(ledger, candidates)
    second = append_new_threshold_events(ledger, candidates)
    assert first["new_event_count"] == 1
    assert second["new_event_count"] == 0
    assert len(load_threshold_events(ledger)) == 1


def test_existing_event_definition_change_fails_closed(tmp_path: Path) -> None:
    report = _report()
    report["directions"]["primary_market_pcf_iopv"]["full_coverage_days"] = 20
    report["directions"]["primary_market_pcf_iopv"]["gates"]["quality_audit"][
        "eligible"
    ] = True
    candidate = derive_threshold_candidates(report)[0]
    ledger = tmp_path / "threshold_events.jsonl"
    append_new_threshold_events(ledger, [candidate])
    changed = copy.deepcopy(candidate)
    changed["action"] = "UNSAFE_CHANGED_ACTION"
    with pytest.raises(ThresholdEventError, match="定义发生变化"):
        append_new_threshold_events(ledger, [changed])


def test_governance_pass_only_emits_separate_protocol_event() -> None:
    report = _report()
    report["directions"]["orthogonal_low_vol_replication"]["eligible_to_start"] = True
    event = derive_threshold_candidates(report)[0]
    assert event["event_id"] == "ORTHOGONAL_LOW_VOL_GOVERNANCE_GATE_PASS"
    assert event["action"] == (
        "REQUIRES_SEPARATE_PREREGISTERED_PROTOCOL_NO_AUTOMATIC_START"
    )
    assert event["automatic_trading_authorized"] is False


def test_ledger_with_trading_authorization_is_rejected(tmp_path: Path) -> None:
    unsafe = {
        "event_id": "UNSAFE",
        "automatic_trading_authorized": True,
        "safety": {
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
        },
    }
    ledger = tmp_path / "threshold_events.jsonl"
    ledger.write_text(json.dumps(unsafe) + "\n", encoding="utf-8")
    with pytest.raises(ThresholdEventError, match="非法授权交易"):
        load_threshold_events(ledger)
