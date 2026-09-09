from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts import run_t_only_forward_v1_daily_v1_2 as daily
from scripts.t_only_forward_maturity_only_v1_2 import (
    FORBIDDEN_PUBLIC_KEYS,
    assert_maturity_only_payload,
    build_maturity_only_snapshot,
    build_operational_failure_snapshot,
    retire_legacy_run_status,
    sanitize_existing_outputs,
)


def detailed_report(*, new_days: int = 4, closed_cycles: int = 0) -> dict[str, Any]:
    mature = new_days >= 252 and closed_cycles >= 3
    return {
        "project_id": "510300_T_ONLY_FORWARD_V1_1",
        "status": "PASS_FORWARD_GATES" if mature else "COLLECTING",
        "current_view": (
            "MATURE_SHADOW_EVIDENCE_ONLY"
            if mature
            else "NO_VIEW_UNTIL_FORWARD_MATURITY"
        ),
        "as_of_market_date": "2026-08-24",
        "forward_signal_start": "2026-08-19",
        "new_trading_days": new_days,
        "closed_cycles": closed_cycles,
        "maturity": {
            "trading_days_required": 252,
            "closed_cycles_required": 3,
            "mature": mature,
        },
        "current_shadow_signal": {
            "signal_date": "2026-08-24",
            "target_exposure": 0.325,
        },
        "metrics": {
            "strategy": {
                "total_return": 0.123,
                "cagr": 0.456,
                "sharpe": 1.2,
                "maximum_drawdown": -0.1,
                "average_exposure": 0.325,
            }
        },
        "gates": {"net_return_above_cash": True},
        "latest_indicator_diagnostic": {"weekly_state": True},
    }


def passing_data_gate() -> dict[str, Any]:
    return {
        "status": "PASS",
        "actual_last_date": "2026-08-24",
        "frozen_input_unchanged": True,
        "dividend_secondary_cross_check": {"status": "PASS"},
    }


def ledger_audit(rows: int) -> dict[str, Any]:
    return {
        "expected_rows": rows,
        "actual_rows": rows,
        "duplicate_dates": 0,
        "status": "PASS",
        "complete": True,
        "sha256": "a" * 64,
    }


def walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(walk_keys(item))
    return keys


def test_immature_snapshot_contains_only_maturity_and_completeness() -> None:
    snapshot = build_maturity_only_snapshot(
        source_report=detailed_report(),
        data_gate=passing_data_gate(),
        ledger_audit=ledger_audit(4),
        run_completeness={"status": "PASS", "complete": True, "run_id": "R1"},
        source_detailed_status_sha256="b" * 64,
        data_gate_sha256="c" * 64,
    )

    assert snapshot["status"] == "COLLECTING"
    assert snapshot["current_view"] == "NO_VIEW_UNTIL_FORWARD_MATURITY"
    assert snapshot["new_trading_days"] == 4
    assert snapshot["closed_cycles"] == 0
    assert snapshot["maturity"] == {
        "trading_days_required": 252,
        "closed_cycles_required": 3,
        "mature": False,
    }
    assert snapshot["completeness"]["overall_complete"] is True
    assert snapshot["evaluation_status"] == "NOT_EVALUATED_BEFORE_MATURITY"
    assert not walk_keys(snapshot).intersection(FORBIDDEN_PUBLIC_KEYS)


def test_mature_snapshot_requires_separate_review_without_auto_disclosure() -> None:
    snapshot = build_maturity_only_snapshot(
        source_report=detailed_report(new_days=252, closed_cycles=3),
        data_gate=passing_data_gate(),
        ledger_audit=ledger_audit(252),
        run_completeness={"status": "PASS", "complete": True, "run_id": "R2"},
    )

    assert snapshot["status"] == "MATURE_REVIEW_REQUIRED"
    assert snapshot["current_view"] == "MATURE_REVIEW_REQUIRED"
    assert snapshot["maturity"]["mature"] is True
    assert not walk_keys(snapshot).intersection(FORBIDDEN_PUBLIC_KEYS)


def test_changed_maturity_threshold_is_rejected() -> None:
    source = detailed_report()
    source["maturity"]["trading_days_required"] = 251
    with pytest.raises(RuntimeError, match="成熟门槛发生变化"):
        build_maturity_only_snapshot(
            source_report=source,
            data_gate=passing_data_gate(),
            ledger_audit=ledger_audit(4),
        )


def test_sanitize_existing_outputs_overwrites_status_and_guide(tmp_path: Path) -> None:
    report_json = tmp_path / "status.json"
    report_markdown = tmp_path / "status.md"
    guide_json = tmp_path / "guide.json"
    guide_markdown = tmp_path / "guide.md"
    data_gate_path = tmp_path / "data_gate.json"
    ledger_path = tmp_path / "ledger.parquet"
    report_json.write_text(
        json.dumps(detailed_report(), ensure_ascii=False), encoding="utf-8"
    )
    data_gate_path.write_text(
        json.dumps(passing_data_gate(), ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(
        {"date": pd.date_range("2026-08-19", periods=4, freq="D")}
    ).to_parquet(ledger_path, index=False)

    snapshot = sanitize_existing_outputs(
        run_completeness={"status": "PASS", "complete": True, "run_id": "R3"},
        report_json=report_json,
        report_markdown=report_markdown,
        guide_json=guide_json,
        guide_markdown=guide_markdown,
        data_gate_path=data_gate_path,
        ledger_path=ledger_path,
    )
    saved_status = json.loads(report_json.read_text(encoding="utf-8"))
    saved_guide = json.loads(guide_json.read_text(encoding="utf-8"))

    assert snapshot == saved_status
    assert saved_guide["artifact_role"] == "T_ONLY_MATURITY_ONLY_DAILY_DELIVERABLE"
    assert not walk_keys(saved_status).intersection(FORBIDDEN_PUBLIC_KEYS)
    assert not walk_keys(saved_guide).intersection(FORBIDDEN_PUBLIC_KEYS)
    assert "0.123" not in report_markdown.read_text(encoding="utf-8")

    second = sanitize_existing_outputs(
        run_completeness={"status": "PASS", "complete": True, "run_id": "R4"},
        report_json=report_json,
        report_markdown=report_markdown,
        guide_json=guide_json,
        guide_markdown=guide_markdown,
        data_gate_path=data_gate_path,
        ledger_path=ledger_path,
    )
    assert second["completeness"]["run"]["run_id"] == "R4"


def test_operational_failure_does_not_reuse_performance_or_signal() -> None:
    failure = build_operational_failure_snapshot(
        previous_snapshot=detailed_report(),
        run_id="R5",
        failure_class="EXTERNAL_FREE_SOURCE_FAILED",
        failure_stage="refresh",
        failure_message="连接失败",
    )

    assert failure["status"] == "FAILED"
    assert failure["current_view"] == "NO_VIEW_OPERATIONAL_FAILURE"
    assert failure["completeness"]["overall_complete"] is False
    assert not walk_keys(failure).intersection(FORBIDDEN_PUBLIC_KEYS)
    assert_maturity_only_payload(failure)


def test_legacy_run_status_is_replaced_by_maturity_only_pointer(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "daily_run_status.json"
    legacy.write_text(
        json.dumps(
            {
                "output_audit": {
                    "current_shadow_signal": {"target_exposure": 0.65},
                    "daily_decision": "SHADOW_ACTION_PENDING_NEXT_OPEN",
                    "daily_headline": "下一交易日影子加仓",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    snapshot = build_maturity_only_snapshot(
        source_report=detailed_report(),
        data_gate=passing_data_gate(),
        ledger_audit=ledger_audit(4),
    )

    payload = retire_legacy_run_status(snapshot, legacy_run_status=legacy)

    saved = json.loads(legacy.read_text(encoding="utf-8"))
    assert payload == saved
    assert saved["status"] == "SUPERSEDED_BY_MATURITY_ONLY_V1_2"
    assert not walk_keys(saved).intersection(FORBIDDEN_PUBLIC_KEYS)


def test_failure_classification_distinguishes_data_and_program_failures() -> None:
    assert daily.classify_stage_failure("refresh", {"exit_code": 3}) == "DATA_GATE_FAILED"
    assert daily.classify_stage_failure("output_audit", None) == "PROGRAM_FAILED"


def test_overall_status_never_claims_temporary_strategy_result() -> None:
    assert (
        daily.overall_success_status("COLLECTING")
        == "SUCCESS_MATURITY_ONLY_COLLECTING"
    )
    assert (
        daily.overall_success_status("MATURE_REVIEW_REQUIRED")
        == "SUCCESS_MATURE_REVIEW_REQUIRED"
    )
    with pytest.raises(RuntimeError, match="成熟度状态非法"):
        daily.overall_success_status("PASS_FORWARD_GATES")
