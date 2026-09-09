"""行业预期差前瞻运行闸门测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_industry_expectation_gap_forward_operations_v1 as operations_runner

from research.industry_expectation_gap_forward_operations import (
    ForwardOperationsError,
    assess_next_origin_gate,
    assess_outcome_input_gate,
    validate_operations_config,
    validate_origin_ledgers,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return {
        "origin_collection": {
            "cadence": "MONTHLY_DISTINCT_PREDICTION_DATE",
            "minimum_sources_per_industry": 2,
            "minimum_new_official_releases": 2,
            "require_source_registry_cutoff_advance": True,
            "require_sector_panel_month_advance": True,
            "require_manual_evidence_package": True,
            "automatic_origin_generation_enabled": False,
            "historical_backfill_enabled": False,
            "duplicate_prediction_date_enabled": False,
        },
        "outcome_collection": {
            "partial_horizon_return_output_enabled": False,
        },
        "safety": {
            "research_only": True,
            "may_upgrade_original_no_view": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
        },
    }


def _registry(*, cutoff: str = "2026-08-18T16:30:00+08:00") -> dict:
    return {
        "information_cutoff": cutoff,
        "sources": {
            "LOCAL": {
                "kind": "LOCAL_POINT_IN_TIME_DATA",
                "available_at": "2026-08-15T00:00:00+08:00",
                "data_as_of": "2026-07-31",
            },
            "OFFICIAL_OLD": {
                "kind": "OFFICIAL_INDUSTRY_RELEASE",
                "available_at": "2026-07-30T10:00:00+08:00",
                "data_as_of": "2026-06-30",
            },
        },
    }


def _ledger() -> dict:
    return {
        "as_of_date": "2026-08-18",
        "information_cutoff": "2026-08-18T16:30:00+08:00",
        "weight_snapshot_date": "2026-07-31",
        "historical_return_read": False,
        "model_fitting_performed": False,
        "judgments": [
            {
                "industry_l1": "电子",
                "source_ids": ["LOCAL", "OFFICIAL_OLD"],
            }
        ],
    }


def test_config_rejects_automatic_origin_generation() -> None:
    config = _config()
    config["origin_collection"]["automatic_origin_generation_enabled"] = True
    with pytest.raises(ForwardOperationsError, match="automatic_origin_generation_enabled"):
        validate_operations_config(config)


def test_duplicate_prediction_date_is_not_a_new_time_sample() -> None:
    ledger = _ledger()
    with pytest.raises(ForwardOperationsError, match="prediction_date重复"):
        validate_origin_ledgers([ledger, dict(ledger)], _registry())


def test_source_available_after_cutoff_is_rejected() -> None:
    registry = _registry()
    registry["sources"]["OFFICIAL_OLD"]["available_at"] = "2026-08-19T10:00:00+08:00"
    with pytest.raises(ForwardOperationsError, match="预测截止后"):
        validate_origin_ledgers([_ledger()], registry)


def test_current_month_and_old_evidence_keep_next_origin_closed() -> None:
    origins = validate_origin_ledgers([_ledger()], _registry())
    result = assess_next_origin_gate(
        origins,
        _registry(),
        sector_panel_latest_date="2026-07-31",
        as_of="2026-08-19T17:05:00+08:00",
        minimum_new_official_releases=2,
    )
    assert result["ready"] is False
    assert result["status"] == "WAITING_FOR_NEXT_ORIGIN_EVIDENCE"
    assert "CURRENT_MONTH_ALREADY_HAS_ORIGIN" in result["blockers"]
    assert "SOURCE_REGISTRY_CUTOFF_NOT_ADVANCED" in result["blockers"]
    assert "SECTOR_PANEL_MONTH_NOT_ADVANCED" in result["blockers"]
    assert result["automatic_origin_generation_enabled"] is False


def test_new_month_panel_and_two_new_official_releases_open_manual_gate() -> None:
    registry = _registry(cutoff="2026-09-20T16:30:00+08:00")
    registry["sources"].update(
        {
            "OFFICIAL_NEW_A": {
                "kind": "OFFICIAL_INDUSTRY_RELEASE",
                "available_at": "2026-09-10T10:00:00+08:00",
                "data_as_of": "2026-08-31",
            },
            "OFFICIAL_NEW_B": {
                "kind": "OFFICIAL_MACRO_RELEASE",
                "available_at": "2026-09-15T10:00:00+08:00",
                "data_as_of": "2026-08-31",
            },
        }
    )
    origins = validate_origin_ledgers([_ledger()], registry)
    result = assess_next_origin_gate(
        origins,
        registry,
        sector_panel_latest_date="2026-08-31",
        as_of="2026-09-20T17:05:00+08:00",
        minimum_new_official_releases=2,
    )
    assert result["ready"] is True
    assert result["status"] == "READY_FOR_MANUAL_EVIDENCE_PACKAGE"
    assert result["blockers"] == []
    assert result["manual_evidence_package_required"] is True
    assert result["candidate_date_is_authorization"] is False


def test_outcome_inputs_before_entry_fail_closed() -> None:
    result = assess_outcome_input_gate(
        entry_date="2026-08-19",
        constituent_latest_date="2026-08-18",
        etf_latest_date="2026-08-11",
        expected_latest_trading_date="2026-08-19",
    )
    assert result["status"] == "BLOCKED_OUTCOME_INPUT_DATE_MISMATCH"
    assert result["ready"] is False
    assert result["partial_horizon_return_output_enabled"] is False


def test_outcome_inputs_current_allow_only_append_only_evaluation() -> None:
    result = assess_outcome_input_gate(
        entry_date="2026-08-19",
        constituent_latest_date="2026-08-19",
        etf_latest_date="2026-08-19",
        expected_latest_trading_date="2026-08-19",
    )
    assert result["status"] == "READY_TO_REFRESH_APPEND_ONLY_EVALUATION"
    assert result["ready"] is True


def test_scheduled_task_contains_no_forecast_or_trading_generator() -> None:
    task = (
        ROOT / "scripts" / "run_industry_expectation_gap_forward_operations_task.ps1"
    ).read_text(encoding="utf-8")
    installer = (
        ROOT / "scripts" / "install_industry_expectation_gap_forward_operations_task.ps1"
    ).read_text(encoding="utf-8")
    assert "refresh_industry_expectation_gap_outcome_inputs.py" in task
    assert "--evaluate-if-ready" in task
    assert task.index("refresh_industry_expectation_gap_outcome_inputs.py") < task.index(
        "--evaluate-if-ready"
    )
    assert "generate_510300_small_account_paper_signal" not in task
    assert "run_industry_expectation_gap_etf_model_v1.py" not in task
    assert "immutable_receipt = $true" in task
    assert "industry_expectation_gap_v1_evaluation\\task_runs" in task
    assert "RestartCount 3" in installer
    assert "WakeToRun" in installer


def test_operations_use_frozen_recovery_runner_and_persist_fail_closed_health() -> None:
    config = (
        ROOT / "config" / "industry_expectation_gap_forward_operations_v1.yaml"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT / "scripts" / "run_industry_expectation_gap_forward_operations_v1.py"
    ).read_text(encoding="utf-8")
    assert "industry_expectation_gap_forward_evaluation_v1_1.py" in config
    assert "industry_expectation_gap_forward_evaluation_v1_1_recovery_manifest.json" in config
    assert "EVALUATION_FAILED_CLOSED" in runner
    assert '"operations_health"' in runner
    assert 'return 1 if report["operations_health"] != "PASS" else 0' in runner


def test_subprocess_failure_is_captured_before_fail_closed_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        operations_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=7,
            stdout="",
            stderr="模拟评价失败",
        ),
    )
    record = operations_runner._run_python_capture(
        "scripts/run_industry_expectation_gap_forward_evaluation_v1_1.py"
    )
    assert record["exit_code"] == 7
    assert record["stderr"] == "模拟评价失败"
    with pytest.raises(operations_runner.ForwardOperationsError, match="退出码7"):
        operations_runner._run_python(
            "scripts/run_industry_expectation_gap_forward_evaluation_v1_1.py"
        )
