"""优先前瞻研究统一状态测试。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from scripts.render_priority_forward_research_status import (
    _task_status_sort_key,
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "priority_forward_research_operations_v1.yaml"


def _config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_current_report_preserves_all_research_safety_boundaries() -> None:
    report = build_report(
        _config(),
        datetime(2026, 8, 19, 17, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert report["overall_status"] == "COLLECTING_FORWARD_WITH_BLOCKERS"
    assert report["safety"] == {
        "research_only": True,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }
    primary = report["directions"]["primary_market_pcf_iopv"]
    assert primary["gates"]["first_unseen_evaluation"]["required"] == 80
    assert primary["gates"]["replication"]["required"] == 120
    industry = report["directions"]["industry_expectation_gap"]
    assert industry["origin_cluster_count"] >= industry["mature_origin_cluster_count"]
    assert industry["calibration"]["minimum_origin_clusters"] == 20
    assert industry["model_comparison"]["minimum_origin_clusters"] == 40
    assert report["scheduler"]["registered_tasks_verified"] is False
    assert report["scheduler"]["manual_runners_verified"] is True
    assert report["scheduler"]["windows_powershell_5_1_parser_verified"] is True
    assert report["scheduler"]["login_autostart_verified"] is False
    assert report["scheduler"]["session_supervisor_status"].startswith("RUNNING_")
    assert report["scheduler"]["codex_heartbeats_active"] is True
    assert report["scheduler"]["primary_market_heartbeat_id"] == "510300-pcf-iopv"
    assert report["scheduler"]["close_validation_heartbeat_id"] == "510300"
    assert any(
        alert["direction"] == "WINDOWS_TASK_SCHEDULER"
        and alert["severity"] == "WARNING"
        and alert["code"] == "CODEX_HEARTBEATS_ACTIVE_WINDOWS_FALLBACK_BLOCKED"
        for alert in report["alerts"]
    )


def test_low_vol_direction_is_gated_by_current_governance() -> None:
    report = build_report(
        _config(),
        datetime(2026, 8, 19, 17, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    low_vol = report["directions"]["orthogonal_low_vol_replication"]
    if report["governance_status"] == "PASS":
        assert low_vol["eligible_to_start"] is True
    else:
        assert low_vol["eligible_to_start"] is False
        assert low_vol["status"] == "NOT_STARTED_GOVERNANCE_GATE_BLOCKED"


def test_daily_status_task_is_reporting_only() -> None:
    runner = (ROOT / "scripts" / "run_priority_forward_research_status_task.ps1").read_text(
        encoding="utf-8-sig"
    )
    installer = (
        ROOT / "scripts" / "install_priority_forward_research_status_task.ps1"
    ).read_text(encoding="utf-8-sig")
    assert "render_priority_forward_research_status.py" in runner
    assert "generate_510300_small_account_paper_signal" not in runner
    assert "order_generation_enabled = $false" in runner
    assert "broker_connection_enabled = $false" in runner
    assert "immutable_receipt = $true" in runner
    assert "priority_forward_status_task_runs" in runner
    assert "priority_forward_status_runs" in runner
    assert "report_snapshot_file" in runner
    assert "WriteAllBytes" in runner
    assert "WakeToRun" in installer
    assert "RestartCount 2" in installer


def test_status_config_tracks_immutable_task_receipts_and_threshold_ledger() -> None:
    config = _config()
    assert config["inputs"]["primary_market_task_receipt_glob"].endswith("/*.json")
    assert config["inputs"]["industry_task_receipt_glob"].endswith("/*.json")
    assert config["inputs"]["scheduler_installation_status"].endswith(".json")
    assert config["inputs"]["supervisor_status"].endswith(".json")
    assert config["inputs"]["codex_automation_status"].endswith(".json")
    assert config["outputs"]["threshold_event_ledger"].endswith(".jsonl")


def test_immutable_receipt_wins_tie_with_mutable_current_pointer() -> None:
    ended_at = "2026-08-19T17:05:00+08:00"
    current = {"ended_at": ended_at, "immutable_receipt": False}
    receipt = {"ended_at": ended_at, "immutable_receipt": True}
    path = ROOT / "does_not_need_to_exist.json"
    assert _task_status_sort_key(receipt, path) > _task_status_sort_key(current, path)


def test_unified_status_fails_closed_if_any_trading_switch_is_enabled() -> None:
    config = _config()
    config["safety"]["live_trading_enabled"] = True
    with pytest.raises(ValueError, match="安全开关"):
        build_report(
            config,
            datetime(2026, 8, 19, 17, 15, tzinfo=ZoneInfo("Asia/Shanghai")),
        )
