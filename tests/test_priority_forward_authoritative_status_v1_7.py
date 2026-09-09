from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.render_priority_forward_authoritative_status_v1_7 import (
    ROOT,
    build_report,
    load_config,
    select_latest_direct_task_status,
)


CONFIG_PATH = ROOT / "config" / "priority_forward_authoritative_status_v1_7.yaml"


def task_payload(receipt_id: str, ended_at: str, status: str) -> dict:
    return {
        "receipt_id": receipt_id,
        "immutable_receipt": True,
        "task_name": "Codex-510300-Primary-Market-Collector",
        "ended_at": ended_at,
        "collection_status": status,
    }


def test_selector_uses_latest_direct_receipt_and_prefers_receipt_directory() -> None:
    old = task_payload("old", "2026-08-26T09:25:03+08:00", "PROGRAM_FAILED")
    latest = task_payload("latest", "2026-08-28T09:31:19+08:00", "EXTERNAL_FREE_SOURCE_FAILED")
    selected_path, selected = select_latest_direct_task_status(
        [
            (Path("reports/data_quality/current.json"), latest),
            (Path("reports/data_quality/primary_market_task_runs_v1_2_1/latest.json"), latest),
            (Path("reports/data_quality/primary_market_task_runs_v1_2_1/old.json"), old),
        ]
    )
    assert selected["receipt_id"] == "latest"
    assert "primary_market_task_runs_v1_2_1" in selected_path.as_posix()


def test_v1_7_contract_matches_final_resource_allocation() -> None:
    config = load_config(CONFIG_PATH)
    contract = config["contract"]
    assert contract["original_90_day_decision"] == "RUN_ONE_NEW_FROZEN_TEST"
    assert contract["decision"] == "STOP_NO_HISTORICAL_RETURN_EVALUATION"
    assert contract["historical_return_attempts_allowed"] == 0
    assert contract["only_new_frozen_test"] == "A_SHARE_HS_CSI300_OFFICIAL_ADDITION_FORCED_DEMAND_V1"
    assert [item["resource_percent"] for item in contract["active_research_streams"]] == [70, 30]
    assert config["execution_authorization"]["live_trading_enabled"] is False


def test_current_report_uses_v1_2_1_external_failure_without_new_quality_day() -> None:
    config = load_config(CONFIG_PATH)
    report = build_report(
        config,
        datetime(2026, 8, 29, 3, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert report["overall_research_status"] == "NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET"
    assert report["original_90_day_decision"] == "RUN_ONE_NEW_FROZEN_TEST"
    assert report["decision"] == "STOP_NO_HISTORICAL_RETURN_EVALUATION"
    assert report["frozen_v1_8_manifest"]["hash_verified"] is True
    csi300 = report["active_research_streams"][0]
    assert csi300["status"] == "NO_VIEW_DATA_CONTRACT_FAILED"
    assert csi300["return_evaluation"] == "NOT_ALLOWED"
    assert csi300["failed_data_gates"] == ["D4", "D5"]
    assert csi300["historical_return_attempts_allowed"] == 0
    assert csi300["historical_return_attempts_used"] == 0
    assert csi300["source_rescue_allowed"] is False
    assert csi300["data_admission"]["sha256"]
    assert csi300["protocol_manifest"]["hash_verified"] is True
    pcf = report["active_research_streams"][1]
    assert pcf["full_quality_days"] == 2
    latest = pcf["latest_task_authority"]
    assert latest["path"].startswith("reports/data_quality/primary_market_task_runs_v1_2_1/")
    assert latest["collection_status"] == "EXTERNAL_FREE_SOURCE_FAILED"
    assert latest["task_exit_code"] == 3
    assert latest["quality_day_counted"] is False
    assert latest["raw_response_present"] is False
    assert latest["receipt_evidence"]["sha256"]
    assert latest["log_evidence"]["sha256"]
    assert report["execution_authorization"]["order_generation_enabled"] is False
