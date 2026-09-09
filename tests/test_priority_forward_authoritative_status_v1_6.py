from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from scripts.render_priority_forward_authoritative_status_v1_6 import (
    build_report,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "priority_forward_authoritative_status_v1_6.yaml"
TASK_WRAPPER = (
    ROOT / "scripts" / "run_priority_forward_authoritative_status_task_v1_6.ps1"
)


def test_contract_freezes_zero_budget_and_two_active_streams() -> None:
    config = load_config(CONFIG_PATH)
    contract = config["contract"]
    assert contract["overall_research_status"] == (
        "NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET"
    )
    assert contract["decision"] == "PAUSE_AND_FIX_FOUNDATION"
    assert contract["data_purchase_budget_cny"] == 0
    assert contract["zero_purchase_lock_days"] >= 180
    assert contract["maximum_active_research_streams"] == 2
    assert [item["id"] for item in contract["active_research_streams"]] == [
        "PRIMARY_MARKET_PCF_IOPV",
        "INDUSTRY_EXPECTATION_GAP",
    ]


def test_report_uses_v1_2_readiness_and_admitted_second_endpoint() -> None:
    config = load_config(CONFIG_PATH)
    report = build_report(
        config,
        datetime(2026, 8, 26, 3, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert report["overall_research_status"] == (
        "NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET"
    )
    assert report["decision"] == "PAUSE_AND_FIX_FOUNDATION"
    assert report["contract"]["active_research_stream_count"] == 2
    primary = report["active_research_streams"][0]
    assert primary["id"] == "PRIMARY_MARKET_PCF_IOPV"
    assert primary["quality_day_gate"] == 20
    assert primary["first_unseen_gate"] == 80
    assert primary["replication_gate"] == 120
    second = report["foundation_gates"]["second_official_etf_endpoint"]
    assert second["passed"] is True
    assert second["independent_vendor"] is False
    assert report["free_source_registry"]["nonzero_cost_source_ids"] == []
    cash = report["data_admission_only"]
    assert cash["event_count"] == 21
    assert cash["data_gate_pass_count"] == 0
    assert cash["status"] == "NO_VIEW_FREE_DATA_INSUFFICIENT"
    assert cash["attempt_consumed"] is True
    assert cash["second_repair_allowed"] is False
    assert cash["candidate_closed_for_free_source_repair"] is True
    assert cash["price_values_read"] is False
    assert cash["return_values_read"] is False


def test_supervisor_routes_close_status_to_v1_6() -> None:
    supervisor = yaml.safe_load(
        (ROOT / "config" / "priority_forward_supervisor_v1_2.yaml").read_text(
            encoding="utf-8"
        )
    )
    task = next(
        item for item in supervisor["tasks"] if item["id"] == "PRIORITY_FORWARD_STATUS"
    )
    assert task["launcher"].endswith(
        "run_priority_forward_authoritative_status_task_v1_6.ps1"
    )
    assert supervisor["outputs"]["codex_task_claim_directory"].endswith("_v1_6")


def test_v1_6_status_task_is_bom_and_powershell_5_parseable() -> None:
    assert TASK_WRAPPER.read_bytes().startswith(b"\xef\xbb\xbf")
    environment = os.environ.copy()
    environment["CODEX_POWERSHELL_PARSE_TARGET"] = str(TASK_WRAPPER)
    command = (
        "$tokens = $null; $errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "$env:CODEX_POWERSHELL_PARSE_TARGET, [ref]$tokens, [ref]$errors) "
        "| Out-Null; if ($errors.Count -gt 0) { exit 1 }"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
