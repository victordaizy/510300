from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from scripts import render_priority_forward_authoritative_status_v1_8 as status


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/priority_forward_authoritative_status_v1_8.yaml"
TASK_RUNNER = ROOT / "scripts/run_priority_forward_authoritative_status_task_v1_8.ps1"


def test_v1_8_status_config_keeps_terminal_csi_and_all_execution_gates_closed() -> None:
    config = status.load_config(CONFIG)

    assert config["contract"]["decision"] == "STOP_NO_HISTORICAL_RETURN_EVALUATION"
    assert config["contract"]["historical_return_attempts_allowed"] == 0
    assert config["inputs"]["priority_forward_manifest"].endswith("v1_9_manifest.json")
    assert config["inputs"]["expected_runtime_patch_id"] == (
        "PCF_IOPV_STRICT_TLS_ROUTE_V1_3"
    )
    authorization = config["execution_authorization"]
    assert authorization == {
        "research_only": True,
        "shadow_enabled": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }


def test_v1_8_status_selects_latest_immutable_direct_receipt() -> None:
    config = status.load_config(CONFIG)

    path, payload = status.select_latest_direct_task_receipt(
        config["inputs"]["primary_market_task_receipt_globs"]
    )

    assert path.name == "20260828T0125025110922Z_20016.json"
    assert payload["immutable_receipt"] is True
    assert payload["collection_status"] == "EXTERNAL_FREE_SOURCE_FAILED"
    assert payload["task_exit_code"] == 3


def test_v1_8_status_report_verifies_v1_9_manifest_and_counts_no_probe_day() -> None:
    config = status.load_config(CONFIG)

    report = status.build_report(
        config,
        datetime(2026, 8, 29, 4, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    csi, pcf = report["active_research_streams"]
    assert csi["status"] == "NO_VIEW_DATA_CONTRACT_FAILED"
    assert csi["return_evaluation"] == "NOT_ALLOWED"
    assert pcf["deployment_status"] == "DEPLOYED_AWAITING_FIRST_ELIGIBLE_WINDOW"
    assert pcf["full_quality_days"] == 2
    assert pcf["latest_task_authority"]["quality_day_counted"] is False
    assert report["frozen_priority_forward_manifest"]["verification_status"] == (
        "PASS_V1_9_FROZEN_ENTRYPOINT_VERIFIED"
    )
    assert report["status_semantics"]["read_only_probe"] == "NOT_A_QUALITY_DAY"
    assert all(
        value is False
        for key, value in report["execution_authorization"].items()
        if key != "research_only"
    )


def test_v1_8_status_task_is_utf8_bom_and_windows_powershell_parseable() -> None:
    assert TASK_RUNNER.read_bytes().startswith(b"\xef\xbb\xbf")
    command = (
        "$tokens = $null; $errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{TASK_RUNNER}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { "
        "$errors | ForEach-Object { [Console]::Error.WriteLine($_.Message) }; "
        "exit 1 }"
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_v1_9_supervisor_routes_close_status_to_v1_8_task() -> None:
    supervisor = yaml.safe_load(
        (ROOT / "config/priority_forward_supervisor_v1_5.yaml").read_text(
            encoding="utf-8"
        )
    )
    task = next(item for item in supervisor["tasks"] if item["id"] == "PRIORITY_FORWARD_STATUS")

    assert task["launcher"] == (
        "scripts/run_priority_forward_authoritative_status_task_v1_8.ps1"
    )
    assert task["evidence_globs"] == [
        "reports/audit/priority_forward_authoritative_status_task_runs_v1_8/*.json"
    ]
