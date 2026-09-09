from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts.run_priority_forward_codex_automation_v1_5 import (
    claim_task_attempt,
    run_phase,
)


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")
RUNNER = ROOT / "scripts" / "run_510300_primary_market_forward_v1_1.ps1"
TASK_WRAPPER = (
    ROOT / "scripts" / "run_510300_primary_market_collection_task_v1_1.ps1"
)


def _powershell_5_parse(path: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CODEX_POWERSHELL_PARSE_TARGET"] = str(path)
    command = (
        "$tokens = $null; $errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "$env:CODEX_POWERSHELL_PARSE_TARGET, [ref]$tokens, [ref]$errors) "
        "| Out-Null; if ($errors.Count -gt 0) { "
        "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("path", [RUNNER, TASK_WRAPPER])
def test_powershell_5_1_entrypoints_are_utf8_bom_and_parse(path: Path) -> None:
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    result = _powershell_5_parse(path)
    assert result.returncode == 0, result.stderr


def _claim_in_subprocess(claim_directory: Path, start_gate: Path) -> subprocess.Popen[str]:
    code = (
        "from datetime import date, datetime; "
        "from pathlib import Path; "
        "from zoneinfo import ZoneInfo; "
        "import time; "
        "from scripts.run_priority_forward_codex_automation_v1_5 "
        "import claim_task_attempt; "
        f"gate=Path({str(start_gate)!r}); "
        "\nwhile not gate.exists(): time.sleep(0.01)\n"
        "path, acquired = claim_task_attempt("
        f"Path({str(claim_directory)!r}), "
        "'PRIMARY_MARKET_PCF_IOPV', date(2026, 8, 26), "
        "datetime(2026, 8, 26, 9, 25, tzinfo=ZoneInfo('Asia/Shanghai')), "
        "'morning'); "
        "print(json.dumps({'path': str(path), 'acquired': acquired}))"
    )
    return subprocess.Popen(
        [sys.executable, "-c", "import json; " + code],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_same_day_claim_is_atomic_across_processes(tmp_path: Path) -> None:
    claim_directory = tmp_path / "claims"
    start_gate = tmp_path / "start.gate"
    processes = [
        _claim_in_subprocess(claim_directory, start_gate),
        _claim_in_subprocess(claim_directory, start_gate),
    ]
    start_gate.write_text("开始\n", encoding="utf-8")
    results: list[dict[str, object]] = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
        results.append(json.loads(stdout))
    assert sorted(bool(item["acquired"]) for item in results) == [False, True]
    claims = list(claim_directory.glob("*.json"))
    assert len(claims) == 1
    payload = json.loads(claims[0].read_text(encoding="utf-8"))
    assert payload["task_id"] == "PRIMARY_MARKET_PCF_IOPV"
    assert payload["target_date"] == "2026-08-26"
    assert payload["claim_status"] == "ATOMIC_ATTEMPT_CLAIMED"


def _isolated_config(root: Path) -> dict[str, object]:
    launcher = root / "launcher.ps1"
    launcher.write_text("exit 0\n", encoding="utf-8")
    return {
        "version": "TEST_PRIORITY_FORWARD_V1_5",
        "timezone": "Asia/Shanghai",
        "outputs": {
            "codex_run_receipt_directory": "reports/orchestration",
            "codex_log_directory": "output/logs",
            "codex_current_status": "reports/current.json",
            "codex_task_claim_directory": "reports/claims",
        },
        "tasks": [
            {
                "id": "PRIMARY_MARKET_PCF_IOPV",
                "schedule": "09:25",
                "latest_start": "09:35",
                "weekdays": [0, 1, 2, 3, 4],
                "launcher": "launcher.ps1",
                "evidence_globs": ["reports/task/*.json"],
            },
            {
                "id": "INDUSTRY_EXPECTATION_GAP",
                "schedule": "17:05",
                "latest_start": "23:30",
                "weekdays": [0, 1, 2, 3, 4],
                "launcher": "launcher.ps1",
                "evidence_globs": ["reports/industry/*.json"],
            },
            {
                "id": "PRIORITY_FORWARD_STATUS",
                "schedule": "17:15",
                "latest_start": "23:50",
                "weekdays": [0, 1, 2, 3, 4],
                "launcher": "launcher.ps1",
                "evidence_globs": ["reports/status/*.json"],
            },
        ],
        "safety": {
            "research_only": True,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
        },
    }


def test_run_phase_race_launches_primary_market_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_priority_forward_codex_automation_v1_5 as module

    config = _isolated_config(tmp_path)
    invocation_count = 0

    def fake_run(task: object, root: Path, log_handle: object) -> int:
        nonlocal invocation_count
        invocation_count += 1
        time.sleep(0.4)
        receipt_directory = root / "reports" / "task"
        receipt_directory.mkdir(parents=True, exist_ok=True)
        receipt = {
            "started_at": "2026-08-26T09:25:00+08:00",
            "research_only": True,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
        }
        (receipt_directory / "attempt.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(module, "_run_powershell", fake_run)
    monkeypatch.setattr(module, "_run_renderer", lambda root, handle: 0)
    now = datetime(2026, 8, 26, 9, 25, tzinfo=TIMEZONE)

    def invoke() -> tuple[dict[str, object], int]:
        return run_phase(
            config,
            "morning",
            now,
            root=tmp_path,
            dry_run=False,
            wait_for_window=False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: invoke(), range(2)))

    decisions = sorted(
        str(payload["task_results"][0]["decision"])
        for payload, _ in outcomes
    )
    assert decisions == ["ALREADY_CLAIMED_ATOMIC", "RUN_NOW"]
    assert invocation_count == 1
    assert len(list((tmp_path / "reports" / "claims").glob("*.json"))) == 1
    assert all(exit_code == 0 for _, exit_code in outcomes)


def test_claim_name_is_stable_and_rejects_invalid_task_id(tmp_path: Path) -> None:
    now = datetime(2026, 8, 26, 9, 25, tzinfo=TIMEZONE)
    first_path, first_acquired = claim_task_attempt(
        tmp_path, "PRIMARY_MARKET_PCF_IOPV", date(2026, 8, 26), now, "morning"
    )
    second_path, second_acquired = claim_task_attempt(
        tmp_path, "PRIMARY_MARKET_PCF_IOPV", date(2026, 8, 26), now, "morning"
    )
    assert first_path == second_path
    assert first_acquired is True
    assert second_acquired is False
    with pytest.raises(ValueError, match="任务ID"):
        claim_task_attempt(tmp_path, "../越界", date(2026, 8, 26), now, "morning")


def test_dry_run_is_read_only(tmp_path: Path) -> None:
    config = _isolated_config(tmp_path)
    now = datetime(2026, 8, 27, 9, 25, tzinfo=TIMEZONE)
    payload, exit_code = run_phase(
        config,
        "morning",
        now,
        root=tmp_path,
        dry_run=True,
        wait_for_window=False,
    )
    assert exit_code == 0
    assert payload["status"] == "DRY_RUN"
    assert payload["immutable_receipt"] is False
    assert payload["receipt_written"] is False
    assert payload["log_written"] is False
    assert payload["task_results"][0]["decision"] == "RUN_NOW"
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "output").exists()
