from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_priority_forward_morning_task_v1_5.ps1"
DEPLOYMENT_AUDIT = (
    ROOT
    / "reports"
    / "audit"
    / "PRIORITY_FORWARD_WINDOWS_SCHEDULER_V1_5_DEPLOYMENT_20260826.json"
)


def _powershell(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSTALLER),
            *arguments,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_windows_task_installer_is_utf8_bom_and_parses_in_powershell_5_1() -> None:
    assert INSTALLER.read_bytes().startswith(b"\xef\xbb\xbf")
    environment = os.environ.copy()
    environment["CODEX_POWERSHELL_PARSE_TARGET"] = str(INSTALLER)
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


def test_registered_windows_task_uses_v1_5_atomic_entrypoint() -> None:
    result = _powershell("-AuditOnly")
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["state"] == "Ready"
    assert payload["action_execute"].endswith(".venv\\Scripts\\python.exe")
    assert "run_priority_forward_codex_automation_v1_5.py" in payload[
        "action_arguments"
    ]
    assert payload["action_arguments"].endswith("--phase morning")
    assert payload["restart_count"] == 0
    assert payload["atomic_claim_entrypoint"] is True
    assert payload["exact_match"] is True


def test_windows_task_deployment_receipt_file_hashes_are_exact() -> None:
    import hashlib

    payload = json.loads(DEPLOYMENT_AUDIT.read_text(encoding="utf-8"))
    assert payload["deployment_status"] == (
        "PASS_SINGLE_ATOMIC_ENTRYPOINT_ACTIVE_AWAITING_NEXT_WINDOW"
    )
    assert payload["change_type"] == (
        "IN_PLACE_ACTION_REPLACEMENT_NO_DUPLICATE_TASK_CREATED"
    )
    assert payload["after"]["restart_count"] == 0
    assert payload["after"]["atomic_claim_entrypoint"] is True
    assert payload["verification"]["next_real_run_observed"] is False
    for item in payload["files"].values():
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
