from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_priority_forward_morning_task_v1_6.ps1"
MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_6_manifest.json"


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


def test_v1_6_installer_is_bom_and_powershell_5_parseable() -> None:
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


def test_v1_6_installer_reports_current_v1_7_task_as_superseded() -> None:
    result = _powershell("-AuditOnly", "-LogonType", "Interactive")
    assert result.returncode == 2, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["action_execute"].endswith(".venv\\Scripts\\python.exe")
    assert "run_priority_forward_codex_automation_v1_7.py" in payload[
        "action_arguments"
    ]
    assert "--phase morning" in payload["action_arguments"]
    assert "priority_forward_supervisor_v1_3.yaml" in payload["action_arguments"]
    assert "priority_forward_research_operations_v1_7_manifest.json" in payload[
        "action_arguments"
    ]
    assert payload["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert payload["restart_count"] == 0
    assert payload["start_when_available"] is False
    assert payload["principal_logon_type"] == "Interactive"
    assert payload["persistence_capability"] == "LOGGED_IN_SESSION_ONLY"
    assert payload["logout_reboot_verified"] is False
    assert payload["atomic_claim_entrypoint"] is True
    assert payload["frozen_manifest_entrypoint"] is True
    assert payload["exact_match"] is False
