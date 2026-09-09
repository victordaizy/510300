from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/priority_forward_supervisor_v1_3.yaml"
MANIFEST = ROOT / "config/priority_forward_research_operations_v1_7_manifest.json"
ENTRYPOINT = ROOT / "scripts/run_priority_forward_codex_automation_v1_7.py"
INSTALLER = ROOT / "scripts/install_priority_forward_morning_task_v1_7.ps1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_status_deduplication_uses_only_immutable_task_receipts() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    status_task = next(
        task for task in config["tasks"] if task["id"] == "PRIORITY_FORWARD_STATUS"
    )
    assert status_task["evidence_globs"] == [
        "reports/audit/priority_forward_authoritative_status_task_runs_v1_6/*.json"
    ]


def test_close_dry_run_accepts_current_mutable_status_schema() -> None:
    expected_hash = sha256_file(MANIFEST)
    completed = subprocess.run(
        [
            str(ROOT / ".venv/Scripts/python.exe"),
            str(ENTRYPOINT),
            "--phase",
            "close",
            "--config",
            str(CONFIG),
            "--manifest",
            str(MANIFEST),
            "--expected-manifest-sha256",
            expected_hash,
            "--dry-run",
            "--now",
            "2026-08-26T19:00:00+08:00",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_RUN"
    assert [item["task_id"] for item in payload["task_results"]] == [
        "INDUSTRY_EXPECTATION_GAP",
        "PRIORITY_FORWARD_STATUS",
    ]


def test_v1_7_manifest_and_entrypoint_verify_exactly() -> None:
    expected_hash = sha256_file(MANIFEST)
    completed = subprocess.run(
        [
            str(ROOT / ".venv/Scripts/python.exe"),
            str(ENTRYPOINT),
            "--manifest",
            str(MANIFEST),
            "--expected-manifest-sha256",
            expected_hash,
            "--verify-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS_V1_7_FROZEN_ENTRYPOINT_VERIFIED"
    assert payload["manifest_sha256"] == expected_hash
    assert payload["failure_count"] == 0


def test_v1_7_installer_is_bom_and_powershell_5_parseable() -> None:
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


def test_registered_windows_task_exactly_uses_v1_7() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSTALLER),
            "-AuditOnly",
            "-LogonType",
            "Interactive",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["exact_match"] is True
    assert payload["manifest_sha256"] == sha256_file(MANIFEST)
    assert "run_priority_forward_codex_automation_v1_7.py" in payload[
        "action_arguments"
    ]
    assert "priority_forward_supervisor_v1_3.yaml" in payload["action_arguments"]
    assert payload["restart_count"] == 0
    assert payload["start_when_available"] is False
    assert payload["multiple_instances"] == "IgnoreNew"
    assert payload["immutable_receipt_deduplication"] is True
