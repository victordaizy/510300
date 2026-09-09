from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts/install_priority_forward_morning_task_v1_10_1.ps1"
MANIFEST = ROOT / "config/priority_forward_research_operations_v1_10_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v1_10_1_installer_uses_orchestrator_module_entrypoint() -> None:
    text = INSTALLER.read_text(encoding="utf-8-sig")

    assert INSTALLER.read_bytes().startswith(b"\xef\xbb\xbf")
    assert '$orchestratorModule = "scripts.run_priority_forward_codex_automation_v1_10"' in text
    assert '$arguments = "-m $orchestratorModule --phase morning' in text
    assert "python_module_entrypoint = $true" in text
    assert "WINDOWS_SCHEDULER_ORCHESTRATOR_MODULE_ENTRYPOINT_V1_10_1" in text
    assert "StartWhenAvailable = $false" in text


def test_v1_10_1_exact_module_entrypoint_verifies_without_writes() -> None:
    completed = subprocess.run(
        [
            str(ROOT / ".venv/Scripts/python.exe"),
            "-m",
            "scripts.run_priority_forward_codex_automation_v1_10",
            "--manifest",
            str(MANIFEST),
            "--expected-manifest-sha256",
            sha256_file(MANIFEST),
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
    assert payload["status"] == "PASS_V1_10_FROZEN_ENTRYPOINT_VERIFIED"
    assert payload["failure_count"] == 0
    assert payload["late_backfill_enabled"] is False


def test_v1_10_1_installer_parses_in_windows_powershell() -> None:
    command = (
        "$tokens = $null; $errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{INSTALLER}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { exit 1 }"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
