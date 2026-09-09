from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from scripts import render_priority_forward_authoritative_status_v1_8_1 as authority


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/priority_forward_supervisor_v1_6.yaml"
MANIFEST = ROOT / "config/priority_forward_research_operations_v1_10_manifest.json"
V1_9_MANIFEST = ROOT / "config/priority_forward_research_operations_v1_9_manifest.json"
ENTRYPOINT = ROOT / "scripts/run_priority_forward_codex_automation_v1_10.py"
FREEZER = ROOT / "scripts/freeze_priority_forward_research_operations_v1_10.py"
INSTALLER = ROOT / "scripts/install_priority_forward_morning_task_v1_10.ps1"
STATUS_TASK = ROOT / "scripts/run_priority_forward_authoritative_status_task_v1_8_1.ps1"
EXPECTED_PATCH_ID = (
    "PCF_IOPV_STRICT_TLS_ROUTE_V1_3_AND_STATUS_MODULE_ENTRYPOINT_V1_8_1"
)
EXPECTED_V1_9_SHA256 = (
    "fb1cdb90c066d002e0283a73ac483702a356bd77cd24560f91e7172aeb3d2b62"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_entrypoint(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ROOT / ".venv/Scripts/python.exe"), "-m", "scripts.run_priority_forward_codex_automation_v1_10", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_v1_10_manifest_and_entrypoint_verify_exactly() -> None:
    expected_hash = sha256_file(MANIFEST)
    completed = run_entrypoint(
        "--manifest",
        str(MANIFEST),
        "--expected-manifest-sha256",
        expected_hash,
        "--verify-only",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS_V1_10_FROZEN_ENTRYPOINT_VERIFIED"
    assert payload["manifest_sha256"] == expected_hash
    assert payload["runtime_patch_id"] == EXPECTED_PATCH_ID
    assert payload["tracked_file_count"] == 49
    assert payload["failure_count"] == 0
    assert payload["tls_verification_required"] is True
    assert payload["insecure_tls_allowed"] is False
    assert payload["late_backfill_enabled"] is False


def test_v1_10_freezer_verifies_without_mutating_predecessor() -> None:
    before = sha256_file(MANIFEST)
    predecessor_before = sha256_file(V1_9_MANIFEST)
    completed = subprocess.run(
        [str(ROOT / ".venv/Scripts/python.exe"), str(FREEZER), "--mode", "verify"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS_PRIORITY_FORWARD_V1_10_MANIFEST_VERIFIED"
    assert payload["manifest_sha256"] == before
    assert sha256_file(MANIFEST) == before
    assert predecessor_before == EXPECTED_V1_9_SHA256
    assert sha256_file(V1_9_MANIFEST) == EXPECTED_V1_9_SHA256


def test_v1_10_morning_and_close_dry_runs_are_write_free() -> None:
    expected_hash = sha256_file(MANIFEST)
    common = (
        "--config",
        str(CONFIG),
        "--manifest",
        str(MANIFEST),
        "--expected-manifest-sha256",
        expected_hash,
        "--dry-run",
    )
    morning = run_entrypoint(
        "--phase",
        "morning",
        *common,
        "--now",
        "2026-08-31T09:25:00+08:00",
    )
    close = run_entrypoint(
        "--phase",
        "close",
        *common,
        "--now",
        "2026-08-31T19:00:00+08:00",
    )

    assert morning.returncode == 0, morning.stderr
    assert close.returncode == 0, close.stderr
    morning_payload = json.loads(morning.stdout)
    close_payload = json.loads(close.stdout)
    assert morning_payload["receipt_written"] is False
    assert morning_payload["log_written"] is False
    assert morning_payload["task_results"][0]["launcher"] == (
        "scripts/run_510300_primary_market_collection_task_v1_3.ps1"
    )
    assert [item["launcher"] for item in close_payload["task_results"]] == [
        "scripts/run_industry_expectation_gap_forward_operations_task_v1_2.ps1",
        "scripts/run_priority_forward_authoritative_status_task_v1_8_1.ps1",
    ]


def test_v1_10_authority_status_uses_module_entrypoint_and_manifest() -> None:
    config = authority.load_config()
    report = authority.build_report(
        config,
        datetime(2026, 8, 29, 4, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert report["schema_version"] == "1.8.1"
    assert report["frozen_priority_forward_manifest"]["manifest_id"] == (
        "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_10"
    )
    assert report["frozen_priority_forward_manifest"]["verification_status"] == (
        "PASS_V1_10_FROZEN_ENTRYPOINT_VERIFIED"
    )
    assert report["active_research_streams"][1]["deployment_status"] == (
        "DEPLOYED_AWAITING_FIRST_ELIGIBLE_WINDOW"
    )
    assert report["active_research_streams"][1]["full_quality_days"] == 2
    assert report["active_research_streams"][1]["latest_task_authority"]["quality_day_counted"] is False


def test_v1_10_supervisor_and_installer_bind_exact_new_runtime() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    status_task = next(item for item in config["tasks"] if item["id"] == "PRIORITY_FORWARD_STATUS")
    installer = INSTALLER.read_text(encoding="utf-8-sig")
    task_text = STATUS_TASK.read_text(encoding="utf-8-sig")

    assert config["version"] == "PRIORITY_FORWARD_SUPERVISOR_V1_6"
    assert config["outputs"]["codex_task_claim_directory"].endswith("_v1_10")
    assert status_task["launcher"].endswith("v1_8_1.ps1")
    assert 'runnerModule = "scripts.render_priority_forward_authoritative_status_v1_8_1"' in task_text
    assert "& $python -m $runnerModule --config $configFile" in task_text
    assert "run_priority_forward_codex_automation_v1_10.py" in installer
    assert "priority_forward_supervisor_v1_6.yaml" in installer
    assert "priority_forward_research_operations_v1_10_manifest.json" in installer
    assert EXPECTED_PATCH_ID in installer
    assert "StartWhenAvailable = $false" in installer


def test_v1_10_powershell_files_are_bom_and_parseable() -> None:
    for path in (STATUS_TASK, INSTALLER):
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")
        command = (
            "$tokens = $null; $errors = $null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{path}', [ref]$tokens, [ref]$errors) | Out-Null; "
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
