from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/priority_forward_supervisor_v1_5.yaml"
MANIFEST = ROOT / "config/priority_forward_research_operations_v1_9_manifest.json"
V1_8_MANIFEST = ROOT / "config/priority_forward_research_operations_v1_8_manifest.json"
ENTRYPOINT = ROOT / "scripts/run_priority_forward_codex_automation_v1_9.py"
FREEZER = ROOT / "scripts/freeze_priority_forward_research_operations_v1_9.py"
INSTALLER = ROOT / "scripts/install_priority_forward_morning_task_v1_9.ps1"
EXPECTED_PATCH_ID = "PCF_IOPV_STRICT_TLS_ROUTE_V1_3"
EXPECTED_V1_8_SHA256 = (
    "ae8215c5b7e35837564d464eb975054a2d79e2dd33a0cc25636b2c667fdcd1c3"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_entrypoint(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ROOT / ".venv/Scripts/python.exe"), str(ENTRYPOINT), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_v1_9_manifest_and_entrypoint_verify_exactly() -> None:
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
    assert payload["status"] == "PASS_V1_9_FROZEN_ENTRYPOINT_VERIFIED"
    assert payload["manifest_sha256"] == expected_hash
    assert payload["runtime_patch_id"] == EXPECTED_PATCH_ID
    assert payload["tracked_file_count"] == 46
    assert payload["failure_count"] == 0
    assert payload["tls_verification_required"] is True
    assert payload["insecure_tls_allowed"] is False
    assert payload["late_backfill_enabled"] is False


def test_v1_9_freezer_verifies_without_mutating_manifests() -> None:
    before = sha256_file(MANIFEST)
    v1_8_before = sha256_file(V1_8_MANIFEST)
    completed = subprocess.run(
        [
            str(ROOT / ".venv/Scripts/python.exe"),
            str(FREEZER),
            "--mode",
            "verify",
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
    assert payload["status"] == "PASS_PRIORITY_FORWARD_V1_9_MANIFEST_VERIFIED"
    assert payload["manifest_sha256"] == before
    assert payload["runtime_patch_id"] == EXPECTED_PATCH_ID
    assert sha256_file(MANIFEST) == before
    assert v1_8_before == EXPECTED_V1_8_SHA256
    assert sha256_file(V1_8_MANIFEST) == EXPECTED_V1_8_SHA256


def test_v1_9_morning_and_close_dry_runs_write_no_claim_or_receipt() -> None:
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
    assert morning_payload["task_results"] == [
        {
            "task_id": "PRIMARY_MARKET_PCF_IOPV",
            "decision": "RUN_NOW",
            "pre_evidence": [],
            "launcher": "scripts/run_510300_primary_market_collection_task_v1_3.ps1",
            "claim_file": None,
            "claim_acquired": None,
            "exit_code": None,
            "post_evidence": [],
        }
    ]
    assert [item["task_id"] for item in close_payload["task_results"]] == [
        "INDUSTRY_EXPECTATION_GAP",
        "PRIORITY_FORWARD_STATUS",
    ]


def test_v1_9_config_isolated_outputs_and_strict_transport_launcher() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    pcf = next(item for item in config["tasks"] if item["id"] == "PRIMARY_MARKET_PCF_IOPV")

    assert config["version"] == "PRIORITY_FORWARD_SUPERVISOR_V1_5"
    assert config["outputs"]["codex_run_receipt_directory"].endswith("_v1_9")
    assert config["outputs"]["codex_task_claim_directory"].endswith("_v1_9")
    assert pcf["launcher"] == (
        "scripts/run_510300_primary_market_collection_task_v1_3.ps1"
    )
    assert pcf["evidence_globs"] == [
        "reports/data_quality/primary_market_task_runs_v1_3/*.json"
    ]
    assert config["safety"]["position_mapping_enabled"] is False
    assert config["safety"]["order_generation_enabled"] is False
    assert config["safety"]["broker_connection_enabled"] is False
    assert config["safety"]["live_trading_enabled"] is False


def test_v1_9_manifest_preserves_failed_day_and_execution_gates() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["failed_predecessor_attempt"]["target_date"] == "2026-08-28"
    assert manifest["failed_predecessor_attempt"]["quality_day_created"] is False
    contract = manifest["deduplication_evidence_contract"]
    assert contract["late_backfill_enabled"] is False
    assert contract["rerun_2026_08_28_enabled"] is False
    assert manifest["source_endpoints_changed"] is False
    assert manifest["maturity_thresholds_changed"] is False
    assert manifest["position_mapping_enabled"] is False
    assert manifest["order_generation_enabled"] is False
    assert manifest["broker_connection_enabled"] is False
    assert manifest["live_trading_enabled"] is False


def test_v1_9_installer_targets_frozen_entrypoint_without_registering() -> None:
    text = INSTALLER.read_text(encoding="utf-8-sig")

    assert INSTALLER.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "run_priority_forward_codex_automation_v1_9.py" in text
    assert "priority_forward_supervisor_v1_5.yaml" in text
    assert "priority_forward_research_operations_v1_9_manifest.json" in text
    assert EXPECTED_PATCH_ID in text
    assert "StartWhenAvailable = $false" in text
    assert "Register-ScheduledTask" in text
