from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.freeze_priority_forward_research_operations_v1_6 import (
    MANIFEST_PATH,
    verify_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts/run_priority_forward_codex_automation_v1_6.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_manifest_verifies_and_freezes_zero_paid_two_stream_contract() -> None:
    result = verify_manifest()
    assert result["status"] == "PASS_PRIORITY_FORWARD_V1_6_MANIFEST_VERIFIED"
    assert result["failure_count"] == 0
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "FROZEN_ZERO_PAID_FOUNDATION_ACTIVE_AWAITING_NEXT_WINDOW"
    assert manifest["data_purchase_budget_cny"] == 0
    assert manifest["zero_purchase_lock_days"] >= 180
    assert manifest["maximum_active_research_streams"] == 2
    assert manifest["active_research_streams"] == [
        "PRIMARY_MARKET_PCF_IOPV",
        "INDUSTRY_EXPECTATION_GAP",
    ]
    assert manifest["position_mapping_enabled"] is False
    assert manifest["order_generation_enabled"] is False
    assert manifest["live_trading_enabled"] is False


def test_manifest_tracks_frozen_entrypoint_and_critical_runtime_files() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    paths = {item["path"] for item in manifest["files"]}
    required = {
        "scripts/run_priority_forward_codex_automation_v1_6.py",
        "scripts/run_priority_forward_codex_automation_v1_5.py",
        "config/priority_forward_supervisor_v1_2.yaml",
        "scripts/run_510300_primary_market_collection_task_v1_2.ps1",
        "scripts/run_industry_expectation_gap_forward_operations_task_v1_2.ps1",
        "scripts/run_priority_forward_authoritative_status_task_v1_6.ps1",
        "data/governance/FREE_SOURCE_REGISTRY_V1_1.csv",
        "reports/audit/A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1_FINAL_FREE_SOURCE_REPAIR.json",
        "reports/audit/OPTION_ORDERBOOK_ZERO_COST_SOURCE_QUALIFICATION_V1.json",
    }
    assert required.issubset(paths)


def test_entrypoint_verify_only_accepts_exact_manifest_hash() -> None:
    expected_hash = sha256_file(MANIFEST_PATH)
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        str(ENTRYPOINT),
        "--expected-manifest-sha256",
        expected_hash,
        "--verify-only",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS_V1_6_FROZEN_ENTRYPOINT_VERIFIED"
    assert payload["manifest_sha256"] == expected_hash
    assert payload["failure_count"] == 0


def test_entrypoint_rejects_wrong_manifest_hash() -> None:
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        str(ENTRYPOINT),
        "--expected-manifest-sha256",
        "0" * 64,
        "--verify-only",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "清单文件哈希不匹配" in completed.stderr
