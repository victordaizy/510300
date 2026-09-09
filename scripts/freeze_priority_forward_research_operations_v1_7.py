"""生成并核验 V1.7 不可变回执去重运行清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from scripts.freeze_priority_forward_research_operations_v1_6 import (
    ROOT,
    create_json_exclusive,
    file_records,
    relative,
    sha256_file,
    verify_manifest as verify_v1_6_manifest,
)


BASE_MANIFEST_PATH = ROOT / "config/priority_forward_research_operations_v1_6_manifest.json"
MANIFEST_PATH = ROOT / "config/priority_forward_research_operations_v1_7_manifest.json"
ENTRYPOINT = ROOT / "scripts/run_priority_forward_codex_automation_v1_7.py"
REPLACED_BASE_PATHS = {
    "config/priority_forward_supervisor_v1_2.yaml",
    "scripts/run_priority_forward_codex_automation_v1_6.py",
}
V1_7_PATHS = {
    "config/priority_forward_supervisor_v1_3.yaml",
    "scripts/run_priority_forward_codex_automation_v1_7.py",
    "scripts/install_priority_forward_morning_task_v1_7.ps1",
}


def tracked_paths() -> list[Path]:
    verify_v1_6_manifest()
    base = json.loads(BASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    base_files = base.get("files")
    if not isinstance(base_files, list) or not base_files:
        raise RuntimeError("V1.6 基础清单没有受控文件")
    relative_paths = {
        str(item["path"])
        for item in base_files
        if isinstance(item, dict) and str(item.get("path")) not in REPLACED_BASE_PATHS
    }
    relative_paths.update(V1_7_PATHS)
    paths = [ROOT / path for path in sorted(relative_paths)]
    missing = [relative(path) if path.exists() else str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"V1.7 清单种子文件缺失：{missing}")
    return paths


def build_manifest() -> dict[str, Any]:
    records = file_records(tracked_paths())
    content_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.7.0",
        "manifest_id": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_7",
        "status": "FROZEN_ZERO_PAID_FOUNDATION_V1_7_ACTIVE",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "change_class": "OPERATIONAL_EVIDENCE_ADMISSION_CORRECTION",
        "supersedes_runtime_manifest": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_6",
        "research_protocol_changed": False,
        "model_or_candidate_changed": False,
        "historical_signal_changed": False,
        "data_purchase_budget_cny": 0,
        "zero_purchase_lock_days": 180,
        "maximum_active_research_streams": 2,
        "active_research_streams": [
            "PRIMARY_MARKET_PCF_IOPV",
            "INDUSTRY_EXPECTATION_GAP",
        ],
        "execution_entrypoint": relative(ENTRYPOINT),
        "supervisor_config": "config/priority_forward_supervisor_v1_3.yaml",
        "deduplication_evidence_contract": {
            "status": "IMMUTABLE_TASK_RECEIPTS_ONLY",
            "mutable_aggregate_status_is_attempt_evidence": False,
            "same_day_atomic_claim_enabled": True,
            "late_backfill_enabled": False,
        },
        "files": records,
        "content_sha256": content_sha256,
        "mutable_outputs_not_hashed": [
            "reports/data_quality/510300_primary_market_readiness_v1_2.json",
            "reports/forward/industry_expectation_gap_v1_evaluation/operations_status_v1_2.json",
            "reports/audit/priority_forward_authoritative_status_v1_6.json",
            "reports/audit/priority_forward_*_runs_v1_6/*.json",
            "reports/audit/priority_forward_*_runs_v1_7/*.json",
            "reports/audit/priority_forward_task_claims_v1_7/*.json",
            "output/**",
        ],
        "terminal_branches": {
            "cash_option_floor": "NO_VIEW_FREE_DATA_INSUFFICIENT",
            "option_orderbook": "BLOCKED_ZERO_COST_SOURCE_UNAVAILABLE",
        },
        "windows_logout_reboot_persistence": "BLOCKED_PERMISSION_S4U_REGISTRATION_ACCESS_DENIED",
        "research_only": True,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }


def verify_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError("V1.7 运行清单不存在")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = file_records(tracked_paths())
    failures: list[dict[str, Any]] = []
    if manifest.get("files") != expected:
        failures.append({"failure": "TRACKED_FILE_RECORDS"})
    content_sha256 = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("content_sha256") != content_sha256:
        failures.append(
            {
                "failure": "CONTENT_SHA256",
                "expected": manifest.get("content_sha256"),
                "actual": content_sha256,
            }
        )
    if manifest.get("data_purchase_budget_cny") != 0:
        failures.append({"failure": "DATA_PURCHASE_BUDGET"})
    if manifest.get("maximum_active_research_streams") != 2:
        failures.append({"failure": "ACTIVE_STREAM_CAP"})
    contract = manifest.get("deduplication_evidence_contract", {})
    if contract.get("status") != "IMMUTABLE_TASK_RECEIPTS_ONLY":
        failures.append({"failure": "DEDUPLICATION_EVIDENCE_CONTRACT"})
    result = {
        "status": (
            "PASS_PRIORITY_FORWARD_V1_7_MANIFEST_VERIFIED"
            if not failures
            else "FAILED_PRIORITY_FORWARD_V1_7_MANIFEST_VERIFICATION"
        ),
        "failure_count": len(failures),
        "failures": failures,
        "manifest_path": relative(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "content_sha256": content_sha256,
        "tracked_file_count": len(expected),
    }
    if failures:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def freeze_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return {**verify_manifest(), "manifest_reused": True}
    manifest = build_manifest()
    create_json_exclusive(MANIFEST_PATH, manifest)
    return {**verify_manifest(), "manifest_reused": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结 V1.7 不可变回执去重运行清单")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    arguments = parser.parse_args()
    result = freeze_manifest() if arguments.mode == "freeze" else verify_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
