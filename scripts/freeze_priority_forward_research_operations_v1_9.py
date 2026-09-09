"""生成并核验 V1.9 严格 TLS 路线修复运行清单。"""

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
)
from scripts.freeze_priority_forward_research_operations_v1_8 import (
    verify_manifest as verify_v1_8_manifest,
)


BASE_MANIFEST_PATH = ROOT / "config/priority_forward_research_operations_v1_8_manifest.json"
MANIFEST_PATH = ROOT / "config/priority_forward_research_operations_v1_9_manifest.json"
ENTRYPOINT = ROOT / "scripts/run_priority_forward_codex_automation_v1_9.py"
RUNTIME_PATCH_ID = "PCF_IOPV_STRICT_TLS_ROUTE_V1_3"
EFFECTIVE_FROM = "2026-08-31T09:20:00+08:00"
REPLACED_BASE_PATHS = {
    "config/priority_forward_authoritative_status_v1_6.yaml",
    "config/priority_forward_supervisor_v1_4.yaml",
    "scripts/render_priority_forward_authoritative_status_v1_6.py",
    "scripts/install_priority_forward_morning_task_v1_8.ps1",
    "scripts/run_510300_primary_market_collection_task_v1_2_1.ps1",
    "scripts/run_510300_primary_market_forward_v1_2_1.ps1",
    "scripts/run_priority_forward_authoritative_status_task_v1_6.ps1",
    "scripts/run_priority_forward_codex_automation_v1_8.py",
}
V1_9_PATHS = {
    "config/primary_market_forward_v1_3.yaml",
    "config/priority_forward_authoritative_status_v1_8.yaml",
    "config/priority_forward_supervisor_v1_5.yaml",
    "market_data/etf_primary_market_strict_v1_3.py",
    "market_data/sse_strict_transport_v1_3.py",
    "reports/audit/PRIORITY_FORWARD_V1_9_STRICT_TLS_ROUTE_REGISTRATION.json",
    "scripts/collect_510300_primary_market_v1_3.py",
    "scripts/install_priority_forward_morning_task_v1_9.ps1",
    "scripts/render_priority_forward_authoritative_status_v1_8.py",
    "scripts/run_510300_primary_market_collection_task_v1_3.ps1",
    "scripts/run_510300_primary_market_forward_v1_3.ps1",
    "scripts/run_priority_forward_authoritative_status_task_v1_8.ps1",
    "scripts/run_priority_forward_codex_automation_v1_9.py",
}


def tracked_paths() -> list[Path]:
    verify_v1_8_manifest()
    base = json.loads(BASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    base_files = base.get("files")
    if not isinstance(base_files, list) or not base_files:
        raise RuntimeError("V1.8 基础清单没有受控文件")
    relative_paths = {
        str(item["path"])
        for item in base_files
        if isinstance(item, dict) and str(item.get("path")) not in REPLACED_BASE_PATHS
    }
    relative_paths.update(V1_9_PATHS)
    paths = [ROOT / path for path in sorted(relative_paths)]
    missing = [
        relative(path) if path.exists() else str(path)
        for path in paths
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"V1.9 清单种子文件缺失：{missing}")
    return paths


def build_manifest() -> dict[str, Any]:
    records = file_records(tracked_paths())
    content_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.9.0",
        "manifest_id": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_9",
        "status": (
            "FROZEN_ZERO_PAID_FOUNDATION_V1_9_STRICT_TLS_ROUTE_"
            "ACTIVE_NEXT_ELIGIBLE_WINDOW"
        ),
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "effective_from": EFFECTIVE_FROM,
        "change_class": (
            "OPERATIONAL_STRICT_TLS_ROUTE_AND_EXISTING_OHLC_TOLERANCE_ALIGNMENT"
        ),
        "runtime_patch_id": RUNTIME_PATCH_ID,
        "supersedes_runtime_manifest": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_8",
        "failed_predecessor_attempt": {
            "target_date": "2026-08-28",
            "status": "EXTERNAL_FREE_SOURCE_FAILED",
            "task_exit_code": 3,
            "same_day_retry_after_terminal_receipt": False,
            "same_day_backfill_performed": False,
            "quality_day_created": False,
            "evidence": (
                "reports/audit/"
                "PRIORITY_FORWARD_V1_9_STRICT_TLS_ROUTE_REGISTRATION.json"
            ),
        },
        "research_protocol_changed": False,
        "model_or_candidate_changed": False,
        "historical_signal_changed": False,
        "source_contract_changed": False,
        "source_endpoints_changed": False,
        "parser_implementation_changed": True,
        "transport_contract_changed": True,
        "maturity_thresholds_changed": False,
        "data_purchase_budget_cny": 0,
        "zero_purchase_lock_days": 180,
        "maximum_active_research_streams": 2,
        "active_research_streams": [
            "PRIMARY_MARKET_PCF_IOPV",
            "INDUSTRY_EXPECTATION_GAP",
        ],
        "execution_entrypoint": relative(ENTRYPOINT),
        "supervisor_config": "config/priority_forward_supervisor_v1_5.yaml",
        "pcf_runtime": {
            "task_launcher": (
                "scripts/run_510300_primary_market_collection_task_v1_3.ps1"
            ),
            "forward_launcher": (
                "scripts/run_510300_primary_market_forward_v1_3.ps1"
            ),
            "collector_invocation": (
                "python -m scripts.collect_510300_primary_market_v1_3"
            ),
            "analyzer_invocation": (
                "python -m scripts.analyze_510300_primary_market_readiness_v1_2"
            ),
            "collector_config": "config/primary_market_forward_v1_3.yaml",
            "task_receipt_schema_version": "1.3.0",
            "transport_contract": "SSE_STRICT_TRANSPORT_CONTRACT_V1",
            "tls_verification_required": True,
            "insecure_tls_allowed": False,
            "certificate_failure_direct_fallback_limit": 1,
            "daily_ohlc_relation_tolerance_cny": 0.005,
        },
        "deduplication_evidence_contract": {
            "status": "IMMUTABLE_TASK_RECEIPTS_ONLY",
            "mutable_aggregate_status_is_attempt_evidence": False,
            "same_day_atomic_claim_enabled": True,
            "late_backfill_enabled": False,
            "rerun_2026_08_28_enabled": False,
        },
        "first_eligible_scheduled_window": "2026-08-31T09:25:00+08:00",
        "deployment_status_until_first_run": (
            "DEPLOYED_AWAITING_FIRST_ELIGIBLE_WINDOW"
        ),
        "files": records,
        "content_sha256": content_sha256,
        "mutable_outputs_not_hashed": [
            "data/raw/primary_market_v1_2/**",
            "reports/data_quality/510300_primary_market_*_v1_2.json",
            "reports/data_quality/510300_primary_market_*_v1_3.json",
            "reports/data_quality/free_source_acquisitions/primary_market_v1_3/*.json",
            "reports/data_quality/primary_market_task_runs_v1_3/*.json",
            "reports/forward/industry_expectation_gap_v1_evaluation/operations_status_v1_2.json",
            "reports/audit/priority_forward_authoritative_status_v1_8.json",
            "reports/audit/priority_forward_*_runs_v1_8/*.json",
            "reports/audit/priority_forward_*_runs_v1_9/*.json",
            "reports/audit/priority_forward_task_claims_v1_9/*.json",
            "output/**",
        ],
        "terminal_branches": {
            "cash_option_floor": "NO_VIEW_FREE_DATA_INSUFFICIENT",
            "option_orderbook": "BLOCKED_ZERO_COST_SOURCE_UNAVAILABLE",
        },
        "windows_logout_reboot_persistence": (
            "BLOCKED_PERMISSION_S4U_REGISTRATION_ACCESS_DENIED"
        ),
        "research_only": True,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }


def verify_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError("V1.9 运行清单不存在")
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
    exact_fields = {
        "status": (
            "FROZEN_ZERO_PAID_FOUNDATION_V1_9_STRICT_TLS_ROUTE_"
            "ACTIVE_NEXT_ELIGIBLE_WINDOW"
        ),
        "runtime_patch_id": RUNTIME_PATCH_ID,
        "effective_from": EFFECTIVE_FROM,
        "data_purchase_budget_cny": 0,
        "maximum_active_research_streams": 2,
        "research_protocol_changed": False,
        "model_or_candidate_changed": False,
        "historical_signal_changed": False,
        "source_endpoints_changed": False,
        "transport_contract_changed": True,
        "maturity_thresholds_changed": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }
    for field, value in exact_fields.items():
        if manifest.get(field) != value:
            failures.append({"failure": field.upper()})
    contract = manifest.get("deduplication_evidence_contract", {})
    if contract.get("status") != "IMMUTABLE_TASK_RECEIPTS_ONLY":
        failures.append({"failure": "DEDUPLICATION_EVIDENCE_CONTRACT"})
    if contract.get("late_backfill_enabled") is not False:
        failures.append({"failure": "LATE_BACKFILL_CONTRACT"})
    if contract.get("rerun_2026_08_28_enabled") is not False:
        failures.append({"failure": "FAILED_DAY_RERUN_CONTRACT"})
    pcf_runtime = manifest.get("pcf_runtime", {})
    runtime_fields = {
        "task_receipt_schema_version": "1.3.0",
        "transport_contract": "SSE_STRICT_TRANSPORT_CONTRACT_V1",
        "tls_verification_required": True,
        "insecure_tls_allowed": False,
        "certificate_failure_direct_fallback_limit": 1,
        "daily_ohlc_relation_tolerance_cny": 0.005,
    }
    for field, value in runtime_fields.items():
        if pcf_runtime.get(field) != value:
            failures.append({"failure": f"PCF_RUNTIME_{field.upper()}"})
    result = {
        "status": (
            "PASS_PRIORITY_FORWARD_V1_9_MANIFEST_VERIFIED"
            if not failures
            else "FAILED_PRIORITY_FORWARD_V1_9_MANIFEST_VERIFICATION"
        ),
        "failure_count": len(failures),
        "failures": failures,
        "manifest_path": relative(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "content_sha256": content_sha256,
        "tracked_file_count": len(expected),
        "runtime_patch_id": RUNTIME_PATCH_ID,
        "effective_from": manifest.get("effective_from"),
        "late_backfill_enabled": False,
        "research_only": True,
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
    parser = argparse.ArgumentParser(description="冻结 V1.9 严格 TLS 路线修复清单")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    arguments = parser.parse_args()
    result = freeze_manifest() if arguments.mode == "freeze" else verify_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
