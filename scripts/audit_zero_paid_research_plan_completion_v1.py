"""逐要求核验零付费研究计划的当前完成度。

该审计只读取当前工作区、Windows 计划任务和本机 Codex 自动化配置，
不会触发采集、研究计算、补跑或订单流程。
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import tomllib
import yaml
from datetime import datetime, time
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")
OUTPUT_JSON = ROOT / "reports" / "audit" / "ZERO_PAID_RESEARCH_PLAN_COMPLETION_V1.json"
OUTPUT_MARKDOWN = ROOT / "reports" / "audit" / "ZERO_PAID_RESEARCH_PLAN_COMPLETION_V1.md"

ATTACHMENTS = (
    {
        "path": Path(
            r"C:\Users\戴周阳\.codex\attachments\bf402842-399b-450f-bc94-6b70992a2d72\pasted-text-1.txt"
        ),
        "sha256": "e2abb5babecbb8f75fb69bce8e28a09b03a2abaa8dbd6ad07658f2679eec9ce0",
        "precedence": 1,
    },
    {
        "path": Path(
            r"C:\Users\戴周阳\.codex\attachments\bf402842-399b-450f-bc94-6b70992a2d72\pasted-text-2.txt"
        ),
        "sha256": "8c84a1f5eab96acd29a7a60cac51af96df375e0fd63a1225023820c88c904d0b",
        "precedence": 2,
    },
)

CENTRAL_STATUS = ROOT / "reports" / "audit" / "priority_forward_authoritative_status_v1_6.json"
MAIN_MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_7_manifest.json"
SUPERVISOR_CONFIG = ROOT / "config" / "priority_forward_supervisor_v1_3.yaml"
V1_7_CORRECTION_AUDIT = ROOT / "reports" / "audit" / "PRIORITY_FORWARD_V1_7_IMMUTABLE_RECEIPT_EVIDENCE_CORRECTION.json"
V1_7_TEST_BOUNDARY_AUDIT = ROOT / "reports" / "audit" / "PRIORITY_FORWARD_V1_7_TEST_BOUNDARY.json"
V1_8_MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_8_manifest.json"
V1_8_SUPERVISOR_CONFIG = ROOT / "config" / "priority_forward_supervisor_v1_4.yaml"
V1_8_TEST_BOUNDARY_AUDIT = ROOT / "reports" / "audit" / "PRIORITY_FORWARD_V1_8_TEST_BOUNDARY.json"
V1_8_DEPLOYMENT_AUDIT = ROOT / "reports" / "audit" / "PRIORITY_FORWARD_V1_8_DEPLOYMENT_VALIDATION.json"
ACTIVE_SCHEDULE_SURFACE_AUDIT = ROOT / "reports" / "audit" / "PRIORITY_FORWARD_ACTIVE_SCHEDULE_SURFACE_V1.json"
ACTIVE_SCHEDULE_SURFACE_AUDIT_V1_8 = (
    ROOT / "reports" / "audit" / "PRIORITY_FORWARD_ACTIVE_SCHEDULE_SURFACE_V1_8.json"
)
PCF_READINESS = ROOT / "reports" / "data_quality" / "510300_primary_market_readiness_v1_2.json"
ZERO_PAID_POLICY = ROOT / "docs" / "ZERO_PAID_DATA_POLICY.md"
CANONICAL_SOURCE_REGISTRY = ROOT / "data" / "governance" / "FREE_SOURCE_REGISTRY.csv"
SOURCE_REGISTRY = ROOT / "data" / "governance" / "FREE_SOURCE_REGISTRY_V1_1.csv"
FROZEN_INPUT_AUDIT = ROOT / "reports" / "audit" / "PRIORITY_FORWARD_FROZEN_INPUT_CONTENT_ADDRESSING_V1.json"
OPTION_AUDIT = ROOT / "reports" / "audit" / "OPTION_ORDERBOOK_ZERO_COST_SOURCE_QUALIFICATION_V1.json"
CASH_AUDIT = ROOT / "reports" / "audit" / "A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1_FINAL_FREE_SOURCE_REPAIR.json"
CB_SUFFICIENCY_RECEIPT = ROOT / "reports" / "a_share_hs_cb_priority_allocation_one_lot_v1_0_4" / "official_terms_lifecycle_sufficiency_receipt.json"
CB_SUFFICIENCY_ARCHIVE = ROOT / "config" / "a_share_hs_cb_priority_allocation_one_lot_v1_0_4_official_terms_lifecycle_sufficiency_archive_manifest.json"
CB_SUFFICIENCY_LEDGER = ROOT / "data" / "curated" / "a_share_hs_cb_priority_allocation_one_lot_v1_0_4" / "official_terms_lifecycle_sufficiency_ledger.parquet"
CB_SOURCE_AVAILABILITY = ROOT / "reports" / "a_share_hs_cb_priority_allocation_one_lot_v1_0_4" / "FREE_DOUBLE_SOURCE_AVAILABILITY_LIST.csv"
CB_LIFECYCLE_RECEIPT = ROOT / "reports" / "a_share_hs_cb_priority_allocation_one_lot_v1_0_5" / "official_listing_lifecycle_receipt.json"
CB_LIFECYCLE_ARCHIVE = ROOT / "config" / "a_share_hs_cb_priority_allocation_one_lot_v1_0_5_official_listing_lifecycle_source_archive_manifest.json"
CB_LIFECYCLE_LEDGER = ROOT / "data" / "curated" / "a_share_hs_cb_priority_allocation_one_lot_v1_0_5" / "official_lifecycle_ledger.parquet"
WINDOWS_PERSISTENCE_AUDIT = ROOT / "reports" / "audit" / "PRIORITY_FORWARD_WINDOWS_SCHEDULER_V1_6_PERSISTENCE_VALIDATION.json"
INDUSTRY_STATUS = ROOT / "reports" / "forward" / "industry_expectation_gap_v1_evaluation" / "operations_status_v1_2.json"
T_ONLY_STATUS = ROOT / "reports" / "forward" / "t_only_forward_v1_1_status.json"
T_ONLY_MANIFEST = ROOT / "config" / "t_only_forward_v1_1_maturity_only_v1_2_manifest.json"
T_ONLY_RUN_STATUS = ROOT / "paper" / "t_only_forward_v1_1" / "daily_run_status_v1_2.json"
T_ONLY_RECEIPT_DIRECTORY = (
    ROOT / "paper" / "t_only_forward_v1_1" / "daily_run_receipts_v1_2"
)
TRADE_CALENDAR = ROOT / "data" / "reference" / "sse_trade_calendar_2026.csv"
PCF_TASK_RECEIPT_DIRECTORY = (
    ROOT / "reports" / "data_quality" / "primary_market_task_runs_v1_2"
)
PCF_TASK_RECEIPT_DIRECTORY_V1_2_1 = (
    ROOT / "reports" / "data_quality" / "primary_market_task_runs_v1_2_1"
)
PCF_CODEX_RECEIPT_DIRECTORY = (
    ROOT / "reports" / "audit" / "priority_forward_codex_runs_v1_7"
)
PCF_CODEX_RECEIPT_DIRECTORY_V1_8 = (
    ROOT / "reports" / "audit" / "priority_forward_codex_runs_v1_8"
)
PCF_CLAIM_DIRECTORY = (
    ROOT / "reports" / "audit" / "priority_forward_task_claims_v1_7"
)
PCF_CLAIM_DIRECTORY_V1_8 = (
    ROOT / "reports" / "audit" / "priority_forward_task_claims_v1_8"
)
PCF_SOURCE_RECEIPT_DIRECTORY = (
    ROOT
    / "reports"
    / "data_quality"
    / "free_source_acquisitions"
    / "primary_market_v1_2"
)
INDUSTRY_TASK_RECEIPT_DIRECTORY = (
    ROOT
    / "reports"
    / "forward"
    / "industry_expectation_gap_v1_evaluation"
    / "task_runs_v1_2"
)
AUTHORITATIVE_STATUS_TASK_RECEIPT_DIRECTORY = (
    ROOT
    / "reports"
    / "audit"
    / "priority_forward_authoritative_status_task_runs_v1_6"
)

PCF_TASK_NAME = "Codex-510300-Primary-Market-Collector"
T_ONLY_TASK_NAME = "Codex-510300-T-Only-Forward-V1"
LEGACY_WINDOWS_TASK_NAMES = {
    "Codex-Industry-Expectation-Gap-Forward-Operations-V1",
    "Codex-Priority-Forward-Research-Daily-Status",
}
EXPECTED_ENABLED_WINDOWS_TASK_NAMES = {PCF_TASK_NAME, T_ONLY_TASK_NAME}
EXPECTED_ACTIVE_AUTOMATION_IDS = {"510300-pcf-iopv", "510300"}
SUPPRESSED_AUTOMATION_IDS = {"510300-2", "v3-forward-1"}
MAIN_MANIFEST_SHA256 = "50d40ecefd3fdf8aa7f6de1f5db11c37edfa41ff04f75fe3edb295800a93aa49"
V1_8_MANIFEST_SHA256 = "ae8215c5b7e35837564d464eb975054a2d79e2dd33a0cc25636b2c667fdcd1c3"
T_ONLY_MANIFEST_SHA256 = "a390cdb2cbd8c1bfa3dc18dbd1055d2acf1792d39b337ff27bd08d4389a01d29"
ZERO_PAID_POLICY_SHA256 = "534bce83d18f78cc4893bbb861caa3826860d2c5a2e21a8143a4064d6858668b"

SOURCE_REGISTRY_REQUIRED_COLUMNS = {
    "source_id",
    "grade",
    "research_line",
    "provider",
    "source_url_or_endpoint",
    "access_cost_cny",
    "allowed_use",
    "prohibited_use",
    "fields_or_evidence",
    "time_coverage",
    "point_in_time_status",
    "raw_preservation_required",
    "parser_version",
    "receipt_schema_version",
    "fallback_source",
    "admission_status",
    "quality_status",
    "failure_reason",
    "terms_or_license_note",
    "last_audited_at",
}
SOURCE_REGISTRY_REQUIRED_NONEMPTY_COLUMNS = SOURCE_REGISTRY_REQUIRED_COLUMNS - {
    "fallback_source",
    "failure_reason",
}
ZERO_PAID_POLICY_REQUIRED_TOKENS = {
    "DATA_PURCHASE_BUDGET = 0 CNY",
    "首次复核日：`2027-02-21`",
    "同时允许的主动研究流：最多 2 条",
    "且仅限 `A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1`",
    "不把采购保留为开放任务",
    "原始响应先落盘，解析结果与原始层分开保存",
    "失败日不得使用以后日期回填",
    "`NO_VIEW`、`BLOCKED`、`FAILED`、`SKIPPED`",
    "source_id",
    "source_url_or_endpoint",
    "acquired_at",
    "market_date",
    "raw_response_hash",
    "raw_response_path",
    "parser_version",
    "schema_version",
    "quality_status",
    "fallback_source",
    "failure_reason",
}

AUTOMATIONS = (
    {
        "id": "510300-pcf-iopv",
        "path": Path.home() / ".codex" / "automations" / "510300-pcf-iopv" / "automation.toml",
        "required_runner": "run_priority_forward_codex_automation_v1_7.py",
        "required_manifest_sha256": MAIN_MANIFEST_SHA256,
        "required_time": "BYHOUR=9;BYMINUTE=20",
    },
    {
        "id": "510300",
        "path": Path.home() / ".codex" / "automations" / "510300" / "automation.toml",
        "required_runner": "run_priority_forward_codex_automation_v1_7.py",
        "required_manifest_sha256": MAIN_MANIFEST_SHA256,
        "required_time": "BYHOUR=19;BYMINUTE=0",
    },
)

FORBIDDEN_T_ONLY_KEYS = {
    "metrics",
    "gates",
    "current_shadow_signal",
    "model_action",
    "shadow_account",
    "latest_indicator_diagnostic",
    "possible_model_actions",
    "daily_decision",
    "daily_headline",
    "target_exposure",
    "total_return",
    "cagr",
    "sharpe",
    "maximum_drawdown",
    "average_exposure",
}

SATISFIED_STATUSES = {"PASS", "TERMINAL_EXPECTED"}
PENDING_STATUSES = {
    "PENDING_WINDOW",
    "PENDING_MATURITY",
    "PENDING_PERMISSION",
    "PENDING_CURRENT_RUN",
    "PENDING_NEXT_WINDOW",
}
CONTRADICTION_STATUSES = {"FAILED", "CONTRADICTION", "MISSING"}
EXECUTION_DISABLED_FIELDS = {
    "position_mapping_enabled",
    "order_generation_enabled",
    "broker_connection_enabled",
    "live_trading_enabled",
}
FREE_SOURCE_RECEIPT_REQUIRED_FIELDS = {
    "source_id",
    "source_url_or_endpoint",
    "acquired_at",
    "market_date",
    "raw_response_hash",
    "raw_response_path",
    "parser_version",
    "schema_version",
    "quality_status",
    "fallback_source",
    "failure_reason",
}
PCF_REQUIRED_SOURCE_IDS = {
    "SSE_PCF_COMMON_QUERY",
    "SSE_YUNHQ_IOPV_SNAPSHOT",
    "SSE_ETF_DAILY_TURNOVER_OFFICIAL",
}
PCF_V1_8_EFFECTIVE_DATE = "2026-08-27"
PCF_V1_8_RUNTIME_PATCH_ID = "PCF_IOPV_PYTHON_MODULE_ENTRYPOINT_V1_2_1"


def pcf_runtime_contract_for_date(
    target_date: str, active_runtime_version: str | None = None
) -> dict[str, Any]:
    use_v1_8 = bool(
        target_date >= PCF_V1_8_EFFECTIVE_DATE
        and (
            active_runtime_version == "V1_8"
            if active_runtime_version is not None
            else True
        )
    )
    if use_v1_8:
        return {
            "runtime_version": "V1_8",
            "task_receipt_directory": PCF_TASK_RECEIPT_DIRECTORY_V1_2_1,
            "codex_receipt_directory": PCF_CODEX_RECEIPT_DIRECTORY_V1_8,
            "claim_directory": PCF_CLAIM_DIRECTORY_V1_8,
            "task_schema_version": "1.2.1",
            "task_launcher": (
                "scripts/run_510300_primary_market_collection_task_v1_2_1.ps1"
            ),
            "forward_runner": (
                "scripts/run_510300_primary_market_forward_v1_2_1.ps1"
            ),
            "runtime_patch_id": PCF_V1_8_RUNTIME_PATCH_ID,
        }
    return {
        "runtime_version": "V1_7",
        "task_receipt_directory": PCF_TASK_RECEIPT_DIRECTORY,
        "codex_receipt_directory": PCF_CODEX_RECEIPT_DIRECTORY,
        "claim_directory": PCF_CLAIM_DIRECTORY,
        "task_schema_version": "1.2.0",
        "task_launcher": "scripts/run_510300_primary_market_collection_task_v1_2.ps1",
        "forward_runner": "scripts/run_510300_primary_market_forward_v1_2.ps1",
        "runtime_patch_id": None,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON 顶层必须是对象：{path}")
    return payload


def active_runtime_contract() -> dict[str, Any]:
    legacy = {
        "runtime_version": "V1_7",
        "manifest_path": MAIN_MANIFEST,
        "manifest_sha256": MAIN_MANIFEST_SHA256,
        "manifest_status": "FROZEN_ZERO_PAID_FOUNDATION_V1_7_ACTIVE",
        "supervisor_config_path": SUPERVISOR_CONFIG,
        "runner": "run_priority_forward_codex_automation_v1_7.py",
        "correction_audit_path": V1_7_CORRECTION_AUDIT,
        "correction_expected_status": "PASS_V1_7_DEPLOYED_AWAITING_REAL_WINDOWS",
        "test_boundary_audit_path": V1_7_TEST_BOUNDARY_AUDIT,
        "schedule_surface_audit_path": ACTIVE_SCHEDULE_SURFACE_AUDIT,
        "deployment_audit_present": False,
        "deployment_audit_valid": True,
    }
    if not V1_8_DEPLOYMENT_AUDIT.is_file():
        return legacy
    try:
        deployment = read_json(V1_8_DEPLOYMENT_AUDIT)
    except Exception as exc:
        return {
            **legacy,
            "deployment_audit_present": True,
            "deployment_audit_valid": False,
            "deployment_audit_error": str(exc),
        }
    frozen_manifest = deployment.get("frozen_manifest")
    deployment_valid = bool(
        deployment.get("status")
        == "PASS_V1_8_DEPLOYED_FOR_NEXT_ELIGIBLE_WINDOW"
        and isinstance(frozen_manifest, Mapping)
        and frozen_manifest.get("path")
        == "config/priority_forward_research_operations_v1_8_manifest.json"
        and frozen_manifest.get("sha256") == V1_8_MANIFEST_SHA256
        and deployment.get("effective_from") == "2026-08-27T09:20:00+08:00"
        and deployment.get("same_day_pcf_retry_performed") is False
    )
    if not deployment_valid:
        return {
            **legacy,
            "deployment_audit_present": True,
            "deployment_audit_valid": False,
            "deployment_audit_status": deployment.get("status"),
        }
    return {
        "runtime_version": "V1_8",
        "manifest_path": V1_8_MANIFEST,
        "manifest_sha256": V1_8_MANIFEST_SHA256,
        "manifest_status": (
            "FROZEN_ZERO_PAID_FOUNDATION_V1_8_ACTIVE_NEXT_ELIGIBLE_WINDOW"
        ),
        "supervisor_config_path": V1_8_SUPERVISOR_CONFIG,
        "runner": "run_priority_forward_codex_automation_v1_8.py",
        "correction_audit_path": V1_8_DEPLOYMENT_AUDIT,
        "correction_expected_status": (
            "PASS_V1_8_DEPLOYED_FOR_NEXT_ELIGIBLE_WINDOW"
        ),
        "test_boundary_audit_path": V1_8_TEST_BOUNDARY_AUDIT,
        "schedule_surface_audit_path": ACTIVE_SCHEDULE_SURFACE_AUDIT_V1_8,
        "deployment_audit_present": True,
        "deployment_audit_valid": True,
    }


def evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "bytes": None}
    try:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    return {
        "path": relative,
        "exists": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_zero_paid_data_contract(
    policy_path: Path,
    canonical_registry_path: Path,
    versioned_registry_path: Path,
) -> dict[str, Any]:
    """核验零付费政策和显式来源注册表交付物。"""

    policy_text = policy_path.read_text(encoding="utf-8-sig")
    missing_policy_tokens = sorted(
        token for token in ZERO_PAID_POLICY_REQUIRED_TOKENS if token not in policy_text
    )
    policy_sha256 = sha256_file(policy_path)
    policy_checks = {
        "sha256_matches": policy_sha256 == ZERO_PAID_POLICY_SHA256,
        "required_tokens_present": not missing_policy_tokens,
    }

    canonical_rows = read_csv_rows(canonical_registry_path)
    versioned_rows = read_csv_rows(versioned_registry_path)
    columns = set(versioned_rows[0]) if versioned_rows else set()
    missing_columns = sorted(SOURCE_REGISTRY_REQUIRED_COLUMNS - columns)
    empty_required_cells = [
        {"row": index, "source_id": row.get("source_id"), "column": column}
        for index, row in enumerate(versioned_rows, start=2)
        for column in sorted(SOURCE_REGISTRY_REQUIRED_NONEMPTY_COLUMNS)
        if not str(row.get(column) or "").strip()
    ]
    duplicate_source_ids = sorted(
        source_id
        for source_id in {row.get("source_id") for row in versioned_rows}
        if source_id
        and sum(row.get("source_id") == source_id for row in versioned_rows) > 1
    )
    nonzero_cost_source_ids = sorted(
        str(row.get("source_id"))
        for row in versioned_rows
        if float(row.get("access_cost_cny") or 0) != 0
    )
    raw_preservation_not_required = sorted(
        str(row.get("source_id"))
        for row in versioned_rows
        if str(row.get("raw_preservation_required") or "").strip().lower()
        != "true"
    )
    registry_checks = {
        "has_rows": bool(versioned_rows),
        "required_columns_present": not missing_columns,
        "required_cells_complete": not empty_required_cells,
        "source_ids_unique": not duplicate_source_ids,
        "all_sources_zero_cost": not nonzero_cost_source_ids,
        "all_sources_require_raw_preservation": not raw_preservation_not_required,
        "canonical_matches_frozen_v1_1": canonical_rows == versioned_rows,
    }
    return {
        "status": "PASS"
        if all(policy_checks.values()) and all(registry_checks.values())
        else "FAILED",
        "policy_status": "PASS" if all(policy_checks.values()) else "FAILED",
        "registry_status": "PASS" if all(registry_checks.values()) else "FAILED",
        "policy_sha256": policy_sha256,
        "policy_checks": policy_checks,
        "missing_policy_tokens": missing_policy_tokens,
        "registry_row_count": len(versioned_rows),
        "registry_columns": sorted(columns),
        "missing_registry_columns": missing_columns,
        "empty_required_registry_cells": empty_required_cells,
        "duplicate_source_ids": duplicate_source_ids,
        "nonzero_cost_source_ids": nonzero_cost_source_ids,
        "raw_preservation_not_required": raw_preservation_not_required,
        "registry_checks": registry_checks,
    }


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        return {
            "status": "FAILED",
            "manifest_sha256": sha256_file(path),
            "failure": "MANIFEST_FILES_MISSING",
        }
    actual_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for expected in records:
        relative = str(expected["path"])
        target = ROOT / relative
        actual = {
            "path": relative,
            "sha256": sha256_file(target) if target.is_file() else None,
            "bytes": target.stat().st_size if target.is_file() else None,
        }
        actual_records.append(actual)
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    actual_records.sort(key=lambda item: str(item["path"]))
    content_sha256 = hashlib.sha256(
        json.dumps(actual_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if content_sha256 != manifest.get("content_sha256"):
        failures.append(
            {
                "path": "<manifest-content>",
                "expected": manifest.get("content_sha256"),
                "actual": content_sha256,
            }
        )
    return {
        "status": "PASS" if not failures else "FAILED",
        "manifest_sha256": sha256_file(path),
        "content_sha256": content_sha256,
        "tracked_file_count": len(actual_records),
        "failure_count": len(failures),
        "failures": failures,
    }


def verify_runtime_test_boundary(
    manifest_path: Path, supervisor_config_path: Path
) -> dict[str, Any]:
    """实时核验测试文件、测试导入和监督器启动项是否越过冻结运行边界。"""

    manifest = read_json(manifest_path)
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        return {
            "status": "FAILED",
            "failure": "MANIFEST_FILES_MISSING",
            "tracked_test_file_count": 0,
            "runtime_test_import_count": 0,
            "config_launcher_count": 0,
            "config_launchers_all_tracked": False,
        }

    tracked_paths = {
        str(record["path"]).replace("\\", "/")
        for record in records
        if isinstance(record, Mapping) and record.get("path")
    }
    tracked_test_paths = sorted(
        relative
        for relative in tracked_paths
        if any(part.lower() in {"test", "tests"} for part in Path(relative).parts)
        or Path(relative).name.lower().startswith("test_")
    )

    test_imports: list[dict[str, Any]] = []
    parse_failures: list[dict[str, str]] = []
    for relative in sorted(path for path in tracked_paths if path.lower().endswith(".py")):
        target = ROOT / relative
        try:
            tree = ast.parse(target.read_text(encoding="utf-8-sig"), filename=relative)
        except (OSError, SyntaxError, UnicodeError) as exc:
            parse_failures.append({"path": relative, "error": str(exc)})
            continue
        for node in ast.walk(tree):
            imported_names: list[str] = []
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names.append(node.module)
                imported_names.extend(alias.name for alias in node.names)
            for imported_name in imported_names:
                parts = imported_name.lower().split(".")
                if any(
                    part in {"test", "tests", "pytest", "unittest"}
                    or part.startswith("test_")
                    for part in parts
                ):
                    test_imports.append(
                        {
                            "path": relative,
                            "line": getattr(node, "lineno", None),
                            "module": imported_name,
                        }
                    )

    config_text = supervisor_config_path.read_text(encoding="utf-8-sig")
    launchers = [
        match.strip().replace("\\", "/")
        for match in re.findall(
            r"(?m)^\s*(?:-\s*)?launcher:\s*[\"']?([^\"'#\r\n]+)", config_text
        )
    ]
    missing_launchers = sorted(
        launcher for launcher in launchers if launcher not in tracked_paths
    )
    config_launchers_all_tracked = bool(launchers) and not missing_launchers
    passed = bool(
        not tracked_test_paths
        and not test_imports
        and not parse_failures
        and config_launchers_all_tracked
    )
    return {
        "status": "PASS" if passed else "FAILED",
        "tracked_file_count": len(tracked_paths),
        "tracked_python_file_count": sum(
            path.lower().endswith(".py") for path in tracked_paths
        ),
        "tracked_test_file_count": len(tracked_test_paths),
        "tracked_test_paths": tracked_test_paths,
        "runtime_test_import_count": len(test_imports),
        "runtime_test_imports": test_imports,
        "python_parse_failure_count": len(parse_failures),
        "python_parse_failures": parse_failures,
        "config_launcher_count": len(launchers),
        "config_launchers": launchers,
        "config_launchers_all_tracked": config_launchers_all_tracked,
        "missing_config_launchers": missing_launchers,
    }


def verify_industry_frozen_input_boundary(
    frozen_input_audit: Mapping[str, Any],
    industry_status: Mapping[str, Any],
    supervisor_config_path: Path,
    *,
    pcf_task_launcher: str = (
        "scripts/run_510300_primary_market_collection_task_v1_2.ps1"
    ),
    pcf_evidence_glob: str = (
        "reports/data_quality/primary_market_task_runs_v1_2/*.json"
    ),
) -> dict[str, Any]:
    """实时核验当前监督器和行业内容寻址快照，不沿用历史latest语义。"""

    failures: list[dict[str, Any]] = []
    frozen_industry = frozen_input_audit.get("industry_expectation_gap") or {}
    frozen_records = frozen_industry.get("records") or []
    current_snapshot = industry_status.get("input_snapshot") or {}
    current_records = current_snapshot.get("records") or []
    current_by_role = {
        str(record.get("role")): record
        for record in current_records
        if isinstance(record, Mapping)
    }
    verified_records: list[dict[str, Any]] = []
    for record in frozen_records:
        role = str(record.get("role") or "")
        relative = str(record.get("snapshot_path") or "")
        expected_hash = str(record.get("sha256") or "")
        target = ROOT / relative
        actual_hash = sha256_file(target) if target.is_file() else None
        filename_content_addressed = bool(
            target.is_file() and target.stem == expected_hash
        )
        current_record = current_by_role.get(role) or {}
        record_pass = bool(
            role
            and relative
            and expected_hash
            and record.get("content_addressed") is True
            and actual_hash == expected_hash
            and filename_content_addressed
            and current_record.get("snapshot_path") == relative
            and current_record.get("sha256") == expected_hash
            and current_record.get("content_addressed") is True
        )
        verified_records.append(
            {
                "role": role,
                "snapshot_path": relative,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "filename_content_addressed": filename_content_addressed,
                "matches_current_status": bool(
                    current_record.get("snapshot_path") == relative
                    and current_record.get("sha256") == expected_hash
                ),
                "status": "PASS" if record_pass else "FAILED",
            }
        )
        if not record_pass:
            failures.append({"role": role, "reason": "SNAPSHOT_RECORD_MISMATCH"})

    supervisor_text = supervisor_config_path.read_text(encoding="utf-8-sig")
    launchers = sorted(
        match.strip().replace("\\", "/")
        for match in re.findall(
            r"(?m)^\s*(?:-\s*)?launcher:\s*[\"']?([^\"'#\r\n]+)",
            supervisor_text,
        )
    )
    evidence_globs = sorted(
        match.strip().replace("\\", "/")
        for match in re.findall(
            r"(?m)^\s*-\s+([^#\r\n]+\.json)\s*$", supervisor_text
        )
    )
    expected_launchers = sorted(
        [
            pcf_task_launcher,
            "scripts/run_industry_expectation_gap_forward_operations_task_v1_2.ps1",
            "scripts/run_priority_forward_authoritative_status_task_v1_6.ps1",
        ]
    )
    expected_evidence_globs = sorted(
        [
            pcf_evidence_glob,
            "reports/forward/industry_expectation_gap_v1_evaluation/task_runs_v1_2/*.json",
            "reports/audit/priority_forward_authoritative_status_task_runs_v1_6/*.json",
        ]
    )
    checks = {
        "snapshot_id_matches": frozen_industry.get("snapshot_id")
        == current_snapshot.get("snapshot_id"),
        "snapshot_manifest_matches": frozen_industry.get("immutable_manifest_path")
        == current_snapshot.get("immutable_manifest_path"),
        "all_snapshot_records_verified": bool(verified_records)
        and not any(record["status"] == "FAILED" for record in verified_records),
        "shared_latest_not_consumed": frozen_industry.get(
            "shared_latest_consumed_by_evaluation"
        )
        is False
        and current_snapshot.get("shared_latest_consumed_by_evaluation") is False,
        "current_supervisor_launchers_exact": launchers == expected_launchers,
        "current_supervisor_evidence_globs_exact": evidence_globs
        == expected_evidence_globs,
        "current_supervisor_excludes_v3": "v3_forward_2" not in supervisor_text.lower(),
    }
    if not all(checks.values()):
        failures.append({"role": "<boundary>", "reason": "BOUNDARY_CHECK_FAILED"})
    return {
        "status": "PASS" if not failures else "FAILED",
        "checks": checks,
        "legacy_audit_supervisor_path": (
            frozen_input_audit.get("active_supervisor") or {}
        ).get("path"),
        "current_supervisor_path": supervisor_config_path.relative_to(ROOT).as_posix(),
        "current_supervisor_sha256": sha256_file(supervisor_config_path),
        "launchers": launchers,
        "evidence_globs": evidence_globs,
        "snapshot_record_count": len(verified_records),
        "verified_records": verified_records,
        "failure_count": len(failures),
        "failures": failures,
    }


def verify_file_map_manifest(path: Path) -> dict[str, Any]:
    """核验以相对路径到SHA-256映射保存的不可变归档清单。"""

    manifest = read_json(path)
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        return {
            "status": "FAILED",
            "manifest_sha256": sha256_file(path),
            "failure": "MANIFEST_FILE_MAP_MISSING",
        }
    failures: list[dict[str, Any]] = []
    for relative, expected_hash in records.items():
        target = ROOT / str(relative)
        actual_hash = sha256_file(target) if target.is_file() else None
        if actual_hash != expected_hash:
            failures.append(
                {
                    "path": relative,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                }
            )
    canonical = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    actual_content_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual_content_sha256 != manifest.get("content_sha256"):
        failures.append(
            {
                "path": "<manifest-content>",
                "expected_sha256": manifest.get("content_sha256"),
                "actual_sha256": actual_content_sha256,
            }
        )
    return {
        "status": "PASS" if not failures else "FAILED",
        "manifest_sha256": sha256_file(path),
        "content_sha256": actual_content_sha256,
        "tracked_file_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
    }


def windows_task_snapshot(task_name: str) -> dict[str, Any]:
    if task_name not in EXPECTED_ENABLED_WINDOWS_TASK_NAMES | LEGACY_WINDOWS_TASK_NAMES:
        raise ValueError(f"未登记的计划任务：{task_name}")
    escaped = task_name.replace("'", "''")
    command = (
        f"$task=Get-ScheduledTask -TaskName '{escaped}' -ErrorAction Stop;"
        f"$info=Get-ScheduledTaskInfo -TaskName '{escaped}';"
        "$action=@($task.Actions)[0];"
        "[ordered]@{"
        "task_name=$task.TaskName;state=[string]$task.State;"
        "next_run_time=if($null-ne$info.NextRunTime){$info.NextRunTime.ToString('o')}else{$null};"
        "last_run_time=if($null-ne$info.LastRunTime){$info.LastRunTime.ToString('o')}else{$null};"
        "last_task_result=$info.LastTaskResult;"
        "execute=[string]$action.Execute;arguments=[string]$action.Arguments;"
        "working_directory=[string]$action.WorkingDirectory;"
        "restart_count=[int]$task.Settings.RestartCount;"
        "start_when_available=[bool]$task.Settings.StartWhenAvailable;"
        "multiple_instances=[string]$task.Settings.MultipleInstances;"
        "logon_type=[string]$task.Principal.LogonType;"
        "run_level=[string]$task.Principal.RunLevel"
        "}|ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        return {
            "task_name": task_name,
            "query_status": "FAILED",
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    payload = json.loads(completed.stdout)
    payload["query_status"] = "PASS"
    return payload


def automation_snapshot(specification: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(specification["path"])
    if not path.is_file():
        return {
            "id": specification["id"],
            "status": "MISSING",
            "path": str(path),
        }
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    prompt = str(config.get("prompt") or "")
    rrule = str(config.get("rrule") or "")
    checks = {
        "active": config.get("status") == "ACTIVE",
        "runner_pinned": str(specification["required_runner"]) in prompt,
        "manifest_sha256_pinned": str(specification["required_manifest_sha256"])
        in prompt,
        "schedule_matches": str(specification["required_time"]) in rrule,
    }
    return {
        "id": specification["id"],
        "status": "PASS" if all(checks.values()) else "FAILED",
        "checks": checks,
        "path": str(path),
        "sha256": sha256_file(path),
        "target_thread_id": config.get("target_thread_id"),
    }


def workspace_windows_task_inventory() -> dict[str, Any]:
    escaped_root = str(ROOT).replace("'", "''")
    command = (
        f"$projectRoot='{escaped_root}';"
        "$records=@();"
        "Get-ScheduledTask|ForEach-Object{"
        "$task=$_;foreach($action in @($task.Actions)){"
        "$joined=([string]$action.Execute)+' '+([string]$action.Arguments)+' '+([string]$action.WorkingDirectory);"
        "if($joined.Contains($projectRoot)){"
        "$records+=[ordered]@{task_name=$task.TaskName;task_path=$task.TaskPath;"
        "state=[string]$task.State;execute=[string]$action.Execute;"
        "arguments=[string]$action.Arguments;working_directory=[string]$action.WorkingDirectory}"
        "}}};@($records)|ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        return {
            "query_status": "FAILED",
            "error": completed.stderr.strip() or completed.stdout.strip(),
            "tasks": [],
        }
    decoded = json.loads(completed.stdout.lstrip("\ufeff") or "[]")
    tasks = decoded if isinstance(decoded, list) else [decoded]
    return {"query_status": "PASS", "tasks": tasks}


def workspace_automation_inventory() -> dict[str, Any]:
    automation_root = Path.home() / ".codex" / "automations"
    records: list[dict[str, Any]] = []
    if not automation_root.is_dir():
        return {"query_status": "FAILED", "error": "AUTOMATION_ROOT_MISSING", "automations": []}
    for path in sorted(automation_root.glob("*/automation.toml")):
        try:
            with path.open("rb") as handle:
                config = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        searchable = str(config).replace("\\\\", "\\")
        if str(ROOT).lower() not in searchable.lower():
            continue
        prompt = str(config.get("prompt") or "")
        runners = sorted(set(re.findall(r"[A-Za-z0-9_-]+\.(?:py|ps1)", prompt)))
        records.append(
            {
                "id": str(config.get("id") or path.parent.name),
                "status": str(config.get("status") or "MISSING"),
                "kind": config.get("kind"),
                "runners": runners,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    return {"query_status": "PASS", "automations": records}


def verify_active_schedule_surface(
    pcf_task: Mapping[str, Any],
    t_only_task: Mapping[str, Any],
    *,
    pcf_runner: str,
    pcf_manifest_sha256: str,
) -> dict[str, Any]:
    windows_inventory = workspace_windows_task_inventory()
    automation_inventory = workspace_automation_inventory()
    windows_tasks = windows_inventory.get("tasks") or []
    automations = automation_inventory.get("automations") or []
    enabled_windows_task_names = sorted(
        str(item.get("task_name"))
        for item in windows_tasks
        if item.get("state") != "Disabled"
    )
    disabled_legacy_task_names = sorted(
        str(item.get("task_name"))
        for item in windows_tasks
        if item.get("state") == "Disabled"
        and item.get("task_name") in LEGACY_WINDOWS_TASK_NAMES
    )
    active_automation_ids = sorted(
        str(item.get("id")) for item in automations if item.get("status") == "ACTIVE"
    )
    automation_status_by_id = {
        str(item.get("id")): str(item.get("status")) for item in automations
    }
    checks = {
        "windows_inventory_readable": windows_inventory.get("query_status") == "PASS",
        "only_authorized_windows_tasks_enabled": set(enabled_windows_task_names)
        == EXPECTED_ENABLED_WINDOWS_TASK_NAMES,
        "legacy_windows_tasks_disabled": set(disabled_legacy_task_names)
        == LEGACY_WINDOWS_TASK_NAMES,
        "pcf_windows_task_exact": task_exact(
            pcf_task,
            runner=pcf_runner,
            manifest_sha256=pcf_manifest_sha256,
        ),
        "t_only_windows_task_exact": task_exact(
            t_only_task,
            runner="run_t_only_forward_v1_daily_v1_2.py",
            manifest_sha256=T_ONLY_MANIFEST_SHA256,
        ),
        "automation_inventory_readable": automation_inventory.get("query_status")
        == "PASS",
        "only_authorized_automations_active": set(active_automation_ids)
        == EXPECTED_ACTIVE_AUTOMATION_IDS,
        "suppressed_automations_paused": all(
            automation_status_by_id.get(automation_id) == "PAUSED"
            for automation_id in SUPPRESSED_AUTOMATION_IDS
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "checks": checks,
        "enabled_windows_task_names": enabled_windows_task_names,
        "disabled_legacy_task_names": disabled_legacy_task_names,
        "active_automation_ids": active_automation_ids,
        "automation_status_by_id": automation_status_by_id,
        "windows_inventory": windows_inventory,
        "automation_inventory": automation_inventory,
    }


def walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(walk_keys(item))
    return keys


def trading_day(today: str) -> bool:
    with TRADE_CALENDAR.open("r", encoding="utf-8-sig", newline="") as handle:
        return any(row["trade_date"] == today for row in csv.DictReader(handle))


def receipts_for_local_date(
    directory: Path,
    target_date: str,
    *,
    phase: str | None = None,
    timestamp_fields: tuple[str, ...] = ("started_at", "attempted_at"),
) -> list[dict[str, Any]]:
    return [
        evidence(path)
        for path in receipt_paths_for_local_date(
            directory,
            target_date,
            phase=phase,
            timestamp_fields=timestamp_fields,
        )
    ]


def receipt_paths_for_local_date(
    directory: Path,
    target_date: str,
    *,
    phase: str | None = None,
    timestamp_fields: tuple[str, ...] = ("started_at", "attempted_at"),
) -> list[Path]:
    if not directory.is_dir():
        return []
    paths: list[Path] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = read_json(path)
        except Exception:
            modified_date = datetime.fromtimestamp(
                path.stat().st_mtime, TIMEZONE
            ).date().isoformat()
            if phase is None and modified_date == target_date:
                paths.append(path)
            continue
        timestamp = next(
            (payload.get(field) for field in timestamp_fields if payload.get(field)),
            None,
        )
        if (
            timestamp
            and str(timestamp)[:10] == target_date
            and (phase is None or payload.get("phase") == phase)
        ):
            paths.append(path)
    return paths


def receipt_payloads_for_local_date(
    directory: Path,
    target_date: str,
    *,
    phase: str | None = None,
    timestamp_fields: tuple[str, ...] = ("started_at", "attempted_at"),
) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    payloads: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        timestamp = next(
            (payload.get(field) for field in timestamp_fields if payload.get(field)),
            None,
        )
        if (
            timestamp
            and str(timestamp)[:10] == target_date
            and (phase is None or payload.get("phase") == phase)
        ):
            payloads.append(payload)
    return payloads


def _execution_fields_disabled(payload: Mapping[str, Any]) -> bool:
    return all(payload.get(field) is False for field in EXECUTION_DISABLED_FIELDS)


def _parse_local_timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(TIMEZONE)


def verify_pcf_atomic_run_evidence(
    task_receipts: list[dict[str, Any]],
    codex_receipts: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    target_date: str,
    readiness: Mapping[str, Any] | None = None,
    *,
    task_schema_version: str = "1.2.0",
    task_launcher: str = "scripts/run_510300_primary_market_collection_task_v1_2.ps1",
    forward_runner: str = "scripts/run_510300_primary_market_forward_v1_2.ps1",
    runtime_patch_id: str | None = None,
) -> dict[str, Any]:
    """核验一次 PCF 当日运行是否形成一致、不可重复的原子证据束。"""

    task_failures: list[dict[str, Any]] = []
    successful_task_receipts = 0
    task_receipt_ids: list[str] = []
    success_collection_statuses = {
        "COMPLETE_QUALITY_DAY",
        "COLLECTED_INCOMPLETE_QUALITY_DAY",
    }
    for payload in task_receipts:
        reasons: list[str] = []
        receipt_id = str(payload.get("receipt_id") or "")
        task_receipt_ids.append(receipt_id)
        started_at = _parse_local_timestamp(payload.get("started_at"))
        ended_at = _parse_local_timestamp(payload.get("ended_at"))
        if payload.get("schema_version") != task_schema_version:
            reasons.append("SCHEMA_VERSION_MISMATCH")
        if not receipt_id:
            reasons.append("RECEIPT_ID_MISSING")
        if payload.get("immutable_receipt") is not True:
            reasons.append("IMMUTABLE_RECEIPT_FALSE")
        receipt_relative = str(payload.get("receipt_file") or "")
        receipt_path = ROOT / receipt_relative
        receipt_file_matches = False
        if receipt_relative and receipt_path.is_file():
            try:
                receipt_file_matches = read_json(receipt_path) == payload
            except Exception:
                receipt_file_matches = False
        if not receipt_file_matches:
            reasons.append("RECEIPT_FILE_SELF_REFERENCE_MISMATCH")
        if payload.get("task_name") != PCF_TASK_NAME:
            reasons.append("TASK_NAME_MISMATCH")
        if runtime_patch_id is not None:
            if payload.get("runtime_patch_id") != runtime_patch_id:
                reasons.append("RUNTIME_PATCH_ID_MISMATCH")
            if payload.get("python_module_entrypoint") is not True:
                reasons.append("PYTHON_MODULE_ENTRYPOINT_FALSE")
            if payload.get("runner_file") != forward_runner:
                reasons.append("FORWARD_RUNNER_MISMATCH")
        if started_at is None or started_at.date().isoformat() != target_date:
            reasons.append("START_DATE_MISMATCH")
        elif not time(9, 25) <= started_at.timetz().replace(tzinfo=None) <= time(9, 35):
            reasons.append("OUTSIDE_LEGAL_START_WINDOW")
        if ended_at is None or started_at is None or ended_at < started_at:
            reasons.append("END_TIMESTAMP_INVALID")
        if not _execution_fields_disabled(payload):
            reasons.append("EXECUTION_FIELDS_NOT_DISABLED")

        run_status = payload.get("run_status")
        collection_status = str(payload.get("collection_status") or "")
        exit_code = payload.get("task_exit_code")
        if run_status == "SUCCESS":
            if exit_code != 0:
                reasons.append("SUCCESS_EXIT_CODE_NOT_ZERO")
            if collection_status not in success_collection_statuses:
                reasons.append("SUCCESS_COLLECTION_STATUS_INVALID")
            if payload.get("latest_observed_trade_date") != target_date:
                reasons.append("SUCCESS_CURRENT_TRADE_DATE_MISSING")
            if not isinstance(readiness, Mapping):
                reasons.append("SUCCESS_READINESS_REPORT_MISSING")
            else:
                readiness_generated_at = _parse_local_timestamp(
                    readiness.get("generated_at")
                )
                target_quality_rows = [
                    item
                    for item in readiness.get("daily_quality") or []
                    if isinstance(item, Mapping)
                    and item.get("trade_date") == target_date
                ]
                expected_collection_status = (
                    "COMPLETE_QUALITY_DAY"
                    if len(target_quality_rows) == 1
                    and target_quality_rows[0].get("complete_quality_day") is True
                    else "COLLECTED_INCOMPLETE_QUALITY_DAY"
                )
                if (
                    readiness_generated_at is None
                    or started_at is None
                    or ended_at is None
                    or not started_at <= readiness_generated_at <= ended_at
                ):
                    reasons.append("READINESS_GENERATION_ORDER_INVALID")
                if readiness.get("last_observed_trade_date") != target_date:
                    reasons.append("READINESS_CURRENT_TRADE_DATE_MISSING")
                if payload.get("full_coverage_days") != readiness.get(
                    "full_coverage_days"
                ):
                    reasons.append("READINESS_FULL_COVERAGE_COUNT_MISMATCH")
                if len(target_quality_rows) != 1:
                    reasons.append("READINESS_TARGET_QUALITY_ROW_COUNT_INVALID")
                if collection_status != expected_collection_status:
                    reasons.append("READINESS_COLLECTION_STATUS_MISMATCH")
            if not reasons:
                successful_task_receipts += 1
        elif run_status == "FAILED":
            if exit_code not in {1, 3}:
                reasons.append("FAILED_EXIT_CODE_INVALID")
            if not (
                collection_status == "EXTERNAL_FREE_SOURCE_FAILED"
                or collection_status.startswith("PROGRAM_FAILED")
                or collection_status.startswith("FAILED_")
            ):
                reasons.append("FAILED_COLLECTION_STATUS_INVALID")
            if (
                collection_status == "EXTERNAL_FREE_SOURCE_FAILED"
                and exit_code != 3
            ):
                reasons.append("EXTERNAL_FAILURE_EXIT_CODE_MISMATCH")
            if (
                collection_status != "EXTERNAL_FREE_SOURCE_FAILED"
                and exit_code != 1
            ):
                reasons.append("INTERNAL_FAILURE_EXIT_CODE_MISMATCH")
        else:
            reasons.append("RUN_STATUS_UNKNOWN")
        if reasons:
            task_failures.append({"receipt_id": receipt_id, "reasons": reasons})

    duplicate_task_receipt_ids = sorted(
        receipt_id
        for receipt_id in set(task_receipt_ids)
        if receipt_id and task_receipt_ids.count(receipt_id) > 1
    )
    task_contract_valid = bool(
        len(task_receipts) == 1
        and not task_failures
        and not duplicate_task_receipt_ids
    )

    claim_failures: list[dict[str, Any]] = []
    for payload in claims:
        reasons: list[str] = []
        claimed_at = _parse_local_timestamp(payload.get("claimed_at"))
        if payload.get("schema_version") != "1.0.0":
            reasons.append("SCHEMA_VERSION_MISMATCH")
        if payload.get("claim_status") != "ATOMIC_ATTEMPT_CLAIMED":
            reasons.append("CLAIM_STATUS_INVALID")
        if payload.get("immutable_receipt") is not True:
            reasons.append("IMMUTABLE_RECEIPT_FALSE")
        if payload.get("task_id") != "PRIMARY_MARKET_PCF_IOPV":
            reasons.append("TASK_ID_MISMATCH")
        if payload.get("target_date") != target_date:
            reasons.append("TARGET_DATE_MISMATCH")
        if claimed_at is None or claimed_at.date().isoformat() != target_date:
            reasons.append("CLAIMED_AT_DATE_MISMATCH")
        elif not time(9, 25) <= claimed_at.timetz().replace(tzinfo=None) <= time(9, 35):
            reasons.append("CLAIM_OUTSIDE_LEGAL_START_WINDOW")
        if payload.get("phase") != "morning":
            reasons.append("PHASE_MISMATCH")
        if payload.get("research_only") is not True:
            reasons.append("RESEARCH_ONLY_FALSE")
        if not _execution_fields_disabled(payload):
            reasons.append("EXECUTION_FIELDS_NOT_DISABLED")
        if reasons:
            claim_failures.append(
                {"task_id": payload.get("task_id"), "reasons": reasons}
            )
    claim_contract_valid = len(claims) == 1 and not claim_failures

    codex_failures: list[dict[str, Any]] = []
    pcf_task_results: list[dict[str, Any]] = []
    expected_task_receipt_path = (
        str(task_receipts[0].get("receipt_file") or "")
        if len(task_receipts) == 1
        else ""
    )
    for payload in codex_receipts:
        reasons: list[str] = []
        started_at = _parse_local_timestamp(payload.get("started_at"))
        if payload.get("schema_version") != "1.1.0":
            reasons.append("SCHEMA_VERSION_MISMATCH")
        if payload.get("immutable_receipt") is not True:
            reasons.append("IMMUTABLE_RECEIPT_FALSE")
        if payload.get("dry_run") is not False:
            reasons.append("DRY_RUN_NOT_FALSE")
        if payload.get("phase") != "morning":
            reasons.append("PHASE_MISMATCH")
        late_orchestration = False
        if started_at is None or started_at.date().isoformat() != target_date:
            reasons.append("START_DATE_MISMATCH")
        else:
            orchestration_time = started_at.timetz().replace(tzinfo=None)
            if orchestration_time < time(9, 15):
                reasons.append("BEFORE_MORNING_ORCHESTRATION_WINDOW")
            late_orchestration = orchestration_time > time(9, 35)
        if payload.get("atomic_same_day_claim_enabled") is not True:
            reasons.append("ATOMIC_CLAIM_DISABLED")
        if payload.get("receipt_written") is not True:
            reasons.append("RECEIPT_WRITTEN_FALSE")
        if payload.get("log_written") is not True:
            reasons.append("LOG_WRITTEN_FALSE")
        if payload.get("research_only") is not True:
            reasons.append("RESEARCH_ONLY_FALSE")
        if not _execution_fields_disabled(payload):
            reasons.append("EXECUTION_FIELDS_NOT_DISABLED")
        task_results = payload.get("task_results")
        if not isinstance(task_results, list):
            reasons.append("TASK_RESULTS_INVALID")
        else:
            matching_results = [
                result
                for result in task_results
                if isinstance(result, dict)
                and result.get("task_id") == "PRIMARY_MARKET_PCF_IOPV"
            ]
            if len(task_results) != 1 or len(matching_results) != 1:
                reasons.append("PCF_TASK_RESULT_COUNT_INVALID")
            for result in matching_results:
                pcf_task_results.append(result)
                decision = result.get("decision")
                if decision not in {
                    "RUN_NOW",
                    "ALREADY_CLAIMED_ATOMIC",
                    "ALREADY_ATTEMPTED",
                    "MISSED_START_WINDOW",
                }:
                    reasons.append("PCF_TASK_DECISION_INVALID")
                if result.get("launcher") != task_launcher:
                    reasons.append("PCF_TASK_LAUNCHER_MISMATCH")
                claim_relative = str(result.get("claim_file") or "")
                claim_path = ROOT / claim_relative
                claim_matches = False
                if claim_relative and claim_path.is_file():
                    try:
                        claim_payload = read_json(claim_path)
                        claim_matches = bool(
                            len(claims) == 1 and claim_payload == claims[0]
                        )
                    except Exception:
                        claim_matches = False
                if decision in {"RUN_NOW", "ALREADY_CLAIMED_ATOMIC"}:
                    if not claim_matches:
                        reasons.append("PCF_TASK_CLAIM_FILE_MISMATCH")
                if decision == "RUN_NOW":
                    if result.get("claim_acquired") is not True:
                        reasons.append("PCF_TASK_CLAIM_NOT_ACQUIRED")
                    expected_task_exit = (
                        task_receipts[0].get("task_exit_code")
                        if len(task_receipts) == 1
                        else None
                    )
                    if result.get("exit_code") != expected_task_exit:
                        reasons.append("PCF_TASK_EXIT_CODE_MISMATCH")
                    post_evidence = result.get("post_evidence")
                    if (
                        not expected_task_receipt_path
                        or not isinstance(post_evidence, list)
                        or expected_task_receipt_path not in post_evidence
                    ):
                        reasons.append("PCF_TASK_POST_EVIDENCE_MISMATCH")
                elif decision == "ALREADY_CLAIMED_ATOMIC":
                    if result.get("claim_acquired") is not False:
                        reasons.append("PCF_TASK_DUPLICATE_CLAIM_STATE_INVALID")
                    if result.get("exit_code") is not None:
                        reasons.append("PCF_TASK_DUPLICATE_HAS_EXIT_CODE")
                    if result.get("post_evidence") != []:
                        reasons.append("PCF_TASK_DUPLICATE_HAS_POST_EVIDENCE")
                elif decision in {"ALREADY_ATTEMPTED", "MISSED_START_WINDOW"}:
                    if result.get("claim_file") is not None:
                        reasons.append("PCF_TASK_NOOP_HAS_CLAIM_FILE")
                    if result.get("claim_acquired") is not None:
                        reasons.append("PCF_TASK_NOOP_HAS_CLAIM_STATE")
                    if result.get("exit_code") is not None:
                        reasons.append("PCF_TASK_NOOP_HAS_EXIT_CODE")
                    if result.get("post_evidence") != []:
                        reasons.append("PCF_TASK_NOOP_HAS_POST_EVIDENCE")
            if late_orchestration and any(
                result.get("decision")
                not in {"MISSED_START_WINDOW", "ALREADY_ATTEMPTED"}
                for result in matching_results
            ):
                reasons.append("LATE_ORCHESTRATION_DID_NOT_CLASSIFY_NO_BACKFILL")
        if reasons:
            codex_failures.append(
                {"receipt_id": payload.get("receipt_id"), "reasons": reasons}
            )
    launcher_results = [
        result for result in pcf_task_results if result.get("decision") == "RUN_NOW"
    ]
    codex_contract_valid = bool(
        codex_receipts
        and not codex_failures
        and pcf_task_results
        and len(launcher_results) == 1
    )
    successful_launcher_results = sum(
        result.get("decision") == "RUN_NOW" and result.get("exit_code") == 0
        for result in pcf_task_results
    )
    contract_valid = bool(
        task_contract_valid and claim_contract_valid and codex_contract_valid
    )
    successful = bool(
        contract_valid
        and successful_task_receipts == 1
        and successful_launcher_results == 1
    )
    no_evidence_observed = not task_receipts and not codex_receipts and not claims
    return {
        "status": (
            "NOT_OBSERVED"
            if no_evidence_observed
            else "PASS"
            if successful
            else "FAILED_RUN"
            if contract_valid
            else "FAILED_CONTRACT"
        ),
        "contract_valid": contract_valid,
        "successful": successful,
        "task_receipt_count": len(task_receipts),
        "successful_task_receipt_count": successful_task_receipts,
        "task_failures": task_failures,
        "duplicate_task_receipt_ids": duplicate_task_receipt_ids,
        "claim_count": len(claims),
        "claim_failures": claim_failures,
        "codex_receipt_count": len(codex_receipts),
        "codex_failures": codex_failures,
        "pcf_task_result_count": len(pcf_task_results),
        "actual_launcher_result_count": len(launcher_results),
        "successful_launcher_result_count": successful_launcher_results,
        "task_schema_version": task_schema_version,
        "task_launcher": task_launcher,
        "runtime_patch_id": runtime_patch_id,
    }


def pcf_source_receipt_paths_for_date(target_date: str) -> list[Path]:
    date_directory_name = target_date.replace("-", "")
    if not PCF_SOURCE_RECEIPT_DIRECTORY.is_dir():
        return []
    return sorted(
        path
        for path in PCF_SOURCE_RECEIPT_DIRECTORY.glob(
            f"*/{date_directory_name}/*.json"
        )
        if path.is_file()
    )


def verify_pcf_source_receipts(target_date: str) -> dict[str, Any]:
    """核验当日三类免费来源回执与内容寻址原始响应。"""

    paths = pcf_source_receipt_paths_for_date(target_date)
    source_receipt_counts = {source_id: 0 for source_id in PCF_REQUIRED_SOURCE_IDS}
    source_success_counts = {source_id: 0 for source_id in PCF_REQUIRED_SOURCE_IDS}
    invalid_receipts: list[dict[str, Any]] = []
    verified_raw_response_count = 0
    for path in paths:
        reasons: list[str] = []
        try:
            payload = read_json(path)
        except Exception as exc:
            invalid_receipts.append(
                {"path": path.relative_to(ROOT).as_posix(), "reasons": [str(exc)]}
            )
            continue
        source_id = str(payload.get("source_id") or "")
        if source_id in source_receipt_counts:
            source_receipt_counts[source_id] += 1
        else:
            reasons.append("SOURCE_ID_NOT_ADMITTED_FOR_PCF_V1_2")
        missing_fields = sorted(FREE_SOURCE_RECEIPT_REQUIRED_FIELDS - payload.keys())
        if missing_fields:
            reasons.append(f"MISSING_FIELDS:{','.join(missing_fields)}")
        if payload.get("market_date") != target_date:
            reasons.append("MARKET_DATE_MISMATCH")
        acquired_at = _parse_local_timestamp(payload.get("acquired_at"))
        if acquired_at is None or acquired_at.date().isoformat() != target_date:
            reasons.append("ACQUIRED_DATE_MISMATCH")
        if payload.get("schema_version") != "FREE_SOURCE_ACQUISITION_RECEIPT_V1":
            reasons.append("SCHEMA_VERSION_MISMATCH")
        if payload.get("access_cost_cny") != 0:
            reasons.append("ACCESS_COST_NOT_ZERO")
        if payload.get("paid_data_used") is not False:
            reasons.append("PAID_DATA_USED_NOT_FALSE")
        if payload.get("immutable_receipt") is not True:
            reasons.append("IMMUTABLE_RECEIPT_FALSE")

        quality_status = str(payload.get("quality_status") or "")
        if quality_status == "PASS_RAW_CAPTURED_AND_PARSED":
            raw_relative = str(payload.get("raw_response_path") or "")
            expected_hash = str(payload.get("raw_response_hash") or "")
            raw_path = ROOT / raw_relative
            raw_valid = bool(
                raw_relative
                and expected_hash
                and raw_path.is_file()
                and sha256_file(raw_path) == expected_hash
                and raw_path.stem == expected_hash
            )
            if not raw_valid:
                reasons.append("RAW_RESPONSE_HASH_OR_PATH_INVALID")
            else:
                verified_raw_response_count += 1
                if source_id in source_success_counts:
                    source_success_counts[source_id] += 1
            if payload.get("failure_reason") not in {None, ""}:
                reasons.append("SUCCESS_RECEIPT_HAS_FAILURE_REASON")
        elif quality_status.startswith("FAILED_"):
            if not str(payload.get("failure_reason") or "").strip():
                reasons.append("FAILED_RECEIPT_MISSING_REASON")
        else:
            reasons.append("QUALITY_STATUS_UNKNOWN")
        if reasons:
            invalid_receipts.append(
                {"path": path.relative_to(ROOT).as_posix(), "reasons": reasons}
            )

    missing_source_receipts = sorted(
        source_id
        for source_id, count in source_receipt_counts.items()
        if count == 0
    )
    missing_success_sources = sorted(
        source_id
        for source_id, count in source_success_counts.items()
        if count == 0
    )
    no_evidence_observed = not paths
    contract_valid = bool(paths and not invalid_receipts and not missing_source_receipts)
    all_required_sources_succeeded = bool(
        contract_valid and not missing_success_sources
    )
    return {
        "status": (
            "NOT_OBSERVED"
            if no_evidence_observed
            else "PASS"
            if all_required_sources_succeeded
            else "PARTIAL_VALID"
            if contract_valid
            else "FAILED_CONTRACT"
        ),
        "contract_valid": contract_valid,
        "all_required_sources_succeeded": all_required_sources_succeeded,
        "receipt_count": len(paths),
        "source_receipt_counts": dict(sorted(source_receipt_counts.items())),
        "source_success_counts": dict(sorted(source_success_counts.items())),
        "verified_raw_response_count": verified_raw_response_count,
        "missing_source_receipts": missing_source_receipts,
        "missing_success_sources": missing_success_sources,
        "invalid_receipt_count": len(invalid_receipts),
        "invalid_receipts": invalid_receipts,
    }


def verify_pcf_operational_acceptance(current: datetime) -> dict[str, Any]:
    """核验修复后首20个交易机会的程序成功率和完整质量日比例。"""

    today = current.astimezone(TIMEZONE).date().isoformat()
    current_time = current.astimezone(TIMEZONE).timetz().replace(tzinfo=None)
    with TRADE_CALENDAR.open("r", encoding="utf-8-sig", newline="") as handle:
        calendar_dates = sorted(
            str(row["trade_date"])
            for row in csv.DictReader(handle)
            if str(row.get("trade_date") or "") >= "2026-08-26"
            and (
                str(row["trade_date"]) < today
                or (str(row["trade_date"]) == today and current_time >= time(15, 10))
            )
        )
    opportunity_dates = calendar_dates[:20]

    payloads_by_date: dict[str, list[dict[str, Any]]] = {}
    unreadable_receipts: list[str] = []
    receipt_directories = {
        PCF_TASK_RECEIPT_DIRECTORY.resolve(),
        PCF_TASK_RECEIPT_DIRECTORY_V1_2_1.resolve(),
    }
    for receipt_directory in sorted(receipt_directories, key=str):
        if not receipt_directory.is_dir():
            continue
        for path in sorted(receipt_directory.glob("*.json")):
            try:
                payload = read_json(path)
            except Exception:
                unreadable_receipts.append(path.relative_to(ROOT).as_posix())
                continue
            started_at = _parse_local_timestamp(payload.get("started_at"))
            if started_at is None:
                unreadable_receipts.append(path.relative_to(ROOT).as_posix())
                continue
            payloads_by_date.setdefault(started_at.date().isoformat(), []).append(payload)

    missing_receipt_dates: list[str] = []
    duplicate_receipt_dates: list[str] = []
    invalid_receipt_dates: list[str] = []
    internal_program_success_dates: list[str] = []
    complete_quality_dates: list[str] = []
    external_source_failure_dates: list[str] = []
    for opportunity_date in opportunity_dates:
        day_payloads = payloads_by_date.get(opportunity_date, [])
        if not day_payloads:
            missing_receipt_dates.append(opportunity_date)
            continue
        if len(day_payloads) != 1:
            duplicate_receipt_dates.append(opportunity_date)
            continue
        payload = day_payloads[0]
        run_status = payload.get("run_status")
        collection_status = str(payload.get("collection_status") or "")
        exit_code = payload.get("task_exit_code")
        runtime_contract = pcf_runtime_contract_for_date(opportunity_date)
        basic_contract_valid = bool(
            payload.get("schema_version") == runtime_contract["task_schema_version"]
            and payload.get("immutable_receipt") is True
            and payload.get("task_name") == PCF_TASK_NAME
            and _execution_fields_disabled(payload)
        )
        if runtime_contract["runtime_patch_id"] is not None:
            basic_contract_valid = bool(
                basic_contract_valid
                and payload.get("runtime_patch_id")
                == runtime_contract["runtime_patch_id"]
                and payload.get("python_module_entrypoint") is True
                and payload.get("runner_file") == runtime_contract["forward_runner"]
            )
        if not basic_contract_valid:
            invalid_receipt_dates.append(opportunity_date)
            continue
        if (
            run_status == "SUCCESS"
            and exit_code == 0
            and collection_status
            in {"COMPLETE_QUALITY_DAY", "COLLECTED_INCOMPLETE_QUALITY_DAY"}
        ):
            internal_program_success_dates.append(opportunity_date)
            if collection_status == "COMPLETE_QUALITY_DAY":
                complete_quality_dates.append(opportunity_date)
        elif (
            run_status == "FAILED"
            and exit_code == 3
            and collection_status == "EXTERNAL_FREE_SOURCE_FAILED"
        ):
            internal_program_success_dates.append(opportunity_date)
            external_source_failure_dates.append(opportunity_date)
        elif not (
            run_status == "FAILED"
            and exit_code == 1
            and collection_status.startswith("PROGRAM_FAILED")
        ):
            invalid_receipt_dates.append(opportunity_date)

    opportunity_count = len(opportunity_dates)
    internal_success_rate = (
        len(internal_program_success_dates) / opportunity_count
        if opportunity_count
        else None
    )
    complete_quality_day_rate = (
        len(complete_quality_dates) / opportunity_count if opportunity_count else None
    )
    acceptance_window_complete = opportunity_count >= 20
    internal_program_gate_pass = bool(
        acceptance_window_complete
        and internal_success_rate is not None
        and internal_success_rate >= 0.95
    )
    external_complete_day_gate_pass = bool(
        acceptance_window_complete
        and complete_quality_day_rate is not None
        and complete_quality_day_rate >= 0.90
    )
    return {
        "status": (
            "PASS"
            if internal_program_gate_pass and external_complete_day_gate_pass
            else "PENDING_MATURITY"
            if not acceptance_window_complete
            else "FAILED_INTERNAL_PROGRAM_GATE"
            if not internal_program_gate_pass
            else "FAILED_EXTERNAL_COMPLETE_DAY_GATE"
        ),
        "effective_from": "2026-08-26",
        "required_opportunities": 20,
        "opportunity_count": opportunity_count,
        "opportunity_dates": opportunity_dates,
        "acceptance_window_complete": acceptance_window_complete,
        "internal_program_success_count": len(internal_program_success_dates),
        "internal_program_success_rate": internal_success_rate,
        "minimum_internal_program_success_rate": 0.95,
        "internal_program_gate_pass": internal_program_gate_pass,
        "complete_quality_day_count": len(complete_quality_dates),
        "complete_quality_day_rate": complete_quality_day_rate,
        "minimum_complete_quality_day_rate": 0.90,
        "external_complete_day_gate_pass": external_complete_day_gate_pass,
        "external_source_failure_dates": external_source_failure_dates,
        "missing_receipt_dates": missing_receipt_dates,
        "duplicate_receipt_dates": duplicate_receipt_dates,
        "invalid_receipt_dates": invalid_receipt_dates,
        "unreadable_receipts": unreadable_receipts,
    }


def verify_t_only_daily_evidence(
    receipt_paths: list[Path],
    latest_status: Mapping[str, Any],
    target_date: str,
) -> dict[str, Any]:
    """核验T-only当日不可变回执、公开状态哈希和成熟度输出合同。"""

    receipt_failures: list[dict[str, Any]] = []
    successful_receipts = 0
    failed_but_valid_receipts = 0
    receipt_records: list[dict[str, Any]] = []
    for path in receipt_paths:
        reasons: list[str] = []
        try:
            payload = read_json(path)
        except Exception as exc:
            receipt_failures.append(
                {"path": path.relative_to(ROOT).as_posix(), "reasons": [str(exc)]}
            )
            continue
        run_id = str(payload.get("run_id") or "")
        started_at = _parse_local_timestamp(payload.get("started_at"))
        if payload.get("schema_version") != "1.2.0":
            reasons.append("SCHEMA_VERSION_MISMATCH")
        if payload.get("project_id") != "510300_T_ONLY_FORWARD_V1_1_DAILY_MATURITY_ONLY":
            reasons.append("PROJECT_ID_MISMATCH")
        if not run_id or path.stem != run_id:
            reasons.append("RUN_ID_OR_FILENAME_MISMATCH")
        if started_at is None or started_at.date().isoformat() != target_date:
            reasons.append("START_DATE_MISMATCH")
        elif not time(16, 30) <= started_at.timetz().replace(tzinfo=None) <= time(16, 40):
            reasons.append("OUTSIDE_SCHEDULE_WINDOW")
        manifest_verification = payload.get("manifest_verification") or {}
        if (
            manifest_verification.get("status")
            != "PASS_V1_2_FROZEN_ENTRYPOINT_VERIFIED"
            or manifest_verification.get("manifest_sha256")
            != T_ONLY_MANIFEST_SHA256
        ):
            reasons.append("MANIFEST_VERIFICATION_INVALID")
        forbidden_keys = sorted(walk_keys(payload).intersection(FORBIDDEN_T_ONLY_KEYS))
        if forbidden_keys:
            reasons.append(f"FORBIDDEN_OUTPUT_KEYS:{','.join(forbidden_keys)}")
        public_status_sha256 = (
            sha256_file(T_ONLY_STATUS) if T_ONLY_STATUS.is_file() else None
        )
        if payload.get("public_status_sha256") != public_status_sha256:
            reasons.append("PUBLIC_STATUS_HASH_MISMATCH")

        overall_status = str(payload.get("overall_status") or "")
        exit_code = payload.get("exit_code")
        output_audit = payload.get("output_audit") or {}
        failure = payload.get("failure")
        receipt_success = bool(
            overall_status
            in {
                "SUCCESS_MATURITY_ONLY_COLLECTING",
                "SUCCESS_MATURE_REVIEW_REQUIRED",
            }
            and exit_code == 0
            and output_audit.get("status") == "PASS"
            and output_audit.get("overall_complete") is True
            and output_audit.get("current_view")
            in {"NO_VIEW_UNTIL_FORWARD_MATURITY", "MATURE_REVIEW_REQUIRED"}
            and failure is None
        )
        receipt_failure_valid = bool(
            overall_status == "NO_VIEW_OPERATIONAL_FAILURE"
            and exit_code == 1
            and isinstance(failure, Mapping)
            and failure.get("class")
            and failure.get("stage")
        )
        if not receipt_success and not receipt_failure_valid:
            reasons.append("OVERALL_STATUS_OR_EXIT_CODE_INVALID")
        if not reasons:
            if receipt_success:
                successful_receipts += 1
            else:
                failed_but_valid_receipts += 1
        receipt_records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "run_id": run_id,
                "overall_status": overall_status,
                "exit_code": exit_code,
                "reasons": reasons,
            }
        )
        if reasons:
            receipt_failures.append(
                {"path": path.relative_to(ROOT).as_posix(), "reasons": reasons}
            )

    latest_receipt = latest_status.get("receipt") or {}
    latest_matches = bool(
        len(receipt_records) == 1
        and latest_status.get("run_id") == receipt_records[0].get("run_id")
        and latest_status.get("overall_status")
        == receipt_records[0].get("overall_status")
        and latest_status.get("exit_code") == receipt_records[0].get("exit_code")
        and latest_receipt.get("path") == receipt_records[0].get("path")
        and latest_receipt.get("sha256") == receipt_records[0].get("sha256")
        and latest_receipt.get("immutable_create_mode") == "CREATE_NEW"
    )
    contract_valid = bool(
        len(receipt_paths) == 1
        and len(receipt_records) == 1
        and not receipt_failures
        and latest_matches
    )
    successful = bool(contract_valid and successful_receipts == 1)
    no_evidence_observed = not receipt_paths
    return {
        "status": (
            "NOT_OBSERVED"
            if no_evidence_observed
            else "PASS"
            if successful
            else "FAILED_RUN"
            if contract_valid and failed_but_valid_receipts == 1
            else "FAILED_CONTRACT"
        ),
        "contract_valid": contract_valid,
        "successful": successful,
        "receipt_count": len(receipt_paths),
        "successful_receipt_count": successful_receipts,
        "failed_but_valid_receipt_count": failed_but_valid_receipts,
        "latest_status_matches_receipt": latest_matches,
        "receipt_failure_count": len(receipt_failures),
        "receipt_failures": receipt_failures,
        "receipt_records": receipt_records,
    }


def verify_pcf_maturity_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """从逐日质量记录重算 PCF 的 20/40/80/120 门槛。"""

    failures: list[str] = []
    threshold_fields = {
        "minimum_full_coverage_days": 20,
        "recommended_full_coverage_days": 40,
        "first_unseen_evaluation_full_coverage_days": 80,
        "replication_full_coverage_days": 120,
    }
    for field, expected in threshold_fields.items():
        if payload.get(field) != expected:
            failures.append(f"THRESHOLD_CHANGED:{field}")
    if payload.get("statistical_cluster_unit") != "trade_date":
        failures.append("STATISTICAL_CLUSTER_UNIT_CHANGED")
    quality_contract = payload.get("quality_contract")
    if not isinstance(quality_contract, Mapping):
        quality_contract = {}
        failures.append("QUALITY_CONTRACT_MISSING")
    if quality_contract.get("official_daily_crosscheck_source_id") != (
        "SSE_ETF_DAILY_TURNOVER_OFFICIAL"
    ):
        failures.append("OFFICIAL_CROSSCHECK_SOURCE_CHANGED")
    if quality_contract.get("official_daily_crosscheck_required_from") != (
        "2026-08-26"
    ):
        failures.append("OFFICIAL_CROSSCHECK_START_CHANGED")
    if quality_contract.get("raw_hash_and_receipt_must_verify") is not True:
        failures.append("RAW_HASH_OR_RECEIPT_VERIFICATION_DISABLED")

    daily_quality = payload.get("daily_quality")
    if not isinstance(daily_quality, list):
        daily_quality = []
        failures.append("DAILY_QUALITY_LIST_MISSING")
    trade_dates = [
        str(item.get("trade_date") or "")
        for item in daily_quality
        if isinstance(item, Mapping)
    ]
    if len(trade_dates) != len(daily_quality) or any(not item for item in trade_dates):
        failures.append("TRADE_DATE_INVALID")
    if len(trade_dates) != len(set(trade_dates)):
        failures.append("TRADE_DATE_DUPLICATE")
    for item in daily_quality:
        if not isinstance(item, Mapping):
            continue
        trade_date = str(item.get("trade_date") or "")
        crosscheck = item.get("crosscheck")
        declared_required = bool(
            isinstance(crosscheck, Mapping) and crosscheck.get("required") is True
        )
        if trade_date and declared_required != (trade_date >= "2026-08-26"):
            failures.append(f"CROSSCHECK_REQUIREMENT_MISMATCH:{trade_date}")
    observed_days_recomputed = len(set(trade_dates))
    daily_complete_recomputed: list[bool] = []
    for item in daily_quality:
        if not isinstance(item, Mapping):
            daily_complete_recomputed.append(False)
            continue
        checks = item.get("checks")
        if not isinstance(checks, Mapping) or not checks:
            failures.append(f"QUALITY_CHECKS_MISSING:{item.get('trade_date')}")
            daily_complete_recomputed.append(False)
            continue
        if any(
            value is not True and value is not False for value in checks.values()
        ):
            failures.append(f"QUALITY_CHECK_VALUE_INVALID:{item.get('trade_date')}")
        complete = all(value is True for value in checks.values())
        daily_complete_recomputed.append(complete)
        if item.get("complete_quality_day") is not complete:
            failures.append(f"COMPLETE_QUALITY_FLAG_MISMATCH:{item.get('trade_date')}")
        declared_failure_reasons = sorted(
            str(value) for value in item.get("failure_reasons") or []
        )
        recomputed_failure_reasons = sorted(
            str(name) for name, passed in checks.items() if passed is not True
        )
        if declared_failure_reasons != recomputed_failure_reasons:
            failures.append(f"FAILURE_REASONS_MISMATCH:{item.get('trade_date')}")
    full_days_recomputed = sum(daily_complete_recomputed)
    required_crosscheck_days_recomputed = sum(
        isinstance(item, Mapping)
        and isinstance(item.get("crosscheck"), Mapping)
        and item["crosscheck"].get("required") is True
        for item in daily_quality
    )
    official_check_names = {
        "official_daily_crosscheck_present_once",
        "official_daily_crosscheck_raw_hash_verified",
        "official_daily_crosscheck_receipt_verified",
        "official_final_snapshot_present",
        "official_daily_ohl_match",
        "official_daily_close_match",
    }
    passed_crosscheck_days_recomputed = 0
    legacy_complete_days_recomputed = 0
    for item in daily_quality:
        if not isinstance(item, Mapping):
            continue
        crosscheck = item.get("crosscheck")
        required = bool(
            isinstance(crosscheck, Mapping) and crosscheck.get("required") is True
        )
        if required:
            checks = item.get("checks")
            if isinstance(checks, Mapping) and all(
                checks.get(name) is True for name in official_check_names
            ):
                passed_crosscheck_days_recomputed += 1
        elif all(
            value is True for value in (item.get("checks") or {}).values()
        ):
            legacy_complete_days_recomputed += 1

    declared_pairs = {
        "observed_trading_days": observed_days_recomputed,
        "full_coverage_days": full_days_recomputed,
        "crosscheck_required_day_count": required_crosscheck_days_recomputed,
        "crosscheck_passed_day_count": passed_crosscheck_days_recomputed,
        "legacy_complete_quality_day_count": legacy_complete_days_recomputed,
    }
    for field, recomputed in declared_pairs.items():
        if payload.get(field) != recomputed:
            failures.append(f"COUNT_MISMATCH:{field}")

    eligibility = {
        "eligible_for_quality_audit": full_days_recomputed >= 20,
        "eligible_for_feature_freeze": full_days_recomputed >= 40,
        "eligible_for_first_unseen_evaluation": full_days_recomputed >= 80,
        "eligible_for_research_evaluation": full_days_recomputed >= 80,
        "eligible_for_replication_evaluation": full_days_recomputed >= 120,
    }
    for field, recomputed in eligibility.items():
        if payload.get(field) is not recomputed:
            failures.append(f"ELIGIBILITY_MISMATCH:{field}")

    contract_valid = not failures
    complete = bool(
        contract_valid
        and full_days_recomputed >= 120
        and passed_crosscheck_days_recomputed >= 120
    )
    return {
        "status": (
            "FAILED_CONTRACT"
            if not contract_valid
            else "PASS"
            if complete
            else "PENDING_MATURITY"
        ),
        "contract_valid": contract_valid,
        "complete": complete,
        "observed_trading_days_recomputed": observed_days_recomputed,
        "full_coverage_days_recomputed": full_days_recomputed,
        "crosscheck_required_day_count_recomputed": (
            required_crosscheck_days_recomputed
        ),
        "crosscheck_passed_day_count_recomputed": passed_crosscheck_days_recomputed,
        "legacy_complete_quality_day_count_recomputed": legacy_complete_days_recomputed,
        "eligibility_recomputed": eligibility,
        "failure_count": len(failures),
        "failures": failures,
    }


def verify_industry_maturity_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """独立重算行业原点与非重叠块门槛，避免仅信任 eligible 布尔值。"""

    failures: list[str] = []
    maturity = payload.get("maturity")
    if not isinstance(maturity, Mapping):
        return {
            "status": "FAILED_CONTRACT",
            "contract_valid": False,
            "comparison_mature": False,
            "failures": ["MATURITY_OBJECT_MISSING"],
        }

    def integer(field: str) -> int:
        try:
            value = int(maturity.get(field))
        except (TypeError, ValueError):
            failures.append(f"INVALID_INTEGER:{field}")
            return -1
        if value < 0:
            failures.append(f"NEGATIVE_INTEGER:{field}")
        return value

    origin_count = integer("origin_cluster_count")
    mature_origin_count = integer("mature_origin_cluster_count")
    block_count = integer("non_overlapping_60d_block_count")
    origins = maturity.get("origin_clusters")
    if not isinstance(origins, list):
        origins = []
        failures.append("ORIGIN_CLUSTER_LIST_MISSING")
    origin_ids = [
        str(item.get("origin_cluster") or "")
        for item in origins
        if isinstance(item, Mapping)
    ]
    if len(origin_ids) != len(origins) or any(not item for item in origin_ids):
        failures.append("ORIGIN_CLUSTER_ID_INVALID")
    if len(origin_ids) != len(set(origin_ids)):
        failures.append("ORIGIN_CLUSTER_ID_DUPLICATE")
    if origin_count != len(origin_ids):
        failures.append("ORIGIN_CLUSTER_COUNT_MISMATCH")
    computed_mature_origin_count = sum(
        item.get("mature") is True for item in origins if isinstance(item, Mapping)
    )
    if mature_origin_count != computed_mature_origin_count:
        failures.append("MATURE_ORIGIN_COUNT_MISMATCH")
    if block_count > mature_origin_count:
        failures.append("NON_OVERLAPPING_BLOCK_COUNT_EXCEEDS_MATURE_ORIGINS")
    if any(
        item.get("counts_as_independent_time_samples") != 1
        for item in origins
        if isinstance(item, Mapping)
    ):
        failures.append("ORIGIN_INDEPENDENCE_WEIGHT_INVALID")
    if maturity.get("origin_cluster_definition") != "prediction_date":
        failures.append("ORIGIN_CLUSTER_DEFINITION_CHANGED")
    if maturity.get("independent_time_sample_unit") != "origin_cluster":
        failures.append("INDEPENDENT_SAMPLE_UNIT_CHANGED")
    if maturity.get("industry_rows_count_as_time_samples") is not False:
        failures.append("INDUSTRY_ROWS_COUNTED_AS_TIME_SAMPLES")

    calibration = maturity.get("calibration")
    comparison = maturity.get("model_comparison")
    if not isinstance(calibration, Mapping) or not isinstance(comparison, Mapping):
        failures.append("MATURITY_GATE_OBJECT_MISSING")
        calibration = {}
        comparison = {}
    if not (
        calibration.get("minimum_origin_clusters") == 20
        and calibration.get("minimum_non_overlapping_60d_blocks") == 4
    ):
        failures.append("CALIBRATION_THRESHOLDS_CHANGED")
    if not (
        comparison.get("minimum_origin_clusters") == 40
        and comparison.get("minimum_non_overlapping_60d_blocks") == 8
    ):
        failures.append("COMPARISON_THRESHOLDS_CHANGED")
    calculated_calibration_eligible = bool(origin_count >= 20 and block_count >= 4)
    calculated_comparison_eligible = bool(origin_count >= 40 and block_count >= 8)
    if calibration.get("eligible") is not calculated_calibration_eligible:
        failures.append("CALIBRATION_ELIGIBILITY_MISMATCH")
    if comparison.get("eligible") is not calculated_comparison_eligible:
        failures.append("COMPARISON_ELIGIBILITY_MISMATCH")

    evaluation = payload.get("evaluation_run")
    evaluation_executed = bool(
        isinstance(evaluation, Mapping) and evaluation.get("executed") is True
    )
    if evaluation_executed and not calculated_calibration_eligible:
        failures.append("EVALUATION_EXECUTED_BEFORE_CALIBRATION_GATE")
    if not calculated_calibration_eligible and payload.get("view_status") != "NO_VIEW":
        failures.append("PRE_CALIBRATION_VIEW_NOT_NO_VIEW")
    if maturity.get("original_prediction_state") != "NO_VIEW":
        failures.append("ORIGINAL_PREDICTION_STATE_CHANGED")
    if maturity.get("may_upgrade_original_no_view") is not False:
        failures.append("ORIGINAL_NO_VIEW_UPGRADE_ENABLED")

    contract_valid = not failures
    comparison_mature = bool(contract_valid and calculated_comparison_eligible)
    return {
        "status": (
            "FAILED_CONTRACT"
            if not contract_valid
            else "PASS"
            if comparison_mature
            else "PENDING_MATURITY"
        ),
        "contract_valid": contract_valid,
        "comparison_mature": comparison_mature,
        "origin_cluster_count": origin_count,
        "origin_cluster_count_recomputed": len(origin_ids),
        "mature_origin_cluster_count": mature_origin_count,
        "mature_origin_cluster_count_recomputed": computed_mature_origin_count,
        "non_overlapping_60d_block_count": block_count,
        "calibration_eligible_recomputed": calculated_calibration_eligible,
        "comparison_eligible_recomputed": calculated_comparison_eligible,
        "evaluation_executed": evaluation_executed,
        "failure_count": len(failures),
        "failures": failures,
    }


def verify_t_only_maturity_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """按冻结的 252 日与 3 周期门槛重算 T-only 成熟状态。"""

    failures: list[str] = []
    try:
        new_trading_days = int(payload.get("new_trading_days"))
        closed_cycles = int(payload.get("closed_cycles"))
    except (TypeError, ValueError):
        new_trading_days = -1
        closed_cycles = -1
        failures.append("MATURITY_COUNTS_INVALID")
    if new_trading_days < 0 or closed_cycles < 0:
        failures.append("MATURITY_COUNTS_NEGATIVE")
    if closed_cycles > new_trading_days:
        failures.append("CLOSED_CYCLES_EXCEED_TRADING_DAYS")

    maturity = payload.get("maturity")
    if not isinstance(maturity, Mapping):
        maturity = {}
        failures.append("MATURITY_OBJECT_MISSING")
    if not (
        maturity.get("trading_days_required") == 252
        and maturity.get("closed_cycles_required") == 3
    ):
        failures.append("MATURITY_THRESHOLDS_CHANGED")
    maturity_recomputed = bool(new_trading_days >= 252 and closed_cycles >= 3)
    if maturity.get("mature") is not maturity_recomputed:
        failures.append("MATURITY_FLAG_MISMATCH")

    completeness = payload.get("completeness")
    if not isinstance(completeness, Mapping):
        completeness = {}
        failures.append("COMPLETENESS_OBJECT_MISSING")
    ledger = completeness.get("ledger")
    if isinstance(ledger, Mapping) and ledger.get("complete") is True:
        if (
            ledger.get("expected_rows") != new_trading_days
            or ledger.get("actual_rows") != new_trading_days
            or ledger.get("duplicate_dates") != 0
        ):
            failures.append("COMPLETE_LEDGER_COUNT_MISMATCH")
    elif maturity_recomputed:
        failures.append("MATURE_LEDGER_NOT_COMPLETE")

    current_view = payload.get("current_view")
    evaluation_status = payload.get("evaluation_status")
    if maturity_recomputed:
        if current_view != "MATURE_REVIEW_REQUIRED":
            failures.append("MATURE_VIEW_STATUS_INVALID")
        if evaluation_status != "MATURE_THRESHOLD_REACHED_REVIEW_SEPARATELY":
            failures.append("MATURE_EVALUATION_STATUS_INVALID")
    else:
        if current_view not in {
            "NO_VIEW_NO_FORWARD_TRADING_DAY",
            "NO_VIEW_UNTIL_FORWARD_MATURITY",
            "NO_VIEW_OPERATIONAL_FAILURE",
        }:
            failures.append("PRE_MATURITY_VIEW_STATUS_INVALID")
        if evaluation_status not in {
            "NOT_EVALUATED_BEFORE_MATURITY",
            "NOT_EVALUATED_OPERATIONAL_FAILURE",
        }:
            failures.append("PRE_MATURITY_EVALUATION_STATUS_INVALID")

    integrity_complete = completeness.get("overall_complete") is True
    contract_valid = not failures
    complete = bool(contract_valid and maturity_recomputed and integrity_complete)
    return {
        "status": (
            "FAILED_CONTRACT"
            if not contract_valid
            else "PASS"
            if complete
            else "MATURE_PENDING_CURRENT_INTEGRITY"
            if maturity_recomputed
            else "PENDING_MATURITY"
        ),
        "contract_valid": contract_valid,
        "complete": complete,
        "new_trading_days": new_trading_days,
        "closed_cycles": closed_cycles,
        "maturity_recomputed": maturity_recomputed,
        "integrity_complete": integrity_complete,
        "current_view": current_view,
        "evaluation_status": evaluation_status,
        "failure_count": len(failures),
        "failures": failures,
    }


def supervisor_close_task_windows(
    supervisor_config_path: Path,
) -> dict[str, tuple[time, time]]:
    payload = yaml.safe_load(supervisor_config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tasks"), list):
        raise RuntimeError(f"监督器任务配置无效：{supervisor_config_path}")
    required_ids = {"INDUSTRY_EXPECTATION_GAP", "PRIORITY_FORWARD_STATUS"}
    windows: dict[str, tuple[time, time]] = {}
    for task in payload["tasks"]:
        if not isinstance(task, Mapping) or task.get("id") not in required_ids:
            continue
        task_id = str(task["id"])
        values: list[time] = []
        for field in ("schedule", "latest_start"):
            raw = str(task.get(field) or "")
            match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", raw)
            if match is None:
                raise RuntimeError(
                    f"监督器 {task_id} 的 {field} 不是 HH:MM：{raw!r}"
                )
            values.append(time(int(match.group(1)), int(match.group(2))))
        if values[1] < values[0]:
            raise RuntimeError(f"监督器 {task_id} 的 latest_start 早于 schedule")
        windows[task_id] = (values[0], values[1])
    if set(windows) != required_ids:
        raise RuntimeError(
            f"监督器缺少收盘任务窗口：{sorted(required_ids - set(windows))}"
        )
    return windows


def verify_close_acceptance_evidence(
    industry_receipt_paths: list[Path],
    status_receipt_paths: list[Path],
    codex_receipt_paths: list[Path],
    claims: list[dict[str, Any]],
    central_status: Mapping[str, Any],
    target_date: str,
    close_task_windows: Mapping[str, tuple[time, time]] | None = None,
) -> dict[str, Any]:
    """核验收盘编排、两项任务回执和权威状态快照的生成顺序。"""

    task_windows = dict(
        close_task_windows
        or {
            "INDUSTRY_EXPECTATION_GAP": (time(17, 5), time(23, 30)),
            "PRIORITY_FORWARD_STATUS": (time(17, 15), time(23, 50)),
        }
    )
    industry_window = task_windows["INDUSTRY_EXPECTATION_GAP"]
    status_window = task_windows["PRIORITY_FORWARD_STATUS"]
    codex_window = (
        min(industry_window[0], status_window[0]),
        min(industry_window[1], status_window[1]),
    )
    structural_failures: list[dict[str, Any]] = []
    industry_payload: dict[str, Any] = {}
    industry_acceptable = False
    industry_receipt_relative: str | None = None
    if len(industry_receipt_paths) != 1:
        structural_failures.append(
            {"part": "industry", "reason": "RECEIPT_COUNT_NOT_ONE"}
        )
    else:
        path = industry_receipt_paths[0]
        industry_receipt_relative = path.relative_to(ROOT).as_posix()
        reasons: list[str] = []
        try:
            industry_payload = read_json(path)
        except Exception as exc:
            reasons.append(str(exc))
        if industry_payload:
            started_at = _parse_local_timestamp(industry_payload.get("started_at"))
            if industry_payload.get("schema_version") != "1.2.0":
                reasons.append("SCHEMA_VERSION_MISMATCH")
            if industry_payload.get("immutable_receipt") is not True:
                reasons.append("IMMUTABLE_RECEIPT_FALSE")
            if industry_payload.get("task_name") != (
                "Codex-Industry-Expectation-Gap-Forward-Operations-V1.2"
            ):
                reasons.append("TASK_NAME_MISMATCH")
            if started_at is None or started_at.date().isoformat() != target_date:
                reasons.append("START_DATE_MISMATCH")
            elif not industry_window[0] <= started_at.timetz().replace(
                tzinfo=None
            ) <= industry_window[1]:
                reasons.append("OUTSIDE_CLOSE_WINDOW")
            if industry_payload.get("paid_provider_call_enabled") is not False:
                reasons.append("PAID_PROVIDER_CALL_ENABLED")
            if industry_payload.get("shared_latest_consumed_by_evaluation") is not False:
                reasons.append("SHARED_LATEST_CONSUMED")
            if not _execution_fields_disabled(industry_payload):
                reasons.append("EXECUTION_FIELDS_NOT_DISABLED")
            run_status = industry_payload.get("run_status")
            collection_status = industry_payload.get("collection_status")
            exit_code = industry_payload.get("task_exit_code")
            industry_acceptable = bool(
                (
                    run_status == "SUCCESS"
                    and collection_status == "COMPLETED_ZERO_PAID_INPUT_VERIFIED"
                    and exit_code == 0
                )
                or (
                    run_status == "FAILED"
                    and collection_status == "EXTERNAL_FREE_SOURCE_FAILED"
                    and exit_code == 3
                )
            )
            if not (
                industry_acceptable
                or (
                    run_status == "FAILED"
                    and str(collection_status).startswith("PROGRAM_FAILED")
                    and exit_code == 1
                )
            ):
                reasons.append("RUN_STATUS_OR_EXIT_CODE_INVALID")
        if reasons:
            structural_failures.append({"part": "industry", "reasons": reasons})

    status_payload: dict[str, Any] = {}
    status_task_success = False
    status_receipt_relative: str | None = None
    status_snapshot_valid = False
    if len(status_receipt_paths) != 1:
        structural_failures.append(
            {"part": "authoritative_status", "reason": "RECEIPT_COUNT_NOT_ONE"}
        )
    else:
        path = status_receipt_paths[0]
        status_receipt_relative = path.relative_to(ROOT).as_posix()
        reasons = []
        try:
            status_payload = read_json(path)
        except Exception as exc:
            reasons.append(str(exc))
        if status_payload:
            started_at = _parse_local_timestamp(status_payload.get("started_at"))
            ended_at = _parse_local_timestamp(status_payload.get("ended_at"))
            if status_payload.get("schema_version") != "1.6.0":
                reasons.append("SCHEMA_VERSION_MISMATCH")
            if status_payload.get("immutable_receipt") is not True:
                reasons.append("IMMUTABLE_RECEIPT_FALSE")
            if status_payload.get("task_name") != (
                "Codex-Priority-Forward-Authoritative-Status-V1.6"
            ):
                reasons.append("TASK_NAME_MISMATCH")
            if started_at is None or started_at.date().isoformat() != target_date:
                reasons.append("START_DATE_MISMATCH")
            elif not status_window[0] <= started_at.timetz().replace(
                tzinfo=None
            ) <= status_window[1]:
                reasons.append("OUTSIDE_CLOSE_WINDOW")
            if not _execution_fields_disabled(status_payload):
                reasons.append("EXECUTION_FIELDS_NOT_DISABLED")
            status_task_success = bool(
                status_payload.get("run_status") == "SUCCESS"
                and status_payload.get("task_exit_code") == 0
                and status_payload.get("overall_research_status")
                == "NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET"
                and status_payload.get("decision") == "PAUSE_AND_FIX_FOUNDATION"
            )
            if not (
                status_task_success
                or (
                    status_payload.get("run_status") == "FAILED"
                    and status_payload.get("task_exit_code") == 1
                )
            ):
                reasons.append("RUN_STATUS_OR_EXIT_CODE_INVALID")
            snapshot_relative = str(status_payload.get("report_snapshot_file") or "")
            snapshot_path = ROOT / snapshot_relative
            central_generated_at = _parse_local_timestamp(
                central_status.get("generated_at")
            )
            status_snapshot_valid = bool(
                status_task_success
                and snapshot_relative
                and snapshot_path.is_file()
                and CENTRAL_STATUS.is_file()
                and sha256_file(snapshot_path) == sha256_file(CENTRAL_STATUS)
                and central_generated_at is not None
                and started_at is not None
                and ended_at is not None
                and started_at <= central_generated_at <= ended_at
                and central_status.get("overall_research_status")
                == status_payload.get("overall_research_status")
                and central_status.get("decision") == status_payload.get("decision")
            )
            if status_task_success and not status_snapshot_valid:
                reasons.append("AUTHORITATIVE_SNAPSHOT_OR_GENERATION_ORDER_INVALID")
        if reasons:
            structural_failures.append(
                {"part": "authoritative_status", "reasons": reasons}
            )

    claim_failures: list[dict[str, Any]] = []
    claim_task_ids: list[str] = []
    for payload in claims:
        reasons = []
        task_id = str(payload.get("task_id") or "")
        claim_task_ids.append(task_id)
        claimed_at = _parse_local_timestamp(payload.get("claimed_at"))
        if payload.get("schema_version") != "1.0.0":
            reasons.append("SCHEMA_VERSION_MISMATCH")
        if payload.get("claim_status") != "ATOMIC_ATTEMPT_CLAIMED":
            reasons.append("CLAIM_STATUS_INVALID")
        if payload.get("immutable_receipt") is not True:
            reasons.append("IMMUTABLE_RECEIPT_FALSE")
        if payload.get("target_date") != target_date or payload.get("phase") != "close":
            reasons.append("DATE_OR_PHASE_MISMATCH")
        if claimed_at is None or claimed_at.date().isoformat() != target_date:
            reasons.append("CLAIMED_AT_DATE_MISMATCH")
        else:
            claim_window = task_windows.get(task_id)
            if claim_window is None:
                reasons.append("CLAIM_TASK_WINDOW_MISSING")
            elif not claim_window[0] <= claimed_at.timetz().replace(
                tzinfo=None
            ) <= claim_window[1]:
                reasons.append("CLAIM_OUTSIDE_CLOSE_WINDOW")
        if payload.get("research_only") is not True or not _execution_fields_disabled(payload):
            reasons.append("RESEARCH_BOUNDARY_INVALID")
        if reasons:
            claim_failures.append({"task_id": task_id, "reasons": reasons})
    expected_close_task_ids = {
        "INDUSTRY_EXPECTATION_GAP",
        "PRIORITY_FORWARD_STATUS",
    }
    claims_valid = bool(
        len(claims) == 2
        and set(claim_task_ids) == expected_close_task_ids
        and not claim_failures
    )

    codex_failures: list[dict[str, Any]] = []
    codex_results: dict[str, dict[str, Any]] = {}
    expected_receipt_by_task = {
        "INDUSTRY_EXPECTATION_GAP": industry_receipt_relative,
        "PRIORITY_FORWARD_STATUS": status_receipt_relative,
    }
    expected_exit_by_task = {
        "INDUSTRY_EXPECTATION_GAP": industry_payload.get("task_exit_code"),
        "PRIORITY_FORWARD_STATUS": status_payload.get("task_exit_code"),
    }
    expected_launcher_by_task = {
        "INDUSTRY_EXPECTATION_GAP": (
            "scripts/run_industry_expectation_gap_forward_operations_task_v1_2.ps1"
        ),
        "PRIORITY_FORWARD_STATUS": (
            "scripts/run_priority_forward_authoritative_status_task_v1_6.ps1"
        ),
    }
    if len(codex_receipt_paths) != 1:
        structural_failures.append(
            {"part": "codex_close", "reason": "RECEIPT_COUNT_NOT_ONE"}
        )
    else:
        path = codex_receipt_paths[0]
        reasons = []
        try:
            payload = read_json(path)
        except Exception as exc:
            payload = {}
            reasons.append(str(exc))
        if payload:
            started_at = _parse_local_timestamp(payload.get("started_at"))
            ended_at = _parse_local_timestamp(payload.get("ended_at"))
            if payload.get("schema_version") != "1.1.0":
                reasons.append("SCHEMA_VERSION_MISMATCH")
            if not payload.get("receipt_id"):
                reasons.append("RECEIPT_ID_MISSING")
            if payload.get("immutable_receipt") is not True or payload.get("dry_run") is not False:
                reasons.append("IMMUTABLE_OR_DRY_RUN_INVALID")
            if payload.get("phase") != "close":
                reasons.append("PHASE_MISMATCH")
            if started_at is None or started_at.date().isoformat() != target_date:
                reasons.append("START_DATE_MISMATCH")
            elif not codex_window[0] <= started_at.timetz().replace(
                tzinfo=None
            ) <= codex_window[1]:
                reasons.append("OUTSIDE_CLOSE_WINDOW")
            if ended_at is None or started_at is None or ended_at < started_at:
                reasons.append("END_TIMESTAMP_INVALID")
            if payload.get("atomic_same_day_claim_enabled") is not True:
                reasons.append("ATOMIC_CLAIM_DISABLED")
            if payload.get("receipt_written") is not True or payload.get("log_written") is not True:
                reasons.append("RECEIPT_OR_LOG_NOT_WRITTEN")
            if payload.get("research_only") is not True or not _execution_fields_disabled(payload):
                reasons.append("RESEARCH_BOUNDARY_INVALID")
            receipt_relative = path.relative_to(ROOT).as_posix()
            if payload.get("receipt_file") != receipt_relative:
                reasons.append("RECEIPT_FILE_SELF_REFERENCE_MISMATCH")
            log_relative = str(payload.get("log_file") or "")
            if not log_relative or not (ROOT / log_relative).is_file():
                reasons.append("LOG_FILE_MISSING")
            task_results = payload.get("task_results")
            if not isinstance(task_results, list):
                reasons.append("TASK_RESULTS_INVALID")
            else:
                task_result_ids = [
                    str(result.get("task_id") or "")
                    for result in task_results
                    if isinstance(result, Mapping)
                ]
                codex_results = {
                    str(result.get("task_id")): result
                    for result in task_results
                    if isinstance(result, Mapping)
                }
                if (
                    len(task_results) != 2
                    or len(task_result_ids) != 2
                    or set(task_result_ids) != expected_close_task_ids
                    or len(set(task_result_ids)) != 2
                ):
                    reasons.append("TASK_RESULT_IDS_MISMATCH")
                for task_id, result in codex_results.items():
                    if result.get("decision") != "RUN_NOW":
                        reasons.append(f"TASK_DECISION_NOT_RUN_NOW:{task_id}")
                    if result.get("launcher") != expected_launcher_by_task.get(task_id):
                        reasons.append(f"TASK_LAUNCHER_MISMATCH:{task_id}")
                    if result.get("exit_code") != expected_exit_by_task.get(task_id):
                        reasons.append(f"TASK_EXIT_CODE_MISMATCH:{task_id}")
                    if result.get("claim_acquired") is not True:
                        reasons.append(f"TASK_CLAIM_NOT_ACQUIRED:{task_id}")
                    claim_relative = str(result.get("claim_file") or "")
                    claim_path = ROOT / claim_relative
                    matching_claim = False
                    if claim_relative and claim_path.is_file():
                        try:
                            claim_payload = read_json(claim_path)
                            matching_claim = any(
                                claim_payload == candidate for candidate in claims
                            )
                        except Exception:
                            matching_claim = False
                    if not matching_claim:
                        reasons.append(f"TASK_CLAIM_FILE_MISMATCH:{task_id}")
                    expected_post_evidence = expected_receipt_by_task.get(task_id)
                    post_evidence = result.get("post_evidence")
                    if (
                        not expected_post_evidence
                        or not isinstance(post_evidence, list)
                        or expected_post_evidence not in post_evidence
                    ):
                        reasons.append(f"TASK_POST_EVIDENCE_MISMATCH:{task_id}")
                industry_exit = codex_results.get(
                    "INDUSTRY_EXPECTATION_GAP", {}
                ).get("exit_code")
                status_exit = codex_results.get("PRIORITY_FORWARD_STATUS", {}).get(
                    "exit_code"
                )
                expected_top_exit = (
                    0 if industry_exit == 0 and status_exit == 0 else 1
                )
                expected_top_status = (
                    "SUCCESS_OR_ALREADY_ATTEMPTED"
                    if expected_top_exit == 0
                    else "FAILED"
                )
                if payload.get("exit_code") != expected_top_exit:
                    reasons.append("TOP_LEVEL_EXIT_CODE_MISMATCH")
                if payload.get("status") != expected_top_status:
                    reasons.append("TOP_LEVEL_STATUS_MISMATCH")
        if reasons:
            codex_failures.append(
                {"path": path.relative_to(ROOT).as_posix(), "reasons": reasons}
            )
            structural_failures.append({"part": "codex_close", "reasons": reasons})

    codex_results_acceptable = bool(
        codex_results.get("PRIORITY_FORWARD_STATUS", {}).get("exit_code") == 0
        and codex_results.get("INDUSTRY_EXPECTATION_GAP", {}).get("exit_code")
        in {0, 3}
    )
    central_industry_latest_path = None
    for stream in central_status.get("active_research_streams") or []:
        if stream.get("id") == "INDUSTRY_EXPECTATION_GAP":
            central_industry_latest_path = (stream.get("latest_task") or {}).get("path")
    central_uses_current_industry_receipt = bool(
        industry_receipt_relative
        and central_industry_latest_path == industry_receipt_relative
    )
    if status_task_success and not central_uses_current_industry_receipt:
        structural_failures.append(
            {"part": "central_status", "reason": "CURRENT_INDUSTRY_RECEIPT_NOT_CONSUMED"}
        )

    no_evidence_observed = not (
        industry_receipt_paths or status_receipt_paths or codex_receipt_paths or claims
    )
    contract_valid = bool(
        not structural_failures and claims_valid and not codex_failures
    )
    successful = bool(
        contract_valid
        and industry_acceptable
        and status_task_success
        and status_snapshot_valid
        and codex_results_acceptable
        and central_uses_current_industry_receipt
    )
    return {
        "status": (
            "NOT_OBSERVED"
            if no_evidence_observed
            else "PASS"
            if successful
            else "FAILED_RUN"
            if contract_valid
            else "FAILED_CONTRACT"
        ),
        "contract_valid": contract_valid,
        "successful": successful,
        "industry_receipt_count": len(industry_receipt_paths),
        "industry_acceptable": industry_acceptable,
        "status_receipt_count": len(status_receipt_paths),
        "status_task_success": status_task_success,
        "status_snapshot_valid": status_snapshot_valid,
        "codex_receipt_count": len(codex_receipt_paths),
        "codex_results_acceptable": codex_results_acceptable,
        "claim_count": len(claims),
        "claims_valid": claims_valid,
        "central_industry_latest_path": central_industry_latest_path,
        "central_uses_current_industry_receipt": central_uses_current_industry_receipt,
        "structural_failure_count": len(structural_failures),
        "structural_failures": structural_failures,
        "task_windows": {
            task_id: {
                "schedule": window[0].strftime("%H:%M"),
                "latest_start": window[1].strftime("%H:%M"),
            }
            for task_id, window in sorted(task_windows.items())
        },
        "codex_close_start_window": {
            "schedule": codex_window[0].strftime("%H:%M"),
            "latest_start": codex_window[1].strftime("%H:%M"),
        },
    }


def task_exact(
    snapshot: Mapping[str, Any], *, runner: str, manifest_sha256: str
) -> bool:
    arguments = str(snapshot.get("arguments") or "")
    return bool(
        snapshot.get("query_status") == "PASS"
        and snapshot.get("state") in {"Ready", "Running"}
        and runner in arguments
        and manifest_sha256 in arguments
        and int(snapshot.get("restart_count", -1)) == 0
        and snapshot.get("start_when_available") is False
        and snapshot.get("multiple_instances") == "IgnoreNew"
    )


def requirement(
    requirement_id: str,
    title: str,
    status: str,
    current_state: Any,
    evidence_items: Iterable[Any],
    next_action: str | None = None,
) -> dict[str, Any]:
    if status not in SATISFIED_STATUSES | PENDING_STATUSES | CONTRADICTION_STATUSES:
        raise ValueError(f"未知完成状态：{status}")
    return {
        "id": requirement_id,
        "title": title,
        "status": status,
        "satisfied": status in SATISFIED_STATUSES,
        "current_state": current_state,
        "evidence": list(evidence_items),
        "next_action": next_action,
    }


def build_audit(now: datetime | None = None) -> dict[str, Any]:
    current = now or datetime.now(TIMEZONE)
    today = current.date().isoformat()
    is_trading_day = trading_day(today)

    attachment_records = []
    for item in ATTACHMENTS:
        path = Path(item["path"])
        record = evidence(path)
        record["expected_sha256"] = item["sha256"]
        record["hash_matches"] = record["sha256"] == item["sha256"]
        record["precedence"] = item["precedence"]
        attachment_records.append(record)

    runtime = active_runtime_contract()
    active_manifest_path = Path(runtime["manifest_path"])
    active_manifest_sha256 = str(runtime["manifest_sha256"])
    active_supervisor_config_path = Path(runtime["supervisor_config_path"])
    active_runner = str(runtime["runner"])
    active_correction_audit_path = Path(runtime["correction_audit_path"])
    active_test_boundary_audit_path = Path(runtime["test_boundary_audit_path"])
    active_schedule_surface_audit_path = Path(runtime["schedule_surface_audit_path"])
    active_pcf_config_contract = pcf_runtime_contract_for_date(
        PCF_V1_8_EFFECTIVE_DATE, str(runtime["runtime_version"])
    )
    runtime_state = {
        key: (
            value.relative_to(ROOT).as_posix()
            if isinstance(value, Path) and value.is_relative_to(ROOT)
            else str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in runtime.items()
    }

    central = read_json(CENTRAL_STATUS)
    active_manifest = read_json(active_manifest_path)
    active_manifest_check = verify_manifest(active_manifest_path)
    test_boundary_check = verify_runtime_test_boundary(
        active_manifest_path, active_supervisor_config_path
    )
    active_correction = read_json(active_correction_audit_path)
    active_test_boundary = read_json(active_test_boundary_audit_path)
    active_schedule_surface_audit = read_json(active_schedule_surface_audit_path)
    pcf = read_json(PCF_READINESS)
    pcf_maturity_contract = verify_pcf_maturity_contract(pcf)
    registry_rows = read_csv_rows(SOURCE_REGISTRY)
    zero_paid_contract = verify_zero_paid_data_contract(
        ZERO_PAID_POLICY, CANONICAL_SOURCE_REGISTRY, SOURCE_REGISTRY
    )
    frozen_inputs = read_json(FROZEN_INPUT_AUDIT)
    industry = read_json(INDUSTRY_STATUS)
    industry_frozen_boundary = verify_industry_frozen_input_boundary(
        frozen_inputs,
        industry,
        active_supervisor_config_path,
        pcf_task_launcher=str(active_pcf_config_contract["task_launcher"]),
        pcf_evidence_glob=(
            Path(active_pcf_config_contract["task_receipt_directory"])
            .relative_to(ROOT)
            .as_posix()
            + "/*.json"
        ),
    )
    option = read_json(OPTION_AUDIT)
    cash = read_json(CASH_AUDIT)
    cb_sufficiency = read_json(CB_SUFFICIENCY_RECEIPT)
    cb_archive_check = verify_file_map_manifest(CB_SUFFICIENCY_ARCHIVE)
    cb_lifecycle = read_json(CB_LIFECYCLE_RECEIPT)
    cb_lifecycle_archive_check = verify_file_map_manifest(CB_LIFECYCLE_ARCHIVE)
    with CB_SOURCE_AVAILABILITY.open("r", encoding="utf-8-sig", newline="") as handle:
        cb_source_rows = list(csv.DictReader(handle))
    windows = read_json(WINDOWS_PERSISTENCE_AUDIT)
    t_only = read_json(T_ONLY_STATUS)
    t_only_manifest_check = verify_manifest(T_ONLY_MANIFEST)
    pcf_task = windows_task_snapshot(PCF_TASK_NAME)
    t_only_task = windows_task_snapshot(T_ONLY_TASK_NAME)
    automation_specs = [
        {
            **item,
            "required_runner": active_runner,
            "required_manifest_sha256": active_manifest_sha256,
        }
        for item in AUTOMATIONS
    ]
    automation_records = [automation_snapshot(item) for item in automation_specs]
    active_schedule_surface = verify_active_schedule_surface(
        pcf_task,
        t_only_task,
        pcf_runner=active_runner,
        pcf_manifest_sha256=active_manifest_sha256,
    )

    active_streams = central["active_research_streams"]
    active_ids = [str(item["id"]) for item in active_streams]
    contract = central["contract"]
    execution = central["execution_authorization"]
    industry_maturity = industry["maturity"]
    industry_maturity_contract = verify_industry_maturity_contract(industry)
    option_closure = option["closure"]
    cash_terminal = cash["terminal_decision"]
    t_only_leaks = sorted(walk_keys(t_only).intersection(FORBIDDEN_T_ONLY_KEYS))
    t_only_maturity_contract = verify_t_only_maturity_contract(t_only)

    pcf_runtime_contract = pcf_runtime_contract_for_date(
        today, str(runtime["runtime_version"])
    )
    active_pcf_collection_task_path = ROOT / str(
        pcf_runtime_contract["task_launcher"]
    )
    pcf_task_receipt_directory = pcf_runtime_contract["task_receipt_directory"]
    pcf_codex_receipt_directory = pcf_runtime_contract["codex_receipt_directory"]
    pcf_claim_directory = pcf_runtime_contract["claim_directory"]
    pcf_receipts = receipts_for_local_date(pcf_task_receipt_directory, today)
    pcf_codex_receipts = receipts_for_local_date(
        pcf_codex_receipt_directory, today, phase="morning"
    )
    pcf_claims = receipts_for_local_date(
        pcf_claim_directory,
        today,
        phase="morning",
        timestamp_fields=("claimed_at",),
    )
    pcf_run_evidence = verify_pcf_atomic_run_evidence(
        receipt_payloads_for_local_date(pcf_task_receipt_directory, today),
        receipt_payloads_for_local_date(
            pcf_codex_receipt_directory, today, phase="morning"
        ),
        receipt_payloads_for_local_date(
            pcf_claim_directory,
            today,
            phase="morning",
            timestamp_fields=("claimed_at",),
        ),
        today,
        readiness=pcf,
        task_schema_version=str(pcf_runtime_contract["task_schema_version"]),
        task_launcher=str(pcf_runtime_contract["task_launcher"]),
        forward_runner=str(pcf_runtime_contract["forward_runner"]),
        runtime_patch_id=pcf_runtime_contract["runtime_patch_id"],
    )
    pcf_source_receipt_audit = verify_pcf_source_receipts(today)
    pcf_source_receipt_evidence = [
        evidence(path) for path in pcf_source_receipt_paths_for_date(today)
    ]
    pcf_operational_acceptance = verify_pcf_operational_acceptance(current)
    t_only_receipt_paths = receipt_paths_for_local_date(
        T_ONLY_RECEIPT_DIRECTORY, today
    )
    t_only_receipts = [evidence(path) for path in t_only_receipt_paths]
    t_only_latest_run_status = (
        read_json(T_ONLY_RUN_STATUS) if T_ONLY_RUN_STATUS.is_file() else {}
    )
    t_only_daily_evidence = verify_t_only_daily_evidence(
        t_only_receipt_paths, t_only_latest_run_status, today
    )
    close_industry_receipt_paths = receipt_paths_for_local_date(
        INDUSTRY_TASK_RECEIPT_DIRECTORY, today
    )
    close_status_receipt_paths = receipt_paths_for_local_date(
        AUTHORITATIVE_STATUS_TASK_RECEIPT_DIRECTORY, today
    )
    close_codex_receipt_paths = receipt_paths_for_local_date(
        pcf_codex_receipt_directory, today, phase="close"
    )
    close_claim_paths = receipt_paths_for_local_date(
        pcf_claim_directory,
        today,
        phase="close",
        timestamp_fields=("claimed_at",),
    )
    close_acceptance = verify_close_acceptance_evidence(
        close_industry_receipt_paths,
        close_status_receipt_paths,
        close_codex_receipt_paths,
        receipt_payloads_for_local_date(
            pcf_claim_directory,
            today,
            phase="close",
            timestamp_fields=("claimed_at",),
        ),
        central,
        today,
        supervisor_close_task_windows(active_supervisor_config_path),
    )
    close_evidence_paths = [
        *close_industry_receipt_paths,
        *close_status_receipt_paths,
        *close_codex_receipt_paths,
        *close_claim_paths,
    ]

    before_pcf_window = current.timetz().replace(tzinfo=None) < time(9, 25)
    before_t_only_window = current.timetz().replace(tzinfo=None) < time(16, 30)
    before_close_window = current.timetz().replace(tzinfo=None) < time(19, 0)

    requirements: list[dict[str, Any]] = []
    requirements.append(
        requirement(
            "R01",
            "两份用户文本均完整读取且第二份冲突时优先",
            "PASS" if all(item["hash_matches"] for item in attachment_records) else "CONTRADICTION",
            {"attachment_count": len(attachment_records), "precedence": "pasted-text-2"},
            attachment_records,
        )
    )
    requirements.append(
        requirement(
            "R02",
            "总体裁决保持无已验证可交易 Alpha/强 Beta，并暂停修复基础设施",
            "PASS"
            if central.get("overall_research_status")
            == "NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET"
            and central.get("decision") == "PAUSE_AND_FIX_FOUNDATION"
            else "CONTRADICTION",
            {
                "overall_research_status": central.get("overall_research_status"),
                "decision": central.get("decision"),
            },
            [evidence(CENTRAL_STATUS)],
        )
    )
    budget_pass = bool(
        contract.get("data_purchase_budget_cny") == 0
        and int(contract.get("zero_purchase_lock_days", 0)) >= 180
        and all(float(row.get("access_cost_cny") or 0) == 0 for row in registry_rows)
        and zero_paid_contract["policy_status"] == "PASS"
    )
    requirements.append(
        requirement(
            "R03",
            "至少 180 天零付费数据合同",
            "PASS" if budget_pass else "CONTRADICTION",
            {
                "budget_cny": contract.get("data_purchase_budget_cny"),
                "lock_days": contract.get("zero_purchase_lock_days"),
                "registry_rows": len(registry_rows),
                "nonzero_cost_rows": sum(
                    float(row.get("access_cost_cny") or 0) != 0 for row in registry_rows
                ),
                "policy_status": zero_paid_contract["policy_status"],
                "policy_sha256": zero_paid_contract["policy_sha256"],
                "missing_policy_tokens": zero_paid_contract["missing_policy_tokens"],
            },
            [
                evidence(CENTRAL_STATUS),
                evidence(ZERO_PAID_POLICY),
                evidence(SOURCE_REGISTRY),
            ],
        )
    )
    stream_pass = bool(
        int(contract.get("maximum_active_research_streams", -1)) == 2
        and int(contract.get("active_research_stream_count", -1)) == 2
        and active_ids == ["PRIMARY_MARKET_PCF_IOPV", "INDUSTRY_EXPECTATION_GAP"]
    )
    requirements.append(
        requirement(
            "R04",
            "活动研究流最多两条",
            "PASS" if stream_pass else "CONTRADICTION",
            {"maximum": 2, "active_ids": active_ids},
            [evidence(CENTRAL_STATUS), evidence(active_manifest_path)],
        )
    )
    registry_pass = bool(
        zero_paid_contract["registry_status"] == "PASS"
        and len(registry_rows) == len({row["source_id"] for row in registry_rows})
        and all(row.get("admission_status") for row in registry_rows)
        and all(row.get("quality_status") for row in registry_rows)
    )
    requirements.append(
        requirement(
            "R05",
            "免费数据源注册表完整覆盖字段、时间、点时性、原始归档、许可和失败合同",
            "PASS" if registry_pass else "CONTRADICTION",
            {
                "rows": len(registry_rows),
                "unique_source_ids": len({row['source_id'] for row in registry_rows}),
                "registry_status": zero_paid_contract["registry_status"],
                "missing_columns": zero_paid_contract["missing_registry_columns"],
                "empty_required_cells": zero_paid_contract[
                    "empty_required_registry_cells"
                ],
                "nonzero_cost_source_ids": zero_paid_contract[
                    "nonzero_cost_source_ids"
                ],
                "raw_preservation_not_required": zero_paid_contract[
                    "raw_preservation_not_required"
                ],
                "canonical_matches_frozen_v1_1": zero_paid_contract[
                    "registry_checks"
                ]["canonical_matches_frozen_v1_1"],
            },
            [
                evidence(CANONICAL_SOURCE_REGISTRY),
                evidence(SOURCE_REGISTRY),
                evidence(ZERO_PAID_POLICY),
            ],
        )
    )
    manifest_pass = bool(
        runtime.get("deployment_audit_valid") is True
        and active_manifest_check["status"] == "PASS"
        and active_manifest_check["manifest_sha256"] == active_manifest_sha256
        and active_manifest.get("status") == runtime["manifest_status"]
        and active_correction.get("status")
        == runtime["correction_expected_status"]
        and active_correction["frozen_manifest"].get("sha256")
        == active_manifest_sha256
        and active_test_boundary.get("status")
        == "PASS_TESTS_EXCLUDED_FROM_RUNTIME_MODEL_BOUNDARY"
        and active_test_boundary["frozen_manifest"].get("manifest_sha256")
        == active_manifest_sha256
        and active_test_boundary["findings"].get("tracked_test_file_count") == 0
        and active_test_boundary["findings"].get("runtime_test_import_count") == 0
        and active_test_boundary["findings"].get("config_launchers_all_tracked")
        is True
        and test_boundary_check.get("status") == "PASS"
        and test_boundary_check.get("tracked_test_file_count")
        == active_test_boundary["findings"].get("tracked_test_file_count")
        and test_boundary_check.get("runtime_test_import_count")
        == active_test_boundary["findings"].get("runtime_test_import_count")
        and test_boundary_check.get("config_launcher_count")
        == active_test_boundary["findings"].get("config_launcher_count")
        and test_boundary_check.get("config_launchers_all_tracked")
        == active_test_boundary["findings"].get("config_launchers_all_tracked")
        and active_test_boundary["boundary"].get("tests_are_runtime_or_model_inputs")
        is False
        and active_test_boundary["boundary"].get("runtime_files_remain_frozen") is True
    )
    requirements.append(
        requirement(
            "R06",
            "主前瞻入口、配置和终局证据受冻结清单约束",
            "PASS" if manifest_pass else "CONTRADICTION",
            {
                **active_manifest_check,
                "active_runtime": runtime_state,
                "operational_correction": active_correction.get("status"),
                "test_boundary": active_test_boundary.get("status"),
                "tracked_test_file_count": active_test_boundary["findings"].get(
                    "tracked_test_file_count"
                ),
                "runtime_test_import_count": active_test_boundary["findings"].get(
                    "runtime_test_import_count"
                ),
                "config_launchers_all_tracked": active_test_boundary["findings"].get(
                    "config_launchers_all_tracked"
                ),
                "live_test_boundary_check": test_boundary_check,
            },
            [
                evidence(active_manifest_path),
                evidence(active_supervisor_config_path),
                evidence(active_correction_audit_path),
                evidence(active_test_boundary_audit_path),
            ],
        )
    )
    pcf_task_pass = task_exact(
        pcf_task,
        runner=active_runner,
        manifest_sha256=active_manifest_sha256,
    )
    if (
        pcf_run_evidence["successful"]
        and pcf_source_receipt_audit["all_required_sources_succeeded"]
    ):
        pcf_run_status = "PASS"
        pcf_next = None
    elif is_trading_day and before_pcf_window:
        pcf_run_status = "PENDING_WINDOW"
        pcf_next = "等待本交易日 09:25 至 09:35 合法窗口"
    elif pcf_run_evidence["status"] == "FAILED_RUN":
        pcf_run_status = "PENDING_NEXT_WINDOW"
        pcf_next = "保留本日失败状态且不重试；等待下一交易日真实窗口"
    elif pcf_run_evidence["status"] == "PASS":
        pcf_run_status = "CONTRADICTION"
        pcf_next = "任务回执宣称成功但三类来源回执未形成完整且可核验的原始证据束"
    elif pcf_run_evidence["actual_launcher_result_count"] > 0:
        pcf_run_status = "CONTRADICTION"
        pcf_next = "修正回执合同缺口；不得把结构错误或重复尝试记为成功"
    else:
        pcf_run_status = "PENDING_CURRENT_RUN"
        pcf_next = (
            "当日原子认领已存在，等待全天采集自然结束并写出任务回执"
            if pcf_claims
            else "读取本日不可变回执；无回执时不得登记成功或质量日"
        )
    requirements.append(
        requirement(
            "R07",
            "PCF/IOPV V1.2 通过冻结入口在合法窗口产生真实当日回执",
            pcf_run_status if pcf_task_pass else "CONTRADICTION",
            {
                "trading_day": is_trading_day,
                "runtime_contract": {
                    key: (
                        value.relative_to(ROOT).as_posix()
                        if isinstance(value, Path)
                        else value
                    )
                    for key, value in pcf_runtime_contract.items()
                },
                "task_exact": pcf_task_pass,
                "task": pcf_task,
                "task_receipts_today": len(pcf_receipts),
                "codex_receipts_today": len(pcf_codex_receipts),
                "claims_today": len(pcf_claims),
                "atomic_run_evidence": pcf_run_evidence,
                "source_receipt_audit": pcf_source_receipt_audit,
            },
            [
                evidence(PCF_READINESS),
                *pcf_receipts,
                *pcf_codex_receipts,
                *pcf_claims,
                *pcf_source_receipt_evidence,
            ],
            pcf_next,
        )
    )
    pcf_mature = pcf_maturity_contract["status"] == "PASS"
    pcf_maturity_status = (
        "PASS"
        if pcf_mature
        else "CONTRADICTION"
        if pcf_maturity_contract["status"] == "FAILED_CONTRACT"
        else "PENDING_MATURITY"
    )
    requirements.append(
        requirement(
            "R08",
            "PCF/IOPV 只按真实完整质量日推进 20/40/80/120 门槛",
            pcf_maturity_status,
            {
                "status": pcf.get("status"),
                "observed_trading_days": pcf.get("observed_trading_days"),
                "legacy_full_coverage_days": pcf.get("full_coverage_days"),
                "v1_2_crosscheck_passed_days": pcf.get("crosscheck_passed_day_count"),
                "gates": [20, 40, 80, 120],
                "evaluation_eligible": pcf.get("eligible_for_research_evaluation"),
                "contract_recalculation": pcf_maturity_contract,
            },
            [evidence(PCF_READINESS)],
            "继续严格前瞻；未达到相应门槛不得评价",
        )
    )
    pcf_operational_status = str(pcf_operational_acceptance["status"])
    if pcf_operational_status == "PASS":
        pcf_operational_requirement_status = "PASS"
        pcf_operational_next = None
    elif pcf_operational_status == "PENDING_MATURITY":
        pcf_operational_requirement_status = "PENDING_MATURITY"
        pcf_operational_next = "继续累计首20个真实交易机会；缺失日计入分母且不得补跑"
    elif (
        pcf_operational_status == "FAILED_INTERNAL_PROGRAM_GATE"
        and pcf.get("status") == "PAUSED_INTERNAL_PROGRAM_SUCCESS_BELOW_95"
    ) or (
        pcf_operational_status == "FAILED_EXTERNAL_COMPLETE_DAY_GATE"
        and pcf.get("status") == "BLOCKED_EXTERNAL_COMPLETE_DAY_RATIO_BELOW_90"
    ):
        pcf_operational_requirement_status = "TERMINAL_EXPECTED"
        pcf_operational_next = None
    else:
        pcf_operational_requirement_status = "CONTRADICTION"
        pcf_operational_next = "按冻结停止线暂停或阻断采集器，不得删除失败机会"
    requirements.append(
        requirement(
            "R25",
            "PCF修复后首20个交易机会达到内部程序95%和完整质量日90%验收门",
            pcf_operational_requirement_status,
            pcf_operational_acceptance,
            [
                evidence(PCF_READINESS),
                *[
                    evidence(path)
                    for directory in (
                        PCF_TASK_RECEIPT_DIRECTORY,
                        PCF_TASK_RECEIPT_DIRECTORY_V1_2_1,
                    )
                    if directory.is_dir()
                    for path in sorted(directory.glob("*.json"))
                ],
            ],
            pcf_operational_next,
        )
    )
    frozen_pass = bool(
        frozen_inputs.get("audit_status")
        == "PASS_FROZEN_INPUTS_CONTENT_ADDRESSED_AND_LATEST_ISOLATED"
        and frozen_inputs["checks"].get("industry_v1_2_snapshot_inputs_verified") is True
        and frozen_inputs["industry_expectation_gap"].get("shared_latest_consumed_by_evaluation")
        is False
        and industry_frozen_boundary.get("status") == "PASS"
    )
    requirements.append(
        requirement(
            "R09",
            "行业预期差使用内容寻址冻结快照且不消费共享 latest",
            "PASS" if frozen_pass else "CONTRADICTION",
            {
                "audit_status": frozen_inputs.get("audit_status"),
                "snapshot_id": frozen_inputs["industry_expectation_gap"].get("snapshot_id"),
                "shared_latest_consumed": frozen_inputs["industry_expectation_gap"].get(
                    "shared_latest_consumed_by_evaluation"
                ),
                "live_boundary": industry_frozen_boundary,
            },
            [
                evidence(FROZEN_INPUT_AUDIT),
                evidence(INDUSTRY_STATUS),
                evidence(active_supervisor_config_path),
                evidence(active_manifest_path),
            ],
        )
    )
    industry_mature = industry_maturity_contract["status"] == "PASS"
    industry_maturity_status = (
        "PASS"
        if industry_mature
        else "CONTRADICTION"
        if industry_maturity_contract["status"] == "FAILED_CONTRACT"
        else "PENDING_MATURITY"
    )
    requirements.append(
        requirement(
            "R10",
            "行业预期差按独立原点和非重叠 60 日块成熟，不提前评价",
            industry_maturity_status,
            {
                "status": industry.get("status"),
                "view_status": industry.get("view_status"),
                "origin_clusters": industry_maturity.get("origin_cluster_count"),
                "mature_origin_clusters": industry_maturity.get("mature_origin_cluster_count"),
                "non_overlapping_60d_blocks": industry_maturity.get(
                    "non_overlapping_60d_block_count"
                ),
                "calibration_gate": "20 origins and 4 blocks",
                "comparison_gate": "40 origins and 8 blocks",
                "evaluation_executed": industry["evaluation_run"].get("executed"),
                "contract_recalculation": industry_maturity_contract,
            },
            [evidence(INDUSTRY_STATUS)],
            "等待新官方原点和成熟结果；当前保持 NO_VIEW",
        )
    )
    t_only_output_pass = bool(
        t_only_manifest_check["status"] == "PASS"
        and t_only_manifest_check["manifest_sha256"] == T_ONLY_MANIFEST_SHA256
        and t_only.get("current_view") == "NO_VIEW_UNTIL_FORWARD_MATURITY"
        and not t_only_leaks
    )
    requirements.append(
        requirement(
            "R11",
            "T-only 成熟前公开输出仅含成熟度和完整性",
            "PASS" if t_only_output_pass else "CONTRADICTION",
            {
                "current_view": t_only.get("current_view"),
                "forbidden_key_matches": t_only_leaks,
                "manifest": t_only_manifest_check,
            },
            [evidence(T_ONLY_STATUS), evidence(T_ONLY_MANIFEST)],
        )
    )
    t_only_task_pass = task_exact(
        t_only_task,
        runner="run_t_only_forward_v1_daily_v1_2.py",
        manifest_sha256=T_ONLY_MANIFEST_SHA256,
    )
    if t_only_daily_evidence["successful"]:
        t_only_run_status = "PASS"
        t_only_next = None
    elif is_trading_day and before_t_only_window:
        t_only_run_status = "PENDING_WINDOW"
        t_only_next = "等待本交易日 16:30 首次 V1.2 计划运行"
    elif t_only_daily_evidence["contract_valid"]:
        t_only_run_status = "PENDING_NEXT_WINDOW"
        t_only_next = "保留本日NO_VIEW运行失败且不重试；等待下一交易日"
    elif t_only_receipts:
        t_only_run_status = "CONTRADICTION"
        t_only_next = "修正当日回执、公开状态哈希或成熟度输出合同"
    else:
        t_only_run_status = "PENDING_CURRENT_RUN"
        t_only_next = "读取本日 V1.2 不可变回执；无回执时保持运行完整性未确认"
    requirements.append(
        requirement(
            "R12",
            "T-only V1.2 计划任务产生首次真实本日回执",
            t_only_run_status if t_only_task_pass else "CONTRADICTION",
            {
                "task_exact": t_only_task_pass,
                "task": t_only_task,
                "receipts_today": len(t_only_receipts),
                "latest_run_status_exists": T_ONLY_RUN_STATUS.is_file(),
                "daily_evidence": t_only_daily_evidence,
            },
            [
                evidence(T_ONLY_STATUS),
                evidence(T_ONLY_RUN_STATUS),
                evidence(T_ONLY_MANIFEST),
                *t_only_receipts,
            ],
            t_only_next,
        )
    )
    t_only_mature = t_only_maturity_contract["status"] == "PASS"
    t_only_maturity_status = (
        "PASS"
        if t_only_mature
        else "CONTRADICTION"
        if t_only_maturity_contract["status"] == "FAILED_CONTRACT"
        else "PENDING_MATURITY"
    )
    requirements.append(
        requirement(
            "R13",
            "T-only 达到 252 个新交易日并闭合至少 3 个周期后才评价",
            t_only_maturity_status,
            {
                "new_trading_days": t_only.get("new_trading_days"),
                "closed_cycles": t_only.get("closed_cycles"),
                "required_new_trading_days": 252,
                "required_closed_cycles": 3,
                "current_view": t_only.get("current_view"),
                "contract_recalculation": t_only_maturity_contract,
            },
            [evidence(T_ONLY_STATUS)],
            "继续自动追加账本，不查看中途表现",
        )
    )
    option_terminal = bool(
        option.get("qualification_status") == "BLOCKED_ZERO_COST_SOURCE_UNAVAILABLE"
        and option_closure.get("scheduled_trial_enabled") is False
        and option_closure.get("historical_backfill_enabled") is False
        and option_closure.get("daily_proxy_enabled") is False
        and option_closure.get("next_action") == "CLOSED_PRESERVE_EVIDENCE"
        and active_schedule_surface["automation_status_by_id"].get("510300-2")
        == "PAUSED"
    )
    requirements.append(
        requirement(
            "R14",
            "期权盘口先做零成本来源准入；未通过即关闭且不用代理",
            "TERMINAL_EXPECTED" if option_terminal else "CONTRADICTION",
            {
                "qualification_status": option.get("qualification_status"),
                "qualified_trial_days": option["forward_evidence"].get(
                    "qualified_trial_days"
                ),
                "required_trial_days": 20,
                "closure": option_closure,
                "codex_automation_status": active_schedule_surface[
                    "automation_status_by_id"
                ].get("510300-2"),
            },
            [evidence(OPTION_AUDIT), evidence(active_schedule_surface_audit_path)],
        )
    )
    cash_terminal_pass = bool(
        cash.get("status") == "NO_VIEW_FREE_DATA_INSUFFICIENT"
        and cash.get("attempt_consumed") is True
        and int(cash.get("attempt_number", 0)) == 1
        and int(cash.get("maximum_formal_attempts", 0)) == 1
        and cash_terminal.get("candidate_closed_for_free_source_repair") is True
        and cash_terminal.get("second_repair_allowed") is False
        and cash_terminal.get("price_values_read") is False
        and cash_terminal.get("return_values_read") is False
    )
    requirements.append(
        requirement(
            "R15",
            "现金选择权只消耗一次最终免费修复，数据门失败即终结且不读取收益",
            "TERMINAL_EXPECTED" if cash_terminal_pass else "CONTRADICTION",
            {
                "status": cash.get("status"),
                "attempt_consumed": cash.get("attempt_consumed"),
                "event_count": central["data_admission_only"].get("event_count"),
                "data_gate_pass_count": central["data_admission_only"].get(
                    "data_gate_pass_count"
                ),
                "price_values_read": cash_terminal.get("price_values_read"),
                "return_values_read": cash_terminal.get("return_values_read"),
                "second_repair_allowed": cash_terminal.get("second_repair_allowed"),
            },
            [evidence(CASH_AUDIT), evidence(CENTRAL_STATUS)],
        )
    )
    prior_persistence_arguments = str(
        windows.get("registered_task", {}).get("action_arguments") or ""
    )
    persistence_audit_bound_to_active_runtime = bool(
        active_runner in prior_persistence_arguments
        and active_manifest_sha256 in prior_persistence_arguments
    )
    persistence_pass = bool(
        windows.get("validation_status") == "PASS_REBOOT_AND_LOGOUT_RUN_OBSERVED"
        and persistence_audit_bound_to_active_runtime
        and windows.get("verification", {}).get("logout_run_observed") is True
        and windows.get("verification", {}).get("reboot_run_observed") is True
        and task_exact(
            pcf_task,
            runner=active_runner,
            manifest_sha256=active_manifest_sha256,
        )
    )
    requirements.append(
        requirement(
            "R16",
            f"当前 {runtime['runtime_version']} Windows 任务在注销和重启后真实运行通过",
            "PASS" if persistence_pass else "PENDING_PERMISSION",
            {
                "validation_status": windows.get("validation_status"),
                "current_task_runner": pcf_task.get("arguments"),
                "current_task_logon_type": pcf_task.get("logon_type"),
                "active_runtime_version": runtime["runtime_version"],
                "audit_bound_to_current_runtime": persistence_audit_bound_to_active_runtime,
                "prior_audit_task_runner": windows["registered_task"].get(
                    "action_arguments"
                ),
                "logout_run_observed": windows["verification"].get(
                    "logout_run_observed"
                ),
                "reboot_run_observed": windows["verification"].get(
                    "reboot_run_observed"
                ),
            },
            [evidence(WINDOWS_PERSISTENCE_AUDIT)],
            (
                f"需用户授权后对当前 {runtime['runtime_version']} 入口各完成一次注销与重启实测；"
                "此前保持未验证"
            ),
        )
    )
    execution_pass = bool(
        execution.get("research_only") is True
        and execution.get("position_mapping_enabled") is False
        and execution.get("order_generation_enabled") is False
        and execution.get("broker_connection_enabled") is False
        and execution.get("live_trading_enabled") is False
    )
    requirements.append(
        requirement(
            "R17",
            "研究、持仓映射、订单、券商连接和实盘授权保持分离",
            "PASS" if execution_pass else "CONTRADICTION",
            execution,
            [evidence(CENTRAL_STATUS), evidence(active_manifest_path)],
        )
    )
    automations_pass = all(item["status"] == "PASS" for item in automation_records)
    requirements.append(
        requirement(
            "R18",
            (
                "Codex 09:20 与 19:00 自动化均为 ACTIVE 且固定 "
                f"{runtime['runtime_version']} 清单哈希"
            ),
            "PASS" if automations_pass else "CONTRADICTION",
            automation_records,
            automation_records,
        )
    )
    schedule_surface_pass = bool(
        active_schedule_surface.get("status") == "PASS"
        and active_schedule_surface_audit.get("status")
        == "PASS_UNAUTHORIZED_SCHEDULES_PAUSED_OR_DISABLED"
        and active_schedule_surface_audit["frozen_manifest"].get("sha256")
        == active_manifest_sha256
    )
    requirements.append(
        requirement(
            "R26",
            (
                "实际调度面只保留授权的"
                f"{runtime['runtime_version']}与T-only入口，期权、V3和旧版重复任务均停用"
            ),
            "PASS" if schedule_surface_pass else "CONTRADICTION",
            active_schedule_surface,
            [
                evidence(active_schedule_surface_audit_path),
                evidence(active_manifest_path),
                *active_schedule_surface["automation_inventory"].get(
                    "automations", []
                ),
            ],
        )
    )
    collection_script = active_pcf_collection_task_path.read_text(encoding="utf-8")
    collector_script = (
        ROOT / "scripts" / "collect_510300_primary_market_v1_2.py"
    ).read_text(encoding="utf-8")
    classification_pass = bool(
        "EXTERNAL_FREE_SOURCE_FAILED" in collection_script
        and "PROGRAM_FAILED" in collection_script
        and "return 3" in collector_script
    )
    requirements.append(
        requirement(
            "R19",
            "外部免费源失败与内部程序失败使用不同状态和退出码",
            "PASS" if classification_pass else "CONTRADICTION",
            {
                "external_status": "EXTERNAL_FREE_SOURCE_FAILED",
                "program_status": "PROGRAM_FAILED",
                "external_exit_code": 3,
                "program_exit_code": 1,
            },
            [
                evidence(active_pcf_collection_task_path),
                evidence(ROOT / "scripts" / "collect_510300_primary_market_v1_2.py"),
            ],
        )
    )
    no_backfill_pass = bool(
        pcf_task.get("restart_count") == 0
        and pcf_task.get("start_when_available") is False
        and industry["origin_collection"].get("historical_backfill_enabled") is False
        and option_closure.get("historical_backfill_enabled") is False
        and read_json(T_ONLY_MANIFEST).get("backfill_enabled") is False
    )
    requirements.append(
        requirement(
            "R20",
            "失败、错过窗口和缺失数据不自动重试、不补跑、不回填",
            "PASS" if no_backfill_pass else "CONTRADICTION",
            {
                "pcf_restart_count": pcf_task.get("restart_count"),
                "pcf_start_when_available": pcf_task.get("start_when_available"),
                "industry_historical_backfill": industry["origin_collection"].get(
                    "historical_backfill_enabled"
                ),
                "option_historical_backfill": option_closure.get(
                    "historical_backfill_enabled"
                ),
                "t_only_backfill": read_json(T_ONLY_MANIFEST).get("backfill_enabled"),
            },
            [
                evidence(active_manifest_path),
                evidence(T_ONLY_MANIFEST),
                evidence(INDUSTRY_STATUS),
            ],
        )
    )
    v3 = frozen_inputs["v3_forward_2_quarantine"]
    v3_pass = bool(
        frozen_inputs["checks"].get("active_supervisor_excludes_v3_forward_2") is True
        and v3.get("active_task") is False
        and v3.get("retry_or_repair_performed") is False
        and v3.get("latest_status_exists") is False
        and active_schedule_surface["automation_status_by_id"].get("v3-forward-1")
        == "PAUSED"
    )
    requirements.append(
        requirement(
            "R21",
            "哈希漂移的 V3 分支保持失败隔离，不重试也不复用旧信号",
            "TERMINAL_EXPECTED" if v3_pass else "CONTRADICTION",
            {
                "daily_status": v3.get("daily_status"),
                "mismatch_count": v3.get("mismatch_count"),
                "active_task": v3.get("active_task"),
                "latest_status_exists": v3.get("latest_status_exists"),
                "retry_or_repair_performed": v3.get("retry_or_repair_performed"),
                "codex_automation_status": active_schedule_surface[
                    "automation_status_by_id"
                ].get("v3-forward-1"),
            },
            [evidence(FROZEN_INPUT_AUDIT), evidence(active_schedule_surface_audit_path)],
        )
    )
    cb_summary = cb_sufficiency.get("summary") or {}
    cb_sufficiency_pass = bool(
        cb_archive_check.get("status") == "PASS"
        and cb_sufficiency.get("status")
        == "CONTINUE_OFFICIAL_LIFECYCLE_EVIDENCE_ONLY"
        and int(cb_summary.get("candidate_count", 0)) == 939
        and int(cb_summary.get("official_right_terms_complete_count", 0)) >= 100
        and int(cb_summary.get("maximum_evaluable_sample_upper_bound_count", 0))
        >= 846
        and float(
            cb_summary.get("maximum_evaluable_sample_upper_bound_fraction", 0.0)
        )
        >= 0.9
        and cb_summary.get("official_right_terms_gate_pass") is True
        and cb_summary.get("complete_lifecycle_upper_bound_gate_pass") is True
        and cb_sufficiency.get("discovery_terms_used_to_fill_official_terms") is False
        and cb_sufficiency.get("market_data_coverage_values_read") is False
        and cb_sufficiency.get("market_price_values_read") is False
        and cb_sufficiency.get("future_returns_read") is False
        and cb_sufficiency.get("formal_return_evaluation_authorized") is False
    )
    requirements.append(
        requirement(
            "R23",
            "可转债官方权利条款与事前最大样本上限完成冻结裁定",
            "PASS" if cb_sufficiency_pass else "CONTRADICTION",
            {
                "status": cb_sufficiency.get("status"),
                "candidate_count": cb_summary.get("candidate_count"),
                "official_right_terms_complete_count": cb_summary.get(
                    "official_right_terms_complete_count"
                ),
                "maximum_evaluable_sample_upper_bound_count": cb_summary.get(
                    "maximum_evaluable_sample_upper_bound_count"
                ),
                "maximum_evaluable_sample_upper_bound_fraction": cb_summary.get(
                    "maximum_evaluable_sample_upper_bound_fraction"
                ),
                "current_official_complete_lifecycle_count": cb_summary.get(
                    "current_official_complete_lifecycle_count"
                ),
                "archive": cb_archive_check,
            },
            [
                evidence(CB_SUFFICIENCY_RECEIPT),
                evidence(CB_SUFFICIENCY_LEDGER),
                evidence(CB_SUFFICIENCY_ARCHIVE),
            ],
        )
    )
    cb_source_inventory_valid = bool(
        len(cb_source_rows) == 4
        and {(row["asset_leg"], row["source_rank"]) for row in cb_source_rows}
        == {
            ("STOCK", "PRIMARY"),
            ("STOCK", "SECONDARY"),
            ("CONVERTIBLE_BOND", "PRIMARY"),
            ("CONVERTIBLE_BOND", "SECONDARY"),
        }
        and all(float(row.get("access_cost_cny") or 0) == 0 for row in cb_source_rows)
        and all(
            str(row.get("market_data_coverage_values_read")).lower() == "false"
            and str(row.get("market_price_values_read")).lower() == "false"
            and str(row.get("future_returns_read")).lower() == "false"
            for row in cb_source_rows
        )
    )
    cb_lifecycle_summary = cb_lifecycle.get("summary") or {}
    cb_lifecycle_terminal = bool(
        cb_lifecycle_archive_check.get("status") == "PASS"
        and cb_lifecycle.get("status") == "NO_VIEW_INSUFFICIENT_EVIDENCE"
        and int(cb_lifecycle_summary.get("official_complete_lifecycle_count", 0))
        < 846
        and cb_lifecycle_summary.get("official_complete_lifecycle_gate_pass") is False
        and cb_lifecycle.get("market_data_coverage_values_read") is False
        and cb_lifecycle.get("market_price_values_read") is False
        and cb_lifecycle.get("future_returns_read") is False
        and cb_lifecycle.get("formal_return_evaluation_authorized") is False
    )
    cb_lifecycle_and_sources_complete = bool(
        cb_source_inventory_valid
        and cb_lifecycle_archive_check.get("status") == "PASS"
        and cb_lifecycle.get("status")
        == "CONTINUE_FREE_DOUBLE_SOURCE_COVERAGE_AUDIT_ONLY"
        and int(cb_lifecycle_summary.get("official_complete_lifecycle_count", 0))
        >= 846
        and all(
            row.get("coverage_test_status") == "PASS_FREE_DOUBLE_SOURCE_COVERAGE"
            for row in cb_source_rows
        )
    )
    requirements.append(
        requirement(
            "R24",
            "可转债官方完整生命周期门：达到846才查双源，不足即NO_VIEW并停止",
            (
                "PASS"
                if cb_lifecycle_and_sources_complete
                else "TERMINAL_EXPECTED"
                if cb_lifecycle_terminal
                else "PENDING_MATURITY"
                if cb_source_inventory_valid
                and cb_lifecycle_archive_check.get("status") == "PASS"
                else "CONTRADICTION"
            ),
            {
                "status": cb_lifecycle.get("status"),
                "official_complete_lifecycle_count": cb_lifecycle_summary.get(
                    "official_complete_lifecycle_count"
                ),
                "required_complete_lifecycle_count": 846,
                "official_complete_lifecycle_fraction": cb_lifecycle_summary.get(
                    "official_complete_lifecycle_fraction"
                ),
                "source_inventory_rows": len(cb_source_rows),
                "source_coverage_statuses": sorted(
                    {str(row.get("coverage_test_status")) for row in cb_source_rows}
                ),
                "market_data_coverage_values_read": cb_lifecycle.get(
                    "market_data_coverage_values_read"
                ),
                "archive": cb_lifecycle_archive_check,
            },
            [
                evidence(CB_LIFECYCLE_RECEIPT),
                evidence(CB_LIFECYCLE_LEDGER),
                evidence(CB_LIFECYCLE_ARCHIVE),
                evidence(CB_SOURCE_AVAILABILITY),
            ],
            (
                None
                if cb_lifecycle_terminal
                else "达到846个后再冻结逐候选免费双源覆盖审计；未达门时不得读取行情或收益"
            ),
        )
    )
    if not is_trading_day:
        close_status = "PENDING_WINDOW"
        close_next = "等待下一交易日 19:00 收盘编排窗口"
    elif before_close_window:
        close_status = "PENDING_WINDOW"
        close_next = "等待本交易日 19:00 收盘后验收"
    elif close_acceptance["status"] == "PASS":
        close_status = "PASS"
        close_next = None
    elif close_acceptance["status"] == "FAILED_RUN":
        close_status = "PENDING_NEXT_WINDOW"
        close_next = "保留本日失败回执且不补跑，等待下一交易日窗口"
    elif close_codex_receipt_paths:
        close_status = "CONTRADICTION"
        close_next = "最终编排回执已存在但同日任务、声明或状态快照未形成有效闭环"
    else:
        close_status = "PENDING_CURRENT_RUN"
        close_next = "最终编排回执尚未形成，不补跑且不得把中间状态视为完成"
    requirements.append(
        requirement(
            "R22",
            (
                "本交易日收盘后由 "
                f"{pcf_runtime_contract['runtime_version']} 编排回执生成最新 V1.6 权威研究状态"
            ),
            close_status,
            {
                "before_close_window": before_close_window,
                "acceptance_status": close_acceptance["status"],
                "contract_valid": close_acceptance["contract_valid"],
                "successful": close_acceptance["successful"],
                "industry_receipt_count": close_acceptance[
                    "industry_receipt_count"
                ],
                "industry_acceptable": close_acceptance["industry_acceptable"],
                "status_receipt_count": close_acceptance["status_receipt_count"],
                "status_task_success": close_acceptance["status_task_success"],
                "status_snapshot_valid": close_acceptance[
                    "status_snapshot_valid"
                ],
                "codex_receipt_count": close_acceptance["codex_receipt_count"],
                "codex_results_acceptable": close_acceptance[
                    "codex_results_acceptable"
                ],
                "claim_count": close_acceptance["claim_count"],
                "claims_valid": close_acceptance["claims_valid"],
                "central_uses_current_industry_receipt": close_acceptance[
                    "central_uses_current_industry_receipt"
                ],
                "structural_failure_count": close_acceptance[
                    "structural_failure_count"
                ],
                "structural_failures": close_acceptance["structural_failures"],
                "task_windows": close_acceptance["task_windows"],
                "codex_close_start_window": close_acceptance[
                    "codex_close_start_window"
                ],
                "central_status_generated_at": central.get("generated_at"),
            },
            [evidence(CENTRAL_STATUS), *[evidence(path) for path in close_evidence_paths]],
            close_next,
        )
    )

    counts = {
        "satisfied": sum(item["status"] in SATISFIED_STATUSES for item in requirements),
        "pending": sum(item["status"] in PENDING_STATUSES for item in requirements),
        "contradictions": sum(
            item["status"] in CONTRADICTION_STATUSES for item in requirements
        ),
        "total": len(requirements),
    }
    pending_ids = [
        item["id"] for item in requirements if item["status"] in PENDING_STATUSES
    ]
    contradiction_ids = [
        item["id"]
        for item in requirements
        if item["status"] in CONTRADICTION_STATUSES
    ]
    completion_proven = counts["pending"] == 0 and counts["contradictions"] == 0
    return {
        "schema_version": "1.0.0",
        "audit_id": "ZERO_PAID_RESEARCH_PLAN_COMPLETION_V1",
        "generated_at": current.isoformat(),
        "current_trading_date": today,
        "is_sse_trading_day": is_trading_day,
        "overall_status": (
            "COMPLETE"
            if completion_proven
            else "CONTRADICTIONS_REQUIRE_ACTION"
            if contradiction_ids
            else "IN_PROGRESS_PENDING_REQUIRED_EVIDENCE"
        ),
        "goal_completion_proven": completion_proven,
        "research_conclusion": central.get("overall_research_status"),
        "decision": central.get("decision"),
        "counts": counts,
        "pending_requirement_ids": pending_ids,
        "contradiction_requirement_ids": contradiction_ids,
        "requirements": requirements,
        "source_files": {
            "attachments": attachment_records,
            "audit_script": evidence(Path(__file__)),
        },
    }


def render_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# 零付费研究计划完成度审计 V1",
        "",
        f"- 生成时间：{audit['generated_at']}",
        f"- 总状态：`{audit['overall_status']}`",
        f"- 完成已被证明：`{str(audit['goal_completion_proven']).upper()}`",
        f"- 研究结论：`{audit['research_conclusion']}`",
        f"- 当前决定：`{audit['decision']}`",
        f"- 已满足：{audit['counts']['satisfied']} / {audit['counts']['total']}",
        f"- 待取得证据：{audit['counts']['pending']}",
        f"- 矛盾或缺失：{audit['counts']['contradictions']}",
        "",
        "| ID | 要求 | 状态 | 当前证据摘要 | 下一步 |",
        "|---|---|---|---|---|",
    ]
    for item in audit["requirements"]:
        state = json.dumps(item["current_state"], ensure_ascii=False, separators=(",", ":"))
        if len(state) > 180:
            state = state[:177] + "..."
        state = state.replace("|", "\\|")
        next_action = str(item.get("next_action") or "—").replace("|", "\\|")
        lines.append(
            f"| {item['id']} | {item['title']} | `{item['status']}` | {state} | {next_action} |"
        )
    lines.extend(
        [
            "",
            "## 当前未完成项",
            "",
        ]
    )
    if audit["pending_requirement_ids"]:
        for requirement_id in audit["pending_requirement_ids"]:
            item = next(
                entry for entry in audit["requirements"] if entry["id"] == requirement_id
            )
            lines.append(f"- `{requirement_id}`：{item['title']}；{item['next_action']}")
    else:
        lines.append("无。")
    lines.extend(["", "## 矛盾或缺失", ""])
    if audit["contradiction_requirement_ids"]:
        for requirement_id in audit["contradiction_requirement_ids"]:
            item = next(
                entry for entry in audit["requirements"] if entry["id"] == requirement_id
            )
            lines.append(f"- `{requirement_id}`：{item['title']}")
    else:
        lines.append("无。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="审计零付费研究计划完成度")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    audit = build_audit()
    if not arguments.no_write:
        atomic_text(OUTPUT_JSON, json.dumps(audit, ensure_ascii=False, indent=2))
        atomic_text(OUTPUT_MARKDOWN, render_markdown(audit))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["contradiction_requirement_ids"]:
        return 1
    if arguments.require_complete and not audit["goal_completion_proven"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
