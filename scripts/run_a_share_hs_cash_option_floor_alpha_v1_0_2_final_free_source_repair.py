from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIMEZONE = ZoneInfo("Asia/Shanghai")
CONFIG_PATH = (
    ROOT
    / "config/a_share_hs_official_cash_option_floor_alpha_v1_0_2_final_free_source_repair.json"
)
DOC_PATH = (
    ROOT
    / "docs/A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1_0_2_FINAL_FREE_SOURCE_REPAIR.md"
)
SCRIPT_PATH = Path(__file__).resolve()
TEST_PATH = (
    ROOT
    / "tests/test_a_share_hs_cash_option_floor_alpha_v1_0_2_final_free_source_repair.py"
)
PROTOCOL_FILES = (CONFIG_PATH, DOC_PATH, SCRIPT_PATH, TEST_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON 顶层必须是对象：{relative(path)}")
    return value


def create_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n").encode(
        "utf-8"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    expected = {
        "repair_id": "A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1_0_2_FINAL_FREE_SOURCE_REPAIR",
        "attempt_id": "CASH_OPTION_V1_0_2_FINAL_FREE_REPAIR_20260826",
        "maximum_formal_attempts": 1,
        "data_purchase_budget_cny": 0,
        "terminal_status_on_any_required_gate_failure": "NO_VIEW_FREE_DATA_INSUFFICIENT",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"最终修复配置字段漂移：{key}，预期={value!r}，实际={config.get(key)!r}"
            )
    governance = config.get("governance") or {}
    required_false = (
        "second_source_repair_allowed",
        "source_hopping_after_attempt",
        "event_set_changed",
        "threshold_changed",
        "costs_changed",
        "benchmark_changed",
        "price_values_may_be_read_before_full_admission",
        "return_values_may_be_read_before_full_admission",
        "price_screen_allowed_on_terminal_no_view",
        "position_mapping",
        "order_generation",
        "shadow_authorized",
        "live_authorized",
    )
    for key in required_false:
        if governance.get(key) is not False:
            raise RuntimeError(f"最终修复治理字段必须为 false：{key}")
    if governance.get("attempt_is_consumed_even_if_preflight_stops_external_requests") is not True:
        raise RuntimeError("请求前停止时也必须消费唯一正式尝试")
    preflight = config.get("preflight") or {}
    if preflight.get("all_required_gates_must_pass_before_any_external_request") is not True:
        raise RuntimeError("全部硬门槛必须在任何外部请求前通过")
    if preflight.get("terms_gate_pass") is not False:
        raise RuntimeError("本冻结版本的原始归档权利门状态必须保持失败")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    specifications = [
        {
            "path": config["parent_protocol"]["config_path"],
            "sha256": config["parent_protocol"]["config_sha256"],
            "role": "PARENT_PROTOCOL_CONFIG",
        },
        {
            "path": config["parent_protocol"]["manifest_path"],
            "sha256": config["parent_protocol"]["manifest_sha256"],
            "role": "PARENT_PROTOCOL_MANIFEST",
        },
        {
            "path": config["source_remediation_parent"]["config_path"],
            "sha256": config["source_remediation_parent"]["config_sha256"],
            "role": "SOURCE_REMEDIATION_CONFIG",
        },
        {
            "path": config["source_remediation_parent"]["manifest_path"],
            "sha256": config["source_remediation_parent"]["manifest_sha256"],
            "role": "SOURCE_REMEDIATION_MANIFEST",
        },
        *config["frozen_inputs"],
    ]
    for specification in specifications:
        path = ROOT / str(specification["path"])
        actual = sha256_file(path) if path.is_file() else None
        expected = str(specification["sha256"])
        item = {
            "path": relative(path) if path.exists() else str(specification["path"]),
            "role": str(specification["role"]),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "PASS" if actual == expected else "FAILED",
        }
        verified.append(item)
        if actual != expected:
            failures.append(item)
    if failures:
        raise RuntimeError(
            "最终修复冻结输入哈希失败："
            + json.dumps(failures, ensure_ascii=False, separators=(",", ":"))
        )
    return verified


def protocol_manifest_path(config: dict[str, Any]) -> Path:
    return ROOT / str(config["artifacts"]["protocol_manifest"])


def terminal_receipt_path(config: dict[str, Any]) -> Path:
    return ROOT / str(config["artifacts"]["terminal_receipt"])


def protocol_file_records() -> list[dict[str, Any]]:
    missing = [relative(path) for path in PROTOCOL_FILES if not path.is_file()]
    if missing:
        raise RuntimeError(f"最终修复协议文件缺失：{missing}")
    return [
        {
            "path": relative(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in PROTOCOL_FILES
    ]


def freeze_protocol() -> dict[str, Any]:
    config = load_config()
    verify_frozen_inputs(config)
    manifest_path = protocol_manifest_path(config)
    if manifest_path.exists():
        result = verify_protocol()
        return {**result, "manifest_reused": True}
    files = protocol_file_records()
    content_sha256 = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "1.0.0",
        "repair_id": config["repair_id"],
        "attempt_id": config["attempt_id"],
        "status": "FROZEN_FINAL_FREE_SOURCE_REPAIR_BEFORE_FORMAL_ATTEMPT",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "evidence_cutoff": config["evidence_cutoff"],
        "maximum_formal_attempts": 1,
        "data_purchase_budget_cny": 0,
        "files": files,
        "content_sha256": content_sha256,
        "frozen_input_count": len(config["frozen_inputs"]) + 4,
        "price_values_read_by_freeze": False,
        "return_values_read_by_freeze": False,
        "price_screen_completed": False,
    }
    create_json_exclusive(manifest_path, manifest)
    return {
        "status": "PASS_FINAL_FREE_SOURCE_REPAIR_PROTOCOL_FROZEN",
        "manifest_path": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "content_sha256": content_sha256,
        "manifest_reused": False,
    }


def verify_protocol() -> dict[str, Any]:
    config = load_config()
    input_verification = verify_frozen_inputs(config)
    manifest_path = protocol_manifest_path(config)
    if not manifest_path.is_file():
        raise RuntimeError("最终修复协议清单尚未冻结")
    manifest = read_json(manifest_path)
    failures: list[dict[str, Any]] = []
    if manifest.get("repair_id") != config["repair_id"]:
        failures.append({"failure": "REPAIR_ID", "actual": manifest.get("repair_id")})
    if manifest.get("attempt_id") != config["attempt_id"]:
        failures.append({"failure": "ATTEMPT_ID", "actual": manifest.get("attempt_id")})
    if manifest.get("maximum_formal_attempts") != 1:
        failures.append(
            {"failure": "MAXIMUM_FORMAL_ATTEMPTS", "actual": manifest.get("maximum_formal_attempts")}
        )
    current_files = protocol_file_records()
    expected_files = manifest.get("files") or []
    if current_files != expected_files:
        failures.append(
            {
                "failure": "PROTOCOL_FILE_HASH_OR_SIZE",
                "expected": expected_files,
                "actual": current_files,
            }
        )
    actual_content_sha256 = hashlib.sha256(
        json.dumps(current_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if actual_content_sha256 != manifest.get("content_sha256"):
        failures.append(
            {
                "failure": "PROTOCOL_CONTENT_SHA256",
                "expected": manifest.get("content_sha256"),
                "actual": actual_content_sha256,
            }
        )
    result = {
        "status": (
            "PASS_FINAL_FREE_SOURCE_REPAIR_PROTOCOL_VERIFIED"
            if not failures
            else "FAILED_FINAL_FREE_SOURCE_REPAIR_PROTOCOL_VERIFICATION"
        ),
        "failure_count": len(failures),
        "failures": failures,
        "manifest_path": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "content_sha256": actual_content_sha256,
        "frozen_input_count": len(input_verification),
    }
    if failures:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, default=str))
    return result


def summarize_admission_matrix(config: dict[str, Any]) -> dict[str, Any]:
    matrix_specifications = [
        item for item in config["frozen_inputs"] if item["role"] == "PRE_REPAIR_DATA_ADMISSION_MATRIX"
    ]
    if len(matrix_specifications) != 1:
        raise RuntimeError("最终修复准入矩阵规范必须唯一")
    path = ROOT / str(matrix_specifications[0]["path"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 21 or len({str(row["event_id"]) for row in rows}) != 21:
        raise RuntimeError("准入矩阵必须包含 21 个唯一事件")
    if any(str(row["price_values_read_by_matrix"]).lower() != "false" for row in rows):
        raise RuntimeError("准入矩阵不应读取行情价格值")
    if any(str(row["return_values_read_by_matrix"]).lower() != "false" for row in rows):
        raise RuntimeError("准入矩阵不应读取收益值")
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "event_count": len(rows),
        "official_terms_pass_count": sum(
            row["official_terms_status"] == "PASS_FROZEN_OFFICIAL_TERMS" for row in rows
        ),
        "primary_sina_hash_verified_count": sum(
            row["primary_sina_raw_status"] == "PASS_FROZEN_HASH_VERIFIED" for row in rows
        ),
        "secondary_eastmoney_present_count": sum(
            str(row["secondary_eastmoney_raw_status"]).startswith("PRESENT_") for row in rows
        ),
        "secondary_eastmoney_missing_count": sum(
            row["secondary_eastmoney_raw_status"] == "FAILED_NOT_ACQUIRED" for row in rows
        ),
        "calendar_admitted_count": sum(
            str(row["calendar_status"]).startswith("PASS_") for row in rows
        ),
        "benchmark_dual_source_admitted_count": sum(
            str(row["benchmark_primary_status"]).startswith("PASS_")
            and str(row["benchmark_secondary_status"]).startswith("PASS_")
            for row in rows
        ),
        "formal_corporate_action_adjudication_count": sum(
            str(row["corporate_action_status"]).startswith("PASS_") for row in rows
        ),
        "formal_lifecycle_adjudication_count": sum(
            str(row["lifecycle_status"]).startswith("PASS_") for row in rows
        ),
        "data_gate_pass_count": sum(row["data_gate_status"] == "PASS_FREE_DATA_ADMISSION" for row in rows),
        "price_values_read_by_matrix": False,
        "return_values_read_by_matrix": False,
    }


def lifecycle_inventory(config: dict[str, Any]) -> dict[str, Any]:
    input_by_role = {str(item["role"]): item for item in config["frozen_inputs"]}
    terms_path = ROOT / str(input_by_role["EXACT_FROZEN_21_EVENT_TERMS"]["path"])
    review_path = ROOT / str(input_by_role["FROZEN_OFFICIAL_DOCUMENT_ROLE_INDEX"]["path"])
    terms = pd.read_parquet(
        terms_path,
        columns=["event_id", "ts_code", "effective_announcement_date", "application_end_date"],
    )
    review = pd.read_parquet(
        review_path,
        columns=[
            "event_id",
            "announcement_id",
            "announcement_internal_date",
            "review_roles_json",
            "pdf_path",
            "pdf_sha256",
            "source_sha256_matches_frozen",
        ],
    )
    if len(terms) != 21 or terms["event_id"].astype(str).nunique() != 21:
        raise RuntimeError("冻结条款数据不是 21 个唯一事件")
    review = review.copy()
    review["event_id"] = review["event_id"].astype(str)
    review["announcement_internal_date"] = pd.to_datetime(
        review["announcement_internal_date"], errors="coerce"
    ).dt.normalize()
    event_records: list[dict[str, Any]] = []
    result_event_count = 0
    settlement_event_count = 0
    for term in terms.sort_values(["effective_announcement_date", "event_id"], kind="stable").itertuples(
        index=False
    ):
        event_id = str(term.event_id)
        group = review.loc[review["event_id"] == event_id].copy()
        application_end = pd.Timestamp(term.application_end_date).normalize()
        roles: set[str] = set()
        documents: list[dict[str, Any]] = []
        for row in group.sort_values(
            ["announcement_internal_date", "announcement_id"], kind="stable"
        ).itertuples(index=False):
            try:
                row_roles = [str(value) for value in json.loads(str(row.review_roles_json))]
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise RuntimeError(f"文档角色 JSON 无法解析：{event_id}/{row.announcement_id}") from error
            roles.update(row_roles)
            document_date = (
                row.announcement_internal_date.date().isoformat()
                if not pd.isna(row.announcement_internal_date)
                else None
            )
            documents.append(
                {
                    "announcement_id": str(row.announcement_id),
                    "date": document_date,
                    "roles": sorted(row_roles),
                    "on_or_after_application_end": (
                        False
                        if pd.isna(row.announcement_internal_date)
                        else bool(row.announcement_internal_date >= application_end)
                    ),
                    "pdf_path": str(row.pdf_path),
                    "pdf_sha256": str(row.pdf_sha256),
                    "source_sha256_matches_frozen": bool(row.source_sha256_matches_frozen),
                }
            )
        has_result = "RESULT" in roles
        has_settlement = "SETTLEMENT" in roles
        result_event_count += int(has_result)
        settlement_event_count += int(has_settlement)
        event_records.append(
            {
                "event_id": event_id,
                "ts_code": str(term.ts_code),
                "official_document_count": len(documents),
                "on_or_after_application_end_document_count": sum(
                    bool(item["on_or_after_application_end"]) for item in documents
                ),
                "result_role_present": has_result,
                "settlement_role_present": has_settlement,
                "formal_corporate_action_adjudication_complete": False,
                "formal_lifecycle_adjudication_complete": False,
                "admission_status": "INCOMPLETE_NO_FORMAL_CORPORATE_ACTION_AND_LIFECYCLE_ADJUDICATION",
                "documents": documents,
            }
        )
    return {
        "event_count": len(event_records),
        "result_role_present_event_count": result_event_count,
        "settlement_role_present_event_count": settlement_event_count,
        "formal_corporate_action_adjudication_complete_count": 0,
        "formal_lifecycle_adjudication_complete_count": 0,
        "role_inventory_is_final_adjudication": False,
        "price_values_read": False,
        "return_values_read": False,
        "events": event_records,
    }


def build_terminal_receipt(config: dict[str, Any]) -> dict[str, Any]:
    started_at = datetime.now(TIMEZONE)
    protocol_verification = verify_protocol()
    matrix = summarize_admission_matrix(config)
    lifecycle = lifecycle_inventory(config)
    if lifecycle["result_role_present_event_count"] != 21:
        raise RuntimeError("冻结官方文档库未覆盖全部 21 个事件的结果角色")
    preflight_failures = [
        {
            "gate": "REQUIRED_RAW_ARCHIVE_RIGHTS",
            "status": config["preflight"]["eastmoney_raw_archive_rights_status"],
            "reason": config["preflight"]["terms_gate_reason"],
        },
        {
            "gate": "SUBJECT_SECONDARY_FREE_SOURCE",
            "status": "FAILED_11_OF_21_RAW_RESPONSES_MISSING",
            "present_count": matrix["secondary_eastmoney_present_count"],
            "missing_count": matrix["secondary_eastmoney_missing_count"],
        },
        {
            "gate": "COMPLETE_TRADING_CALENDAR",
            "status": "FAILED_NOT_ADMITTED",
            "admitted_event_count": matrix["calendar_admitted_count"],
        },
        {
            "gate": "CSI300_DUAL_SOURCE_BENCHMARK",
            "status": "FAILED_NOT_ADMITTED",
            "admitted_event_count": matrix["benchmark_dual_source_admitted_count"],
        },
        {
            "gate": "FORMAL_CORPORATE_ACTION_ADJUDICATION",
            "status": "FAILED_NOT_COMPLETED",
            "complete_event_count": lifecycle[
                "formal_corporate_action_adjudication_complete_count"
            ],
        },
        {
            "gate": "FORMAL_COMPLETE_LIFECYCLE_ADJUDICATION",
            "status": "FAILED_NOT_COMPLETED",
            "complete_event_count": lifecycle["formal_lifecycle_adjudication_complete_count"],
            "result_role_present_event_count": lifecycle["result_role_present_event_count"],
            "settlement_role_present_event_count": lifecycle[
                "settlement_role_present_event_count"
            ],
        },
    ]
    completed_at = datetime.now(TIMEZONE)
    return {
        "schema_version": "1.0.0",
        "repair_id": config["repair_id"],
        "attempt_id": config["attempt_id"],
        "status": "NO_VIEW_FREE_DATA_INSUFFICIENT",
        "final_status": "NO_VIEW_FREE_DATA_INSUFFICIENT",
        "attempt_number": 1,
        "maximum_formal_attempts": 1,
        "attempt_consumed": True,
        "attempt_started_at": started_at.isoformat(),
        "attempt_completed_at": completed_at.isoformat(),
        "evidence_cutoff": config["evidence_cutoff"],
        "data_purchase_budget_cny": 0,
        "protocol_verification": protocol_verification,
        "pre_repair_admission_matrix": matrix,
        "official_lifecycle_evidence_inventory": lifecycle,
        "preflight": {
            "status": "FAILED_MANDATORY_PREFLIGHT",
            "failure_count": len(preflight_failures),
            "failures": preflight_failures,
        },
        "external_source_execution": {
            "status": "SKIPPED_MANDATORY_PREFLIGHT_FAILED",
            "external_request_attempted": False,
            "eastmoney_subject_requests_attempted": 0,
            "baostock_requests_attempted": 0,
            "eastmoney_csi300_requests_attempted": 0,
            "raw_files_created": 0,
            "reason": "所需原始归档权利门未通过，按冻结执行顺序在任何外部请求前停止。",
        },
        "terminal_decision": {
            "candidate_id": "A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1",
            "data_admission_pass": False,
            "candidate_closed_for_free_source_repair": True,
            "second_repair_allowed": False,
            "v1_0_3_or_later_repair_allowed": False,
            "price_screen_executed": False,
            "price_values_read": False,
            "return_values_read": False,
            "qualified_event_count_known": False,
            "alpha_or_beta_conclusion_allowed": False,
            "position_mapping": False,
            "order_generation": False,
            "shadow_authorized": False,
            "live_authorized": False,
        },
    }


def run_final_repair() -> tuple[dict[str, Any], int]:
    config = load_config()
    receipt_path = terminal_receipt_path(config)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        verify_terminal_receipt(receipt)
        return {**receipt, "terminal_receipt_reused": True}, 3
    receipt = build_terminal_receipt(config)
    create_json_exclusive(receipt_path, receipt)
    return {
        **receipt,
        "terminal_receipt_path": relative(receipt_path),
        "terminal_receipt_sha256": sha256_file(receipt_path),
        "terminal_receipt_reused": False,
    }, 3


def verify_terminal_receipt(receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config()
    verify_protocol()
    receipt_path = terminal_receipt_path(config)
    if receipt is None:
        if not receipt_path.is_file():
            raise RuntimeError("最终免费来源修复终局回执不存在")
        receipt = read_json(receipt_path)
    checks = {
        "repair_id": receipt.get("repair_id") == config["repair_id"],
        "attempt_id": receipt.get("attempt_id") == config["attempt_id"],
        "final_status": receipt.get("final_status") == "NO_VIEW_FREE_DATA_INSUFFICIENT",
        "attempt_consumed": receipt.get("attempt_consumed") is True,
        "external_request_not_attempted": (
            (receipt.get("external_source_execution") or {}).get("external_request_attempted")
            is False
        ),
        "price_screen_not_executed": (
            (receipt.get("terminal_decision") or {}).get("price_screen_executed") is False
        ),
        "price_values_not_read": (
            (receipt.get("terminal_decision") or {}).get("price_values_read") is False
        ),
        "return_values_not_read": (
            (receipt.get("terminal_decision") or {}).get("return_values_read") is False
        ),
        "second_repair_forbidden": (
            (receipt.get("terminal_decision") or {}).get("second_repair_allowed") is False
        ),
        "event_count": (
            (receipt.get("official_lifecycle_evidence_inventory") or {}).get("event_count") == 21
        ),
        "result_role_count": (
            (receipt.get("official_lifecycle_evidence_inventory") or {}).get(
                "result_role_present_event_count"
            )
            == 21
        ),
        "settlement_role_count": (
            (receipt.get("official_lifecycle_evidence_inventory") or {}).get(
                "settlement_role_present_event_count"
            )
            == 10
        ),
        "formal_lifecycle_complete_count": (
            (receipt.get("official_lifecycle_evidence_inventory") or {}).get(
                "formal_lifecycle_adjudication_complete_count"
            )
            == 0
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    result = {
        "status": (
            "PASS_FINAL_FREE_SOURCE_REPAIR_TERMINAL_RECEIPT_VERIFIED"
            if not failures
            else "FAILED_FINAL_FREE_SOURCE_REPAIR_TERMINAL_RECEIPT_VERIFICATION"
        ),
        "failure_count": len(failures),
        "failed_checks": failures,
        "checks": checks,
        "terminal_receipt_path": relative(receipt_path),
        "terminal_receipt_sha256": sha256_file(receipt_path) if receipt_path.is_file() else None,
        "final_status": receipt.get("final_status"),
    }
    if failures:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def status() -> dict[str, Any]:
    config = load_config()
    manifest_path = protocol_manifest_path(config)
    receipt_path = terminal_receipt_path(config)
    result: dict[str, Any] = {
        "repair_id": config["repair_id"],
        "attempt_id": config["attempt_id"],
        "protocol_manifest_exists": manifest_path.is_file(),
        "terminal_receipt_exists": receipt_path.is_file(),
        "maximum_formal_attempts": 1,
        "data_purchase_budget_cny": 0,
        "price_values_read_by_status": False,
        "return_values_read_by_status": False,
    }
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        result.update(
            {
                "status": str(receipt.get("final_status")),
                "attempt_consumed": bool(receipt.get("attempt_consumed")),
                "second_repair_allowed": False,
                "terminal_receipt_sha256": sha256_file(receipt_path),
            }
        )
    elif manifest_path.is_file():
        result.update(
            {
                "status": "READY_FOR_SINGLE_FINAL_FREE_SOURCE_REPAIR_ATTEMPT",
                "attempt_consumed": False,
                "second_repair_allowed": False,
            }
        )
    else:
        result.update(
            {
                "status": "PROTOCOL_MANIFEST_NOT_FROZEN",
                "attempt_consumed": False,
                "second_repair_allowed": False,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="执行现金选择权 V1.0.2 最后一次免费来源修复")
    parser.add_argument("--mode", choices=("status", "freeze", "run", "verify"), default="status")
    arguments = parser.parse_args()
    if arguments.mode == "freeze":
        result = freeze_protocol()
        exit_code = 0
    elif arguments.mode == "run":
        result, exit_code = run_final_repair()
    elif arguments.mode == "verify":
        result = verify_terminal_receipt()
        exit_code = 0
    else:
        result = status()
        exit_code = 0
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
