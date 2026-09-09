from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_a_share_hs_cash_option_floor_alpha_v1 import (  # noqa: E402
    load_config as load_parent_config,
    verify_protocol as verify_parent_protocol,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
REMEDIATION_ID = "A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1_0_1_SOURCE_REMEDIATION"
PROTOCOL_STATUS = "FROZEN_SOURCE_ONLY_REMEDIATION_BEFORE_EASTMONEY_AND_BAOSTOCK_READ"
CONFIG_PATH = (
    ROOT / "config/a_share_hs_official_cash_option_floor_alpha_v1_0_1_source_remediation.json"
)
TERMS_PATH = (
    ROOT
    / "data/curated/a_share_hs_official_cash_option_rights_v1/cash_option_terms_v1.parquet"
)
PARENT_FORMAL_RECEIPT = (
    ROOT / "data/staging/a_share_hs_official_cash_option_floor_alpha_v1/market_acquisition_receipt.json"
)
SINA_ROOT = ROOT / "data/raw/a_share_hs_official_cash_option_floor_alpha_v1/sina_history"
CORE_FILES = [
    CONFIG_PATH,
    ROOT / "docs/A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1_0_1_SOURCE_REMEDIATION.md",
    ROOT / "scripts/freeze_a_share_hs_cash_option_floor_alpha_v1_0_1_source_remediation.py",
    ROOT / "tests/test_a_share_hs_cash_option_floor_alpha_v1_0_1_source_remediation.py",
]


class SourceRemediationError(ValueError):
    """数据源修复协议不满足冻结约束。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise SourceRemediationError(
            f"数据源修复字段漂移：{field}，预期={expected!r}，实际={actual!r}"
        )


def require_false(mapping: Mapping[str, Any], key: str) -> None:
    if mapping.get(key) is not False:
        raise SourceRemediationError(f"数据源修复治理字段必须为 false：{key}")


def validate_remediation_config(config: Mapping[str, Any]) -> None:
    require_equal(config.get("remediation_id"), REMEDIATION_ID, "remediation_id")
    require_equal(config.get("evidence_cutoff"), "2026-08-14", "evidence_cutoff")
    require_equal(config.get("research_scope"), "DISCOVERY_ONLY", "research_scope")
    parent = config.get("parent_protocol") or {}
    exact_parent = {
        "protocol_id": "A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1",
        "protocol_manifest_sha256": "3314ff14320686373422b93aa5fb2f80d00b752da10c1cc67dc81f39a3301f2d",
        "protocol_content_sha256": "b632081712f9b3ec1f4d7caab869dbe29e31bdafae44fd46feaae3e2a96f6f65",
        "config_sha256": "58d87400ff86648dbe9f215a4797987227ff2b8afbcac58bc2799a5c8dd6f92b",
        "core_module_sha256": "9eadb8162827d2ef4f6ab38f0a9549c0bfb7e25370a71c57d29d0d80f6693e55",
        "expected_event_count": 21,
    }
    for key, expected in exact_parent.items():
        require_equal(parent.get(key), expected, f"parent_protocol.{key}")

    trigger = config.get("trigger") or {}
    exact_trigger = {
        "attempt_receipt_sha256": "dd17bab3f6ccde51c1635620455e50bb24dbe2827eecc7d03362c209924cb0fb",
        "attempt_status": "ACQUISITION_ATTEMPT_NETWORK_OR_PARSE_FAILURE",
        "failure_count": 26,
        "tushare_daily_failure_count": 21,
        "tushare_trade_calendar_failure_count": 2,
        "tushare_stock_basic_failure_count": 3,
        "required_error_code": "40101",
        "required_error_meaning": "TOKEN_EXPIRED",
        "both_audited_proxy_nodes_failed": True,
        "formal_market_receipt_created": False,
        "source_change_triggered_by_observed_price_or_spread": False,
    }
    for key, expected in exact_trigger.items():
        require_equal(trigger.get(key), expected, f"trigger.{key}")

    preserved = config.get("preserved_parent_contract") or {}
    require_equal(
        preserved.get("minimum_net_conditional_floor_return"),
        "INHERIT_PARENT_0_08",
        "preserved_parent_contract.minimum_net_conditional_floor_return",
    )
    require_equal(
        preserved.get("source_contract_is_only_changed_component"),
        True,
        "preserved_parent_contract.source_contract_is_only_changed_component",
    )
    require_false(preserved, "parameter_search_allowed")

    observed = config.get("information_observed_before_remediation_freeze") or {}
    exact_observed = {
        "sina_raw_response_count": 21,
        "sina_raw_responses_saved_before_remediation_freeze": True,
        "sina_full_history_ohlc_machine_decoded_to_filter_dates": True,
        "operator_visible_ohlc_values": False,
        "cash_option_spreads_computed": False,
        "price_screen_completed": False,
        "qualified_event_count_known": False,
        "future_subject_values_used_for_screening": False,
        "coverage_row_counts_and_dates_observed": True,
        "events_with_at_least_one_sina_row_in_entry_window": 13,
        "events_with_zero_sina_rows_in_entry_window": 8,
        "source_remediation_choice_based_on_coverage_counts": False,
        "source_remediation_choice_based_only_on_parent_primary_access_failure": True,
        "eastmoney_price_values_read": False,
        "baostock_calendar_values_read": False,
    }
    for key, expected in exact_observed.items():
        require_equal(
            observed.get(key), expected, f"information_observed_before_remediation_freeze.{key}"
        )

    market = config.get("remediated_market_data_contract") or {}
    exact_market = {
        "subject_primary_source": "SINA_FINANCE_HISTDATA_KLC2_RAW_UNADJUSTED",
        "subject_primary_raw_responses_must_be_pre_freeze_immutable_files": True,
        "subject_primary_raw_response_count": 21,
        "subject_secondary_source": "EASTMONEY_PUSH2HIS_KLINE_RAW_UNADJUSTED",
        "subject_secondary_url": "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        "eastmoney_raw_response_preservation_required": True,
        "eastmoney_response_security_code_match_required": True,
        "eastmoney_adjustment": "NONE_FQT_0",
        "independent_second_source_required_for_entry_row": True,
        "same_provider_repackaging_counts_as_independent": False,
        "entry_date_exact_match_required": True,
        "entry_ohlc_cross_source_max_absolute_difference_cny": 0.005,
        "primary_pre_close_used_for_one_price_up_rule": True,
        "secondary_pre_close_not_required": True,
        "calendar_primary_source": "BAOSTOCK_QUERY_TRADE_DATES",
        "calendar_start": "2006-09-23",
        "calendar_end": "2026-02-06",
        "calendar_raw_csv_preservation_required": True,
        "calendar_login_and_query_receipt_required": True,
        "raw_response_sha256_required": True,
        "request_parameters_without_secrets_required": True,
        "no_silent_source_substitution": True,
        "unresolved_source_conflict_disposition": "NO_VIEW_MARKET_DATA",
        "unresolved_missing_secondary_entry_row_disposition": "NO_VIEW_MARKET_DATA",
        "two_sources_both_return_no_rows_in_complete_window_disposition": "NO_REPLICABLE_ENTRY_WINDOW",
    }
    for key, expected in exact_market.items():
        require_equal(market.get(key), expected, f"remediated_market_data_contract.{key}")
    request = market.get("subject_secondary_request") or {}
    exact_request = {
        "klt": "101",
        "fqt": "0",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "beg": "EFFECTIVE_ANNOUNCEMENT_DATE_MINUS_30_CALENDAR_DAYS",
        "end": "RIGHTS_REGISTRATION_DATE",
        "secid": "1.DIGITS_FOR_SH_OR_0.DIGITS_FOR_SZ",
    }
    for key, expected in exact_request.items():
        require_equal(request.get(key), expected, f"subject_secondary_request.{key}")

    governance = config.get("governance") or {}
    require_equal(governance.get("source_access_remediation_versioned"), True, "governance.source_access_remediation_versioned")
    for key in (
        "parent_protocol_edited",
        "parent_result_rescued",
        "event_set_changed",
        "threshold_changed",
        "costs_changed",
        "benchmark_changed",
        "acceptance_gates_changed",
        "eastmoney_price_values_read_before_remediation_freeze",
        "baostock_calendar_values_read_before_remediation_freeze",
        "observed_sina_ohlc_used_to_change_any_parameter",
        "observed_sina_coverage_used_to_exclude_any_event",
        "liquidity_filter_applied",
        "position_mapping",
        "order_generation",
        "shadow_authorized",
        "live_authorized",
    ):
        require_false(governance, key)


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_remediation_config(config)
    return config


def verify_parent(config: dict[str, Any]) -> dict[str, Any]:
    parent_config = load_parent_config()
    verification = verify_parent_protocol(parent_config)
    parent = config["parent_protocol"]
    paths = {
        "protocol_manifest": ROOT / str(parent["protocol_manifest"]),
        "config": ROOT / str(parent["config"]),
        "core_module": ROOT / str(parent["core_module"]),
    }
    expected = {
        "protocol_manifest": parent["protocol_manifest_sha256"],
        "config": parent["config_sha256"],
        "core_module": parent["core_module_sha256"],
    }
    for key, path in paths.items():
        if sha256_file(path) != expected[key]:
            raise SourceRemediationError(f"父协议文件哈希漂移：{relative(path)}")
    if verification["content_sha256"] != parent["protocol_content_sha256"]:
        raise SourceRemediationError("父协议内容哈希漂移")
    if PARENT_FORMAL_RECEIPT.exists():
        raise SourceRemediationError("父协议意外生成了正式市场回执，触发事实已改变")
    return {
        "status": "PASS_PARENT_FROZEN_PROTOCOL_UNCHANGED",
        "protocol_manifest_sha256": sha256_file(paths["protocol_manifest"]),
        "config_sha256": sha256_file(paths["config"]),
        "core_module_sha256": sha256_file(paths["core_module"]),
        "content_sha256": verification["content_sha256"],
        "formal_market_receipt_exists": False,
    }


def verify_trigger(config: dict[str, Any]) -> dict[str, Any]:
    trigger = config["trigger"]
    path = ROOT / str(trigger["attempt_receipt"])
    if sha256_file(path) != trigger["attempt_receipt_sha256"]:
        raise SourceRemediationError("TuShare失败尝试回执哈希漂移")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != trigger["attempt_status"]:
        raise SourceRemediationError("TuShare失败尝试状态漂移")
    failures = payload.get("failures") or []
    if len(failures) != trigger["failure_count"]:
        raise SourceRemediationError("TuShare失败数量漂移")
    counts = Counter(str(item.get("source") or "") for item in failures)
    exact_counts = {
        "TUSHARE_PRO_DAILY_RAW_UNADJUSTED": trigger["tushare_daily_failure_count"],
        "TUSHARE_PRO_TRADE_CAL": trigger["tushare_trade_calendar_failure_count"],
        "TUSHARE_PRO_STOCK_BASIC": trigger["tushare_stock_basic_failure_count"],
    }
    if counts != Counter(exact_counts):
        raise SourceRemediationError(f"TuShare失败来源计数漂移：{dict(counts)}")
    if any(trigger["required_error_code"] not in str(item.get("error") or "") for item in failures):
        raise SourceRemediationError("存在不是40101的TuShare失败")
    if payload.get("future_subject_values_used_for_screening") is not False:
        raise SourceRemediationError("失败尝试错误地使用了未来标的值筛选")
    return {
        "status": "PASS_TUSHARE_40101_TRIGGER_VERIFIED",
        "attempt_receipt": relative(path),
        "attempt_receipt_sha256": sha256_file(path),
        "failure_count": len(failures),
        "failure_source_counts": dict(sorted(counts.items())),
        "all_failures_contain_40101": True,
        "formal_market_receipt_created": False,
    }


def inventory_sina_raw(config: dict[str, Any]) -> dict[str, Any]:
    terms = pd.read_parquet(TERMS_PATH, columns=["event_id", "ts_code"])
    if len(terms) != 21 or terms["event_id"].astype(str).duplicated().any():
        raise SourceRemediationError("冻结条款事件集不是21个唯一事件")
    files: list[dict[str, Any]] = []
    for row in terms.sort_values(["ts_code", "event_id"], kind="stable").itertuples(index=False):
        path = SINA_ROOT / f"{safe_name(str(row.ts_code))}.js"
        if not path.is_file() or path.stat().st_size <= 0:
            raise SourceRemediationError(f"新浪原始响应缺失：{relative(path)}")
        files.append(
            {
                "event_id": str(row.event_id),
                "ts_code": str(row.ts_code),
                "path": relative(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    if len({item["path"] for item in files}) != 21:
        raise SourceRemediationError("新浪原始响应没有21个唯一文件")
    expected_count = config["remediated_market_data_contract"]["subject_primary_raw_response_count"]
    if len(files) != expected_count:
        raise SourceRemediationError("新浪原始响应数量偏离修复协议")
    return {
        "status": "PASS_PREEXISTING_SINA_RAW_INVENTORY_VERIFIED",
        "file_count": len(files),
        "files": files,
        "operator_visible_ohlc_values": False,
        "cash_option_spreads_computed": False,
    }


def artifact_path(config: dict[str, Any], key: str) -> Path:
    return ROOT / str(config["artifacts"][key])


def validate_plan(config: dict[str, Any]) -> dict[str, Any]:
    parent = verify_parent(config)
    trigger = verify_trigger(config)
    sina = inventory_sina_raw(config)
    missing = [relative(path) for path in CORE_FILES if not path.exists()]
    return {
        "status": (
            "PASS_SOURCE_REMEDIATION_PLAN_VALIDATED"
            if not missing
            else "FAILED_SOURCE_REMEDIATION_CORE_FILES_MISSING"
        ),
        "parent_verification": parent,
        "trigger_verification": trigger,
        "sina_raw_inventory": sina,
        "missing_core_files": missing,
        "protocol_manifest_exists": artifact_path(config, "protocol_manifest").exists(),
        "eastmoney_price_values_read": False,
        "baostock_calendar_values_read": False,
        "price_screen_completed": False,
    }


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    parent = verify_parent(config)
    trigger = verify_trigger(config)
    sina = inventory_sina_raw(config)
    path = artifact_path(config, "protocol_manifest")
    if path.exists():
        raise SourceRemediationError("数据源修复冻结清单已存在，禁止覆盖")
    missing = [relative(item) for item in CORE_FILES if not item.exists()]
    if missing:
        raise SourceRemediationError(f"数据源修复核心文件缺失：{missing}")
    files = [{"path": relative(item), "sha256": sha256_file(item)} for item in CORE_FILES]
    content = {item["path"]: item["sha256"] for item in files}
    manifest = {
        "remediation_id": REMEDIATION_ID,
        "status": PROTOCOL_STATUS,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "evidence_cutoff": config["evidence_cutoff"],
        "files": files,
        "content_sha256": hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest(),
        "parent_verification": parent,
        "trigger_verification": trigger,
        "sina_raw_inventory": sina,
        "source_contract_is_only_changed_component": True,
        "parent_threshold_changed": False,
        "parent_costs_changed": False,
        "parent_acceptance_gates_changed": False,
        "operator_visible_ohlc_values_before_freeze": False,
        "cash_option_spreads_computed_before_freeze": False,
        "eastmoney_price_values_read": False,
        "baostock_calendar_values_read": False,
        "price_screen_completed": False,
        "position_mapping": False,
        "order_generation": False,
        "shadow_authorized": False,
        "live_authorized": False,
    }
    atomic_text(path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest["manifest_sha256"] = sha256_file(path)
    return manifest


def verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    parent = verify_parent(config)
    trigger = verify_trigger(config)
    sina = inventory_sina_raw(config)
    path = artifact_path(config, "protocol_manifest")
    if not path.exists():
        raise SourceRemediationError("数据源修复冻结清单不存在")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    if manifest.get("status") != PROTOCOL_STATUS:
        failures.append({"failure": "STATUS", "actual": manifest.get("status")})
    for flag in (
        "parent_threshold_changed",
        "parent_costs_changed",
        "parent_acceptance_gates_changed",
        "operator_visible_ohlc_values_before_freeze",
        "cash_option_spreads_computed_before_freeze",
        "eastmoney_price_values_read",
        "baostock_calendar_values_read",
        "price_screen_completed",
        "position_mapping",
        "order_generation",
        "shadow_authorized",
        "live_authorized",
    ):
        if manifest.get(flag) is not False:
            failures.append({"failure": "GOVERNANCE_FLAG", "field": flag, "actual": manifest.get(flag)})
    for item in manifest.get("files") or []:
        item_path = ROOT / str(item["path"])
        actual = sha256_file(item_path) if item_path.exists() else None
        if actual != item["sha256"]:
            failures.append(
                {
                    "failure": "FILE_HASH",
                    "path": item["path"],
                    "expected": item["sha256"],
                    "actual": actual,
                }
            )
    frozen_sina = {
        item["path"]: item["sha256"]
        for item in (manifest.get("sina_raw_inventory") or {}).get("files") or []
    }
    current_sina = {item["path"]: item["sha256"] for item in sina["files"]}
    if frozen_sina != current_sina:
        failures.append({"failure": "SINA_RAW_INVENTORY_DRIFT"})
    if failures:
        raise SourceRemediationError(json.dumps(failures, ensure_ascii=False, default=str))
    return {
        "status": "PASS_FROZEN_SOURCE_REMEDIATION_VERIFIED",
        "protocol_manifest": relative(path),
        "protocol_manifest_sha256": sha256_file(path),
        "content_sha256": manifest["content_sha256"],
        "checked_core_file_count": len(manifest["files"]),
        "sina_raw_file_count": len(sina["files"]),
        "parent_verification": parent,
        "trigger_verification": trigger,
        "eastmoney_price_values_read": False,
        "baostock_calendar_values_read": False,
        "price_screen_completed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在读取东方财富和BaoStock前冻结现金选择权V1.0.1纯数据源修复"
    )
    parser.add_argument(
        "--mode", choices=("validate-plan", "freeze-protocol", "verify-protocol"), default="validate-plan"
    )
    args = parser.parse_args()
    config = load_config()
    if args.mode == "validate-plan":
        result = validate_plan(config)
    elif args.mode == "freeze-protocol":
        result = freeze_protocol(config)
    else:
        result = verify_protocol(config)
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
