from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_hs_partial_cash_tender_20x20_source_v1 import (  # noqa: E402
    PROTOCOL_ID,
    build_lifecycle_candidate_ledger,
    build_machine_source_ledger,
    canonical_frame_sha256_payload,
    validate_source_inputs,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
CONFIG_PATH = ROOT / "config/a_share_hs_official_partial_cash_tender_20x20_v1.yaml"
PROTOCOL_DOC = (
    ROOT / "docs/A_SHARE_HS_OFFICIAL_PARTIAL_CASH_TENDER_20X20_V1_PROTOCOL.md"
)
MODULE_PATH = ROOT / "research/a_share_hs_partial_cash_tender_20x20_source_v1.py"
RUNNER_PATH = Path(__file__).resolve()
TEST_PATH = ROOT / "tests/test_a_share_hs_partial_cash_tender_20x20_source_v1.py"
PROTOCOL_STATUS = (
    "FROZEN_PARTIAL_TENDER_20X20_PROTOCOL_AND_INPUTS_"
    "BEFORE_PARTIAL_SUBJECT_MARKET_READ"
)
SOURCE_RUN_STATUS = "PASS_PARTIAL_TENDER_20X20_OFFICIAL_SOURCE_EXTRACTION_COMPLETE"


PARQUET_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "formal_report_text_review": {
        "announcement_id",
        "ts_code",
        "official_pdf_date",
        "extracted_text",
        "offer_price_candidates_json",
        "offer_period_candidates_json",
        "pdf_path",
        "pdf_sha256",
    },
    "formal_report_adjudication_ledger": {
        "announcement_id",
        "ts_code",
        "official_pdf_date",
        "reason_code",
        "market_price_read",
        "future_return_read",
        "tender_outcome_read",
    },
    "cninfo_metadata_archive": {
        "announcement_id",
        "ts_code",
        "official_pdf_date",
        "announcement_title",
        "pdf_path",
        "pdf_sha256",
        "status",
    },
    "unified_daily_market": {
        "con_code",
        "date",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "amount",
        "volume",
        "total_return_open",
        "total_return_close",
        "is_suspended",
        "observed_traded_row",
    },
    "industry_peer_return_panel": {
        "con_code",
        "date",
        "industry_l1_code",
        "industry_peer_count",
        "industry_peer_log_return",
    },
    "security_master": {
        "ts_code",
        "exchange",
        "security_type",
        "list_date",
        "delist_date",
        "status_from",
        "status_to",
        "available_at",
    },
    "security_status_intervals": {
        "ts_code",
        "exchange",
        "status",
        "valid_from",
        "valid_to",
        "available_at",
    },
    "trading_calendar": {"exchange", "date", "is_open", "available_at"},
    "h00300_close_reference": {"date", "symbol", "close", "source"},
    "csi300_price_index_ohlc_bridge": {
        "date",
        "open",
        "high",
        "low",
        "close",
        "symbol",
        "source",
    },
}


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


def frame_sha256(frame: pd.DataFrame) -> str:
    payload = canonical_frame_sha256_payload(frame)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("20×20配置不是YAML对象")
    return value


def project_path(config: dict[str, Any], section: str, key: str) -> Path:
    return ROOT / str(config[section][key])


def validate_static_contract(config: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if config.get("protocol_id") != PROTOCOL_ID:
        failures.append("PROTOCOL_ID_MISMATCH")
    source = config.get("source_universe") or {}
    clock = config.get("clock_and_execution") or {}
    costs = config.get("costs") or {}
    gates = config.get("acceptance_gates") or {}
    governance = config.get("governance") or {}
    if source.get("minimum_offer_quantity_share_of_total_shares") != 0.20:
        failures.append("SOURCE_OFFER_RATIO_NOT_20_PERCENT")
    if clock.get("minimum_gross_offer_premium") != 0.20:
        failures.append("MARKET_PREMIUM_NOT_20_PERCENT")
    if clock.get("entry_limit_price_formula") != "OFFER_PRICE_DIV_1_20":
        failures.append("ENTRY_LIMIT_FORMULA_MISMATCH")
    for key in ("liquidity_filter", "market_cap_filter", "adv_filter", "turnover_filter"):
        if source.get(key) is not False:
            failures.append(f"FORBIDDEN_FILTER_ENABLED:{key}")
    if source.get("st_exclusion") is not False:
        failures.append("FORBIDDEN_ST_FILTER_ENABLED")
    if source.get("minimum_acceptance_condition_allowed") is not False:
        failures.append("MINIMUM_ACCEPTANCE_CONDITION_ALLOWED")
    stress = costs.get("stress") or {}
    if [stress.get(key) for key in (
        "buy_bps",
        "accepted_tender_and_settlement_bps",
        "residual_or_failed_sale_bps",
    )] != [100, 100, 100]:
        failures.append("STRESS_LEG_COSTS_NOT_100_BPS_EACH")
    if gates.get("stress_industry_excess_win_rate_at_least") != 0.75:
        failures.append("HIGH_ODDS_WIN_RATE_GATE_DRIFT")
    if gates.get("stress_industry_excess_payoff_ratio_at_least") != 2.0:
        failures.append("HIGH_PAYOFF_RATIO_GATE_DRIFT")
    if gates.get("stress_industry_excess_profit_factor_at_least") != 4.0:
        failures.append("HIGH_PROFIT_FACTOR_GATE_DRIFT")
    required_false = (
        "partial_tender_subject_market_price_read_before_protocol_freeze",
        "partial_tender_subject_future_return_read_before_protocol_freeze",
        "observed_partial_tender_premium_filter_applied_before_protocol_freeze",
        "parameter_search_allowed",
        "position_mapping",
        "order_generation",
        "shadow_authorized",
        "live_authorized",
    )
    for key in required_false:
        if governance.get(key) is not False:
            failures.append(f"GOVERNANCE_FALSE_REQUIRED:{key}")
    if governance.get("independent_new_study_not_prior_result_rescue") is not True:
        failures.append("INDEPENDENT_STUDY_FLAG_MISSING")
    if failures:
        raise RuntimeError("20×20静态合同失败：" + json.dumps(failures, ensure_ascii=False))
    return {
        "status": "PASS_PARTIAL_TENDER_20X20_STATIC_CONTRACT_VALIDATED",
        "offer_quantity_ratio_floor": 0.20,
        "gross_offer_premium_floor": 0.20,
        "stress_leg_cost_bps": [100, 100, 100],
        "liquidity_filter": False,
        "market_cap_filter": False,
        "parameter_search_allowed": False,
    }


def verify_manifest_tree(path: Path, expected_status: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"上游冻结清单不存在：{relative(path)}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != expected_status:
        raise RuntimeError(
            f"上游冻结状态错误：{relative(path)} {manifest.get('status')}"
        )
    raw_files = manifest.get("files") or {}
    if isinstance(raw_files, dict):
        tracked = {str(key): str(value) for key, value in raw_files.items()}
    elif isinstance(raw_files, list):
        tracked = {
            str(item["path"]): str(item["sha256"])
            for item in [*raw_files, *(manifest.get("references") or [])]
        }
    else:
        raise RuntimeError(f"上游冻结清单files结构非法：{relative(path)}")
    failures: list[dict[str, str | None]] = []
    for item_path, expected_hash in tracked.items():
        item = ROOT / item_path
        actual = sha256_file(item) if item.exists() else None
        if actual != expected_hash:
            failures.append(
                {
                    "path": item_path,
                    "expected": expected_hash,
                    "actual": actual,
                }
            )
    if failures:
        raise RuntimeError(
            "上游冻结文件哈希失败：" + json.dumps(failures, ensure_ascii=False)
        )
    return {
        "status": "PASS_UPSTREAM_MANIFEST_TREE_VERIFIED",
        "manifest": relative(path),
        "manifest_sha256": sha256_file(path),
        "checked_file_count": len(tracked),
    }


def verify_upstream(config: dict[str, Any]) -> dict[str, Any]:
    results_path = project_path(
        config, "inputs", "prior_full_tender_results_archive_manifest"
    )
    results = verify_manifest_tree(
        results_path, "FROZEN_TENDER_EVALUATION_RESULTS_NO_PARAMETER_RESCUE"
    )
    results_manifest = json.loads(results_path.read_text(encoding="utf-8"))
    expected_historical_status = (
        "NO_VIEW_INSUFFICIENT_FULL_CASH_TENDER_SPREAD_EVIDENCE"
    )
    if results_manifest.get("historical_status") != expected_historical_status:
        raise RuntimeError("前序全面现金要约历史状态漂移")
    if results_manifest.get("parameter_rescue_applied") is not False:
        raise RuntimeError("前序全面现金要约结果存在救参标记")
    event_path = project_path(
        config, "inputs", "prior_formal_report_event_archive_manifest"
    )
    event_archive = verify_manifest_tree(
        event_path, "FROZEN_BEFORE_MARKET_PRICE_AND_TENDER_OUTCOME_READ"
    )
    event_manifest = json.loads(event_path.read_text(encoding="utf-8"))
    for key in ("market_price_read", "future_return_read", "tender_outcome_read"):
        if event_manifest.get(key) is not False:
            raise RuntimeError(f"前序正式报告来源冻结标记非法：{key}")
    return {
        "status": "PASS_PARTIAL_TENDER_20X20_UPSTREAM_VERIFIED",
        "prior_full_tender_results": results,
        "prior_full_tender_historical_status": expected_historical_status,
        "formal_report_source_archive": event_archive,
        "partial_subject_market_price_read": False,
        "partial_subject_future_return_read": False,
        "partial_tender_outcome_read": False,
    }


def validate_parquet_metadata(config: dict[str, Any]) -> dict[str, Any]:
    expected_rows = config.get("expected_rows") or {}
    rows: dict[str, int] = {}
    for key, required_columns in PARQUET_REQUIRED_COLUMNS.items():
        path = project_path(config, "inputs", key)
        if not path.exists():
            raise RuntimeError(f"冻结输入不存在：{relative(path)}")
        parquet = pq.ParquetFile(path)
        actual_rows = int(parquet.metadata.num_rows)
        expected = int(expected_rows[key])
        if actual_rows != expected:
            raise RuntimeError(f"Parquet行数漂移：{key} {actual_rows}!={expected}")
        missing = sorted(required_columns - set(parquet.schema_arrow.names))
        if missing:
            raise RuntimeError(f"Parquet字段缺失：{key} {missing}")
        rows[key] = actual_rows
    return {
        "status": "PASS_PARTIAL_TENDER_20X20_PARQUET_METADATA_VALIDATED",
        "row_counts": rows,
        "partial_subject_market_values_read": False,
        "partial_subject_future_returns_read": False,
        "benchmark_values_read": False,
    }


def load_official_source_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    text_review = pd.read_parquet(
        project_path(config, "inputs", "formal_report_text_review")
    )
    adjudication = pd.read_parquet(
        project_path(config, "inputs", "formal_report_adjudication_ledger")
    )
    metadata = pd.read_parquet(
        project_path(config, "inputs", "cninfo_metadata_archive")
    )
    return text_review, adjudication, metadata


def protocol_file_paths(config: dict[str, Any]) -> list[Path]:
    paths = [CONFIG_PATH, PROTOCOL_DOC, MODULE_PATH, RUNNER_PATH, TEST_PATH]
    paths.extend(ROOT / str(value) for value in config["inputs"].values())
    unique = {path.resolve(): path for path in paths}
    return [unique[key] for key in sorted(unique, key=lambda item: item.as_posix())]


def source_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    keys = ("machine_source_ledger", "lifecycle_candidate_ledger", "source_run_receipt")
    return {key: project_path(config, "artifacts", key) for key in keys}


def validate_plan(config: dict[str, Any]) -> dict[str, Any]:
    static = validate_static_contract(config)
    upstream = verify_upstream(config)
    metadata = validate_parquet_metadata(config)
    text_review, adjudication, cninfo_metadata = load_official_source_inputs(config)
    source_validation = validate_source_inputs(
        text_review, adjudication, cninfo_metadata
    )
    existing_outputs = [
        relative(path) for path in source_output_paths(config).values() if path.exists()
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_PARTIAL_TENDER_20X20_PLAN_VALIDATED",
        "static_validation": static,
        "upstream_verification": upstream,
        "parquet_metadata_validation": metadata,
        "official_source_validation": source_validation,
        "existing_source_outputs": existing_outputs,
        "protocol_manifest_exists": project_path(
            config, "artifacts", "protocol_manifest"
        ).exists(),
        "partial_subject_market_price_read": False,
        "partial_subject_future_return_read": False,
        "partial_tender_outcome_read": False,
        "observed_partial_tender_premium_filter_applied": False,
    }


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config, "artifacts", "protocol_manifest")
    if manifest_path.exists():
        raise RuntimeError(f"20×20协议已冻结，禁止覆盖：{relative(manifest_path)}")
    plan = validate_plan(config)
    if plan["existing_source_outputs"]:
        raise RuntimeError("源解析产物在协议冻结前已经存在")
    tracked_paths = protocol_file_paths(config)
    missing = [relative(path) for path in tracked_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"协议冻结前缺少文件：{missing}")
    files = {relative(path): sha256_file(path) for path in tracked_paths}
    manifest = {
        "protocol_id": f"{PROTOCOL_ID}_FREEZE",
        "status": PROTOCOL_STATUS,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "evidence_cutoff": config["evidence_cutoff"],
        "files": files,
        "content_sha256": hashlib.sha256(
            canonical_json(files).encode("utf-8")
        ).hexdigest(),
        "plan_validation": plan,
        "candidate_definition": {
            "minimum_offer_quantity_ratio": 0.20,
            "minimum_gross_offer_premium": 0.20,
            "entry_limit_formula": "OFFER_PRICE_DIV_1_20",
        },
        "independent_new_study_not_prior_result_rescue": True,
        "prior_full_tender_branch_historical_status": (
            "NO_VIEW_INSUFFICIENT_FULL_CASH_TENDER_SPREAD_EVIDENCE"
        ),
        "partial_subject_market_price_read": False,
        "partial_subject_future_return_read": False,
        "partial_tender_outcome_read": False,
        "observed_partial_tender_premium_filter_applied": False,
        "liquidity_filter_applied": False,
        "market_cap_filter_applied": False,
        "parameter_search_applied": False,
        "position_mapping": False,
        "order_generation": False,
        "shadow_authorized": False,
        "live_authorized": False,
        "immutable_rule": (
            "冻结后候选、输入、实现、时钟、成本、基准和门槛只能通过新版本变更"
        ),
    }
    atomic_write_text(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config, "artifacts", "protocol_manifest")
    if not manifest_path.exists():
        raise RuntimeError("20×20协议冻结清单不存在")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[dict[str, str | None]] = []
    if manifest.get("status") != PROTOCOL_STATUS:
        failures.append(
            {
                "path": relative(manifest_path),
                "expected": PROTOCOL_STATUS,
                "actual": str(manifest.get("status")),
            }
        )
    files = manifest.get("files") or {}
    if not isinstance(files, dict) or not files:
        raise RuntimeError("20×20协议冻结清单缺少files映射")
    for item_path, expected_hash in files.items():
        item = ROOT / item_path
        actual = sha256_file(item) if item.exists() else None
        if actual != expected_hash:
            failures.append(
                {"path": item_path, "expected": expected_hash, "actual": actual}
            )
    if failures:
        raise RuntimeError(
            "20×20协议冻结哈希失败：" + json.dumps(failures, ensure_ascii=False)
        )
    return {
        "status": "PASS_PARTIAL_TENDER_20X20_PROTOCOL_FREEZE_VERIFIED",
        "checked_file_count": len(files),
        "manifest_sha256": sha256_file(manifest_path),
        "content_sha256": manifest["content_sha256"],
        "upstream_verification": verify_upstream(config),
        "parquet_metadata_validation": validate_parquet_metadata(config),
        "partial_subject_market_price_read_before_freeze": False,
        "partial_subject_future_return_read_before_freeze": False,
        "observed_partial_tender_premium_filter_applied_before_freeze": False,
    }


def source_summary(
    machine: pd.DataFrame, lifecycle: pd.DataFrame
) -> dict[str, Any]:
    ratio = pd.to_numeric(machine["machine_offer_quantity_ratio"], errors="coerce")
    return {
        "machine_source_row_count": len(machine),
        "machine_source_unique_security_count": machine["ts_code"].nunique(),
        "machine_ratio_resolved_count": int(ratio.notna().sum()),
        "machine_ratio_at_least_20_percent_count": int(ratio.ge(0.20).sum()),
        "machine_potential_20x20_source_count": int(
            machine["machine_status"].eq(
                "POTENTIAL_20X20_SOURCE_EVENT_MANUAL_CONFIRMATION_REQUIRED"
            ).sum()
        ),
        "machine_high_priority_review_count": int(
            machine["manual_review_priority"].eq("HIGH").sum()
        ),
        "minimum_acceptance_condition_detected_count": int(
            machine["minimum_acceptance_condition_detected"].sum()
        ),
        "remaining_material_condition_detected_count": int(
            machine["remaining_material_condition_detected"].sum()
        ),
        "lifecycle_candidate_row_count": len(lifecycle),
        "lifecycle_candidate_unique_security_count": lifecycle["ts_code"].nunique(),
        "machine_source_canonical_sha256": frame_sha256(machine),
        "lifecycle_candidate_canonical_sha256": frame_sha256(lifecycle),
    }


def run_source(config: dict[str, Any]) -> dict[str, Any]:
    protocol = verify_protocol(config)
    outputs = source_output_paths(config)
    existing = [relative(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError(f"源解析产物已存在，禁止覆盖：{existing}")
    text_review, adjudication, metadata = load_official_source_inputs(config)
    source_validation = validate_source_inputs(text_review, adjudication, metadata)
    machine = build_machine_source_ledger(text_review, adjudication)
    lifecycle = build_lifecycle_candidate_ledger(metadata, machine)
    summary = source_summary(machine, lifecycle)
    atomic_write_parquet(outputs["machine_source_ledger"], machine)
    atomic_write_parquet(outputs["lifecycle_candidate_ledger"], lifecycle)
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "status": SOURCE_RUN_STATUS,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "evidence_cutoff": config["evidence_cutoff"],
        "protocol_verification": protocol,
        "official_source_validation": source_validation,
        **summary,
        "machine_source_ledger": relative(outputs["machine_source_ledger"]),
        "machine_source_ledger_sha256": sha256_file(
            outputs["machine_source_ledger"]
        ),
        "lifecycle_candidate_ledger": relative(
            outputs["lifecycle_candidate_ledger"]
        ),
        "lifecycle_candidate_ledger_sha256": sha256_file(
            outputs["lifecycle_candidate_ledger"]
        ),
        "partial_subject_market_price_read": False,
        "partial_subject_future_return_read": False,
        "partial_tender_outcome_read": False,
        "observed_partial_tender_premium_filter_applied": False,
        "market_input_values_read": False,
        "benchmark_input_values_read": False,
        "source_only_not_trade_candidate_yet": True,
        "manual_adjudication_required": True,
        "position_mapping": False,
        "order_generation": False,
        "shadow_authorized": False,
        "live_authorized": False,
    }
    atomic_write_text(
        outputs["source_run_receipt"],
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
    )
    return receipt


def verify_source(config: dict[str, Any]) -> dict[str, Any]:
    protocol = verify_protocol(config)
    outputs = source_output_paths(config)
    missing = [relative(path) for path in outputs.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"源解析产物缺失：{missing}")
    receipt = json.loads(outputs["source_run_receipt"].read_text(encoding="utf-8"))
    failures: list[str] = []
    if receipt.get("status") != SOURCE_RUN_STATUS:
        failures.append(f"STATUS:{receipt.get('status')}")
    for key in ("machine_source_ledger", "lifecycle_candidate_ledger"):
        actual = sha256_file(outputs[key])
        if actual != receipt.get(f"{key}_sha256"):
            failures.append(f"FILE_HASH:{key}")
    stored_machine = pd.read_parquet(outputs["machine_source_ledger"])
    stored_lifecycle = pd.read_parquet(outputs["lifecycle_candidate_ledger"])
    text_review, adjudication, metadata = load_official_source_inputs(config)
    rebuilt_machine = build_machine_source_ledger(text_review, adjudication)
    rebuilt_lifecycle = build_lifecycle_candidate_ledger(metadata, rebuilt_machine)
    stored_summary = source_summary(stored_machine, stored_lifecycle)
    rebuilt_summary = source_summary(rebuilt_machine, rebuilt_lifecycle)
    if canonical_json(stored_summary) != canonical_json(rebuilt_summary):
        failures.append("STORED_OUTPUT_REBUILD_MISMATCH")
    for key, value in rebuilt_summary.items():
        if receipt.get(key) != value:
            failures.append(f"RECEIPT_SUMMARY:{key}")
    for frame_name, frame in (
        ("machine_source", stored_machine),
        ("lifecycle_candidate", stored_lifecycle),
    ):
        for flag in ("market_price_read", "future_return_read", "tender_outcome_read"):
            if flag not in frame.columns or frame[flag].fillna(True).astype(bool).any():
                failures.append(f"FORBIDDEN_READ_FLAG:{frame_name}:{flag}")
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": (
            "PASS_PARTIAL_TENDER_20X20_SOURCE_OUTPUTS_VERIFIED"
            if not failures
            else "FAILED_PARTIAL_TENDER_20X20_SOURCE_OUTPUTS_VERIFICATION"
        ),
        "failure_count": len(failures),
        "failures": failures,
        **rebuilt_summary,
        "protocol_verification": protocol,
        "partial_subject_market_price_read": False,
        "partial_subject_future_return_read": False,
        "partial_tender_outcome_read": False,
        "market_input_values_read": False,
        "position_mapping": False,
        "order_generation": False,
        "live_authorized": False,
    }
    if failures:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="冻结并构建A股正式部分现金要约20×20官方来源档案"
    )
    parser.add_argument(
        "--mode",
        choices=(
            "validate-plan",
            "freeze-protocol",
            "verify-protocol",
            "run-source",
            "verify-source",
        ),
        default="validate-plan",
    )
    args = parser.parse_args()
    config = load_config()
    if args.mode == "validate-plan":
        result = validate_plan(config)
    elif args.mode == "freeze-protocol":
        result = freeze_protocol(config)
    elif args.mode == "verify-protocol":
        result = verify_protocol(config)
    elif args.mode == "run-source":
        result = run_source(config)
    else:
        result = verify_source(config)
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
