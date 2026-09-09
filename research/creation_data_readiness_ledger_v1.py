from __future__ import annotations

import json
import re
from datetime import datetime, time
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.frozen_protocol_support_v1 import (
    ProtocolError,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    canonical_json_sha256,
    frozen_file_inventory,
    relative_path,
    resolve_path,
    sha256_file,
    strict_json_dumps,
    validate_frozen_manifest,
)


PROJECT_ID = "510300_CREATION_DATA_READINESS_LEDGER_V1"
BUNDLE_STATUS = "FROZEN_DATA_ONLY_NO_RETURN_CODE"
SHANGHAI = ZoneInfo("Asia/Shanghai")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

LEDGER_SCHEMA: dict[str, str] = {
    "trade_date": "string",
    "record_class": "string",
    "observed_at": "string",
    "authoritative_forward_record": "boolean",
    "actual_fund_shares": "float64",
    "actual_share_available_at": "string",
    "actual_share_available_at_verified": "boolean",
    "actual_share_raw_source_document_id": "string",
    "actual_share_raw_document_sha256": "string",
    "revision_flag": "boolean",
    "split_flag": "boolean",
    "iopv_snapshot_count": "Int64",
    "iopv_valid_snapshot_count": "Int64",
    "iopv_first_exchange_timestamp": "string",
    "iopv_last_exchange_timestamp": "string",
    "iopv_valid_fraction": "float64",
    "bid_ask_valid_snapshot_count": "Int64",
    "bid_ask_coverage": "float64",
    "iopv_raw_source_document_id": "string",
    "iopv_raw_document_sha256": "string",
    "source_status": "string",
    "trading_calendar_status": "string",
    "complete_close_coverage": "boolean",
    "strict_complete_day": "boolean",
    "failure_reasons_json": "string",
    "input_snapshot_sha256": "string",
    "return_view": "string",
    "position_impact": "float64",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ProtocolError(f"配置不是对象：{path}")
    if config.get("protocol", {}).get("project_id") != PROJECT_ID:
        raise ProtocolError("项目标识不匹配")
    return config


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in LEDGER_SCHEMA.items()})


def _artifact_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = config["artifacts"]
    return {
        "ledger": resolve_path(root, artifacts["readiness_ledger"]),
        "status_json": resolve_path(root, artifacts["status_json"]),
        "status_markdown": resolve_path(root, artifacts["status_markdown"]),
    }


def freeze_bundle(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(config_path.resolve())
    protocol = config["protocol"]
    if protocol.get("state") != "PRE_DATA_ONLY_FREEZE":
        raise ProtocolError("协议不是数据账冻结前状态")
    if protocol.get("research_mode") != "FORWARD_DATA_BUILD_ONLY":
        raise ProtocolError("协议不是只建设前向数据")
    if protocol.get("return_view") != "FORBIDDEN" or protocol.get("strategy_code") != "FORBIDDEN":
        raise ProtocolError("协议没有同时禁止收益读取与策略代码")
    if int(protocol.get("maturity_complete_days", 0)) != 120:
        raise ProtocolError("新数据成熟门没有冻结为120个严格完整日")
    if protocol.get("parent_strategy_protocol_change") != "FORBIDDEN":
        raise ProtocolError("协议没有禁止改写父V1")
    if protocol.get("backfill_allowed") is not False:
        raise ProtocolError("协议没有禁止回填")
    if any(bool(value) for value in config["boundaries"].values()):
        raise ProtocolError("数据账边界中存在被打开的策略或交易开关")
    if any("return" in column.lower() and column != "return_view" for column in LEDGER_SCHEMA):
        raise ProtocolError("数据成熟度账出现收益字段")

    manifest_path = resolve_path(root, config["artifacts"]["freeze_manifest"])
    if manifest_path.exists():
        raise ProtocolError(f"冻结清单已存在，禁止覆盖：{manifest_path}")
    paths = _artifact_paths(root, config)
    existing = [relative_path(root, path) for path in paths.values() if path.exists()]
    if existing:
        raise ProtocolError(f"冻结前已存在数据成熟度账结果：{existing}")
    entries: list[tuple[str | Path, str]] = [
        (path, "PROTOCOL_IMPLEMENTATION") for path in config["freeze"]["files"]
    ]
    entries.extend(
        (dependency["path"], dependency["role"])
        for dependency in config["immutable_dependencies"]
    )
    manifest: dict[str, Any] = {
        "schema_version": "510300_CREATION_DATA_READINESS_FREEZE_MANIFEST_V1",
        "project_id": PROJECT_ID,
        "protocol_version": protocol["version"],
        "bundle_status": BUNDLE_STATUS,
        "frozen_at_utc": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
        "return_code_present": False,
        "strategy_code_present": False,
        "parent_strategy_status": protocol["parent_strategy_status"],
        "parent_strategy_protocol_changed": False,
        "maturity_complete_days": 120,
        "backfill_allowed": False,
        "position_impact": 0,
        "live_trading_authorized": False,
        "frozen_files": frozen_file_inventory(root, entries),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(
        manifest,
        excluded_keys=("manifest_sha256",),
    )
    atomic_write_json(manifest_path, manifest)
    return {
        "status": BUNDLE_STATUS,
        "manifest_path": relative_path(root, manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "frozen_file_count": len(manifest["frozen_files"]),
        "maturity_complete_days": 120,
        "return_view": "FORBIDDEN",
        "position_impact": 0,
    }


def validate_bundle(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path.resolve())
    manifest = validate_frozen_manifest(
        root,
        resolve_path(root, config["artifacts"]["freeze_manifest"]),
        expected_manifest_sha256,
        project_id=PROJECT_ID,
        bundle_status=BUNDLE_STATUS,
    )
    return config, manifest


def _qualification(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[bool, bool, list[str]]:
    contract = config["strict_day_contract"]
    reasons: list[str] = []
    available_verified = row.get("actual_share_available_at_verified")
    if not isinstance(available_verified, (bool, np.bool_)) or not bool(available_verified):
        reasons.append("ACTUAL_SHARE_AVAILABLE_AT_NOT_VERIFIED")
    available_at = row.get("actual_share_available_at")
    if not available_at:
        reasons.append("ACTUAL_SHARE_AVAILABLE_AT_MISSING")
    else:
        try:
            available_timestamp = pd.Timestamp(available_at)
            if pd.isna(available_timestamp) or available_timestamp.tzinfo is None:
                reasons.append("ACTUAL_SHARE_AVAILABLE_AT_TIMEZONE_MISSING_OR_INVALID")
        except (TypeError, ValueError):
            reasons.append("ACTUAL_SHARE_AVAILABLE_AT_TIMEZONE_MISSING_OR_INVALID")
    actual_fund_shares = pd.to_numeric(row.get("actual_fund_shares"), errors="coerce")
    if not np.isfinite(actual_fund_shares) or float(actual_fund_shares) <= 0:
        reasons.append("ACTUAL_FUND_SHARES_MISSING_OR_NOT_POSITIVE")
    if not str(row.get("actual_share_raw_source_document_id") or "").strip():
        reasons.append("ACTUAL_SHARE_RAW_SOURCE_DOCUMENT_ID_MISSING")
    actual_share_hash = str(row.get("actual_share_raw_document_sha256") or "")
    if not SHA256_PATTERN.fullmatch(actual_share_hash):
        reasons.append("ACTUAL_SHARE_RAW_DOCUMENT_SHA256_MISSING_OR_INVALID")
    if not isinstance(row.get("revision_flag"), (bool, np.bool_)):
        reasons.append("REVISION_FLAG_MISSING")
    if not isinstance(row.get("split_flag"), (bool, np.bool_)):
        reasons.append("SPLIT_FLAG_MISSING")
    try:
        snapshot_count = int(row.get("iopv_snapshot_count") or 0)
        valid_count = int(row.get("iopv_valid_snapshot_count") or 0)
        bid_ask_count = int(row.get("bid_ask_valid_snapshot_count") or 0)
    except (TypeError, ValueError, OverflowError):
        snapshot_count = 0
        valid_count = 0
        bid_ask_count = 0
        reasons.append("IOPV_SNAPSHOT_COUNTS_INVALID")
    if snapshot_count <= 0:
        reasons.append("IOPV_SNAPSHOT_COUNT_NOT_POSITIVE")
    if valid_count < 0 or valid_count > snapshot_count:
        reasons.append("IOPV_VALID_SNAPSHOT_COUNT_OUT_OF_RANGE")
    if bid_ask_count < 0 or bid_ask_count > snapshot_count:
        reasons.append("BID_ASK_VALID_SNAPSHOT_COUNT_OUT_OF_RANGE")
    valid_fraction = valid_count / snapshot_count if snapshot_count > 0 else 0.0
    bid_ask_coverage = bid_ask_count / snapshot_count if snapshot_count > 0 else 0.0
    if valid_count < int(contract["iopv_minimum_valid_snapshots"]):
        reasons.append("IOPV_VALID_SNAPSHOT_COUNT_BELOW_120")
    if valid_fraction < float(contract["iopv_valid_fraction_minimum"]):
        reasons.append("IOPV_VALID_FRACTION_BELOW_95_PERCENT")
    if bid_ask_coverage < float(contract["bid_ask_coverage_minimum"]):
        reasons.append("BID_ASK_COVERAGE_BELOW_95_PERCENT")
    first_timestamp = row.get("iopv_first_exchange_timestamp")
    last_timestamp = row.get("iopv_last_exchange_timestamp")
    complete_close_coverage = False
    if not first_timestamp or not last_timestamp:
        reasons.append("IOPV_FIRST_OR_LAST_TIMESTAMP_MISSING")
    else:
        try:
            first = pd.Timestamp(first_timestamp)
            last = pd.Timestamp(last_timestamp)
            if pd.isna(first) or pd.isna(last) or first.tzinfo is None or last.tzinfo is None:
                raise ValueError("时间戳缺少时区或无效")
            first = first.tz_convert(SHANGHAI)
            last = last.tz_convert(SHANGHAI)
            first_clock = first.time()
            last_clock = last.time()
            first_limit = time.fromisoformat(contract["iopv_first_snapshot_no_later_than"])
            last_limit = time.fromisoformat(contract["iopv_last_snapshot_no_earlier_than"])
            if first > last:
                reasons.append("IOPV_TIMESTAMP_ORDER_INVALID")
            if first_clock > first_limit:
                reasons.append("IOPV_FIRST_SNAPSHOT_TOO_LATE")
            if last_clock < last_limit:
                reasons.append("IOPV_CLOSE_COVERAGE_FAILED")
            trade_date_value = row.get("trade_date")
            if trade_date_value:
                trade_date = pd.Timestamp(trade_date_value).date()
                if first.date() != trade_date or last.date() != trade_date:
                    reasons.append("IOPV_TIMESTAMPS_NOT_ON_TRADE_DATE")
            complete_close_coverage = (
                first <= last and first_clock <= first_limit and last_clock >= last_limit
            )
        except (TypeError, ValueError):
            reasons.append("IOPV_FIRST_OR_LAST_TIMESTAMP_INVALID_OR_TIMEZONE_MISSING")
    if not str(row.get("iopv_raw_source_document_id") or "").strip():
        reasons.append("IOPV_RAW_SOURCE_DOCUMENT_ID_MISSING")
    iopv_hash = str(row.get("iopv_raw_document_sha256") or "")
    if not SHA256_PATTERN.fullmatch(iopv_hash):
        reasons.append("IOPV_RAW_DOCUMENT_SHA256_MISSING_OR_INVALID")
    strict_complete = len(reasons) == 0
    return strict_complete, complete_close_coverage, reasons


def _diagnostic_rows(root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    share_path = resolve_path(
        root,
        config["admission_inputs"]["retrospective_actual_share_candidate"]["path"],
    )
    iopv_path = resolve_path(root, config["admission_inputs"]["legacy_iopv_snapshots"]["path"])
    shares = pd.read_parquet(share_path)
    iopv = pd.read_parquet(iopv_path)
    if "date" not in shares.columns or "fund_shares" not in shares.columns:
        raise ProtocolError("历史份额候选文件缺少date或fund_shares")
    required_iopv = {"trade_date", "exchange_timestamp", "last_price", "iopv"}
    if not required_iopv.issubset(iopv.columns):
        raise ProtocolError(f"旧IOPV文件缺少字段：{sorted(required_iopv - set(iopv.columns))}")
    shares = shares.copy()
    iopv = iopv.copy()
    shares["date"] = pd.to_datetime(shares["date"], errors="coerce").dt.normalize()
    shares["fund_shares"] = pd.to_numeric(shares["fund_shares"], errors="coerce")
    share_lookup = shares.drop_duplicates("date", keep="last").set_index("date")["fund_shares"].to_dict()
    iopv["trade_date"] = pd.to_datetime(iopv["trade_date"], errors="coerce").dt.normalize()
    iopv["exchange_timestamp"] = pd.to_datetime(iopv["exchange_timestamp"], errors="coerce")
    iopv["last_price"] = pd.to_numeric(iopv["last_price"], errors="coerce")
    iopv["iopv"] = pd.to_numeric(iopv["iopv"], errors="coerce")
    start, end = map(pd.Timestamp, config["protocol"]["pre_start_admission_period"])
    iopv = iopv[iopv["trade_date"].between(start, end)].copy()
    input_hash = canonical_json_sha256(
        {
            "retrospective_actual_share_candidate_sha256": sha256_file(share_path),
            "legacy_iopv_snapshots_sha256": sha256_file(iopv_path),
        }
    )
    rows: list[dict[str, Any]] = []
    for trade_date, group in iopv.groupby("trade_date", sort=True):
        valid = group["last_price"].gt(0) & group["iopv"].gt(0)
        if {"bid_price_1", "ask_price_1"}.issubset(group.columns):
            bid = pd.to_numeric(group["bid_price_1"], errors="coerce")
            ask = pd.to_numeric(group["ask_price_1"], errors="coerce")
            bid_ask_valid = bid.gt(0) & ask.gt(0) & ask.ge(bid)
        else:
            bid_ask_valid = pd.Series(False, index=group.index)
        timestamps = group["exchange_timestamp"].dropna().sort_values()
        candidate = {
            "trade_date": str(pd.Timestamp(trade_date).date()),
            "actual_fund_shares": share_lookup.get(pd.Timestamp(trade_date)),
            "actual_share_available_at_verified": False,
            "actual_share_available_at": None,
            "actual_share_raw_source_document_id": None,
            "actual_share_raw_document_sha256": None,
            "revision_flag": None,
            "split_flag": None,
            "iopv_snapshot_count": int(len(group)),
            "iopv_valid_snapshot_count": int(valid.sum()),
            "bid_ask_valid_snapshot_count": int(bid_ask_valid.sum()),
            "iopv_first_exchange_timestamp": timestamps.iloc[0].isoformat() if len(timestamps) else None,
            "iopv_last_exchange_timestamp": timestamps.iloc[-1].isoformat() if len(timestamps) else None,
            "iopv_raw_source_document_id": None,
            "iopv_raw_document_sha256": None,
        }
        strict, close_coverage, reasons = _qualification(candidate, config)
        row = {column: None for column in LEDGER_SCHEMA}
        row.update(
            {
                "trade_date": str(pd.Timestamp(trade_date).date()),
                "record_class": "PRE_START_DATA_FORMAT_ADMISSION_ONLY",
                "observed_at": datetime.now(tz=SHANGHAI).isoformat(),
                "authoritative_forward_record": False,
                "actual_fund_shares": candidate["actual_fund_shares"],
                **candidate,
                "iopv_valid_fraction": int(valid.sum()) / len(group) if len(group) else 0.0,
                "bid_ask_coverage": int(bid_ask_valid.sum()) / len(group) if len(group) else 0.0,
                "source_status": "LEGACY_INPUT_ADMISSION_DIAGNOSTIC",
                "trading_calendar_status": "NOT_COUNTED_PRE_START",
                "complete_close_coverage": close_coverage,
                "strict_complete_day": strict,
                "failure_reasons_json": strict_json_dumps(reasons, indent=None),
                "input_snapshot_sha256": input_hash,
                "return_view": "FORBIDDEN",
                "position_impact": 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=LEDGER_SCHEMA) if rows else _empty_ledger()


def _status_payload(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    authoritative = ledger[ledger["authoritative_forward_record"].fillna(False).astype(bool)] if len(ledger) else ledger
    strict = authoritative[authoritative["strict_complete_day"].fillna(False).astype(bool)] if len(authoritative) else authoritative
    maturity = int(config["protocol"]["maturity_complete_days"])
    complete_days = int(len(strict))
    data_contract_mature = complete_days >= maturity
    return {
        "schema_version": "510300_CREATION_DATA_READINESS_STATUS_V1",
        "project_id": PROJECT_ID,
        "updated_at": datetime.now(tz=SHANGHAI).isoformat(),
        "scientific_status": (
            "DATA_CONTRACT_MATURE_SEPARATE_PREFREEZE_RETURN_PROTOCOL_REQUIRED"
            if data_contract_mature
            else "NO_VIEW_DATA_CONTRACT_NOT_MATURE"
        ),
        "resource_allocation_status": (
            "DATA_MATURE_NO_RETURN_VIEW_AWAIT_SEPARATE_AUTHORIZATION"
            if data_contract_mature
            else "FORWARD_DATA_BUILD_ONLY_NO_RETURN_VIEW"
        ),
        "operational_status": "COLLECTING_DATA_ONLY" if len(authoritative) else "READY_FOR_FIRST_FORWARD_DATA_DAY",
        "parent_strategy_status": config["protocol"]["parent_strategy_status"],
        "parent_strategy_protocol_changed": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "pre_start_admission_rows": int((ledger["record_class"] == "PRE_START_DATA_FORMAT_ADMISSION_ONLY").sum()) if len(ledger) else 0,
        "authoritative_forward_rows": int(len(authoritative)),
        "strict_complete_days": complete_days,
        "maturity_complete_days_required": maturity,
        "strict_complete_days_remaining": max(maturity - complete_days, 0),
        "data_contract_mature": data_contract_mature,
        "first_strict_complete_day": strict["trade_date"].min() if len(strict) else None,
        "return_view": "FORBIDDEN",
        "strategy_code_present": False,
        "current_validated_high_sharpe_strategy": "NONE",
        "current_holding_route": "CASH_CNY",
        "position_impact": 0,
        "live_trading_authorized": False,
    }


def _status_markdown(status: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# 510300 一级市场数据成熟度账 V1：状态",
            "",
            f"- 科学状态：`{status['scientific_status']}`",
            f"- 资源状态：`{status['resource_allocation_status']}`",
            f"- 准入诊断行：{status['pre_start_admission_rows']}",
            f"- 权威前向行：{status['authoritative_forward_rows']}",
            f"- 严格完整日：{status['strict_complete_days']}/{status['maturity_complete_days_required']}",
            f"- 尚缺：{status['strict_complete_days_remaining']} 日",
            f"- 收益读取：`{status['return_view']}`",
            f"- 仓位影响：`{status['position_impact']}`",
            "",
            "达到120个严格完整日也只获得另行预冻结收益研究的资格，不自动产生因子、收益结论或仓位。",
            "",
        ]
    )


def _write_state(paths: Mapping[str, Path], config: Mapping[str, Any], manifest: Mapping[str, Any], ledger: pd.DataFrame) -> dict[str, Any]:
    status = _status_payload(config, manifest, ledger)
    atomic_write_parquet(paths["ledger"], ledger)
    atomic_write_json(paths["status_json"], status)
    atomic_write_text(paths["status_markdown"], _status_markdown(status))
    return status


def initialize_readiness_ledger(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    root = root.resolve()
    config, manifest = validate_bundle(root, config_path, expected_manifest_sha256)
    paths = _artifact_paths(root, config)
    existing = [relative_path(root, path) for path in paths.values() if path.exists()]
    if existing:
        raise ProtocolError(f"数据成熟度账已经存在，禁止重新初始化：{existing}")
    ledger = _diagnostic_rows(root, config)
    status = _write_state(paths, config, manifest, ledger)
    return {
        "status": "DATA_READINESS_LEDGER_INITIALIZED_NO_RETURN_VIEW",
        "pre_start_admission_rows": status["pre_start_admission_rows"],
        "authoritative_forward_rows": status["authoritative_forward_rows"],
        "strict_complete_days": status["strict_complete_days"],
        "maturity_complete_days_required": status["maturity_complete_days_required"],
        "return_view": "FORBIDDEN",
        "position_impact": 0,
    }


def _validate_hash(value: Any, field: str) -> str:
    text = str(value)
    if not SHA256_PATTERN.fullmatch(text):
        raise ProtocolError(f"{field}不是64位SHA-256")
    return text.lower()


def _parse_trade_date(value: Any, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field}无法解析") from exc
    if pd.isna(timestamp):
        raise ProtocolError(f"{field}为空或无效")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(SHANGHAI).tz_localize(None)
    return timestamp.normalize()


def append_forward_day(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
    payload_path: Path,
) -> dict[str, Any]:
    root = root.resolve()
    config, manifest = validate_bundle(root, config_path, expected_manifest_sha256)
    paths = _artifact_paths(root, config)
    if not paths["ledger"].is_file():
        raise ProtocolError("数据成熟度账尚未初始化")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("每日数据回执不是JSON对象")
    required = set(config["required_forward_receipt_fields"])
    missing = sorted(required - set(payload))
    if missing:
        raise ProtocolError(f"每日数据回执缺少字段：{missing}")
    trade_date = _parse_trade_date(payload["trade_date"], "trade_date")
    today = pd.Timestamp(datetime.now(tz=SHANGHAI).date())
    if trade_date != today:
        raise ProtocolError("只允许记录当天数据，禁止以后回填")
    if trade_date < pd.Timestamp("2026-09-01"):
        raise ProtocolError("权威前向数据日不得早于2026-09-01")
    calendar = pd.read_csv(root / "data/reference/sse_trade_calendar_2026.csv")
    trade_dates = set(pd.to_datetime(calendar["trade_date"], errors="coerce").dropna().dt.normalize())
    if trade_date not in trade_dates:
        raise ProtocolError("输入日期不是冻结日历中的上交所交易日")
    for field, expected in config["required_status_values"].items():
        if payload[field] != expected:
            raise ProtocolError(f"{field}={payload[field]}，要求={expected}")
    ledger = pd.read_parquet(paths["ledger"])
    if str(trade_date.date()) in set(ledger["trade_date"].astype(str)):
        raise ProtocolError("该交易日已经存在，禁止重复或覆盖")
    for field in ("actual_share_raw_document_sha256", "iopv_raw_document_sha256"):
        payload[field] = _validate_hash(payload[field], field)
    for field in ("actual_share_raw_source_document_id", "iopv_raw_source_document_id"):
        if not str(payload[field]).strip():
            raise ProtocolError(f"{field}为空")
    for field in ("actual_share_available_at_verified", "revision_flag", "split_flag"):
        if not isinstance(payload[field], bool):
            raise ProtocolError(f"{field}必须是JSON布尔值")
    for field in ("observed_at", "actual_share_available_at", "iopv_first_exchange_timestamp", "iopv_last_exchange_timestamp"):
        try:
            timestamp = pd.Timestamp(payload[field])
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"{field}无法解析为时间戳") from exc
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ProtocolError(f"{field}必须包含时区")
        payload[field] = timestamp.tz_convert(SHANGHAI).isoformat()
    row = {column: None for column in LEDGER_SCHEMA}
    row.update(
        {
            "trade_date": str(trade_date.date()),
            "record_class": "AUTHORITATIVE_FORWARD_DATA_READINESS",
            "observed_at": payload["observed_at"],
            "authoritative_forward_record": True,
            "actual_fund_shares": float(payload["actual_fund_shares"]),
            "actual_share_available_at": payload["actual_share_available_at"],
            "actual_share_available_at_verified": payload["actual_share_available_at_verified"],
            "actual_share_raw_source_document_id": str(payload["actual_share_raw_source_document_id"]),
            "actual_share_raw_document_sha256": payload["actual_share_raw_document_sha256"],
            "revision_flag": payload["revision_flag"],
            "split_flag": payload["split_flag"],
            "iopv_snapshot_count": int(payload["iopv_snapshot_count"]),
            "iopv_valid_snapshot_count": int(payload["iopv_valid_snapshot_count"]),
            "iopv_first_exchange_timestamp": payload["iopv_first_exchange_timestamp"],
            "iopv_last_exchange_timestamp": payload["iopv_last_exchange_timestamp"],
            "bid_ask_valid_snapshot_count": int(payload["bid_ask_valid_snapshot_count"]),
            "iopv_raw_source_document_id": str(payload["iopv_raw_source_document_id"]),
            "iopv_raw_document_sha256": payload["iopv_raw_document_sha256"],
            "source_status": str(payload["source_status"]),
            "trading_calendar_status": str(payload["trading_calendar_status"]),
            "input_snapshot_sha256": sha256_file(payload_path),
            "return_view": "FORBIDDEN",
            "position_impact": 0.0,
        }
    )
    count = row["iopv_snapshot_count"]
    row["iopv_valid_fraction"] = row["iopv_valid_snapshot_count"] / count if count else 0.0
    row["bid_ask_coverage"] = row["bid_ask_valid_snapshot_count"] / count if count else 0.0
    strict, close_coverage, reasons = _qualification(row, config)
    row["complete_close_coverage"] = close_coverage
    row["strict_complete_day"] = strict
    row["failure_reasons_json"] = strict_json_dumps(reasons, indent=None)
    ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    status = _write_state(paths, config, manifest, ledger)
    return {
        "status": "STRICT_COMPLETE_DAY" if strict else "FORWARD_DATA_DAY_INCOMPLETE",
        "trade_date": str(trade_date.date()),
        "strict_complete_day": strict,
        "failure_reasons": reasons,
        "strict_complete_days": status["strict_complete_days"],
        "maturity_complete_days_required": status["maturity_complete_days_required"],
        "return_view": "FORBIDDEN",
        "position_impact": 0,
    }
