from __future__ import annotations

import json
import math
import re
from datetime import datetime, time
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from research.frozen_protocol_support_v1 import (
    ProtocolError,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    canonical_json_sha256,
    frozen_file_inventory,
    relative_path,
    resolve_path,
    validate_frozen_manifest,
)


PROJECT_ID = "510300_POST_CLOSE_OFFSHORE_INFORMATION_TRANSFER_V1"
BUNDLE_STATUS = "FROZEN_FORWARD_ONLY_ZERO_POSITION"
SHANGHAI = ZoneInfo("Asia/Shanghai")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


OBSERVATION_SCHEMA: dict[str, str] = {
    "trade_date": "string",
    "observed_at": "string",
    "authoritative_forward_record": "boolean",
    "a50_contract": "string",
    "a50_start_price": "float64",
    "a50_end_price": "float64",
    "a50_start_exchange_timestamp": "string",
    "a50_end_exchange_timestamp": "string",
    "a50_start_received_at": "string",
    "a50_end_received_at": "string",
    "a50_raw_source_document_id": "string",
    "a50_raw_source_sha256": "string",
    "a50_source_url": "string",
    "sse_rule_status": "string",
    "sgx_session_status": "string",
    "trading_calendar_status": "string",
    "instrument_status": "string",
    "etf_close_price": "float64",
    "etf_close_source_document_id": "string",
    "etf_close_source_sha256": "string",
    "r_post": "float64",
    "beta_sample_n_before_signal": "Int64",
    "beta_hat_before_signal": "float64",
    "beta_standard_error_before_signal": "float64",
    "beta_lcb_before_signal": "float64",
    "beta_hac_lag_before_signal": "Int64",
    "gross_edge_lcb": "float64",
    "signal_qualified": "boolean",
    "target_mature": "boolean",
    "exit_trade_date": "string",
    "matured_at": "string",
    "etf_next_0935_price": "float64",
    "etf_next_0935_exchange_timestamp": "string",
    "etf_next_0935_received_at": "string",
    "etf_next_0935_source_document_id": "string",
    "etf_next_0935_source_sha256": "string",
    "target_return_close_to_next_0935": "float64",
    "historical_evidence_label": "string",
    "position_impact": "float64",
}

SIGNAL_SCHEMA: dict[str, str] = {
    "trade_date": "string",
    "signal_timestamp": "string",
    "a50_contract": "string",
    "r_post": "float64",
    "beta_sample_n": "Int64",
    "beta_lcb": "float64",
    "gross_edge_lcb": "float64",
    "entry_gate_bps": "float64",
    "theoretical_entry_price": "float64",
    "theoretical_entry_method": "string",
    "exit_trade_date": "string",
    "signal_return": "float64",
    "signal_return_status": "string",
    "actual_fill_required_for_strategy_return": "boolean",
    "historical_evidence_label": "string",
    "position_impact": "float64",
}

ORDER_INTENT_SCHEMA: dict[str, str] = {
    "trade_date": "string",
    "intent_time": "string",
    "instrument": "string",
    "side": "string",
    "price_method": "string",
    "reference_price": "float64",
    "authorized_quantity_shares": "Int64",
    "submitted_to_broker": "boolean",
    "broker_order_id": "string",
    "future_live_cancel_time": "string",
    "intent_status": "string",
    "separate_live_authorization_present": "boolean",
    "position_impact": "float64",
}

ACTUAL_FILL_SCHEMA: dict[str, str] = {
    "trade_date": "string",
    "broker_order_id": "string",
    "fill_timestamp": "string",
    "filled_shares": "Int64",
    "submitted_shares": "Int64",
    "fill_price": "float64",
    "fill_ratio": "float64",
    "exit_trade_date": "string",
    "exit_fill_timestamp": "string",
    "exit_fill_price": "float64",
    "actual_fill_return": "float64",
    "actual_cost": "float64",
    "net_edge": "float64",
    "capacity_adjusted_account_return": "float64",
    "authorization_receipt_sha256": "string",
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


def _empty_frame(schema: Mapping[str, str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in schema.items()})


def _artifact_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    ledgers = config["ledgers"]
    return {
        "observation": resolve_path(root, ledgers["transmission_observation_ledger"]),
        "signal": resolve_path(root, ledgers["signal_ledger"]),
        "order_intent": resolve_path(root, ledgers["order_intent_ledger"]),
        "actual_fill": resolve_path(root, ledgers["actual_fill_ledger"]),
        "status_json": resolve_path(root, ledgers["status_json"]),
        "status_markdown": resolve_path(root, ledgers["status_markdown"]),
    }


def freeze_bundle(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(config_path.resolve())
    protocol = config["protocol"]
    if protocol.get("state") != "PRE_FORWARD_ONLY_FREEZE":
        raise ProtocolError("协议不是前向冻结前状态")
    if protocol.get("research_mode") != "FORWARD_ONLY_DISCOVERY":
        raise ProtocolError("研究模式不是纯前向发现")
    if protocol.get("historical_tradable_backtest") != "NOT_ALLOWED":
        raise ProtocolError("协议没有禁止历史可交易回测")
    if protocol.get("pre_start_strategy_return_reading") != "FORBIDDEN":
        raise ProtocolError("协议没有禁止准入期收益读取")
    if protocol.get("backfill_allowed") is not False:
        raise ProtocolError("协议没有禁止回填")
    if any(bool(value) for value in config["boundaries"].values()):
        raise ProtocolError("交易隔离开关中存在被打开的项目")
    if int(config["ledgers"]["order_intent_authorized_quantity_shares"]) != 0:
        raise ProtocolError("当前订单意向授权数量必须严格为0")
    if config["ledgers"]["actual_fill_recording_before_separate_live_authorization"] is not False:
        raise ProtocolError("当前版本不得记录实盘成交")
    clock = config["signal_clock"]
    expected_clocks = {
        "observation_start": "15:00:00",
        "observation_end": "15:04:30",
        "order_intent_time": "15:05:00",
        "future_live_cancel_time": "15:29:50",
        "exit_time": "NEXT_SSE_TRADING_DAY_09_35",
    }
    for field, expected in expected_clocks.items():
        if clock.get(field) != expected:
            raise ProtocolError(f"冻结时钟{field}={clock.get(field)}，要求={expected}")
    costs = config["cost_and_entry_gate"]
    stress_cost = float(costs["stress_round_trip_bps"])
    if not math.isclose(
        float(costs["entry_gross_edge_lcb_bps"]),
        float(costs["entry_multiple_of_stress_cost"]) * stress_cost,
    ):
        raise ProtocolError("42bp进入门不是冻结的3倍压力成本")
    if not math.isclose(float(costs["decay_gross_edge_bps"]), 2.0 * stress_cost):
        raise ProtocolError("28bp衰退门不是冻结的2倍压力成本")
    estimator = config["beta_estimator"]
    if int(estimator.get("minimum_mature_observations", 0)) != 60:
        raise ProtocolError("beta最少成熟观测没有冻结为60日")
    if estimator.get("assumed_beta_forbidden") is not True:
        raise ProtocolError("协议没有禁止假定beta")
    if estimator.get("beta_lcb_must_be_strictly_positive") is not True:
        raise ProtocolError("协议没有要求beta保守下界严格为正")

    source_receipt_path = resolve_path(root, config["artifacts"]["source_receipt"])
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    if source_receipt["sse_rule"]["status"] != "OFFICIAL_RULE_VERIFIED_CURRENT_AND_NOT_DELAYED":
        raise ProtocolError("上交所盘后固定价格规则尚未通过官方来源核验")
    if source_receipt["sse_rule"]["effective_date"] != "2026-07-06":
        raise ProtocolError("上交所规则生效日与协议不一致")

    manifest_path = resolve_path(root, config["artifacts"]["freeze_manifest"])
    if manifest_path.exists():
        raise ProtocolError(f"冻结清单已存在，禁止覆盖：{manifest_path}")
    existing = [relative_path(root, path) for path in _artifact_paths(root, config).values() if path.exists()]
    if existing:
        raise ProtocolError(f"冻结前已存在本分支前向账或状态：{existing}")

    entries: list[tuple[str | Path, str]] = [
        (path, "PROTOCOL_IMPLEMENTATION") for path in config["freeze"]["files"]
    ]
    entries.extend(
        (dependency["path"], dependency["role"])
        for dependency in config["immutable_dependencies"]
    )
    manifest: dict[str, Any] = {
        "schema_version": "510300_POST_CLOSE_OFFSHORE_FREEZE_MANIFEST_V1",
        "project_id": PROJECT_ID,
        "protocol_version": protocol["version"],
        "bundle_status": BUNDLE_STATUS,
        "frozen_at_utc": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
        "historical_strategy_returns_observed_before_freeze": False,
        "historical_tradable_backtest_allowed": False,
        "authoritative_forward_start_rule": protocol["forward_start_rule"],
        "backfill_allowed": False,
        "assumed_beta_allowed": False,
        "assumed_fills_allowed": False,
        "authorized_order_quantity_shares": 0,
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
        "historical_strategy_returns_observed_before_freeze": False,
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


def _hac_beta_estimate(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ y
    residuals = y - design @ coefficients
    meat = np.zeros((2, 2), dtype=float)
    for index in range(len(x)):
        vector = design[index][:, None]
        meat += residuals[index] ** 2 * (vector @ vector.T)
    for current_lag in range(1, min(lag, len(x) - 1) + 1):
        weight = 1.0 - current_lag / (lag + 1.0)
        cross = np.zeros((2, 2), dtype=float)
        for index in range(current_lag, len(x)):
            left = design[index][:, None]
            right = design[index - current_lag][:, None]
            cross += residuals[index] * residuals[index - current_lag] * (left @ right.T)
        meat += weight * (cross + cross.T)
    covariance = xtx_inverse @ meat @ xtx_inverse
    if len(x) > 2:
        covariance *= len(x) / (len(x) - 2)
    beta = float(coefficients[1])
    variance = float(covariance[1, 1])
    standard_error = math.sqrt(max(variance, 0.0))
    return beta, standard_error


def estimate_beta_lcb(
    observations: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    estimator = config["beta_estimator"]
    if observations.empty:
        mature = observations.copy()
    else:
        mature = observations[
            observations["authoritative_forward_record"].fillna(False).astype(bool)
            & observations["target_mature"].fillna(False).astype(bool)
        ].copy()
    x = pd.to_numeric(mature.get("r_post", pd.Series(dtype=float)), errors="coerce")
    y = pd.to_numeric(
        mature.get("target_return_close_to_next_0935", pd.Series(dtype=float)),
        errors="coerce",
    )
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_values = x[valid].to_numpy(dtype=float)
    y_values = y[valid].to_numpy(dtype=float)
    sample_size = int(len(x_values))
    minimum = int(estimator["minimum_mature_observations"])
    base = {
        "sample_n": sample_size,
        "minimum_sample_n": minimum,
        "beta_hat": None,
        "beta_standard_error": None,
        "beta_lcb": None,
        "hac_lag": None,
        "status": "NO_VIEW_INSUFFICIENT_MATURE_OBSERVATIONS",
    }
    if sample_size < minimum:
        return base
    if float(np.std(x_values, ddof=1)) <= 1e-10:
        base["status"] = "NO_VIEW_REGRESSOR_VARIATION_ZERO"
        return base
    lag = int(math.floor(4.0 * (sample_size / 100.0) ** (2.0 / 9.0)))
    beta, standard_error = _hac_beta_estimate(x_values, y_values, lag)
    confidence = float(estimator["one_sided_confidence_level"])
    critical = float(stats.t.ppf(confidence, sample_size - 2))
    beta_lcb = beta - critical * standard_error
    base.update(
        {
            "beta_hat": beta,
            "beta_standard_error": standard_error,
            "beta_lcb": beta_lcb,
            "hac_lag": lag,
            "status": "BETA_LCB_POSITIVE" if beta_lcb > 0 else "NO_VIEW_BETA_LCB_NOT_POSITIVE",
        }
    )
    return base


def evaluate_entry_gate(r_post: float, beta_lcb: float | None, config: Mapping[str, Any]) -> dict[str, Any]:
    threshold = float(config["cost_and_entry_gate"]["entry_gross_edge_lcb_bps"]) / 10000.0
    if not np.isfinite(r_post):
        return {
            "gross_edge_lcb": None,
            "threshold": threshold,
            "signal_qualified": False,
            "status": "NO_VIEW_INVALID_R_POST",
        }
    if beta_lcb is None or not np.isfinite(beta_lcb) or beta_lcb <= 0:
        return {
            "gross_edge_lcb": None,
            "threshold": threshold,
            "signal_qualified": False,
            "status": "NO_VIEW_BETA_LCB_NOT_POSITIVE",
        }
    gross_edge = float(beta_lcb * r_post)
    qualified = bool(r_post > 0 and gross_edge >= threshold)
    return {
        "gross_edge_lcb": gross_edge,
        "threshold": threshold,
        "signal_qualified": qualified,
        "status": "SIGNAL_QUALIFIED_ZERO_POSITION_INTENT" if qualified else "NO_SIGNAL_EDGE_BELOW_FROZEN_GATE",
    }


def _status_markdown(status: Mapping[str, Any]) -> str:
    beta = status["beta_state"]
    counts = status["ledger_counts"]
    return "\n".join(
        [
            "# 510300 盘后离岸信息传导 V1：前向状态",
            "",
            f"- 科学状态：`{status['scientific_status']}`",
            f"- 资源状态：`{status['resource_allocation_status']}`",
            f"- 运行状态：`{status['operational_status']}`",
            f"- 权威前向起点：`{status['authoritative_forward_start']}`",
            f"- 回填：`{status['backfill_allowed']}`",
            f"- beta 成熟样本：{beta['sample_n']}/{beta['minimum_sample_n']}",
            f"- beta LCB：{beta['beta_lcb']}",
            f"- 观测/信号/意向/实际成交：{counts['observations']}/{counts['signals']}/{counts['order_intents']}/{counts['actual_fills']}",
            f"- 当前持仓：`{status['current_holding_route']}`",
            f"- 仓位影响：`{status['position_impact']}`",
            "",
            "`ACTUAL_FILL_RETURN`、`FILL_PROBABILITY` 与 `CAPACITY_ADJUSTED_RETURN` 在单独实盘授权和真实成交出现前均为 `NO_VIEW_NO_LIVE_AUTHORIZATION`。",
            "",
        ]
    )


def _build_status(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    observations: pd.DataFrame,
    signals: pd.DataFrame,
    intents: pd.DataFrame,
    fills: pd.DataFrame,
) -> dict[str, Any]:
    beta = estimate_beta_lcb(observations, config)
    authoritative_dates = (
        observations.loc[
            observations["authoritative_forward_record"].fillna(False).astype(bool), "trade_date"
        ].dropna().astype(str).tolist()
        if not observations.empty
        else []
    )
    if beta["status"] == "BETA_LCB_POSITIVE":
        scientific_status = "FORWARD_BETA_LCB_POSITIVE_SIGNAL_GATE_ELIGIBLE"
        no_new_entry = False
    else:
        scientific_status = beta["status"]
        no_new_entry = True
    return {
        "schema_version": "510300_POST_CLOSE_OFFSHORE_FORWARD_STATUS_V1",
        "project_id": PROJECT_ID,
        "updated_at": datetime.now(tz=SHANGHAI).isoformat(),
        "scientific_status": scientific_status,
        "resource_allocation_status": "NEXT_PRIORITY_FORWARD_ONLY_DISCOVERY",
        "operational_status": (
            "COLLECTING_FORWARD_ZERO_POSITION"
            if len(observations)
            else "READY_FOR_FIRST_COMPLIANT_FORWARD_DAY"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "authoritative_forward_start_rule": config["protocol"]["forward_start_rule"],
        "authoritative_forward_start": min(authoritative_dates) if authoritative_dates else None,
        "backfill_allowed": False,
        "beta_state": beta,
        "ledger_counts": {
            "observations": int(len(observations)),
            "mature_observations": int(observations["target_mature"].fillna(False).sum()) if len(observations) else 0,
            "signals": int(len(signals)),
            "order_intents": int(len(intents)),
            "actual_fills": int(len(fills)),
        },
        "reporting_state": {
            "SIGNAL_RETURN": "PENDING_OR_OBSERVATIONAL_ONLY",
            "ACTUAL_FILL_RETURN": "NO_VIEW_NO_LIVE_AUTHORIZATION",
            "FILL_PROBABILITY": "NO_VIEW_NO_LIVE_AUTHORIZATION",
            "CAPACITY_ADJUSTED_RETURN": "NO_VIEW_NO_LIVE_AUTHORIZATION",
        },
        "stop_gate_state": {
            "configured_rules": list(config["stop_gates"]["rules"]),
            "beta_gate": (
                "PASS"
                if beta["status"] == "BETA_LCB_POSITIVE"
                else "NO_NEW_ENTRY_BETA_LCB_NOT_PROVEN_POSITIVE"
            ),
            "actual_fill_dependent_gates": "NOT_EVALUABLE_NO_LIVE_AUTHORIZATION",
            "gross_edge_decay_gate": "EVALUATED_PER_VALID_SIGNAL_BEFORE_ZERO_POSITION_INTENT",
            "rule_or_data_contract_gate": "EVALUATED_ON_EACH_DAILY_RECEIPT",
            "stop_action": config["stop_gates"]["stop_action"],
        },
        "no_new_entry": no_new_entry,
        "order_intent_authorized_quantity_shares": 0,
        "actual_fill_assumption_allowed": False,
        "current_validated_high_sharpe_strategy": "NONE",
        "current_holding_route": "CASH_CNY",
        "position_impact": 0,
        "live_trading_authorized": False,
    }


def _write_status(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    observations: pd.DataFrame,
    signals: pd.DataFrame,
    intents: pd.DataFrame,
    fills: pd.DataFrame,
) -> dict[str, Any]:
    status = _build_status(config, manifest, observations, signals, intents, fills)
    atomic_write_json(paths["status_json"], status)
    atomic_write_text(paths["status_markdown"], _status_markdown(status))
    return status


def initialize_forward_ledgers(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    root = root.resolve()
    config, manifest = validate_bundle(root, config_path, expected_manifest_sha256)
    paths = _artifact_paths(root, config)
    existing = [relative_path(root, path) for path in paths.values() if path.exists()]
    if existing:
        raise ProtocolError(f"前向账或状态已经存在，禁止重新初始化：{existing}")
    observations = _empty_frame(OBSERVATION_SCHEMA)
    signals = _empty_frame(SIGNAL_SCHEMA)
    intents = _empty_frame(ORDER_INTENT_SCHEMA)
    fills = _empty_frame(ACTUAL_FILL_SCHEMA)
    atomic_write_parquet(paths["observation"], observations)
    atomic_write_parquet(paths["signal"], signals)
    atomic_write_parquet(paths["order_intent"], intents)
    atomic_write_parquet(paths["actual_fill"], fills)
    status = _write_status(paths, config, manifest, observations, signals, intents, fills)
    return {
        "status": "FORWARD_LEDGERS_INITIALIZED_EMPTY_ZERO_POSITION",
        "scientific_status": status["scientific_status"],
        "authoritative_forward_start": status["authoritative_forward_start"],
        "ledger_paths": {key: relative_path(root, path) for key, path in paths.items()},
        "position_impact": 0,
    }


def _load_ledgers(paths: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = ("observation", "signal", "order_intent", "actual_fill")
    missing = [str(paths[key]) for key in required if not paths[key].is_file()]
    if missing:
        raise ProtocolError(f"前向账尚未初始化：{missing}")
    return tuple(pd.read_parquet(paths[key]) for key in required)  # type: ignore[return-value]


def _parse_timestamp(value: Any, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field}无法解析为时间戳") from exc
    if pd.isna(timestamp):
        raise ProtocolError(f"{field}为空或不是有效时间戳")
    if timestamp.tzinfo is None:
        raise ProtocolError(f"{field}必须包含时区")
    return timestamp.tz_convert(SHANGHAI)


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


def _validate_hash(value: Any, field: str) -> str:
    text = str(value)
    if not SHA256_PATTERN.fullmatch(text):
        raise ProtocolError(f"{field}不是64位SHA-256")
    return text.lower()


def _target_timestamp(trade_date: pd.Timestamp, clock: str) -> pd.Timestamp:
    parsed = time.fromisoformat(clock)
    return pd.Timestamp(
        datetime.combine(trade_date.date(), parsed, tzinfo=SHANGHAI)
    )


def _validate_target_trade(
    exchange_timestamp: pd.Timestamp,
    received_at: pd.Timestamp,
    target: pd.Timestamp,
    maximum_staleness_seconds: float,
    maximum_latency_seconds: float,
    field_prefix: str,
) -> None:
    staleness = (target - exchange_timestamp).total_seconds()
    latency = (received_at - exchange_timestamp).total_seconds()
    if staleness < 0 or staleness > maximum_staleness_seconds:
        raise ProtocolError(f"{field_prefix}交易时间戳不满足目标前最多{maximum_staleness_seconds}秒")
    if latency < 0 or latency > maximum_latency_seconds:
        raise ProtocolError(f"{field_prefix}到达延迟不满足最多{maximum_latency_seconds}秒")


def _read_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("输入回执不是JSON对象")
    return payload


def record_signal_window(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
    payload_path: Path,
) -> dict[str, Any]:
    root = root.resolve()
    config, manifest = validate_bundle(root, config_path, expected_manifest_sha256)
    paths = _artifact_paths(root, config)
    observations, signals, intents, fills = _load_ledgers(paths)
    payload = _read_payload(payload_path)
    required = set(config["data_contract"]["required_window_fields"])
    missing = sorted(required - set(payload))
    if missing:
        raise ProtocolError(f"信号窗口回执缺少字段：{missing}")

    trade_date = _parse_trade_date(payload["trade_date"], "trade_date")
    today = pd.Timestamp(datetime.now(tz=SHANGHAI).date())
    start_date = pd.Timestamp("2026-09-01")
    if trade_date != today:
        raise ProtocolError("只允许记录当天窗口，禁止盘后补跑或回填")
    if trade_date < start_date:
        raise ProtocolError("2026-09-01前只允许离线准入检查，不得写入权威前向账")
    calendar = pd.read_csv(root / "data/reference/sse_trade_calendar_2026.csv")
    valid_dates = set(pd.to_datetime(calendar["trade_date"], errors="coerce").dropna().dt.normalize())
    if trade_date not in valid_dates:
        raise ProtocolError("输入日期不是冻结日历中的上交所交易日")
    if not observations.empty and str(trade_date.date()) in set(observations["trade_date"].astype(str)):
        raise ProtocolError("该交易日已经记录，禁止重复追加")

    for field, expected in config["data_contract"]["required_status_values"].items():
        if payload[field] != expected:
            raise ProtocolError(f"{field}={payload[field]}，要求={expected}，本日NO_VIEW")
    start_price = float(payload["a50_start_price"])
    end_price = float(payload["a50_end_price"])
    etf_close = float(payload["etf_close_price"])
    if not all(np.isfinite(value) and value > 0 for value in (start_price, end_price, etf_close)):
        raise ProtocolError("A50或510300价格不是正数")
    if not str(payload["a50_contract"]).strip():
        raise ProtocolError("a50_contract为空")
    start_exchange = _parse_timestamp(payload["a50_start_exchange_timestamp"], "a50_start_exchange_timestamp")
    end_exchange = _parse_timestamp(payload["a50_end_exchange_timestamp"], "a50_end_exchange_timestamp")
    start_received = _parse_timestamp(payload["a50_start_received_at"], "a50_start_received_at")
    end_received = _parse_timestamp(payload["a50_end_received_at"], "a50_end_received_at")
    observed_at = _parse_timestamp(payload["observed_at"], "observed_at")
    clock = config["signal_clock"]
    start_target = _target_timestamp(trade_date, clock["observation_start"])
    end_target = _target_timestamp(trade_date, clock["observation_end"])
    _validate_target_trade(
        start_exchange,
        start_received,
        start_target,
        float(clock["maximum_exchange_timestamp_staleness_seconds"]),
        float(clock["maximum_arrival_latency_seconds"]),
        "A50起点",
    )
    _validate_target_trade(
        end_exchange,
        end_received,
        end_target,
        float(clock["maximum_exchange_timestamp_staleness_seconds"]),
        float(clock["maximum_arrival_latency_seconds"]),
        "A50终点",
    )
    deadline = _target_timestamp(trade_date, clock["signal_must_be_available_by"])
    if end_received > deadline or observed_at > deadline or observed_at < end_received:
        raise ProtocolError("信号没有在冻结截止时点前完整到达")
    for field in ("a50_raw_source_sha256", "etf_close_source_sha256"):
        payload[field] = _validate_hash(payload[field], field)
    for field in ("a50_raw_source_document_id", "etf_close_source_document_id", "a50_source_url"):
        if not str(payload[field]).strip():
            raise ProtocolError(f"{field}为空")

    beta = estimate_beta_lcb(observations, config)
    r_post = end_price / start_price - 1.0
    gate = evaluate_entry_gate(r_post, beta["beta_lcb"], config)
    row = {column: None for column in OBSERVATION_SCHEMA}
    row.update(
        {
            "trade_date": str(trade_date.date()),
            "observed_at": observed_at.isoformat(),
            "authoritative_forward_record": True,
            "a50_contract": str(payload["a50_contract"]),
            "a50_start_price": start_price,
            "a50_end_price": end_price,
            "a50_start_exchange_timestamp": start_exchange.isoformat(),
            "a50_end_exchange_timestamp": end_exchange.isoformat(),
            "a50_start_received_at": start_received.isoformat(),
            "a50_end_received_at": end_received.isoformat(),
            "a50_raw_source_document_id": str(payload["a50_raw_source_document_id"]),
            "a50_raw_source_sha256": payload["a50_raw_source_sha256"],
            "a50_source_url": str(payload["a50_source_url"]),
            "sse_rule_status": str(payload["sse_rule_status"]),
            "sgx_session_status": str(payload["sgx_session_status"]),
            "trading_calendar_status": str(payload["trading_calendar_status"]),
            "instrument_status": str(payload["instrument_status"]),
            "etf_close_price": etf_close,
            "etf_close_source_document_id": str(payload["etf_close_source_document_id"]),
            "etf_close_source_sha256": payload["etf_close_source_sha256"],
            "r_post": r_post,
            "beta_sample_n_before_signal": beta["sample_n"],
            "beta_hat_before_signal": beta["beta_hat"],
            "beta_standard_error_before_signal": beta["beta_standard_error"],
            "beta_lcb_before_signal": beta["beta_lcb"],
            "beta_hac_lag_before_signal": beta["hac_lag"],
            "gross_edge_lcb": gate["gross_edge_lcb"],
            "signal_qualified": gate["signal_qualified"],
            "target_mature": False,
            "historical_evidence_label": "FORWARD_ONLY_DISCOVERY",
            "position_impact": 0.0,
        }
    )
    observations = pd.concat([observations, pd.DataFrame([row])], ignore_index=True)
    atomic_write_parquet(paths["observation"], observations)

    if gate["signal_qualified"]:
        signal_row = {column: None for column in SIGNAL_SCHEMA}
        signal_row.update(
            {
                "trade_date": str(trade_date.date()),
                "signal_timestamp": observed_at.isoformat(),
                "a50_contract": str(payload["a50_contract"]),
                "r_post": r_post,
                "beta_sample_n": beta["sample_n"],
                "beta_lcb": beta["beta_lcb"],
                "gross_edge_lcb": gate["gross_edge_lcb"],
                "entry_gate_bps": float(config["cost_and_entry_gate"]["entry_gross_edge_lcb_bps"]),
                "theoretical_entry_price": etf_close,
                "theoretical_entry_method": "SAME_DAY_OFFICIAL_CLOSE_ACTUAL_FILL_NOT_ASSUMED",
                "signal_return_status": "PENDING_NEXT_TRADING_DAY_09_35",
                "actual_fill_required_for_strategy_return": True,
                "historical_evidence_label": "FORWARD_ONLY_DISCOVERY",
                "position_impact": 0.0,
            }
        )
        signals = pd.concat([signals, pd.DataFrame([signal_row])], ignore_index=True)
        intent_row = {column: None for column in ORDER_INTENT_SCHEMA}
        intent_row.update(
            {
                "trade_date": str(trade_date.date()),
                "intent_time": _target_timestamp(trade_date, clock["order_intent_time"]).isoformat(),
                "instrument": "510300.SH",
                "side": "BUY",
                "price_method": "SSE_AFTER_HOURS_SAME_DAY_CLOSE",
                "reference_price": etf_close,
                "authorized_quantity_shares": 0,
                "submitted_to_broker": False,
                "future_live_cancel_time": clock["future_live_cancel_time"],
                "intent_status": "ZERO_POSITION_RESEARCH_INTENT_NOT_AN_ORDER",
                "separate_live_authorization_present": False,
                "position_impact": 0.0,
            }
        )
        intents = pd.concat([intents, pd.DataFrame([intent_row])], ignore_index=True)
        atomic_write_parquet(paths["signal"], signals)
        atomic_write_parquet(paths["order_intent"], intents)
    status = _write_status(paths, config, manifest, observations, signals, intents, fills)
    return {
        "status": gate["status"],
        "trade_date": str(trade_date.date()),
        "r_post": r_post,
        "beta_lcb": beta["beta_lcb"],
        "gross_edge_lcb": gate["gross_edge_lcb"],
        "signal_qualified": gate["signal_qualified"],
        "authorized_quantity_shares": 0,
        "submitted_to_broker": False,
        "scientific_status": status["scientific_status"],
    }


def _next_sse_trade_date(root: Path, signal_date: pd.Timestamp) -> pd.Timestamp:
    calendar = pd.read_csv(root / "data/reference/sse_trade_calendar_2026.csv")
    dates = pd.to_datetime(calendar["trade_date"], errors="coerce").dropna().dt.normalize().sort_values()
    later = dates[dates > signal_date]
    if later.empty:
        raise ProtocolError("冻结交易日历中没有下一交易日")
    return pd.Timestamp(later.iloc[0])


def mature_observation(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
    payload_path: Path,
) -> dict[str, Any]:
    root = root.resolve()
    config, manifest = validate_bundle(root, config_path, expected_manifest_sha256)
    paths = _artifact_paths(root, config)
    observations, signals, intents, fills = _load_ledgers(paths)
    payload = _read_payload(payload_path)
    required = set(config["data_contract"]["required_maturity_fields"])
    missing = sorted(required - set(payload))
    if missing:
        raise ProtocolError(f"成熟回执缺少字段：{missing}")
    signal_date = _parse_trade_date(payload["signal_trade_date"], "signal_trade_date")
    exit_date = _parse_trade_date(payload["exit_trade_date"], "exit_trade_date")
    today = pd.Timestamp(datetime.now(tz=SHANGHAI).date())
    if exit_date != today:
        raise ProtocolError("成熟回执只能在下一交易日当天写入，禁止回填")
    expected_exit = _next_sse_trade_date(root, signal_date)
    if exit_date != expected_exit:
        raise ProtocolError("退出日期不是冻结日历中的下一上交所交易日")
    matches = observations.index[observations["trade_date"].astype(str).eq(str(signal_date.date()))]
    if len(matches) != 1:
        raise ProtocolError("找不到唯一的待成熟传导观测")
    index = int(matches[0])
    if bool(observations.at[index, "target_mature"]):
        raise ProtocolError("该观测已经成熟，禁止重复写入")
    price = float(payload["etf_next_0935_price"])
    if not np.isfinite(price) or price <= 0:
        raise ProtocolError("09:35价格不是正数")
    exchange_timestamp = _parse_timestamp(
        payload["etf_next_0935_exchange_timestamp"], "etf_next_0935_exchange_timestamp"
    )
    received_at = _parse_timestamp(payload["etf_next_0935_received_at"], "etf_next_0935_received_at")
    matured_at = _parse_timestamp(payload["matured_at"], "matured_at")
    target = _target_timestamp(exit_date, "09:35:00")
    clock = config["signal_clock"]
    _validate_target_trade(
        exchange_timestamp,
        received_at,
        target,
        float(clock["maximum_exchange_timestamp_staleness_seconds"]),
        float(clock["maximum_arrival_latency_seconds"]),
        "510300退出观察",
    )
    if matured_at < received_at or matured_at > target + pd.Timedelta(seconds=10):
        raise ProtocolError("成熟记录写入时点不合法")
    source_hash = _validate_hash(payload["etf_next_0935_source_sha256"], "etf_next_0935_source_sha256")
    if not str(payload["etf_next_0935_source_document_id"]).strip():
        raise ProtocolError("etf_next_0935_source_document_id为空")
    close_price = float(observations.at[index, "etf_close_price"])
    target_return = price / close_price - 1.0
    updates = {
        "target_mature": True,
        "exit_trade_date": str(exit_date.date()),
        "matured_at": matured_at.isoformat(),
        "etf_next_0935_price": price,
        "etf_next_0935_exchange_timestamp": exchange_timestamp.isoformat(),
        "etf_next_0935_received_at": received_at.isoformat(),
        "etf_next_0935_source_document_id": str(payload["etf_next_0935_source_document_id"]),
        "etf_next_0935_source_sha256": source_hash,
        "target_return_close_to_next_0935": target_return,
    }
    for column, value in updates.items():
        observations.at[index, column] = value
    atomic_write_parquet(paths["observation"], observations)
    signal_matches = signals.index[signals["trade_date"].astype(str).eq(str(signal_date.date()))]
    if len(signal_matches) == 1:
        signal_index = int(signal_matches[0])
        signals.at[signal_index, "exit_trade_date"] = str(exit_date.date())
        signals.at[signal_index, "signal_return"] = target_return
        signals.at[signal_index, "signal_return_status"] = "OBSERVATIONAL_SIGNAL_RETURN_NOT_ACTUAL_FILL_RETURN"
        atomic_write_parquet(paths["signal"], signals)
    status = _write_status(paths, config, manifest, observations, signals, intents, fills)
    return {
        "status": "FORWARD_OBSERVATION_MATURED",
        "signal_trade_date": str(signal_date.date()),
        "exit_trade_date": str(exit_date.date()),
        "target_return_close_to_next_0935": target_return,
        "beta_state_after_maturity": status["beta_state"],
        "actual_fill_return": "NO_VIEW_NO_LIVE_AUTHORIZATION",
        "position_impact": 0,
    }
