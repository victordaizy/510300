"""510300 有锚稀疏均值回归网格 V1 的输入审计与事件信息预算。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_anchored_sparse_mean_reversion_grid_v1.yaml"
MANIFEST_PATH = (
    ROOT / "config" / "510300_anchored_sparse_mean_reversion_grid_v1_manifest.json"
)


class DataContractError(RuntimeError):
    """输入或冻结合同不满足时抛出。"""


@dataclass(frozen=True)
class CostModel:
    """单层交易的完整往返成本。"""

    notional_cny: float
    commission_rate_per_leg: float
    minimum_commission_cny_per_leg: float
    slippage_bps_per_leg: float
    stress_round_trip_multiplier: float

    @property
    def commission_rate_effective_per_leg(self) -> float:
        commission_cny = max(
            self.notional_cny * self.commission_rate_per_leg,
            self.minimum_commission_cny_per_leg,
        )
        return commission_cny / self.notional_cny

    @property
    def base_round_trip_rate(self) -> float:
        per_leg = self.commission_rate_effective_per_leg + self.slippage_bps_per_leg / 10_000.0
        return 2.0 * per_leg

    @property
    def stress_round_trip_rate(self) -> float:
        return self.base_round_trip_rate * self.stress_round_trip_multiplier


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """读取并验证冻结前协议。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataContractError("协议根节点必须为对象")
    validate_config(payload)
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    """验证不可被悄悄放宽的核心研究边界。"""

    failures: list[str] = []
    protocol = config.get("protocol", {})
    scope = config.get("scope", {})
    account = config.get("account", {})
    event = config.get("event_definition", {})
    costs = config.get("costs", {})
    grid = config.get("grid_bands", {})
    gates = config.get("phase_a_gates", {})
    phase_b = config.get("phase_b", {})
    governance = config.get("governance", {})

    if protocol.get("model_id") != "510300_ANCHORED_SPARSE_MEAN_REVERSION_GRID_V1":
        failures.append("model_id")
    if protocol.get("research_stage") != "DISCOVERY":
        failures.append("research_stage")
    if bool(protocol.get("outcome_used_to_choose_parameters")):
        failures.append("outcome_used_to_choose_parameters")
    if scope.get("execution_asset") != "510300.SH":
        failures.append("execution_asset")
    if set(scope.get("allowed_holdings", [])) != {"510300.SH", "CASH_CNY"}:
        failures.append("allowed_holdings")
    for key in (
        "constituent_execution_allowed",
        "derivative_execution_allowed",
        "leverage_allowed",
        "short_selling_allowed",
        "martingale_position_sizing_allowed",
    ):
        if bool(scope.get(key)):
            failures.append(key)
    if not math.isclose(float(account.get("initial_capital_cny", -1.0)), 200_000.0):
        failures.append("initial_capital_cny")
    if not math.isclose(float(account.get("grid_notional_cny", -1.0)), 20_000.0):
        failures.append("grid_notional_cny")
    if int(account.get("maximum_buy_layers", -1)) != 2:
        failures.append("maximum_buy_layers")
    if int(account.get("maximum_sell_layers", -1)) != 2:
        failures.append("maximum_sell_layers")
    if event.get("forward_horizons_trading_minutes") != [5, 15, 30, 45, 60]:
        failures.append("forward_horizons_trading_minutes")
    if int(event.get("primary_horizon_trading_minutes", -1)) != 45:
        failures.append("primary_horizon_trading_minutes")
    if event.get("candidate_open_time") != "10:00":
        failures.append("candidate_open_time")
    if event.get("candidate_last_signal_time") != "14:15":
        failures.append("candidate_last_signal_time")
    if not math.isclose(float(costs.get("commission_rate_per_leg", -1.0)), 0.0002):
        failures.append("commission_rate_per_leg")
    if not math.isclose(float(costs.get("minimum_commission_cny_per_leg", -1.0)), 5.0):
        failures.append("minimum_commission_cny_per_leg")
    if not math.isclose(float(costs.get("base_slippage_bps_per_leg", -1.0)), 5.0):
        failures.append("base_slippage_bps_per_leg")
    if not math.isclose(float(costs.get("stress_round_trip_multiplier", -1.0)), 1.5):
        failures.append("stress_round_trip_multiplier")
    if not bool(costs.get("omitted_cost_is_failure")):
        failures.append("omitted_cost_is_failure")
    if not math.isclose(float(grid.get("first_scale_multiple", -1.0)), 2.0):
        failures.append("first_scale_multiple")
    if not math.isclose(
        float(grid.get("first_minimum_round_trip_cost_multiple", -1.0)), 2.5
    ):
        failures.append("first_minimum_round_trip_cost_multiple")
    if not math.isclose(float(grid.get("second_scale_multiple", -1.0)), 3.0):
        failures.append("second_scale_multiple")
    if not math.isclose(float(grid.get("second_minimum_increment", -1.0)), 0.002):
        failures.append("second_minimum_increment")
    if int(gates.get("minimum_completed_primary_events", -1)) != 300:
        failures.append("minimum_completed_primary_events")
    if not bool(gates.get("all_gates_must_pass_before_phase_b")):
        failures.append("all_gates_must_pass_before_phase_b")
    if bool(phase_b.get("portfolio_backtest_implemented_in_v1")):
        failures.append("portfolio_backtest_implemented_in_v1")
    if phase_b.get("status_before_phase_a_pass") != "NOT_ALLOWED":
        failures.append("phase_b_status_before_phase_a_pass")
    if governance.get("failed_phase_a_runs_phase_b") is not False:
        failures.append("failed_phase_a_runs_phase_b")
    if governance.get("order_generation") != "DISABLED":
        failures.append("order_generation")
    if governance.get("broker_connection") != "DISABLED":
        failures.append("broker_connection")
    if bool(governance.get("live_trading_authorized")):
        failures.append("live_trading_authorized")
    if failures:
        raise DataContractError(f"协议核心边界校验失败：{failures}")


def cost_model(config: Mapping[str, Any]) -> CostModel:
    """从具名字段构建成本模型，避免位置参数错位。"""

    account = config["account"]
    costs = config["costs"]
    return CostModel(
        notional_cny=float(account["grid_notional_cny"]),
        commission_rate_per_leg=float(costs["commission_rate_per_leg"]),
        minimum_commission_cny_per_leg=float(costs["minimum_commission_cny_per_leg"]),
        slippage_bps_per_leg=float(costs["base_slippage_bps_per_leg"]),
        stress_round_trip_multiplier=float(costs["stress_round_trip_multiplier"]),
    )


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(relative: str) -> Path:
    """把协议内 POSIX 相对路径解析到项目根目录。"""

    return ROOT / PurePosixPath(relative)


def json_safe(value: Any) -> Any:
    """把 pandas/numpy 值转为严格 JSON 可编码对象。"""

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """严格 JSON 原子写入。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(dict(payload)), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    """UTF-8 文本原子写入。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Parquet 原子写入。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """读取协议绑定输入，不执行任何未来收益计算。"""

    specifications = config["inputs"]
    inputs: dict[str, Any] = {}
    for key in ("etf_minute", "index_minute"):
        inputs[key] = pd.read_parquet(project_path(specifications[key]["path"]))
        metadata_path = project_path(specifications[key]["metadata_path"])
        inputs[f"{key}_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    inputs["etf_daily"] = pd.read_parquet(
        project_path(specifications["etf_daily_crosscheck"]["path"])
    )
    inputs["index_daily"] = pd.read_parquet(
        project_path(specifications["index_daily_crosscheck"]["path"])
    )
    inputs["dividends"] = pd.read_csv(
        project_path(specifications["dividends"]["path"]), encoding="utf-8"
    )
    return inputs


def _minute_audit(
    frame: pd.DataFrame,
    specification: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    require_full_ohlc: bool,
) -> dict[str, Any]:
    required = list(specification["required_columns"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return {
            "passed": False,
            "failures": [f"missing_columns:{missing}"],
            "missing_columns": missing,
        }

    work = frame.copy()
    work["trade_time"] = pd.to_datetime(work["trade_time"], errors="coerce")
    work["date"] = work["trade_time"].dt.normalize()
    work["clock"] = work["trade_time"].dt.strftime("%H:%M")
    counts = work.groupby("date", sort=True).size()
    model_columns = list(specification["model_price_columns"])
    invalid_model_price = (~np.isfinite(work[model_columns])).any(axis=1) | (
        work[model_columns] <= 0
    ).any(axis=1)
    invalid_amount = (~np.isfinite(work["amount"])) | (work["amount"] < 0)
    invalid_full_ohlc = (
        (~np.isfinite(work[["open", "high", "low", "close"]])).any(axis=1)
        | (work[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (work["high"] < work[["open", "close"]].max(axis=1))
        | (work["low"] > work[["open", "close"]].min(axis=1))
        | (work["high"] < work["low"])
    )
    duplicate_count = int(work["trade_time"].duplicated().sum())
    non_monotonic_count = int(
        (work["trade_time"].diff().dropna() <= pd.Timedelta(0)).sum()
    )
    expected_bars = int(contract["expected_bars_per_complete_day"])
    failures: list[str] = []
    checks = {
        "row_count": len(work) == int(contract["expected_row_count_each"]),
        "trading_day_count": work["date"].nunique()
        == int(contract["expected_trading_day_count"]),
        "bars_per_day": bool((counts == expected_bars).all()),
        "first_record_label": bool(
            (
                work.groupby("date", sort=True)["clock"].first()
                == contract["expected_first_record_label"]
            ).all()
        ),
        "last_record_label": bool(
            (
                work.groupby("date", sort=True)["clock"].last()
                == contract["expected_last_record_label"]
            ).all()
        ),
        "duplicate_timestamps": duplicate_count
        <= int(contract["duplicate_timestamp_tolerance"]),
        "strict_timestamp_order": non_monotonic_count == 0,
        "model_prices_positive_finite": int(invalid_model_price.sum()) == 0,
        "amount_nonnegative_finite": int(invalid_amount.sum()) == 0,
        "symbol_exact": set(work["ts_code"].dropna().astype(str).unique())
        == {str(specification["symbol"])},
        "date_range": work["date"].min() == pd.Timestamp(contract["raw_start"])
        and work["date"].max() == pd.Timestamp(contract["raw_end"]),
    }
    if require_full_ohlc:
        checks["full_ohlc_valid"] = int(invalid_full_ohlc.sum()) == 0
    else:
        checks["unused_high_low_anomalies_within_limit"] = int(invalid_full_ohlc.sum()) <= int(
            contract["maximum_unused_index_high_low_anomalies"]
        )
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "row_count": int(len(work)),
        "trading_day_count": int(work["date"].nunique()),
        "start": work["trade_time"].min(),
        "end": work["trade_time"].max(),
        "bar_count_distribution": {
            str(int(key)): int(value) for key, value in counts.value_counts().sort_index().items()
        },
        "duplicate_timestamp_count": duplicate_count,
        "non_monotonic_timestamp_count": non_monotonic_count,
        "invalid_model_price_count": int(invalid_model_price.sum()),
        "invalid_amount_count": int(invalid_amount.sum()),
        "unused_or_full_ohlc_anomaly_count": int(invalid_full_ohlc.sum()),
        "zero_amount_row_count": int((work["amount"] == 0).sum()),
    }


def _daily_crosscheck(
    minute: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    columns: Iterable[str],
    tolerance: float,
) -> dict[str, Any]:
    minute_work = minute.copy()
    minute_work["trade_time"] = pd.to_datetime(minute_work["trade_time"])
    minute_work["date"] = minute_work["trade_time"].dt.normalize()
    aggregate = (
        minute_work.groupby("date", sort=True)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .reset_index()
    )
    daily_work = daily.copy()
    daily_work["date"] = pd.to_datetime(daily_work["date"]).dt.normalize()
    duplicate_daily_dates = int(daily_work["date"].duplicated().sum())
    daily_work = daily_work.drop_duplicates("date", keep="last")
    merged = aggregate.merge(
        daily_work[["date", "open", "high", "low", "close"]],
        on="date",
        how="left",
        suffixes=("_minute", "_daily"),
        validate="one_to_one",
    )
    metrics: dict[str, Any] = {}
    checks: dict[str, bool] = {
        "daily_dates_unique": duplicate_daily_dates == 0,
        "daily_coverage_complete": int(merged["close_daily"].isna().sum()) == 0,
    }
    for column in columns:
        difference = (merged[f"{column}_minute"] - merged[f"{column}_daily"]).abs()
        maximum = float(difference.max()) if difference.notna().any() else math.nan
        metrics[column] = {
            "maximum_absolute_difference": maximum,
            "p99_absolute_difference": float(difference.quantile(0.99)),
            "exact_match_count": int((difference == 0).sum()),
        }
        checks[f"{column}_within_tolerance"] = bool(maximum <= tolerance)
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "metrics": metrics,
        "overlap_day_count": int(len(merged)),
        "duplicate_daily_date_count": duplicate_daily_dates,
    }


def audit_inputs(inputs: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """只做输入合同审计，不构造候选未来收益。"""

    contract = config["data_contract"]
    specifications = config["inputs"]
    etf = inputs["etf_minute"]
    index = inputs["index_minute"]
    etf_audit = _minute_audit(
        etf, specifications["etf_minute"], contract, require_full_ohlc=True
    )
    index_audit = _minute_audit(
        index, specifications["index_minute"], contract, require_full_ohlc=False
    )

    etf_times = pd.to_datetime(etf["trade_time"], errors="coerce")
    index_times = pd.to_datetime(index["trade_time"], errors="coerce")
    timestamps_equal = len(etf_times) == len(index_times) and np.array_equal(
        etf_times.to_numpy(), index_times.to_numpy()
    )
    alignment = {
        "passed": bool(timestamps_equal),
        "exact_timestamp_match_count": int(
            (etf_times.to_numpy() == index_times.to_numpy()).sum()
        )
        if len(etf_times) == len(index_times)
        else 0,
        "etf_row_count": int(len(etf_times)),
        "index_row_count": int(len(index_times)),
        "etf_only_timestamp_count": int(
            len(pd.Index(etf_times).difference(pd.Index(index_times)))
        ),
        "index_only_timestamp_count": int(
            len(pd.Index(index_times).difference(pd.Index(etf_times)))
        ),
    }

    etf_daily = _daily_crosscheck(
        etf,
        inputs["etf_daily"],
        columns=("open", "high", "low", "close"),
        tolerance=float(contract["etf_daily_ohlc_max_absolute_difference"]),
    )
    index_daily = _daily_crosscheck(
        index,
        inputs["index_daily"],
        columns=("open", "close"),
        tolerance=float(contract["index_daily_open_close_max_absolute_difference"]),
    )

    dividends = inputs["dividends"].copy()
    required_dividend_columns = {
        "symbol",
        "record_date",
        "ex_date",
        "payment_date",
        "cash_dividend_per_share",
        "source",
    }
    missing_dividend_columns = sorted(required_dividend_columns.difference(dividends.columns))
    if not missing_dividend_columns:
        dividends["ex_date"] = pd.to_datetime(dividends["ex_date"], errors="coerce")
        within = dividends.loc[
            dividends["ex_date"].between(
                pd.Timestamp(contract["raw_start"]), pd.Timestamp(contract["raw_end"])
            )
        ].copy()
        official_sources = within["source"].astype(str).map(
            lambda value: (urlparse(value).hostname or "").lower().endswith("sse.com.cn")
        )
        dividend_checks = {
            "required_columns": True,
            "symbol_exact": set(dividends["symbol"].dropna().astype(str).unique())
            == {specifications["dividends"]["symbol"]},
            "ex_dates_valid": bool(dividends["ex_date"].notna().all()),
            "ex_dates_unique": int(dividends["ex_date"].duplicated().sum()) == 0,
            "cash_dividend_positive": bool(
                (pd.to_numeric(dividends["cash_dividend_per_share"], errors="coerce") > 0).all()
            ),
            "in_sample_sources_official_sse": bool(official_sources.all()),
            "in_sample_dividend_count_positive": len(within) > 0,
        }
        in_sample_dividends = within[
            ["ex_date", "cash_dividend_per_share", "source"]
        ].to_dict("records")
    else:
        dividend_checks = {"required_columns": False}
        in_sample_dividends = []
    dividend_failures = [name for name, passed in dividend_checks.items() if not passed]
    dividend_audit = {
        "passed": not dividend_failures,
        "failures": dividend_failures,
        "missing_columns": missing_dividend_columns,
        "checks": dividend_checks,
        "in_sample_records": in_sample_dividends,
    }

    metadata_checks = {
        "etf_metadata_status_pass": inputs["etf_minute_metadata"].get("status") == "PASS",
        "index_metadata_status_pass": inputs["index_minute_metadata"].get("status") == "PASS",
        "etf_adjustment_raw": inputs["etf_minute_metadata"].get("price_adjustment")
        == "raw_unadjusted",
        "index_official_idx_mins_not_claimed": inputs["index_minute_metadata"].get(
            "official_idx_mins_permission"
        )
        is False,
    }
    metadata_failures = [name for name, passed in metadata_checks.items() if not passed]
    metadata_audit = {
        "passed": not metadata_failures,
        "failures": metadata_failures,
        "checks": metadata_checks,
        "index_source_limit": specifications["index_minute"]["source_limit"],
        "iopv_available": False,
        "iopv_claim_allowed": False,
    }

    sections = {
        "etf_minute": etf_audit,
        "index_minute": index_audit,
        "timestamp_alignment": alignment,
        "etf_daily_crosscheck": etf_daily,
        "index_daily_crosscheck": index_daily,
        "dividends": dividend_audit,
        "metadata": metadata_audit,
    }
    failures: list[str] = []
    for name, section in sections.items():
        if not bool(section.get("passed")):
            failures.append(name)
    return {
        "audit_id": "510300_ANCHORED_SPARSE_MEAN_REVERSION_GRID_V1_INPUT_AUDIT",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")),
        "status": "PASS_DISCOVERY_INDEX_PROXY_INPUTS" if not failures else "BLOCKED_INPUT_CONTRACT",
        "passed": not failures,
        "return_evaluation": "ALLOWED_AFTER_MANIFEST_VERIFICATION" if not failures else "NOT_ALLOWED",
        "failures": failures,
        "sections": sections,
        "evidence_limits": {
            "historical_iopv_complete": False,
            "anchor_is_index_implied_proxy_not_fair_value": True,
            "minute_record_start_or_end_semantics_proven": False,
            "unused_index_high_low_raw_rows_modified": False,
        },
    }


def _prior_rolling_location_scale(
    series: pd.Series, window: int
) -> tuple[pd.Series, pd.Series]:
    """同一桶内只用此前 window 条记录计算中位数与 MAD。"""

    shifted = series.shift(1)
    location = shifted.rolling(window=window, min_periods=window).median()

    def robust_mad(values: np.ndarray) -> float:
        median = float(np.median(values))
        return float(np.median(np.abs(values - median))) * 1.4826

    scale = shifted.rolling(window=window, min_periods=window).apply(
        robust_mad, raw=True
    )
    return location, scale


def _time_segment(total_minutes: int) -> str:
    if total_minutes < 10 * 60:
        return "OPEN_0930_0959"
    if total_minutes <= 11 * 60 + 30:
        return "MORNING_MIDDLE_1000_1130"
    if total_minutes <= 14 * 60 + 15:
        return "AFTERNOON_MIDDLE_1300_1415"
    return "TAIL_1416_1500"


def _label_path_efficiency(value: float) -> str:
    if not math.isfinite(value):
        return "UNKNOWN"
    if value < 0.3:
        return "LOW"
    if value < 0.6:
        return "MEDIUM"
    return "HIGH"


def _label_volume_ratio(value: float) -> str:
    if not math.isfinite(value):
        return "UNKNOWN"
    if value < 1.0:
        return "NORMAL"
    if value < 2.0:
        return "ELEVATED"
    return "EXTREME"


def _label_severity(value: float) -> str:
    if not math.isfinite(value):
        return "UNKNOWN"
    if value < 1.5:
        return "1.0_1.5"
    if value < 2.0:
        return "1.5_2.0"
    if value < 3.0:
        return "2.0_3.0"
    return "3.0_PLUS"


def build_minute_features(
    inputs: Mapping[str, Any], config: Mapping[str, Any]
) -> pd.DataFrame:
    """构造完全点时的分钟锚点、尺度、成交额与路径特征。"""

    etf = inputs["etf_minute"].copy().rename(
        columns={
            "open": "etf_open",
            "high": "etf_high",
            "low": "etf_low",
            "close": "etf_close",
            "vol": "etf_volume",
            "amount": "etf_amount",
        }
    )
    index = inputs["index_minute"].copy().rename(
        columns={
            "open": "index_open",
            "high": "index_high",
            "low": "index_low",
            "close": "index_close",
            "vol": "index_volume",
            "amount": "index_amount",
        }
    )
    etf["trade_time"] = pd.to_datetime(etf["trade_time"])
    index["trade_time"] = pd.to_datetime(index["trade_time"])
    panel = etf.drop(columns=["ts_code"]).merge(
        index.drop(columns=["ts_code"]),
        on="trade_time",
        how="inner",
        validate="one_to_one",
    )
    panel = panel.sort_values("trade_time", kind="stable").reset_index(drop=True)
    panel["date"] = panel["trade_time"].dt.normalize()
    panel["minute_label"] = panel["trade_time"].dt.strftime("%H:%M")
    panel["bar_index"] = panel.groupby("date", sort=False).cumcount()
    panel["minute_of_day"] = panel["trade_time"].dt.hour * 60 + panel["trade_time"].dt.minute
    panel["time_segment"] = panel["minute_of_day"].map(_time_segment)

    daily_close = panel.groupby("date", sort=True).agg(
        etf_day_close=("etf_close", "last"),
        index_day_close=("index_close", "last"),
    )
    daily_close["previous_etf_close"] = daily_close["etf_day_close"].shift(1)
    daily_close["previous_index_close"] = daily_close["index_day_close"].shift(1)
    panel = panel.merge(
        daily_close[["previous_etf_close", "previous_index_close"]],
        left_on="date",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    dividends = inputs["dividends"].copy()
    dividends["ex_date"] = pd.to_datetime(dividends["ex_date"]).dt.normalize()
    dividend_map = dividends.set_index("ex_date")["cash_dividend_per_share"].astype(float)
    panel["cash_dividend_per_share"] = panel["date"].map(dividend_map).fillna(0.0)
    panel["dividend_adjusted_previous_etf_close"] = (
        panel["previous_etf_close"] - panel["cash_dividend_per_share"]
    )
    panel["index_implied_price_proxy"] = (
        panel["dividend_adjusted_previous_etf_close"]
        * panel["index_close"]
        / panel["previous_index_close"]
    )
    panel["raw_log_basis"] = np.log(
        panel["etf_close"] / panel["index_implied_price_proxy"]
    )

    history_days = int(config["anchor"]["rolling_history_complete_days"])
    center = pd.Series(np.nan, index=panel.index, dtype=float)
    scale_raw = pd.Series(np.nan, index=panel.index, dtype=float)
    for _, indices in panel.groupby("minute_label", sort=False).groups.items():
        values = panel.loc[indices, "raw_log_basis"]
        group_center, group_scale = _prior_rolling_location_scale(values, history_days)
        center.loc[indices] = group_center
        scale_raw.loc[indices] = group_scale
    panel["basis_center_prior20"] = center
    panel["basis_scale_mad_raw_prior20"] = scale_raw
    tick = float(config["anchor"]["etf_tick_cny"])
    panel["basis_scale_tick_floor"] = tick / panel["previous_etf_close"]
    panel["basis_scale"] = np.maximum(
        panel["basis_scale_mad_raw_prior20"], panel["basis_scale_tick_floor"]
    )
    panel["centered_log_basis"] = panel["raw_log_basis"] - panel["basis_center_prior20"]
    panel["basis_z"] = panel["centered_log_basis"] / panel["basis_scale"]

    amount_window = int(config["features"]["trailing_amount_minutes"])
    panel["etf_amount_5m"] = panel.groupby("date", sort=False)["etf_amount"].transform(
        lambda values: values.rolling(amount_window, min_periods=amount_window).sum()
    )
    amount_history = int(config["features"]["trailing_amount_seasonal_days"])
    panel["etf_amount_5m_prior20_same_minute_median"] = panel.groupby(
        "minute_label", sort=False
    )["etf_amount_5m"].transform(
        lambda values: values.shift(1).rolling(
            amount_history, min_periods=amount_history
        ).median()
    )
    panel["etf_amount_5m_ratio"] = (
        panel["etf_amount_5m"]
        / panel["etf_amount_5m_prior20_same_minute_median"]
    )

    panel["etf_log_price"] = np.log(panel["etf_close"])
    panel["index_log_price"] = np.log(panel["index_close"])
    panel["etf_log_return_1m"] = panel.groupby("date", sort=False)[
        "etf_log_price"
    ].diff()
    panel["index_log_return_1m"] = panel.groupby("date", sort=False)[
        "index_log_price"
    ].diff()
    efficiency_window = int(config["features"]["path_efficiency_minutes"])
    displacement = panel["etf_log_price"] - panel.groupby("date", sort=False)[
        "etf_log_price"
    ].shift(efficiency_window)
    path_length = panel.groupby("date", sort=False)["etf_log_return_1m"].transform(
        lambda values: values.abs().rolling(
            efficiency_window, min_periods=efficiency_window
        ).sum()
    )
    panel["path_efficiency_30m"] = (displacement.abs() / path_length).clip(0.0, 1.0)
    panel["etf_log_return_5m"] = panel["etf_log_price"] - panel.groupby(
        "date", sort=False
    )["etf_log_price"].shift(5)
    panel["index_log_return_5m"] = panel["index_log_price"] - panel.groupby(
        "date", sort=False
    )["index_log_price"].shift(5)
    panel["relative_log_return_5m"] = (
        panel["etf_log_return_5m"] - panel["index_log_return_5m"]
    )

    daily_rv = panel.groupby("date", sort=True)["etf_log_return_1m"].apply(
        lambda values: float(np.sqrt(np.nansum(np.square(values.to_numpy(dtype=float)))))
    )
    market_window = int(config["features"]["market_state_prior_day_realized_volatility_days"])
    prior_rv = daily_rv.shift(1)
    lower_quantile = prior_rv.rolling(market_window, min_periods=market_window).quantile(
        float(config["features"]["market_state_quantiles"][0])
    )
    upper_quantile = prior_rv.rolling(market_window, min_periods=market_window).quantile(
        float(config["features"]["market_state_quantiles"][1])
    )
    market_state = pd.Series("UNKNOWN", index=daily_rv.index, dtype="string")
    known = prior_rv.notna() & lower_quantile.notna() & upper_quantile.notna()
    market_state.loc[known & (prior_rv <= lower_quantile)] = "LOW_VOL"
    market_state.loc[known & (prior_rv > lower_quantile) & (prior_rv <= upper_quantile)] = (
        "MEDIUM_VOL"
    )
    market_state.loc[known & (prior_rv > upper_quantile)] = "HIGH_VOL"
    panel["prior_day_realized_volatility"] = panel["date"].map(prior_rv)
    panel["market_state"] = panel["date"].map(market_state).fillna("UNKNOWN")

    full_day_efficiency: dict[pd.Timestamp, float] = {}
    for date, day in panel.groupby("date", sort=True):
        log_prices = day["etf_log_price"].to_numpy(dtype=float)
        path = float(np.abs(np.diff(log_prices)).sum())
        displacement_day = float(abs(log_prices[-1] - log_prices[0]))
        full_day_efficiency[pd.Timestamp(date)] = displacement_day / path if path > 0 else 0.0
    panel["outcome_only_full_day_path_efficiency"] = panel["date"].map(
        full_day_efficiency
    )

    model = cost_model(config)
    base_cost = model.base_round_trip_rate
    grid = config["grid_bands"]
    panel["base_round_trip_cost"] = base_cost
    panel["stress_round_trip_cost"] = model.stress_round_trip_rate
    panel["first_grid_band"] = np.maximum(
        float(grid["first_scale_multiple"]) * panel["basis_scale"],
        float(grid["first_minimum_round_trip_cost_multiple"]) * base_cost,
    )
    panel["second_grid_band"] = np.maximum(
        float(grid["second_scale_multiple"]) * panel["basis_scale"],
        panel["first_grid_band"] + float(grid["second_minimum_increment"]),
    )
    panel["feature_ready"] = panel[
        [
            "centered_log_basis",
            "basis_scale",
            "basis_z",
            "etf_amount_5m_ratio",
        ]
    ].notna().all(axis=1)
    return panel


def extract_events(panel: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """按因果片段规则提取互不重叠的偏离事件。"""

    definition = config["event_definition"]
    entry_abs_z = float(definition["diagnostic_episode_entry_abs_z"])
    reset_abs_z = float(definition["center_or_reset_abs_z"])
    minimum_spacing = int(definition["minimum_event_spacing_trading_minutes"])
    minimum_volume_ratio = float(
        definition["liquidity_shock_minimum_trailing_amount_ratio"]
    )
    maximum_decay_fraction = float(
        definition["liquidity_decay_maximum_fraction_of_episode_peak_amount"]
    )
    open_minutes = 10 * 60
    last_minutes = 14 * 60 + 15
    events: list[dict[str, Any]] = []

    for date, day in panel.groupby("date", sort=True):
        day = day.sort_values("bar_index", kind="stable")
        active = False
        active_sign = 0
        emitted = False
        previous_abs_deviation = math.nan
        episode_peak_abs_z = math.nan
        episode_peak_amount = math.nan
        episode_start_time: pd.Timestamp | None = None
        last_event_bar = -10_000

        for row in day.itertuples(index=True):
            if not bool(row.feature_ready) or not math.isfinite(float(row.basis_z)):
                continue
            current_sign = 1 if float(row.centered_log_basis) > 0 else -1
            current_abs_z = abs(float(row.basis_z))
            current_abs_deviation = abs(float(row.centered_log_basis))
            current_amount = float(row.etf_amount_5m)

            if active and (current_abs_z < reset_abs_z or current_sign != active_sign):
                active = False
                emitted = False

            if not active and current_abs_z >= entry_abs_z:
                active = True
                active_sign = current_sign
                emitted = False
                previous_abs_deviation = current_abs_deviation
                episode_peak_abs_z = current_abs_z
                episode_peak_amount = current_amount
                episode_start_time = pd.Timestamp(row.trade_time)
                continue

            if not active:
                continue

            episode_peak_abs_z = max(episode_peak_abs_z, current_abs_z)
            episode_peak_amount = max(episode_peak_amount, current_amount)
            non_widening = current_abs_deviation <= previous_abs_deviation
            if non_widening and not emitted:
                emitted = True
                if int(row.bar_index) - last_event_bar >= minimum_spacing:
                    decay_fraction = (
                        current_amount / episode_peak_amount
                        if episode_peak_amount > 0
                        else math.nan
                    )
                    amount_ratio = float(row.etf_amount_5m_ratio)
                    liquidity_qualified = (
                        amount_ratio >= minimum_volume_ratio
                        and math.isfinite(decay_fraction)
                        and decay_fraction <= maximum_decay_fraction
                    )
                    candidate_time_allowed = open_minutes <= int(row.minute_of_day) <= last_minutes
                    first_grid_qualified = (
                        abs(float(row.centered_log_basis)) >= float(row.first_grid_band)
                    )
                    second_grid_qualified = (
                        abs(float(row.centered_log_basis)) >= float(row.second_grid_band)
                    )
                    path_efficiency = float(row.path_efficiency_30m)
                    event_time = pd.Timestamp(row.trade_time)
                    events.append(
                        {
                            "event_id": f"{event_time:%Y%m%dT%H%M}_{'PREMIUM' if active_sign > 0 else 'DISCOUNT'}",
                            "date": pd.Timestamp(date),
                            "signal_time": event_time,
                            "signal_bar_index": int(row.bar_index),
                            "panel_row_index": int(row.Index),
                            "episode_start_time": episode_start_time,
                            "direction_sign": int(active_sign),
                            "direction": "PREMIUM_SELL_OLD_INVENTORY"
                            if active_sign > 0
                            else "DISCOUNT_BUY_WITH_CASH",
                            "raw_log_basis": float(row.raw_log_basis),
                            "basis_center_prior20": float(row.basis_center_prior20),
                            "centered_log_basis": float(row.centered_log_basis),
                            "abs_centered_log_basis": abs(float(row.centered_log_basis)),
                            "basis_scale": float(row.basis_scale),
                            "basis_z": float(row.basis_z),
                            "abs_basis_z": current_abs_z,
                            "episode_peak_abs_z_observed": episode_peak_abs_z,
                            "severity_bin": _label_severity(current_abs_z),
                            "time_segment": str(row.time_segment),
                            "candidate_time_allowed": bool(candidate_time_allowed),
                            "etf_amount_5m": current_amount,
                            "etf_amount_5m_ratio": amount_ratio,
                            "volume_ratio_bin": _label_volume_ratio(amount_ratio),
                            "episode_peak_amount_5m_observed": episode_peak_amount,
                            "liquidity_decay_fraction": decay_fraction,
                            "liquidity_shock_qualified": bool(liquidity_qualified),
                            "path_efficiency_30m": path_efficiency,
                            "path_efficiency_bin": _label_path_efficiency(path_efficiency),
                            "market_state": str(row.market_state),
                            "etf_log_return_5m": float(row.etf_log_return_5m),
                            "index_log_return_5m": float(row.index_log_return_5m),
                            "relative_log_return_5m": float(row.relative_log_return_5m),
                            "pre_shock_direction": "UP"
                            if float(row.etf_log_return_5m) > 0
                            else "DOWN"
                            if float(row.etf_log_return_5m) < 0
                            else "FLAT",
                            "base_round_trip_cost": float(row.base_round_trip_cost),
                            "stress_round_trip_cost": float(row.stress_round_trip_cost),
                            "first_grid_band": float(row.first_grid_band),
                            "second_grid_band": float(row.second_grid_band),
                            "first_grid_qualified": bool(first_grid_qualified),
                            "second_grid_qualified": bool(second_grid_qualified),
                            "cost_grid_qualified": bool(
                                candidate_time_allowed
                                and liquidity_qualified
                                and first_grid_qualified
                            ),
                            "outcome_only_full_day_path_efficiency": float(
                                row.outcome_only_full_day_path_efficiency
                            ),
                            "future_information_used_in_signal": False,
                        }
                    )
                    last_event_bar = int(row.bar_index)
            previous_abs_deviation = current_abs_deviation

    return pd.DataFrame(events)


def attach_event_outcomes(
    events: pd.DataFrame, panel: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """从下一条记录开盘开始计算预声明期限的方向调整结果。"""

    if events.empty:
        return events.copy()
    definition = config["event_definition"]
    horizons = [int(value) for value in definition["forward_horizons_trading_minutes"]]
    center_multiple = float(definition["center_or_reset_abs_z"])
    stop_multiple = float(definition["stop_abs_z"])
    by_date = {
        pd.Timestamp(date): day.sort_values("bar_index", kind="stable").reset_index(drop=True)
        for date, day in panel.groupby("date", sort=True)
    }
    output_rows: list[dict[str, Any]] = []

    for event in events.to_dict("records"):
        row = dict(event)
        day = by_date[pd.Timestamp(event["date"])]
        signal_index = int(event["signal_bar_index"])
        entry_index = signal_index + 1
        direction_sign = int(event["direction_sign"])
        if entry_index >= len(day):
            for horizon in horizons:
                row[f"outcome_{horizon}m_complete"] = False
            output_rows.append(row)
            continue
        entry = day.iloc[entry_index]
        etf_entry = float(entry["etf_open"])
        index_entry = float(entry["index_open"])
        row["entry_time"] = pd.Timestamp(entry["trade_time"])
        row["entry_etf_open"] = etf_entry
        row["entry_index_open"] = index_entry

        for horizon in horizons:
            exit_index = signal_index + horizon
            complete = exit_index < len(day)
            row[f"outcome_{horizon}m_complete"] = bool(complete)
            if not complete:
                continue
            exit_row = day.iloc[exit_index]
            etf_return = math.log(float(exit_row["etf_close"]) / etf_entry)
            index_return = math.log(float(exit_row["index_close"]) / index_entry)
            etf_reversal = -direction_sign * etf_return
            anchor_catchup = direction_sign * index_return
            convergence = etf_reversal + anchor_catchup
            path = day.iloc[entry_index : exit_index + 1]
            direction_returns = -direction_sign * np.log(
                path["etf_close"].to_numpy(dtype=float) / etf_entry
            )
            first_hit = "NONE"
            first_hit_time: pd.Timestamp | None = None
            for path_row in path.itertuples(index=False):
                if not (
                    math.isfinite(float(path_row.centered_log_basis))
                    and math.isfinite(float(path_row.basis_scale))
                ):
                    continue
                center_hit = abs(float(path_row.centered_log_basis)) <= (
                    center_multiple * float(path_row.basis_scale)
                )
                stop_hit = direction_sign * float(path_row.centered_log_basis) >= (
                    stop_multiple * float(path_row.basis_scale)
                )
                if stop_hit:
                    first_hit = "STOP_4SIGMA"
                    first_hit_time = pd.Timestamp(path_row.trade_time)
                    break
                if center_hit:
                    first_hit = "CENTER_0_5SIGMA"
                    first_hit_time = pd.Timestamp(path_row.trade_time)
                    break
            row.update(
                {
                    f"exit_time_{horizon}m": pd.Timestamp(exit_row["trade_time"]),
                    f"etf_log_return_{horizon}m": etf_return,
                    f"index_log_return_{horizon}m": index_return,
                    f"etf_reversal_score_{horizon}m": etf_reversal,
                    f"anchor_catchup_score_{horizon}m": anchor_catchup,
                    f"convergence_score_{horizon}m": convergence,
                    f"net_base_score_{horizon}m": etf_reversal
                    - float(event["base_round_trip_cost"]),
                    f"net_stress_score_{horizon}m": etf_reversal
                    - float(event["stress_round_trip_cost"]),
                    f"maximum_favorable_score_{horizon}m": float(np.max(direction_returns)),
                    f"maximum_adverse_score_{horizon}m": float(np.min(direction_returns)),
                    f"first_hit_{horizon}m": first_hit,
                    f"first_hit_time_{horizon}m": first_hit_time,
                    f"center_before_stop_{horizon}m": first_hit == "CENTER_0_5SIGMA",
                    f"stop_before_center_{horizon}m": first_hit == "STOP_4SIGMA",
                }
            )
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def _mean_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else None


def _table_records(
    frame: pd.DataFrame, group_column: str, outcome_column: str
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for label, group in frame.groupby(group_column, dropna=False, sort=True):
        values = pd.to_numeric(group[outcome_column], errors="coerce").dropna()
        records.append(
            {
                group_column: str(label),
                "event_count": int(len(values)),
                "mean": float(values.mean()) if len(values) else None,
                "median": float(values.median()) if len(values) else None,
                "sum": float(values.sum()) if len(values) else None,
                "positive_rate": float((values > 0).mean()) if len(values) else None,
            }
        )
    return records


def moving_block_day_bootstrap_lower_bound(
    event_dates: pd.Series,
    event_values: pd.Series,
    calendar_dates: pd.Series,
    *,
    repetitions: int,
    block_length: int,
    random_seed: int,
    lower_quantile: float,
) -> dict[str, Any]:
    """按完整交易日日历移动块抽样，并在每次样本内计算等事件权重均值。"""

    normalized_calendar = pd.to_datetime(calendar_dates)
    if isinstance(normalized_calendar, pd.Series):
        normalized_calendar = normalized_calendar.dt.normalize()
    else:
        normalized_calendar = normalized_calendar.normalize()
    calendar = pd.Index(normalized_calendar.unique()).sort_values()
    if len(calendar) == 0:
        return {
            "status": "NO_CALENDAR",
            "lower_bound": None,
            "repetitions": int(repetitions),
        }
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(event_dates).dt.normalize(),
            "value": pd.to_numeric(event_values, errors="coerce"),
        }
    ).dropna()
    grouped = data.groupby("date")["value"].agg(["sum", "count"])
    daily_sum = grouped["sum"].reindex(calendar, fill_value=0.0).to_numpy(dtype=float)
    daily_count = grouped["count"].reindex(calendar, fill_value=0).to_numpy(dtype=int)
    if int(daily_count.sum()) == 0:
        return {
            "status": "NO_EVENTS",
            "lower_bound": None,
            "repetitions": int(repetitions),
        }
    block = min(int(block_length), len(calendar))
    possible_starts = len(calendar) - block + 1
    blocks_needed = int(math.ceil(len(calendar) / block))
    generator = np.random.default_rng(int(random_seed))
    estimates = np.empty(int(repetitions), dtype=float)
    for repetition in range(int(repetitions)):
        starts = generator.integers(0, possible_starts, size=blocks_needed)
        indices = np.concatenate(
            [np.arange(start, start + block, dtype=int) for start in starts]
        )[: len(calendar)]
        denominator = int(daily_count[indices].sum())
        estimates[repetition] = (
            float(daily_sum[indices].sum()) / denominator if denominator else math.nan
        )
    finite = estimates[np.isfinite(estimates)]
    return {
        "status": "PASS_COMPUTED" if len(finite) else "NO_FINITE_REPLICATES",
        "lower_bound": float(np.quantile(finite, lower_quantile)) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "upper_bound": float(np.quantile(finite, 1.0 - lower_quantile))
        if len(finite)
        else None,
        "repetitions": int(repetitions),
        "finite_repetitions": int(len(finite)),
        "block_length_trading_days": int(block),
        "calendar_day_count": int(len(calendar)),
        "event_count": int(daily_count.sum()),
        "random_seed": int(random_seed),
    }


def _monotonic_gate(
    records: list[dict[str, Any]],
    order: list[str],
    *,
    minimum_count: int,
    nondecreasing: bool,
) -> bool:
    mapping = {str(record[next(iter(record.keys()))]): record for record in records}
    counts: list[int] = []
    means: list[float] = []
    for label in order:
        record = mapping.get(label)
        if record is None or record.get("mean") is None:
            return False
        counts.append(int(record["event_count"]))
        means.append(float(record["mean"]))
    if any(count < minimum_count for count in counts):
        return False
    if nondecreasing:
        return all(right >= left for left, right in zip(means, means[1:]))
    return all(right <= left for left, right in zip(means, means[1:]))


def evaluate_phase_a(
    outcomes: pd.DataFrame,
    panel: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """执行第一阶段硬门并明确第二阶段是否获准。"""

    gates_config = config["phase_a_gates"]
    primary_horizon = int(config["event_definition"]["primary_horizon_trading_minutes"])
    complete_column = f"outcome_{primary_horizon}m_complete"
    base_column = f"net_base_score_{primary_horizon}m"
    stress_column = f"net_stress_score_{primary_horizon}m"
    etf_column = f"etf_reversal_score_{primary_horizon}m"
    anchor_column = f"anchor_catchup_score_{primary_horizon}m"
    convergence_column = f"convergence_score_{primary_horizon}m"

    if outcomes.empty or complete_column not in outcomes.columns:
        primary = outcomes.iloc[0:0].copy()
        diagnostic = outcomes.iloc[0:0].copy()
    else:
        completed = outcomes[complete_column].fillna(False).astype(bool)
        primary = outcomes.loc[
            completed & outcomes["cost_grid_qualified"].fillna(False).astype(bool)
        ].copy()
        diagnostic = outcomes.loc[
            completed
            & outcomes["candidate_time_allowed"].fillna(False).astype(bool)
            & outcomes["liquidity_shock_qualified"].fillna(False).astype(bool)
        ].copy()

    primary_count = int(len(primary))
    base_mean = _mean_or_none(primary[base_column]) if base_column in primary else None
    stress_mean = _mean_or_none(primary[stress_column]) if stress_column in primary else None
    etf_mean = _mean_or_none(primary[etf_column]) if etf_column in primary else None
    anchor_mean = _mean_or_none(primary[anchor_column]) if anchor_column in primary else None
    convergence_mean = (
        _mean_or_none(primary[convergence_column]) if convergence_column in primary else None
    )

    calendar_dates = panel["date"].drop_duplicates().sort_values()
    bootstrap_config = gates_config["bootstrap"]
    bootstrap = moving_block_day_bootstrap_lower_bound(
        primary["date"] if "date" in primary else pd.Series(dtype="datetime64[ns]"),
        primary[base_column] if base_column in primary else pd.Series(dtype=float),
        calendar_dates,
        repetitions=int(bootstrap_config["repetitions"]),
        block_length=int(bootstrap_config["block_length_trading_days"]),
        random_seed=int(bootstrap_config["random_seed"]),
        lower_quantile=float(bootstrap_config["lower_quantile"]),
    )

    direction_table = _table_records(primary, "direction", base_column)
    year_frame = primary.copy()
    if not year_frame.empty:
        year_frame["calendar_year"] = pd.to_datetime(year_frame["date"]).dt.year.astype(str)
    year_table = _table_records(year_frame, "calendar_year", base_column)
    positive_years = [
        record for record in year_table if record.get("mean") is not None and record["mean"] > 0
    ]
    positive_contributions = [
        max(float(record["sum"]), 0.0)
        for record in year_table
        if record.get("sum") is not None
    ]
    total_positive_contribution = float(sum(positive_contributions))
    maximum_year_contribution_share = (
        max(positive_contributions) / total_positive_contribution
        if total_positive_contribution > 0 and positive_contributions
        else None
    )

    severity_table = _table_records(diagnostic, "severity_bin", etf_column)
    path_table = _table_records(primary, "path_efficiency_bin", base_column)
    volume_table = _table_records(diagnostic, "volume_ratio_bin", base_column)
    time_table = _table_records(
        outcomes.loc[
            outcomes.get(complete_column, pd.Series(False, index=outcomes.index)).fillna(False)
            & outcomes.get(
                "liquidity_shock_qualified", pd.Series(False, index=outcomes.index)
            ).fillna(False)
        ]
        if not outcomes.empty
        else outcomes,
        "time_segment",
        base_column,
    )
    market_table = _table_records(primary, "market_state", base_column)

    platform_table: list[dict[str, Any]] = []
    for multiple in config["grid_bands"]["nearby_scale_multiples_for_platform"]:
        if diagnostic.empty:
            subset = diagnostic
        else:
            threshold = np.maximum(
                float(multiple) * diagnostic["basis_scale"].astype(float),
                float(config["grid_bands"]["first_minimum_round_trip_cost_multiple"])
                * diagnostic["base_round_trip_cost"].astype(float),
            )
            subset = diagnostic.loc[
                diagnostic["abs_centered_log_basis"].astype(float) >= threshold
            ]
        platform_table.append(
            {
                "scale_multiple": float(multiple),
                "event_count": int(len(subset)),
                "mean_net_base": _mean_or_none(subset[base_column])
                if base_column in subset
                else None,
            }
        )

    direction_means = {
        str(record["direction"]): record for record in direction_table
    }
    direction_gate = len(direction_means) == 2 and all(
        int(record["event_count"]) >= int(gates_config["minimum_direction_events"])
        and record.get("mean") is not None
        and float(record["mean"]) > 0
        for record in direction_means.values()
    )
    severity_gate = _monotonic_gate(
        severity_table,
        ["1.0_1.5", "1.5_2.0", "2.0_3.0", "3.0_PLUS"],
        minimum_count=int(gates_config["minimum_events_per_monotonic_bin"]),
        nondecreasing=True,
    )
    path_gate = _monotonic_gate(
        path_table,
        ["LOW", "MEDIUM", "HIGH"],
        minimum_count=int(gates_config["minimum_events_per_monotonic_bin"]),
        nondecreasing=False,
    )
    platform_gate = all(
        int(record["event_count"])
        >= int(gates_config["minimum_events_per_platform_threshold"])
        and record.get("mean_net_base") is not None
        and float(record["mean_net_base"]) > 0
        for record in platform_table
    )

    available_sample_years = sorted(
        int(value) for value in pd.to_datetime(panel["date"]).dt.year.unique()
    )
    latest_two_years = available_sample_years[-2:]
    year_mapping = {int(record["calendar_year"]): record for record in year_table}
    latest_two_positive = len(latest_two_years) == 2 and all(
        year in year_mapping
        and year_mapping[year].get("mean") is not None
        and float(year_mapping[year]["mean"]) > 0
        for year in latest_two_years
    )

    all_time_completed = outcomes.iloc[0:0]
    if not outcomes.empty and complete_column in outcomes.columns:
        all_time_completed = outcomes.loc[
            outcomes[complete_column].fillna(False).astype(bool)
            & outcomes["liquidity_shock_qualified"].fillna(False).astype(bool)
        ].copy()
    special_segments = {"OPEN_0930_0959", "TAIL_1416_1500"}
    if all_time_completed.empty:
        special_time_profit_share = None
        special_time_gate = False
    else:
        segment_sums = all_time_completed.groupby("time_segment")[base_column].sum()
        positive_total = float(segment_sums.clip(lower=0.0).sum())
        special_positive = float(
            segment_sums.loc[segment_sums.index.isin(special_segments)].clip(lower=0.0).sum()
        )
        special_time_profit_share = (
            special_positive / positive_total if positive_total > 0 else None
        )
        special_time_gate = (
            special_time_profit_share is not None and special_time_profit_share <= 0.50
        )

    daily_efficiency = (
        panel[["date", "outcome_only_full_day_path_efficiency"]]
        .drop_duplicates("date")
        .set_index("date")["outcome_only_full_day_path_efficiency"]
    )
    trend_threshold = float(daily_efficiency.quantile(0.90))
    if primary.empty:
        top_trend_sum = None
        ordinary_positive_sum = None
        trend_tail_gate = False
    else:
        daily_event_sum = primary.groupby("date")[base_column].sum()
        top_dates = set(daily_efficiency.loc[daily_efficiency >= trend_threshold].index)
        top_trend_sum = float(daily_event_sum.loc[daily_event_sum.index.isin(top_dates)].sum())
        ordinary = daily_event_sum.loc[~daily_event_sum.index.isin(top_dates)]
        ordinary_positive_sum = float(ordinary.clip(lower=0.0).sum())
        trend_tail_gate = top_trend_sum >= 0 or abs(top_trend_sum) <= ordinary_positive_sum

    gates = {
        "completed_event_count_at_least_300": primary_count
        >= int(gates_config["minimum_completed_primary_events"]),
        "base_mean_net_return_positive": base_mean is not None and base_mean > 0,
        "day_block_bootstrap_95pct_lower_bound_positive": bootstrap.get("lower_bound")
        is not None
        and float(bootstrap["lower_bound"])
        > float(bootstrap_config["minimum_lower_bound"]),
        "at_least_four_positive_calendar_years": len(positive_years)
        >= int(gates_config["minimum_positive_calendar_years"]),
        "single_positive_year_contribution_not_above_50pct": maximum_year_contribution_share
        is not None
        and maximum_year_contribution_share
        <= float(gates_config["maximum_single_positive_year_contribution"]),
        "stress_1_5x_cost_mean_positive": stress_mean is not None and stress_mean > 0,
        "both_directions_count_and_net_mean_positive": direction_gate,
        "etf_reversal_positive_and_dominates_anchor_catchup": etf_mean is not None
        and anchor_mean is not None
        and etf_mean > 0
        and etf_mean > anchor_mean,
        "severity_monotonic_with_minimum_bin_count": severity_gate,
        "lower_path_efficiency_stronger_with_minimum_bin_count": path_gate,
        "nearby_threshold_platform_positive_with_minimum_counts": platform_gate,
        "latest_two_calendar_years_positive": latest_two_positive,
        "profit_not_mainly_open_or_tail": special_time_gate,
        "strongest_10pct_trend_days_do_not_consume_ordinary_positive_sum": trend_tail_gate,
        "phase_a_uses_no_limit_fill_or_same_record_round_trip": True,
    }
    all_pass = all(gates.values())

    horizon_table: list[dict[str, Any]] = []
    for horizon in config["event_definition"]["forward_horizons_trading_minutes"]:
        horizon = int(horizon)
        complete = f"outcome_{horizon}m_complete"
        subset = outcomes.iloc[0:0]
        if not outcomes.empty and complete in outcomes.columns:
            subset = outcomes.loc[
                outcomes["cost_grid_qualified"].fillna(False).astype(bool)
                & outcomes[complete].fillna(False).astype(bool)
            ]
        horizon_table.append(
            {
                "horizon_minutes": horizon,
                "event_count": int(len(subset)),
                "mean_etf_reversal": _mean_or_none(
                    subset[f"etf_reversal_score_{horizon}m"]
                )
                if f"etf_reversal_score_{horizon}m" in subset
                else None,
                "mean_anchor_catchup": _mean_or_none(
                    subset[f"anchor_catchup_score_{horizon}m"]
                )
                if f"anchor_catchup_score_{horizon}m" in subset
                else None,
                "mean_convergence": _mean_or_none(
                    subset[f"convergence_score_{horizon}m"]
                )
                if f"convergence_score_{horizon}m" in subset
                else None,
                "mean_net_base": _mean_or_none(subset[f"net_base_score_{horizon}m"])
                if f"net_base_score_{horizon}m" in subset
                else None,
                "mean_net_stress": _mean_or_none(
                    subset[f"net_stress_score_{horizon}m"]
                )
                if f"net_stress_score_{horizon}m" in subset
                else None,
                "center_before_stop_rate": float(
                    subset[f"center_before_stop_{horizon}m"].astype(float).mean()
                )
                if len(subset) and f"center_before_stop_{horizon}m" in subset
                else None,
                "stop_before_center_rate": float(
                    subset[f"stop_before_center_{horizon}m"].astype(float).mean()
                )
                if len(subset) and f"stop_before_center_{horizon}m" in subset
                else None,
            }
        )

    failed_gates = [name for name, passed in gates.items() if not passed]
    stop_codes = {
        "STOP_1_COST_ERASES_REVERSAL": not gates["base_mean_net_return_positive"],
        "STOP_2_CONVERGENCE_NOT_ETF_LED": not gates[
            "etf_reversal_positive_and_dominates_anchor_catchup"
        ],
        "STOP_3_PROFIT_MAINLY_OPEN_OR_TAIL": not gates["profit_not_mainly_open_or_tail"],
        "STOP_4_TREND_TAIL_LOSS_TOO_LARGE": not gates[
            "strongest_10pct_trend_days_do_not_consume_ordinary_positive_sum"
        ],
        "STOP_5_WIDER_GRID_EVENT_BUDGET_INSUFFICIENT": not gates[
            "completed_event_count_at_least_300"
        ]
        or not gates["nearby_threshold_platform_positive_with_minimum_counts"],
        "STOP_6_FICTITIOUS_SAME_RECORD_FILL": False,
        "STOP_7_RECENT_TWO_YEARS_DISAPPEAR": not gates[
            "latest_two_calendar_years_positive"
        ],
        "STOP_8_SAME_EXPOSURE_BENCHMARK_NO_EXCESS": "NOT_EVALUATED_PHASE_A",
    }
    return {
        "status": "PHASE_A_PASS_PHASE_B_ALLOWED_NOT_RUN"
        if all_pass
        else "REJECTED_PHASE_A_NO_GRID_BACKTEST_NO_RESCUE",
        "phase_a_passed": bool(all_pass),
        "phase_b_portfolio_backtest": "ALLOWED_NOT_RUN" if all_pass else "NOT_ALLOWED",
        "portfolio_sharpe_uplift_vs_same_average_exposure": "NOT_EVALUATED_PHASE_A",
        "primary_horizon_minutes": primary_horizon,
        "metrics": {
            "diagnostic_event_count": int(len(outcomes)),
            "liquidity_shock_candidate_time_event_count": int(len(diagnostic)),
            "primary_cost_grid_qualified_completed_event_count": primary_count,
            "base_mean_net_event_return": base_mean,
            "stress_mean_net_event_return": stress_mean,
            "mean_etf_reversal": etf_mean,
            "mean_anchor_catchup": anchor_mean,
            "mean_total_convergence": convergence_mean,
            "positive_calendar_year_count": int(len(positive_years)),
            "maximum_single_positive_year_contribution_share": maximum_year_contribution_share,
            "special_open_tail_positive_profit_share": special_time_profit_share,
            "full_day_path_efficiency_p90": trend_threshold,
            "top_10pct_trend_day_event_sum": top_trend_sum,
            "ordinary_day_positive_event_sum": ordinary_positive_sum,
            "latest_two_calendar_years": latest_two_years,
        },
        "bootstrap": bootstrap,
        "gates": gates,
        "failed_gates": failed_gates,
        "stop_codes": stop_codes,
        "tables": {
            "horizons": horizon_table,
            "directions": direction_table,
            "calendar_years": year_table,
            "severity": severity_table,
            "path_efficiency": path_table,
            "volume_ratio": volume_table,
            "time_segment": time_table,
            "market_state": market_table,
            "nearby_threshold_platform": platform_table,
        },
    }


def build_report(
    inputs: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    panel: pd.DataFrame,
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    evaluation: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """构建严格、可审计且不含组合回报冒充的结果对象。"""

    input_hashes: dict[str, Any] = {}
    for key, specification in config["inputs"].items():
        for field in ("path", "metadata_path"):
            relative = specification.get(field)
            if relative:
                path = project_path(relative)
                input_hashes[f"{key}.{field}"] = {
                    "path": relative,
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
    model = cost_model(config)
    return {
        "schema_version": "1.0.0",
        "report_id": "510300_ANCHORED_SPARSE_MEAN_REVERSION_GRID_V1_PHASE_A",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")),
        "status": evaluation["status"],
        "goal_achieved": False,
        "research_stage": "DISCOVERY",
        "return_evaluation": "EVENT_INFORMATION_BUDGET_COMPLETED",
        "portfolio_return_evaluation": "NOT_RUN_PHASE_A",
        "manifest_verification": manifest_verification,
        "input_audit": input_audit,
        "scope": config["scope"],
        "mechanism": {
            "anchor_type": config["anchor"]["type"],
            "anchor_is_fair_value": False,
            "historical_iopv_used": False,
            "grid_role": config["protocol"]["grid_role"],
        },
        "cost_model": {
            "grid_notional_cny": model.notional_cny,
            "effective_commission_rate_per_leg": model.commission_rate_effective_per_leg,
            "base_round_trip_rate": model.base_round_trip_rate,
            "stress_round_trip_rate": model.stress_round_trip_rate,
        },
        "coverage": {
            "minute_feature_rows": int(len(panel)),
            "minute_feature_days": int(panel["date"].nunique()),
            "first_feature_time": panel["trade_time"].min(),
            "last_feature_time": panel["trade_time"].max(),
            "diagnostic_event_rows": int(len(events)),
            "event_outcome_rows": int(len(outcomes)),
        },
        "phase_a_evaluation": evaluation,
        "decision": {
            "phase_b_grid_backtest_allowed": bool(evaluation["phase_a_passed"]),
            "phase_b_grid_backtest_run": False,
            "parameter_rescue_allowed": False,
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
            "next_step": "冻结第二阶段组合协议后独立实现，不得改动第一阶段"
            if evaluation["phase_a_passed"]
            else "停止本分支；不运行网格组合回测，不改参数、窗口、成本、代理或子区间救援",
        },
        "inputs": input_hashes,
        "outputs": {
            "feature_table": config["paths"]["feature_table"],
            "event_table": config["paths"]["event_table"],
            "event_outcome_table": config["paths"]["event_outcome_table"],
            "gate_receipt": config["paths"]["gate_receipt"],
        },
        "official_rules": config["official_rules"],
        "governance": config["governance"],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """生成人工可复核的中文结论报告。"""

    evaluation = report["phase_a_evaluation"]
    metrics = evaluation["metrics"]
    costs = report["cost_model"]
    failed = evaluation["failed_gates"]

    def percent(value: Any) -> str:
        return "NOT_AVAILABLE" if value is None else f"{float(value):.4%}"

    lines = [
        "# 510300 有锚稀疏均值回归网格 V1：第一阶段事件信息预算",
        "",
        f"- 状态：`{report['status']}`",
        "- 研究阶段：`DISCOVERY_ONLY / NO_TRADE`",
        "- 组合回测：未运行；第一阶段失败时明确禁止运行。",
        "- 可交易资产仍只有 `510300.SH` 与 `CASH_CNY`；000300 只作观察锚点。",
        "",
        "## 输入结论",
        "",
        f"- 输入审计：`{report['input_audit']['status']}`。",
        f"- 分钟特征：{report['coverage']['minute_feature_rows']:,} 行，{report['coverage']['minute_feature_days']:,} 个交易日。",
        "- 锚点是除息调整后的指数隐含价格代理，不是历史 IOPV，也不宣称真实公平价值。",
        "- 000300 原始 high/low 异常只登记，不回填；模型使用的同步 open/close 通过硬门。",
        "",
        "## 固定成本",
        "",
        f"- 单层名义金额：{costs['grid_notional_cny']:,.0f} 元。",
        f"- 每腿有效佣金率：{costs['effective_commission_rate_per_leg']:.4%}。",
        f"- 基础完整往返成本：{costs['base_round_trip_rate']:.4%}。",
        f"- 1.5 倍压力成本：{costs['stress_round_trip_rate']:.4%}。",
        "",
        "## 45 分钟主结果",
        "",
        f"- 全部诊断事件：{metrics['diagnostic_event_count']:,}。",
        f"- 流动性冲击且处于候选时段的事件：{metrics['liquidity_shock_candidate_time_event_count']:,}。",
        f"- 达到第一层成本边界且结果完整的事件：{metrics['primary_cost_grid_qualified_completed_event_count']:,}。",
        f"- 基础成本后平均事件收益：{percent(metrics['base_mean_net_event_return'])}。",
        f"- 压力成本后平均事件收益：{percent(metrics['stress_mean_net_event_return'])}。",
        f"- ETF 自身反转贡献：{percent(metrics['mean_etf_reversal'])}。",
        f"- 指数追赶贡献：{percent(metrics['mean_anchor_catchup'])}。",
        f"- 总收敛：{percent(metrics['mean_total_convergence'])}。",
        f"- 交易日移动块 Bootstrap 95% 下界：{percent(evaluation['bootstrap'].get('lower_bound'))}。",
        f"- 正收益日历年数量：{metrics['positive_calendar_year_count']}。",
        f"- 最大单一正年份贡献占比：{percent(metrics['maximum_single_positive_year_contribution_share'])}。",
        "",
        "## 硬门裁决",
        "",
    ]
    for name, passed in evaluation["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}：`{name}`")
    lines.extend(
        [
            "",
            f"未通过门槛：{', '.join(failed) if failed else '无'}。",
            "",
            "## 停止线",
            "",
        ]
    )
    for name, triggered in evaluation["stop_codes"].items():
        lines.append(f"- `{name}`：{triggered}")
    lines.extend(
        [
            "",
            "## 最终决定",
            "",
            report["decision"]["next_step"] + "。",
            "",
            "本报告不是收益承诺、仓位建议、订单或实盘授权。研究、Paper/Shadow、券商连接、持仓、订单和实盘状态保持严格分离。",
            "",
        ]
    )
    return "\n".join(lines)


def feature_output_columns() -> list[str]:
    """冻结分钟特征产物列，避免输出无关原始字段。"""

    return [
        "trade_time",
        "date",
        "minute_label",
        "bar_index",
        "time_segment",
        "etf_open",
        "etf_close",
        "etf_amount",
        "index_open",
        "index_close",
        "previous_etf_close",
        "previous_index_close",
        "cash_dividend_per_share",
        "dividend_adjusted_previous_etf_close",
        "index_implied_price_proxy",
        "raw_log_basis",
        "basis_center_prior20",
        "basis_scale_mad_raw_prior20",
        "basis_scale_tick_floor",
        "basis_scale",
        "centered_log_basis",
        "basis_z",
        "etf_amount_5m",
        "etf_amount_5m_prior20_same_minute_median",
        "etf_amount_5m_ratio",
        "path_efficiency_30m",
        "etf_log_return_5m",
        "index_log_return_5m",
        "relative_log_return_5m",
        "prior_day_realized_volatility",
        "market_state",
        "outcome_only_full_day_path_efficiency",
        "base_round_trip_cost",
        "stress_round_trip_cost",
        "first_grid_band",
        "second_grid_band",
        "feature_ready",
    ]


def blocked_report(
    input_audit: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """输入失败时返回真实阻断状态，不构造事件或收益。"""

    return {
        "schema_version": "1.0.0",
        "report_id": "510300_ANCHORED_SPARSE_MEAN_REVERSION_GRID_V1_PHASE_A",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")),
        "status": "BLOCKED_INPUT_CONTRACT",
        "goal_achieved": False,
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_return_evaluation": "NOT_ALLOWED",
        "manifest_verification": manifest_verification,
        "input_audit": input_audit,
        "decision": {
            "phase_b_grid_backtest_allowed": False,
            "phase_b_grid_backtest_run": False,
            "next_step": "修复或补齐同一冻结输入合同后创建机械修正版；不得替代代理或读取候选收益",
            "live_trading_authorized": False,
        },
        "governance": config["governance"],
    }
