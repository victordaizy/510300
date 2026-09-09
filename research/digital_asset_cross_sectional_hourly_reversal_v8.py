"""15个USD本位永续的小时横截面反转V8。"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.digital_asset_spot_volatility_scaled_trend_v1 import (
    CONSERVATIVE_EVALUATOR_CONFIG,
    atomic_json,
    atomic_parquet,
    atomic_text,
)
from research.qdii_cross_market_discount_reversion_zero_variance_v2 import (
    evaluate_historical_returns_zero_variance_v2,
    load_correction_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "digital_asset_cross_sectional_hourly_reversal_v8.yaml"
FIXED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BCHUSDT",
    "XRPUSDT",
    "EOSUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "ETCUSDT",
    "LINKUSDT",
    "XLMUSDT",
    "ADAUSDT",
    "BNBUSDT",
    "DASHUSDT",
    "ZECUSDT",
    "XTZUSDT",
]


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("小时横截面反转V8配置必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    failures: list[str] = []
    protocol = contract.get("protocol", {})
    if protocol.get("candidate_id") != "DIGITAL_ASSET_CROSS_SECTIONAL_HOURLY_REVERSAL_V8":
        failures.append("candidate_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V18":
        failures.append("parent_protocol_id")
    if contract.get("account", {}).get("initial_capital_cny") != 500000.0:
        failures.append("initial_capital_cny")
    if contract.get("account", {}).get("user_transaction_fee_rate_per_leg") != 0.0001:
        failures.append("user_transaction_fee_rate_per_leg")
    universe = contract.get("universe", {})
    if universe.get("fixed_symbols") != FIXED_SYMBOLS:
        failures.append("fixed_symbols")
    if int(universe.get("bar_interval_hours", 0)) != 1:
        failures.append("bar_interval_hours")
    if int(universe.get("turnover_lookback_hours", 0)) != 168:
        failures.append("turnover_lookback_hours")
    if float(universe.get("maximum_order_fraction_of_prior_median_quote_volume", math.nan)) != 0.001:
        failures.append("maximum_order_fraction")
    signal = contract.get("signal", {})
    for key, expected in {
        "long_count": 3,
        "short_count": 3,
        "long_side_target_gross": 1.0,
        "short_side_target_gross": 1.0,
        "maximum_target_gross_exposure": 2.0,
        "target_net_exposure": 0.0,
    }.items():
        value = signal.get(key)
        if isinstance(expected, int):
            if int(value or 0) != expected:
                failures.append(key)
        elif float(value if value is not None else math.nan) != expected:
            failures.append(key)
    if signal.get("prior_return_formula") != "PRIOR_BAR_CLOSE_DIV_PRIOR_BAR_OPEN_MINUS_1":
        failures.append("prior_return_formula")
    if signal.get("long_selection") != "LOWEST_PRIOR_RETURN_THEN_SYMBOL_ASCENDING":
        failures.append("long_selection")
    if signal.get("short_selection") != "HIGHEST_PRIOR_RETURN_THEN_SYMBOL_ASCENDING":
        failures.append("short_selection")
    if signal.get("long_short_overlap_allowed") is not False:
        failures.append("long_short_overlap_allowed")
    if signal.get("short_selection_excludes_selected_longs") is not True:
        failures.append("short_selection_excludes_selected_longs")
    portfolio = contract.get("portfolio", {})
    if float(portfolio.get("stress_cost_reserve_multiplier", math.nan)) != 0.98:
        failures.append("stress_cost_reserve_multiplier")
    if float(portfolio.get("minimum_trade_notional_usdt", math.nan)) != 10.0:
        failures.append("minimum_trade_notional_usdt")
    if int(portfolio.get("terminal_winddown_calendar_days", 0)) != 7:
        failures.append("terminal_winddown_calendar_days")
    if portfolio.get("terminal_winddown_start") != "2023-12-25 00:00:00":
        failures.append("terminal_winddown_start")
    expected_increments = {
        "BTCUSDT": 0.001,
        "ETHUSDT": 0.001,
        "BCHUSDT": 0.001,
        "XRPUSDT": 0.1,
        "EOSUSDT": 0.1,
        "LTCUSDT": 0.001,
        "TRXUSDT": 1.0,
        "ETCUSDT": 0.01,
        "LINKUSDT": 0.01,
        "XLMUSDT": 1.0,
        "ADAUSDT": 1.0,
        "BNBUSDT": 0.01,
        "DASHUSDT": 0.001,
        "ZECUSDT": 0.001,
        "XTZUSDT": 0.1,
    }
    if portfolio.get("quantity_increment") != expected_increments:
        failures.append("quantity_increment")
    if portfolio.get("capacity_aware_global_proportional_slice") is not True:
        failures.append("global_capacity_slice")
    costs = contract.get("costs", {})
    for key, expected in {
        "user_transaction_fee_rate_per_leg": 0.0001,
        "base_total_perpetual_cost_bps_per_leg": 4.0,
        "stress_total_perpetual_cost_bps_per_leg": 16.0,
        "base_funding_receipt_multiplier": 1.0,
        "base_funding_payment_multiplier": 1.0,
        "stress_funding_receipt_multiplier": 0.75,
        "stress_funding_payment_multiplier": 1.25,
        "base_fx_conversion_spread_bps_per_conversion": 10.0,
        "stress_fx_conversion_spread_bps_per_conversion": 30.0,
    }.items():
        if float(costs.get(key, math.nan)) != expected:
            failures.append(key)
    risk = contract.get("risk", {})
    if float(risk.get("maximum_gross_exposure", math.nan)) != 2.0:
        failures.append("maximum_gross_exposure")
    if float(risk.get("maximum_absolute_net_exposure", math.nan)) != 0.05:
        failures.append("maximum_absolute_net_exposure")
    if float(risk.get("minimum_combined_equity_fraction_of_gross_notional", math.nan)) != 0.05:
        failures.append("maintenance_fraction")
    gates = contract.get("visible_gates", {})
    if float(gates.get("minimum_annualized_net_excess", math.nan)) != 0.40:
        failures.append("minimum_annualized_net_excess")
    if float(gates.get("minimum_strategy_net_sharpe", math.nan)) != 1.50:
        failures.append("minimum_strategy_net_sharpe")
    if int(gates.get("annualization_trading_days", 0)) != 365:
        failures.append("annualization_trading_days")
    if any(bool(value) for value in contract.get("safety", {}).values()):
        failures.append("safety")
    if failures:
        raise ValueError(f"小时横截面反转V8配置被弱化或损坏：{sorted(set(failures))}")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_time(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    converted = pd.to_datetime(result[column], errors="coerce")
    if isinstance(converted.dtype, pd.DatetimeTZDtype):
        converted = converted.dt.tz_localize(None)
    result[column] = converted.astype("datetime64[ns]")
    if result[column].isna().any():
        raise DataContractError(f"{column}含无效时间")
    return result


def _exact_symbol_grid(
    frame: pd.DataFrame,
    *,
    time_column: str,
    symbols: list[str],
    expected: pd.DatetimeIndex,
    label: str,
) -> None:
    if sorted(frame["symbol"].astype(str).unique()) != sorted(symbols):
        raise DataContractError(f"{label}产品集合不正确")
    for symbol in symbols:
        observed = pd.DatetimeIndex(
            frame.loc[frame["symbol"].eq(symbol), time_column]
        ).sort_values().unique()
        if not observed.equals(expected):
            raise DataContractError(f"{label} {symbol}未精确覆盖冻结共同网格")


def load_visible_inputs(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inputs = contract["inputs"]
    status_path = ROOT / inputs["source_status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    forbidden_true = (
        "hourly_return_or_rank_computed",
        "long_or_short_direction_computed",
        "signal_or_target_computed",
        "strategy_or_benchmark_return_computed",
        "sealed_period_2024_onward_read",
        "account_or_authenticated_endpoint_used",
    )
    if status.get("status") != "PASS_INPUT_ACQUISITION_ONLY" or any(
        bool(status.get(key)) for key in forbidden_true
    ):
        raise DataContractError("V8原始采集状态不是严格的输入采集通过")
    paths = {
        "perpetual": ROOT / inputs["perpetual_1h"],
        "funding": ROOT / inputs["funding_rates"],
        "fx": ROOT / inputs["visible_fx"],
        "benchmark": ROOT / inputs["visible_benchmark"],
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少V8输入：{key}={path}")
    perpetual = _normalize_time(pd.read_parquet(paths["perpetual"]), "open_time")
    funding = _normalize_time(pd.read_parquet(paths["funding"]), "funding_time")
    fx = pd.read_parquet(paths["fx"])
    benchmark = pd.read_parquet(paths["benchmark"])
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce").astype("datetime64[ns]").dt.normalize()
    benchmark["date"] = (
        pd.to_datetime(benchmark["date"], errors="coerce")
        .astype("datetime64[ns]")
        .dt.normalize()
    )
    if fx["date"].isna().any() or benchmark["date"].isna().any():
        raise DataContractError("V8汇率或基准含无效日期")
    symbols = list(contract["universe"]["fixed_symbols"])
    source_start = pd.Timestamp(contract["historical_partition"]["acquisition_start"])
    visible_end = pd.Timestamp(f"{contract['historical_partition']['visible_end']} 23:00:00")
    funding_end = pd.Timestamp(f"{contract['historical_partition']['visible_end']} 16:00:00")
    _exact_symbol_grid(
        perpetual,
        time_column="open_time",
        symbols=symbols,
        expected=pd.date_range(source_start, visible_end, freq="1h"),
        label="V8永续小时K线",
    )
    _exact_symbol_grid(
        funding,
        time_column="funding_time",
        symbols=symbols,
        expected=pd.date_range(source_start, funding_end, freq="8h"),
        label="V8资金费率",
    )
    price_columns = ["open", "high", "low", "close"]
    required_numeric = price_columns + ["quote_volume"]
    numeric_values = perpetual[required_numeric].apply(pd.to_numeric, errors="coerce")
    if numeric_values.isna().any().any() or not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
        raise DataContractError("V8永续小时K线存在缺失或非有限数值")
    if not (numeric_values[price_columns] > 0.0).all().all() or not (
        numeric_values["quote_volume"] > 0.0
    ).all():
        raise DataContractError("V8执行价格或成交额非正")
    if not (
        numeric_values["high"] >= numeric_values[["open", "close"]].max(axis=1)
    ).all() or not (
        numeric_values["low"] <= numeric_values[["open", "close"]].min(axis=1)
    ).all():
        raise DataContractError("V8小时高低价关系不成立")
    rates = pd.to_numeric(funding["funding_rate"], errors="coerce")
    if rates.isna().any() or not np.isfinite(rates.to_numpy(dtype=float)).all():
        raise DataContractError("V8资金费率含缺失或非有限值")
    audit = {
        "source_status": status["status"],
        "perpetual_rows": int(len(perpetual)),
        "funding_rows": int(len(funding)),
        "fx_rows": int(len(fx)),
        "benchmark_rows": int(len(benchmark)),
        "symbol_count": int(len(symbols)),
        "input_sha256": {key: _sha256(path) for key, path in paths.items()},
        "hourly_grid_complete": True,
        "funding_grid_complete": True,
        "future_price_or_strategy_return_read": True,
        "sealed_replication_read": False,
        "authenticated_endpoint_used": False,
    }
    return perpetual, funding, fx, benchmark, audit


def build_hourly_targets(
    perpetual: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """只用上一根完整小时K线生成下一小时执行目标。"""

    symbols = sorted(contract["universe"]["fixed_symbols"])
    data = _normalize_time(perpetual, "open_time")
    data["open"] = pd.to_numeric(data["open"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["prior_return"] = data["close"] / data["open"] - 1.0
    data["execution_time"] = data["open_time"] + pd.Timedelta(hours=1)
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    end = pd.Timestamp(f"{contract['historical_partition']['visible_end']} 23:00:00")
    winddown = pd.Timestamp(contract["portfolio"]["terminal_winddown_start"])
    eligible = data.loc[
        data["execution_time"].between(start, end)
        & data["execution_time"].lt(winddown),
        ["execution_time", "symbol", "prior_return"],
    ]
    matrix = (
        eligible.pivot(index="execution_time", columns="symbol", values="prior_return")
        .reindex(columns=symbols)
        .sort_index()
    )
    expected_times = pd.date_range(start, winddown - pd.Timedelta(hours=1), freq="1h")
    if not matrix.index.equals(expected_times) or matrix.isna().any().any():
        raise DataContractError("V8上一小时收益未精确覆盖全部执行时点和标的")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataContractError("V8上一小时收益含非有限值")
    long_count = int(contract["signal"]["long_count"])
    short_count = int(contract["signal"]["short_count"])
    low_order = np.argsort(values, axis=1, kind="stable")
    high_order = np.argsort(-values, axis=1, kind="stable")
    low = low_order[:, :long_count]
    short_scores = -values.copy()
    np.put_along_axis(short_scores, low, np.inf, axis=1)
    high = np.argsort(short_scores, axis=1, kind="stable")[:, :short_count]
    if any(set(a).intersection(set(b)) for a, b in zip(low.tolist(), high.tolist())):
        raise DataContractError("V8多空选择发生重叠")
    selected = np.concatenate([low, high], axis=1)
    symbol_array = np.asarray(symbols, dtype=object)
    times = matrix.index.to_numpy(dtype="datetime64[ns]")
    target_weights = np.tile(
        np.concatenate(
            [
                np.full(long_count, 1.0 / long_count),
                np.full(short_count, -1.0 / short_count),
            ]
        ),
        len(matrix),
    )
    targets = pd.DataFrame(
        {
            "signal_time": np.repeat(times - np.timedelta64(1, "h"), long_count + short_count),
            "execution_time": np.repeat(times, long_count + short_count),
            "symbol": symbol_array[selected.reshape(-1)],
            "target_weight": target_weights,
        }
    ).sort_values(["execution_time", "symbol"]).reset_index(drop=True)
    long_rank = np.empty_like(low_order)
    short_rank = np.empty_like(high_order)
    ranks = np.broadcast_to(np.arange(1, len(symbols) + 1), low_order.shape)
    np.put_along_axis(long_rank, low_order, ranks, axis=1)
    np.put_along_axis(short_rank, high_order, ranks, axis=1)
    long_selected = np.zeros(values.shape, dtype=bool)
    short_selected = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(long_selected, low, True, axis=1)
    np.put_along_axis(short_selected, high, True, axis=1)
    selection = np.full(values.shape, "NONE", dtype=object)
    selection[long_selected] = "LONG"
    selection[short_selected] = "SHORT"
    features = pd.DataFrame(
        {
            "signal_time": np.repeat(times - np.timedelta64(1, "h"), len(symbols)),
            "execution_time": np.repeat(times, len(symbols)),
            "symbol": np.tile(symbol_array, len(matrix)),
            "prior_bar_return": values.reshape(-1),
            "long_rank": long_rank.reshape(-1),
            "short_rank": short_rank.reshape(-1),
            "selection": selection.reshape(-1),
        }
    )
    grouped = targets.groupby("execution_time", observed=True)
    long_sums = grouped["target_weight"].apply(lambda item: float(item[item > 0.0].sum()))
    short_sums = grouped["target_weight"].apply(lambda item: float(item[item < 0.0].sum()))
    if not np.allclose(long_sums, 1.0) or not np.allclose(short_sums, -1.0):
        raise DataContractError("V8多空目标权重未分别归一至正负1")
    audit = {
        "hourly_signal_count": int(len(matrix)),
        "target_row_count": int(len(targets)),
        "feature_row_count": int(len(features)),
        "first_signal_time": pd.Timestamp(features["signal_time"].min()).isoformat(),
        "last_signal_time": pd.Timestamp(features["signal_time"].max()).isoformat(),
        "first_execution_time": pd.Timestamp(targets["execution_time"].min()).isoformat(),
        "last_execution_time": pd.Timestamp(targets["execution_time"].max()).isoformat(),
        "strict_execution_lag_hours": 1,
        "long_count_each_hour": long_count,
        "short_count_each_hour": short_count,
        "tie_break": "SYMBOL_ASCENDING",
        "current_execution_bar_return_used": False,
        "future_price_or_return_used_in_signal": False,
        "sealed_replication_read": False,
    }
    return targets, features, audit


def _to_datetime64_ns(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    converted = pd.to_datetime(values, errors="coerce")
    if isinstance(converted, pd.Series):
        if isinstance(converted.dtype, pd.DatetimeTZDtype):
            converted = converted.dt.tz_localize(None)
        return converted.astype("datetime64[ns]")
    if converted.tz is not None:
        converted = converted.tz_localize(None)
    return pd.Series(converted.to_numpy(dtype="datetime64[ns]"))


def daily_context(
    fx: pd.DataFrame,
    benchmark: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    calendar = pd.DataFrame(
        {"date": pd.date_range(start.normalize() - pd.Timedelta(days=1), end.normalize(), freq="D")}
    )
    calendar["date"] = _to_datetime64_ns(calendar["date"])
    fx_data = fx[["date", "cny_per_usd"]].dropna().copy()
    fx_data["date"] = _to_datetime64_ns(fx_data["date"])
    fx_data = fx_data.sort_values("date").rename(columns={"date": "fx_source_date"})
    context = pd.merge_asof(
        calendar.sort_values("date"),
        fx_data,
        left_on="date",
        right_on="fx_source_date",
        direction="backward",
        allow_exact_matches=False,
    )
    benchmark_data = benchmark[["date", "close"]].dropna().copy()
    benchmark_data["date"] = _to_datetime64_ns(benchmark_data["date"])
    benchmark_data = benchmark_data.sort_values("date").rename(
        columns={"date": "benchmark_source_date", "close": "benchmark_close"}
    )
    context = pd.merge_asof(
        context.sort_values("date"),
        benchmark_data,
        left_on="date",
        right_on="benchmark_source_date",
        direction="backward",
        allow_exact_matches=True,
    )
    numeric = context[["cny_per_usd", "benchmark_close"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not (numeric > 0.0).all().all():
        raise DataContractError("V8日度汇率或H00300映射缺失或非正")
    context["cny_per_usd"] = numeric["cny_per_usd"]
    context["benchmark_close"] = numeric["benchmark_close"]
    context["benchmark_total_return"] = context["benchmark_close"].pct_change(fill_method=None)
    context["fx_age_calendar_days"] = (context["date"] - context["fx_source_date"]).dt.days
    return context.loc[
        context["date"].between(start.normalize(), end.normalize())
    ].reset_index(drop=True)


def _market_panels(
    perpetual: pd.DataFrame,
    symbols: list[str],
    lookback: int,
) -> dict[str, pd.DataFrame]:
    data = _normalize_time(perpetual, "open_time")
    if data.duplicated(["open_time", "symbol"]).any():
        raise DataContractError("V8永续小时K线存在重复产品时点")
    panels: dict[str, pd.DataFrame] = {}
    for column in ("open", "high", "low", "close", "quote_volume"):
        panel = (
            data.pivot(index="open_time", columns="symbol", values=column)
            .reindex(columns=symbols)
            .sort_index()
            .apply(pd.to_numeric, errors="coerce")
        )
        panels[column] = panel
    panels["prior_median_quote_volume"] = (
        panels["quote_volume"].shift(1).rolling(lookback, min_periods=lookback).median()
    )
    return panels


def _truncate_toward_zero(values: np.ndarray, increments: np.ndarray) -> np.ndarray:
    scaled = np.abs(values) / increments
    return np.sign(values) * np.floor(scaled + 1e-9) * increments


def _cost_rate(contract: dict[str, Any], scenario: str) -> float:
    return float(contract["costs"][f"{scenario}_total_perpetual_cost_bps_per_leg"]) / 10000.0


def _fx_spread(contract: dict[str, Any], scenario: str) -> float:
    return float(contract["costs"][f"{scenario}_fx_conversion_spread_bps_per_conversion"]) / 10000.0


def run_portfolio_backtest(
    targets: pd.DataFrame,
    perpetual: pd.DataFrame,
    funding: pd.DataFrame,
    fx: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    end = pd.Timestamp(f"{contract['historical_partition']['visible_end']} 23:00:00")
    times = pd.date_range(start, end, freq="1h")
    symbols = list(contract["universe"]["fixed_symbols"])
    symbol_count = len(symbols)
    panels = _market_panels(
        perpetual,
        symbols,
        int(contract["universe"]["turnover_lookback_hours"]),
    )
    selected: dict[str, np.ndarray] = {}
    for name in ("open", "high", "low", "close", "prior_median_quote_volume"):
        frame = panels[name].reindex(times)
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataContractError(f"V8可见期{name}矩阵存在缺口或非有限值")
        selected[name] = values
    if not (selected["open"] > 0.0).all() or not (
        selected["prior_median_quote_volume"] > 0.0
    ).all():
        raise DataContractError("V8可见期执行价或滞后中位成交额非正")
    next_open = np.empty_like(selected["open"])
    next_open[:-1] = selected["open"][1:]
    next_open[-1] = selected["close"][-1]

    target_matrix = (
        targets.pivot(index="execution_time", columns="symbol", values="target_weight")
        .reindex(index=times, columns=symbols)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    winddown = pd.Timestamp(contract["portfolio"]["terminal_winddown_start"])
    pre_winddown = times < winddown
    if not np.allclose(np.clip(target_matrix[pre_winddown], 0.0, None).sum(axis=1), 1.0):
        raise DataContractError("V8每小时多头目标毛敞口不为1")
    if not np.allclose(np.clip(-target_matrix[pre_winddown], 0.0, None).sum(axis=1), 1.0):
        raise DataContractError("V8每小时空头目标毛敞口不为1")
    if not np.allclose(target_matrix[~pre_winddown], 0.0):
        raise DataContractError("V8末端减仓期仍存在新目标")

    funding_data = _normalize_time(funding, "funding_time")
    funding_rate_panel = (
        funding_data.pivot(index="funding_time", columns="symbol", values="funding_rate")
        .reindex(columns=symbols)
        .sort_index()
    )
    mark_panel = (
        funding_data.pivot(index="funding_time", columns="symbol", values="mark_price")
        .reindex(columns=symbols)
        .sort_index()
    )
    settlement_times = times + pd.Timedelta(hours=1)
    funding_rates = funding_rate_panel.reindex(settlement_times).to_numpy(dtype=float)
    funding_marks = mark_panel.reindex(settlement_times).to_numpy(dtype=float)
    context = daily_context(fx, benchmark, start, end)
    first_fx = float(context.iloc[0]["cny_per_usd"])
    initial_cny = float(contract["account"]["initial_capital_cny"])
    nav = np.asarray(
        [
            initial_cny / first_fx * (1.0 - _fx_spread(contract, "base")),
            initial_cny / first_fx * (1.0 - _fx_spread(contract, "stress")),
        ],
        dtype=float,
    )
    positions = np.zeros(symbol_count, dtype=float)
    increments = np.asarray(
        [float(contract["portfolio"]["quantity_increment"][symbol]) for symbol in symbols]
    )
    reserve = float(contract["portfolio"]["stress_cost_reserve_multiplier"])
    capacity_fraction = float(
        contract["universe"]["maximum_order_fraction_of_prior_median_quote_volume"]
    )
    minimum_trade = float(contract["portfolio"]["minimum_trade_notional_usdt"])
    maximum_gross = maximum_net = maximum_order_fraction = 0.0
    maximum_intrabar_loss_fraction = 0.0
    global_slice_count = minimum_trade_skip_count = 0
    entry_count = exit_count = flip_count = resize_count = winddown_trade_count = 0
    total_transaction_cost = np.zeros(2, dtype=float)
    total_funding = np.zeros(2, dtype=float)
    trade_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    cash_hourly_rate = (1.0 + float(contract["account"]["cash_annual_rate"])) ** (1.0 / 8760.0) - 1.0

    for time_index, timestamp in enumerate(times):
        open_prices = selected["open"][time_index]
        target_weights = target_matrix[time_index]
        target_notional_basis = float(nav.min()) * reserve
        desired_quantities = _truncate_toward_zero(
            target_weights * target_notional_basis / open_prices,
            increments,
        )
        full_delta = desired_quantities - positions
        full_order_notional = np.abs(full_delta) * open_prices
        capacities = capacity_fraction * selected["prior_median_quote_volume"][time_index]
        active_orders = full_order_notional > 1e-12
        scale = 1.0
        if active_orders.any():
            scale = min(1.0, float(np.min(capacities[active_orders] / full_order_notional[active_orders])))
        if scale < 1.0 - 1e-12:
            global_slice_count += 1
        executed_delta = _truncate_toward_zero(scale * full_delta, increments)
        order_notional = np.abs(executed_delta) * open_prices
        terminal_reduction = timestamp >= winddown
        small_regular = (order_notional > 1e-12) & (order_notional < minimum_trade)
        if not terminal_reduction:
            minimum_trade_skip_count += int(small_regular.sum())
            executed_delta[small_regular] = 0.0
            order_notional[small_regular] = 0.0
        new_positions = np.rint((positions + executed_delta) / increments) * increments
        executed_delta = new_positions - positions
        order_notional = np.abs(executed_delta) * open_prices
        order_fractions = np.divide(
            order_notional,
            selected["prior_median_quote_volume"][time_index],
            out=np.zeros_like(order_notional),
            where=selected["prior_median_quote_volume"][time_index] > 0.0,
        )
        maximum_order_fraction = max(maximum_order_fraction, float(order_fractions.max()))
        if (order_fractions > capacity_fraction + 1e-10).any():
            raise DataContractError(f"{timestamp} V8全组合切片后订单仍越过冻结容量")
        traded = order_notional > 1e-12
        if traded.any():
            costs = np.asarray(
                [
                    float(order_notional.sum()) * _cost_rate(contract, "base"),
                    float(order_notional.sum()) * _cost_rate(contract, "stress"),
                ]
            )
            nav -= costs
            total_transaction_cost += costs
            for index in np.flatnonzero(traded):
                before = float(positions[index])
                after = float(new_positions[index])
                if terminal_reduction:
                    reason = "TERMINAL_WINDDOWN"
                    winddown_trade_count += 1
                elif abs(before) <= 1e-12 and abs(after) > 1e-12:
                    reason = "ENTRY"
                    entry_count += 1
                elif abs(before) > 1e-12 and abs(after) <= 1e-12:
                    reason = "EXIT"
                    exit_count += 1
                elif before * after < 0.0:
                    reason = "FLIP"
                    flip_count += 1
                else:
                    reason = "RESIZE"
                    resize_count += 1
                trade_rows.append(
                    {
                        "trade_time": timestamp,
                        "signal_time": timestamp - pd.Timedelta(hours=1),
                        "symbol": symbols[index],
                        "reason": reason,
                        "side": "BUY" if executed_delta[index] > 0.0 else "SELL",
                        "quantity": abs(float(executed_delta[index])),
                        "signed_quantity_change": float(executed_delta[index]),
                        "execution_open": float(open_prices[index]),
                        "order_notional_usdt": float(order_notional[index]),
                        "prior_median_quote_volume_usdt": float(
                            selected["prior_median_quote_volume"][time_index, index]
                        ),
                        "capacity_fraction": float(order_fractions[index]),
                        "global_slice_scale": float(scale),
                        "target_weight": float(target_weights[index]),
                        "position_after": after,
                    }
                )
        positions = new_positions
        if float(nav.min()) <= 0.0:
            raise DataContractError(f"{timestamp} V8交易成本后净值非正")

        gross_notional = float(np.sum(np.abs(positions) * open_prices))
        net_notional = float(np.sum(positions * open_prices))
        minimum_nav = float(nav.min())
        gross_exposure = gross_notional / minimum_nav
        net_exposure = net_notional / minimum_nav
        maximum_gross = max(maximum_gross, gross_exposure)
        maximum_net = max(maximum_net, abs(net_exposure))
        if gross_exposure > float(contract["risk"]["maximum_gross_exposure"]) + 1e-10:
            raise DataContractError(f"{timestamp} V8实际毛敞口超过冻结上限")
        if abs(net_exposure) > float(contract["risk"]["maximum_absolute_net_exposure"]) + 1e-10:
            raise DataContractError(f"{timestamp} V8实际净敞口超过冻结上限")
        long_worst = np.where(
            positions > 0.0,
            positions * (selected["low"][time_index] - open_prices),
            0.0,
        )
        short_worst = np.where(
            positions < 0.0,
            positions * (selected["high"][time_index] - open_prices),
            0.0,
        )
        worst_intrabar_pnl = float(long_worst.sum() + short_worst.sum())
        worst_equity = minimum_nav + worst_intrabar_pnl
        maintenance = (
            float(contract["risk"]["minimum_combined_equity_fraction_of_gross_notional"])
            * gross_notional
        )
        maximum_intrabar_loss_fraction = max(
            maximum_intrabar_loss_fraction,
            -min(0.0, worst_intrabar_pnl) / minimum_nav,
        )
        if worst_equity <= maintenance:
            raise DataContractError(f"{timestamp} V8合并权益代理触发保证金失败")

        price_pnl = float(np.sum(positions * (next_open[time_index] - open_prices)))
        rates = funding_rates[time_index]
        marks = funding_marks[time_index]
        settlement = np.isfinite(rates)
        funding_base = funding_stress = 0.0
        if settlement.any():
            valid_marks = np.where(
                np.isfinite(marks) & (marks > 0.0), marks, next_open[time_index]
            )
            payments = -positions[settlement] * valid_marks[settlement] * rates[settlement]
            funding_base = float(
                np.where(
                    payments >= 0.0,
                    payments * float(contract["costs"]["base_funding_receipt_multiplier"]),
                    payments * float(contract["costs"]["base_funding_payment_multiplier"]),
                ).sum()
            )
            funding_stress = float(
                np.where(
                    payments >= 0.0,
                    payments * float(contract["costs"]["stress_funding_receipt_multiplier"]),
                    payments * float(contract["costs"]["stress_funding_payment_multiplier"]),
                ).sum()
            )
        cash_interest = np.zeros(2, dtype=float)
        if not np.any(np.abs(positions) > 1e-12):
            cash_interest = nav * cash_hourly_rate
        nav += np.asarray([price_pnl + funding_base, price_pnl + funding_stress]) + cash_interest
        total_funding += np.asarray([funding_base, funding_stress])
        if float(nav.min()) <= 0.0:
            raise DataContractError(f"{timestamp} V8区间结算后净值非正")
        interval_rows.append(
            {
                "interval_open_time": timestamp,
                "interval_end_time": min(timestamp + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1)),
                "base_nav_usdt": float(nav[0]),
                "stress_nav_usdt": float(nav[1]),
                "price_pnl_usdt": price_pnl,
                "base_funding_usdt": funding_base,
                "stress_funding_usdt": funding_stress,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "capacity_pass": True,
                "active_position_count": int(np.count_nonzero(np.abs(positions) > 1e-12)),
                "global_slice_scale": float(scale),
            }
        )

    terminal_count = int(np.count_nonzero(np.abs(positions) > 1e-12))
    if terminal_count:
        raise DataContractError("V8末端七日容量减仓后仍有非零永续持仓")
    intervals = pd.DataFrame(interval_rows)
    trades = pd.DataFrame(
        trade_rows,
        columns=[
            "trade_time",
            "signal_time",
            "symbol",
            "reason",
            "side",
            "quantity",
            "signed_quantity_change",
            "execution_open",
            "order_notional_usdt",
            "prior_median_quote_volume_usdt",
            "capacity_fraction",
            "global_slice_scale",
            "target_weight",
            "position_after",
        ],
    )
    daily_last = (
        intervals.assign(
            trade_date=pd.to_datetime(intervals["interval_open_time"]).astype("datetime64[ns]").dt.normalize()
        )
        .groupby("trade_date", observed=True, as_index=False)
        .last()
    )
    if len(daily_last) != len(context):
        raise DataContractError("V8日度净值与评价日历不一致")
    daily = context.merge(
        daily_last[
            [
                "trade_date",
                "base_nav_usdt",
                "stress_nav_usdt",
                "gross_exposure",
                "net_exposure",
                "capacity_pass",
            ]
        ],
        left_on="date",
        right_on="trade_date",
        how="left",
        validate="one_to_one",
    )
    if daily[["base_nav_usdt", "stress_nav_usdt"]].isna().any().any():
        raise DataContractError("V8日度净值存在缺口")
    daily["base_nav_cny"] = daily["base_nav_usdt"] * daily["cny_per_usd"]
    daily["stress_nav_cny"] = daily["stress_nav_usdt"] * daily["cny_per_usd"]
    daily.loc[daily.index[-1], "base_nav_cny"] *= 1.0 - _fx_spread(contract, "base")
    daily.loc[daily.index[-1], "stress_nav_cny"] *= 1.0 - _fx_spread(contract, "stress")
    prior_base = daily["base_nav_cny"].shift(1, fill_value=initial_cny)
    prior_stress = daily["stress_nav_cny"].shift(1, fill_value=initial_cny)
    daily["strategy_base_net_return"] = daily["base_nav_cny"] / prior_base - 1.0
    daily["strategy_stress_net_return"] = daily["stress_nav_cny"] / prior_stress - 1.0
    daily["quality_complete"] = True
    daily["capacity_pass"] = daily["capacity_pass"].astype(bool)
    daily.drop(columns=["trade_date"], inplace=True)
    daily.rename(columns={"date": "trade_date"}, inplace=True)
    daily = daily[
        [
            "trade_date",
            "strategy_base_net_return",
            "strategy_stress_net_return",
            "benchmark_total_return",
            "gross_exposure",
            "net_exposure",
            "quality_complete",
            "capacity_pass",
            "base_nav_cny",
            "stress_nav_cny",
            "cny_per_usd",
            "fx_source_date",
            "benchmark_source_date",
        ]
    ]
    portfolio_audit = {
        "initial_capital_cny": initial_cny,
        "final_base_nav_cny": float(daily.iloc[-1]["base_nav_cny"]),
        "final_stress_nav_cny": float(daily.iloc[-1]["stress_nav_cny"]),
        "entry_trade_count": entry_count,
        "exit_trade_count": exit_count,
        "flip_trade_count": flip_count,
        "resize_trade_count": resize_count,
        "terminal_winddown_trade_count": winddown_trade_count,
        "global_capacity_sliced_hour_count": global_slice_count,
        "minimum_trade_skipped_order_count": minimum_trade_skip_count,
        "maximum_order_capacity_fraction": maximum_order_fraction,
        "maximum_gross_exposure": maximum_gross,
        "maximum_absolute_net_exposure": maximum_net,
        "maximum_intrabar_combined_loss_fraction": maximum_intrabar_loss_fraction,
        "base_total_funding_usdt": float(total_funding[0]),
        "stress_total_funding_usdt": float(total_funding[1]),
        "base_total_transaction_cost_usdt": float(total_transaction_cost[0]),
        "stress_total_transaction_cost_usdt": float(total_transaction_cost[1]),
        "terminal_position_count": terminal_count,
        "global_proportional_slice_used": True,
        "sealed_replication_read": False,
    }
    context_audit = {
        "rows": int(len(context)),
        "maximum_fx_age_calendar_days": int(context["fx_age_calendar_days"].max()),
        "strictly_lagged_fx": bool((context["fx_source_date"] < context["date"]).all()),
        "benchmark_first_return_uses_prior_calendar_day_context": True,
    }
    return daily, trades, portfolio_audit, context_audit


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["evaluation"]["metrics"]
    gates = report["evaluation"]["gates"]
    return "\n".join(
        [
            "# 多币种小时横截面反转V8可见期结果",
            "",
            f"状态：`{report['status']}`",
            "",
            f"- H00300全收益CAGR：{metrics['benchmark_total_return_cagr']:.2%}。",
            f"- 基础/压力策略净CAGR：{metrics['strategy_base_net_cagr']:.2%}/{metrics['strategy_stress_net_cagr']:.2%}。",
            f"- 基础/压力年化净超额：{metrics['base_annualized_excess']:.2%}/{metrics['stress_annualized_excess']:.2%}。",
            f"- 基础/压力净夏普：{metrics['base_strategy_net_sharpe']}/{metrics['stress_strategy_net_sharpe']}。",
            f"- 基础/压力40个百分点门：{gates['base_annualized_excess_at_least_40pct']}/{gates['stress_annualized_excess_at_least_40pct']}。",
            f"- 基础/压力1.50夏普门：{gates['base_strategy_sharpe_at_least_1_5']}/{gates['stress_strategy_sharpe_at_least_1_5']}。",
            f"- 全部门通过：{report['evaluation']['all_visible_gates_pass']}。",
            "",
            f"{report['decision']['next_step']}。",
            "",
            "Paper、Shadow、订单、交易所账户连接和实盘均未打开。",
            "",
        ]
    )


def run_visible(
    contract: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    validate_contract(contract)
    perpetual, funding, fx, benchmark, input_audit = load_visible_inputs(contract)
    targets, features, signal_audit = build_hourly_targets(perpetual, contract)
    daily, trades, portfolio_audit, context_audit = run_portfolio_backtest(
        targets, perpetual, funding, fx, benchmark, contract
    )
    correction = load_correction_contract(CONSERVATIVE_EVALUATOR_CONFIG)
    evaluation = evaluate_historical_returns_zero_variance_v2(
        daily,
        contract,
        correction,
        bootstrap_repetitions_override=bootstrap_repetitions_override,
    )
    passed = bool(evaluation["all_visible_gates_pass"])
    report = {
        "schema_version": "1.0.0",
        "report_id": "DIGITAL_ASSET_CROSS_SECTIONAL_HOURLY_REVERSAL_V8_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": (
            "VISIBLE_PASS_FORMULA_FAMILY_REPLICATION_AUTHORIZED_NOT_OPENED"
            if passed
            else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
        ),
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {
            "start": contract["historical_partition"]["visible_start"],
            "end": contract["historical_partition"]["visible_end"],
        },
        "objective": {
            "initial_capital_cny": contract["account"]["initial_capital_cny"],
            "minimum_annualized_net_excess": contract["visible_gates"]["minimum_annualized_net_excess"],
            "minimum_strategy_net_sharpe": contract["visible_gates"]["minimum_strategy_net_sharpe"],
            "user_transaction_fee_rate_per_leg": contract["account"]["user_transaction_fee_rate_per_leg"],
        },
        "manifest_verification": manifest_verification,
        "data_audit": {
            "inputs": input_audit,
            "signal": signal_audit,
            "portfolio": portfolio_audit,
            "daily_context": context_audit,
            "signal_feature_rows": int(len(features)),
        },
        "evaluation": evaluation,
        "historical_evidence_limits": contract["historical_evidence_limits"],
        "decision": {
            "formula_family_replication_conditionally_eligible": passed,
            "sealed_replication_authorized": passed,
            "sealed_replication_open": False,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "paper_or_live_authorized": False,
            "next_step": (
                "可见期全部通过；另行核对复验边界后才可打开2024年起的封存公式族复验，当前仍未打开"
                if passed
                else "冻结拒绝V8，不修改1小时反转、15标的、多空各3、2倍总毛敞口、成本、资金费率或容量门救回"
            ),
        },
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    outputs = contract["outputs"]
    atomic_parquet(ROOT / outputs["visible_daily_returns"], daily)
    atomic_parquet(ROOT / outputs["visible_targets"], targets)
    atomic_parquet(ROOT / outputs["visible_trades"], trades)
    atomic_json(ROOT / outputs["input_audit_json"], report["data_audit"])
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
