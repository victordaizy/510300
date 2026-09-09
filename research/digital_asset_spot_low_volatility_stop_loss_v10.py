"""半年低波动选币并实施月内止损的现货V10。"""

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
from research.digital_asset_cross_sectional_hourly_reversal_v8 import (
    FIXED_SYMBOLS,
    _fx_spread,
    _normalize_time,
    _sha256,
    _truncate_toward_zero,
    daily_context,
)
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
CONFIG = ROOT / "config" / "digital_asset_spot_low_volatility_stop_loss_v10.yaml"


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("现货低波动V10配置必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    failures: list[str] = []
    protocol = contract.get("protocol", {})
    if protocol.get("candidate_id") != "DIGITAL_ASSET_SPOT_LOW_VOLATILITY_STOP_LOSS_V10":
        failures.append("candidate_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V20":
        failures.append("parent_protocol_id")
    account = contract.get("account", {})
    if float(account.get("initial_capital_cny", math.nan)) != 500000.0:
        failures.append("initial_capital_cny")
    if float(account.get("user_transaction_fee_rate_per_leg", math.nan)) != 0.0001:
        failures.append("user_transaction_fee_rate_per_leg")
    universe = contract.get("universe", {})
    if universe.get("fixed_symbols") != FIXED_SYMBOLS:
        failures.append("fixed_symbols")
    if int(universe.get("turnover_lookback_observed_venue_hours", 0)) != 168:
        failures.append("turnover_lookback")
    if float(universe.get("maximum_order_fraction_of_prior_median_quote_volume", math.nan)) != 0.001:
        failures.append("maximum_order_fraction")
    signal = contract.get("signal", {})
    if signal.get("daily_return_formula") != "LOG_23UTC_CLOSE_DIV_PRIOR_23UTC_CLOSE":
        failures.append("daily_return_formula")
    if int(signal.get("volatility_lookback_calendar_days", 0)) != 180:
        failures.append("volatility_lookback")
    if int(signal.get("volatility_standard_deviation_ddof", -1)) != 1:
        failures.append("volatility_ddof")
    if int(signal.get("selected_count", 0)) != 3:
        failures.append("selected_count")
    if signal.get("selection_execution_months") != [3, 9]:
        failures.append("selection_execution_months")
    if float(signal.get("target_long_gross", math.nan)) != 1.0:
        failures.append("target_long_gross")
    if signal.get("short_position_allowed") is not False or signal.get("derivative_position_allowed") is not False:
        failures.append("long_only")
    stop = contract.get("stop_loss", {})
    if float(stop.get("threshold", math.nan)) != 0.05:
        failures.append("stop_threshold")
    if stop.get("stopped_slot_holds_cash") is not True:
        failures.append("stopped_slot_holds_cash")
    portfolio = contract.get("portfolio", {})
    if float(portfolio.get("stress_cost_reserve_multiplier", math.nan)) != 0.98:
        failures.append("reserve")
    if float(portfolio.get("slot_target_notional_fraction", math.nan)) != 1.0 / 3.0:
        failures.append("slot_fraction")
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
    costs = contract.get("costs", {})
    for key, expected in {
        "user_transaction_fee_rate_per_leg": 0.0001,
        "base_total_spot_cost_bps_per_leg": 4.0,
        "stress_total_spot_cost_bps_per_leg": 16.0,
        "base_fx_conversion_spread_bps_per_conversion": 10.0,
        "stress_fx_conversion_spread_bps_per_conversion": 30.0,
    }.items():
        if float(costs.get(key, math.nan)) != expected:
            failures.append(key)
    risk = contract.get("risk", {})
    if float(risk.get("maximum_gross_exposure", math.nan)) != 1.0:
        failures.append("maximum_gross_exposure")
    if any(bool(risk.get(key)) for key in ("account_borrowing_allowed", "account_margin_allowed", "short_sale_allowed", "derivative_position_allowed")):
        failures.append("risk_long_only")
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
        raise ValueError(f"现货低波动V10配置被弱化或损坏：{sorted(set(failures))}")


def _spot_cost_rate(contract: dict[str, Any], scenario: str) -> float:
    return float(contract["costs"][f"{scenario}_total_spot_cost_bps_per_leg"]) / 10000.0


def selection_execution_times(contract: dict[str, Any]) -> pd.DatetimeIndex:
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    end = pd.Timestamp(contract["portfolio"]["terminal_winddown_start"]) - pd.Timedelta(days=1)
    months = set(int(value) for value in contract["signal"]["selection_execution_months"])
    day = int(contract["signal"]["selection_execution_day"])
    hour = int(contract["signal"]["selection_execution_hour"])
    calendar = pd.date_range(start.normalize(), end.normalize(), freq="D")
    selected = [
        timestamp + pd.Timedelta(hours=hour)
        for timestamp in calendar
        if timestamp.month in months and timestamp.day == day
    ]
    return pd.DatetimeIndex(selected)


def load_visible_inputs(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inputs = contract["inputs"]
    status_path = ROOT / inputs["source_status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    forbidden = (
        "basis_value_or_rank_computed",
        "long_or_short_direction_computed",
        "signal_or_target_computed",
        "strategy_or_benchmark_return_computed",
        "sealed_period_2024_onward_read",
        "account_or_authenticated_endpoint_used",
    )
    if status.get("status") != "PASS_INPUT_ACQUISITION_ONLY" or any(
        bool(status.get(key)) for key in forbidden
    ):
        raise DataContractError("V10复用的现货采集状态不合格")
    paths = {
        "spot_1h": ROOT / inputs["spot_1h"],
        "fx": ROOT / inputs["visible_fx"],
        "benchmark": ROOT / inputs["visible_benchmark"],
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少V10输入：{key}={path}")
        receipt = status.get("outputs", {}).get(key, {})
        if receipt.get("path") != path.relative_to(ROOT).as_posix() or receipt.get("sha256") != _sha256(path):
            raise DataContractError(f"V10输入与采集回执不一致：{key}")
    spot = _normalize_time(pd.read_parquet(paths["spot_1h"]), "open_time")
    fx = pd.read_parquet(paths["fx"])
    benchmark = pd.read_parquet(paths["benchmark"])
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce").astype("datetime64[ns]").dt.normalize()
    benchmark["date"] = (
        pd.to_datetime(benchmark["date"], errors="coerce")
        .astype("datetime64[ns]")
        .dt.normalize()
    )
    symbols = list(contract["universe"]["fixed_symbols"])
    if sorted(spot["symbol"].astype(str).unique()) != sorted(symbols):
        raise DataContractError("V10现货产品集合不正确")
    observed_grids: list[pd.DatetimeIndex] = []
    for symbol in symbols:
        item = spot.loc[spot["symbol"].eq(symbol), "open_time"]
        if item.duplicated().any() or len(item) != int(
            contract["prefreeze_coverage_gate"]["required_spot_hourly_rows_per_symbol"]
        ):
            raise DataContractError(f"V10现货 {symbol}覆盖或重复不正确")
        observed_grids.append(pd.DatetimeIndex(item).sort_values().unique())
    if not all(grid.equals(observed_grids[0]) for grid in observed_grids[1:]):
        raise DataContractError("V10现货缺口不是全市场同步停市")
    source_start = pd.Timestamp(contract["historical_partition"]["acquisition_start"])
    source_end = pd.Timestamp(f"{contract['historical_partition']['visible_end']} 23:00:00")
    missing = pd.date_range(source_start, source_end, freq="1h").difference(observed_grids[0])
    if len(missing) != int(
        contract["prefreeze_coverage_gate"]["required_common_venue_wide_missing_hours"]
    ) or any(timestamp.hour in (23, 0) for timestamp in missing):
        raise DataContractError("V10同步缺口数量变化或触及日度决策时点")
    daily_times = pd.date_range(
        pd.Timestamp(contract["historical_partition"]["visible_start"]),
        pd.Timestamp(contract["historical_partition"]["visible_end"]),
        freq="D",
    )
    for hour in (0, 23):
        required = daily_times + pd.Timedelta(hours=hour)
        observed = spot.loc[spot["open_time"].isin(required)]
        counts = observed.groupby("open_time", observed=True)["symbol"].nunique()
        if len(counts) != len(required) or not counts.eq(len(symbols)).all():
            raise DataContractError(f"V10每日{hour:02d}点覆盖不完整")
    numeric_columns = ["open", "high", "low", "close", "quote_volume"]
    numeric = spot[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DataContractError("V10现货含缺失或非有限数值")
    if not (numeric[["open", "high", "low", "close"]] > 0.0).all().all():
        raise DataContractError("V10现货价格非正")
    if not (numeric["quote_volume"] >= 0.0).all():
        raise DataContractError("V10现货成交额为负")
    audit = {
        "source_status": status["status"],
        "spot_rows": int(len(spot)),
        "fx_rows": int(len(fx)),
        "benchmark_rows": int(len(benchmark)),
        "symbol_count": len(symbols),
        "venue_wide_missing_hour_count": int(len(missing)),
        "missing_daily_signal_or_execution_hour_count": 0,
        "missing_values_backfilled": False,
        "input_sha256": {key: _sha256(path) for key, path in paths.items()},
        "future_price_or_strategy_return_read": True,
        "sealed_replication_read": False,
    }
    return spot, fx, benchmark, audit


def build_semiannual_selections(
    spot: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    symbols = sorted(contract["universe"]["fixed_symbols"])
    data = _normalize_time(spot, "open_time")
    close_23 = (
        data.loc[data["open_time"].dt.hour.eq(23)]
        .pivot(index="open_time", columns="symbol", values="close")
        .reindex(columns=symbols)
        .sort_index()
        .astype(float)
    )
    returns = np.log(close_23 / close_23.shift(1))
    executions = selection_execution_times(contract)
    lookback = int(contract["signal"]["volatility_lookback_calendar_days"])
    ddof = int(contract["signal"]["volatility_standard_deviation_ddof"])
    selected_count = int(contract["signal"]["selected_count"])
    target_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for execution_time in executions:
        signal_time = execution_time - pd.Timedelta(hours=1)
        history = returns.loc[returns.index <= signal_time].tail(lookback)
        if len(history) != lookback or history.isna().any().any():
            raise DataContractError(f"V10 {execution_time}不具备完整180日波动率历史")
        volatility = history.std(axis=0, ddof=ddof)
        if volatility.isna().any() or not np.isfinite(volatility.to_numpy(dtype=float)).all():
            raise DataContractError(f"V10 {execution_time}波动率含非有限值")
        ranked = sorted(symbols, key=lambda symbol: (float(volatility[symbol]), symbol))
        chosen = ranked[:selected_count]
        for rank, symbol in enumerate(ranked, start=1):
            feature_rows.append(
                {
                    "signal_time": signal_time,
                    "execution_time": execution_time,
                    "symbol": symbol,
                    "realized_volatility": float(volatility[symbol]),
                    "volatility_rank": rank,
                    "selected": symbol in chosen,
                }
            )
        for symbol in chosen:
            target_rows.append(
                {
                    "signal_time": signal_time,
                    "execution_time": execution_time,
                    "symbol": symbol,
                    "slot_weight": 1.0 / selected_count,
                    "realized_volatility": float(volatility[symbol]),
                }
            )
    targets = pd.DataFrame(target_rows).sort_values(["execution_time", "symbol"]).reset_index(drop=True)
    features = pd.DataFrame(feature_rows).sort_values(["execution_time", "volatility_rank"]).reset_index(drop=True)
    if not targets.groupby("execution_time", observed=True)["slot_weight"].sum().apply(
        lambda value: math.isclose(float(value), 1.0, abs_tol=1e-12)
    ).all():
        raise DataContractError("V10半年选择权重不为1")
    audit = {
        "selection_event_count": int(len(executions)),
        "selected_count_each_event": selected_count,
        "target_row_count": int(len(targets)),
        "feature_row_count": int(len(features)),
        "first_signal_time": pd.Timestamp(features["signal_time"].min()).isoformat(),
        "last_signal_time": pd.Timestamp(features["signal_time"].max()).isoformat(),
        "volatility_lookback_calendar_days": lookback,
        "volatility_ddof": ddof,
        "strict_execution_lag_hours": 1,
        "future_price_or_return_used_in_selection": False,
        "sealed_replication_read": False,
    }
    return targets, features, audit


def _daily_market_panels(
    spot: pd.DataFrame,
    symbols: list[str],
    days: pd.DatetimeIndex,
    lookback: int,
) -> dict[str, np.ndarray]:
    data = _normalize_time(spot, "open_time").sort_values(["symbol", "open_time"])
    panels: dict[str, pd.DataFrame] = {}
    for column in ("open", "close", "low", "quote_volume"):
        panels[column] = (
            data.pivot(index="open_time", columns="symbol", values=column)
            .reindex(columns=symbols)
            .sort_index()
            .astype(float)
        )
    opens = panels["open"].reindex(days)
    closes_23 = panels["close"].reindex(days + pd.Timedelta(hours=23))
    closes_23.index = days
    dated = data.assign(trade_date=data["open_time"].dt.normalize())
    lows = (
        dated.groupby(["trade_date", "symbol"], observed=True)["low"]
        .min()
        .unstack("symbol")
        .reindex(index=days, columns=symbols)
    )
    prior_median_parts: list[pd.DataFrame] = []
    for symbol, group in data.groupby("symbol", observed=True, sort=True):
        item = group[["open_time", "quote_volume"]].copy().sort_values("open_time")
        item["symbol"] = symbol
        item["prior_median_quote_volume"] = (
            item["quote_volume"].shift(1).rolling(lookback, min_periods=lookback).median()
        )
        prior_median_parts.append(item)
    prior_median_long = pd.concat(prior_median_parts, ignore_index=True)
    prior_median = (
        prior_median_long.pivot(
            index="open_time", columns="symbol", values="prior_median_quote_volume"
        )
        .reindex(index=days, columns=symbols)
    )
    result_frames = {
        "open": opens,
        "close_23": closes_23,
        "low": lows,
        "prior_median_quote_volume": prior_median,
    }
    result: dict[str, np.ndarray] = {}
    for key, frame in result_frames.items():
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataContractError(f"V10日度{key}矩阵存在缺口或非有限值")
        result[key] = values
    return result


def run_portfolio_backtest(
    selections: pd.DataFrame,
    spot: pd.DataFrame,
    fx: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    end = pd.Timestamp(contract["historical_partition"]["visible_end"])
    days = pd.date_range(start, end, freq="D")
    symbols = list(contract["universe"]["fixed_symbols"])
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    market = _daily_market_panels(
        spot,
        symbols,
        days,
        int(contract["universe"]["turnover_lookback_observed_venue_hours"]),
    )
    next_open = np.empty_like(market["open"])
    next_open[:-1] = market["open"][1:]
    next_open[-1] = market["close_23"][-1]
    selection_by_time = {
        pd.Timestamp(timestamp): list(group.sort_values("symbol")["symbol"].astype(str))
        for timestamp, group in selections.groupby("execution_time", observed=True)
    }
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
    increments = np.asarray(
        [float(contract["portfolio"]["quantity_increment"][symbol]) for symbol in symbols]
    )
    positions = np.zeros(len(symbols), dtype=float)
    target_positions = np.zeros(len(symbols), dtype=float)
    reserve = float(contract["portfolio"]["stress_cost_reserve_multiplier"])
    slot_fraction = float(contract["portfolio"]["slot_target_notional_fraction"])
    capacity_fraction = float(
        contract["universe"]["maximum_order_fraction_of_prior_median_quote_volume"]
    )
    minimum_trade = float(contract["portfolio"]["minimum_trade_notional_usdt"])
    stop_threshold = float(contract["stop_loss"]["threshold"])
    winddown = pd.Timestamp(contract["portfolio"]["terminal_winddown_start"])
    selected_symbols: list[str] = []
    stopped_symbols: set[str] = set()
    month_reference: dict[str, float] = {}
    maximum_gross = maximum_order_fraction = maximum_daily_intraday_loss = 0.0
    selection_trade_days = monthly_reentry_trade_days = stop_trade_days = terminal_trade_days = 0
    capacity_continuation_trade_days = 0
    stop_event_count = global_slice_count = minimum_trade_skip_count = 0
    total_transaction_cost = np.zeros(2, dtype=float)
    total_cash_interest = np.zeros(2, dtype=float)
    trade_rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    daily_cash_rate = (1.0 + float(contract["account"]["cash_annual_rate"])) ** (1.0 / 365.0) - 1.0

    for day_index, timestamp in enumerate(days):
        open_prices = market["open"][day_index]
        close_23 = market["close_23"][day_index - 1] if day_index > 0 else None
        desired = target_positions.copy()
        reason: str | None = None
        newly_stopped: list[str] = []
        if timestamp >= winddown:
            target_positions[:] = 0.0
            desired = target_positions.copy()
            reason = "TERMINAL_WINDDOWN"
        elif timestamp in selection_by_time:
            selected_symbols = selection_by_time[timestamp]
            stopped_symbols = set()
            target_positions[:] = 0.0
            slot_notional = float(nav.min()) * reserve * slot_fraction
            for symbol in selected_symbols:
                index = symbol_index[symbol]
                target_positions[index] = _truncate_toward_zero(
                    np.asarray([slot_notional / open_prices[index]]),
                    np.asarray([increments[index]]),
                )[0]
                month_reference[symbol] = float(open_prices[index])
            desired = target_positions.copy()
            reason = "SELECTION_REBALANCE"
        elif timestamp.day == 1:
            slot_notional = float(nav.min()) * reserve * slot_fraction
            for symbol in selected_symbols:
                index = symbol_index[symbol]
                if symbol in stopped_symbols or positions[index] <= 1e-12:
                    target_positions[index] = _truncate_toward_zero(
                        np.asarray([slot_notional / open_prices[index]]),
                        np.asarray([increments[index]]),
                    )[0]
                month_reference[symbol] = float(open_prices[index])
            stopped_symbols = set()
            desired = target_positions.copy()
            reason = "MONTHLY_REENTRY"
        else:
            if close_23 is None:
                raise DataContractError("V10止损缺少上一日23点收盘")
            for symbol in selected_symbols:
                if symbol in stopped_symbols:
                    continue
                index = symbol_index[symbol]
                reference = month_reference.get(symbol)
                if reference is None:
                    raise DataContractError(f"V10 {timestamp} {symbol}缺少月度参考价")
                cumulative_return = float(close_23[index]) / reference - 1.0
                if cumulative_return <= -stop_threshold:
                    stopped_symbols.add(symbol)
                    newly_stopped.append(symbol)
                    stop_event_count += 1
                    target_positions[index] = 0.0
            if newly_stopped or any(
                positions[symbol_index[symbol]] > 1e-12 for symbol in stopped_symbols
            ):
                for symbol in stopped_symbols:
                    target_positions[symbol_index[symbol]] = 0.0
                desired = target_positions.copy()
                reason = "STOP_LOSS" if newly_stopped else "STOP_EXIT_CONTINUATION"

        full_delta = desired - positions
        if reason is None and (np.abs(full_delta) > 1e-12).any():
            reason = "CAPACITY_CONTINUATION"
        full_order_notional = np.abs(full_delta) * open_prices
        capacities = capacity_fraction * market["prior_median_quote_volume"][day_index]
        active_orders = full_order_notional > 1e-12
        slice_scale = 1.0
        if active_orders.any():
            slice_scale = min(
                1.0,
                float(np.min(capacities[active_orders] / full_order_notional[active_orders])),
            )
        if slice_scale < 1.0 - 1e-12:
            global_slice_count += 1
        executed_delta = _truncate_toward_zero(slice_scale * full_delta, increments)
        order_notional = np.abs(executed_delta) * open_prices
        small_buys = (
            (executed_delta > 1e-12)
            & (order_notional > 1e-12)
            & (order_notional < minimum_trade)
        )
        minimum_trade_skip_count += int(small_buys.sum())
        executed_delta[small_buys] = 0.0
        new_positions = np.rint((positions + executed_delta) / increments) * increments
        executed_delta = new_positions - positions
        if (new_positions < -1e-12).any():
            raise DataContractError(f"{timestamp} V10生成了现货空头")
        order_notional = np.abs(executed_delta) * open_prices
        order_fractions = np.divide(
            order_notional,
            market["prior_median_quote_volume"][day_index],
            out=np.zeros_like(order_notional),
            where=market["prior_median_quote_volume"][day_index] > 0.0,
        )
        maximum_order_fraction = max(maximum_order_fraction, float(order_fractions.max()))
        if (order_fractions > capacity_fraction + 1e-10).any():
            raise DataContractError(f"{timestamp} V10订单越过冻结容量")
        traded = order_notional > 1e-12
        if traded.any():
            costs = np.asarray(
                [
                    float(order_notional.sum()) * _spot_cost_rate(contract, "base"),
                    float(order_notional.sum()) * _spot_cost_rate(contract, "stress"),
                ]
            )
            nav -= costs
            total_transaction_cost += costs
            if reason == "SELECTION_REBALANCE":
                selection_trade_days += 1
            elif reason == "MONTHLY_REENTRY":
                monthly_reentry_trade_days += 1
            elif reason in ("STOP_LOSS", "STOP_EXIT_CONTINUATION"):
                stop_trade_days += 1
            elif reason == "TERMINAL_WINDDOWN":
                terminal_trade_days += 1
            elif reason == "CAPACITY_CONTINUATION":
                capacity_continuation_trade_days += 1
            for index in np.flatnonzero(traded):
                trade_rows.append(
                    {
                        "trade_time": timestamp,
                        "symbol": symbols[index],
                        "reason": reason,
                        "side": "BUY" if executed_delta[index] > 0.0 else "SELL",
                        "quantity": abs(float(executed_delta[index])),
                        "signed_quantity_change": float(executed_delta[index]),
                        "execution_open": float(open_prices[index]),
                        "order_notional_usdt": float(order_notional[index]),
                        "prior_median_quote_volume_usdt": float(
                            market["prior_median_quote_volume"][day_index, index]
                        ),
                        "capacity_fraction": float(order_fractions[index]),
                        "global_slice_scale": float(slice_scale),
                        "position_after": float(new_positions[index]),
                        "new_stop_trigger": symbols[index] in newly_stopped,
                    }
                )
        positions = new_positions
        if float(nav.min()) <= 0.0:
            raise DataContractError(f"{timestamp} V10交易成本后净值非正")
        spot_notional = float(np.sum(positions * open_prices))
        minimum_nav = float(nav.min())
        gross_exposure = spot_notional / minimum_nav
        maximum_gross = max(maximum_gross, gross_exposure)
        if gross_exposure > float(contract["risk"]["maximum_gross_exposure"]) + 1e-10:
            raise DataContractError(f"{timestamp} V10现货毛敞口超过冻结上限")
        cash = nav - spot_notional
        if float(cash.min()) < -1e-8:
            raise DataContractError(f"{timestamp} V10出现负现金或借款")
        worst_intraday_pnl = float(
            np.sum(positions * (market["low"][day_index] - open_prices))
        )
        maximum_daily_intraday_loss = max(
            maximum_daily_intraday_loss,
            -min(0.0, worst_intraday_pnl) / minimum_nav,
        )
        if minimum_nav + worst_intraday_pnl <= 0.0:
            raise DataContractError(f"{timestamp} V10日内最坏权益非正")
        price_pnl = float(np.sum(positions * (next_open[day_index] - open_prices)))
        interest = np.maximum(cash, 0.0) * daily_cash_rate
        nav += price_pnl + interest
        total_cash_interest += interest
        if float(nav.min()) <= 0.0:
            raise DataContractError(f"{timestamp} V10日结净值非正")
        day_rows.append(
            {
                "trade_date": timestamp,
                "base_nav_usdt": float(nav[0]),
                "stress_nav_usdt": float(nav[1]),
                "price_pnl_usdt": price_pnl,
                "base_cash_interest_usdt": float(interest[0]),
                "stress_cash_interest_usdt": float(interest[1]),
                "gross_exposure": gross_exposure,
                "net_exposure": gross_exposure,
                "capacity_pass": True,
                "active_position_count": int(np.count_nonzero(positions > 1e-12)),
                "stopped_slot_count": int(len(stopped_symbols)),
            }
        )

    terminal_count = int(np.count_nonzero(positions > 1e-12))
    if terminal_count:
        raise DataContractError("V10末端七日容量减仓后仍有非零现货持仓")
    path = pd.DataFrame(day_rows)
    trades = pd.DataFrame(
        trade_rows,
        columns=[
            "trade_time",
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
            "position_after",
            "new_stop_trigger",
        ],
    )
    daily = context.merge(path, left_on="date", right_on="trade_date", how="left", validate="one_to_one")
    if daily[["base_nav_usdt", "stress_nav_usdt"]].isna().any().any():
        raise DataContractError("V10日度净值存在缺口")
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
        "selection_trade_day_count": selection_trade_days,
        "monthly_reentry_trade_day_count": monthly_reentry_trade_days,
        "stop_loss_event_count": stop_event_count,
        "stop_exit_trade_day_count": stop_trade_days,
        "terminal_winddown_trade_day_count": terminal_trade_days,
        "capacity_continuation_trade_day_count": capacity_continuation_trade_days,
        "global_capacity_sliced_day_count": global_slice_count,
        "minimum_trade_skipped_order_count": minimum_trade_skip_count,
        "maximum_order_capacity_fraction": maximum_order_fraction,
        "maximum_gross_exposure": maximum_gross,
        "maximum_daily_intraday_loss_fraction": maximum_daily_intraday_loss,
        "base_total_transaction_cost_usdt": float(total_transaction_cost[0]),
        "stress_total_transaction_cost_usdt": float(total_transaction_cost[1]),
        "base_total_cash_interest_usdt": float(total_cash_interest[0]),
        "stress_total_cash_interest_usdt": float(total_cash_interest[1]),
        "terminal_position_count": terminal_count,
        "short_position_count": 0,
        "derivative_position_count": 0,
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
            "# 数字资产现货低波动止损V10可见期结果",
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
    spot, fx, benchmark, input_audit = load_visible_inputs(contract)
    selections, features, signal_audit = build_semiannual_selections(spot, contract)
    daily, trades, portfolio_audit, context_audit = run_portfolio_backtest(
        selections, spot, fx, benchmark, contract
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
        "report_id": "DIGITAL_ASSET_SPOT_LOW_VOLATILITY_STOP_LOSS_V10_VISIBLE",
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
            "selection": signal_audit,
            "portfolio": portfolio_audit,
            "daily_context": context_audit,
            "selection_feature_rows": int(len(features)),
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
                else "冻结拒绝V10，不修改180日波动、半年持有、三币集中、5%月内止损、成本或容量门救回"
            ),
        },
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    outputs = contract["outputs"]
    atomic_parquet(ROOT / outputs["visible_daily_returns"], daily)
    atomic_parquet(ROOT / outputs["visible_targets"], selections)
    atomic_parquet(ROOT / outputs["visible_trades"], trades)
    atomic_json(ROOT / outputs["input_audit_json"], report["data_audit"])
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
