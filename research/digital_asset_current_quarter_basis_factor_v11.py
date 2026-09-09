"""BTC与ETH当季期货基差高低组合V11。"""

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
CONFIG = ROOT / "config" / "digital_asset_current_quarter_basis_factor_v11.yaml"
FIXED_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("当季期货基差V11配置必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    failures: list[str] = []
    protocol = contract.get("protocol", {})
    if protocol.get("candidate_id") != "DIGITAL_ASSET_CURRENT_QUARTER_BASIS_FACTOR_V11":
        failures.append("candidate_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V21":
        failures.append("parent_protocol_id")
    account = contract.get("account", {})
    if float(account.get("initial_capital_cny", math.nan)) != 500000.0:
        failures.append("initial_capital_cny")
    if float(account.get("user_transaction_fee_rate_per_leg", math.nan)) != 0.0001:
        failures.append("user_transaction_fee_rate_per_leg")
    history = contract.get("historical_partition", {})
    if history.get("visible_start") != "2021-09-01" or history.get("visible_end") != "2026-08-14":
        failures.append("visible_period")
    universe = contract.get("universe", {})
    if universe.get("fixed_symbols") != FIXED_SYMBOLS:
        failures.append("fixed_symbols")
    if int(universe.get("turnover_lookback_observed_venue_hours", 0)) != 168:
        failures.append("turnover_lookback")
    if float(universe.get("maximum_order_fraction_of_prior_median_quote_volume", math.nan)) != 0.001:
        failures.append("maximum_order_fraction")
    signal = contract.get("signal", {})
    if signal.get("daily_basis_formula") != "LOG_SPOT_23UTC_CLOSE_DIV_CURRENT_QUARTER_23UTC_CLOSE":
        failures.append("basis_formula")
    if int(signal.get("basis_lookback_calendar_days", 0)) != 5:
        failures.append("basis_lookback")
    if int(signal.get("long_count", 0)) != 1 or int(signal.get("short_count", 0)) != 1:
        failures.append("long_short_count")
    for key, expected in {
        "long_side_target_gross": 0.45,
        "short_side_target_gross": 0.45,
        "total_target_gross": 0.90,
        "target_net": 0.0,
    }.items():
        if float(signal.get(key, math.nan)) != expected:
            failures.append(key)
    if signal.get("spot_position_allowed") is not False:
        failures.append("spot_position_allowed")
    delivery = contract.get("delivery", {})
    if delivery.get("delivery_calendar") != "LAST_FRIDAY_OF_MARCH_JUNE_SEPTEMBER_DECEMBER":
        failures.append("delivery_calendar")
    if delivery.get("delivery_day_position_policy") != "ZERO_FROM_00UTC_THROUGH_23UTC":
        failures.append("delivery_policy")
    portfolio = contract.get("portfolio", {})
    if portfolio.get("quantity_increment") != {"BTCUSDT": 0.001, "ETHUSDT": 0.001}:
        failures.append("quantity_increment")
    if portfolio.get("terminal_winddown_start") != "2026-08-08 00:00:00":
        failures.append("terminal_winddown_start")
    costs = contract.get("costs", {})
    for key, expected in {
        "user_transaction_fee_rate_per_leg": 0.0001,
        "base_total_futures_cost_bps_per_leg": 8.0,
        "stress_total_futures_cost_bps_per_leg": 30.0,
        "base_fx_conversion_spread_bps_per_conversion": 10.0,
        "stress_fx_conversion_spread_bps_per_conversion": 30.0,
        "funding_rate_assumed": 0.0,
    }.items():
        if float(costs.get(key, math.nan)) != expected:
            failures.append(key)
    risk = contract.get("risk", {})
    if float(risk.get("maximum_gross_exposure", math.nan)) != 1.05:
        failures.append("maximum_gross_exposure")
    if float(risk.get("maximum_absolute_net_exposure", math.nan)) != 0.10:
        failures.append("maximum_absolute_net_exposure")
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
        raise ValueError(f"当季期货基差V11配置被弱化或损坏：{sorted(set(failures))}")


def _cost_rate(contract: dict[str, Any], scenario: str) -> float:
    return float(contract["costs"][f"{scenario}_total_futures_cost_bps_per_leg"]) / 10000.0


def delivery_days(contract: dict[str, Any]) -> pd.DatetimeIndex:
    start = pd.Timestamp(contract["historical_partition"]["visible_start"]).normalize()
    end = pd.Timestamp(contract["historical_partition"]["visible_end"]).normalize()
    days: list[pd.Timestamp] = []
    for year in range(start.year, end.year + 1):
        for month in (3, 6, 9, 12):
            last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            last_friday = last - pd.Timedelta(days=(last.weekday() - 4) % 7)
            if start <= last_friday <= end:
                days.append(last_friday)
    return pd.DatetimeIndex(days)


def load_visible_inputs(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inputs = contract["inputs"]
    status_path = ROOT / inputs["source_status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    forbidden = (
        "basis_value_or_rank_computed",
        "long_or_short_direction_computed",
        "signal_or_target_computed",
        "strategy_or_benchmark_return_computed",
        "account_or_authenticated_endpoint_used",
    )
    if (
        status.get("status") != "PASS_INPUT_ACQUISITION_ONLY"
        or status.get("candidate_id") != contract["protocol"]["candidate_id"]
        or any(bool(status.get(key)) for key in forbidden)
    ):
        raise DataContractError("V11官方输入采集状态不合格")
    paths = {
        "spot_1h": ROOT / inputs["spot_1h"],
        "current_quarter_1h": ROOT / inputs["current_quarter_1h"],
        "fx": ROOT / inputs["visible_fx"],
        "benchmark": ROOT / inputs["visible_benchmark"],
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少V11输入：{key}={path}")
        receipt = status.get("outputs", {}).get(key, {})
        if receipt.get("path") != path.relative_to(ROOT).as_posix() or receipt.get("sha256") != _sha256(path):
            raise DataContractError(f"V11输入与采集回执不一致：{key}")
    spot = _normalize_time(pd.read_parquet(paths["spot_1h"]), "open_time")
    futures = _normalize_time(pd.read_parquet(paths["current_quarter_1h"]), "open_time")
    fx = pd.read_parquet(paths["fx"])
    benchmark = pd.read_parquet(paths["benchmark"])
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce").astype("datetime64[ns]").dt.normalize()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce").astype("datetime64[ns]").dt.normalize()
    symbols = list(contract["universe"]["fixed_symbols"])
    gate = contract["prefreeze_coverage_gate"]
    expected = pd.date_range(
        pd.Timestamp(gate["required_grid_start"]),
        pd.Timestamp(gate["required_grid_end"]),
        freq="1h",
    )
    grid_audit: dict[str, Any] = {}
    for label, frame, required_rows, required_missing in (
        (
            "spot",
            spot,
            int(gate["required_spot_hourly_rows_per_symbol"]),
            int(gate["required_common_spot_missing_hours"]),
        ),
        (
            "current_quarter",
            futures,
            int(gate["required_current_quarter_hourly_rows_per_symbol"]),
            int(gate["required_common_current_quarter_missing_hours"]),
        ),
    ):
        if sorted(frame["symbol"].astype(str).unique()) != sorted(symbols):
            raise DataContractError(f"V11 {label}产品集合不正确")
        observed_grids: list[pd.DatetimeIndex] = []
        for symbol in symbols:
            subset = frame.loc[frame["symbol"].eq(symbol), "open_time"]
            if subset.duplicated().any() or len(subset) != required_rows:
                raise DataContractError(f"V11 {label} {symbol}覆盖或重复不正确")
            observed_grids.append(pd.DatetimeIndex(subset).sort_values().unique())
        if not observed_grids[0].equals(observed_grids[1]):
            raise DataContractError(f"V11 {label}两产品小时网格不同")
        missing = expected.difference(observed_grids[0])
        missing_decision = [timestamp for timestamp in missing if timestamp.hour in (0, 23)]
        if len(missing) != required_missing or missing_decision:
            raise DataContractError(f"V11 {label}缺口数量变化或触及决策时点")
        grid_audit[label] = {
            "rows_per_symbol": required_rows,
            "common_missing_hour_count": int(len(missing)),
            "missing_decision_hour_count": int(len(missing_decision)),
        }
        numeric_columns = ["open", "high", "low", "close", "quote_volume"]
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise DataContractError(f"V11 {label}含缺失或非有限数值")
        if not (numeric[["open", "high", "low", "close"]] > 0.0).all().all():
            raise DataContractError(f"V11 {label}价格非正")
        if not (numeric["quote_volume"] >= 0.0).all():
            raise DataContractError(f"V11 {label}成交额为负")
    for label, frame, column in (
        ("fx", fx, "cny_per_usd"),
        ("benchmark", benchmark, "close"),
    ):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if frame["date"].isna().any() or numeric.isna().any() or not (numeric > 0.0).all():
            raise DataContractError(f"V11 {label}上下文不完整或非正")
    audit = {
        "source_status": status["status"],
        "spot_rows": int(len(spot)),
        "current_quarter_rows": int(len(futures)),
        "fx_rows": int(len(fx)),
        "benchmark_rows": int(len(benchmark)),
        "symbol_count": len(symbols),
        "grid": grid_audit,
        "missing_values_backfilled": False,
        "input_sha256": {key: _sha256(path) for key, path in paths.items()},
        "basis_rank_or_strategy_return_computed_during_collection": False,
    }
    return spot, futures, fx, benchmark, audit


def build_daily_targets(
    spot: pd.DataFrame,
    futures: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    symbols = sorted(contract["universe"]["fixed_symbols"])
    spot_data = _normalize_time(spot, "open_time")
    futures_data = _normalize_time(futures, "open_time")
    spot_close = (
        spot_data.loc[spot_data["open_time"].dt.hour.eq(23)]
        .pivot(index="open_time", columns="symbol", values="close")
        .reindex(columns=symbols)
        .sort_index()
        .astype(float)
    )
    futures_close = (
        futures_data.loc[futures_data["open_time"].dt.hour.eq(23)]
        .pivot(index="open_time", columns="symbol", values="close")
        .reindex(columns=symbols)
        .sort_index()
        .astype(float)
    )
    common = spot_close.index.intersection(futures_close.index)
    spot_close = spot_close.reindex(common)
    futures_close = futures_close.reindex(common)
    basis = np.log(spot_close / futures_close)
    lookback = int(contract["signal"]["basis_lookback_calendar_days"])
    mean_basis = basis.rolling(lookback, min_periods=lookback).mean()
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    winddown = pd.Timestamp(contract["portfolio"]["terminal_winddown_start"])
    executions = pd.date_range(start, winddown - pd.Timedelta(days=1), freq="1D")
    delivery = set(delivery_days(contract))
    long_weight = float(contract["signal"]["long_side_target_gross"])
    short_weight = float(contract["signal"]["short_side_target_gross"])
    target_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    delivery_count = 0
    for execution_time in executions:
        signal_time = execution_time - pd.Timedelta(hours=1)
        if signal_time not in mean_basis.index:
            raise DataContractError(f"V11 {execution_time}缺少23点基差信号")
        current = basis.loc[signal_time]
        averaged = mean_basis.loc[signal_time]
        if current.isna().any() or averaged.isna().any():
            raise DataContractError(f"V11 {execution_time}不具备完整5日基差历史")
        if not np.isfinite(current.to_numpy(dtype=float)).all() or not np.isfinite(
            averaged.to_numpy(dtype=float)
        ).all():
            raise DataContractError(f"V11 {execution_time}基差含非有限值")
        ranked = sorted(symbols, key=lambda symbol: (-float(averaged[symbol]), symbol))
        is_delivery = execution_time.normalize() in delivery
        if is_delivery:
            delivery_count += 1
        for rank, symbol in enumerate(ranked, start=1):
            if is_delivery:
                selection = "DELIVERY_CASH"
                target_weight = 0.0
            elif rank == 1:
                selection = "LONG_HIGH_BASIS"
                target_weight = long_weight
            else:
                selection = "SHORT_LOW_BASIS"
                target_weight = -short_weight
            feature_rows.append(
                {
                    "signal_time": signal_time,
                    "execution_time": execution_time,
                    "symbol": symbol,
                    "daily_log_basis": float(current[symbol]),
                    "mean_log_basis_5d": float(averaged[symbol]),
                    "mean_basis_rank_desc": rank,
                    "selection": selection,
                    "delivery_blackout": is_delivery,
                }
            )
            target_rows.append(
                {
                    "signal_time": signal_time,
                    "execution_time": execution_time,
                    "symbol": symbol,
                    "target_weight": target_weight,
                    "mean_log_basis_5d": float(averaged[symbol]),
                    "delivery_blackout": is_delivery,
                }
            )
    targets = pd.DataFrame(target_rows).sort_values(["execution_time", "symbol"]).reset_index(drop=True)
    features = pd.DataFrame(feature_rows).sort_values(["execution_time", "mean_basis_rank_desc"]).reset_index(drop=True)
    grouped = targets.groupby("execution_time", observed=True)["target_weight"]
    gross = grouped.apply(lambda values: float(values.abs().sum()))
    net = grouped.sum()
    blackout = targets.groupby("execution_time", observed=True)["delivery_blackout"].first()
    expected_gross = float(contract["signal"]["total_target_gross"])
    if not np.allclose(gross.loc[~blackout], expected_gross) or not np.allclose(net, 0.0):
        raise DataContractError("V11非交割日多空目标或净目标不正确")
    if not np.allclose(gross.loc[blackout], 0.0):
        raise DataContractError("V11交割日目标未归零")
    audit = {
        "daily_signal_count": int(len(executions)),
        "delivery_blackout_day_count": delivery_count,
        "target_row_count": int(len(targets)),
        "feature_row_count": int(len(features)),
        "first_signal_time": pd.Timestamp(features["signal_time"].min()).isoformat(),
        "last_signal_time": pd.Timestamp(features["signal_time"].max()).isoformat(),
        "basis_lookback_calendar_days": lookback,
        "strict_execution_lag_hours": 1,
        "tie_break": "SYMBOL_ASCENDING",
        "spot_position_generated": False,
        "future_price_or_return_used_in_signal": False,
    }
    return targets, features, audit


def _market_panels(
    futures: pd.DataFrame,
    symbols: list[str],
    times: pd.DatetimeIndex,
    lookback: int,
) -> dict[str, np.ndarray]:
    data = _normalize_time(futures, "open_time").sort_values(["symbol", "open_time"])
    frames: dict[str, pd.DataFrame] = {}
    for column in ("open", "high", "low", "close", "quote_volume"):
        frames[column] = (
            data.pivot(index="open_time", columns="symbol", values=column)
            .reindex(columns=symbols)
            .sort_index()
            .astype(float)
        )
    frames["prior_median_quote_volume"] = (
        frames["quote_volume"].shift(1).rolling(lookback, min_periods=lookback).median()
    )
    result: dict[str, np.ndarray] = {}
    for key in ("open", "high", "low", "close", "prior_median_quote_volume"):
        values = frames[key].reindex(index=times, columns=symbols).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataContractError(f"V11可见期{key}矩阵存在缺口或非有限值")
        result[key] = values
    return result


def run_portfolio_backtest(
    targets: pd.DataFrame,
    futures: pd.DataFrame,
    fx: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    end = pd.Timestamp(f"{contract['historical_partition']['visible_end']} 23:00:00")
    times = pd.date_range(start, end, freq="1h")
    symbols = list(contract["universe"]["fixed_symbols"])
    market = _market_panels(
        futures,
        symbols,
        times,
        int(contract["universe"]["turnover_lookback_observed_venue_hours"]),
    )
    next_open = np.empty_like(market["open"])
    next_open[:-1] = market["open"][1:]
    next_open[-1] = market["close"][-1]
    daily_target = (
        targets.pivot(index="execution_time", columns="symbol", values="target_weight")
        .reindex(columns=symbols)
        .fillna(0.0)
    )
    target_by_time = {
        pd.Timestamp(timestamp): row.to_numpy(dtype=float)
        for timestamp, row in daily_target.iterrows()
    }
    delivery_set = set(delivery_days(contract))
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
    capacity_fraction = float(
        contract["universe"]["maximum_order_fraction_of_prior_median_quote_volume"]
    )
    minimum_trade = float(contract["portfolio"]["minimum_trade_notional_usdt"])
    winddown = pd.Timestamp(contract["portfolio"]["terminal_winddown_start"])
    maximum_gross_limit = float(contract["risk"]["maximum_gross_exposure"])
    maximum_net_limit = float(contract["risk"]["maximum_absolute_net_exposure"])
    risk_buffer = float(contract["risk"]["intraday_risk_deleveraging_buffer_multiplier"])
    maintenance_fraction = float(
        contract["risk"]["minimum_combined_equity_fraction_of_gross_notional"]
    )
    maximum_gross = maximum_net = maximum_order_fraction = 0.0
    maximum_intrabar_loss_fraction = 0.0
    daily_rebalance_count = delivery_blackout_count = risk_deleverage_count = 0
    terminal_trade_count = capacity_continuation_count = global_slice_count = 0
    minimum_trade_skip_count = 0
    total_transaction_cost = np.zeros(2, dtype=float)
    total_cash_interest = np.zeros(2, dtype=float)
    trade_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    cash_hourly_rate = (1.0 + float(contract["account"]["cash_annual_rate"])) ** (
        1.0 / 8760.0
    ) - 1.0

    for time_index, timestamp in enumerate(times):
        open_prices = market["open"][time_index]
        minimum_nav_before_trade = float(nav.min())
        if minimum_nav_before_trade <= 0.0:
            raise DataContractError(f"{timestamp} V11交易前净值非正")
        current_gross = float(np.sum(np.abs(positions) * open_prices)) / minimum_nav_before_trade
        current_net = float(np.sum(positions * open_prices)) / minimum_nav_before_trade
        reason: str | None = None
        target_weights = np.zeros(len(symbols), dtype=float)
        if timestamp >= winddown:
            target_positions[:] = 0.0
            reason = "TERMINAL_WINDDOWN"
        elif timestamp in target_by_time:
            target_weights = target_by_time[timestamp]
            target_positions = _truncate_toward_zero(
                target_weights * minimum_nav_before_trade / open_prices,
                increments,
            )
            reason = (
                "DELIVERY_BLACKOUT"
                if timestamp.normalize() in delivery_set
                else "DAILY_REBALANCE"
            )
        elif current_gross > maximum_gross_limit or abs(current_net) > maximum_net_limit:
            scale = 1.0
            if current_gross > 0.0:
                scale = min(scale, risk_buffer * maximum_gross_limit / current_gross)
            if abs(current_net) > 0.0:
                scale = min(scale, risk_buffer * maximum_net_limit / abs(current_net))
            target_positions = _truncate_toward_zero(positions * scale, increments)
            reason = "INTRADAY_RISK_DELEVERAGE"
        desired = target_positions.copy()
        full_delta = desired - positions
        if reason is None and (np.abs(full_delta) > 1e-12).any():
            reason = "CAPACITY_CONTINUATION"
        full_order_notional = np.abs(full_delta) * open_prices
        capacities = capacity_fraction * market["prior_median_quote_volume"][time_index]
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
        proposed_positions = positions + executed_delta
        order_notional = np.abs(executed_delta) * open_prices
        exposure_increasing = np.abs(proposed_positions) > np.abs(positions) + 1e-12
        small_increases = (
            exposure_increasing
            & (order_notional > 1e-12)
            & (order_notional < minimum_trade)
        )
        minimum_trade_skip_count += int(small_increases.sum())
        executed_delta[small_increases] = 0.0
        new_positions = np.rint((positions + executed_delta) / increments) * increments
        executed_delta = new_positions - positions
        order_notional = np.abs(executed_delta) * open_prices
        order_fractions = np.divide(
            order_notional,
            market["prior_median_quote_volume"][time_index],
            out=np.zeros_like(order_notional),
            where=market["prior_median_quote_volume"][time_index] > 0.0,
        )
        maximum_order_fraction = max(maximum_order_fraction, float(order_fractions.max()))
        if (order_fractions > capacity_fraction + 1e-10).any():
            raise DataContractError(f"{timestamp} V11订单越过冻结容量")
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
            if reason == "DAILY_REBALANCE":
                daily_rebalance_count += 1
            elif reason == "DELIVERY_BLACKOUT":
                delivery_blackout_count += 1
            elif reason == "INTRADAY_RISK_DELEVERAGE":
                risk_deleverage_count += 1
            elif reason == "TERMINAL_WINDDOWN":
                terminal_trade_count += 1
            elif reason == "CAPACITY_CONTINUATION":
                capacity_continuation_count += 1
            for index in np.flatnonzero(traded):
                trade_rows.append(
                    {
                        "trade_time": timestamp,
                        "signal_time": (
                            timestamp - pd.Timedelta(hours=1)
                            if reason in ("DAILY_REBALANCE", "DELIVERY_BLACKOUT")
                            else pd.NaT
                        ),
                        "symbol": symbols[index],
                        "reason": reason,
                        "side": "BUY" if executed_delta[index] > 0.0 else "SELL",
                        "quantity": abs(float(executed_delta[index])),
                        "signed_quantity_change": float(executed_delta[index]),
                        "execution_open": float(open_prices[index]),
                        "order_notional_usdt": float(order_notional[index]),
                        "prior_median_quote_volume_usdt": float(
                            market["prior_median_quote_volume"][time_index, index]
                        ),
                        "capacity_fraction": float(order_fractions[index]),
                        "global_slice_scale": float(slice_scale),
                        "target_weight": float(target_weights[index]),
                        "position_after": float(new_positions[index]),
                    }
                )
        positions = new_positions
        if float(nav.min()) <= 0.0:
            raise DataContractError(f"{timestamp} V11交易成本后净值非正")
        gross_notional = float(np.sum(np.abs(positions) * open_prices))
        net_notional = float(np.sum(positions * open_prices))
        minimum_nav = float(nav.min())
        gross_exposure = gross_notional / minimum_nav
        net_exposure = net_notional / minimum_nav
        maximum_gross = max(maximum_gross, gross_exposure)
        maximum_net = max(maximum_net, abs(net_exposure))
        if gross_exposure > maximum_gross_limit + 1e-10:
            raise DataContractError(f"{timestamp} V11风险减仓后实际毛敞口仍超过冻结上限")
        if abs(net_exposure) > maximum_net_limit + 1e-10:
            raise DataContractError(f"{timestamp} V11风险减仓后实际净敞口仍超过冻结上限")
        long_worst = np.where(
            positions > 0.0,
            positions * (market["low"][time_index] - open_prices),
            0.0,
        )
        short_worst = np.where(
            positions < 0.0,
            positions * (market["high"][time_index] - open_prices),
            0.0,
        )
        worst_intrabar_pnl = float(long_worst.sum() + short_worst.sum())
        worst_equity = minimum_nav + worst_intrabar_pnl
        maintenance = maintenance_fraction * gross_notional
        maximum_intrabar_loss_fraction = max(
            maximum_intrabar_loss_fraction,
            -min(0.0, worst_intrabar_pnl) / minimum_nav,
        )
        if worst_equity <= maintenance:
            raise DataContractError(f"{timestamp} V11合并权益代理触发保证金失败")
        price_pnl = float(np.sum(positions * (next_open[time_index] - open_prices)))
        cash_interest = np.zeros(2, dtype=float)
        if not np.any(np.abs(positions) > 1e-12):
            cash_interest = nav * cash_hourly_rate
        nav += price_pnl + cash_interest
        total_cash_interest += cash_interest
        if float(nav.min()) <= 0.0:
            raise DataContractError(f"{timestamp} V11区间结算后净值非正")
        interval_rows.append(
            {
                "interval_open_time": timestamp,
                "base_nav_usdt": float(nav[0]),
                "stress_nav_usdt": float(nav[1]),
                "price_pnl_usdt": price_pnl,
                "base_cash_interest_usdt": float(cash_interest[0]),
                "stress_cash_interest_usdt": float(cash_interest[1]),
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "capacity_pass": True,
                "active_position_count": int(np.count_nonzero(np.abs(positions) > 1e-12)),
            }
        )

    terminal_count = int(np.count_nonzero(np.abs(positions) > 1e-12))
    if terminal_count:
        raise DataContractError("V11末端七日容量减仓后仍有非零当季期货持仓")
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
            trade_date=pd.to_datetime(intervals["interval_open_time"])
            .astype("datetime64[ns]")
            .dt.normalize()
        )
        .groupby("trade_date", observed=True, as_index=False)
        .last()
    )
    if len(daily_last) != len(context):
        raise DataContractError("V11日度净值与评价日历不一致")
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
        raise DataContractError("V11日度净值存在缺口")
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
        "daily_rebalance_trade_hour_count": daily_rebalance_count,
        "delivery_blackout_trade_hour_count": delivery_blackout_count,
        "intraday_risk_deleveraging_hour_count": risk_deleverage_count,
        "terminal_winddown_trade_hour_count": terminal_trade_count,
        "capacity_continuation_trade_hour_count": capacity_continuation_count,
        "global_capacity_sliced_hour_count": global_slice_count,
        "minimum_trade_skipped_order_count": minimum_trade_skip_count,
        "maximum_order_capacity_fraction": maximum_order_fraction,
        "maximum_gross_exposure": maximum_gross,
        "maximum_absolute_net_exposure": maximum_net,
        "maximum_intrabar_combined_loss_fraction": maximum_intrabar_loss_fraction,
        "base_total_transaction_cost_usdt": float(total_transaction_cost[0]),
        "stress_total_transaction_cost_usdt": float(total_transaction_cost[1]),
        "base_total_cash_interest_usdt": float(total_cash_interest[0]),
        "stress_total_cash_interest_usdt": float(total_cash_interest[1]),
        "funding_rate_applied": 0.0,
        "terminal_position_count": terminal_count,
        "spot_position_count": 0,
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
            "# 数字资产当季期货基差因子V11可见期结果",
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
    spot, futures, fx, benchmark, input_audit = load_visible_inputs(contract)
    targets, features, signal_audit = build_daily_targets(spot, futures, contract)
    daily, trades, portfolio_audit, context_audit = run_portfolio_backtest(
        targets, futures, fx, benchmark, contract
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
        "report_id": "DIGITAL_ASSET_CURRENT_QUARTER_BASIS_FACTOR_V11_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": (
            "VISIBLE_PASS_40PCT_HIGH_SHARPE_RESEARCH_SCREEN_ONLY"
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
            "selection_feature_rows": int(len(features)),
        },
        "evaluation": evaluation,
        "historical_evidence_limits": contract["historical_evidence_limits"],
        "decision": {
            "external_forward_validation_required": passed,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "paper_or_live_authorized": False,
            "next_step": (
                "可见期全部通过；锁定V11并另设真正前瞻复验，当前不生成任何交易信号"
                if passed
                else "冻结拒绝V11，不修改5日基差、两资产、0.45/0.45、成本、交割日、容量或风险门救回"
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
