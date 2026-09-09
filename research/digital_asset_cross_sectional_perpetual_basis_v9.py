"""现货生成信号、仅交易永续的日频横截面基差因子V9。"""

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
    _cost_rate,
    _exact_symbol_grid,
    _fx_spread,
    _market_panels,
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
CONFIG = ROOT / "config" / "digital_asset_cross_sectional_perpetual_basis_v9.yaml"


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("永续基差因子V9配置必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    failures: list[str] = []
    protocol = contract.get("protocol", {})
    if protocol.get("candidate_id") != "DIGITAL_ASSET_CROSS_SECTIONAL_PERPETUAL_BASIS_V9":
        failures.append("candidate_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V19":
        failures.append("parent_protocol_id")
    account = contract.get("account", {})
    if float(account.get("initial_capital_cny", math.nan)) != 500000.0:
        failures.append("initial_capital_cny")
    if float(account.get("user_transaction_fee_rate_per_leg", math.nan)) != 0.0001:
        failures.append("user_transaction_fee_rate_per_leg")
    universe = contract.get("universe", {})
    if universe.get("fixed_symbols") != FIXED_SYMBOLS:
        failures.append("fixed_symbols")
    if int(universe.get("turnover_lookback_hours", 0)) != 168:
        failures.append("turnover_lookback_hours")
    if float(universe.get("maximum_order_fraction_of_prior_median_quote_volume", math.nan)) != 0.001:
        failures.append("maximum_order_fraction")
    signal = contract.get("signal", {})
    fixed_signal = {
        "basis_formula": "SPOT_CLOSE_DIV_PERPETUAL_CLOSE_MINUS_1",
        "long_count": 3,
        "short_count": 3,
        "long_selection": "HIGHEST_BASIS_THEN_SYMBOL_ASCENDING",
        "short_selection": "LOWEST_BASIS_THEN_SYMBOL_ASCENDING_EXCLUDING_LONGS",
        "long_side_target_gross": 0.5,
        "short_side_target_gross": 0.5,
        "maximum_target_gross_exposure": 1.0,
        "target_net_exposure": 0.0,
        "spot_position_allowed": False,
    }
    for key, expected in fixed_signal.items():
        value = signal.get(key)
        if isinstance(expected, bool):
            if value is not expected:
                failures.append(key)
        elif isinstance(expected, int):
            if int(value or 0) != expected:
                failures.append(key)
        elif isinstance(expected, float):
            if float(value if value is not None else math.nan) != expected:
                failures.append(key)
        elif value != expected:
            failures.append(key)
    portfolio = contract.get("portfolio", {})
    if float(portfolio.get("stress_cost_reserve_multiplier", math.nan)) != 0.98:
        failures.append("stress_cost_reserve_multiplier")
    if portfolio.get("intraday_risk_only_proportional_deleveraging") is not True:
        failures.append("intraday_risk_deleveraging")
    if float(portfolio.get("intraday_risk_deleveraging_gross_trigger", math.nan)) != 0.99:
        failures.append("risk_gross_trigger")
    if float(portfolio.get("intraday_risk_deleveraging_absolute_net_trigger", math.nan)) != 0.029:
        failures.append("risk_net_trigger")
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
    if float(risk.get("maximum_gross_exposure", math.nan)) != 1.0:
        failures.append("maximum_gross_exposure")
    if float(risk.get("maximum_absolute_net_exposure", math.nan)) != 0.03:
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
        raise ValueError(f"永续基差因子V9配置被弱化或损坏：{sorted(set(failures))}")


def load_visible_inputs(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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
        raise DataContractError("V9原始采集状态不是严格的输入采集通过")
    paths = {
        "spot_1h": ROOT / inputs["spot_1h"],
        "perpetual_1h": ROOT / inputs["perpetual_1h"],
        "funding_rates": ROOT / inputs["funding_rates"],
        "fx": ROOT / inputs["visible_fx"],
        "benchmark": ROOT / inputs["visible_benchmark"],
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少V9输入：{key}={path}")
        receipt = status.get("outputs", {}).get(key, {})
        if receipt.get("path") != path.relative_to(ROOT).as_posix() or receipt.get("sha256") != _sha256(path):
            raise DataContractError(f"V9输入与采集回执不一致：{key}")
    spot = _normalize_time(pd.read_parquet(paths["spot_1h"]), "open_time")
    perpetual = _normalize_time(pd.read_parquet(paths["perpetual_1h"]), "open_time")
    funding = _normalize_time(pd.read_parquet(paths["funding_rates"]), "funding_time")
    fx = pd.read_parquet(paths["fx"])
    benchmark = pd.read_parquet(paths["benchmark"])
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce").astype("datetime64[ns]").dt.normalize()
    benchmark["date"] = (
        pd.to_datetime(benchmark["date"], errors="coerce")
        .astype("datetime64[ns]")
        .dt.normalize()
    )
    symbols = list(contract["universe"]["fixed_symbols"])
    source_start = pd.Timestamp(contract["historical_partition"]["acquisition_start"])
    visible_end = pd.Timestamp(f"{contract['historical_partition']['visible_end']} 23:00:00")
    funding_end = pd.Timestamp(f"{contract['historical_partition']['visible_end']} 16:00:00")
    _exact_symbol_grid(
        perpetual,
        time_column="open_time",
        symbols=symbols,
        expected=pd.date_range(source_start, visible_end, freq="1h"),
        label="V9永续小时K线",
    )
    _exact_symbol_grid(
        funding,
        time_column="funding_time",
        symbols=symbols,
        expected=pd.date_range(source_start, funding_end, freq="8h"),
        label="V9资金费率",
    )
    if sorted(spot["symbol"].astype(str).unique()) != sorted(symbols):
        raise DataContractError("V9现货产品集合不正确")
    observed_grids: list[pd.DatetimeIndex] = []
    for symbol in symbols:
        item = spot.loc[spot["symbol"].eq(symbol), "open_time"]
        if item.duplicated().any() or len(item) != int(
            contract["prefreeze_coverage_gate"]["required_spot_hourly_rows_per_symbol"]
        ):
            raise DataContractError(f"V9现货 {symbol}覆盖或重复不正确")
        observed_grids.append(pd.DatetimeIndex(item).sort_values().unique())
    if not all(grid.equals(observed_grids[0]) for grid in observed_grids[1:]):
        raise DataContractError("V9现货缺口不是15标的同步停市")
    missing = pd.date_range(source_start, visible_end, freq="1h").difference(observed_grids[0])
    if len(missing) != int(
        contract["prefreeze_coverage_gate"]["required_common_venue_wide_spot_missing_hours"]
    ) or any(timestamp.hour in (23, 0) for timestamp in missing):
        raise DataContractError("V9现货同步缺口数量变化或触及关键决策时点")
    signal_times = pd.date_range(
        pd.Timestamp(contract["historical_partition"]["visible_start"]) - pd.Timedelta(hours=1),
        pd.Timestamp(contract["portfolio"]["terminal_winddown_start"]) - pd.Timedelta(hours=25),
        freq="1D",
    )
    for label, frame in (("现货", spot), ("永续", perpetual)):
        observed = frame.loc[frame["open_time"].isin(signal_times)]
        counts = observed.groupby("open_time", observed=True)["symbol"].nunique()
        if len(counts) != len(signal_times) or not counts.eq(len(symbols)).all():
            raise DataContractError(f"V9{label}每日23点信号覆盖不完整")
    for label, frame, price_columns in (
        ("现货", spot, ["open", "high", "low", "close"]),
        ("永续", perpetual, ["open", "high", "low", "close"]),
    ):
        numeric = frame[price_columns].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise DataContractError(f"V9{label}价格含缺失或非有限值")
        if not (numeric > 0.0).all().all():
            raise DataContractError(f"V9{label}价格非正")
    rates = pd.to_numeric(funding["funding_rate"], errors="coerce")
    if rates.isna().any() or not np.isfinite(rates.to_numpy(dtype=float)).all():
        raise DataContractError("V9资金费率含缺失或非有限值")
    audit = {
        "source_status": status["status"],
        "spot_rows": int(len(spot)),
        "perpetual_rows": int(len(perpetual)),
        "funding_rows": int(len(funding)),
        "fx_rows": int(len(fx)),
        "benchmark_rows": int(len(benchmark)),
        "symbol_count": len(symbols),
        "venue_wide_spot_missing_hour_count": int(len(missing)),
        "missing_signal_or_execution_hour_count": 0,
        "missing_values_backfilled": False,
        "input_sha256": {key: _sha256(path) for key, path in paths.items()},
        "future_price_or_strategy_return_read": True,
        "sealed_replication_read": False,
        "authenticated_endpoint_used": False,
    }
    return spot, perpetual, funding, fx, benchmark, audit


def build_daily_targets(
    spot: pd.DataFrame,
    perpetual: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    symbols = sorted(contract["universe"]["fixed_symbols"])
    spot_data = _normalize_time(spot, "open_time")
    perpetual_data = _normalize_time(perpetual, "open_time")
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    winddown = pd.Timestamp(contract["portfolio"]["terminal_winddown_start"])
    signal_times = pd.date_range(start - pd.Timedelta(hours=1), winddown - pd.Timedelta(hours=25), freq="1D")
    spot_close = (
        spot_data.loc[spot_data["open_time"].isin(signal_times)]
        .pivot(index="open_time", columns="symbol", values="close")
        .reindex(index=signal_times, columns=symbols)
    )
    perpetual_close = (
        perpetual_data.loc[perpetual_data["open_time"].isin(signal_times)]
        .pivot(index="open_time", columns="symbol", values="close")
        .reindex(index=signal_times, columns=symbols)
    )
    if spot_close.isna().any().any() or perpetual_close.isna().any().any():
        raise DataContractError("V9每日23点基差输入不完整")
    basis = spot_close.to_numpy(dtype=float) / perpetual_close.to_numpy(dtype=float) - 1.0
    if not np.isfinite(basis).all():
        raise DataContractError("V9基差信号含非有限值")
    high_order = np.argsort(-basis, axis=1, kind="stable")
    long_count = int(contract["signal"]["long_count"])
    short_count = int(contract["signal"]["short_count"])
    long_selected = high_order[:, :long_count]
    short_scores = basis.copy()
    np.put_along_axis(short_scores, long_selected, np.inf, axis=1)
    short_selected = np.argsort(short_scores, axis=1, kind="stable")[:, :short_count]
    if any(set(a).intersection(set(b)) for a, b in zip(long_selected.tolist(), short_selected.tolist())):
        raise DataContractError("V9多空选择发生重叠")
    selected = np.concatenate([long_selected, short_selected], axis=1)
    symbol_array = np.asarray(symbols, dtype=object)
    executions = (signal_times + pd.Timedelta(hours=1)).to_numpy(dtype="datetime64[ns]")
    signals = signal_times.to_numpy(dtype="datetime64[ns]")
    weights = np.tile(
        np.concatenate(
            [
                np.full(long_count, 0.5 / long_count),
                np.full(short_count, -0.5 / short_count),
            ]
        ),
        len(signal_times),
    )
    targets = pd.DataFrame(
        {
            "signal_time": np.repeat(signals, long_count + short_count),
            "execution_time": np.repeat(executions, long_count + short_count),
            "symbol": symbol_array[selected.reshape(-1)],
            "target_weight": weights,
        }
    ).sort_values(["execution_time", "symbol"]).reset_index(drop=True)
    high_rank = np.empty_like(high_order)
    low_order = np.argsort(basis, axis=1, kind="stable")
    low_rank = np.empty_like(low_order)
    ranks = np.broadcast_to(np.arange(1, len(symbols) + 1), high_order.shape)
    np.put_along_axis(high_rank, high_order, ranks, axis=1)
    np.put_along_axis(low_rank, low_order, ranks, axis=1)
    long_mask = np.zeros(basis.shape, dtype=bool)
    short_mask = np.zeros(basis.shape, dtype=bool)
    np.put_along_axis(long_mask, long_selected, True, axis=1)
    np.put_along_axis(short_mask, short_selected, True, axis=1)
    selection = np.full(basis.shape, "NONE", dtype=object)
    selection[long_mask] = "LONG"
    selection[short_mask] = "SHORT"
    features = pd.DataFrame(
        {
            "signal_time": np.repeat(signals, len(symbols)),
            "execution_time": np.repeat(executions, len(symbols)),
            "symbol": np.tile(symbol_array, len(signal_times)),
            "basis": basis.reshape(-1),
            "high_basis_rank": high_rank.reshape(-1),
            "low_basis_rank": low_rank.reshape(-1),
            "selection": selection.reshape(-1),
        }
    )
    grouped = targets.groupby("execution_time", observed=True)["target_weight"]
    long_sums = grouped.apply(lambda values: float(values[values > 0.0].sum()))
    short_sums = grouped.apply(lambda values: float(values[values < 0.0].sum()))
    if not np.allclose(long_sums, 0.5) or not np.allclose(short_sums, -0.5):
        raise DataContractError("V9多空目标权重未分别归一至正负0.5")
    audit = {
        "daily_signal_count": int(len(signal_times)),
        "target_row_count": int(len(targets)),
        "feature_row_count": int(len(features)),
        "first_signal_time": signal_times.min().isoformat(),
        "last_signal_time": signal_times.max().isoformat(),
        "first_execution_time": pd.Timestamp(targets["execution_time"].min()).isoformat(),
        "last_execution_time": pd.Timestamp(targets["execution_time"].max()).isoformat(),
        "strict_execution_lag_hours": 1,
        "long_count_each_day": long_count,
        "short_count_each_day": short_count,
        "tie_break": "SYMBOL_ASCENDING",
        "spot_position_generated": False,
        "next_day_return_used_in_signal": False,
        "sealed_replication_read": False,
    }
    return targets, features, audit


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
    panels = _market_panels(
        perpetual,
        symbols,
        int(contract["universe"]["turnover_lookback_hours"]),
    )
    market: dict[str, np.ndarray] = {}
    for name in ("open", "high", "low", "close", "prior_median_quote_volume"):
        values = panels[name].reindex(times).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataContractError(f"V9可见期{name}矩阵存在缺口或非有限值")
        market[name] = values
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
    funding_data = _normalize_time(funding, "funding_time")
    rates_panel = (
        funding_data.pivot(index="funding_time", columns="symbol", values="funding_rate")
        .reindex(columns=symbols)
        .sort_index()
    )
    marks_panel = (
        funding_data.pivot(index="funding_time", columns="symbol", values="mark_price")
        .reindex(columns=symbols)
        .sort_index()
    )
    settlements = times + pd.Timedelta(hours=1)
    funding_rates = rates_panel.reindex(settlements).to_numpy(dtype=float)
    funding_marks = marks_panel.reindex(settlements).to_numpy(dtype=float)
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
    reserve = float(contract["portfolio"]["stress_cost_reserve_multiplier"])
    capacity_fraction = float(
        contract["universe"]["maximum_order_fraction_of_prior_median_quote_volume"]
    )
    minimum_trade = float(contract["portfolio"]["minimum_trade_notional_usdt"])
    winddown = pd.Timestamp(contract["portfolio"]["terminal_winddown_start"])
    gross_trigger = float(contract["portfolio"]["intraday_risk_deleveraging_gross_trigger"])
    net_trigger = float(contract["portfolio"]["intraday_risk_deleveraging_absolute_net_trigger"])
    maximum_gross = maximum_net = maximum_order_fraction = 0.0
    maximum_intrabar_loss_fraction = 0.0
    daily_rebalance_count = risk_deleverage_count = terminal_trade_count = 0
    global_slice_count = minimum_trade_skip_count = 0
    total_transaction_cost = np.zeros(2, dtype=float)
    total_funding = np.zeros(2, dtype=float)
    trade_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    cash_hourly_rate = (1.0 + float(contract["account"]["cash_annual_rate"])) ** (1.0 / 8760.0) - 1.0

    for time_index, timestamp in enumerate(times):
        open_prices = market["open"][time_index]
        minimum_nav_before_trade = float(nav.min())
        current_gross = float(np.sum(np.abs(positions) * open_prices)) / minimum_nav_before_trade
        current_net = float(np.sum(positions * open_prices)) / minimum_nav_before_trade
        reason: str | None = None
        target_weights = np.zeros(len(symbols), dtype=float)
        if timestamp >= winddown:
            desired = np.zeros(len(symbols), dtype=float)
            reason = "TERMINAL_WINDDOWN"
        elif timestamp in target_by_time:
            target_weights = target_by_time[timestamp]
            desired = _truncate_toward_zero(
                target_weights * minimum_nav_before_trade * reserve / open_prices,
                increments,
            )
            reason = "DAILY_REBALANCE"
        elif current_gross > gross_trigger or abs(current_net) > net_trigger:
            scale = 1.0
            if current_gross > 0.0:
                scale = min(scale, gross_trigger / current_gross)
            if abs(current_net) > 0.0:
                scale = min(scale, net_trigger / abs(current_net))
            desired = _truncate_toward_zero(positions * scale, increments)
            reason = "INTRADAY_RISK_DELEVERAGE"
        else:
            desired = positions.copy()
        full_delta = desired - positions
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
        order_notional = np.abs(executed_delta) * open_prices
        if reason == "DAILY_REBALANCE":
            small = (order_notional > 1e-12) & (order_notional < minimum_trade)
            minimum_trade_skip_count += int(small.sum())
            executed_delta[small] = 0.0
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
            raise DataContractError(f"{timestamp} V9订单越过冻结容量")
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
            elif reason == "INTRADAY_RISK_DELEVERAGE":
                risk_deleverage_count += 1
            elif reason == "TERMINAL_WINDDOWN":
                terminal_trade_count += 1
            for index in np.flatnonzero(traded):
                trade_rows.append(
                    {
                        "trade_time": timestamp,
                        "signal_time": (
                            timestamp - pd.Timedelta(hours=1)
                            if reason == "DAILY_REBALANCE"
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
            raise DataContractError(f"{timestamp} V9交易成本后净值非正")

        gross_notional = float(np.sum(np.abs(positions) * open_prices))
        net_notional = float(np.sum(positions * open_prices))
        minimum_nav = float(nav.min())
        gross_exposure = gross_notional / minimum_nav
        net_exposure = net_notional / minimum_nav
        maximum_gross = max(maximum_gross, gross_exposure)
        maximum_net = max(maximum_net, abs(net_exposure))
        if gross_exposure > float(contract["risk"]["maximum_gross_exposure"]) + 1e-10:
            raise DataContractError(f"{timestamp} V9风险减仓后实际毛敞口仍超过冻结上限")
        if abs(net_exposure) > float(contract["risk"]["maximum_absolute_net_exposure"]) + 1e-10:
            raise DataContractError(f"{timestamp} V9风险减仓后实际净敞口仍超过冻结上限")
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
        maintenance = (
            float(contract["risk"]["minimum_combined_equity_fraction_of_gross_notional"])
            * gross_notional
        )
        maximum_intrabar_loss_fraction = max(
            maximum_intrabar_loss_fraction,
            -min(0.0, worst_intrabar_pnl) / minimum_nav,
        )
        if worst_equity <= maintenance:
            raise DataContractError(f"{timestamp} V9合并权益代理触发保证金失败")

        price_pnl = float(np.sum(positions * (next_open[time_index] - open_prices)))
        rates = funding_rates[time_index]
        marks = funding_marks[time_index]
        settlement_mask = np.isfinite(rates)
        base_funding = stress_funding = 0.0
        if settlement_mask.any():
            valid_marks = np.where(
                np.isfinite(marks) & (marks > 0.0), marks, next_open[time_index]
            )
            payments = (
                -positions[settlement_mask]
                * valid_marks[settlement_mask]
                * rates[settlement_mask]
            )
            base_funding = float(
                np.where(
                    payments >= 0.0,
                    payments * float(contract["costs"]["base_funding_receipt_multiplier"]),
                    payments * float(contract["costs"]["base_funding_payment_multiplier"]),
                ).sum()
            )
            stress_funding = float(
                np.where(
                    payments >= 0.0,
                    payments * float(contract["costs"]["stress_funding_receipt_multiplier"]),
                    payments * float(contract["costs"]["stress_funding_payment_multiplier"]),
                ).sum()
            )
        cash_interest = np.zeros(2, dtype=float)
        if not np.any(np.abs(positions) > 1e-12):
            cash_interest = nav * cash_hourly_rate
        nav += np.asarray([price_pnl + base_funding, price_pnl + stress_funding]) + cash_interest
        total_funding += np.asarray([base_funding, stress_funding])
        if float(nav.min()) <= 0.0:
            raise DataContractError(f"{timestamp} V9区间结算后净值非正")
        interval_rows.append(
            {
                "interval_open_time": timestamp,
                "interval_end_time": min(timestamp + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1)),
                "base_nav_usdt": float(nav[0]),
                "stress_nav_usdt": float(nav[1]),
                "price_pnl_usdt": price_pnl,
                "base_funding_usdt": base_funding,
                "stress_funding_usdt": stress_funding,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "capacity_pass": True,
                "active_position_count": int(np.count_nonzero(np.abs(positions) > 1e-12)),
            }
        )

    terminal_count = int(np.count_nonzero(np.abs(positions) > 1e-12))
    if terminal_count:
        raise DataContractError("V9末端七日容量减仓后仍有非零永续持仓")
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
        raise DataContractError("V9日度净值与评价日历不一致")
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
        raise DataContractError("V9日度净值存在缺口")
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
        "daily_rebalance_hour_count": daily_rebalance_count,
        "intraday_risk_deleveraging_hour_count": risk_deleverage_count,
        "terminal_winddown_trade_hour_count": terminal_trade_count,
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
        "spot_position_count": 0,
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
            "# 多币种永续基差因子V9可见期结果",
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
    spot, perpetual, funding, fx, benchmark, input_audit = load_visible_inputs(contract)
    targets, features, signal_audit = build_daily_targets(spot, perpetual, contract)
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
        "report_id": "DIGITAL_ASSET_CROSS_SECTIONAL_PERPETUAL_BASIS_V9_VISIBLE",
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
                else "冻结拒绝V9，不修改日频基差方向、多空各3、1倍总毛敞口、成本、资金费率或容量门救回"
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
