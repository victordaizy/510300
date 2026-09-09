"""以当季期货基差排序、只交易现货多头的V12。"""

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
from research.digital_asset_current_quarter_basis_factor_v11 import delivery_days
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
CONFIG = ROOT / "config" / "digital_asset_basis_ranked_spot_long_v12.yaml"
FIXED_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("基差排序现货单多V12配置必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    failures: list[str] = []
    protocol = contract.get("protocol", {})
    if protocol.get("candidate_id") != "DIGITAL_ASSET_BASIS_RANKED_SPOT_LONG_V12":
        failures.append("candidate_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V22":
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
    if int(signal.get("selected_count", 0)) != 1:
        failures.append("selected_count")
    if float(signal.get("target_long_gross", math.nan)) != 0.90:
        failures.append("target_long_gross")
    if float(signal.get("target_cash_fraction", math.nan)) != 0.10:
        failures.append("target_cash_fraction")
    if signal.get("short_position_allowed") is not False or signal.get("derivative_position_allowed") is not False:
        failures.append("spot_long_only")
    if contract.get("delivery", {}).get("delivery_day_position_policy") != "CASH_FROM_00UTC_THROUGH_23UTC":
        failures.append("delivery_policy")
    portfolio = contract.get("portfolio", {})
    if portfolio.get("quantity_increment") != {"BTCUSDT": 0.001, "ETHUSDT": 0.001}:
        failures.append("quantity_increment")
    if portfolio.get("terminal_winddown_start") != "2026-08-08 00:00:00":
        failures.append("terminal_winddown_start")
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
    if any(
        bool(risk.get(key))
        for key in (
            "account_borrowing_allowed",
            "account_margin_allowed",
            "short_sale_allowed",
            "derivative_position_allowed",
        )
    ):
        failures.append("risk_long_only")
    gates = contract.get("visible_gates", {})
    if float(gates.get("minimum_annualized_net_excess", math.nan)) != 0.40:
        failures.append("minimum_annualized_net_excess")
    if float(gates.get("minimum_strategy_net_sharpe", math.nan)) != 1.50:
        failures.append("minimum_strategy_net_sharpe")
    if any(bool(value) for value in contract.get("safety", {}).values()):
        failures.append("safety")
    if failures:
        raise ValueError(f"基差排序现货单多V12配置被弱化或损坏：{sorted(set(failures))}")


def _spot_cost_rate(contract: dict[str, Any], scenario: str) -> float:
    return float(contract["costs"][f"{scenario}_total_spot_cost_bps_per_leg"]) / 10000.0


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
    if status.get("status") != "PASS_INPUT_ACQUISITION_ONLY" or any(
        bool(status.get(key)) for key in forbidden
    ):
        raise DataContractError("V12复用的V11输入采集状态不合格")
    paths = {
        "spot_1h": ROOT / inputs["spot_1h"],
        "current_quarter_1h": ROOT / inputs["current_quarter_1h"],
        "fx": ROOT / inputs["visible_fx"],
        "benchmark": ROOT / inputs["visible_benchmark"],
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少V12输入：{key}={path}")
        receipt = status.get("outputs", {}).get(key, {})
        if receipt.get("path") != path.relative_to(ROOT).as_posix() or receipt.get("sha256") != _sha256(path):
            raise DataContractError(f"V12输入与采集回执不一致：{key}")
    spot = _normalize_time(pd.read_parquet(paths["spot_1h"]), "open_time")
    futures = _normalize_time(pd.read_parquet(paths["current_quarter_1h"]), "open_time")
    fx = pd.read_parquet(paths["fx"])
    benchmark = pd.read_parquet(paths["benchmark"])
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce").astype("datetime64[ns]").dt.normalize()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce").astype("datetime64[ns]").dt.normalize()
    symbols = list(contract["universe"]["fixed_symbols"])
    for label, frame in (("spot", spot), ("current_quarter", futures)):
        if sorted(frame["symbol"].astype(str).unique()) != sorted(symbols):
            raise DataContractError(f"V12 {label}产品集合不正确")
        if frame.duplicated(["open_time", "symbol"]).any():
            raise DataContractError(f"V12 {label}存在重复产品时点")
        numeric = frame[["open", "high", "low", "close", "quote_volume"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise DataContractError(f"V12 {label}含缺失或非有限数值")
        if not (numeric[["open", "high", "low", "close"]] > 0.0).all().all():
            raise DataContractError(f"V12 {label}价格非正")
    daily_times = pd.date_range(
        pd.Timestamp(contract["historical_partition"]["visible_start"]),
        pd.Timestamp(contract["historical_partition"]["visible_end"]),
        freq="D",
    )
    for label, frame in (("spot", spot), ("current_quarter", futures)):
        for hour in (0, 23):
            required = daily_times + pd.Timedelta(hours=hour)
            counts = frame.loc[frame["open_time"].isin(required)].groupby(
                "open_time", observed=True
            )["symbol"].nunique()
            if len(counts) != len(required) or not counts.eq(len(symbols)).all():
                raise DataContractError(f"V12 {label}每日{hour:02d}点覆盖不完整")
    audit = {
        "source_status": status["status"],
        "source_candidate_id": status.get("candidate_id"),
        "spot_rows": int(len(spot)),
        "current_quarter_rows": int(len(futures)),
        "fx_rows": int(len(fx)),
        "benchmark_rows": int(len(benchmark)),
        "symbol_count": len(symbols),
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
    spot_close = (
        _normalize_time(spot, "open_time")
        .loc[lambda frame: frame["open_time"].dt.hour.eq(23)]
        .pivot(index="open_time", columns="symbol", values="close")
        .reindex(columns=symbols)
        .sort_index()
        .astype(float)
    )
    futures_close = (
        _normalize_time(futures, "open_time")
        .loc[lambda frame: frame["open_time"].dt.hour.eq(23)]
        .pivot(index="open_time", columns="symbol", values="close")
        .reindex(columns=symbols)
        .sort_index()
        .astype(float)
    )
    common = spot_close.index.intersection(futures_close.index)
    basis = np.log(spot_close.reindex(common) / futures_close.reindex(common))
    lookback = int(contract["signal"]["basis_lookback_calendar_days"])
    mean_basis = basis.rolling(lookback, min_periods=lookback).mean()
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    winddown = pd.Timestamp(contract["portfolio"]["terminal_winddown_start"])
    executions = pd.date_range(start, winddown - pd.Timedelta(days=1), freq="1D")
    delivery = set(delivery_days(contract))
    target_weight = float(contract["signal"]["target_long_gross"])
    target_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    delivery_count = 0
    for execution_time in executions:
        signal_time = execution_time - pd.Timedelta(hours=1)
        if signal_time not in mean_basis.index:
            raise DataContractError(f"V12 {execution_time}缺少23点基差信号")
        current = basis.loc[signal_time]
        averaged = mean_basis.loc[signal_time]
        if current.isna().any() or averaged.isna().any():
            raise DataContractError(f"V12 {execution_time}不具备完整5日基差历史")
        ranked = sorted(symbols, key=lambda symbol: (-float(averaged[symbol]), symbol))
        chosen = ranked[0]
        is_delivery = execution_time.normalize() in delivery
        if is_delivery:
            delivery_count += 1
        for rank, symbol in enumerate(ranked, start=1):
            weight = 0.0 if is_delivery or symbol != chosen else target_weight
            target_rows.append(
                {
                    "signal_time": signal_time,
                    "execution_time": execution_time,
                    "symbol": symbol,
                    "target_weight": weight,
                    "mean_log_basis_5d": float(averaged[symbol]),
                    "delivery_blackout": is_delivery,
                }
            )
            feature_rows.append(
                {
                    "signal_time": signal_time,
                    "execution_time": execution_time,
                    "symbol": symbol,
                    "daily_log_basis": float(current[symbol]),
                    "mean_log_basis_5d": float(averaged[symbol]),
                    "mean_basis_rank_desc": rank,
                    "selected": bool(not is_delivery and symbol == chosen),
                    "delivery_blackout": is_delivery,
                }
            )
    targets = pd.DataFrame(target_rows).sort_values(["execution_time", "symbol"]).reset_index(drop=True)
    features = pd.DataFrame(feature_rows).sort_values(["execution_time", "mean_basis_rank_desc"]).reset_index(drop=True)
    grouped = targets.groupby("execution_time", observed=True)
    gross = grouped["target_weight"].sum()
    blackout = grouped["delivery_blackout"].first()
    if not np.allclose(gross.loc[~blackout], target_weight) or not np.allclose(
        gross.loc[blackout], 0.0
    ):
        raise DataContractError("V12现货目标权重或交割日现金目标不正确")
    audit = {
        "daily_signal_count": int(len(executions)),
        "delivery_blackout_day_count": delivery_count,
        "target_row_count": int(len(targets)),
        "feature_row_count": int(len(features)),
        "basis_lookback_calendar_days": lookback,
        "strict_execution_lag_hours": 1,
        "selected_count_each_non_delivery_day": 1,
        "tie_break": "SYMBOL_ASCENDING",
        "short_or_derivative_position_generated": False,
        "future_price_or_return_used_in_signal": False,
    }
    return targets, features, audit


def _daily_spot_panels(
    spot: pd.DataFrame,
    symbols: list[str],
    days: pd.DatetimeIndex,
    lookback: int,
) -> dict[str, np.ndarray]:
    data = _normalize_time(spot, "open_time").sort_values(["symbol", "open_time"])
    opens = (
        data.pivot(index="open_time", columns="symbol", values="open")
        .reindex(index=days, columns=symbols)
        .astype(float)
    )
    closes_23 = (
        data.pivot(index="open_time", columns="symbol", values="close")
        .reindex(index=days + pd.Timedelta(hours=23), columns=symbols)
        .astype(float)
    )
    closes_23.index = days
    lows = (
        data.assign(trade_date=data["open_time"].dt.normalize())
        .groupby(["trade_date", "symbol"], observed=True)["low"]
        .min()
        .unstack("symbol")
        .reindex(index=days, columns=symbols)
    )
    prior_parts: list[pd.DataFrame] = []
    for symbol, group in data.groupby("symbol", observed=True, sort=True):
        item = group[["open_time", "quote_volume"]].copy().sort_values("open_time")
        item["symbol"] = symbol
        item["prior_median_quote_volume"] = (
            item["quote_volume"].shift(1).rolling(lookback, min_periods=lookback).median()
        )
        prior_parts.append(item)
    prior = (
        pd.concat(prior_parts, ignore_index=True)
        .pivot(index="open_time", columns="symbol", values="prior_median_quote_volume")
        .reindex(index=days, columns=symbols)
    )
    frames = {"open": opens, "close_23": closes_23, "low": lows, "prior_median_quote_volume": prior}
    result: dict[str, np.ndarray] = {}
    for key, frame in frames.items():
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataContractError(f"V12日度{key}矩阵存在缺口或非有限值")
        result[key] = values
    return result


def run_portfolio_backtest(
    targets: pd.DataFrame,
    spot: pd.DataFrame,
    fx: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    end = pd.Timestamp(contract["historical_partition"]["visible_end"])
    days = pd.date_range(start, end, freq="D")
    symbols = list(contract["universe"]["fixed_symbols"])
    market = _daily_spot_panels(
        spot,
        symbols,
        days,
        int(contract["universe"]["turnover_lookback_observed_venue_hours"]),
    )
    next_open = np.empty_like(market["open"])
    next_open[:-1] = market["open"][1:]
    next_open[-1] = market["close_23"][-1]
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
    maximum_gross = maximum_order_fraction = maximum_intraday_loss = 0.0
    daily_rebalance_count = delivery_count = terminal_count_days = continuation_count = 0
    global_slice_count = minimum_trade_skip_count = 0
    total_transaction_cost = np.zeros(2, dtype=float)
    total_cash_interest = np.zeros(2, dtype=float)
    trade_rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    daily_cash_rate = (1.0 + float(contract["account"]["cash_annual_rate"])) ** (
        1.0 / 365.0
    ) - 1.0
    for day_index, timestamp in enumerate(days):
        open_prices = market["open"][day_index]
        reason: str | None = None
        target_weights = np.zeros(len(symbols), dtype=float)
        if timestamp >= winddown:
            target_positions[:] = 0.0
            reason = "TERMINAL_WINDDOWN"
        elif timestamp in target_by_time:
            target_weights = target_by_time[timestamp]
            target_positions = _truncate_toward_zero(
                target_weights * float(nav.min()) / open_prices,
                increments,
            )
            reason = (
                "DELIVERY_BLACKOUT"
                if timestamp.normalize() in delivery_set
                else "DAILY_REBALANCE"
            )
        desired = target_positions.copy()
        full_delta = desired - positions
        if reason is None and (np.abs(full_delta) > 1e-12).any():
            reason = "CAPACITY_CONTINUATION"
        full_notional = np.abs(full_delta) * open_prices
        capacities = capacity_fraction * market["prior_median_quote_volume"][day_index]
        active = full_notional > 1e-12
        slice_scale = 1.0
        if active.any():
            slice_scale = min(1.0, float(np.min(capacities[active] / full_notional[active])))
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
            raise DataContractError(f"{timestamp} V12生成了现货空头")
        order_notional = np.abs(executed_delta) * open_prices
        fractions = np.divide(
            order_notional,
            market["prior_median_quote_volume"][day_index],
            out=np.zeros_like(order_notional),
            where=market["prior_median_quote_volume"][day_index] > 0.0,
        )
        maximum_order_fraction = max(maximum_order_fraction, float(fractions.max()))
        if (fractions > capacity_fraction + 1e-10).any():
            raise DataContractError(f"{timestamp} V12订单越过冻结容量")
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
            if reason == "DAILY_REBALANCE":
                daily_rebalance_count += 1
            elif reason == "DELIVERY_BLACKOUT":
                delivery_count += 1
            elif reason == "TERMINAL_WINDDOWN":
                terminal_count_days += 1
            elif reason == "CAPACITY_CONTINUATION":
                continuation_count += 1
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
                            market["prior_median_quote_volume"][day_index, index]
                        ),
                        "capacity_fraction": float(fractions[index]),
                        "global_slice_scale": float(slice_scale),
                        "target_weight": float(target_weights[index]),
                        "position_after": float(new_positions[index]),
                    }
                )
        positions = new_positions
        if float(nav.min()) <= 0.0:
            raise DataContractError(f"{timestamp} V12交易成本后净值非正")
        spot_notional = float(np.sum(positions * open_prices))
        minimum_nav = float(nav.min())
        gross_exposure = spot_notional / minimum_nav
        maximum_gross = max(maximum_gross, gross_exposure)
        if gross_exposure > float(contract["risk"]["maximum_gross_exposure"]) + 1e-10:
            raise DataContractError(f"{timestamp} V12现货毛敞口超过冻结上限")
        cash = nav - spot_notional
        if float(cash.min()) < -1e-8:
            raise DataContractError(f"{timestamp} V12出现负现金或借款")
        worst_intraday_pnl = float(
            np.sum(positions * (market["low"][day_index] - open_prices))
        )
        maximum_intraday_loss = max(
            maximum_intraday_loss,
            -min(0.0, worst_intraday_pnl) / minimum_nav,
        )
        if minimum_nav + worst_intraday_pnl <= 0.0:
            raise DataContractError(f"{timestamp} V12日内最坏权益非正")
        price_pnl = float(np.sum(positions * (next_open[day_index] - open_prices)))
        interest = np.maximum(cash, 0.0) * daily_cash_rate
        nav += price_pnl + interest
        total_cash_interest += interest
        if float(nav.min()) <= 0.0:
            raise DataContractError(f"{timestamp} V12日结净值非正")
        day_rows.append(
            {
                "trade_date": timestamp,
                "base_nav_usdt": float(nav[0]),
                "stress_nav_usdt": float(nav[1]),
                "gross_exposure": gross_exposure,
                "net_exposure": gross_exposure,
                "capacity_pass": True,
                "active_position_count": int(np.count_nonzero(positions > 1e-12)),
            }
        )
    final_position_count = int(np.count_nonzero(positions > 1e-12))
    if final_position_count:
        raise DataContractError("V12末端七日容量减仓后仍有非零现货持仓")
    path = pd.DataFrame(day_rows)
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
    daily = context.merge(path, left_on="date", right_on="trade_date", how="left", validate="one_to_one")
    if daily[["base_nav_usdt", "stress_nav_usdt"]].isna().any().any():
        raise DataContractError("V12日度净值存在缺口")
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
        "daily_rebalance_trade_day_count": daily_rebalance_count,
        "delivery_blackout_trade_day_count": delivery_count,
        "terminal_winddown_trade_day_count": terminal_count_days,
        "capacity_continuation_trade_day_count": continuation_count,
        "global_capacity_sliced_day_count": global_slice_count,
        "minimum_trade_skipped_order_count": minimum_trade_skip_count,
        "maximum_order_capacity_fraction": maximum_order_fraction,
        "maximum_gross_exposure": maximum_gross,
        "maximum_daily_intraday_loss_fraction": maximum_intraday_loss,
        "base_total_transaction_cost_usdt": float(total_transaction_cost[0]),
        "stress_total_transaction_cost_usdt": float(total_transaction_cost[1]),
        "base_total_cash_interest_usdt": float(total_cash_interest[0]),
        "stress_total_cash_interest_usdt": float(total_cash_interest[1]),
        "terminal_position_count": final_position_count,
        "short_position_count": 0,
        "derivative_position_count": 0,
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
            "# 数字资产基差排序现货单多V12可见期结果",
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
        targets, spot, fx, benchmark, contract
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
        "report_id": "DIGITAL_ASSET_BASIS_RANKED_SPOT_LONG_V12_VISIBLE",
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
                "可见期全部通过；锁定V12并另设真正前瞻复验，当前不生成任何交易信号"
                if passed
                else "冻结拒绝V12，不修改5日基差、90%现货仓位、两资产、成本、交割日、容量或风险门救回"
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
