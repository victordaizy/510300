"""围绕固定前收盘锚、保持隔夜核心库存不变的日内做T引擎。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from backtest.intraday_t_engine import (
    CostModel,
    commission_for_notional,
    maximum_affordable_lot,
    maximum_sell_first_lot_with_repurchase_reserve,
    roundtrip_cost_fraction,
    unfavorable_execution_price,
)


@dataclass(frozen=True)
class AnchorTResult:
    scenario: str
    strategy_name: str
    ledger: pd.DataFrame
    trades: pd.DataFrame


def build_fixed_anchor_features(
    minute: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """构造只依赖当时及更早数据的固定前收盘锚与分钟反转特征。"""

    minute_required = {"trade_time", "open", "high", "low", "close", "vol", "amount"}
    if missing := minute_required - set(minute.columns):
        raise ValueError(f"一分钟数据缺少字段：{sorted(missing)}")
    if missing := {"date", "close"} - set(daily.columns):
        raise ValueError(f"日线数据缺少字段：{sorted(missing)}")
    result = minute.copy()
    result["trade_time"] = pd.to_datetime(result["trade_time"])
    result = result.sort_values("trade_time").reset_index(drop=True)
    result["trade_date"] = result["trade_time"].dt.normalize()
    grouped = result.groupby("trade_date", sort=False)
    result["record_number"] = grouped.cumcount()
    result["previous_record_close"] = grouped["close"].shift(1)
    result["return_3_records"] = grouped["close"].pct_change(3, fill_method=None)
    result["record_time"] = result["trade_time"].dt.time

    anchors = daily[["date", "close"]].copy()
    anchors["date"] = pd.to_datetime(anchors["date"]).dt.normalize()
    anchors = anchors.sort_values("date").drop_duplicates("date")
    anchors["anchor_close"] = anchors["close"].shift(1)
    anchors = anchors.rename(columns={"date": "trade_date"})[["trade_date", "anchor_close"]]
    result = result.merge(anchors, on="trade_date", how="left", validate="many_to_one")
    result["anchor_deviation"] = result["close"] / result["anchor_close"] - 1.0
    return result


def _cost_model(config: dict[str, Any]) -> CostModel:
    costs = config["costs"]
    return CostModel(
        commission_rate=float(costs["commission_rate"]),
        minimum_commission_cny=float(costs["minimum_commission_cny"]),
        stamp_duty_rate=float(costs["stamp_duty_rate"]),
        price_tick_cny=float(costs["price_tick_cny"]),
        base_slippage_bps_per_leg=float(costs["base_slippage_bps_per_leg"]),
    )


def _in_window(record_time: time, windows: list[dict[str, str]]) -> bool:
    return any(
        time.fromisoformat(item["start"]) <= record_time <= time.fromisoformat(item["end"])
        for item in windows
    )


def _direction(row: dict[str, Any], threshold: float) -> str | None:
    deviation = float(row["anchor_deviation"])
    return_3 = float(row["return_3_records"])
    close = float(row["close"])
    previous = float(row["previous_record_close"])
    if deviation <= -threshold and return_3 > 0.0 and close > previous:
        return "BUY_LOW"
    if deviation >= threshold and return_3 < 0.0 and close < previous:
        return "SELL_HIGH"
    return None


def _find_fixed_anchor_exit(
    day: list[dict[str, Any]],
    entry_position: int,
    direction: str,
    entry_execution_price: float,
    anchor: float,
    config: dict[str, Any],
) -> tuple[int, int, str]:
    stop = float(config["exit"]["stop_loss_fraction_from_entry_execution"])
    maximum_holding = int(config["exit"]["maximum_holding_trading_records"])
    forced_time = time.fromisoformat(config["exit"]["forced_exit_signal_time"])
    for signal_position in range(entry_position, len(day) - 1):
        row = day[signal_position]
        close = float(row["close"])
        holding = signal_position - entry_position + 1
        if direction == "BUY_LOW":
            stop_hit = (entry_execution_price - close) / entry_execution_price >= stop
            target_hit = close >= anchor
        else:
            stop_hit = (close - entry_execution_price) / entry_execution_price >= stop
            target_hit = close <= anchor
        if stop_hit:
            return signal_position, signal_position + 1, "STOP_LOSS"
        if target_hit:
            return signal_position, signal_position + 1, "FIXED_ANCHOR_TARGET"
        if holding >= maximum_holding:
            return signal_position, signal_position + 1, "MAX_HOLDING"
        if row["record_time"] >= forced_time:
            return signal_position, signal_position + 1, "FORCED_EXIT"
    raise RuntimeError("日内T未能在最后一条记录前形成退出成交")


def _execute_day(
    day: list[dict[str, Any]],
    available_cash: float,
    old_shares: int,
    config: dict[str, Any],
    model: CostModel,
    slippage_bps: float,
) -> dict[str, Any] | None:
    account = config["account"]
    entry = config["entry"]
    minimum_shares = int(account["minimum_t_shares"])
    maximum_shares = min(int(account["maximum_t_shares"]), int(old_shares))
    lot_size = int(account["lot_size"])
    if maximum_shares < minimum_shares:
        return None
    windows = entry["signal_windows"]
    edge_multiple = float(entry["expected_anchor_distance_to_roundtrip_cost_multiple"])
    revalidation = float(entry["next_open_displacement_revalidation_fraction"])
    volume_ratio = float(entry["maximum_order_to_signal_record_volume_ratio"])
    stop = float(config["exit"]["stop_loss_fraction_from_entry_execution"])
    repurchase_reserve = float(entry.get("sell_first_repurchase_reserve_fraction", stop))

    for signal_position in range(3, len(day) - 1):
        signal = day[signal_position]
        if not _in_window(signal["record_time"], windows):
            continue
        values = np.asarray(
            [
                signal["anchor_close"],
                signal["anchor_deviation"],
                signal["return_3_records"],
                signal["previous_record_close"],
                signal["close"],
                signal["vol"],
            ],
            dtype=float,
        )
        if not np.isfinite(values).all() or float(signal["vol"]) <= 0:
            continue
        signal_price = float(signal["close"])
        buy_size = maximum_affordable_lot(
            available_cash,
            signal_price,
            maximum_shares,
            minimum_shares,
            lot_size,
            model,
            model.base_slippage_bps_per_leg,
        )
        sell_size = maximum_sell_first_lot_with_repurchase_reserve(
            available_cash,
            signal_price,
            maximum_shares,
            minimum_shares,
            lot_size,
            repurchase_reserve,
            model,
            model.base_slippage_bps_per_leg,
        )
        provisional_size = max(buy_size, sell_size)
        if provisional_size < minimum_shares:
            continue
        provisional_threshold = edge_multiple * roundtrip_cost_fraction(
            signal_price, provisional_size, model
        )
        direction = _direction(signal, provisional_threshold)
        if direction is None:
            continue
        planned_size = buy_size if direction == "BUY_LOW" else sell_size
        if planned_size < minimum_shares:
            continue
        threshold = edge_multiple * roundtrip_cost_fraction(signal_price, planned_size, model)
        if _direction(signal, threshold) != direction:
            continue
        if planned_size / float(signal["vol"]) > volume_ratio:
            continue

        entry_position = signal_position + 1
        entry_record = day[entry_position]
        entry_raw = float(entry_record["open"])
        if direction == "BUY_LOW":
            actual_size = maximum_affordable_lot(
                available_cash,
                entry_raw,
                maximum_shares,
                minimum_shares,
                lot_size,
                model,
                slippage_bps,
            )
        else:
            actual_size = maximum_sell_first_lot_with_repurchase_reserve(
                available_cash,
                entry_raw,
                maximum_shares,
                minimum_shares,
                lot_size,
                repurchase_reserve,
                model,
                slippage_bps,
            )
        if actual_size < minimum_shares:
            continue
        threshold = edge_multiple * roundtrip_cost_fraction(signal_price, actual_size, model)
        next_deviation = entry_raw / float(signal["anchor_close"]) - 1.0
        if direction == "BUY_LOW" and next_deviation > -threshold * revalidation:
            continue
        if direction == "SELL_HIGH" and next_deviation < threshold * revalidation:
            continue
        if actual_size / float(signal["vol"]) > volume_ratio:
            continue

        entry_side = "BUY" if direction == "BUY_LOW" else "SELL"
        entry_price = unfavorable_execution_price(
            entry_raw, entry_side, slippage_bps, model.price_tick_cny
        )
        entry_notional = entry_price * actual_size
        entry_commission = commission_for_notional(
            entry_notional, model.commission_rate, model.minimum_commission_cny
        )
        entry_stamp = entry_notional * model.stamp_duty_rate if entry_side == "SELL" else 0.0
        if direction == "BUY_LOW" and entry_notional + entry_commission > available_cash + 1e-9:
            continue

        exit_signal_position, exit_position, exit_reason = _find_fixed_anchor_exit(
            day,
            entry_position,
            direction,
            entry_price,
            float(signal["anchor_close"]),
            config,
        )
        exit_signal = day[exit_signal_position]
        exit_record = day[exit_position]
        exit_raw = float(exit_record["open"])
        exit_side = "SELL" if direction == "BUY_LOW" else "BUY"
        exit_price = unfavorable_execution_price(
            exit_raw, exit_side, slippage_bps, model.price_tick_cny
        )
        exit_notional = exit_price * actual_size
        exit_commission = commission_for_notional(
            exit_notional, model.commission_rate, model.minimum_commission_cny
        )
        exit_stamp = exit_notional * model.stamp_duty_rate if exit_side == "SELL" else 0.0
        if direction == "SELL_HIGH":
            cash_after_entry = available_cash + entry_notional - entry_commission - entry_stamp
            if exit_notional + exit_commission > cash_after_entry + 1e-9:
                raise RuntimeError("卖旧买回的退出资金超过保守预留")
            raw_pnl = (entry_raw - exit_raw) * actual_size
            execution_pnl = (entry_price - exit_price) * actual_size
        else:
            raw_pnl = (exit_raw - entry_raw) * actual_size
            execution_pnl = (exit_price - entry_price) * actual_size
        commission = entry_commission + exit_commission
        stamp_duty = entry_stamp + exit_stamp
        net_pnl = execution_pnl - commission - stamp_duty
        return {
            "trade_date": pd.Timestamp(signal["trade_date"]),
            "direction": direction,
            "sequence": (
                "BUY_NEW_THEN_SELL_OLD" if direction == "BUY_LOW" else "SELL_OLD_THEN_BUY_NEW"
            ),
            "shares": int(actual_size),
            "anchor_close_cny": float(signal["anchor_close"]),
            "signal_deviation": float(signal["anchor_deviation"]),
            "dynamic_threshold": threshold,
            "entry_signal_time": pd.Timestamp(signal["trade_time"]),
            "entry_execution_time": pd.Timestamp(entry_record["trade_time"]),
            "entry_raw_open_cny": entry_raw,
            "entry_execution_price_cny": entry_price,
            "exit_signal_time": pd.Timestamp(exit_signal["trade_time"]),
            "exit_execution_time": pd.Timestamp(exit_record["trade_time"]),
            "exit_raw_open_cny": exit_raw,
            "exit_execution_price_cny": exit_price,
            "exit_reason": exit_reason,
            "holding_trading_records": int(exit_signal_position - entry_position + 1),
            "raw_gross_pnl_cny": raw_pnl,
            "execution_gross_pnl_cny": execution_pnl,
            "commission_cny": commission,
            "stamp_duty_cny": stamp_duty,
            "slippage_and_tick_cost_cny": raw_pnl - execution_pnl,
            "net_pnl_cny": net_pnl,
            "old_shares_available": int(old_shares),
            "cash_available_cny": float(available_cash),
            "signal_record_volume_shares": int(signal["vol"]),
            "order_to_signal_volume_ratio": actual_size / float(signal["vol"]),
        }
    return None


def run_anchor_t_overlay(
    minute_features: pd.DataFrame,
    base_ledger: pd.DataFrame,
    core_trade_dates: set[pd.Timestamp],
    config: dict[str, Any],
    scenario: str,
    slippage_bps: float,
    strategy_name: str,
) -> AnchorTResult:
    """在不改变每日核心份额的前提下叠加最多一个日内T回合。"""

    required = {"date", "shares", "cash", "equity", "actual_position"}
    if missing := required - set(base_ledger.columns):
        raise ValueError(f"核心净值表缺少字段：{sorted(missing)}")
    ledger = base_ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"]).dt.normalize()
    ledger = ledger.sort_values("date").reset_index(drop=True)
    by_date = ledger.set_index("date")
    model = _cost_model(config)
    cumulative_t_pnl = 0.0
    rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    normalized_trade_dates = {pd.Timestamp(value).normalize() for value in core_trade_dates}

    for date_value, day_frame in minute_features.groupby("trade_date", sort=True):
        date = pd.Timestamp(date_value).normalize()
        if date not in by_date.index:
            continue
        base = by_date.loc[date]
        trade = None
        skipped_for_core_trade = date in normalized_trade_dates
        if not skipped_for_core_trade:
            available_cash = float(base["cash"]) + cumulative_t_pnl
            if available_cash >= 0.0:
                trade = _execute_day(
                    day_frame.to_dict("records"),
                    available_cash,
                    int(base["shares"]),
                    config,
                    model,
                    slippage_bps,
                )
        if trade is not None:
            trade["scenario"] = scenario
            trade["strategy_name"] = strategy_name
            trade_rows.append(trade)
            cumulative_t_pnl += float(trade["net_pnl_cny"])
        combined_equity = float(base["equity"]) + cumulative_t_pnl
        rows.append(
            {
                "date": date,
                "base_equity": float(base["equity"]),
                "equity": combined_equity,
                "base_shares": int(base["shares"]),
                "base_cash": float(base["cash"]),
                "base_actual_position": float(base["actual_position"]),
                "round_trip_today": int(trade is not None),
                "skipped_for_core_trade": skipped_for_core_trade,
                "cumulative_t_net_pnl_cny": cumulative_t_pnl,
            }
        )
    output = pd.DataFrame(rows)
    output["daily_return"] = output["equity"].pct_change(fill_method=None).fillna(0.0)
    output["drawdown"] = output["equity"] / output["equity"].cummax() - 1.0
    trades = pd.DataFrame(trade_rows)
    return AnchorTResult(scenario, strategy_name, output, trades)


def summarize_anchor_t(
    result: AnchorTResult,
    initial_cash: float,
    trading_days: int = 242,
) -> dict[str, Any]:
    ledger = result.ledger
    trades = result.trades
    elapsed_days = max((ledger["date"].iloc[-1] - ledger["date"].iloc[0]).days, 1)
    total_return = float(ledger["equity"].iloc[-1] / initial_cash - 1.0)
    cagr = float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)
    volatility = float(ledger["daily_return"].std(ddof=1) * np.sqrt(trading_days))
    if trades.empty:
        t_metrics = {
            "round_trip_count": 0,
            "raw_gross_pnl_cny": 0.0,
            "net_pnl_cny": 0.0,
            "commission_cny": 0.0,
            "slippage_and_tick_cost_cny": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
    else:
        wins = trades.loc[trades["net_pnl_cny"].gt(0.0), "net_pnl_cny"].sum()
        losses = -trades.loc[trades["net_pnl_cny"].lt(0.0), "net_pnl_cny"].sum()
        t_metrics = {
            "round_trip_count": int(len(trades)),
            "raw_gross_pnl_cny": float(trades["raw_gross_pnl_cny"].sum()),
            "net_pnl_cny": float(trades["net_pnl_cny"].sum()),
            "commission_cny": float(trades["commission_cny"].sum()),
            "slippage_and_tick_cost_cny": float(trades["slippage_and_tick_cost_cny"].sum()),
            "win_rate": float(trades["net_pnl_cny"].gt(0.0).mean()),
            "profit_factor": float(wins / losses) if losses > 0 else math.inf,
        }
    return {
        "strategy_name": result.strategy_name,
        "scenario": result.scenario,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": (
            float(ledger["daily_return"].mean() * trading_days / volatility)
            if volatility > 0
            else None
        ),
        "max_drawdown": float(ledger["drawdown"].min()),
        "average_base_exposure": float(ledger["base_actual_position"].mean()),
        "ending_equity": float(ledger["equity"].iloc[-1]),
        "intraday_t": t_metrics,
    }
