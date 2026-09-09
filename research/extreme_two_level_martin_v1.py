"""510300极端两级类马丁：50%首层、下跌2ATR后100%第二层。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from research.graph_regime_martin_turtle_v2 import (
    _cost_model,
    _events_by_date,
    _target_order,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "extreme_two_level_martin_v1.yaml"


VARIANTS = {
    "XM0_EXTREME_TWO_LEVEL_ONLY": {"turtle": False, "graph_exit": False},
    "XS1_EXTREME_TWO_LEVEL_TURTLE_SWITCH": {"turtle": True, "graph_exit": True},
}


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if [item["id"] for item in config["candidates"]] != list(VARIANTS):
        raise ValueError("类马丁候选集合与冻结实现不一致")
    return config


def simulate_variant(
    features: pd.DataFrame,
    dividends: pd.DataFrame,
    source_config: dict[str, Any],
    variant_id: str,
    *,
    cost_multiplier: float = 1.0,
) -> dict[str, pd.DataFrame]:
    if variant_id not in VARIANTS:
        raise KeyError(f"未知类马丁轨道：{variant_id}")
    variant = VARIANTS[variant_id]
    execution = source_config["price_and_execution"]
    lot_size = int(execution["lot_size_shares"])
    daily_cash_rate = (1.0 + float(execution["cash_annual_rate"])) ** (
        1.0 / int(execution["trading_days_per_year"])
    ) - 1.0
    costs = _cost_model(source_config, cost_multiplier)
    record_events = _events_by_date(dividends, "record_date")
    ex_events = _events_by_date(dividends, "ex_date")
    payment_events = _events_by_date(dividends, "payment_date")
    entitlements: dict[pd.Timestamp, float] = {}
    receivables_by_payment: dict[pd.Timestamp, float] = defaultdict(float)

    cash = float(execution["initial_capital_cny"])
    shares = 0
    receivable = 0.0
    pending: dict[str, Any] | None = None
    position_mode = "CASH"
    turtle_rearmed = True
    layer = 0
    first_layer_adjusted_price = np.nan
    fixed_entry_atr = np.nan
    hard_invalidation = np.nan
    holding_days = 0
    used_bottom_events: set[str] = set()
    cycle: dict[str, Any] | None = None
    ledger: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []

    for index, row in features.iterrows():
        date = row["date"]
        for event in ex_events.get(date, []):
            amount = entitlements.get(event.ex_date, 0.0)
            if amount:
                receivable += amount
                receivables_by_payment[event.payment_date] += amount
        if date in payment_events:
            amount = receivables_by_payment.pop(date, 0.0)
            cash += amount
            receivable -= amount

        shares_before = shares
        mode_before = position_mode
        executed_reason = None
        day_commission = 0.0
        day_slippage = 0.0
        day_notional = 0.0
        if pending is not None:
            cash, shares, trade = _target_order(
                target_exposure=float(pending["target_exposure"]),
                raw_open=float(row["open"]),
                cash=cash,
                shares=shares,
                receivable=receivable,
                lot_size=lot_size,
                costs=costs,
            )
            executed_reason = pending["reason"]
            if trade is not None:
                day_commission = float(trade["commission"])
                day_slippage = float(trade["slippage_cost"])
                day_notional = float(trade["raw_notional"])
                executions.append(
                    {
                        "variant_id": variant_id,
                        "signal_date": pending["signal_date"],
                        "execution_date": date,
                        "reason": pending["reason"],
                        "target_exposure": pending["target_exposure"],
                        "mode_before": mode_before,
                        "side": trade["side"],
                        "trade_shares": trade["shares"],
                        "raw_open": float(row["open"]),
                        "execution_price": trade["execution_price"],
                        "raw_notional": trade["raw_notional"],
                        "commission": trade["commission"],
                        "slippage_cost": trade["slippage_cost"],
                        "shares_after": shares,
                        "bottom_event_id": pending.get("bottom_event_id"),
                        "top_event_id": pending.get("top_event_id"),
                    }
                )
            reason = pending["reason"]
            if reason in {"TURTLE_ENTRY", "EXTREME_TO_TURTLE"} and shares > 0:
                position_mode = "TURTLE_LONG"
                layer = 0
                holding_days = 0
            elif reason == "EXTREME_LAYER_1" and trade is not None:
                position_mode = "EXTREME_MARTIN"
                layer = 1
                holding_days = 0
                fixed_entry_atr = float(pending["entry_atr"])
                first_layer_adjusted_price = float(trade["execution_price"]) * float(
                    row["adjustment_factor"]
                )
                hard_invalidation = first_layer_adjusted_price - 3.0 * fixed_entry_atr
            elif reason == "EXTREME_LAYER_2" and trade is not None:
                position_mode = "EXTREME_MARTIN"
                layer = 2
            elif pending["target_exposure"] == 0.0:
                position_mode = "CASH"
                layer = 0
                holding_days = 0
                first_layer_adjusted_price = np.nan
                fixed_entry_atr = np.nan
                hard_invalidation = np.nan
                if reason == "TURTLE_GRAPH_EXIT":
                    turtle_rearmed = False
            pending = None

        if shares_before == 0 and shares > 0:
            cycle = {
                "variant_id": variant_id,
                "entry_date": date,
                "entry_reason": executed_reason,
                "entry_equity": cash + shares * float(row["open"]) + receivable,
                "full_position_date": None,
                "full_position_reason": None,
                "full_position_equity": None,
                "fixed_entry_atr": fixed_entry_atr,
                "first_layer_adjusted_price": first_layer_adjusted_price,
                "hard_invalidation": hard_invalidation,
            }
        elif (
            shares_before > 0
            and shares > 0
            and cycle is not None
            and executed_reason in {"EXTREME_LAYER_2", "EXTREME_TO_TURTLE"}
        ):
            cycle["full_position_date"] = date
            cycle["full_position_reason"] = executed_reason
            cycle["full_position_equity"] = cash + shares * float(row["open"]) + receivable
        elif shares_before > 0 and shares == 0 and cycle is not None:
            exit_equity = cash + receivable
            full_equity = cycle.get("full_position_equity")
            cycles.append(
                {
                    **cycle,
                    "exit_date": date,
                    "exit_reason": executed_reason,
                    "exit_equity": exit_equity,
                    "net_pnl": exit_equity - float(cycle["entry_equity"]),
                    "pnl_before_full": float(full_equity) - float(cycle["entry_equity"])
                    if full_equity is not None
                    else None,
                    "pnl_after_full": exit_equity - float(full_equity)
                    if full_equity is not None
                    else None,
                    "holding_calendar_days": (date - cycle["entry_date"]).days,
                }
            )
            cycle = None

        cash *= 1.0 + daily_cash_rate
        equity = cash + shares * float(row["close"]) + receivable
        exposure = shares * float(row["close"]) / equity if equity > 0 else np.nan
        for event in record_events.get(date, []):
            entitlements[event.ex_date] = shares * float(event.cash_dividend_per_share)
        if not turtle_rearmed and pd.notna(row["ma20"]) and row["adjusted_close"] <= row["ma20"]:
            turtle_rearmed = True
        if position_mode == "EXTREME_MARTIN":
            holding_days += 1

        bottom_event_id = row["bottom_divergence_active_id"]
        top_event_id = row["top_divergence_event"]
        has_bottom = pd.notna(bottom_event_id) and str(bottom_event_id) != ""
        has_top = pd.notna(top_event_id) and str(top_event_id) != ""
        breakout = bool(row["turtle_breakout"]) if pd.notna(row["turtle_breakout"]) else False
        next_pending: dict[str, Any] | None = None
        if index + 1 < len(features):
            if position_mode == "TURTLE_LONG":
                if variant["graph_exit"] and has_top:
                    next_pending = {
                        "signal_date": date,
                        "target_exposure": 0.0,
                        "reason": "TURTLE_GRAPH_EXIT",
                        "top_event_id": top_event_id,
                    }
                elif bool(row["turtle_regular_exit"]):
                    next_pending = {
                        "signal_date": date,
                        "target_exposure": 0.0,
                        "reason": "TURTLE_REGULAR_EXIT",
                    }
            elif position_mode == "EXTREME_MARTIN":
                if variant["turtle"] and breakout:
                    next_pending = {
                        "signal_date": date,
                        "target_exposure": 1.0,
                        "reason": "EXTREME_TO_TURTLE",
                    }
                else:
                    exit_reason = None
                    if not variant["turtle"] and breakout:
                        exit_reason = "CONTROL_BREAKOUT_EXIT"
                    elif has_top:
                        exit_reason = "EXTREME_TOP_DIVERGENCE_EXIT"
                    elif bool(row["trend_down"]):
                        exit_reason = "EXTREME_TREND_DOWN_EXIT"
                    elif np.isfinite(hard_invalidation) and row["adjusted_close"] < hard_invalidation:
                        exit_reason = "EXTREME_HARD_INVALIDATION_EXIT"
                    elif row["adjusted_close"] >= row["ma28"]:
                        exit_reason = "EXTREME_MA28_EXIT"
                    elif holding_days >= 60:
                        exit_reason = "EXTREME_TIME_EXIT"
                    if exit_reason:
                        next_pending = {
                            "signal_date": date,
                            "target_exposure": 0.0,
                            "reason": exit_reason,
                            "top_event_id": top_event_id,
                        }
                    elif (
                        layer == 1
                        and bool(row["range_eligible"])
                        and row["adjusted_close"]
                        <= first_layer_adjusted_price - 2.0 * fixed_entry_atr
                    ):
                        next_pending = {
                            "signal_date": date,
                            "target_exposure": 1.0,
                            "reason": "EXTREME_LAYER_2",
                        }
            else:
                if variant["turtle"] and turtle_rearmed and breakout:
                    next_pending = {
                        "signal_date": date,
                        "target_exposure": 1.0,
                        "reason": "TURTLE_ENTRY",
                    }
                elif (
                    bool(row["range_eligible"])
                    and bool(row["bottom_divergence_active"])
                    and has_bottom
                    and bottom_event_id not in used_bottom_events
                ):
                    next_pending = {
                        "signal_date": date,
                        "target_exposure": 0.50,
                        "reason": "EXTREME_LAYER_1",
                        "bottom_event_id": bottom_event_id,
                        "entry_atr": float(row["atr14"]),
                    }
                    used_bottom_events.add(str(bottom_event_id))
        pending = next_pending
        ledger.append(
            {
                "variant_id": variant_id,
                "date": date,
                "raw_open": float(row["open"]),
                "raw_close": float(row["close"]),
                "adjusted_close": float(row["adjusted_close"]),
                "chart_state": row["chart_state"],
                "position_mode": position_mode,
                "layer": layer,
                "shares": shares,
                "cash": cash,
                "receivable": receivable,
                "equity": equity,
                "exposure": exposure,
                "day_commission": day_commission,
                "day_slippage": day_slippage,
                "day_raw_notional": day_notional,
                "executed_reason": executed_reason,
                "next_signal_reason": next_pending["reason"] if next_pending else None,
                "holding_days": holding_days,
                "first_layer_adjusted_price": first_layer_adjusted_price,
                "fixed_entry_atr": fixed_entry_atr,
                "hard_invalidation": hard_invalidation,
                "bottom_divergence_active_id": bottom_event_id,
                "top_divergence_event": top_event_id,
                "range_eligible": bool(row["range_eligible"]),
                "trend_down": bool(row["trend_down"]),
                "turtle_breakout": bool(row["turtle_breakout"]),
                "turtle_regular_exit": bool(row["turtle_regular_exit"]),
            }
        )
    return {
        "ledger": pd.DataFrame(ledger),
        "executions": pd.DataFrame(executions),
        "cycles": pd.DataFrame(cycles),
    }
