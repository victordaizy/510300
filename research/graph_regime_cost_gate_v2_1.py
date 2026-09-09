"""图形状态策略V2.1：只拦截显式成本过高的买入或加仓。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from research.graph_regime_martin_turtle_v2 import (
    CONFIG_FILE as SOURCE_CONFIG_FILE,
    VARIANTS,
    _cost_model,
    _events_by_date,
    _target_order,
)


ROOT = Path(__file__).resolve().parents[1]
COST_GATE_CONFIG_FILE = ROOT / "config" / "graph_regime_cost_gate_v2_1.yaml"


def load_cost_gate_config(path: Path = COST_GATE_CONFIG_FILE) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    gate = config["cost_gate"]
    if float(gate["maximum_one_way_explicit_cost_bps"]) != 10.0:
        raise ValueError("V2.1成本门必须固定为单边10bp")
    if float(gate["derived_minimum_buy_notional_cny"]) != 10_000.0:
        raise ValueError("V2.1最低买入金额必须固定为10000元")
    if gate["sell_and_exposure_reduction_exempt"] is not True:
        raise ValueError("卖出和减仓必须豁免成本门")
    return config


def load_source_config(path: Path = SOURCE_CONFIG_FILE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def preview_buy_cost(
    *,
    target_exposure: float,
    raw_open: float,
    cash: float,
    shares: int,
    receivable: float,
    lot_size: int,
    costs: Any,
) -> dict[str, Any] | None:
    """按整手和现金约束预演实际可成交买单，再计算显式成本率。"""

    wealth_open = cash + shares * raw_open + receivable
    desired = int(
        np.floor(target_exposure * wealth_open / raw_open / lot_size) * lot_size
    )
    desired = max(0, desired)
    delta = desired - shares
    if delta <= 0:
        return None
    execution_price = raw_open * (1.0 + costs.slippage_rate * costs.multiplier)
    while delta > 0:
        raw_notional = delta * raw_open
        commission = costs.commission(raw_notional)
        if delta * execution_price + commission <= cash + 1e-9:
            break
        delta -= lot_size
    if delta <= 0:
        return {
            "shares": 0,
            "raw_notional": 0.0,
            "commission": 0.0,
            "slippage_cost": 0.0,
            "one_way_explicit_cost_bps": np.inf,
        }
    raw_notional = delta * raw_open
    commission = costs.commission(raw_notional)
    slippage_cost = delta * (execution_price - raw_open)
    return {
        "shares": delta,
        "raw_notional": raw_notional,
        "commission": commission,
        "slippage_cost": slippage_cost,
        "one_way_explicit_cost_bps": 10_000.0
        * (commission + slippage_cost)
        / raw_notional,
    }


def simulate_variant_cost_gated(
    features: pd.DataFrame,
    dividends: pd.DataFrame,
    source_config: dict[str, Any],
    cost_gate_config: dict[str, Any],
    variant_id: str,
    *,
    cost_multiplier: float = 1.0,
) -> dict[str, pd.DataFrame]:
    if variant_id not in VARIANTS:
        raise KeyError(f"未知V2轨道：{variant_id}")
    variant = VARIANTS[variant_id]
    execution = source_config["price_and_execution"]
    gate = cost_gate_config["cost_gate"]
    maximum_cost_bps = float(gate["maximum_one_way_explicit_cost_bps"])
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
    martin_layer = 0
    blocked_martin_layers: set[int] = set()
    last_layer_adjusted_price = np.nan
    module_holding_days = 0
    module_invalidation = np.nan
    used_bottom_events: set[str] = set()
    cycle: dict[str, Any] | None = None
    cycles: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    blocked_orders: list[dict[str, Any]] = []

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
        blocked_reason = None
        day_commission = 0.0
        day_slippage = 0.0
        day_notional = 0.0
        if pending is not None:
            preview = preview_buy_cost(
                target_exposure=float(pending["target_exposure"]),
                raw_open=float(row["open"]),
                cash=cash,
                shares=shares,
                receivable=receivable,
                lot_size=lot_size,
                costs=costs,
            )
            is_blocked = (
                preview is not None
                and float(preview["one_way_explicit_cost_bps"]) > maximum_cost_bps
            )
            if is_blocked:
                blocked_reason = pending["reason"]
                blocked_orders.append(
                    {
                        "variant_id": variant_id,
                        "signal_date": pending["signal_date"],
                        "decision_date": date,
                        "reason": pending["reason"],
                        "target_exposure": pending["target_exposure"],
                        "mode_before": mode_before,
                        "intended_buy_shares": preview["shares"],
                        "raw_open": float(row["open"]),
                        "intended_raw_notional": preview["raw_notional"],
                        "estimated_commission": preview["commission"],
                        "estimated_slippage_cost": preview["slippage_cost"],
                        "one_way_explicit_cost_bps": preview[
                            "one_way_explicit_cost_bps"
                        ],
                        "maximum_allowed_bps": maximum_cost_bps,
                        "bottom_event_id": pending.get("bottom_event_id"),
                        "action": "BLOCK_BUY_KEEP_EXISTING_SHARES",
                    }
                )
                if pending["reason"].startswith("MARTINGALE_LAYER_"):
                    blocked_martin_layers.add(int(pending["reason"].rsplit("_", 1)[-1]))
                if pending["reason"] == "TURTLE_ENTRY":
                    turtle_rearmed = False
                if (
                    shares > 0
                    and pending["reason"] in {"MARTIN_TO_TURTLE", "PRING_TO_TURTLE"}
                    and gate["blocked_transition_changes_management_state_without_trade"]
                ):
                    position_mode = "TURTLE_LONG"
                    executed_reason = f"{pending['reason']}_COST_GATE_NO_BUY"
                    martin_layer = 0
                    blocked_martin_layers.clear()
                    module_holding_days = 0
                    module_invalidation = np.nan
            else:
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
                if reason in {"TURTLE_ENTRY", "MARTIN_TO_TURTLE", "PRING_TO_TURTLE"}:
                    if shares > 0:
                        position_mode = "TURTLE_LONG"
                        martin_layer = 0
                        blocked_martin_layers.clear()
                        module_holding_days = 0
                        module_invalidation = np.nan
                elif reason.startswith("MARTINGALE_LAYER_") and trade is not None:
                    position_mode = "MARTINGALE"
                    martin_layer = int(reason.rsplit("_", 1)[-1])
                    module_holding_days = 0 if martin_layer == 1 else module_holding_days
                    module_invalidation = float(pending["invalidation_price"])
                    if trade["side"] == "BUY":
                        last_layer_adjusted_price = float(trade["execution_price"]) * float(
                            row["adjustment_factor"]
                        )
                elif reason == "PRING_ENTRY" and trade is not None:
                    position_mode = "PRING_STYLE"
                    module_holding_days = 0
                    module_invalidation = float(pending["invalidation_price"])
                elif pending["target_exposure"] == 0.0:
                    position_mode = "CASH"
                    martin_layer = 0
                    blocked_martin_layers.clear()
                    module_holding_days = 0
                    module_invalidation = np.nan
                    last_layer_adjusted_price = np.nan
                    if reason == "TURTLE_GRAPH_EXIT":
                        turtle_rearmed = False
            pending = None

        if shares_before == 0 and shares > 0:
            cycle = {
                "variant_id": variant_id,
                "entry_date": date,
                "entry_reason": executed_reason,
                "entry_equity": cash + shares * float(row["open"]) + receivable,
                "switch_date": None,
                "switch_reason": None,
                "switch_equity": None,
            }
        elif (
            shares_before > 0
            and shares > 0
            and cycle is not None
            and executed_reason
            in {
                "MARTIN_TO_TURTLE",
                "PRING_TO_TURTLE",
                "MARTIN_TO_TURTLE_COST_GATE_NO_BUY",
                "PRING_TO_TURTLE_COST_GATE_NO_BUY",
            }
        ):
            cycle["switch_date"] = date
            cycle["switch_reason"] = executed_reason
            cycle["switch_equity"] = cash + shares * float(row["open"]) + receivable
        elif shares_before > 0 and shares == 0 and cycle is not None:
            exit_equity = cash + receivable
            switch_equity = cycle.get("switch_equity")
            cycles.append(
                {
                    **cycle,
                    "exit_date": date,
                    "exit_reason": executed_reason,
                    "exit_equity": exit_equity,
                    "net_pnl": exit_equity - float(cycle["entry_equity"]),
                    "pnl_to_switch": float(switch_equity) - float(cycle["entry_equity"])
                    if switch_equity is not None
                    else None,
                    "pnl_after_switch": exit_equity - float(switch_equity)
                    if switch_equity is not None
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
        if position_mode in {"MARTINGALE", "PRING_STYLE"}:
            module_holding_days += 1

        next_pending: dict[str, Any] | None = None
        bottom_event_id = row["bottom_divergence_active_id"]
        top_event_id = row["top_divergence_event"]
        has_bottom_event = pd.notna(bottom_event_id) and str(bottom_event_id) != ""
        has_top_event = pd.notna(top_event_id) and str(top_event_id) != ""
        breakout = bool(row["turtle_breakout"]) if pd.notna(row["turtle_breakout"]) else False
        if index + 1 < len(features):
            if position_mode == "TURTLE_LONG":
                if variant["graph_exit"] and has_top_event:
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
            elif position_mode in {"MARTINGALE", "PRING_STYLE"}:
                if variant["turtle"] and breakout:
                    next_pending = {
                        "signal_date": date,
                        "target_exposure": 1.0,
                        "reason": "MARTIN_TO_TURTLE"
                        if position_mode == "MARTINGALE"
                        else "PRING_TO_TURTLE",
                    }
                else:
                    exit_reason = None
                    if not variant["turtle"] and breakout:
                        exit_reason = "CONTROL_BREAKOUT_EXIT"
                    elif has_top_event:
                        exit_reason = "MODULE_TOP_DIVERGENCE_EXIT"
                    elif bool(row["trend_down"]):
                        exit_reason = "MODULE_TREND_DOWN_EXIT"
                    elif np.isfinite(module_invalidation) and row["adjusted_close"] < module_invalidation:
                        exit_reason = "MODULE_INVALIDATION_EXIT"
                    elif row["adjusted_close"] >= row["ma28"]:
                        exit_reason = "MODULE_MA28_EXIT"
                    elif module_holding_days >= 20:
                        exit_reason = "MODULE_TIME_EXIT"
                    if exit_reason:
                        next_pending = {
                            "signal_date": date,
                            "target_exposure": 0.0,
                            "reason": exit_reason,
                            "top_event_id": top_event_id,
                        }
                    elif (
                        position_mode == "MARTINGALE"
                        and martin_layer < 3
                        and martin_layer + 1 not in blocked_martin_layers
                        and bool(row["range_eligible"])
                        and np.isfinite(last_layer_adjusted_price)
                        and row["adjusted_close"]
                        <= last_layer_adjusted_price - row["atr14"]
                    ):
                        layer = martin_layer + 1
                        next_pending = {
                            "signal_date": date,
                            "target_exposure": [0.125, 0.375, 0.875][layer - 1],
                            "reason": f"MARTINGALE_LAYER_{layer}",
                            "invalidation_price": module_invalidation,
                        }
            else:
                if variant["turtle"] and turtle_rearmed and breakout:
                    next_pending = {
                        "signal_date": date,
                        "target_exposure": 1.0,
                        "reason": "TURTLE_ENTRY",
                    }
                elif (
                    variant["module"] in {"MARTINGALE", "PRING_STYLE"}
                    and bool(row["range_eligible"])
                    and bool(row["bottom_divergence_active"])
                    and has_bottom_event
                    and bottom_event_id not in used_bottom_events
                ):
                    if variant["module"] == "MARTINGALE":
                        next_pending = {
                            "signal_date": date,
                            "target_exposure": 0.125,
                            "reason": "MARTINGALE_LAYER_1",
                            "bottom_event_id": bottom_event_id,
                            "invalidation_price": float(row["bottom_divergence_invalidation"]),
                        }
                    elif row["bias28"] < -10.0:
                        next_pending = {
                            "signal_date": date,
                            "target_exposure": 0.50,
                            "reason": "PRING_ENTRY",
                            "bottom_event_id": bottom_event_id,
                            "invalidation_price": float(row["bottom_divergence_invalidation"]),
                        }
                    if next_pending is not None:
                        used_bottom_events.add(str(bottom_event_id))
        pending = next_pending
        ledger.append(
            {
                "variant_id": variant_id,
                "date": date,
                "raw_open": float(row["open"]),
                "raw_close": float(row["close"]),
                "adjusted_close": float(row["adjusted_close"]),
                "adjustment_factor": float(row["adjustment_factor"]),
                "chart_state": row["chart_state"],
                "position_mode": position_mode,
                "turtle_rearmed": turtle_rearmed,
                "range_eligible": bool(row["range_eligible"]),
                "trend_down": bool(row["trend_down"]),
                "turtle_breakout": bool(row["turtle_breakout"]),
                "turtle_regular_exit": bool(row["turtle_regular_exit"]),
                "bottom_divergence_event": row["bottom_divergence_event"],
                "bottom_divergence_active_id": bottom_event_id,
                "top_divergence_event": top_event_id,
                "bias28": row["bias28"],
                "atr14": row["atr14"],
                "er20": row["er20"],
                "shares": shares,
                "cash": cash,
                "receivable": receivable,
                "equity": equity,
                "exposure": exposure,
                "day_commission": day_commission,
                "day_slippage": day_slippage,
                "day_raw_notional": day_notional,
                "executed_reason": executed_reason,
                "blocked_reason": blocked_reason,
                "next_signal_reason": next_pending["reason"] if next_pending else None,
                "next_target_exposure": next_pending["target_exposure"]
                if next_pending
                else np.nan,
                "martin_layer": martin_layer,
                "module_holding_days": module_holding_days,
            }
        )
    return {
        "ledger": pd.DataFrame(ledger),
        "executions": pd.DataFrame(executions),
        "cycles": pd.DataFrame(cycles),
        "blocked_orders": pd.DataFrame(blocked_orders),
    }
