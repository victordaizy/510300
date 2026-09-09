"""510300底仓做T的无前视、T+1约束回测引擎。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "intraday_t_strategy.yaml"


@dataclass(frozen=True)
class CostModel:
    commission_rate: float
    minimum_commission_cny: float
    stamp_duty_rate: float
    price_tick_cny: float
    base_slippage_bps_per_leg: float


@dataclass
class SimulationResult:
    scenario: str
    period: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    daily_ledger: pd.DataFrame
    trades: pd.DataFrame
    metadata: dict[str, Any]


def load_strategy_config(path: Path | None = None) -> dict[str, Any]:
    config_file = path or DEFAULT_CONFIG_FILE
    with config_file.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def cost_model_from_config(config: dict[str, Any]) -> CostModel:
    costs = config["costs"]
    return CostModel(
        commission_rate=float(costs["commission_rate"]),
        minimum_commission_cny=float(costs["minimum_commission_cny"]),
        stamp_duty_rate=float(costs["stamp_duty_rate"]),
        price_tick_cny=float(costs["price_tick_cny"]),
        base_slippage_bps_per_leg=float(costs["base_slippage_bps_per_leg"]),
    )


def commission_for_notional(
    notional_cny: float,
    commission_rate: float,
    minimum_commission_cny: float,
) -> float:
    if notional_cny < 0:
        raise ValueError("成交金额不能为负数")
    if notional_cny == 0:
        return 0.0
    return round(max(notional_cny * commission_rate, minimum_commission_cny), 10)


def unfavorable_execution_price(
    raw_price_cny: float,
    side: str,
    slippage_bps_per_leg: float,
    price_tick_cny: float,
) -> float:
    if raw_price_cny <= 0 or price_tick_cny <= 0:
        raise ValueError("价格和最小价位必须为正数")
    slippage_fraction = slippage_bps_per_leg / 10_000.0
    if side == "BUY":
        adjusted = raw_price_cny * (1.0 + slippage_fraction)
        return round(math.ceil((adjusted - 1e-12) / price_tick_cny) * price_tick_cny, 6)
    if side == "SELL":
        adjusted = raw_price_cny * (1.0 - slippage_fraction)
        return round(math.floor((adjusted + 1e-12) / price_tick_cny) * price_tick_cny, 6)
    raise ValueError(f"不支持的成交方向：{side}")


def roundtrip_cost_fraction(
    signal_price_cny: float,
    shares: int,
    cost_model: CostModel,
) -> float:
    if signal_price_cny <= 0 or shares <= 0:
        raise ValueError("信号价格和交易份额必须为正数")
    one_way_notional = signal_price_cny * shares
    one_way_commission = commission_for_notional(
        one_way_notional,
        cost_model.commission_rate,
        cost_model.minimum_commission_cny,
    )
    two_leg_slippage = (
        2.0
        * one_way_notional
        * cost_model.base_slippage_bps_per_leg
        / 10_000.0
    )
    return (2.0 * one_way_commission + two_leg_slippage) / one_way_notional


def maximum_affordable_lot(
    cash_cny: float,
    raw_buy_price_cny: float,
    maximum_shares: int,
    minimum_shares: int,
    lot_size: int,
    cost_model: CostModel,
    slippage_bps_per_leg: float,
) -> int:
    if lot_size <= 0 or maximum_shares < 0 or minimum_shares < 0:
        raise ValueError("整手与份额参数无效")
    execution_price = unfavorable_execution_price(
        raw_buy_price_cny,
        "BUY",
        slippage_bps_per_leg,
        cost_model.price_tick_cny,
    )
    upper = maximum_shares - maximum_shares % lot_size
    lower = max(minimum_shares, lot_size)
    lower = int(math.ceil(lower / lot_size) * lot_size)
    for shares in range(upper, lower - 1, -lot_size):
        notional = execution_price * shares
        total_cost = notional + commission_for_notional(
            notional,
            cost_model.commission_rate,
            cost_model.minimum_commission_cny,
        )
        if total_cost <= cash_cny + 1e-9:
            return shares
    return 0


def maximum_sell_first_lot_with_repurchase_reserve(
    cash_cny: float,
    raw_sell_price_cny: float,
    maximum_shares: int,
    minimum_shares: int,
    lot_size: int,
    stop_loss_fraction: float,
    cost_model: CostModel,
    slippage_bps_per_leg: float,
) -> int:
    """按注册止损价预留买回资金，避免先卖后买后出现负现金。"""
    sell_execution_price = unfavorable_execution_price(
        raw_sell_price_cny,
        "SELL",
        slippage_bps_per_leg,
        cost_model.price_tick_cny,
    )
    reserve_raw_buy_price = sell_execution_price * (1.0 + stop_loss_fraction)
    reserve_buy_execution_price = unfavorable_execution_price(
        reserve_raw_buy_price,
        "BUY",
        slippage_bps_per_leg,
        cost_model.price_tick_cny,
    )
    upper = maximum_shares - maximum_shares % lot_size
    lower = int(math.ceil(max(minimum_shares, lot_size) / lot_size) * lot_size)
    for shares in range(upper, lower - 1, -lot_size):
        sell_notional = sell_execution_price * shares
        sell_commission = commission_for_notional(
            sell_notional,
            cost_model.commission_rate,
            cost_model.minimum_commission_cny,
        )
        sell_stamp_duty = sell_notional * cost_model.stamp_duty_rate
        available_after_sell = cash_cny + sell_notional - sell_commission - sell_stamp_duty
        reserve_buy_notional = reserve_buy_execution_price * shares
        reserve_buy_commission = commission_for_notional(
            reserve_buy_notional,
            cost_model.commission_rate,
            cost_model.minimum_commission_cny,
        )
        if reserve_buy_notional + reserve_buy_commission <= available_after_sell + 1e-9:
            return shares
    return 0


def build_intraday_features(minute: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_time", "open", "high", "low", "close", "vol", "amount"}
    missing = required - set(minute.columns)
    if missing:
        raise ValueError(f"一分钟数据缺少字段：{sorted(missing)}")
    output = minute.copy()
    output["trade_time"] = pd.to_datetime(output["trade_time"])
    output = output.sort_values("trade_time").reset_index(drop=True)
    output["trade_date"] = output["trade_time"].dt.normalize()
    grouped = output.groupby("trade_date", sort=False)
    output["record_number"] = grouped.cumcount()
    output["cumulative_volume"] = grouped["vol"].cumsum()
    output["cumulative_amount"] = grouped["amount"].cumsum()
    output["session_vwap"] = (
        output["cumulative_amount"]
        / output["cumulative_volume"].replace(0.0, np.nan)
    )
    output["previous_close"] = grouped["close"].shift(1)
    output["return_3_records"] = grouped["close"].pct_change(periods=3, fill_method=None)
    output["vwap_deviation"] = output["close"] / output["session_vwap"] - 1.0
    output["record_time"] = output["trade_time"].dt.time
    return output


def _in_signal_window(record_time: time, windows: list[dict[str, str]]) -> bool:
    return any(
        time.fromisoformat(window["start"]) <= record_time <= time.fromisoformat(window["end"])
        for window in windows
    )


def _pay_due_receivables(
    current_date: pd.Timestamp,
    receivables: list[dict[str, Any]],
) -> float:
    paid = 0.0
    for item in receivables:
        if not item["paid"] and item["payment_date"] <= current_date:
            paid += float(item["amount_cny"])
            item["paid"] = True
    return paid


def _outstanding_receivables(receivables: list[dict[str, Any]]) -> float:
    return float(sum(item["amount_cny"] for item in receivables if not item["paid"]))


def _add_dividend_entitlement(
    current_date: pd.Timestamp,
    shares: int,
    dividend_by_record_date: dict[pd.Timestamp, list[dict[str, Any]]],
    receivables: list[dict[str, Any]],
) -> float:
    added = 0.0
    for event in dividend_by_record_date.get(current_date, []):
        amount = shares * float(event["cash_dividend_per_share"])
        receivables.append(
            {
                "record_date": current_date,
                "payment_date": pd.Timestamp(event["payment_date"]).normalize(),
                "amount_cny": amount,
                "paid": False,
            }
        )
        added += amount
    return added


def _prepare_dividends(dividends: pd.DataFrame) -> dict[pd.Timestamp, list[dict[str, Any]]]:
    required = {"record_date", "payment_date", "cash_dividend_per_share"}
    missing = required - set(dividends.columns)
    if missing:
        raise ValueError(f"分红数据缺少字段：{sorted(missing)}")
    output = dividends.copy()
    output["record_date"] = pd.to_datetime(output["record_date"]).dt.normalize()
    output["payment_date"] = pd.to_datetime(output["payment_date"]).dt.normalize()
    grouped: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for row in output.to_dict("records"):
        grouped.setdefault(pd.Timestamp(row["record_date"]), []).append(row)
    return grouped


def _candidate_direction(row: pd.Series, threshold: float) -> str | None:
    if (
        row["vwap_deviation"] <= -threshold
        and row["return_3_records"] > 0.0
        and row["close"] > row["previous_close"]
    ):
        return "BUY_LOW"
    if (
        row["vwap_deviation"] >= threshold
        and row["return_3_records"] < 0.0
        and row["close"] < row["previous_close"]
    ):
        return "SELL_HIGH"
    return None


def _revalidate_displacement(
    direction: str,
    next_raw_open: float,
    signal_vwap: float,
    threshold: float,
    revalidation_fraction: float,
) -> bool:
    next_deviation = next_raw_open / signal_vwap - 1.0
    required = threshold * revalidation_fraction
    if direction == "BUY_LOW":
        return bool(next_deviation <= -required)
    return bool(next_deviation >= required)


def _find_exit(
    day: list[dict[str, Any]],
    entry_execution_position: int,
    direction: str,
    entry_execution_price: float,
    config: dict[str, Any],
) -> tuple[int, int, str]:
    exit_config = config["exit"]
    stop_loss = float(exit_config["stop_loss_fraction_from_entry_execution"])
    maximum_holding = int(exit_config["maximum_holding_trading_records"])
    forced_time = time.fromisoformat(exit_config["forced_exit_signal_time"])
    for signal_position in range(entry_execution_position, len(day) - 1):
        row = day[signal_position]
        close = float(row["close"])
        vwap = float(row["session_vwap"])
        holding_records = signal_position - entry_execution_position + 1
        if direction == "BUY_LOW":
            stop_hit = (entry_execution_price - close) / entry_execution_price >= stop_loss
            target_hit = close >= vwap
        else:
            stop_hit = (close - entry_execution_price) / entry_execution_price >= stop_loss
            target_hit = close <= vwap
        if stop_hit:
            return signal_position, signal_position + 1, "STOP_LOSS"
        if target_hit:
            return signal_position, signal_position + 1, "VWAP_TARGET"
        if holding_records >= maximum_holding:
            return signal_position, signal_position + 1, "MAX_HOLDING"
        if row["record_time"] >= forced_time:
            return signal_position, signal_position + 1, "FORCED_EXIT"
    raise RuntimeError("交易未能在当日下一条记录开盘前平仓")


def _execute_one_round_trip(
    day: list[dict[str, Any]],
    cash_cny: float,
    shares_held: int,
    sellable_old_shares: int,
    config: dict[str, Any],
    cost_model: CostModel,
    scenario_slippage_bps: float,
) -> tuple[float, int, int, dict[str, Any] | None]:
    account = config["account"]
    entry = config["entry"]
    minimum_t_shares = int(account["minimum_t_shares"])
    maximum_t_shares = int(account["maximum_t_shares"])
    lot_size = int(account["lot_size"])
    edge_multiple = float(entry["expected_edge_to_roundtrip_cost_multiple"])
    maximum_volume_ratio = float(entry["maximum_order_to_signal_record_volume_ratio"])
    revalidation_fraction = float(entry["next_open_displacement_revalidation_fraction"])
    windows = entry["signal_windows"]

    for signal_position in range(3, len(day) - 1):
        signal = day[signal_position]
        if not _in_signal_window(signal["record_time"], windows):
            continue
        signal_values = np.asarray(
            [signal["session_vwap"], signal["vwap_deviation"], signal["return_3_records"]],
            dtype=float,
        )
        if not np.isfinite(signal_values).all():
            continue
        if signal["vol"] <= 0:
            continue

        signal_price = float(signal["close"])
        buy_capacity_at_signal = maximum_affordable_lot(
            cash_cny,
            signal_price,
            maximum_t_shares,
            minimum_t_shares,
            lot_size,
            cost_model,
            cost_model.base_slippage_bps_per_leg,
        )
        direction_sizes = {
            "BUY_LOW": buy_capacity_at_signal,
            "SELL_HIGH": min(maximum_t_shares, sellable_old_shares),
        }
        provisional_shares = max(direction_sizes.values())
        if provisional_shares < minimum_t_shares:
            continue
        provisional_threshold = edge_multiple * roundtrip_cost_fraction(
            signal_price,
            provisional_shares,
            cost_model,
        )
        direction = _candidate_direction(signal, provisional_threshold)
        if direction is None:
            other_shares = direction_sizes["BUY_LOW"] if signal["vwap_deviation"] < 0 else direction_sizes["SELL_HIGH"]
            if other_shares < minimum_t_shares:
                continue
            other_threshold = edge_multiple * roundtrip_cost_fraction(
                signal_price,
                other_shares,
                cost_model,
            )
            direction = _candidate_direction(signal, other_threshold)
            if direction is None:
                continue

        planned_shares = direction_sizes[direction]
        if planned_shares < minimum_t_shares:
            continue
        threshold = edge_multiple * roundtrip_cost_fraction(signal_price, planned_shares, cost_model)
        if _candidate_direction(signal, threshold) != direction:
            continue
        if planned_shares / float(signal["vol"]) > maximum_volume_ratio:
            continue

        entry_execution_position = signal_position + 1
        entry_record = day[entry_execution_position]
        entry_raw_open = float(entry_record["open"])
        if direction == "BUY_LOW":
            actual_shares = maximum_affordable_lot(
                cash_cny,
                entry_raw_open,
                maximum_t_shares,
                minimum_t_shares,
                lot_size,
                cost_model,
                scenario_slippage_bps,
            )
        else:
            actual_shares = maximum_sell_first_lot_with_repurchase_reserve(
                cash_cny,
                entry_raw_open,
                min(maximum_t_shares, sellable_old_shares),
                minimum_t_shares,
                lot_size,
                float(config["exit"]["stop_loss_fraction_from_entry_execution"]),
                cost_model,
                scenario_slippage_bps,
            )
        if actual_shares < minimum_t_shares:
            continue
        threshold = edge_multiple * roundtrip_cost_fraction(signal_price, actual_shares, cost_model)
        if _candidate_direction(signal, threshold) != direction:
            continue
        if actual_shares / float(signal["vol"]) > maximum_volume_ratio:
            continue
        if not _revalidate_displacement(
            direction,
            entry_raw_open,
            float(signal["session_vwap"]),
            threshold,
            revalidation_fraction,
        ):
            continue

        cash_before = cash_cny
        shares_before = shares_held
        if direction == "BUY_LOW":
            entry_side = "BUY"
        else:
            entry_side = "SELL"
        entry_execution_price = unfavorable_execution_price(
            entry_raw_open,
            entry_side,
            scenario_slippage_bps,
            cost_model.price_tick_cny,
        )
        entry_notional = entry_execution_price * actual_shares
        entry_commission = commission_for_notional(
            entry_notional,
            cost_model.commission_rate,
            cost_model.minimum_commission_cny,
        )
        entry_stamp_duty = entry_notional * cost_model.stamp_duty_rate if entry_side == "SELL" else 0.0
        if direction == "BUY_LOW":
            total_debit = entry_notional + entry_commission
            if total_debit > cash_cny + 1e-9:
                continue
            cash_cny -= total_debit
            shares_held += actual_shares
        else:
            if actual_shares > sellable_old_shares:
                raise RuntimeError("先卖后买数量超过当日可卖旧底仓")
            cash_cny += entry_notional - entry_commission - entry_stamp_duty
            shares_held -= actual_shares
            sellable_old_shares -= actual_shares

        exit_signal_position, exit_execution_position, exit_reason = _find_exit(
            day,
            entry_execution_position,
            direction,
            entry_execution_price,
            config,
        )
        exit_signal = day[exit_signal_position]
        exit_record = day[exit_execution_position]
        exit_raw_open = float(exit_record["open"])
        exit_side = "SELL" if direction == "BUY_LOW" else "BUY"
        exit_execution_price = unfavorable_execution_price(
            exit_raw_open,
            exit_side,
            scenario_slippage_bps,
            cost_model.price_tick_cny,
        )
        exit_notional = exit_execution_price * actual_shares
        exit_commission = commission_for_notional(
            exit_notional,
            cost_model.commission_rate,
            cost_model.minimum_commission_cny,
        )
        exit_stamp_duty = exit_notional * cost_model.stamp_duty_rate if exit_side == "SELL" else 0.0
        if direction == "BUY_LOW":
            if actual_shares > sellable_old_shares:
                raise RuntimeError("买新卖旧时错误地试图卖出当日新买份额")
            cash_cny += exit_notional - exit_commission - exit_stamp_duty
            shares_held -= actual_shares
            sellable_old_shares -= actual_shares
            raw_gross_pnl = (exit_raw_open - entry_raw_open) * actual_shares
            execution_gross_pnl = (exit_execution_price - entry_execution_price) * actual_shares
            inventory_source = "OLD_CORE_INVENTORY"
        else:
            total_debit = exit_notional + exit_commission
            if total_debit > cash_cny + 1e-9:
                raise RuntimeError(
                    "先卖后买实际退出跳空超过预留资金："
                    f"现金{cash_cny:.2f}元，买回需要{total_debit:.2f}元，"
                    f"交易日{pd.Timestamp(signal['trade_date']).date()}"
                )
            cash_cny -= total_debit
            shares_held += actual_shares
            raw_gross_pnl = (entry_raw_open - exit_raw_open) * actual_shares
            execution_gross_pnl = (entry_execution_price - exit_execution_price) * actual_shares
            inventory_source = "OLD_CORE_INVENTORY_SOLD_AT_ENTRY"

        total_commission = entry_commission + exit_commission
        total_stamp_duty = entry_stamp_duty + exit_stamp_duty
        slippage_cost = raw_gross_pnl - execution_gross_pnl
        net_pnl = execution_gross_pnl - total_commission - total_stamp_duty
        cash_change = cash_cny - cash_before
        if not math.isclose(cash_change, net_pnl, abs_tol=1e-7):
            raise RuntimeError("交易现金变动与净损益不一致")

        trade = {
            "trade_date": pd.Timestamp(signal["trade_date"]),
            "direction": direction,
            "sequence": "BUY_NEW_THEN_SELL_OLD" if direction == "BUY_LOW" else "SELL_OLD_THEN_BUY_NEW",
            "shares": int(actual_shares),
            "entry_signal_time": pd.Timestamp(signal["trade_time"]),
            "entry_execution_time": pd.Timestamp(entry_record["trade_time"]),
            "entry_raw_open_cny": entry_raw_open,
            "entry_execution_price_cny": entry_execution_price,
            "entry_commission_cny": entry_commission,
            "entry_stamp_duty_cny": entry_stamp_duty,
            "exit_signal_time": pd.Timestamp(exit_signal["trade_time"]),
            "exit_execution_time": pd.Timestamp(exit_record["trade_time"]),
            "exit_raw_open_cny": exit_raw_open,
            "exit_execution_price_cny": exit_execution_price,
            "exit_commission_cny": exit_commission,
            "exit_stamp_duty_cny": exit_stamp_duty,
            "exit_reason": exit_reason,
            "holding_trading_records": int(exit_signal_position - entry_execution_position + 1),
            "signal_close_cny": signal_price,
            "signal_vwap_cny": float(signal["session_vwap"]),
            "signal_vwap_deviation": float(signal["vwap_deviation"]),
            "signal_return_3_records": float(signal["return_3_records"]),
            "dynamic_entry_threshold": threshold,
            "signal_record_volume_shares": int(signal["vol"]),
            "order_to_signal_volume_ratio": actual_shares / float(signal["vol"]),
            "raw_gross_pnl_cny": raw_gross_pnl,
            "execution_gross_pnl_cny": execution_gross_pnl,
            "commission_cny": total_commission,
            "stamp_duty_cny": total_stamp_duty,
            "slippage_and_tick_cost_cny": slippage_cost,
            "net_pnl_cny": net_pnl,
            "cash_before_cny": cash_before,
            "cash_after_cny": cash_cny,
            "shares_before": int(shares_before),
            "shares_after": int(shares_held),
            "sold_inventory_source": inventory_source,
        }
        return cash_cny, shares_held, sellable_old_shares, trade
    return cash_cny, shares_held, sellable_old_shares, None


def _maximum_buy_and_hold_shares(
    initial_cash_cny: float,
    raw_open_cny: float,
    lot_size: int,
    cost_model: CostModel,
    slippage_bps_per_leg: float,
) -> int:
    rough_upper = int(initial_cash_cny // raw_open_cny)
    rough_upper -= rough_upper % lot_size
    return maximum_affordable_lot(
        initial_cash_cny,
        raw_open_cny,
        rough_upper,
        lot_size,
        lot_size,
        cost_model,
        slippage_bps_per_leg,
    )


def simulate_period(
    minute: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    period: str,
    scenario: str,
    slippage_bps_per_leg: float,
) -> SimulationResult:
    required_features = {
        "trade_date",
        "record_number",
        "session_vwap",
        "previous_close",
        "return_3_records",
        "vwap_deviation",
        "record_time",
    }
    if required_features.issubset(minute.columns):
        features = minute.copy()
        features["trade_time"] = pd.to_datetime(features["trade_time"])
        features["trade_date"] = pd.to_datetime(features["trade_date"]).dt.normalize()
    else:
        features = build_intraday_features(minute)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    features = features.loc[features["trade_date"].between(start, end)].copy()
    if features.empty:
        raise ValueError(f"区间{start.date()}至{end.date()}没有一分钟数据")

    account = config["account"]
    inventory = config["inventory_constraints"]
    cost_model = cost_model_from_config(config)
    initial_cash = float(account["initial_cash_cny"])
    base_shares = int(account["base_target_shares"])
    lot_size = int(account["lot_size"])
    dividend_by_record_date = _prepare_dividends(dividends)

    first_row = features.iloc[0]
    first_raw_open = float(first_row["open"])
    acquisition_price = unfavorable_execution_price(
        first_raw_open,
        "BUY",
        slippage_bps_per_leg,
        cost_model.price_tick_cny,
    )
    acquisition_notional = acquisition_price * base_shares
    acquisition_commission = commission_for_notional(
        acquisition_notional,
        cost_model.commission_rate,
        cost_model.minimum_commission_cny,
    )
    acquisition_total = acquisition_notional + acquisition_commission
    if acquisition_total > initial_cash + 1e-9:
        raise ValueError("初始资金不足以建立注册底仓")

    strategy_cash = initial_cash - acquisition_total
    strategy_shares = base_shares
    static_cash = initial_cash - acquisition_total
    static_shares = base_shares
    full_shares = _maximum_buy_and_hold_shares(
        initial_cash,
        first_raw_open,
        lot_size,
        cost_model,
        slippage_bps_per_leg,
    )
    full_acquisition_notional = acquisition_price * full_shares
    full_acquisition_commission = commission_for_notional(
        full_acquisition_notional,
        cost_model.commission_rate,
        cost_model.minimum_commission_cny,
    )
    full_cash = initial_cash - full_acquisition_notional - full_acquisition_commission

    strategy_receivables: list[dict[str, Any]] = []
    static_receivables: list[dict[str, Any]] = []
    full_receivables: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    cumulative_t_net_pnl = 0.0
    cumulative_commission = 0.0
    cumulative_slippage = 0.0
    trade_dates = sorted(features["trade_date"].unique())
    first_trade_date = pd.Timestamp(trade_dates[0])

    for date_value, day in features.groupby("trade_date", sort=True):
        current_date = pd.Timestamp(date_value).normalize()
        day_records = day.to_dict("records")
        strategy_cash += _pay_due_receivables(current_date, strategy_receivables)
        static_cash += _pay_due_receivables(current_date, static_receivables)
        full_cash += _pay_due_receivables(current_date, full_receivables)
        sellable_old_shares = 0 if current_date == first_trade_date else base_shares
        trade: dict[str, Any] | None = None
        if current_date != first_trade_date:
            try:
                strategy_cash, strategy_shares, sellable_old_shares, trade = _execute_one_round_trip(
                    day_records,
                    strategy_cash,
                    strategy_shares,
                    sellable_old_shares,
                    config,
                    cost_model,
                    slippage_bps_per_leg,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"{scenario}/{period}/{current_date.date()}：{error}"
                ) from error
        if strategy_shares != int(inventory["end_of_day_required_shares"]):
            raise RuntimeError(f"{current_date.date()}日终份额未恢复到底仓目标")
        if strategy_cash < -1e-7:
            raise RuntimeError(f"{current_date.date()}出现负现金")

        if trade is not None:
            trade["scenario"] = scenario
            trade["period"] = period
            trade_rows.append(trade)
            cumulative_t_net_pnl += float(trade["net_pnl_cny"])
            cumulative_commission += float(trade["commission_cny"])
            cumulative_slippage += float(trade["slippage_and_tick_cost_cny"])

        strategy_dividend_added = _add_dividend_entitlement(
            current_date,
            strategy_shares,
            dividend_by_record_date,
            strategy_receivables,
        )
        static_dividend_added = _add_dividend_entitlement(
            current_date,
            static_shares,
            dividend_by_record_date,
            static_receivables,
        )
        full_dividend_added = _add_dividend_entitlement(
            current_date,
            full_shares,
            dividend_by_record_date,
            full_receivables,
        )
        close = float(day_records[-1]["close"])
        strategy_receivable = _outstanding_receivables(strategy_receivables)
        static_receivable = _outstanding_receivables(static_receivables)
        full_receivable = _outstanding_receivables(full_receivables)
        strategy_nav = strategy_cash + strategy_shares * close + strategy_receivable
        static_nav = static_cash + static_shares * close + static_receivable
        full_nav = full_cash + full_shares * close + full_receivable
        attribution_error = strategy_nav - static_nav - cumulative_t_net_pnl
        if not math.isclose(attribution_error, 0.0, abs_tol=1e-6):
            raise RuntimeError("策略净值与底仓基准、累计做T净损益之间归因不一致")

        ledger_rows.append(
            {
                "date": current_date,
                "scenario": scenario,
                "period": period,
                "close_cny": close,
                "strategy_cash_cny": strategy_cash,
                "strategy_shares": int(strategy_shares),
                "strategy_dividend_receivable_cny": strategy_receivable,
                "strategy_nav_cny": strategy_nav,
                "static_core_cash_cny": static_cash,
                "static_core_shares": int(static_shares),
                "static_core_dividend_receivable_cny": static_receivable,
                "static_core_nav_cny": static_nav,
                "full_buyhold_cash_cny": full_cash,
                "full_buyhold_shares": int(full_shares),
                "full_buyhold_dividend_receivable_cny": full_receivable,
                "full_buyhold_nav_cny": full_nav,
                "round_trips_today": int(trade is not None),
                "cumulative_t_net_pnl_cny": cumulative_t_net_pnl,
                "cumulative_t_commission_cny": cumulative_commission,
                "cumulative_t_slippage_and_tick_cost_cny": cumulative_slippage,
                "strategy_dividend_entitlement_added_cny": strategy_dividend_added,
                "static_dividend_entitlement_added_cny": static_dividend_added,
                "full_dividend_entitlement_added_cny": full_dividend_added,
                "attribution_error_cny": attribution_error,
            }
        )

    ledger = pd.DataFrame(ledger_rows)
    trades = pd.DataFrame(trade_rows)
    metadata = {
        "initial_cash_cny": initial_cash,
        "first_trade_date": first_trade_date,
        "last_trade_date": pd.Timestamp(trade_dates[-1]),
        "core_acquisition_raw_open_cny": first_raw_open,
        "core_acquisition_execution_price_cny": acquisition_price,
        "core_acquisition_shares": base_shares,
        "core_acquisition_commission_cny": acquisition_commission,
        "cash_after_core_acquisition_cny": initial_cash - acquisition_total,
        "full_buyhold_acquisition_shares": full_shares,
        "full_buyhold_acquisition_commission_cny": full_acquisition_commission,
        "slippage_bps_per_leg": slippage_bps_per_leg,
        "trading_day_count": len(ledger),
        "round_trip_count": len(trades),
    }
    return SimulationResult(
        scenario=scenario,
        period=period,
        start_date=start,
        end_date=end,
        daily_ledger=ledger,
        trades=trades,
        metadata=metadata,
    )


def performance_metrics(
    nav: pd.Series,
    initial_cash_cny: float,
    annual_trading_days: int,
) -> dict[str, float | int]:
    values = pd.Series(nav, dtype=float).reset_index(drop=True)
    if values.empty or (values <= 0).any():
        raise ValueError("净值序列必须非空且全部为正数")
    path = pd.concat([pd.Series([initial_cash_cny]), values], ignore_index=True)
    returns = path.pct_change().dropna()
    total_return = float(values.iloc[-1] / initial_cash_cny - 1.0)
    years = len(values) / float(annual_trading_days)
    cagr = float((values.iloc[-1] / initial_cash_cny) ** (1.0 / years) - 1.0)
    annualized_volatility = float(returns.std(ddof=1) * math.sqrt(annual_trading_days))
    if returns.std(ddof=1) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(annual_trading_days))
    else:
        sharpe = float("nan")
    drawdown = path / path.cummax() - 1.0
    maximum_drawdown = float(-drawdown.min())
    calmar = float(cagr / maximum_drawdown) if maximum_drawdown > 0 else float("inf")
    rolling_returns = path.pct_change(periods=annual_trading_days).dropna()
    return {
        "trading_days": int(len(values)),
        "final_nav_cny": float(values.iloc[-1]),
        "total_pnl_cny": float(values.iloc[-1] - initial_cash_cny),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe_zero_cash_rate": sharpe,
        "maximum_drawdown": maximum_drawdown,
        "calmar": calmar,
        "rolling_242_day_observations": int(len(rolling_returns)),
        "rolling_242_day_median_return": float(rolling_returns.median()) if len(rolling_returns) else float("nan"),
        "rolling_242_day_minimum_return": float(rolling_returns.min()) if len(rolling_returns) else float("nan"),
        "rolling_242_day_maximum_return": float(rolling_returns.max()) if len(rolling_returns) else float("nan"),
        "rolling_242_day_return_above_20pct_ratio": float(rolling_returns.gt(0.20).mean()) if len(rolling_returns) else float("nan"),
    }


def trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "round_trip_count": 0,
            "winning_round_trip_count": 0,
            "win_rate": float("nan"),
            "profit_factor": 0.0,
            "gross_raw_pnl_cny": 0.0,
            "execution_gross_pnl_cny": 0.0,
            "commission_cny": 0.0,
            "stamp_duty_cny": 0.0,
            "slippage_and_tick_cost_cny": 0.0,
            "net_pnl_cny": 0.0,
            "average_net_pnl_cny": float("nan"),
            "by_direction": {},
            "by_exit_reason": {},
        }
    wins = trades.loc[trades["net_pnl_cny"] > 0.0, "net_pnl_cny"]
    losses = trades.loc[trades["net_pnl_cny"] < 0.0, "net_pnl_cny"]
    loss_total = float(-losses.sum())
    profit_factor = float(wins.sum() / loss_total) if loss_total > 0 else float("inf")
    return {
        "round_trip_count": int(len(trades)),
        "winning_round_trip_count": int((trades["net_pnl_cny"] > 0.0).sum()),
        "win_rate": float((trades["net_pnl_cny"] > 0.0).mean()),
        "profit_factor": profit_factor,
        "gross_raw_pnl_cny": float(trades["raw_gross_pnl_cny"].sum()),
        "execution_gross_pnl_cny": float(trades["execution_gross_pnl_cny"].sum()),
        "commission_cny": float(trades["commission_cny"].sum()),
        "stamp_duty_cny": float(trades["stamp_duty_cny"].sum()),
        "slippage_and_tick_cost_cny": float(trades["slippage_and_tick_cost_cny"].sum()),
        "net_pnl_cny": float(trades["net_pnl_cny"].sum()),
        "average_net_pnl_cny": float(trades["net_pnl_cny"].mean()),
        "median_net_pnl_cny": float(trades["net_pnl_cny"].median()),
        "average_holding_trading_records": float(trades["holding_trading_records"].mean()),
        "by_direction": {
            str(key): int(value) for key, value in trades["direction"].value_counts().items()
        },
        "by_exit_reason": {
            str(key): int(value) for key, value in trades["exit_reason"].value_counts().items()
        },
    }


def calendar_year_metrics(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash_cny: float,
    complete_year_minimum_days: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    previous_year_end_nav = initial_cash_cny
    for year, frame in ledger.groupby(ledger["date"].dt.year, sort=True):
        frame = frame.sort_values("date")
        if trades.empty:
            year_trades = trades
        else:
            trade_years = pd.to_datetime(trades["trade_date"]).dt.year
            year_trades = trades.loc[trade_years.eq(year)]
        end_nav = float(frame.iloc[-1]["strategy_nav_cny"])
        static_end_nav = float(frame.iloc[-1]["static_core_nav_cny"])
        start_nav = previous_year_end_nav
        if output:
            static_start_nav = float(output[-1]["static_core_end_nav_cny"])
        else:
            static_start_nav = initial_cash_cny
        output.append(
            {
                "year": int(year),
                "trading_days": int(len(frame)),
                "is_complete_calendar_year": bool(len(frame) >= complete_year_minimum_days),
                "strategy_start_nav_cny": start_nav,
                "strategy_end_nav_cny": end_nav,
                "strategy_return": end_nav / start_nav - 1.0,
                "static_core_end_nav_cny": static_end_nav,
                "static_core_return": static_end_nav / static_start_nav - 1.0,
                "round_trip_count": int(len(year_trades)),
                "t_net_pnl_cny": float(year_trades["net_pnl_cny"].sum()) if not year_trades.empty else 0.0,
            }
        )
        previous_year_end_nav = end_nav
    return output
