"""510300周线状态—日线执行V1：固定指标、下一开盘执行与纸面账本。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from research.graph_regime_martin_turtle_v2 import (
    CostModel,
    _cost_model,
    _events_by_date,
    _target_order,
    add_point_in_time_adjusted_ohlc,
    load_inputs,
    performance_metrics,
    simulate_static_exposure,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "weekly_daily_technical_v1.yaml"


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    """读取并检查V1冻结配置。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("V1配置必须是YAML对象")
    ids = [item["id"] for item in config["variants"]]
    if ids != ["T_ONLY", "M_ONLY", "SWITCH"]:
        raise ValueError("V1轨道集合或顺序与冻结协议不一致")
    if config["protocol"]["parameter_rescue_after_results"] is not False:
        raise ValueError("V1禁止结果后参数救援")
    return config


def _wilder_average(values: pd.Series, window: int) -> pd.Series:
    """Wilder递推平均，首值为第一个完整窗口的算术平均。"""

    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(array), np.nan, dtype=float)
    if len(array) < window:
        return pd.Series(output, index=values.index, dtype=float)
    for end in range(window - 1, len(array)):
        sample = array[end - window + 1 : end + 1]
        if np.all(np.isfinite(sample)):
            output[end] = float(sample.mean())
            first = end
            break
    else:
        return pd.Series(output, index=values.index, dtype=float)
    for index in range(first + 1, len(array)):
        value = array[index]
        if np.isfinite(value):
            output[index] = (output[index - 1] * (window - 1) + value) / window
    return pd.Series(output, index=values.index, dtype=float)


def wilder_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """计算Wilder ATR。"""

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _wilder_average(true_range, window)


def wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    """计算使用Wilder递推和首个算术平均种子的RSI。"""

    delta = pd.to_numeric(close, errors="coerce").diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = _wilder_average(gains.iloc[1:].reset_index(drop=True), window)
    average_loss = _wilder_average(losses.iloc[1:].reset_index(drop=True), window)
    output = pd.Series(np.nan, index=close.index, dtype=float)
    for offset in range(len(average_gain)):
        target = offset + 1
        gain = average_gain.iloc[offset]
        loss = average_loss.iloc[offset]
        if not np.isfinite(gain) or not np.isfinite(loss):
            continue
        if gain == 0.0 and loss == 0.0:
            output.iloc[target] = 50.0
        elif loss == 0.0:
            output.iloc[target] = 100.0
        else:
            relative_strength = gain / loss
            output.iloc[target] = 100.0 - 100.0 / (1.0 + relative_strength)
    return output


def _weekly_features(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """聚合完整周线并计算当日收盘时点可知的周线状态。"""

    weekly_config = config["weekly_regime"]
    frame = data.copy()
    frame["week_period"] = frame["date"].dt.to_period(weekly_config["period"])
    weekly = (
        frame.groupby("week_period", observed=True)
        .agg(
            first_date=("date", "first"),
            last_date=("date", "last"),
            open=("adjusted_open", "first"),
            high=("adjusted_high", "max"),
            low=("adjusted_low", "min"),
            close=("adjusted_close", "last"),
        )
        .reset_index()
    )
    close = weekly["close"].astype(float)
    weekly["ema20"] = close.ewm(
        span=int(weekly_config["ema_window"]), adjust=False
    ).mean()
    fast = close.ewm(span=int(weekly_config["macd_fast"]), adjust=False).mean()
    slow = close.ewm(span=int(weekly_config["macd_slow"]), adjust=False).mean()
    weekly["dif"] = fast - slow
    weekly["dea"] = weekly["dif"].ewm(
        span=int(weekly_config["macd_signal"]), adjust=False
    ).mean()
    slope_reference = weekly["ema20"].shift(
        int(weekly_config["ema_slope_lookback_weeks"])
    )
    warmed = weekly.index >= int(weekly_config["macd_slow"]) - 1
    bull = (
        warmed
        & close.gt(weekly["ema20"])
        & weekly["ema20"].gt(slope_reference)
        & weekly["dif"].gt(weekly["dea"])
    )
    bear = (
        warmed
        & close.lt(weekly["ema20"])
        & weekly["ema20"].lt(slope_reference)
        & weekly["dif"].lt(weekly["dea"])
    )
    weekly["state"] = np.select(
        [bull, bear, warmed],
        ["W_BULL", "W_BEAR", "W_RANGE"],
        default="W_UNKNOWN",
    )
    return weekly


def build_features(
    market: pd.DataFrame, dividends: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建因果日线特征和周线审计表。"""

    data = add_point_in_time_adjusted_ohlc(market, dividends)
    data = data.sort_values("date").reset_index(drop=True)
    daily = config["daily_indicators"]
    close = data["adjusted_close"].astype(float)
    high = data["adjusted_high"].astype(float)
    low = data["adjusted_low"].astype(float)
    data["ma20"] = close.rolling(int(daily["mean_days"])).mean()
    data["ma28"] = close.rolling(int(daily["bias_days"])).mean()
    data["bias28"] = 100.0 * (close / data["ma28"] - 1.0)
    data["atr14"] = wilder_atr(high, low, close, int(daily["atr_wilder_days"]))
    data["rsi5"] = wilder_rsi(close, int(daily["rsi_wilder_days"]))
    data["hh20"] = high.shift(1).rolling(int(daily["donchian_entry_days"])).max()
    data["ll10"] = low.shift(1).rolling(int(daily["donchian_exit_days"])).min()

    weekly = _weekly_features(data, config)
    data["week_period"] = data["date"].dt.to_period(config["weekly_regime"]["period"])
    previous_state = dict(zip(weekly["week_period"], weekly["state"].shift(1).fillna("W_UNKNOWN")))
    previous_source = dict(zip(weekly["week_period"], weekly["last_date"].shift(1)))
    current_state_by_period = dict(zip(weekly["week_period"], weekly["state"]))
    current_last_date = dict(zip(weekly["week_period"], weekly["last_date"]))
    data["weekly_state_at_close"] = data["week_period"].map(previous_state).fillna("W_UNKNOWN")
    data["weekly_state_source_date"] = data["week_period"].map(previous_source)
    is_week_last = data.apply(
        lambda row: row["date"] == current_last_date[row["week_period"]], axis=1
    )
    data.loc[is_week_last, "weekly_state_at_close"] = data.loc[
        is_week_last, "week_period"
    ].map(current_state_by_period)
    data.loc[is_week_last, "weekly_state_source_date"] = data.loc[is_week_last, "date"]
    data["is_completed_week_close"] = is_week_last
    return data, weekly


def _variant_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in config["variants"]}


def _target_with_minimum_gate(
    *,
    target_exposure: float,
    raw_open: float,
    cash: float,
    shares: int,
    receivable: float,
    lot_size: int,
    costs: CostModel,
    minimum_notional: float,
    allow_small_liquidation: bool,
) -> tuple[float, int, dict[str, Any] | None, str | None, float]:
    """预览目标订单并执行5,000元闸门。"""

    next_cash, next_shares, trade = _target_order(
        target_exposure=target_exposure,
        raw_open=raw_open,
        cash=cash,
        shares=shares,
        receivable=receivable,
        lot_size=lot_size,
        costs=costs,
    )
    preview_notional = float(trade["raw_notional"]) if trade is not None else 0.0
    if trade is None:
        return cash, shares, None, "NO_QUANTITY_CHANGE", preview_notional
    is_liquidation = trade["side"] == "SELL" and target_exposure == 0.0
    if preview_notional < minimum_notional and not (
        is_liquidation and allow_small_liquidation
    ):
        return cash, shares, None, "BELOW_MINIMUM_NOTIONAL", preview_notional
    return next_cash, next_shares, trade, None, preview_notional


def simulate_variant(
    features: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    variant_id: str,
    *,
    cost_multiplier: float = 1.0,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """运行一条固定轨道，返回日账本、执行、拒单、状态切换和闭合周期。"""

    variants = _variant_map(config)
    if variant_id not in variants:
        raise KeyError(f"未知V1轨道：{variant_id}")
    variant = variants[variant_id]
    execution = config["price_and_execution"]
    trend = config["trend_module"]
    mean = config["mean_reversion_module"]
    target_exposures = [float(value) for value in execution["target_exposures"]]
    lot_size = int(execution["lot_size_shares"])
    minimum_notional = float(execution["minimum_normal_trade_notional_cny"])
    costs = _cost_model(config, cost_multiplier)
    daily_cash_rate = (1.0 + float(execution["cash_annual_rate"])) ** (
        1.0 / int(execution["trading_days_per_year"])
    ) - 1.0

    record_events = _events_by_date(dividends, "record_date")
    ex_events = _events_by_date(dividends, "ex_date")
    payment_events = _events_by_date(dividends, "payment_date")
    entitlements: dict[pd.Timestamp, float] = {}
    receivables_by_payment: dict[pd.Timestamp, float] = defaultdict(float)

    cash = float(execution["initial_capital_cny"])
    shares = 0
    receivable = 0.0
    pending: dict[str, Any] | None = None
    mode = "CASH"
    tier = 0
    frozen_atr = np.nan
    last_fill_adjusted = np.nan
    first_entry_adjusted = np.nan
    holding_days = 0
    active_cycle: dict[str, Any] | None = None
    ledger_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []

    for index, row in features.iterrows():
        date = row["date"]
        for event in ex_events.get(date, []):
            amount = entitlements.get(event.ex_date, 0.0)
            receivable += amount
            receivables_by_payment[event.payment_date] += amount
        if date in payment_events:
            amount = receivables_by_payment.pop(date, 0.0)
            cash += amount
            receivable -= amount

        shares_before = shares
        mode_before = mode
        executed_reason: str | None = None
        day_commission = 0.0
        day_slippage = 0.0
        day_notional = 0.0

        if pending is not None:
            executed_reason = str(pending["reason"])
            if pending["action"] == "SWITCH":
                mode = "TREND"
                frozen_atr = float(pending["atr14"])
                last_fill_adjusted = float(row["open"]) * float(row["adjustment_factor"])
                transition_rows.append(
                    {
                        "variant_id": variant_id,
                        "signal_date": pending["signal_date"],
                        "effective_date": date,
                        "from_mode": mode_before,
                        "to_mode": mode,
                        "reason": executed_reason,
                        "tier": tier,
                    }
                )
                if active_cycle is not None:
                    active_cycle["switch_date"] = date
                    active_cycle["switch_reason"] = executed_reason
                    active_cycle["switch_equity"] = cash + shares * float(row["open"]) + receivable
            else:
                target = float(pending["target_exposure"])
                cash_after, shares_after, trade, rejection, preview_notional = (
                    _target_with_minimum_gate(
                        target_exposure=target,
                        raw_open=float(row["open"]),
                        cash=cash,
                        shares=shares,
                        receivable=receivable,
                        lot_size=lot_size,
                        costs=costs,
                        minimum_notional=minimum_notional,
                        allow_small_liquidation=bool(
                            execution["risk_liquidation_exempt_from_minimum_notional"]
                        ),
                    )
                )
                if rejection is not None:
                    skipped_rows.append(
                        {
                            "variant_id": variant_id,
                            "signal_date": pending["signal_date"],
                            "execution_date": date,
                            "reason": executed_reason,
                            "rejection": rejection,
                            "target_exposure": target,
                            "preview_raw_notional": preview_notional,
                        }
                    )
                else:
                    cash, shares = cash_after, shares_after
                    if trade is not None:
                        day_commission = float(trade["commission"])
                        day_slippage = float(trade["slippage_cost"])
                        day_notional = float(trade["raw_notional"])
                        execution_rows.append(
                            {
                                "variant_id": variant_id,
                                "signal_date": pending["signal_date"],
                                "execution_date": date,
                                "reason": executed_reason,
                                "mode_before": mode_before,
                                "side": trade["side"],
                                "trade_shares": int(trade["shares"]),
                                "raw_open": float(row["open"]),
                                "execution_price": float(trade["execution_price"]),
                                "raw_notional": day_notional,
                                "commission": day_commission,
                                "slippage_cost": day_slippage,
                                "target_exposure": target,
                                "shares_after": shares,
                            }
                        )
                    reason = executed_reason
                    if reason == "TREND_ENTRY":
                        mode, tier, holding_days = "TREND", 1, 0
                        frozen_atr = float(pending["atr14"])
                        last_fill_adjusted = float(trade["execution_price"]) * float(
                            row["adjustment_factor"]
                        )
                        first_entry_adjusted = last_fill_adjusted
                    elif reason == "MEAN_ENTRY":
                        mode, tier, holding_days = "MEAN", 1, 0
                        frozen_atr = float(pending["atr14"])
                        last_fill_adjusted = float(trade["execution_price"]) * float(
                            row["adjustment_factor"]
                        )
                        first_entry_adjusted = last_fill_adjusted
                    elif reason.startswith("TREND_ADD_"):
                        mode = "TREND"
                        tier = int(reason.rsplit("_", 1)[-1])
                        last_fill_adjusted = float(trade["execution_price"]) * float(
                            row["adjustment_factor"]
                        )
                    elif reason.startswith("MEAN_ADD_"):
                        mode = "MEAN"
                        tier = int(reason.rsplit("_", 1)[-1])
                        last_fill_adjusted = float(trade["execution_price"]) * float(
                            row["adjustment_factor"]
                        )
                    elif target == 0.0:
                        mode, tier, holding_days = "CASH", 0, 0
                        frozen_atr = np.nan
                        last_fill_adjusted = np.nan
                        first_entry_adjusted = np.nan
            pending = None

        if shares_before == 0 and shares > 0:
            active_cycle = {
                "variant_id": variant_id,
                "entry_date": date,
                "entry_reason": executed_reason,
                "entry_mode": mode,
                "entry_equity": cash + shares * float(row["open"]) + receivable,
                "switch_date": None,
                "switch_reason": None,
                "switch_equity": None,
            }
        elif shares_before > 0 and shares == 0 and active_cycle is not None:
            exit_equity = cash + receivable
            switch_equity = active_cycle.get("switch_equity")
            cycle_rows.append(
                {
                    **active_cycle,
                    "exit_date": date,
                    "exit_reason": executed_reason,
                    "exit_equity": exit_equity,
                    "net_pnl": exit_equity - float(active_cycle["entry_equity"]),
                    "pnl_to_switch": (
                        float(switch_equity) - float(active_cycle["entry_equity"])
                        if switch_equity is not None
                        else None
                    ),
                    "pnl_after_switch": (
                        exit_equity - float(switch_equity)
                        if switch_equity is not None
                        else None
                    ),
                    "holding_calendar_days": (date - active_cycle["entry_date"]).days,
                }
            )
            active_cycle = None

        cash *= 1.0 + daily_cash_rate
        equity = cash + shares * float(row["close"]) + receivable
        exposure = shares * float(row["close"]) / equity if equity > 0 else np.nan
        for event in record_events.get(date, []):
            entitlements[event.ex_date] = shares * float(event.cash_dividend_per_share)
        if shares > 0:
            holding_days += 1

        next_pending: dict[str, Any] | None = None
        weekly_state = str(row["weekly_state_at_close"])
        adjusted_close = float(row["adjusted_close"])
        atr14 = float(row["atr14"]) if pd.notna(row["atr14"]) else np.nan
        hh20 = float(row["hh20"]) if pd.notna(row["hh20"]) else np.nan
        ll10 = float(row["ll10"]) if pd.notna(row["ll10"]) else np.nan
        ma20 = float(row["ma20"]) if pd.notna(row["ma20"]) else np.nan
        bias28 = float(row["bias28"]) if pd.notna(row["bias28"]) else np.nan
        rsi5 = float(row["rsi5"]) if pd.notna(row["rsi5"]) else np.nan
        has_next_day = index + 1 < len(features)

        def order(reason: str, target_tier: int) -> dict[str, Any]:
            return {
                "action": "ORDER",
                "signal_date": date,
                "reason": reason,
                "target_exposure": target_exposures[target_tier - 1]
                if target_tier > 0
                else 0.0,
                "atr14": atr14,
            }

        if has_next_day:
            if mode == "TREND":
                if weekly_state == "W_BEAR":
                    next_pending = order("TREND_EXIT_WEEKLY_BEAR", 0)
                elif np.isfinite(ll10) and adjusted_close < ll10:
                    next_pending = order("TREND_EXIT_DONCHIAN_10", 0)
                elif (
                    np.isfinite(frozen_atr)
                    and adjusted_close <= last_fill_adjusted - float(trend["stop_from_last_fill_atr"]) * frozen_atr
                ):
                    next_pending = order("TREND_EXIT_2ATR_STOP", 0)
                elif (
                    weekly_state == "W_BULL"
                    and tier < int(trend["maximum_tier"])
                    and np.isfinite(frozen_atr)
                    and adjusted_close >= last_fill_adjusted + float(trend["add_atr_interval"]) * frozen_atr
                ):
                    next_pending = order(f"TREND_ADD_{tier + 1}", tier + 1)
            elif mode == "MEAN":
                if weekly_state == "W_BEAR":
                    next_pending = order("MEAN_EXIT_WEEKLY_BEAR", 0)
                elif (
                    np.isfinite(frozen_atr)
                    and adjusted_close <= first_entry_adjusted - float(mean["disaster_stop_from_first_entry_atr"]) * frozen_atr
                ):
                    next_pending = order("MEAN_EXIT_4ATR_STOP", 0)
                elif holding_days >= int(mean["maximum_holding_trading_days"]):
                    next_pending = order("MEAN_EXIT_TIME_25", 0)
                elif (
                    variant["switch_enabled"]
                    and weekly_state == "W_BULL"
                    and np.isfinite(hh20)
                    and adjusted_close > hh20
                    and np.isfinite(atr14)
                ):
                    next_pending = {
                        "action": "SWITCH",
                        "signal_date": date,
                        "reason": "MEAN_TO_TREND_SWITCH",
                        "atr14": atr14,
                    }
                elif (np.isfinite(ma20) and adjusted_close >= ma20) or (
                    np.isfinite(rsi5) and rsi5 >= float(mean["mean_exit_rsi_minimum"])
                ):
                    next_pending = order("MEAN_EXIT_REVERSION", 0)
                elif weekly_state == "W_RANGE" and tier < int(mean["maximum_tier"]):
                    required_bias = (
                        float(mean["tier2_bias_maximum"])
                        if tier == 1
                        else float(mean["tier3_bias_maximum"])
                    )
                    if (
                        np.isfinite(bias28)
                        and bias28 <= required_bias
                        and np.isfinite(frozen_atr)
                        and adjusted_close <= last_fill_adjusted - float(mean["add_atr_interval"]) * frozen_atr
                    ):
                        next_pending = order(f"MEAN_ADD_{tier + 1}", tier + 1)
            else:
                if (
                    variant["trend_enabled"]
                    and weekly_state == "W_BULL"
                    and np.isfinite(hh20)
                    and adjusted_close > hh20
                    and np.isfinite(atr14)
                ):
                    next_pending = order("TREND_ENTRY", 1)
                elif (
                    variant["mean_reversion_enabled"]
                    and weekly_state == "W_RANGE"
                    and np.isfinite(bias28)
                    and bias28 <= float(mean["entry_bias_maximum"])
                    and np.isfinite(rsi5)
                    and rsi5 <= float(mean["entry_rsi_maximum"])
                    and np.isfinite(atr14)
                ):
                    next_pending = order("MEAN_ENTRY", 1)
        pending = next_pending

        ledger_rows.append(
            {
                "variant_id": variant_id,
                "date": date,
                "equity": equity,
                "shares": shares,
                "cash": cash,
                "receivable": receivable,
                "exposure": exposure,
                "mode": mode,
                "tier": tier,
                "holding_days": holding_days,
                "weekly_state_at_close": weekly_state,
                "day_commission": day_commission,
                "day_slippage": day_slippage,
                "day_raw_notional": day_notional,
                "executed_reason": executed_reason,
                "next_signal_reason": next_pending["reason"] if next_pending else None,
            }
        )

    return {
        "ledger": pd.DataFrame(ledger_rows),
        "executions": pd.DataFrame(execution_rows),
        "skipped_orders": pd.DataFrame(skipped_rows),
        "transitions": pd.DataFrame(transition_rows),
        "cycles": pd.DataFrame(cycle_rows),
        "state": {
            "ending_mode": mode,
            "ending_tier": tier,
            "open_cycle": active_cycle is not None,
            "pending_signal_suppressed": pending is not None,
        },
    }


def period_return(ledger: pd.DataFrame, start: str, end: str) -> float | None:
    """计算预注册阶段的首尾权益收益。"""

    subset = ledger.loc[
        ledger["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ]
    if len(subset) < 2:
        return None
    return float(subset["equity"].iloc[-1] / subset["equity"].iloc[0] - 1.0)


def h00300_metrics(
    benchmark: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, initial: float
) -> dict[str, float]:
    """将H00300全收益指数转换为可比较权益曲线。"""

    frame = benchmark.loc[benchmark["date"].between(start, end), ["date", "close"]].copy()
    frame = frame.dropna().sort_values("date").reset_index(drop=True)
    frame["equity"] = initial * frame["close"].astype(float) / float(frame["close"].iloc[0])
    frame["exposure"] = 1.0
    frame["day_commission"] = 0.0
    frame["day_slippage"] = 0.0
    frame["day_raw_notional"] = 0.0
    return performance_metrics(frame)


__all__ = [
    "ROOT",
    "build_features",
    "h00300_metrics",
    "load_config",
    "load_inputs",
    "performance_metrics",
    "period_return",
    "simulate_static_exposure",
    "simulate_variant",
    "wilder_atr",
    "wilder_rsi",
]

