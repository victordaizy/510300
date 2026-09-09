"""LC + FVG + PV 原型的纯策略逻辑。

本模块回答一个问题：流动性扫单、FVG 回踩和价量确认组合后，是否仍有
覆盖成本的分钟级方向优势。所有信号只使用当前及历史记录，成交固定在下一
根分钟线开盘，避免同一根 K 线收盘识别、又按其收盘成交的前视偏差。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "ts_code",
    "trade_time",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
}


@dataclass(frozen=True)
class BacktestResult:
    """原型回测的完整内存状态。"""

    features: pd.DataFrame
    trades: pd.DataFrame
    daily_equity: pd.DataFrame
    summaries: pd.DataFrame


def _rolling_by_session(
    frame: pd.DataFrame,
    column: str,
    window: int,
    minimum: int,
    aggregation: str,
    shift: int = 1,
) -> pd.Series:
    grouped = frame.groupby("trade_date", sort=False)[column]

    def calculate(series: pd.Series) -> pd.Series:
        rolling = series.shift(shift).rolling(window, min_periods=minimum)
        return getattr(rolling, aggregation)()

    return grouped.transform(calculate)


def prepare_features(minute: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """从原始分钟数据构造不含未来信息的 LC、FVG 与 PV 特征。"""

    missing = REQUIRED_COLUMNS - set(minute.columns)
    if missing:
        raise ValueError(f"分钟数据缺少字段：{sorted(missing)}")

    feature_config = config["feature"]
    output = minute.copy()
    output["trade_time"] = pd.to_datetime(output["trade_time"])
    output = output.sort_values("trade_time").reset_index(drop=True)
    output["row_number"] = np.arange(len(output), dtype=np.int64)
    output["trade_date"] = output["trade_time"].dt.normalize()
    output["record_time"] = output["trade_time"].dt.time
    output["minute_key"] = (
        output["trade_time"].dt.hour * 60 + output["trade_time"].dt.minute
    )
    session = output.groupby("trade_date", sort=False)
    output["bar_number"] = session.cumcount()
    output["previous_close"] = session["close"].shift(1)

    previous_close = output["previous_close"]
    output["true_range"] = np.maximum.reduce(
        [
            (output["high"] - output["low"]).to_numpy(),
            (output["high"] - previous_close).abs().fillna(0.0).to_numpy(),
            (output["low"] - previous_close).abs().fillna(0.0).to_numpy(),
        ]
    )
    output["atr"] = _rolling_by_session(
        output,
        "true_range",
        int(feature_config["atr_window_bars"]),
        int(feature_config["atr_window_bars"]),
        "mean",
    )

    liquidity_window = int(feature_config["liquidity_lookback_bars"])
    liquidity_minimum = int(feature_config["liquidity_minimum_bars"])
    output["liquidity_high"] = _rolling_by_session(
        output,
        "high",
        liquidity_window,
        liquidity_minimum,
        "max",
    )
    output["liquidity_low"] = _rolling_by_session(
        output,
        "low",
        liquidity_window,
        liquidity_minimum,
        "min",
    )

    tick = float(feature_config["price_tick"])
    output["bullish_sweep"] = (
        (output["low"] <= output["liquidity_low"] - tick + 1e-12)
        & (output["close"] > output["liquidity_low"])
    )
    output["bearish_sweep"] = (
        (output["high"] >= output["liquidity_high"] + tick - 1e-12)
        & (output["close"] < output["liquidity_high"])
    )

    output["bullish_sweep_row"] = output["row_number"].where(output["bullish_sweep"])
    output["bearish_sweep_row"] = output["row_number"].where(output["bearish_sweep"])
    output["bullish_sweep_low"] = output["low"].where(output["bullish_sweep"])
    output["bearish_sweep_high"] = output["high"].where(output["bearish_sweep"])
    output["bullish_sweep_time"] = output["trade_time"].where(output["bullish_sweep"])
    output["bearish_sweep_time"] = output["trade_time"].where(output["bearish_sweep"])
    for column in (
        "bullish_sweep_row",
        "bearish_sweep_row",
        "bullish_sweep_low",
        "bearish_sweep_high",
        "bullish_sweep_time",
        "bearish_sweep_time",
    ):
        output[column] = output.groupby("trade_date", sort=False)[column].ffill()

    output["bullish_sweep_age_bars"] = (
        output["row_number"] - output["bullish_sweep_row"]
    )
    output["bearish_sweep_age_bars"] = (
        output["row_number"] - output["bearish_sweep_row"]
    )
    output["bullish_sweep_age_minutes"] = (
        output["trade_time"] - output["bullish_sweep_time"]
    ).dt.total_seconds() / 60.0
    output["bearish_sweep_age_minutes"] = (
        output["trade_time"] - output["bearish_sweep_time"]
    ).dt.total_seconds() / 60.0

    high_two_bars_ago = session["high"].shift(2)
    low_two_bars_ago = session["low"].shift(2)
    time_two_bars_ago = session["trade_time"].shift(2)
    output["three_bar_span_minutes"] = (
        output["trade_time"] - time_two_bars_ago
    ).dt.total_seconds() / 60.0
    output["body_size"] = (output["close"] - output["open"]).abs()
    output["bullish_fvg_lower"] = high_two_bars_ago
    output["bullish_fvg_upper"] = output["low"]
    output["bearish_fvg_lower"] = output["high"]
    output["bearish_fvg_upper"] = low_two_bars_ago
    output["bullish_fvg_width"] = (
        output["bullish_fvg_upper"] - output["bullish_fvg_lower"]
    )
    output["bearish_fvg_width"] = (
        output["bearish_fvg_upper"] - output["bearish_fvg_lower"]
    )

    maximum_sweep_bars = int(feature_config["sweep_to_fvg_maximum_bars"])
    maximum_sweep_minutes = float(feature_config["sweep_to_fvg_maximum_minutes"])
    minimum_gap_fraction = float(feature_config["minimum_fvg_atr_fraction"])
    minimum_displacement_fraction = float(
        feature_config["minimum_displacement_atr_fraction"]
    )
    common_fvg_condition = (
        output["atr"].notna()
        & (output["three_bar_span_minutes"] <= 3.0)
        & (output["body_size"] >= output["atr"] * minimum_displacement_fraction)
    )
    output["bullish_fvg_candidate"] = (
        common_fvg_condition
        & (output["close"] > output["open"])
        & (output["bullish_fvg_width"] >= output["atr"] * minimum_gap_fraction)
        & output["bullish_sweep_age_bars"].between(0, maximum_sweep_bars)
        & output["bullish_sweep_age_minutes"].between(0.0, maximum_sweep_minutes)
    )
    output["bearish_fvg_candidate"] = (
        common_fvg_condition
        & (output["close"] < output["open"])
        & (output["bearish_fvg_width"] >= output["atr"] * minimum_gap_fraction)
        & output["bearish_sweep_age_bars"].between(0, maximum_sweep_bars)
        & output["bearish_sweep_age_minutes"].between(0.0, maximum_sweep_minutes)
    )

    volume_days = int(feature_config["volume_baseline_trading_days"])
    volume_minimum = int(feature_config["volume_baseline_minimum_days"])
    output["seasonal_volume_median"] = output.groupby(
        "minute_key", sort=False
    )["vol"].transform(
        lambda series: series.shift(1).rolling(
            volume_days, min_periods=volume_minimum
        ).median()
    )
    output["relative_volume"] = (
        output["vol"] / output["seasonal_volume_median"].replace(0.0, np.nan)
    )
    output["cumulative_volume"] = session["vol"].cumsum()
    output["cumulative_amount"] = session["amount"].cumsum()
    output["session_vwap"] = (
        output["cumulative_amount"]
        / output["cumulative_volume"].replace(0.0, np.nan)
    )
    output["vwap_deviation"] = output["close"] / output["session_vwap"] - 1.0
    return output


def _execution_price(raw_price: float, side: str, slippage_bps: float) -> float:
    fraction = slippage_bps / 10_000.0
    if side == "BUY":
        return raw_price * (1.0 + fraction)
    if side == "SELL":
        return raw_price * (1.0 - fraction)
    raise ValueError(f"未知成交方向：{side}")


def _complete_trade(
    position: dict[str, Any],
    record: Any,
    exit_raw_price: float,
    exit_reason: str,
    cost_config: dict[str, Any],
) -> dict[str, Any]:
    direction = position["direction"]
    exit_side = "SELL" if direction == "LONG" else "BUY"
    exit_execution_price = _execution_price(
        float(exit_raw_price),
        exit_side,
        float(cost_config["slippage_bps_per_leg"]),
    )
    entry_raw_price = float(position["entry_raw_price"])
    entry_execution_price = float(position["entry_execution_price"])
    if direction == "LONG":
        gross_return = float(exit_raw_price) / entry_raw_price - 1.0
        execution_return = exit_execution_price / entry_execution_price - 1.0
    else:
        gross_return = 1.0 - float(exit_raw_price) / entry_raw_price
        execution_return = 1.0 - exit_execution_price / entry_execution_price
    net_return = execution_return - 2.0 * float(
        cost_config["commission_rate_per_leg"]
    )
    return {
        **position,
        "exit_time": pd.Timestamp(record.trade_time),
        "exit_raw_price": float(exit_raw_price),
        "exit_execution_price": float(exit_execution_price),
        "exit_reason": exit_reason,
        "holding_bars": int(record.bar_number - position["entry_bar"] + 1),
        "gross_return": float(gross_return),
        "net_return": float(net_return),
    }


def simulate_trades(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """用显式状态机模拟 FVG 建立、回踩、下单和离场。"""

    confirmation = config["confirmation"]
    risk = config["risk"]
    cost = config["cost"]
    feature_config = config["feature"]
    signal_start = time.fromisoformat(str(confirmation["signal_start_time"]))
    signal_end = time.fromisoformat(str(confirmation["signal_end_time"]))
    force_exit_time = time.fromisoformat(str(risk["force_exit_time"]))
    relative_volume_minimum = float(confirmation["relative_volume_minimum"])
    vwap_tolerance = float(confirmation["vwap_tolerance_fraction"])
    retest_expiry = int(confirmation["retest_expiry_bars"])
    maximum_holding = int(risk["maximum_holding_bars"])
    reward_to_risk = float(risk["reward_to_risk"])
    minimum_stop_fraction = float(risk["minimum_stop_fraction"])
    maximum_stop_fraction = float(risk["maximum_stop_fraction"])
    maximum_trades_per_day = int(risk["maximum_trades_per_day"])
    tick = float(feature_config["price_tick"])

    trades: list[dict[str, Any]] = []
    for trade_date, day in features.groupby("trade_date", sort=True):
        active_zones: list[dict[str, Any]] = []
        pending_entry: dict[str, Any] | None = None
        position: dict[str, Any] | None = None
        trades_today = 0
        records = list(day.itertuples(index=False))

        for record_index, record in enumerate(records):
            current_time = record.record_time
            is_last_record = record_index == len(records) - 1

            if pending_entry is not None:
                if int(record.bar_number) == pending_entry["signal_bar"] + 1:
                    entry_raw_price = float(record.open)
                    direction = pending_entry["direction"]
                    stop_price = float(pending_entry["stop_price"])
                    risk_distance = (
                        entry_raw_price - stop_price
                        if direction == "LONG"
                        else stop_price - entry_raw_price
                    )
                    risk_fraction = risk_distance / entry_raw_price
                    if minimum_stop_fraction <= risk_fraction <= maximum_stop_fraction:
                        entry_side = "BUY" if direction == "LONG" else "SELL"
                        entry_execution_price = _execution_price(
                            entry_raw_price,
                            entry_side,
                            float(cost["slippage_bps_per_leg"]),
                        )
                        target_price = (
                            entry_raw_price + reward_to_risk * risk_distance
                            if direction == "LONG"
                            else entry_raw_price - reward_to_risk * risk_distance
                        )
                        position = {
                            **pending_entry,
                            "trade_date": pd.Timestamp(trade_date),
                            "entry_time": pd.Timestamp(record.trade_time),
                            "entry_bar": int(record.bar_number),
                            "entry_raw_price": entry_raw_price,
                            "entry_execution_price": float(entry_execution_price),
                            "target_price": float(target_price),
                            "risk_fraction": float(risk_fraction),
                        }
                        trades_today += 1
                    pending_entry = None
                elif int(record.bar_number) > pending_entry["signal_bar"] + 1:
                    pending_entry = None

            if position is not None:
                direction = position["direction"]
                stop_touched = (
                    float(record.low) <= position["stop_price"]
                    if direction == "LONG"
                    else float(record.high) >= position["stop_price"]
                )
                target_touched = (
                    float(record.high) >= position["target_price"]
                    if direction == "LONG"
                    else float(record.low) <= position["target_price"]
                )
                holding_bars = int(record.bar_number - position["entry_bar"] + 1)
                if stop_touched:
                    trades.append(
                        _complete_trade(
                            position,
                            record,
                            position["stop_price"],
                            "止损",
                            cost,
                        )
                    )
                    position = None
                elif target_touched:
                    trades.append(
                        _complete_trade(
                            position,
                            record,
                            position["target_price"],
                            "止盈",
                            cost,
                        )
                    )
                    position = None
                elif holding_bars >= maximum_holding:
                    trades.append(
                        _complete_trade(position, record, float(record.close), "超时", cost)
                    )
                    position = None
                elif current_time >= force_exit_time or is_last_record:
                    trades.append(
                        _complete_trade(position, record, float(record.close), "收盘", cost)
                    )
                    position = None

            valid_zones: list[dict[str, Any]] = []
            for zone in active_zones:
                age = int(record.bar_number) - zone["created_bar"]
                invalidated = (
                    float(record.close) < zone["lower"]
                    if zone["direction"] == "LONG"
                    else float(record.close) > zone["upper"]
                )
                if age <= retest_expiry and not invalidated:
                    valid_zones.append(zone)
            active_zones = valid_zones

            can_signal = (
                position is None
                and pending_entry is None
                and trades_today < maximum_trades_per_day
                and signal_start <= current_time <= signal_end
                and pd.notna(record.relative_volume)
                and pd.notna(record.session_vwap)
            )
            if can_signal:
                for zone in reversed(active_zones):
                    if int(record.bar_number) <= zone["created_bar"]:
                        continue
                    midpoint = (zone["lower"] + zone["upper"]) / 2.0
                    overlaps = (
                        float(record.low) <= zone["upper"]
                        and float(record.high) >= zone["lower"]
                    )
                    if not overlaps or float(record.relative_volume) < relative_volume_minimum:
                        continue
                    bullish_confirmation = (
                        zone["direction"] == "LONG"
                        and float(record.close) >= midpoint
                        and float(record.close) > float(record.open)
                        and float(record.close)
                        >= float(record.session_vwap) * (1.0 - vwap_tolerance)
                    )
                    bearish_confirmation = (
                        zone["direction"] == "SHORT"
                        and float(record.close) <= midpoint
                        and float(record.close) < float(record.open)
                        and float(record.close)
                        <= float(record.session_vwap) * (1.0 + vwap_tolerance)
                    )
                    if not (bullish_confirmation or bearish_confirmation):
                        continue
                    stop_price = (
                        min(zone["sweep_extreme"], zone["lower"]) - tick
                        if zone["direction"] == "LONG"
                        else max(zone["sweep_extreme"], zone["upper"]) + tick
                    )
                    pending_entry = {
                        "direction": zone["direction"],
                        "signal_time": pd.Timestamp(record.trade_time),
                        "signal_bar": int(record.bar_number),
                        "fvg_created_time": zone["created_time"],
                        "fvg_lower": float(zone["lower"]),
                        "fvg_upper": float(zone["upper"]),
                        "sweep_time": zone["sweep_time"],
                        "sweep_extreme": float(zone["sweep_extreme"]),
                        "stop_price": float(stop_price),
                        "signal_relative_volume": float(record.relative_volume),
                        "signal_vwap_deviation": float(record.vwap_deviation),
                    }
                    active_zones = []
                    break

            if current_time <= signal_end:
                if bool(record.bullish_fvg_candidate):
                    active_zones.append(
                        {
                            "direction": "LONG",
                            "created_bar": int(record.bar_number),
                            "created_time": pd.Timestamp(record.trade_time),
                            "lower": float(record.bullish_fvg_lower),
                            "upper": float(record.bullish_fvg_upper),
                            "sweep_time": pd.Timestamp(record.bullish_sweep_time),
                            "sweep_extreme": float(record.bullish_sweep_low),
                        }
                    )
                if bool(record.bearish_fvg_candidate):
                    active_zones.append(
                        {
                            "direction": "SHORT",
                            "created_bar": int(record.bar_number),
                            "created_time": pd.Timestamp(record.trade_time),
                            "lower": float(record.bearish_fvg_lower),
                            "upper": float(record.bearish_fvg_upper),
                            "sweep_time": pd.Timestamp(record.bearish_sweep_time),
                            "sweep_extreme": float(record.bearish_sweep_high),
                        }
                    )

        if position is not None:
            last_record = records[-1]
            trades.append(
                _complete_trade(
                    position,
                    last_record,
                    float(last_record.close),
                    "数据日结束",
                    cost,
                )
            )

    return pd.DataFrame(trades)


def _calculate_summary(
    trades: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(trading_dates).normalize().unique()))
    daily = pd.DataFrame({"trade_date": dates})
    if trades.empty:
        daily["net_return"] = 0.0
    else:
        by_date = trades.groupby("trade_date")["net_return"].sum()
        daily["net_return"] = daily["trade_date"].map(by_date).fillna(0.0)
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0

    count = len(trades)
    positive = trades.loc[trades["net_return"] > 0, "net_return"].sum() if count else 0.0
    negative = trades.loc[trades["net_return"] < 0, "net_return"].sum() if count else 0.0
    total_return = float(daily["equity"].iloc[-1] - 1.0) if len(daily) else 0.0
    annualized_return = (
        float((1.0 + total_return) ** (242.0 / len(daily)) - 1.0)
        if len(daily) and total_return > -1.0
        else np.nan
    )
    summary = {
        "区间": label,
        "交易日": len(daily),
        "交易数": count,
        "多头数": int((trades["direction"] == "LONG").sum()) if count else 0,
        "空头数": int((trades["direction"] == "SHORT").sum()) if count else 0,
        "胜率": float((trades["net_return"] > 0).mean()) if count else np.nan,
        "平均毛收益": float(trades["gross_return"].mean()) if count else np.nan,
        "平均净收益": float(trades["net_return"].mean()) if count else np.nan,
        "累计净收益": total_return,
        "年化净收益": annualized_return,
        "最大回撤": float(daily["drawdown"].min()) if len(daily) else np.nan,
        "盈利因子": float(positive / abs(negative)) if negative < 0 else np.nan,
    }
    return summary, daily


def evaluate_backtest(
    features: pd.DataFrame,
    trades: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算全样本及预先存在的数据分段指标。"""

    summaries: list[dict[str, Any]] = []
    all_dates = pd.DatetimeIndex(features["trade_date"].unique())
    full_summary, full_daily = _calculate_summary(trades, all_dates, "全样本")
    summaries.append(full_summary)
    full_daily["区间"] = "全样本"
    daily_frames = [full_daily]

    for partition in config["partitions"].values():
        start = pd.Timestamp(partition["start"])
        end = pd.Timestamp(partition["end"])
        label = str(partition["label"])
        partition_dates = all_dates[(all_dates >= start) & (all_dates <= end)]
        if trades.empty:
            partition_trades = trades.copy()
        else:
            partition_trades = trades[
                trades["trade_date"].between(start, end)
            ].copy()
        summary, daily = _calculate_summary(partition_trades, partition_dates, label)
        summaries.append(summary)
        daily["区间"] = label
        daily_frames.append(daily)

    return pd.DataFrame(summaries), pd.concat(daily_frames, ignore_index=True)


def run_backtest(minute: pd.DataFrame, config: dict[str, Any]) -> BacktestResult:
    """执行完整的内存原型。"""

    features = prepare_features(minute, config)
    trades = simulate_trades(features, config)
    summaries, daily_equity = evaluate_backtest(features, trades, config)
    return BacktestResult(
        features=features,
        trades=trades,
        daily_equity=daily_equity,
        summaries=summaries,
    )
