"""把标准技术指标与既有失败轨道统一转换成双向执行事件。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EventMap = dict[pd.Timestamp, dict[str, Any]]


def _first_events(
    data: pd.DataFrame,
    mask: pd.Series,
    candidate_id: str,
    side: str,
) -> EventMap:
    frame = data.copy().sort_values("signal_asof").reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["next_open"] = pd.to_numeric(frame["open"], errors="coerce").shift(-1)
    frame["next_trade_date"] = frame["trade_date"].shift(-1)
    eligible = (
        pd.Series(mask, index=data.index).reindex(frame.index, fill_value=False).fillna(False).astype(bool)
        & frame["bar_slot"].between(3, 15)
        & frame["next_trade_date"].eq(frame["trade_date"])
        & frame["next_open"].gt(0)
    )
    events: EventMap = {}
    for row in frame.loc[eligible].itertuples(index=False):
        date = pd.Timestamp(row.trade_date).normalize()
        events.setdefault(
            date,
            {
                "signal_asof": pd.Timestamp(row.signal_asof),
                "technical_signal": f"{candidate_id}_{side}",
                "raw_execution_price": float(row.next_open),
            },
        )
    return events


def _cross_up(left: pd.Series, right: pd.Series, same_day: pd.Series) -> pd.Series:
    return same_day & left.gt(right) & left.shift(1).le(right.shift(1))


def _cross_down(left: pd.Series, right: pd.Series, same_day: pd.Series) -> pd.Series:
    return same_day & left.lt(right) & left.shift(1).ge(right.shift(1))


def build_standard_candidate_events(
    factors: pd.DataFrame,
    indicator: pd.DataFrame,
) -> dict[str, tuple[EventMap, EventMap, str]]:
    """生成固定参数技术候选；不根据回测结果搜索窗口或阈值。"""

    data = factors.copy().sort_values("signal_asof").reset_index(drop=True)
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    same_day = data["trade_date"].eq(data["trade_date"].shift(1))
    zero = pd.Series(0.0, index=data.index)
    candidates: dict[str, tuple[pd.Series, pd.Series, str]] = {}

    candidates["MOMENTUM_4BAR_ZERO_CROSS"] = (
        _cross_up(data["momentum_4bar"], zero, same_day),
        _cross_down(data["momentum_4bar"], zero, same_day),
        "4根15分钟动量上穿/下穿零轴",
    )
    candidates["MOMENTUM_16BAR_ZERO_CROSS"] = (
        _cross_up(data["momentum_16bar"], zero, same_day),
        _cross_down(data["momentum_16bar"], zero, same_day),
        "16根15分钟动量上穿/下穿零轴",
    )

    close = data["close"].astype(float)
    ema12 = close.ewm(span=12, adjust=False, min_periods=26).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    candidates["MACD_12_26_9_CROSS"] = (
        _cross_up(macd, macd_signal, same_day),
        _cross_down(macd, macd_signal, same_day),
        "标准MACD(12,26,9)金叉/死叉",
    )
    candidates["EMA_8_21_CROSS"] = (
        _cross_up(data["ema_8bar"], data["ema_21bar"], same_day),
        _cross_down(data["ema_8bar"], data["ema_21bar"], same_day),
        "EMA8上穿/下穿EMA21",
    )
    candidates["RSI_7_REVERSAL_30_70"] = (
        same_day & data["rsi_7bar"].gt(30.0) & data["rsi_7bar"].shift(1).le(30.0),
        same_day & data["rsi_7bar"].lt(70.0) & data["rsi_7bar"].shift(1).ge(70.0),
        "RSI7从超卖区向上离开30/从超买区向下离开70",
    )
    candidates["SESSION_VWAP_CROSS"] = (
        _cross_up(close, data["session_vwap"], same_day),
        _cross_down(close, data["session_vwap"], same_day),
        "价格上穿/下穿当日累计VWAP",
    )

    middle = close.rolling(20, min_periods=20).mean()
    band_std = close.rolling(20, min_periods=20).std(ddof=1)
    lower = middle - 2.0 * band_std
    upper = middle + 2.0 * band_std
    candidates["BOLLINGER_20_2_REENTRY"] = (
        same_day & close.gt(lower) & close.shift(1).le(lower.shift(1)),
        same_day & close.lt(upper) & close.shift(1).ge(upper.shift(1)),
        "布林带(20,2)下轨/上轨回归",
    )
    high_volume = data["time_of_day_rvol_median_20d"].gt(1.3)
    candidates["ORB_30M_RVOL_BREAKOUT"] = (
        data["orb_breakout_up_bps"].gt(0.0) & high_volume,
        data["orb_breakout_down_bps"].lt(0.0) & high_volume,
        "前30分钟区间突破并由同时间段RVOL>1.3确认",
    )
    candidates["FVG_15M_FORMATION"] = (
        data["bullish_fvg"].fillna(False).astype(bool),
        data["bearish_fvg"].fillna(False).astype(bool),
        "标准三柱15分钟多头/空头FVG形成",
    )
    candidates["ADX_DI_14_CROSS"] = (
        _cross_up(data["plus_di_14bar"], data["minus_di_14bar"], same_day)
        & data["adx_14bar"].gt(20.0),
        _cross_down(data["plus_di_14bar"], data["minus_di_14bar"], same_day)
        & data["adx_14bar"].gt(20.0),
        "ADX14>20时+DI/-DI方向交叉",
    )

    output: dict[str, tuple[EventMap, EventMap, str]] = {}
    for candidate_id, (buy, sell, description) in candidates.items():
        output[candidate_id] = (
            _first_events(data, buy, candidate_id, "BUY"),
            _first_events(data, sell, candidate_id, "SELL"),
            description,
        )

    existing = indicator.copy().sort_values("signal_asof").reset_index(drop=True)
    output["COMPOSITE_RVOL_VWAP_EMA_ADX_RSI_ORB"] = (
        _first_events(existing, existing["is_entry_event"], "COMPOSITE", "BUY"),
        _first_events(existing, existing["is_exit_event"], "COMPOSITE", "SELL"),
        "既有RVOL、VWAP、EMA、ADX、RSI与ORB状态组合",
    )
    return output


def events_from_trade_records(
    trades: pd.DataFrame,
    candidate_id: str,
    direction_column: str = "direction",
    time_column: str = "entry_execution_time",
    price_column: str = "entry_raw_open_cny",
) -> tuple[EventMap, EventMap]:
    """把既有预注册策略的实际信号成交记录转换为时点事件。"""

    frame = trades.copy()
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame = frame.sort_values(time_column)
    buy: EventMap = {}
    sell: EventMap = {}
    for row in frame.itertuples(index=False):
        timestamp = pd.Timestamp(getattr(row, time_column))
        direction = str(getattr(row, direction_column))
        event = {
            "signal_asof": timestamp,
            "technical_signal": f"{candidate_id}_{direction}",
            "raw_execution_price": float(getattr(row, price_column)),
        }
        target = buy if direction in {"BUY_LOW", "LONG", "BUY"} else sell
        target.setdefault(timestamp.normalize(), event)
    return buy, sell


def events_from_close_pressure_forecasts(
    forecasts: pd.DataFrame,
    daily: pd.DataFrame,
) -> tuple[EventMap, EventMap]:
    """概率高于/低于当时扩展基准率时，分别视为次日买入/卖出确认。"""

    calendar = pd.to_datetime(daily["date"]).dt.normalize().drop_duplicates().sort_values().tolist()
    next_date = {calendar[index]: calendar[index + 1] for index in range(len(calendar) - 1)}
    open_by_date = daily.assign(date=pd.to_datetime(daily["date"]).dt.normalize()).set_index("date")["open"]
    frame = forecasts.copy().sort_values("signal_date")
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    buy: EventMap = {}
    sell: EventMap = {}
    for row in frame.itertuples(index=False):
        execution_date = next_date.get(pd.Timestamp(row.signal_date))
        if execution_date is None or execution_date not in open_by_date.index:
            continue
        probability = float(row.probability_positive_2d_net)
        baseline = float(row.expanding_base_rate_probability)
        event = {
            "signal_asof": pd.Timestamp(row.signal_date) + pd.Timedelta(hours=15),
            "technical_signal": "CLOSE_PRESSURE_PROBABILITY_ABOVE_BASE" if probability > baseline else "CLOSE_PRESSURE_PROBABILITY_BELOW_BASE",
            "raw_execution_price": float(open_by_date.loc[execution_date]),
        }
        (buy if probability > baseline else sell).setdefault(execution_date, event)
    return buy, sell
