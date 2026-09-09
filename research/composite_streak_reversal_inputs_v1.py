"""三项短期反转指标；只读取当前及以前的分红连续价格。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require


def wilder_rsi(values, period):
    """以首批完整变动的简单均值启动，缺失后重新准备。"""
    require(isinstance(period, int) and period > 0, "相对强弱周期必须为正整数")
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    previous, gain, loss, count = np.nan, 0., 0., 0
    for t, value in enumerate(values):
        if not np.isfinite(value):
            previous, gain, loss, count = np.nan, 0., 0., 0
            continue
        if not np.isfinite(previous):
            previous = value
            continue
        change, previous = value - previous, value
        positive, negative = max(change, 0.), max(-change, 0.)
        if count < period:
            gain, loss, count = gain + positive, loss + negative, count + 1
            if count < period:
                continue
            gain, loss = gain / period, loss / period
        else:
            gain = (gain * (period - 1) + positive) / period
            loss = (loss * (period - 1) + negative) / period
        result[t] = 50. if gain + loss == 0 else 100. * gain / (gain + loss)
    return result


def signed_streak(values):
    """涨跌变向从正负一天重新计数，平盘归零。"""
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    previous, streak = np.nan, 0
    for t, value in enumerate(values):
        if not np.isfinite(value):
            previous, streak = np.nan, 0
            continue
        if not np.isfinite(previous) or value == previous:
            streak = 0
        elif value > previous:
            streak = max(streak, 0) + 1
        else:
            streak = min(streak, 0) - 1
        result[t], previous = streak, value
    return result


def prior_return_rank(values, window):
    """当前日收益与此前完整窗口比较；平手不算严格较小。"""
    require(isinstance(window, int) and window > 0, "排名窗口必须为正整数")
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    for t in range(window, len(values)):
        previous = values[t-window:t]
        if np.isfinite(values[t]) and np.isfinite(previous).all():
            result[t] = 100. * np.count_nonzero(previous < values[t]) / window
    return result


def factor_frame(data):
    """唯一生产参数为价格三期、天数两期、历史排名一百日和均线二百日。"""
    wealth = pd.Series(data.wealth.to_numpy(float))
    require((wealth.isna() | (wealth > 0)).all(), "连续价格不能非正")
    daily_return = wealth / wealth.shift(1) - 1
    streak = signed_streak(wealth)
    price_rsi, streak_rsi = wilder_rsi(wealth, 3), wilder_rsi(streak, 2)
    rank = prior_return_rank(daily_return, 100)
    score = (price_rsi + streak_rsi + rank) / 3.
    average = wealth.rolling(200, min_periods=200).mean().to_numpy()
    valid = np.isfinite(score) & np.isfinite(average) & wealth.notna().to_numpy()
    return pd.DataFrame({"date": data.date.to_numpy(), "wealth_close": wealth, "price_rsi3": price_rsi,
        "signed_streak": streak, "streak_rsi2": streak_rsi, "daily_wealth_return": daily_return,
        "strict_rank100": rank, "composite_score": score, "wealth_sma200": average,
        "factor_valid": valid, "factor_state": np.where(valid, "VIEW_COMPOSITE_AVAILABLE", "NO_VIEW_INCOMPLETE_INPUT")})


def trading_rule(factors):
    """没有因子看法时不新进场；独立可知的均线退出仍然有效。"""
    above = factors.wealth_close > factors.wealth_sma200
    entry = factors.factor_valid & above & (factors.composite_score < 10)
    exit_flag = (factors.composite_score > 70) | (factors.wealth_close <= factors.wealth_sma200)
    return {"entry": entry.to_numpy(int), "exit": {1: exit_flag.to_numpy(bool)}}
