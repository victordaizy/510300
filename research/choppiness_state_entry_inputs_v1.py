"""十四日价格路径曲折度与保留状态的趋势、震荡进出场。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def classify_choppiness(values, lower=38.2, upper=61.8):
    state, output = np.nan, []
    for value in values:
        if not np.isfinite(value):
            state = np.nan
        elif value < lower:
            state = 1.
        elif value > upper:
            state = 2.
        output.append(state)
    return np.asarray(output, float)


def factor_frame(data, period=14, lower=38.2, upper=61.8):
    require(isinstance(period, int) and period > 1 and 0 < lower < upper < 100, "曲折度设置不合法")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "曲折度日期必须唯一递增")
    raw = data[["high", "low", "close", "dividend", "wealth"]].to_numpy(float)
    valid = np.isfinite(raw).all(axis=1)
    require((raw[valid, :3] > 0).all() and (raw[valid, 4] > 0).all() and (raw[valid, 3] >= 0).all(), "价格财富或股息不合法")
    require((raw[valid, 1] <= raw[valid, 2]).all() and (raw[valid, 2] <= raw[valid, 0]).all(), "收盘不在高低价范围")
    high = raw[:, 4]*(raw[:, 0]+raw[:, 3])/(raw[:, 2]+raw[:, 3])
    low = raw[:, 4]*(raw[:, 1]+raw[:, 3])/(raw[:, 2]+raw[:, 3])
    close = raw[:, 4]
    previous = np.r_[np.nan, close[:-1]]
    true_range = np.maximum.reduce([high-low, np.abs(high-previous), np.abs(low-previous)])
    true_range[~valid] = np.nan
    full_range = pd.Series(high).rolling(period).max()-pd.Series(low).rolling(period).min()
    path_length = pd.Series(true_range).rolling(period).sum()
    quotient = path_length/full_range.where(full_range > 0)
    score = (100*np.log10(quotient.where(quotient > 0))/np.log10(period)).to_numpy()
    state = classify_choppiness(score, lower, upper)
    w = pd.Series(close)
    high20, low10 = w.shift().rolling(20).max(), w.shift().rolling(10).min()
    bull, z = data.sma120.to_numpy(float), data.z20.to_numpy(float)
    trend_entry = (state == 1) & (bull > 0) & (close > high20.to_numpy())
    range_entry = (state == 2) & (z < -1.5) & (close > previous)
    allowed = data.feature_valid.fillna(False).to_numpy(bool)
    entry = np.where(allowed & trend_entry, 1, np.where(allowed & range_entry, 2, 0))
    trend_exit = (state == 2) | (bull <= 0) | (close < low10.to_numpy())
    range_exit = (state == 1) | (z >= 0)
    status = np.where(np.isfinite(score), np.where(np.isfinite(state), "CHOPPINESS_STATE_AVAILABLE", "NO_VIEW_INITIAL_MIDDLE_STATE"), "NO_VIEW_INCOMPLETE_OR_DEGENERATE_CHOPPINESS")
    return pd.DataFrame({"date": dates, "wealth_high": high, "wealth_low": low, "wealth_close": close, "true_range": true_range,
                         "range_span": full_range, "path_length": path_length, "choppiness": score, "market_state": state,
                         "factor_status": status, "sma120": bull, "z20": z, "previous_high20": high20, "previous_low10": low10,
                         "raw_entry": entry, "raw_exit_trend": trend_exit, "raw_exit_range": range_exit})


def rules(factors):
    return {"entry": factors.raw_entry.to_numpy(int), "exit": {1: factors.raw_exit_trend.to_numpy(bool), 2: factors.raw_exit_range.to_numpy(bool)}}


def attach_factor_context(decisions, factors):
    saved = decisions.merge(factors.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
    unknown = saved.factor_status.ne("CHOPPINESS_STATE_AVAILABLE")
    require((saved.loc[unknown, "requested_quantity"] <= 0).all(), "曲折度无状态时新增买入")
    keep = unknown & saved.requested_quantity.eq(0)
    saved.loc[keep, "reference_weight"] = np.nan
    saved.loc[keep, "action"] = "曲折度无新状态，保持已有份额；原价格与期限退出继续"
    return saved
