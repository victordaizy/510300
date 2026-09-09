"""以当时可见的趋势与短期动量形成四个市场状态，不计算动作收益。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require

STATE_NAMES = {0: "长期不在均线上方、短期未上涨", 1: "长期不在均线上方、短期上涨",
               2: "长期在均线上方、短期未上涨", 3: "长期在均线上方、短期上涨"}


def market_state_frame(data):
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "联合状态日期必须唯一递增")
    values = data[["sma120", "mom5"]].to_numpy(float)
    require((np.isfinite(values) | np.isnan(values)).all(), "联合市场状态来源不能是无穷值")
    valid = np.isfinite(values).all(axis=1)
    trend, momentum = values[:, 0] > 0, values[:, 1] > 0
    regime = np.where(valid, 2*trend.astype(int)+momentum.astype(int), np.nan)
    return pd.DataFrame({"date": dates, "sma120": values[:, 0], "mom5": values[:, 1],
                         "market_state": regime, "state_status": np.where(valid, "MARKET_STATE_AVAILABLE", "NO_VIEW_MARKET_STATE_INPUTS")})
