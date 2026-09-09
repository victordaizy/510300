"""因果含分红财富尺度的真实波幅与趋势通道。"""
import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require


def band_frame(data, window=10, multiplier=3.0):
    require(isinstance(window, int) and window > 0, "波幅窗口必须为正整数")
    require(np.isfinite(multiplier) and multiplier > 0, "通道倍数必须为正")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "价格日期必须严格递增")
    raw = data[["high", "low", "close", "dividend", "wealth"]].to_numpy(float)
    require((np.isnan(raw) | np.isfinite(raw)).all(), "价格或分红不能为无穷大")
    for column in [0, 1, 2, 4]:
        require((np.isnan(raw[:, column]) | (raw[:, column] > 0)).all(), "价格及财富必须为正")
    require((np.isnan(raw[:, 3]) | (raw[:, 3] >= 0)).all(), "现金分红不能为负")
    valid = np.isfinite(raw).all(axis=1)
    require((raw[valid, 1] <= raw[valid, 2]).all() and (raw[valid, 2] <= raw[valid, 0]).all(), "收盘不在最高最低价格范围内")
    high = raw[:, 4] * (raw[:, 0] + raw[:, 3]) / (raw[:, 2] + raw[:, 3])
    low = raw[:, 4] * (raw[:, 1] + raw[:, 3]) / (raw[:, 2] + raw[:, 3])
    close = raw[:, 4]
    n = len(data)
    values = {k: np.full(n, np.nan) for k in ["true_range", "atr", "basic_upper", "basic_lower", "upper_band", "lower_band", "active_band", "trend_state", "target"]}
    statuses = np.full(n, "NO_VIEW_WARMUP", object)
    seed, average, upper, lower, direction = [], np.nan, np.nan, np.nan, None
    for t in range(n):
        if t == 0 or not valid[t] or not np.isfinite(close[t - 1]):
            seed, average, upper, lower, direction = [], np.nan, np.nan, np.nan, None
            statuses[t] = "NO_VIEW_FIRST_PREVIOUS_CLOSE" if t == 0 else "NO_VIEW_INPUT_RESET"
            continue
        tr = max(high[t] - low[t], abs(high[t] - close[t - 1]), abs(low[t] - close[t - 1]))
        values["true_range"][t] = tr
        if not np.isfinite(average):
            seed.append(tr)
            if len(seed) < window:
                continue
            average = float(np.mean(seed))
        else:
            average = ((window - 1) * average + tr) / window
        midpoint = .5 * (high[t] + low[t])
        basic_upper, basic_lower = midpoint + multiplier * average, midpoint - multiplier * average
        if direction is None:
            upper, lower, direction = basic_upper, basic_lower, 0
            statuses[t] = "VIEW_INITIAL_DOWN_STATE"
        else:
            upper = basic_upper if basic_upper < upper or close[t - 1] > upper else upper
            lower = basic_lower if basic_lower > lower or close[t - 1] < lower else lower
            direction = int(close[t] > upper) if direction == 0 else int(not close[t] < lower)
            statuses[t] = "VIEW_UP_STATE" if direction else "VIEW_DOWN_STATE"
        for key, value in {"atr": average, "basic_upper": basic_upper, "basic_lower": basic_lower,
            "upper_band": upper, "lower_band": lower, "active_band": lower if direction else upper,
            "trend_state": direction, "target": direction}.items():
            values[key][t] = value
    return pd.DataFrame({"date": dates, "wealth_high": high, "wealth_low": low, "wealth_close": close,
        "input_valid": valid, "source_state": statuses, **values})
