"""使用过去区间中点及完整二十六期滞后构造云图，保留三值信号。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require

PERIODS = {"conversion": 9, "base": 26, "span_b": 52, "displacement": 26}


def strictly_above(left, right):
    """只排除浮点舍入造成的假突破，容差远小于价格最小刻度。"""
    return left-right > 1e-12*np.maximum(np.abs(left), np.abs(right))


def cloud_frame(data):
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "云图日期必须唯一递增")
    required = ["high", "low", "close", "previous_close", "dividend", "wealth"]
    raw = data[required].to_numpy(float)
    require((np.isfinite(raw) | np.isnan(raw)).all(), "云图来源不能包含无穷值")
    for column in ["high", "low", "close", "previous_close", "wealth"]:
        values = data[column].to_numpy(float)
        require((np.isnan(values) | (values > 0)).all(), "云图价格和财富必须为正")
    require((data.dividend.isna() | data.dividend.ge(0)).all(), "云图分红不能为负")
    valid_prices = data[["high", "low", "close"]].notna().all(axis=1)
    require((data.loc[valid_prices, "high"] >= data.loc[valid_prices, "close"]).all() and
            (data.loc[valid_prices, "close"] >= data.loc[valid_prices, "low"]).all(), "云图收盘不在高低价内")
    factor = data.wealth.shift(1)/data.previous_close
    high, low, close = (data.high+data.dividend)*factor, (data.low+data.dividend)*factor, (data.close+data.dividend)*factor
    out = pd.DataFrame({"date": dates, "wealth_high": high.to_numpy(), "wealth_low": low.to_numpy(), "wealth_close": close.to_numpy()})
    for name, count in [("conversion", 9), ("base", 26), ("range52", 52)]:
        upper = high.rolling(count, min_periods=count).max()
        lower = low.rolling(count, min_periods=count).min()
        out[name] = ((upper+lower)/2).to_numpy()
    out["computed_span_a"] = (out.conversion+out.base)/2
    out["computed_span_b"] = out.range52
    out["current_cloud_a"] = out.computed_span_a.shift(26)
    out["current_cloud_b"] = out.computed_span_b.shift(26)
    out["current_cloud_top"] = out[["current_cloud_a", "current_cloud_b"]].max(axis=1, skipna=False)
    out["current_cloud_bottom"] = out[["current_cloud_a", "current_cloud_b"]].min(axis=1, skipna=False)
    out["cloud_calculation_date"] = out.date.shift(26)
    entry_fields = ["wealth_close", "conversion", "base", "computed_span_a", "computed_span_b", "current_cloud_top"]
    entry_known = np.isfinite(out[entry_fields].to_numpy(float)).all(axis=1)
    bullish = strictly_above(out.wealth_close, out.current_cloud_top) & strictly_above(out.conversion, out.base) & strictly_above(out.computed_span_a, out.computed_span_b)
    out["entry_signal"] = np.where(entry_known, bullish.astype(float), np.nan)
    base_known = out[["wealth_close", "base"]].notna().all(axis=1)
    cloud_known = out[["wealth_close", "current_cloud_bottom"]].notna().all(axis=1)
    base_exit = base_known & strictly_above(out.base, out.wealth_close)
    cloud_exit = cloud_known & strictly_above(out.current_cloud_bottom, out.wealth_close)
    any_exit = base_exit | cloud_exit
    exit_known = any_exit | (base_known & cloud_known)
    out["exit_signal"] = np.where(exit_known, any_exit.astype(float), np.nan)
    out["entry_state"] = np.where(entry_known, np.where(bullish, "ENTRY_CONDITION_TRUE", "ENTRY_CONDITION_FALSE"), "NO_VIEW_ENTRY_INPUT_OR_HISTORY")
    out["exit_state"] = np.where(exit_known, np.where(any_exit, "PRICE_EXIT_TRUE", "PRICE_EXIT_FALSE"), "NO_VIEW_EXIT_INPUT_OR_HISTORY")
    out["exit_by_base"] = base_exit
    out["exit_by_cloud"] = cloud_exit
    require(out.entry_signal.dropna().isin([0., 1.]).all() and out.exit_signal.dropna().isin([0., 1.]).all(), "云图信号不是预定三值状态")
    return out


def cloud_rule(states):
    return {"entry": states.entry_signal.to_numpy(float), "exit": {1: states.exit_signal.to_numpy(float)}}
