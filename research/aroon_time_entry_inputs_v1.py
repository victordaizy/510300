"""用局部含分红高低财富的极值时间产生一个固定进入退出信号。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def adjusted_window(high, low, close, dividend):
    high, low, close, dividend = [np.asarray(x, float) for x in [high, low, close, dividend]]
    increments = 1+dividend/close
    before = np.r_[1., np.cumprod(increments[:-1])]
    return before*(high+dividend), before*(low+dividend)


def most_recent_age(values, maximum, tolerance):
    values = np.asarray(values, float)
    extreme = values.max() if maximum else values.min()
    equal = np.abs(values-extreme) <= tolerance*max(1., abs(extreme))
    return len(values)-1-int(np.flatnonzero(equal)[-1])


def factor_frame(data, period=25, entry_up=70., entry_down=30., tie_tolerance=1e-12):
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates and period >= 2, "高低点时间日历或周期错误")
    raw = data[["high", "low", "close", "dividend"]].to_numpy(float)
    results = []
    for t in range(len(data)):
        up, down, age_high, age_low = [float("nan")]*4
        status, start = "NO_VIEW_INSUFFICIENT_EXTREMA_WINDOW", pd.NaT
        if t >= period:
            start = dates[t-period]
            window = raw[t-period:t+1]
            valid = np.isfinite(window).all() and (window[:, :3] > 0).all() and (window[:, 3] >= 0).all() and (window[:, 0] >= window[:, 2]).all() and (window[:, 2] >= window[:, 1]).all()
            if valid:
                high, low = adjusted_window(*window.T)
                require(np.isfinite(high).all() and np.isfinite(low).all(), "含分红极值价格发生数值失效")
                age_high, age_low = most_recent_age(high, True, tie_tolerance), most_recent_age(low, False, tie_tolerance)
                up, down = 100.*(period-age_high)/period, 100.*(period-age_low)/period
                status = "AROON_TIME_FACTORS_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_OR_INVALID_EXTREMA_WINDOW"
        valid = status == "AROON_TIME_FACTORS_AVAILABLE"
        results.append({"date": dates[t], "factor_status": status, "window_start": start, "window_observations": min(t+1, period+1), "high_age_trading_days": age_high, "low_age_trading_days": age_low,
            "aroon_up": up, "aroon_down": down, "raw_entry": int(valid and up >= entry_up and down <= entry_down), "raw_exit": bool(valid and down > up)})
    return pd.DataFrame(results)


def rules(factors):
    return {"entry": factors.raw_entry.to_numpy(int), "exit": {1: factors.raw_exit.to_numpy(bool)}}


def attach_factor_context(decisions, factors):
    saved = decisions.merge(factors.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
    missing = saved.factor_status.ne("AROON_TIME_FACTORS_AVAILABLE")
    require((saved.loc[missing, "requested_quantity"] <= 0).all(), "高低点因子无观点时申请了新的买入")
    unchanged = missing & saved.requested_quantity.eq(0)
    saved.loc[unchanged, "reference_weight"] = np.nan
    saved.loc[unchanged, "action"] = "高低点时间因子无观点，保持已有份额；原价格和期限退出继续"
    return saved
