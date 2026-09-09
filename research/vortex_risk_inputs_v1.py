"""相邻经济高低价的14日方向与20日普通波动预算。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = "VORTEX_RISK"
CANDIDATES = {PRIMARY: "相邻高低价方向与普通波动预算"}


def economic_ohlc(data):
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 0 and dates.is_monotonic_increasing and not dates.has_duplicates, "经济价格日历必须完整递增")
    columns = ["open", "high", "low", "close", "previous_close", "dividend", "wealth"]
    values = data[columns].to_numpy(float)
    require((np.isnan(values) | np.isfinite(values)).all(), "经济价格来源包含无穷值")
    for column in ["open", "high", "low", "close", "previous_close", "wealth"]:
        x = data[column].to_numpy(float)
        require((np.isnan(x) | (x > 0)).all(), "经济价格或财富已知但非正")
    require((data.dividend.isna() | data.dividend.ge(0)).all(), "现金分红不允许负值")
    raw = data[["open", "high", "low", "close"]].to_numpy(float)
    complete = np.isfinite(raw).all(axis=1)
    require((raw[complete, 1] >= np.maximum(raw[complete, 0], raw[complete, 3])).all() and
            (raw[complete, 2] <= np.minimum(raw[complete, 0], raw[complete, 3])).all(), "真实开高低收次序不合法")
    previous = data.previous_close.to_numpy(float)
    paired = np.isfinite(previous[1:]) & np.isfinite(raw[:-1, 3])
    require(np.allclose(previous[1:][paired], raw[:-1, 3][paired], atol=1e-12, rtol=1e-12), "前收盘没有对应上一真实收盘")
    wealth, dividends = data.wealth.to_numpy(float), data.dividend.to_numpy(float)
    scales = np.r_[wealth[0]/(raw[0, 3]+dividends[0]), wealth[:-1]/previous[1:]]
    adjusted = (raw+dividends[:, None])*scales[:, None]
    known = complete & np.isfinite(dividends) & np.isfinite(wealth) & np.isfinite(scales)
    adjusted[~known] = np.nan
    require(np.allclose(adjusted[known, 3], wealth[known], atol=1e-12, rtol=1e-12), "经济收盘不能还原含分红财富")
    return pd.DataFrame({"date": dates, **{"economic_"+name: adjusted[:, j] for j, name in enumerate(["open", "high", "low", "close"])}})


def vortex_factors(data, window=14):
    require(window == 14, "涡旋方向只允许固定14日窗口")
    f = economic_ohlc(data)
    high, low, close = f.economic_high, f.economic_low, f.economic_close
    plus = (high-low.shift()).abs()
    minus = (low-high.shift()).abs()
    ranges = np.column_stack([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()])
    tr = pd.Series(np.max(ranges, axis=1))
    f["positive_movement"], f["negative_movement"], f["true_range"] = plus, minus, tr
    for name, values in [("positive_sum14", plus), ("negative_sum14", minus), ("true_range_sum14", tr)]:
        f[name] = values.rolling(window, min_periods=window).sum()
    denominator = f.true_range_sum14.where(f.true_range_sum14.gt(0))
    f["vortex_plus"] = f.positive_sum14 / denominator
    f["vortex_minus"] = f.negative_sum14 / denominator
    known = np.isfinite(f[["vortex_plus", "vortex_minus"]]).all(axis=1)
    f["positive_direction"] = np.where(known, f.vortex_plus.gt(f.vortex_minus).astype(float), np.nan)
    volatility = data.vol20.to_numpy(float)
    require((np.isnan(volatility) | (np.isfinite(volatility) & (volatility >= 0))).all(), "普通波动输入存在无穷或负值")
    f["volatility20"] = volatility
    return f


def vortex_risk_frames(data, parents_by_cost, cfg, start):
    require(cfg["candidate_models"] == list(CANDIDATES) and cfg["vortex_window"] == 14 and cfg["risk_target"] == .1, "相邻高低价固定设置改变")
    frames, first = aligned_target_frames(data, parents_by_cost, [], cfg, start)
    factors = vortex_factors(data, cfg["vortex_window"])
    indices = np.arange(first-1, len(data)-1)
    directions, volatility = factors.positive_direction.to_numpy(), factors.volatility20.to_numpy()
    target = np.full(len(data), np.nan)
    nonpositive = indices[directions[indices] == 0]
    target[nonpositive] = 0.
    positive = indices[(directions[indices] == 1) & np.isfinite(volatility[indices]) & (volatility[indices] > 0)]
    target[positive] = np.minimum(1., cfg["risk_target"] / volatility[positive])
    summaries = []
    for cost, frame in frames.items():
        for column in factors.columns[1:]:
            frame[column] = factors[column].to_numpy()
        frame[PRIMARY+"_target"] = target.copy()
        x = target[indices]
        summaries.append({"model": PRIMARY, "cost": cost, "decision_origins": len(indices),
            "positive_target_origins": int((x > 0).sum()), "zero_target_origins": int((x == 0).sum()),
            "unknown_target_origins": int(np.isnan(x).sum()), "equal_direction_origins": int((factors.vortex_plus.iloc[indices] == factors.vortex_minus.iloc[indices]).sum()),
            "risk_capped_origins": int(((x > 0) & (x < 1)).sum()), "full_target_origins": int((x == 1).sum())})
    return frames, summaries
