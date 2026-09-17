"""六十日全部成对对数财富斜率中位数与普通波动仓位。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = "MEDIAN_SLOPE_RISK"
CANDIDATES = {PRIMARY: "六十日成对斜率中位数与普通波动预算"}


def median_slope_factors(data):
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 0 and not dates.hasnans and dates.is_monotonic_increasing and not dates.has_duplicates,
            "斜率来源日期必须已知、唯一并递增")
    wealth = data.wealth.to_numpy(float)
    volatility = data.vol20.to_numpy(float)
    require((np.isnan(wealth) | (np.isfinite(wealth) & (wealth > 0))).all(), "斜率财富包含非正或无穷值")
    require((np.isnan(volatility) | (np.isfinite(volatility) & (volatility >= 0))).all(), "普通波动包含负数或无穷值")
    logs = np.log(wealth)
    slopes = np.full(len(data), np.nan)
    if len(data) >= 60:
        windows = np.lib.stride_tricks.sliding_window_view(logs, 60)
        complete = np.isfinite(windows).all(axis=1)
        earlier, later = np.triu_indices(60, 1)
        paired = (windows[complete][:, later] - windows[complete][:, earlier]) / (later - earlier)
        slopes[np.flatnonzero(complete) + 59] = np.median(paired, axis=1)
    direction = np.where(np.isfinite(slopes), (slopes > 0).astype(float), np.nan)
    return pd.DataFrame({"date": dates, "log_wealth": logs, "median_pair_slope60": slopes,
                         "positive_direction": direction, "volatility20": volatility})


def median_slope_risk_frames(data, parents_by_cost, cfg, start):
    require(cfg["candidate_models"] == list(CANDIDATES) and cfg["slope_window"] == 60 and
            cfg["volatility_window"] == 20 and cfg["risk_target"] == .1, "成对斜率固定规则改变")
    frames, first = aligned_target_frames(data, parents_by_cost, [], cfg, start)
    factors = median_slope_factors(data)
    indices = np.arange(first - 1, len(data) - 1)
    direction, volatility = factors.positive_direction.to_numpy(), factors.volatility20.to_numpy()
    target = np.full(len(data), np.nan)
    target[indices[direction[indices] == 0]] = 0.
    positive = indices[(direction[indices] == 1) & np.isfinite(volatility[indices]) & (volatility[indices] > 0)]
    target[positive] = np.minimum(1., .1 / volatility[positive])
    summaries = []
    for cost, frame in frames.items():
        for column in factors.columns[1:]:
            frame[column] = factors[column].to_numpy()
        frame[PRIMARY + "_target"] = target.copy()
        current = target[indices]
        summaries.append({"model": PRIMARY, "cost": cost, "decision_origins": len(indices),
            "positive_target_origins": int((current > 0).sum()), "zero_target_origins": int((current == 0).sum()),
            "unknown_target_origins": int(np.isnan(current).sum()),
            "exact_zero_slope_origins": int((factors.median_pair_slope60.iloc[indices] == 0).sum()),
            "risk_capped_origins": int(((current > 0) & (current < 1)).sum()),
            "full_target_origins": int((current == 1).sum())})
    return frames, summaries
