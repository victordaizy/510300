"""用完整对数财富窗口的上升方向及拟合优度分配来源预算。"""
import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = "TREND_COHERENCE_BLEND"
MODELS = ["VINTAGE_REFERENCE_RISK", "TREND_NOISE_REFERENCE_BLEND"]


def rolling_trend_coherence(data, cfg):
    require(cfg["trend_window"] == 120, "线性连贯程度窗口改变")
    wealth = data.wealth.to_numpy(float)
    require((np.isnan(wealth) | (np.isfinite(wealth) & (wealth > 0))).all(), "对数财富出现非正值或无穷值")
    slopes, quality = np.full(len(data), np.nan), np.full(len(data), np.nan)
    if len(data) < 120:
        return slopes, quality
    windows = np.lib.stride_tricks.sliding_window_view(np.log(wealth), 120)
    valid = np.isfinite(windows).all(axis=1)
    chosen = windows[valid]
    centered = chosen-chosen.mean(axis=1, keepdims=True)
    x = np.arange(120, dtype=float)-59.5
    xx = np.dot(x, x)
    cross = centered @ x
    yy = np.einsum("ij,ij->i", centered, centered)
    constant = (chosen == chosen[:, :1]).all(axis=1)
    cross[constant], yy[constant] = 0., 0.
    slopes[119+np.flatnonzero(valid)] = cross/xx
    quality[119+np.flatnonzero(valid)] = np.clip(np.divide(cross*cross, xx*yy, out=np.zeros(len(yy)), where=yy > 0), 0, 1)
    return slopes, quality


def trend_coherence_frames(data, parents_by_cost, cfg, start):
    require(cfg["combination"] == "POSITIVE_LOG_TREND_R_SQUARED_BUDGET", "趋势连贯程度预算定义改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    slopes, quality = rolling_trend_coherence(data, cfg)
    known = np.isfinite(slopes) & np.isfinite(quality)
    budget = np.full(len(data), np.nan)
    budget[known] = np.where(slopes[known] > 0, quality[known], 0.)
    summaries = []
    for cost, frame in frames.items():
        frame["log_wealth_slope120"] = slopes
        frame["trend_r_squared120"] = quality
        frame["reference_budget131"] = budget
        frame["reference_budget143"] = 1-budget
        x, y = (frame[model+"_parent_target"].to_numpy(float) for model in MODELS)
        available = known & np.isfinite(x) & np.isfinite(y)
        target = np.full(len(data), np.nan)
        target[available] = np.clip(budget[available]*x[available]+(1-budget[available])*y[available], 0, 1)
        frame["target"] = target
        eligible = frame.iloc[first-1:-1]
        summaries.append({"cost": cost, "decision_origins": len(eligible), "positive_target_origins": int(eligible.target.gt(0).sum()),
            "zero_target_origins": int(eligible.target.eq(0).sum()), "unknown_target_origins": int(eligible.target.isna().sum()),
            "mean_131_budget": float(eligible.reference_budget131.mean()), "maximum_131_budget": float(eligible.reference_budget131.max()),
            "positive_131_budget_origins": int(eligible.reference_budget131.gt(0).sum()),
            "complete_rolling_statistic_windows": int(eligible.trend_r_squared120.notna().sum()), "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
