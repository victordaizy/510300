"""同一父信号段内分别生成只减和只增的两条预算路径。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = "EPISODE_BUDGET_NONINCREASING"
CANDIDATES = {PRIMARY: "信号段内预算只减不增", "EPISODE_BUDGET_NONDECREASING": "信号段内预算只增不减"}
MODELS = ["TREND_NOISE_REFERENCE_BLEND"]


def monotone_episode_frames(data, parents_by_cost, cfg, start):
    require(cfg["combination"] == "BOTH_MONOTONE_PARENT_EPISODE_BUDGETS" and cfg["candidate_models"] == list(CANDIDATES), "单向预算固定候选改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    summaries = []
    for cost, frame in frames.items():
        parent = frame[MODELS[0]+"_parent_target"].to_numpy(float)
        lower, upper = np.full(len(data), np.nan), np.full(len(data), np.nan)
        episode = np.zeros(len(data), dtype=int)
        starts = np.full(len(data), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        active, number, low, high, origin, ended = False, 0, np.nan, np.nan, None, 0
        for t in range(first-1, len(data)-1):
            value = parent[t]
            if pd.isna(value):
                episode[t] = number if active else 0
                if active:
                    starts[t] = origin
                continue
            if value == 0:
                if active:
                    ended += 1
                active, low, high, origin = False, np.nan, np.nan, None
                lower[t], upper[t] = 0., 0.
                continue
            if not active:
                active, low, high = True, value, value
                number += 1
                origin = data.date.iloc[t].to_datetime64()
            else:
                low, high = min(low, value), max(high, value)
            lower[t], upper[t], episode[t], starts[t] = low, high, number, origin
        frame["parent_signal_episode"] = episode
        frame["parent_episode_start"] = starts
        for model, target in zip(CANDIDATES, [lower, upper]):
            frame[model+"_target"] = target
            eligible = target[first-1:-1]
            old = parent[first-1:-1]
            summaries.append({"model": model, "cost": cost, "decision_origins": len(eligible), "parent_signal_episodes": number,
                "ended_signal_episodes": ended, "last_signal_episode_active": active, "positive_target_origins": int((eligible > 0).sum()),
                "zero_target_origins": int((eligible == 0).sum()), "unknown_target_origins": int(np.isnan(eligible).sum()),
                "changed_parent_target_origins": int((np.isfinite(eligible) & np.isfinite(old) & (eligible != old)).sum()),
                "mean_target": float(np.nanmean(eligible))})
    return frames, summaries
