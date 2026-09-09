"""只在父正目标段开始时用长期趋势确定进入资格。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = "EPISODE_TREND_ADMISSION"
MODELS = ["TREND_NOISE_REFERENCE_BLEND"]


def checked_long_trend(data, cfg):
    require(cfg["entry_trend_window"] == 200 and cfg["entry_trend_threshold"] == 0, "长期趋势进入条件改变")
    wealth = data.wealth.to_numpy(float)
    require((np.isnan(wealth) | (np.isfinite(wealth) & (wealth > 0))).all(), "长期趋势财富非法")
    expected = data.wealth/data.wealth.rolling(200, min_periods=200).mean()-1
    np.testing.assert_allclose(data.sma200, expected, atol=1e-12, rtol=1e-12, equal_nan=True)
    return data.sma200.to_numpy(float)


def episode_trend_frames(data, parents_by_cost, cfg, start):
    require(cfg["combination"] == "FIRST_POSITIVE_PARENT_EPISODE_LONG_TREND_ADMISSION", "父正目标段进入机制改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    trend = checked_long_trend(data, cfg)
    summaries = []
    for cost, frame in frames.items():
        parent = frame[MODELS[0]+"_parent_target"].to_numpy(float)
        target, entry_trend, episode_number = [np.full(len(data), np.nan) for _ in range(3)]
        gate = np.full(len(data), "OUTSIDE_DECISION_CALENDAR", dtype=object)
        entry_origin = np.full(len(data), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        started, ended = np.zeros(len(data), bool), np.zeros(len(data), bool)
        current_gate, current_entry, current_trend = None, None, np.nan
        count, counts = 0, {"ADMITTED": 0, "REJECTED": 0, "NO_VIEW_ENTRY_TREND": 0}
        for t in range(first-1, len(data)-1):
            if np.isnan(parent[t]):
                gate[t] = current_gate if current_gate is not None else "NO_ACTIVE_EPISODE"
            elif parent[t] == 0:
                target[t], gate[t] = 0., "PARENT_ZERO"
                ended[t] = current_gate is not None
                current_gate, current_entry, current_trend = None, None, np.nan
            else:
                if current_gate is None:
                    count += 1
                    current_entry, current_trend = t, trend[t]
                    current_gate = "NO_VIEW_ENTRY_TREND" if np.isnan(trend[t]) else "ADMITTED" if trend[t] > 0 else "REJECTED"
                    counts[current_gate] += 1
                    started[t] = True
                gate[t] = current_gate
                if current_gate == "ADMITTED":
                    target[t] = parent[t]
                elif current_gate == "REJECTED":
                    target[t] = 0.
            if current_gate is not None:
                episode_number[t], entry_trend[t] = count, current_trend
                entry_origin[t] = data.date.iloc[current_entry].to_datetime64()
        frame["long_trend_deviation200"] = trend
        frame["episode_gate"] = gate
        frame["episode_number"] = episode_number
        frame["episode_entry_origin"] = entry_origin
        frame["episode_entry_trend200"] = entry_trend
        frame["episode_started"] = started
        frame["episode_ended"] = ended
        frame["target"] = target
        eligible = frame.iloc[first-1:-1]
        summaries.append({"cost": cost, "decision_origins": len(eligible), "positive_target_origins": int(eligible.target.gt(0).sum()),
            "zero_target_origins": int(eligible.target.eq(0).sum()), "unknown_target_origins": int(eligible.target.isna().sum()),
            "parent_signal_episodes": count, "admission_counts": counts, "ended_signal_episodes": int(ended.sum()),
            "last_signal_episode_still_active": current_gate is not None, "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
