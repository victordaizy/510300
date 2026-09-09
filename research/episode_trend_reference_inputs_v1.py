"""把趋势选择限制到信号周期起点，明确零目标结束来源承诺。"""
import numpy as np
import pandas as pd
from research.trend_reference_router_inputs_v1 import trend_routed_frames, MODELS
from research.intraday_overnight_increment_v1 import require

PRIMARY = "EPISODE_TREND_REFERENCE"


def episode_routed_frames(data, parents_by_cost, cfg, start):
    """复用已有趋势与目标对齐，分别保存两费用的来源状态。"""
    require(cfg["source_commitment"] == "UNTIL_SELECTED_PARENT_EXPLICIT_ZERO", "信号周期来源释放规则改变")
    frames, _ = trend_routed_frames(data, parents_by_cost, cfg, start)
    first = int(np.flatnonzero(data.date.ge(start))[0])
    summaries = []
    for cost_id, factors in frames.items():
        factors["daily_trend_choice"] = factors.selected_parent.copy()
        selected = np.full(len(data), None, object)
        before, after = selected.copy(), selected.copy()
        targets = np.full(len(data), np.nan)
        ids = np.full(len(data), np.nan)
        started, released, changes = (np.zeros(len(data), bool) for _ in range(3))
        states = np.full(len(data), "NO_VIEW_OUTSIDE_DECISION_PERIOD", object)
        parents = {MODELS[0]: factors.trend_parent_target.to_numpy(float), MODELS[1]: factors.other_parent_target.to_numpy(float)}
        locked, previous, serial = None, None, 0
        for t in range(first-1, len(data)-1):
            before[t] = locked
            candidate = locked if locked is not None else factors.daily_trend_choice.iloc[t]
            candidate = None if pd.isna(candidate) else candidate
            selected[t] = candidate
            if candidate is None:
                states[t] = "NO_VIEW_ENTRY_TREND"
                continue
            targets[t] = parents[candidate][t]
            if previous is not None:
                changes[t] = candidate != previous
            previous = candidate
            if locked is not None:
                ids[t] = serial
                if np.isnan(targets[t]):
                    states[t] = "LOCK_HELD_NO_VIEW_PARENT_TARGET"
                elif targets[t] == 0:
                    released[t], locked = True, None
                    states[t] = "LOCK_RELEASED_EXPLICIT_ZERO"
                else:
                    states[t] = "LOCK_HELD_POSITIVE_TARGET"
            elif np.isnan(targets[t]):
                states[t] = "NO_VIEW_ENTRY_PARENT_TARGET"
            elif targets[t] == 0:
                states[t] = "NO_LOCK_KNOWN_ZERO_TARGET"
            else:
                serial += 1
                ids[t], locked, started[t] = serial, candidate, True
                states[t] = "LOCK_STARTED_POSITIVE_TARGET"
            after[t] = locked
        factors["selected_parent"] = selected
        factors["locked_parent_before"] = before
        factors["locked_parent_after"] = after
        factors["episode_id"] = ids
        factors["episode_started"] = started
        factors["episode_released"] = released
        factors["selection_changed"] = changes
        factors["episode_status"] = states
        factors["target"] = targets
        eligible = factors.iloc[first-1:-1]
        summaries.append({"cost": cost_id, "decision_origins": len(eligible), "selection_changes": int(changes.sum()),
            "signal_episodes_started": int(started.sum()), "signal_episodes_released_by_zero": int(released.sum()),
            "selected_parent_at_last_decision": locked,
            "selected_origins": {str(k): int(v) for k, v in eligible.selected_parent.value_counts().items()},
            "positive_target_origins": int(eligible.target.gt(0).sum()), "zero_target_origins": int(eligible.target.eq(0).sum()),
            "unknown_target_origins": int(eligible.target.isna().sum()), "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
