"""用两条等额起始参考账户的已实现净值分配预算。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def budget_frame(dates, reference_equities, expert_states, first):
    dates = pd.DatetimeIndex(dates)
    equities, states = np.asarray(reference_equities, float), np.asarray(expert_states, float)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "财富预算日历必须严格递增")
    require(equities.shape == states.shape == (len(dates), 2), "两参考净值及状态形状不同")
    require(1 <= first < len(dates)-1, "财富预算起点缺少准备收盘或终点")
    require((np.isnan(states) | np.isin(states, [0., 1.])).all(), "参考状态须为零、一或缺失")
    rows, weights, last = [], np.array([.5, .5]), pd.NaT
    for t, date in enumerate(dates):
        active = first-1 <= t < len(dates)-1
        if not active:
            row_weights, target, status = [np.nan, np.nan], np.nan, "NO_VIEW_OUTSIDE_DECISION_PERIOD"
        else:
            status = "INITIAL_EQUAL_REFERENCE_WEALTH"
            if t >= first:
                pair = equities[t]
                if np.isfinite(pair).all() and (pair > 0).all():
                    normalized = pair/pair.max()
                    weights = normalized/normalized.sum()
                    last, status = date, "REFERENCE_WEALTH_BUDGET_AVAILABLE"
                else:
                    status = "NO_VIEW_INVALID_REFERENCE_EQUITY_KEEP_BUDGET"
            row_weights = weights
            target = float(states[t] @ weights) if np.isfinite(states[t]).all() else np.nan
        rows.append({"date": date, "panic_budget": row_weights[0], "learned_budget": row_weights[1],
            "panic_reference_equity": equities[t, 0], "learned_reference_equity": equities[t, 1],
            "panic_state": states[t, 0], "learned_state": states[t, 1], "target": target,
            "budget_status": status, "budget_update_scheduled": active and t >= first,
            "last_successful_budget_origin": last if active else pd.NaT})
    return pd.DataFrame(rows)
