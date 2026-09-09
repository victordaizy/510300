"""用正系数多项式的精确积分汇总全部固定混合比例，不挑历史赢家。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def advance(coefficients, gross):
    c, g = np.asarray(coefficients, float), np.asarray(gross, float)
    require(c.ndim == 1 and len(c) > 0 and np.isfinite(c).all() and (c >= 0).all() and c.max() > 0, "混合财富系数失效")
    require(g.shape == (2,) and np.isfinite(g).all() and (g > 0).all(), "参考完整净增长必须有效且为正")
    n = len(c)
    updated = np.zeros(n+1)
    updated[:-1] += g[1]*c*(n-np.arange(n))/n
    updated[1:] += g[0]*c*(np.arange(n)+1)/n
    updated /= updated.max()
    require(np.isfinite(updated).all() and updated.sum() > 0, "混合积分数值失效")
    return updated


def integral_budget(coefficients):
    c = np.asarray(coefficients, float)
    # 积分分母中各基函数具有相同积分；乘混合比例后的积分只差此项。
    budget = float(np.dot(c, (np.arange(len(c))+1)/(len(c)+1))/c.sum())
    require(np.isfinite(budget) and 0 <= budget <= 1, "混合预算超出无杠杆范围")
    return budget


def budget_frame(dates, reference_returns, expert_states, first):
    dates = pd.DatetimeIndex(dates)
    returns, states = np.asarray(reference_returns, float), np.asarray(expert_states, float)
    require(dates.is_monotonic_increasing and not dates.has_duplicates and returns.shape == states.shape == (len(dates), 2), "连续混合参考日历或形状错误")
    require(1 <= first < len(dates)-1 and (np.isnan(states) | np.isin(states, [0., 1.])).all(), "参考起点或持有状态错误")
    c, broken, rows = np.ones(1), False, []
    count = 0
    for t in range(len(dates)):
        budget, target = float("nan"), float("nan")
        status = "NO_VIEW_OUTSIDE_DECISION_PERIOD"
        if first-1 <= t < len(dates)-1:
            if t >= first:
                if not np.isfinite(returns[t]).all() or (returns[t] <= -1).any():
                    broken = True
                if not broken:
                    c = advance(c, 1+returns[t])
                    count += 1
            if broken:
                status = "NO_VIEW_INCOMPLETE_CUMULATIVE_HISTORY"
            else:
                budget = integral_budget(c)
                status = "INITIAL_UNIFORM_MIXTURE" if t == first-1 else "UNIVERSAL_MIXTURE_BUDGET_AVAILABLE"
                if np.isfinite(states[t]).all():
                    target = float(budget*states[t, 0]+(1-budget)*states[t, 1])
        rows.append({"date": dates[t], "panic_budget": budget, "learned_budget": 1-budget, "budget_status": status,
            "complete_reference_days": count, "panic_state": states[t, 0], "learned_state": states[t, 1], "target": target,
            "panic_reference_return": returns[t, 0], "learned_reference_return": returns[t, 1]})
    return pd.DataFrame(rows)
