"""用固定的自适应后悔公式，在严格收盘时钟下更新不同起点的策略记录。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require

EXPERTS = ("panic", "learned", "cash")
NUMERICAL_TOLERANCE = 1e-12


def normalized_record_weights(regret, magnitude):
    """保留指数差的精度；先验对每条已经生效的记录相同。"""
    regret, magnitude = np.asarray(regret, float), np.asarray(magnitude, float)
    require(regret.shape == magnitude.shape and regret.size > 0, "累计比较数组形状不一致")
    require(np.isfinite(regret).all() and np.isfinite(magnitude).all(), "累计比较出现非有限数")
    require((magnitude >= 0).all() and (np.abs(regret) <= magnitude+NUMERICAL_TOLERANCE).all(), "累计绝对差小于累计差的绝对值")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        upper = np.maximum(regret+1., 0.)**2/(3.*(magnitude+1.))
        lower = np.maximum(regret-1., 0.)**2/(3.*(magnitude+1.))
        positive = upper > lower
        if not positive.any():
            return np.full_like(regret, 1./regret.size), True
        log_weight = np.full_like(regret, -np.inf)
        log_weight[positive] = upper[positive]+np.log(-np.expm1(lower[positive]-upper[positive]))-np.log(2.)
        require(np.isfinite(log_weight[positive]).all(), "权重对数出现异常")
        weight = np.exp(log_weight-np.max(log_weight))
        weight /= weight.sum()
    require(np.isfinite(weight).all() and (weight >= 0).all() and abs(weight.sum()-1.) <= NUMERICAL_TOLERANCE, "比较权重不守恒")
    return weight, False


def empty_row(date, returns, states):
    row = {"date": date, "panic_reference_return": returns[0], "learned_reference_return": returns[1],
           "panic_state": states[0], "learned_state": states[1], "virtual_loss": np.nan, "target": np.nan,
           "learning_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD", "birth_scheduled": False, "birth_added": False,
           "active_groups": 0, "updated_records": 0, "zero_weight_fallback": False, "record_start": -1, "record_stop": -1}
    row.update({f"{expert}_{field}": np.nan for expert in EXPERTS for field in ["loss", "regret", "budget"]})
    return row


def adaptive_regret_frame(dates, reference_returns, expert_states, first):
    dates = pd.DatetimeIndex(dates)
    returns, states = np.asarray(reference_returns, float), np.asarray(expert_states, float)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "参考日历必须严格递增")
    require(returns.shape == states.shape == (len(dates), 2), "连续收益、意向与日历的形状不符")
    require(isinstance(first, int) and 1 <= first < len(dates)-1, "连续参考起点无效")
    rows, born, snapshots = [], [], {key: [] for key in ["regret", "magnitude", "weight"]}
    regret, magnitude, weights = (np.empty((0, 3)) for _ in range(3))
    offset, failure = 0, None
    for t in range(len(dates)):
        row = empty_row(dates[t], returns[t], states[t])
        if t < first-1 or t == len(dates)-1:
            rows.append(row)
            continue
        scheduled = t == first-1 or dates[t].to_period("M") != dates[t-1].to_period("M")
        row.update(birth_scheduled=scheduled, active_groups=len(born))
        if failure is None and not np.isin(states[t], [0., 1.]).all():
            failure = "NO_VIEW_SOURCE_GAP_REMAINDER"
        if failure is None and t >= first and (not np.isfinite(returns[t]).all() or not ((returns[t] > -1.) & (returns[t] <= 1.)).all()):
            failure = "NO_VIEW_SOURCE_GAP_REMAINDER"
        if failure is not None:
            row["learning_status"] = failure
            rows.append(row)
            continue
        try:
            if t >= first:
                losses = (1.-np.r_[returns[t], 0.])/2.
                old_budget = weights.sum(axis=0)
                mixture = float(old_budget @ losses)
                daily_regret = mixture-losses
                regret += daily_regret
                magnitude += np.abs(daily_regret)
                row.update(virtual_loss=mixture, updated_records=regret.size)
                row.update({f"{expert}_loss": losses[j] for j, expert in enumerate(EXPERTS)})
                row.update({f"{expert}_regret": daily_regret[j] for j, expert in enumerate(EXPERTS)})
            if scheduled:
                regret = np.vstack([regret, np.zeros(3)])
                magnitude = np.vstack([magnitude, np.zeros(3)])
                born.append(t)
                row["birth_added"] = True
            weights, fallback = normalized_record_weights(regret, magnitude)
            budgets = weights.sum(axis=0)
            target = float(budgets[:2] @ states[t])
            require(np.isfinite(target) and -NUMERICAL_TOLERANCE <= target <= 1.+NUMERICAL_TOLERANCE, "股票目标超出完整资金")
        except (FloatingPointError, ValueError, AssertionError, RuntimeError):
            failure = "NO_VIEW_NUMERICAL_REMAINDER"
            row.update(learning_status=failure, active_groups=len(born))
            rows.append(row)
            continue
        row.update(learning_status="ADAPTIVE_REGRET_AVAILABLE", active_groups=len(born), target=target,
                   zero_weight_fallback=fallback, record_start=offset, record_stop=offset+weights.size)
        row.update({f"{expert}_budget": budgets[j] for j, expert in enumerate(EXPERTS)})
        for key, array in [("regret", regret), ("magnitude", magnitude), ("weight", weights)]:
            snapshots[key].append(array.ravel().copy())
        offset += weights.size
        rows.append(row)
    records = {key: np.concatenate(values) if values else np.array([], float) for key, values in snapshots.items()}
    records["group_birth_index"] = np.asarray(born, dtype=np.int64)
    return pd.DataFrame(rows), records


def prefix_factors(full, source):
    require(len(source) <= len(full) and pd.DatetimeIndex(full.date.iloc[:len(source)]).equals(pd.DatetimeIndex(source.date)), "较早日历不是完整日期前缀")
    answer = full.iloc[:len(source)].copy()
    source_row = source.iloc[-1]
    answer.loc[answer.index[-1], :] = empty_row(source_row.date,
        [source_row.panic_reference_return, source_row.learned_reference_return], [source_row.panic_state, source_row.learned_state])
    return answer
