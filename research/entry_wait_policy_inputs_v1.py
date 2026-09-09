"""在已成熟机会中评价买入或等待的完整现金结果，进行有限次策略改进。"""
from __future__ import annotations
from bisect import bisect_right
import hashlib
import numpy as np
import pandas as pd
from research.entry_payoff_gate_inputs_v1 import FEATURES as MARKET_FEATURES, CN as MARKET_CN
from research.intraday_overnight_increment_v1 import require

FEATURES = [*MARKET_FEATURES, "signal_age_log"]
CN = [*MARKET_CN, "连续已观测进入信号天数的对数"]
FILLED = "FILLED_BUY_NATURAL_EXIT_TERMINAL_PAYOFF"
UNFILLED = "UNFILLED_BUY_ZERO_STEP_REWARD_THEN_SAME_CASH_STATE"


def policy_hash(policy):
    return hashlib.sha256(np.asarray(policy, dtype=np.uint8).tobytes()).hexdigest()


def select_training_rows(links, fit_date, recent_groups):
    require(links.path_id.is_unique, "等待动作原点重复")
    eligible = links[links.original_group_mature_date.notna() & links.original_group_mature_date.le(pd.Timestamp(fit_date))]
    ordered = eligible[["episode_id", "original_group_mature_date"]].drop_duplicates().sort_values(["original_group_mature_date", "episode_id"])
    groups = ordered.tail(recent_groups).episode_id.to_list()
    rows = eligible[eligible.episode_id.isin(groups)].copy().sort_values(["episode_id", "path_id"]).reset_index(drop=True)
    rows["signal_age_log"] = np.log1p(rows.signal_age_observed_days)
    rows["sample_weight"] = 1/rows.groupby("episode_id").path_id.transform("count") if len(rows) else pd.Series(dtype=float)
    return rows, groups


def action_arrays(rows):
    require(np.isfinite(rows[FEATURES]).all().all(), "动作价值因子缺失")
    require(rows.buy_outcome_kind.isin([FILLED, UNFILLED]).all(), "训练组包含未解决的买入动作")
    positions = {int(value): i for i, value in enumerate(rows.path_id)}
    successors = np.full(len(rows), -1, dtype=int)
    for i, row in enumerate(rows.itertuples()):
        if pd.notna(row.next_path_id):
            require(int(row.next_path_id) in positions, "下一状态未随完整组进入训练")
            j = positions[int(row.next_path_id)]
            require(j > i and int(row.next_path_id) == row.path_id+1 and rows.episode_id.iloc[j] == row.episode_id, "等待连接跨组、跳日或倒序")
            successors[i] = j
        else:
            require(row.wait_outcome_kind == "WAIT_EPISODE_END_CASH", "未知等待边界不能填终止零价值")
    reward = rows.buy_terminal_or_immediate_return.to_numpy(float)
    require(np.isfinite(reward).all(), "成熟买入收益缺失")
    filled = rows.buy_outcome_kind.eq(FILLED).to_numpy(bool)
    require((reward[~filled] == 0).all(), "未成交的已知现金单步收益不为零")
    return reward, filled, successors


def evaluate_fixed_policy(policy, reward, filled, successors):
    """一次反向递推得到历史固定策略下的实现结果；不对未来实际收益取最大。"""
    n = len(reward)
    require(len(policy) == n and len(filled) == n and len(successors) == n, "策略评价输入长度不符")
    values = np.zeros(n)
    targets = np.zeros((n, 2))
    for i in range(n-1, -1, -1):
        j = int(successors[i])
        require(j == -1 or i < j < n, "反向评价不是有限向前连接")
        waiting = 0. if j < 0 else values[j]
        buying = reward[i] if filled[i] else reward[i]+waiting
        targets[i] = [buying, waiting]
        values[i] = buying if policy[i] else waiting
    require(np.isfinite(targets).all(), "固定策略的动作结果非有限")
    return targets, values


def decision_mask(buy, wait):
    return np.asarray(buy) > np.maximum(0., np.asarray(wait))


def transition_state(current, following, seen):
    if np.array_equal(current, following):
        return "TRAINING_POLICY_STABLE"
    if policy_hash(following) in seen:
        return "NO_VIEW_REPEATED_TRAINING_POLICY"
    return "CONTINUE_POLICY_IMPROVEMENT"


def fit_policy(rows, cfg):
    reward, filled, successors = action_arrays(rows)
    x, weights = rows[FEATURES].to_numpy(float), rows.sample_weight.to_numpy(float)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale[scale <= 1e-12] = 1
    design = np.c_[np.ones(len(x)), np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])]
    penalty = np.diag([0.] + [cfg["ridge_alpha"]]*len(FEATURES))
    gram = design.T@(weights[:, None]*design)+penalty
    # 同一月的矩阵只求解一次；后续只更新两个动作的目标，不重复分解输入。
    projector = np.linalg.solve(gram, design.T*weights)
    policy = np.zeros(len(rows), dtype=bool)
    seen = {policy_hash(policy)}
    records = []
    status, final_model = "NO_VIEW_POLICY_ITERATION_LIMIT", None
    for iteration in range(1, cfg["maximum_policy_iterations"]+1):
        target, realized = evaluate_fixed_policy(policy, reward, filled, successors)
        beta = projector@target
        prediction = design@beta
        require(np.isfinite(beta).all() and np.isfinite(prediction).all(), "动作价值回归数值失效")
        following = decision_mask(prediction[:, 0], prediction[:, 1])
        state = transition_state(policy, following, seen)
        record = {"iteration": iteration, "evaluation_policy_hash": policy_hash(policy), "improved_policy_hash": policy_hash(following),
            "evaluation_entries": int(policy.sum()), "improved_entries": int(following.sum()), "changed_states": int(np.count_nonzero(policy != following)),
            "transition_status": state, "buy_coefficients_with_intercept": beta[:, 0].tolist(), "wait_coefficients_with_intercept": beta[:, 1].tolist(),
            "maximum_absolute_evaluation_target": float(np.abs(target).max()), "maximum_absolute_fitted_value": float(np.abs(prediction).max())}
        records.append(record)
        if state == "TRAINING_POLICY_STABLE":
            status = "FIT_COMPLETE"
            final_model = {"mean": mean.tolist(), "scale": scale.tolist(), "clip": cfg["feature_clip"],
                "buy_coefficients_with_intercept": beta[:, 0].tolist(), "wait_coefficients_with_intercept": beta[:, 1].tolist(), "iterations": iteration}
            break
        if state == "NO_VIEW_REPEATED_TRAINING_POLICY":
            status = state
            break
        policy = following
        seen.add(policy_hash(policy))
    return {"status": status, "model": final_model, "iterations": records, "scalar_action_regressions": 2*len(records), "matrix_factorizations": 1}


def observed_signal_age(factors):
    age, run = [], 0
    for valid, raw in zip(factors.signal_available, factors.raw_entry):
        if not bool(valid):
            run = 0
            age.append(float("nan"))
        elif raw:
            run += 1
            age.append(float(run))
        else:
            run = 0
            age.append(0.)
    return np.array(age)


def action_views(data, factors, models):
    require(pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(factors.date)), "进入动作日历不符")
    ages = observed_signal_age(factors)
    fits = [row["fit_index"] for row in models]
    results = []
    for t in range(len(data)):
        idx = bisect_right(fits, t)-1
        stored = models[idx] if idx >= 0 else None
        x = np.r_[data[MARKET_FEATURES[:-1]].iloc[t].to_numpy(float), factors.d60_factor.iloc[t], np.log1p(ages[t])]
        buy, wait = float("nan"), float("nan")
        status = "NO_VIEW_NO_MATURE_ACTION_MODEL"
        if stored and stored["status"] == "FIT_COMPLETE":
            require(pd.Timestamp(stored["latest_mature_group_date"]) <= data.date.iloc[stored["fit_index"]] <= data.date.iloc[t], "使用了未来动作模型")
            if np.isfinite(x).all():
                model = stored["model"]
                design = np.r_[1., np.clip((x-np.array(model["mean"]))/np.array(model["scale"]), -model["clip"], model["clip"])]
                buy = float(design@np.array(model["buy_coefficients_with_intercept"]))
                wait = float(design@np.array(model["wait_coefficients_with_intercept"]))
                status = "ACTION_VALUES_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_ACTION_CONTEXT"
        elif stored:
            status = stored["status"]
        results.append({"date": data.date.iloc[t], "entry_model_status": status, "entry_fit_origin": data.date.iloc[stored["fit_index"]] if stored else pd.NaT,
            "buy_value": buy, "wait_value": wait, "waiting_policy_margin": buy-max(0., wait) if np.isfinite(buy) and np.isfinite(wait) else float("nan"),
            "immediate_value_margin": buy, "signal_age_observed_days": ages[t], **dict(zip(FEATURES, x))})
    return pd.DataFrame(results)
