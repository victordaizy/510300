"""固定来源的连续目标适配，最后真实收盘也产生下一开盘目标。"""
from bisect import bisect_right
import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.joint_downside_reference_pair_inputs_v1 import minimum_joint_downside
from research.model_support_reference_router_inputs_v1 import validate_support_records


def align_decisions(data,decisions):
    values=np.full(len(data),np.nan)
    indices=decisions.origin_index.to_numpy(int)
    require(pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(data.date.iloc[indices])), '连续目标原时点错位')
    values[indices]=decisions.reference_weight.to_numpy(float)
    return values


def align_returns(data,ledger):
    values=np.full(len(data),np.nan)
    indices=pd.DatetimeIndex(data.date).get_indexer(ledger.date)
    require((indices>=0).all() and ledger.mark_clock.eq('CLOSE').all(),'连续风险来源必须都是已完成收盘')
    values[indices]=ledger.net_return.to_numpy(float)
    return values


def planned_weights(data,decisions,ledger,cfg,first):
    indices=np.arange(first-1,len(data))
    require(len(decisions)==len(ledger)+1==len(indices),'连续计划份额长度不同')
    require(np.array_equal(decisions.origin_index,indices),'连续计划份额索引不同')
    require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[first:])), '连续来源账本日期不同')
    prior=np.r_[cfg['initial_capital'],ledger.equity].astype(float)
    shares=np.r_[0.,ledger.shares].astype(float)
    planned=shares+decisions.requested_quantity.to_numpy(float)
    require(np.isfinite(prior).all() and (prior>0).all() and np.isfinite(planned).all() and (planned>=0).all() and (planned%cfg['lot']==0).all(),'计划份额或净值不合法')
    weights=planned*data.close.iloc[indices].to_numpy(float)/prior
    require(np.isfinite(weights).all() and (weights>=-1e-12).all() and (weights<=1+1e-12).all(),'计划持仓超过无融资范围')
    weights=np.clip(weights,0.,1.)
    weights[decisions.reference_weight.isna()]=np.nan
    result=np.full(len(data),np.nan);result[indices]=weights
    return result


def ordinary_multiplier(data):
    vol=data.vol20.to_numpy(float)
    valid=np.isfinite(vol)&(vol>0)
    values=np.full(len(data),np.nan)
    values[valid]=np.minimum(1.,.1/vol[valid])
    return pd.Series(values).ffill().fillna(1.).to_numpy(float)


def direction_targets(factors,first):
    direction=factors.positive_direction.to_numpy(float)
    volatility=factors.volatility20.to_numpy(float)
    target=np.full(len(factors),np.nan)
    origins=np.arange(first-1,len(factors))
    target[origins[direction[origins]==0.]]=0.
    positive=origins[(direction[origins]==1.)&np.isfinite(volatility[origins])&(volatility[origins]>0.)]
    target[positive]=np.minimum(1.,.1/volatility[positive])
    return target


def support_choice(data,models,first):
    times=validate_support_records(data,models)
    choices=np.zeros(len(data),bool)
    for t in range(first-1,len(data)):
        k=bisect_right(times,pd.Timestamp(data.date.iloc[t])+pd.Timedelta(hours=15,minutes=5))-1
        choices[t]=k>=0 and models[k]['status']=='FIT_COMPLETE'
    return choices


def joint_downside_budgets(data,returns,first):
    dates=pd.DatetimeIndex(data.date)
    weights=np.full((len(data),2),np.nan)
    rows=[];weight=.5
    for t,date in enumerate(dates):
        updated=False
        if t<first-1:
            rows.append({'date':date,'downside_budget':np.nan,'continuous_budget':np.nan,'updated':False})
            continue
        if t>=first and date.to_period('M')!=dates[t-1].to_period('M'):
            count=min(242,t-first+1)
            sample=returns[t-count+1:t+1]
            if count==242 and np.isfinite(sample).all():
                weight=minimum_joint_downside(sample,weight)['downside_budget'];updated=True
        weights[t]=[weight,1.-weight]
        rows.append({'date':date,'downside_budget':weight,'continuous_budget':1.-weight,'updated':updated})
    return weights,pd.DataFrame(rows)


def continuous_minimum_variance_budget(dates, reference_returns, expert_states, first, window=242):
    dates = pd.DatetimeIndex(dates)
    returns = np.asarray(reference_returns, dtype=float)
    states = np.asarray(expert_states, dtype=float)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "风险预算日历必须严格递增")
    require(returns.shape == states.shape == (len(dates), 2), "两条参考策略的日历及形状不符")
    require(1 <= first < len(dates)-1 and isinstance(window, int) and window >= 2, "评价起点或风险窗口无效")
    require((np.isnan(returns) | (np.isfinite(returns) & (returns > -1))).all(), "参考收益存在无穷或净值失效")
    require((np.isnan(states) | np.isin(states, [0., 1.])).all(), "参考状态必须为零、一或缺失")
    rows = []
    weights = np.array([.5, .5])
    sd = np.full(2, np.nan)
    status, attempted, successful, start = "INITIAL_EQUAL_BUDGET_NO_RISK_ESTIMATE", pd.NaT, pd.NaT, pd.NaT
    count = 0
    covariance, difference_variance, raw_panic_budget = np.nan, np.nan, np.nan
    for t in range(len(dates)):
        scheduled = False
        if t < first-1:
            rows.append({"date": dates[t], "panic_budget": np.nan, "learned_budget": np.nan, "panic_sd": np.nan, "learned_sd": np.nan,
                "reference_covariance": np.nan, "difference_variance": np.nan, "raw_panic_budget": np.nan,
                "risk_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD", "risk_update_scheduled": False, "risk_attempt_origin": pd.NaT,
                "last_successful_risk_origin": pd.NaT, "risk_window_start": pd.NaT, "risk_window_observations": 0,
                "panic_state": states[t, 0], "learned_state": states[t, 1], "target": np.nan})
            continue
        if t >= first and dates[t].to_period("M") != dates[t-1].to_period("M"):
            scheduled, attempted = True, dates[t]
            count = min(window, t-first+1)
            start = dates[t-count+1]
            values = returns[t-count+1:t+1]
            sd = np.full(2, np.nan)
            covariance, difference_variance, raw_panic_budget = np.nan, np.nan, np.nan
            if count < window:
                status = "NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all():
                status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            else:
                sd = np.std(values, axis=0, ddof=1)
                if not np.isfinite(sd).all() or (sd <= 0).any():
                    status = "NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"
                else:
                    centered = values-values.mean(axis=0)
                    covariance = float((centered[:, 0]*centered[:, 1]).sum()/(window-1))
                    # 直接计算收益差的方差，避免两项近似相等方差相减造成负值。
                    difference_variance = float(np.var(values[:, 0]-values[:, 1], ddof=1))
                    if not np.isfinite(difference_variance) or difference_variance <= 0:
                        status = "NO_VIEW_DEGENERATE_DIFFERENCE_KEEP_BUDGET"
                    else:
                        raw_panic_budget = float((sd[1]**2-covariance)/difference_variance)
                        panic_weight = float(np.clip(raw_panic_budget, 0., 1.))
                        weights = np.array([panic_weight, 1.-panic_weight])
                        successful, status = dates[t], "MIN_VARIANCE_BUDGET_AVAILABLE"
        target = float(states[t] @ weights) if np.isfinite(states[t]).all() else np.nan
        require(np.isfinite(weights).all() and (weights >= 0).all() and np.isclose(weights.sum(), 1.), "风险预算超出完整无杠杆资金")
        rows.append({"date": dates[t], "panic_budget": weights[0], "learned_budget": weights[1], "panic_sd": sd[0], "learned_sd": sd[1],
            "reference_covariance": covariance, "difference_variance": difference_variance, "raw_panic_budget": raw_panic_budget,
            "risk_status": status, "risk_update_scheduled": scheduled, "risk_attempt_origin": attempted,
            "last_successful_risk_origin": successful, "risk_window_start": start, "risk_window_observations": count,
            "panic_state": states[t, 0], "learned_state": states[t, 1], "target": target})
    return pd.DataFrame(rows)
