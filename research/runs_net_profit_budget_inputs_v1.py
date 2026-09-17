"""每月依据完整父账户平均净收益的正值分配预算。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.runs_covariance_budget_inputs_v1 import aligned_base_returns
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'RUNS_NET_PROFIT_BUDGET'
MODELS = ['TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE']
CANDIDATES = {PRIMARY: '两父平均净收益正值的月度预算'}
WINDOW = 242


def net_profit_budget(left, right, previous):
    a, b = np.asarray(left, float), np.asarray(right, float)
    require(a.ndim == b.ndim == 1 and len(a) == len(b) and len(a) <= WINDOW, '净收益窗口配对或长度不同')
    require(len(previous) == 2 and all(np.isfinite(v) and 0 <= v <= 1 for v in previous)
        and (sum(previous) == 0 or abs(sum(previous)-1.) < 1e-12), '上一净收益预算非法')
    row = {'sample_count': len(a), 'mean_reference': np.nan, 'mean_runs': np.nan,
        'positive_mean_reference': np.nan, 'positive_mean_runs': np.nan,
        'old_budget_reference': float(previous[0]), 'old_budget_runs': float(previous[1]),
        'budget_reference': float(previous[0]), 'budget_runs': float(previous[1]), 'status': 'INSUFFICIENT_WINDOW'}
    if len(a) < WINDOW:
        return row
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        row['status'] = 'MISSING_RETURN'
        return row
    ma, mb = float(a.mean()), float(b.mean())
    pa, pb = max(ma, 0.), max(mb, 0.)
    total = pa+pb
    row.update(mean_reference=ma, mean_runs=mb, positive_mean_reference=pa, positive_mean_runs=pb,
        budget_reference=pa/total if total > 0 else 0., budget_runs=pb/total if total > 0 else 0.,
        status='UPDATED_POSITIVE_MEAN' if total > 0 else 'CASH_NONPOSITIVE_MEANS')
    return row


def runs_net_profit_budget_frames(data, parents_by_cost, base_ledgers, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['parent_models'] == MODELS and
        cfg['profit_window'] == WINDOW and cfg['initial_budgets'] == [.5, .5] and
        cfg['budget_update'] == 'FIRST_TRADING_DAY_MONTH_CLOSE' and cfg['budget_source_cost'] == 'BASE',
        '完整净收益正值月度预算规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    returns = aligned_base_returns(data, base_ledgers, first, cfg)
    dates = pd.DatetimeIndex(data.date)
    origins = np.arange(first-1, len(data)-1)
    budgets = np.full((len(data), 2), np.nan)
    status = np.full(len(data), 'OUTSIDE_DECISION', dtype=object)
    monthly = np.zeros(len(data), bool)
    previous = (.5, .5)
    events = []
    for t in origins:
        if t == first-1:
            row = net_profit_budget([], [], previous)
            row['status'] = 'INITIAL_HALF'
            events.append({'origin_index': int(t), 'origin': dates[t], 'window_start': pd.NaT, 'window_end': pd.NaT, **row})
            status[t] = 'INITIAL_HALF'
        elif dates[t].to_period('M') != dates[t-1].to_period('M'):
            monthly[t] = True
            left = max(first, t-WINDOW+1)
            row = net_profit_budget(returns[MODELS[0]][left:t+1], returns[MODELS[1]][left:t+1], previous)
            previous = row['budget_reference'], row['budget_runs']
            events.append({'origin_index': int(t), 'origin': dates[t], 'window_start': dates[left], 'window_end': dates[t], **row})
            status[t] = row['status']
        else:
            status[t] = 'MONTHLY_BUDGET_HELD'
        budgets[t] = previous
    summaries = []
    for cost, frame in frames.items():
        a = frame[MODELS[0]+'_parent_target'].to_numpy(float)
        b = frame[MODELS[1]+'_parent_target'].to_numpy(float)
        target = budgets[:, 0]*a+budgets[:, 1]*b
        frame['budget_reference'], frame['budget_runs'] = budgets[:, 0], budgets[:, 1]
        frame['budget_status'], frame['month_first_update_attempt'] = status, monthly
        frame['budget_source_cost'] = 'BASE'
        frame[PRIMARY+'_target'] = target
        current = target[origins]
        summaries.append({'model': PRIMARY, 'cost': cost, 'decision_origins': len(origins),
            'positive_target_origins': int((current > 0).sum()), 'zero_target_origins': int((current == 0).sum()),
            'unknown_target_origins': int(np.isnan(current).sum()), 'month_first_attempts': int(monthly.sum()),
            'positive_mean_updates': sum(row['status'] == 'UPDATED_POSITIVE_MEAN' for row in events),
            'nonpositive_cash_updates': sum(row['status'] == 'CASH_NONPOSITIVE_MEANS' for row in events),
            'mean_reference_budget': float(budgets[origins, 0].mean()), 'mean_runs_budget': float(budgets[origins, 1].mean()),
            'cash_budget_origins': int((budgets[origins].sum(axis=1) == 0.).sum()),
            'mean_target': float(np.nanmean(current)) if np.isfinite(current).any() else None})
    return frames, summaries, events
