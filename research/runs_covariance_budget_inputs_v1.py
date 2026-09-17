"""只用已知基础父账户收益，每月按完整样本风险更新两套策略预算。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'RUNS_COVARIANCE_BUDGET'
MODELS = ['TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE']
CANDIDATES = {PRIMARY: '两套父策略每月历史最小方差预算'}
WINDOW = 242


def covariance_budget(left, right, previous):
    """二百四十二日样本统计；无完整风险证据时保留已知预算。"""
    a, b = np.asarray(left, float), np.asarray(right, float)
    require(a.ndim == b.ndim == 1 and len(a) == len(b) and len(a) <= WINDOW,
        '风险窗口维度、配对或长度不同')
    require(np.isfinite(previous) and 0 <= previous <= 1, '上一预算非法')
    row = {'sample_count': len(a), 'mean_reference': np.nan, 'mean_runs': np.nan,
        'variance_reference': np.nan, 'variance_runs': np.nan, 'covariance': np.nan,
        'difference_variance': np.nan, 'old_budget_reference': float(previous),
        'budget_reference': float(previous), 'status': 'INSUFFICIENT_WINDOW'}
    if len(a) < WINDOW:
        return row
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        row['status'] = 'MISSING_RETURN'
        return row
    mean_a, mean_b = float(a.mean()), float(b.mean())
    ca, cb = a-mean_a, b-mean_b
    diff = a-b
    centered_diff = diff-diff.mean()
    va = 0. if np.ptp(a) == 0 else float(np.dot(ca, ca)/(WINDOW-1))
    vb = 0. if np.ptp(b) == 0 else float(np.dot(cb, cb)/(WINDOW-1))
    cov = float(np.dot(ca, cb)/(WINDOW-1))
    vd = 0. if np.ptp(diff) == 0 else float(np.dot(centered_diff, centered_diff)/(WINDOW-1))
    row.update(mean_reference=mean_a, mean_runs=mean_b, variance_reference=va,
        variance_runs=vb, covariance=cov, difference_variance=vd)
    if va <= 0 or vb <= 0 or vd <= 0:
        row['status'] = 'DEGENERATE_RISK'
        return row
    row.update(budget_reference=float(np.clip((vb-cov)/vd, 0., 1.)), status='UPDATED')
    return row


def aligned_base_returns(data, base_ledgers, first, cfg):
    """核对净收益与净资产，终点开盘清算只供核对而不进入收盘窗口。"""
    require(set(base_ledgers) == set(MODELS), '风险父账户集合不同')
    dates = pd.DatetimeIndex(data.date)
    values = {}
    for model, ledger in base_ledgers.items():
        require(ledger.source_model.eq(model).all() and ledger.source_cost.eq('BASE').all(),
            '风险估计必须使用对应身份的基础费用完整父账户')
        require(pd.DatetimeIndex(ledger.date).equals(dates[first:]), '风险父账户完整日历不匹配')
        require(ledger.mark_clock.iloc[:-1].eq('CLOSE').all() and
            ledger.mark_clock.iloc[-1] == 'OPEN_TERMINAL', '风险父账户收盘与终点时钟不同')
        equity = ledger.equity.to_numpy(float)
        daily = ledger.net_return.to_numpy(float)
        require(np.isfinite(equity).all() and (equity > 0).all(), '风险父账户净资产非法')
        require((np.isnan(daily) | (np.isfinite(daily) & (daily > -1))).all(), '风险父账户净收益非法')
        calculated = equity/np.r_[cfg['initial_capital'], equity[:-1]]-1.
        valid = np.isfinite(daily)
        np.testing.assert_allclose(daily[valid], calculated[valid], atol=1e-12, rtol=0)
        aligned = np.full(len(data), np.nan)
        aligned[first:-1] = daily[:-1]
        values[model] = aligned
    return values


def runs_covariance_budget_frames(data, parents_by_cost, base_ledgers, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['parent_models'] == MODELS and
        cfg['risk_window'] == WINDOW and cfg['initial_budget_reference'] == .5 and
        cfg['budget_update'] == 'FIRST_TRADING_DAY_MONTH_CLOSE' and cfg['risk_source_cost'] == 'BASE',
        '月度历史风险预算的固定设置改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    returns = aligned_base_returns(data, base_ledgers, first, cfg)
    dates = pd.DatetimeIndex(data.date)
    origins = np.arange(first-1, len(data)-1)
    budget = np.full(len(data), np.nan)
    status = np.full(len(data), 'OUTSIDE_DECISION', dtype=object)
    monthly = np.zeros(len(data), bool)
    events = []
    previous = .5
    for origin in origins:
        if origin == first-1:
            row = covariance_budget([], [], previous)
            row['status'] = 'INITIAL_HALF'
            events.append({'origin_index': int(origin), 'origin': dates[origin],
                'window_start': pd.NaT, 'window_end': pd.NaT, **row})
            status[origin] = 'INITIAL_HALF'
        elif dates[origin].to_period('M') != dates[origin-1].to_period('M'):
            monthly[origin] = True
            left = max(first, origin-WINDOW+1)
            row = covariance_budget(returns[MODELS[0]][left:origin+1], returns[MODELS[1]][left:origin+1], previous)
            events.append({'origin_index': int(origin), 'origin': dates[origin],
                'window_start': dates[left], 'window_end': dates[origin], **row})
            previous = row['budget_reference']
            status[origin] = row['status']
        else:
            status[origin] = 'MONTHLY_BUDGET_HELD'
        budget[origin] = previous
    summaries = []
    for cost, frame in frames.items():
        a = frame[MODELS[0]+'_parent_target'].to_numpy(float)
        b = frame[MODELS[1]+'_parent_target'].to_numpy(float)
        target = budget*a+(1.-budget)*b
        frame['budget_reference'] = budget
        frame['budget_runs'] = 1.-budget
        frame['budget_status'] = status
        frame['month_first_update_attempt'] = monthly
        frame['risk_source_cost'] = 'BASE'
        frame[PRIMARY+'_target'] = target
        current = target[origins]
        summaries.append({'model': PRIMARY, 'cost': cost, 'decision_origins': len(origins),
            'positive_target_origins': int((current > 0).sum()), 'zero_target_origins': int((current == 0).sum()),
            'unknown_target_origins': int(np.isnan(current).sum()), 'month_first_attempts': int(monthly.sum()),
            'valid_risk_updates': sum(row['status'] == 'UPDATED' for row in events),
            'mean_reference_budget': float(budget[origins].mean()),
            'reference_only_budget_origins': int((budget[origins] == 1).sum()),
            'runs_only_budget_origins': int((budget[origins] == 0).sum()),
            'mean_target': float(np.nanmean(current)) if np.isfinite(current).any() else None})
    return frames, summaries, events
