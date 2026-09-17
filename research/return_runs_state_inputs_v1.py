"""统计收益相对窗口中位数的强弱连续段，确认实际进出场方向。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'RETURN_RUNS_STATE'
CANDIDATES = {PRIMARY: '收益强弱连续段与六十日上涨动量'}
RUN_COLUMNS = ['median_return60', 'strong_count60', 'weak_count60', 'non_tie_count60', 'tie_count60',
    'actual_runs60', 'expected_runs60', 'run_variance60', 'run_score60']
STATE_COLUMNS = ['entry_confirmation_count', 'runs_exit_confirmation_count',
    'momentum_exit_confirmation_count', 'positive_direction']


def run_count_summary(values):
    values = np.asarray(values, float)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), '连续段需要完整有限的一维收益窗口')
    median = float(np.median(values))
    signs = np.sign(values-median)
    non_tie = signs[signs != 0.]
    strong, weak, n = int((non_tie > 0).sum()), int((non_tie < 0).sum()), len(non_tie)
    actual = int(1+np.count_nonzero(non_tie[1:] != non_tie[:-1])) if n else 0
    product = 2*strong*weak
    expected = 1.+product/n if n else np.nan
    variance = product*(product-n)/(n*n*(n-1)) if n > 1 else np.nan
    score = (actual-expected)/np.sqrt(variance) if strong > 0 and weak > 0 and variance > 0. else np.nan
    return dict(zip(RUN_COLUMNS, [median, strong, weak, n, len(values)-n, actual, expected, variance, score]))


def confirmed_runs_directions(scores, momentums):
    scores, momentums = np.asarray(scores, float), np.asarray(momentums, float)
    require(scores.ndim == momentums.ndim == 1 and len(scores) == len(momentums) and
        not np.isinf(scores).any() and not np.isinf(momentums).any(), '连续段方向输入维度或有限性错误')
    entry = runs_exit = momentum_exit = direction = 0
    rows = []
    for score, momentum in zip(scores, momentums):
        if np.isnan(score) or np.isnan(momentum):
            entry = runs_exit = momentum_exit = direction = 0
            rows.append([np.nan]*4)
            continue
        entry = entry+1 if score < -1. and momentum > 0. else 0
        runs_exit = runs_exit+1 if score >= 0. else 0
        momentum_exit = momentum_exit+1 if momentum <= 0. else 0
        if entry >= 2:
            direction = 1
        if runs_exit >= 2 or momentum_exit >= 2:
            direction = 0
        rows.append([entry, runs_exit, momentum_exit, direction])
    return pd.DataFrame(rows, columns=STATE_COLUMNS)


def return_runs_state_factors(data):
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 0 and not dates.hasnans and dates.is_monotonic_increasing and not dates.has_duplicates,
        '连续段因素日期必须已知、唯一并递增')
    returns, volatility = data.total_log.to_numpy(float), data.vol20.to_numpy(float)
    require(not np.isinf(returns).any(), '含分红对数收益不能为无穷')
    require((np.isnan(volatility) | (np.isfinite(volatility) & (volatility >= 0.))).all(), '已知普通波动必须有限且非负')
    values = np.full((len(data), len(RUN_COLUMNS)), np.nan)
    momentum = np.full(len(data), np.nan)
    for t in range(59, len(data)):
        sample = returns[t-59:t+1]
        if np.isfinite(sample).all():
            summary = run_count_summary(sample)
            values[t] = [summary[column] for column in RUN_COLUMNS]
            momentum[t] = sample.sum()
    factors = pd.DataFrame(values, columns=RUN_COLUMNS)
    factors['momentum60'] = momentum
    states = confirmed_runs_directions(factors.run_score60, momentum)
    for column in states:
        factors[column] = states[column]
    factors.insert(0, 'date', dates)
    factors['total_daily_log'] = returns
    factors['volatility20'] = volatility
    return factors


def return_runs_state_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['runs_window'] == 60 and cfg['momentum_window'] == 60 and
        cfg['volatility_window'] == 20 and cfg['risk_target'] == .1 and cfg['entry_threshold'] == -1. and
        cfg['exit_threshold'] == 0. and cfg['confirmation_closes'] == 2, '收益强弱连续段固定规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, [], cfg, start)
    factors = return_runs_state_factors(data)
    origins = np.arange(first-1, len(data)-1)
    direction, volatility = factors.positive_direction.to_numpy(), factors.volatility20.to_numpy()
    target = np.full(len(data), np.nan)
    target[origins[direction[origins] == 0.]] = 0.
    positive = origins[(direction[origins] == 1.) & np.isfinite(volatility[origins]) & (volatility[origins] > 0.)]
    target[positive] = np.minimum(1., .1/volatility[positive])
    summaries = []
    for cost, frame in frames.items():
        for column in factors.columns[1:]:
            frame[column] = factors[column].to_numpy()
        frame[PRIMARY+'_target'] = target.copy()
        current = target[origins]
        summaries.append({'model': PRIMARY, 'cost': cost, 'decision_origins': len(origins),
            'positive_target_origins': int((current > 0.).sum()), 'zero_target_origins': int((current == 0.).sum()),
            'unknown_target_origins': int(np.isnan(current).sum()),
            'entry_confirmation_origins': int(factors.entry_confirmation_count.iloc[origins].eq(2).sum()),
            'runs_exit_confirmation_origins': int(factors.runs_exit_confirmation_count.iloc[origins].eq(2).sum()),
            'momentum_exit_confirmation_origins': int(factors.momentum_exit_confirmation_count.iloc[origins].eq(2).sum()),
            'risk_capped_origins': int(((current > 0.) & (current < 1.)).sum()), 'full_target_origins': int((current == 1.).sum())})
    return frames, summaries
