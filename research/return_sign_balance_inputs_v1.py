"""统计六十日收益的上涨、下跌及持平天数，联合累计方向确认进出场。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'RETURN_SIGN_BALANCE'
CANDIDATES = {PRIMARY: '上涨日占优与六十日累计方向'}
SIGN_COLUMNS = ['up_days60', 'down_days60', 'flat_days60', 'sign_balance60']
STATE_COLUMNS = ['entry_confirmation_count', 'sign_exit_confirmation_count',
    'momentum_exit_confirmation_count', 'positive_direction']


def sign_balance_summary(values):
    values = np.asarray(values, float)
    require(values.ndim == 1 and len(values) == 60 and np.isfinite(values).all(), '上涨日优势需要完整六十日有限收益')
    up, down, flat = int((values > 0).sum()), int((values < 0).sum()), int((values == 0).sum())
    return dict(zip(SIGN_COLUMNS, [up, down, flat, (up-down)/60.]))


def confirmed_sign_directions(scores, momentums):
    scores, momentums = np.asarray(scores, float), np.asarray(momentums, float)
    require(scores.ndim == momentums.ndim == 1 and len(scores) == len(momentums) and
        not np.isinf(scores).any() and not np.isinf(momentums).any(), '上涨日优势方向输入维度或有限性错误')
    entry = sign_exit = momentum_exit = direction = 0
    rows = []
    for score, momentum in zip(scores, momentums):
        if np.isnan(score) or np.isnan(momentum):
            entry = sign_exit = momentum_exit = direction = 0
            rows.append([np.nan]*4)
            continue
        entry = entry+1 if score > 0. and momentum > 0. else 0
        sign_exit = sign_exit+1 if score <= 0. else 0
        momentum_exit = momentum_exit+1 if momentum <= 0. else 0
        if entry >= 2:
            direction = 1
        if sign_exit >= 2 or momentum_exit >= 2:
            direction = 0
        rows.append([entry, sign_exit, momentum_exit, direction])
    return pd.DataFrame(rows, columns=STATE_COLUMNS)


def return_sign_balance_factors(data):
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 0 and not dates.hasnans and dates.is_monotonic_increasing and not dates.has_duplicates,
        '上涨日优势因素日期必须已知、唯一并递增')
    returns, volatility = data.total_log.to_numpy(float), data.vol20.to_numpy(float)
    require(not np.isinf(returns).any(), '含分红对数收益不能为无穷')
    require((np.isnan(volatility) | (np.isfinite(volatility) & (volatility >= 0.))).all(), '已知普通波动必须有限且非负')
    values = np.full((len(data), len(SIGN_COLUMNS)), np.nan)
    momentum = np.full(len(data), np.nan)
    for t in range(59, len(data)):
        sample = returns[t-59:t+1]
        if np.isfinite(sample).all():
            summary = sign_balance_summary(sample)
            values[t] = [summary[column] for column in SIGN_COLUMNS]
            momentum[t] = sample.sum()
    factors = pd.DataFrame(values, columns=SIGN_COLUMNS)
    factors['momentum60'] = momentum
    states = confirmed_sign_directions(factors.sign_balance60, momentum)
    for column in states:
        factors[column] = states[column]
    factors.insert(0, 'date', dates)
    factors['total_daily_log'] = returns
    factors['volatility20'] = volatility
    return factors


def return_sign_balance_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['sign_window'] == 60 and cfg['momentum_window'] == 60 and
        cfg['volatility_window'] == 20 and cfg['risk_target'] == .1 and cfg['entry_threshold'] == 0. and
        cfg['exit_threshold'] == 0. and cfg['confirmation_closes'] == 2, '上涨日优势固定规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, [], cfg, start)
    factors = return_sign_balance_factors(data)
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
            'sign_exit_confirmation_origins': int(factors.sign_exit_confirmation_count.iloc[origins].eq(2).sum()),
            'momentum_exit_confirmation_origins': int(factors.momentum_exit_confirmation_count.iloc[origins].eq(2).sum()),
            'risk_capped_origins': int(((current > 0.) & (current < 1.)).sum()), 'full_target_origins': int((current == 1.).sum())})
    return frames, summaries
