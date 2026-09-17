"""以同期价量变化相关和上涨动量确认进入，并分别确认两种退出条件。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'PRICE_VOLUME_COHERENCE'
CANDIDATES = {PRIMARY: '二十日价量相关与上涨动量确认及普通波动仓位'}
CORRELATION_COLUMNS = ['return_mean20', 'volume_change_mean20', 'return_centered_square_sum20',
    'volume_centered_square_sum20', 'centered_cross_sum20', 'price_volume_correlation20']
STATE_COLUMNS = ['entry_confirmation_count', 'correlation_exit_confirmation_count',
    'momentum_exit_confirmation_count', 'positive_direction']


def pearson_summary(returns, volume_changes):
    x, y = np.asarray(returns, float), np.asarray(volume_changes, float)
    require(x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) >= 2 and
        np.isfinite(x).all() and np.isfinite(y).all(), '价量相关需要两列等长完整有限序列')
    mx, my = float(x.mean()), float(y.mean())
    dx, dy = x-mx, y-my
    xx, yy, xy = float(dx@dx), float(dy@dy), float(dx@dy)
    correlation = np.nan
    if np.ptp(x) > 0. and np.ptp(y) > 0. and xx > 0. and yy > 0.:
        correlation = float(np.clip(xy/np.sqrt(xx*yy), -1., 1.))
    return dict(zip(CORRELATION_COLUMNS, [mx, my, xx, yy, xy, correlation]))


def confirmed_coherence_directions(correlations, momentums):
    correlations, momentums = np.asarray(correlations, float), np.asarray(momentums, float)
    require(correlations.ndim == momentums.ndim == 1 and len(correlations) == len(momentums) and
        not np.isinf(correlations).any() and not np.isinf(momentums).any(), '价量方向输入维度或有限性错误')
    require((np.isnan(correlations) | ((correlations >= -1.) & (correlations <= 1.))).all(), '相关系数超出正负一')
    entry = correlation_exit = momentum_exit = direction = 0
    rows = []
    for correlation, momentum in zip(correlations, momentums):
        if np.isnan(correlation) or np.isnan(momentum):
            entry = correlation_exit = momentum_exit = direction = 0
            rows.append([np.nan]*4)
            continue
        entry = entry+1 if correlation > .2 and momentum > 0. else 0
        correlation_exit = correlation_exit+1 if correlation < -.2 else 0
        momentum_exit = momentum_exit+1 if momentum <= 0. else 0
        if entry >= 2:
            direction = 1
        if correlation_exit >= 2 or momentum_exit >= 2:
            direction = 0
        rows.append([entry, correlation_exit, momentum_exit, direction])
    return pd.DataFrame(rows, columns=STATE_COLUMNS)


def price_volume_coherence_factors(data):
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 0 and not dates.hasnans and dates.is_monotonic_increasing and not dates.has_duplicates,
        '价量因素日期必须已知、唯一并递增')
    require(data.volume_unit.eq('share').all(), '本轮成交量单位必须为份')
    close, previous, dividend, volume, volatility = [data[column].to_numpy(float) for column in
        ['close', 'previous_close', 'dividend', 'volume', 'vol20']]
    for values in [close, previous, volume]:
        require((np.isnan(values) | (np.isfinite(values) & (values > 0.))).all(), '已知价格和成交量必须有限且为正')
    for values in [dividend, volatility]:
        require((np.isnan(values) | (np.isfinite(values) & (values >= 0.))).all(), '已知分红和波动必须有限且非负')
    both = np.isfinite(previous[1:]) & np.isfinite(close[:-1])
    require(np.array_equal(previous[1:][both], close[:-1][both]), '前收盘与上一市场日收盘不一致')
    returns = np.log((close+dividend)/previous)
    volume_changes = np.r_[np.nan, np.log(volume[1:]/volume[:-1])]
    require(not np.isinf(returns).any() and not np.isinf(volume_changes).any(), '价量变化不能为无穷')
    values = np.full((len(data), len(CORRELATION_COLUMNS)), np.nan)
    for t in range(19, len(data)):
        x, y = returns[t-19:t+1], volume_changes[t-19:t+1]
        if np.isfinite(x).all() and np.isfinite(y).all():
            summary = pearson_summary(x, y)
            values[t] = [summary[column] for column in CORRELATION_COLUMNS]
    factors = pd.DataFrame(values, columns=CORRELATION_COLUMNS)
    factors['momentum20'] = pd.Series(returns).rolling(20, min_periods=20).sum()
    states = confirmed_coherence_directions(factors.price_volume_correlation20, factors.momentum20)
    for column in states:
        factors[column] = states[column]
    factors.insert(0, 'date', dates)
    factors['total_daily_log'] = returns
    factors['volume_shares'] = volume
    factors['volume_daily_log_change'] = volume_changes
    factors['volatility20'] = volatility
    return factors


def price_volume_coherence_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['correlation_window'] == 20 and cfg['momentum_window'] == 20 and
        cfg['volatility_window'] == 20 and cfg['risk_target'] == .1 and cfg['entry_threshold'] == .2 and
        cfg['exit_threshold'] == -.2 and cfg['confirmation_closes'] == 2, '价量相关固定规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, [], cfg, start)
    factors = price_volume_coherence_factors(data)
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
            'correlation_exit_confirmation_origins': int(factors.correlation_exit_confirmation_count.iloc[origins].eq(2).sum()),
            'momentum_exit_confirmation_origins': int(factors.momentum_exit_confirmation_count.iloc[origins].eq(2).sum()),
            'risk_capped_origins': int(((current > 0.) & (current < 1.)).sum()), 'full_target_origins': int((current == 1.).sum())})
    return frames, summaries
