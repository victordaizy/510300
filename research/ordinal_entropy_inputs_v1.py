"""按三日价格顺序分布的集中程度及上涨动量决定进出场。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'ORDINAL_ENTROPY'
CANDIDATES = {PRIMARY: '三日价格顺序分布与六十日上涨动量'}
PATTERNS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
COUNT_COLUMNS = ['pattern_' + ''.join(map(str, pattern)) + '_count60' for pattern in PATTERNS]
STATE_COLUMNS = ['entry_confirmation_count', 'entropy_exit_confirmation_count',
    'momentum_exit_confirmation_count', 'positive_direction']


def three_day_patterns(values):
    values = np.asarray(values, float)
    require(values.ndim == 1 and not np.isinf(values).any(), '价格顺序需要一维且没有无穷的序列')
    codes = np.full(len(values), -1, dtype=int)
    mapping = {pattern: code for code, pattern in enumerate(PATTERNS)}
    for t in range(2, len(values)):
        sample = values[t-2:t+1]
        if np.isfinite(sample).all():
            codes[t] = mapping[tuple(np.argsort(sample, kind='stable'))]
    return codes


def normalized_entropy(counts):
    counts = np.asarray(counts, float)
    require(counts.shape == (6,) and np.isfinite(counts).all() and (counts >= 0).all() and counts.sum() > 0,
        '排列熵需要六个非负完整计数且总数为正')
    probabilities = counts[counts > 0]/counts.sum()
    return float(np.clip(-(probabilities*np.log(probabilities)).sum()/np.log(6.), 0., 1.))


def ordinal_summary(values):
    values = np.asarray(values, float)
    require(values.ndim == 1 and len(values) >= 3 and np.isfinite(values).all(), '顺序分布需要至少三个完整值')
    counts = np.bincount(three_day_patterns(values)[2:], minlength=6)
    return counts, normalized_entropy(counts)


def confirmed_entropy_directions(entropies, momentums):
    entropies, momentums = np.asarray(entropies, float), np.asarray(momentums, float)
    require(entropies.ndim == momentums.ndim == 1 and len(entropies) == len(momentums) and
        not np.isinf(entropies).any() and not np.isinf(momentums).any(), '排列熵方向输入维度或有限性错误')
    require((np.isnan(entropies) | ((entropies >= 0.) & (entropies <= 1.))).all(), '标准化排列熵超出零到一')
    entry = entropy_exit = momentum_exit = direction = 0
    rows = []
    for entropy, momentum in zip(entropies, momentums):
        if np.isnan(entropy) or np.isnan(momentum):
            entry = entropy_exit = momentum_exit = direction = 0
            rows.append([np.nan]*4)
            continue
        entry = entry+1 if entropy < .9 and momentum > 0. else 0
        entropy_exit = entropy_exit+1 if entropy > .95 else 0
        momentum_exit = momentum_exit+1 if momentum <= 0. else 0
        if entry >= 2:
            direction = 1
        if entropy_exit >= 2 or momentum_exit >= 2:
            direction = 0
        rows.append([entry, entropy_exit, momentum_exit, direction])
    return pd.DataFrame(rows, columns=STATE_COLUMNS)


def ordinal_entropy_factors(data):
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 0 and not dates.hasnans and dates.is_monotonic_increasing and not dates.has_duplicates,
        '排列熵日期必须已知、唯一并递增')
    wealth, volatility = data.wealth.to_numpy(float), data.vol20.to_numpy(float)
    require((np.isnan(wealth) | (np.isfinite(wealth) & (wealth > 0.))).all(), '已知含分红财富必须有限且为正')
    require((np.isnan(volatility) | (np.isfinite(volatility) & (volatility >= 0.))).all(), '已知普通波动必须有限且非负')
    codes = three_day_patterns(wealth)
    counts = np.full((len(data), 6), np.nan)
    entropy, momentum = np.full(len(data), np.nan), np.full(len(data), np.nan)
    for t in range(59, len(data)):
        sample = codes[t-57:t+1]
        if (sample >= 0).all():
            counts[t] = np.bincount(sample, minlength=6)
            entropy[t] = normalized_entropy(counts[t])
        if t >= 60 and np.isfinite(wealth[t-60:t+1]).all():
            momentum[t] = np.log(wealth[t]/wealth[t-60])
    factors = pd.DataFrame(counts, columns=COUNT_COLUMNS)
    factors['entropy60'] = entropy
    factors['momentum60'] = momentum
    states = confirmed_entropy_directions(entropy, momentum)
    for column in states:
        factors[column] = states[column]
    factors.insert(0, 'date', dates)
    factors['wealth'] = wealth
    factors['three_day_pattern'] = np.where(codes >= 0, codes.astype(float), np.nan)
    factors['volatility20'] = volatility
    return factors


def ordinal_entropy_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['entropy_window'] == 60 and cfg['pattern_order'] == 3 and
        cfg['momentum_window'] == 60 and cfg['volatility_window'] == 20 and cfg['risk_target'] == .1 and
        cfg['entry_threshold'] == .9 and cfg['exit_threshold'] == .95 and cfg['confirmation_closes'] == 2,
        '价格顺序分布固定规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, [], cfg, start)
    factors = ordinal_entropy_factors(data)
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
            'entropy_exit_confirmation_origins': int(factors.entropy_exit_confirmation_count.iloc[origins].eq(2).sum()),
            'momentum_exit_confirmation_origins': int(factors.momentum_exit_confirmation_count.iloc[origins].eq(2).sum()),
            'risk_capped_origins': int(((current > 0.) & (current < 1.)).sum()), 'full_target_origins': int((current == 1.).sum())})
    return frames, summaries
