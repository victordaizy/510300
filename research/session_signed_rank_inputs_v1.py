"""以六十日日内相对隔夜差值的带符号排序确定进出场方向。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'SESSION_SIGNED_RANK'
CANDIDATES = {PRIMARY: '日内隔夜带符号排序及普通波动仓位'}


def signed_rank_summary(values):
    values = np.asarray(values, float)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), '带符号排序需要完整有限的一维差值')
    nonzero = values[values != 0.]
    if len(nonzero) == 0:
        return {'nonzero_count': 0, 'positive_rank_sum': 0., 'negative_rank_sum': 0., 'rank_square_sum': 0., 'rank_score60': 0.}
    _, inverse, counts = np.unique(abs(nonzero), return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    average_ranks = (last-counts+1+last)/2.
    ranks = average_ranks[inverse]
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    squared = float(np.square(ranks).sum())
    return {'nonzero_count': len(nonzero), 'positive_rank_sum': positive, 'negative_rank_sum': negative,
        'rank_square_sum': squared, 'rank_score60': (positive-negative)/np.sqrt(squared)}


def confirmed_rank_directions(scores):
    direction = positives = negatives = 0
    records = []
    for score in np.asarray(scores, float):
        require(not np.isinf(score), '排序分数不能为无穷')
        if np.isnan(score):
            direction = positives = negatives = 0
            records.append([np.nan, np.nan, np.nan])
            continue
        positives = positives+1 if score > 1.96 else 0
        negatives = negatives+1 if score < 0. else 0
        if positives >= 2:
            direction = 1
        if negatives >= 2:
            direction = 0
        records.append([positives, negatives, direction])
    return pd.DataFrame(records, columns=['positive_confirmation_count', 'negative_confirmation_count', 'positive_direction'])


def session_signed_rank_factors(data):
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 0 and not dates.hasnans and dates.is_monotonic_increasing and not dates.has_duplicates,
        '排序因素日期必须已知、唯一并递增')
    intraday = data.intraday_log.to_numpy(float)
    overnight = data.overnight_log.to_numpy(float)
    volatility = data.vol20.to_numpy(float)
    require(not np.isinf(intraday).any() and not np.isinf(overnight).any(), '日内隔夜来源包含无穷值')
    require((np.isnan(volatility) | (np.isfinite(volatility) & (volatility >= 0.))).all(), '普通波动包含无效值')
    difference = intraday-overnight
    columns = ['nonzero_count', 'positive_rank_sum', 'negative_rank_sum', 'rank_square_sum', 'rank_score60']
    values = np.full((len(data), len(columns)), np.nan)
    for t in range(59, len(data)):
        sample = difference[t-59:t+1]
        if np.isfinite(sample).all():
            summary = signed_rank_summary(sample)
            values[t] = [summary[name] for name in columns]
    factors = pd.DataFrame(values, columns=columns)
    directions = confirmed_rank_directions(factors.rank_score60)
    for column in directions:
        factors[column] = directions[column]
    factors.insert(0, 'date', dates)
    factors['intraday_log'] = intraday
    factors['overnight_log'] = overnight
    factors['session_difference'] = difference
    factors['volatility20'] = volatility
    return factors


def session_signed_rank_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['rank_window'] == 60 and cfg['volatility_window'] == 20 and
        cfg['risk_target'] == .1 and cfg['entry_threshold'] == 1.96 and cfg['exit_threshold'] == 0. and cfg['confirmation_closes'] == 2,
        '带符号排序及确认固定规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, [], cfg, start)
    factors = session_signed_rank_factors(data)
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
            'entry_confirmation_origins': int(factors.positive_confirmation_count.iloc[origins].eq(2).sum()),
            'exit_confirmation_origins': int(factors.negative_confirmation_count.iloc[origins].eq(2).sum()),
            'all_zero_difference_windows': int(factors.nonzero_count.iloc[origins].eq(0).sum()),
            'risk_capped_origins': int(((current > 0.) & (current < 1.)).sum()), 'full_target_origins': int((current == 1.).sum())})
    return frames, summaries
