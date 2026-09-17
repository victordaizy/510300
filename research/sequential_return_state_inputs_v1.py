"""在完整市场日历累计双向收益证据，产生明确进出场及风险仓位。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'SEQUENTIAL_RETURN_STATE'
CANDIDATES = {PRIMARY: '双向累积收益证据与普通波动预算'}


def sequential_states(standardized):
    values = np.asarray(standardized, float)
    require(values.ndim == 1 and not np.isinf(values).any(), '标准化收益维度或数值无效')
    up = down = 0.
    direction = 0
    records = []
    for value in values:
        if not np.isfinite(value):
            up = down = 0.
            direction = 0
            records.append([np.nan]*6)
            continue
        up, down = max(0., up+value-.5), max(0., down-value-.5)
        upward, downward = up >= 5., down >= 5.
        require(not (upward and downward), '同一收盘出现相互矛盾的双方向触发')
        before_up, before_down = up, down
        trigger = 1 if upward else -1 if downward else 0
        if trigger:
            direction = 1 if upward else 0
            up = down = 0.
        records.append([before_up, before_down, up, down, trigger, direction])
    return pd.DataFrame(records, columns=['up_evidence_before_reset', 'down_evidence_before_reset',
        'up_evidence_after_reset', 'down_evidence_after_reset', 'direction_trigger', 'positive_direction'])


def sequential_return_factors(data):
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 0 and not dates.hasnans and dates.is_monotonic_increasing and not dates.has_duplicates,
        '双向证据日期必须已知、唯一并递增')
    daily = data.total_simple.to_numpy(float)
    volatility = data.vol20.to_numpy(float)
    require((np.isnan(daily) | (np.isfinite(daily) & (daily > -1.))).all(), '含分红收益包含无效值')
    require((np.isnan(volatility) | (np.isfinite(volatility) & (volatility >= 0.))).all(), '仓位波动包含无效值')
    lagged_scale = pd.Series(daily).rolling(20, min_periods=20).std(ddof=1).shift(1).to_numpy()
    standardized = np.full(len(data), np.nan)
    available = np.isfinite(daily) & np.isfinite(lagged_scale) & (lagged_scale > 1e-12)
    standardized[available] = daily[available]/lagged_scale[available]
    states = sequential_states(standardized)
    states.insert(0, 'date', dates)
    states['daily_total_return'] = daily
    states['previous_twenty_day_scale'] = lagged_scale
    states['standardized_daily_return'] = standardized
    states['current_volatility20'] = volatility
    return states


def sequential_return_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['volatility_window'] == 20 and
        cfg['risk_target'] == .1 and cfg['drift_allowance'] == .5 and cfg['alarm_threshold'] == 5.,
        '双向收益状态固定设置改变')
    frames, first = aligned_target_frames(data, parents_by_cost, [], cfg, start)
    factors = sequential_return_factors(data)
    indices = np.arange(first-1, len(data)-1)
    direction = factors.positive_direction.to_numpy()
    volatility = factors.current_volatility20.to_numpy()
    target = np.full(len(data), np.nan)
    target[indices[direction[indices] == 0]] = 0.
    positive = indices[(direction[indices] == 1) & np.isfinite(volatility[indices]) & (volatility[indices] > 0)]
    target[positive] = np.minimum(1., .1/volatility[positive])
    summaries = []
    for cost, frame in frames.items():
        for column in factors.columns[1:]:
            frame[column] = factors[column].to_numpy()
        frame[PRIMARY+'_target'] = target.copy()
        current = target[indices]
        summaries.append({'model': PRIMARY, 'cost': cost, 'decision_origins': len(indices),
            'positive_target_origins': int((current > 0).sum()), 'zero_target_origins': int((current == 0).sum()),
            'unknown_target_origins': int(np.isnan(current).sum()),
            'upward_alarm_origins': int(factors.direction_trigger.iloc[indices].eq(1).sum()),
            'downward_alarm_origins': int(factors.direction_trigger.iloc[indices].eq(-1).sum()),
            'risk_capped_origins': int(((current > 0) & (current < 1)).sum()), 'full_target_origins': int((current == 1).sum())})
    return frames, summaries
