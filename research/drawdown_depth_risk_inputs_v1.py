"""以一百二十日趋势准入，普通波动与六十日持续回撤共同限制仓位。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'DRAWDOWN_DEPTH_RISK'
CANDIDATES = {PRIMARY: '趋势准入及普通波动与持续回撤双上限'}


def drawdown_depth_factors(data):
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 0 and not dates.hasnans and dates.is_monotonic_increasing and not dates.has_duplicates,
        '持续回撤日期必须已知、唯一并递增')
    wealth, volatility = data.wealth.to_numpy(float), data.vol20.to_numpy(float)
    require((np.isnan(wealth) | (np.isfinite(wealth) & (wealth > 0))).all(), '持续回撤财富包含无效值')
    require((np.isnan(volatility) | (np.isfinite(volatility) & (volatility >= 0))).all(), '普通波动包含无效值')
    trend = wealth/pd.Series(wealth).rolling(120, min_periods=120).mean().to_numpy()-1.
    strength = np.full(len(data), np.nan)
    if len(data) >= 60:
        windows = np.lib.stride_tricks.sliding_window_view(wealth, 60)
        valid = np.isfinite(windows).all(axis=1)
        complete = windows[valid]
        peaks = np.maximum.accumulate(complete, axis=1)
        drawdowns = complete/peaks-1.
        strength[np.flatnonzero(valid)+59] = np.sqrt(np.mean(drawdowns**2, axis=1))
    ordinary_cap, drawdown_cap = np.full(len(data), np.nan), np.full(len(data), np.nan)
    valid_vol = np.isfinite(volatility) & (volatility > 0.)
    ordinary_cap[valid_vol] = np.minimum(1., .1/volatility[valid_vol])
    drawdown_cap[strength == 0.] = 1.
    positive = np.isfinite(strength) & (strength > 0.)
    drawdown_cap[positive] = np.minimum(1., .04/strength[positive])
    return pd.DataFrame({'date': dates, 'wealth': wealth, 'trend_deviation120': trend, 'volatility20': volatility,
        'drawdown_strength60': strength, 'ordinary_volatility_cap': ordinary_cap, 'drawdown_depth_cap': drawdown_cap})


def drawdown_depth_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['trend_window'] == 120 and
        cfg['volatility_window'] == 20 and cfg['drawdown_window'] == 60 and cfg['risk_target'] == .1 and
        cfg['drawdown_budget'] == .04, '趋势及双风险上限固定规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, [], cfg, start)
    factors = drawdown_depth_factors(data)
    origins = np.arange(first-1, len(data)-1)
    trend = factors.trend_deviation120.to_numpy()
    ordinary, drawdown = factors.ordinary_volatility_cap.to_numpy(), factors.drawdown_depth_cap.to_numpy()
    target = np.full(len(data), np.nan)
    nonpositive = origins[np.isfinite(trend[origins]) & (trend[origins] <= 0.)]
    target[nonpositive] = 0.
    positive = origins[(trend[origins] > 0.) & np.isfinite(ordinary[origins]) & np.isfinite(drawdown[origins])]
    target[positive] = np.minimum(ordinary[positive], drawdown[positive])
    summaries = []
    for cost, frame in frames.items():
        for column in factors.columns[1:]:
            frame[column] = factors[column].to_numpy()
        frame[PRIMARY+'_target'] = target.copy()
        current = target[origins]
        summaries.append({'model': PRIMARY, 'cost': cost, 'decision_origins': len(origins),
            'positive_target_origins': int((current > 0.).sum()), 'zero_target_origins': int((current == 0.).sum()),
            'unknown_target_origins': int(np.isnan(current).sum()),
            'drawdown_cap_binding_origins': int(((trend[origins] > 0.) & (drawdown[origins] < ordinary[origins])).sum()),
            'risk_capped_origins': int(((current > 0.) & (current < 1.)).sum()), 'full_target_origins': int((current == 1.).sum())})
    return frames, summaries
