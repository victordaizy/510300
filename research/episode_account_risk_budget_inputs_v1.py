"""在每个连续正目标区间开始时固定已知账户风险倍率。"""
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

ROOT = Path(__file__).resolve().parents[1]
RISK_SOURCE = ROOT / 'reports/research/510300_account_volatility_exposure_v1'
PARENT = 'EITHER_CONFIRMED_RUNS_AUXILIARY'
MODELS = [PARENT]
PRIMARY = 'EPISODE_ACCOUNT_RISK_BUDGET'
CANDIDATES = {PRIMARY: '连续正目标区间固定账户风险倍率'}


def fixed_episode_budget(source, current_multiplier):
    source = np.asarray(source, float)
    current_multiplier = np.asarray(current_multiplier, float)
    require(source.shape == current_multiplier.shape and source.ndim == 1, '目标与倍率长度不同')
    require((np.isnan(source) | (np.isfinite(source) & (source >= 0) & (source <= 1))).all(), '来源目标越界')
    require((np.isnan(current_multiplier) | (np.isfinite(current_multiplier) & (current_multiplier > 0))).all(), '来源倍率无效')
    target = np.full(len(source), np.nan)
    frozen = np.full(len(source), np.nan)
    starts = np.zeros(len(source), bool)
    active, multiplier = False, np.nan
    for t, value in enumerate(source):
        if np.isnan(value):
            if active:
                frozen[t] = multiplier
            continue
        if value == 0:
            target[t] = 0.
            active, multiplier = False, np.nan
            continue
        if not active:
            active, multiplier = True, current_multiplier[t]
            starts[t] = True
        frozen[t] = multiplier
        if np.isfinite(multiplier):
            target[t] = min(1., value * multiplier)
    return target, frozen, starts


def episode_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['budget_clock'] == 'FIRST_POSITIVE_ORIGIN_UNTIL_KNOWN_ZERO', '预算区间定义改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    period = 'evaluation' if pd.Timestamp(start) == pd.Timestamp(cfg['evaluation_start']) else 'earlier_diagnostic'
    origins = np.arange(first-1, len(data)-1)
    rows = []
    for cost, frame in frames.items():
        saved = pd.read_parquet(RISK_SOURCE / period / cost / 'factors.parquet')
        require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(data.date)) and saved.source_cost.eq(cost).all(), '保存风险日期或费用错位')
        source = frame[PARENT+'_parent_target'].to_numpy(float)
        np.testing.assert_allclose(saved[PARENT+'_parent_target'], source, atol=0, rtol=0, equal_nan=True)
        current = saved.account_risk_multiplier.to_numpy(float)
        target, fixed, starts = fixed_episode_budget(source, current)
        frame['current_account_risk_multiplier'] = current
        frame['fixed_episode_multiplier'] = fixed
        frame['positive_episode_start'] = starts
        frame[PRIMARY+'_target'] = target
        v = target[origins]
        rows.append({'model': PRIMARY, 'cost': cost, 'decision_origins': len(origins),
                     'positive_target_origins': int((v>0).sum()), 'zero_target_origins': int((v==0).sum()),
                     'unknown_target_origins': int(np.isnan(v).sum()), 'full_target_origins': int((v==1).sum()),
                     'positive_signal_episodes': int(starts[origins].sum()), 'mean_target': float(np.nanmean(v))})
    return frames, rows
