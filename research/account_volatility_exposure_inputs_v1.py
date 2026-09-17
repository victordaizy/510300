"""用原完整账户截至收盘的六十日波动确定下一开盘股票预算。"""
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/research/510300_return_confirmation_auxiliary_batch_v1'
PARENT = 'EITHER_CONFIRMED_RUNS_AUXILIARY'
MODELS = [PARENT]
PRIMARY = 'ACCOUNT_VOLATILITY_EXPOSURE'
CANDIDATES = {PRIMARY: '按六十日完整账户波动分配股票预算'}


def risk_budget(source, aligned_returns, first, annual_days=242, window=60, risk_target=.10):
    source = np.asarray(source, float)
    returns = np.asarray(aligned_returns, float)
    require(len(source) == len(returns), '原目标与账户收益长度不同')
    require((np.isnan(source) | (np.isfinite(source) & (source >= 0) & (source <= 1))).all(), '来源目标超出允许范围')
    risk = np.full(len(source), np.nan)
    if len(source) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(returns, window)
        risk[window - 1:] = windows.std(axis=1, ddof=1) * np.sqrt(annual_days)
    multiplier = np.full(len(source), np.nan)
    multiplier[np.arange(len(source)) < first + window - 1] = 1.0
    positive_risk = np.isfinite(risk) & (risk > 0)
    multiplier[positive_risk] = risk_target / risk[positive_risk]
    multiplier[np.isfinite(risk) & (risk == 0)] = 1.0
    target = np.minimum(1.0, source * multiplier)
    target[source == 0] = 0.0
    return target, risk, multiplier


def account_volatility_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['account_risk_window'] == 60
            and cfg['account_risk_target'] == .10, '本轮窗口或风险预算改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    period = 'evaluation' if pd.Timestamp(start) == pd.Timestamp(cfg['evaluation_start']) else 'earlier_diagnostic'
    origins = np.arange(first - 1, len(data) - 1)
    summaries = []
    for cost, frame in frames.items():
        ledger = pd.read_parquet(SOURCE / period / cost / f'{PARENT}_ledger.parquet')
        require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[first:])), '原账户日期错位')
        realized = np.full(len(data), np.nan)
        realized[first:] = ledger.net_return.to_numpy(float)
        parent = frame[PARENT + '_parent_target'].to_numpy(float)
        target, risk, multiplier = risk_budget(parent, realized, first, cfg['annual_days'],
                                             cfg['account_risk_window'], cfg['account_risk_target'])
        frame['source_realized_net_return'] = realized
        frame['source_account_volatility60'] = risk
        frame['account_risk_multiplier'] = multiplier
        frame[PRIMARY + '_target'] = target
        v = target[origins]
        summaries.append({'model': PRIMARY, 'cost': cost, 'decision_origins': len(origins),
                          'positive_target_origins': int((v > 0).sum()), 'zero_target_origins': int((v == 0).sum()),
                          'unknown_target_origins': int(np.isnan(v).sum()), 'full_target_origins': int((v == 1).sum()),
                          'mean_target': float(np.nanmean(v))})
    return frames, summaries
