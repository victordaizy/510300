"""复用既有风险与区间预算函数，一次构建十四套未计算过的设置。"""
import numpy as np
import pandas as pd

from research.account_volatility_exposure_inputs_v1 import PARENT, SOURCE, risk_budget
from research.episode_account_risk_budget_inputs_v1 import fixed_episode_budget
from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

MODELS = [PARENT]
SKIP = {('DAILY', 60, 10), ('DAILY', 60, 12), ('DAILY', 60, 15), ('EPISODE', 60, 10)}
SETTINGS = {f'RISK_{clock}_{window}_{budget}': {'clock': clock, 'window': window, 'budget_percent': budget}
    for clock in ['DAILY', 'EPISODE'] for window in [30, 60, 120] for budget in [10, 12, 15]
    if (clock, window, budget) not in SKIP}
CANDIDATES = {model: f"{s['window']}日风险、{s['budget_percent']}%预算、{'每日调整' if s['clock']=='DAILY' else '区间固定'}"
              for model, s in SETTINGS.items()}
PRIMARY = 'RISK_EPISODE_60_12'


def paths(source, returns, first, cfg):
    require(cfg['candidate_settings'] == SETTINGS and cfg['candidate_models'] == list(CANDIDATES), '批次设置与登记不同')
    result = {}
    for model, setting in SETTINGS.items():
        target, risk, current = risk_budget(source, returns, first, cfg['annual_days'],
                                           setting['window'], setting['budget_percent']/100)
        fixed = current
        if setting['clock'] == 'EPISODE':
            target, fixed, _ = fixed_episode_budget(source, current)
        result[model] = (target, risk, current, fixed)
    return result


def batch_frames(data, parents_by_cost, cfg, start):
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    period = 'evaluation' if pd.Timestamp(start) == pd.Timestamp(cfg['evaluation_start']) else 'earlier_diagnostic'
    origins, summary = np.arange(first-1, len(data)-1), []
    for cost_id, frame in frames.items():
        ledger = pd.read_parquet(SOURCE/period/cost_id/f'{PARENT}_ledger.parquet')
        require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[first:])), '风险来源日期不同')
        returns = np.r_[np.full(first, np.nan), ledger.net_return.to_numpy(float)]
        frame['source_realized_net_return'] = returns
        for model, (target, risk, current, applied) in paths(frame[PARENT+'_parent_target'], returns, first, cfg).items():
            frame[model+'_target'] = target
            frame[model+'_risk'] = risk
            frame[model+'_current_multiplier'] = current
            frame[model+'_applied_multiplier'] = applied
            values = target[origins]
            summary.append({'model': model, 'cost': cost_id, 'decision_origins': len(origins),
                'positive_target_origins': int((values > 0).sum()), 'zero_target_origins': int((values == 0).sum()),
                'unknown_target_origins': int(np.isnan(values).sum()), 'full_target_origins': int((values == 1).sum()),
                'mean_target': float(np.nanmean(values))})
    return frames, summary
