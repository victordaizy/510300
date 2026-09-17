"""只合并来源收盘已经提出的计划份额，保留原退出确认等待。"""
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'TWO_CLOSE_ZERO_EXIT': '510300_two_close_zero_exit_v1',
    'CONFIRMED_RISK_EPISODE_30_10': '510300_confirmed_exit_existing_budget_batch_v1',
    'MEAN_REBOUND_AUX_50': '510300_mean_rebound_auxiliary_v1',
}
MODELS = list(SOURCES)
PRIMARY = 'INTENT_MIX_80_15_05'
CANDIDATES = {PRIMARY: '八成、一成半、半成计划份额合并',
              'INTENT_MIX_DIAGNOSTIC_WEIGHTS': '按已保存筛选权重合并计划份额'}


def source_plan(data, decisions, ledger, cfg, first):
    """只用决定当时的份额、净值与申请，不读取次日成交反推目标。"""
    indices = np.arange(first-1, len(data)-1)
    dates = pd.DatetimeIndex(data.date)
    require(len(decisions) == len(ledger) == len(indices), '来源计划日历长度不同')
    require(np.array_equal(decisions.origin_index, indices), '来源计划索引不同')
    require(pd.DatetimeIndex(decisions.origin).equals(dates[indices]), '来源计划收盘日期不同')
    require(pd.DatetimeIndex(decisions.execution_date).equals(dates[indices+1]), '来源计划不是下一开盘')
    require(pd.DatetimeIndex(ledger.date).equals(dates[indices+1]), '来源账户日期不同')
    require(pd.DatetimeIndex(decisions.decision_time).equals(dates[indices]+pd.Timedelta(hours=15, minutes=5)), '来源决定时钟不同')
    prior = np.r_[cfg['initial_capital'], ledger.equity.iloc[:-1]].astype(float)
    shares = np.r_[0, ledger.shares.iloc[:-1]].astype(float)
    requests = decisions.requested_quantity.to_numpy(float)
    prices = data.close.iloc[indices].to_numpy(float)
    planned = shares+requests
    require((np.isfinite(prior) & (prior > 0) & np.isfinite(prices) & (prices > 0)).all(), '来源收盘净值或价格无效')
    require((np.isfinite(planned) & (planned >= 0) & (planned % cfg['lot'] == 0)).all(), '来源计划份额无效')
    weights = planned*prices/prior
    require((np.isfinite(weights) & (weights >= -1e-12) & (weights <= 1+1e-12)).all(), '来源计划比例越界')
    weights = np.clip(weights, 0., 1.)
    original = decisions.reference_weight.to_numpy(float)
    require((np.isnan(original) | (np.isfinite(original) & (original >= 0) & (original <= 1))).all(), '来源原目标无效')
    weights[np.isnan(original)] = np.nan
    return pd.DataFrame({'prior_equity': prior, 'prior_shares': shares,
        'source_requested_quantity': requests, 'planned_shares': planned, 'planned_weight': weights,
        'zero_signal_but_holding_plan': ((original == 0) & (planned > 0)).astype(float)})


def validate_weights(cfg):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['source_folders'] == SOURCES, '固定候选或来源身份改变')
    weights = cfg['source_weights']
    require(set(weights) == set(CANDIDATES), '固定权重候选不同')
    require(weights[PRIMARY] == dict(zip(MODELS, [.8, .15, .05])), '主候选固定权重改变')
    for mapping in weights.values():
        require(set(mapping) == set(MODELS), '固定权重来源不同')
        values = np.array([mapping[m] for m in MODELS], dtype=float)
        require((np.isfinite(values) & (values >= 0)).all() and abs(values.sum()-1) < 1e-12, '权重不是非负全额分配')
    return weights


def intent_frames(data, parents_by_cost, cfg, start, ledger_loader=None):
    weights = validate_weights(cfg)
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    period = 'evaluation' if pd.Timestamp(start) == pd.Timestamp(cfg['evaluation_start']) else 'earlier_diagnostic'
    indices = np.arange(first-1, len(data)-1)
    if ledger_loader is None:
        def ledger_loader(source_period, cost, model):
            return pd.read_parquet(ROOT/'reports/research'/SOURCES[model]/source_period/cost/f'{model}_ledger.parquet')
    summaries = []
    for cost, frame in frames.items():
        for model in MODELS:
            source = parents_by_cost[cost][model]
            plan = source_plan(data, source, ledger_loader(period, cost, model), cfg, first)
            for column in plan:
                values = np.full(len(data), np.nan)
                values[indices] = plan[column]
                frame[model+'_'+column] = values
        for candidate, mapping in weights.items():
            target = sum(mapping[model]*frame[model+'_planned_weight'].to_numpy(float) for model in MODELS)
            frame[candidate+'_target'] = target
            values = target[indices]
            summaries.append({'model': candidate, 'cost': cost, 'decision_origins': len(indices),
                'positive_target_origins': int((values > 0).sum()), 'zero_target_origins': int((values == 0).sum()),
                'unknown_target_origins': int(np.isnan(values).sum()), 'mean_target': float(np.nanmean(values)),
                'source_zero_but_holding_origins': int(frame.loc[indices, [m+'_zero_signal_but_holding_plan' for m in MODELS]].max(axis=1).eq(1).sum())})
    return frames, summaries
