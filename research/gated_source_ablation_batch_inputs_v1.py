"""从收盘计划份额删去辅助来源，保留已有加仓准入及正常减仓退出。"""
from pathlib import Path
import numpy as np
import pandas as pd
from research.three_source_order_intent_mix_inputs_v1 import SOURCES, source_plan

from research.adaptive_allocation_v1 import target_request
from research.close_return_buy_gate_batch_inputs_v1 import market_factors as upper_market
from research.close_return_buy_strength_gate_batch_inputs_v1 import market_factors as lower_market
from research.event_account_indexed_request_v1 import simulate_indexed_request_account
from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

ROOT = Path(__file__).resolve().parents[1]
MODELS = list(SOURCES)
MIXES = {'NO_EPISODE': ([16/17, 0., 1/17], '去掉区间固定预算'),
         'NO_REBOUND': ([16/19, 3/19, 0.], '去掉反弹辅助'),
         'CORE_ONLY': ([1., 0., 0.], '只保留两次零确认核心')}
SETTINGS, CANDIDATES = {}, {}
for percent in [100, 105]:
    for mix, (weights, label) in MIXES.items():
        model = f'ABLATE_{percent}_{mix}'
        SETTINGS[model] = {'parent': mix, 'band': .2, 'direction': 'LOWER', 'return_threshold': .01,
            'exposure_multiplier': percent/100, 'source_weights': dict(zip(MODELS, weights))}
        CANDIDATES[model] = f'目标{percent/100:.2f}倍、{label}'
PRIMARY = 'ABLATE_105_NO_EPISODE'


def market_factors(data):
    upper, lower = upper_market(data), lower_market(data)
    output = upper[['daily_total_simple', 'market_inputs_known']].copy()
    for direction, source in [('upper', upper), ('lower', lower)]:
        for percent in [0, 1]:
            output[f'buy_allowed_{direction}_r{percent:02d}'] = source[f'buy_allowed_r{percent:02d}']
    return output


def allowance_column(setting):
    return f"buy_allowed_{setting['direction'].lower()}_r{int(setting['return_threshold']*100):02d}"


def gate_frames(data, parents_by_cost, cfg, start, ledger_loader=None):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['candidate_settings'] == SETTINGS
        and cfg['source_folders'] == SOURCES, '来源删减设置不同')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    indices = np.arange(first-1, len(data)-1)
    period = 'evaluation' if pd.Timestamp(start) == pd.Timestamp(cfg['evaluation_start']) else 'earlier_diagnostic'
    if ledger_loader is None:
        def ledger_loader(source_period, cost, model):
            return pd.read_parquet(ROOT/'reports/research'/SOURCES[model]/source_period/cost/f'{model}_ledger.parquet')
    market, summaries = market_factors(data), []
    for cost, frame in frames.items():
        for column in market:
            frame[column] = market[column].to_numpy()
        for parent in MODELS:
            plan = source_plan(data, parents_by_cost[cost][parent], ledger_loader(period, cost, parent), cfg, first)
            for column in plan:
                values = np.full(len(data), np.nan)
                values[indices] = plan[column]
                frame[parent+'_'+column] = values
        for model, setting in SETTINGS.items():
            target = np.zeros(len(data))
            for parent, weight in setting['source_weights'].items():
                if weight > 0:
                    target += weight*frame[parent+'_planned_weight'].to_numpy(float)
            target = np.minimum(1., target*setting['exposure_multiplier'])
            frame[model+'_target'] = target
            active = target[indices]
            summaries.append({'model': model, 'cost': cost, 'decision_origins': len(indices),
                'positive_source_origins': int((active > 0).sum()), 'zero_source_origins': int((active == 0).sum()),
                'unknown_source_origins': int(np.isnan(active).sum()), 'source_mix': setting['parent'],
                'exposure_multiplier': setting['exposure_multiplier']})
    return frames, summaries


def gate_request(account, price, raw, cfg, model, allowed, daily_return):
    require(model in SETTINGS and cfg['candidate_settings'] == SETTINGS, '仅加仓过滤身份不同')
    setting = SETTINGS[model]
    normal = target_request(account, price, raw, {**cfg, 'weight_band': setting['band']})
    quantity = normal['requested_quantity']
    suppressed = account.shares > 0 and quantity > 0 and not allowed
    result = dict(normal)
    if suppressed:
        result['requested_quantity'] = 0
        result['action'] = '已有持仓，本次收益或资料条件不允许加仓，保留原目标和份额'
    result.update(output_candidate=model, budget_source=setting['parent'], used_band=float(setting['band']),
        exposure_multiplier=float(setting['exposure_multiplier']), filter_direction=setting['direction'], return_threshold=float(setting['return_threshold']),
        observed_daily_return=float(daily_return), addition_allowed=float(allowed),
        normal_requested_quantity=float(quantity), addition_suppressed=float(suppressed),
        own_close_equity=float(account.value(price)), own_close_shares=float(account.shares))
    return result


def simulate_gate_account(data, dividends, cfg, cost, start, model, targets=None, prediction=None, horizon=1, event_mask=None):
    require(model in SETTINGS and targets is not None and prediction is None, '本批只使用原保存目标')
    require(event_mask is not None and np.asarray(event_mask).all(), '加仓条件必须逐收盘判断')
    market = market_factors(data)
    allowed = market[allowance_column(SETTINGS[model])].to_numpy(bool)
    returns = market.daily_total_simple.to_numpy(float)

    def request(account, price, value, settings, model_id, origin_index):
        return gate_request(account, price, value, settings, model_id, bool(allowed[origin_index]), returns[origin_index])

    return simulate_indexed_request_account(data, dividends, cfg, cost, start, model,
        targets=targets, event_mask=event_mask, request_policy=request)
