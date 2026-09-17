"""同一批次仅改变最外层调仓门槛，分别保留两类退出语义。"""
import numpy as np

from research.adaptive_allocation_v1 import target_request
from research.event_account_indexed_request_v1 import simulate_indexed_request_account
from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.two_close_zero_exit_inputs_v1 import zero_streak

SOURCES = {'TWO_CLOSE_ZERO_EXIT': '510300_two_close_zero_exit_v1',
           'INTENT_MIX_80_15_05': '510300_three_source_order_intent_mix_v1'}
MODELS = list(SOURCES)
SETTINGS = {}
CANDIDATES = {}
for prefix, parent, confirmations, label in [
    ('ZERO_CONFIRM', MODELS[0], 2, '两次归零确认'),
    ('INTENT_MIX', MODELS[1], 1, '简单三来源合并')]:
    for percent in [0, 5, 15, 20]:
        model = f'{prefix}_BAND_{percent:02d}'
        SETTINGS[model] = {'parent': parent, 'band': percent/100, 'zero_confirmations': confirmations}
        CANDIDATES[model] = f'{label}、{percent}个百分点调仓门槛'
PRIMARY = 'INTENT_MIX_BAND_15'


def band_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['candidate_settings'] == SETTINGS, '批次候选或门槛改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    summaries = []
    for cost, frame in frames.items():
        for model, setting in SETTINGS.items():
            values = frame[setting['parent']+'_parent_target'].to_numpy(float)
            frame[model+'_target'] = values
            frame[model+'_zero_streak'] = zero_streak(values)
            current = values[first-1:-1]
            summaries.append({'model': model, 'cost': cost, 'decision_origins': len(current),
                'positive_target_origins': int((current > 0).sum()), 'zero_target_origins': int((current == 0).sum()),
                'unknown_target_origins': int(np.isnan(current).sum()), 'mean_target': float(np.nanmean(current)),
                'rebalance_band': setting['band'], 'zero_confirmations': setting['zero_confirmations']})
    return frames, summaries


def band_request(account, price, value, cfg, model, count):
    require(model in SETTINGS and cfg['candidate_settings'] == SETTINGS, '调仓候选身份或设置不同')
    require(np.isfinite(value) and 0 <= value <= 1, '调仓目标无效')
    setting = SETTINGS[model]
    waiting = value == 0 and count < setting['zero_confirmations']
    if waiting:
        result = {'requested_quantity': 0, 'reference_weight': 0., 'action': '来源第一次明确归零，保持自身份额等待确认'}
    else:
        result = target_request(account, price, value, {**cfg, 'weight_band': setting['band']})
    result.update(output_candidate=model, budget_source=setting['parent'], used_band=float(setting['band']),
        used_zero_count=float(count), required_zero_confirmations=float(setting['zero_confirmations']),
        waiting_zero_confirmation=float(waiting and account.shares > 0),
        own_close_shares=float(account.shares), own_close_equity=account.value(price))
    return result


def simulate_band_account(data, dividends, cfg, cost, start, model, targets=None, prediction=None, horizon=1, event_mask=None):
    require(model in SETTINGS and targets is not None and prediction is None, '本批只使用已保存来源目标')
    require(event_mask is not None and np.asarray(event_mask).all(), '调仓门槛需要逐收盘判断')
    counts = zero_streak(targets)

    def request(account, price, value, settings, model_id, origin_index):
        return band_request(account, price, value, settings, model_id, int(counts[origin_index]))

    return simulate_indexed_request_account(data, dividends, cfg, cost, start, model,
        targets=targets, event_mask=event_mask, request_policy=request)
