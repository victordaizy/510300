"""固定加仓条件，在100%以内小幅放大目标投入，并重新计算完整账户。"""
import numpy as np

from research.adaptive_allocation_v1 import target_request
from research.close_return_buy_gate_batch_inputs_v1 import market_factors as upper_market
from research.close_return_buy_strength_gate_batch_inputs_v1 import market_factors as lower_market
from research.event_account_indexed_request_v1 import simulate_indexed_request_account
from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

MODELS = ['INTENT_MIX_BAND_00', 'INTENT_MIX_BAND_20']
SETTINGS, CANDIDATES = {}, {}
for percent in [105, 110, 115, 120]:
    model = f'ADD_GATE_EXPOSURE_{percent}'
    SETTINGS[model] = {'parent': 'INTENT_MIX_BAND_20', 'band': .2, 'direction': 'LOWER',
        'return_threshold': .01, 'exposure_multiplier': percent/100}
    CANDIDATES[model] = f'固定1%加仓条件、原目标乘{percent/100:.2f}并限制100%'
PRIMARY = 'ADD_GATE_EXPOSURE_110'


def market_factors(data):
    upper, lower = upper_market(data), lower_market(data)
    output = upper[['daily_total_simple', 'market_inputs_known']].copy()
    for direction, source in [('upper', upper), ('lower', lower)]:
        for percent in [0, 1]:
            output[f'buy_allowed_{direction}_r{percent:02d}'] = source[f'buy_allowed_r{percent:02d}']
    return output


def allowance_column(setting):
    return f"buy_allowed_{setting['direction'].lower()}_r{int(setting['return_threshold']*100):02d}"


def gate_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['candidate_settings'] == SETTINGS, '仅加仓过滤候选或设置改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    market, summaries = market_factors(data), []
    for cost, frame in frames.items():
        for column in market:
            frame[column] = market[column].to_numpy()
        for model, setting in SETTINGS.items():
            values = np.minimum(1., frame[setting['parent']+'_parent_target'].to_numpy(float)*setting['exposure_multiplier'])
            frame[model+'_target'] = values
            raw, allowed = values[first-1:-1], market[allowance_column(setting)].iloc[first-1:-1]
            summaries.append({'model': model, 'cost': cost, 'decision_origins': len(raw),
                'positive_source_origins': int((raw > 0).sum()), 'zero_source_origins': int((raw == 0).sum()),
                'unknown_source_origins': int(np.isnan(raw).sum()), 'addition_allowed_origins': int(allowed.sum()),
                'rebalance_band': setting['band'], 'direction': setting['direction'], 'return_threshold': setting['return_threshold']})
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
