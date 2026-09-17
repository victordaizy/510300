"""收盘已知涨幅只限制新增买入，原目标与卖出请求保持。"""
from decimal import Decimal

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import target_request
from research.event_account_indexed_request_v1 import simulate_indexed_request_account
from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

MODELS = ['INTENT_MIX_BAND_00', 'INTENT_MIX_BAND_20']
SETTINGS, CANDIDATES = {}, {}
for band in [0, 20]:
    for cap in [0, 1]:
        model = f'BUY_STRENGTH_BAND{band:02d}_R{cap:02d}'
        SETTINGS[model] = {'parent': f'INTENT_MIX_BAND_{band:02d}', 'band': band/100, 'return_floor': cap/100}
        CANDIDATES[model] = f'{band}个百分点调仓、当日涨幅至少达到{cap}%才新增买入'
PRIMARY = 'BUY_STRENGTH_BAND00_R01'


def market_factors(data):
    close = data.close.to_numpy(float)
    previous = data.previous_close.to_numpy(float)
    dividend = data.dividend.to_numpy(float)
    known = np.isfinite(close) & (close > 0) & np.isfinite(previous) & (previous > 0) & np.isfinite(dividend) & (dividend >= 0)
    returns = np.full(len(data), np.nan)
    returns[known] = (close[known]+dividend[known])/previous[known]-1
    output = pd.DataFrame({'daily_total_simple': returns, 'market_inputs_known': known})
    for percent in [0, 1]:
        allowed = np.zeros(len(data), dtype=bool)
        multiplier = Decimal(1)+Decimal(percent)/Decimal(100)
        for i in np.flatnonzero(known):
            # 按原始十进制价格比较，避免恰好零或百分之一被二进制舍入误判。
            allowed[i] = Decimal(str(close[i]))+Decimal(str(dividend[i])) >= Decimal(str(previous[i]))*multiplier
        output[f'buy_allowed_r{percent:02d}'] = allowed
    return output


def gate_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['candidate_settings'] == SETTINGS, '买入过滤候选或阈值改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    market, summaries = market_factors(data), []
    for cost, frame in frames.items():
        for column in market:
            frame[column] = market[column].to_numpy()
        for model, setting in SETTINGS.items():
            values = frame[setting['parent']+'_parent_target'].to_numpy(float)
            frame[model+'_target'] = values
            raw = values[first-1:-1]
            allowed = market[f"buy_allowed_r{int(setting['return_floor']*100):02d}"].iloc[first-1:-1]
            summaries.append({'model': model, 'cost': cost, 'decision_origins': len(raw),
                'positive_source_origins': int((raw > 0).sum()), 'zero_source_origins': int((raw == 0).sum()),
                'unknown_source_origins': int(np.isnan(raw).sum()), 'buy_allowed_origins': int(allowed.sum()),
                'rebalance_band': setting['band'], 'return_floor': setting['return_floor']})
    return frames, summaries


def gate_request(account, price, raw, cfg, model, allowed, daily_return):
    require(model in SETTINGS and cfg['candidate_settings'] == SETTINGS, '买入过滤候选身份不同')
    setting = SETTINGS[model]
    normal = target_request(account, price, raw, {**cfg, 'weight_band': setting['band']})
    quantity = normal['requested_quantity']
    suppressed = quantity > 0 and not allowed
    result = dict(normal)
    if suppressed:
        result['requested_quantity'] = 0
        result['action'] = '原目标保留，本次涨幅或资料条件不允许新增买入'
    result.update(output_candidate=model, budget_source=setting['parent'], used_band=float(setting['band']),
        return_floor=float(setting['return_floor']), observed_daily_return=float(daily_return), buy_allowed=float(allowed),
        normal_requested_quantity=float(quantity), buy_suppressed=float(suppressed),
        own_close_equity=float(account.value(price)), own_close_shares=float(account.shares))
    return result


def simulate_gate_account(data, dividends, cfg, cost, start, model, targets=None, prediction=None, horizon=1, event_mask=None):
    require(model in SETTINGS and targets is not None and prediction is None, '本批只使用保存来源目标')
    require(event_mask is not None and np.asarray(event_mask).all(), '买入过滤必须逐收盘判断')
    market = market_factors(data)
    setting = SETTINGS[model]
    allowed = market[f"buy_allowed_r{int(setting['return_floor']*100):02d}"].to_numpy(bool)
    returns = market.daily_total_simple.to_numpy(float)

    def request(account, price, value, settings, model_id, origin_index):
        return gate_request(account, price, value, settings, model_id, bool(allowed[origin_index]), returns[origin_index])

    return simulate_indexed_request_account(data, dividends, cfg, cost, start, model,
        targets=targets, event_mask=event_mask, request_policy=request)
