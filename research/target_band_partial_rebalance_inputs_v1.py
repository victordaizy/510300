"""来源目标不变，仅在已有持仓偏离较大时部分调整。"""
import numpy as np

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

PARENT = 'ACCOUNT_VOLATILITY_EXPOSURE'
MODELS = [PARENT]
PRIMARY = 'TARGET_BAND_EDGE'
CANDIDATES = {'TARGET_BAND_EDGE': '偏离后调到允许区间边缘', 'TARGET_HALF_GAP': '偏离后只调整一半差额'}


def partial_request(account, price, target, cfg, model):
    require(model in CANDIDATES and cfg['weight_band'] == .1, '部分调仓规则或偏离带不同')
    require(np.isnan(target) or np.isfinite(target) and 0 <= target <= 1, '原目标超出范围')
    nav = account.value(price)
    actual = account.shares*price/nav
    desired, applied = account.shares, np.nan
    if np.isnan(target):
        action = '目标未知，保持实际份额'
    elif target == 0:
        desired, applied, action = 0, 0., '原目标归零，全部退出'
    elif account.shares == 0:
        applied, action = target, '按原完整正目标首次进入'
        desired = int(np.floor(applied*nav/price/cfg['lot'])*cfg['lot'])
    elif abs(target-actual) < cfg['weight_band']:
        applied, action = actual, '仍在允许偏离带内，保持份额'
    else:
        if model == 'TARGET_BAND_EDGE':
            applied = target-cfg['weight_band'] if actual < target else target+cfg['weight_band']
            action = '偏离较大，仅调到允许区间边缘'
        else:
            applied, action = (actual+target)/2, '偏离较大，仅调整一半差额'
        applied = float(np.clip(applied, 0., 1.))
        desired = int(np.floor(applied*nav/price/cfg['lot'])*cfg['lot'])
    return {'requested_quantity': desired-account.shares, 'reference_weight': target,
        'application_weight': applied, 'own_close_equity': nav, 'own_close_exposure': actual,
        'own_close_shares': float(account.shares), 'action': action}


def partial_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES), '本批候选集合不同')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    rows = []
    for cost_id, frame in frames.items():
        source = frame[PARENT+'_parent_target'].to_numpy(float)
        for model in CANDIDATES:
            frame[model+'_target'] = source
            values = source[first-1:-1]
            rows.append({'model': model, 'cost': cost_id, 'decision_origins': len(values),
                'positive_target_origins': int((values > 0).sum()), 'zero_target_origins': int((values == 0).sum()),
                'unknown_target_origins': int(np.isnan(values).sum()), 'mean_target': float(np.nanmean(values))})
    return frames, rows
