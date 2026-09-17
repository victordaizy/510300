"""原正目标照常调整，第一次明确零保留份额，第二次明确零全退。"""
import numpy as np

from research.adaptive_allocation_v1 import target_request
from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

PARENT = 'ACCOUNT_VOLATILITY_EXPOSURE'
MODELS = [PARENT]
PRIMARY = 'TWO_CLOSE_ZERO_EXIT'
CANDIDATES = {PRIMARY: '连续两个收盘明确归零才退出'}


def zero_streak(targets):
    values = np.asarray(targets, float)
    require(values.ndim == 1 and (np.isnan(values) | (np.isfinite(values) & (values >= 0) & (values <= 1))).all(), '来源目标无效')
    output, count = np.zeros(len(values), dtype=int), 0
    for t, value in enumerate(values):
        count = count+1 if value == 0 else 0
        output[t] = count
    return output


def confirmation_request(account, price, target, cfg, model, count):
    require(model == PRIMARY and cfg['zero_confirmations'] == 2 and cfg['weight_band'] == .1, '明确零确认设置不同')
    require(np.isfinite(target), '本申请入口只处理已知目标')
    waiting = target == 0 and count == 1
    if waiting:
        result = {'requested_quantity': 0, 'reference_weight': 0., 'action': '原目标首次明确归零，保持份额等待下一收盘确认'}
    else:
        result = target_request(account, price, target, cfg)
        if target == 0:
            result['action'] = '原目标连续两次或以上明确归零，下一开盘全部退出'
    result.update(used_zero_count=float(count), waiting_zero_confirmation=float(waiting and account.shares > 0),
                  own_close_shares=float(account.shares), own_close_equity=account.value(price))
    return result


def confirmation_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['zero_confirmations'] == 2, '本轮候选或确认次数不同')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    rows = []
    for cost, frame in frames.items():
        source = frame[PARENT+'_parent_target'].to_numpy(float)
        frame[PRIMARY+'_target'] = source
        counts = zero_streak(source)
        frame['source_zero_streak'] = counts
        values = source[first-1:-1]
        rows.append({'model': PRIMARY, 'cost': cost, 'decision_origins': len(values),
            'positive_target_origins': int((values > 0).sum()), 'zero_target_origins': int((values == 0).sum()),
            'unknown_target_origins': int(np.isnan(values).sum()),
            'first_zero_origins': int((counts[first-1:-1] == 1).sum()), 'mean_target': float(np.nanmean(values))})
    return frames, rows
