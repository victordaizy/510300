"""固定信号的两档资金预算；目标最高一，不读取下一时点价格或收益。"""
import numpy as np

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

PARENT = 'EITHER_CONFIRMED_RUNS_AUXILIARY'
MODELS = [PARENT]
PRIMARY = 'EXPOSURE_EXPANSION_150'
MULTIPLIERS = {PRIMARY: 1.5, 'EXPOSURE_EXPANSION_200': 2.0}
CANDIDATES = {PRIMARY: '原目标一倍半、最高满仓', 'EXPOSURE_EXPANSION_200': '原目标两倍、最高满仓'}


def expanded_target(values, multiplier):
    values = np.asarray(values, dtype=float)
    require(multiplier in (1.5, 2.0), '仓位倍率不属于两档事前设置')
    require((np.isnan(values) | (np.isfinite(values) & (values >= 0) & (values <= 1))).all(), '来源目标越界')
    return np.minimum(1.0, values * multiplier)


def expansion_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['multipliers'] == MULTIPLIERS, '两档预算设置发生改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    indices = np.arange(first - 1, len(data) - 1)
    summaries = []
    for cost, frame in frames.items():
        source = frame[PARENT + '_parent_target'].to_numpy(float)
        for model, multiplier in MULTIPLIERS.items():
            target = expanded_target(source, multiplier)
            frame[model + '_target'] = target
            v = target[indices]
            summaries.append({
                'model': model, 'cost': cost, 'decision_origins': len(indices),
                'positive_target_origins': int((v > 0).sum()), 'zero_target_origins': int((v == 0).sum()),
                'unknown_target_origins': int(np.isnan(v).sum()), 'full_target_origins': int((v == 1).sum()),
                'clipped_origins': int((source[indices] * multiplier > 1).sum()), 'mean_target': float(np.nanmean(v)),
            })
    return frames, summaries
