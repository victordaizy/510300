"""将两套既定收盘目标各取一半，交由单独资金账户执行。"""
import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'RUNS_REFERENCE_BLEND'
MODELS = ['TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE']
CANDIDATES = {PRIMARY: '趋势波动参考与收益连续段各半组合'}


def runs_reference_blend_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['parent_models'] == MODELS and
        cfg['blend_weights'] == {model: .5 for model in MODELS}, '两套既定目标的等额组合规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    origins = np.arange(first-1, len(data)-1)
    summaries = []
    for cost, frame in frames.items():
        a = frame[MODELS[0]+'_parent_target'].to_numpy(float)
        b = frame[MODELS[1]+'_parent_target'].to_numpy(float)
        target = (a+b)/2.
        frame[PRIMARY+'_target'] = target
        current, left, right = target[origins], a[origins], b[origins]
        known = np.isfinite(left) & np.isfinite(right)
        summaries.append({'model': PRIMARY, 'cost': cost, 'decision_origins': len(origins),
            'positive_target_origins': int((current > 0).sum()), 'zero_target_origins': int((current == 0).sum()),
            'unknown_target_origins': int(np.isnan(current).sum()),
            'both_parent_positive_origins': int((known & (left > 0) & (right > 0)).sum()),
            'only_reference_positive_origins': int((known & (left > 0) & (right == 0)).sum()),
            'only_runs_positive_origins': int((known & (left == 0) & (right > 0)).sum()),
            'both_parent_zero_origins': int((known & (left == 0) & (right == 0)).sum()),
            'mean_target': float(np.nanmean(current)) if known.any() else None})
    return frames, summaries
