"""两种事前固定的父策略进场机会合并，不另外缩减单边目标。"""
import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'RUNS_OPPORTUNITY_MAX'
MODELS = ['TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE']
CANDIDATES = {PRIMARY: '两父目标取较大值', 'RUNS_OPPORTUNITY_CAPPED_SUM': '两父目标相加至百分之一百'}


def runs_opportunity_union_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['parent_models'] == MODELS and
        cfg['combination'] == 'MAXIMUM_AND_CAPPED_SUM_OF_SAVED_PARENT_TARGETS', '两套进场机会合并规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    origins = np.arange(first-1, len(data)-1)
    summaries = []
    for cost, frame in frames.items():
        a = frame[MODELS[0]+'_parent_target'].to_numpy(float)
        b = frame[MODELS[1]+'_parent_target'].to_numpy(float)
        targets = {PRIMARY: np.maximum(a, b), 'RUNS_OPPORTUNITY_CAPPED_SUM': np.minimum(a+b, 1.)}
        known = np.isfinite(a[origins]) & np.isfinite(b[origins])
        for model, target in targets.items():
            frame[model+'_target'] = target
            current, left, right = target[origins], a[origins], b[origins]
            summaries.append({'model': model, 'cost': cost, 'decision_origins': len(origins),
                'positive_target_origins': int((current > 0).sum()), 'zero_target_origins': int((current == 0).sum()),
                'unknown_target_origins': int(np.isnan(current).sum()),
                'both_parent_positive_origins': int((known & (left > 0) & (right > 0)).sum()),
                'only_reference_positive_origins': int((known & (left > 0) & (right == 0)).sum()),
                'only_runs_positive_origins': int((known & (left == 0) & (right > 0)).sum()),
                'both_parent_zero_origins': int((known & (left == 0) & (right == 0)).sum()),
                'full_weight_origins': int((current == 1.).sum()),
                'mean_target': float(np.nanmean(current)) if known.any() else None})
    return frames, summaries
