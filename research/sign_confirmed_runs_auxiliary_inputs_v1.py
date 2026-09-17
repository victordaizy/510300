"""核心目标持续保留，以保存的上涨方向决定辅助目标是否加入。"""
import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'SIGN_CONFIRMED_RUNS_AUXILIARY'
MODELS = ['TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE', 'RETURN_SIGN_BALANCE']
CANDIDATES = {PRIMARY: '核心加上涨方向确认的辅助仓位'}


def sign_confirmed_runs_auxiliary_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['parent_models'] == MODELS
        and cfg['combination'] == 'CORE_PLUS_SIGN_ALLOWED_AUXILIARY_CAPPED_AT_ONE'
        and cfg['direction_column'] == 'positive_direction', '方向确认辅助规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    origins = np.arange(first-1, len(data)-1)
    summaries = []
    for cost, frame in frames.items():
        direction = np.full(len(data), np.nan)
        values = parents_by_cost[cost][MODELS[2]][cfg['direction_column']].to_numpy(float)
        require((np.isnan(values) | (values == 0.) | (values == 1.)).all(), '保存方向不是允许、不允许或未知')
        direction[origins] = values
        core = frame[MODELS[0]+'_parent_target'].to_numpy(float)
        auxiliary = frame[MODELS[1]+'_parent_target'].to_numpy(float)
        known = np.isfinite(core) & np.isfinite(auxiliary) & np.isfinite(direction)
        enabled = known & (direction == 1.)
        effective = np.full(len(data), np.nan)
        effective[known] = 0.
        effective[enabled] = auxiliary[enabled]
        target = np.full(len(data), np.nan)
        target[known] = np.minimum(1., core[known]+effective[known])
        frame['auxiliary_direction'] = direction
        frame['effective_auxiliary_target'] = effective
        frame[PRIMARY+'_target'] = target
        current, a, b, d = target[origins], core[origins], auxiliary[origins], direction[origins]
        available = known[origins]
        summaries.append({'model': PRIMARY, 'cost': cost, 'decision_origins': len(origins),
            'positive_target_origins': int((current > 0).sum()), 'zero_target_origins': int((current == 0).sum()),
            'unknown_target_origins': int(np.isnan(current).sum()),
            'direction_allowed_origins': int((d == 1.).sum()), 'direction_disallowed_origins': int((d == 0.).sum()),
            'auxiliary_admitted_positive_origins': int((available & (d == 1.) & (b > 0)).sum()),
            'auxiliary_rejected_positive_origins': int((available & (d == 0.) & (b > 0)).sum()),
            'core_kept_while_direction_disallowed_origins': int((available & (d == 0.) & (a > 0)).sum()),
            'admitted_auxiliary_only_positive_origins': int((available & (d == 1.) & (a == 0.) & (b > 0.)).sum()),
            'both_contributing_positive_origins': int((available & (d == 1.) & (a > 0.) & (b > 0.)).sum()),
            'full_weight_origins': int((current == 1.).sum()),
            'mean_target': float(np.nanmean(current)) if available.any() else None})
    return frames, summaries
