"""一次形成三个事先固定的方向确认辅助目标，保留原核心目标。"""
import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'DUAL_CONFIRMED_RUNS_AUXILIARY'
MODELS = ['TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE', 'RETURN_LAG_STATE', 'RETURN_SIGN_BALANCE']
CANDIDATES = {PRIMARY: '两方向共同确认辅助', 'LAG_CONFIRMED_RUNS_AUXILIARY': '仅相邻相关方向确认辅助',
    'EITHER_CONFIRMED_RUNS_AUXILIARY': '任一方向确认辅助'}
FORMULAS = {PRIMARY: 'BOTH', 'LAG_CONFIRMED_RUNS_AUXILIARY': 'LAG', 'EITHER_CONFIRMED_RUNS_AUXILIARY': 'EITHER'}


def return_confirmation_auxiliary_batch_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['parent_models'] == MODELS
        and cfg['combination'] == 'FIXED_THREE_DIRECTION_GATES_CORE_PLUS_AUXILIARY_CAPPED_AT_ONE'
        and cfg['direction_formulas'] == FORMULAS, '三套方向确认辅助规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    origins = np.arange(first-1, len(data)-1)
    summaries = []
    for cost, frame in frames.items():
        directions = []
        for model, column in zip(MODELS[2:], ['lag_direction', 'sign_direction']):
            values = parents_by_cost[cost][model].positive_direction.to_numpy(float)
            require((np.isnan(values) | (values == 0.) | (values == 1.)).all(), '保存方向不是允许、不允许或未知')
            direction = np.full(len(data), np.nan)
            direction[origins] = values
            frame[column] = direction
            directions.append(direction)
        lag, sign = directions
        core = frame[MODELS[0]+'_parent_target'].to_numpy(float)
        auxiliary = frame[MODELS[1]+'_parent_target'].to_numpy(float)
        for model, formula in FORMULAS.items():
            known = np.isfinite(core) & np.isfinite(auxiliary) & np.isfinite(lag)
            if formula != 'LAG':
                known &= np.isfinite(sign)
            allowed = {'LAG': lag == 1., 'BOTH': (lag == 1.) & (sign == 1.),
                'EITHER': (lag == 1.) | (sign == 1.)}[formula]
            admitted = np.full(len(data), np.nan)
            effective = np.full(len(data), np.nan)
            target = np.full(len(data), np.nan)
            admitted[known] = allowed[known].astype(float)
            effective[known] = 0.
            effective[known & allowed] = auxiliary[known & allowed]
            target[known] = np.minimum(1., core[known]+effective[known])
            frame[model+'_admitted'] = admitted
            frame[model+'_effective_auxiliary_target'] = effective
            frame[model+'_target'] = target
            current, a, b, g = target[origins], core[origins], auxiliary[origins], admitted[origins]
            summaries.append({'model': model, 'cost': cost, 'decision_origins': len(origins),
                'positive_target_origins': int((current > 0).sum()), 'zero_target_origins': int((current == 0).sum()),
                'unknown_target_origins': int(np.isnan(current).sum()),
                'direction_allowed_origins': int((g == 1.).sum()), 'direction_disallowed_origins': int((g == 0.).sum()),
                'auxiliary_admitted_positive_origins': int(((g == 1.) & (b > 0)).sum()),
                'auxiliary_rejected_positive_origins': int(((g == 0.) & (b > 0)).sum()),
                'core_kept_while_direction_disallowed_origins': int(((g == 0.) & (a > 0)).sum()),
                'admitted_auxiliary_only_positive_origins': int(((g == 1.) & (a == 0.) & (b > 0.)).sum()),
                'both_contributing_positive_origins': int(((g == 1.) & (a > 0.) & (b > 0.)).sum()),
                'full_weight_origins': int((current == 1.).sum()),
                'mean_target': float(np.nanmean(current)) if np.isfinite(current).any() else None})
    return frames, summaries
