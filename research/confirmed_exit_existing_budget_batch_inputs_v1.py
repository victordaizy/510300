"""将十七套已保存预算接入同一个两次明确零退出规则。"""
import numpy as np

from research.event_account_indexed_request_v1 import simulate_indexed_request_account
from research.intraday_overnight_increment_v1 import require
from research.risk_window_clock_batch_inputs_v1 import SETTINGS as OLD_SETTINGS
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.two_close_zero_exit_inputs_v1 import PRIMARY as EXIT_POLICY, confirmation_request, zero_streak

SOURCES = {model: {'folder': '510300_risk_window_clock_batch_v1', **setting} for model, setting in OLD_SETTINGS.items()}
SOURCES.update({
    'EPISODE_ACCOUNT_RISK_BUDGET': {'folder': '510300_episode_account_risk_budget_v1', 'clock': 'EPISODE', 'window': 60, 'budget_percent': 10},
    'ACCOUNT_RISK_12': {'folder': '510300_finite_account_risk_calibration_v1', 'clock': 'DAILY', 'window': 60, 'budget_percent': 12},
    'ACCOUNT_RISK_15': {'folder': '510300_finite_account_risk_calibration_v1', 'clock': 'DAILY', 'window': 60, 'budget_percent': 15},
})
MODELS = list(SOURCES)
MAPPING = {'CONFIRMED_'+model: model for model in MODELS}
CANDIDATES = {new: f"{SOURCES[old]['window']}日、{SOURCES[old]['budget_percent']}%预算、"
    f"{'每日调整' if SOURCES[old]['clock']=='DAILY' else '区间固定'}、两次零确认" for new, old in MAPPING.items()}
PRIMARY = 'CONFIRMED_RISK_DAILY_120_10'


def check_identity(cfg):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['candidate_sources'] == SOURCES
        and cfg['candidate_mapping'] == MAPPING and cfg['zero_confirmations'] == 2, '本批来源、候选映射或确认次数不同')


def mapped_frames(data, parents_by_cost, cfg, start):
    check_identity(cfg)
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    summaries = []
    for cost, frame in frames.items():
        for model, parent in MAPPING.items():
            source = frame[parent+'_parent_target'].to_numpy(float)
            counts = zero_streak(source)
            frame[model+'_target'] = source
            frame[model+'_zero_streak'] = counts
            values = source[first-1:-1]
            summaries.append({'model': model, 'parent': parent, 'cost': cost, 'decision_origins': len(values),
                'positive_target_origins': int((values > 0).sum()), 'zero_target_origins': int((values == 0).sum()),
                'unknown_target_origins': int(np.isnan(values).sum()),
                'first_zero_origins': int((counts[first-1:-1] == 1).sum()), 'mean_target': float(np.nanmean(values))})
    return frames, summaries


def simulate_mapped_confirmation(data, dividends, cfg, cost, start, model, targets=None, prediction=None, horizon=1, event_mask=None):
    check_identity(cfg)
    require(model in MAPPING and targets is not None and prediction is None, '新组合没有明确预算目标')
    require(event_mask is not None and np.asarray(event_mask).all(), '新组合必须逐个收盘判断')
    counts = zero_streak(targets)

    def request(account, price, value, settings, output_model, origin_index):
        require(output_model == model, '实际输出候选身份改变')
        result = confirmation_request(account, price, value, settings, EXIT_POLICY, int(counts[origin_index]))
        result.update(output_candidate=output_model, budget_source=MAPPING[model], exit_policy=EXIT_POLICY)
        return result

    return simulate_indexed_request_account(data, dividends, cfg, cost, start, model,
        targets=targets, event_mask=event_mask, request_policy=request)
