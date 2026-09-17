"""保存账户的跨阶段失效和收益恒等式解释，不运行策略。"""
import argparse
import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import now, write_json, digest
from research.monthly_single_factor_walkforward_v1 import read, csv


from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/research/510300_single_factor_direct_stump_v1'
OUT = ROOT / 'reports/research/510300_direct_stump_failure_explanation_v1'
INDEX = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
STUDY = '510300_DIRECT_STUMP_FAILURE_EXPLANATION_V1'
PAIRS = [('CONFIRMED_STUMP', 'NO_FACTOR_LEARNER'), ('NO_FACTOR_LEARNER', 'FIXED_WEEKLY_VOL10'),
         ('CONFIRMED_STUMP', 'FIXED_WEEKLY_VOL10')]
MODELS = ['CONFIRMED_STUMP', 'NO_FACTOR_LEARNER', 'FIXED_WEEKLY_VOL10']


def decomposition(ledger, market, capital=200000., annual=242):
    previous_nav = np.r_[capital, ledger.equity.to_numpy(float)[:-1]]
    weight = ledger.shares_before.to_numpy(float) * market.previous_close.to_numpy(float) / previous_nav
    market_ret = market.total_simple.to_numpy(float)
    actual = ledger.net_return.to_numpy(float)
    friction = (ledger.commission.to_numpy(float) + ledger.slippage_cost.to_numpy(float)) / previous_nav
    before_cost_residual = actual - weight * market_ret + friction
    average_weight_part = np.full(len(ledger), weight.mean()) * market_ret
    timing_part = (weight - weight.mean()) * market_ret
    daily = pd.DataFrame({'date': ledger.date.to_numpy(), 'net_return': actual, 'prior_weight': weight,
                          'average_weight_part': average_weight_part, 'timing_part': timing_part,
                          'execution_rights_part': before_cost_residual, 'friction_part': -friction})
    np.testing.assert_allclose(daily[['average_weight_part', 'timing_part', 'execution_rights_part', 'friction_part']].sum(axis=1),
                               actual, atol=1e-12, rtol=0)
    components = {key: float(daily[key].mean() * annual) for key in
                  ['net_return', 'average_weight_part', 'timing_part', 'execution_rights_part', 'friction_part']}
    components['mean_prior_weight'] = float(weight.mean())
    return daily, components


def active_fit_indices(decisions, fit_origins):
    # 原日账每行对应前一天的一条判断；最后一条待执行判断留在账户之外。
    executed = decisions.iloc[:-1]
    scheduled = (executed.weekly_event | executed.initialization_event).to_numpy(bool)
    assert scheduled[0]
    last_scheduled_row = np.maximum.accumulate(np.where(scheduled, np.arange(len(executed)), 0))
    last_origin = executed.origin_index.to_numpy(int)[last_scheduled_row]
    fit_indices = np.searchsorted(np.asarray(fit_origins, int), last_origin, side='right') - 1
    assert (fit_indices >= 0).all()
    assert (np.asarray(fit_origins)[fit_indices] <= last_origin).all()
    return fit_indices


def freeze():
    assert not (OUT / 'protocol.json').exists()
    tests = read(OUT / 'tests_receipt.json')
    assert tests['passed'] == 2 and tests['failed'] == 0
    cfg = read(SOURCE / 'protocol.json')
    files = [Path(__file__), ROOT / 'docs/510300_DIRECT_STUMP_FAILURE_EXPLANATION_V1.md',
             ROOT / 'tests/test_direct_stump_failure_explanation_v1.py',
             SOURCE / 'protocol.json', SOURCE / 'learning_completed.json', SOURCE / 'result.json',
             SOURCE / 'saved_verification.json', ROOT / cfg['features']]
    for period in cfg['periods']:
        for cost in cfg['costs']:
            for model in MODELS:
                folder = SOURCE / 'evaluation' / period / cost / model
                files.extend([folder / 'ledger.parquet', folder / 'decisions.parquet'])
    write_json(OUT / 'protocol.json', {'study_id': STUDY, 'registered_at': now(),
        'source_study': cfg['study_id'], 'known_before_this_analysis': '四情景未达标、142次更新、20次分界确认已知',
        'analysis_scope': '三组固定差值、两个时期、两档费用、全部学习更新及后续窗口；仅已保存结果',
        'new_accounts': 0, 'new_model_fits': 0, 'new_candidates': 0,
        'files': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in files]}, exclusive=True)
    index = read(INDEX)
    index['running_studies'].append({'study': STUDY, 'status': 'REGISTERED_SAVED_ACCOUNT_FAILURE_EXPLANATION'})
    index['updated_at'] = now()
    index['next_work'].update(registered=True, next_historical_question_status='SAVED_FAILURE_EXPLANATION_REGISTERED')
    write_json(INDEX, index)
    print('保存账户失效定位的范围已固定。', flush=True)


def run():
    assert not (OUT / 'result.json').exists(), '已完成本次解释，不重复统计'
    protocol = read(OUT / 'protocol.json')
    for f in protocol['files']:
        assert digest(ROOT / f['path']) == f['sha256'], '原文件变化：' + f['path']
    cfg = read(SOURCE / 'protocol.json')
    learned = read(SOURCE / 'learning_completed.json')['records']
    result = read(SOURCE / 'result.json')
    data = pd.read_parquet(ROOT / cfg['features']).set_index('date')
    transitions = []
    for record in learned:
        winner = record['training_winner']
        baseline = record['no_factor_policy']
        training, confirmation = record['training_scores'], record['confirmation_scores']
        transitions.append({'fit_origin': record['fit_origin'], 'winner': winner, 'baseline': baseline,
            'training_feature_winner': winner not in ['CASH', 'PASSIVE'], 'accepted': record['feature_policy_accepted'],
            'training_utility_gap': training[winner]['utility'] - training[baseline]['utility'],
            'confirmation_utility_gap': confirmation[winner]['utility'] - confirmation[baseline]['utility'],
            'primary_policy': record['primary_policy']})
    csv(OUT / 'all_training_confirmation_transitions.csv', transitions)
    phase = pd.DataFrame(transitions)
    learned_origins = [r['fit_origin_index'] for r in learned]
    components, comparisons, daily_comparisons, windows, states = [], [], [], [], []
    rows_checked = decisions_checked = 0
    for period in cfg['periods']:
        for cost in cfg['costs']:
            ledgers, daily_parts, aggregate_parts = {}, {}, {}
            active = None
            for model in MODELS:
                folder = SOURCE / 'evaluation' / period / cost / model
                ledger = pd.read_parquet(folder / 'ledger.parquet')
                decisions = pd.read_parquet(folder / 'decisions.parquet')
                assert np.array_equal(decisions.execution_date.iloc[:-1].to_numpy(), ledger.date.to_numpy())
                assignments = active_fit_indices(decisions, learned_origins)
                if active is None:
                    active = assignments
                else:
                    assert np.array_equal(active, assignments)
                assert len(ledger) == len(active)
                ledgers[model] = ledger
                daily_parts[model], aggregate_parts[model] = decomposition(ledger, data.loc[ledger.date], cfg['initial_capital'], cfg['annual_days'])
                components.append({'period': period, 'cost': cost, 'model': model, **aggregate_parts[model]})
                rows_checked += len(ledger)
                decisions_checked += len(decisions)
            for left, right in PAIRS:
                label = {'period': period, 'cost': cost, 'left': left, 'right': right}
                diff = {key: aggregate_parts[left][key] - aggregate_parts[right][key] for key in aggregate_parts[left]}
                np.testing.assert_allclose(sum(diff[k] for k in ['average_weight_part', 'timing_part', 'execution_rights_part', 'friction_part']), diff['net_return'], atol=1e-12, rtol=0)
                comparisons.append({**label, **diff})
                p, b = ledgers[left], ledgers[right]
                parts = daily_parts[left].copy()
                for key in parts.columns:
                    if key != 'date':
                        parts[key] -= daily_parts[right][key]
                parts['active_fit_number'] = active
                parts['active_primary_policy'] = [learned[i]['primary_policy'] for i in active]
                parts['active_primary_kind'] = [learned[i]['primary_spec']['kind'] for i in active]
                parts['active_no_factor_policy'] = [learned[i]['no_factor_policy'] for i in active]
                parts['accepted_feature'] = [learned[i]['feature_policy_accepted'] for i in active]
                daily_comparisons.append(parts.assign(**label))
                for fit_number in np.unique(active):
                    indices = np.flatnonzero(active == fit_number)
                    assert np.all(np.diff(indices) == 1)
                    start, end = int(indices[0]), int(indices[-1])
                    r = learned[fit_number]
                    left_growth = float(np.prod(1 + p.net_return.iloc[indices]) - 1)
                    right_growth = float(np.prod(1 + b.net_return.iloc[indices]) - 1)
                    left_initial = cfg['initial_capital'] if start == 0 else p.equity.iloc[start - 1]
                    right_initial = cfg['initial_capital'] if start == 0 else b.equity.iloc[start - 1]
                    np.testing.assert_allclose(left_growth, p.equity.iloc[end] / left_initial - 1, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(right_growth, b.equity.iloc[end] / right_initial - 1, atol=1e-12, rtol=0)
                    windows.append({**label, 'fit_origin': r['fit_origin'], 'window_start': p.date.iloc[start],
                        'window_end': p.date.iloc[end], 'days': len(indices), 'end_censored': end == len(p) - 1,
                        'accepted_feature': r['feature_policy_accepted'], 'primary_policy': r['primary_policy'],
                        'no_factor_policy': r['no_factor_policy'], 'left_window_return': left_growth,
                        'right_window_return': right_growth, 'window_return_gap': left_growth - right_growth,
                        'annual_arithmetic_contribution': float(parts.net_return.iloc[indices].sum() * 242 / len(p)),
                        'left_start_shares': int(p.shares_before.iloc[start]), 'right_start_shares': int(b.shares_before.iloc[start])})
                for policy, group in parts.groupby('active_primary_policy'):
                    states.append({**label, 'primary_policy': policy, 'days': len(group),
                        'annual_arithmetic_contribution': float(group.net_return.sum() * 242 / len(parts))})
    csv(OUT / 'account_components.csv', components)
    csv(OUT / 'paired_annual_arithmetic_components.csv', comparisons)
    csv(OUT / 'daily_paired_components.csv', pd.concat(daily_comparisons, ignore_index=True))
    csv(OUT / 'subsequent_active_rule_windows.csv', windows)
    csv(OUT / 'contributions_by_active_rule.csv', states)
    lookup = {(r['period'], r['cost'], r['left'], r['right']): r for r in comparisons}
    summary = []
    window_data = pd.DataFrame(windows)
    for period in cfg['periods']:
        for cost in cfg['costs']:
            parts = [lookup[period, cost, a, b] for a, b in PAIRS]
            for key in ['net_return', 'average_weight_part', 'timing_part', 'execution_rights_part', 'friction_part']:
                np.testing.assert_allclose(parts[0][key] + parts[1][key], parts[2][key], atol=1e-12, rtol=0)
            for left, right in PAIRS:
                subset = window_data[(window_data.period == period) & (window_data.cost == cost) & (window_data.left == left) & (window_data.right == right)]
                np.testing.assert_allclose(subset.annual_arithmetic_contribution.sum(), lookup[period, cost, left, right]['net_return'], atol=1e-12, rtol=0)
                accepted = subset[subset.accepted_feature & ~subset.end_censored]
                summary.append({'period': period, 'cost': cost, 'left': left, 'right': right,
                    'all_windows': len(subset), 'end_censored_windows': int(subset.end_censored.sum()),
                    'accepted_complete_windows': len(accepted),
                    'accepted_window_wins': int(accepted.window_return_gap.gt(1e-12).sum()),
                    'accepted_window_losses': int(accepted.window_return_gap.lt(-1e-12).sum()),
                    'accepted_window_ties': int(accepted.window_return_gap.abs().le(1e-12).sum()),
                    'accepted_windows_total_annual_arithmetic_contribution': float(accepted.annual_arithmetic_contribution.sum())})
    observed_winners = phase[phase.training_feature_winner]
    result_out = {'study_id': STUDY, 'completed_at': now(), 'status': 'COMPLETED_SAVED_PHASE_AND_COST_EXPLANATION',
        'training_updates': len(phase), 'training_feature_winners': len(observed_winners),
        'feature_winners_rejected_in_confirmation': int((~observed_winners.accepted).sum()),
        'feature_winners_accepted': int(observed_winners.accepted.sum()),
        'training_feature_winner_mean_utility_gap': float(observed_winners.training_utility_gap.mean()),
        'confirmation_feature_winner_mean_utility_gap': float(observed_winners.confirmation_utility_gap.mean()),
        'comparisons': comparisons, 'account_components': components, 'subsequent_summaries': summary,
        'verified_account_rows': rows_checked, 'verified_decision_rows': decisions_checked,
        'source_study_goal_achieved': result['goal_achieved'],
        'new_accounts': 0, 'new_model_fits': 0, 'new_candidates': 0, 'independent_validation': False,
        'goal_achieved': False, 'position_impact': 0,
        'interpretation_limit': '年化算术恒等式与继承原持仓的后续窗口描述；不证明因果，不是零费用重跑或独立验证'}
    write_json(OUT / 'result.json', result_out, exclusive=True)
    print('训练确认与随后账户失效定位完成，未新增模型或账户。', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='解释保存账户的规则失效阶段与费用')
    parser.add_argument('stage', choices=['freeze', 'run'])
    {'freeze': freeze, 'run': run}[parser.parse_args().stage]()
