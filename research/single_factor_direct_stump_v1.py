"""单因素单层交易规则，以过去完整账户训练并在随后历史段确认。"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, target_request
from research.intraday_overnight_increment_v1 import digest, now, write_json
from research.monthly_single_factor_walkforward_v1 import read, csv
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account
from research.strategy_review_diagnostics_v1 import metrics, cycles


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/510300_single_factor_direct_stump_v1.json'
OUT = ROOT / 'reports/research/510300_single_factor_direct_stump_v1'
INDEX = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
MODELS = ['CONFIRMED_STUMP', 'NO_FACTOR_LEARNER', 'FIXED_WEEKLY_VOL10']
POLICY_ORDER = ['CASH', 'PASSIVE', 'LOW_Q25', 'HIGH_Q25', 'LOW_Q50', 'HIGH_Q50', 'LOW_Q75', 'HIGH_Q75']


def inputs(cfg):
    data = pd.read_parquet(ROOT / cfg['features'])
    data['date'] = pd.to_datetime(data.date)
    data['five_day_total_return'] = np.expm1(data.mom5)
    assert data.date.is_monotonic_increasing and not data.date.duplicated().any()
    assert data.date.iloc[-1] == pd.Timestamp(cfg['data_cutoff'])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg['dividends']))
    return data, dividends


def policies_from_training(values, cfg):
    values = np.asarray(values, float)
    assert np.isfinite(values).all()
    thresholds = np.quantile(values, cfg['threshold_quantiles'], method=cfg['quantile_method'])
    policies = {'CASH': {'kind': 'CASH', 'threshold': None}, 'PASSIVE': {'kind': 'PASSIVE', 'threshold': None}}
    for quantile, threshold in zip(cfg['threshold_quantiles'], thresholds):
        suffix = str(int(round(quantile * 100)))
        policies['LOW_Q' + suffix] = {'kind': 'LOW', 'threshold': float(threshold)}
        policies['HIGH_Q' + suffix] = {'kind': 'HIGH', 'threshold': float(threshold)}
    assert list(policies) == POLICY_ORDER
    return policies


def targets_for_policy(data, policy, cfg):
    risk = np.minimum(1.0, cfg['target_volatility'] / data.vol20.where(data.vol20 > 0).to_numpy(float))
    if policy['kind'] == 'CASH':
        return np.zeros(len(data))
    if policy['kind'] == 'PASSIVE':
        return risk
    feature = data.five_day_total_return.to_numpy(float)
    allowed = feature < policy['threshold'] if policy['kind'] == 'LOW' else feature >= policy['threshold']
    output = np.where(allowed, risk, 0.0)
    output[~np.isfinite(feature) | ~np.isfinite(risk)] = np.nan
    return output


def request(account, price, target, cfg):
    desired_lots = np.floor(target * account.value(price) / price / cfg['lot'])
    return target_request(account, price, 0.0 if desired_lots == 0 else target, cfg)


def utility(returns, cfg):
    returns = np.asarray(returns, float)
    return float(cfg['annual_days'] * (returns.mean() - cfg['utility_mean_variance_gamma'] / 2 * returns.var(ddof=1)))


def choose(scores, allowed, cfg):
    chosen = None
    for key in POLICY_ORDER:
        if key not in allowed:
            continue
        if chosen is None:
            chosen = key
            continue
        difference = scores[key]['utility'] - scores[chosen]['utility']
        if difference > cfg['utility_comparison_tolerance']:
            chosen = key
        elif abs(difference) <= cfg['utility_comparison_tolerance'] and scores[key]['fills'] < scores[chosen]['fills']:
            chosen = key
    assert chosen is not None
    return chosen


def confirm_choices(training, confirmation, cfg):
    no_factor_winner = choose(training, ['CASH', 'PASSIVE'], cfg)
    candidate_winner = choose(training, POLICY_ORDER, cfg)
    tolerance = cfg['utility_comparison_tolerance']
    baseline = no_factor_winner if confirmation[no_factor_winner]['utility'] > confirmation['PASSIVE']['utility'] + tolerance else 'PASSIVE'
    accepted = candidate_winner not in ['CASH', 'PASSIVE'] and confirmation[candidate_winner]['utility'] > confirmation[baseline]['utility'] + tolerance
    return {
        'training_winner': candidate_winner, 'no_factor_training_winner': no_factor_winner,
        'no_factor_policy': baseline, 'primary_policy': candidate_winner if accepted else baseline,
        'feature_policy_accepted': bool(accepted),
        'reason': '分界规则在随后确认段优于无因子规则' if accepted else '训练或确认没有支持分界规则，采用无因子规则',
    }


def account_piece(data, dividends, cfg, start_index, end_index, next_date, policy, policy_id):
    targets = targets_for_policy(data, policy, cfg)
    event_mask = data.date.dt.weekday.eq(cfg['execution_decision_weekday']).to_numpy()
    ledger, decisions, _ = simulate_indexed_request_account(
        data, dividends, cfg, cfg['costs'][cfg['training_cost']], str(data.date.iloc[start_index].date()),
        policy_id, targets=targets, event_mask=event_mask, stop_index=end_index,
        next_execution_date=next_date,
        request_policy=lambda account, price, value, config, identity, t: request(account, price, value, config))
    score = {'policy': policy_id, 'utility': utility(ledger.net_return, cfg), **metrics(ledger)}
    assert ledger.date.iloc[0] == data.date.iloc[start_index] and ledger.date.iloc[-1] == data.date.iloc[end_index]
    assert decisions.origin.iloc[0] == data.date.iloc[start_index - 1]
    return score, ledger


def learn_at(data, dividends, origin_index, cfg, next_date):
    end = int(origin_index)
    start = end - cfg['training_days'] - cfg['confirmation_days'] + 1
    assert start > 0, '过去完整训练与确认记录不足'
    train_end = start + cfg['training_days'] - 1
    confirm_start = train_end + 1
    prefix = data.iloc[:end + 1].reset_index(drop=True)
    signal_indices = np.arange(start - 1, train_end)
    scheduled = prefix.date.iloc[signal_indices].dt.weekday.eq(cfg['execution_decision_weekday']).to_numpy(copy=True)
    scheduled[0] = True
    signal_indices = signal_indices[scheduled]
    x = prefix.five_day_total_return.iloc[signal_indices].to_numpy(float)
    assert np.isfinite(x).all() and len(x) >= cfg['minimum_training_signal_origins']
    policies = policies_from_training(x, cfg)
    training, confirmation, frames = {}, {}, []
    for policy_id in POLICY_ORDER:
        score, ledger = account_piece(prefix, dividends, cfg, start, train_end, next_date, policies[policy_id], policy_id)
        training[policy_id] = score
        frames.append(ledger.assign(phase='TRAINING', policy=policy_id))
    chosen = choose(training, POLICY_ORDER, cfg)
    no_factor_chosen = choose(training, ['CASH', 'PASSIVE'], cfg)
    needed = {chosen, no_factor_chosen, 'PASSIVE'}
    for policy_id in POLICY_ORDER:
        if policy_id not in needed:
            continue
        score, ledger = account_piece(prefix, dividends, cfg, confirm_start, end, next_date, policies[policy_id], policy_id)
        confirmation[policy_id] = score
        frames.append(ledger.assign(phase='CONFIRMATION', policy=policy_id))
    selection = confirm_choices(training, confirmation, cfg)
    record = {
        'fit_origin': prefix.date.iloc[end], 'fit_origin_index': end,
        'training_start': prefix.date.iloc[start], 'training_end': prefix.date.iloc[train_end],
        'confirmation_start': prefix.date.iloc[confirm_start], 'confirmation_end': prefix.date.iloc[end],
        'training_start_index': start, 'training_end_index': train_end,
        'confirmation_start_index': confirm_start, 'confirmation_end_index': end,
        'training_signal_indices': signal_indices.tolist(), 'training_signal_origins': len(signal_indices),
        'policies': policies, 'training_scores': training, 'confirmation_scores': confirmation,
        'internal_training_accounts': len(training), 'internal_confirmation_accounts': len(confirmation),
        'primary_spec': policies[selection['primary_policy']],
        'no_factor_spec': policies[selection['no_factor_policy']], **selection,
    }
    return record, pd.concat(frames, ignore_index=True)


def learning_origins(data, cfg):
    month = data.date.dt.to_period('M')
    events = month.ne(month.shift(1)).to_numpy(copy=True)
    events[data.date.lt(cfg['initial_fit_origin'])] = False
    initial = np.flatnonzero(data.date.eq(cfg['initial_fit_origin']))
    assert len(initial) == 1
    events[initial[0]] = True
    return np.flatnonzero(events)


def reference_source(period, cost, model):
    return ROOT / f'reports/research/510300_simple_core_window_diagnostic_v1/comparators/{period}_{model}_{cost}/ledger.parquet'


def freeze():
    cfg = read(CONFIG)
    assert not (OUT / 'protocol.json').exists(), '本轮已固定，不能重新登记'
    tests = read(OUT / 'tests_receipt.json')
    assert tests['passed'] == 5 and tests['failed'] == 0
    data, _ = inputs(cfg)
    fit_indices = learning_origins(data, cfg)
    paths = [CONFIG, ROOT / cfg['rules'], Path(__file__), ROOT / cfg['features'], ROOT / cfg['dividends'],
             ROOT / 'tests/test_single_factor_direct_stump_v1.py',
             ROOT / 'research/adaptive_allocation_v1.py', ROOT / 'research/post_selection_continuous_accounts_v1.py',
             ROOT / 'research/intraday_overnight_increment_v1.py', ROOT / 'research/strategy_review_diagnostics_v1.py']
    for period in cfg['periods']:
        for cost in cfg['costs']:
            for model in ['BUY_HOLD', 'ETF_VOL10']:
                paths.append(reference_source(period, cost, model))
    protocol = {**cfg, 'registered_at': now(), 'planned_learning_batches': len(fit_indices),
                'planned_internal_training_accounts': len(fit_indices) * 8,
                'planned_internal_confirmation_accounts_max': len(fit_indices) * 3,
                'own_actual_results_read_before_freeze': False,
                'frozen_files': [{'path': str(path.relative_to(ROOT)), 'sha256': digest(path)} for path in paths]}
    write_json(OUT / 'protocol.json', protocol, exclusive=True)
    index = read(INDEX)
    index['updated_at'] = now()
    index['running_studies'].append({'study': cfg['study_id'], 'status': 'FROZEN_READY_FOR_FULL_ACCOUNT_LEARNING'})
    index['next_work'].update(registered=True, planned_settings=1, planned_new_accounts=12,
                              next_historical_question_status='SINGLE_FACTOR_DIRECT_STUMP_REGISTERED')
    index['last_goal_turn_classification'] = 'PROGRESS_LOW_COMPLEXITY_DIRECT_ACCOUNT_LEARNING_REGISTERED'
    write_json(INDEX, index)
    print(f'单一直接交易学习方法已固定：{len(fit_indices)}个月度批次，内部训练账户另计。', flush=True)


def check_protocol():
    cfg = read(OUT / 'protocol.json')
    for item in cfg['frozen_files']:
        assert digest(ROOT / item['path']) == item['sha256'], '固定方法或输入发生变化：' + item['path']
    return cfg


def learn():
    cfg = check_protocol()
    assert not (OUT / 'learning_completed.json').exists(), '学习已完成，不重复选择规则'
    began = time.perf_counter()
    data, dividends = inputs(cfg)
    origins = learning_origins(data, cfg)
    records = []
    for number, origin in enumerate(origins, 1):
        folder = OUT / 'monthly_learning' / data.date.iloc[origin].strftime('%Y%m%d')
        completed = folder / 'fit.json'
        if completed.exists():
            record = read(completed)
        else:
            next_date = data.date.iloc[origin + 1] if origin + 1 < len(data) else cfg['periods']['main'][2]
            record, ledgers = learn_at(data, dividends, origin, cfg, next_date)
            folder.mkdir(parents=True, exist_ok=True)
            ledgers.to_parquet(folder / 'internal_ledgers.parquet', index=False)
            write_json(completed, record, exclusive=True)
        records.append(record)
        if number % 12 == 0 or number == len(origins):
            print(f'完整账户训练与确认完成：{number}/{len(origins)}个月度批次。', flush=True)
    targets = {model: np.full(len(data), np.nan) for model in MODELS}
    targets['FIXED_WEEKLY_VOL10'] = targets_for_policy(data, {'kind': 'PASSIVE'}, cfg)
    for n, record in enumerate(records):
        start = int(record['fit_origin_index'])
        end = int(records[n + 1]['fit_origin_index']) if n + 1 < len(records) else len(data)
        for model, key in [('CONFIRMED_STUMP', 'primary_spec'), ('NO_FACTOR_LEARNER', 'no_factor_spec')]:
            targets[model][start:end] = targets_for_policy(data.iloc[start:end], record[key], cfg)
    events = data.date.dt.weekday.eq(cfg['execution_decision_weekday']).to_numpy()
    np.savez_compressed(OUT / 'targets.npz', events=events, **targets)
    brief = [{k: record[k] for k in ['fit_origin', 'training_start', 'training_end', 'confirmation_start', 'confirmation_end',
                                     'training_signal_origins', 'training_winner', 'no_factor_training_winner', 'no_factor_policy',
                                     'primary_policy', 'feature_policy_accepted', 'internal_training_accounts', 'internal_confirmation_accounts', 'reason']}
             for record in records]
    csv(OUT / 'monthly_selected_rules.csv', brief)
    write_json(OUT / 'learning_completed.json', {
        'completed_at': now(), 'learning_batches': len(records), 'new_model_fits': len(records) * 2,
        'model_fit_count_scope': '每批一个主规则学习输出、一个无因子学习输出；另列内部比较账户',
        'internal_training_accounts': sum(r['internal_training_accounts'] for r in records),
        'internal_confirmation_accounts': sum(r['internal_confirmation_accounts'] for r in records),
        'accepted_feature_policy_months': sum(r['feature_policy_accepted'] for r in records),
        'new_evaluation_accounts_so_far': 0, 'run_seconds': time.perf_counter() - began,
        'records': records,
    }, exclusive=True)


def evaluate():
    cfg = check_protocol()
    assert (OUT / 'learning_completed.json').exists()
    assert not (OUT / 'evaluation_completed.json').exists(), '正式评价账户已完成，不重复运行'
    began = time.perf_counter()
    data, dividends = inputs(cfg)
    with np.load(OUT / 'targets.npz') as saved:
        events = saved['events'].copy()
        targets = {model: saved[model].copy() for model in MODELS}
    results = []
    for period, (start, end, next_date) in cfg['periods'].items():
        frame = data[data.date.le(end)].reset_index(drop=True)
        for model in MODELS:
            for cost_id, cost in cfg['costs'].items():
                folder = OUT / 'evaluation' / period / cost_id / model
                completed = folder / 'completed.json'
                if completed.exists():
                    results.append(read(completed))
                    continue
                ledger, decisions, checkpoint = simulate_indexed_request_account(
                    frame, dividends, cfg, cost, start, model, targets=targets[model][:len(frame)],
                    event_mask=events[:len(frame)], next_execution_date=next_date,
                    request_policy=lambda account, price, value, config, identity, t: request(account, price, value, config))
                decisions['weekly_event'] = events[decisions.origin_index.to_numpy(int)]
                decisions['initialization_event'] = False
                decisions.loc[0, 'initialization_event'] = True
                folder.mkdir(parents=True, exist_ok=True)
                ledger.to_parquet(folder / 'ledger.parquet', index=False)
                decisions.to_parquet(folder / 'decisions.parquet', index=False)
                csv(folder / 'trades.csv', ledger[ledger.filled_quantity.ne(0)])
                write_json(folder / 'checkpoint.json', checkpoint)
                result = {'period': period, 'cost': cost_id, 'model': model, 'reused': False, **metrics(ledger),
                          'ending_shares': int(ledger.shares.iloc[-1]),
                          'unfilled_requests': int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum())}
                write_json(completed, result, exclusive=True)
                results.append(result)
                print('正式完整账户完成：' + period + '／' + cost_id + '／' + model, flush=True)
        for model in ['BUY_HOLD', 'ETF_VOL10']:
            for cost_id in cfg['costs']:
                source = reference_source(period, cost_id, model)
                ledger = pd.read_parquet(source)
                assert ledger.date.iloc[0] == pd.Timestamp(start) and ledger.date.iloc[-1] == pd.Timestamp(end)
                assert ledger.mark_clock.eq('CLOSE').all()
                results.append({'period': period, 'cost': cost_id, 'model': model, 'reused': True,
                                'source': str(source.relative_to(ROOT)), **metrics(ledger)})
    csv(OUT / 'evaluation_metrics.csv', results)
    write_json(OUT / 'evaluation_completed.json', {'completed_at': now(), 'new_evaluation_accounts': 12,
               'reused_reference_accounts': 8, 'metrics': results, 'run_seconds': time.perf_counter() - began}, exclusive=True)


def summarize():
    cfg = check_protocol()
    learning = read(OUT / 'learning_completed.json')
    evaluated = read(OUT / 'evaluation_completed.json')
    yearly, all_cycles, concentration = [], [], []
    for item in evaluated['metrics']:
        source = ROOT / item['source'] if item['reused'] else OUT / 'evaluation' / item['period'] / item['cost'] / item['model'] / 'ledger.parquet'
        ledger = pd.read_parquet(source)
        identity = {key: item[key] for key in ['period', 'cost', 'model']}
        for year, piece in ledger.groupby(ledger.date.dt.year):
            initial = float(piece.equity.iloc[0] / (1 + piece.net_return.iloc[0]))
            yearly.append({**identity, 'year': int(year), 'partial_year': int(year) == 2026, **metrics(piece, initial)})
        complete, active = cycles(ledger)
        unfinished = float(ledger.equity.iloc[-1] - active['start_equity']) if active else 0.0
        np.testing.assert_allclose((float(complete.profit.sum()) if len(complete) else 0.0) + unfinished, item['profit'], atol=1e-6, rtol=0)
        winners = complete[complete.profit.gt(0)].sort_values('profit', ascending=False) if len(complete) else complete
        concentration.append({**identity, 'complete_cycles': len(complete), 'unfinished_profit': unfinished,
                              'total_profit': item['profit'], 'top_five_completed_profit': float(winners.profit.head(5).sum()) if len(winners) else 0.0})
        all_cycles.extend({**identity, **row} for row in complete.to_dict('records'))
    csv(OUT / 'yearly_metrics.csv', yearly)
    csv(OUT / 'completed_cycles.csv', all_cycles if all_cycles else pd.DataFrame(columns=['period', 'cost', 'model', 'entry', 'exit', 'profit']))
    csv(OUT / 'cycle_concentration.csv', concentration)
    lookup = {(r['period'], r['cost'], r['model']): r for r in evaluated['metrics']}
    comparisons = []
    for period in cfg['periods']:
        for cost in cfg['costs']:
            candidate = lookup[period, cost, cfg['primary']]
            comparisons.append({
                'period': period, 'cost': cost, 'annual_return': candidate['annual_return'], 'sharpe': candidate['sharpe'],
                'point_goal_pass': candidate['annual_return'] >= .10 and candidate['sharpe'] is not None and candidate['sharpe'] >= 1.2,
                'both_baselines_better': all(candidate['annual_return'] > lookup[period, cost, base]['annual_return']
                    and candidate['sharpe'] is not None and lookup[period, cost, base]['sharpe'] is not None
                    and candidate['sharpe'] > lookup[period, cost, base]['sharpe'] for base in cfg['comparators']),
                'vs_baselines': [{'model': base,
                                  'annual_return_difference': candidate['annual_return'] - lookup[period, cost, base]['annual_return'],
                                  'sharpe_difference': candidate['sharpe'] - lookup[period, cost, base]['sharpe']
                                  if candidate['sharpe'] is not None and lookup[period, cost, base]['sharpe'] is not None else None}
                                 for base in cfg['comparators']],
            })
    years = {(r['year'], r['model']): r for r in yearly if r['cost'] == 'STRESS' and not r['partial_year']}
    wins = {base: sum(years[year, cfg['primary']]['annual_return'] > years[year, base]['annual_return']
                      for year in range(2015, 2026)) for base in cfg['comparators']}
    eligible = all(r['both_baselines_better'] for r in comparisons) and all(v >= cfg['continuation_full_year_wins_min'] for v in wins.values())
    passed = all(r['point_goal_pass'] for r in comparisons)
    result = {
        'study_id': cfg['study_id'], 'completed_at': now(),
        'status': 'COMPLETED_LOW_COMPLEXITY_DIRECT_STUMP_POINT_PASS_UNVALIDATED' if passed else 'COMPLETED_LOW_COMPLEXITY_DIRECT_STUMP_TARGET_NOT_MET',
        'new_complete_learning_methods': 1, 'monthly_learning_batches': learning['learning_batches'],
        'new_model_fits': learning['new_model_fits'], 'model_fit_count_scope': learning['model_fit_count_scope'],
        'internal_training_accounts': learning['internal_training_accounts'],
        'internal_confirmation_accounts': learning['internal_confirmation_accounts'],
        'new_evaluation_accounts': 12, 'reused_reference_accounts': 8,
        'accepted_feature_policy_months': learning['accepted_feature_policy_months'],
        'metrics': evaluated['metrics'], 'comparisons': comparisons, 'full_year_wins_vs_baselines': wins,
        'continuation_criteria_pass': eligible, 'all_four_historical_point_goals_pass': passed,
        'learning_seconds': learning['run_seconds'], 'evaluation_seconds': evaluated['run_seconds'],
        'independent_validation': False, 'strict_forward_evidence_days': 0, 'goal_achieved': False, 'position_impact': 0,
    }
    write_json(OUT / 'result.json', result, exclusive=True)
    print('直接交易规则检验完成；四情景目标通过：' + str(passed) + '，深化依据通过：' + str(eligible), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='单因素直接交易学习与完整账户')
    parser.add_argument('stage', choices=['freeze', 'learn', 'evaluate', 'summarize', 'run'])
    stage = parser.parse_args().stage
    if stage == 'run':
        learn()
        evaluate()
        summarize()
    else:
        {'freeze': freeze, 'learn': learn, 'evaluate': evaluate, 'summarize': summarize}[stage]()
