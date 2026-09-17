"""风险预测门通过后，按预先写定的映射执行完整账户并复核保存结果。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, target_request
from research.intraday_overnight_increment_v1 import digest, now, write_json
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account
from research.strategy_review_diagnostics_v1 import metrics, cycles
from research.monthly_downside_forecast_v1 import (
    ROOT, OUT, CANDIDATE, BASELINES, INDEX, check_protocol, read, csv, inputs,
)


def risk_target(risk, cfg):
    if not np.isfinite(risk) or risk < 0:
        return np.nan
    return min(1.0, cfg['target_volatility'] / math.sqrt(2 * max(risk, cfg['positive_risk_floor'])))


def risk_request(account, price, target, cfg):
    target_lots = math.floor(target * account.value(price) / price / cfg['lot'])
    return target_request(account, price, 0.0 if target_lots == 0 else target, cfg)


def reference_source(period, cost, model):
    if model == 'BUY_HOLD':
        return ROOT / f'reports/research/510300_simple_core_window_diagnostic_v1/comparators/{period}_BUY_HOLD_{cost}/ledger.parquet'
    assert model == 'MONTHLY_VOL10'
    return ROOT / f'reports/research/510300_monthly_single_factor_walkforward_v1/accounts/{period}/{cost}/MONTHLY_VOL10/ledger.parquet'


def freeze():
    cfg = check_protocol()
    prediction = read(OUT / 'prediction_result.json')
    assert prediction['prediction_gate_pass'], '预测门未通过，不开放账户阶段'
    assert read(OUT / 'saved_verification_receipt.json')['status'] == 'PASS_SAVED_LABEL_TIMING_AND_PREDICTION_LOSS_CHECKS'
    assert not (OUT / 'account_protocol.json').exists(), '账户阶段已登记，不能重登记'
    tests = read(OUT / 'account_tests_receipt.json')
    assert tests['passed'] == 2 and tests['failed'] == 0
    paths = [Path(__file__), ROOT / 'tests/test_monthly_downside_accounts_v1.py',
             OUT / 'forecasts.csv', OUT / 'prediction_result.json', OUT / 'saved_verification_receipt.json',
             ROOT / 'research/strategy_review_diagnostics_v1.py',
             ROOT / 'research/intraday_overnight_increment_v1.py']
    for period in cfg['periods']:
        for cost in cfg['costs']:
            for model in ['BUY_HOLD', 'MONTHLY_VOL10']:
                paths.append(reference_source(period, cost, model))
    write_json(OUT / 'account_protocol.json', {
        'registered_at': now(), 'account_rules': cfg['rules'],
        'prediction_gate_passed_before_account_results': True,
        'mapping_changed_after_prediction_results': False,
        'new_accounts': 12, 'reused_accounts': 8, 'new_model_fits_in_account_stage': 0,
        'frozen_files': [{'path': str(path.relative_to(ROOT)), 'sha256': digest(path)} for path in paths],
        'goal_achieved': False, 'independent_validation': False,
    }, exclusive=True)
    index = read(INDEX)
    index['updated_at'] = now()
    for study in index['running_studies']:
        if study.get('study') == cfg['study_id']:
            study['status'] = 'PREDICTION_PASS_ACCOUNT_STAGE_REGISTERED'
    index['next_work'].update(planned_new_accounts=12, planned_new_model_fits=0,
                              next_historical_question_status='PREDICTION_PASS_ACCOUNT_STAGE_REGISTERED')
    index['last_goal_turn_classification'] = 'PROGRESS_RISK_PREDICTION_INCREMENT_PASS'
    write_json(INDEX, index)
    print('预测门已通过，按原定仓位映射登记12份新账户；不重新拟合模型。', flush=True)


def check_account_protocol():
    cfg = check_protocol()
    registered = read(OUT / 'account_protocol.json')
    for item in registered['frozen_files']:
        assert digest(ROOT / item['path']) == item['sha256'], '账户固定输入或方法变化：' + item['path']
    return cfg


def run():
    cfg = check_account_protocol()
    assert not (OUT / 'accounts_completed.json').exists(), '账户已完成，不重复计算'
    began = time.perf_counter()
    data = inputs(cfg)
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg['dividends']))
    forecasts = pd.read_csv(OUT / 'forecasts.csv', parse_dates=['origin'])
    events = np.zeros(len(data), bool)
    targets = {model: np.full(len(data), np.nan) for model in [CANDIDATE] + BASELINES}
    for row in forecasts.to_dict('records'):
        origin_index = int(row['origin_index'])
        assert data.date.iloc[origin_index] == row['origin']
        events[origin_index] = True
        for model in targets:
            targets[model][origin_index] = risk_target(row[model], cfg)
    np.savez_compressed(OUT / 'account_targets.npz', events=events, **targets)
    accounts = []
    for period, (start, end, next_date) in cfg['periods'].items():
        frame = data[data.date.le(end)].reset_index(drop=True)
        for model in targets:
            for cost_id, cost in cfg['costs'].items():
                folder = OUT / 'accounts' / period / cost_id / model
                completed = folder / 'completed.json'
                if completed.exists():
                    accounts.append(read(completed))
                    continue
                ledger, decisions, state = simulate_indexed_request_account(
                    frame, dividends, cfg, cost, start, model,
                    targets=targets[model][:len(frame)], event_mask=events[:len(frame)],
                    next_execution_date=next_date,
                    request_policy=lambda account, price, value, config, identity, t: risk_request(account, price, value, config))
                decisions['monthly_event'] = events[decisions.origin_index.to_numpy(int)]
                folder.mkdir(parents=True, exist_ok=True)
                ledger.to_parquet(folder / 'ledger.parquet', index=False)
                decisions.to_parquet(folder / 'decisions.parquet', index=False)
                csv(folder / 'trades.csv', ledger[ledger.filled_quantity.ne(0)])
                write_json(folder / 'checkpoint.json', state)
                result = {'period': period, 'model': model, 'cost': cost_id, 'reused': False, **metrics(ledger),
                          'ending_shares': int(ledger.shares.iloc[-1]),
                          'unfilled_requests': int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum())}
                write_json(completed, result, exclusive=True)
                accounts.append(result)
                print('完整账户完成：' + period + '／' + cost_id + '／' + model, flush=True)
        for model in ['BUY_HOLD', 'MONTHLY_VOL10']:
            for cost_id in cfg['costs']:
                source = reference_source(period, cost_id, model)
                ledger = pd.read_parquet(source)
                assert ledger.date.iloc[0] == pd.Timestamp(start) and ledger.date.iloc[-1] == pd.Timestamp(end)
                assert ledger.mark_clock.eq('CLOSE').all()
                accounts.append({'period': period, 'model': model, 'cost': cost_id, 'reused': True,
                                 'source': str(source.relative_to(ROOT)), **metrics(ledger),
                                 'ending_shares': int(ledger.shares.iloc[-1])})
    csv(OUT / 'account_metrics.csv', accounts)
    write_json(OUT / 'accounts_completed.json', {
        'completed_at': now(), 'new_accounts': 12, 'reused_accounts': 8, 'new_model_fits': 0,
        'metrics': accounts, 'run_seconds': time.perf_counter() - began,
    }, exclusive=True)


def summarize():
    cfg = check_account_protocol()
    completed = read(OUT / 'accounts_completed.json')
    yearly, cycle_rows, concentrations = [], [], []
    for item in completed['metrics']:
        source = ROOT / item['source'] if item['reused'] else OUT / 'accounts' / item['period'] / item['cost'] / item['model'] / 'ledger.parquet'
        ledger = pd.read_parquet(source)
        identity = {key: item[key] for key in ['period', 'cost', 'model']}
        for year, group in ledger.groupby(ledger.date.dt.year):
            initial = float(group.equity.iloc[0] / (1 + group.net_return.iloc[0]))
            yearly.append({**identity, 'year': int(year), 'partial_year': int(year) == 2026, **metrics(group, initial)})
        history, active = cycles(ledger)
        realized = float(history.profit.sum()) if len(history) else 0.0
        unfinished = float(ledger.equity.iloc[-1] - active['start_equity']) if active else 0.0
        np.testing.assert_allclose(realized + unfinished, item['profit'], atol=1e-6, rtol=0)
        cycle_rows.extend({**identity, **row} for row in history.to_dict('records'))
        winners = history[history.profit.gt(0)].sort_values('profit', ascending=False) if len(history) else history
        concentrations.append({**identity, 'completed_cycles': len(history), 'unfinished_cycle': active is not None,
                               'total_profit': item['profit'], 'unfinished_cycle_profit': unfinished,
                               'top_five_completed_cycle_profit': float(winners.profit.head(5).sum()) if len(winners) else 0.0})
    csv(OUT / 'account_yearly_metrics.csv', yearly)
    csv(OUT / 'completed_cycles.csv', cycle_rows if cycle_rows else
        pd.DataFrame(columns=['period', 'cost', 'model', 'entry', 'exit', 'profit']))
    csv(OUT / 'cycle_concentration.csv', concentrations)
    rows = {(r['period'], r['cost'], r['model']): r for r in completed['metrics']}
    comparisons = []
    for period in cfg['periods']:
        for cost in cfg['costs']:
            candidate = rows[period, cost, CANDIDATE]
            comparisons.append({
                'period': period, 'cost': cost, 'annual_return': candidate['annual_return'],
                'sharpe': candidate['sharpe'], 'max_drawdown': candidate['max_drawdown'],
                'historical_point_targets_pass': candidate['annual_return'] >= .10 and candidate['sharpe'] >= 1.2,
                'vs_comparators': [{
                    'model': baseline,
                    'annual_return_difference': candidate['annual_return'] - rows[period, cost, baseline]['annual_return'],
                    'sharpe_difference': candidate['sharpe'] - rows[period, cost, baseline]['sharpe'],
                } for baseline in BASELINES + ['BUY_HOLD', 'MONTHLY_VOL10']],
            })
    all_targets = all(r['historical_point_targets_pass'] for r in comparisons)
    prediction = read(OUT / 'prediction_result.json')
    result = {
        'study_id': cfg['study_id'], 'completed_at': now(),
        'status': 'RISK_INCREMENT_PASS_HISTORICAL_ACCOUNT_POINT_TARGETS_PASS_UNVALIDATED' if all_targets
                  else 'RISK_INCREMENT_PASS_ACCOUNT_TARGETS_NOT_MET',
        'new_candidate_methods': 1, 'new_model_fits': prediction['new_model_fits'],
        'paired_prediction_months': prediction['paired_months'], 'prediction_gate_pass': True,
        'account_stage': 'COMPLETED', 'new_accounts': 12, 'reused_accounts': 8,
        'metrics': completed['metrics'], 'candidate_comparisons': comparisons,
        'all_four_historical_point_targets_pass': all_targets,
        'no_retuning_of_registered_risk_to_position_mapping': True,
        'risk_forecast_role': 'FIXED_INFORMATION_BENCHMARK_NOT_A_VALIDATED_HIGH_SHARPE_STRATEGY',
        'account_run_seconds': completed['run_seconds'],
        'independent_validation': False, 'strict_forward_evidence_days': 0,
        'goal_achieved': False, 'position_impact': 0,
    }
    write_json(OUT / 'result.json', result, exclusive=True)
    print('完整账户结果已汇总；全部四场景历史目标通过：' + str(all_targets), flush=True)


def verify():
    cfg = check_account_protocol()
    result = read(OUT / 'result.json')
    dates = inputs(cfg).date
    checked_rows, checked_decisions = 0, 0
    with np.load(OUT / 'account_targets.npz') as saved:
        events = saved['events'].copy()
        targets = {model: saved[model].copy() for model in [CANDIDATE] + BASELINES}
    forecasts = pd.read_csv(OUT / 'forecasts.csv')
    for row in forecasts.to_dict('records'):
        for model in targets:
            expected = risk_target(row[model], cfg)
            np.testing.assert_allclose(targets[model][int(row['origin_index'])], expected, atol=1e-14, rtol=0)
    for item in result['metrics']:
        folder = OUT / 'accounts' / item['period'] / item['cost'] / item['model']
        ledger = pd.read_parquet(ROOT / item['source'] if item['reused'] else folder / 'ledger.parquet')
        qty = ledger.filled_quantity.to_numpy(float)
        price = ledger.fill_price.fillna(0).to_numpy(float)
        old_cash = np.r_[cfg['initial_capital'], ledger.cash.to_numpy()[:-1]]
        old_shares = np.r_[0, ledger.shares.to_numpy()[:-1]]
        old_receivable = np.r_[0, ledger.dividend_receivable.to_numpy()[:-1]]
        np.testing.assert_allclose(ledger.cash, old_cash + ledger.dividend_paid - qty * price - ledger.commission, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(ledger.shares, old_shares + qty)
        np.testing.assert_allclose(ledger.dividend_receivable, old_receivable + ledger.dividend_recognized - ledger.dividend_paid, atol=1e-7, rtol=0)
        np.testing.assert_allclose(ledger.equity, ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, atol=1e-7, rtol=0)
        assert (qty % 100 == 0).all() and ((-qty[qty < 0]) <= old_shares[qty < 0]).all()
        cost = cfg['costs'][item['cost']]
        fee = np.where(qty != 0, np.maximum(abs(qty) * price * cost['commission'], cost['minimum']), 0)
        np.testing.assert_allclose(ledger.commission, fee, atol=1e-9, rtol=0)
        returns = ledger.equity.to_numpy() / np.r_[cfg['initial_capital'], ledger.equity.to_numpy()[:-1]] - 1
        np.testing.assert_allclose(ledger.net_return, returns, atol=1e-12, rtol=0)
        annual = (float(ledger.equity.iloc[-1]) / cfg['initial_capital']) ** (cfg['annual_days'] / len(ledger)) - 1
        sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(cfg['annual_days'])
        np.testing.assert_allclose([annual, sharpe], [item['annual_return'], item['sharpe']], atol=1e-12, rtol=0)
        assert ledger.mark_clock.eq('CLOSE').all()
        if not item['reused']:
            decisions = pd.read_parquet(folder / 'decisions.parquet')
            origin_indices = decisions.origin_index.to_numpy(int)
            np.testing.assert_array_equal(decisions.origin, dates.iloc[origin_indices])
            np.testing.assert_array_equal(decisions.monthly_event, events[origin_indices])
            np.testing.assert_array_equal(decisions.execution_date.iloc[:-1], ledger.date)
            assert decisions.origin.lt(decisions.execution_date).all()
            assert decisions.loc[~decisions.monthly_event, 'requested_quantity'].eq(0).all()
            np.testing.assert_array_equal(ledger.requested_quantity, decisions.requested_quantity.iloc[:-1])
            checked_decisions += len(decisions)
        checked_rows += len(ledger)
    assert len(result['metrics']) == 20 and sum(not r['reused'] for r in result['metrics']) == 12
    write_json(OUT / 'account_verification_receipt.json', {
        'verified_at': now(), 'status': 'PASS_SAVED_20_FULL_ACCOUNTS_AND_MONTHLY_EXECUTION',
        'accounts': 20, 'new_accounts': 12, 'reused_accounts': 8,
        'ledger_rows': checked_rows, 'new_decision_rows': checked_decisions,
        'new_accounts_generated_by_verification': 0, 'new_model_fits': 0,
        'independent_validation': False, 'goal_achieved': False,
    }, exclusive=True)
    print(f'保存账户复核通过：20份账户，{checked_rows}行日账。', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='风险预测通过后的完整账户阶段')
    parser.add_argument('stage', choices=['freeze', 'run', 'summarize', 'verify'])
    argument = parser.parse_args()
    {'freeze': freeze, 'run': run, 'summarize': summarize, 'verify': verify}[argument.stage]()
