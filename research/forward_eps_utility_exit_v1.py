"""在原前瞻EPS仓位上比较两种退出增量，保留全部真实账户结果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_utility_exit_account_v1 import simulate_utility_exit_account
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import summarize, save_account
from research.intraday_overnight_increment_v1 import normalize_dividends, block_indices, return_metrics, interval

CONFIG = ROOT / 'config/510300_forward_eps_utility_exit_v1.json'
MANIFEST = ROOT / 'config/510300_forward_eps_utility_exit_v1_manifest.json'
OUT = ROOT / 'reports/research/510300_forward_eps_utility_exit_v1'
SOURCE = ROOT / 'reports/research/510300_forward_eps_monthly_policy_v2_csi'
PARENT = ROOT / 'reports/research/510300_adaptive_allocation_v1'
VARIANTS = {'Z0_ORIGINAL_MONTHLY': (False, False), 'Z1_FORECAST_EXPIRY': (True, False),
            'Z2_FORECAST_EXPIRY_AND_TREND': (True, True)}


def freeze():
    config = read(ROOT / 'config/510300_forward_eps_monthly_policy_v2.json')
    config.update({'study_id': '510300_FORWARD_EPS_UTILITY_EXIT_V1',
                   'primary': 'Z2_FORECAST_EXPIRY_AND_TREND', 'baseline': 'Z0_ORIGINAL_MONTHLY',
                   'forecast_validity_trading_days': 60, 'bootstrap_day_blocks': [20, 60],
                   'candidate_configurations': 2, 'reused_method_controls': 1,
                   'planned_evaluation_accounts': 8, 'new_models_fit': 0,
                   'original_entry_and_sizing_function_preserved': True,
                   'source_model_chosen_after_its_history_was_observed': True,
                   'prepare_gpt_numerical_review_package': False})
    save(CONFIG, config, exclusive=True)
    paths = [CONFIG, Path(__file__), ROOT / 'research/forward_eps_utility_exit_account_v1.py',
             ROOT / 'tests/test_forward_eps_utility_exit_account_v1.py',
             ROOT / 'docs/510300_FORWARD_EPS_UTILITY_EXIT_V1.md',
             ROOT / 'research/event_clock_account_v1.py', ROOT / 'research/adaptive_allocation_v1.py',
             ROOT / 'research/intraday_overnight_increment_v1.py', ROOT / 'research/financial_annual_components_v1.py',
             ROOT / 'research/forward_eps_guosen_history_v1.py', ROOT / 'config/510300_research_authority_v6.json',
             SOURCE / 'signals.parquet', SOURCE / 'result.json', PARENT / 'features.parquet',
             ROOT / config['inputs']['dividends']]
    for cost in config['costs']:
        for model in ['E3_FORWARD_EPS', 'BUY_HOLD']:
            paths.append(SOURCE / 'evaluation' / cost / (model + '_ledger.parquet'))
    save(MANIFEST, {'registered_at': now(), 'candidate_configurations': 2, 'planned_evaluation_accounts': 8,
                    'new_candidate_returns_not_read': True, 'existing_eps_model_result_already_observed': True,
                    'files': [identity(path) for path in paths]}, exclusive=True)
    print('保留原入场与仓位、只改变退出的两个新增方案已登记，共八条账户。', flush=True)


def run():
    config = read(CONFIG)
    for row in read(MANIFEST)['files']:
        if identity(ROOT / row['path'])['sha256'] != row['sha256']:
            raise ValueError('已登记方法或输入变化')
    if (OUT / 'RUN_STARTED.json').exists():
        raise FileExistsError('本轮已经开始，先确认现有进程和保存结果')
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'RUN_STARTED.json', {'started_at': now(), 'manifest': identity(MANIFEST)}, exclusive=True)
    data = pd.read_parquet(PARENT / 'features.parquet')
    signals = pd.read_parquet(SOURCE / 'signals.parquet')
    assert data.date.tolist() == signals.date.tolist()
    prediction = signals.E3_FORWARD_EPS.to_numpy(float)
    mask = signals.event_mask.to_numpy(bool)
    dividends = normalize_dividends(pd.read_csv(ROOT / config['inputs']['dividends']))
    metrics, yearly, eras, checks, uncertainties, triggers = [], [], [], [], {}, {}
    for cost_name, cost in config['costs'].items():
        accounts = {}
        for model, (expiry, trend) in VARIANTS.items():
            ledger, decisions = simulate_utility_exit_account(data, dividends, config, cost, prediction, mask, expiry, trend)
            assert not ledger.terminal_unliquidated.iloc[-1]
            assert decisions.loc[~decisions.origin_index.isin(np.flatnonzero(mask)), 'requested_quantity'].le(0).all()
            selected = decisions.loc[decisions.new_exit_trigger]
            selected.to_csv(OUT / f'{cost_name}_{model}_退出首次触发.csv', index=False, encoding='utf-8-sig')
            ledger.loc[ledger.filled_quantity.ne(0)].to_csv(OUT / f'{cost_name}_{model}_全部进出场成交.csv', index=False, encoding='utf-8-sig')
            triggers[cost_name + '_' + model] = selected.exit_reasons.value_counts().to_dict()
            if model == 'Z0_ORIGINAL_MONTHLY':
                original = pd.read_parquet(SOURCE / 'evaluation' / cost_name / 'E3_FORWARD_EPS_ledger.parquet')
                for field in ['equity','shares','cash','net_return','commission','slippage_cost','dividend_receivable','filled_quantity']:
                    np.testing.assert_array_equal(ledger[field], original[field])
                checks.append({'cost':cost_name, 'original_monthly_account_exactly_reproduced': True})
            save_account(OUT / 'evaluation' / cost_name, model, ledger, decisions)
            accounts[model] = ledger
        ledger, decisions = simulate_event_account(data, dividends, config, cost, config['evaluation_start'],
                                                    'BUY_HOLD', horizon=60, event_mask=mask)
        original = pd.read_parquet(SOURCE / 'evaluation' / cost_name / 'BUY_HOLD_ledger.parquet')
        np.testing.assert_array_equal(ledger.equity, original.equity)
        save_account(OUT / 'evaluation' / cost_name, 'BUY_HOLD', ledger, decisions)
        accounts['BUY_HOLD'] = ledger
        benchmark = summarize(ledger, config)
        for model, ledger in accounts.items():
            metric = {'cost':cost_name, 'model':model, **summarize(ledger,config)}
            metric['annualized_return_excess_vs_buy_hold'] = metric['annualized_return'] - benchmark['annualized_return']
            metric['meets_point_target'] = metric['net_sharpe'] is not None and metric['net_sharpe'] >= 1.2
            metrics.append(metric)
            for year, group in ledger.groupby(ledger.date.dt.year):
                yearly.append({'cost':cost_name, 'model':model, 'year':int(year), **summarize(group,config)})
            for label, a, b in [('2020—2021','2020-01-01','2021-12-31'),('2022—2023','2022-01-01','2023-12-31'),
                                ('2024—终点','2024-01-01',config['data_cutoff'])]:
                eras.append({'cost':cost_name, 'model':model, 'era':label, **summarize(ledger.loc[ledger.date.between(a,b)],config)})
        returns = pd.DataFrame({'date':accounts['BUY_HOLD'].date,
                                **{model:ledger.net_return.to_numpy() for model,ledger in accounts.items()}})
        returns.to_parquet(OUT / f'{cost_name}_all_evaluation_returns.parquet', index=False)
        uncertainties[cost_name] = {}
        for block in config['bootstrap_day_blocks']:
            rng = np.random.default_rng(config['random_seed'])
            draws = []
            for _ in range(config['bootstrap_repetitions']):
                ix = block_indices(rng, len(returns), block)
                draw = {}
                for model in VARIANTS:
                    values = returns[model].to_numpy()[ix]
                    draw[model + '_sharpe'] = return_metrics(values, config['annual_days'])['net_sharpe']
                    if model != 'Z0_ORIGINAL_MONTHLY':
                        draw[model + '_minus_original'] = float((values-returns.Z0_ORIGINAL_MONTHLY.to_numpy()[ix]).mean()*config['annual_days'])
                        draw[model + '_minus_buy_hold'] = float((values-returns.BUY_HOLD.to_numpy()[ix]).mean()*config['annual_days'])
                draws.append(draw)
            saved = pd.DataFrame(draws)
            saved.to_parquet(OUT / f'{cost_name}_block{block}_saved_bootstrap_statistics.parquet', index=False)
            uncertainties[cost_name][str(block)] = {key + '_95_interval': interval(saved[key].dropna().tolist()) for key in saved}
        print('原仓位与两种退出增量的四条完整账户已完成：', cost_name, flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / 'metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(yearly).to_csv(OUT / 'yearly_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(eras).to_csv(OUT / 'era_metrics.csv', index=False, encoding='utf-8-sig')
    save(OUT / 'execution_checks.json', {'checks':checks, 'trigger_counts':triggers}, exclusive=True)
    save(OUT / 'uncertainty.json', uncertainties, exclusive=True)
    result = {'study_id':config['study_id'], 'completed_at':now(), 'status':'UTILITY_ENTRY_PRESERVED_EXIT_COMPARISON_COMPLETE',
              'primary':frame.loc[frame.model.eq(config['primary'])].to_dict('records'), 'all_metrics':metrics,
              'uncertainty':uncertainties, 'candidate_configurations':2, 'reused_method_controls':1, 'evaluation_accounts':8,
              'new_models_fit':0, 'trigger_counts':triggers, 'goal_achieved':False,
              'historical_point_target_met':bool(frame.loc[frame.model.eq(config['primary']),'meets_point_target'].any()),
              'independent_validation':'NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY', 'position_impact':0}
    save(OUT / 'result.json', result, exclusive=True)
    print(json.dumps({'status':result['status'], 'primary':result['primary']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    freeze() if args.freeze else run()
