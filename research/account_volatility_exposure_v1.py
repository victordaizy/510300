"""第181轮完整账户风险预算，一次冻结后计算和核对。"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.account_volatility_exposure_inputs_v1 import CANDIDATES, MODELS, PARENT, PRIMARY, SOURCE, account_volatility_frames
from research.adaptive_allocation_v1 import normalize_dividends
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.saved_target_account_checks_v1 import verify_saved_target_accounts
from research.saved_target_batch_runner_v1 import run_saved_target_batch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_account_volatility_exposure_v1'
CONFIG = ROOT / 'config/510300_account_volatility_exposure_v1.json'
PARENTS = {PARENT: SOURCE}
CONTROLS = {PARENT: (SOURCE, '第174轮原资金预算'),
            'EXPOSURE_EXPANSION_200': (ROOT/'reports/research/510300_unlevered_exposure_expansion_v1', '第180轮固定两倍预算'),
            'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '第181轮已存在，请接续')
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT/'tests_receipt.json'
    require(not receipt.exists(), '必要测试已有回执')
    began = time.perf_counter()
    test = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_account_volatility_exposure_v1.py',
                           '-q', '-p', 'no:cacheprovider'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
    print(test.stdout, flush=True)
    require(test.returncode == 0 and '3 passed' in test.stdout, '第181轮必要测试未通过')
    write_json(receipt, {'recorded_at': now(), 'exit_code': test.returncode, 'passed': 3,
                         'seconds': time.perf_counter()-began, 'timing_scope': '完整测试进程墙钟'}, exclusive=True)
    source_config = ROOT/'config/510300_return_confirmation_auxiliary_batch_v1.json'
    old = json.loads(source_config.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
            'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends',
            'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {k: old[k] for k in keys}
    cfg.update(study_id='510300_ACCOUNT_VOLATILITY_EXPOSURE_V1', round=181, primary=PRIMARY,
        candidate_models=list(CANDIDATES), candidate_configurations=1, parent_models=MODELS,
        account_risk_window=60, account_risk_target=.10, annual_return_target=.10,
        registered_at=now(), decision_clock='15:05:00', source_budget_cny=0, new_model_fits=0,
        new_reference_accounts=0, position_impact=0, goal_achieved=False, independent_validation='NOT_ESTABLISHED',
        rules='docs/510300_ACCOUNT_VOLATILITY_EXPOSURE_V1.md',
        evidence_class='RETROSPECTIVE_REPLAY_PREVIOUSLY_OBSERVED_HISTORY',
        legacy_meets_point_target_field='SHARPE_ONLY_USE_JOINT_TARGET_ASSESSMENT',
        previous_goal_turn_classification='PROGRESS_ROUND180_EIGHT_ACCOUNTS_JOINT_TARGET_FAILED')
    rules = ROOT/cfg['rules']
    with rules.open('x', encoding='utf-8') as stream:
        stream.write((ROOT/'docs/510300_ACCOUNT_VOLATILITY_EXPOSURE_NEXT_20260913.md').read_text(encoding='utf-8'))
        stream.write('\n\n## 原来源全部中文因素与进入退出规则\n\n')
        stream.write((ROOT/old['rules']).read_text(encoding='utf-8'))
    paths = [Path(__file__), ROOT/'research/account_volatility_exposure_inputs_v1.py',
             ROOT/'research/saved_target_batch_runner_v1.py', ROOT/'research/saved_parent_target_alignment_v1.py',
             ROOT/'research/saved_target_account_checks_v1.py', ROOT/'research/event_clock_account_v1.py',
             ROOT/'research/adaptive_allocation_v1.py', ROOT/'research/intraday_overnight_increment_v1.py',
             ROOT/'tests/test_account_volatility_exposure_v1.py', receipt, OUT/'tests_output.txt',
             ROOT/cfg['features'], ROOT/cfg['dividends'], rules, source_config, ROOT/old['rules'],
             SOURCE/'saved_verification_receipt.json', ROOT/'config/510300_research_authority_v6.json']
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in cfg['costs']:
            paths += [SOURCE/period/cost/f'{PARENT}_decisions.parquet']
            paths += [folder/period/cost/f'{model}_ledger.parquet' for model, (folder, _) in CONTROLS.items()]
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    p = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i = json.loads(p.read_text(encoding='utf-8'))
    require(i['latest_completed_round']['round'] == 180, '前序轮次不同')
    i['running_studies'] = [{'round': 181, 'study': cfg['study_id'], 'status': 'FROZEN_NOT_STARTED', 'config': str(CONFIG.relative_to(ROOT))}]
    i['next_work'].update(registered=True, status='ACCOUNT_VOLATILITY_EXPOSURE_FROZEN', source=cfg['rules'])
    write_json(p, i)
    print('第181轮单一风险预算已冻结。', flush=True)


def run():
    result = run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, account_volatility_frames)
    rows = []
    for period, key in [('evaluation', 'all_metrics'), ('earlier_diagnostic', 'earlier_diagnostics')]:
        for row in result[key]:
            if row['model'] == PRIMARY:
                rows.append({'period': period, **row, 'joint_point_pass': row['net_sharpe'] is not None and
                             row['net_sharpe'] >= 1.2 and row['annualized_return'] >= .10})
    pd.DataFrame(rows).to_csv(OUT/'joint_target_metrics.csv', index=False, encoding='utf-8-sig')
    assessment = {'recorded_at': now(), 'goal_achieved': False, 'independent_validation': 'NOT_ESTABLISHED',
                  'main_two_cost_joint_pass': all(x['joint_point_pass'] for x in rows if x['period']=='evaluation'),
                  'earlier_two_cost_joint_pass': all(x['joint_point_pass'] for x in rows if x['period']=='earlier_diagnostic'),
                  'four_scenario_joint_pass': all(x['joint_point_pass'] for x in rows)}
    write_json(OUT/'joint_target_assessment.json', assessment, exclusive=True)
    print(json.dumps(assessment, ensure_ascii=False), flush=True)


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(), '第181轮已经核对')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path']) == item['sha256'], '冻结来源改变')
    data = pd.read_parquet(ROOT/cfg['features'])
    dividends = normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    checks = []

    def expected(period, cost_id, frame, start):
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        parent = pd.read_parquet(SOURCE/period/cost_id/f'{PARENT}_decisions.parquet')
        saved = pd.read_parquet(SOURCE/period/cost_id/f'{PARENT}_ledger.parquet')
        np.testing.assert_array_equal(parent.origin_index, indices)
        require(pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), '来源执行错位')
        require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])), '来源收益日期错位')
        risk = np.full(len(frame), np.nan)
        multiplier = np.full(len(frame), np.nan)
        target = np.full(len(frame), np.nan)
        returns = np.r_[np.full(first, np.nan), saved.net_return.to_numpy(float)]
        for t in range(len(frame)):
            if t < first+59:
                multiplier[t] = 1.
            else:
                sample = returns[t-59:t+1]
                if np.isfinite(sample).all():
                    mean = math.fsum(sample)/60
                    risk[t] = math.sqrt(math.fsum((value-mean)**2 for value in sample)/59*cfg['annual_days'])
                    multiplier[t] = .10/risk[t] if risk[t] > 0 else 1.
        for t, value in zip(indices, parent.reference_weight):
            if value == 0:
                target[t] = 0.
            elif np.isfinite(value) and np.isfinite(multiplier[t]):
                target[t] = min(1., value*multiplier[t])
        folder = OUT/period/cost_id
        factors = pd.read_parquet(folder/'factors.parquet')
        np.testing.assert_allclose(factors.source_realized_net_return, returns, atol=0, rtol=0, equal_nan=True)
        np.testing.assert_allclose(factors.source_account_volatility60, risk, atol=1e-12, rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(factors.account_risk_multiplier, multiplier, atol=1e-10, rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(factors[PRIMARY+'_target'], target, atol=1e-12, rtol=1e-12, equal_nan=True)
        ledger = pd.read_parquet(folder/f'{PRIMARY}_ledger.parquet')
        require(ledger.cash.ge(-1e-7).all() and ledger.shares.ge(0).all(), '无融资账户出现负现金或份额')
        np.testing.assert_allclose(ledger.open, frame.open.iloc[first:], atol=0, rtol=0)
        np.testing.assert_allclose(ledger.mark.iloc[:-1], frame.close.iloc[first:-1], atol=0, rtol=0)
        filled = ledger.filled_quantity.ne(0)
        qty = ledger.loc[filled, 'filled_quantity'].to_numpy()
        op = ledger.loc[filled, 'open'].to_numpy()
        cost = cfg['costs'][cost_id]
        units = op*(1+np.sign(qty)*cost['slippage'])/cfg['tick']
        prices = np.where(qty>0, np.ceil(units-1e-10), np.floor(units+1e-10))*cfg['tick']
        np.testing.assert_allclose(ledger.loc[filled, 'fill_price'], prices, atol=1e-12, rtol=0)
        np.testing.assert_allclose(ledger.loc[filled, 'commission'], np.maximum(abs(qty)*prices*cost['commission'], cost['minimum']), atol=1e-8, rtol=0)
        np.testing.assert_allclose(ledger.loc[filled, 'slippage_cost'], abs(qty)*abs(prices-op), atol=1e-8, rtol=0)
        checks.append({'period': period, 'cost': cost_id, 'decisions': len(indices), 'fills': int(filled.sum()),
                       'maximum_target_error': float(np.nanmax(abs(factors[PRIMARY+'_target'].to_numpy()-target)))})
        return target

    accounts, cycles, differences, count = verify_saved_target_accounts(OUT, cfg, result, data, dividends, expected, comparison_models=list(CONTROLS))
    require(len(accounts)==4 and count==5646, '第181轮核对范围不同')
    pd.DataFrame(checks).to_csv(OUT/'saved_account_volatility_checks.csv', index=False, encoding='utf-8-sig')
    receipt = {'verified_at': now(), 'status': 'PASS_FOUR_SIMULATED_ACCOUNTS_CAUSAL_RISK_BUDGET_AND_COSTS',
               'actual_accounts': 4, 'actual_decisions_checked': count, 'complete_actual_cycles': len(cycles),
               'simulated_fills_checked': sum(x['fills'] for x in checks),
               'new_models_or_accounts': 0, 'independent_performance_validation': False,
               'security_audit_performed': False, 'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'verify': verify}[sys.argv[1]]()
