"""第180轮固定预算试验：准备、一次运行和保存账户复核。"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.saved_target_account_checks_v1 import verify_saved_target_accounts
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.unlevered_exposure_expansion_inputs_v1 import CANDIDATES, MODELS, MULTIPLIERS, PARENT, PRIMARY, expansion_frames

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_unlevered_exposure_expansion_v1'
CONFIG = ROOT / 'config/510300_unlevered_exposure_expansion_v1.json'
P174 = ROOT / 'reports/research/510300_return_confirmation_auxiliary_batch_v1'
P32 = ROOT / 'reports/research/510300_rearmed_session_exit_v1'
PARENTS = {PARENT: P174}
CONTROLS = {PARENT: (P174, '原第174轮任一方向确认'), 'BUY_HOLD': (P32, '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT / 'RUN_STARTED.json').exists(), '第180轮已有冻结或运行，请接续')
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / 'tests_receipt.json'
    require(not receipt_path.exists(), '必要测试已有回执，不重复准备')
    began = time.perf_counter()
    completed = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_unlevered_exposure_expansion_v1.py', '-q', '-p', 'no:cacheprovider'],
                               cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT / 'tests_output.txt').write_text(completed.stdout + completed.stderr, encoding='utf-8')
    print(completed.stdout, flush=True)
    require(completed.returncode == 0 and '4 passed' in completed.stdout, '必要测试未通过')
    write_json(receipt_path, {'recorded_at': now(), 'exit_code': completed.returncode, 'passed': 4,
                             'seconds': time.perf_counter() - began, 'timing_scope': '完整测试进程墙钟'}, exclusive=True)
    old_config = ROOT / 'config/510300_return_confirmation_auxiliary_batch_v1.json'
    source_cfg = json.loads(old_config.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
            'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends',
            'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {key: source_cfg[key] for key in keys}
    cfg.update(study_id='510300_UNLEVERED_EXPOSURE_EXPANSION_V1', round=180, primary=PRIMARY,
               candidate_models=list(CANDIDATES), candidate_configurations=2, multipliers=MULTIPLIERS,
               registered_at=now(), decision_clock='15:05:00', parent_models=MODELS, annual_return_target=.10,
               new_model_fits=0, new_reference_accounts=0, source_budget_cny=0, position_impact=0,
               goal_achieved=False, independent_validation='NOT_ESTABLISHED',
               rules='docs/510300_UNLEVERED_EXPOSURE_EXPANSION_V1.md',
               previous_goal_turn_classification='PROGRESS_ROUND179_COMPLETED_NEW_DUAL_TARGET_REGISTERED',
               evidence_class='RETROSPECTIVE_REAL_ACCOUNT_REPLAY_PREVIOUSLY_OBSERVED_HISTORY',
               joint_gate='NET_SHARPE_GE_1_2_AND_COMPOUND_CAGR_GE_10_PERCENT_EACH_COST',
               legacy_meets_point_target_field='SHARPE_ONLY_USE_JOINT_TARGET_ASSESSMENT_FOR_CURRENT_GOAL')
    rules = ROOT / cfg['rules']
    with rules.open('a', encoding='utf-8') as stream:
        stream.write('\n\n' + (ROOT / source_cfg['rules']).read_text(encoding='utf-8'))
    paths = [Path(__file__), ROOT / 'research/unlevered_exposure_expansion_inputs_v1.py',
             ROOT / 'research/saved_target_batch_runner_v1.py', ROOT / 'research/saved_parent_target_alignment_v1.py',
             ROOT / 'research/saved_target_account_checks_v1.py', ROOT / 'research/event_clock_account_v1.py',
             ROOT / 'research/adaptive_allocation_v1.py', ROOT / 'research/intraday_overnight_increment_v1.py',
             ROOT / 'tests/test_unlevered_exposure_expansion_v1.py', receipt_path, OUT / 'tests_output.txt',
             ROOT / cfg['features'], ROOT / cfg['dividends'], rules, old_config, ROOT / source_cfg['rules'],
             P174 / 'saved_verification_receipt.json', ROOT / 'config/510300_research_authority_v6.json']
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in cfg['costs']:
            paths.append(P174 / period / cost / f'{PARENT}_decisions.parquet')
            for model, (folder, _) in CONTROLS.items():
                paths.append(folder / period / cost / f'{model}_ledger.parquet')
    cfg['frozen_files'] = [{'path': str(path.relative_to(ROOT)), 'sha256': digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    index_path = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 179, '第180轮前序不是179')
    index['running_studies'] = [{'round': 180, 'study': cfg['study_id'], 'status': 'FROZEN_NOT_STARTED',
                                'config': str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(candidate_round=180, registered=True, planned_settings=2, planned_new_accounts=8,
                              status='UNLEVERED_EXPOSURE_EXPANSION_FROZEN', source=cfg['rules'],
                              focus='新增年化10%目标下，一倍半与两倍股票目标资金预算，最高百分之一百')
    write_json(index_path, index)
    print('第180轮两档资金预算已冻结。', flush=True)


def run():
    result = run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, expansion_frames)
    rows = []
    for period, key in [('evaluation', 'all_metrics'), ('earlier_diagnostic', 'earlier_diagnostics')]:
        for row in result[key]:
            if row['model'] in CANDIDATES:
                rows.append({'period': period, **row, 'joint_point_pass': row['net_sharpe'] is not None and
                             row['net_sharpe'] >= 1.2 and row['annualized_return'] >= .10})
    pd.DataFrame(rows).to_csv(OUT / 'joint_target_metrics.csv', index=False, encoding='utf-8-sig')
    assessment = {'recorded_at': now(), 'definition': '成本后净夏普至少1.2且复合年化至少10%，两档费用分别核对',
                  'goal_achieved': False, 'independent_validation': 'NOT_ESTABLISHED', 'candidates': {}}
    for model in CANDIDATES:
        groups = {period: [r for r in rows if r['model'] == model and r['period'] == period]
                  for period in ['evaluation', 'earlier_diagnostic']}
        assessment['candidates'][model] = {
            'main_two_cost_joint_pass': all(r['joint_point_pass'] for r in groups['evaluation']),
            'earlier_two_cost_joint_pass': all(r['joint_point_pass'] for r in groups['earlier_diagnostic']),
            'four_scenario_joint_pass': all(r['joint_point_pass'] for group in groups.values() for r in group),
        }
    write_json(OUT / 'joint_target_assessment.json', assessment, exclusive=True)
    print(json.dumps(assessment, ensure_ascii=False), flush=True)


def verify():
    require(not (OUT / 'saved_verification_receipt.json').exists(), '第180轮已经核对')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    result = json.loads((OUT / 'result.json').read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT / item['path']) == item['sha256'], '冻结文件发生改变')
    data = pd.read_parquet(ROOT / cfg['features'])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg['dividends']))
    checks, accounts, cycles, differences = [], [], [], []
    decision_count = 0

    def expected(period, cost_id, frame, start):
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        origins = np.arange(first - 1, len(frame) - 1)
        parent = pd.read_parquet(P174 / period / cost_id / f'{PARENT}_decisions.parquet')
        np.testing.assert_array_equal(parent.origin_index, origins)
        require(pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[origins])), '来源收盘错位')
        require(pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[origins + 1])), '来源执行错位')
        target = np.full(len(frame), np.nan)
        for origin, weight in zip(origins, parent.reference_weight):
            if np.isfinite(weight):
                target[origin] = min(1., weight * cfg['multipliers'][active_model])
        folder = OUT / period / cost_id
        factors = pd.read_parquet(folder / 'factors.parquet')
        np.testing.assert_allclose(factors[active_model + '_target'], target, atol=0, rtol=0, equal_nan=True)
        require(pd.DatetimeIndex(factors.decision_time.iloc[origins]).equals(
            pd.DatetimeIndex(frame.date.iloc[origins]) + pd.Timedelta(hours=15, minutes=5)), '收盘判断时间改变')
        ledger = pd.read_parquet(folder / f'{active_model}_ledger.parquet')
        np.testing.assert_allclose(ledger.open, frame.open.iloc[first:], atol=0, rtol=0)
        np.testing.assert_allclose(ledger.mark.iloc[:-1], frame.close.iloc[first:-1], atol=0, rtol=0)
        require(ledger.cash.ge(-1e-7).all() and ledger.shares.ge(0).all(), '账户出现融资或负份额')
        require((ledger.shares % cfg['lot'] == 0).all(), '实际份额违反整手')
        filled = ledger.filled_quantity.ne(0)
        quantities, opens = ledger.loc[filled, 'filled_quantity'].to_numpy(), ledger.loc[filled, 'open'].to_numpy()
        cost = cfg['costs'][cost_id]
        tick = cfg['tick']
        units = opens * (1 + np.sign(quantities) * cost['slippage']) / tick
        prices = np.where(quantities > 0, np.ceil(units - 1e-10), np.floor(units + 1e-10)) * tick
        np.testing.assert_allclose(ledger.loc[filled, 'fill_price'], prices, atol=1e-12, rtol=0)
        np.testing.assert_allclose(ledger.loc[filled, 'commission'], np.maximum(abs(quantities) * prices * cost['commission'], cost['minimum']), atol=1e-8, rtol=0)
        np.testing.assert_allclose(ledger.loc[filled, 'slippage_cost'], abs(quantities) * abs(prices - opens), atol=1e-8, rtol=0)
        for model, (parent_folder, _) in CONTROLS.items():
            pd.testing.assert_frame_equal(pd.read_parquet(folder / f'{model}_ledger.parquet'),
                                          pd.read_parquet(parent_folder / period / cost_id / f'{model}_ledger.parquet'))
        checks.append({'period': period, 'cost': cost_id, 'model': active_model, 'decisions': len(origins),
                       'fills': int(filled.sum()), 'target_max_error': 0.0, 'cash_nonnegative': True})
        return target

    for active_model in CANDIDATES:
        aa, cc, dd, count = verify_saved_target_accounts(OUT, {**cfg, 'primary': active_model}, result,
                                                       data, dividends, expected, comparison_models=list(CONTROLS))
        accounts.extend({'model': active_model, **r} for r in aa)
        cycles.extend({'model': active_model, **r} for r in cc)
        differences.extend({'model': active_model, **r} for r in dd)
        decision_count += count
    require(len(accounts) == 8 and decision_count == 11292, '账户覆盖范围不同')
    for name, rows in [('saved_account_checks.csv', accounts), ('saved_actual_cycles.csv', cycles),
                       ('saved_comparison_differences.csv', differences), ('saved_expansion_checks.csv', checks)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')
    receipt = {'verified_at': now(), 'status': 'PASS_EIGHT_SIMULATED_ACCOUNTS_TARGETS_AND_EXECUTION_COSTS',
               'actual_accounts': len(accounts), 'actual_decisions_checked': decision_count,
               'complete_actual_cycles': len(cycles), 'simulated_fills_checked': sum(r['fills'] for r in checks),
               'new_models_or_accounts': 0, 'independent_performance_validation': False,
               'security_audit_performed': False, 'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT / 'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'verify': verify}[sys.argv[1]]()
