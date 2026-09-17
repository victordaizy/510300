"""一次检验原目标明确零的两次收盘确认，保留完整账户与时间顺序。"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_requested_account_checks_v1 import verify_requested_target_accounts
from research.saved_target_custom_execution_v1 import run_custom_execution_batch
from research.two_close_zero_exit_account_v1 import simulate_confirmed_zero
from research.two_close_zero_exit_inputs_v1 import CANDIDATES, MODELS, PARENT, PRIMARY, confirmation_frames
from scripts.verify_round195_20260913 import prices_and_costs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_two_close_zero_exit_v1'
CONFIG = ROOT/'config/510300_two_close_zero_exit_v1.json'
SOURCE = ROOT/'reports/research/510300_account_volatility_exposure_v1'
PARENTS = {PARENT: SOURCE}
CONTROLS = {PARENT: (SOURCE, '原181立即执行明确零退出'),
    'ENTRY_SHARES_PRESERVATION': (ROOT/'reports/research/510300_entry_shares_preservation_v1', '原193正目标保持份额'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '本轮已经登记或运行')
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    test = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_two_close_zero_exit_v1.py', '-q', '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
    print(test.stdout, flush=True)
    require(test.returncode == 0 and '7 passed' in test.stdout, '七项必要测试未通过')
    write_json(OUT/'tests_receipt.json', {'recorded_at': now(), 'exit_code': test.returncode, 'passed': 7,
        'seconds': time.perf_counter()-began}, exclusive=True)
    old_path = ROOT/'config/510300_account_volatility_exposure_v1.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {key: old[key] for key in keys}
    cfg.update(study_id='510300_TWO_CLOSE_ZERO_EXIT_V1', round=198, registered_at=now(), primary=PRIMARY,
        candidate_models=list(CANDIDATES), candidate_configurations=1, parent_models=MODELS, zero_confirmations=2,
        decision_clock='15:05:00', annual_return_target=.10, new_model_fits=0, new_reference_accounts=0,
        source_budget_cny=0, goal_achieved=False, position_impact=0, independent_validation='NOT_ESTABLISHED',
        evidence_class='RETROSPECTIVE_EXIT_TIMING_SELECTED_AFTER_ONE_AND_TWO_DAY_SAVED_PRICE_DIAGNOSTIC',
        rules='docs/510300_TWO_CLOSE_ZERO_EXIT_V1.md',
        source_preflight='reports/research/510300_saved_zero_target_exit_preflight_20260913/result.json')
    with (ROOT/cfg['rules']).open('x', encoding='utf-8') as stream:
        stream.write((ROOT/'docs/510300_TWO_CLOSE_ZERO_EXIT_NEXT_20260913.md').read_text(encoding='utf-8').replace(
            '当前只准备规则，尚未实现、测试、冻结或计算新账户。', '当前在必要测试通过后、首次完整账户计算前冻结规则。'))
        stream.write('\n\n## 原181来源全部中文因子与进出场\n\n')
        stream.write((ROOT/old['rules']).read_text(encoding='utf-8'))
    paths = [Path(__file__), ROOT/'research/two_close_zero_exit_inputs_v1.py', ROOT/'research/two_close_zero_exit_account_v1.py',
        ROOT/'research/event_account_indexed_request_v1.py', ROOT/'research/event_account_request_callback_v1.py',
        ROOT/'research/event_clock_account_v1.py', ROOT/'research/saved_requested_account_checks_v1.py',
        ROOT/'research/saved_target_custom_execution_v1.py', ROOT/'research/saved_parent_target_alignment_v1.py',
        ROOT/'research/adaptive_allocation_v1.py', ROOT/'research/intraday_overnight_increment_v1.py',
        ROOT/'research/joint_account_acceptance_v1.py', ROOT/'scripts/verify_round195_20260913.py',
        ROOT/'scripts/build_indexed_request_account_20260913.py', ROOT/'tests/test_two_close_zero_exit_v1.py',
        ROOT/'tests/test_target_band_partial_rebalance_v1.py', old_path, ROOT/old['rules'],
        OUT/'tests_output.txt', OUT/'tests_receipt.json', SOURCE/'saved_verification_receipt.json']
    paths += [ROOT/cfg[key] for key in ['features', 'dividends', 'rules', 'source_preflight']]
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in cfg['costs']:
            paths.append(SOURCE/period/cost/f'{PARENT}_decisions.parquet')
            paths += [folder/period/cost/f'{model}_ledger.parquet' for model, (folder, _) in CONTROLS.items()]
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 197 and not index['running_studies'], '前序状态不同')
    index['running_studies'] = [{'round': 198, 'study': cfg['study_id'], 'status': 'FROZEN_NOT_STARTED', 'config': str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True, status='TWO_CLOSE_ZERO_EXIT_FROZEN', source=cfg['rules'])
    write_json(path, index)
    print('第198轮两个收盘明确零确认已冻结，四条新账户。', flush=True)


def run():
    result = run_custom_execution_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, confirmation_frames, simulate_confirmed_zero)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    print(json.dumps(save_joint_assessment(OUT, cfg, result), ensure_ascii=False), flush=True)


def independent_counts(values):
    counts = []
    last_nonzero = -1
    for j, value in enumerate(values):
        if not np.isfinite(value) or value != 0:
            last_nonzero = j
            counts.append(0)
        else:
            counts.append(j-last_nonzero)
    return np.array(counts, dtype=int)


def expected_requests(targets, prior, old_shares, prices, cfg):
    known = np.isfinite(targets)
    actual = old_shares*prices/prior
    desired = old_shares.copy()
    desired[known] = (np.floor(targets[known]*prior[known]/prices[known]/cfg['lot'])*cfg['lot']).astype(int)
    keeping = known & (targets > 0) & (old_shares > 0) & (np.abs(targets-actual) < .1)
    waiting = (targets == 0) & (independent_counts(targets) == 1)
    desired[keeping | waiting] = old_shares[keeping | waiting]
    return desired-old_shares


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(), '本轮核对已经完成')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path']) == item['sha256'], '冻结来源改变')
    data = pd.read_parquet(ROOT/cfg['features'])
    dividends = normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    checks = []

    def expected(period, cost, frame, start):
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        source = pd.read_parquet(SOURCE/period/cost/f'{PARENT}_decisions.parquet')
        np.testing.assert_array_equal(source.origin_index, indices)
        require(pd.DatetimeIndex(source.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), '来源判断日历不同')
        require(pd.DatetimeIndex(source.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), '来源执行日历不同')
        target = np.full(len(frame), np.nan)
        target[indices] = source.reference_weight
        counts = independent_counts(target)
        folder = OUT/period/cost
        factors = pd.read_parquet(folder/'factors.parquet')
        np.testing.assert_array_equal(factors.source_zero_streak, counts)
        np.testing.assert_allclose(factors[PRIMARY+'_target'], target, atol=0, rtol=0, equal_nan=True)
        decisions = pd.read_parquet(folder/f'{PRIMARY}_decisions.parquet')
        ledger = pd.read_parquet(folder/f'{PRIMARY}_ledger.parquet')
        require(pd.DatetimeIndex(decisions.decision_time).equals(pd.DatetimeIndex(decisions.origin)+pd.Timedelta(hours=15, minutes=5)), '决定时钟不同')
        known = np.isfinite(target[indices])
        old_shares = np.r_[0, ledger.shares.iloc[:-1]]
        prior = np.r_[cfg['initial_capital'], ledger.equity.iloc[:-1]]
        np.testing.assert_array_equal(decisions.loc[known, 'used_zero_count'], counts[indices][known])
        np.testing.assert_array_equal(decisions.loc[known, 'own_close_shares'], old_shares[known])
        np.testing.assert_allclose(decisions.loc[known, 'own_close_equity'], prior[known], atol=1e-6, rtol=0)
        waiting = (counts[indices] == 1) & (old_shares > 0)
        np.testing.assert_array_equal(decisions.loc[known, 'waiting_zero_confirmation'], waiting[known].astype(float))
        require(decisions.loc[waiting, 'requested_quantity'].eq(0).all(), '首次明确零没有保留份额')
        for control, (old, _) in CONTROLS.items():
            pd.testing.assert_frame_equal(pd.read_parquet(old/period/cost/f'{control}_ledger.parquet'),
                                          pd.read_parquet(folder/f'{control}_ledger.parquet'))
        checks.append({'period': period, 'cost': cost, 'decisions': len(indices),
            'first_zero_held_origins': int(waiting.sum()), 'confirmed_zero_held_origins': int(((counts[indices] >= 2) & (old_shares > 0)).sum()),
            'fills': prices_and_costs(ledger, frame, first, cfg, cfg['costs'][cost])})
        return target

    accounts, cycles, differences, count = verify_requested_target_accounts(OUT, cfg, result, data, dividends,
        expected, expected_requests, comparison_models=list(CONTROLS))
    pd.DataFrame(checks).to_csv(OUT/'saved_zero_confirmation_checks.csv', index=False, encoding='utf-8-sig')
    require(len(accounts) == 4 and count == 5646, '四账户核对数量不同')
    receipt = {'verified_at': now(), 'status': 'PASS_FOUR_TWO_CLOSE_ZERO_EXIT_ACCOUNTS_AND_ACTUAL_REQUESTS',
        'actual_accounts': 4, 'actual_decisions_checked': count, 'complete_actual_cycles': len(cycles),
        'simulated_fills_checked': sum(r['fills'] for r in checks), 'new_models_or_accounts': 0,
        'independent_performance_validation': False, 'security_audit_performed': False,
        'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'verify': verify}[sys.argv[1]]()
