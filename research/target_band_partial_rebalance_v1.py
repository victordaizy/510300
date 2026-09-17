"""两种部分调仓一次冻结计算，来源信号保持不变。"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.event_account_request_callback_v1 import simulate_request_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_requested_account_checks_v1 import verify_requested_target_accounts
from research.saved_target_custom_execution_v1 import run_custom_execution_batch
from research.target_band_partial_rebalance_inputs_v1 import CANDIDATES, MODELS, PARENT, PRIMARY, partial_frames, partial_request
from scripts.verify_round195_20260913 import prices_and_costs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_target_band_partial_rebalance_v1'
CONFIG = ROOT/'config/510300_target_band_partial_rebalance_v1.json'
SOURCE = ROOT/'reports/research/510300_account_volatility_exposure_v1'
PARENTS = {PARENT: SOURCE}
CONTROLS = {PARENT: (SOURCE, '原181调整到目标中心'),
    'ENTRY_SHARES_PRESERVATION': (ROOT/'reports/research/510300_entry_shares_preservation_v1', '原193正目标完全保持份额'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def simulate_partial(*args, **kwargs):
    return simulate_request_account(*args, **kwargs, request_policy=partial_request)


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '本轮已有登记或运行')
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    test = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_target_band_partial_rebalance_v1.py', '-q', '-p', 'no:cacheprovider'],
                          cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
    print(test.stdout, flush=True)
    require(test.returncode == 0 and '14 passed' in test.stdout, '本轮十四项必要测试未通过')
    write_json(OUT/'tests_receipt.json', {'recorded_at': now(), 'exit_code': test.returncode, 'passed': 14,
        'seconds': time.perf_counter()-began}, exclusive=True)
    old_path = ROOT/'config/510300_account_volatility_exposure_v1.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {key: old[key] for key in keys}
    cfg.update(study_id='510300_TARGET_BAND_PARTIAL_REBALANCE_V1', round=197, registered_at=now(),
        primary=PRIMARY, candidate_models=list(CANDIDATES), candidate_configurations=2, parent_models=MODELS,
        decision_clock='15:05:00', annual_return_target=.10, half_gap_fraction=.5, new_model_fits=0,
        new_reference_accounts=0, goal_achieved=False, position_impact=0, source_budget_cny=0,
        independent_validation='NOT_ESTABLISHED', evidence_class='RETROSPECTIVE_FINITE_EXECUTION_RULE_COMPARISON',
        rules='docs/510300_TARGET_BAND_PARTIAL_REBALANCE_V1.md')
    with (ROOT/cfg['rules']).open('x', encoding='utf-8') as stream:
        stream.write((ROOT/'docs/510300_TARGET_BAND_PARTIAL_REBALANCE_NEXT_20260913.md').read_text(encoding='utf-8').replace(
            '尚未实现、测试、冻结或计算本轮账户。', '本轮在必要测试通过后、首次账户计算前冻结。'))
        stream.write('\n\n## 原181来源的全部中文因素与进出场规则\n\n')
        stream.write((ROOT/old['rules']).read_text(encoding='utf-8'))
    paths = [Path(__file__), ROOT/'research/target_band_partial_rebalance_inputs_v1.py',
        ROOT/'research/event_account_request_callback_v1.py', ROOT/'research/event_clock_account_v1.py',
        ROOT/'research/saved_requested_account_checks_v1.py', ROOT/'research/saved_target_account_checks_v1.py',
        ROOT/'research/saved_target_custom_execution_v1.py', ROOT/'research/saved_parent_target_alignment_v1.py',
        ROOT/'research/adaptive_allocation_v1.py', ROOT/'research/intraday_overnight_increment_v1.py',
        ROOT/'research/joint_account_acceptance_v1.py', ROOT/'scripts/verify_round195_20260913.py',
        ROOT/'scripts/build_requested_account_helpers_20260913.py', ROOT/'tests/test_target_band_partial_rebalance_v1.py',
        old_path, ROOT/old['rules'], OUT/'tests_output.txt', OUT/'tests_receipt.json', SOURCE/'saved_verification_receipt.json']
    paths += [ROOT/cfg[key] for key in ['features', 'dividends', 'rules']]
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost_id in cfg['costs']:
            paths.append(SOURCE/period/cost_id/f'{PARENT}_decisions.parquet')
            paths += [folder/period/cost_id/f'{model}_ledger.parquet' for model, (folder, _) in CONTROLS.items()]
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 196 and not index['running_studies'], '前序或运行状态不同')
    index['running_studies'] = [{'round': 197, 'study': cfg['study_id'], 'status': 'FROZEN_NOT_STARTED', 'config': str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True, status='TARGET_BAND_PARTIAL_REBALANCE_FROZEN', source=cfg['rules'])
    write_json(path, index)
    print('第197轮两种部分调仓规则已冻结，共八条新账户。', flush=True)


def run():
    result = run_custom_execution_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, partial_frames, simulate_partial)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    print(json.dumps(save_joint_assessment(OUT, cfg, result), ensure_ascii=False), flush=True)


def independently_requested(targets, prior, old_shares, prices, cfg):
    actual = old_shares*prices/prior
    known = np.isfinite(targets)
    held_positive = known & (targets > 0) & (old_shares > 0)
    keep = held_positive & (np.abs(targets-actual) < .1)
    adjust = held_positive & ~keep
    applied = targets.copy()
    applied[keep] = actual[keep]
    if cfg['primary'] == 'TARGET_BAND_EDGE':
        applied[adjust] = targets[adjust] - np.sign(targets[adjust]-actual[adjust])*.1
    elif cfg['primary'] == 'TARGET_HALF_GAP':
        applied[adjust] = (actual[adjust]+targets[adjust])/2
    else:
        raise ValueError('独立申请核对遇到未登记方案')
    applied = np.clip(applied, 0., 1.)
    desired = old_shares.copy()
    recompute = known & ~keep
    desired[recompute] = (np.floor(applied[recompute]*prior[recompute]/prices[recompute]/cfg['lot'])*cfg['lot']).astype(int)
    return desired-old_shares


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(), '本轮保存核对已经完成')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    require(cfg['half_gap_fraction'] == .5, '一半差额设置不同')
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path']) == item['sha256'], '冻结输入改变')
    data = pd.read_parquet(ROOT/cfg['features'])
    dividends = normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    checks = []

    def expected(period, cost_id, frame, start, model):
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        source = pd.read_parquet(SOURCE/period/cost_id/f'{PARENT}_decisions.parquet')
        np.testing.assert_array_equal(source.origin_index, indices)
        require(pd.DatetimeIndex(source.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), '来源收盘日不同')
        require(pd.DatetimeIndex(source.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), '来源不是下一开盘')
        target = np.full(len(frame), np.nan)
        target[indices] = source.reference_weight
        folder = OUT/period/cost_id
        factors = pd.read_parquet(folder/'factors.parquet')
        np.testing.assert_allclose(factors[model+'_target'], target, atol=0, rtol=0, equal_nan=True)
        ledger = pd.read_parquet(folder/f'{model}_ledger.parquet')
        decisions = pd.read_parquet(folder/f'{model}_decisions.parquet')
        require(pd.DatetimeIndex(decisions.decision_time).equals(pd.DatetimeIndex(decisions.origin)+pd.Timedelta(hours=15, minutes=5)), '收盘时钟不同')
        prior = np.r_[cfg['initial_capital'], ledger.equity.iloc[:-1]]
        shares = np.r_[0, ledger.shares.iloc[:-1]].astype(int)
        prices = frame.close.iloc[indices].to_numpy(float)
        values, actual = target[indices], shares*prices/prior
        known = np.isfinite(values)
        np.testing.assert_allclose(decisions.loc[known, 'own_close_equity'], prior[known], atol=1e-6, rtol=0)
        np.testing.assert_allclose(decisions.loc[known, 'own_close_exposure'], actual[known], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(decisions.loc[known, 'own_close_shares'], shares[known])
        applying = values.copy()
        held = known & (values > 0) & (shares > 0)
        keeping = held & (np.abs(values-actual) < .1)
        moving = held & ~keeping
        applying[keeping] = actual[keeping]
        if model == 'TARGET_BAND_EDGE':
            applying[moving] = values[moving]-np.sign(values[moving]-actual[moving])*.1
        else:
            applying[moving] = (values[moving]+actual[moving])/2
        np.testing.assert_allclose(decisions.application_weight, np.clip(applying, 0., 1.), atol=1e-12, rtol=0, equal_nan=True)
        for control, (old, _) in CONTROLS.items():
            pd.testing.assert_frame_equal(pd.read_parquet(old/period/cost_id/f'{control}_ledger.parquet'),
                                          pd.read_parquet(folder/f'{control}_ledger.parquet'))
        checks.append({'model': model, 'period': period, 'cost': cost_id, 'decisions': len(indices),
            'fills': prices_and_costs(ledger, frame, first, cfg, cfg['costs'][cost_id]),
            'positive_adjustment_origins': int(moving.sum()), 'within_band_origins': int(keeping.sum())})
        return target

    accounts, cycles, differences, count = [], [], [], 0
    for model in CANDIDATES:
        def selected(period, cost_id, frame, start):
            return expected(period, cost_id, frame, start, model)
        a, c, d, n = verify_requested_target_accounts(OUT, {**cfg, 'primary': model}, result, data, dividends,
            selected, independently_requested, comparison_models=list(CONTROLS))
        for destination, records in [(accounts, a), (cycles, c), (differences, d)]:
            destination.extend({'model': model, **row} for row in records)
        count += n
    for name, rows in [('saved_account_checks.csv', accounts), ('saved_actual_cycles.csv', cycles),
        ('saved_comparison_differences.csv', differences), ('saved_target_checks.csv', checks)]:
        pd.DataFrame(rows).to_csv(OUT/name, index=False, encoding='utf-8-sig')
    require(len(accounts) == 8 and count == 11292, '八条新账户核对范围不同')
    receipt = {'verified_at': now(), 'status': 'PASS_EIGHT_PARTIAL_REBALANCE_ACCOUNTS_OWN_NAV_REQUESTS_AND_COSTS',
        'actual_accounts': 8, 'actual_decisions_checked': count, 'complete_actual_cycles': len(cycles),
        'simulated_fills_checked': sum(row['fills'] for row in checks), 'new_models_or_accounts': 0,
        'independent_performance_validation': False, 'security_audit_performed': False,
        'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'verify': verify}[sys.argv[1]]()
