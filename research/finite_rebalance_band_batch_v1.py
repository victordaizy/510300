"""八套已定调仓门槛集中计算与核对，复用全部旧目标和对照。"""
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.finite_rebalance_band_batch_inputs_v1 import CANDIDATES, MODELS, PRIMARY, SETTINGS, SOURCES, band_frames, simulate_band_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_requested_account_checks_v1 import verify_requested_target_accounts
from research.saved_target_custom_execution_v1 import run_custom_execution_batch
from research.two_close_zero_exit_v1 import independent_counts
from scripts.verify_round195_20260913 import prices_and_costs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_finite_rebalance_band_batch_v1'
CONFIG = ROOT/'config/510300_finite_rebalance_band_batch_v1.json'
PARENTS = {m: ROOT/'reports/research'/f for m, f in SOURCES.items()}
CONTROLS = {
    MODELS[0]: (PARENTS[MODELS[0]], '原198两次归零确认、十个百分点'),
    MODELS[1]: (PARENTS[MODELS[1]], '原200简单三来源合并、十个百分点'),
    'ACCOUNT_VOLATILITY_EXPOSURE': (ROOT/'reports/research/510300_account_volatility_exposure_v1', '原181完整账户波动预算'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '本批已经登记或运行')
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    test = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_finite_rebalance_band_batch_v1.py', '-q', '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
    print(test.stdout, flush=True)
    require(test.returncode == 0 and '6 passed' in test.stdout, '六项调仓门槛测试未通过')
    write_json(OUT/'tests_receipt.json', {'recorded_at': now(), 'exit_code': test.returncode, 'passed': 6,
        'seconds': time.perf_counter()-began}, exclusive=True)
    old_path = ROOT/'config/510300_three_source_order_intent_mix_v1.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {key: old[key] for key in keys}
    cfg.update(study_id='510300_FINITE_REBALANCE_BAND_BATCH_V1', round=201, registered_at=now(),
        primary=PRIMARY, candidate_models=list(CANDIDATES), candidate_configurations=8,
        candidate_settings=SETTINGS, parent_models=MODELS, source_folders=SOURCES,
        decision_clock='15:05:00', annual_return_target=.10, new_model_fits=0, new_reference_accounts=0,
        source_budget_cny=0, goal_achieved=False, position_impact=0, independent_validation='NOT_ESTABLISHED',
        evidence_class='RETROSPECTIVE_FINITE_REBALANCE_BAND_CALIBRATION',
        prior_goal_turn_classification='PROGRESS_ROUND200_COMPLETED_EIGHT_ACCOUNTS_AND_NEW_EVIDENCE',
        rules='docs/510300_FINITE_REBALANCE_BAND_BATCH_V1.md')
    text = (ROOT/'docs/510300_FINITE_REBALANCE_BAND_BATCH_NEXT_20260913.md').read_text(encoding='utf-8')
    text = text.replace('本轮目前仅准备规则，尚未实现、测试、冻结或计算新账户。', '本批在六项必要测试通过后、首次新账户计算前冻结规则。')
    text += '\n\n## 八套实际冻结设置\n\n|方案|调仓门槛|外层明确零退出|\n|---|---:|---|\n'
    for model, setting in SETTINGS.items():
        text += f"|{CANDIDATES[model]}|{setting['band']:.0%}|{'连续两次确认' if setting['zero_confirmations'] == 2 else '立即申请退出'}|\n"
    text += '\n\n## 全部中文来源因素与原进出场规则\n\n下文为第200轮及其内部来源规则，完整保留以说明目标如何产生。本批最外层正目标调仓门槛按上表替代，内部参考账户仍沿用原门槛；下文旧轮次状态不代表本轮运行状态。\n\n'
    text += (ROOT/old['rules']).read_text(encoding='utf-8')
    with (ROOT/cfg['rules']).open('x', encoding='utf-8') as stream:
        stream.write(text)
    paths = [Path(__file__), ROOT/'research/finite_rebalance_band_batch_inputs_v1.py',
        ROOT/'tests/test_finite_rebalance_band_batch_v1.py', ROOT/'tests/test_two_close_zero_exit_v1.py',
        ROOT/'tests/test_target_band_partial_rebalance_v1.py', ROOT/'research/two_close_zero_exit_inputs_v1.py',
        ROOT/'research/two_close_zero_exit_v1.py', ROOT/'research/event_account_indexed_request_v1.py',
        ROOT/'research/saved_requested_account_checks_v1.py', ROOT/'research/saved_target_custom_execution_v1.py',
        ROOT/'research/saved_parent_target_alignment_v1.py', ROOT/'research/adaptive_allocation_v1.py',
        ROOT/'research/intraday_overnight_increment_v1.py', ROOT/'research/joint_account_acceptance_v1.py',
        ROOT/'scripts/verify_round195_20260913.py', old_path, ROOT/old['rules'], OUT/'tests_output.txt', OUT/'tests_receipt.json']
    paths += [ROOT/cfg[k] for k in ['features', 'dividends', 'rules']]
    for model, folder in PARENTS.items():
        paths += [folder/'saved_verification_receipt.json', ROOT/'config'/f'{SOURCES[model]}.json']
        paths += [folder/p/c/f'{model}_decisions.parquet' for p in ['evaluation', 'earlier_diagnostic'] for c in cfg['costs']]
    for model, (folder, _) in CONTROLS.items():
        paths += [folder/p/c/f'{model}_ledger.parquet' for p in ['evaluation', 'earlier_diagnostic'] for c in cfg['costs']]
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(paths))]
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 200 and not index['running_studies'], '前序完成状态不同')
    write_json(CONFIG, cfg, exclusive=True)
    index['running_studies'] = [{'round': 201, 'study': cfg['study_id'], 'status': 'FROZEN_NOT_STARTED', 'config': str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True, status='FINITE_REBALANCE_BAND_BATCH_FROZEN', source=cfg['rules'])
    write_json(path, index)
    print('第201轮八套门槛已冻结，计划三十二条完整账户。', flush=True)


def run():
    with (OUT/'run_output.txt').open('x', encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
        result = run_custom_execution_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, band_frames, simulate_band_account)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    joint = save_joint_assessment(OUT, cfg, result)
    ranking = []
    for model in [*CANDIDATES, *MODELS]:
        row, ratios = {'model': model, 'reused_ten_point_band': model in MODELS}, []
        for period, key in [('main', 'all_metrics'), ('earlier', 'earlier_diagnostics')]:
            for cost in ['BASE', 'STRESS']:
                metric = next(r for r in result[key] if r['model'] == model and r['cost'] == cost)
                row['name'] = metric['name']
                row.update({period+'_'+cost.lower()+'_'+k: metric[k] for k in ['net_sharpe', 'annualized_return', 'max_drawdown', 'trade_count']})
                ratios.extend([metric['net_sharpe']/1.2, metric['annualized_return']/.1])
        row['minimum_joint_ratio'] = min(ratios)
        ranking.append(row)
    ranked = pd.DataFrame(ranking).sort_values('minimum_joint_ratio', ascending=False)
    ranked.to_csv(OUT/'ten_setting_joint_comparison.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({'核心计算秒数': result['run_seconds'], '联合验收': joint['candidates'],
        '全批排名': ranked.to_dict('records')}, ensure_ascii=False), flush=True)


def expected_requests(targets, prior, old_shares, prices, cfg):
    setting = cfg['candidate_settings'][cfg['primary']]
    known = np.isfinite(targets)
    desired = old_shares.copy()
    desired[known] = (np.floor(targets[known]*prior[known]/prices[known]/cfg['lot'])*cfg['lot']).astype(int)
    within = known & (targets > 0) & (old_shares > 0) & (np.abs(targets-old_shares*prices/prior) < setting['band'])
    waiting = (targets == 0) & (independent_counts(targets) < setting['zero_confirmations'])
    desired[within | waiting] = old_shares[within | waiting]
    return desired-old_shares


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(), '本批核对已经完成')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path']) == item['sha256'], '冻结来源改变')
    data = pd.read_parquet(ROOT/cfg['features'])
    dividends = normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    cache, checks = {}, []

    def expected(period, cost, frame, start, model):
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        folder = OUT/period/cost
        if (period, cost) not in cache:
            targets = {}
            for parent, source_folder in PARENTS.items():
                source = pd.read_parquet(source_folder/period/cost/f'{parent}_decisions.parquet')
                np.testing.assert_array_equal(source.origin_index, indices)
                require(pd.DatetimeIndex(source.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), '来源收盘不同')
                require(pd.DatetimeIndex(source.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), '来源不是下一开盘')
                require(pd.DatetimeIndex(source.decision_time).equals(pd.DatetimeIndex(source.origin)+pd.Timedelta(hours=15, minutes=5)), '来源决定时钟不同')
                values = np.full(len(frame), np.nan)
                values[indices] = source.reference_weight
                targets[parent] = values
            for control, (old, _) in CONTROLS.items():
                pd.testing.assert_frame_equal(pd.read_parquet(old/period/cost/f'{control}_ledger.parquet'),
                    pd.read_parquet(folder/f'{control}_ledger.parquet'))
            cache[period, cost] = targets, pd.read_parquet(folder/'factors.parquet')
        targets, factors = cache[period, cost]
        setting = SETTINGS[model]
        values = targets[setting['parent']]
        counts = independent_counts(values)
        np.testing.assert_allclose(factors[model+'_target'], values, atol=0, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(factors[model+'_zero_streak'], counts)
        decisions = pd.read_parquet(folder/f'{model}_decisions.parquet')
        ledger = pd.read_parquet(folder/f'{model}_ledger.parquet')
        known = np.isfinite(values[indices])
        require(pd.DatetimeIndex(decisions.decision_time).equals(pd.DatetimeIndex(decisions.origin)+pd.Timedelta(hours=15, minutes=5)), '新账户决定时钟不同')
        require(decisions.loc[known, 'output_candidate'].eq(model).all() and decisions.loc[known, 'budget_source'].eq(setting['parent']).all(), '来源或候选身份改变')
        require(decisions.loc[known, 'used_band'].eq(setting['band']).all() and decisions.loc[known, 'required_zero_confirmations'].eq(setting['zero_confirmations']).all(), '执行门槛或退出类型不同')
        np.testing.assert_array_equal(decisions.loc[known, 'used_zero_count'], counts[indices][known])
        prior = np.r_[cfg['initial_capital'], ledger.equity.iloc[:-1]]
        shares = np.r_[0, ledger.shares.iloc[:-1]]
        np.testing.assert_array_equal(decisions.loc[known, 'own_close_shares'], shares[known])
        np.testing.assert_allclose(decisions.loc[known, 'own_close_equity'], prior[known], atol=1e-6, rtol=0)
        waiting = (values[indices] == 0) & (counts[indices] < setting['zero_confirmations']) & (shares > 0)
        np.testing.assert_array_equal(decisions.loc[known, 'waiting_zero_confirmation'], waiting[known].astype(float))
        checks.append({'model': model, 'period': period, 'cost': cost, 'band': setting['band'],
            'waiting_held_origins': int(waiting.sum()), 'fills': prices_and_costs(ledger, frame, first, cfg, cfg['costs'][cost])})
        return values

    accounts, cycles, differences, count = [], [], [], 0
    for model in CANDIDATES:
        a, c, d, n = verify_requested_target_accounts(OUT, {**cfg, 'primary': model}, result, data, dividends,
            lambda p, c, f, s: expected(p, c, f, s, model), expected_requests, comparison_models=list(CONTROLS))
        for destination, rows in [(accounts, a), (cycles, c), (differences, d)]:
            destination.extend({'model': model, **row} for row in rows)
        count += n
    for name, rows in [('saved_account_checks.csv', accounts), ('saved_actual_cycles.csv', cycles),
        ('saved_comparison_differences.csv', differences), ('saved_band_and_exit_checks.csv', checks)]:
        pd.DataFrame(rows).to_csv(OUT/name, index=False, encoding='utf-8-sig')
    require(len(accounts) == 32 and count == 45168, '三十二账户核对范围不同')
    receipt = {'verified_at': now(), 'status': 'PASS_THIRTY_TWO_BAND_ACCOUNTS_AND_BOTH_EXIT_TYPES',
        'actual_accounts': 32, 'actual_decisions_checked': count, 'complete_actual_cycles': len(cycles),
        'simulated_fills_checked': sum(r['fills'] for r in checks), 'new_models_or_accounts': 0,
        'independent_performance_validation': False, 'security_audit_performed': False,
        'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'verify': verify}[sys.argv[1]]()
