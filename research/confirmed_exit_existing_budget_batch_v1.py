"""十七套既有预算与新退出规则集中计算、核对和排序。"""
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.confirmed_exit_existing_budget_batch_inputs_v1 import CANDIDATES, MAPPING, MODELS, PRIMARY, SOURCES, mapped_frames, simulate_mapped_confirmation
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_requested_account_checks_v1 import verify_requested_target_accounts
from research.saved_target_custom_execution_v1 import run_custom_execution_batch
from research.two_close_zero_exit_v1 import expected_requests, independent_counts
from scripts.verify_round195_20260913 import prices_and_costs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_confirmed_exit_existing_budget_batch_v1'
CONFIG = ROOT/'config/510300_confirmed_exit_existing_budget_batch_v1.json'
PARENTS = {model: ROOT/'reports/research'/s['folder'] for model, s in SOURCES.items()}
CONTROLS = {
    'TWO_CLOSE_ZERO_EXIT': (ROOT/'reports/research/510300_two_close_zero_exit_v1', '原198每日六十日10%预算、两次零确认'),
    'ACCOUNT_VOLATILITY_EXPOSURE': (ROOT/'reports/research/510300_account_volatility_exposure_v1', '原181明确零立即退出'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '本批已有登记或运行')
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    test = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_confirmed_exit_existing_budget_batch_v1.py', '-q', '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
    print(test.stdout, flush=True)
    require(test.returncode == 0 and '5 passed' in test.stdout, '五项批次必要测试未通过')
    write_json(OUT/'tests_receipt.json', {'recorded_at': now(), 'exit_code': test.returncode, 'passed': 5,
        'seconds': time.perf_counter()-began}, exclusive=True)
    previous_path = ROOT/'config/510300_two_close_zero_exit_v1.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {key: previous[key] for key in keys}
    cfg.update(study_id='510300_CONFIRMED_EXIT_EXISTING_BUDGET_BATCH_V1', round=199, registered_at=now(),
        primary=PRIMARY, candidate_models=list(CANDIDATES), candidate_configurations=17, candidate_sources=SOURCES,
        candidate_mapping=MAPPING, parent_models=MODELS, zero_confirmations=2, decision_clock='15:05:00',
        annual_return_target=.10, new_model_fits=0, new_reference_accounts=0, source_budget_cny=0,
        goal_achieved=False, position_impact=0, independent_validation='NOT_ESTABLISHED',
        evidence_class='RETROSPECTIVE_FIXED_EXIT_AND_EXISTING_BUDGET_INTERACTION_BATCH',
        rules='docs/510300_CONFIRMED_EXIT_EXISTING_BUDGET_BATCH_V1.md')
    original_cfg = json.loads((ROOT/'config/510300_return_confirmation_auxiliary_batch_v1.json').read_text(encoding='utf-8'))
    with (ROOT/cfg['rules']).open('x', encoding='utf-8') as stream:
        stream.write((ROOT/'docs/510300_CONFIRMED_EXIT_EXISTING_BUDGET_BATCH_NEXT_20260913.md').read_text(encoding='utf-8').replace(
            '尚未实现、测试、冻结或计算新组合。', '本批在必要测试通过后、首次新组合计算前冻结。'))
        stream.write('\n\n## 十七套新组合的来源清单\n\n|新组合|来源研究|来源模型标识|\n|---|---|---|\n')
        for model, parent in MAPPING.items():
            stream.write(f"|{CANDIDATES[model]}|{SOURCES[parent]['folder']}|{parent}|\n")
        stream.write('\n\n## 共同174来源的全部中文因素和参考账户规则\n\n')
        stream.write('下文说明原参考目标如何生成。新组合本身的明确零退出由上文两次确认规则决定；参考账户保持原逻辑，不回流新组合收益。\n\n')
        stream.write((ROOT/original_cfg['rules']).read_text(encoding='utf-8'))
    paths = [Path(__file__), ROOT/'research/confirmed_exit_existing_budget_batch_inputs_v1.py',
        ROOT/'research/risk_window_clock_batch_inputs_v1.py', ROOT/'research/two_close_zero_exit_inputs_v1.py',
        ROOT/'research/two_close_zero_exit_account_v1.py', ROOT/'research/two_close_zero_exit_v1.py',
        ROOT/'research/event_account_indexed_request_v1.py', ROOT/'research/saved_requested_account_checks_v1.py',
        ROOT/'research/saved_target_custom_execution_v1.py', ROOT/'research/saved_parent_target_alignment_v1.py',
        ROOT/'research/adaptive_allocation_v1.py', ROOT/'research/intraday_overnight_increment_v1.py',
        ROOT/'research/joint_account_acceptance_v1.py', ROOT/'scripts/verify_round195_20260913.py',
        ROOT/'tests/test_confirmed_exit_existing_budget_batch_v1.py', ROOT/'tests/test_two_close_zero_exit_v1.py',
        ROOT/'tests/test_target_band_partial_rebalance_v1.py', previous_path, ROOT/original_cfg['rules'],
        ROOT/'config/510300_return_confirmation_auxiliary_batch_v1.json', OUT/'tests_output.txt', OUT/'tests_receipt.json',
        CONTROLS['TWO_CLOSE_ZERO_EXIT'][0]/'saved_verification_receipt.json']
    paths += [ROOT/cfg[key] for key in ['features', 'dividends', 'rules']]
    paths += [ROOT/'config'/f"{s['folder']}.json" for s in SOURCES.values()]
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in cfg['costs']:
            paths += [folder/period/cost/f'{model}_decisions.parquet' for model, folder in PARENTS.items()]
            paths += [folder/period/cost/f'{model}_ledger.parquet' for model, (folder, _) in CONTROLS.items()]
    require(len(PARENTS) == 17 and 'ACCOUNT_VOLATILITY_EXPOSURE' not in PARENTS, '已有198组合重复进入计算')
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 198 and not index['running_studies'], '前序完成状态不同')
    index['running_studies'] = [{'round': 199, 'study': cfg['study_id'], 'status': 'FROZEN_NOT_STARTED', 'config': str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True, status='CONFIRMED_EXIT_EXISTING_BUDGET_BATCH_FROZEN', source=cfg['rules'])
    write_json(path, index)
    print('第199轮十七套新组合已冻结，计划六十八条完整账户。', flush=True)


def run():
    with (OUT/'run_output.txt').open('x', encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
        result = run_custom_execution_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, mapped_frames, simulate_mapped_confirmation)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    joint = save_joint_assessment(OUT, cfg, result)
    ranking = []
    for model in [*CANDIDATES, 'TWO_CLOSE_ZERO_EXIT']:
        row = {'model': model, 'reused_198': model == 'TWO_CLOSE_ZERO_EXIT'}
        ratios = []
        for period, key in [('main', 'all_metrics'), ('earlier', 'earlier_diagnostics')]:
            for cost in ['BASE', 'STRESS']:
                metric = next(r for r in result[key] if r['model'] == model and r['cost'] == cost)
                row['name'] = metric['name']
                row.update({period+'_'+cost.lower()+'_'+field: metric[field] for field in ['net_sharpe', 'annualized_return', 'max_drawdown']})
                ratios += [metric['net_sharpe']/1.2, metric['annualized_return']/.1]
        row['minimum_joint_ratio'] = min(ratios)
        ranking.append(row)
    ranked = pd.DataFrame(ranking).sort_values(['minimum_joint_ratio', 'model'], ascending=[False, True])
    ranked.to_csv(OUT/'eighteen_confirmed_budget_comparison.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({'核心计算秒数': result['run_seconds'], '四场景联合通过': [m for m, a in joint['candidates'].items() if a['four_scenario_joint_pass']],
        '前五': ranked.head(5).to_dict('records')}, ensure_ascii=False), flush=True)


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(), '本批核对已经完成')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path']) == item['sha256'], '冻结输入改变')
    data = pd.read_parquet(ROOT/cfg['features'])
    dividends = normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    factor_cache, checks = {}, []

    def expected(period, cost, frame, start, model):
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        parent = MAPPING[model]
        source = pd.read_parquet(PARENTS[parent]/period/cost/f'{parent}_decisions.parquet')
        np.testing.assert_array_equal(source.origin_index, indices)
        require(pd.DatetimeIndex(source.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), '预算来源收盘日期不同')
        require(pd.DatetimeIndex(source.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), '预算来源不是下一开盘')
        require(pd.DatetimeIndex(source.decision_time).equals(pd.DatetimeIndex(source.origin)+pd.Timedelta(hours=15, minutes=5)), '预算来源时钟不同')
        target = np.full(len(frame), np.nan)
        target[indices] = source.reference_weight
        counts = independent_counts(target)
        folder = OUT/period/cost
        if (period, cost) not in factor_cache:
            factor_cache[(period, cost)] = pd.read_parquet(folder/'factors.parquet')
            for control, (old, _) in CONTROLS.items():
                pd.testing.assert_frame_equal(pd.read_parquet(old/period/cost/f'{control}_ledger.parquet'),
                    pd.read_parquet(folder/f'{control}_ledger.parquet'))
        factors = factor_cache[(period, cost)]
        np.testing.assert_allclose(factors[model+'_target'], target, atol=0, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(factors[model+'_zero_streak'], counts)
        decisions = pd.read_parquet(folder/f'{model}_decisions.parquet')
        ledger = pd.read_parquet(folder/f'{model}_ledger.parquet')
        require(pd.DatetimeIndex(decisions.decision_time).equals(pd.DatetimeIndex(decisions.origin)+pd.Timedelta(hours=15, minutes=5)), '组合决定时钟不同')
        known = np.isfinite(target[indices])
        prior = np.r_[cfg['initial_capital'], ledger.equity.iloc[:-1]]
        shares = np.r_[0, ledger.shares.iloc[:-1]]
        require(decisions.loc[known, 'output_candidate'].eq(model).all() and decisions.loc[known, 'budget_source'].eq(parent).all(), '组合或预算身份混用')
        np.testing.assert_array_equal(decisions.loc[known, 'used_zero_count'], counts[indices][known])
        np.testing.assert_array_equal(decisions.loc[known, 'own_close_shares'], shares[known])
        np.testing.assert_allclose(decisions.loc[known, 'own_close_equity'], prior[known], atol=1e-6, rtol=0)
        waiting = (counts[indices] == 1) & (shares > 0)
        np.testing.assert_array_equal(decisions.loc[known, 'waiting_zero_confirmation'], waiting[known].astype(float))
        checks.append({'model': model, 'period': period, 'cost': cost, 'decisions': len(indices),
            'first_zero_held_origins': int(waiting.sum()), 'fills': prices_and_costs(ledger, frame, first, cfg, cfg['costs'][cost])})
        return target

    accounts, cycles, differences, count = [], [], [], 0
    for model in CANDIDATES:
        def selected(period, cost, frame, start):
            return expected(period, cost, frame, start, model)
        a, c, d, n = verify_requested_target_accounts(OUT, {**cfg, 'primary': model}, result, data, dividends,
            selected, expected_requests, comparison_models=list(CONTROLS))
        for destination, rows in [(accounts, a), (cycles, c), (differences, d)]:
            destination.extend({'model': model, **row} for row in rows)
        count += n
    for name, rows in [('saved_account_checks.csv', accounts), ('saved_actual_cycles.csv', cycles),
        ('saved_comparison_differences.csv', differences), ('saved_zero_confirmation_checks.csv', checks)]:
        pd.DataFrame(rows).to_csv(OUT/name, index=False, encoding='utf-8-sig')
    require(len(accounts) == 68 and count == 95982, '六十八账户核对范围不同')
    receipt = {'verified_at': now(), 'status': 'PASS_SIXTY_EIGHT_CONFIRMED_BUDGET_ACCOUNTS_AND_SOURCE_IDENTITIES',
        'actual_accounts': 68, 'actual_decisions_checked': count, 'complete_actual_cycles': len(cycles),
        'simulated_fills_checked': sum(r['fills'] for r in checks), 'new_models_or_accounts': 0,
        'independent_performance_validation': False, 'security_audit_performed': False,
        'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'verify': verify}[sys.argv[1]]()
