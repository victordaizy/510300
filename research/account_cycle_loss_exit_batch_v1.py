"""集中计算六套完整账户周期保护，并独立重建申请、基准和退出状态。"""
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.account_cycle_loss_exit_batch_inputs_v1 import CANDIDATES, MODELS, PRIMARY, SETTINGS, loss_frames, simulate_loss_account
from research.adaptive_allocation_v1 import normalize_dividends
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_requested_account_checks_v1 import verify_requested_target_accounts
from research.saved_target_custom_execution_v1 import run_custom_execution_batch
from scripts.verify_round195_20260913 import prices_and_costs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_account_cycle_loss_exit_batch_v1'
CONFIG = ROOT/'config/510300_account_cycle_loss_exit_batch_v1.json'
SOURCE = ROOT/'reports/research/510300_finite_rebalance_band_batch_v1'
PARENTS = {m: SOURCE for m in MODELS}
CONTROLS = {MODELS[0]: (SOURCE, '原201零门槛合并、无新增亏损保护'),
    MODELS[1]: (SOURCE, '原201二十个百分点合并、无新增亏损保护'),
    'TWO_CLOSE_ZERO_EXIT': (ROOT/'reports/research/510300_two_close_zero_exit_v1', '原198两次归零确认'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '本批已经登记或运行')
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    test = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_account_cycle_loss_exit_batch_v1.py', '-q', '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
    print(test.stdout, flush=True)
    require(test.returncode == 0 and '6 passed' in test.stdout, '六项周期保护测试未通过')
    write_json(OUT/'tests_receipt.json', {'recorded_at': now(), 'exit_code': test.returncode, 'passed': 6,
        'seconds': time.perf_counter()-began}, exclusive=True)
    old_path = ROOT/'config/510300_finite_rebalance_band_batch_v1.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {k: old[k] for k in keys}
    cfg.update(study_id='510300_ACCOUNT_CYCLE_LOSS_EXIT_BATCH_V1', round=202, registered_at=now(), primary=PRIMARY,
        candidate_models=list(CANDIDATES), candidate_configurations=6, candidate_settings=SETTINGS, parent_models=MODELS,
        decision_clock='15:05:00', annual_return_target=.10, new_model_fits=0, new_reference_accounts=0, source_budget_cny=0,
        goal_achieved=False, position_impact=0, independent_validation='NOT_ESTABLISHED',
        evidence_class='RETROSPECTIVE_COMPLETE_ACCOUNT_CYCLE_LOSS_EXITS',
        prior_goal_turn_classification='PROGRESS_ROUND201_COMPLETED_32_ACCOUNTS_AND_ENTRY_AGE_EVIDENCE',
        source_preflight='reports/research/510300_saved_entry_age_attribution_through201/result.json',
        rules='docs/510300_ACCOUNT_CYCLE_LOSS_EXIT_BATCH_V1.md')
    text = (ROOT/'docs/510300_ACCOUNT_CYCLE_LOSS_EXIT_BATCH_NEXT_20260913.md').read_text(encoding='utf-8')
    text = text.replace('本批目前只准备方案，尚未实现、测试、冻结或计算账户。', '本批在六项必要测试通过后、首次新账户计算前冻结规则。')
    text += '\n\n## 六套实际冻结设置\n\n|方案|已有持仓调仓门槛|完整账户周期亏损触发值|\n|---|---:|---:|\n'
    for model, s in SETTINGS.items():
        text += f"|{CANDIDATES[model]}|{s['band']:.0%}|{s['loss_fraction']:.0%}|\n"
    text += '\n\n## 来源的全部中文因素和原进出场\n\n下文说明第201轮及其内部目标如何产生。本批选择零和二十个百分点两个已有方案，并在本批自己的实际账户增加上述周期亏损保护；不把其他六套旧门槛当成本批候选。旧轮次状态只属于来源记录。\n\n'
    text += (ROOT/old['rules']).read_text(encoding='utf-8')
    with (ROOT/cfg['rules']).open('x', encoding='utf-8') as stream:
        stream.write(text)
    paths = [Path(__file__), ROOT/'research/account_cycle_loss_exit_batch_inputs_v1.py',
        ROOT/'tests/test_account_cycle_loss_exit_batch_v1.py', ROOT/'tests/test_two_close_zero_exit_v1.py',
        ROOT/'tests/test_target_band_partial_rebalance_v1.py', ROOT/'research/event_account_indexed_request_v1.py',
        ROOT/'research/saved_requested_account_checks_v1.py', ROOT/'research/saved_target_custom_execution_v1.py',
        ROOT/'research/saved_parent_target_alignment_v1.py', ROOT/'research/adaptive_allocation_v1.py',
        ROOT/'research/intraday_overnight_increment_v1.py', ROOT/'research/joint_account_acceptance_v1.py',
        ROOT/'scripts/verify_round195_20260913.py', old_path, ROOT/old['rules'], SOURCE/'saved_verification_receipt.json',
        OUT/'tests_output.txt', OUT/'tests_receipt.json']
    paths += [ROOT/cfg[k] for k in ['features', 'dividends', 'rules', 'source_preflight']]
    for model, folder in PARENTS.items():
        paths += [folder/p/c/f'{model}_decisions.parquet' for p in ['evaluation', 'earlier_diagnostic'] for c in cfg['costs']]
    for model, (folder, _) in CONTROLS.items():
        paths += [folder/p/c/f'{model}_ledger.parquet' for p in ['evaluation', 'earlier_diagnostic'] for c in cfg['costs']]
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(paths))]
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 201 and not index['running_studies'], '前序完成状态不同')
    write_json(CONFIG, cfg, exclusive=True)
    index['running_studies'] = [{'round': 202, 'study': cfg['study_id'], 'status': 'FROZEN_NOT_STARTED', 'config': str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True, status='ACCOUNT_CYCLE_LOSS_EXIT_BATCH_FROZEN', source=cfg['rules'])
    write_json(path, index)
    print('第202轮六套完整账户周期保护已冻结，计划二十四条账户。', flush=True)


def run():
    with (OUT/'run_output.txt').open('x', encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
        result = run_custom_execution_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, loss_frames, simulate_loss_account)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    joint = save_joint_assessment(OUT, cfg, result)
    ranking = []
    for model in [*CANDIDATES, *MODELS, 'TWO_CLOSE_ZERO_EXIT']:
        row, ratios = {'model': model, 'reused_control': model not in CANDIDATES}, []
        for period, key in [('main', 'all_metrics'), ('earlier', 'earlier_diagnostics')]:
            for cost in ['BASE', 'STRESS']:
                m = next(r for r in result[key] if r['model'] == model and r['cost'] == cost)
                row['name'] = m['name']
                row.update({period+'_'+cost.lower()+'_'+k: m[k] for k in ['net_sharpe', 'annualized_return', 'max_drawdown', 'trade_count']})
                ratios.extend([m['net_sharpe']/1.2, m['annualized_return']/.1])
        row['minimum_joint_ratio'] = min(ratios)
        ranking.append(row)
    ranked = pd.DataFrame(ranking).sort_values('minimum_joint_ratio', ascending=False)
    ranked.to_csv(OUT/'nine_setting_joint_comparison.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({'核心计算秒数': result['run_seconds'], '联合验收': joint['candidates'],
        '全批排名': ranked.to_dict('records')}, ensure_ascii=False), flush=True)


def independent_policy(source, ledger, cfg, model):
    """从实际份额转变和前序净值重建基准，使用分开的卖出锁与等待归零状态。"""
    setting = cfg['candidate_settings'][model]
    prior = np.r_[cfg['initial_capital'], ledger.equity.iloc[:-1]]
    shares = np.r_[0, ledger.shares.iloc[:-1]].astype(int)
    forced_sell, wait_zero, basis, cycles = False, False, np.nan, 0
    output = []
    for i, raw in enumerate(source):
        holding = shares[i] > 0
        if holding and (i == 0 or shares[i-1] == 0):
            require(i > 0 and ledger.filled_quantity.iloc[i-1] > 0 and not forced_sell and not wait_zero, '独立重建发现不允许的新入场')
            basis, cycles = prior[i-1], cycles+1
        if not holding:
            basis, forced_sell = np.nan, False
        period_return = prior[i]/basis-1 if holding else np.nan
        new_trigger = holding and not forced_sell and prior[i] <= basis*(1-setting['loss_fraction'])
        if new_trigger:
            forced_sell, wait_zero = True, True
        if raw == 0:
            wait_zero = False
        protected = forced_sell or wait_zero
        effective = 0. if protected else raw
        output.append({'reference_weight': effective, 'cycle_basis': basis, 'cycle_return': period_return,
            'cycle_number': float(cycles if holding else 0), 'triggered_protection': float(new_trigger),
            'pending_protective_exit': float(forced_sell), 'waiting_source_zero': float(wait_zero),
            'protection_active': float(protected), 'own_close_equity': prior[i], 'own_close_shares': float(shares[i])})
    return pd.DataFrame(output)


def expected_requests(targets, prior, old_shares, prices, cfg):
    setting = cfg['candidate_settings'][cfg['primary']]
    known = np.isfinite(targets)
    desired = old_shares.copy()
    desired[known] = (np.floor(targets[known]*prior[known]/prices[known]/cfg['lot'])*cfg['lot']).astype(int)
    within = known & (targets > 0) & (old_shares > 0) & (np.abs(targets-old_shares*prices/prior) < setting['band'])
    desired[within] = old_shares[within]
    return desired-old_shares


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(), '本批核对已经完成')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path']) == item['sha256'], '冻结来源改变')
    data = pd.read_parquet(ROOT/cfg['features'])
    dividends = normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    cache, checks, events = {}, [], []

    def expected(period, cost, frame, start, model):
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        folder = OUT/period/cost
        if (period, cost) not in cache:
            originals = {}
            for parent, old in PARENTS.items():
                source = pd.read_parquet(old/period/cost/f'{parent}_decisions.parquet')
                np.testing.assert_array_equal(source.origin_index, indices)
                require(pd.DatetimeIndex(source.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), '来源收盘不同')
                require(pd.DatetimeIndex(source.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), '来源不是下一开盘')
                require(pd.DatetimeIndex(source.decision_time).equals(pd.DatetimeIndex(source.origin)+pd.Timedelta(hours=15, minutes=5)), '来源时钟不同')
                originals[parent] = source.reference_weight.to_numpy(float)
            for control, (old, _) in CONTROLS.items():
                pd.testing.assert_frame_equal(pd.read_parquet(old/period/cost/f'{control}_ledger.parquet'),
                    pd.read_parquet(folder/f'{control}_ledger.parquet'))
            cache[period, cost] = originals, pd.read_parquet(folder/'factors.parquet')
        originals, factors = cache[period, cost]
        setting = SETTINGS[model]
        source = originals[setting['parent']]
        decisions = pd.read_parquet(folder/f'{model}_decisions.parquet')
        ledger = pd.read_parquet(folder/f'{model}_ledger.parquet')
        computed = independent_policy(source, ledger, cfg, model)
        for column in computed:
            np.testing.assert_allclose(decisions[column], computed[column], atol=1e-10 if column == 'cycle_basis' else 1e-12,
                rtol=0, equal_nan=True, err_msg='独立周期状态或资金基准不同：'+column)
        require(pd.DatetimeIndex(decisions.decision_time).equals(pd.DatetimeIndex(decisions.origin)+pd.Timedelta(hours=15, minutes=5)), '保护决定时钟不同')
        require(decisions.output_candidate.eq(model).all() and decisions.budget_source.eq(setting['parent']).all(), '保护候选身份混用')
        require(decisions.used_band.eq(setting['band']).all() and decisions.loss_fraction.eq(setting['loss_fraction']).all(), '保护设置混用')
        np.testing.assert_allclose(decisions.source_original_target, source, atol=0, rtol=0, equal_nan=True)
        np.testing.assert_allclose(factors.loc[indices, model+'_target'], source, atol=0, rtol=0, equal_nan=True)
        require(decisions.loc[decisions.reference_weight.isna(), 'signal_state'].eq('NO_VIEW_KEEP_EXISTING_SHARES').all(), '未知状态被改成明确空仓')
        for j in np.flatnonzero(computed.triggered_protection):
            events.append({'model': model, 'period': period, 'cost': cost, 'origin': decisions.origin.iloc[j],
                'next_execution': decisions.execution_date.iloc[j], **computed.iloc[j].to_dict()})
        checks.append({'model': model, 'period': period, 'cost': cost,
            'triggers': int(computed.triggered_protection.sum()), 'waiting_zero_origins': int(computed.waiting_source_zero.sum()),
            'pending_exit_origins': int(computed.pending_protective_exit.sum()),
            'unknown_source_protection_origins': int((np.isnan(source) & computed.protection_active.eq(1)).sum()),
            'fills': prices_and_costs(ledger, frame, first, cfg, cfg['costs'][cost])})
        targets = np.full(len(frame), np.nan)
        targets[indices] = computed.reference_weight
        return targets

    accounts, cycles, differences, count = [], [], [], 0
    for model in CANDIDATES:
        a, c, d, n = verify_requested_target_accounts(OUT, {**cfg, 'primary': model}, result, data, dividends,
            lambda p, c, f, s: expected(p, c, f, s, model), expected_requests, comparison_models=list(CONTROLS))
        for destination, rows in [(accounts, a), (cycles, c), (differences, d)]:
            destination.extend({'model': model, **row} for row in rows)
        count += n
    for name, rows in [('saved_account_checks.csv', accounts), ('saved_actual_cycles.csv', cycles),
        ('saved_comparison_differences.csv', differences), ('saved_protection_checks.csv', checks), ('protection_trigger_events.csv', events)]:
        pd.DataFrame(rows).to_csv(OUT/name, index=False, encoding='utf-8-sig')
    require(len(accounts) == 24 and count == 33876, '二十四账户核对范围不同')
    receipt = {'verified_at': now(), 'status': 'PASS_TWENTY_FOUR_CYCLE_LOSS_ACCOUNTS_AND_INDEPENDENT_ENTRY_BASES',
        'actual_accounts': 24, 'actual_decisions_checked': count, 'complete_actual_cycles': len(cycles),
        'simulated_fills_checked': sum(r['fills'] for r in checks), 'protection_triggers_checked': len(events),
        'new_models_or_accounts': 0, 'independent_performance_validation': False, 'security_audit_performed': False,
        'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'verify': verify}[sys.argv[1]]()
