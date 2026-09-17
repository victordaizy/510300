"""四套收盘买入过滤一次计算、独立重建申请并保存完整比较。"""
import contextlib
import json
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.close_return_buy_gate_batch_inputs_v1 import CANDIDATES, MODELS, PRIMARY, SETTINGS, gate_frames, simulate_gate_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_requested_account_checks_v1 import verify_requested_target_accounts
from research.saved_target_custom_execution_v1 import run_custom_execution_batch
from scripts.verify_round195_20260913 import prices_and_costs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_close_return_buy_gate_batch_v1'
CONFIG = ROOT/'config/510300_close_return_buy_gate_batch_v1.json'
SOURCE = ROOT/'reports/research/510300_finite_rebalance_band_batch_v1'
PARENTS = {m: SOURCE for m in MODELS}
CONTROLS = {MODELS[0]: (SOURCE, '原201零门槛合并'), MODELS[1]: (SOURCE, '原201二十个百分点合并'),
    'TWO_CLOSE_ZERO_EXIT': (ROOT/'reports/research/510300_two_close_zero_exit_v1', '原198两次归零确认'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '本批已经登记或运行')
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    test = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_close_return_buy_gate_batch_v1.py', '-q', '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
    print(test.stdout, flush=True)
    require(test.returncode == 0 and '5 passed' in test.stdout, '五项买入过滤测试未通过')
    write_json(OUT/'tests_receipt.json', {'recorded_at': now(), 'exit_code': test.returncode, 'passed': 5,
        'seconds': time.perf_counter()-began}, exclusive=True)
    old_path = ROOT/'config/510300_finite_rebalance_band_batch_v1.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {k: old[k] for k in keys}
    cfg.update(study_id='510300_CLOSE_RETURN_BUY_GATE_BATCH_V1', round=203, registered_at=now(), primary=PRIMARY,
        candidate_models=list(CANDIDATES), candidate_configurations=4, candidate_settings=SETTINGS, parent_models=MODELS,
        decision_clock='15:05:00', annual_return_target=.10, new_model_fits=0, new_reference_accounts=0,
        source_budget_cny=0, goal_achieved=False, position_impact=0, independent_validation='NOT_ESTABLISHED',
        evidence_class='RETROSPECTIVE_CLOSE_RETURN_GATE_ON_POSITIVE_REQUESTS',
        prior_goal_turn_classification='PROGRESS_ROUND202_COMPLETED_24_ACCOUNTS_AND_REJECTED_LOSS_EXIT_FAMILY',
        rules='docs/510300_CLOSE_RETURN_BUY_GATE_BATCH_V1.md')
    text = (ROOT/'docs/510300_CLOSE_RETURN_BUY_GATE_BATCH_NEXT_20260913.md').read_text(encoding='utf-8')
    text = text.replace('本批目前只准备规则，尚未实现、测试、冻结或计算账户。', '本批在五项必要测试通过后、首次新账户计算前冻结规则。')
    text += '\n\n## 四套实际冻结设置\n\n|方案|已有持仓调仓门槛|允许新增买入的当日涨幅上限|\n|---|---:|---:|\n'
    for model, s in SETTINGS.items():
        text += f"|{CANDIDATES[model]}|{s['band']:.0%}|{s['return_cap']:.0%}|\n"
    text += '\n\n## 来源全部中文因素和原进出场\n\n下文保留第201轮及内部参考规则，说明原目标如何产生。本批只选择零与二十个百分点门槛，并增加上述买入过滤；不加入第202轮亏损保护。旧轮次状态属于来源记录。\n\n'
    text += (ROOT/old['rules']).read_text(encoding='utf-8')
    with (ROOT/cfg['rules']).open('x', encoding='utf-8') as stream:
        stream.write(text)
    paths = [Path(__file__), ROOT/'research/close_return_buy_gate_batch_inputs_v1.py',
        ROOT/'tests/test_close_return_buy_gate_batch_v1.py', ROOT/'tests/test_two_close_zero_exit_v1.py',
        ROOT/'tests/test_target_band_partial_rebalance_v1.py', ROOT/'research/event_account_indexed_request_v1.py',
        ROOT/'research/saved_requested_account_checks_v1.py', ROOT/'research/saved_target_custom_execution_v1.py',
        ROOT/'research/saved_parent_target_alignment_v1.py', ROOT/'research/adaptive_allocation_v1.py',
        ROOT/'research/intraday_overnight_increment_v1.py', ROOT/'research/joint_account_acceptance_v1.py',
        ROOT/'scripts/verify_round195_20260913.py', old_path, ROOT/old['rules'], SOURCE/'saved_verification_receipt.json',
        OUT/'tests_output.txt', OUT/'tests_receipt.json']
    paths += [ROOT/cfg[k] for k in ['features', 'dividends', 'rules']]
    for model, folder in PARENTS.items():
        paths += [folder/p/c/f'{model}_decisions.parquet' for p in ['evaluation', 'earlier_diagnostic'] for c in cfg['costs']]
    for model, (folder, _) in CONTROLS.items():
        paths += [folder/p/c/f'{model}_ledger.parquet' for p in ['evaluation', 'earlier_diagnostic'] for c in cfg['costs']]
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(paths))]
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 202 and not index['running_studies'], '前序完成状态不同')
    write_json(CONFIG, cfg, exclusive=True)
    index['running_studies'] = [{'round': 203, 'study': cfg['study_id'], 'status': 'FROZEN_NOT_STARTED', 'config': str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True, status='CLOSE_RETURN_BUY_GATE_BATCH_FROZEN', source=cfg['rules'])
    write_json(path, index)
    print('第203轮四套买入过滤已冻结，计划十六条账户。', flush=True)


def run():
    with (OUT/'run_output.txt').open('x', encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
        result = run_custom_execution_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, gate_frames, simulate_gate_account)
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
    ranked.to_csv(OUT/'seven_setting_joint_comparison.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({'核心计算秒数': result['run_seconds'], '联合验收': joint['candidates'],
        '全批排名': ranked.to_dict('records')}, ensure_ascii=False), flush=True)


def normal_requests(targets, prior, shares, prices, cfg):
    known = np.isfinite(targets)
    desired = shares.copy()
    desired[known] = (np.floor(targets[known]*prior[known]/prices[known]/cfg['lot'])*cfg['lot']).astype(int)
    band = cfg['candidate_settings'][cfg['primary']]['band']
    within = known & (targets > 0) & (shares > 0) & (np.abs(targets-shares*prices/prior) < band)
    desired[within] = shares[within]
    return desired-shares


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(), '本批核对已经完成')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path']) == item['sha256'], '冻结来源改变')
    data = pd.read_parquet(ROOT/cfg['features'])
    dividends = normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    cache, checks, suppressions = {}, [], []

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
            derived = []
            for r in frame.itertuples():
                valid = all(np.isfinite(x) for x in [r.close, r.previous_close, r.dividend]) and r.close > 0 and r.previous_close > 0 and r.dividend >= 0
                ret = (r.close+r.dividend)/r.previous_close-1 if valid else np.nan
                allowed = {percent: valid and Decimal(str(r.close))+Decimal(str(r.dividend)) <= Decimal(str(r.previous_close))*(Decimal(100)+Decimal(percent))/Decimal(100) for percent in [0, 1]}
                derived.append({'daily_total_simple': ret, 'market_inputs_known': valid,
                    'buy_allowed_r00': allowed[0], 'buy_allowed_r01': allowed[1]})
            factors = pd.read_parquet(folder/'factors.parquet')
            market = pd.DataFrame(derived)
            for column in market:
                np.testing.assert_allclose(factors[column], market[column], atol=1e-12, rtol=0, equal_nan=True)
            cache[period, cost] = originals, factors, market
        originals, factors, market = cache[period, cost]
        setting = SETTINGS[model]
        raw = originals[setting['parent']]
        allowed = market[f"buy_allowed_r{int(setting['return_cap']*100):02d}"].iloc[indices].to_numpy(bool)
        decisions = pd.read_parquet(folder/f'{model}_decisions.parquet')
        ledger = pd.read_parquet(folder/f'{model}_ledger.parquet')
        known = np.isfinite(raw)
        np.testing.assert_allclose(factors.loc[indices, model+'_target'], raw, atol=0, rtol=0, equal_nan=True)
        require(pd.DatetimeIndex(decisions.decision_time).equals(pd.DatetimeIndex(decisions.origin)+pd.Timedelta(hours=15, minutes=5)), '新决定时钟不同')
        require(decisions.loc[known, 'output_candidate'].eq(model).all() and decisions.loc[known, 'budget_source'].eq(setting['parent']).all(), '过滤候选身份混用')
        require(decisions.loc[known, 'used_band'].eq(setting['band']).all() and decisions.loc[known, 'return_cap'].eq(setting['return_cap']).all(), '过滤设置不同')
        prior, shares = np.r_[cfg['initial_capital'], ledger.equity.iloc[:-1]], np.r_[0, ledger.shares.iloc[:-1]].astype(int)
        normal = normal_requests(raw, prior, shares, frame.close.iloc[indices].to_numpy(float), {**cfg, 'primary': model})
        blocked = (normal > 0) & ~allowed
        np.testing.assert_array_equal(decisions.loc[known, 'normal_requested_quantity'], normal[known])
        np.testing.assert_array_equal(decisions.loc[known, 'buy_allowed'], allowed[known].astype(float))
        np.testing.assert_array_equal(decisions.loc[known, 'buy_suppressed'], blocked[known].astype(float))
        np.testing.assert_allclose(decisions.loc[known, 'observed_daily_return'], market.daily_total_simple.iloc[indices].to_numpy()[known], atol=1e-12, rtol=0, equal_nan=True)
        np.testing.assert_allclose(decisions.loc[known, 'own_close_equity'], prior[known], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(decisions.loc[known, 'own_close_shares'], shares[known])
        for j in np.flatnonzero(blocked):
            suppressions.append({'model': model, 'period': period, 'cost': cost, 'origin': decisions.origin.iloc[j],
                'source_target': raw[j], 'normal_requested_quantity': normal[j], 'own_shares': shares[j],
                'observed_daily_return': market.daily_total_simple.iloc[indices[j]], 'return_cap': setting['return_cap']})
        checks.append({'model': model, 'period': period, 'cost': cost, 'suppressed_buy_requests': int(blocked.sum()),
            'suppressed_initial_entry_requests': int((blocked & (shares == 0)).sum()),
            'suppressed_addition_requests': int((blocked & (shares > 0)).sum()),
            'fills': prices_and_costs(ledger, frame, first, cfg, cfg['costs'][cost])})
        targets = np.full(len(frame), np.nan)
        targets[indices] = raw
        return targets, allowed

    accounts, cycles, differences, count = [], [], [], 0
    for model in CANDIDATES:
        active = {}

        def target_callback(period, cost, frame, start):
            targets, allowed = expected(period, cost, frame, start, model)
            active['allowed'] = allowed
            return targets

        def request_callback(targets, prior, shares, prices, settings):
            requests = normal_requests(targets, prior, shares, prices, settings)
            requests[(requests > 0) & ~active['allowed']] = 0
            return requests

        a, c, d, n = verify_requested_target_accounts(OUT, {**cfg, 'primary': model}, result, data, dividends,
            target_callback, request_callback, comparison_models=list(CONTROLS))
        for destination, rows in [(accounts, a), (cycles, c), (differences, d)]:
            destination.extend({'model': model, **row} for row in rows)
        count += n
    for name, rows in [('saved_account_checks.csv', accounts), ('saved_actual_cycles.csv', cycles),
        ('saved_comparison_differences.csv', differences), ('saved_buy_gate_checks.csv', checks), ('suppressed_buy_requests.csv', suppressions)]:
        pd.DataFrame(rows).to_csv(OUT/name, index=False, encoding='utf-8-sig')
    require(len(accounts) == 16 and count == 22584, '十六账户核对范围不同')
    receipt = {'verified_at': now(), 'status': 'PASS_SIXTEEN_BUY_GATE_ACCOUNTS_AND_INDEPENDENT_POSITIVE_REQUEST_FILTERS',
        'actual_accounts': 16, 'actual_decisions_checked': count, 'complete_actual_cycles': len(cycles),
        'simulated_fills_checked': sum(r['fills'] for r in checks), 'suppressed_buy_requests_checked': len(suppressions),
        'new_models_or_accounts': 0, 'independent_performance_validation': False, 'security_audit_performed': False,
        'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'verify': verify}[sys.argv[1]]()
