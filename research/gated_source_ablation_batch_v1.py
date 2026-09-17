"""六套固定加仓条件来源删减一次计算、独立重建申请并保存完整比较。"""
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
from research.gated_source_ablation_batch_inputs_v1 import CANDIDATES, MODELS, PRIMARY, SETTINGS, SOURCES, allowance_column, gate_frames, simulate_gate_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_requested_account_checks_v1 import verify_requested_target_accounts
from research.saved_target_custom_execution_v1 import run_custom_execution_batch
from scripts.verify_round195_20260913 import prices_and_costs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_gated_source_ablation_batch_v1'
CONFIG = ROOT/'config/510300_gated_source_ablation_batch_v1.json'
SOURCE = ROOT/'reports/research/510300_finite_rebalance_band_batch_v1'
PARENTS = {m: ROOT/'reports/research'/folder for m, folder in SOURCES.items()}
CONTROLS = {
    'ADD_GATE_EXPOSURE_105': (ROOT/'reports/research/510300_addition_gate_exposure_batch_v1', '原206目标1.05倍'),
    'ADD_GATE_BAND20_LOWER01': (ROOT/'reports/research/510300_addition_only_return_gate_batch_v1', '原205仅强势加仓'),
    'INTENT_MIX_BAND_00': (SOURCE, '原201零门槛合并'),
    'INTENT_MIX_BAND_20': (SOURCE, '原201二十个百分点合并'),
    'TWO_CLOSE_ZERO_EXIT': (ROOT/'reports/research/510300_two_close_zero_exit_v1', '原198两次归零确认'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}



def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '本批已经登记或运行')
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    test = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_gated_source_ablation_batch_v1.py', '-q', '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
    print(test.stdout, flush=True)
    require(test.returncode == 0 and '6 passed' in test.stdout, '六项买入过滤测试未通过')
    write_json(OUT/'tests_receipt.json', {'recorded_at': now(), 'exit_code': test.returncode, 'passed': 6,
        'seconds': time.perf_counter()-began}, exclusive=True)
    old_path = ROOT/'config/510300_finite_rebalance_band_batch_v1.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {k: old[k] for k in keys}
    cfg.update(study_id='510300_GATED_SOURCE_ABLATION_BATCH_V1', round=208, registered_at=now(), primary=PRIMARY,
        candidate_models=list(CANDIDATES), candidate_configurations=6, candidate_settings=SETTINGS, parent_models=MODELS, source_folders=SOURCES,
        decision_clock='15:05:00', annual_return_target=.10, new_model_fits=0, new_reference_accounts=0,
        source_budget_cny=0, goal_achieved=False, position_impact=0, independent_validation='NOT_ESTABLISHED',
        evidence_class='RETROSPECTIVE_GATED_SOURCE_ABLATION_WITH_PLANNED_EXPOSURE',
        prior_goal_turn_classification='PROGRESS_ROUND207_COMPLETED_32_ACCOUNTS_AND_REDUCTION_BRANCH_REJECTED',
        rules='docs/510300_GATED_SOURCE_ABLATION_BATCH_V1.md')
    text = (ROOT/'docs/510300_GATED_SOURCE_ABLATION_BATCH_NEXT_20260913.md').read_text(encoding='utf-8')
    text = text.replace('本批目前只准备规则，尚未实现、测试、冻结或计算账户。', '本批在六项必要测试通过后、首次新账户计算前冻结规则。')
    text += '\n\n## 六套实际冻结设置\n\n|方案|已有持仓调仓门槛|已有持仓加仓的当日收益条件|目标倍率|\n|---|---:|---:|---:|\n'
    for model, s in SETTINGS.items():
        text += f"|{CANDIDATES[model]}|{s['band']:.0%}|{'不超过' if s['direction']=='UPPER' else '至少为'}{s['return_threshold']:.0%}|{s['exposure_multiplier']:.2f}|\n"
    text += '\n\n## 来源全部中文因素和原进出场\n\n下文保留第201轮及内部参考规则，说明原目标如何产生。本批固定二十个百分点门槛及至少1%的加仓准入，三来源按上述保留权重重建计划目标，再乘1.00或1.05并限制100%；普通减仓恢复原规则，明确零及终点全部退出保持。旧轮次状态属于来源记录。\n\n'
    text += (ROOT/old['rules']).read_text(encoding='utf-8')
    with (ROOT/cfg['rules']).open('x', encoding='utf-8') as stream:
        stream.write(text)
    paths = [Path(__file__), ROOT/'research/gated_source_ablation_batch_inputs_v1.py',
        ROOT/'tests/test_gated_source_ablation_batch_v1.py', ROOT/'tests/test_two_close_zero_exit_v1.py',
        ROOT/'tests/test_target_band_partial_rebalance_v1.py', ROOT/'research/event_account_indexed_request_v1.py',
        ROOT/'research/saved_requested_account_checks_v1.py', ROOT/'research/saved_target_custom_execution_v1.py',
        ROOT/'research/saved_parent_target_alignment_v1.py', ROOT/'research/adaptive_allocation_v1.py',
        ROOT/'research/three_source_order_intent_mix_inputs_v1.py', ROOT/'tests/test_three_source_order_intent_mix_v1.py',
        ROOT/'research/close_return_buy_gate_batch_inputs_v1.py', ROOT/'research/close_return_buy_strength_gate_batch_inputs_v1.py',
        ROOT/'research/intraday_overnight_increment_v1.py', ROOT/'research/joint_account_acceptance_v1.py',
        ROOT/'scripts/verify_round195_20260913.py', old_path, ROOT/old['rules'], SOURCE/'saved_verification_receipt.json',
        OUT/'tests_output.txt', OUT/'tests_receipt.json']
    paths += [ROOT/cfg[k] for k in ['features', 'dividends', 'rules']]
    for model, folder in PARENTS.items():
        paths += [folder/p/c/f'{model}_{kind}.parquet' for p in ['evaluation', 'earlier_diagnostic'] for c in cfg['costs'] for kind in ['decisions', 'ledger']]
        paths += [folder/'saved_verification_receipt.json', ROOT/'config'/f'{SOURCES[model]}.json']
    for model, (folder, _) in CONTROLS.items():
        paths += [folder/p/c/f'{model}_ledger.parquet' for p in ['evaluation', 'earlier_diagnostic'] for c in cfg['costs']]
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(paths))]
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 207 and not index['running_studies'], '前序完成状态不同')
    write_json(CONFIG, cfg, exclusive=True)
    index['running_studies'] = [{'round': 208, 'study': cfg['study_id'], 'status': 'FROZEN_NOT_STARTED', 'config': str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True, status='GATED_SOURCE_ABLATION_BATCH_FROZEN', source=cfg['rules'])
    write_json(path, index)
    print('第208轮六套来源删减已冻结，计划二十四条账户。', flush=True)


def run():
    with (OUT/'run_output.txt').open('x', encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
        result = run_custom_execution_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, gate_frames, simulate_gate_account)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    joint = save_joint_assessment(OUT, cfg, result)
    ranking = []
    for model in [*CANDIDATES, *[m for m in CONTROLS if m != 'BUY_HOLD']]:
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
    ranked.to_csv(OUT/'eleven_setting_joint_comparison.csv', index=False, encoding='utf-8-sig')
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
                source_ledger = pd.read_parquet(old/period/cost/f'{parent}_ledger.parquet')
                require(pd.DatetimeIndex(source_ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), '来源账本日期不同')
                source_equity = np.r_[cfg['initial_capital'], source_ledger.equity.iloc[:-1]].astype(float)
                source_shares = np.r_[0, source_ledger.shares.iloc[:-1]].astype(float)
                planned = source_shares+source.requested_quantity.to_numpy(float)
                source_price = frame.close.iloc[indices].to_numpy(float)
                require((planned >= 0).all() and (planned % cfg['lot'] == 0).all(), '来源计划份额无效')
                weights = planned*source_price/source_equity
                require((weights >= -1e-12).all() and (weights <= 1+1e-12).all(), '来源计划比例越界')
                weights = np.clip(weights, 0., 1.)
                weights[source.reference_weight.isna().to_numpy()] = np.nan
                originals[parent] = weights
            for control, (old, _) in CONTROLS.items():
                pd.testing.assert_frame_equal(pd.read_parquet(old/period/cost/f'{control}_ledger.parquet'),
                    pd.read_parquet(folder/f'{control}_ledger.parquet'))
            derived = []
            for r in frame.itertuples():
                valid = all(np.isfinite(x) for x in [r.close, r.previous_close, r.dividend]) and r.close > 0 and r.previous_close > 0 and r.dividend >= 0
                ret = (r.close+r.dividend)/r.previous_close-1 if valid else np.nan
                allowed = {}
                for direction in ['upper', 'lower']:
                    for percent in [0, 1]:
                        key = f'buy_allowed_{direction}_r{percent:02d}'
                        if valid:
                            value = Decimal(str(r.close))+Decimal(str(r.dividend))
                            boundary = Decimal(str(r.previous_close))*(Decimal(100)+Decimal(percent))/Decimal(100
                            )
                            allowed[key] = value <= boundary if direction == 'upper' else value >= boundary
                        else:
                            allowed[key] = False
                derived.append({'daily_total_simple': ret, 'market_inputs_known': valid,
                    **allowed})
            factors = pd.read_parquet(folder/'factors.parquet')
            for parent, weights in originals.items():
                np.testing.assert_allclose(factors.loc[indices, parent+'_planned_weight'], weights, atol=0, rtol=0, equal_nan=True)
            market = pd.DataFrame(derived)
            for column in market:
                np.testing.assert_allclose(factors[column], market[column], atol=1e-12, rtol=0, equal_nan=True)
            cache[period, cost] = originals, factors, market
        originals, factors, market = cache[period, cost]
        setting = SETTINGS[model]
        raw = np.zeros(len(indices))
        for parent, weight in setting['source_weights'].items():
            if weight > 0:
                raw += weight*originals[parent]
        raw = np.minimum(1., raw*setting['exposure_multiplier'])
        allowed = market[allowance_column(setting)].iloc[indices].to_numpy(bool)
        decisions = pd.read_parquet(folder/f'{model}_decisions.parquet')
        ledger = pd.read_parquet(folder/f'{model}_ledger.parquet')
        known = np.isfinite(raw)
        np.testing.assert_allclose(factors.loc[indices, model+'_target'], raw, atol=0, rtol=0, equal_nan=True)
        require(pd.DatetimeIndex(decisions.decision_time).equals(pd.DatetimeIndex(decisions.origin)+pd.Timedelta(hours=15, minutes=5)), '新决定时钟不同')
        require(decisions.loc[known, 'output_candidate'].eq(model).all() and decisions.loc[known, 'budget_source'].eq(setting['parent']).all(), '过滤候选身份混用')
        require(decisions.loc[known, 'used_band'].eq(setting['band']).all() and decisions.loc[known, 'return_threshold'].eq(setting['return_threshold']).all() and decisions.loc[known, 'filter_direction'].eq(setting['direction']).all(), '过滤设置不同')
        require(decisions.loc[known, 'exposure_multiplier'].eq(setting['exposure_multiplier']).all(), '投入倍率不同')
        prior, shares = np.r_[cfg['initial_capital'], ledger.equity.iloc[:-1]], np.r_[0, ledger.shares.iloc[:-1]].astype(int)
        normal = normal_requests(raw, prior, shares, frame.close.iloc[indices].to_numpy(float), {**cfg, 'primary': model})
        blocked = (normal > 0) & (shares > 0) & ~allowed
        np.testing.assert_array_equal(decisions.loc[known, 'normal_requested_quantity'], normal[known])
        np.testing.assert_array_equal(decisions.loc[known, 'addition_allowed'], allowed[known].astype(float))
        np.testing.assert_array_equal(decisions.loc[known, 'addition_suppressed'], blocked[known].astype(float))
        np.testing.assert_allclose(decisions.loc[known, 'observed_daily_return'], market.daily_total_simple.iloc[indices].to_numpy()[known], atol=1e-12, rtol=0, equal_nan=True)
        np.testing.assert_allclose(decisions.loc[known, 'own_close_equity'], prior[known], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(decisions.loc[known, 'own_close_shares'], shares[known])
        for j in np.flatnonzero(blocked):
            suppressions.append({'model': model, 'period': period, 'cost': cost, 'origin': decisions.origin.iloc[j],
                'source_target': raw[j], 'normal_requested_quantity': normal[j], 'own_shares': shares[j],
                'observed_daily_return': market.daily_total_simple.iloc[indices[j]], 'return_threshold': setting['return_threshold']})
        require(not (blocked & (shares == 0)).any(), '初次入场被加仓条件误拦')
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
            requests[(requests > 0) & (shares > 0) & ~active['allowed']] = 0
            return requests

        a, c, d, n = verify_requested_target_accounts(OUT, {**cfg, 'primary': model}, result, data, dividends,
            target_callback, request_callback, comparison_models=list(CONTROLS))
        for destination, rows in [(accounts, a), (cycles, c), (differences, d)]:
            destination.extend({'model': model, **row} for row in rows)
        count += n
    for name, rows in [('saved_account_checks.csv', accounts), ('saved_actual_cycles.csv', cycles),
        ('saved_comparison_differences.csv', differences), ('saved_buy_gate_checks.csv', checks), ('suppressed_buy_requests.csv', suppressions)]:
        pd.DataFrame(rows).to_csv(OUT/name, index=False, encoding='utf-8-sig')
    require(len(accounts) == 24 and count == 33876, '二十四账户核对范围不同')
    receipt = {'verified_at': now(), 'status': 'PASS_TWENTY_FOUR_SOURCE_ABLATION_ACCOUNTS_AND_INDEPENDENT_PLANS',
        'actual_accounts': 24, 'actual_decisions_checked': count, 'complete_actual_cycles': len(cycles),
        'simulated_fills_checked': sum(r['fills'] for r in checks), 'suppressed_buy_requests_checked': len(suppressions),
        'new_models_or_accounts': 0, 'independent_performance_validation': False, 'security_audit_performed': False,
        'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'verify': verify}[sys.argv[1]]()
