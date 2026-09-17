"""十四套有限风险参数一次冻结、一次批量计算、一次汇总核对。"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.risk_window_clock_batch_inputs_v1 import CANDIDATES, MODELS, PARENT, PRIMARY, SETTINGS, SOURCE, batch_frames
from research.saved_target_account_checks_v1 import verify_saved_target_accounts
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from scripts.verify_round195_20260913 import prices_and_costs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_risk_window_clock_batch_v1'
CONFIG = ROOT/'config/510300_risk_window_clock_batch_v1.json'
PARENTS = {PARENT: SOURCE}
CONTROLS = {PARENT: (SOURCE, '原174任一方向确认组合'),
    'ACCOUNT_VOLATILITY_EXPOSURE': (ROOT/'reports/research/510300_account_volatility_exposure_v1', '原181每日六十日10%预算'),
    'EPISODE_ACCOUNT_RISK_BUDGET': (ROOT/'reports/research/510300_episode_account_risk_budget_v1', '原182区间六十日10%预算'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '本批已登记或运行')
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    test = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_risk_window_clock_batch_v1.py', '-q', '-p', 'no:cacheprovider'],
                          cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
    print(test.stdout, flush=True)
    require(test.returncode == 0 and '5 passed' in test.stdout, '五项必要测试没有通过')
    write_json(OUT/'tests_receipt.json', {'recorded_at': now(), 'exit_code': test.returncode, 'passed': 5,
        'seconds': time.perf_counter()-began}, exclusive=True)
    old_path = ROOT/'config/510300_return_confirmation_auxiliary_batch_v1.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {k: old[k] for k in keys}
    cfg.update(study_id='510300_RISK_WINDOW_CLOCK_BATCH_V1', round=196, registered_at=now(), primary=PRIMARY,
        candidate_models=list(CANDIDATES), candidate_settings=SETTINGS, candidate_configurations=14, parent_models=MODELS,
        decision_clock='15:05:00', annual_return_target=.10, new_model_fits=0, new_reference_accounts=0,
        goal_achieved=False, position_impact=0, source_budget_cny=0, independent_validation='NOT_ESTABLISHED',
        evidence_class='EXPLICIT_RETROSPECTIVE_FINITE_GRID_CALIBRATION_FOUR_OLD_SETTINGS_NOT_RERUN',
        source_preflight='reports/research/510300_saved_combination_fast_screen_20260913/result.json',
        rules='docs/510300_RISK_WINDOW_CLOCK_BATCH_V1.md')
    with (ROOT/cfg['rules']).open('x', encoding='utf-8') as stream:
        stream.write((ROOT/'docs/510300_RISK_WINDOW_CLOCK_BATCH_NEXT_20260913.md').read_text(encoding='utf-8'))
        stream.write('\n\n## 十四套固定候选清单\n\n|候选|风险窗口|风险预算|调整方式|\n|---|---:|---:|---|\n')
        for model, s in SETTINGS.items():
            stream.write(f"|{CANDIDATES[model]}|{s['window']}日|{s['budget_percent']}%|{'每日调整' if s['clock']=='DAILY' else '区间固定'}|\n")
        stream.write('\n\n## 原174来源的全部中文因素与进出规则\n\n')
        stream.write((ROOT/old['rules']).read_text(encoding='utf-8'))
    paths = [Path(__file__), ROOT/'research/risk_window_clock_batch_inputs_v1.py',
        ROOT/'research/account_volatility_exposure_inputs_v1.py', ROOT/'research/episode_account_risk_budget_inputs_v1.py',
        ROOT/'research/saved_parent_target_alignment_v1.py', ROOT/'research/saved_target_batch_runner_v1.py',
        ROOT/'research/saved_target_account_checks_v1.py', ROOT/'research/event_clock_account_v1.py',
        ROOT/'research/adaptive_allocation_v1.py', ROOT/'research/intraday_overnight_increment_v1.py',
        ROOT/'research/joint_account_acceptance_v1.py', ROOT/'scripts/verify_round195_20260913.py',
        ROOT/'tests/test_risk_window_clock_batch_v1.py', old_path, ROOT/old['rules'],
        OUT/'tests_output.txt', OUT/'tests_receipt.json']
    paths += [ROOT/cfg[k] for k in ['features', 'dividends', 'rules', 'source_preflight']]
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in cfg['costs']:
            paths.append(SOURCE/period/cost/f'{PARENT}_decisions.parquet')
            paths += [folder/period/cost/f'{model}_ledger.parquet' for model, (folder, _) in CONTROLS.items()]
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    p = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(p.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 195 and not index['running_studies'], '前序或运行状态不同')
    index['running_studies'] = [{'round': 196, 'study': cfg['study_id'], 'status': 'FROZEN_NOT_STARTED', 'config': str(CONFIG.relative_to(ROOT))}]
    index['next_work'] = {'candidate_round': 196, 'registered': True, 'status': 'RISK_WINDOW_CLOCK_BATCH_FROZEN',
        'source': cfg['rules'], 'focus': '一次批量比较十四套未算过的风险窗口与预算时点设置',
        'planned_settings': 14, 'planned_new_accounts': 56, 'planned_new_model_fits': 0,
        'planned_new_reference_accounts': 0, 'external_data_required': False}
    write_json(p, index)
    print('第196轮十四套设置已一次冻结；四套旧设置不重跑。', flush=True)


def run():
    result = run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, batch_frames)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    joint = save_joint_assessment(OUT, cfg, result)
    print(json.dumps({'四场景联合点值通过': [m for m, v in joint['candidates'].items() if v['four_scenario_joint_pass']]}, ensure_ascii=False), flush=True)


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(), '本批已经完成核对')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path']) == item['sha256'], '冻结来源改变')
    data = pd.read_parquet(ROOT/cfg['features'])
    dividends = normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    cached, checks = {}, []

    def expected(period, cost_id, frame, start, model):
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        key = (period, cost_id)
        if key not in cached:
            parent = pd.read_parquet(SOURCE/period/cost_id/f'{PARENT}_decisions.parquet')
            old_ledger = pd.read_parquet(SOURCE/period/cost_id/f'{PARENT}_ledger.parquet')
            np.testing.assert_array_equal(parent.origin_index, indices)
            require(pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), '父来源判断日期不符')
            require(pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), '父来源执行日期不符')
            require(pd.DatetimeIndex(old_ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])), '父收益日期不符')
            returns = np.r_[np.full(first, np.nan), old_ledger.net_return.to_numpy(float)]
            source = np.full(len(frame), np.nan)
            source[indices] = parent.reference_weight
            saved = pd.read_parquet(OUT/period/cost_id/'factors.parquet')
            np.testing.assert_allclose(saved.source_realized_net_return, returns, atol=0, rtol=0, equal_nan=True)
            np.testing.assert_allclose(saved[PARENT+'_parent_target'], source, atol=0, rtol=0, equal_nan=True)
            windows = {}
            for window in [30, 60, 120]:
                risk = np.full(len(frame), np.nan)
                for t in range(first+window-1, len(frame)):
                    sample = returns[t-window+1:t+1]
                    if np.isfinite(sample).all():
                        mean = math.fsum(sample)/window
                        risk[t] = math.sqrt(math.fsum((v-mean)**2 for v in sample)/(window-1)*242)
                windows[window] = risk
            for control, (folder, _) in CONTROLS.items():
                pd.testing.assert_frame_equal(pd.read_parquet(folder/period/cost_id/f'{control}_ledger.parquet'),
                    pd.read_parquet(OUT/period/cost_id/f'{control}_ledger.parquet'))
            cached[key] = (source, saved, windows)
        source, saved, windows = cached[key]
        setting = SETTINGS[model]
        risk = windows[setting['window']]
        current = np.full(len(frame), np.nan)
        current[:first+setting['window']-1] = 1.
        current[np.isfinite(risk) & (risk == 0)] = 1.
        positive = np.isfinite(risk) & (risk > 0)
        current[positive] = setting['budget_percent']/100/risk[positive]
        target, applied = np.full(len(frame), np.nan), current.copy()
        if setting['clock'] == 'DAILY':
            target = np.minimum(1., source*current)
            target[source == 0] = 0.
        else:
            applied[:] = np.nan
            active, fixed = False, np.nan
            for t, value in enumerate(source):
                if np.isnan(value):
                    if active:
                        applied[t] = fixed
                    continue
                if value == 0:
                    target[t], active, fixed = 0., False, np.nan
                    continue
                if not active:
                    active, fixed = True, current[t]
                applied[t] = fixed
                if np.isfinite(fixed):
                    target[t] = min(1., value*fixed)
        for suffix, value in [('target', target), ('risk', risk), ('current_multiplier', current), ('applied_multiplier', applied)]:
            np.testing.assert_allclose(saved[model+'_'+suffix], value, atol=1e-10, rtol=1e-12, equal_nan=True)
        ledger = pd.read_parquet(OUT/period/cost_id/f'{model}_ledger.parquet')
        checks.append({'model': model, 'period': period, 'cost': cost_id, 'decisions': len(indices),
            'fills': prices_and_costs(ledger, frame, first, cfg, cfg['costs'][cost_id])})
        return target

    accounts, cycles, differences, count = [], [], [], 0
    for model in CANDIDATES:
        def selected(period, cost_id, frame, start):
            return expected(period, cost_id, frame, start, model)
        a, c, d, n = verify_saved_target_accounts(OUT, {**cfg, 'primary': model}, result, data, dividends,
            selected, comparison_models=['ACCOUNT_VOLATILITY_EXPOSURE', 'EPISODE_ACCOUNT_RISK_BUDGET', 'BUY_HOLD'])
        for destination, source in [(accounts, a), (cycles, c), (differences, d)]:
            destination.extend({'model': model, **row} for row in source)
        count += n
    for filename, records in [('saved_account_checks.csv', accounts), ('saved_actual_cycles.csv', cycles),
        ('saved_comparison_differences.csv', differences), ('saved_target_checks.csv', checks)]:
        pd.DataFrame(records).to_csv(OUT/filename, index=False, encoding='utf-8-sig')
    require(len(accounts) == 56 and count == 79044, '完整批次核对数量不同')
    receipt = {'verified_at': now(), 'status': 'PASS_FIFTY_SIX_ACCOUNTS_AND_FOUR_REUSED_REFERENCE_FAMILIES',
        'actual_accounts': 56, 'actual_decisions_checked': count, 'complete_actual_cycles': len(cycles),
        'simulated_fills_checked': sum(r['fills'] for r in checks), 'new_models_or_accounts': 0,
        'independent_performance_validation': False, 'security_audit_performed': False,
        'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'verify': verify}[sys.argv[1]]()
