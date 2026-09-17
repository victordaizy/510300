"""冻结月度成熟周期预算，读取保存父收益及目标后计算四个独立账户。"""
import json
import pandas as pd
import numpy as np
from research.adaptive_allocation_v1 import normalize_dividends
from pathlib import Path
from research.runs_closed_cycle_budget_inputs_v1 import PRIMARY, MODELS, CANDIDATES, mature_saved_cycles, runs_closed_cycle_budget_frames
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_runs_closed_cycle_budget_v1'
CONFIG = ROOT/'config/510300_runs_closed_cycle_budget_v1.json'
P143 = ROOT/'reports/research/510300_trend_noise_reference_blend_v1'
P165 = ROOT/'reports/research/510300_return_runs_state_v1'
P169 = ROOT/'reports/research/510300_runs_net_profit_budget_v1'
P168 = ROOT/'reports/research/510300_runs_opportunity_union_v1'
P167 = ROOT/'reports/research/510300_runs_covariance_budget_v1'
P166 = ROOT/'reports/research/510300_runs_reference_blend_v1'
P32 = ROOT/'reports/research/510300_rearmed_session_exit_v1'
PARENTS = {MODELS[0]: P143, MODELS[1]: P165}
CONTROLS = {'RUNS_NET_PROFIT_BUDGET': (P169, '第169轮月度正均值预算'),
    'RUNS_OPPORTUNITY_CAPPED_SUM': (P168, '第168轮相加封顶'),
    'RUNS_COVARIANCE_BUDGET': (P167, '第167轮月度风险预算'), 'RUNS_REFERENCE_BLEND': (P166, '第166轮固定各半组合'), MODELS[0]: (P143, '第143轮趋势波动连续预算'),
    MODELS[1]: (P165, '第165轮收益强弱连续段'), 'BUY_HOLD': (P32, '买入持有')}


def freeze():
    require(not CONFIG.exists(), '第170轮月度成熟周期预算已经冻结')
    prior_path = ROOT/'config/510300_runs_net_profit_budget_v1.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    tests = json.loads((OUT/'tests_receipt.json').read_text(encoding='utf-8'))
    require(tests['exit_code'] == 0 and tests['passed'] == 7, '第170轮七项必要测试尚未通过')
    require((P169/'acceptance_outcome.json').is_file() and (P169/'saved_verification_receipt.json').is_file(), '第169轮正均值预算尚未完成核对及关闭')
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {key: prior[key] for key in keys}
    cfg.update(study_id='510300_RUNS_CLOSED_CYCLE_BUDGET_V1', round=170, primary=PRIMARY, candidate_configurations=1,
        candidate_models=list(CANDIDATES), registered_at=now(), decision_clock='15:05:00', parent_models=MODELS,
        cycle_window=20, minimum_mature_cycles=5, initial_budgets=[.5, .5], budget_update='FIRST_TRADING_DAY_MONTH_CLOSE',
        budget_source_cost='BASE', return_clock='CLOSE_ONLY_TERMINAL_OPEN_EXCLUDED',
        cycle_score='POSITIVE_SUM_RETURNS_OVER_ABSOLUTE_RETURN_SUM',
        cycle_maturity='MAX_ACTUAL_EXIT_AND_ALL_OWNED_EX_DIVIDEND_DATES', nonpositive_scores='BOTH_BUDGETS_ZERO',
        new_model_fits=0, new_reference_accounts=0,
        combination='MONTHLY_POSITIVE_EFFICIENCY_OF_MATURE_CLOSED_PARENT_CYCLES',
        unknown_parent='ANY_UNKNOWN_PARENT_MAKES_COMBINED_TARGET_UNKNOWN', parent_feedback='NONE',
        source_budget_cny=0, goal_achieved=False, position_impact=0, rules='docs/510300_RUNS_CLOSED_CYCLE_BUDGET_V1.md',
        independent_validation='NOT_ESTABLISHED', outer_exit_retry='RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE',
        reentry='ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT',
        previous_goal_turn_classification='PROGRESS_ROUND169_COMPLETED_CLOSED_AND_DELIVERED_FULL_GOAL_NOT_MET')
    paths = [Path(__file__), prior_path, ROOT/'research/runs_closed_cycle_budget_inputs_v1.py',
        ROOT/'research/saved_target_batch_runner_v1.py', ROOT/'research/saved_parent_target_alignment_v1.py',
        ROOT/'research/event_clock_account_v1.py', ROOT/'research/adaptive_allocation_v1.py',
        ROOT/'research/intraday_overnight_increment_v1.py', ROOT/'tests/test_runs_closed_cycle_budget_v1.py',
        ROOT/'tests/test_heikin_price_state_v1.py', ROOT/'tests/test_trend_reference_router_v1.py',
        OUT/'tests_receipt.json', ROOT/cfg['features'], ROOT/cfg['dividends'], ROOT/cfg['rules'],
        ROOT/'docs/510300_RUNS_CLOSED_CYCLE_BUDGET_NEXT_20260910.md', ROOT/'config/510300_research_authority_v6.json',
        P169/'acceptance_outcome.json', P169/'saved_verification_receipt.json',
        ROOT/'research/runs_covariance_budget_inputs_v1.py', ROOT/'tests/test_runs_covariance_budget_v1.py',
        ROOT/'tests/test_runs_reference_blend_v1.py', ROOT/'research/runs_reference_blend_inputs_v1.py',
        ROOT/'docs/510300_RUNS_REFERENCE_BLEND_EXECUTION_CLARIFICATION_20260910.md']
    known = {str(Path(item['path'])): item['sha256'] for item in prior['frozen_files']}
    for key in ['features', 'dividends']:
        require(digest(ROOT/cfg[key]) == known[str(Path(cfg[key]))], '固定行情或分红版本改变')
    for model, folder in PARENTS.items():
        parent_cfg_path = ROOT/'config'/f'{folder.name}.json'
        parent_cfg = json.loads(parent_cfg_path.read_text(encoding='utf-8'))
        require(parent_cfg['primary'] == model, '保存父规则身份不同')
        for key in ['features', 'dividends', 'evaluation_start', 'data_cutoff', 'earlier_start', 'earlier_terminal', 'costs']:
            require(parent_cfg[key] == cfg[key], '父规则与组合的来源、日历或费用不同')
        parent_metadata = [parent_cfg_path, ROOT/parent_cfg['rules'], folder/'saved_verification_receipt.json']
        for path in parent_metadata:
            require(digest(path) == known[str(path.relative_to(ROOT))], '冻结父规则或核对回执版本改变')
        paths.extend(parent_metadata)
        paths.append(folder/'saved_actual_cycles.csv')
        for period in ['evaluation', 'earlier_diagnostic']:
            for cost in cfg['costs']:
                path = folder/period/cost/f'{model}_decisions.parquet'
                require(digest(path) == known[str(path.relative_to(ROOT))], '冻结父收盘目标版本改变')
                paths.append(path)
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in cfg['costs']:
            for model, (folder, _) in CONTROLS.items():
                path = folder/period/cost/f'{model}_ledger.parquet'
                if model != 'RUNS_NET_PROFIT_BUDGET':
                    require(digest(path) == known[str(path.relative_to(ROOT))], '已有对照账户版本改变')
                paths.append(path)
    cfg['frozen_files'] = [{'path': str(path.relative_to(ROOT)), 'sha256': digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print('第170轮最近二十个成熟周期、至少五个的月度预算已冻结，尚未生成新账户。', flush=True)


def run():
    monthly_events, cycle_selections, maturity_sources = [], [], []

    def build_frames(frame, sources, cfg, start):
        require(start in [cfg['evaluation_start'], cfg['earlier_start']], '周期预算评价起点不同')
        period = 'evaluation' if start == cfg['evaluation_start'] else 'earlier_diagnostic'
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        dividends = normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
        source_cycles = {}
        for model, folder in PARENTS.items():
            ledger = pd.read_parquet(folder/period/'BASE'/f'{model}_ledger.parquet').assign(source_cost='BASE', source_model=model)
            saved = pd.read_csv(folder/'saved_actual_cycles.csv', parse_dates=['entry_date', 'exit_date'])
            saved = saved[saved.period.eq(period) & saved.cost.eq('BASE')].assign(source_model=model)
            checked = mature_saved_cycles(frame, ledger, saved, dividends, model, first)
            source_cycles[model] = checked
            maturity_sources.extend({'period': period, **row} for row in checked.to_dict('records'))
        factors, summaries, events, selected = runs_closed_cycle_budget_frames(frame, sources, source_cycles, cfg, start)
        monthly_events.extend({'period': period, 'budget_source_cost': 'BASE', **event} for event in events)
        cycle_selections.extend({'period': period, **row} for row in selected)
        return factors, summaries

    result = run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, build_frames)
    for name, rows in [('monthly_budget_updates.csv', monthly_events), ('monthly_cycle_selections.csv', cycle_selections),
        ('cycle_maturity_sources.csv', maturity_sources)]:
        pd.DataFrame(rows).to_csv(OUT/name, index=False, encoding='utf-8-sig', mode='x')
    print(f'已保存{len(monthly_events)}条月度预算、{len(cycle_selections)}条周期入选和{len(maturity_sources)}条成熟周期来源。', flush=True)
    return result


if __name__ == '__main__':
    import sys
    {'freeze': freeze, 'run': run}[sys.argv[1]]()
