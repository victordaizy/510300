"""冻结两种进场机会合并，读取保存父目标并计算八个独立资金账户。"""
import json
from pathlib import Path
from research.runs_opportunity_union_inputs_v1 import PRIMARY, MODELS, CANDIDATES, runs_opportunity_union_frames
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_runs_opportunity_union_v1'
CONFIG = ROOT/'config/510300_runs_opportunity_union_v1.json'
P143 = ROOT/'reports/research/510300_trend_noise_reference_blend_v1'
P165 = ROOT/'reports/research/510300_return_runs_state_v1'
P167 = ROOT/'reports/research/510300_runs_covariance_budget_v1'
P166 = ROOT/'reports/research/510300_runs_reference_blend_v1'
P32 = ROOT/'reports/research/510300_rearmed_session_exit_v1'
PARENTS = {MODELS[0]: P143, MODELS[1]: P165}
CONTROLS = {'RUNS_COVARIANCE_BUDGET': (P167, '第167轮月度风险预算'),
    'RUNS_REFERENCE_BLEND': (P166, '第166轮固定各半'), MODELS[0]: (P143, '第143轮趋势波动连续预算'),
    MODELS[1]: (P165, '第165轮收益强弱连续段'), 'BUY_HOLD': (P32, '买入持有')}


def freeze():
    require(not CONFIG.exists(), '第168轮两套进场合并规则已经冻结')
    prior_path = ROOT/'config/510300_runs_covariance_budget_v1.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    tests = json.loads((OUT/'tests_receipt.json').read_text(encoding='utf-8'))
    require(tests['exit_code'] == 0 and tests['passed'] == 6, '第168轮六项必要测试尚未通过')
    require((P167/'acceptance_outcome.json').is_file() and (P167/'saved_verification_receipt.json').is_file(), '第167轮月度风险预算尚未完成核对及关闭')
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {key: prior[key] for key in keys}
    cfg.update(study_id='510300_RUNS_OPPORTUNITY_UNION_V1', round=168, primary=PRIMARY, candidate_configurations=2,
        candidate_models=list(CANDIDATES), registered_at=now(), decision_clock='15:05:00', parent_models=MODELS,
        target_formulas={PRIMARY: 'MAXIMUM', 'RUNS_OPPORTUNITY_CAPPED_SUM': 'SUM_CAPPED_AT_ONE'},
        new_model_fits=0, new_reference_accounts=0,
        combination='MAXIMUM_AND_CAPPED_SUM_OF_SAVED_PARENT_TARGETS',
        unknown_parent='ANY_UNKNOWN_PARENT_MAKES_COMBINED_TARGET_UNKNOWN', parent_feedback='NONE',
        source_budget_cny=0, goal_achieved=False, position_impact=0, rules='docs/510300_RUNS_OPPORTUNITY_UNION_V1.md',
        independent_validation='NOT_ESTABLISHED', outer_exit_retry='RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE',
        reentry='ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT',
        previous_goal_turn_classification='PROGRESS_ROUND167_COMPLETED_CLOSED_AND_DELIVERED_FULL_GOAL_NOT_MET')
    paths = [Path(__file__), prior_path, ROOT/'research/runs_opportunity_union_inputs_v1.py',
        ROOT/'research/saved_target_batch_runner_v1.py', ROOT/'research/saved_parent_target_alignment_v1.py',
        ROOT/'research/event_clock_account_v1.py', ROOT/'research/adaptive_allocation_v1.py',
        ROOT/'research/intraday_overnight_increment_v1.py', ROOT/'tests/test_runs_opportunity_union_v1.py',
        ROOT/'tests/test_heikin_price_state_v1.py', ROOT/'tests/test_trend_reference_router_v1.py',
        OUT/'tests_receipt.json', ROOT/cfg['features'], ROOT/cfg['dividends'], ROOT/cfg['rules'],
        ROOT/'docs/510300_RUNS_OPPORTUNITY_UNION_NEXT_20260910.md', ROOT/'config/510300_research_authority_v6.json',
        P167/'acceptance_outcome.json', P167/'saved_verification_receipt.json',
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
            require(digest(path) == known[str(path.relative_to(ROOT))], '冻结父规则或回执改变')
        paths.extend(parent_metadata)
        for period in ['evaluation', 'earlier_diagnostic']:
            for cost in cfg['costs']:
                path = folder/period/cost/f'{model}_decisions.parquet'
                require(digest(path) == known[str(path.relative_to(ROOT))], '冻结父收盘目标改变')
                paths.append(path)
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in cfg['costs']:
            for model, (folder, _) in CONTROLS.items():
                path = folder/period/cost/f'{model}_ledger.parquet'
                if model != 'RUNS_COVARIANCE_BUDGET':
                    require(digest(path) == known[str(path.relative_to(ROOT))], '已有对照账户版本改变')
                paths.append(path)
    cfg['frozen_files'] = [{'path': str(path.relative_to(ROOT)), 'sha256': digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print('第168轮较大值与相加封顶两套规则已冻结，尚未生成八个新账户。', flush=True)


def run():
    return run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, runs_opportunity_union_frames)


if __name__ == '__main__':
    import sys
    {'freeze': freeze, 'run': run}[sys.argv[1]]()
