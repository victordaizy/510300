"""冻结一个方向确认辅助组合，复用保存来源并计算四个新资金账户。"""
import json
from pathlib import Path
from research.sign_confirmed_runs_auxiliary_inputs_v1 import PRIMARY, MODELS, CANDIDATES, sign_confirmed_runs_auxiliary_frames
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_sign_confirmed_runs_auxiliary_v1'
CONFIG = ROOT/'config/510300_sign_confirmed_runs_auxiliary_v1.json'
P143 = ROOT/'reports/research/510300_trend_noise_reference_blend_v1'
P165 = ROOT/'reports/research/510300_return_runs_state_v1'
P172 = ROOT/'reports/research/510300_return_sign_balance_v1'
P168 = ROOT/'reports/research/510300_runs_opportunity_union_v1'
P32 = ROOT/'reports/research/510300_rearmed_session_exit_v1'
PARENTS = {MODELS[0]: P143, MODELS[1]: P165, MODELS[2]: P172}
CONTROLS = {'RUNS_OPPORTUNITY_CAPPED_SUM': (P168, '第168轮不加方向条件的相加封顶'),
    MODELS[0]: (P143, '第143轮核心策略'), MODELS[1]: (P165, '第165轮收益强弱连续段'),
    MODELS[2]: (P172, '第172轮独立上涨日优势'), 'BUY_HOLD': (P32, '买入持有')}


def freeze():
    require(not CONFIG.exists(), '第173轮方向确认辅助组合已经冻结')
    prior_path = ROOT/'config/510300_return_sign_balance_v1.json'
    binding_path = ROOT/'config/510300_runs_opportunity_union_v1.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    binding = json.loads(binding_path.read_text(encoding='utf-8'))
    tests = json.loads((OUT/'tests_receipt.json').read_text(encoding='utf-8'))
    require(tests['exit_code'] == 0 and tests['passed'] == 7, '第173轮七项必要测试尚未通过')
    require((P172/'acceptance_outcome.json').is_file() and (P172/'saved_verification_receipt.json').is_file(),
        '第172轮尚未核对并关闭')
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {key: prior[key] for key in keys}
    cfg.update(study_id='510300_SIGN_CONFIRMED_RUNS_AUXILIARY_V1', round=173, primary=PRIMARY,
        candidate_configurations=1, candidate_models=list(CANDIDATES), registered_at=now(), decision_clock='15:05:00',
        parent_models=MODELS, combination='CORE_PLUS_SIGN_ALLOWED_AUXILIARY_CAPPED_AT_ONE',
        direction_column='positive_direction', direction_parent=MODELS[2],
        ignored_direction_parent_target='DIRECTION_ONLY_NO_SECOND_VOLATILITY_SCALING',
        unknown_parent='ANY_UNKNOWN_CORE_AUXILIARY_OR_DIRECTION_MAKES_COMBINED_TARGET_UNKNOWN',
        parent_feedback='NONE', new_model_fits=0, new_reference_accounts=0, source_budget_cny=0,
        goal_achieved=False, position_impact=0, rules='docs/510300_SIGN_CONFIRMED_RUNS_AUXILIARY_V1.md',
        independent_validation='NOT_ESTABLISHED', outer_exit_retry='RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE',
        reentry='ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT',
        previous_goal_turn_classification='PROGRESS_ROUND172_COMPLETED_CLOSED_AND_DELIVERED_FULL_GOAL_NOT_MET')
    paths = [Path(__file__), prior_path, binding_path,
        ROOT/'research/sign_confirmed_runs_auxiliary_inputs_v1.py', ROOT/'research/saved_target_batch_runner_v1.py',
        ROOT/'research/saved_parent_target_alignment_v1.py', ROOT/'research/event_clock_account_v1.py',
        ROOT/'research/adaptive_allocation_v1.py', ROOT/'research/intraday_overnight_increment_v1.py',
        ROOT/'tests/test_sign_confirmed_runs_auxiliary_v1.py', ROOT/'tests/test_heikin_price_state_v1.py',
        ROOT/'tests/test_trend_reference_router_v1.py', ROOT/'tests/test_runs_reference_blend_v1.py',
        OUT/'tests_receipt.json', ROOT/cfg['features'], ROOT/cfg['dividends'], ROOT/cfg['rules'],
        ROOT/'docs/510300_SIGN_CONFIRMED_RUNS_AUXILIARY_NEXT_20260911.md',
        ROOT/'config/510300_research_authority_v6.json', P172/'acceptance_outcome.json',
        P168/'saved_verification_receipt.json', P168/'acceptance_outcome.json',
        ROOT/'docs/510300_RUNS_REFERENCE_BLEND_EXECUTION_CLARIFICATION_20260910.md']
    known = {str(Path(item['path'])): item['sha256'] for cfg_old in [binding, prior] for item in cfg_old['frozen_files']}
    for key in ['features', 'dividends']:
        require(digest(ROOT/cfg[key]) == known[str(Path(cfg[key]))], '固定行情或分红版本改变')
    for model, folder in PARENTS.items():
        parent_cfg_path = ROOT/'config'/f'{folder.name}.json'
        parent_cfg = json.loads(parent_cfg_path.read_text(encoding='utf-8'))
        require(parent_cfg['primary'] == model, '保存父规则身份不同')
        for key in ['features', 'dividends', 'evaluation_start', 'data_cutoff', 'earlier_start', 'earlier_terminal', 'costs']:
            require(parent_cfg[key] == cfg[key], '来源、日历或费用不同')
        parent_paths = [parent_cfg_path, ROOT/parent_cfg['rules'], folder/'saved_verification_receipt.json']
        for period in ['evaluation', 'earlier_diagnostic']:
            for cost in cfg['costs']:
                parent_paths.append(folder/period/cost/f'{model}_decisions.parquet')
        for path in parent_paths:
            if model != MODELS[2]:
                require(digest(path) == known[str(path.relative_to(ROOT))], '旧父规则或收盘目标版本改变')
        paths.extend(parent_paths)
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in cfg['costs']:
            for model, (folder, _) in CONTROLS.items():
                path = folder/period/cost/f'{model}_ledger.parquet'
                old_hash = known.get(str(path.relative_to(ROOT)))
                if old_hash is not None:
                    require(digest(path) == old_hash, '已有对照账户版本改变')
                paths.append(path)
    cfg['frozen_files'] = [{'path': str(path.relative_to(ROOT)), 'sha256': digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print('第173轮方向确认辅助组合已冻结，尚未生成四个新账户。', flush=True)


def run():
    return run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, sign_confirmed_runs_auxiliary_frames)


if __name__ == '__main__':
    import sys
    {'freeze': freeze, 'run': run}[sys.argv[1]]()
