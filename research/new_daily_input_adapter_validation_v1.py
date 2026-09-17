"""固定免费输入适配实现，运行一次真实日历检查并保存结果。"""
import argparse
import json
from pathlib import Path
import time

from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.new_daily_input_adapter_v1 import ROOT, read, run as check_and_run

OUT = ROOT / 'reports/research/510300_new_daily_input_adapter_v1'
CONFIG = ROOT / 'config/510300_new_daily_input_adapter_v1.json'
RUNTIME = ROOT / 'config/510300_new_daily_input_adapter_runtime_v1.json'
DAILY = ROOT / 'reports/research/510300_fixed_daily_continuation_v1'


def prepare():
    require(not CONFIG.exists() and not RUNTIME.exists(), '输入适配已登记，请接续保存状态')
    index_path = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
    index = read(index_path)
    require(index['latest_completed_round']['round'] == 215 and not index['running_studies'], '前序入口尚未完整交付')
    require(read(OUT / 'tests_receipt.json')['passed'] == 10, '十项新增验证未通过')
    source = ROOT / 'reports/research/510300_september_monthly_continuation_v1'
    inputs = ROOT / 'reports/research/510300_post_selection_extension_inputs_v1'
    coverage = read(ROOT / 'data/reference/510300_dividends_coverage.json')
    prior_coverage = read(inputs / 'candidate_dividend_coverage.json')
    require(prior_coverage['coverage']['requested_coverage_end'] == '2026-09-11' and
        prior_coverage['ledger_sha256'] == coverage['distribution_file_sha256'] == digest(ROOT / 'data/reference/510300_dividends.csv'), '旧接纳分红身份或覆盖不同')
    coverage.update(coverage_end='2026-09-11', coverage_extension_evidence=str((inputs / 'candidate_dividend_coverage.json').relative_to(ROOT)),
        original_global_coverage_preserved=True)
    write_json(OUT / 'initial_dividend_coverage.json', coverage, exclusive=True)
    runtime = {'adapter': 'research/new_daily_input_adapter_v1.py', 'registered_at': now(),
        'output_root': str(DAILY.relative_to(ROOT)), 'calendar': 'data/reference/sse_trade_calendar_2026.csv',
        'official_configuration': 'config/510300_official_dividend_coverage_refresh_v1.json',
        'strategy_configuration': 'config/510300_september_monthly_continuation_v1.json',
        'research_origin': index['latest_fixed_research_origin'],
        'initial_source': {'prices': str((inputs / 'candidate_prices.parquet').relative_to(ROOT)),
            'features': str((inputs / 'candidate_features.parquet').relative_to(ROOT)), 'dividends': 'data/reference/510300_dividends.csv',
            'coverage': str((OUT / 'initial_dividend_coverage.json').relative_to(ROOT)),
            'ridge_models': str((source / 'ridge_models.json').relative_to(ROOT)), 'within_models': str((source / 'within_models.json').relative_to(ROOT)),
            'source_folder': str(source.relative_to(ROOT))}, 'new_parameter_search': False, 'source_budget_cny': 0,
        'position_impact': 0, 'goal_achieved': False}
    write_json(RUNTIME, runtime, exclusive=True)
    old = read(ROOT / 'config/510300_fixed_date_continuation_validation_v1.json')
    fields = ['evaluation_start', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'annual_return_target', 'costs', 'dividends',
        'earlier_start', 'earlier_terminal', 'weight_band', 'features', 'primary', 'candidate_models']
    cfg = {k: old[k] for k in fields}
    cfg.update(study_id='510300_NEW_DAILY_INPUT_ADAPTER_V1', round=216, registered_at=now(),
        rules='docs/510300_NEW_DAILY_INPUT_ADAPTER_V1.md', data_cutoff='2026-09-11', candidate_configurations=0,
        runtime_settings=str(RUNTIME.relative_to(ROOT)), new_model_fits=0, new_reference_accounts=0,
        goal_achieved=False, independent_validation='NOT_ESTABLISHED', position_impact=0)
    files = [Path(__file__), ROOT / 'research/new_daily_input_adapter_v1.py', ROOT / 'tests/test_new_daily_input_adapter_v1.py',
        ROOT / 'research/fixed_date_continuation_v1.py', ROOT / 'research/adaptive_allocation_v1.py',
        ROOT / 'research/official_dividend_coverage_refresh_v1.py', ROOT / cfg['rules'], OUT / 'tests_receipt.json', RUNTIME,
        OUT / 'initial_dividend_coverage.json', ROOT / runtime['research_origin'], ROOT / runtime['official_configuration'],
        source / 'result.json', source / 'saved_verification_receipt.json', inputs / 'candidate_dividend_coverage.json']
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in files]
    write_json(CONFIG, cfg, exclusive=True)
    index['running_studies'] = [{'round': 216, 'study': cfg['study_id'], 'config': str(CONFIG.relative_to(ROOT)), 'status': 'FROZEN_ADAPTER_NOT_CHECKED'}]
    index['next_work'].update(registered=True, status='FREE_INPUT_ADAPTER_FROZEN')
    write_json(index_path, index)
    print('免费输入适配已固定，直接执行一次实际日历检查。', flush=True)


def run():
    require(not (OUT / 'result.json').exists(), '本轮检查已经完成，不重复建立研究结果')
    cfg = read(CONFIG)
    for item in cfg['frozen_files']:
        require(digest(ROOT / item['path']) == item['sha256'], '固定适配入口或依据改变')
    began = time.perf_counter()
    checked = check_and_run(RUNTIME)
    write_json(OUT / 'actual_calendar_check.json', checked, exclusive=True)
    require(checked['status'] == 'NO_NEW_COMPLETE_TRADING_DAY', '当前日期状态发生变化，先使用实际新增结果再交付')
    old = read(ROOT / 'reports/research/510300_september_monthly_continuation_v1/result.json')
    result = {k: old[k] for k in ['candidate_models', 'all_metrics', 'earlier_diagnostics', 'primary', 'post_selected_best_base']}
    result.update(study_id=cfg['study_id'], completed_at=now(), status='COMPLETED_FREE_INPUT_ADAPTER_NO_NEW_COMPLETE_DAY',
        candidate_configurations=0, evaluation_accounts=0, new_accounts_generated=0, reused_control_accounts=2,
        earlier_diagnostic_accounts=0, new_earlier_diagnostic_accounts=0, new_model_fits=0, new_reference_accounts=0,
        metrics_reused_from_round214=True, new_strategy_performance_computed=False, new_observed_trading_days=0,
        strict_forward_evidence_days=0, independent_validation='NOT_ESTABLISHED', goal_achieved=False, position_impact=0,
        network_requests=0, actual_calendar_check=checked, compatible_saved_raw_sources=2, compatible_saved_price_dates=30,
        compatible_price_cells=3476*15, compatible_feature_cells=3476*63, runtime_entrypoint='research/new_daily_input_adapter_v1.py',
        runtime_settings=str(RUNTIME.relative_to(ROOT)), run_seconds=time.perf_counter()-began,
        run_seconds_scope='当前实际官方日历检查与无新增日期返回；不含实现或十项兼容测试',
        expected_next_useful_check_after=checked['next_useful_check_after'], prior_goal_turn_classification='PROGRESS_ROUND215_DATE_ENTRY_VERIFIED',
        current_goal_turn_classification='PROGRESS_INPUT_ADAPTER_IMPLEMENTED_TESTED_AND_REAL_NO_NEW_DATE_CHECKED',
        remaining='需2026年9月14日或之后完整新收盘、实际免费来源接纳与后续独立样本积累')
    write_json(OUT / 'result.json', result, exclusive=True)
    print(json.dumps({'status':result['status'],'actual_check':checked,'seconds':result['run_seconds']},ensure_ascii=False),flush=True)


def verify():
    result = read(OUT / 'result.json')
    checked = read(OUT / 'actual_calendar_check.json')
    require(checked == read(DAILY / 'latest_check.json'), '实际入口保存回执不同')
    require(checked['network_requests'] == checked['new_account_rows'] == 0 and
        checked['latest_complete_official_day'] == checked['accepted_price_cutoff'] == '2026-09-11', '实际无新增日期状态不一致')
    require(not (DAILY / '2026-09-14').exists() and not (DAILY / 'latest_completed.json').exists(), '未出现数据时提前生成了日线续算结果')
    old = read(ROOT / 'reports/research/510300_september_monthly_continuation_v1/result.json')
    require(result['all_metrics'] == old['all_metrics'] and result['new_observed_trading_days'] == 0, '旧结果引用或样本数量错误')
    for item in read(CONFIG)['frozen_files']:
        require(digest(ROOT / item['path']) == item['sha256'], '固定程序或输入改变')
    write_json(OUT / 'saved_verification_receipt.json', {'verified_at':now(),
        'status':'PASS_SAVED_PARSER_INPUT_COMPATIBILITY_AND_ACTUAL_ZERO_REQUEST_NO_NEW_DAY',
        'new_tests':10, 'prior_tests_reused':19, 'network_requests':0, 'new_accounts':0, 'new_observed_trading_days':0,
        'actual_latest_complete_official_day':'2026-09-11', 'independent_performance_validation':False}, exclusive=True)
    print('免费输入适配保存核对通过：实际当前零新收盘、零请求、零新账户。', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='固定免费输入适配，保存当前实际检查及必要验证。')
    parser.add_argument('action', choices=['prepare', 'run', 'verify'])
    globals()[parser.parse_args().action]()
