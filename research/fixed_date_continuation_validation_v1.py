"""登记并验证一次1日加8日续接，不增加策略候选或绩效样本。"""
import argparse
import json
from pathlib import Path
import time

import pandas as pd

from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.fixed_date_continuation_v1 import run as run_dates, read, ROOT, PRIMARY

OUT = ROOT / 'reports/research/510300_fixed_date_continuation_validation_v1'
CONFIG = ROOT / 'config/510300_fixed_date_continuation_validation_v1.json'
SOURCE = ROOT / 'reports/research/510300_september_monthly_continuation_v1'


def prepare():
    require(not CONFIG.exists(), '本项已经登记，直接接续已保存步骤')
    index_path = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
    index = read(index_path)
    require(index['latest_completed_round']['round'] == 214 and not index['running_studies'], '前序完整状态不同')
    OUT.mkdir(parents=True, exist_ok=True)
    require(read(OUT / 'tests_receipt.json')['passed'] == 6, '六项新增入口测试未完成')
    old = read(ROOT / 'config/510300_september_monthly_continuation_v1.json')
    fields = ['evaluation_start', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
              'cash_annual_rate_assumption', 'high_sharpe_target', 'annual_return_target', 'costs', 'dividends',
              'earlier_start', 'earlier_terminal', 'weight_band', 'features']
    cfg = {k: old[k] for k in fields}
    cfg.update(study_id='510300_FIXED_DATE_CONTINUATION_VALIDATION_V1', round=215, registered_at=now(),
        primary=PRIMARY, candidate_models=[PRIMARY], candidate_configurations=0, evaluation_accounts=0,
        data_cutoff='2026-09-11', new_model_fits=0, new_reference_accounts=0,
        rules='docs/510300_FIXED_DATE_CONTINUATION_VALIDATION_V1.md', mode='HISTORICAL_ENGINE_EQUIVALENCE',
        planned_splits=[{'source_close': '2026-08-31', 'cutoff': '2026-09-01', 'days': 1},
                        {'source_close': '2026-09-01', 'cutoff': '2026-09-11', 'days': 8}],
        planned_validation_segments=44, planned_validation_incremental_rows=198,
        goal_achieved=False, strict_forward_evidence_days=0, position_impact=0)
    model_folder = SOURCE
    for name, cutoff, source in [('first_day', '2026-09-01', ROOT / 'reports/research/510300_post_selection_continuous_replay_v1'),
                                  ('remaining_days', '2026-09-11', OUT / 'first_day')]:
        admission = OUT / (name + '_admission.json')
        write_json(admission, {'recorded_at': now(), 'accepted': True, 'cutoff': cutoff,
            'completed_at': now(), 'dividend_coverage_through': '2026-09-11',
            'features_sha256': digest(ROOT / cfg['features']), 'dividends_sha256': digest(ROOT / cfg['dividends']),
            'evidence_class': 'PREVIOUSLY_ACCEPTED_HISTORY_ENGINE_EQUIVALENCE_ONLY',
            'original_input_admission': 'reports/research/510300_post_selection_extension_inputs_v1/saved_verification_receipt.json',
            'not_new_data_collection': True}, exclusive=True)
        write_json(OUT / (name + '_settings.json'), {
            'configuration': str(CONFIG.relative_to(ROOT)), 'source_folder': str(source.relative_to(ROOT)),
            'output_folder': str((OUT / name).relative_to(ROOT)), 'features': cfg['features'], 'source_features': cfg['features'],
            'dividends': cfg['dividends'], 'ridge_models': str((model_folder / 'ridge_models.json').relative_to(ROOT)),
            'within_models': str((model_folder / 'within_models.json').relative_to(ROOT)),
            'cutoff': cutoff, 'admission_receipt': str(admission.relative_to(ROOT)), 'mode': 'HISTORICAL_ENGINE_EQUIVALENCE'}, exclusive=True)
    files = [Path(__file__), ROOT / 'research/fixed_date_continuation_v1.py', ROOT / 'tests/test_fixed_date_continuation_v1.py',
        ROOT / 'research/post_selection_continuous_replay_v1.py', ROOT / 'research/post_selection_continuous_accounts_v1.py',
        ROOT / 'research/post_selection_continuous_factors_v1.py', ROOT / cfg['rules'], OUT / 'tests_receipt.json',
        SOURCE / 'result.json', SOURCE / 'saved_verification_receipt.json', SOURCE / 'ridge_models.json', SOURCE / 'within_models.json',
        ROOT / cfg['features'], ROOT / cfg['dividends'], ROOT / index['latest_fixed_research_origin'],
        ROOT / 'config/510300_september_monthly_continuation_v1.json']
    files += [OUT / (name + suffix) for name in ['first_day', 'remaining_days'] for suffix in ['_settings.json', '_admission.json']]
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in files]
    write_json(CONFIG, cfg, exclusive=True)
    index['running_studies'] = [{'round': 215, 'study': cfg['study_id'], 'config': str(CONFIG.relative_to(ROOT)), 'status': 'FROZEN_ENGINE_EQUIVALENCE_NOT_STARTED'}]
    index['next_work'].update(registered=True, status='DATE_PARAMETERIZED_ENTRY_FROZEN', planned_new_accounts=0,
        planned_existing_account_incremental_continuations=0, planned_validation_segments=44, external_data_required=False)
    write_json(index_path, index)
    print('第215轮入口已固定，直接做1日加8日等价检验；零新增策略及绩效样本。', flush=True)


def run():
    cfg = read(CONFIG)
    require(not (OUT / 'result.json').exists(), '验证已经完成，不重复运行')
    for item in cfg['frozen_files']:
        require(digest(ROOT / item['path']) == item['sha256'], '固定入口或输入已改变')
    if not (OUT / 'RUN_STARTED.json').exists():
        write_json(OUT / 'RUN_STARTED.json', {'started_at': now(), 'config_sha256': digest(CONFIG)}, exclusive=True)
    started = time.perf_counter()
    first = run_dates(OUT / 'first_day_settings.json')
    remaining = run_dates(OUT / 'remaining_days_settings.json')
    require(first['incremental_account_rows'] == 22 and remaining['incremental_account_rows'] == 176, '实际拆分日期不同')
    rows = []
    for folder in sorted((SOURCE / 'accounts').glob('*/*')):
        compare = OUT / 'remaining_days/accounts' / folder.parent.name / folder.name
        for filename in ['ledger.parquet', 'decisions.parquet']:
            left, right = pd.read_parquet(folder / filename), pd.read_parquet(compare / filename)
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        require(read(folder / 'checkpoint.json') == read(compare / 'checkpoint.json'), '拆分后最终控制器或账户状态不同')
        if (folder / 'cycles.csv').is_file():
            pd.testing.assert_frame_equal(pd.read_csv(folder / 'cycles.csv'), pd.read_csv(compare / 'cycles.csv'), check_exact=True)
        rows.append({'node': folder.name, 'cost': folder.parent.name, 'ledger_rows': len(pd.read_parquet(folder / 'ledger.parquet', columns=['date'])),
            'ledger_exact': True, 'decisions_exact': True, 'complete_state_exact': True})
    require(len(rows) == 22, '实际等价来源不足')
    pd.DataFrame(rows).to_csv(OUT / 'saved_equivalence.csv', index=False, encoding='utf-8-sig')
    before = {str(p.relative_to(OUT / 'remaining_days')): digest(p) for p in sorted((OUT / 'remaining_days').rglob('*')) if p.is_file() and p.name != 'run.lock'}
    cached = run_dates(OUT / 'remaining_days_settings.json')
    after = {str(p.relative_to(OUT / 'remaining_days')): digest(p) for p in sorted((OUT / 'remaining_days').rglob('*')) if p.is_file() and p.name != 'run.lock'}
    require(cached == remaining and before == after, '重复同日未直接复用保存结果')
    old = read(SOURCE / 'result.json')
    result = {k: old[k] for k in ['candidate_models', 'all_metrics', 'earlier_diagnostics', 'primary', 'post_selected_best_base']}
    result.update(study_id=cfg['study_id'], completed_at=now(), status='COMPLETED_DATE_ENTRY_TWO_SPLIT_EQUIVALENCE',
        candidate_configurations=0, evaluation_accounts=0, new_accounts_generated=0, reused_control_accounts=2,
        earlier_diagnostic_accounts=0, new_earlier_diagnostic_accounts=0, new_model_fits=0, new_reference_accounts=0,
        metrics_reused_from_round214=True, new_strategy_performance_computed=False, new_observed_trading_days=0,
        validation_engine_segments=44, validation_incremental_account_rows=198, final_equivalent_accounts=22,
        final_equivalent_account_rows=sum(r['ledger_rows'] for r in rows), idempotent_repeat_no_recomputation=True,
        run_seconds=time.perf_counter() - started, run_seconds_scope='两个日期段的入口计算、保存等价及重复日期复用核对；不含开发和测试',
        first_segment_core_seconds=first['core_seconds'], remaining_segment_core_seconds=remaining['core_seconds'],
        strict_forward_evidence_days=0, independent_validation='NOT_ESTABLISHED', goal_achieved=False, position_impact=0,
        remaining='下一官方交易日完整行情和分红接纳；当前无新增日线，入口尚未执行未来日期')
    write_json(OUT / 'result.json', result, exclusive=True)
    print(json.dumps({'status': result['status'], 'seconds': result['run_seconds'], 'equivalent_accounts': len(rows),
        'new_performance_accounts': 0, 'new_independent_days': 0}, ensure_ascii=False), flush=True)


def verify():
    result = read(OUT / 'result.json')
    rows = pd.read_csv(OUT / 'saved_equivalence.csv')
    require(len(rows) == 22 and rows[['ledger_exact', 'decisions_exact', 'complete_state_exact']].all().all(), '保存等价记录未通过')
    require(result['new_observed_trading_days'] == 0 and result['evaluation_accounts'] == 0, '工程检验被计为新绩效')
    source = read(SOURCE / 'result.json')
    require(result['all_metrics'] == source['all_metrics'] and result['earlier_diagnostics'] == source['earlier_diagnostics'], '引用历史指标发生变化')
    for item in read(CONFIG)['frozen_files']:
        require(digest(ROOT / item['path']) == item['sha256'], '固定入口或输入已改变')
    write_json(OUT / 'saved_verification_receipt.json', {'verified_at': now(),
        'status': 'PASS_22_SAVED_ACCOUNTS_TWO_SPLIT_EXACT_AND_REPEAT_NO_RECOMPUTATION',
        'accounts': 22, 'account_rows': int(rows.ledger_rows.sum()), 'new_engine_tests': 6, 'prior_tests_reused': 13,
        'validation_segments': 44, 'new_observed_trading_days': 0, 'independent_performance_validation': False}, exclusive=True)
    print('日期入口的22条实际拆分结果与214保存账户、决定及完整状态一致；重复同日未重算。', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='只检验通用新增日入口，不新增策略或回测收益。')
    parser.add_argument('action', choices=['prepare', 'run', 'verify'])
    globals()[parser.parse_args().action]()
