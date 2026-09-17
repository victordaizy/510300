"""只核对已直接关联的份额回执，缺少历史披露时点即结束局部补充。"""
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, write_json
from research.monthly_single_factor_walkforward_v1 import read


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_fund_share_publication_receipts_closure_v1'
INDEX = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
SNAPSHOT = ROOT / 'data/external_validation/510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_2/snapshots/20260828T125705_0800'
REPORT = ROOT / 'deliverables/510300有限历史挖掘_申赎时间口径收尾_20260914/已有回执核对结论.md'


def link(path, title):
    return f'[{title}](<{path.as_posix()}>)'


def main():
    assert not (OUT / 'result.json').exists(), '本项局部补充已结束，不重复扫描或累计进展'
    receipt_path = ROOT / 'reports/data_quality/daily_flow_data_tushare_status.json'
    merge_path = ROOT / 'reports/data_quality/510300_etf_share_premium_level_full_v1.json'
    receipt, merged = read(receipt_path), read(merge_path)
    source = ROOT / receipt['datasets']['fund_share']['file']
    assert digest(source) == receipt['datasets']['fund_share']['sha256']
    shares = pd.read_parquet(source)
    extension_path = ROOT / 'data/raw/flow/510300_fund_share_extension_20260818_v2.parquet'
    extension = pd.read_parquet(extension_path)
    historical = pd.read_parquet(SNAPSHOT / '510300_fund_share_daily.parquet')
    index_path = SNAPSHOT / 'raw_response_index.json'
    index = read(index_path)
    collection = read(SNAPSHOT / 'collection_status.json')
    manifest = read(SNAPSHOT / 'snapshot_manifest.json')
    assert digest(index_path) == manifest['collection_audit']['raw_response_index']['sha256']
    raw = [item for item in index if item['kind'] == 'fund_share']
    fields, raw_dates, rows, actual_times = Counter(), [], [], []
    for item in raw:
        path = ROOT / item['file']
        assert digest(path) == item['sha256'], '份额响应与原索引身份不一致'
        response = read(path)
        assert not response.get('actionErrors') and not response.get('fieldErrors')
        assert len(response['result']) == 1
        row = response['result'][0]
        assert row['SEC_CODE'] == '510300' and row['STAT_DATE'] == item['identifier']
        fields.update(row.keys())
        raw_dates.append(row['STAT_DATE'])
        actual_times.append(item['stored_at_utc_from_file_mtime'])
        rows.append({'date': pd.Timestamp(row['STAT_DATE']), 'fund_shares': float(row['TOT_VOL']) * 10000})
    parsed = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    original = historical.sort_values('date').reset_index(drop=True)
    assert len(raw) == len(historical) == len(set(raw_dates)) == 2220
    assert np.array_equal(parsed.date.to_numpy(), pd.to_datetime(original.date).to_numpy())
    np.testing.assert_allclose(parsed.fund_shares, original.fund_shares, atol=1e-5, rtol=0)
    publication_keys = [key for key in fields if any(token in key.lower() for token in ['publish', 'announce', 'available', 'release', 'disclos'])]
    assert not publication_keys
    trading_dates = pd.to_datetime(pd.read_parquet(ROOT / 'data/raw/r6/510300_daily.parquet', columns=['date']).date)
    source_dates = pd.to_datetime(shares.date)
    extra = sorted(str(x.date()) for x in source_dates[~source_dates.isin(trading_dates)])
    assert extra == sorted(merged['share_only_dates_excluded_without_fill'])
    expected_dates = trading_dates[trading_dates.between(source_dates.min(), source_dates.max())]
    missing_trading_dates = sorted(str(x.date()) for x in expected_dates[~expected_dates.isin(source_dates)])
    assert not missing_trading_dates
    paths = [source, extension_path, receipt_path, merge_path, index_path,
             SNAPSHOT / 'collection_status.json', SNAPSHOT / 'snapshot_manifest.json', SNAPSHOT / 'run_contract.json',
             SNAPSHOT / '510300_fund_share_daily.parquet',
             ROOT / 'scripts/download_daily_flow_data_tushare.py',
             ROOT / 'scripts/build_510300_etf_share_premium_level_inputs_v1.py',
             ROOT / 'research/daily_etf_flow_direction_challenger.py']
    result = {
        'study_id': '510300_FUND_SHARE_PUBLICATION_RECEIPTS_CLOSURE_V1', 'completed_at': now(),
        'status': 'CLOSED_NO_HISTORICAL_PUBLICATION_TIME_IN_DIRECTLY_LINKED_RECEIPTS',
        'source_rows': len(shares), 'source_receipt_at': receipt['checked_at'],
        'source_receipt_api_host': receipt['api_host'],
        'source_label_does_not_prove_direct_official_api_collection': True,
        'source_first_date': str(source_dates.min().date()), 'source_last_date': str(source_dates.max().date()),
        'actual_trading_dates_covered': len(expected_dates), 'extra_non_trading_dates': extra,
        'extra_dates_already_excluded_in_old_merge': True, 'missing_trading_dates_in_range': missing_trading_dates,
        'historical_raw_response_count': len(raw), 'historical_response_row_fields': dict(fields),
        'historical_publication_fields_found': publication_keys,
        'historical_snapshot_completed_at': collection['completed_at_asia_shanghai'],
        'historical_raw_file_timestamp_range_utc': [min(actual_times), max(actual_times)],
        'historical_raw_file_timestamp_meaning': '原索引标注文件保存时间，不能解释为历史首次公布时间',
        'historical_normalized_values_match_raw_responses': True,
        'extension_rows': len(extension),
        'extension_dates': [str(pd.Timestamp(x).date()) for x in extension.date],
        'extension_retrieval_times': [str(x) for x in extension.retrieved_at.unique()],
        'extension_does_not_backfill_earlier_publication_proof': True,
        'decision': '结束此输入的局部补充；无新原始披露证据不再重查相同回执，不延迟或反向改造旧失败因子',
        'scope_limit': '仅本结果列出的直接关联回执和份额响应；不声称全项目或外部世界不存在其他证据',
        'new_downloads': 0, 'new_candidates': 0, 'new_accounts': 0, 'new_model_fits': 0,
        'independent_validation': False, 'goal_achieved': False, 'position_impact': 0,
        'files': [{'path': str(path.relative_to(ROOT)), 'sha256': digest(path)} for path in paths],
    }
    write_json(OUT / 'result.json', result, exclusive=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join([
        '# 510300申赎数据：已有回执核对结论', '',
        '**检查已完成：直接关联的现成材料仍未建立逐日历史首次披露时间，因此结束这项输入的局部补充。旧申赎方法的失败保留，没有重新拟合、延迟一天后重跑或倒转因子方向。**', '',
        '## 实际查了哪些材料', '',
        '| 材料 | 核对结果 |', '|---|---|',
        '| 原1216行份额缓存与下载回执 | 文件身份一致；2026年8月13日取得历史，回执不含逐日历史公布时点。 |',
        '| 2220条上交所历史份额原始响应 | 全部逐条读取，代码、统计日期和份额数值与历史快照一致；返回的六种字段均不含历史公布时间。 |',
        '| 历史快照的收集记录、原始索引及边界说明 | 2026年8月28日完成历史抓取；原记录自身已说明不是干净的未来样本。 |',
        '| 2026年8月13、14、17、18日四条扩展份额 | 在8月19日凌晨00时38分取得；不能据此补成此前每个收盘时已经可见。 |',
        '| 已有份额与净值拼接回执 | 五个非交易日已被排除，不能把这些记录误报成旧拼接因子新增问题。 |', '',
        '原份额下载回执记载使用兼容接口的第三方主机；源标签写着Tushare接口名，并不等于由官方API直接取得。更早的2220条材料则明确记录上交所查询地址。此次只是区分来源事实，没有进行额外网络安全审计，也没有新调用数据服务。', '',
        '## 为什么仍不能把它当作已证明当时可用的信号', '',
        '统计日说明数值属于哪一天。后来取得数据的时间说明那次取得时已经可见。历史第一次公开时间才决定交易当时是否能使用。原始响应只有统计日期、ETF类型、代码、简称、规模份额和扩展简称六种字段；文件在2026年8月保存的时间不能替代2012—2021年的首次公开时间。', '',
        '这证明本次所查直接材料的时间证据不足；并不证明每个历史点一定前视，也不证明整个项目或外部不存在别的证据。原快照的数值对得上，仍然不能推出历史可交易时点也对得上。', '',
        '## 五个额外日期不是新的收益解释', '',
        '原1216行中，有1211个510300交易日和5个非交易日：2022年12月31日、2023年9月30日、2023年12月31日、2024年3月31日、2024年6月30日。覆盖区间内交易日没有缺失，旧拼接程序也已经排除这五条。因此不能因为原下载覆盖率显示超过100%，就认定这五条造成了策略亏损。', '',
        '## 后续处理', '',
        '该输入的局部补充结束。重新检查只由新的原始公布时点证据触发；不再扫描相同文件，不猜测固定延迟后当作证据已补齐，不开展长期补数。此次新下载、候选、账户、拟合均为零。', '',
        '近期五个有限候选已完成检验，没有一个达到要求；旧复杂策略仍保留高过拟合风险判断。夏普1.2、年化10%的目标未完成，历史研究授权保留，但当前没有已登记待执行的新候选。继续固定未来观察，等待新信息或明确的新机制后再开展下一项有限检验。', '',
        '下一次已知的日常数据处理时点为2026年9月14日收盘后，沿用原15时30分安排与既定完整日线入口。此项日期说明不是对某个正在运行任务的成功确认。', '',
        '## 文件入口', '',
        '- ' + link(OUT / 'result.json', '完整来源范围、逐字段计数与核对结果'),
        '- ' + link(receipt_path, '原下载回执'),
        '- ' + link(index_path, '2220条份额响应所在的原始索引'),
        '- ' + link(merge_path, '旧拼接程序排除非交易日的原记录'), '',
        '完成时间：' + result['completed_at'], '',
    ]), encoding='utf-8')
    state = read(INDEX)
    counters = {key: state[key] for key in ['registered_configurations_in_this_resumption', 'evaluation_accounts_in_this_resumption', 'finite_historical_mining_totals']}
    record = {'study_id': result['study_id'], 'status': result['status'], 'completed_at': result['completed_at'],
        'result': str((OUT / 'result.json').relative_to(ROOT)).replace('\\', '/'),
        'report': str(REPORT.relative_to(ROOT)).replace('\\', '/'), 'report_sha256': digest(REPORT),
        'raw_fund_share_responses_checked': len(raw), 'local_remediation_closed': True,
        'new_candidates': 0, 'new_accounts': 0, 'new_model_fits': 0, 'goal_achieved': False}
    state.update(updated_at=now(), status='FUND_SHARE_PUBLICATION_RECEIPT_CHECK_CLOSED_NO_NEW_ELIGIBLE_CANDIDATE',
        latest_fund_share_publication_receipt_closure=record, latest_continuation_note=record['report'],
        last_goal_turn_classification='PROGRESS_DIRECT_FUND_SHARE_RECEIPTS_CHECK_COMPLETED_LOCAL_REMEDIATION_CLOSED',
        consecutive_external_data_blocked_goal_turns=0, goal_achieved=False)
    state['latest_fund_share_date_semantics_check'].update(local_receipt_check_completed=True,
        local_remediation_closed=True, closure_record=record['result'])
    state['next_work'].update(status='NO_REGISTERED_NEW_CANDIDATE_FIXED_DAILY_OBSERVATION_AND_NEW_MECHANISM_REQUIRED',
        source=record['report'], registered=False, planned_settings=0, planned_new_accounts=0, planned_new_model_fits=0,
        historical_mining_allowed=True, implementation_remaining=False,
        implementation_remaining_scope='近期有限研究与直接份额回执核对已完成；无已登记未执行的新候选',
        next_historical_question_status='NO_NEW_ELIGIBLE_CANDIDATE_IDENTIFIED_AFTER_COMPLETED_FINITE_BATCH',
        next_historical_question='仅在出现不同于旧失败的新经济机制和当时可知的信息后，登记下一项有限历史检验',
        focus='原固定每日观察接收新完整日线；历史研究授权保留，旧份额局部补充和已失败方法不循环重开',
        fund_share_local_remediation_closed=True,
        fund_share_resume_condition='提供此前未检查的逐期首次披露或当时接收证据，能够对应原数值和实际判断截止时间',
        no_live_process_handle_confirmed=True, current_state_is_verified_wait=False)
    assert all(state[key] == value for key, value in counters.items())
    write_json(INDEX, state)
    write_json(OUT / 'delivery_receipt.json', {'status': 'RECEIPT_CHECK_CLOSED_AND_CHINESE_REPORT_DELIVERED',
        'completed_at': now(), 'report': record['report'], 'report_sha256': record['report_sha256'],
        'result_sha256': digest(OUT / 'result.json'), 'old_counters_preserved': True, 'goal_achieved': False}, exclusive=True)
    print('已核对2220条历史份额响应并结束局部补充；没有新增回测或达标声明。', flush=True)


if __name__ == '__main__':
    main()
