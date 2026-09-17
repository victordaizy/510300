"""把已完成研究的延续状态一次保存，供新增日期到齐后的同规则离线续算。"""
import json
from pathlib import Path

import pandas as pd

from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.post_selection_continuous_accounts_v1 import unpack

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_fixed_research_origin_v1'
SOURCE = ROOT / 'reports/research/510300_september_monthly_continuation_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}


def main():
    require(not (OUT / 'origin.json').exists(), '研究起点已保存，请直接使用原记录')
    index_path = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
    index = read(index_path)
    require(index['latest_completed_round']['round'] == 214 and not index['running_studies'], '完整九月状态尚未交付')
    require(read(SOURCE / 'saved_verification_receipt.json')['full_pipeline_incremental_resume_verified'], '实际增量核对未完成')
    recorded_at = now()
    require(pd.Timestamp(recorded_at) < pd.Timestamp('2026-09-14T09:30:00+08:00'), '该起点已晚于下一开盘，不能保存为此前记录')
    summary = pd.read_csv(SOURCE / 'ending_positions_and_next_requests.csv')
    require(len(summary) == 22 and summary[['model', 'cost']].duplicated().sum() == 0, '22条来源状态不完整')
    rows = []
    for row in summary.to_dict('records'):
        folder = SOURCE / 'accounts' / row['cost'] / row['model']
        checkpoint = read(folder / 'checkpoint.json')
        state = unpack(checkpoint['state'])
        require(state['asof_date'] == pd.Timestamp('2026-09-11'), '来源截止时间不同')
        require(int(state['account']['shares']) == row['shares'], '保存状态份额与末日表不一致')
        require(state['pending']['requested_quantity'] == row['next_request'], '待续算申请不同')
        require(row['next_execution_date'] == '2026-09-14', '下一官方交易日不同')
        rows.append({**row, 'scope': 'FIXED_RESEARCH_ACCOUNT_CONTINUATION_ONLY',
            'checkpoint': identity(folder / 'checkpoint.json'), 'ledger': identity(folder / 'ledger.parquet'),
            'decisions': identity(folder / 'decisions.parquet')})
    cfg_path = ROOT / 'config/510300_september_monthly_continuation_v1.json'
    cfg = read(cfg_path)
    paths = [cfg_path, SOURCE / 'ridge_models.json', SOURCE / 'within_models.json',
        SOURCE / 'result.json', SOURCE / 'saved_verification_receipt.json',
        SOURCE / 'factors/all_required_targets.parquet', ROOT / cfg['features'], ROOT / cfg['dividends'],
        ROOT / 'reports/research/510300_post_selection_extension_inputs_v1/dependency_graph.json',
        ROOT / 'data/reference/sse_trade_calendar_2026.csv', ROOT / 'docs/510300_FIXED_INCREMENTAL_RESEARCH_NEXT_20260913.md',
        ROOT / 'research/september_saved_continuation_v1.py', Path(__file__)]
    origin = {'recorded_at': recorded_at, 'status': 'FIXED_RESEARCH_CONTINUATION_ORIGIN_SAVED_BEFORE_NEXT_OPEN',
        'source_completed_round': 214, 'known_price_cutoff': '2026-09-11', 'next_official_trading_day': '2026-09-14',
        'fixed_strategy': 'SELECTED_MIX_BAND10_SIMPLE2', 'weights': [.85, .15], 'ordinary_weight_band': .1,
        'accounts': rows, 'bound_files': [identity(p) for p in paths], 'new_model_fits': 0,
        'new_account_rows': 0, 'strict_forward_evidence_days': 0, 'independent_validation': 'NOT_ESTABLISHED',
        'timestamp_scope': 'ACTUAL_LOCAL_RECORD_CREATION_TIME_NOT_EXTERNAL_TIMESTAMP_CERTIFICATION',
        'research_scope': '按已固定规则延续离线研究账户；已有历史申请原样保存，新增完整数据到齐后评价',
        'actual_holdings_state': 'UNKNOWN_OUT_OF_SCOPE', 'broker_connection_authorized': False,
        'paper_signal_authorized': False, 'shadow_signal_authorized': False, 'order_authorization': 'NOT_AUTHORIZED',
        'position_impact': 0, 'goal_achieved': False}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'origin.json', origin, exclusive=True)
    pd.testing.assert_frame_equal(summary, pd.DataFrame([{k: r[k] for k in summary.columns} for r in read(OUT / 'origin.json')['accounts']]))
    index['latest_fixed_research_origin'] = str((OUT / 'origin.json').relative_to(ROOT))
    index['next_work'].update(research_origin_saved=True, research_origin=str((OUT / 'origin.json').relative_to(ROOT)),
        status='FIXED_RESEARCH_ORIGIN_SAVED_DATE_PARAMETERIZATION_NEXT')
    index['updated_at'] = now()
    write_json(index_path, index)
    print(json.dumps({'已保存': str((OUT / 'origin.json').relative_to(ROOT)), '实际时间': recorded_at,
        '来源状态': len(rows), '新增交易日': 0, '新增模型': 0}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
