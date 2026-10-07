"""滞后原文不能改写统计日、成员范围和原先的同日失败。"""
from research.all_factor_disclosed_context_v1 import build_context


def test_lagged_report_preserves_statistics_date_and_has_later_processing_upper_bound():
    rows = '\n'.join(f'{300000+i} 公司{i} 工业 深圳 1.00%' for i in range(10))
    text = '2026年8月31日 沪深300 000300 滚动市盈率 14.65 市净率 1.44 股息率 2.27% 计算用股本\n'+rows
    context = build_context(text, '2026-10-06T13:17:00+08:00', '2026-10-06T13:30:00+08:00', {'statistics_date':'2026-09-30','raw_peg':13.15})
    assert context['document_statistics_date']=='2026-08-31'
    assert not context['quote_and_factsheet_same_statistics_date']
    assert not context['api_peg_current_semantics_verified']
    assert context['available_at']=='2026-10-06T13:30:00+08:00'
    assert not context['historical_use_authorized']
    assert context['financial_model_admission']=='NOT_ADMITTED'


def test_missing_or_duplicate_top10_is_not_backfilled():
    import pytest
    text='2026年8月31日 沪深300 000300 滚动市盈率 14.65 计算用股本\n300750 宁德时代 工业 深圳 3.66%'
    with pytest.raises(ValueError):
        build_context(text, '2026-10-06T13:17:00+08:00', '2026-10-06T13:30:00+08:00', {})
