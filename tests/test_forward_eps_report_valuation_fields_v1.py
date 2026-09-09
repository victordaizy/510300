"""明确原文字段、报价区间及缺失和多值处理。"""
from decimal import Decimal
from research.forward_eps_report_valuation_fields_v1 import parse_front_page


def test_explicit_interval_and_quote_stay_on_report_clock():
    result=parse_front_page('合理估值 190.00 - 200.00 元\n收盘价 135.90 元\n总市值/流通市值 99778/99778 百万元\n')
    assert result['quote']['value']=='135.90'
    assert result['target']['lower']=='190.00' and result['target']['upper']=='200.00'
    assert Decimal(result['target_upside']['midpoint'])==Decimal(195)/Decimal('135.90')-1
    assert not result['exact_total_shares_inferred'] and not result['target_is_current_month_end_fair_value']


def test_missing_target_is_not_zero_or_filled_from_body():
    result=parse_front_page('合理估值\n收盘价 10.00 元\n我们认为未来有望达到20元。\n')
    assert result['target_upside'] is None and result['target']['lower'] is None


def test_multiple_quotes_are_not_silently_selected():
    result=parse_front_page('合理估值 20.00 元\n收盘价 10.00 元\n收盘价 12.00 元\n')
    assert result['quote']['value'] is None and result['target_upside'] is None


def test_target_may_be_below_quote_without_forcing_positive_upside():
    result=parse_front_page('合理估值 8.00 至 9.00 元\n收盘价 10.00 元\n')
    assert Decimal(result['target_upside']['midpoint'])==Decimal('-.15')


def test_reversed_or_zero_target_is_not_admitted():
    for line in ['合理估值 12 - 10 元','合理估值 0 元']:
        result=parse_front_page(line+'\n收盘价 10 元\n')
        assert result['target_upside'] is None
