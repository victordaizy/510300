"""针对金融报告范围与季度列错配的回归测试。"""
from research.bank_financial_scope_adapter_v1 import parse_consolidated_q3_page


def page(scope="合并", periods="1-9 月期间 7-9 月期间"):
    return f"""未经审计{scope}利润表
货币单位均以人民币百万元列示
{periods}
2021 年 2020 年 2021 年 2020 年
营业收入合计 100 90 30 20
营业支出合计 (60) (55) (18) (12)
营业利润 40 35 12 8
净利润 30 25 9 6
本行股东的净利润 29 24 8 5
少数股东的净利润 1 1 1 1
基本及稀释每股收益（人民币元） 3.62 3.02 1.27 1.05
"""


def test_parent_only_statement_is_not_consolidated():
    r = parse_consolidated_q3_page(page(scope=""), 2021)
    assert r["status"] == "NO_VIEW_NOT_EXPLICIT_CONSOLIDATED_INCOME_STATEMENT" and not r["facts"]


def test_ytd_column_comes_from_period_header_not_fixed_position():
    first = parse_consolidated_q3_page(page(), 2021)
    third = parse_consolidated_q3_page(page(periods="7-9 月期间 1-9 月期间"), 2021)
    assert first['facts'][1]['metric_value'] == 40_000_000
    assert third['facts'][1]['metric_value'] == 12_000_000
    assert first['facts'][-1]['metric_value'] == 3.62
    assert third['facts'][-1]['metric_value'] == 1.27


def test_missing_header_or_year_cannot_be_guessed():
    assert not parse_consolidated_q3_page(page().replace("1-9 月期间", "累计"), 2021)["facts"]
    assert not parse_consolidated_q3_page(page(), 2022)["facts"]


def test_accounting_identity_violation_is_rejected():
    bad = page().replace("营业利润 40 35 12 8", "营业利润 140 35 12 8")
    assert parse_consolidated_q3_page(bad, 2021)["status"] == "NO_VIEW_CONSOLIDATED_ACCOUNTING_IDENTITY_MISMATCH"
