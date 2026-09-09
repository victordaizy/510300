"""覆盖无总标题的财务预测表、明确行名换行和拒绝混读。"""
from research.forward_eps_soochow_layout_v3 import appendix_page_selected, rows_with_headers


def test_financial_appendix_without_overall_caption():
    assert appendix_page_selected("资产负债表（百万元）2015 2016E 2017E\n重要财务与估值指标\n每股收益（元）1 2 3")


def test_forecast_income_and_bare_eps():
    assert appendix_page_selected("利润表 2015 2016E 2017E\nEPS 1 2 3")


def test_body_eps_without_financial_table_is_not_appendix():
    assert not appendix_page_selected("投资建议预计2016E和2017E的EPS分别为1和2")


def test_actual_financial_table_is_not_forecast_appendix():
    assert not appendix_page_selected("利润表 2015 2016 2017\n每股收益 1 2 3")


def test_wrapped_parent_profit_keeps_reported_label():
    text = "利润表(百万元) 2015 2016E 2017E 2018E\n归属母公司净利\n润 342.1 211.3 244.8 320.3\n"
    rows = rows_with_headers(text, "profit")
    assert len(rows) == 1
    assert rows[0]["label"] == "归母净利润"
    assert rows[0]["label_as_reported"] == "归属母公司净利\n润"
    assert rows[0]["values"] == ["342.1", "211.3", "244.8", "320.3"]


def test_wrapped_parent_profit_requires_amount_unit():
    assert rows_with_headers("2015 2016E 2017E\n归属母公司净利\n润 1 2 3", "profit") == []


def test_wrapped_growth_percent_is_not_amount():
    assert rows_with_headers("利润表(百万元) 2015 2016E 2017E\n归属母公司净利\n润 1% 2% 3%", "profit") == []
