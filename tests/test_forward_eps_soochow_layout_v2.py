"""检查旧版表格列边界、分栏标题和明确利润单位。"""
import pytest

from research.forward_eps_soochow_layout_v2 import date_with_author, headers, rows_with_headers, choose_optional


def test_author_date_excludes_old_related_report_date():
    text = "2017 年 1 月 13 日\n证券分析师 徐力\n相关研究\n2016 年 4 月 26 日"
    assert date_with_author(text) == "2017-01-13"


def test_eps_yuan_row_and_bare_pe_keep_forecast_years():
    text = "2015 2016E 2017E 2018E\n每股收益(元) 1.60 1.83 2.01 2.17\nP/E 12 11 10 9"
    eps = rows_with_headers(text, "eps")[0]
    assert eps["values"] == ["1.60", "1.83", "2.01", "2.17"]
    assert eps["unit_as_reported"] == "元"
    assert rows_with_headers(text, "pe")[0]["values"] == ["12", "11", "10", "9"]


def test_nearest_matching_column_header_and_trailing_other_column_text():
    text = "重要财务与估值指标 2015 2016E 2017E 2018E\n8 每股收益(元) 0.04 -0.41 0.02 0.14\n"
    row = rows_with_headers(text, "eps")[0]
    assert row["values"] == ["0.04", "-0.41", "0.02", "0.14"]


def test_forecast_header_required_for_eps():
    assert rows_with_headers("2014 2015 2016 2017\nEPS 1 2 3 4", "eps") == []
    assert rows_with_headers("2014 2015 2016E 2018E\nEPS 1 2 3 4", "eps") == []


@pytest.mark.parametrize("values", ["1 2 3", "1 2 3 4 5"])
def test_misaligned_cell_count_rejected(values):
    assert rows_with_headers("2015 2016E 2017E 2018E\nEPS " + values, "eps") == []


def test_profit_growth_percent_cannot_be_profit_amount():
    text = "利润表(百万元) 2014 2015 2016E 2017E\n归属母公司净利润 1.1% 1.2% 1.3% 1.4%"
    assert rows_with_headers(text, "profit") == []


def test_profit_unit_and_parent_scope_preserved():
    text = "利润表(百万元) 2014 2015 2016E 2017E\n归属母公司净利润 40692 41158 41614 45386"
    row = rows_with_headers(text, "profit")[0]
    assert row["label"] == "归母净利润"
    assert row["label_as_reported"] == "归属母公司净利润"
    assert row["unit_as_reported"] == "利润表(百万元)"
    assert rows_with_headers(text.replace("利润表(百万元)", "利润表"), "profit") == []


def test_bare_eps_does_not_invent_unit():
    row = rows_with_headers("每股指标 2014 2015 2016E 2017E\nEPS 0.87 0.84 0.85 0.93", "eps")[0]
    assert row["unit_as_reported"] is None


def test_optional_mismatched_year_cannot_join():
    row = rows_with_headers("2014 2015 2016E 2017E\nP/E 7 6 5 4", "pe")[0]
    assert choose_optional([row], [[2015, ""], [2016, ""], [2017, "E"], [2018, "E"]]) is None
