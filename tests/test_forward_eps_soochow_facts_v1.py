"""检验原件预测年度、EPS列边界、日期时钟及未明确口径的保留。"""
from copy import deepcopy

import pytest

from research.forward_eps_soochow_facts_v1 import parse, report_date


def fixture():
    page = """贵州茅台（600519） 东吴证券研究所
2024 年 12 月 29 日
盈利预测与估值 [Table_EPS] 2022A 2023A 2024E 2025E 2026E
营业总收入（百万元） 127554 150560 173848 188752 205238
归母净利润（百万元） 62717 74734 85973 92060 99879
EPS-最新摊薄（元/股） 49.93 59.49 68.44 73.28 79.51
P/E（现价&最新摊薄） 30.62 25.70 22.34 20.86 19.23
[Table_Tag]
投资要点
收盘价(元) 1,528.97
总股本(百万股) 1,256.20
"""
    meta = {"info_code": "APTEST", "company_code": "80000031", "security": [{"stock": "600519"}],
            "notice_date": "2024-12-29 00:00:00", "eitime": "2024-12-30 09:01:00"}
    row = {"infoCode": "APTEST", "ts_code": "600519.SH", "stockName": "贵州茅台", "publishDate": "2024-12-29 00:00:00"}
    return [page], meta, row


def test_actual_columns_excluded_and_absolute_years_preserved():
    pages, meta, row = fixture()
    row["predictThisYearEps"] = "79.51"
    row["currentYear"] = 2026
    facts = parse(pages, meta, row)["facts"]
    assert [f["target_fiscal_year"] for f in facts] == [2024, 2025, 2026]
    assert [f["eps_value_exact"] for f in facts] == ["68.44", "73.28", "79.51"]
    assert facts[0]["conservative_information_date"] == "2024-12-30"


def test_unknown_eps_basis_not_relabelled_latest_diluted():
    pages, meta, row = fixture()
    pages[0] = pages[0].replace("EPS-最新摊薄", "每股收益")
    fact = parse(pages, meta, row)["facts"][0]
    assert not fact["latest_diluted_basis_explicit"]
    assert "未明确" in fact["eps_definition"]
    assert fact["share_snapshot_million"] == "1256.20"
    assert not fact["historical_immutable_snapshot_proven"]


@pytest.mark.parametrize("replacement", ["68.44 73.28", "68.44 73.28 79.51 80.00"])
def test_missing_or_extra_eps_cell_rejected(replacement):
    pages, meta, row = fixture()
    pages[0] = pages[0].replace("68.44 73.28 79.51", replacement)
    with pytest.raises(ValueError):
        parse(pages, meta, row)


@pytest.mark.parametrize("replacement", ["2022A 2023A 2024A 2025A 2026A", "2022A 2023A 2024E 2027E 2026E"])
def test_forecast_flag_and_contiguous_years_required(replacement):
    pages, meta, row = fixture()
    pages[0] = pages[0].replace("2022A 2023A 2024E 2025E 2026E", replacement)
    with pytest.raises(ValueError):
        parse(pages, meta, row)


def test_negative_forecast_preserved():
    pages, meta, row = fixture()
    pages[0] = pages[0].replace("68.44 73.28 79.51", "-0.20 (0.10) 0.05")
    assert [f["eps_value_exact"] for f in parse(pages, meta, row)["facts"]] == ["-0.20", "-0.10", "0.05"]


def test_other_security_rejected():
    pages, meta, row = fixture()
    meta["security"] = [{"stock": "600036"}]
    with pytest.raises(ValueError):
        parse(pages, meta, row)


def test_author_column_date_supported_and_ambiguous_date_rejected():
    assert report_date("原件\n[Table_Author] 2018 年 10 月 29 日\n分析师") == "2018-10-29"
    with pytest.raises(ValueError):
        report_date("2024 年 12 月 29 日\n2024 年 12 月 30 日")


def test_duplicate_table_rejected():
    pages, meta, row = fixture()
    with pytest.raises(ValueError):
        parse(pages + pages, meta, row)
