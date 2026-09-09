"""份额会计关系、基金身份与可用时点的针对性检查。"""
import pandas as pd

from research.original_fund_quarterly_facts_v1 import parse_flow_table,available_clock,validate_identity_and_period


def table():
    return [["本报告期期初基金份额总额","9,591,787,690.00"],
            ["本报告期基金总申购份额","2,709,900,000.00"],
            ["减:本报告期基金总赎回份额","3,392,100,000.00"],
            ["本报告期基金拆分变动份额","-"],
            ["本报告期期末基金份额总额","8,909,587,690.00"]]


def test_original_share_identity_and_explicit_split_dash():
    result=parse_flow_table(table())
    assert result["status"].startswith("PASS_")
    assert result["exact_values"]["ending_units"]=="8909587690.00"
    assert result["raw_rows"]["split_delta_units"]["raw_cell"]=="-"
    changed=table();changed[4][1]="8,900,587,690.00"
    assert parse_flow_table(changed)["status"]=="NO_VIEW_SHARE_FLOW_ACCOUNTING_MISMATCH"


def test_missing_value_is_not_a_zero_flow():
    changed=table();changed[1][1]=""
    assert parse_flow_table(changed)["status"]=="NO_VIEW_UNPROVEN_SHARE_FLOW_VALUE"
    changed=table();changed[1][1]="-"
    assert parse_flow_table(changed)["status"]=="NO_VIEW_UNPROVEN_SHARE_FLOW_VALUE"


def test_fund_code_linked_fund_and_period_identity():
    cover="华泰柏瑞沪深300交易型开放式指数证券投资基金2012年第3季度报告 报告送出日期：2012年10月26日"
    overview="交易代码510300 本报告期自2012年7月1日起至2012年9月30日止。"
    assert validate_identity_and_period(cover,overview,"2012Q3")["status"].startswith("PASS_")
    assert not validate_identity_and_period(cover.replace("投资基金","投资基金联接基金"),overview,"2012Q3")["status"].startswith("PASS_")
    assert not validate_identity_and_period(cover,overview.replace("510300","510310"),"2012Q3")["status"].startswith("PASS_")
    assert not validate_identity_and_period(cover,overview,"2012Q4")["status"].startswith("PASS_")


def test_publication_clock_does_not_use_file_creation_as_publication():
    days=pd.date_range("2021-10-25","2021-10-29",freq="B")
    source={"publication_date":"2021-10-27","official_publication_catalog_verified":True}
    result=available_clock(source,"2021-10-27",{"CreationDate":"D:20211026113628"},days)
    assert result["source_usable"] and result["feature_available_session"]=="2021-10-28"
    missing={"publication_date":None,"official_publication_catalog_verified":False}
    assert not available_clock(missing,"2021-10-27",{"CreationDate":"D:20211026113628"},days)["source_usable"]
    assert not available_clock(source,"2021-10-27",{"CreationDate":"D:20231121120000"},days)["source_usable"]
