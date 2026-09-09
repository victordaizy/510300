"""验证跨页只拼接明确份额项目，缺行、重复、颠倒顺序仍不可用。"""
from research.original_fund_quarterly_facts_v1_1 import join_adjacent_tables


def test_adjacent_split_label_and_other_asset_table():
    a=[["报告期期初基金份额总额","11109187690.00"],
       ["报告期期间基金总申购份额","2040300000.00"],
       ["减:报告期期间基金总赎回份额","4979700000.00"],
       ['报告期期间基金拆分变动份额（份额减少以"-"',"-"]]
    b=[["填列）", ""],["报告期期末基金份额总额","8169787690.00"]]
    actual=join_adjacent_tables([[["应收申购款","99999"]],a,b])
    assert actual["status"]=="PASS_ORIGINAL_SHARE_FLOW_TABLE_IDENTITY"
    assert actual["exact_values"]["ending_units"]=="8169787690.00"
    assert not join_adjacent_tables([a])["status"].startswith("PASS_")
    assert not join_adjacent_tables([a,b,b])["status"].startswith("PASS_")
    assert not join_adjacent_tables([b,a])["status"].startswith("PASS_")
