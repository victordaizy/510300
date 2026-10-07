"""验证公布钟、表格合并单元格归属、版本和未知成员保留。"""
import pandas as pd

from research import official_industry_classification_inputs_v1 as source


def test_publication_date_is_not_period_end_or_same_day_close():
    rows = [{"id": "2023H2", "published": "2024-04-03", "parse_passed": True, "taxonomy": "CAPCO_2023"},
        {"id": "2024H1", "published": "2024-09-30", "parse_passed": True, "taxonomy": "CAPCO_2023"}]
    assert source.choose_snapshot(rows, "2024-09-24 15:00")["id"] == "2023H2"
    assert source.choose_snapshot(rows, "2024-09-30 15:00")["id"] == "2023H2"
    assert source.choose_snapshot(rows, "2024-10-08 15:00")["id"] == "2024H1"
    assert source.choose_snapshot(rows, "2024-01-01 15:00") is None
    rows[1]["parse_passed"] = False
    assert source.choose_snapshot(rows, "2024-10-08 15:00") is None


def test_leading_zero_and_exchange_identity_are_preserved():
    assert source.symbol("000001") == "000001.SZ"
    assert source.symbol("600000") == "600000.SH"
    assert source.symbol("900900") == "900900.SH"
    assert source.symbol("920001") == "920001.BJ"
    assert source.symbol("1") is None


def test_csrc_merged_cells_keep_origin_but_new_section_does_not_inherit_major():
    state = {}
    first = source.parse_row(["金融业(J)", "66", "货币金融服务", "000001", "平安银行"], "CSRC_QUARTERLY", state, "1:0:1")
    second = source.parse_row([None, None, None, "600000", "浦发银行"], "CSRC_QUARTERLY", state, "1:0:2")
    assert first["industry_key"] == second["industry_key"] == "J66"
    assert second["major_origin"] == "1:0:1"
    continued = source.parse_row(["金融业(J)", None, None, "601988", "中国银行"], "CSRC_QUARTERLY", state, "2:0:0")
    assert continued["industry_key"] == "J66" and continued["major_origin"] == "1:0:1"
    unknown = source.parse_row(["制造业(C)", None, None, "600001", "示例证券"], "CSRC_QUARTERLY", state, "2:0:1")
    assert not unknown["classification_known"] and unknown["major_code"] is None


def test_capco_manufacturing_subclass_and_taxonomy_are_not_sw_labels():
    row = source.parse_row(["000008", "神州高铁", "C", "制造业", "CG", "设备", "37", "铁路等运输设备制造业"],
        "CAPCO_2023", {}, "1:0:7")
    assert row["industry_key"] == "C37" and row["manufacturing_subclass"] == "CG"
    assert row["taxonomy"] == "CAPCO_2023" and row["layout_inheritance"] == ""


def test_duplicate_or_missing_security_is_not_a_passed_pdf():
    row = source.parse_row(["000001", "平安银行", "J", "金融业", "", "", "66", "货币金融服务"], "CAPCO_2023", {}, "1:0:1")
    assert source.quality([row], {"000001"})["parse_passed"]
    assert not source.quality([row, row], {"000001"})["parse_passed"]
    assert not source.quality([row], {"000001", "600000"})["parse_passed"]


def test_missing_member_is_retained_unknown_without_older_or_future_backfill():
    members = pd.DataFrame({"symbol": ["000001.SZ", "600000.SH"]})
    snapshot = pd.DataFrame({"symbol": ["000001.SZ"], "classification_known": [True], "taxonomy": ["CAPCO_2023"]})
    result = source.classify_members(members, snapshot)
    assert len(result) == 2 and result.classification_known.tolist() == [True, False]
    assert pd.isna(result.loc[1, "taxonomy"])
