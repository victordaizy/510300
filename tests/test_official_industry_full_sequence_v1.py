"""逐行用途保留部分未知、失败最新节点和分类行出处的必要验证。"""
import pandas as pd

from research import official_industry_classification_inputs_v1 as original
from research import official_industry_full_sequence_inputs_v1 as source


def metadata(**changes):
    row = {"id": "2018Q3", "published": "2018-11-02", "parse_passed": False, "title_passed": True,
        "rows": 2, "duplicate_security_rows": 0, "unknown_classification_rows": 1,
        "unparsed_text_codes": [], "codes_absent_from_text": [], "page_clock_verified": True}
    row.update(changes)
    return source.qualify_metadata(row)


def two_rows():
    known = original.parse_row(["金融业(J)", "66", "货币金融服务", "000001", "平安银行"], "CSRC_QUARTERLY", {}, "1:0:1")
    unknown = original.parse_row(["居民服务、修理和", "80", "维修", "300736", "百邦科技"], "CSRC_QUARTERLY", {}, "83:0:28")
    return pd.DataFrame([known, unknown])


def test_known_security_row_is_usable_while_other_unknown_row_and_original_snapshot_failure_remain():
    node = metadata()
    assert not node["complete_snapshot_passed"] and node["row_source_snapshot_eligible"]
    rows = source.qualified_rows(two_rows(), node)
    assert rows.row_source_known.tolist() == [True, False]
    members = pd.DataFrame({"symbol": ["000001.SZ", "300736.SZ", "600000.SH"]})
    joined = members.merge(rows, on="symbol", how="left", validate="one_to_one")
    assert len(joined) == 3 and joined.row_source_known.eq(True).tolist() == [True, False, False]
    assert pd.isna(joined.loc[1, "section_code"]) and pd.isna(joined.loc[2, "industry_key"])


def test_latest_structurally_failed_node_blocks_rows_and_does_not_fall_back_to_older_source():
    older = metadata(id="2015Q1", published="2015-05-04", parse_passed=True, unknown_classification_rows=0)
    newer = metadata(unparsed_text_codes=["600000"])
    latest = source.latest_published([older, newer], "2019-01-08 15:00")
    assert latest["id"] == "2018Q3" and not latest["row_source_snapshot_eligible"]
    assert not source.qualified_rows(two_rows(), latest).row_source_known.any()


def test_a_known_category_without_its_original_row_lineage_is_not_an_admitted_security_row():
    rows = two_rows().iloc[[0]].copy()
    rows.loc[rows.index[0], "major_origin"] = None
    qualified = source.qualified_rows(rows, metadata())
    assert qualified.classification_known.iloc[0] and not qualified.row_source_known.iloc[0]
