"""空成员源和失败最新快照的必要案例汇总回归。"""
import pandas as pd

from research.official_industry_classification_parse_v1_0_1 import case_tables


def test_no_membership_source_keeps_unknown_case_without_zero_denominator_or_grouping():
    metadata = [{"id": "2014Q3", "published": "2014-10-13", "parse_passed": True, "taxonomy": "CSRC_QUARTERLY"}]
    members = pd.DataFrame({"symbol": pd.Series(dtype="object")})
    def forbidden_loader(chosen):
        raise AssertionError("缺成员源不应加载并归组分类。")
    case, classified, industries = case_tables("2015-01-05", "2014-12-31", members, metadata, forbidden_loader)
    assert case["view_state"] == "NO_VIEW_MEMBERSHIP_SOURCE_MISSING"
    assert case["members"] == 0 and case["coverage"] is None and case["unknown_members"] is None
    assert case["snapshot_id"] == "2014Q3" and len(classified) == 0 and industries == []


def test_latest_failed_snapshot_keeps_all_three_hundred_members_unknown_and_retains_failure_id():
    metadata = [{"id": "2015Q1", "published": "2015-05-04", "parse_passed": True, "taxonomy": "CSRC_QUARTERLY"},
        {"id": "2018Q3", "published": "2018-11-02", "parse_passed": False, "taxonomy": "CSRC_QUARTERLY"}]
    members = pd.DataFrame({"symbol": [f"{i:06d}.SZ" for i in range(300)]})
    def forbidden_loader(chosen):
        raise AssertionError("最新源解析失败不应回退旧分类。")
    case, classified, industries = case_tables("2019-01-08", "2019-01-07", members, metadata, forbidden_loader)
    assert case["view_state"] == "NO_VIEW_LATEST_PUBLISHED_SNAPSHOT_PARSE_FAILED"
    assert case["latest_published_snapshot_id"] == "2018Q3" and case["snapshot_id"] is None
    assert case["members"] == case["unknown_members"] == 300 and case["coverage"] == 0
    assert len(classified) == 300 and not classified.classification_known.any() and industries == []
