"""防止将多报告期更正公告错误连接为单一原报告。"""
import pandas as pd
import pytest

from research.factor96_t11_correction_catalogue_v1 import candidate_record, report_period_from_title


@pytest.mark.parametrize("title,kind,period", [
    ("2024年年度报告（更正后）", "FY", "2024-12-31"),
    ("2025年半年度报告（修订版）", "H1", "2025-06-30"),
    ("关于2023年第三季度报告的更正公告", "Q3", "2023-09-30"),
    ("2021年一季度报告（更新后）", "Q1", "2021-03-31"),
])
def test_explicit_single_report(title, kind, period):
    found_kind, found_period, status = report_period_from_title(title)
    assert (found_kind, found_period, status) == (kind, pd.Timestamp(period), "UNIQUE_EXPLICIT_YEAR_AND_REPORT_TYPE")


@pytest.mark.parametrize("title", [
    "关于2023年、2024年年度报告的更正公告",
    "2024年年度报告及半年度报告更正公告",
    "2024年年度报告及2025年第一季度报告更正公告",
    "关于前期会计差错更正的公告",
])
def test_ambiguous_period_stays_unknown(title):
    assert report_period_from_title(title) == (None, None, "NO_VIEW_AMBIGUOUS_OR_MISSING_REPORT_PERIOD")


def test_revision_title_does_not_establish_direction_or_date_identity():
    raw = {"announcementId": "sample", "secCode": "600000", "announcementTitle": "2024年年度报告（更正后）",
           "announcementTime": int(pd.Timestamp("2025-04-26", tz="Asia/Shanghai").timestamp() * 1000),
           "adjunctUrl": "finalpage/2025-04-27/sample.PDF"}
    result = candidate_record(raw)
    assert result["official_internal_date_equal"] is False
    assert result["content_admission"] == "NOT_READ_NUMERICAL_REVISION_DIRECTION_UNKNOWN"
    assert result["ts_code"] == "600000.SH"


def test_original_title_is_not_revision_candidate():
    assert candidate_record({"announcementTitle": "2024年年度报告"}) is None
