from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from research.a_share_hs_cninfo_periodic_report_metadata_v1 import (
    classify_title,
    monthly_intervals,
    normalize_records,
    ts_code_from_cninfo_code,
)


EXCLUSIONS = ["摘要", "取消", "更正", "修订", "更新后", "英文", "英文版", "提示性公告"]


def test_monthly_intervals_cover_partial_boundaries_without_overlap() -> None:
    intervals = monthly_intervals(date(2016, 1, 15), date(2016, 3, 2))
    assert [(item.label, item.start_date, item.end_date) for item in intervals] == [
        ("2016-01", date(2016, 1, 15), date(2016, 1, 31)),
        ("2016-02", date(2016, 2, 1), date(2016, 2, 29)),
        ("2016-03", date(2016, 3, 1), date(2016, 3, 2)),
    ]


@pytest.mark.parametrize(
    ("title", "period_type", "period"),
    [
        ("2024年一季度报告", "Q1", "2024-03-31"),
        ("2024年第一季度报告全文", "Q1", "2024-03-31"),
        ("北辰实业2020年半年报", "H1", "2020-06-30"),
        ("2024年半年度报告", "H1", "2024-06-30"),
        ("2024年三季度报告", "Q3", "2024-09-30"),
        ("2024年第三季度报告正文", "Q3", "2024-09-30"),
        ("北辰实业2024年年报", "FY", "2024-12-31"),
        ("2024年年度报告", "FY", "2024-12-31"),
    ],
)
def test_periodic_title_variants(title: str, period_type: str, period: str) -> None:
    actual_type, actual_period, status = classify_title(
        title, excluded_tokens=EXCLUSIONS
    )
    assert status == "ACCEPTED_ORIGINAL_FULL"
    assert actual_type == period_type
    assert actual_period == pd.Timestamp(period)


@pytest.mark.parametrize(
    "title",
    [
        "2024年年度报告摘要",
        "2024年半年度报告（更正版）",
        "2017年年度报告（更新后）",
        "2017年年度报告（已取消）",
        "2024年年度报告英文版",
    ],
)
def test_non_original_or_non_full_titles_are_excluded(title: str) -> None:
    period_type, report_period, status = classify_title(
        title, excluded_tokens=EXCLUSIONS
    )
    assert period_type is None
    assert report_period is None
    assert status.startswith("EXCLUDED_TITLE_TOKEN_")


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("600000", "600000.SH"),
        ("688001", "688001.SH"),
        ("000001", "000001.SZ"),
        ("300001", "300001.SZ"),
        ("200001", None),
        ("900901", None),
        ("830001", None),
    ],
)
def test_security_code_scope(code: str, expected: str | None) -> None:
    assert ts_code_from_cninfo_code(code) == expected


def test_normalization_uses_official_url_date_and_deduplicates() -> None:
    timezone = ZoneInfo("Asia/Shanghai")
    timestamp_ms = int(datetime(2024, 4, 30, tzinfo=timezone).timestamp() * 1000)
    records = [
        {
            "announcementId": "2",
            "secCode": "000001",
            "secName": "平安银行",
            "orgId": "gssz0000001",
            "pageColumn": "SZZB",
            "announcementTitle": "2024年一季度报告全文",
            "announcementTime": timestamp_ms,
            "adjunctUrl": "finalpage/2024-04-30/2.PDF",
            "adjunctType": "PDF",
            "adjunctSize": 100,
            "query_interval": "2024-04",
        },
        {
            "announcementId": "1",
            "secCode": "000001",
            "secName": "平安银行",
            "orgId": "gssz0000001",
            "pageColumn": "SZZB",
            "announcementTitle": "2024年一季度报告正文",
            "announcementTime": timestamp_ms,
            "adjunctUrl": "finalpage/2024-04-30/1.PDF",
            "adjunctType": "PDF",
            "adjunctSize": 90,
            "query_interval": "2024-04",
        },
    ]
    config = {
        "source": {"pdf_base_url": "https://static.cninfo.com.cn/"},
        "event_contract": {"excluded_title_tokens": EXCLUSIONS},
    }
    all_records, events = normalize_records(
        records,
        config=config,
        retrieved_at=datetime(2026, 8, 24, tzinfo=timezone),
    )
    assert len(all_records) == 2
    assert len(events) == 1
    assert events.iloc[0]["announcement_id"] == "1"
    assert events.iloc[0]["event_publication_date"] == pd.Timestamp("2024-04-30")
    assert bool(events.iloc[0]["official_internal_date_equal"])
