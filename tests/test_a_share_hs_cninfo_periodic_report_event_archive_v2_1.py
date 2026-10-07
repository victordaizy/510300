from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from research.a_share_hs_cninfo_periodic_report_event_archive_v2_1 import (
    normalize_records_with_chronology,
)


def config() -> dict:
    return {
        "source": {"pdf_base_url": "https://static.cninfo.com.cn/"},
        "event_contract": {
            "excluded_title_tokens": [
                "摘要",
                "取消",
                "更正",
                "修订",
                "更新后",
                "英文",
                "英文版",
                "提示性公告",
            ]
        },
    }


def timestamp_ms(value: str) -> int:
    return int(
        pd.Timestamp(value, tz="Asia/Shanghai").tz_convert("UTC").value // 1_000_000
    )


def record(identifier: str, publication_date: str) -> dict:
    return {
        "announcementId": identifier,
        "secCode": "600000",
        "secName": "测试公司",
        "orgId": "gssh0600000",
        "pageColumn": "SSE",
        "announcementTitle": "测试公司2024年年度报告",
        "announcementTime": timestamp_ms(publication_date),
        "adjunctUrl": f"finalpage/{publication_date}/{identifier}.PDF",
        "adjunctType": "PDF",
        "adjunctSize": 100,
        "query_interval": publication_date[:7],
    }


def test_invalid_early_candidate_is_removed_before_period_deduplication() -> None:
    all_records, events, invalid = normalize_records_with_chronology(
        [record("bad", "2024-04-26"), record("good", "2025-03-22")],
        config=config(),
        retrieved_at=datetime(2026, 8, 24, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert len(all_records) == 2
    assert events["announcement_id"].tolist() == ["good"]
    assert invalid["announcement_id"].tolist() == ["bad"]
    assert invalid["chronology_status"].tolist() == [
        "REJECTED_EVENT_PUBLICATION_BEFORE_REPORT_PERIOD"
    ]


def test_all_invalid_candidates_leave_security_period_without_event() -> None:
    _, events, invalid = normalize_records_with_chronology(
        [record("bad", "2024-04-26")],
        config=config(),
        retrieved_at=datetime(2026, 8, 24, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert events.empty
    assert len(invalid) == 1


def test_valid_candidate_is_unchanged() -> None:
    _, events, invalid = normalize_records_with_chronology(
        [record("good", "2025-03-22")],
        config=config(),
        retrieved_at=datetime(2026, 8, 24, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert events["announcement_id"].tolist() == ["good"]
    assert invalid.empty
