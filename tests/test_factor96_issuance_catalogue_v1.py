"""发行目录的范围、字段和标题分派边界。"""
import pandas as pd
import pytest

from research.factor96_issuance_catalogue_v1 import classify_title, normalize_page


def row(**changes):
    original = {"secCode": "601555", "orgId": "issuer", "announcementId": "123", "announcementTitle": "东吴证券配股发行结果公告",
        "announcementTime": int(pd.Timestamp("2021-12-20", tz="Asia/Shanghai").timestamp() * 1000),
        "adjunctUrl": "finalpage/2021-12-20/123.PDF", "announcementType": "0105", "columnId": "0901"}
    return {**original, **changes}


def parse(value):
    return normalize_page(value, {"601555": "issuer"}, "rights_title", "2021-01-01", "2021-12-31")


def test_source_clock_and_no_admission():
    total, rows = parse({"totalAnnouncement": 1, "announcements": [row()]})
    assert total == 1 and rows[0]["catalogue_date"] == "2021-12-20"
    assert rows[0]["title_role"] == "ISSUER_CALENDAR_DOCUMENT_CANDIDATE"
    assert not rows[0]["trading_feature_admitted"]


def test_empty_query_is_only_empty_filtered_source():
    assert parse({"totalAnnouncement": 0, "announcements": None}) == (0, [])


@pytest.mark.parametrize("change", [{"secCode": "600030"}, {"orgId": "another"}, {"announcementTitle": "增发公告"},
    {"announcementTime": int(pd.Timestamp("2022-01-01", tz="Asia/Shanghai").timestamp() * 1000)},
    {"adjunctUrl": "finalpage/../private.pdf"}])
def test_rejects_outside_source_contract(change):
    with pytest.raises(ValueError):
        parse({"totalAnnouncement": 1, "announcements": [row(**change)]})


@pytest.mark.parametrize("body", [{"totalAnnouncement": -1, "announcements": []}, {"totalAnnouncement": True, "announcements": []},
    {"totalAnnouncement": 1, "announcements": []}, {"totalAnnouncement": 0, "announcements": [row()]}])
def test_rejects_invalid_counts(body):
    with pytest.raises(ValueError):
        parse(body)


def test_title_roles_do_not_turn_bonds_or_ipo_into_equity_supply():
    assert classify_title("首次公开发行股票上市公告书") == "IPO_OUTSIDE_M06_CORE"
    assert classify_title("非公开发行公司债券发行结果公告") == "OTHER_FINANCING_OR_MIXED_TITLE"
    assert classify_title("关于分配股息款的公告") == "LEXICAL_FALSE_POSITIVE_CANDIDATE"
    assert classify_title("关于终止向特定对象发行股票的公告") == "TERMINATION_OR_REJECTION_CANDIDATE"
    assert classify_title("配股发行结果更正公告") == "AMENDMENT_CANDIDATE"
    assert classify_title("非公开发行股票法律意见书") == "SUPPORTING_DOCUMENT"
