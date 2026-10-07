"""针对真实目录故障验证原文、发行人和响应完整性的边界。"""
import pytest

from research.factor96_issuance_catalogue_completion_v1 import clean_title, normalize


COMPANIES = [{"symbol": "600029.SH", "code": "600029", "org_id": "gssh0600029"}]


def body(title="发行公告", code="600029", organization="gssh0600029"):
    return {"totalAnnouncement": 1, "announcements": [{"secCode": code, "orgId": organization,
        "announcementId": "1202334276", "announcementTitle": title, "announcementTime": 1463932800000,
        "adjunctUrl": "finalpage/2016-05-23/1202334276.PDF"}]}


def decode(value):
    return normalize(value, COMPANIES, "issuance_title", "2015-01-01", "2025-12-31")


def test_literal_chinese_angle_brackets_are_not_html():
    title = "关于<非公开发行人民币普通股>的独立意见"
    assert clean_title(title) == title
    assert decode(body(title))[1][0]["title"] == title


def test_only_em_highlights_and_entities_are_decoded():
    assert clean_title("<em>发行</em>与&lt;股份认购&gt;") == "发行与<股份认购>"


def test_bond_security_keeps_reported_code_without_equity_assertion():
    row = decode(body(code="136053"))[1][0]
    assert row["reported_security_code"] == "136053"
    assert row["query_security_code"] == "600029"
    assert row["code_relationship"] == "OTHER_CODE_SAME_ORGANIZATION_UNRESOLVED"
    assert row["trading_feature_admitted"] is False


def test_wrong_organization_rejected_even_if_security_code_matches():
    with pytest.raises(ValueError, match="组织"):
        decode(body(organization="other"))


def test_wrong_keyword_rejected():
    with pytest.raises(ValueError, match="关键词"):
        decode(body("股东大会公告"))


def test_zero_total_requires_no_rows():
    assert decode({"totalAnnouncement": 0, "announcements": None}) == (0, [])
    invalid = body()
    invalid["totalAnnouncement"] = 0
    with pytest.raises(ValueError, match="长度"):
        decode(invalid)


def test_duplicate_response_cannot_prove_completion():
    value = body()
    value["announcements"] *= 2
    value["totalAnnouncement"] = 2
    with pytest.raises(ValueError, match="重复"):
        decode(value)


def test_server_truncation_is_visible_as_total_greater_than_rows():
    value = body()
    value["totalAnnouncement"] = 207
    count, rows = decode(value)
    assert count == 207 and len(rows) == 1
