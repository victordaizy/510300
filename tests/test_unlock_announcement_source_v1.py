"""覆盖公告搜索分页和标题分类会影响历史完整性的边界。"""
import json

import pandas as pd

import scripts.collect_510300_unlock_announcements_v1 as source


def config():
    return json.loads(source.CONFIG.read_text(encoding="utf-8"))


def test_page_count_includes_partial_last_page_despite_api_floor_value():
    assert source.expected_pages(190, 30) == 7
    assert source.expected_pages(303, 30) == 11
    assert source.expected_pages(348, 30) == 12
    assert source.expected_pages(0, 30) == 1


def test_title_filter_excludes_segmented_search_false_positive_and_opinions():
    cfg = config()
    assert source.title_kind("关于限售股份<em>上市</em><em>流通</em>的提示性公告", cfg) == "ELIGIBLE_ORIGINAL_DISCLOSURE"
    assert source.title_kind("首次公开发行前已发行股份上市流通提示性公告", cfg) == "ELIGIBLE_ORIGINAL_DISCLOSURE"
    assert source.title_kind("关于公司限售股份上市流通的核查意见", cfg) == "SUPPORTING_OPINION_NOT_COUNTED"
    assert source.title_kind("关于限售股份上市流通日期更正的公告", cfg) == "REVISION_TITLE_RETAINED_SEPARATELY"
    assert source.title_kind("可转换债券上市流通公告", cfg) == "NOT_TARGET_TITLE"


def test_month_windows_preserve_partial_end_date():
    cfg = config()
    windows = source.month_windows(cfg)
    assert len(windows) == 140
    assert windows[0] == ("2015-01", "2015-01-01", "2015-01-31")
    assert windows[-1] == ("2026-08", "2026-08-01", "2026-08-14")


def install_fake_session(monkeypatch, duplicate=False, changing_total=False):
    class Response:
        status_code = 200
        url = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
        headers = {}

        def __init__(self, page):
            count = 30 if page == 1 else 1
            first = 0 if duplicate or page == 1 else 30
            items = [{"announcementId": str(i), "announcementTime": int(pd.Timestamp("2024-01-03", tz="Asia/Shanghai").timestamp() * 1000),
                      "announcementTitle": "限售股份上市流通提示性公告"} for i in range(first, first + count)]
            self.payload = {"totalAnnouncement": 32 if changing_total and page == 2 else 31, "totalpages": 1, "announcements": items}
            self.content = json.dumps(self.payload).encode("utf-8")

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.headers = {}

        def post(self, url, data, timeout):
            return Response(int(data["pageNum"]))

        def close(self):
            return None

    monkeypatch.setattr(source.requests, "Session", Session)


def test_collection_gets_omitted_partial_page_and_all_unique_records(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "ROOT", tmp_path)
    install_fake_session(monkeypatch)
    report, rows = source.collect_month(("2024-01", "2024-01-01", "2024-01-31"), tmp_path, config())
    assert report["status"] == "PASS_COMPLETE_DECLARED_MONTH"
    assert len(rows) == 31 and report["actual_pages"] == 2
    assert report["reported_totalpages_first_response"] == 1


def test_repeated_id_is_incomplete_not_silently_deduplicated_to_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "ROOT", tmp_path)
    install_fake_session(monkeypatch, duplicate=True)
    report, rows = source.collect_month(("2024-01", "2024-01-01", "2024-01-31"), tmp_path, config())
    assert report["status"] == "INCOMPLETE_MONTH_NO_ZERO_FILL" and len(rows) == 31
    assert "唯一性" in report["error"]


def test_changing_declared_total_stops_month_and_preserves_raw_responses(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "ROOT", tmp_path)
    install_fake_session(monkeypatch, changing_total=True)
    report, rows = source.collect_month(("2024-01", "2024-01-01", "2024-01-31"), tmp_path, config())
    assert report["status"] == "INCOMPLETE_MONTH_NO_ZERO_FILL" and len(rows) == 30
    assert (tmp_path / "2024-01/page_002_attempt_1.json").exists()
