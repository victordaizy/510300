"""以实际留存官方页面检查单位、分页、身份和日期边界，不访问网络。"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

import json
import pytest

from research import official_dividend_coverage_refresh_v1 as source

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "data/forward/510300_official_dividend_coverage_survey_v1/20260906T033228560286"
CONFIG = json.loads((ROOT / source.CONFIG).read_text(encoding="utf-8"))
BASELINE = json.loads((ROOT / CONFIG["baseline_coverage"]).read_text(encoding="utf-8"))


def test_real_official_table_matches_all_fourteen_events_and_unit() -> None:
    rows = source.parse_manager((CAPTURE / "fund_510300_product.html").read_bytes(), "510300")
    source.compare_manager(rows, source.ledger_rows(ROOT / CONFIG["baseline_dividends"]))
    assert len(rows) == 14
    assert rows[0] == {"record_date": "2026-01-16", "payment_date": "2026-01-27", "cash_dividend_per_share": "0.123"}


@pytest.mark.parametrize("change", ["wrong_fund", "wrong_unit", "missing_pagination"])
def test_wrong_identity_unit_or_partial_table_is_not_accepted(change: str) -> None:
    body = (CAPTURE / "fund_510300_product.html").read_bytes()
    if change == "wrong_fund":
        body = body.replace(b"seekProject('510300')", b"seekProject('510500')")
    elif change == "wrong_unit":
        body = body.replace("每10份基金份额分红（元）".encode(), "每份基金份额分红（元）".encode())
    else:
        body = body.replace(b'showFhTable(1,allPage,10', b'unknownPagination(1,allPage,10')
    with pytest.raises(source.CoverageError):
        source.parse_manager(body, "510300")


def test_new_manager_event_requires_reconciliation() -> None:
    rows = source.parse_manager((CAPTURE / "fund_510300_product.html").read_bytes(), "510300")
    rows.append({"record_date": "2026-09-07", "payment_date": "2026-09-15", "cash_dividend_per_share": "0.01"})
    with pytest.raises(source.CoverageError, match="RECONCILIATION"):
        source.compare_manager(rows, source.ledger_rows(ROOT / CONFIG["baseline_dividends"]))


def test_real_sse_full_page_has_sixteen_records_and_known_dividend_anchor() -> None:
    page = json.loads((CAPTURE / "sse_fund_announcements_page_1.json").read_text(encoding="utf-8"))
    checked = source.validate_announcements([page], CONFIG, date(2026, 9, 6), BASELINE)
    assert checked["total_announcements"] == 16
    assert checked["pages"] == 1
    assert len(checked["known_dividend_announcements"]) == 1
    assert checked["new_unreconciled_dividend_announcements"] == 0


@pytest.mark.parametrize("change", ["partial", "wrong_fund", "anchor_missing", "new_dividend"])
def test_query_defects_do_not_become_no_new_dividend(change: str) -> None:
    page = json.loads((CAPTURE / "sse_fund_announcements_page_1.json").read_text(encoding="utf-8"))
    if change == "partial":
        page["result"].pop()
    elif change == "wrong_fund":
        page["result"][0]["SECURITY_CODE"] = "510500"
    elif change == "anchor_missing":
        page["result"] = [row for row in page["result"] if "分红" not in row["TITLE"]]
        page["pageHelp"]["total"] -= 1
    else:
        new = deepcopy(page["result"][0])
        new.update({"TITLE": "华泰柏瑞沪深300交易型开放式指数证券投资基金分红公告", "URL": "/disclosure/fund/announcement/c/new/2026-09-04/510300_new.pdf", "SSEDATE": "2026-09-04"})
        page["result"].append(new)
        page["pageHelp"]["total"] += 1
    with pytest.raises(source.CoverageError):
        source.validate_announcements([page], CONFIG, date(2026, 9, 6), BASELINE)


@pytest.mark.parametrize("moment,expected", [
    ("2026-09-04T14:59:00+08:00", date(2026, 9, 3)),
    ("2026-09-04T15:00:00+08:00", date(2026, 9, 4)),
    ("2026-09-06T03:00:00+08:00", date(2026, 9, 4)),
])
def test_coverage_never_extends_to_an_unclosed_trading_day(moment: str, expected: date) -> None:
    calendar = [date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 7)]
    assert source.completed_trade_day(calendar, datetime.fromisoformat(moment)) == expected


def test_stale_http_date_cannot_support_current_coverage() -> None:
    moment = datetime.fromisoformat("2026-09-06T03:37:48+08:00")
    source.check_http_clock({"date": "Sat, 05 Sep 2026 19:37:53 GMT"}, moment, date(2026, 9, 4), CONFIG)
    with pytest.raises(source.CoverageError):
        source.check_http_clock({"Date": "Fri, 14 Aug 2026 07:00:00 GMT"}, moment, date(2026, 9, 4), CONFIG)
