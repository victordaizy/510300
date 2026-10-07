"""检验公布时钟、计数去重和实际持有期的关键行为。"""
import numpy as np
import pandas as pd

from research.shareholder_disclosure_breadth_v1 import normalize, build_daily, nonoverlap


def raw_row(**kwargs):
    row = {"NOTICE_DATE": "2017-02-01", "START_DATE": "2017-01-30", "END_DATE": "2017-01-31",
           "CHANGE_NUM": 2.0, "SECURITY_CODE": "600000", "HOLDER_NAME": "甲", "DIRECTION": "增持", "MARKET": "二级市场"}
    row.update(kwargs)
    return row


def test_notice_clock_and_duplicates():
    raw = pd.DataFrame([raw_row(), raw_row(), raw_row(HOLDER_NAME="乙")])
    events, annotated = normalize(raw)
    assert len(events) == 2 and annotated.duplicate.sum() == 1
    calendar = pd.bdate_range("2016-12-01", "2017-02-03")
    members = pd.DataFrame([(d, f"{600000+i}.SH") for d in calendar for i in range(300)], columns=["membership_date", "symbol"])
    daily = build_daily(calendar, members, events).set_index("date")
    assert daily.loc["2017-02-01", "buy_count"] == 0
    assert daily.loc["2017-02-02", "buy_count"] == 1


def test_two_sides_cancel_and_missing_query_is_not_zero():
    events, _ = normalize(pd.DataFrame([raw_row(), raw_row(HOLDER_NAME="乙", DIRECTION="减持")]))
    calendar = pd.bdate_range("2016-12-01", "2017-02-03")
    members = pd.DataFrame([(d, f"{600000+i}.SH") for d in calendar for i in range(300)], columns=["membership_date", "symbol"])
    daily = build_daily(calendar, members, events).iloc[-1]
    assert daily.buy_count == 1 and daily.sell_count == 1 and not daily.selected
    unavailable = build_daily(calendar, members, events, False)
    assert unavailable.breadth.isna().all() and unavailable.status.eq("NO_VIEW_SOURCE").all()


def test_exclude_nonmarket_and_future_completion():
    events, annotated = normalize(pd.DataFrame([raw_row(), raw_row(MARKET="股权激励"), raw_row(END_DATE="2017-02-02")]))
    assert len(events) == 1
    assert annotated.valid_clock.tolist() == [True, True, False]


def test_overlapping_triggers_cannot_reset_expiry():
    labels = pd.DataFrame({"information_status": ["READY"]*5, "selected": [True]*5,
        "idx": [0, 5, 10, 22, 27], "status": ["MATURE", "MATURE", "MATURE", "UNFILLED_UPPER_LIMIT", "CENSORED_EXIT_AFTER_SAMPLE"],
        "exit_idx": [21, 26, 31, 23, np.nan]})
    assert nonoverlap(labels).tolist() == [True, False, False, True, True]
