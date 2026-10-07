"""检查现金事件、实施时钟、成员缺口与季节阈值的关键边界。"""
import numpy as np
import pandas as pd
import pytest

from research.constituent_dividend_calendar_v1 import normalize_schedules, build_daily, seasonal_threshold, matched_values


def raw_event(**changes):
    row = {"ts_code": "S000", "div_proc": "实施", "cash_div_tax": 1., "imp_ann_date": "20150105",
           "record_date": "20150108", "ex_date": "20150109", "pay_date": "20150109"}
    return {**row, **changes}


def test_cash_implementation_not_proposal_and_missing_pay_date_fails():
    rows = [raw_event(), raw_event(), raw_event(div_proc="预案"), raw_event(cash_div_tax=0.)]
    schedules, receipt = normalize_schedules(pd.DataFrame(rows))
    assert len(schedules) == 1 and receipt["collapsed_same_schedule_rows"] == 1
    with pytest.raises(AssertionError, match="必要日期"):
        normalize_schedules(pd.DataFrame([raw_event(pay_date=None)]))


def test_known_after_day_end_payment_removed_and_missing_members_not_zero():
    calendar = pd.bdate_range("2015-01-05", periods=30)
    membership = pd.DataFrame([(d, f"S{i:03}") for d in calendar for i in range(300)], columns=["membership_date", "symbol"])
    schedules, _ = normalize_schedules(pd.DataFrame([raw_event()]))
    covered = [f"S{i:03}" for i in range(290)]
    daily = build_daily(calendar, membership, schedules, covered)
    assert daily.scheduled_issuer_count.iloc[0] == 0
    assert daily.scheduled_issuer_count.iloc[1] == 1
    assert daily.breadth.iloc[1] == 1 / 290
    assert daily.scheduled_issuer_count.iloc[4] == 0
    missing = build_daily(calendar, membership, schedules, covered[:280])
    assert missing.breadth.isna().all()


def test_threshold_excludes_current_future_expired_and_other_month_values():
    dates = pd.to_datetime(["2014-01-02", "2015-01-02", "2016-01-04", "2016-02-02", "2017-01-03", "2018-01-03"])
    frame = pd.DataFrame({"date": dates, "breadth": [999., .2, .4, 999., .5, 999.], "status": "SOURCE_OK"})
    result = seasonal_threshold(frame, minimum=1)
    # 2015年1月2日已在两年窗口之外，仅2016年1月4日进入当前基准。
    assert result.seasonal_median.iloc[4] == .4 and result.selected.iloc[4]
    changed = frame.copy()
    changed.loc[[0, 1, 3, 5], "breadth"] = -999.
    assert seasonal_threshold(changed, minimum=1).seasonal_median.iloc[4] == .4


def test_matched_control_uses_same_year_month_and_censor_stays_missing():
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-06", "2020-01-13", "2020-02-03", "2020-02-10"]),
                          "gross_return": [.02, 0., -.1, np.nan], "selected": [True, False, True, True]})
    valid = matched_values(frame)
    assert len(valid) == 3
    np.testing.assert_allclose(valid.matched_month_mean, [.01, .01, -.1])
