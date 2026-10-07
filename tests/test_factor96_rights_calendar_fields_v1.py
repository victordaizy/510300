"""验证公告更正不提前生效、已知上市与缴款分开以及数字单位。"""
from research.factor96_rights_calendar_fields_v1 import asof_case, case_versions, extract_candidates, number_literals, MONEY, SHARES


def versions():
    return case_versions([{"document_id": key, "catalogue_date": day, "catalogue_timestamp": day + "T00:00:00+08:00", "source_url": "source"}
        for key, day in zip(["1201854885", "1201897631", "1201900999", "1201908424"], ["2015-12-24", "2016-01-07", "2016-01-08", "2016-01-13"])])


def test_before_first_notice_is_no_view():
    assert asof_case(versions(), "2015-12-23T23:59:59+08:00")["status"] == "NO_VIEW"


def test_plan_budget_is_not_actual_subscription():
    state = asof_case(versions(), "2015-12-24T23:59:59+08:00")
    assert state["actual_subscription_cents"] is None
    assert state["plan_shares_times_price_cents"] == 1277640000000
    assert state["stated_proceeds_budget_cap_cents"] == 1500000000000


def test_clearing_day_is_outside_payment_window():
    state = asof_case(versions(), "2016-01-06T12:00:00+08:00")
    assert state["payment_phase"] == "ANNOUNCED_PAYMENT_WINDOW_ENDED"
    assert state["actual_subscribed_shares"] is None


def test_correction_not_backfilled_before_known_time():
    state = asof_case(versions(), "2016-01-08T12:00:00+08:00")
    assert state["actual_subscribed_shares"] == 1496665905
    assert state["actual_subscription_cents"] == 1225769376195


def test_correction_updates_exact_amount_after_known_time():
    state = asof_case(versions(), "2016-01-08T23:59:59+08:00")
    assert state["actual_subscribed_shares"] == 1496671674
    assert state["actual_subscription_cents"] == 1225774101006


def test_exrights_date_is_not_new_share_listing():
    state = asof_case(versions(), "2016-01-07T23:59:59+08:00")
    assert state["scheduled_exrights_resumption_date"] == "2016-01-07"
    assert state["announced_listing_date"] is None


def test_listing_date_only_visible_from_listing_notice():
    assert asof_case(versions(), "2016-01-13T12:00:00+08:00")["announced_listing_date"] is None
    assert asof_case(versions(), "2016-01-13T23:59:59+08:00")["announced_listing_date"] == "2016-01-18"


def test_conflicting_original_year_is_not_silently_fixed():
    state = asof_case(versions(), "2015-12-24T23:59:59+08:00")
    assert state["scheduled_result_publication_date"] is None
    assert state["result_date_conflict"] == ["2015-01-07", "2016-01-07"]


def test_units_are_exact_decimals_without_float_rounding():
    assert number_literals("12,257,741,010.06元", MONEY)[0]["normalized_number"] == "12257741010.06"
    assert number_literals("1.496671674亿股", SHARES)[0]["normalized_number"] == "149667167.400000000"


def test_generic_candidates_never_admit_economic_fields():
    rows = extract_candidates([{"page": 1, "text": "配股缴款及清算期间2015年12月29日至2016年1月6日，实际缴款1月5日止。"}], "sample")
    assert rows and all(r["event_field_admitted"] is False for r in rows)
    assert rows[0]["date_literals"][0]["value"] == "2015-12-29"
