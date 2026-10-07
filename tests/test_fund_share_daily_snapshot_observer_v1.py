"""验证观察器不会以基线、未来版本、缺日或错误单位伪造可用份额变化。"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from research import fund_share_daily_snapshot_observer_v1 as observer

ZONE = ZoneInfo("Asia/Shanghai")
SESSIONS = [date(2026, 9, 30), date(2026, 10, 8), date(2026, 10, 9), date(2026, 10, 12)]


def payload(value="1,234.50", code="510300", day="2026-10-08"):
    return {"result": [{"SEC_CODE": code, "STAT_DATE": day, "TOT_VOL": value}]}


def observed(day="2026-10-08", value="1,234.50"):
    d = date.fromisoformat(day)
    return observer.normalize(payload(value=value, day=day), datetime(d.year, d.month, d.day, 23, 10, tzinfo=ZONE), d, SESSIONS, "OBSERVE")


def test_units_are_shares_and_never_cash_flow():
    row = observed()
    assert row["total_units"] == "12345000.00"
    assert row["original_unit"] == "万份"
    assert row["cash_flow_cny"] == "NOT_IDENTIFIED"


def test_another_security_is_not_silently_used():
    row = observer.normalize(payload(code="510310"), datetime(2026, 10, 8, 23, 10, tzinfo=ZONE), SESSIONS[1], SESSIONS, "OBSERVE")
    assert row["status"] == "NO_VIEW_TARGET_ROW_MISSING_OR_DUPLICATED"


def test_scale_field_does_not_replace_share_field():
    value = {"result": [{"SEC_CODE": "510300", "STAT_DATE": "2026-10-08", "SCALE": "1234"}]}
    row = observer.normalize(value, datetime(2026, 10, 8, 23, 10, tzinfo=ZONE), SESSIONS[1], SESSIONS, "OBSERVE")
    assert row["status"] == "NO_VIEW_REQUIRED_SHARE_FIELDS_MISSING"


def test_stale_response_cannot_become_requested_new_day():
    row = observer.normalize(payload(day="2026-09-30"), datetime(2026, 10, 8, 23, 10, tzinfo=ZONE), SESSIONS[1], SESSIONS, "OBSERVE")
    assert row["status"] == "NO_VIEW_STALE_OR_DIFFERENT_REQUESTED_STATISTICS_DATE"


def test_baseline_cannot_form_first_forward_change():
    seed = observer.normalize(payload(day="2026-09-30"), datetime(2026, 10, 6, 10, 0, tzinfo=ZONE), None, SESSIONS, "SEED")
    value = observer.daily_change(observed(), seed, datetime(2026, 10, 9, 16, 0, tzinfo=ZONE), SESSIONS)
    assert value["status"] == "NOT_COMPUTED_NEEDS_TWO_NEW_OBSERVED_VERSIONS"
    assert value["fractional_unit_change"] is None


def test_missing_exchange_date_is_not_interpolated():
    value = observer.daily_change(observed("2026-10-12"), observed("2026-10-08"), datetime(2026, 10, 13, 16, 0, tzinfo=ZONE), SESSIONS)
    assert value["status"] == "NOT_COMPUTED_NONCONSECUTIVE_EXCHANGE_DATES"


def test_later_capture_cannot_be_used_at_prior_close():
    value = observer.daily_change(observed("2026-10-09"), observed(), datetime(2026, 10, 9, 16, 0, tzinfo=ZONE), SESSIONS)
    assert value["status"] == "NOT_COMPUTED_VERSION_NOT_YET_KNOWN"


def test_known_consecutive_observations_can_form_descriptive_change():
    value = observer.daily_change(observed("2026-10-09", "1357.95"), observed(), datetime(2026, 10, 12, 16, 0, tzinfo=ZONE), SESSIONS)
    assert value["fractional_unit_change"] == "0.1"
    assert value["trade_signal"] == "NOT_GENERATED"
    assert value["financial_admission"] == "NOT_ADMITTED"


def test_invalid_numerical_values_are_not_zero_filled():
    for value in ["NaN", "Infinity", "-1", "0", "", None, True]:
        with pytest.raises(ValueError):
            observer.positive_decimal(value)


def test_jsonp_is_parsed_without_executing_code():
    assert observer.decode_response(b'callback({"result":[]});') == {"result": []}
    with pytest.raises(ValueError):
        observer.decode_response(b'callback({"result":[]});dangerous()')
