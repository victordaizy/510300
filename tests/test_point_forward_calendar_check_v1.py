"""官方日历覆盖的回归检查：不得遗漏空仓日或把空窗口当验证通过。"""
import pandas as pd
import pytest

from research.point_forward_calendar_check_v1 import coverage_from_calendar
from research.point_forward_observer_inputs_v1 import coverage_rows


def fixture_rows():
    calendar = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    signals = pd.DataFrame({"candidate": ["A"] * 3, "origin": calendar[:-1],
                            "execution_date": calendar[1:], "target": [0., 0., 1.]})
    registry = signals.rename(columns={"execution_date": "planned_execution_date"}).copy()
    registry["source_data_max_date"] = registry.origin
    registry["actual_generated_at"] = [(x.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)).isoformat()
                                       for x in registry.origin]
    registry["signal_status"] = "AVAILABLE"
    return signals, registry, calendar


def checked(signals, registry, calendar, last="2026-01-08"):
    return coverage_from_calendar(signals, registry, calendar, "2026-01-06", last, "2026-12-31", ("A",))


def test_both_tables_missing_flat_day_cannot_shrink_expected_calendar():
    signals, registry, calendar = fixture_rows()
    signals, registry = signals.drop(index=1), registry.drop(index=1)
    previous = coverage_rows(signals, registry, "2026-01-06", "2026-01-08")
    assert previous.timely_matching.all()
    table, result = checked(signals, registry, calendar)
    assert len(table) == 3
    assert result["invalid_candidate_sessions"] == 1
    assert not result["full_calendar_coverage"]
    absent = table.loc[table.execution_date.eq(pd.Timestamp("2026-01-07"))].iloc[0]
    assert absent.reason == "MISSING_SIGNAL|MISSING_REGISTRATION"


def test_future_intents_and_empty_window_do_not_prove_coverage():
    signals, registry, calendar = fixture_rows()
    table, result = checked(signals, registry, calendar, "2026-01-05")
    assert table.empty
    assert result["status"] == "NO_NEW_OBSERVATIONS"
    assert result["full_calendar_coverage"] is False


@pytest.mark.parametrize("fault", ["late", "no_timezone", "future_source", "wrong_previous_session", "duplicate"])
def test_registration_clock_source_date_and_duplicate_remain_invalid(fault):
    signals, registry, calendar = fixture_rows()
    if fault == "late":
        registry.loc[1, "actual_generated_at"] = "2026-01-07T09:30:00+08:00"
    elif fault == "no_timezone":
        registry.loc[1, "actual_generated_at"] = "2026-01-06T16:00:00"
    elif fault == "future_source":
        registry.loc[1, "source_data_max_date"] = pd.Timestamp("2026-01-07")
    elif fault == "wrong_previous_session":
        signals.loc[1, "origin"] = registry.loc[1, "origin"] = pd.Timestamp("2026-01-05")
    else:
        registry = pd.concat([registry, registry.iloc[[1]]], ignore_index=True)
    _, result = checked(signals, registry, calendar)
    assert result["invalid_candidate_sessions"] == 1
    assert not result["full_calendar_coverage"]


def test_calendar_exhaustion_cannot_hide_later_days():
    signals, registry, calendar = fixture_rows()
    with pytest.raises(ValueError, match="日历覆盖范围不足"):
        checked(signals, registry, calendar, "2027-01-05")


def test_full_clock_and_calendar_coverage_is_not_strategy_validation():
    signals, registry, calendar = fixture_rows()
    _, result = checked(signals, registry, calendar)
    assert result["full_calendar_coverage"]
    assert result["independent_validation"] == "NOT_ESTABLISHED"
