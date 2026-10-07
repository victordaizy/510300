"""检验公布钟、来源恢复、重复计数和未来不影响既有描述。"""
import numpy as np
import pandas as pd

from research import entry_information_sequence_inputs_v1 as inputs


def data():
    dates = pd.bdate_range("2024-01-02", periods=8)
    clock = dates[0].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    return pd.DataFrame({"date": dates, "decision_time": dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16),
        "stage_entry_type": ["NONE", "REPRICING", "NONE", "REPRICING", "PULLBACK", "NONE", "REPRICING", "NONE"],
        "orders_available_at": clock, "orders_known": True, "orders_reference_period": "2023-12",
        "funding_policy_known_at": clock, "rate": 1.8,
        "funding_available_at": clock, "funding_known": True, "fund_stat_date": dates[0]-pd.Timedelta(days=1),
        "margin_available_at": clock, "margin_known": True, "margin_stat_date": dates[0]-pd.Timedelta(days=1)})


def test_prefix_and_future_change_do_not_modify_past():
    frame = data()
    full = inputs.describe(frame)
    future = frame.copy()
    future.loc[5:, "orders_reference_period"] = "FUTURE"
    future.loc[5:, "stage_entry_type"] = "REPAIR"
    changed = inputs.describe(future)
    cut = inputs.describe(frame.iloc[:5])
    pd.testing.assert_frame_equal(full.iloc[:5], cut, check_exact=True)
    pd.testing.assert_frame_equal(changed.iloc[:5], cut, check_exact=True)


def test_known_restored_old_identity_is_not_new_publication():
    frame = data()
    frame.loc[2, "orders_known"] = False
    result = inputs.describe(frame)
    assert not result.orders_clock_admitted.iloc[2]
    assert not result.orders_first_observed_here.iloc[3]
    assert not result.orders_publication_since_previous_decision.iloc[3]
    assert result.orders_earlier_same_source_route_events.iloc[3] == 1


def test_first_seen_old_clock_is_not_new_news():
    frame = data()
    frame.loc[:2, "orders_known"] = False
    result = inputs.describe(frame)
    assert result.orders_first_observed_here.iloc[3]
    assert result.orders_first_observed_without_new_publication.iloc[3]
    assert not result.orders_publication_since_previous_decision.iloc[3]


def test_future_clock_excludes_even_original_known_flag():
    frame = data()
    frame.loc[3, "orders_available_at"] = frame.decision_time.iloc[3] + pd.Timedelta(seconds=1)
    result = inputs.describe(frame)
    assert result.orders_future_clock.iloc[3]
    assert not result.orders_clock_admitted.iloc[3]
    assert not result.orders_first_observed_here.iloc[3]
    assert np.isnan(result.orders_earlier_same_source_route_events.iloc[3])


def test_repeat_count_is_by_source_and_route_and_uses_only_earlier_events():
    result = inputs.describe(data())
    assert result.orders_earlier_same_source_route_events.iloc[1] == 0
    assert result.orders_earlier_same_source_route_events.iloc[3] == 1
    assert result.orders_earlier_same_source_route_events.iloc[4] == 0
    assert result.orders_earlier_same_source_route_events.iloc[6] == 2


def test_publication_interval_is_open_left_closed_right_and_timezone_aware():
    frame = data()
    frame.loc[3, "orders_reference_period"] = "NEW"
    frame.loc[3, "orders_available_at"] = frame.decision_time.iloc[2]
    result = inputs.describe(frame)
    assert not result.orders_publication_since_previous_decision.iloc[3]
    frame.loc[3, "orders_available_at"] = frame.decision_time.iloc[3].tz_convert("UTC")
    result = inputs.describe(frame)
    assert result.orders_clock_admitted.iloc[3]
    assert result.orders_publication_since_previous_decision.iloc[3]
