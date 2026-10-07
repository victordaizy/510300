"""验证实际进入后的信息身份和因果时序，不用交易盈亏优化规则。"""
import numpy as np
import pandas as pd
import pytest

from research import actual_holding_information_inputs_v1 as inputs


def row(date, **changes):
    last = pd.Timestamp(date).to_period("W-FRI").start_time.normalize() - pd.Timedelta(days=1)
    result = {"date": pd.Timestamp(date), "decision_time": inputs.clock(date) + pd.Timedelta(hours=16),
        "ac": 10.2, "cash_high": 10.5, "cash_low": 10., "daily_hist": .1, "weekly_hist": .1,
        "complete_week_first": last - pd.Timedelta(days=4), "complete_week_last": last,
        "complete_week_low": 10.1, "complete_week_close": 10.8, "higher_complete_week_low_and_close": True,
        "log_relative_volume": 0., "up_volume_balance5": 0., "volatility20_60": 1., "funding_gap_pp": np.nan,
        "funding_gap_change5": np.nan, "financing_net_change5": np.nan, "pmi_orders_level": np.nan,
        "pmi_orders_change": np.nan, "industry_positive5_fraction": np.nan, "industry_relative_positive5_fraction": np.nan,
        "industry_source_age_days": np.nan, "industry_snapshot_id": "UNKNOWN", "leader_ids": "", "leader_names": "",
        "leader_mean_return5": np.nan, "leader_mean_relative5": np.nan, "rotation_churn5blocks": np.nan}
    result.update(changes)
    return pd.Series(result)


def receipts():
    return pd.DataFrame(columns=["first_original_decision_date", "node_id", "source_available_upper", "actual_information_role", "common_source_id"])


def sample_account(dates):
    daily = pd.DataFrame({"date": pd.to_datetime(dates), "shares": [100, 100, 0], "exposure": [.1, .1, 0.]})
    trades = pd.DataFrame({"cycle_id": [1], "entry_date": pd.to_datetime([dates[0]]), "net_pnl": [99.]})
    return {"daily": daily, "trades": trades}


def test_week_with_four_pre_entry_days_is_not_wholly_new():
    entry = row("2021-12-10")
    observation = row("2021-12-13", ac=10.8)
    result = inputs.price_view(observation, entry)
    assert result["previous_week_low_and_close_raised"]
    assert not result["complete_week_all_observed_days_after_entry"]
    later = row("2021-12-20", ac=10.8, complete_week_first=pd.Timestamp("2021-12-13"), complete_week_last=pd.Timestamp("2021-12-17"))
    assert inputs.price_view(later, entry)["post_entry_complete_week_acceptance_now"]


def test_fixed_actual_entry_anchor_and_same_day_not_entry_signal():
    entry = row("2021-12-10")
    assert not inputs.price_view(entry, entry)["post_entry_price_accepted_now"]
    later = row("2021-12-13", ac=10.7, cash_high=11., cash_low=10.4)
    result = inputs.price_view(later, entry)
    assert result["post_entry_price_accepted_now"]
    assert result["entry_day_high_anchor"] == 10.5 and result["entry_day_low_anchor"] == 10.


def test_policy_clock_new_old_confirmation_and_common_source():
    entered = inputs.clock("2024-09-24 09:30")
    nodes = [dict(node_id="OLD", source_available_upper=inputs.clock("2024-09-24 09:19"), actual_information_role="RECORDED_SUPPORT_INFORMATION", common_source_id="OLD"),
        dict(node_id="NEW1", source_available_upper=inputs.clock("2024-09-24 11:00"), actual_information_role="RECORDED_SUPPORT_INFORMATION", common_source_id="ONE"),
        dict(node_id="NEW2", source_available_upper=inputs.clock("2024-09-24 11:00"), actual_information_role="RECORDED_SUPPORT_INFORMATION", common_source_id="ONE"),
        dict(node_id="CONFIRM", source_available_upper=inputs.clock("2024-09-24 12:00"), actual_information_role="ANNOUNCED_TARGET_OPERATION_CONFIRMATION", common_source_id="TWO")]
    result = inputs.policy_view(nodes, entered, inputs.clock("2024-09-24 16:00"))
    assert result["policy_fresh_common_sources_today"] == 1
    assert result["policy_older_support_first_observed_after_entry_ids_today"] == "OLD"
    assert result["policy_target_confirmation_ids_today"] == "CONFIRM"
    with pytest.raises(ValueError, match="过去公布钟"):
        inputs.policy_view(nodes, entered, inputs.clock("2024-09-24 10:00"))


def test_macro_old_first_observation_is_not_fresh_publication():
    observation = row("2021-12-13", orders_known=True, orders_available_at=inputs.clock("2021-11-30 09:30"),
                      orders_first_observed_here=True, orders_publication_since_previous_decision=False)
    result = inputs.macro_view(observation, inputs.clock("2021-12-10 09:30"))
    assert result["orders_first_observed_today_without_new_publication"]
    assert not result["orders_source_published_after_actual_entry_open"]
    assert not result["orders_fresh_publication_event_today_after_entry"]
    observation.orders_available_at = inputs.clock("2021-12-13 09:31")
    observation.orders_publication_since_previous_decision = True
    assert inputs.macro_view(observation, inputs.clock("2021-12-10 09:30"))["orders_fresh_publication_event_today_after_entry"]


def test_future_next_entry_and_period_endpoint_cannot_rewrite_history():
    dates = ["2021-12-10", "2021-12-13", "2021-12-14"]
    data = pd.DataFrame([row(date) for date in dates])
    account = sample_account(dates)
    before = inputs.sequences(data, account, receipts(), "PERIOD", "STRESS", "A", dates[-1])
    account["trades"] = pd.concat([account["trades"], pd.DataFrame({"cycle_id": [2], "entry_date": [pd.Timestamp("2021-12-20")], "net_pnl": [-999.]})], ignore_index=True)
    after = inputs.sequences(data, account, receipts(), "PERIOD", "STRESS", "A", dates[-1])
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    endpoint = inputs.sequences(data, account, receipts(), "PERIOD", "STRESS", "A", "2021-12-13")
    assert endpoint.date.max() == pd.Timestamp("2021-12-13")
    assert after.actual_account_phase.iloc[-1] == "AFTER_ACTUAL_EXIT_BEFORE_NEXT_ENTRY"


def test_net_outcome_never_changes_information_states():
    dates = ["2021-12-10", "2021-12-13", "2021-12-14"]
    data = pd.DataFrame([row(date) for date in dates])
    account = sample_account(dates)
    positive = inputs.sequences(data, account, receipts(), "PERIOD", "BASE", "A", dates[-1])
    account["trades"]["net_pnl"] = -1e9
    negative = inputs.sequences(data, account, receipts(), "PERIOD", "BASE", "A", dates[-1])
    pd.testing.assert_frame_equal(positive, negative, check_exact=True)


def test_duplicate_original_calendar_rejected():
    frame = pd.DataFrame([row("2021-12-10"), row("2021-12-10")])
    with pytest.raises(ValueError, match="重复"):
        inputs.normalize_calendar(frame)
