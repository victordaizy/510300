"""支持与价格接受：必要公开钟、锚点、持有与现金执行测试。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research import support_price_acceptance_inputs_v1 as candidate
from research.support_price_acceptance_account_v1 import account
from research.point_account_nr7_complement_v1 import verify_account


def node(identity, at, shared="test", action="RECORDED_SUPPORT_INFORMATION", target=np.nan, value=np.nan):
    return {"node_id": identity, "source_available_upper": pd.Timestamp(at, tz="Asia/Shanghai"),
        "common_source_id": shared, "registered_action": action, "announced_seven_day_target": target,
        "observed_seven_day_value": value, "source_url": "test", "source_path": "test"}


def watch_frame(close, supports):
    dates = pd.bdate_range("2020-01-06", periods=len(close))
    return pd.DataFrame({"date": dates, "ac": close, "cash_high": np.asarray(close)+.1, "cash_low": np.asarray(close)-.1,
        "daily_hist": .01, "atr14": .1, "support_activation_allowed": supports,
        "support_new_node_ids": ["TEST" if value else "" for value in supports], "reverse_operation_arrived": False})


def test_after_sixteen_announcement_only_enters_next_actual_decision():
    dates = pd.to_datetime(["2020-04-03", "2020-04-07"])
    data = pd.DataFrame({"date": dates, "decision_time": dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=16)})
    rows, receipts = candidate.source_states(data, pd.DataFrame([node("SUPPORT", "2020-04-03 16:57:32")]))
    assert not rows.support_activation_allowed.iloc[0]
    assert rows.support_activation_allowed.iloc[1]
    assert receipts.first_original_decision_date.iloc[0] == dates[1]


def test_same_source_and_later_target_operation_are_not_new_independent_activation():
    dates = pd.bdate_range("2024-09-23", periods=3)
    data = pd.DataFrame({"date": dates, "decision_time": dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=16)})
    records = pd.DataFrame([node("CHAIN_R01", "2024-09-24 09:19:36", "shared", target=1.5),
        node("CHAIN_C01", "2024-09-24 11:42:50", "shared"),
        node("RATE_24", "2024-09-25 09:20:30", "operation", "OBSERVED_RATE_LOWER", value=1.5)])
    states, receipts = candidate.source_states(data, records)
    assert states.support_unique_common_sources.iloc[1] == 1
    assert not states.support_activation_allowed.iloc[2]
    assert receipts.actual_information_role.iloc[-1] == "ANNOUNCED_TARGET_OPERATION_CONFIRMATION"


def test_additional_support_does_not_replace_original_price_anchor():
    data = watch_frame([1.,1.02,1.03,1.12,1.13], [True,False,True,False,False])
    points = candidate.price_states(data, True)
    assert points.support_entry_event.iloc[3]
    assert points.support_setup_idx.iloc[3] == 0
    assert points.support_setup_high.iloc[3] == 1.1
    assert points.support_entry_event.sum() == 1


def test_invalidated_support_does_not_rearm_on_old_information():
    data = watch_frame([1.,.8,1.2,1.2,1.4], [True,False,False,True,False])
    points = candidate.price_states(data, True)
    assert points.support_observation_reason.iloc[1] == "ANCHOR_LOW_FAILED"
    assert not points.support_entry_event.iloc[2]
    assert points.support_entry_event.iloc[4]


def test_future_support_and_unfinished_week_do_not_change_past_states():
    dates = pd.bdate_range("2020-01-06", periods=17)
    close = np.linspace(3.,3.2,len(dates))
    data = pd.DataFrame({"date": dates, "decision_time": dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=16),
        "ac": close, "close": close, "high": close+.015, "low": close-.015, "cash_shift": 0., "daily_hist": .01, "atr14": .05})
    records = pd.DataFrame([node("EARLY", "2020-01-07 10:00:00"), node("FUTURE", "2020-01-27 10:00:00")])
    complete, _ = candidate.observations(data, records)
    for count in [4,8,11,14]:
        prefix, _ = candidate.observations(data.iloc[:count], records)
        pd.testing.assert_frame_equal(prefix, complete.iloc[:count].reset_index(drop=True), check_exact=True)


def holding_row(**changes):
    values = {"date": pd.Timestamp("2020-04-01"), "ac": 3.2, "reverse_operation_arrived": False,
        "higher_complete_week_low_and_close": True, "complete_week_last": pd.Timestamp("2020-03-27"),
        "complete_week_low": 3., "daily_hist": .01, "ema20": 3.}
    values.update(changes)
    return SimpleNamespace(**values)


def test_promotion_requires_week_completed_after_actual_entry_and_never_lowers_stop():
    active = {"entry_idx": 1, "entry_date": pd.Timestamp("2020-03-30"), "structural_stop": 2.8,
        "holding_stage": "EARLY", "promotion_date": pd.NaT, "fixed_stop": 2.9, "fixed_target": 3.4}
    assert candidate.exit_decision(active, holding_row(), 2, candidate.POLICIES[0]) is None
    assert active["holding_stage"] == "EARLY"
    assert candidate.exit_decision(active, holding_row(complete_week_last=pd.Timestamp("2020-04-03"),date=pd.Timestamp("2020-04-06")), 3, candidate.POLICIES[0]) is None
    assert active["holding_stage"] == "CARRY" and active["structural_stop"] == 3.
    candidate.exit_decision(active, holding_row(complete_week_low=2.9), 4, candidate.POLICIES[0])
    assert active["structural_stop"] == 3.
    assert candidate.exit_decision(active, holding_row(ac=2.99), 5, candidate.POLICIES[0]) == "KNOWN_ANCHOR_OR_TRAILING_LOW_FAILED"


def execution_frame():
    dates = pd.bdate_range("2020-01-06", periods=7)
    d = pd.DataFrame({"date": dates, "open": [3.,3.,2.4,3.,3.,3.,3.], "close": [3.,2.8,3.1,3.,3.,3.,3.],
        "dividend": [0.,0.,.05,0.,0.,0.,0.], "cash_shift": [0.,0.,.05,.05,.05,.05,.05], "atr14": .1,
        "daily_hist": .01, "ema20": 3., "higher_complete_week_low_and_close": False,
        "complete_week_last": pd.NaT, "complete_week_low": 2., "reverse_operation_arrived": False})
    d["ac"] = d.close+d.cash_shift
    for prefix in ["support", "price"]:
        d[prefix+"_entry_event"] = [True,False,False,False,False,False,False]
        d[prefix+"_setup_low"] = 2.
        d[prefix+"_setup_date"] = dates[0]-pd.Timedelta(days=1)
        d[prefix+"_setup_source_ids"] = "TEST"
    div = pd.DataFrame({"record_date": [dates[1]], "ex_date": [dates[2]], "payment_date": [dates[4]], "cash_dividend_per_share": [.05]})
    risks = pd.DataFrame({"idx": np.arange(len(d)), "es95": .02})
    return d, div, risks


def test_locked_gap_exit_and_dividend_book_survive_position_close():
    data, div, risks = execution_frame()
    result = account(data, div, risks, candidate.POLICIES[2], "STRESS", data.date.iloc[1])
    verify_account(result)
    trade = result["trades"].iloc[0]
    assert trade.entry_date == data.date.iloc[1]
    assert trade.exit_date == data.date.iloc[3]
    assert result["rejections"].date.eq(data.date.iloc[2]).any()
    assert result["daily"].dividend_accrual.sum() == trade.entry_quantity*.05
    assert result["daily"].dividend_paid.sum() == trade.entry_quantity*.05
    assert result["terminal"]["unpaid_dividend_cny"] == 0
    assert result["orders"].iloc[-1].reason == "LOCKED_EXIT"


def test_open_below_known_anchor_cancels_entry_for_all_three_policies():
    data, div, risks = execution_frame()
    data.loc[1,"open"] = 1.9
    for policy in candidate.POLICIES:
        result = account(data, div, risks, policy, "STRESS", data.date.iloc[1])
        verify_account(result)
        assert result["orders"].empty
        assert result["rejections"].reason.eq("OPEN_ALREADY_BELOW_KNOWN_ANCHOR").any()
