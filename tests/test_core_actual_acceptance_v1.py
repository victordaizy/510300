"""必要的先知信息、真实进入、保护顺序及共同账户测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import core_actual_acceptance_inputs_v1 as rules
from research.core_actual_acceptance_account_v1 import account
from research.point_account_nr7_complement_v1 import verify_account


def active():
    return {"entry_date": pd.Timestamp("2020-01-06"), "entry_day_high_anchor": np.nan,
        "entry_day_low_anchor": np.nan, "holding_stage": "CORE", "protection_armed": False,
        "structural_stop": np.nan, "first_price_acceptance": pd.NaT, "promotion_date": pd.NaT,
        "extension_revoked": False, "first_extension_revocation": pd.NaT}


def row(date, **changes):
    date = pd.Timestamp(date)
    r = {"date": date, "decision_time": date.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=16),
        "cash_high": 11., "cash_low": 9.9, "ac": 10.5, "daily_hist": 1.,
        "complete_week_first": pd.NaT, "complete_week_last": pd.NaT,
        "complete_week_low": np.nan, "complete_week_close": np.nan,
        "higher_complete_week_low_and_close": False, "up_volume_balance5": .4,
        "industry_positive5_fraction": .7, "leader_mean_return5": .02, "industry_view_allowed": True,
        "industry_source_date": date-pd.Timedelta(days=1), "orders_known": False,
        "orders_available_at": pd.NaT, "orders_publication_since_previous_decision": False,
        "pmi_orders_change": np.nan, "reverse_source_available_upper": pd.NaT,
        "reverse_operation_arrived": False}
    r.update(changes)
    return pd.Series(r)


def initialized():
    a = active()
    rules.advance(a, row("2020-01-06"), rules.POLICIES[0])
    return a


def new_week(date="2020-01-13", **changes):
    return row(date, ac=11.5, complete_week_first=pd.Timestamp("2020-01-06"),
        complete_week_last=pd.Timestamp("2020-01-10"), complete_week_low=10.1,
        complete_week_close=11.4, **changes)


def test_entry_day_range_only_known_at_own_close_and_not_accepted():
    a = active()
    result = rules.advance(a, row("2020-01-06", ac=11.5), rules.POLICIES[0])
    assert a["entry_day_high_anchor"] == 11 and not result["protection_armed"]
    result = rules.advance(a, row("2020-01-07", ac=11.5), rules.POLICIES[0])
    assert result["protection_armed"] and a["first_price_acceptance"] == pd.Timestamp("2020-01-07")


def test_low_failure_before_acceptance_does_not_predict_later_fate():
    a = initialized()
    result = rules.advance(a, row("2020-01-07", ac=9.5), rules.POLICIES[0])
    assert result["entry_day_low_failed_now"] and result["extra_exit_reason"] == "NONE"
    result = rules.advance(a, row("2020-01-08", ac=11.5), rules.POLICIES[0])
    assert result["protection_armed"] and result["extra_exit_reason"] == "NONE"


def test_low_failure_after_acceptance_is_an_actual_current_exit_request():
    a = initialized()
    rules.advance(a, row("2020-01-07", ac=11.5), rules.POLICIES[0])
    result = rules.advance(a, row("2020-01-08", ac=9.9), rules.POLICIES[0])
    assert result["extra_exit_reason"] == "ACTUAL_ACCEPTANCE_STRUCTURE_FAILED"


def test_week_with_pre_entry_day_cannot_promote_and_missing_participation_keeps_core_action():
    a = initialized()
    result = rules.advance(a, row("2020-01-13", ac=11.5, complete_week_first=pd.Timestamp("2020-01-03"),
        complete_week_last=pd.Timestamp("2020-01-10"), complete_week_low=10.1, complete_week_close=11.4), rules.POLICIES[0])
    assert not result["complete_week_all_observed_days_after_entry"] and a["holding_stage"] == "ACCEPTED"
    result = rules.advance(a, new_week(industry_view_allowed=False), rules.POLICIES[0])
    assert not result["participation_known"] and a["holding_stage"] == "ACCEPTED"


def test_price_control_and_protection_control_separate_information_and_extension():
    for policy, expected in ((rules.POLICIES[0], "ACCEPTED"), (rules.POLICIES[1], "CARRY"), (rules.POLICIES[2], "ACCEPTED")):
        a = initialized()
        rules.advance(a, new_week(industry_view_allowed=False), policy)
        assert a["holding_stage"] == expected


def test_new_order_publication_revokes_but_old_negative_background_does_not():
    a = initialized()
    rules.advance(a, new_week(), rules.POLICIES[0])
    old = row("2020-01-14", ac=11.5, orders_known=True, pmi_orders_change=-1,
              orders_available_at=pd.Timestamp("2020-01-02 09:00", tz="Asia/Shanghai"))
    rules.advance(a, old, rules.POLICIES[0])
    assert not a["extension_revoked"] and a["holding_stage"] == "CARRY"
    fresh = old.copy()
    fresh["orders_available_at"] = pd.Timestamp("2020-01-14 09:00", tz="Asia/Shanghai")
    fresh["orders_publication_since_previous_decision"] = True
    rules.advance(a, fresh, rules.POLICIES[0])
    assert a["extension_revoked"] and a["holding_stage"] == "ACCEPTED"
    rules.advance(a, new_week("2020-01-15"), rules.POLICIES[0])
    assert a["holding_stage"] == "ACCEPTED"


def test_future_macro_clock_rejected_and_price_control_has_no_macro_veto():
    a = initialized()
    future = row("2020-01-07", orders_known=True, orders_available_at=pd.Timestamp("2020-01-08 09:00", tz="Asia/Shanghai"))
    with pytest.raises(ValueError, match="晚于"):
        rules.advance(a, future, rules.POLICIES[0])
    a = initialized()
    fresh = new_week(orders_known=True, pmi_orders_change=-1, orders_publication_since_previous_decision=True,
                     orders_available_at=pd.Timestamp("2020-01-13 09:00", tz="Asia/Shanghai"))
    rules.advance(a, fresh, rules.POLICIES[1])
    assert a["holding_stage"] == "CARRY" and not a["extension_revoked"]


def test_carry_stop_never_falls_with_later_week_and_revocation():
    a = initialized()
    rules.advance(a, new_week(), rules.POLICIES[0])
    assert a["structural_stop"] == 10.1
    r = new_week("2020-01-20")
    r["complete_week_first"], r["complete_week_last"] = pd.Timestamp("2020-01-13"), pd.Timestamp("2020-01-17")
    r["complete_week_low"] = 10.6
    rules.advance(a, r, rules.POLICIES[0])
    r["date"], r["decision_time"] = pd.Timestamp("2020-01-21"), pd.Timestamp("2020-01-21 16:00", tz="Asia/Shanghai")
    r["complete_week_low"] = 10.2
    rules.advance(a, r, rules.POLICIES[0])
    assert a["structural_stop"] == 10.6


def test_unknown_parent_does_not_rearm_consumed_episode():
    p = pd.DataFrame({"origin": pd.date_range("2020-01-01", periods=7), rules.PARENT_A: [.3, np.nan, .4, 0, np.nan, .2, .3]})
    assert rules.parent_episodes(p).parent_episode_id.tolist() == [1, 1, 1, 1, 1, 2, 2]


def fixture():
    dates = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13", "2020-01-14", "2020-01-15", "2020-01-16"])
    closes = [10., 10., 11.2, 9.8, 9.8, 9.8, 10., 10., 10., 10.]
    rows = []
    for i, date in enumerate(dates):
        r = row(date, ac=closes[i], industry_view_allowed=False).to_dict()
        r.update(open=closes[i-1] if i else 10., close=closes[i], high=max(11., closes[i]), low=min(9.9, closes[i]),
                 volume=1000., dividend=.1 if i == 3 else 0., cash_shift=.1 if i >= 3 else 0., symbol="510300.SH")
        r["cash_high"], r["cash_low"] = r["high"]+r["cash_shift"], r["low"]+r["cash_shift"]
        r["ac"] = r["close"]+r["cash_shift"]
        rows.append(r)
    d = pd.DataFrame(rows)
    parents = pd.DataFrame({"origin": dates, rules.PARENT_A: [.5, .5, .5, .5, .5, 0, .5, .5, .5, .5]})
    risks = pd.DataFrame({"idx": range(len(d)), "es95": .005})
    dividends = pd.DataFrame({"record_date": [dates[2]], "ex_date": [dates[3]], "payment_date": [dates[5]], "cash_dividend_per_share": [.1]})
    return d, parents, risks, dividends


def test_account_cash_dividend_next_open_consumption_and_future_invariance():
    d, p, risk, dividends = fixture()
    result = account(d, dividends, p, risk, rules.POLICIES[0], "STRESS", "2020-01-06")
    verify_account(result)
    first_buys = result["trades"].entry_date.tolist()
    assert first_buys == [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-14")]
    assert result["trades"].exit_date.iloc[0] == pd.Timestamp("2020-01-09")
    assert result["trades"].dividend_cny.iloc[0] > 0
    assert result["daily"].receivable.iloc[-1] == 0
    changed = d.copy()
    changed.loc[changed.date.eq(pd.Timestamp("2020-01-16")), ["close", "ac", "cash_high", "cash_low", "high", "low"]] = [10.2, 10.3, 10.4, 10.1, 10.3, 10.]
    other = account(changed, dividends, p, risk, rules.POLICIES[0], "STRESS", "2020-01-06")
    for name, clock in (("daily", "date"), ("orders", "date"), ("decisions", "origin"), ("holding_evidence", "origin")):
        left, right = result[name], other[name]
        pd.testing.assert_frame_equal(left.loc[left[clock].lt(pd.Timestamp("2020-01-16"))].reset_index(drop=True),
                                      right.loc[right[clock].lt(pd.Timestamp("2020-01-16"))].reset_index(drop=True), check_exact=True)


def test_all_nat_reverse_clock_prefix_keeps_declared_timezone_and_resolution():
    d, _, _, _ = fixture()
    receipts = pd.DataFrame({"first_original_decision_date": [pd.Timestamp("2020-01-13")],
        "source_available_upper": [pd.Timestamp("2020-01-13 09:00", tz="Asia/Shanghai")],
        "actual_information_role": ["OBSERVED_RATE_HIGHER"], "node_id": ["SYNTHETIC_REVERSE"]})
    full = rules.observations(d, receipts)
    cut = rules.observations(d.loc[d.date.lt(pd.Timestamp("2020-01-13"))], receipts)
    assert str(full.reverse_source_available_upper.dtype) == "datetime64[ns, Asia/Shanghai]"
    assert str(cut.reverse_source_available_upper.dtype) == "datetime64[ns, Asia/Shanghai]"
    pd.testing.assert_frame_equal(cut, full.loc[full.date.lt(pd.Timestamp("2020-01-13"))].reset_index(drop=True), check_exact=True)
