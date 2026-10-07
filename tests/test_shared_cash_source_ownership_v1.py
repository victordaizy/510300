"""来源优先、共同库存、一次消费、锁定及现金时序的必要反例。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import shared_cash_source_ownership_rules_v1 as rules
from research.shared_cash_source_ownership_account_v1 import account
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.point_account_nr7_complement_v1 import verify_account


def fixture():
    dates = pd.bdate_range("2020-01-06",periods=9)
    d = pd.DataFrame({"date":dates,"open":10.,"close":10.,"high":10.2,"low":9.8,"volume":100000.,
        "cash_shift":0.,"dividend":0.,"ac":10.,"atr14":.3,"daily_hist":.1,"ema20":9.9,
        "reverse_operation_arrived":False,"higher_complete_week_low_and_close":False,
        "complete_week_last":pd.NaT,"complete_week_low":np.nan})
    for prefix in ("support","price"):
        d[prefix+"_entry_event"] = False
        d[prefix+"_setup_low"] = 9.8
        d[prefix+"_setup_high"] = 9.9
        d[prefix+"_setup_date"] = dates[0]-pd.Timedelta(days=3)
        d[prefix+"_setup_source_ids"] = "KNOWN_SUPPORT" if prefix == "support" else "PRICE_ONLY_NO_POLICY"
    parents = pd.DataFrame({"origin":dates,PARENT_A:0.})
    risks = pd.DataFrame({"idx":np.arange(len(d)),"es95":.03})
    dividends = pd.DataFrame(columns=["record_date","ex_date","payment_date","cash_dividend_per_share"])
    return d,dividends,parents,risks


def run(d,v,p,k,policy=rules.POLICIES[0]):
    return account(d,v,p,k,policy,"STRESS",d.date.iloc[1])


def test_event_priority_held_unknown_and_stop_are_not_zero():
    assert rules.event_role(100,0,False,True,True,rules.POLICIES[0]).startswith("CONSUMED_BY_EXISTING")
    assert rules.event_role(0,np.nan,False,False,True,rules.POLICIES[0]) == "CONSUMED_UNKNOWN_CORE_NOT_ZERO"
    assert rules.event_role(0,.2,False,False,True,rules.POLICIES[0]) == "CONSUMED_CORE_POSITIVE_PRIORITY"
    assert rules.event_role(0,0,True,False,True,rules.POLICIES[0]) == "CONSUMED_ACCOUNT_STOP_OR_LOCKED_EXIT"


def test_core_positive_wins_same_signal_and_real_owner_is_core():
    d,v,p,k = fixture()
    d.loc[0,"support_entry_event"] = True
    p[PARENT_A] = .5
    a = run(d,v,p,k)
    assert a["trades"].owner.tolist() == ["CORE"]
    assert a["orders"].source.eq("CORE_WEIGHT").all()
    assert a["decisions"].current_complement_event_role.iloc[0] == "CONSUMED_CORE_POSITIVE_PRIORITY"
    verify_account(a)


def test_support_owner_survives_later_core_positive_without_add_or_transfer():
    d,v,p,k = fixture()
    d.loc[0,"support_entry_event"] = True
    p.loc[2:,PARENT_A] = .1
    a = run(d,v,p,k)
    assert a["trades"].owner.tolist() == ["COMPLEMENT"]
    assert len(a["orders"].loc[a["orders"].side.eq("BUY")]) == 1
    assert a["daily"].owner.eq("COMPLEMENT").all()
    assert a["terminal"]["open_owner"] == "COMPLEMENT"
    verify_account(a)


def test_unknown_core_cannot_make_support_entry_from_missing():
    d,v,p,k = fixture()
    d.loc[0,"support_entry_event"] = True
    p[PARENT_A] = np.nan
    a = run(d,v,p,k)
    assert a["orders"].empty and a["trades"].empty
    assert a["decisions"].complement_event_consumed_today.iloc[0]


def test_held_event_and_pending_full_exit_are_consumed_without_same_open_rebuy():
    d,v,p,k = fixture()
    p.loc[:1,PARENT_A] = .5
    d.loc[2,"support_entry_event"] = True
    a = run(d,v,p,k)
    assert a["trades"].owner.tolist() == ["CORE"]
    assert a["daily"].shares.iloc[1] > 0 and a["daily"].shares.iloc[2] == 0
    assert a["decisions"].current_complement_event_role.iloc[2] == "CONSUMED_BY_EXISTING_OWNER_NO_SAME_OPEN_REBUY"
    assert len(a["orders"].loc[a["orders"].side.eq("BUY")]) == 1


def test_failed_open_anchor_consumes_signal_and_never_retries_old_support():
    d,v,p,k = fixture()
    d.loc[0,"support_entry_event"] = True
    d.loc[1,"open"] = 9.7
    a = run(d,v,p,k)
    assert a["orders"].empty and a["trades"].empty
    assert a["rejections"].reason.tolist() == ["OPEN_ALREADY_BELOW_KNOWN_ANCHOR"]
    assert a["decisions"].complement_event_consumed_today.sum() == 1


def test_blocked_exit_keeps_owner_and_allows_core_only_on_later_close():
    d,v,p,k = fixture()
    d.loc[0,"support_entry_event"] = True
    d.loc[2,"reverse_operation_arrived"] = True
    d.loc[3,"open"] = 8.9
    p.loc[2:,PARENT_A] = .5
    a = run(d,v,p,k)
    assert a["daily"].owner.iloc[2] == "COMPLEMENT" and a["daily"].owner.iloc[3] == "FLAT"
    first = a["trades"].iloc[0]
    assert first.exit_date == d.date.iloc[4]
    second = a["trades"].iloc[1]
    assert second.owner == "CORE" and second.entry_date == d.date.iloc[5]
    assert a["rejections"].reason.iloc[0] == "OPEN_LIMIT_OR_MISSING_PRICE"
    verify_account(a)


def test_price_control_can_enter_without_support_and_information_main_cannot():
    d,v,p,k = fixture()
    d.loc[0,"price_entry_event"] = True
    main,price = run(d,v,p,k),run(d,v,p,k,rules.POLICIES[1])
    assert main["trades"].empty
    assert price["trades"].source.tolist() == ["PRICE_ACCEPTED"]


def test_original_core_unknown_after_buy_keeps_original_position():
    d,v,p,k = fixture()
    p.loc[0,PARENT_A] = .5
    p.loc[1:,PARENT_A] = np.nan
    a = run(d,v,p,k,rules.BASELINE_CORE)
    assert a["trades"].owner.tolist() == ["CORE"] and a["orders"].side.tolist() == ["BUY"]
    assert a["decisions"].reason.iloc[1] == "UNKNOWN_KEEP_EXISTING"


def test_dividend_accrual_and_payment_are_one_shared_cash_ledger():
    d,v,p,k = fixture()
    d.loc[0,"support_entry_event"] = True
    v = pd.DataFrame({"record_date":[d.date.iloc[2]],"ex_date":[d.date.iloc[3]],
        "payment_date":[d.date.iloc[6]],"cash_dividend_per_share":[.1]})
    d.loc[3:,["open","close"]] = 9.9
    d.loc[3:,"cash_shift"] = .1
    d.loc[3,"dividend"] = .1
    a = run(d,v,p,k)
    accrued = a["daily"].dividend_accrual.sum()
    assert accrued > 0 and a["daily"].dividend_paid.sum() == accrued
    assert a["daily"].receivable.iloc[2] == accrued and a["terminal"]["unpaid_dividend_cny"] == 0
    assert a["trades"].dividend_cny.sum() == accrued
    verify_account(a)


def test_future_core_and_support_changes_do_not_change_past_daily_or_orders():
    d,v,p,k = fixture()
    d.loc[0,"support_entry_event"] = True
    full = run(d,v,p,k)
    changed = d.copy()
    changed.loc[6,"reverse_operation_arrived"] = True
    changed.loc[7,"support_entry_event"] = True
    different = p.copy()
    different.loc[6:,PARENT_A] = .5
    later = run(changed,v,different,k)
    cut = d.date.iloc[5]
    pd.testing.assert_frame_equal(full["daily"].loc[full["daily"].date.le(cut)],later["daily"].loc[later["daily"].date.le(cut)],check_exact=True)
    pd.testing.assert_frame_equal(full["orders"].loc[full["orders"].date.le(cut)],later["orders"].loc[later["orders"].date.le(cut)],check_exact=True)
