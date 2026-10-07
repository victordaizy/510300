"""来源身份、真实持仓、未知与事后开盘容量的必要检查。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import core_support_complement_inputs_v1 as inputs


def test_held_position_cannot_turn_pending_exit_into_same_open_new_entry():
    assert inputs.decision_role(100,0,False,.03,True) == "OWN_POSITION_HELD_EVENT_CONSUMED_NO_SAME_OPEN_REBUY"
    assert inputs.decision_role(0,0,False,.03,True) == "POTENTIAL_COMPLEMENT_ATTEMPT_ONLY_NOT_NEW_STRATEGY_ENTRY"


def test_original_core_positive_or_unknown_is_not_zero_even_after_forced_cash_exit():
    assert inputs.decision_role(0,.3,False,.03,True) == "ORIGINAL_CORE_POSITIVE_PRIORITY"
    assert inputs.decision_role(0,np.nan,False,.03,True) == "NO_VIEW_ORIGINAL_CORE_PARENT"
    assert inputs.decision_role(0,0,False,np.nan,True) == "NO_VIEW_MATURE_RISK"


def test_support_identity_requires_prior_public_and_prior_observation_not_confirmation():
    row = pd.Series({"date":pd.Timestamp("2020-01-09"),"support_setup_date":pd.Timestamp("2020-01-08"),"support_setup_source_ids":"S1|S2"})
    receipts = pd.DataFrame({"node_id":["S1","S2"],"common_source_id":["ONE_FILE","ONE_FILE"],
        "source_available_upper":[pd.Timestamp("2020-01-08 09:00",tz="Asia/Shanghai")]*2,
        "first_original_decision_date":[pd.Timestamp("2020-01-08")]*2,
        "actual_information_role":["RECORDED_SUPPORT_INFORMATION"]*2})
    info = inputs.source_identity(row,receipts)
    assert info["source_common_count_not_independent_votes"] == 1
    future = receipts.copy()
    future.loc[0,"source_available_upper"] = pd.Timestamp("2020-01-08 17:00",tz="Asia/Shanghai")
    with pytest.raises(ValueError,match="以后"):
        inputs.source_identity(row,future)
    bad = receipts.copy()
    bad.loc[0,"actual_information_role"] = "ANNOUNCED_TARGET_OPERATION_CONFIRMATION"
    with pytest.raises(ValueError,match="非支持"):
        inputs.source_identity(row,bad)


def fixture():
    d = pd.DataFrame({"date":pd.to_datetime(["2020-01-08","2020-01-09"]),
        "open":[10.,10.],"close":[10.,10.],"cash_shift":[0.,0.],"dividend":[0.,0.]})
    snapshot = {"cash":200000.,"receivable":0.,"shares":0,"peak":200000.,"next_dividend_accrual":0.}
    return d,snapshot


def test_future_open_is_unknown_and_next_seen_capacity_does_not_update_snapshot():
    d,snapshot = fixture()
    original = snapshot.copy()
    hidden = inputs.open_capacity(d.iloc[:1],0,snapshot,.03,9.5,"STRESS")
    assert hidden["next_open_role"] == "NOT_OBSERVED_YET" and pd.isna(hidden["next_open_observed_at"])
    seen = inputs.open_capacity(d,0,snapshot,.03,9.5,"STRESS")
    assert seen["next_open_one_shot_quantity_not_combined_position"] > 0 and snapshot == original
    assert seen["next_open_observed_at"] == pd.Timestamp("2020-01-09 09:30",tz="Asia/Shanghai")


def test_next_open_anchor_or_limit_rejection_and_timezone_not_silently_repaired():
    d,snapshot = fixture()
    d.loc[1,"open"] = 9.4
    assert inputs.open_capacity(d,0,snapshot,.03,9.5,"STRESS")["next_open_role"] == "OPEN_ALREADY_BELOW_KNOWN_ANCHOR"
    d.loc[1,"open"] = 11.
    assert inputs.open_capacity(d,0,snapshot,.03,9.5,"STRESS")["next_open_role"] == "OPEN_LIMIT_OR_MISSING_PRICE"
    with pytest.raises(ValueError,match="时区"):
        inputs.clock(pd.Timestamp("2020-01-08 09:00"))
