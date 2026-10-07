"""验证阶段规则的时序、未知分支、实际退出和股息账本。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research import broker_stage_policy_inputs_v1 as policy
from research.point_account_nr7_complement_v1 import verify_account


def observation_frame(n=125):
    dates = pd.bdate_range("2020-01-06", periods=n)
    changes = np.r_[np.resize([.01, -.008], n-25), np.full(25, .001)]
    close = 3*np.cumprod(1+changes)
    frame = pd.DataFrame({"date": dates, "open": close, "high": close+.02, "low": close-.02, "close": close,
                          "volume": 10000, "dividend": 0., "cash_shift": 0., "ac": close, "atr14": .05,
                          "daily_hist_atr": .2, "weekly_hist_atr": -.1, "weekly_last_date": dates-pd.Timedelta(days=7),
                          "log_relative_volume": 0., "orders_known": True, "funding_known": True, "margin_known": True,
                          "pmi_orders_level": -5., "pmi_orders_change": 1., "funding_gap_pp": -.1,
                          "funding_gap_change5": -.01, "financing_net_change5": -.01})
    return frame


def holding_row(**changes):
    values = {"date": pd.Timestamp("2020-04-01"), "ac": 3.2, "previous_week_low": 2.9, "trend_confirmation": False,
              "weekly_hist": -.01, "daily_hist": .01, "ema20": 3., "funding_distribution": False}
    values.update(changes)
    return SimpleNamespace(**values)


def active():
    return {"entry_idx": 1, "fixed_stop": 2.9, "fixed_target": 3.2, "structural_stop": 2.8,
            "holding_stage": "EARLY", "promotion_date": pd.NaT}


def execution_frame():
    dates = pd.bdate_range("2020-01-06", periods=7)
    frame = pd.DataFrame({"date": dates, "open": [3.,3.,2.4,3.,3.,3.,3.], "close": [3.,2.8,3.1,3.,3.,3.,3.],
                          "dividend": [0.,0.,.05,0.,0.,0.,0.], "cash_shift": [0.,0.,.05,.05,.05,.05,.05],
                          "atr14": .1, "previous_week_low": 2., "daily_hist": .01, "weekly_hist": -.01, "ema20": 3.,
                          "trend_confirmation": False, "funding_distribution": False,
                          "stage_entry_type": ["REPRICING", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
                          "raw_entry_type": ["REPRICING", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
                          "orders_known": False, "funding_known": False, "margin_known": False})
    frame["ac"] = frame.close+frame.cash_shift
    div = pd.DataFrame({"record_date": [dates[1]], "ex_date": [dates[2]], "payment_date": [dates[4]], "cash_dividend_per_share": [.05]})
    risks = pd.DataFrame({"idx": np.arange(len(frame)), "es95": .02})
    return frame, div, risks


def test_prefix_does_not_use_future_week_low():
    frame = observation_frame()
    complete = policy.observations(frame)
    selected = ["date", "previous_week_low", "support_week_last_date", "support_available_at", "stage_entry_type", "raw_entry_type"]
    for index in [80,81,82,83,84,85,104]:
        prefix = policy.observations(frame.iloc[:index+1])
        pd.testing.assert_frame_equal(prefix[selected], complete.iloc[:index+1][selected], check_exact=True)


def test_unknown_macro_can_still_observe_independent_repricing():
    frame = observation_frame()
    frame[["orders_known", "funding_known", "margin_known"]] = False
    frame.loc[frame.index[-1], "ac"] = frame.ac.max()+.3
    frame.loc[frame.index[-1], "log_relative_volume"] = np.log(2.)
    result = policy.observations(frame)
    assert result.stage_entry_type.iloc[-1] == "REPRICING"
    assert not result.stage_repair_condition.iloc[-1]
    assert not result.stage_pullback_condition.iloc[-1]


def test_repair_uses_improvement_not_positive_pmi_level():
    result = policy.observations(observation_frame())
    assert result.pmi_orders_level.iloc[-1] < 0
    assert result.volatility20_60.iloc[-1] <= 1
    assert result.stage_repair_condition.iloc[-1]


def test_missing_macro_flags_are_not_zero_or_positive_votes():
    frame = observation_frame()
    frame[["orders_known", "funding_known", "margin_known"]] = False
    result = policy.observations(frame)
    assert not result.repair_macro_support.any()
    assert not result.trend_confirmation.any()
    assert not result.funding_distribution.any()


def test_continuous_condition_has_only_one_rising_edge():
    result = policy.observations(observation_frame())
    condition = result.stage_repair_condition
    expected = condition & ~condition.shift(1, fill_value=False)
    pd.testing.assert_series_equal(result.stage_repair_event, expected, check_names=False)
    assert not (result.stage_repair_event & condition.shift(1, fill_value=False)).any()


def test_fixed_boundary_and_twenty_close_decisions_are_inclusive():
    assert policy.exit_decision(active(), holding_row(ac=2.9), 2, "SAME_ENTRY_FIXED_EXIT") == "FIXED_LOSS_CLOSE"
    assert policy.exit_decision(active(), holding_row(ac=3.2), 2, "SAME_ENTRY_FIXED_EXIT") == "FIXED_PROFIT_CLOSE"
    assert policy.exit_decision(active(), holding_row(ac=3.1), 20, "SAME_ENTRY_FIXED_EXIT") == "FIXED_TWENTY_CLOSES"
    assert policy.exit_decision(active(), holding_row(ac=3.1), 19, "SAME_ENTRY_FIXED_EXIT") is None


def test_promotion_and_trailing_support_never_loosen():
    trade = active()
    assert policy.exit_decision(trade, holding_row(), 2, "STAGE_ENTRY_AND_EXIT") is None
    assert trade["holding_stage"] == "EARLY"
    assert policy.exit_decision(trade, holding_row(trend_confirmation=True, weekly_hist=.02, previous_week_low=3.), 3, "STAGE_ENTRY_AND_EXIT") is None
    assert trade["holding_stage"] == "TREND"
    assert trade["structural_stop"] == 3.
    policy.exit_decision(trade, holding_row(weekly_hist=.02, previous_week_low=2.9), 4, "STAGE_ENTRY_AND_EXIT")
    assert trade["structural_stop"] == 3.
    assert policy.exit_decision(trade, holding_row(ac=2.99, weekly_hist=.02), 5, "STAGE_ENTRY_AND_EXIT") == "KNOWN_STRUCTURAL_LOW_FAILED"


def test_blocked_exit_remains_locked_and_dividend_is_paid_after_exit():
    frame, div, risks = execution_frame()
    result = policy.account(frame, div, risks, "SAME_ENTRY_FIXED_EXIT", "STRESS", frame.date.iloc[1])
    verify_account(result)
    trade = result["trades"].iloc[0]
    assert trade.entry_date == frame.date.iloc[1]
    assert trade.exit_date == frame.date.iloc[3]
    assert result["rejections"].date.eq(frame.date.iloc[2]).any()
    assert result["daily"].dividend_accrual.sum() == trade.entry_quantity*.05
    assert result["daily"].dividend_paid.sum() == trade.entry_quantity*.05
    assert result["terminal"]["unpaid_dividend_cny"] == 0
    assert result["orders"].iloc[-1].reason == "LOCKED_EXIT"


def test_open_below_known_support_cancels_entry():
    frame, div, risks = execution_frame()
    frame.loc[1, "open"] = 1.9
    for mode in policy.POLICIES:
        result = policy.account(frame, div, risks, mode, "STRESS", frame.date.iloc[1])
        verify_account(result)
        assert result["orders"].empty
        assert result["rejections"].reason.eq("OPEN_ALREADY_BELOW_KNOWN_SUPPORT").any()
