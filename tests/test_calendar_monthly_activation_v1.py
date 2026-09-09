"""只用过去收益月初启停，保持月中规则、零门槛及无有效状态。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from research.calendar_liquidity_timing_v1 import prepare
from research.calendar_monthly_activation_v1 import active_targets, monthly_activation
from research.self_performance_entry_v1 import performance_flags


def example():
    dates = pd.bdate_range("2019-12-02", "2020-03-10")
    data = pd.DataFrame({"date": dates, "feature_valid": True})
    cfg = {"calendar_first_month_sessions": 3, "calendar_month_end_natural_days": 5, "long_break_minimum_date_gap_days": 4, "post_break_sessions": 3}
    prepared, _ = prepare(data, pd.DataFrame({"trade_date": dates, "is_open": True}), cfg)
    flags = pd.DataFrame({"date": dates, "history_available": True, "reference_year_net_return": .01})
    return prepared, flags


def test_monthly_decision_uses_previous_close_and_stays_fixed_midmonth():
    p, f = example()
    f.loc[f.date >= "2020-01-02", "reference_year_net_return"] = -.02
    a = monthly_activation(p, f, "2020-01-01").set_index("market_date")
    assert a.loc["2019-12-31", "is_update"] and a.loc["2019-12-31", "calendar_enabled"]
    assert a.loc["2020-01-15", "calendar_enabled"] and not a.loc["2020-01-15", "is_update"]
    assert a.loc["2020-01-31", "decision_time"] == pd.Timestamp("2020-02-03 09:00")
    assert not a.loc["2020-01-31", "calendar_enabled"]
    assert a.loc["2020-01-31", "last_valid_evidence_date"] == pd.Timestamp("2020-01-31")


def test_exact_zero_disables_without_relabeling_as_missing():
    p, f = example()
    f.reference_year_net_return = 0.
    a = monthly_activation(p, f, "2020-01-01")
    updates = a[a.is_update]
    assert updates.has_valid_activation_decision.all() and not updates.calendar_enabled.any()
    assert updates.learned_budget.eq(1).all()


def test_missing_update_keeps_previous_allocation_and_initial_default_is_explicit():
    p, f = example()
    f.loc[f.date.eq("2020-01-31"), "history_available"] = False
    a = monthly_activation(p, f, "2020-01-01").set_index("market_date")
    assert a.loc["2020-01-31", "calendar_enabled"]
    assert a.loc["2020-01-31", "activation_status"].startswith("NO_VIEW")
    assert a.loc["2020-01-31", "last_valid_evidence_date"] == pd.Timestamp("2019-12-31")
    f.loc[f.date.eq("2019-12-31"), "history_available"] = False
    b = monthly_activation(p, f, "2020-01-01").set_index("market_date")
    assert not b.loc["2019-12-31", "has_valid_activation_decision"] and b.loc["2019-12-31", "learned_budget"] == 1


def test_future_reference_returns_do_not_revise_past_activation():
    p, f = example()
    original = monthly_activation(p, f, "2020-01-01")
    cut = p.index[p.date.eq("2020-02-10")][0]
    f.loc[cut:, "reference_year_net_return"] = -9.
    assert_frame_equal(original.iloc[:cut], monthly_activation(p, f, "2020-01-01").iloc[:cut])


def test_activation_changes_budget_not_underlying_state_and_keeps_missing_target():
    dates = pd.bdate_range("2020-01-01", periods=5)
    state = pd.DataFrame({"market_date": dates, "calendar_state": [1., 0., 1., 0., np.nan],
                          "learned_state": [0., 1., 1., 0., 1.], "inputs_complete": [True] * 4 + [False]})
    enabled = pd.DataFrame({"market_date": dates, "calendar_budget": .5, "learned_budget": .5})
    disabled = pd.DataFrame({"market_date": dates, "calendar_budget": 0., "learned_budget": 1.})
    np.testing.assert_allclose(active_targets(state, enabled), [.5, .5, 1., 0., np.nan], equal_nan=True)
    np.testing.assert_allclose(active_targets(state, disabled), [0., 1., 1., 0., np.nan], equal_nan=True)


def test_reference_window_compounds_cash_days_and_rejects_terminal_open():
    dates = pd.bdate_range("2020-01-01", periods=6)
    data = pd.DataFrame({"date": dates})
    reference = pd.DataFrame({"date": dates, "net_return": [.1, 0., -.1, .05, 0., .02], "mark_clock": ["CLOSE"] * 5 + ["OPEN_TERMINAL"]})
    flags = performance_flags(data, reference, 3)
    assert not flags.history_available.iloc[:2].any()
    assert np.isclose(flags.reference_year_net_return.iloc[2], -.01)
    assert not flags.history_available.iloc[-1]
