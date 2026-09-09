"""核对新增进入退出条件的边界、未知状态与实际执行行为。"""
import numpy as np
import pandas as pd
import pytest
from research.downside_balance_gate_inputs_v1 import balance_gate_targets, PARENT, PRIMARY
from research.event_clock_account_v1 import simulate_event_account

CFG = {"balance_multiple": 2., "weight_band": .10, "initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}


def fixture(n=14):
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": 10., "close": 10.,
        "previous_close": 10., "dividend": 0., "variance60": .0004})
    target = np.full(n, .6); target[[0, -1]] = np.nan
    parent = pd.DataFrame({"date": data.date, "origin_index": np.arange(n), "reference_cost": "BASE", "model": PARENT,
        "target": target, "target_status": "PARENT_TARGET", "full_second_moment20": .04, "downside_second_moment20": .01})
    return data, parent, data.date.iloc[2]


def test_directional_boundary_equality_and_all_zero_window():
    data, parent, start = fixture()
    parent.loc[2, "full_second_moment20"] = .02
    parent.loc[3, "full_second_moment20"] = .019
    parent.loc[4, ["full_second_moment20", "downside_second_moment20"]] = 0.
    f, summary = balance_gate_targets(data, parent, CFG, start, "BASE")
    assert f.target.iloc[2] == .6 and f.balance_allowed.iloc[2] == 1.
    assert f.target.iloc[3] == 0. and f.balance_allowed.iloc[3] == 0.
    assert f.target.iloc[4] == .6 and f.balance_allowed.iloc[4] == 1.
    assert summary["parent_positive_origins_rejected"] == 1


def test_known_parent_exit_does_not_need_risk_but_unknown_is_not_exit():
    data, parent, start = fixture()
    parent.loc[2, ["target", "full_second_moment20", "downside_second_moment20"]] = [0., np.nan, np.nan]
    parent.loc[3, "target"] = np.nan
    parent.loc[3, "full_second_moment20"] = .015
    parent.loc[4, "downside_second_moment20"] = np.nan
    f, summary = balance_gate_targets(data, parent, CFG, start, "BASE")
    assert f.target.iloc[2] == 0. and f.target_status.iloc[2] == "EXPLICIT_PARENT_ZERO"
    assert pd.isna(f.target.iloc[3]) and pd.isna(f.target.iloc[4])
    assert f.target_status.iloc[4] == "NO_VIEW_BALANCE_INPUT"
    assert f.target.iloc[[0, -1]].isna().all() and summary["unknown_target_origins"] == 2


def test_cost_dates_indices_and_unknown_final_close_are_required():
    data, parent, start = fixture()
    with pytest.raises(ValueError):
        balance_gate_targets(data, parent, CFG, start, "STRESS")
    shifted = parent.copy(); shifted.date += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        balance_gate_targets(data, shifted, CFG, start, "BASE")
    shifted = parent.copy(); shifted.loc[3, "origin_index"] = 8
    with pytest.raises(ValueError):
        balance_gate_targets(data, shifted, CFG, start, "BASE")
    shifted = parent.copy(); shifted.loc[len(parent)-1, "target"] = .6
    with pytest.raises(ValueError):
        balance_gate_targets(data, shifted, CFG, start, "BASE")


def test_future_inputs_and_short_prefix_cannot_change_prior_targets():
    data, parent, start = fixture()
    original, _ = balance_gate_targets(data, parent, CFG, start, "BASE")
    changed = parent.copy(); changed.loc[8:, "full_second_moment20"] = .015
    modified, _ = balance_gate_targets(data, changed, CFG, start, "BASE")
    pd.testing.assert_frame_equal(original.iloc[:8], modified.iloc[:8])
    prefix = parent.iloc[:8].copy(); prefix.loc[7, "target"] = np.nan
    short, _ = balance_gate_targets(data.iloc[:8], prefix, CFG, start, "BASE")
    pd.testing.assert_frame_equal(original.iloc[:7], short.iloc[:7])
    assert pd.isna(short.target.iloc[-1])


def test_invalid_risk_values_sizes_identity_and_changed_boundary_are_rejected():
    data, parent, start = fixture()
    for column, value in [("full_second_moment20", -.1), ("full_second_moment20", np.inf),
        ("downside_second_moment20", .1), ("target", 1.1), ("target", np.inf), ("model", "ANOTHER_PARENT")]:
        changed = parent.copy(); changed.loc[3, column] = value
        with pytest.raises(ValueError):
            balance_gate_targets(data, changed, CFG, start, "BASE")
    with pytest.raises(ValueError):
        balance_gate_targets(data, parent, {**CFG, "balance_multiple": 1.5}, start, "BASE")


def test_actual_entry_partial_reduction_exit_reentry_dividend_and_latest_retry():
    for blocked in [False, True]:
        data, parent, start = fixture()
        parent.loc[3, "target"] = .2
        parent.loc[[4, 9], "full_second_moment20"] = .015
        parent.loc[8, "downside_second_moment20"] = np.nan
        parent.loc[10, "target"] = np.nan
        if blocked:
            data.loc[5, "open"] = 9.
        data.loc[3, "dividend"] = .1
        dividends = pd.DataFrame({"record_date": [data.date.iloc[2]], "ex_date": [data.date.iloc[3]],
            "payment_date": [data.date.iloc[8]], "cash_dividend_per_share": [.1]})
        factors, _ = balance_gate_targets(data, parent, CFG, start, "BASE")
        ledger, decisions = simulate_event_account(data, dividends, CFG, {"commission": .0002, "minimum": 5., "slippage": .0005}, start, PRIMARY,
            targets=factors.target.to_numpy(float), event_mask=np.ones(len(data), bool))
        by_date = ledger.set_index("date")
        assert ledger.filled_quantity.gt(0).sum() == 3
        assert by_date.loc[data.date.iloc[4], "filled_quantity"] < 0 and by_date.loc[data.date.iloc[4], "shares"] > 0
        if blocked:
            assert by_date.loc[data.date.iloc[5], "status"] == "UNFILLED_DIRECTIONAL_LIMIT"
            assert by_date.loc[data.date.iloc[6], "filled_quantity"] > 0
        else:
            assert by_date.loc[data.date.iloc[5], "shares"] == 0
            assert by_date.loc[data.date.iloc[6], "filled_quantity"] > 0
        assert by_date.loc[data.date.iloc[9], "requested_quantity"] == 0
        assert by_date.loc[data.date.iloc[10], "shares"] == 0
        assert by_date.loc[data.date.iloc[12], "filled_quantity"] > 0
        assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
        assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[2], "shares"]*.1)
        assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum())
        assert ledger.accounting_error.abs().max() < 1e-6
        assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
