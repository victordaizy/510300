"""仅验证自身峰值、因果目标以及新增账户连接与原记账的一致性。"""
import numpy as np
import pandas as pd
import pytest
from research.own_cushion_account_v1 import cushion_target, simulate_cushion_account
from research.event_clock_account_v1 import simulate_event_account


def fixture():
    dates = pd.bdate_range("2020-01-01", periods=12)
    cl = np.array([4., 4.08, 4.02, 3.96, 3.72, 3.64, 3.88, 3.96, 4.02, 4.08, 4.10, 4.12])
    frame = pd.DataFrame({"date": dates, "open": cl, "close": cl, "previous_close": np.r_[4., cl[:-1]], "dividend": [0.]*12, "variance60": [.01]*12})
    frame.loc[5, "dividend"] = .05
    div = pd.DataFrame({"record_date": [dates[4]], "ex_date": [dates[5]], "payment_date": [dates[8]], "cash_dividend_per_share": [.05]})
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1, "floor_fraction": .8, "cushion_multiplier": 5.}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    targets = np.array([.8, .8, .8, .8, .4, 0., .5, .5, 0., .8, .8, 0.])
    return frame, div, cfg, cost, targets


def test_peak_and_loss_reduce_sleeve_fraction_without_leverage():
    peak = cushion_target(200000, 200000, .6)
    loss = cushion_target(180000, peak["high_water"], .6)
    rebound = cushion_target(190000, loss["high_water"], .6)
    new_peak = cushion_target(220000, rebound["high_water"], .6)
    assert peak["target"] == .6 and new_peak["target"] == .6
    assert loss["floor_value"] == 160000 and loss["high_water"] == 200000
    assert abs(loss["target"]-.6*5*20000/180000) < 1e-12
    assert loss["target"] < rebound["target"] < peak["target"]
    assert new_peak["floor_value"] == 176000


def test_breached_floor_requests_zero_and_unknown_parent_stays_unknown():
    assert cushion_target(159000, 200000, .8)["target"] == 0
    assert cushion_target(160000, 200000, .8)["risk_status"] == "EXPLICIT_FLOOR_EXIT"
    assert np.isnan(cushion_target(190000, 200000, np.nan)["target"])
    with pytest.raises(ValueError):
        cushion_target(200000, 200000, 1.1)


def test_zero_floor_diagnostic_matches_original_event_account_with_dividend():
    frame, div, cfg, cost, targets = fixture()
    diagnostic = {**cfg, "floor_fraction": 0.}
    new, new_decisions = simulate_cushion_account(frame, div, diagnostic, cost, str(frame.date.iloc[1].date()), "TEST", targets)
    old, old_decisions = simulate_event_account(frame, div, cfg, cost, str(frame.date.iloc[1].date()), "TEST", targets=targets, event_mask=np.ones(len(frame), bool))
    pd.testing.assert_frame_equal(new[old.columns], old)
    np.testing.assert_array_equal(new_decisions.requested_quantity, old_decisions.requested_quantity)
    assert new.dividend_recognized.sum() > 0 and new.dividend_paid.sum() == new.dividend_recognized.sum()
    assert new.shares.iloc[-1] == 0


def test_future_changes_cannot_change_previous_targets_or_filled_account():
    frame, div, cfg, cost, targets = fixture()
    ledger, decisions = simulate_cushion_account(frame, div, cfg, cost, str(frame.date.iloc[1].date()), "TEST", targets)
    changed = frame.copy()
    changed.loc[8:, ["open", "close", "previous_close"]] *= .85
    later_targets = targets.copy()
    later_targets[8:] = .2
    second, altered_decisions = simulate_cushion_account(changed, div, cfg, cost, str(frame.date.iloc[1].date()), "TEST", later_targets)
    pd.testing.assert_frame_equal(ledger.iloc[:7], second.iloc[:7])
    pd.testing.assert_frame_equal(decisions.iloc[:8], altered_decisions.iloc[:8])


def test_control_uses_own_real_nav_and_next_open_quantities():
    frame, div, cfg, cost, targets = fixture()
    ledger, decisions = simulate_cushion_account(frame, div, cfg, cost, str(frame.date.iloc[1].date()), "TEST", targets)
    equity = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
    peaks = np.maximum.accumulate(equity)
    expected = targets[:-1]*np.clip(5*(equity-.8*peaks)/equity, 0, 1)
    np.testing.assert_allclose(decisions.high_water, peaks, atol=1e-10, rtol=0)
    np.testing.assert_allclose(decisions.target, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(decisions.origin_equity, equity, atol=1e-10, rtol=0)
    assert decisions.execution_date.to_list() == ledger.date.to_list()
    assert (ledger.filled_quantity.abs() % 100 == 0).all()
    assert (ledger.cash >= 0).all()
