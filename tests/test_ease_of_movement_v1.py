"""简易波动的数量单位、除息连续、缺失和下一真实开盘。"""
import numpy as np
import pandas as pd
from research.ease_of_movement_inputs_v1 import factor_frame, EaseOfMovementController, PRIMARY
from research.observed_return_state_account_v1 import simulate_observed_state_account
from research.adaptive_allocation_v1 import Account

CFG = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}
COST = {"commission": .0002, "minimum": 5., "slippage": .0005}
EMPTY_DIV = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])


def market(n=25):
    close = 10.+np.arange(n)*.1
    return pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=n), "high": close+1, "low": close-1, "open": close, "close": close,
                         "previous_close": np.r_[10., close[:-1]], "wealth": close, "volume": 100000000., "dividend": 0.})


def fake_factors(d, values):
    s = pd.Series(values)
    return pd.DataFrame({"date": d.date, "ease_of_movement": s, "entry_cross": s.gt(0)&s.shift().le(0), "exit_nonpositive": s.le(0),
                         "source_state": np.where(s.notna(), "EASE_OF_MOVEMENT_AVAILABLE", "NO_VIEW_INCOMPLETE_MOVEMENT_WINDOW")})


def test_midpoint_range_and_actual_share_volume_have_hand_calculated_values():
    d = market(4)
    d.loc[1, ["high", "low", "close", "wealth"]] = [13., 9., 11., 11.]
    d.loc[2, ["high", "low", "close", "wealth", "volume"]] = [14., 10., 12., 12., 200000000.]
    f = factor_frame(d, 2)
    assert f.movement_per_volume.iloc[1] == 4. and f.movement_per_volume.iloc[2] == 2.
    assert f.ease_of_movement.iloc[2] == 3. and f.ease_of_movement.iloc[:2].isna().all()


def test_cash_dividend_does_not_make_mechanical_midpoint_drop():
    d = market(4)
    d.loc[1, ["high", "low", "close", "wealth", "dividend"]] = [10., 8., 9., 10., 1.]
    f = factor_frame(d, 2)
    assert f.wealth_midpoint.iloc[1] == f.wealth_midpoint.iloc[0] == 10.
    assert f.movement_per_volume.iloc[1] == 0.


def test_missing_volume_does_not_become_zero_and_zero_range_remains_observed():
    d = market(12)
    d.loc[4, "volume"] = np.nan
    d.loc[8, ["high", "low"]] = d.close.iloc[8]
    f = factor_frame(d, 2)
    assert len(f) == len(d) and f.movement_per_volume.iloc[4:6].isna().all()
    assert f.ease_of_movement.iloc[4:7].isna().all()
    assert f.input_valid.iloc[8] and f.movement_per_volume.iloc[8] == 0.


def test_future_changes_do_not_change_past_factors():
    d = market(35)
    f = factor_frame(d)
    d.loc[25:, ["high", "low", "close", "wealth"]] *= 3.
    pd.testing.assert_frame_equal(f.iloc[:25], factor_frame(d).iloc[:25])


def test_entry_needs_new_cross_and_missing_state_preserves_actual_shares():
    d = market(8)
    f = fake_factors(d, [-1., 1., 2., np.nan, -1., 1., 2., 3.])
    c = EaseOfMovementController(d, f, COST, CFG)
    flat = Account(200000.)
    assert c(1, flat)["requested_quantity"] > 0 and c(2, flat)["requested_quantity"] == 0
    held = Account(100., shares=1000, purchase_lots=[(0, 1000)])
    missing = c(3, held)
    assert missing["requested_quantity"] == 0 and pd.isna(missing["reference_weight"])
    assert c(4, held)["requested_quantity"] == -1000
    assert c(5, held)["requested_quantity"] == -1000


def test_next_real_open_and_failed_exit_lock_preserve_full_account():
    d = market(9)
    d[["open", "close", "previous_close"]] = 10.
    d.loc[3, ["close", "previous_close"]] = [9.9, 10.]
    d.loc[4, ["open", "close", "previous_close"]] = [8.9, 9.95, 9.9]
    d.loc[5, ["open", "close", "previous_close"]] = [9.96, 9.96, 9.95]
    f = fake_factors(d, [-1., -1., 1., -1., 1., 1., 1., 1., 1.])
    c = EaseOfMovementController(d, f, COST, CFG)
    ledger, decisions = simulate_observed_state_account(d, EMPTY_DIV, CFG, COST, str(d.date.iloc[3].date()), PRIMARY, c)
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].to_list() == [d.date.iloc[3]]
    assert ledger.loc[ledger.date.eq(d.date.iloc[4]), "status"].iloc[0] == "UNFILLED_DIRECTIONAL_LIMIT"
    assert decisions.loc[decisions.origin.eq(d.date.iloc[4]), "requested_quantity"].iloc[0] < 0
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].to_list() == [d.date.iloc[5]]
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]
