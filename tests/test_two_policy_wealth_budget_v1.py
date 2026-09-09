"""财富预算的观察时钟、缺失与真实成交验证。"""
import numpy as np
import pandas as pd
from research.two_policy_wealth_budget_inputs_v1 import budget_frame
from research.event_clock_account_v1 import simulate_event_account


def fixture():
    dates = pd.bdate_range("2020-01-02", periods=10)
    equity = np.full((10, 2), 200000.)
    states = np.tile([1., 0.], (10, 1))
    return dates, equity, states


def test_prepare_is_equal_and_each_known_close_can_update_without_warmup():
    dates, equity, states = fixture()
    equity[0] = [999999., 1.]
    equity[1] = [400000., 200000.]
    result = budget_frame(dates, equity, states, 1)
    assert result.target.iloc[0] == .5
    assert np.isclose(result.target.iloc[1], 2/3)
    assert result.last_successful_budget_origin.iloc[1] == dates[1]
    assert result.target.iloc[2] == .5


def test_missing_or_nonpositive_equity_keeps_prior_budget_and_can_recover():
    dates, equity, states = fixture()
    equity[1] = [400000., 200000.]
    equity[2:5, 0] = [np.nan, 0., np.inf]
    equity[5] = [100000., 300000.]
    result = budget_frame(dates, equity, states, 1)
    np.testing.assert_allclose(result.target.iloc[1:5], 2/3)
    assert (result.budget_status.iloc[2:5] == "NO_VIEW_INVALID_REFERENCE_EQUITY_KEEP_BUDGET").all()
    assert (result.last_successful_budget_origin.iloc[2:5] == dates[1]).all()
    assert result.target.iloc[5] == .25


def test_unknown_signal_keeps_no_view_without_preventing_known_wealth_update():
    dates, equity, states = fixture()
    states[2, 0] = np.nan
    equity[2] = [100000., 300000.]
    result = budget_frame(dates, equity, states, 1)
    assert np.isnan(result.target.iloc[2]) and result.panic_budget.iloc[2] == .25


def test_future_values_and_terminal_open_do_not_affect_prior_targets():
    dates, equity, states = fixture()
    original = budget_frame(dates, equity, states, 1)
    equity[5:, 0] = 900000.
    states[5:] = [0., 1.]
    changed = budget_frame(dates, equity, states, 1)
    pd.testing.assert_frame_equal(original.iloc[:5], changed.iloc[:5])
    prefix = budget_frame(dates[:6], equity[:6], states[:6], 1)
    pd.testing.assert_frame_equal(original.iloc[:5], prefix.iloc[:5])
    assert np.isnan(changed.target.iloc[-1]) and not changed.budget_update_scheduled.iloc[-1]


def test_identical_reference_scaling_does_not_change_budget():
    dates, equity, states = fixture()
    equity[:, 0] *= 1.3
    original = budget_frame(dates, equity, states, 1)
    scaled = budget_frame(dates, equity*100, states, 1)
    np.testing.assert_allclose(original.target, scaled.target, atol=1e-15, equal_nan=True)


def test_real_account_adjusts_next_open_keeps_dividend_and_full_exit():
    dates, equity, states = fixture()
    equity[2:] = [100000., 300000.]
    states[5:] = 0.
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0., "variance60": .001})
    data.loc[4:, ["open", "close", "previous_close"]] = 9.9
    data.loc[4, ["previous_close", "dividend"]] = [10., .1]
    div = pd.DataFrame({"record_date": [dates[3]], "ex_date": [dates[4]], "payment_date": [dates[7]], "cash_dividend_per_share": [.1]})
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    f = budget_frame(dates, equity, states, 1)
    ledger, decisions = simulate_event_account(data, div, cfg, cost, str(dates[1].date()), "WEALTH_BUDGET", targets=f.target.to_numpy(), event_mask=np.ones(len(data), bool))
    reduction = decisions[decisions.origin.eq(dates[2])].iloc[0]
    assert reduction.requested_quantity < 0
    assert ledger[ledger.date.eq(dates[3])].filled_quantity.iloc[0] == reduction.requested_quantity
    assert ledger[ledger.date.eq(dates[6])].shares.iloc[0] == 0
    entitled = ledger[ledger.date.eq(dates[3])].shares.iloc[0]
    assert entitled > 0 and np.isclose(ledger.dividend_recognized.sum(), entitled*.1)
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.shares.iloc[-1] == 0
