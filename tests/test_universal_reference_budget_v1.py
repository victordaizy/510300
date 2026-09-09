"""验证积分恒等式、当前时点、缺失连续性和实际组合成交。"""
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from research.universal_reference_budget_inputs_v1 import advance, integral_budget, budget_frame
from research.event_clock_account_v1 import simulate_event_account


def fixture():
    dates = pd.bdate_range("2020-01-02", periods=10)
    return dates, np.zeros((10, 2)), np.tile([1., 0.], (10, 1))


def test_uniform_prior_one_day_integral_differs_from_endpoint_wealth_rule():
    c = advance([1.], [1.2, .8])
    assert integral_budget([1.]) == .5
    assert np.isclose(integral_budget(c), (.8+2*1.2)/(3*(1.2+.8)))
    assert not np.isclose(integral_budget(c), 1.2/(1.2+.8))


def test_positive_basis_matches_independent_polynomial_integrals():
    c, p = np.ones(1), Polynomial([1.])
    for a, b in [[1.2, .8], [.7, 1.3], [1.1, 1.4], [.9, .8]]:
        c = advance(c, [a, b])
        p *= Polynomial([b, a-b])
        numerator, denominator = (p*Polynomial([0., 1.])).integ(), p.integ()
        expected = (numerator(1)-numerator(0))/(denominator(1)-denominator(0))
        assert np.isclose(integral_budget(c), expected, atol=1e-14)
    assert np.isclose(integral_budget(c*10000), integral_budget(c), atol=1e-15)


def test_identical_returns_keep_equal_budget_including_zero_cash_days():
    c = np.ones(1)
    for gross in [1.]*30+[1.5]*30+[.8]*30:
        c = advance(c, [gross, gross])
        assert np.isclose(integral_budget(c), .5, atol=1e-14)


def test_future_and_terminal_cannot_change_prior_budgets():
    dates, returns, states = fixture()
    returns[0] = [999., -5.]
    base = budget_frame(dates, returns, states, 1)
    returns[5:] = [.3, -.2]
    altered = budget_frame(dates, returns, states, 1)
    pd.testing.assert_frame_equal(base.iloc[:5], altered.iloc[:5])
    prefix = budget_frame(dates[:6], returns[:6], states[:6], 1)
    pd.testing.assert_frame_equal(base.iloc[:5], prefix.iloc[:5])
    assert np.isnan(altered.target.iloc[-1]) and base.target.iloc[0] == .5


def test_missing_return_breaks_cumulative_history_but_missing_state_is_local():
    dates, returns, states = fixture()
    states[2, 0] = np.nan
    returns[4, 0] = np.nan
    result = budget_frame(dates, returns, states, 1)
    assert np.isnan(result.target.iloc[2]) and result.panic_budget.iloc[2] == .5
    assert result.target.iloc[3] == .5
    assert result.target.iloc[4:].isna().all()
    assert result.budget_status.iloc[5] == "NO_VIEW_INCOMPLETE_CUMULATIVE_HISTORY"


def test_actual_account_uses_next_open_own_cash_dividends_and_zero_target_exit():
    dates, returns, states = fixture()
    states[:3] = [1., 1.]
    states[6:] = [0., 0.]
    f = budget_frame(dates, returns, states, 1)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0., "variance60": .001})
    data.loc[4:, ["open", "close", "previous_close"]] = 9.9
    data.loc[4, ["previous_close", "dividend"]] = [10., .1]
    div = pd.DataFrame({"record_date": [dates[3]], "ex_date": [dates[4]], "payment_date": [dates[8]], "cash_dividend_per_share": [.1]})
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    ledger, decisions = simulate_event_account(data, div, cfg, cost, str(dates[1].date()), "UNIVERSAL_BUDGET", targets=f.target.to_numpy(), event_mask=np.ones(len(data), bool))
    reduction = decisions[decisions.origin.eq(dates[3])].iloc[0]
    assert reduction.requested_quantity < 0
    assert ledger.loc[ledger.date.eq(dates[4]), "filled_quantity"].iloc[0] == reduction.requested_quantity
    assert ledger.loc[ledger.date.eq(dates[7]), "shares"].iloc[0] == 0
    own = ledger.loc[ledger.date.eq(dates[3]), "shares"].iloc[0]
    assert np.isclose(ledger.dividend_recognized.sum(), own*.1)
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.shares.iloc[-1] == 0
