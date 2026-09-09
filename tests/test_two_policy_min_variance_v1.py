"""检验月度风险预算的过去窗口、零风险、缺失及真实账户。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from research.two_policy_min_variance_inputs_v1 import budget_frame
from research.event_clock_account_v1 import simulate_event_account


def fixture():
    dates = pd.bdate_range("2020-01-27", periods=75)
    x = np.tile([-.01, .01, .02], 25)
    returns = np.column_stack([x * 2, x])
    states = np.tile([1., 0.], (len(dates), 1))
    return dates, returns, states


def test_only_month_first_close_changes_budget_and_keeps_full_warmup():
    dates, returns, states = fixture()
    result = budget_frame(dates, returns, states, first=1, window=5)
    first_month = dates.get_loc(pd.Timestamp("2020-02-03"))
    assert result.panic_budget.iloc[first_month-1] == .5
    assert np.isclose(result.panic_budget.iloc[first_month], 0.)
    assert result.risk_window_observations.iloc[first_month] == 5
    assert result.risk_window_start.iloc[first_month] == dates[1]
    returns[first_month+1] = [3., .01]
    changed = budget_frame(dates, returns, states, first=1, window=5)
    assert np.isclose(changed.panic_budget.iloc[first_month+1], 0.)


def test_incomplete_history_does_not_start_early_or_borrow_before_evaluation():
    dates, returns, states = fixture()
    answer = budget_frame(dates, returns, states, first=1, window=242)
    assert (answer.panic_budget.iloc[:-1] == .5).all()
    assert answer.last_successful_risk_origin.isna().all()
    assert "NO_VIEW_WARMUP_KEEP_BUDGET" in set(answer.risk_status)


def test_future_returns_and_states_cannot_change_past_allocation():
    dates, returns, states = fixture()
    original = budget_frame(dates, returns, states, 1, 5)
    changed_r, changed_s = returns.copy(), states.copy()
    changed_r[25:], changed_s[25:] = 10., [0., 1.]
    changed = budget_frame(dates, changed_r, changed_s, 1, 5)
    assert_frame_equal(original.iloc[:25], changed.iloc[:25])
    prefix = budget_frame(dates[:26], returns[:26], states[:26], 1, 5)
    assert_frame_equal(original.iloc[:25], prefix.iloc[:25])


def test_zero_or_missing_window_keeps_prior_budget_without_filling_return():
    dates, returns, states = fixture()
    march = dates.get_loc(pd.Timestamp("2020-03-02"))
    for value, expected in [(0., "NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"), (np.nan, "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET")]:
        copy = returns.copy()
        copy[march-4:march+1, 0] = value
        result = budget_frame(dates, copy, states, 1, 5)
        assert result.risk_status.iloc[march] == expected
        assert np.isclose(result.panic_budget.iloc[march], 0.)
        assert result.last_successful_risk_origin.iloc[march] < dates[march]


def test_missing_expert_state_is_not_a_cash_instruction():
    dates, returns, states = fixture()
    states[10, 0] = np.nan
    result = budget_frame(dates, returns, states, 1, 5)
    assert pd.isna(result.target.iloc[10]) and np.isclose(result.panic_budget.iloc[10], 0.)
    assert np.isclose(result.target.iloc[11], 0.)


def test_reference_terminal_return_is_never_used_for_new_order():
    dates, returns, states = fixture()
    out = budget_frame(dates, returns, states, 1, 5)
    returns[-1] = 100.
    changed = budget_frame(dates, returns, states, 1, 5)
    assert_frame_equal(out, changed)
    assert pd.isna(out.target.iloc[-1]) and not out.risk_update_scheduled.iloc[-1]


def test_real_account_uses_next_open_and_keeps_cash_costs_dividend_and_terminal():
    dates, returns, states = fixture()
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0., "variance60": .001})
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    div = pd.DataFrame({"record_date": [dates[7]], "ex_date": [dates[8]], "payment_date": [dates[9]], "cash_dividend_per_share": [.1]})
    data.loc[8:, ["open", "close", "previous_close"]] = 9.9
    data.loc[8, ["previous_close", "dividend"]] = [10., .1]
    result = budget_frame(dates, returns, states, 1, 5)
    ledger, decisions = simulate_event_account(data, div, cfg, cost, str(dates[1].date()), "SYNTHETIC_RISK_BUDGET", targets=result.target.to_numpy(), event_mask=np.ones(len(dates), bool))
    month = dates.get_loc(pd.Timestamp("2020-02-03"))
    request = decisions[decisions.origin.eq(dates[month])].iloc[0]
    assert request.requested_quantity < 0
    assert ledger[ledger.date.eq(dates[month+1])].filled_quantity.iloc[0] == request.requested_quantity
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    held_on_record = ledger[ledger.date.eq(dates[7])].shares.iloc[0]
    assert np.isclose(ledger.dividend_recognized.sum(), held_on_record*.1)
    assert ledger.shares.iloc[-1] == 0 and ledger.accounting_error.abs().max() < 1e-6


def test_covariance_changes_budget_with_same_individual_variances():
    dates, returns, states = fixture()
    same = budget_frame(dates, returns, states, 1, 5)
    opposite = returns.copy()
    opposite[:, 0] *= -1
    hedge = budget_frame(dates, opposite, states, 1, 5)
    month = dates.get_loc(pd.Timestamp("2020-02-03"))
    assert same.panic_budget.iloc[month] == 0.
    assert np.isclose(hedge.panic_budget.iloc[month], 1/3)
    a = hedge.panic_budget.iloc[month]
    assert np.var(opposite[month-4:month+1] @ [a, 1-a], ddof=1) < 1e-25


def test_identical_nonzero_returns_keep_previous_budget_as_unidentified():
    dates, returns, states = fixture()
    march = dates.get_loc(pd.Timestamp("2020-03-02"))
    returns[march-4:march+1, 0] = returns[march-4:march+1, 1]
    result = budget_frame(dates, returns, states, 1, 5)
    assert result.risk_status.iloc[march] == "NO_VIEW_DEGENERATE_DIFFERENCE_KEEP_BUDGET"
    assert result.panic_budget.iloc[march] == 0.
    assert result.last_successful_risk_origin.iloc[march] < dates[march]


def test_interior_and_boundary_budgets_minimize_the_observed_quadratic():
    dates, returns, states = fixture()
    rng = np.random.default_rng(826)
    returns = rng.normal(0, .01, (len(dates), 2))
    returns[:, 0] *= 2
    result = budget_frame(dates, returns, states, 1, 5)
    for t in np.flatnonzero(result.risk_update_scheduled.to_numpy()):
        if result.risk_status.iloc[t] != "MIN_VARIANCE_BUDGET_AVAILABLE":
            continue
        window = returns[t-4:t+1]
        w = result.panic_budget.iloc[t]
        actual = np.var(window @ [w, 1-w], ddof=1)
        grid = np.linspace(0, 1, 101)
        alternatives = window[:, 0, None]*grid + window[:, 1, None]*(1-grid)
        assert actual <= np.var(alternatives, axis=0, ddof=1).min()+1e-15
