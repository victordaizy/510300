"""检验回撤路径、最优预算、时钟和真实账户，合成数据不读取研究业绩。"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from research.conditional_drawdown_budget_optimizer_v1 import drawdown_path, optimal_drawdown_budget
from research.two_policy_tail_loss_optimizer_v1 import empirical_tail_loss
from research.conditional_drawdown_budget_inputs_v1 import budget_frame, prefix_factors
from research.event_clock_account_v1 import simulate_event_account


def fixture():
    dates = pd.bdate_range("2020-01-27", periods=75)
    x = np.tile([-.01, .01, .02], 25)
    returns = np.column_stack([x*2, x])
    states = np.tile([1., 0.], (len(dates), 1))
    return dates, returns, states


def test_hand_drawdown_uses_zero_anchor_and_uncompounded_path():
    np.testing.assert_allclose(drawdown_path([-.1, .05, -.02, .2, -.03]), [.1, .05, .07, 0., .03], atol=1e-15)
    assert not np.isclose(drawdown_path([.1, -.1])[-1], .11)


def test_fractional_tail_includes_days_instead_of_only_distinct_events():
    values = np.r_[np.repeat(.04, 12), .02, np.zeros(229)]
    assert np.isclose(empirical_tail_loss(values), (.04*12+.02*.1)/12.1)
    assert np.isclose(empirical_tail_loss(np.r_[.04, np.zeros(241)]), .04/12.1)


def test_same_daily_losses_have_different_chronological_drawdown():
    a, b = np.array([.1, -.06, -.06, .1]), np.array([.1, -.06, .1, -.06])
    assert np.isclose(empirical_tail_loss(-a), empirical_tail_loss(-b))
    assert np.isclose(empirical_tail_loss(drawdown_path(a)), .12)
    assert np.isclose(empirical_tail_loss(drawdown_path(b)), .06)


def test_opposite_paths_have_zero_risk_at_interior_budget():
    x = np.array([-.02, .02, -.02, .02])
    answer = optimal_drawdown_budget(np.column_stack([x, -x]), .1)
    assert answer["status"] == "CONDITIONAL_DRAWDOWN_BUDGET_AVAILABLE"
    assert np.isclose(answer["panic_budget"], .5, atol=1e-8)
    assert abs(answer["selected_drawdown"]) < 1e-8 and answer["linear_programs"] == 3


def test_lower_loss_path_chooses_boundary_budget():
    x = np.array([-.01, -.02, -.01, -.03])
    answer = optimal_drawdown_budget(np.column_stack([x, x*2]), .2)
    assert np.isclose(answer["panic_budget"], 1.)
    assert np.isclose(answer["selected_drawdown"], .07)


def test_tied_interval_chooses_nearest_old_budget():
    x = np.array([.01, .02, .01, .02])
    values = np.column_stack([x, -x])
    left, inside = optimal_drawdown_budget(values, .2), optimal_drawdown_budget(values, .8)
    assert np.isclose(left["optimal_budget_lower"], .5) and np.isclose(left["optimal_budget_upper"], 1.)
    assert np.isclose(left["panic_budget"], .5) and inside["panic_budget"] == .8


def test_solver_failure_preserves_budget_and_missing_risk(monkeypatch):
    from research import conditional_drawdown_budget_optimizer_v1 as optimizer
    monkeypatch.setattr(optimizer, "linprog", lambda *args, **kwargs: SimpleNamespace(success=False, status=2, message="合成求解失败", fun=None, x=None))
    answer = optimizer.optimal_drawdown_budget(np.array([[.1, -.1], [-.1, .1]]), .37)
    assert answer["status"] == "NO_VIEW_OPTIMIZER_FAILURE_KEEP_BUDGET"
    assert answer["panic_budget"] == .37 and np.isnan(answer["selected_drawdown"])


def test_month_first_clock_warmup_and_budget_persistence():
    dates, returns, states = fixture()
    answer, certificates = budget_frame(dates, returns, states, 1, 5)
    feb = dates.get_loc(pd.Timestamp("2020-02-03"))
    assert answer.panic_budget.iloc[feb-1] == .5 and np.isclose(answer.panic_budget.iloc[feb], 0.)
    assert answer.risk_window_start.iloc[feb] == dates[1] and answer.risk_window_observations.iloc[feb] == 5
    assert np.isclose(answer.panic_budget.iloc[feb+1], answer.panic_budget.iloc[feb]) and len(certificates) == 4
    warm, cert = budget_frame(dates, returns, states, 1, 242)
    assert (warm.panic_budget.iloc[:-1] == .5).all() and not cert


def test_future_changes_prefix_reuse_and_terminal_do_not_change_past():
    dates, returns, states = fixture()
    original, cert = budget_frame(dates, returns, states, 1, 5)
    changed_returns, changed_states = returns.copy(), states.copy()
    changed_returns[25:], changed_states[25:] = .3, [0., 1.]
    changed, _ = budget_frame(dates, changed_returns, changed_states, 1, 5)
    assert_frame_equal(original.iloc[:25], changed.iloc[:25])
    source = pd.DataFrame({"date": dates[:26], "panic_state": states[:26, 0], "learned_state": states[:26, 1]})
    shortened = prefix_factors(original, source)
    prefix, _ = budget_frame(dates[:26], returns[:26], states[:26], 1, 5)
    assert_frame_equal(shortened, prefix)
    returns[-1] = 100.
    terminal, _ = budget_frame(dates, returns, states, 1, 5)
    assert_frame_equal(terminal, original)


def test_missing_window_or_state_preserves_budget_without_forcing_cash():
    dates, returns, states = fixture()
    march = dates.get_loc(pd.Timestamp("2020-03-02"))
    returns[march-4:march+1, 0] = np.nan
    states[march+1, 0] = np.nan
    answer, _ = budget_frame(dates, returns, states, 1, 5)
    assert answer.risk_status.iloc[march] == "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
    assert np.isclose(answer.panic_budget.iloc[march], answer.panic_budget.iloc[march-1])
    assert pd.isna(answer.target.iloc[march+1])


def test_zero_variance_and_degenerate_difference_preserve_original_gate():
    dates, returns, states = fixture()
    march = dates.get_loc(pd.Timestamp("2020-03-02"))
    for same, status in [(False, "NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"), (True, "NO_VIEW_DEGENERATE_DIFFERENCE_KEEP_BUDGET")]:
        values = returns.copy()
        values[march-4:march+1, 0] = values[march-4:march+1, 1] if same else 0.
        answer, _ = budget_frame(dates, values, states, 1, 5)
        assert answer.risk_status.iloc[march] == status
        assert np.isclose(answer.panic_budget.iloc[march], answer.panic_budget.iloc[march-1])


def test_actual_account_next_open_dividend_and_full_zero_target_exit():
    dates, returns, states = fixture()
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0., "variance60": .001})
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    div = pd.DataFrame({"record_date": [dates[3]], "ex_date": [dates[4]], "payment_date": [dates[5]], "cash_dividend_per_share": [.1]})
    data.loc[4:, ["open", "close", "previous_close"]] = 9.9
    data.loc[4, ["previous_close", "dividend"]] = [10., .1]
    factors, _ = budget_frame(dates, returns, states, 1, 5)
    ledger, decisions = simulate_event_account(data, div, cfg, cost, str(dates[1].date()), "合成回撤预算", targets=factors.target.to_numpy(), event_mask=np.ones(len(dates), bool))
    feb = dates.get_loc(pd.Timestamp("2020-02-03"))
    request = decisions[decisions.origin.eq(dates[feb])].iloc[0]
    assert request.requested_quantity < 0 and ledger[ledger.date.eq(dates[feb+1])].filled_quantity.iloc[0] == request.requested_quantity
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert np.isclose(ledger.dividend_recognized.sum(), ledger[ledger.date.eq(dates[3])].shares.iloc[0]*.1)
    assert ledger.shares.iloc[-1] == 0 and ledger.accounting_error.abs().max() < 1e-6
