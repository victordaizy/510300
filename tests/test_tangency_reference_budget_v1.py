"""验证解析最优、现金和缺失语义以及月度因果时钟。"""
import numpy as np
import pandas as pd
from research.tangency_reference_budget_inputs_v1 import tangency_budget, budget_frame


def test_analytic_tangency_matches_independent_inverse_and_dense_objective_check():
    mean, covariance = np.array([.002, .001]), np.array([[.0004, .00002], [.00002, .0001]])
    actual, status, score = tangency_budget(mean, covariance, [.5, .5])
    expected = np.linalg.solve(covariance, mean)
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    w = np.linspace(0, 1, 10001)
    weights = np.c_[w, 1-w]
    independent = weights@mean/np.sqrt(np.einsum('ij,jk,ik->i', weights, covariance, weights))
    assert score >= independent.max()-1e-12 and status == "TANGENCY_BUDGET_AVAILABLE"
    endpoint, _, _ = tangency_budget([.002, -.001], np.diag([.0001, .0001]), [.5, .5])
    np.testing.assert_allclose(endpoint, [1., 0.], atol=0, rtol=0)


def test_no_positive_mean_is_cash_but_missing_or_degenerate_risk_is_no_view():
    previous = [.3, .7]
    cash, status, score = tangency_budget([0., -.01], np.eye(2), previous)
    np.testing.assert_array_equal(cash, [0., 0.])
    assert status == "EXPLICIT_CASH_NO_POSITIVE_PAST_MEAN" and np.isnan(score)
    for mean, covariance in [([np.nan, .1], np.eye(2)), ([.1, .1], np.ones((2, 2)))]:
        weights, status, _ = tangency_budget(mean, covariance, previous)
        np.testing.assert_array_equal(weights, previous)
        assert status.startswith("NO_VIEW")


def test_month_first_uses_past_prefix_and_missing_window_keeps_previous_budget():
    dates = pd.bdate_range("2020-01-02", periods=100)
    rng = np.random.default_rng(23)
    returns = rng.normal(.003, .01, (100, 2))
    states = np.tile([1., 0.], (100, 1))
    base = budget_frame(dates, returns, states, 1, 20)
    assert base.panic_budget.iloc[0] == .5
    expected = np.array([t > 0 and dates[t].month != dates[t-1].month and t < 99 for t in range(100)])
    np.testing.assert_array_equal(base.budget_update_scheduled, expected)
    changed_returns = returns.copy()
    changed_returns[60:] = 2.
    changed = budget_frame(dates, changed_returns, states, 1, 20)
    pd.testing.assert_frame_equal(base.iloc[:60], changed.iloc[:60])
    prefix = budget_frame(dates[:61], returns[:61], states[:61], 1, 20)
    pd.testing.assert_frame_equal(base.iloc[:60], prefix.iloc[:60])
    update = int(np.flatnonzero(expected)[1])
    missing = returns.copy()
    missing[update-1, 0] = np.nan
    broken = budget_frame(dates, missing, states, 1, 20)
    assert broken.budget_status.iloc[update] == "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
    assert broken.panic_budget.iloc[update] == broken.panic_budget.iloc[update-1]


def test_explicit_cash_target_does_not_require_unknown_expert_signal():
    dates = pd.bdate_range("2020-01-02", periods=50)
    returns = np.full((50, 2), -.001)
    states = np.tile([np.nan, 1.], (50, 1))
    result = budget_frame(dates, returns, states, 1, 20)
    assert np.isnan(result.target.iloc[0])
    cash = result.budget_status.eq("EXPLICIT_CASH_NO_POSITIVE_PAST_MEAN")
    assert cash.any() and result.target[cash].eq(0).all()
    assert np.isnan(result.target.iloc[-1])
