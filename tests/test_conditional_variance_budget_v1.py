"""条件方差的数学、时点、缺失和实际调整路径测试。"""
import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative
from research.conditional_variance_budget_inputs_v1 import objective_gradient, variance_path, fit_variance, train_schedule, forecast_frame, budget_frame, MODELS
from research.event_clock_account_v1 import simulate_event_account


def market(n=300):
    r = .004*np.sin(np.arange(n)*.91)+.002*np.cos(np.arange(n)*.31)
    close = 10.*np.cumprod(1+r)
    return pd.DataFrame({"date": pd.bdate_range("2010-01-04", periods=n), "total_simple": r, "vol20": pd.Series(r).rolling(20).std()*np.sqrt(242),
                         "open": close, "close": close, "previous_close": np.r_[10., close[:-1]], "dividend": 0., "variance60": .0001})


def test_vector_recursion_and_gradient_match_independent_scalar_and_difference():
    r = np.array([1., -2., .5, 3., -.4])
    expected = [2.]
    for shock in r[:-1]:
        expected.append(.1+.07*shock**2+.85*expected[-1])
    np.testing.assert_allclose(variance_path(r, .1, .07, .85, 2.), expected, rtol=0, atol=1e-12)
    theta = np.array([-2., -.4, 2.])
    numerical = approx_derivative(lambda x: objective_gradient(x, r, 2.)[0], theta, method="3-point").ravel()
    np.testing.assert_allclose(objective_gradient(theta, r, 2.)[1], numerical, rtol=1e-5, atol=1e-8)


def test_estimator_is_stationary_and_does_not_substitute_zero_variance():
    cfg = {"maximum_iterations": 200}
    fitted = fit_variance(market().total_simple.to_numpy(), cfg)
    assert fitted["status"] == "FIT_COMPLETE"
    m = fitted["model"]
    assert m["omega"] > 0 and 0 <= m["alpha"]+m["beta"] < 1.
    assert fitted["objective"] <= fitted["initial_objective"]+1e-12
    assert fit_variance(np.zeros(260), cfg)["status"] == "NO_VIEW_ZERO_TRAINING_VARIANCE"


def test_full_training_window_and_future_changes_do_not_change_old_model():
    d = market(320)
    cfg = {"training_window": 756, "minimum_training_rows": 242, "maximum_iterations": 200}
    schedule = [{"fit_index": 250}, {"fit_index": 280}]
    original = train_schedule(d, schedule, cfg)
    changed = d.copy()
    changed.loc[281:, "total_simple"] = 1000.
    assert original == train_schedule(changed, schedule, cfg)
    assert original == train_schedule(d.iloc[:281], schedule, cfg)
    changed.loc[260, "total_simple"] = np.nan
    missing = train_schedule(changed, schedule, cfg)
    assert missing[0] == original[0] and missing[1]["status"] == "NO_VIEW_INCOMPLETE_VARIANCE_WINDOW"


def test_forecast_uses_known_current_shock_then_next_variance_and_keeps_gap():
    d = market(12)
    r = {"fit_index": 3, "latest_observed_return_index": 3, "status": "FIT_COMPLETE", "model": {"omega": .1, "alpha": .1, "beta": .8, "next_variance_percent_squared": 2.}}
    f = forecast_frame(d, [r])
    assert f.forecast_annual_volatility.iloc[:3].isna().all()
    assert abs(f.next_variance_percent_squared.iloc[4]-(.1+.1*(100*d.total_simple.iloc[4])**2+.8*2.)) < 1e-12
    pd.testing.assert_frame_equal(f.iloc[:8], forecast_frame(d.iloc[:8], [r]))
    d.loc[6, "total_simple"] = np.nan
    g = forecast_frame(d, [r])
    assert g.forecast_annual_volatility.iloc[6:].isna().all()


def test_missing_forecast_preserves_prior_policy_multiplier_and_known_exit():
    dates = pd.bdate_range("2020-01-02", periods=6)
    parent = pd.DataFrame({"date": dates, "target": [1., 1., 1., 0., np.nan, 1.]})
    forecast = pd.DataFrame({"date": dates, "forecast_annual_volatility": [.2, .4, np.nan, np.nan, .1, .05], "realized_annual_volatility20": .2})
    f = budget_frame(parent, forecast)
    np.testing.assert_allclose(f[MODELS[0]], [.5, .25, .25, 0., np.nan, 1.], equal_nan=True)
    assert f[f"{MODELS[0]}_status"].iloc[2].startswith("NO_VIEW")


def test_actual_next_open_reduction_and_exit_with_same_trade_band():
    d = market(16)
    d[["open", "close", "previous_close"]] = 10.
    targets = np.zeros(len(d))
    targets[3:6] = 1.
    targets[6:9] = .5
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, decisions = simulate_event_account(d, div, cfg, cost, str(d.date.iloc[3].date()), MODELS[0], targets=targets, event_mask=np.ones(len(d), bool))
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].iloc[0] == d.date.iloc[4]
    sells = ledger[ledger.filled_quantity.lt(0)]
    assert sells.date.to_list() == [d.date.iloc[7], d.date.iloc[10]]
    assert sells.shares.iloc[0] > 0 and sells.shares.iloc[1] == 0
    assert ledger.accounting_error.abs().max() < 1e-6
