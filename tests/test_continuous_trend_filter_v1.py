"""用独立矩阵公式、条件联合分布与交易时钟核对趋势估计。"""
import math

import numpy as np
import pandas as pd
import pytest

from research.continuous_trend_inputs_v1 import DEFAULT_SETTINGS, filter_step, filter_likelihood, fit_noise, policy_target, walk_forward_states
from research.event_clock_account_v1 import simulate_event_account


def test_scalar_step_equals_independent_joseph_matrix_update():
    state = (.3, .02, .4, .07, .2)
    q, r, observation = .002, .03, .42
    transition = np.array([[1., 1.], [0., 1.]])
    covariance = np.array([[state[2], state[3]], [state[3], state[4]]])
    mean = transition @ state[:2]
    predicted = transition @ covariance @ transition.T + np.diag([0., q])
    innovation_variance = predicted[0, 0] + r
    gain = predicted[:, 0] / innovation_variance
    innovation = observation - mean[0]
    mean += gain * innovation
    residual = np.eye(2) - np.outer(gain, [1., 0.])
    posterior = residual @ predicted @ residual.T + r * np.outer(gain, gain)
    actual, e, variance = filter_step(state, observation, q, r)
    np.testing.assert_allclose(actual, [*mean, posterior[0, 0], posterior[0, 1], posterior[1, 1]], rtol=0, atol=1e-14)
    assert e == pytest.approx(innovation) and variance == pytest.approx(innovation_variance)


def test_innovation_likelihood_matches_dense_joint_gaussian():
    y = np.array([.3, .31, .34, .33, .37, .39])
    q, r = .002, .03
    transition = np.array([[1., 1.], [0., 1.]])
    q_matrix, prior = np.diag([0., q]), np.eye(2)
    powers = [np.linalg.matrix_power(transition, i) for i in range(len(y))]
    covariance = np.empty((len(y) - 1, len(y) - 1))
    for i in range(1, len(y)):
        for j in range(1, len(y)):
            value = powers[i] @ prior @ powers[j].T
            for k in range(1, min(i, j) + 1):
                value += powers[i - k] @ q_matrix @ powers[j - k].T
            covariance[i - 1, j - 1] = value[0, 0] + (r if i == j else 0)
    residual = y[1:] - y[0]
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    expected = .5 * ((len(y) - 1) * math.log(2 * math.pi) + logdet + residual @ np.linalg.solve(covariance, residual))
    actual, state = filter_likelihood(y, q, r)
    assert actual == pytest.approx(expected, abs=1e-10)
    sequential = (y[0], 0., 1., 0., 1.)
    for observation in y[1:]:
        sequential, _, _ = filter_step(sequential, observation, q, r)
    np.testing.assert_allclose(state, sequential, rtol=0, atol=1e-14)


@pytest.mark.parametrize("q,r", [(math.exp(-30), 1.), (1., math.exp(-30))])
def test_parameter_bounds_keep_filter_finite_and_positive(q, r):
    y = np.sin(np.arange(120) / 9) * .02
    objective, state = filter_likelihood(y, q, r)
    assert np.isfinite(objective) and state[2] > 0 and state[4] > 0
    assert state[2] * state[4] - state[3] ** 2 >= -1e-12 * state[2] * state[4]


def test_fitting_synthetic_prices_improves_conditional_likelihood():
    rng = np.random.default_rng(20260908)
    level, slope, observations = 0., .001, []
    for _ in range(160):
        observations.append(level + rng.normal(0, .01))
        level += slope
        slope += rng.normal(0, .0001)
    fit = fit_noise(observations, DEFAULT_SETTINGS)
    assert fit["success"]
    assert fit["objective"] <= fit["initial_objective"] + 1e-6
    assert math.exp(-30) <= fit["slope_noise"] <= 1 and math.exp(-30) <= fit["observation_noise"] <= 1


def test_policy_hysteresis_and_exact_boundaries():
    assert policy_target(.5, 1., None)[0] == 0
    assert policy_target(1., 1., 0)[0] == 0
    assert policy_target(1., 1., 1)[0] == 1
    assert policy_target(1.01, 1., 0)[0] == 1
    assert policy_target(0., 1., 1)[0] == 0
    assert policy_target(-.1, 1., 1)[0] == 0


def test_unknown_uncertainty_is_not_treated_as_zero():
    with pytest.raises(ValueError, match="不确定性"):
        policy_target(.01, np.nan, 0)


def fixture(length=115):
    data = pd.DataFrame({"date": pd.bdate_range("2019-12-02", periods=length), "wealth": np.exp(.005 * np.arange(length))})
    settings = {**DEFAULT_SETTINGS, "training_observations": 20}
    return data, settings


def successful_fit(observations, settings):
    return {"success": True, "slope_noise": 1e-8, "observation_noise": 1e-4,
        "state": [float(observations[-1]), .005, 1e-4, 0., 1e-6], "training_observations": len(observations)}


def test_quarterly_windows_end_at_origin_and_current_day_is_not_updated_twice():
    data, settings = fixture()
    states, fits = walk_forward_states(data, settings, fitter=successful_fit)
    fitted = [f for f in fits if f["status"] == "FIT_CONVERGED"]
    assert len(fitted) == 2
    for f in fitted:
        t = f["origin_index"]
        assert f["training_end_index"] == t and f["training_start_index"] == t - 19
        assert states.level.iloc[t] == pytest.approx(math.log(data.wealth.iloc[t]))
        assert states.daily_filter_updates.iloc[t] == 0
        assert states.quarter_fit_consumed_current_observation.iloc[t]
        assert states.daily_filter_updates.iloc[t + 1] == 1
    assert states.target.iloc[:fitted[0]["origin_index"]].isna().all()


def test_failed_new_fit_keeps_prior_model_and_updates_current_day_once():
    data, settings = fixture()
    calls = []

    def fitter(observations, config):
        calls.append(len(observations))
        return successful_fit(observations, config) if len(calls) == 1 else {"success": False}

    states, fits = walk_forward_states(data, settings, fitter=fitter)
    failed = next(f for f in fits if f["status"] == "FIT_FAILED_PRIOR_MODEL_RETAINED")
    t = failed["origin_index"]
    assert states.source_state.iloc[t] == "MODEL_VIEW_AVAILABLE"
    assert states.last_fit_origin_index.iloc[t] < t
    assert states.daily_filter_updates.iloc[t] == 1 and not states.quarter_fit_consumed_current_observation.iloc[t]


def test_first_fit_failure_leaves_all_targets_unknown():
    data, settings = fixture(60)
    states, fits = walk_forward_states(data, settings, fitter=lambda y, cfg: {"success": False})
    assert any(f["status"] == "NO_VIEW_FIT_FAILED" for f in fits)
    assert states.target.isna().all() and states.slope.isna().all()


def test_missing_observation_preserves_unknown_target_and_advances_time():
    data, settings = fixture(60)
    data.loc[30, "wealth"] = np.nan
    states, _ = walk_forward_states(data, settings, fitter=successful_fit)
    assert pd.isna(states.target.iloc[30]) and states.source_state.iloc[30] == "NO_VIEW_MISSING_OBSERVATION"
    assert states.slope_uncertainty.iloc[31] > 0 and states.daily_filter_updates.iloc[31] == 1


def test_future_prices_do_not_change_past_states_or_parameters():
    data, settings = fixture()
    original, fits = walk_forward_states(data, settings, fitter=successful_fit)
    changed = data.copy()
    changed.loc[95:, "wealth"] *= 1.2
    updated, later_fits = walk_forward_states(changed, settings, fitter=successful_fit)
    pd.testing.assert_frame_equal(original.iloc[:95], updated.iloc[:95])
    assert fits == later_fits


def test_first_real_trade_occurs_after_first_available_model():
    data, settings = fixture(60)
    states, fits = walk_forward_states(data, settings, fitter=successful_fit)
    market = data.copy()
    market["open"] = market["close"] = market.wealth * 10
    market["previous_close"] = market.close.shift()
    market["dividend"] = 0.
    market["variance60"] = .0001
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    config = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    ledger, decisions = simulate_event_account(market, dividends, config, cost, str(market.date.iloc[1].date()), "连续趋势测试",
        targets=states.target.to_numpy(), event_mask=np.ones(len(market), bool))
    first_fit = next(f for f in fits if f["status"] == "FIT_CONVERGED")
    first_buy = ledger[ledger.filled_quantity.gt(0)].iloc[0]
    assert first_buy.date == market.date.iloc[first_fit["origin_index"] + 1]
    assert decisions.reference_weight.iloc[:first_fit["origin_index"]].isna().all()
    assert ledger.accounting_error.abs().max() < 1e-7 and ledger.shares.iloc[-1] == 0
