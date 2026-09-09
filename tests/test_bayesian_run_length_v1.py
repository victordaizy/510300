"""核对共轭更新、时间顺序、完整概率和实际进出场确认。"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t
from research.bayesian_run_length_inputs_v1 import GaussianRunLengthFilter, sequential_forecasts, MeanConfirmationController, PRIMARY, CONTROL
from research.joint_entry_exit_account_v1 import simulate_joint_policy

PRIOR = {"mu": 0., "kappa": 1., "alpha": 2., "beta": .0001}
CFG = {"prior": PRIOR, "hazard": 1/242, "warmup_observations": 2}


def frame(values):
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(values)+1), "total_log": np.r_[np.nan, values]})


def test_one_observation_matches_student_t_and_hand_conjugate_update():
    f = GaussianRunLengthFilter(PRIOR, .2)
    assert f.predictive_log_density(.02)[0] == pytest.approx(student_t.logpdf(.02, df=4, loc=0, scale=np.sqrt(.0001)))
    got = f.update(.02)
    np.testing.assert_allclose(np.exp(f.log_probability), [.2, .8], atol=1e-14)
    np.testing.assert_allclose(f.mu, [0, .01], atol=1e-14)
    np.testing.assert_allclose(f.kappa, [1, 2], atol=0)
    np.testing.assert_allclose(f.alpha, [2, 2.5], atol=0)
    np.testing.assert_allclose(f.beta, [.0001, .0002], atol=1e-14)
    assert got["raw_forward_mean"] == pytest.approx(.008)
    assert got["expected_run_length"] == pytest.approx(.8)
    assert got["predictive_variance"] == pytest.approx(.000216)


def test_second_step_uses_old_likelihood_then_resets_prior_and_grows_both_branches():
    f = GaussianRunLengthFilter(PRIOR, .2)
    f.update(.02)
    density = student_t.pdf(-.01, df=2*f.alpha, loc=f.mu, scale=np.sqrt(f.beta*(f.kappa+1)/(f.alpha*f.kappa)))
    weighted = np.array([.2, .8])*density
    expected = np.r_[.2, .8*weighted/weighted.sum()]
    got = f.update(-.01)
    np.testing.assert_allclose(np.exp(f.log_probability), expected, atol=1e-14)
    np.testing.assert_allclose(f.mu, [0., -.005, .01/3], atol=1e-14)
    assert got["raw_forward_mean"] == pytest.approx(expected@f.mu)
    assert got["log_evidence"] == pytest.approx(np.log(weighted.sum()))


def test_no_change_filter_equals_full_history_closed_form_after_every_observation():
    values = np.array([.01, -.02, .015, .03, -.004])
    f = GaussianRunLengthFilter(PRIOR, 0.)
    for count, value in enumerate(values, start=1):
        result = f.update(value)
        prefix = values[:count]
        mu = prefix.sum()/(1+count)
        beta = .0001+.5*((prefix**2).sum()-prefix.sum()**2/(1+count))
        assert result["raw_forward_mean"] == pytest.approx(mu)
        assert f.beta[0] == pytest.approx(beta)
        assert f.kappa[0] == 1+count and f.alpha[0] == 2+count/2
        assert result["expected_run_length"] == count and result["run_zero_probability"] == 0


def test_full_run_posterior_normalizes_and_zero_run_mass_remains_the_fixed_hazard():
    f = GaussianRunLengthFilter(PRIOR, 1/242)
    values = [.02, .019]*20+[-.02, -.019]*20
    for count, value in enumerate(values, start=1):
        result = f.update(value)
        assert len(f.log_probability) == count+1
        assert np.exp(f.log_probability).sum() == pytest.approx(1., abs=1e-12)
        assert result["run_zero_probability"] == pytest.approx(1/242, abs=1e-12)
    assert result["raw_forward_mean"] < 0
    assert result["most_probable_run_length"] < len(values)


def test_future_perturbation_never_changes_saved_prefix_and_warmup_needs_two_ready_days():
    d = frame([.01, -.005, .015, -.001, .012, -.01])
    changed = d.copy()
    changed.loc[5:, "total_log"] = [-.2, .3]
    for method in [PRIMARY, CONTROL]:
        first, archive, final = sequential_forecasts(d, CFG, method)
        second, altered, _ = sequential_forecasts(changed, CFG, method)
        pd.testing.assert_frame_equal(first.iloc[:5], second.iloc[:5])
        end = archive["offsets"][5]
        np.testing.assert_array_equal(archive["log_probabilities"][:end], altered["log_probabilities"][:end])
        assert first.prediction.iloc[:2].isna().all() and pd.notna(first.prediction.iloc[2])
        assert MeanConfirmationController(first)(2, 0)["policy_action"] is None
        assert len(archive["offsets"]) == len(d)+1 and final["observations"] == 6


def test_internal_missing_return_keeps_calendar_and_makes_remaining_forecasts_no_view():
    d = frame([.01, .01, np.nan, .01, .01])
    for method in [PRIMARY, CONTROL]:
        predictions, archive, final = sequential_forecasts(d, CFG, method)
        assert len(predictions) == len(d) and predictions.date.equals(d.date)
        assert predictions.model_status.iloc[3:].eq("NO_VIEW_SOURCE_GAP_REMAINDER").all()
        assert predictions.prediction.iloc[3:].isna().all() and final["observations"] == 2
        assert (np.diff(archive["offsets"])[3:] == 0).all()


def test_positive_negative_zero_mixed_and_missing_confirmation_are_distinct():
    predictions = pd.DataFrame({"prediction": [.01, .01, 0., .01, -.01, -.01, np.nan], "model_status": "PREDICTION_AVAILABLE"})
    c = MeanConfirmationController(predictions)
    assert c(1, 0)["policy_action"] == 1
    assert c(2, 1)["policy_action"] == 1 and c(2, 0)["policy_action"] == 0
    assert c(4, 1)["policy_action"] == 1 and c(4, 0)["policy_action"] == 0
    assert c(5, 1)["policy_action"] == 0
    assert c(6, 1)["policy_action"] is None and c(6, 2)["policy_action"] == 0


def test_real_account_next_open_exit_lock_reentry_and_terminal_priority():
    d = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=10), "open": 10., "close": 10., "previous_close": 10., "dividend": 0.})
    d.loc[4, "open"] = 9.
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    forecasts = pd.DataFrame({"prediction": [.01, .01, -.01, -.01, .01, .01, .01, .01, .01, .01], "model_status": "PREDICTION_AVAILABLE"})
    ledger, decisions, cycles = simulate_joint_policy(d, div, cfg, cost, d.date.iloc[1], MeanConfirmationController(forecasts))
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [d.date.iloc[2], d.date.iloc[6]]
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [d.date.iloc[5], d.date.iloc[9]]
    assert decisions[decisions.origin_index.eq(4)].iloc[0].decision_status == "LOCKED_EXIT_CONTINUES"
    assert ledger[ledger.date.eq(d.date.iloc[4])].iloc[0].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert len(cycles) == 2 and cycles.holding_intervals.ge(1).all()
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]
