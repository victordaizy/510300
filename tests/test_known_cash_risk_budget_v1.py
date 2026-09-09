"""验证已知现金的数学边界、证据与当前月度时钟。"""
import numpy as np
import pandas as pd
import pytest
from research.known_cash_risk_budget_inputs_v1 import risk_support_budget, budget_frame, observed_cash_days
from research.tangency_reference_budget_inputs_v1 import tangency_budget


def test_known_cash_reduces_risk_dimension_and_mean_breaks_equal_sharpe_size_tie():
    for active in [0, 1]:
        mean, covariance, proof = np.zeros(2), np.zeros((2, 2)), np.ones(2, bool)
        mean[active], covariance[active, active], proof[active] = .002, .0004, False
        weights, status, ratio = risk_support_budget(mean, covariance, [0., 0.], proof)
        assert weights[active] == 1 and weights[1-active] == 0
        assert status == "KNOWN_CASH_SINGLE_RISK_BUDGET_AVAILABLE"
        for fraction in [.1, .5, 1.]:
            scaled = weights*fraction
            assert np.isclose(scaled@mean/np.sqrt(scaled@covariance@scaled), ratio)
            assert scaled@mean <= weights@mean


def test_zero_return_without_cash_proof_cannot_activate_the_new_branch():
    previous = [.3, .7]
    for flags in [[False, False], [np.nan, False]]:
        weights, status, _ = risk_support_budget([0., .002], [[0., 0.], [0., .0004]], previous, flags)
        np.testing.assert_array_equal(weights, previous)
        assert status.startswith("NO_VIEW")
    with pytest.raises(ValueError, match="证据与收益矩矛盾"):
        risk_support_budget([.001, .002], [[.0001, 0.], [0., .0004]], previous, [True, False])


def test_full_rank_and_nonpositive_mean_keep_the_preexisting_mathematical_rule():
    covariance = [[.0004, .00002], [.00002, .0001]]
    for mean in [[.002, .001], [-.001, .002], [0., -.001], [np.nan, .002]]:
        new, status, score = risk_support_budget(mean, covariance, [.4, .6], [False, False])
        old, old_status, old_score = tangency_budget(mean, covariance, [.4, .6])
        np.testing.assert_array_equal(new, old)
        assert status == old_status and (np.isnan(score) and np.isnan(old_score) or score == old_score)


def test_cash_evidence_requires_actual_zero_shares_no_fills_and_zero_net_return():
    dates = pd.bdate_range("2020-01-02", periods=6)
    ledger = pd.DataFrame({"date": dates[1:], "shares": [0, 0, 100, 0, 0], "filled_quantity": [0, 0, 0, -100, 0], "net_return": [0., .001, 0., 0., 0.], "cash": 200000., "dividend_receivable": 0., "mark": 10.})
    ledger["equity"] = ledger.cash+ledger.shares*ledger.mark
    known = observed_cash_days(ledger, dates, 1)
    np.testing.assert_allclose(known, [np.nan, 1., 0., 0., 0., 1.], equal_nan=True)


def test_cash_support_uses_complete_past_window_and_cannot_change_earlier_months():
    dates = pd.bdate_range("2020-01-02", periods=80)
    returns = np.c_[np.zeros(80), .003+np.sin(np.arange(80))*.002]
    states = np.tile([0., 1.], (80, 1))
    proofs = np.tile([1., 0.], (80, 1))
    result = budget_frame(dates, returns, states, proofs, 1, 20)
    valid = result.budget_status.eq("KNOWN_CASH_SINGLE_RISK_BUDGET_AVAILABLE")
    assert valid.any() and result.learned_budget[valid].eq(1).all()
    changed_proof = proofs.copy()
    changed_proof[50:, 0] = np.nan
    changed = budget_frame(dates, returns, states, changed_proof, 1, 20)
    pd.testing.assert_frame_equal(result.iloc[:50], changed.iloc[:50])
    prefix = budget_frame(dates[:51], returns[:51], states[:51], proofs[:51], 1, 20)
    pd.testing.assert_frame_equal(result.iloc[:50], prefix.iloc[:50])
    assert np.isnan(result.target.iloc[-1])
