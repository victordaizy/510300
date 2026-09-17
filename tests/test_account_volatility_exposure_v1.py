import numpy as np

from research.account_volatility_exposure_inputs_v1 import risk_budget


def test_hand_window_today_and_future_separation():
    r = np.r_[np.nan, .01, -.01, .02, 100.]
    p = np.full(5, .2)
    target, risk, multiplier = risk_budget(p, r, first=1, annual_days=1, window=3, risk_target=.02)
    assert np.isnan(risk[:3]).all()
    np.testing.assert_allclose(risk[3], np.std([.01, -.01, .02], ddof=1), atol=1e-14)
    changed = r.copy()
    changed[4] = -999
    other = risk_budget(p, changed, first=1, annual_days=1, window=3, risk_target=.02)
    np.testing.assert_allclose(target[:4], other[0][:4])
    np.testing.assert_allclose(multiplier[:3], 1.)


def test_unknown_risk_and_known_exit_are_different():
    r = np.array([np.nan, .01, .02, np.nan, .03, .01])
    p = np.array([np.nan, .5, .5, .5, 0., np.nan])
    target, risk, _ = risk_budget(p, r, first=1, window=3)
    assert np.isnan(target[0]) and np.isnan(target[3]) and np.isnan(target[5])
    assert target[4] == 0
    assert np.isnan(risk[3:]).all()


def test_zero_risk_and_cash_days_are_included():
    r = np.array([0., 0., 0., .01, -.01])
    p = np.array([.2, .2, .2, .9, .9])
    target, risk, multiplier = risk_budget(p, r, first=0, window=3)
    assert multiplier[2] == 1 and target[2] == .2
    np.testing.assert_allclose(risk[3], np.std([0, 0, .01], ddof=1)*np.sqrt(242))
    assert 0 <= target[3] <= 1 and 0 <= target[4] <= 1
