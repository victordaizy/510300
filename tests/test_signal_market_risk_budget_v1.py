"""必要边界：零风险、缺失、区间、未来隔离及来源范围。"""
import numpy as np
import pytest

from research.signal_market_risk_budget_inputs_v1 import EPISODE,PRIMARY,market_targets


def test_daily_budget_ignores_source_magnitude_and_respects_zero():
    targets,current,_,_=market_targets([.1,.8,1.,0.,.2],[.2,.4,0.,np.nan,.05])
    np.testing.assert_allclose(targets[PRIMARY],[.5,.25,1.,0.,1.])
    np.testing.assert_allclose(current,[.5,.25,1.,np.nan,1.],equal_nan=True)


def test_episode_fixed_through_unknown_then_reset():
    targets,_,frozen,starts=market_targets([.1,.8,np.nan,.2,0.,.3],[.2,.4,.9,.8,.1,.5])
    np.testing.assert_allclose(targets[EPISODE],[.5,.5,np.nan,.5,0.,.2],equal_nan=True)
    np.testing.assert_allclose(frozen,[.5,.5,.5,.5,np.nan,.2],equal_nan=True)
    np.testing.assert_array_equal(starts,[True,False,False,False,False,True])


def test_bad_initial_risk_is_not_repaired_and_zero_exit_independent():
    targets,_,_,_=market_targets([.1,.1,0.,.2,.2,0.,.4],[np.nan,.2,np.inf,-.2,.2,np.nan,np.inf])
    np.testing.assert_allclose(targets[EPISODE],[np.nan,np.nan,0.,np.nan,np.nan,0.,np.nan],equal_nan=True)
    np.testing.assert_allclose(targets[PRIMARY],[np.nan,.5,0.,np.nan,.5,0.,np.nan],equal_nan=True)


def test_future_cannot_change_prior_episode_budget():
    source=np.array([0.,.2,.6,np.nan,.1,0.])
    risk=np.array([.1,.2,.4,.2,.8,.1])
    expected=market_targets(source[:4],risk[:4])
    source[4:]=[0.,.9]
    risk[4:]=[0.,100.]
    actual=market_targets(source,risk)
    for model in [PRIMARY,EPISODE]:
        np.testing.assert_allclose(actual[0][model][:4],expected[0][model],equal_nan=True)


def test_invalid_source_and_shape_rejected():
    for source,risk in [([-.1],[.2]),([1.1],[.2]),([np.inf],[.2]),([0.,1.],[.2])]:
        with pytest.raises((RuntimeError,AssertionError,ValueError)):
            market_targets(source,risk)
