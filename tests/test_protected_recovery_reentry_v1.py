import numpy as np

from research.protected_recovery_reentry_inputs_v1 import apply_gate,recovery_gate


def test_exact_exit_and_recovery_boundaries_multiple_times():
    r=recovery_gate([.2]*6,[1.,.97,.99,1.,.96,1.01],[.01]*6)
    np.testing.assert_allclose(r[0],[1,0,0,1,0,1])
    assert r[5].sum()==2 and r[6].sum()==2


def test_paused_reference_and_distance_are_fixed():
    r=recovery_gate([.2]*5,[1.,.97,.98,.99,1.],[.01,.02,.03,.04,.05])
    np.testing.assert_allclose(r[1],.03)
    np.testing.assert_allclose(r[3][1:4],1.)
    assert r[6][-1]


def test_unknown_direction_cannot_authorize_reentry():
    r=recovery_gate([.2,.2,np.nan,.2],[1.,.97,2.,1.],[.01]*4)
    np.testing.assert_allclose(r[0],[1,0,0,1])
    assert not r[6][2] and r[6][3]


def test_known_zero_clears_episode_and_risk_distance():
    r=recovery_gate([.2,.2,0.,.2,.2],[1.,.97,1.,1.,.96],[.01,.01,.01,.02,.02])
    np.testing.assert_allclose(r[0],[1,0,0,1,1])
    assert r[1][3]==.06 and r[4].sum()==2


def test_invalid_initial_risk_cannot_be_repaired():
    r=recovery_gate([.2,.2,0.,.2],[1.]*4,[np.nan,.01,.01,.01])
    np.testing.assert_allclose(r[0],[np.nan,np.nan,0,1],equal_nan=True)


def test_future_price_cannot_change_previous_qualification():
    a=recovery_gate([.2]*5,[1.,.97,1.,1.01,1.02],[.01]*5)
    b=recovery_gate([.2]*5,[1.,.97,1.,1.01,.01],[.01]*5)
    for x,y in zip(a,b):
        np.testing.assert_allclose(x[:4],y[:4],equal_nan=True)
    assert b[0][-1]==0 and a[0][-1]==1


def test_protective_zero_does_not_require_known_budget():
    np.testing.assert_allclose(apply_gate([0,1,1,np.nan],[np.nan,.3,np.nan,0]),[0,.3,np.nan,np.nan],equal_nan=True)
