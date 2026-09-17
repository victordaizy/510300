import numpy as np

from research.signal_episode_protective_exit_inputs_v1 import protective_targets


def test_fixed_distance_and_exact_barrier_exit():
    p=[0.,.2,.2,.2,.2]
    w=[1.,1.,1.,.97,1.1]
    s=[.01,.01,.20,.20,.20]
    target,distance,peak,starts,triggers,locked=protective_targets(p,w,s)
    np.testing.assert_allclose(target,[0.,1.,1.,0.,0.])
    np.testing.assert_allclose(distance[1:],.03)
    assert triggers.tolist()==[False,False,False,True,False]
    assert locked[-1] and starts.sum()==1


def test_known_zero_unlocks_and_next_signal_restarts():
    target,distance,_,starts,triggers,_=protective_targets(
        [.2,.2,0.,.2,.2],[1.,.96,.97,1.,.96],[.01,.01,.01,.02,.02])
    np.testing.assert_allclose(target,[1.,0.,0.,1.,1.])
    assert distance[3]==.06 and starts.sum()==2 and triggers.sum()==1


def test_unknown_source_can_still_have_independent_protective_exit():
    target,_,_,_,triggers,_=protective_targets(
        [.2,np.nan,np.nan,.2,0.],[1.,1.01,.97,1.2,1.2],[.01]*5)
    np.testing.assert_allclose(target,[1.,np.nan,0.,0.,0.],equal_nan=True)
    assert triggers[2]


def test_missing_or_zero_initial_risk_is_not_repaired_mid_episode():
    target,_,_,_,_,_=protective_targets([.2,.2,0.,.2,.2],[1.]*5,[np.nan,.01,.01,0.,.01])
    np.testing.assert_allclose(target,[np.nan,np.nan,0.,np.nan,np.nan],equal_nan=True)


def test_future_peak_cannot_stop_an_earlier_day():
    p=[.2]*5
    s=[.01]*5
    first=protective_targets(p,[1.,1.01,1.02,1.03,1.04],s)
    changed=protective_targets(p,[1.,1.01,1.02,100.,1.04],s)
    for a,b in zip(first,changed):
        np.testing.assert_allclose(a[:3],b[:3],equal_nan=True)
    assert changed[4][-1]
