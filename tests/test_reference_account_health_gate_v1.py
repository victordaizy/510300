"""健康暂停的必要窗口、等号、缺失、独立退出及未来隔离边界。"""
import numpy as np

from research.reference_account_health_gate_inputs_v1 import MEAN,PRIMARY,health_targets


def test_sixty_realized_days_required_and_equality_allows():
    source=np.full(65,.4)
    nav=np.full(65,100.)
    nav[1:60]=90.
    target,gate,_,_=health_targets(source,nav,1)
    assert gate[PRIMARY][59]==1 and gate[PRIMARY][60]==1
    assert target[PRIMARY][60]==.4
    assert gate[MEAN][60]==1


def test_negative_health_exits_unknown_source_and_recovery_reenters():
    nav=np.full(70,100.)
    nav[60:63]=[90.,95.,105.]
    source=np.full(70,.4)
    source[60]=np.nan
    target,gate,_,_=health_targets(source,nav,1)
    for model in [PRIMARY,MEAN]:
        assert target[model][60]==0 and target[model][61]==0
        assert gate[model][62]==1 and target[model][62]==.4


def test_unknown_window_and_independent_known_zero():
    nav=np.full(70,100.)
    nav[30]=np.nan
    source=np.full(70,.4)
    source[61]=0.
    target,gate,_,_=health_targets(source,nav,1)
    for model in [PRIMARY,MEAN]:
        assert np.isnan(gate[model][60]) and np.isnan(target[model][60])
        assert target[model][61]==0


def test_flat_nav_exact_tie_and_two_conditions_differ():
    nav=np.full(70,123456.789123)
    source=np.full(70,.4)
    _,gate,_,gap=health_targets(source,nav,1)
    assert (gate[MEAN][60:]==1).all() and (gap[60:]==0).all()
    nav=np.full(70,110.)
    nav[0]=100.
    nav[60]=105.
    _,gate,_,_=health_targets(source,nav,1)
    assert gate[PRIMARY][60]==1 and gate[MEAN][60]==0


def test_future_nav_and_targets_do_not_change_past():
    nav=np.linspace(100.,110.,80)
    source=np.full(80,.4)
    expected=health_targets(source[:65],nav[:65],1)
    nav[65:]=10.
    source[65:]=0.
    actual=health_targets(source,nav,1)
    for model in [PRIMARY,MEAN]:
        np.testing.assert_allclose(actual[0][model][:65],expected[0][model],equal_nan=True)
