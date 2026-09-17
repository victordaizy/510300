"""必要边界：二十日方向、原仓位优先、未知、双否决及未来隔离。"""
import numpy as np

from research.confirmed_month_edge_idle_inputs_v1 import confirmation_state,confirmed_target


def test_twenty_session_comparison_equality_and_invalid_wealth():
    w=np.ones(25)
    w[21]=2.
    w[22]=.5
    w[23]=np.nan
    w[24]=0.
    result=confirmation_state(w)
    assert np.isnan(result[:20]).all()
    np.testing.assert_allclose(result[20:],[0.,1.,0.,np.nan,np.nan],equal_nan=True)
    near=np.ones(21)
    near[-1]+=5e-13
    assert confirmation_state(near)[-1]==0.


def test_positive_source_unchanged_and_unknown_parent_remains_unknown():
    result=confirmed_target([.3,.7,np.nan],[1.,np.nan,0.],[0.,np.nan,0.])
    np.testing.assert_allclose(result,[.3,.7,np.nan],equal_nan=True)


def test_each_negative_condition_independently_prevents_supplement():
    result=confirmed_target([0.,0.,0.,0.,0.],[0.,np.nan,1.,1.,np.nan],[np.nan,0.,1.,np.nan,1.])
    np.testing.assert_allclose(result,[0.,0.,1.,np.nan,np.nan],equal_nan=True)


def test_calendar_expiry_and_price_reversal_exit_supplement():
    result=confirmed_target([0.,0.,0.,.4],[1.,0.,1.,0.],[1.,1.,0.,0.])
    np.testing.assert_array_equal(result,[1.,0.,0.,.4])


def test_future_prices_do_not_change_known_confirmation():
    w=np.linspace(1.,2.,40)
    before=confirmation_state(w[:30])
    w[30:]=.1
    np.testing.assert_allclose(confirmation_state(w)[:30],before,equal_nan=True)
