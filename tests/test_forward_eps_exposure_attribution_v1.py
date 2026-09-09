"""核对现金、分红残差和规模与变化分解的具体边界。"""
import numpy as np
import pandas as pd
from research.forward_eps_exposure_attribution_v1 import components,pair_contributions,paired_bootstrap


def test_constant_exposure_has_no_variation_component():
    c=components([.5,.5],[.5,.5],[.1,-.1],[0,0],[0,0],[0,0])
    np.testing.assert_allclose(c['time_variation'],[0,0])
    np.testing.assert_allclose(c['net'],[.05,-.05])


def test_buy_only_before_up_day_shows_covariation_not_mean_size():
    c=components([1,0],[0,0],[.1,-.1],[0,0],[0,0],[0,0])
    assert abs(c['mean_size'].sum())<1e-15
    assert abs(c['time_variation'].sum()-.1)<1e-15


def test_dividend_entitlement_and_cost_not_hidden_in_timing():
    c=components([0,0],[0,0],[.1,-.1],[0,0],[.01,0],[.001,.002])
    np.testing.assert_allclose(c['net'],[.009,-.002])
    np.testing.assert_allclose(c['time_variation'],[0,0])


def frame(weight):
    return pd.DataFrame({'r_on':[.01,.01],'r_day':[0.,0.],'w_on':[weight,weight],'w_day':[0.,0.],
                         'dividend_residual_rate':[0.,0.],'cost_rate':[0.,0.],'net_return':[weight*.01,weight*.01]})


def test_constant_exposure_difference_is_size_effect():
    d=pair_contributions(frame(.8),frame(.2))
    assert np.isclose(d['mean_size'],.006*242)
    assert np.isclose(d['time_variation'],0)
    assert np.isclose(d['net'],d['mean_size'])


def test_identical_accounts_have_zero_increment():
    d=pair_contributions(frame(.8),frame(.8),[1,0,1])
    assert all(v==0 for v in d.values())


def test_constant_share_dollar_pnl_has_zero_share_variation():
    d=components([100,100],[100,100],[.2,-.1],[.03,.02],[0,0],[5,0])
    assert np.isclose(d['net'].sum(),10)
    assert np.isclose(d['time_variation'].sum(),0)


def test_large_notional_reconciliation_uses_floating_point_scale():
    d=components([9234567,6543210],[10000000,1234567],[.123456789,-.87654321],
                 [.987654321,-.123456789],[1234.56,-789.12],[456.78,912.34])
    np.testing.assert_allclose(sum(d[k] for k in ['mean_size','time_variation','dividend_entitlement','cost']),d['net'],atol=1e-8,rtol=0)


def test_numpy_bootstrap_matches_direct_resampled_pairs():
    a,b=frame(.8),frame(.2)
    a['date']=pd.date_range('2025-01-01',periods=2);b['date']=a.date
    a.loc[1,['w_on','net_return']]=[.3,.003]
    actual=paired_bootstrap(a,b,1,5,17)
    rng=np.random.default_rng(17)
    expected=pd.DataFrame([pair_contributions(a,b,rng.integers(0,2,size=2)) for _ in range(5)])
    pd.testing.assert_frame_equal(actual,expected,atol=1e-14,rtol=0)
