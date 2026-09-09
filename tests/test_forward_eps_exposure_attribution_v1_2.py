"""验证隔夜与日内损益抵消时仍按实际运算量级核对浮点误差。"""
import numpy as np
from research.forward_eps_exposure_attribution_v1_2 import components


def test_offsetting_overnight_intraday_pnl_preserves_zero_net():
    shares=np.repeat(48400.,500);shares[-1]=0.
    c=components(np.repeat(48400.,500),shares,np.repeat(.004000000000000448,500),
                 np.repeat(-.004000000000000448,500),np.zeros(500),np.zeros(500))
    np.testing.assert_allclose(c['net'][:-1],0,atol=1e-12,rtol=0)
    np.testing.assert_allclose(c['mean_size']+c['time_variation'],c['net'],atol=1e-10,rtol=0)


def test_return_decomposition_retains_strict_accuracy():
    c=components([.25,.9,.1],[.3,.8,.2],[.04,-.08,.01],[-.03,.02,-.01],[0,.001,0],[.0001,.0002,.0003])
    direct=np.array([.25,.9,.1])*[.04,-.08,.01]+np.array([.3,.8,.2])*[-.03,.02,-.01]+[0,.001,0]-np.array([.0001,.0002,.0003])
    np.testing.assert_allclose(sum(c[k] for k in ['mean_size','time_variation','dividend_entitlement','cost']),direct,atol=1e-12,rtol=0)
