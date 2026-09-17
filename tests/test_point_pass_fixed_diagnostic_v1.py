"""核对连续循环抽样、样本标准差和含亏损周期的集中度分母。"""
import numpy as np
from research.intraday_overnight_increment_v1 import block_indices, return_metrics
from research.point_pass_fixed_diagnostic_v1 import batch_indices, sampled_metrics, concentration


def test_batched_contiguous_circular_indices_match_scalar_reference_and_chunking():
    for size,block in [(11,5),(10,20),(3,1)]:
        a=np.random.default_rng(209)
        b=np.random.default_rng(209)
        all_indices=batch_indices(a,17,size,block)
        scalar=np.vstack([block_indices(b,size,block) for _ in range(17)])
        np.testing.assert_array_equal(all_indices,scalar)
        c=np.random.default_rng(209)
        chunks=np.vstack([batch_indices(c,7,size,block),batch_indices(c,10,size,block)])
        np.testing.assert_array_equal(all_indices,chunks)


def test_sampled_sharpe_and_compound_return_match_original_reference_with_cash_days():
    values=np.array([[0.,.02,-.01,.03,0.],[-.02,.01,0.,.02,-.01],[0.,0.,0.,0.,0.]])
    actual=sampled_metrics(values)
    for j,row in enumerate(values[:2]):
        expected=return_metrics(row,242)
        np.testing.assert_allclose(actual[j],[expected['net_sharpe'],expected['annualized_return']],atol=1e-12,rtol=0)
    assert np.isnan(actual[2,0]) and actual[2,1]==0


def test_cycle_concentration_retains_losing_cycles_and_can_exceed_one():
    r=concentration(np.array([100.,60.,-80.,-20.]),60.)
    assert r['complete_cycles']==4 and r['winning_cycles']==2 and r['losing_cycles']==2
    assert r['largest_cycle_net_profit']==100 and r['top_five_winning_profit']==160
    assert r['largest_cycle_share_of_net_profit']==100/60 and r['top_five_share_of_net_profit']==160/60
