"""检验三维最小方差、过去时钟和自身真实账户执行。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from research.three_policy_min_variance_inputs_v1 import minimum_variance,budget_frame
from research.event_clock_account_v1 import simulate_event_account


def fixture():
    dates=pd.bdate_range("2020-01-27",periods=75)
    rng=np.random.default_rng(88);returns=rng.normal(0,.01,(75,3))*[2.,1.,1.5]
    states=np.tile([1.,0.,0.],(75,1))
    return dates,returns,states


def test_known_interior_boundary_and_scale_invariance():
    diagonal=np.diag([1.,4.,9.]);expected=(1/np.diag(diagonal));expected/=expected.sum()
    answer=minimum_variance(diagonal)
    np.testing.assert_allclose(answer["weights"],expected,atol=1e-14,rtol=0)
    boundary=np.array([[1.,2.,3.],[2.,5.,6.],[3.,6.,10.]])
    np.testing.assert_allclose(minimum_variance(boundary)["weights"],[1,0,0],atol=1e-14,rtol=0)
    np.testing.assert_allclose(minimum_variance(diagonal*1e-10)["weights"],expected,atol=1e-14,rtol=0)


def test_correlations_change_allocation_and_convex_global_optimum():
    rng=np.random.default_rng(882)
    grid=np.array([[i/100,j/100,1-(i+j)/100] for i in range(101) for j in range(101-i)])
    for trial in range(8):
        a=rng.normal(size=(3,3));cov=a@a.T+.1*np.eye(3)
        w=minimum_variance(cov)["weights"]
        assert w@cov@w<=np.einsum("ij,jk,ik->i",grid,cov,grid).min()+1e-12
    c=np.array([[1.,.7,0.],[.7,1.,0.],[0.,0.,1.]])
    assert minimum_variance(c)["weights"][2]>1/3


def test_singular_covariance_does_not_invent_unique_budget():
    result=minimum_variance(np.ones((3,3)))
    assert result["weights"] is None and result["status"]=="NO_VIEW_SINGULAR_COVARIANCE_KEEP_BUDGET"
    dates,returns,states=fixture();march=dates.get_loc(pd.Timestamp("2020-03-02"))
    returns[march-4:march+1,1]=returns[march-4:march+1,0]
    result=budget_frame(dates,returns,states,1,5)
    assert result.risk_status.iloc[march]=="NO_VIEW_SINGULAR_COVARIANCE_KEEP_BUDGET"
    np.testing.assert_array_equal(result[["panic_budget","learned_budget","breakout_budget"]].iloc[march],result[["panic_budget","learned_budget","breakout_budget"]].iloc[march-1])


def test_monthly_clock_and_no_borrowing_before_account_start():
    dates,returns,states=fixture();result=budget_frame(dates,returns,states,1,242)
    np.testing.assert_allclose(result[["panic_budget","learned_budget","breakout_budget"]].iloc[:-1],1/3,atol=0,rtol=0)
    assert result.last_successful_risk_origin.isna().all()
    short=budget_frame(dates,returns,states,1,5);month=dates.get_loc(pd.Timestamp("2020-02-03"))
    assert short.risk_update_scheduled.iloc[month] and short.risk_window_start.iloc[month]==dates[1]
    for t in range(1,len(short)-1):
        if not short.risk_update_scheduled.iloc[t]:
            np.testing.assert_array_equal(short[["panic_budget","learned_budget","breakout_budget"]].iloc[t],short[["panic_budget","learned_budget","breakout_budget"]].iloc[t-1])


def test_future_returns_and_terminal_do_not_change_past_decisions():
    dates,returns,states=fixture();original=budget_frame(dates,returns,states,1,5)
    changed_r=returns.copy();changed_s=states.copy();changed_r[25:]=10.;changed_s[25:]=[0.,1.,1.]
    changed=budget_frame(dates,changed_r,changed_s,1,5);assert_frame_equal(original.iloc[:25],changed.iloc[:25])
    prefix=budget_frame(dates[:26],returns[:26],states[:26],1,5);assert_frame_equal(original.iloc[:25],prefix.iloc[:25])
    returns[-1]=100.;assert_frame_equal(original,budget_frame(dates,returns,states,1,5))
    assert pd.isna(original.target.iloc[-1])


def test_zero_or_missing_risk_keeps_previous_and_does_not_delete_cash_days():
    dates,returns,states=fixture();march=dates.get_loc(pd.Timestamp("2020-03-02"))
    for value,status in [(0.,"NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"),(np.nan,"NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET")]:
        data=returns.copy();data[march-4:march+1,2]=value;result=budget_frame(dates,data,states,1,5)
        assert result.risk_status.iloc[march]==status and result.risk_window_observations.iloc[march]==5
        np.testing.assert_array_equal(result[["panic_budget","learned_budget","breakout_budget"]].iloc[march],result[["panic_budget","learned_budget","breakout_budget"]].iloc[march-1])


def test_state_target_uses_three_weights_and_missing_is_not_cash():
    dates,returns,states=fixture();states[10,1]=np.nan;states[11]=[1.,1.,1.];states[12]=[0.,0.,0.]
    result=budget_frame(dates,returns,states,1,5)
    assert pd.isna(result.target.iloc[10]) and np.isclose(result.target.iloc[11],1.) and result.target.iloc[12]==0.
    assert result.target.iloc[9]==result.panic_budget.iloc[9]


def test_real_account_next_open_dividend_actual_holdings_and_terminal():
    dates,returns,states=fixture();states[:]=1.;states[5:]=0.;states[2,1]=np.nan
    data=pd.DataFrame({"date":dates,"open":10.,"close":10.,"previous_close":10.,"dividend":0.,"variance60":.001})
    cfg={"initial_capital":200000.,"lot":100,"tick":.001,"limit_fraction":.1,"weight_band":.1}
    cost={"commission":.0002,"minimum":5.,"slippage":.0005}
    div=pd.DataFrame({"record_date":[dates[3]],"ex_date":[dates[4]],"payment_date":[dates[5]],"cash_dividend_per_share":[.1]})
    data.loc[4:,["open","close","previous_close"]]=9.9;data.loc[4,["previous_close","dividend"]]=[10.,.1]
    targets=budget_frame(dates,returns,states,1,5)
    ledger,decisions=simulate_event_account(data,div,cfg,cost,str(dates[1].date()),"SYNTHETIC_THREE_BUDGET",targets=targets.target.to_numpy(),event_mask=np.ones(len(dates),bool))
    assert ledger.shares.iloc[0]>0 and ledger.shares.iloc[1]==ledger.shares.iloc[0]
    missing=decisions[decisions.origin.eq(dates[2])].iloc[0];assert missing.requested_quantity==0 and pd.isna(missing.reference_weight)
    sell=decisions[decisions.origin.eq(dates[5])].iloc[0];assert sell.requested_quantity<0
    assert ledger[ledger.date.eq(dates[6])].filled_quantity.iloc[0]==sell.requested_quantity
    held=ledger[ledger.date.eq(dates[3])].shares.iloc[0];assert np.isclose(ledger.dividend_recognized.sum(),held*.1)
    assert ledger.shares.iloc[-1]==0 and ledger.accounting_error.abs().max()<1e-6
    assert ledger.commission.sum()>0 and ledger.slippage_cost.sum()>0
