"""检验相对收益目标成熟、月度训练及真实账户时钟。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from research.contextual_parent_advantage_inputs_v1 import FEATURES, PARENTS, target_frame, mature_rows, fit_advantage, predict_advantage, build_policy
from research.event_clock_account_v1 import simulate_event_account


def fixture():
    dates=pd.bdate_range("2020-01-20",periods=100)
    t=np.arange(len(dates))
    data=pd.DataFrame({"date":dates,"mom20":np.sin(t/7),"sma120":np.cos(t/11),"vol20":.2+.01*np.sin(t/3)})
    returns=np.column_stack([.001+.002*np.sin(t),.003+.003*np.cos(t)])
    targets=np.tile([.2,.8],(len(dates),1))
    cfg={"target_horizon":2,"training_window":12,"minimum_training_rows":6,"ridge_alpha":1.,"feature_clip":5.}
    return data,returns,targets,cfg


def test_label_uses_compounded_future_complete_closes_and_excludes_origin_terminal():
    data,returns,targets,cfg=fixture()
    returns[0]=[-.9,.9]
    labels=target_frame(data,returns,1,20)
    expected=np.prod(1+returns[1:21],axis=0)-1
    np.testing.assert_allclose(labels.iloc[0][["two_future_return","three_future_return"]].to_numpy(float),expected,atol=1e-14)
    assert labels.target.iloc[0]==expected[1]-expected[0]
    assert labels.maturity_index.iloc[0]==20 and labels.maturity_date.iloc[0]==data.date.iloc[20]
    assert labels.target.iloc[-20:].isna().all()
    assert labels.target_status.iloc[-20:].eq("NO_VIEW_TERMINAL_OR_UNFINISHED_TARGET").all()
    changed=returns.copy();changed[-1]=[100.,200.]
    assert_frame_equal(labels,target_frame(data,changed,1,20))


def test_mature_window_keeps_missing_and_zero_rows_without_using_future():
    data,returns,targets,cfg=fixture();returns[9,0]=np.nan;returns[2:5]=0.
    labels=target_frame(data,returns,1,2)
    before=mature_rows(labels,12,cfg);at=mature_rows(labels,13,cfg)
    assert before.origin_index.max()==10 and before.maturity_index.max()==12
    assert at.origin_index.max()==11 and len(at)==12
    assert before.target.isna().sum()==2 and before.target.eq(0).sum()==2
    assert before.origin_index.tolist()==list(range(11))


def test_saved_ridge_matches_penalized_normal_equations_with_unpenalized_intercept():
    data,returns,targets,cfg=fixture();rows=data.loc[:39,FEATURES].copy();rows["vol20"]=.2
    rows["target"]=.02+.005*rows.mom20-.003*rows.sma120
    model=fit_advantage(rows,cfg)
    x=rows[FEATURES].to_numpy();mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-12,scale,1.)
    z=np.clip((x-mean)/scale,-5,5);design=np.column_stack([np.ones(len(z)),z])
    expected=np.linalg.solve(design.T@design+np.diag([0,1,1,1]),design.T@rows.target)
    np.testing.assert_allclose([model["intercept"],*model["coefficients"]],expected,atol=1e-12,rtol=0)
    assert model["scale"][2]==1.
    v=np.array([.3,.5,.2]);pred=expected[0]+np.clip((v-mean)/scale,-5,5)@expected[1:]
    assert np.isclose(predict_advantage(model,v),pred,atol=1e-12)


def test_future_mutations_and_truncation_do_not_change_prior_models_or_selection():
    data,returns,targets,cfg=fixture();original,labels,models=build_policy(data,returns,targets,1,cfg)
    altered=data.copy();altered.loc[40:,FEATURES]=[-10,20,8]
    changed_r=returns.copy();changed_r[40:]=[-.5,.8];changed_t=targets.copy();changed_t[40:]=[1.,0.]
    changed,_,changed_models=build_policy(altered,changed_r,changed_t,1,cfg)
    assert_frame_equal(original.iloc[:40],changed.iloc[:40])
    assert [r for r in models if r["fit_index"]<40]==[r for r in changed_models if r["fit_index"]<40]
    prefix,_,_=build_policy(data.iloc[:41],returns[:41],targets[:41],1,cfg)
    assert_frame_equal(original.iloc[:40],prefix.iloc[:40])


def test_only_month_start_refits_and_monthly_prediction_origin_is_preserved():
    data,returns,targets,cfg=fixture();f,labels,models=build_policy(data,returns,targets,1,cfg)
    expected=[t for t in range(1,len(data)-1) if data.date.iloc[t].month!=data.date.iloc[t-1].month]
    assert [r["fit_index"] for r in models]==expected
    assert all(r["latest_maturity_index"]<=r["fit_index"] for r in models)
    assert all(r["training_end_index"]<=r["fit_index"]-2 for r in models)
    for left,right in zip(expected,expected[1:]+[len(data)-1]):
        interval=f.iloc[left:right]
        assert interval.prediction_origin.eq(data.date.iloc[left]).all()
        assert interval.relative_prediction.nunique()==1 and interval.selected_parent.nunique()==1
        assert not interval.selection_update_scheduled.iloc[1:].any()


def test_no_model_is_explicit_two_parent_with_missing_prediction_and_targets_stay_missing():
    data,returns,targets,cfg=fixture();cfg["minimum_training_rows"]=252;targets[12,0]=np.nan;targets[13,1]=np.nan
    f,labels,models=build_policy(data,returns,targets,1,cfg)
    assert f.selected_parent.iloc[:-1].eq(PARENTS[0]).all() and f.relative_prediction.isna().all()
    assert f.target.isna().iloc[12] and f.target.iloc[13]==.2
    assert not any(r["fit_attempted"] for r in models)
    cfg["minimum_training_rows"]=6;data.loc[5,"mom20"]=np.nan
    f,_,models=build_policy(data,returns,targets,1,cfg)
    assert models[0]["status"]=="NO_VIEW_INCOMPLETE_TRAINING_INPUT"
    assert models[0]["training_rows"]==9 and models[0]["missing_training_rows"]==1


def test_zero_relative_prediction_chooses_two_and_current_missing_feature_does_not_predict():
    data,returns,targets,cfg=fixture();returns[:,1]=returns[:,0]
    f,_,models=build_policy(data,returns,targets,1,cfg)
    assert all(r["prediction"]==0. and r["selected_parent"]==PARENTS[0] for r in models)
    data.loc[models[0]["fit_index"],"sma120"]=np.nan
    f,_,new=build_policy(data,returns,targets,1,cfg)
    assert new[0]["status"]=="FIT_COMPLETE" and new[0]["prediction_status"]=="NO_VIEW_INCOMPLETE_CURRENT_FEATURES"
    assert pd.isna(new[0]["prediction"]) and new[0]["selected_parent"]==PARENTS[0]


def test_selected_intent_executes_next_open_on_own_cash_and_preserves_dividend_rights():
    data,returns,targets,cfg=fixture();returns[:]=[.001,.003]
    dates=data.date;month=int(np.flatnonzero(dates.dt.month.ne(dates.shift().dt.month))[1])
    targets[20:]=0.;targets[21]=np.nan
    f,_,models=build_policy(data,returns,targets,1,cfg)
    data=data.assign(open=10.,close=10.,previous_close=10.,dividend=0.,variance60=.001)
    div=pd.DataFrame({"record_date":[dates.iloc[15]],"ex_date":[dates.iloc[16]],"payment_date":[dates.iloc[22]],"cash_dividend_per_share":[.1]})
    data.loc[16:,["open","close","previous_close"]]=9.9;data.loc[16,["previous_close","dividend"]]=[10.,.1]
    account={"initial_capital":200000.,"lot":100,"tick":.001,"limit_fraction":.1,"weight_band":.1}
    cost={"commission":.0002,"minimum":5.,"slippage":.0005}
    ledger,decisions=simulate_event_account(data,div,account,cost,str(dates.iloc[1].date()),"SYNTHETIC_CONTEXTUAL_PARENT",targets=f.target.to_numpy(),event_mask=np.ones(len(data),bool))
    assert f.selected_parent.iloc[month]==PARENTS[1]
    assert decisions.loc[decisions.origin.eq(dates.iloc[month]),"requested_quantity"].iloc[0]>0
    assert ledger.loc[ledger.date.eq(dates.iloc[month+1]),"filled_quantity"].iloc[0]>0
    assert ledger.loc[ledger.date.eq(dates.iloc[21]),"shares"].iloc[0]==0
    assert decisions.loc[decisions.origin.eq(dates.iloc[21]),"requested_quantity"].iloc[0]==0
    held=ledger.loc[ledger.date.eq(dates.iloc[15]),"shares"].iloc[0]
    assert np.isclose(ledger.dividend_recognized.sum(),held*.1)
    assert ledger.shares.iloc[-1]==0 and ledger.accounting_error.abs().max()<1e-6
    assert ledger.commission.sum()>0 and ledger.slippage_cost.sum()>0
