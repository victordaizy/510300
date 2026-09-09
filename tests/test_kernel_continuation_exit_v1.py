"""核相似、周期加权截距求解、成熟时钟和实际退出检验。"""
import numpy as np
import pandas as pd
import pytest
from research.kernel_continuation_exit_inputs_v1 import FEATURES, similarities, fit_kernel_exit, kernel_prediction, KernelExitController
from research.learned_cycle_exit_v1 import training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def rows():
    rng=np.random.default_rng(87); frame=pd.DataFrame(rng.normal(size=(24,8)),columns=FEATURES)
    frame["entry_mode"]=1.;frame["cycle_id"]=np.repeat([1,2],[6,18]);frame["origin_index"]=np.arange(24)
    frame["target"]=.03*(frame.mom20**2-1)+.01*frame.cycle_return
    frame["sample_weight"]=np.repeat([1/6,1/18],[6,18]);return frame


def cfg(): return {"feature_clip":5.,"ridge_alpha":1.,"kernel_gamma":.125}


def stored(value=-.01,t=0,latest=None):
    return {"fit_index":t,"latest_exit_index":t if latest is None else latest,"status":"FIT_COMPLETE","model":{
        "kind":"WEIGHTED_RBF_CONTINUATION_WITH_INTERCEPT","features":FEATURES.copy(),"mean":[0.]*8,"scale":[1.]*8,
        "centers":[[0.]*8],"dual_coefficients":[0.],"intercept":value,"feature_clip":5.,"kernel_gamma":.125}}


def test_kernel_is_fixed_squared_distance_and_symmetric():
    x=np.array([[1.,2.,3.],[-1.,1.,2.],[0.,0.,1.]])
    actual=similarities(x,x,.125)
    expected=np.exp(-.125*np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
    np.testing.assert_allclose(actual,expected,atol=1e-15,rtol=0)
    np.testing.assert_array_equal(np.diag(actual),1.)
    assert np.linalg.eigvalsh(actual).min()>0


def test_cycle_weighted_intercept_matches_independent_block_system():
    data=rows(); model=fit_kernel_exit(data,cfg()); z=np.asarray(model["centers"]); w=data.sample_weight.to_numpy()
    kernel=np.exp(-.125*np.sum((z[:,None,:]-z[None,:,:])**2,axis=2))
    system=np.block([[kernel+np.diag(1/w),np.ones((len(z),1))],[np.ones((1,len(z))),np.zeros((1,1))]])
    solution=np.linalg.solve(system,np.r_[data.target.to_numpy(),0.])
    np.testing.assert_allclose(np.r_[model["dual_coefficients"],model["intercept"]],solution,atol=1e-12,rtol=0)
    assert model["scale"][3]==1.
    assert abs(np.sum(model["dual_coefficients"]))<1e-12


def test_response_translation_moves_only_intercept_and_keeps_kernel_predictions():
    data=rows();first=fit_kernel_exit(data,cfg());data["target"]+=.7;second=fit_kernel_exit(data,cfg())
    np.testing.assert_allclose(first["dual_coefficients"],second["dual_coefficients"],atol=1e-12,rtol=0)
    assert second["intercept"]-first["intercept"]==pytest.approx(.7)
    assert kernel_prediction(second,np.zeros(8))-kernel_prediction(first,np.zeros(8))==pytest.approx(.7)
    data.loc[0,"sample_weight"]=0
    with pytest.raises(ValueError,match="权重"):fit_kernel_exit(data,cfg())


def test_whole_cycle_maturity_and_future_fit_exclusion():
    data=rows();data["exit_index"]=np.repeat([7,30],[6,18])
    selected,ids=training_rows(data,25,{"recent_cycles":20})
    assert ids==[1] and len(selected)==6
    assert selected.sample_weight.sum()==pytest.approx(1.)
    prices=fixture()[0];cycle={"cycle_id":1,"entry_index":1,"entry_cost_cny":100000.,"mode":1}
    assert KernelExitController(prices,[stored(t=5)])(2,cycle,100000.,100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError,match="未来周期"):KernelExitController(prices,[stored(latest=5)])(2,cycle,100000.,100000.)


def test_missing_zero_and_new_cycle_reset_negative_confirmation():
    data=fixture()[0];cycle={"cycle_id":1,"entry_index":1,"entry_cost_cny":100000.,"mode":1}
    controller=KernelExitController(data,[stored(),stored(0.,t=3),stored(t=4)])
    assert controller(1,cycle,100000.,100000.)["negative_confirmation_count"]==1
    data.loc[2,"mom20"]=np.nan;missing=controller(2,cycle,100000.,100000.)
    assert missing["continuation_prediction"] is None and missing["negative_confirmation_count"]==0
    assert controller(3,cycle,100000.,100000.)["negative_confirmation_count"]==0
    assert controller(4,cycle,100000.,100000.)["negative_confirmation_count"]==1
    cycle.update(cycle_id=2,entry_index=5);assert controller(5,cycle,100000.,100000.)["negative_confirmation_count"]==1


def test_real_blocked_exit_persists_and_no_model_retains_price_exit():
    args=fixture();args[0].loc[3,"open"]=9.;args[5]["entry"][5]=0
    ledger,decisions,cycles=simulate_rearmed_exit(*args,KernelExitController(args[0],[stored(),stored(.03,t=3)]))
    assert ledger.iloc[2].status=="UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0),"date"].tolist()==[args[0].date.iloc[4],args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0),"date"].tolist()==[args[0].date.iloc[1],args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity<0
    assert ledger.accounting_error.abs().max()<1e-7 and ledger.shares.iloc[-1]==0
    args=fixture();args[5]["exit"][1][2]=True
    ledger,decisions,cycles=simulate_rearmed_exit(*args,KernelExitController(args[0],[]))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.loc[ledger.filled_quantity.lt(0),"date"].iloc[0]==args[0].date.iloc[3]
