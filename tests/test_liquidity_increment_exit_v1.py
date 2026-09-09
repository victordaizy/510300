"""检验流动性过去时钟、单位、九项重拟合及真实退出。"""
import numpy as np
import pandas as pd
import pytest
from research.liquidity_increment_exit_inputs_v1 import FEATURES, LIQUIDITY, liquidity_features, attach_sample_liquidity, fit_liquidity_exit, liquidity_prediction, LiquidityExitController
from research.learned_cycle_exit_v1 import training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def prices(n=65):
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "amount": np.arange(n)*1000000.+100000000., "amount_unit": "CNY", "total_simple": np.linspace(-.03, .02, n)})


def stored(value=-.01, t=0, latest=None):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE",
        "model": {"kind": "LIQUIDITY_INCREMENT_RIDGE", "features": FEATURES.copy(), "mean": [0.]*9, "scale": [1.]*9, "coefficients": [0.]*9, "intercept": value, "feature_clip": 5.}}


def account_fixture():
    args=fixture(); args[0][LIQUIDITY]=-.8; return args


def test_factor_formula_past_prefix_and_wrong_units():
    data=prices(); actual=liquidity_features(data)
    for t in range(19,len(data)):
        rows=data.iloc[t-19:t+1]; expected=np.log(np.mean(np.abs(rows.total_simple.to_numpy())/(rows.amount.to_numpy()/1e8)))
        assert actual[LIQUIDITY].iloc[t] == pytest.approx(expected, abs=1e-12)
    assert actual[LIQUIDITY].iloc[:19].isna().all()
    changed=data.copy(); changed.loc[40:,"amount"]*=1000
    pd.testing.assert_frame_equal(actual.iloc[:40],liquidity_features(changed).iloc[:40])
    pd.testing.assert_frame_equal(actual.iloc[:40],liquidity_features(data.iloc[:40]))
    data["amount_unit"]="THOUSAND_CNY"
    with pytest.raises(ValueError, match="人民币元"): liquidity_features(data)


def test_missing_or_zero_window_is_not_zero_or_partial_mean():
    data=prices(); data.loc[25,"amount"]=0; actual=liquidity_features(data)
    assert actual[LIQUIDITY].iloc[25:45].isna().all() and np.isfinite(actual[LIQUIDITY].iloc[45])
    data=prices();data["total_simple"]=0;assert liquidity_features(data)[LIQUIDITY].isna().all()
    data=prices();data.loc[25,"total_simple"]=np.nan;assert liquidity_features(data)[LIQUIDITY].iloc[25:45].isna().all()


def test_samples_preserve_all_original_rows_and_whole_cycle_weights():
    data=prices();data[LIQUIDITY]=liquidity_features(data)[LIQUIDITY]
    s=pd.DataFrame({"cycle_id":[1,1,2],"origin_index":[10,20,25],"origin":data.date.iloc[[10,20,25]].to_numpy(),"exit_index":[21,21,50]})
    out=attach_sample_liquidity(s,data);pd.testing.assert_frame_equal(out.drop(columns=LIQUIDITY),s)
    rows,ids=training_rows(out,30,{"recent_cycles":20});assert ids==[1] and len(rows)==2 and rows[LIQUIDITY].isna().sum()==1
    np.testing.assert_allclose(rows.sample_weight,[.5,.5])
    s.loc[0,"origin"]=data.date.iloc[9]
    with pytest.raises(ValueError,match="日期和行情"):attach_sample_liquidity(s,data)


def test_ninth_factor_is_fitted_and_weighted_solution_matches_normal_equation():
    rng=np.random.default_rng(86); rows=pd.DataFrame(rng.normal(size=(80,9)),columns=FEATURES)
    rows["entry_mode"]=1.;rows["target"]=.04*rows[LIQUIDITY]+.02*rows.cycle_return
    rows["sample_weight"]=np.repeat([1/20,1/60],[20,60]);cfg={"feature_clip":5.,"ridge_alpha":1.}
    model=fit_liquidity_exit(rows,cfg)
    x=np.clip((rows[FEATURES].to_numpy()-model["mean"])/model["scale"],-5,5); a=np.c_[np.ones(len(x)),x]
    penalty=np.diag([0.]+[1.]*9);w=rows.sample_weight.to_numpy()
    coef=np.linalg.solve(a.T@(w[:,None]*a)+penalty,a.T@(w*rows.target.to_numpy()))
    np.testing.assert_allclose([model["intercept"]]+model["coefficients"],coef,atol=1e-12,rtol=0)
    assert model["scale"][3]==1 and abs(model["coefficients"][-1])>.001
    v=np.zeros(9);pred=liquidity_prediction(model,v);v[-1]=1.;assert liquidity_prediction(model,v)!=pred


def test_missing_input_future_models_and_maturity():
    data=account_fixture()[0];cycle={"cycle_id":1,"entry_index":1,"entry_cost_cny":100000.,"mode":1}
    controller=LiquidityExitController(data,[stored()]);assert controller(1,cycle,100000.,100000.)["negative_confirmation_count"]==1
    data.loc[2,LIQUIDITY]=np.nan;missing=controller(2,cycle,100000.,100000.)
    assert missing["continuation_prediction"] is None and missing["negative_confirmation_count"]==0
    assert LiquidityExitController(data,[stored(t=5)])(3,cycle,100000.,100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError,match="未来周期"):LiquidityExitController(data,[stored(latest=5)])(3,cycle,100000.,100000.)


def test_actual_exit_persists_after_blocked_fill_and_positive_prediction():
    args=account_fixture();args[0].loc[3,"open"]=9.;args[5]["entry"][5]=0
    ledger,decisions,cycles=simulate_rearmed_exit(*args,LiquidityExitController(args[0],[stored(),stored(.03,t=3)]))
    assert ledger.iloc[2].status=="UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0),"date"].tolist()==[args[0].date.iloc[4],args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0),"date"].tolist()==[args[0].date.iloc[1],args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity<0
    assert ledger.accounting_error.abs().max()<1e-7 and ledger.shares.iloc[-1]==0


def test_no_model_retains_price_exit_and_new_actual_cycle_resets_count():
    args=account_fixture();args[5]["exit"][1][2]=True
    ledger,decisions,cycles=simulate_rearmed_exit(*args,LiquidityExitController(args[0],[]))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.loc[ledger.filled_quantity.lt(0),"date"].iloc[0]==args[0].date.iloc[3]
    cycle={"cycle_id":1,"entry_index":1,"entry_cost_cny":100000.,"mode":1}; c=LiquidityExitController(args[0],[stored()])
    assert c(1,cycle,100000.,100000.)["negative_confirmation_count"]==1
    assert c(2,cycle,100000.,100000.)["negative_confirmation_count"]==2
    cycle.update(cycle_id=2,entry_index=4);assert c(4,cycle,100000.,100000.)["negative_confirmation_count"]==1
