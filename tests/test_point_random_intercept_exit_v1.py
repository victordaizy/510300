"""方法必要测试：两个退化端点、REML等价、时钟、未知和预测边界。"""
import copy
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from research.learned_cycle_exit_v1 import FEATURES,training_rows
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit
from research.point_random_intercept_exit_v1 import (
    KIND,RandomInterceptController,components,fit_random_intercept,random_intercept_prediction,reml_profile,solve_pooling,
)

CFG={"feature_clip":5.,"ridge_alpha":1.,"recent_cycles":20}


def sample_rows():
    rng=np.random.default_rng(7128);counts=[11,12,13,14]
    x=rng.normal(size=(sum(counts),8));x[:,3]=1.
    cycle=np.repeat(np.arange(1,5),counts)
    y=x[:,0]*.03-x[:,1]*.02+np.repeat([-.1,.04,.08,-.03],counts)+rng.normal(0,.008,len(x))
    frame=pd.DataFrame(x,columns=FEATURES)
    frame["cycle_id"],frame["origin_index"],frame["target"]=cycle,np.arange(len(x)),y
    frame["exit_index"]=cycle*100
    frame["sample_weight"]=1/frame.groupby("cycle_id").origin_index.transform("size")
    return frame


def test_zero_mean_information_matches_original_within_cycle_ridge():
    rows=sample_rows();comp=components(rows,CFG)
    model=solve_pooling(comp,CFG,0.);original=fit_within_cycle_exit(rows,CFG)
    for key in ["mean","scale","coefficients","intercept"]:
        np.testing.assert_allclose(model[key],original[key],atol=1e-12,rtol=0)


def test_zero_random_variance_matches_cycle_equal_ordinary_ridge():
    comp=components(sample_rows(),CFG);model=solve_pooling(comp,CFG,1.)
    original=Ridge(alpha=1.,fit_intercept=True,solver="svd").fit(comp["z"],comp["y"],sample_weight=comp["weights"])
    np.testing.assert_allclose(model["coefficients"],original.coef_,atol=1e-12,rtol=0)
    assert abs(model["intercept"]-original.intercept_)<1e-12


def test_compressed_reml_equals_full_covariance_reml():
    rows=sample_rows();comp=components(rows,CFG);k=.27;ratio=(1-k)/k
    covariance=np.zeros((len(rows),len(rows)))
    for _,positions in rows.groupby("cycle_id").indices.items():
        covariance[np.ix_(positions,positions)]=len(positions)*np.eye(len(positions))+ratio*np.ones((len(positions),len(positions)))
    design=np.column_stack([np.ones(len(rows)),comp["z"]@comp["basis"]])
    inv_design=np.linalg.solve(covariance,design);inv_y=np.linalg.solve(covariance,comp["y"])
    beta=np.linalg.solve(design.T@inv_design,design.T@inv_y)
    residual=comp["y"]-design@beta
    rss=float(residual@np.linalg.solve(covariance,residual));degrees=len(rows)-design.shape[1]
    dense=np.linalg.slogdet(covariance)[1]+np.linalg.slogdet(design.T@inv_design)[1]+degrees*np.log(rss/degrees)
    compressed,scale=reml_profile(comp,k)
    constant=float(np.sum(comp["counts"]*np.log(comp["counts"])))
    assert abs(dense-compressed-constant)<1e-9
    assert abs(scale-rss/degrees)<1e-12


def test_reml_unknown_new_cycle_effect_cannot_use_training_effects():
    model=fit_random_intercept(components(sample_rows(),CFG),CFG)
    assert 0<model["between_cycle_mean_weight"]<=1 and model["random_intercept_variance"]>=0
    x=np.arange(8,dtype=float)/10;x[3]=1.
    prediction=random_intercept_prediction(model,x)
    altered=copy.deepcopy(model)
    for group in altered["training_cycle_effects"]:group["conditional_training_effect"]=1e10
    assert random_intercept_prediction(altered,x)==prediction
    altered["new_cycle_random_intercept"]=.01
    with pytest.raises(ValueError):random_intercept_prediction(altered,x)


def test_appended_unmatured_future_cycle_does_not_change_current_model():
    rows=sample_rows();past,_=training_rows(rows,400,CFG)
    future=rows.copy();future["cycle_id"]+=100;future["exit_index"]+=1000;future["origin_index"]+=1000;future["target"]+=1000
    after,_=training_rows(pd.concat([rows,future],ignore_index=True),400,CFG)
    left=fit_random_intercept(components(past,CFG),CFG);right=fit_random_intercept(components(after,CFG),CFG)
    for key in ["mean","scale","coefficients","intercept","between_cycle_mean_weight"]:
        np.testing.assert_allclose(left[key],right[key],atol=1e-12,rtol=0)


def test_missing_training_information_rejects_without_deleting_rows():
    rows=sample_rows();rows.loc[0,"mom5"]=np.nan
    with pytest.raises(ValueError):components(rows,CFG)


def test_first_close_version_stays_unknown_until_next_real_cycle():
    data=pd.DataFrame({"date":pd.date_range("2020-01-01",periods=7),"mom5":0.,"mom20":0.,"sma120":0.,"vol20":.2})
    model={"kind":KIND,"features":FEATURES.copy(),"mean":[0.]*8,"scale":[1.]*8,"feature_clip":5.,
           "coefficients":[0.]*8,"intercept":-.1,"new_cycle_random_intercept":0.}
    records=[{"fit_index":1,"status":"NO_VIEW_NO_MATURE_MODEL","model":None},
             {"fit_index":4,"status":"FIT_COMPLETE","latest_exit_index":3,"model":model}]
    controller=RandomInterceptController(data,records,2)
    first={"cycle_id":1,"entry_index":2,"entry_cost_cny":1000.,"mode":1}
    assert controller(2,first,1000.,1000.)["continuation_prediction"] is None
    assert controller(4,first,1000.,1000.)["continuation_prediction"] is None
    second={"cycle_id":2,"entry_index":4,"entry_cost_cny":1000.,"mode":1}
    assert not controller(4,second,1000.,1000.)["learned_exit_requested"]
    assert controller(5,second,1000.,1000.)["learned_exit_requested"]
