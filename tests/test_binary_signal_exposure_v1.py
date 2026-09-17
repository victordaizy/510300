import numpy as np
import pandas as pd
import pytest

from research.binary_signal_exposure_inputs_v1 import CANDIDATES,PARENT,PRIMARY,binary_frames,binary_target


def fixture():
    d=pd.DataFrame({'date':pd.bdate_range('2020-01-01',periods=7)})
    idx=np.arange(1,6)
    c={'candidate_models':list(CANDIDATES),'target_formula':'STRICT_POSITIVE_ONE_ZERO_ZERO_UNKNOWN_NAN',
       'decision_clock':'15:05:00','weight_band':.1,'costs':{'BASE':{},'STRESS':{}}}
    sources={cost:{PARENT:pd.DataFrame({'origin_index':idx,'origin':d.date.iloc[idx].to_numpy(),
        'execution_date':d.date.iloc[idx+1].to_numpy(),'source_cost':cost,'source_model':PARENT,
        'reference_weight':[0.,1e-15,np.nan,1.,0.]})} for cost in c['costs']}
    return d,sources,c,str(d.date.iloc[2].date())


def test_strict_positive_zero_and_unknown():
    np.testing.assert_allclose(binary_target([0.,1e-15,.3,1.,np.nan]),[0,1,1,1,np.nan],equal_nan=True)
    with pytest.raises(ValueError):
        binary_target([np.inf])


def test_future_and_other_cost_are_isolated():
    d,p,c,s=fixture()
    first,_=binary_frames(d,p,c,s)
    p['BASE'][PARENT].loc[4,'reference_weight']=.01
    changed,_=binary_frames(d,p,c,s)
    np.testing.assert_allclose(first['BASE'][PRIMARY+'_target'].iloc[:5],changed['BASE'][PRIMARY+'_target'].iloc[:5],equal_nan=True)
    np.testing.assert_allclose(first['STRESS'][PRIMARY+'_target'],changed['STRESS'][PRIMARY+'_target'],equal_nan=True)
    assert changed['BASE'][PRIMARY+'_target'].iloc[[0,3,6]].isna().all()


def test_wrong_source_clock_is_rejected():
    d,p,c,s=fixture()
    p['BASE'][PARENT].loc[0,'execution_date']=d.date.iloc[1]
    with pytest.raises(ValueError):
        binary_frames(d,p,c,s)
