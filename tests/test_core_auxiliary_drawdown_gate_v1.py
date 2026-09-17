import numpy as np
import pandas as pd
import pytest
from research.core_auxiliary_drawdown_gate_inputs_v1 import PRIMARY,MODELS,CANDIDATES,core_auxiliary_drawdown_gate_frames
from tests.test_runs_reference_blend_v1 import fixture
from tests.test_trend_reference_router_v1 import CFG as OLD
CFG={**OLD,'candidate_models':list(CANDIDATES),'parent_models':MODELS,'drawdown_window':60,'drawdown_gate':-.05}
def sample():
 d,p,s=fixture(); d['wealth']=np.r_[np.ones(2),np.linspace(1,1.1,8),np.linspace(1.04,1.08,8)];return d,p,s
def test_short_history_is_unknown_before_sixty_days():
 d,p,s=sample();f,x=core_auxiliary_drawdown_gate_frames(d,p,CFG,s);r=f['BASE']; assert r.market_drawdown60.isna().all();assert r[PRIMARY+'_target'].isna().all()
def test_cap_and_unknown_propagate():
 d,p,s=sample();p['BASE'][MODELS[1]].loc[0,'reference_weight']=np.nan;f,_=core_auxiliary_drawdown_gate_frames(d,p,CFG,s);assert pd.isna(f['BASE'][PRIMARY+'_target'].iloc[2]);assert f['BASE'][PRIMARY+'_target'].dropna().le(1).all()
def test_invalid_gate_rejected():
 d,p,s=sample()
 with pytest.raises(ValueError):core_auxiliary_drawdown_gate_frames(d,p,{**CFG,'drawdown_gate':-.04},s)
