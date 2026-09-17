import numpy as np
import pandas as pd
import pytest
from research.confirmed_auxiliary_episode_batch_inputs_v1 import PRIMARY,MODELS,CANDIDATES,confirmed_auxiliary_episode_batch_frames
from tests.test_runs_reference_blend_v1 import fixture as base_fixture
from tests.test_trend_reference_router_v1 import CFG as OLD
CFG={**OLD,'candidate_models':list(CANDIDATES),'parent_models':MODELS,'combination':'FIXED_EPISODE_START_AND_WAIT_DIRECTION_QUALIFICATIONS'}
def fixture():
 d,p,s=base_fixture(); lag=[0,0,1,0,0,1,0,0,1,0,0,1,0,0,1]; sign=[0,1,0,0,0,0,0,1,0,0,0,0,0,1,0]
 for g in p.values():
  for m,v in zip(MODELS[2:],[lag,sign]): g[m]=g[MODELS[0]].assign(source_model=m,positive_direction=v,reference_weight=np.array(v)*.2)
 return d,p,s
def test_start_and_wait_are_different():
 d,p,s=fixture(); f,_=confirmed_auxiliary_episode_batch_frames(d,p,CFG,s); a=f['BASE']; assert a[PRIMARY+'_qualified'].iloc[2]==0 and a['EPISODE_WAIT_CONFIRMED_AUXILIARY_qualified'].iloc[2]==0; assert a['EPISODE_WAIT_CONFIRMED_AUXILIARY_qualified'].iloc[3]==1
def test_qualified_holds_when_direction_turns_down():
 d,p,s=fixture(); f,_=confirmed_auxiliary_episode_batch_frames(d,p,CFG,s); a=f['BASE']; assert a[PRIMARY+'_qualified'].iloc[3]==1 and a[PRIMARY+'_qualified'].iloc[4]==1
def test_unknown_start_propagates_until_auxiliary_zero():
 d,p,s=fixture(); row=p['BASE'][MODELS[1]].index[p['BASE'][MODELS[1]].reference_weight.gt(0)][0]; origin=int(p['BASE'][MODELS[1]].loc[row,'origin_index']); p['BASE'][MODELS[2]]['positive_direction']=p['BASE'][MODELS[2]].positive_direction.astype(float); p['BASE'][MODELS[2]].loc[row,'positive_direction']=np.nan; f,_=confirmed_auxiliary_episode_batch_frames(d,p,CFG,s); assert pd.isna(f['BASE'][PRIMARY+'_target'].iloc[origin])
def test_cost_and_future_prefix():
 d,p,s=fixture(); a,_=confirmed_auxiliary_episode_batch_frames(d,p,CFG,s); q={c:{m:x.copy() for m,x in g.items()} for c,g in p.items()}; [x.__setitem__('reference_weight',0.) for g in q.values() for x in g.values()]; b,_=confirmed_auxiliary_episode_batch_frames(d,q,CFG,s); pd.testing.assert_frame_equal(a['BASE'].iloc[:2],b['BASE'].iloc[:2])
def test_invalid_direction_rejected():
 d,p,s=fixture();p['BASE'][MODELS[2]]['positive_direction']=p['BASE'][MODELS[2]].positive_direction.astype(float);p['BASE'][MODELS[2]].loc[0,'positive_direction']=.5
 with pytest.raises(ValueError): confirmed_auxiliary_episode_batch_frames(d,p,CFG,s)
