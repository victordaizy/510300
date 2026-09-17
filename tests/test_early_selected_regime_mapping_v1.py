import numpy as np,pandas as pd,pytest
from pathlib import Path
from research.early_selected_regime_mapping_inputs_v1 import PRIMARY,MODELS,CANDIDATES,early_selected_regime_mapping_frames
CFG={'candidate_models':list(CANDIDATES),'parent_models':MODELS,'state_mapping':{'上升稳定':'CORE','上升高波动':'CAPPED_SUM','压力回撤':'CAPPED_SUM','非上升':'CASH'},'decision_clock':'15:05:00','weight_band':.1,'costs':{'BASE':{},'STRESS':{}}}
def fixture():
 d=pd.read_parquet(Path('reports/research/510300_adaptive_allocation_v1/features.parquet')).iloc[:140].copy();start=d.date.iloc[125];idx=np.arange(124,len(d)-1)
 p={}
 for cost in CFG['costs']:
  p[cost]={m:pd.DataFrame({'origin':d.date.iloc[idx].to_numpy(),'execution_date':d.date.iloc[idx+1].to_numpy(),'origin_index':idx,'reference_weight':np.where(m==MODELS[0],.4,.3),'source_cost':cost,'source_model':m}) for m in MODELS}
 return d,p,start
def test_known_states_have_bounded_targets():
 d,p,s=fixture();f,_=early_selected_regime_mapping_frames(d,p,CFG,s);x=f['BASE'];v=x[PRIMARY+'_target'].dropna();assert v.between(0,1).all();assert set(x.market_state.unique())<=set(['上升稳定','上升高波动','压力回撤','非上升','未知'])
def test_non_up_is_cash_and_stable_is_core_when_present():
 d,p,s=fixture();f,_=early_selected_regime_mapping_frames(d,p,CFG,s);x=f['BASE'];a=x[MODELS[0]+'_parent_target'];assert (x.loc[x.market_state.eq('非上升'),PRIMARY+'_target']==0).all();mask=x.market_state.eq('上升稳定');assert np.allclose(x.loc[mask,PRIMARY+'_target'],a.loc[mask],equal_nan=True)
def test_mapping_change_rejected():
 d,p,s=fixture()
 with pytest.raises(ValueError):early_selected_regime_mapping_frames(d,p,{**CFG,'state_mapping':{}},s)
