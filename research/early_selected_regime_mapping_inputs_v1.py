"""较早历史预先选择的四状态映射，读取两个保存父目标形成一个真实账户目标。"""
import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require
PRIMARY='EARLY_SELECTED_REGIME_MAPPING'
MODELS=['TREND_NOISE_REFERENCE_BLEND','RETURN_RUNS_STATE']
CANDIDATES={PRIMARY:'较早历史固定选择的四状态映射'}
def early_selected_regime_mapping_frames(data,parents_by_cost,cfg,start):
 require(cfg['candidate_models']==list(CANDIDATES) and cfg['state_mapping']=={'上升稳定':'CORE','上升高波动':'CAPPED_SUM','压力回撤':'CAPPED_SUM','非上升':'CASH'},'状态映射改变')
 frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start);origins=np.arange(first-1,len(data)-1);w=data.wealth.to_numpy(float);sma=np.full(len(data),np.nan);peak=np.full(len(data),np.nan)
 for t in range(119,len(data)):sma[t]=np.nanmean(w[t-119:t+1])
 for t in range(59,len(data)):peak[t]=np.nanmax(w[t-59:t+1])
 dd=w/peak-1;vol=data.vol20.to_numpy(float);state=np.full(len(data),'未知',dtype=object);known=np.isfinite(sma)&np.isfinite(dd)&np.isfinite(vol);state[known&(dd<=-.05)]='压力回撤';state[known&(dd>-.05)&(w>sma)&(vol>.2)]='上升高波动';state[known&(dd>-.05)&(w>sma)&(vol<=.2)]='上升稳定';state[known&(dd>-.05)&(w<=sma)]='非上升';out=[]
 for cost,f in frames.items():
  a=f[MODELS[0]+'_parent_target'].to_numpy(float);b=f[MODELS[1]+'_parent_target'].to_numpy(float);target=np.full(len(data),np.nan);both=np.isfinite(a)&np.isfinite(b);core=np.isfinite(a)
  target[(state=='上升稳定')&core]=a[(state=='上升稳定')&core];mask=np.isin(state,['上升高波动','压力回撤'])&both;target[mask]=np.minimum(1,a[mask]+b[mask]);target[(state=='非上升')&core]=0
  f['market_wealth']=w;f['wealth_sma120']=sma;f['market_drawdown60']=dd;f['volatility20']=vol;f['market_state']=state;f[PRIMARY+'_target']=target;cur=target[origins];out.append({'model':PRIMARY,'cost':cost,'decision_origins':len(origins),'positive_target_origins':int((cur>0).sum()),'zero_target_origins':int((cur==0).sum()),'unknown_target_origins':int(np.isnan(cur).sum()),'stable_origins':int((state[origins]=='上升稳定').sum()),'high_vol_origins':int((state[origins]=='上升高波动').sum()),'stress_origins':int((state[origins]=='压力回撤').sum()),'non_up_origins':int((state[origins]=='非上升').sum()),'mean_target':float(np.nanmean(cur))})
 return frames,out
