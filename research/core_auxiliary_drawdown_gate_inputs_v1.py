"""市场六十日回撤只控制辅助仓位，核心目标保持原有来源。"""
import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require
PRIMARY='CORE_AUXILIARY_DRAWDOWN_GATE'
MODELS=['TREND_NOISE_REFERENCE_BLEND','RETURN_RUNS_STATE']
CANDIDATES={PRIMARY:'六十日市场回撤撤去辅助仓位'}
def core_auxiliary_drawdown_gate_frames(data,parents_by_cost,cfg,start):
 require(cfg['candidate_models']==list(CANDIDATES) and cfg['parent_models']==MODELS and cfg['drawdown_window']==60 and cfg['drawdown_gate']==-.05,'回撤分层规则改变')
 frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start); origins=np.arange(first-1,len(data)-1); summaries=[]
 wealth=data.wealth.to_numpy(float); peak=np.full(len(data),np.nan)
 for t in range(59,len(data)): peak[t]=np.nanmax(wealth[t-59:t+1])
 dd=wealth/peak-1
 for cost,f in frames.items():
  core=f[MODELS[0]+'_parent_target'].to_numpy(float);aux=f[MODELS[1]+'_parent_target'].to_numpy(float); known=np.isfinite(core)&np.isfinite(aux)&np.isfinite(dd); allowed=dd>cfg['drawdown_gate']; target=np.full(len(data),np.nan); effective=np.full(len(data),np.nan); effective[known]=0;effective[known&allowed]=aux[known&allowed];target[known]=np.minimum(1,core[known]+effective[known]);f['market_wealth']=wealth;f['market_peak60']=peak;f['market_drawdown60']=dd;f['auxiliary_allowed']=np.where(known,allowed.astype(float),np.nan);f[PRIMARY+'_effective_auxiliary_target']=effective;f[PRIMARY+'_target']=target; cur=target[origins];summaries.append({'model':PRIMARY,'cost':cost,'decision_origins':len(origins),'positive_target_origins':int((cur>0).sum()),'zero_target_origins':int((cur==0).sum()),'unknown_target_origins':int(np.isnan(cur).sum()),'auxiliary_allowed_origins':int((allowed[origins]&known[origins]).sum()),'auxiliary_blocked_origins':int(((~allowed[origins])&known[origins]).sum()),'full_weight_origins':int((cur==1).sum()),'mean_target':float(np.nanmean(cur))})
 return frames,summaries
