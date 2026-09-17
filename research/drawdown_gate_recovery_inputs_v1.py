import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require
PRIMARY='DRAWDOWN_GATE_TWO_CLOSE_RECOVERY'
MODELS=['TREND_NOISE_REFERENCE_BLEND','RETURN_RUNS_STATE']
CANDIDATES={PRIMARY:'回撤恢复连续两日后加入辅助'}
def drawdown_gate_recovery_frames(data,parents_by_cost,cfg,start):
 require(cfg['candidate_models']==list(CANDIDATES) and cfg['recovery_closes']==2 and cfg['drawdown_window']==60 and cfg['drawdown_gate']==-.05,'恢复规则改变')
 frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start); origins=np.arange(first-1,len(data)-1);w=data.wealth.to_numpy(float);p=np.full(len(data),np.nan)
 for t in range(59,len(data)):p[t]=np.nanmax(w[t-59:t+1])
 dd=w/p-1;allow=np.full(len(data),np.nan);counts=np.full(len(data),np.nan);count=0
 for t in range(len(data)):
  if not np.isfinite(dd[t]):count=0;continue
  count=count+1 if dd[t]>-.05 else 0; counts[t]=min(2,count)
  allow[t]=float(count>=2)
 out=[]
 for cost,f in frames.items():
  a=f[MODELS[0]+'_parent_target'].to_numpy(float);b=f[MODELS[1]+'_parent_target'].to_numpy(float);known=np.isfinite(a)&np.isfinite(b)&np.isfinite(allow);target=np.full(len(data),np.nan);target[known]=np.minimum(1,a[known]+np.where(allow[known]==1,b[known],0));f['market_drawdown60']=dd;f['recovery_confirmation_count']=counts;f['auxiliary_allowed']=allow;f[PRIMARY+'_target']=target;cur=target[origins];out.append({'model':PRIMARY,'cost':cost,'decision_origins':len(origins),'positive_target_origins':int((cur>0).sum()),'zero_target_origins':int((cur==0).sum()),'unknown_target_origins':int(np.isnan(cur).sum()),'auxiliary_allowed_origins':int((allow[origins]==1).sum()),'auxiliary_blocked_origins':int((allow[origins]==0).sum()),'mean_target':float(np.nanmean(cur))})
 return frames,out
