"""将原181目标与独立均值反弹参考目标按预定比例相加封顶。"""
import numpy as np

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

CORE='ACCOUNT_VOLATILITY_EXPOSURE'
AUX='R2_Z_CONFIRM'
MODELS=[CORE,AUX]
PRIMARY='MEAN_REBOUND_AUX_25'
WEIGHTS={'MEAN_REBOUND_AUX_25':.25,'MEAN_REBOUND_AUX_50':.5}
CANDIDATES={'MEAN_REBOUND_AUX_25':'原181加四分之一均值反弹','MEAN_REBOUND_AUX_50':'原181加二分之一均值反弹'}


def combined_targets(core,aux,cfg):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['auxiliary_weights']==WEIGHTS,'固定两档辅助比例不同')
    core,aux=np.asarray(core,float),np.asarray(aux,float)
    require(core.ndim==1 and core.shape==aux.shape,'来源目标形状不同')
    for values in [core,aux]:
        require((np.isnan(values)|(np.isfinite(values)&(values>=0)&(values<=1))).all(),'来源目标越界')
    return {model:np.minimum(1.,core+weight*aux) for model,weight in WEIGHTS.items()}


def rebound_frames(data,parents_by_cost,cfg,start):
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    origins=np.arange(first-1,len(data)-1)
    summaries=[]
    for cost,frame in frames.items():
        core=frame[CORE+'_parent_target'].to_numpy(float)
        aux=frame[AUX+'_parent_target'].to_numpy(float)
        for model,target in combined_targets(core,aux,cfg).items():
            frame[model+'_target']=target
            values=target[origins]
            summaries.append({'model':model,'cost':cost,'decision_origins':len(origins),
                'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
                'unknown_target_origins':int(np.isnan(values).sum()),'full_target_origins':int((values==1).sum()),
                'core_idle_auxiliary_positive_origins':int(((core[origins]==0)&(aux[origins]>0)).sum()),
                'both_positive_origins':int(((core[origins]>0)&(aux[origins]>0)).sum()),'mean_target':float(np.nanmean(values))})
    return frames,summaries
