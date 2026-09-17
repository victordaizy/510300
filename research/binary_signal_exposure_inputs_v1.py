"""将来源明确正目标映射为满仓，明确零映射为现金，未知继续未知。"""
import numpy as np

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

PARENT='EITHER_CONFIRMED_RUNS_AUXILIARY'
MODELS=[PARENT]
PRIMARY='BINARY_SIGNAL_EXPOSURE'
CANDIDATES={PRIMARY:'明确正信号目标满仓、零信号退出'}


def binary_target(source):
    source=np.asarray(source,float)
    require((np.isnan(source)|(np.isfinite(source)&(source>=0)&(source<=1))).all(),'来源目标越界')
    result=np.full(source.shape,np.nan)
    result[source>0]=1.
    result[source==0]=0.
    return result


def binary_frames(data,parents_by_cost,cfg,start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['target_formula']=='STRICT_POSITIVE_ONE_ZERO_ZERO_UNKNOWN_NAN','二元目标规则改变')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    origins=np.arange(first-1,len(data)-1)
    rows=[]
    for cost,f in frames.items():
        target=binary_target(f[PARENT+'_parent_target'].to_numpy(float))
        f[PRIMARY+'_target']=target
        values=target[origins]
        rows.append({'model':PRIMARY,'cost':cost,'decision_origins':len(origins),
                     'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
                     'unknown_target_origins':int(np.isnan(values).sum()),'mean_target':float(np.nanmean(values))})
    return frames,rows
