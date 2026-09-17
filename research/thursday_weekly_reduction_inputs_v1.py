"""仅按已经结束的判断日星期，周四决定下一开盘清仓。"""
import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

PARENT='ACCOUNT_VOLATILITY_EXPOSURE'
MODELS=[PARENT]
PRIMARY='THURSDAY_WEEKLY_REDUCTION'
CANDIDATES={PRIMARY:'周四收盘决定下一开盘清仓'}


def weekly_target(source,dates):
    source=np.asarray(source,float)
    dates=pd.DatetimeIndex(dates)
    require(source.ndim==1 and len(source)==len(dates),'周度来源与日期形状不同')
    require(not dates.hasnans and dates.is_unique and dates.is_monotonic_increasing,'判断日期无效')
    require((np.isnan(source)|(np.isfinite(source)&(source>=0)&(source<=1))).all(),'来源目标越界')
    target=source.copy()
    target[dates.dayofweek==3]=0.
    return target


def weekly_frames(data,parents_by_cost,cfg,start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['zero_origin_weekday']==3,'固定周四条件改变')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    origins=np.arange(first-1,len(data)-1)
    rows=[]
    for cost,f in frames.items():
        source=f[PARENT+'_parent_target'].to_numpy(float)
        target=weekly_target(source,data.date)
        target[:first-1]=np.nan
        target[-1]=np.nan
        f['origin_weekday']=pd.DatetimeIndex(data.date).dayofweek
        f[PRIMARY+'_target']=target
        values=target[origins]
        rows.append({'model':PRIMARY,'cost':cost,'decision_origins':len(origins),
            'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
            'unknown_target_origins':int(np.isnan(values).sum()),
            'thursday_decisions':int((f.origin_weekday.iloc[origins]==3).sum()),
            'positive_source_removed_origins':int(((source[origins]>0)&(values==0)).sum()),
            'mean_target':float(np.nanmean(values))})
    return frames,rows
