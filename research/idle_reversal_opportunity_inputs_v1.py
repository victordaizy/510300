"""原策略优先，仅在明确空仓时使用保存的反弹状态。"""
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

PARENT='ACCOUNT_VOLATILITY_EXPOSURE'
MODELS=[PARENT]
PRIMARY='IDLE_RSI2_OPPORTUNITY'
TREND='IDLE_TREND_RSI2_OPPORTUNITY'
CANDIDATES={PRIMARY:'空仓补充两日超跌',TREND:'空仓补充上涨趋势内两日超跌'}
SIGNALS={PRIMARY:'R09_RSI2',TREND:'R10_RSI2_TREND'}


def priority_target(source,reversal):
    source,reversal=np.asarray(source,float),np.asarray(reversal,float)
    require(source.ndim==1 and source.shape==reversal.shape,'原目标与补充状态形状不同')
    require((np.isnan(source)|(np.isfinite(source)&(source>=0)&(source<=1))).all(),'原目标越界')
    require((np.isnan(reversal)|(reversal==0)|(reversal==1)).all(),'反弹状态必须为零、一或未知')
    target=np.full(source.shape,np.nan)
    active=source>0
    target[active]=source[active]
    idle=source==0
    target[idle]=reversal[idle]
    return target


def idle_frames(data,parents_by_cost,cfg,start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['target_formula']=='PARENT_POSITIVE_PRIORITY_ELSE_KNOWN_ZERO_REVERSAL','空仓补充规则改变')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    saved=pd.read_parquet(Path(__file__).resolve().parents[1]/cfg['saved_reversal_targets'])
    saved=saved.loc[saved.date.le(data.date.iloc[-1])].reset_index(drop=True)
    require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(data.date)),'反弹保存日历不同')
    origins=np.arange(first-1,len(data)-1)
    valid=data.feature_valid.fillna(False).to_numpy(bool)
    rows=[]
    for cost,f in frames.items():
        source=f[PARENT+'_parent_target'].to_numpy(float)
        for model,signal in SIGNALS.items():
            reversal=saved[signal].to_numpy(float).copy()
            reversal[~valid]=np.nan
            target=priority_target(source,reversal)
            f[signal+'_saved_state']=reversal
            f[model+'_target']=target
            values=target[origins]
            added=(source[origins]==0)&(reversal[origins]>0)
            rows.append({'model':model,'cost':cost,'decision_origins':len(origins),
                'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
                'unknown_target_origins':int(np.isnan(values).sum()),'supplemented_idle_origins':int(added.sum()),
                'parent_positive_origins':int((source[origins]>0).sum()),'mean_target':float(np.nanmean(values))})
    return frames,rows
