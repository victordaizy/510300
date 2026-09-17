"""正资格采用市场波动预算，比较逐日目标与首次正资格固定目标。"""
import numpy as np

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

PARENT='EITHER_CONFIRMED_RUNS_AUXILIARY'
MODELS=[PARENT]
PRIMARY='SIGNAL_MARKET_RISK_DAILY'
EPISODE='SIGNAL_MARKET_RISK_EPISODE'
CANDIDATES={PRIMARY:'每日市场风险仓位',EPISODE:'区间固定市场风险仓位'}


def market_targets(source,volatility):
    source,volatility=np.asarray(source,float),np.asarray(volatility,float)
    require(source.ndim==1 and source.shape==volatility.shape,'来源与市场波动形状不同')
    require((np.isnan(source)|(np.isfinite(source)&(source>=0)&(source<=1))).all(),'来源目标越界')
    current=np.full(len(source),np.nan)
    valid=np.isfinite(volatility)&(volatility>=0)
    current[valid&(volatility<=.1)]=1.
    larger=valid&(volatility>.1)
    current[larger]=.1/volatility[larger]
    daily=np.full(len(source),np.nan)
    daily[source==0]=0.
    daily[source>0]=current[source>0]
    episode=np.full(len(source),np.nan)
    frozen=np.full(len(source),np.nan)
    starts=np.zeros(len(source),bool)
    active=False
    selected=np.nan
    for t,value in enumerate(source):
        if np.isnan(value):
            if active:
                frozen[t]=selected
            continue
        if value==0:
            episode[t]=0.
            active,selected=False,np.nan
            continue
        if not active:
            active,selected=True,current[t]
            starts[t]=True
        frozen[t]=selected
        episode[t]=selected
    return {PRIMARY:daily,EPISODE:episode},current,frozen,starts


def market_frames(data,parents_by_cost,cfg,start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['market_volatility_window']==60 and cfg['market_risk_budget']==.1,'市场波动预算改变')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    origins=np.arange(first-1,len(data)-1)
    rows=[]
    for cost,f in frames.items():
        source=f[PARENT+'_parent_target'].to_numpy(float)
        targets,current,frozen,starts=market_targets(source,data.vol60.to_numpy(float))
        f['market_volatility60']=data.vol60.to_numpy(float)
        f['current_market_budget']=current
        f['frozen_episode_budget']=frozen
        f['positive_episode_start']=starts
        for model,target in targets.items():
            f[model+'_target']=target
            values=target[origins]
            rows.append({'model':model,'cost':cost,'decision_origins':len(origins),
                'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
                'unknown_target_origins':int(np.isnan(values).sum()),'full_target_origins':int((values==1).sum()),
                'positive_signal_episodes':int(starts[origins].sum()),'mean_target':float(np.nanmean(values))})
    return frames,rows
