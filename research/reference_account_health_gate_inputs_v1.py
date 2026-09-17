"""读取独立来源模拟净资产，按最近收益与均线决定暂停恢复。"""
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/research/510300_account_volatility_exposure_v1'
PARENT='ACCOUNT_VOLATILITY_EXPOSURE'
MODELS=[PARENT]
PRIMARY='REFERENCE_ACCOUNT_NET_RETURN_GATE'
MEAN='REFERENCE_ACCOUNT_NAV_MEAN_GATE'
CANDIDATES={PRIMARY:'六十日净收益暂停恢复',MEAN:'六十日净资产均线暂停恢复'}


def health_targets(source,nav,first):
    source,nav=np.asarray(source,float),np.asarray(nav,float)
    require(source.ndim==1 and source.shape==nav.shape and 1<=first<len(source),'健康条件来源形状或起点不同')
    require((np.isnan(source)|(np.isfinite(source)&(source>=0)&(source<=1))).all(),'原策略目标越界')
    gates={m:np.full(len(source),np.nan) for m in CANDIDATES}
    return_gap,mean_gap=np.full(len(source),np.nan),np.full(len(source),np.nan)
    anchor=first-1
    for gate in gates.values():
        gate[anchor:min(anchor+60,len(source))]=1.
    for t in range(anchor+60,len(source)):
        full=nav[t-60:t+1]
        if (np.isfinite(full)&(full>0)).all():
            return_gap[t]=nav[t]/full[0]-1.
            gates[PRIMARY][t]=float(nav[t]>=full[0])
        window=full[1:]
        if (np.isfinite(window)&(window>0)).all():
            mean_gap[t]=np.mean(nav[t]-window)
            gates[MEAN][t]=float(mean_gap[t]>=0)
    targets={}
    for model,gate in gates.items():
        target=np.full(len(source),np.nan)
        target[gate==1]=source[gate==1]
        target[(source==0)|(gate==0)]=0.
        targets[model]=target
    return targets,gates,return_gap,mean_gap


def health_frames(data,parents_by_cost,cfg,start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['account_health_window']==60,'来源账户健康窗口改变')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    period='evaluation' if pd.Timestamp(start)==pd.Timestamp(cfg['evaluation_start']) else 'earlier_diagnostic'
    origins=np.arange(first-1,len(data)-1)
    rows=[]
    for cost,f in frames.items():
        ledger=pd.read_parquet(SOURCE/period/cost/f'{PARENT}_ledger.parquet')
        require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[first:])),'健康来源完整日期错位')
        nav=np.full(len(data),np.nan)
        nav[first-1]=cfg['initial_capital']
        nav[first:-1]=ledger.equity.iloc[:-1].to_numpy(float)
        source=f[PARENT+'_parent_target'].to_numpy(float)
        targets,gates,returns,mean_gap=health_targets(source,nav,first)
        f['reference_close_nav']=nav
        f['reference_return60']=returns
        f['reference_nav_minus_mean60']=mean_gap
        for model,target in targets.items():
            gate=gates[model]
            f[model+'_health']=gate
            f[model+'_target']=target
            values=target[origins]
            previous=np.r_[np.nan,gate[:-1]]
            rows.append({'model':model,'cost':cost,'decision_origins':len(origins),
                'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
                'unknown_target_origins':int(np.isnan(values).sum()),
                'active_source_paused_origins':int(((source[origins]>0)&(gate[origins]==0)).sum()),
                'health_pause_transitions':int(((gate[origins]==0)&(previous[origins]!=0)).sum()),
                'health_recovery_transitions':int(((gate[origins]==1)&(previous[origins]==0)).sum()),
                'mean_target':float(np.nanmean(values))})
    return frames,rows
