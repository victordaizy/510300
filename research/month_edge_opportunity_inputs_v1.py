"""来源收盘目标与执行日九点的既有月内两端状态组合。"""
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

ROOT=Path(__file__).resolve().parents[1]
CALENDAR_SOURCE=ROOT/'reports/research/510300_calendar_learned_equal_blend_v1'
PARENT='ACCOUNT_VOLATILITY_EXPOSURE'
MODELS=[PARENT]
PRIMARY='IDLE_MONTH_EDGE_OPPORTUNITY'
FULL='FULL_MONTH_EDGE_OPPORTUNITY'
CANDIDATES={PRIMARY:'原空仓时补充月内两端',FULL:'月内两端使用全部预算'}


def calendar_targets(source,calendar):
    source,calendar=np.asarray(source,float),np.asarray(calendar,float)
    require(source.ndim==1 and source.shape==calendar.shape,'来源与日历形状不同')
    require((np.isnan(source)|(np.isfinite(source)&(source>=0)&(source<=1))).all(),'原目标越界')
    require((np.isnan(calendar)|(calendar==0)|(calendar==1)).all(),'日历必须为零、一或未知')
    idle=source.copy()
    idle[source==0]=calendar[source==0]
    full=np.full(len(source),np.nan)
    full[calendar==0]=source[calendar==0]
    full[(calendar==1)|(source==1)]=1.
    return {PRIMARY:idle,FULL:full}


def calendar_frames(data,parents_by_cost,cfg,start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['decision_clock']=='09:00:00'
        and cfg['source_signal_clock']=='15:05:00' and cfg['month_first_sessions']==3 and cfg['month_end_natural_days']==5,'日历组合或时钟改变')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,{**cfg,'decision_clock':cfg['source_signal_clock']},start)
    period='evaluation' if pd.Timestamp(start)==pd.Timestamp(cfg['evaluation_start']) else 'earlier_diagnostic'
    saved=pd.read_parquet(CALENDAR_SOURCE/f'{period}_states.parquet')
    require(pd.DatetimeIndex(saved.market_date).equals(pd.DatetimeIndex(data.date)),'日历来源市场日期不同')
    origins=np.arange(first-1,len(data)-1)
    execution=pd.DatetimeIndex(data.date.iloc[origins+1])
    require(pd.DatetimeIndex(saved.execution_date.iloc[origins]).equals(execution),'日历执行日错位')
    require(pd.DatetimeIndex(saved.decision_time.iloc[origins]).equals(execution+pd.Timedelta(hours=9)),'日历来源不是执行日九点')
    calendar=np.full(len(data),np.nan)
    calendar[origins]=saved.calendar_state.iloc[origins].to_numpy(float)
    rows=[]
    for cost,f in frames.items():
        source=f[PARENT+'_parent_target'].to_numpy(float)
        targets=calendar_targets(source,calendar)
        f['parent_signal_time']=f.decision_time.copy()
        f['decision_time']=pd.NaT
        f.loc[origins,'decision_time']=execution+pd.Timedelta(hours=9)
        f['calendar_state']=calendar
        for model,target in targets.items():
            f[model+'_target']=target
            values=target[origins]
            rows.append({'model':model,'cost':cost,'decision_origins':len(origins),
                'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
                'unknown_target_origins':int(np.isnan(values).sum()),
                'calendar_positive_origins':int((calendar[origins]==1).sum()),
                'supplemented_idle_origins':int(((source[origins]==0)&(values>0)).sum()),
                'positive_parent_increased_origins':int(((source[origins]>0)&(values>source[origins])).sum()),
                'mean_target':float(np.nanmean(values))})
    return frames,rows
