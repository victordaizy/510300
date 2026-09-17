"""原空仓日历补充需获得前一收盘二十日含分红方向确认。"""
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

ROOT=Path(__file__).resolve().parents[1]
CALENDAR_SOURCE=ROOT/'reports/research/510300_calendar_learned_equal_blend_v1'
PARENT='ACCOUNT_VOLATILITY_EXPOSURE'
MODELS=[PARENT]
PRIMARY='CONFIRMED_MONTH_EDGE_IDLE'
CANDIDATES={PRIMARY:'二十日上涨确认的原空仓日历补充'}


def confirmation_state(wealth):
    wealth=np.asarray(wealth,float)
    require(wealth.ndim==1,'含分红财富必须为一维')
    result=np.full(len(wealth),np.nan)
    if len(wealth)>20:
        left,right=wealth[:-20],wealth[20:]
        valid=np.isfinite(left)&np.isfinite(right)&(left>0)&(right>0)
        ratio=np.full(len(left),np.nan)
        np.divide(right,left,out=ratio,where=valid)
        result[20:]=np.where(valid,(ratio>1.+1e-12).astype(float),np.nan)
    return result


def confirmed_target(source,calendar,confirmation):
    source,calendar,confirmation=(np.asarray(x,float) for x in [source,calendar,confirmation])
    require(source.ndim==1 and source.shape==calendar.shape==confirmation.shape,'确认组合形状不同')
    require((np.isnan(source)|(np.isfinite(source)&(source>=0)&(source<=1))).all(),'原目标越界')
    for values in [calendar,confirmation]:
        require((np.isnan(values)|(values==0)|(values==1)).all(),'进入条件必须为零、一或未知')
    supplement=np.full(len(source),np.nan)
    supplement[(calendar==0)|(confirmation==0)]=0.
    supplement[(calendar==1)&(confirmation==1)]=1.
    target=source.copy()
    target[source==0]=supplement[source==0]
    return target


def confirmed_frames(data,parents_by_cost,cfg,start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['decision_clock']=='09:00:00'
        and cfg['source_signal_clock']=='15:05:00' and cfg['confirmation_window']==20
        and cfg['confirmation_rounding_tolerance']==1e-12 and cfg['month_first_sessions']==3
        and cfg['month_end_natural_days']==5,'确认组合时钟或窗口改变')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,{**cfg,'decision_clock':cfg['source_signal_clock']},start)
    period='evaluation' if pd.Timestamp(start)==pd.Timestamp(cfg['evaluation_start']) else 'earlier_diagnostic'
    saved=pd.read_parquet(CALENDAR_SOURCE/f'{period}_states.parquet')
    require(pd.DatetimeIndex(saved.market_date).equals(pd.DatetimeIndex(data.date)),'日历来源市场日不同')
    origins=np.arange(first-1,len(data)-1)
    execution=pd.DatetimeIndex(data.date.iloc[origins+1])
    require(pd.DatetimeIndex(saved.execution_date.iloc[origins]).equals(execution),'日历执行日错位')
    require(pd.DatetimeIndex(saved.decision_time.iloc[origins]).equals(execution+pd.Timedelta(hours=9)),'日历来源时钟不同')
    calendar=np.full(len(data),np.nan)
    calendar[origins]=saved.calendar_state.iloc[origins].to_numpy(float)
    confirm=confirmation_state(data.wealth)
    confirm[:first-1]=np.nan
    confirm[-1]=np.nan
    rows=[]
    for cost,f in frames.items():
        source=f[PARENT+'_parent_target'].to_numpy(float)
        target=confirmed_target(source,calendar,confirm)
        f['parent_signal_time']=f.decision_time.copy()
        f['decision_time']=pd.NaT
        f.loc[origins,'decision_time']=execution+pd.Timedelta(hours=9)
        f['calendar_state']=calendar
        f['prior_close_confirmation20']=confirm
        f[PRIMARY+'_target']=target
        values=target[origins]
        rows.append({'model':PRIMARY,'cost':cost,'decision_origins':len(origins),
            'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
            'unknown_target_origins':int(np.isnan(values).sum()),
            'supplemented_idle_origins':int(((source[origins]==0)&(values>0)).sum()),
            'calendar_idle_rejected_origins':int(((source[origins]==0)&(calendar[origins]==1)&(confirm[origins]==0)).sum()),
            'positive_parent_changed_origins':int(((source[origins]>0)&(values!=source[origins])).sum()),
            'mean_target':float(np.nanmean(values))})
    return frames,rows
