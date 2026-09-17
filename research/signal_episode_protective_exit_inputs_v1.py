"""固定信号起点波动尺度的收盘保护退出；下一开盘成交由共同账户处理。"""
import numpy as np

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

PARENT='EITHER_CONFIRMED_RUNS_AUXILIARY'
MODELS=[PARENT]
PRIMARY='SIGNAL_EPISODE_PROTECTIVE_EXIT'
CANDIDATES={PRIMARY:'正信号满仓与固定三倍日波动回撤保护'}


def protective_targets(source,wealth,daily_volatility,multiple=3.):
    source,wealth,daily_volatility=(np.asarray(x,float) for x in [source,wealth,daily_volatility])
    require(source.ndim==1 and source.shape==wealth.shape==daily_volatility.shape,'信号、财富与波动长度不同')
    require(multiple==3.,'保护倍数不是事前固定值')
    require((np.isnan(source)|(np.isfinite(source)&(source>=0)&(source<=1))).all(),'来源目标越界')
    target=np.full(len(source),np.nan)
    distance=np.full(len(source),np.nan)
    peak_values=np.full(len(source),np.nan)
    starts=np.zeros(len(source),bool)
    triggers=np.zeros(len(source),bool)
    locked_values=np.zeros(len(source),bool)
    active,locked=False,False
    peak,width=np.nan,np.nan
    for t,value in enumerate(source):
        if value==0:
            target[t]=0.
            active,locked=False,False
            peak,width=np.nan,np.nan
            continue
        if not active and value>0:
            active=True
            starts[t]=True
            if np.isfinite(wealth[t]) and wealth[t]>0 and np.isfinite(daily_volatility[t]) and daily_volatility[t]>0:
                peak,width=wealth[t],multiple*daily_volatility[t]
        if not active:
            continue
        distance[t]=width
        valid_price=np.isfinite(wealth[t]) and wealth[t]>0
        if np.isfinite(peak) and valid_price:
            peak=max(peak,wealth[t])
            if not locked and wealth[t]<=peak*(1-width):
                locked=True
                triggers[t]=True
        peak_values[t]=peak
        locked_values[t]=locked
        if locked:
            target[t]=0.
        elif value>0 and np.isfinite(width) and valid_price:
            target[t]=1.
    return target,distance,peak_values,starts,triggers,locked_values


def protective_frames(data,parents_by_cost,cfg,start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['protective_multiple']==3.
            and cfg['protective_volatility_window']==20,'固定保护参数改变')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    origins=np.arange(first-1,len(data)-1)
    wealth=data.wealth.to_numpy(float)
    returns=data.total_simple.to_numpy(float)
    sigma=np.full(len(data),np.nan)
    if len(data)>=20:
        sigma[19:]=np.lib.stride_tricks.sliding_window_view(returns,20).std(axis=1,ddof=1)
    rows=[]
    for cost,f in frames.items():
        target,width,peak,starts,stops,locked=protective_targets(f[PARENT+'_parent_target'].to_numpy(float),wealth,sigma)
        usable=np.zeros(len(data),bool)
        usable[origins]=True
        target[~usable]=np.nan
        f['market_total_wealth']=wealth
        f['market_daily_volatility20']=sigma
        f['fixed_protection_distance']=width
        f['signal_episode_peak']=peak
        f['positive_episode_start']=starts
        f['protection_trigger']=stops
        f['protective_exit_locked']=locked
        f[PRIMARY+'_target']=target
        v=target[origins]
        rows.append({'model':PRIMARY,'cost':cost,'decision_origins':len(origins),
            'positive_target_origins':int((v>0).sum()),'zero_target_origins':int((v==0).sum()),
            'unknown_target_origins':int(np.isnan(v).sum()),'positive_signal_episodes':int(starts[origins].sum()),
            'protective_exit_triggers':int(stops[origins].sum()),'mean_target':float(np.nanmean(v))})
    return frames,rows
