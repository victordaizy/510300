"""固定保护距离、退出前高点恢复后的重新进入资格。"""
import numpy as np

from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

PARENT='EITHER_CONFIRMED_RUNS_AUXILIARY'
BUDGET_PARENT='ACCOUNT_VOLATILITY_EXPOSURE'
MODELS=[PARENT,BUDGET_PARENT]
PRIMARY='RECOVERY_PROTECTED_ACCOUNT_BUDGET'
FULL='RECOVERY_PROTECTED_FULL_EXPOSURE'
CANDIDATES={PRIMARY:'保护恢复机制与原完整账户风险预算',FULL:'保护恢复机制与满仓预算'}


def recovery_gate(source,wealth,sigma):
    source,wealth,sigma=(np.asarray(x,float) for x in [source,wealth,sigma])
    require(source.ndim==1 and source.shape==wealth.shape==sigma.shape,'方向、财富、波动长度不同')
    require((np.isnan(source)|(np.isfinite(source)&(source>=0)&(source<=1))).all(),'来源目标越界')
    n=len(source)
    gate=np.full(n,np.nan)
    width_values=np.full(n,np.nan)
    peak_values=np.full(n,np.nan)
    recovery_values=np.full(n,np.nan)
    starts,exits,entries,paused_values=(np.zeros(n,bool) for _ in range(4))
    active,paused=False,False
    width,peak,recovery=np.nan,np.nan,np.nan
    for t,value in enumerate(source):
        if value==0:
            gate[t]=0.
            active,paused=False,False
            width,peak,recovery=np.nan,np.nan,np.nan
            continue
        valid_price=np.isfinite(wealth[t]) and wealth[t]>0
        if not active and value>0:
            active=True
            starts[t]=True
            if valid_price and np.isfinite(sigma[t]) and sigma[t]>0:
                width,peak=3.*sigma[t],wealth[t]
        if not active:
            continue
        if np.isfinite(width):
            if paused:
                if value>0 and valid_price and wealth[t]>=recovery:
                    paused=False
                    entries[t]=True
                    peak=wealth[t]
                    recovery=np.nan
            elif valid_price:
                peak=max(peak,wealth[t])
                if wealth[t]<=peak*(1.-width):
                    paused=True
                    exits[t]=True
                    recovery=peak
            if paused:
                gate[t]=0.
            elif value>0 and valid_price:
                gate[t]=1.
        width_values[t]=width
        peak_values[t]=peak
        recovery_values[t]=recovery
        paused_values[t]=paused
    return gate,width_values,peak_values,recovery_values,starts,exits,entries,paused_values


def apply_gate(gate,budget):
    gate,budget=np.asarray(gate,float),np.asarray(budget,float)
    require(gate.shape==budget.shape,'资格与预算长度不同')
    result=np.full(gate.shape,np.nan)
    result[gate==0]=0.
    known=(gate==1)&np.isfinite(budget)
    result[known]=budget[known]
    return result


def recovery_frames(data,parents_by_cost,cfg,start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['recovery_rule']=='KNOWN_POSITIVE_SOURCE_AND_PREEXIT_HIGH_RECLAIM'
            and cfg['protective_multiple']==3. and cfg['protective_volatility_window']==20,'固定恢复规则改变')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    origins=np.arange(first-1,len(data)-1)
    wealth=data.wealth.to_numpy(float)
    sigma=np.full(len(data),np.nan)
    if len(data)>=20:
        sigma[19:]=np.lib.stride_tricks.sliding_window_view(data.total_simple.to_numpy(float),20).std(axis=1,ddof=1)
    rows=[]
    for cost,f in frames.items():
        values=recovery_gate(f[PARENT+'_parent_target'].to_numpy(float),wealth,sigma)
        gate,width,peak,recovery,starts,exits,entries,paused=values
        active_origins=np.zeros(len(data),bool)
        active_origins[origins]=True
        gate[~active_origins]=np.nan
        f['market_total_wealth']=wealth
        f['market_daily_volatility20']=sigma
        for key,value in [('recovery_gate',gate),('fixed_protection_distance',width),('observation_peak',peak),
                          ('fixed_recovery_high',recovery),('positive_episode_start',starts),
                          ('protective_exit_trigger',exits),('reentry_trigger',entries),('waiting_recovery',paused)]:
            f[key]=value
        f[PRIMARY+'_target']=apply_gate(gate,f[BUDGET_PARENT+'_parent_target'].to_numpy(float))
        f[FULL+'_target']=gate
        for model in CANDIDATES:
            target=f[model+'_target'].to_numpy(float)[origins]
            rows.append({'model':model,'cost':cost,'decision_origins':len(origins),
                'positive_target_origins':int((target>0).sum()),'zero_target_origins':int((target==0).sum()),
                'unknown_target_origins':int(np.isnan(target).sum()),'positive_signal_episodes':int(starts[origins].sum()),
                'protective_exit_triggers':int(exits[origins].sum()),'reentries':int(entries[origins].sum()),
                'mean_target':float(np.nanmean(target))})
    return frames,rows
