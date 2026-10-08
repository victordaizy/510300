"""只用现有固定规则的边际资金机会；所有176种组合都保存，不隐藏失败。"""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.research.annual10_capital_20260910 import load_json,read_frame,aligned,accounting,FOLDERS,MODELS
from research.adaptive_allocation_v1 import normalize_dividends,summarize
from research.event_clock_account_v1 import simulate_event_account
from research.simple_price_entry_exit_v1 import signals, specifications,simulate_policy
from research.simple_volume_reversal_v1 import make_rules,specifications as vspecifications
from research.intraday_overnight_increment_v1 import digest,now,require,write_json
OUT=ROOT/'research_runs/annual10_sharpe12_20260910/residual'
CNAMES=['CAL_MONTH_FIRST3','CAL_MONTH_EDGE','CAL_QUARTER_FIRST3','CAL_POST_BREAK3']

def calendar_targets(data):
    dates=pd.DatetimeIndex(data.date);n=len(dates)
    ordinal=np.zeros(n,int);postbreak=np.full(n,100000,int);seen=100000
    for i,d in enumerate(dates):
        ordinal[i]=1 if i==0 or d.to_period('M')!=dates[i-1].to_period('M') else ordinal[i-1]+1
        if i and (d-dates[i-1]).days>=4:seen=0
        postbreak[i]=seen;seen+=1
    remaining=(dates.to_period('M').end_time.normalize()-dates).days
    flags=[ordinal<=3,(ordinal<=3)|(remaining<5),np.isin(dates.month,[1,4,7,10])&(ordinal<=3),postbreak<3]
    return {key:np.r_[v[1:].astype(float),np.nan] for key,v in zip(CNAMES,flags)}

def residual_target(core,sleeve,mode,a):
    require(mode in ['IDLE_ONLY','REMAINING'] and a in [.5,1.],'非登记组合')
    known=np.isfinite(core)&np.isfinite(sleeve);out=np.full(len(core),np.nan)
    if mode=='IDLE_ONLY':out[known]=np.where(core[known]==0,a*sleeve[known],core[known])
    else:out[known]=core[known]+(1-core[known])*a*sleeve[known]
    require((np.isnan(out)|((out>=0)&(out<=1))).all(),'组合超范围')
    return out

def run(period,cost_id):
    cfg=load_json(ROOT/'config/510300_continuous_reference_min_variance_v1.json')
    p=OUT/'protocol.json';OUT.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        used=[Path(__file__),ROOT/'tools/research/annual10_capital_20260910.py',ROOT/cfg['features'],ROOT/cfg['dividends'],
            ROOT/'research/simple_price_entry_exit_v1.py',ROOT/'research/simple_volume_reversal_v1.py',ROOT/'research/event_clock_account_v1.py']
        write_json(p,{'registered_at':now(),'protocol_commit':'c2b9441c71a9dca567c6ba5ad93b4a729a68878c',
            'sources':list(specifications())+list(vspecifications())+CNAMES,'core_multipliers':[2,3],'allocations':[.5,1.],
            'modes':['IDLE_ONLY','REMAINING'],'combinations':176,'accounts':704,'reference_accounts':88,
            'code_sha256':digest(Path(__file__)),'files':[{'path':str(f.relative_to(ROOT)),'sha256':digest(f)} for f in used]},exclusive=True)
    require(load_json(p)['code_sha256']==digest(Path(__file__)),'登记代码已变动')
    data=read_frame(ROOT/cfg['features']);div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    start,end=(cfg['evaluation_start'],cfg['data_cutoff']) if period=='evaluation' else (cfg['earlier_start'],cfg['earlier_terminal'])
    data=data[data.date.le(end)].copy();cost=cfg['costs'][cost_id];first=int(np.flatnonzero(data.date>=start)[0])
    folder=OUT/period/cost_id;require(not (folder/'metrics.csv').exists(),'该子批已完成不得覆盖');folder.mkdir(parents=True,exist_ok=True)
    old=read_frame(ROOT/'reports/research'/FOLDERS['R150']/period/cost_id/f"{MODELS['R150']}_decisions.parquet")
    parent=aligned(data,old)
    rules={**signals(data),**make_rules(data)[0]};specs={**specifications(),**vspecifications()}
    states={};references=[]
    for key in rules:
        led,dec,cycles=simulate_policy(data,div,cfg,cost,start,rules[key],specs[key])
        sdir=folder/'sources';sdir.mkdir(exist_ok=True)
        led.to_parquet(sdir/f'{key}_ledger.parquet',index=False);dec.to_parquet(sdir/f'{key}_decisions.parquet',index=False)
        cycles.to_csv(sdir/f'{key}_cycles.csv',index=False)
        states[key]=aligned(data,dec);references.append({'model':key,**summarize(led,cfg),**accounting(led,cfg)})
    for key,tar in calendar_targets(data).items():
        tar[:first-1]=np.nan
        led,dec=simulate_event_account(data,div,cfg,cost,start,key,targets=tar,event_mask=np.ones(len(data),bool))
        led.to_parquet(folder/'sources'/f'{key}_ledger.parquet',index=False);dec.to_parquet(folder/'sources'/f'{key}_decisions.parquet',index=False)
        states[key]=aligned(data,dec);references.append({'model':key,**summarize(led,cfg),**accounting(led,cfg)})
    pd.DataFrame(references).to_csv(folder/'reference_metrics.csv',index=False)
    rows=[];checks=[]
    targets=pd.DataFrame({'date':data.date,'parent150':parent,**states})
    targets.to_parquet(folder/'source_targets.parquet',index=False)
    for k in [2,3]:
        core=np.minimum(1.,k*parent)
        for key,sleeve in states.items():
            for mode in ['IDLE_ONLY','REMAINING']:
                for a in [.5,1.]:
                    model=f'CORE{k}__{key}__{mode}__A{int(a*100)}'
                    target=residual_target(core,sleeve,mode,a)
                    led,dec=simulate_event_account(data,div,cfg,cost,start,model,targets=target,event_mask=np.ones(len(data),bool))
                    m={'period':period,'cost':cost_id,'model':model,'sleeve':key,'core_multiplier':k,'mode':mode,'allocation':a,**summarize(led,cfg)}
                    m['point_met']=bool(m['annualized_return']>=.10 and m['net_sharpe'] is not None and m['net_sharpe']>=1.2)
                    rows.append(m);checks.append({'model':model,**accounting(led,cfg)})
                    led.to_parquet(folder/f'{model}_ledger.parquet',index=False);dec.to_parquet(folder/f'{model}_decisions.parquet',index=False)
        print('DONE',period,cost_id,'CORE',k,'accounts',len(rows),flush=True)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False);pd.DataFrame(checks).to_csv(folder/'checks.csv',index=False)
    print(pd.DataFrame(rows).sort_values('net_sharpe',ascending=False).head(10)[['model','annualized_return','net_sharpe','max_drawdown','point_met']].to_string(index=False),flush=True)
    print('POINT_MET',sum(r['point_met'] for r in rows),flush=True)

if __name__=='__main__':run(*sys.argv[1:])
