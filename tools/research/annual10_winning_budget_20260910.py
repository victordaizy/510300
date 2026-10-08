"""盈利确认后的有限仓位扩张；参考状态与实际资金账户严格分离。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.research.annual10_capital_20260910 import load_json,read_frame,aligned,accounting,FOLDERS,MODELS
from research.adaptive_allocation_v1 import normalize_dividends,summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import write_json,now,digest,require
BASE=ROOT/'research_runs/annual10_sharpe12_20260910';OUT=BASE/'winning_budget'

def episode_state(data,ledger):
    dates=pd.DatetimeIndex(data.date);profit=np.full(len(data),np.nan);sigma=profit.copy();old=0;capital=np.nan;width=np.nan
    for r in ledger.itertuples():
        t=int(dates.get_loc(r.date))
        if r.shares>0 and old==0:
            capital=r.equity-r.pnl;width=float(data.vol20.iloc[t-1]/np.sqrt(242))
        if r.shares>0:profit[t]=r.equity/capital-1.;sigma[t]=width
        else:capital=np.nan;width=np.nan
        old=r.shares
    return profit,sigma

def winning_target(parent,profit,sigma,mult,threshold,mode):
    result=np.full(len(parent),np.nan);latched=False
    for t,v in enumerate(parent):
        if not np.isfinite(v):continue
        if v==0:result[t]=0.;latched=False;continue
        barrier=0. if threshold=='ZERO' else sigma[t]
        confirmed=np.isfinite(profit[t]) and np.isfinite(barrier) and profit[t]>barrier
        if confirmed:latched=True
        increase=confirmed if mode=='CURRENT' else latched
        result[t]=1. if increase else min(1.,mult*v)
    return result

def run(period,cost_id):
    cfg=load_json(ROOT/'config/510300_continuous_reference_min_variance_v1.json');OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'protocol.json').exists():
        files=[Path(__file__),ROOT/cfg['features'],ROOT/cfg['dividends'],ROOT/'tools/research/annual10_capital_20260910.py']
        for per in ['evaluation','earlier_diagnostic']:
            for co in cfg['costs']:
                for key in ['R143','R150']:
                    files.append(ROOT/'reports/research'/FOLDERS[key]/per/co/f'{MODELS[key]}_decisions.parquet')
                    files.append(BASE/'capital'/per/co/f'{key}__FULL_SUPPORT_ledger.parquet')
                files.append(BASE/'residual'/per/co/'source_targets.parquet')
        write_json(OUT/'protocol.json',{'registered_at':now(),'remote_protocol_commit':'42eca2fd3e762677091ee5434b980ae96e0cbff2','parents':['R143','R150'],'multipliers':[2,3],
          'thresholds':['ZERO','ENTRY_SIGMA'],'modes':['CURRENT','LATCHED'],'supplements':['NONE','CLIMAX'],'policy_count':32,'account_count':128,'code_sha256':digest(Path(__file__)),
          'files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in files]},exclusive=True)
    require(load_json(OUT/'protocol.json')['code_sha256']==digest(Path(__file__)),'冻结后代码变化')
    data=read_frame(ROOT/cfg['features']);div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']));start,end=(cfg['evaluation_start'],cfg['data_cutoff']) if period=='evaluation' else (cfg['earlier_start'],cfg['earlier_terminal'])
    data=data[data.date.le(end)].copy();n=len(data);folder=OUT/period/cost_id;folder.mkdir(parents=True,exist_ok=True);require(not (folder/'metrics.csv').exists(),'完成的结果不覆盖')
    climax=read_frame(BASE/'residual'/period/cost_id/'source_targets.parquet').V1_CLIMAX_RECOVERY.to_numpy(float)
    rows=[];checks=[]
    for key in ['R143','R150']:
        parent=aligned(data,read_frame(ROOT/'reports/research'/FOLDERS[key]/period/cost_id/f'{MODELS[key]}_decisions.parquet'))
        profit,sigma=episode_state(data,read_frame(BASE/'capital'/period/cost_id/f'{key}__FULL_SUPPORT_ledger.parquet'))
        pd.DataFrame({'date':data.date,'parent':parent,'known_reference_profit':profit,'entry_sigma':sigma}).to_parquet(folder/f'{key}_known_reference.parquet',index=False)
        for mult in [2,3]:
            for threshold in ['ZERO','ENTRY_SIGMA']:
                for mode in ['CURRENT','LATCHED']:
                    core=winning_target(parent,profit,sigma,mult,threshold,mode)
                    for supplement in ['NONE','CLIMAX']:
                        target=core.copy()
                        if supplement=='CLIMAX':
                            use=np.isfinite(core)&(core==0);target[use]=climax[use]
                        model=f'{key}__X{mult}__{threshold}__{mode}__{supplement}'
                        led,dec=simulate_event_account(data,div,cfg,cfg['costs'][cost_id],start,model,targets=target,event_mask=np.ones(n,bool))
                        led.to_parquet(folder/f'{model}_ledger.parquet',index=False);dec.to_parquet(folder/f'{model}_decisions.parquet',index=False)
                        m={'period':period,'cost':cost_id,'model':model,**summarize(led,cfg)};m['point_met']=bool(m['annualized_return']>=.10 and m['net_sharpe'] is not None and m['net_sharpe']>=1.2)
                        rows.append(m);checks.append({'model':model,**accounting(led,cfg)})
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False);pd.DataFrame(checks).to_csv(folder/'checks.csv',index=False)
    print('EVAL',period,cost_id,flush=True)
    print(pd.DataFrame(rows).sort_values('annualized_return',ascending=False).head(10)[['model','annualized_return','net_sharpe','max_drawdown','point_met']].to_string(index=False),flush=True)
    print('POINT_MET',sum(r['point_met'] for r in rows),flush=True)

if __name__=='__main__':run(*sys.argv[1:])
