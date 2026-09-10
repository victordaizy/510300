"""正向创新和序贯证据的固定机会研究；只做历史模拟，不连接交易账户。"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.research.annual10_capital_20260910 import read_frame,load_json,aligned,accounting
from research.adaptive_allocation_v1 import normalize_dividends,summarize
from research.simple_price_entry_exit_v1 import simulate_policy
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import require,write_json,digest,now
OUT=ROOT/'research_runs/annual10_sequential_20260910'
CORE='CORE2__V1_CLIMAX_RECOVERY__IDLE_ONLY__A100'
OLD=ROOT/'research_runs/annual10_sharpe12_20260910/residual'
RULES=[(event,confirm,h) for event in ['P','C'] for confirm in ['IMMEDIATE','CONFIRM_NEXT'] for h in [5,10,20]]
SIZES=['FULL','VOL10']
MODES=['STANDALONE','MAX','REMAINING']

def inputs():
    cfg=load_json(ROOT/'config/510300_continuous_reference_min_variance_v1.json')
    data=read_frame(ROOT/cfg['features'])
    return cfg,data,normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))

def events(data):
    log=data.total_log.astype(float)
    prior_sigma=log.shift(1).rolling(60,min_periods=60).std(ddof=1)
    z=log/prior_sigma.where(prior_sigma>0)
    vr=data.volume/data.volume.shift(1).rolling(20,min_periods=20).mean()
    clv=(data.close-data.low)/(data.high-data.low).replace(0,np.nan)
    valid=np.isfinite(np.column_stack([z,vr,clv])).all(axis=1)&data.feature_valid.to_numpy(bool)
    p=valid & (z.to_numpy()>=2.) & (vr.to_numpy()>=1.5)&(clv.to_numpy()>=.75)
    c=np.zeros(len(data),bool);stat=np.zeros(len(data));s=0.
    for t in range(len(data)):
        if not valid[t]:s=0.;continue
        s=max(0.,s+float(z.iloc[t])-.5);stat[t]=s
        if s>=5.:
            c[t]=(vr.iloc[t]>=1.5) and (clv.iloc[t]>=.75)
            s=0.
    result={}
    for key,event in [('P',p),('C',c)]:
        result[(key,'IMMEDIATE')]=event.astype(int)
        previous=np.r_[False,event[:-1]]
        result[(key,'CONFIRM_NEXT')]=(previous & valid & (log.to_numpy()>0)&(clv.to_numpy()>=.5)).astype(int)
    factors=pd.DataFrame({'date':data.date,'z':z,'prior_sigma':prior_sigma,'volume_ratio':vr,'close_location':clv,'cusum_before_reset':stat,'valid':valid})
    return result,factors

def sleeve_size(state,vol,size):
    out=np.full(len(state),np.nan);active=False;amount=0.
    for t,v in enumerate(state):
        if not np.isfinite(v):continue
        if v==0:active=False;amount=0.
        elif not active:
            if not np.isfinite(vol[t]) or vol[t]<=0:continue
            active=True;amount=1. if size=='FULL' else min(1.,.1/vol[t])
        out[t]=amount
    return out

def combine(core,sleeve,mode):
    if mode=='STANDALONE':return sleeve.copy()
    out=np.maximum(core,sleeve) if mode=='MAX' else core+(1-core)*sleeve
    out[~np.isfinite(core)|~np.isfinite(sleeve)]=np.nan
    return out

def test_functions(data):
    rows=[]
    original,_=events(data)
    for cut in [650,1500,2500,3200]:
        altered=data.copy();cols=['total_log','volume','close','low','high']
        altered.loc[cut+1:,cols]=altered.loc[cut+1:,cols].to_numpy()*2
        changed,_=events(altered)
        for key in original:np.testing.assert_array_equal(original[key][:cut+1],changed[key][:cut+1])
        rows.append({'test':f'event_future_prefix_{cut}','status':'PASS'})
    for key in ['P','C']:
        event=original[(key,'IMMEDIATE')];confirmed=original[(key,'CONFIRM_NEXT')]
        require(not bool(confirmed[0]) and (confirmed[1:]<=event[:-1]).all(),'确认信号未延迟')
    rows.append({'test':'confirmation_not_backdated','status':'PASS'})
    damaged=data.copy();damaged.loc[650:660,'total_log']=np.nan
    res,_=events(damaged)
    require(not any(x[650:661].any() for x in res.values()),'缺失日产生进入')
    rows.append({'test':'missing_not_entry','status':'PASS'})
    a=np.array([np.nan,0.,.2,.8,1.]);b=np.array([0.,0.,.4,1.,.5])
    for mode in MODES:
        t=combine(a,b,mode);require(((t[np.isfinite(t)]>=0)&(t[np.isfinite(t)]<=1)).all(),'目标越界')
        if mode!='STANDALONE':require(np.isnan(t[0]),'未知核心被伪造')
    rows.append({'test':'combined_range_missing','status':'PASS'})
    s=np.array([np.nan,0.,1.,1.,np.nan,1.,0.,1.]);v=np.array([.2,.2,.2,.1,.1,.5,.5,.5])
    np.testing.assert_allclose(sleeve_size(s,v,'VOL10'),[np.nan,0.,.5,.5,np.nan,.5,0.,.2],equal_nan=True)
    rows.append({'test':'episode_size_frozen_until_zero','status':'PASS'})
    return rows

def freeze():
    cfg,data,div=inputs();OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'protocol.json').exists(),'已登记，不覆盖')
    tests=test_functions(data)
    files=[Path(__file__),ROOT/cfg['features'],ROOT/cfg['dividends'],ROOT/'research/event_clock_account_v1.py',ROOT/'research/simple_price_entry_exit_v1.py',ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',ROOT/'tools/research/annual10_capital_20260910.py']
    for period in ['evaluation','earlier_diagnostic']:
        for cost in cfg['costs']:
            files.extend(OLD/period/cost/f'{CORE}_{suffix}.parquet' for suffix in ['ledger','decisions'])
    write_json(OUT/'protocol.json',{'study_id':'510300_ANNUAL10_SEQUENTIAL_OPPORTUNITY_20260910','registered_at':now(),'remote_protocol_commit':'5f5628762f6976377cf14408e6709c1b2d292fb3','rules':RULES,'sizes':SIZES,'modes':MODES,'planned_policies':72,'planned_accounts':288,'planned_source_accounts':48,'goal_cagr':.10,'goal_sharpe':1.2,'code_sha256':digest(Path(__file__)),'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files],'history':'PREVIOUSLY_OBSERVED','position_impact':0,'live_trading_authorized':False},exclusive=True)
    write_json(OUT/'tests_receipt.json',{'tests':tests,'passed':True,'count':len(tests)})
    print('FROZEN',len(tests),'TESTS_PASS',flush=True)

def verify():
    pro=load_json(OUT/'protocol.json')
    for row in pro['files']:
        p=ROOT/row['path'];require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'输入或代码改变：'+row['path'])

def run(period,cost):
    verify();cfg,data,div=inputs()
    start,end=(cfg['evaluation_start'],cfg['data_cutoff']) if period=='evaluation' else (cfg['earlier_start'],cfg['earlier_terminal'])
    data=data[data.date<=pd.Timestamp(end)].copy();n=len(data)
    folder=OUT/period/cost;folder.mkdir(parents=True,exist_ok=True)
    require(not (folder/'metrics.csv').exists(),'本情景已完成，不覆盖')
    pdflags,factors=events(data);factors.to_parquet(folder/'factors.parquet',index=False)
    old=read_frame(OLD/period/cost/f'{CORE}_ledger.parquet');dec=read_frame(OLD/period/cost/f'{CORE}_decisions.parquet')
    core=aligned(data,dec)
    replay,_=simulate_event_account(data,div,cfg,cfg['costs'][cost],start,'REPLAY_CORE',targets=core,event_mask=np.ones(n,bool))
    for col in ['equity','cash','shares','commission','slippage_cost','net_return','filled_quantity']:
        np.testing.assert_allclose(replay[col].to_numpy(),old[col].to_numpy(),atol=1e-8,rtol=0)
    write_json(folder/'core_replay.json',{'status':'PASS','fields':['equity','cash','shares','commission','slippage_cost','net_return','filled_quantity'],'rows':len(old)})
    metrics=[];checks=[];source_metrics=[]
    for event,confirm,h in RULES:
        key=f'{event}__{confirm}__H{h}'
        rule={'entry':pdflags[(event,confirm)],'exit':{1:np.zeros(n,bool)}}
        spec={'cooldown':2,'modes':{1:{'loss':.04,'trail':.06,'take':None,'days':h}}}
        source,sd,cycles=simulate_policy(data,div,cfg,cfg['costs'][cost],start,rule,spec)
        src=folder/'sources';src.mkdir(exist_ok=True)
        for suffix,df in [('ledger',source),('decisions',sd)]:df.to_parquet(src/f'{key}_{suffix}.parquet',index=False)
        cycles.to_csv(src/f'{key}_cycles.csv',index=False)
        checks.append({'path':str((src/f'{key}_ledger.parquet').relative_to(ROOT)),**accounting(source,cfg)})
        source_metrics.append({'period':period,'cost':cost,'model':key,**summarize(source,cfg)})
        state=aligned(data,sd)
        for size in SIZES:
            sleeve=sleeve_size(state,data.vol20.to_numpy(),size)
            for mode in MODES:
                model=key+'__'+size+'__'+mode;target=combine(core,sleeve,mode)
                led,dec=simulate_event_account(data,div,cfg,cfg['costs'][cost],start,model,targets=target,event_mask=np.ones(n,bool))
                for suffix,df in [('ledger',led),('decisions',dec)]:df.to_parquet(folder/f'{model}_{suffix}.parquet',index=False)
                checks.append({'path':str((folder/f'{model}_ledger.parquet').relative_to(ROOT)),**accounting(led,cfg)})
                m={'period':period,'cost':cost,'model':model,**summarize(led,cfg)}
                m['point_met']=m['annualized_return']>=.10 and m['net_sharpe'] is not None and m['net_sharpe']>=1.2
                metrics.append(m)
        print('COMPLETED',period,cost,key,flush=True)
    pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(source_metrics).to_csv(folder/'source_metrics.csv',index=False)
    pd.DataFrame(checks).to_csv(folder/'checks.csv',index=False)
    print(pd.DataFrame(metrics).sort_values('annualized_return',ascending=False)[['model','annualized_return','net_sharpe','max_drawdown','point_met']].head(12).to_string(index=False),flush=True)
    print('POINT_PASSES',sum(m['point_met'] for m in metrics),flush=True)

def finish():
    verify();frames=[pd.read_csv(OUT/p/c/'metrics.csv') for p in ['evaluation','earlier_diagnostic'] for c in ['BASE','STRESS']]
    metrics=pd.concat(frames,ignore_index=True);require(len(metrics)==288,'账户不全')
    metrics.to_csv(OUT/'all_metrics.csv',index=False)
    adj=[]
    for model,g in metrics.groupby('model'):
        adj.append({'model':model,'main_base_met':bool(g[g.period.eq('evaluation')&g.cost.eq('BASE')].point_met.all()),'main_both_met':bool(g[g.period.eq('evaluation')].point_met.all()),'all_four_met':bool(g.point_met.all())})
    a=pd.DataFrame(adj);a.to_csv(OUT/'adjudications.csv',index=False)
    write_json(OUT/'result.json',{'status':'COMPLETED_DISCOVERY','completed_at':now(),'policies':72,'accounts':288,'reference_accounts':48,'original_core_replays':4,'point_pass_counts':{'main_base':int(a.main_base_met.sum()),'main_both_costs':int(a.main_both_met.sum()),'all_four':int(a.all_four_met.sum())},'independent_validation':'NOT_ESTABLISHED','goal_achieved':False,'position_impact':0,'live_trading_authorized':False})
    print(a[a.main_base_met].to_string(index=False))

if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='finish':finish()
    else:run(sys.argv[1],sys.argv[2])
