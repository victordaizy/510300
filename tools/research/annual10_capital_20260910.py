"""年化10%/夏普1.2：完整资金边界实验；所有原输入只读。"""
from __future__ import annotations
import hashlib, json, math, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
P=ROOT/'reports/research'
OUT=ROOT/'research_runs/annual10_sharpe12_20260910/capital'
FOLDERS={'R91':'510300_continuous_reference_min_variance_v1','R143':'510300_trend_noise_reference_blend_v1','R150':'510300_monotone_episode_budget_v1'}
MODELS={'R91':'CONTINUOUS_REFERENCE_MIN_VARIANCE','R143':'TREND_NOISE_REFERENCE_BLEND','R150':'EPISODE_BUDGET_NONINCREASING'}
POLICIES=['ORIGINAL','X2','X3','X4','SQRT','FULL_SUPPORT','RISK_CONTRACT']

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def read_frame(path):
    f=pd.read_parquet(path)
    for c in f:
        if getattr(f[c].dtype,'kind',None)=='M':f[c]=f[c].astype('datetime64[ns]')
    return f

def aligned(data,decisions):
    idx=pd.DatetimeIndex(data.date).get_indexer(decisions.origin)
    require((idx>=0).all() and (idx<len(data)-1).all() and len(set(idx))==len(idx),'意向日历错误')
    require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(data.date.iloc[idx+1])),'出现非下一开盘意向')
    out=np.full(len(data),np.nan);out[idx]=decisions.reference_weight.to_numpy(float)
    return out

def capital_target(data,parent,returns,policy):
    require((np.isnan(parent)|((parent>=0)&(parent<=1))).all(),'父目标范围非法')
    if policy=='ORIGINAL':return parent.copy()
    if policy in {'X2','X3','X4'}:return np.minimum(1.,int(policy[-1])*parent)
    if policy=='SQRT':return np.sqrt(parent)
    if policy=='FULL_SUPPORT':return np.where(np.isnan(parent),np.nan,(parent>0).astype(float))
    require(policy=='RISK_CONTRACT','政策未知')
    out=np.full(len(parent),np.nan);mult=1.
    for t in range(len(parent)-1):
        if t and data.date.iloc[t].month!=data.date.iloc[t-1].month:
            history=returns[max(0,t-241):t+1]
            if len(history)==242 and np.isfinite(history).all():
                vol=history.std(ddof=1)*np.sqrt(242)
                if vol>0:mult=min(4.,(.10/1.2)/vol)
        out[t]=min(1.,parent[t]*mult) if np.isfinite(parent[t]) else np.nan
    return out

def accounting(ledger,cfg):
    eq=ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable
    np.testing.assert_allclose(eq,ledger.equity,atol=1e-7,rtol=0)
    previous=np.r_[cfg['initial_capital'],ledger.equity.to_numpy()[:-1]]
    np.testing.assert_allclose(ledger.net_return,ledger.equity.to_numpy()/previous-1,atol=1e-13,rtol=0)
    np.testing.assert_allclose(ledger.pnl,ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost,atol=1e-7,rtol=0)
    require(ledger.cash.min()>=-1e-7 and ledger.shares.min()>=0 and (ledger.shares%100==0).all(),'透支或股数非法')
    require(ledger.shares.iloc[-1]==0,'终点未清算')
    m=summarize(ledger,cfg);r=ledger.net_return.to_numpy()
    c=float(np.expm1(np.log1p(r).sum()*242/len(r)))
    s=float(r.mean()/r.std(ddof=1)*np.sqrt(242)) if r.std(ddof=1)>1e-15 else None
    require(abs(c-m['annualized_return'])<1e-12 and (s is None or abs(s-m['net_sharpe'])<1e-12),'指标独立复算不符')
    return {'rows':len(ledger),'accounting_max_abs_error':float(ledger.accounting_error.abs().max()),'status':'PASS'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'protocol.json').exists(),'禁止覆盖本次结果')
    cfg=load_json(ROOT/'config/510300_continuous_reference_min_variance_v1.json')
    inputs=[ROOT/cfg['features'],ROOT/cfg['dividends']]
    inputs.extend(ROOT/'research'/f for f in ['adaptive_allocation_v1.py','event_clock_account_v1.py','intraday_overnight_increment_v1.py'])
    for key in FOLDERS:
        for per in ['evaluation','earlier_diagnostic']:
            for cost in cfg['costs']:
                inputs.extend(P/FOLDERS[key]/per/cost/f'{MODELS[key]}_{x}.parquet' for x in ['ledger','decisions'])
    protocol={'study_id':'510300_ANNUAL10_SHARPE12_20260910_CAPITAL','registered_at':now(),
        'remote_protocol_commit':'82d3fce7df3dac55e6f8ad3086571546fbb705bf','source_commit':'e5433a24e6aecc2e3333ac12e050fbb84f9169ce',
        'policies':POLICIES,'parents':FOLDERS,'planned_accounts':84,'code_sha256':digest(Path(__file__)),
        'target_cagr':.10,'target_sharpe':1.2,'max_position':1.,'cash_rate':0.,'parameter_selection':'ALL_FIXED_POLICIES_REPORTED',
        'files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in inputs]}
    write_json(OUT/'protocol.json',protocol,exclusive=True)
    data=read_frame(ROOT/cfg['features']);div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    rows=[];checks=[];replays=[]
    for period,start,end in [('evaluation',cfg['evaluation_start'],cfg['data_cutoff']),('earlier_diagnostic',cfg['earlier_start'],cfg['earlier_terminal'])]:
        frame=data[data.date.le(end)].copy()
        for cost,cost_cfg in cfg['costs'].items():
            for key in FOLDERS:
                base=P/FOLDERS[key]/period/cost
                old=read_frame(base/f'{MODELS[key]}_ledger.parquet');dec=read_frame(base/f'{MODELS[key]}_decisions.parquet')
                parent=aligned(frame,dec);rr=np.full(len(frame),np.nan);rr[pd.DatetimeIndex(frame.date).get_indexer(old.date)]=old.net_return
                for policy in POLICIES:
                    target=capital_target(frame,parent,rr,policy)
                    model=key+'__'+policy
                    ledger,decision=simulate_event_account(frame,div,cfg,cost_cfg,start,model,targets=target,event_mask=np.ones(len(frame),bool))
                    if policy=='ORIGINAL':
                        for col in ['equity','cash','shares','commission','slippage_cost','net_return']:
                            np.testing.assert_allclose(ledger[col],old[col],atol=1e-8,rtol=0)
                        replays.append({'period':period,'cost':cost,'parent':key,'status':'PASS'})
                    folder=OUT/period/cost;folder.mkdir(parents=True,exist_ok=True)
                    ledger.to_parquet(folder/f'{model}_ledger.parquet',index=False)
                    decision.to_parquet(folder/f'{model}_decisions.parquet',index=False)
                    checks.append({'period':period,'cost':cost,'model':model,**accounting(ledger,cfg)})
                    m={'period':period,'cost':cost,'model':model,**summarize(ledger,cfg)}
                    m['point_met']=bool(m['annualized_return']>=.10 and m['net_sharpe'] is not None and m['net_sharpe']>=1.2)
                    rows.append(m)
                print('COMPLETED',period,cost,key,flush=True)
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'accounting_checks.csv',index=False)
    pd.DataFrame(replays).to_csv(OUT/'original_replays.csv',index=False)
    result={'status':'COMPLETE','completed_at':now(),'accounts':len(rows),'replays':len(replays),'point_pass_rows':metrics[metrics.point_met].to_dict('records'),
        'independent_validation':'NOT_ESTABLISHED','live_trading_authorized':False,'position_impact':0}
    write_json(OUT/'result.json',result)
    print(metrics[metrics.period.eq('evaluation')&metrics.cost.eq('BASE')][['model','annualized_return','net_sharpe','max_drawdown','mean_exposure','point_met']].to_string(index=False),flush=True)

if __name__=='__main__':main()
