"""独立算式、完整前缀、费用情景及事后删除事件诊断；不设计新交易政策。"""
from __future__ import annotations
import sys,json,hashlib,math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.research.annual10_sequential_20260910 import (OUT,CORE,OLD,events,inputs,combine,aligned,read_frame,verify,load_json,simulate_policy,simulate_event_account,summarize,accounting)
from research.intraday_overnight_increment_v1 import write_json,digest,now
MODEL='C__IMMEDIATE__H5__FULL__MAX'
SOURCE='C__IMMEDIATE__H5'

def independent_check(l,cfg):
    q=l.shares.to_numpy(float);dq=l.filled_quantity.to_numpy(float)
    old=np.r_[0.,q[:-1]];eq=l.equity.to_numpy(float);prev=np.r_[cfg['initial_capital'],eq[:-1]]
    mark=l.mark.to_numpy(float);prevmark=np.r_[l.open.iloc[0],mark[:-1]]
    fill=l.fill_price.fillna(0).to_numpy(float)
    cash=l.cash.to_numpy(float);oldcash=np.r_[cfg['initial_capital'],cash[:-1]]
    np.testing.assert_allclose(q-old,dq,rtol=0,atol=0)
    np.testing.assert_allclose(cash-oldcash,-dq*fill-l.commission+l.dividend_paid,rtol=0,atol=1e-7)
    np.testing.assert_allclose(eq,cash+q*mark+l.dividend_receivable,rtol=0,atol=1e-7)
    pnl=old*(mark-prevmark)+dq*(mark-fill)+l.dividend_recognized-l.commission
    np.testing.assert_allclose(eq-prev,pnl,rtol=0,atol=1e-7)
    np.testing.assert_allclose(l.net_return,eq/prev-1,rtol=0,atol=1e-13)
    assert np.all(-np.minimum(dq,0)<=old) and np.all(q%100==0) and min(cash)>=-1e-7
    assert q[-1]==0 and l.mark_clock.iloc[-1]=='OPEN_TERMINAL'
    r=eq/prev-1;sr=np.sqrt(242)*np.mean(r)/np.std(r,ddof=1) if np.std(r,ddof=1)>0 else None
    cagr=float((eq[-1]/cfg['initial_capital'])**(242/len(eq))-1)
    reported=summarize(l,cfg)
    assert abs(cagr-reported['annualized_return'])<1e-12
    if sr is not None:assert abs(sr-reported['net_sharpe'])<1e-12
    return {'status':'PASS','rows':len(l),'cagr':cagr,'sharpe':sr,'nav_sha256':hashlib.sha256(np.column_stack([eq,r,q]).tobytes()).hexdigest(),'max_pnl_error':float(np.max(np.abs(pnl-(eq-prev))))}

def bootstrap_matrix(a,b):
    rng=np.random.default_rng(20260910);n=len(a);rows=[]
    for i in range(20):
        ix=(rng.integers(n,size=(100,math.ceil(n/20),1))+np.arange(20)[None,None,:])%n
        ix=ix.reshape(100,-1)[:,:n];x=a[ix];y=b[ix]
        ca=np.expm1(np.log1p(x).sum(axis=1)*242/n);cb=np.expm1(np.log1p(y).sum(axis=1)*242/n)
        sr=np.sqrt(242)*x.mean(axis=1)/x.std(axis=1,ddof=1)
        rows.extend(zip(ca,sr,ca-cb))
    z=np.array(rows)
    return {'draws':2000,'block':20,'seed':20260910,'conditional_cagr_ci':np.quantile(z[:,0],[.025,.5,.975]).tolist(),'conditional_sharpe_ci':np.quantile(z[:,1],[.025,.5,.975]).tolist(),'conditional_increment_ci':np.quantile(z[:,2],[.025,.5,.975]).tolist(),'selection_adjusted':False,'future_success_probability':None}

def cycles(l):
    groups=[];active=False;old=0
    for row in l.itertuples():
        if not active and row.shares>0:active=True;entry=row.date;profit=0.;days=0
        if active:profit+=row.pnl;days+=int(row.shares>0)
        if active and row.shares==0 and old>0:
            groups.append({'entry_date':entry,'exit_date':row.date,'net_profit_cny':profit,'holding_closes':days});active=False
        old=row.shares
    assert not active
    return groups

def main():
    verify();cfg,data,div=inputs();audit=OUT/'audit';audit.mkdir(exist_ok=True)
    assert not (audit/'completion.json').exists(),'诊断已完成；不覆盖'
    check=[]
    for p in sorted(OUT.glob('*/*/*_ledger.parquet'))+sorted(OUT.glob('*/*/sources/*_ledger.parquet')):
        check.append({'path':str(p.relative_to(ROOT)),**independent_check(read_frame(p),cfg)})
    assert len(check)==336,len(check)
    pd.DataFrame(check).to_csv(audit/'independent_account_checks.csv',index=False)
    yearly=[];selected=[];boot=[];allcy=[];concentration=[];prefix=[];unique=[]
    for period in ['evaluation','earlier_diagnostic']:
        start,end=(cfg['evaluation_start'],cfg['data_cutoff']) if period=='evaluation' else (cfg['earlier_start'],cfg['earlier_terminal'])
        f=data[data.date<=pd.Timestamp(end)].copy()
        for cost in ['BASE','STRESS']:
            folder=OUT/period/cost;l=read_frame(folder/f'{MODEL}_ledger.parquet')
            old=read_frame(OLD/period/cost/f'{CORE}_ledger.parquet')
            m=summarize(l,cfg);selected.append({'period':period,'cost':cost,'model':MODEL,**m,'terminal_equity':float(l.equity.iloc[-1]),'net_profit':float(l.pnl.sum()),'increment_vs_core':m['annualized_return']-summarize(old,cfg)['annualized_return']})
            boot.append({'period':period,'cost':cost,**bootstrap_matrix(l.net_return.to_numpy(),old.net_return.to_numpy())})
            for year,g in l.groupby(l.date.dt.year):
                yearly.append({'period':period,'cost':cost,'year':int(year),**summarize(g,cfg)})
            cs=cycles(l);allcy.extend({'period':period,'cost':cost,**c} for c in cs)
            net=float(l.pnl.sum());profits=sorted([c['net_profit_cny'] for c in cs],reverse=True)
            concentration.append({'period':period,'cost':cost,'cycles':len(cs),'net_profit':net,'top1_share':sum(profits[:1])/net,'top3_share':sum(profits[:3])/net,'top5_share':sum(profits[:5])/net,'holding_closes':int(l.shares.gt(0).sum()),'pnl_outside_cycles':net-sum(c['net_profit_cny'] for c in cs)})
            other=read_frame(folder/'C__IMMEDIATE__H5__FULL__REMAINING_ledger.parquet')
            pd.testing.assert_frame_equal(l,other)
            unique.append({'period':period,'cost':cost,'max_remaining_same_path':True})
            # 在截断数据上重算事件、源参考和最终目标；除去截断终点的强制清算行。
            first=int(np.flatnonzero(f.date>=pd.Timestamp(start))[0]);full_dec=read_frame(folder/f'{MODEL}_decisions.parquet')
            for cut in [first+200,first+600,len(f)-100]:
                part=f.iloc[:cut+2].copy();ev,_=events(part)
                ref,rd,_=simulate_policy(part,div,cfg,cfg['costs'][cost],start,{'entry':ev[('C','IMMEDIATE')],'exit':{1:np.zeros(len(part),bool)}},{'cooldown':2,'modes':{1:{'loss':.04,'trail':.06,'take':None,'days':5}}})
                coredec=read_frame(OLD/period/cost/f'{CORE}_decisions.parquet');coredec=coredec[coredec.origin<=part.date.iloc[-2]].copy()
                core=aligned(part,coredec);target=combine(core,aligned(part,rd),'MAX')
                replay,decs=simulate_event_account(part,div,cfg,cfg['costs'][cost],start,'PREFIX_REPLAY',targets=target,event_mask=np.ones(len(part),bool))
                for col in ['cash','shares','equity','net_return','commission','slippage_cost','filled_quantity']:
                    np.testing.assert_allclose(replay[col].to_numpy()[:-1],l[col].to_numpy()[:len(replay)-1],rtol=0,atol=1e-8)
                np.testing.assert_allclose(decs.reference_weight,full_dec.reference_weight.iloc[:len(decs)],equal_nan=True,rtol=0,atol=0)
                prefix.append({'period':period,'cost':cost,'last_compared_close':part.date.iloc[-2],'source_and_combined_prefix':'PASS'})
    for name,rows in [('selected_metrics',selected),('yearly',yearly),('cycles',allcy),('concentration',concentration),('prefix_checks',prefix),('identical_paths',unique)]:
        pd.DataFrame(rows).to_csv(audit/f'{name}.csv',index=False)
    write_json(audit/'conditional_bootstrap.json',{'note':'POST_SELECTION_DESCRIPTIVE_ONLY; no correction for all research trials','rows':boot})
    # 逐个移除新增源策略已发生的机会：事后反事实，不能晋升成可交易过滤器。
    period,cost='evaluation','BASE';folder=OUT/period/cost
    dec=read_frame(folder/f'{MODEL}_decisions.parquet');core=aligned(data,read_frame(OLD/period/cost/f'{CORE}_decisions.parquet'))
    state=aligned(data,read_frame(folder/'sources'/f'{SOURCE}_decisions.parquet'))
    src_cycles=pd.read_csv(folder/'sources'/f'{SOURCE}_cycles.csv',parse_dates=['entry_origin','exit_origin'])
    perturb=[]
    for row in src_cycles.itertuples():
        modified=state.copy();mask=(data.date>=row.entry_origin)&(data.date<row.exit_origin);modified[mask]=0.
        target=combine(core,modified,'MAX');name=f'HINDSIGHT_DELETE_SOURCE_EPISODE_{row.cycle_id}'
        led,_=simulate_event_account(data,div,cfg,cfg['costs'][cost],cfg['evaluation_start'],name,targets=target,event_mask=np.ones(len(data),bool))
        independent_check(led,cfg);led.to_parquet(audit/f'{name}_ledger.parquet',index=False)
        perturb.append({'diagnostic':name,'interpretation':'HINDSIGHT_DIAGNOSTIC_NOT_TRADABLE','removed_entry_origin':row.entry_origin,**summarize(led,cfg)})
    pd.DataFrame(perturb).to_csv(audit/'leave_one_source_episode_out.csv',index=False)
    # 不改变信号内容，仅检验在执行上多延迟一个交易日的后果。
    delays=[]
    for period in ['evaluation','earlier_diagnostic']:
        for cost in ['BASE','STRESS']:
            start,end=(cfg['evaluation_start'],cfg['data_cutoff']) if period=='evaluation' else (cfg['earlier_start'],cfg['earlier_terminal'])
            f=data[data.date<=pd.Timestamp(end)].copy();t=aligned(f,read_frame(OUT/period/cost/f'{MODEL}_decisions.parquet'));t=np.r_[np.nan,t[:-1]]
            led,_=simulate_event_account(f,div,cfg,cfg['costs'][cost],start,'EXECUTION_DELAY_ONE_EXTRA_DAY',targets=t,event_mask=np.ones(len(f),bool))
            independent_check(led,cfg);led.to_parquet(audit/f'DELAY_{period}_{cost}_ledger.parquet',index=False)
            delays.append({'period':period,'cost':cost,'diagnostic':'ONE_EXTRA_EXECUTION_DAY',**summarize(led,cfg)})
    pd.DataFrame(delays).to_csv(audit/'extra_execution_day.csv',index=False)
    # 在前2020年规则比较中检查本轮点达标者的排名；不事后把它说成训练选择结果。
    m=pd.read_csv(OUT/'all_metrics.csv');early=m[m.period.eq('earlier_diagnostic')&m.cost.eq('BASE')].copy()
    early['early_cagr_rank']=early.annualized_return.rank(method='min',ascending=False)
    early['early_sharpe_rank']=early.net_sharpe.rank(method='min',ascending=False)
    early.to_csv(audit/'earlier_selection_diagnostic.csv',index=False)
    conclusion={'status':'VERIFIED_HISTORICAL_POINT_ONLY','completed_at':now(),'new_policy_accounts':288,'source_accounts':48,'original_core_replays':4,'independent_account_checks':336,'prefix_checks':len(prefix),'diagnostic_accounts':len(perturb)+len(delays),'new_unit_tests':8,'historical_main_base_joint_goal':True,'historical_main_both_costs_joint_goal':False,'all_four_joint_goal':False,'passing_configurations':2,'distinct_passing_paths':1,'selected_model':MODEL,'new_model_fits':0,'independent_validation':'NOT_ESTABLISHED','validated_strategy_goal_achieved':False,'position_impact':0,'live_trading_authorized':False,'source_files':load_json(OUT/'protocol.json')['files'],'audit_code_sha256':digest(Path(__file__))}
    write_json(audit/'completion.json',conclusion,exclusive=True)
    print('AUDIT_PASSED',len(check),'PREFIX',len(prefix),'DIAGNOSTIC',len(perturb)+len(delays),flush=True)
    print(pd.DataFrame(perturb)[['removed_entry_origin','annualized_return','net_sharpe']].to_string(index=False))
    print(pd.DataFrame(delays)[['period','cost','annualized_return','net_sharpe']].to_string(index=False))
    print(pd.DataFrame(concentration).to_string(index=False))

if __name__=='__main__':main()
