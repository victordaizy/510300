"""汇总全候选而非只汇总胜者，按同账户同时检查10%与1.2。"""
from __future__ import annotations
import sys,hashlib,json,math,platform
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.research.annual10_capital_20260910 import load_json,read_frame
from research.adaptive_allocation_v1 import summarize
from research.intraday_overnight_increment_v1 import write_json,now,digest
OUT=ROOT/'research_runs/annual10_sharpe12_20260910'
PHASES={'capital':84,'residual':704,'episode_value':192,'daily_opportunity':192,'etf_flow':256,'winning_budget':128}
SELECTED=[('residual','CORE2__V1_CLIMAX_RECOVERY__IDLE_ONLY__A100'),('residual','CORE2__V1_CLIMAX_RECOVERY__IDLE_ONLY__A50'),('residual','CORE3__CAL_MONTH_EDGE__IDLE_ONLY__A100')]

def bootstrap(a,b,seed=20260910,reps=2000,block=20):
    rng=np.random.default_rng(seed);n=len(a);c=[]
    for _ in range(math.ceil(reps/100)):
        ids=(rng.integers(n,size=(100,math.ceil(n/block),1))+np.arange(block)[None,None,:])%n;ids=ids.reshape(100,-1)[:,:n]
        aa=a[ids];bb=b[ids]
        ca=np.expm1(np.log1p(aa).sum(axis=1)*242/n);cb=np.expm1(np.log1p(bb).sum(axis=1)*242/n)
        c.extend((ca-cb).tolist())
    return np.quantile(c[:reps],[.025,.5,.975]).tolist()

def cycles(ledger):
    records=[];active=False;amount=0.;prev=0
    for r in ledger.itertuples():
        if not active and r.shares>0:active=True;entry=r.date;amount=0.;held=0
        if active:amount+=r.pnl;held+=int(r.shares>0)
        if active and r.shares==0 and prev>0:records.append({'entry':entry,'exit':r.date,'pnl':amount,'holding_closes':held});active=False
        prev=r.shares
    assert not active
    net=float(ledger.pnl.sum());profits=sorted([r['pnl'] for r in records if r['pnl']>0],reverse=True)
    return records,{'cycles':len(records),'net_profit':net,'holding_days':int(ledger.shares.gt(0).sum()),'top_three_profit_share':sum(profits[:3])/net if net else None,'pnl_outside_cycles':net-sum(r['pnl'] for r in records)}

def main():
    cfg=load_json(ROOT/'config/510300_continuous_reference_min_variance_v1.json');frames=[];checks=[]
    for phase,expected in PHASES.items():
        paths=[OUT/phase/'metrics.csv'] if phase=='capital' else sorted((OUT/phase).glob('*/*/metrics.csv'))
        f=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);assert len(f)==expected,(phase,len(f))
        f['phase']=phase;f['candidate_id']=phase+'/'+f.model;frames.append(f)
        checks.append(pd.read_csv(OUT/phase/'independent_account_checks.csv'))
    d=pd.concat(frames,ignore_index=True);assert not d[['candidate_id','period','cost']].duplicated().any()
    d['point_met']=(d.annualized_return>=.10)&(d.net_sharpe>=1.2)
    d.to_csv(OUT/'all_metrics.csv',index=False)
    c=pd.concat(checks,ignore_index=True);c.to_csv(OUT/'all_account_checks.csv',index=False)
    main=d[d.period.eq('evaluation')&d.cost.eq('BASE')].copy()
    main.sort_values(['annualized_return','net_sharpe'],ascending=False).to_csv(OUT/'main_return_ranking.csv',index=False)
    main[main.net_sharpe>=1.2].sort_values('annualized_return',ascending=False).to_csv(OUT/'main_sharpe12_ranking.csv',index=False)
    summary=[]
    for cid,g in d.groupby('candidate_id',sort=True):
        assert len(g)==4
        mb=g[g.period.eq('evaluation')&g.cost.eq('BASE')].iloc[0]
        summary.append({'candidate_id':cid,'MAIN_BASE_POINT_MET':bool(mb.point_met),'MAIN_BOTH_COSTS_MET':bool(g[g.period.eq('evaluation')].point_met.all()),'ALL_FOUR_SCENARIOS_MET':bool(g.point_met.all()),'minimum_sharpe':g.net_sharpe.min(),'minimum_cagr':g.annualized_return.min()})
    pd.DataFrame(summary).to_csv(OUT/'candidate_adjudications.csv',index=False)
    selected=[];cyrows=[];cyinfo=[];yearly=[];boots=[]
    for phase,model in SELECTED:
        for period in ['evaluation','earlier_diagnostic']:
            for cost in ['BASE','STRESS']:
                l=read_frame(OUT/phase/period/cost/f'{model}_ledger.parquet');m=d[(d.phase==phase)&(d.model==model)&(d.period==period)&(d.cost==cost)].iloc[0]
                selected.append(m.to_dict());cr,info=cycles(l)
                cyrows.extend({'model':model,'period':period,'cost':cost,**r} for r in cr);cyinfo.append({'model':model,'period':period,'cost':cost,**info})
                for year,g in l.groupby(l.date.dt.year):yearly.append({'model':model,'period':period,'cost':cost,'year':int(year),**summarize(g,cfg)})
                parent=read_frame(OUT/'capital'/period/cost/'R150__X2_ledger.parquet')
                ci=bootstrap(l.net_return.to_numpy(),parent.net_return.to_numpy())
                boots.append({'model':model,'period':period,'cost':cost,'annualized_increment':float(m.annualized_return-summarize(parent,cfg)['annualized_return']),'conditional_ci_low':ci[0],'conditional_ci_median':ci[1],'conditional_ci_high':ci[2],'selection_adjusted':False})
    for name,rows in [('selected_four_scenarios',selected),('selected_cycles',cyrows),('selected_cycle_concentration',cyinfo),('selected_yearly',yearly),('conditional_bootstrap',boots)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    unique={}
    for phase in PHASES:
        for p in (OUT/phase).rglob('*_ledger.parquet'):
            rel=p.relative_to(OUT).as_posix()
            if '/sources/' in rel or '/references/' in rel:continue
            parts=rel.split('/');period,cost=parts[1:3];model=p.name.removesuffix('_ledger.parquet');cid=phase+'/'+model
            row=c[c.path.eq(str(p.relative_to(ROOT)))].iloc[0]
            unique.setdefault(cid,{})[(period,cost)]=row.nav_path_sha256
    hashes=[hashlib.sha256(json.dumps(sorted((str(k),v) for k,v in x.items())).encode()).hexdigest() for x in unique.values()]
    strongest=main[main.net_sharpe>=1.2].sort_values('annualized_return',ascending=False).iloc[0]
    richest=main.sort_values('annualized_return',ascending=False).iloc[0]
    newtests=load_json(OUT/'new_tests_receipt.json');ep=load_json(OUT/'episode_value/training_records.json');dd=load_json(OUT/'daily_opportunity/training_records.json')
    fit_count=ep['actual_fits']+dd['completed_model_fits']+sum(load_json(OUT/'etf_flow'/f'training_H{h}.json')['completed_model_fits'] for h in [5,20])
    result={'study_id':'510300_ANNUAL10_SHARPE12_20260910','status':'BATCH_COMPUTED_TARGET_NOT_MET','completed_at':now(),'source_commit':'e5433a24e6aecc2e3333ac12e050fbb84f9169ce',
      'goal_cagr':.10,'goal_sharpe':1.2,'goal_achieved':False,'main_base_point_met':bool(main.point_met.any()),'main_both_costs_met':any(x['MAIN_BOTH_COSTS_MET'] for x in summary),'all_four_scenarios_met':any(x['ALL_FOUR_SCENARIOS_MET'] for x in summary),
      'registered_policies':len(main),'policy_scenario_accounts':len(d),'distinct_four_scenario_nav_versions':len(set(hashes)),'reference_accounts':156,'daily_hypothetical_label_accounts':3182,
      'independent_account_checks':len(c),'independent_account_checks_passed':bool(c.status.eq('PASS').all()),'original_account_replays':12,'new_unit_tests':newtests['tests'],'new_unit_tests_passed':newtests['passed'],'original_pytest_passed':26,
      'completed_saved_predictive_models':fit_count,'partial_unpersisted_fit_calls':'UNKNOWN_DUE_TO_TOOL_TIMEOUT; not added as independent evidence','best_main_cagr_with_sharpe12':strongest.to_dict(),'highest_main_cagr':richest.to_dict(),
      'independent_validation':'NOT_ESTABLISHED','etf_flow_publication_vintage':'UNVERIFIED_PUBLICATION_VINTAGE','position_impact':0,'live_trading_authorized':False,
      'code_files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted((ROOT/'tools/research').glob('annual10*20260910.py'))]}
    assert not result['main_base_point_met'],'出现点目标应停止追加研究并单独复核，不可用失败模板覆盖'
    write_json(OUT/'result.json',result)
    print(json.dumps({k:result[k] for k in ['status','registered_policies','policy_scenario_accounts','distinct_four_scenario_nav_versions','independent_account_checks','completed_saved_predictive_models','main_base_point_met']},ensure_ascii=False,indent=2))
    print(pd.DataFrame(selected)[['model','period','cost','annualized_return','net_sharpe','max_drawdown','mean_exposure']].to_string(index=False))

if __name__=='__main__':main()
