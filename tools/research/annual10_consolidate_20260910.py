"""合并七组实际账户裁决；缺失值不替代成收益或通过状态。"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.intraday_overnight_increment_v1 import write_json,digest,now,require

def main():
    six=ROOT/'research_runs/annual10_sharpe12_20260910'
    seven=ROOT/'research_runs/annual10_consensus_complement_20260910'
    out=ROOT/'research_runs/annual10_consolidated_20260910';out.mkdir(exist_ok=True)
    a=pd.read_csv(six/'all_metrics.csv');b=pd.read_csv(seven/'metrics.csv')
    b['phase']='consensus_complement';b['candidate_id']=b.phase+'/'+b.model
    metrics=pd.concat([a,b],ignore_index=True)
    require(len(metrics)==1812 and metrics.candidate_id.nunique()==453,'预登记计数改变')
    require(not metrics.duplicated(['candidate_id','period','cost']).any(),'情景重复')
    metrics['point_met']=(metrics.annualized_return>=.10)&(metrics.net_sharpe>=1.2)
    main_rows=metrics[metrics.period.eq('evaluation')&metrics.cost.eq('BASE')]
    adjudications=[]
    for cid,g in metrics.groupby('candidate_id'):
        require(len(g)==4,'未完成四情景')
        adjudications.append({'candidate_id':cid,'main_base_point_met':bool(g[g.period.eq('evaluation')&g.cost.eq('BASE')].point_met.all()),'main_both_costs_met':bool(g[g.period.eq('evaluation')].point_met.all()),'all_four_met':bool(g.point_met.all())})
    phases=[]
    for name,g in main_rows.groupby('phase'):
        best=g.sort_values('annualized_return',ascending=False).iloc[0]
        phases.append({'phase':name,'policies':len(g),'accounts':len(g)*4,'max_main_base_cagr':best.annualized_return,'sharpe_of_same_max_cagr_account':best.net_sharpe,'joint_target_count':int(g.point_met.sum())})
    metrics.to_csv(out/'all_metrics_1812.csv',index=False)
    pd.DataFrame(adjudications).to_csv(out/'adjudications_453.csv',index=False)
    pd.DataFrame(phases).to_csv(out/'phase_summary.csv',index=False)
    c1=pd.read_csv(six/'all_account_checks.csv');c2=pd.read_csv(seven/'checks.csv')
    require(len(c1)==1712 and len(c2)==302 and c1.status.eq('PASS').all() and c2.status.eq('PASS').all(),'账户检查未通过')
    best=main_rows[main_rows.net_sharpe>=1.2].sort_values('annualized_return',ascending=False).iloc[0]
    richest=main_rows.sort_values('annualized_return',ascending=False).iloc[0]
    cols=['phase','model','candidate_id','annualized_return','net_sharpe','annualized_volatility','max_drawdown','mean_exposure','trade_count']
    passed=bool(main_rows.point_met.any())
    result={'study_id':'510300_ANNUAL10_SHARPE12_CONSOLIDATED_20260910','completed_at':now(),'status':'HISTORICAL_POINT_FOUND_REQUIRES_VALIDATION' if passed else 'RESEARCH_BATCH_COMPLETE_TARGET_NOT_ACHIEVED','goal_achieved':False,'requested_cagr':.10,'requested_sharpe':1.2,'main_base_point_met':passed,'main_both_costs_met':any(r['main_both_costs_met'] for r in adjudications),'all_four_scenarios_met':any(r['all_four_met'] for r in adjudications),'registered_policy_or_control_configurations':453,'policy_scenario_accounts':1812,'reference_accounts':162,'additional_control_accounts':40,'account_paths_checked':2014,'all_accounting_checks_passed':True,'saved_predictive_model_versions':453,'completed_phases':7,'best_cagr_among_main_sharpe12':best[cols].to_dict(),'highest_main_cagr':richest[cols].to_dict(),'phase_summary':phases,'six_stage_ci_run_id':34465032366,'six_stage_ci_status':'SUCCESS','six_stage_replay_new_candidates':0,'seventh_stage_exact_replay_paths':302,'independent_validation':'NOT_ESTABLISHED','flow_source_vintage':'UNVERIFIED_PUBLICATION_VINTAGE','position_impact':0,'live_trading_authorized':False,'source_main_commit':'e5433a24e6aecc2e3333ac12e050fbb84f9169ce','six_stage_code_commit':'1b14608cc0e7da4589a876e4a16fc9eb73b9776d','seventh_stage_protocol_commit':'38334a89318b2ffa85e9f74dc6bd440998eb8cd2','seventh_stage_code_commit':'00551b8376d4738ca13ba2112081a0ffa5d956f0','seventh_code_sha256':digest(ROOT/'tools/research/annual10_consensus_complement_20260910.py')}
    write_json(out/'result.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
