"""固定双信号的有限权重与风险缩减研究。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.simple_signal_blend_v1 import decision_state
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import normalize_dividends,save_account,summarize
from research.intraday_overnight_increment_v1 import digest,now,require,write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_simple_pair_risk_v1'
CONFIG=ROOT/'config/510300_simple_pair_risk_v1.json'
CONTROL='W50_NONE'

def candidates():
    return [{'id':f'W{int(w*100)}_{risk}','trend_weight':w,'risk':risk} for w in [.25,.5,.75] for risk in ['NONE','VOL10','VOL15','SHOCK']]

def target_for(data,state,item):
    target=state[:,0]*item['trend_weight']+state[:,1]*(1-item['trend_weight'])
    if item['risk'] in ['VOL10','VOL15']:
        limit=.10 if item['risk']=='VOL10' else .15
        target=target*np.minimum(1,limit/np.maximum(data.vol20.to_numpy(float),1e-12))
    elif item['risk']=='SHOCK':
        shock=(data.vol5.to_numpy(float)>1.5*data.vol60.to_numpy(float))&(data.mom5.to_numpy(float)<0)
        target=np.where(shock,0.,target)
    target=np.where(np.isfinite(state).all(axis=1),target,np.nan)
    return target

def name_for(item):
    labels={'NONE':'不额外缩减','VOL10':'按10%波动水平缩减','VOL15':'按15%波动水平缩减','SHOCK':'短期波动冲击时退出'}
    return f'趋势{item["trend_weight"]:.0%}、日内强弱{1-item["trend_weight"]:.0%}；{labels[item["risk"]]}'

def freeze():
    require(not CONFIG.exists(),'双信号风险方案已经登记，不覆盖')
    old=json.loads((ROOT/'config/510300_simple_signal_blend_v1.json').read_text(encoding='utf-8'))
    cfg={k:v for k,v in old.items() if k not in ['candidate_names','frozen_files','experts']}
    cfg.update({'study_id':'510300_SIMPLE_PAIR_RISK_V1','round':29,'registered_at':now(),'primary':'W50_VOL15','candidates':candidates(),
                'candidate_count':12,'new_candidate_configurations':11,'rules':'docs/510300_SIMPLE_PAIR_RISK_V1.md'})
    paths=[Path(__file__),ROOT/'research/event_clock_account_v1.py',ROOT/cfg['features'],ROOT/cfg['dividends'],ROOT/cfg['rules'],
           ROOT/'reports/research/510300_simple_signal_blend_v1/expert_states.parquet']
    for key in ['S1_TREND_REBOUND','D60_INTRA']:
        paths.append(ROOT/'reports/research/510300_simple_signal_blend_v1/earlier_experts'/f'{key}_decisions.parquet')
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in paths]
    write_json(CONFIG,cfg,exclusive=True)
    print('第29轮登记12个双信号权重与风险设置，其中1个原样对照、11个新增。',flush=True)

def run():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    for p in cfg['frozen_files']:require(digest(ROOT/p['path'])==p['sha256'],'双信号风险登记文件变化')
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'RUN_STARTED.json',{'started_at':now(),'config_sha256':digest(CONFIG)},exclusive=True)
    data=pd.read_parquet(ROOT/cfg['features']);div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    table=pd.read_parquet(ROOT/'reports/research/510300_simple_signal_blend_v1/expert_states.parquet')
    require(pd.DatetimeIndex(table.date).equals(pd.DatetimeIndex(data.date)),'双信号日期不一致')
    main_state=table[['S1_TREND_REBOUND','D60_INTRA']].to_numpy(float)
    earlier=data[data.date<=cfg['earlier_terminal']].copy();old_states=[]
    for key in ['S1_TREND_REBOUND','D60_INTRA']:
        d=pd.read_parquet(ROOT/'reports/research/510300_simple_signal_blend_v1/earlier_experts'/f'{key}_decisions.parquet')
        old_states.append(decision_state(earlier,d))
    old_state=np.column_stack(old_states)
    metrics=[];diagnostics=[];yearly=[];eras=[]
    for period,frame,state,start,dest in [('evaluation',data,main_state,cfg['evaluation_start'],metrics),
                                         ('earlier_diagnostic',earlier,old_state,cfg['earlier_start'],diagnostics)]:
        targets={item['id']:target_for(frame,state,item) for item in cfg['candidates']}
        pd.DataFrame({'date':frame.date,**targets}).to_parquet(OUT/f'{period}_targets.parquet',index=False)
        for cost_id,cost in cfg['costs'].items():
            accounts={}
            for item in cfg['candidates']:
                key=item['id']
                if key==CONTROL:
                    p=ROOT/'reports/research/510300_simple_signal_blend_v1'/period/cost_id
                    ledger=pd.read_parquet(p/'B2_TREND_SESSION_ledger.parquet');decisions=pd.read_parquet(p/'B2_TREND_SESSION_decisions.parquet')
                else:
                    ledger,decisions=simulate_event_account(frame,div,cfg,cost,start,key,targets=targets[key],event_mask=np.ones(len(frame),bool))
                save_account(OUT/period/cost_id,key,ledger,decisions);accounts[key]=ledger
            if period=='evaluation':bp=ROOT/'reports/research/510300_simple_price_entry_exit_v1/evaluation'/cost_id/'BUY_HOLD_ledger.parquet'
            else:bp=ROOT/'reports/research/510300_simple_signal_blend_v1/earlier_diagnostic'/cost_id/'BUY_HOLD_ledger.parquet'
            bh=pd.read_parquet(bp);base=summarize(bh,cfg);accounts['BUY_HOLD']=bh
            for key,ledger in accounts.items():
                name=name_for(next(item for item in cfg['candidates'] if item['id']==key)) if key!='BUY_HOLD' else '买入持有'
                m={'cost':cost_id,'model':key,'name':name,**summarize(ledger,cfg)}
                m['annualized_return_excess_vs_buy_hold']=m['annualized_return']-base['annualized_return']
                m['meets_point_target']=m['net_sharpe'] is not None and m['net_sharpe']>=1.2;dest.append(m)
                if period=='evaluation':
                    for y,g in ledger.groupby(ledger.date.dt.year):yearly.append({'cost':cost_id,'model':key,'year':int(y),**summarize(g,cfg)})
                    for label,left,right in [('2020—2021','2020-01-01','2021-12-31'),('2022—2023','2022-01-01','2023-12-31'),('2024—终点','2024-01-01',cfg['data_cutoff'])]:
                        g=ledger[(ledger.date>=left)&(ledger.date<=right)];eras.append({'cost':cost_id,'model':key,'era':label,**summarize(g,cfg)})
            pd.DataFrame({'date':bh.date,**{key:ledger.net_return.to_numpy() for key,ledger in accounts.items()}}).to_parquet(OUT/f'{period}_{cost_id}_returns.parquet',index=False)
            print(f'{period}／{cost_id}：12个双信号风险设置已完成。',flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diagnostics).to_csv(OUT/'earlier_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(eras).to_csv(OUT/'era_metrics.csv',index=False,encoding='utf-8-sig')
    best=max((m for m in metrics if m['cost']=='BASE' and m['model']!='BUY_HOLD'),key=lambda m:m['net_sharpe'] if m['net_sharpe'] is not None else -999)
    result={'study_id':cfg['study_id'],'completed_at':now(),'status':'PAIR_RISK_ACCOUNTS_AND_EARLIER_DIAGNOSTICS_COMPLETE','candidate_configurations':11,
            'total_candidates_evaluated':12,'evaluation_accounts':26,'new_accounts_generated':22,'reused_control_accounts':4,
            'earlier_diagnostic_accounts':26,'new_earlier_diagnostic_accounts':22,'all_metrics':metrics,'earlier_diagnostics':diagnostics,
            'primary':[m for m in metrics if m['model']==cfg['primary']],'post_selected_best_base':best,
            'historical_point_target_met':any(m['meets_point_target'] for m in metrics if m['model']!='BUY_HOLD'),'goal_achieved':False,
            'independent_validation':'NOT_ESTABLISHED_PREVIOUSLY_OBSERVED_HISTORY','position_impact':0}
    write_json(OUT/'result.json',result,exclusive=True)
    selected_early=[m for m in diagnostics if m['model']==best['model']]
    print(json.dumps({'状态':result['status'],'主评价最高':best,'同参数更早历史':selected_early,'预定主方案':result['primary']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1:]==['freeze']:freeze()
    elif sys.argv[1:]==['run']:run()
    else:raise SystemExit('请指定 freeze 或 run')
