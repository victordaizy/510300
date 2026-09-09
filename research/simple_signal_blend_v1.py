"""使用三种已存在的独立信号状态，检验固定组合与共识仓位。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.simple_price_entry_exit_v1 import signals, specifications as price_specs, simulate_policy
from research.simple_session_divergence_v1 import specifications as session_specs, make_rule as session_rule
from research.simple_volume_reversal_v1 import specifications as volume_specs, make_rules as volume_rules
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_simple_signal_blend_v1'
CONFIG=ROOT/'config/510300_simple_signal_blend_v1.json'
EXPERTS=[('S1_TREND_REBOUND','510300_simple_price_entry_exit_v1'),('D60_INTRA','510300_simple_session_divergence_v1'),('V1_CLIMAX_RECOVERY','510300_simple_volume_reversal_v1')]
NAMES={'B1_EQUAL_THREE':'三种状态固定等权','B2_TREND_SESSION':'趋势与日内强弱各半','B3_TREND_VOLUME':'趋势与放量反弹各半',
       'B4_SESSION_VOLUME':'日内强弱与放量反弹各半','B5_MAJORITY':'至少两种状态同意才进入','B6_PARTIAL_CONSENSUS':'一种同意半仓、两种同意满仓',
       'B7_TREND_ANCHOR':'趋势占一半、其余各四分之一','B8_VOL_SCALED_EQUAL':'三种等权并按波动缩减仓位'}


def targets(expert_state,vol20):
    a,b,c=expert_state.T
    total=expert_state.sum(axis=1)
    result={'B1_EQUAL_THREE':(a+b+c)/3,'B2_TREND_SESSION':(a+b)/2,'B3_TREND_VOLUME':(a+c)/2,
            'B4_SESSION_VOLUME':(b+c)/2,'B5_MAJORITY':(total>=2).astype(float),
            'B6_PARTIAL_CONSENSUS':np.where(total>=2,1.,np.where(total>=1,.5,0.)),
            'B7_TREND_ANCHOR':a*.5+b*.25+c*.25,
            'B8_VOL_SCALED_EQUAL':(a+b+c)/3*np.minimum(1,.10/np.maximum(vol20,1e-12))}
    missing=~np.isfinite(expert_state).all(axis=1)
    for k in result:result[k]=np.where(missing,np.nan,result[k])
    return result


def decision_state(data,decisions):
    result=np.full(len(data),np.nan)
    ids=pd.DatetimeIndex(data.date).get_indexer(pd.to_datetime(decisions.origin))
    require((ids>=0).all(),'专家判断时钟不在日历')
    result[ids]=decisions.reference_weight.to_numpy(float)
    return result


def earlier_experts(data,div,cfg):
    p=price_specs();s=session_specs();v=volume_specs()
    def_s=next(item for item in s if item['id']=='D60_INTRA')
    session_cfg=json.loads((ROOT/'config/510300_simple_session_divergence_v1.json').read_text(encoding='utf-8'))
    rules=[signals(data)['S1_TREND_REBOUND'],session_rule(data,def_s)[0],volume_rules(data)[0]['V1_CLIMAX_RECOVERY']]
    specs=[p['S1_TREND_REBOUND'],session_cfg['common_specification'],v['V1_CLIMAX_RECOVERY']]
    states=[]
    for (key,_),rule,spec in zip(EXPERTS,rules,specs):
        ledger,decisions,cycles=simulate_policy(data,div,cfg,cfg['costs']['BASE'],cfg['earlier_start'],rule,spec)
        save_account(OUT/'earlier_experts',key,ledger,decisions)
        states.append(decision_state(data,decisions))
    return np.column_stack(states)


def freeze():
    require(not CONFIG.exists(),'固定信号组合已经登记，不覆盖')
    old=json.loads((ROOT/'config/510300_simple_price_entry_exit_v1.json').read_text(encoding='utf-8'))
    cfg={k:v for k,v in old.items() if k not in ['candidate_specs','candidate_names','frozen_files']}
    cfg.update({'study_id':'510300_SIMPLE_SIGNAL_BLEND_V1','round':28,'registered_at':now(),'primary':'B1_EQUAL_THREE',
                'candidate_names':NAMES,'candidate_count':8,'earlier_start':'2015-01-05','earlier_terminal':'2019-12-31',
                'rules':'docs/510300_SIMPLE_SIGNAL_BLEND_V1.md','experts':EXPERTS,'expert_state_cost':'BASE',
                'weight_band':.1,'primary_rebalance':'DAILY_CLOSE_TO_NEXT_OPEN','execution_renewal':'未成交指令次日收盘按新目标重新判断',
                'all_candidates_earlier_diagnostic':True})
    paths=[Path(__file__),ROOT/'research/event_clock_account_v1.py',ROOT/'research/simple_price_entry_exit_v1.py',
           ROOT/'research/simple_session_divergence_v1.py',ROOT/'research/simple_volume_reversal_v1.py',ROOT/cfg['rules'],ROOT/cfg['features'],ROOT/cfg['dividends']]
    for key,folder in EXPERTS:
        paths.append(ROOT/'reports/research'/folder/'evaluation/BASE'/f'{key}_decisions.parquet')
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in paths]
    write_json(CONFIG,cfg,exclusive=True)
    print('第28轮8种固定组合与共识规则已登记，全部同时检验较早历史。',flush=True)


def run():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    for p in cfg['frozen_files']:require(digest(ROOT/p['path'])==p['sha256'],'固定组合登记文件发生变化')
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'RUN_STARTED.json',{'started_at':now(),'config_sha256':digest(CONFIG)},exclusive=True)
    data=pd.read_parquet(ROOT/cfg['features']);div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    experts=[]
    for key,folder in EXPERTS:
        dec=pd.read_parquet(ROOT/'reports/research'/folder/'evaluation/BASE'/f'{key}_decisions.parquet')
        experts.append(decision_state(data,dec))
    expert_state=np.column_stack(experts)
    pd.DataFrame({'date':data.date,**{key:expert_state[:,j] for j,(key,_) in enumerate(EXPERTS)}}).to_parquet(OUT/'expert_states.parquet',index=False)
    earlier=data[data.date<=cfg['earlier_terminal']].copy();earlier_state=earlier_experts(earlier,div,cfg)
    metrics=[];diagnostics=[];yearly=[];eras=[]
    for period,frame,state,start,dest in [('evaluation',data,expert_state,cfg['evaluation_start'],metrics),
                                         ('earlier_diagnostic',earlier,earlier_state,cfg['earlier_start'],diagnostics)]:
        all_targets=targets(state,frame.vol20.to_numpy(float))
        pd.DataFrame({'date':frame.date,**all_targets}).to_parquet(OUT/f'{period}_targets.parquet',index=False)
        for cost_id,cost in cfg['costs'].items():
            accounts={}
            for key,target in all_targets.items():
                ledger,decisions=simulate_event_account(frame,div,cfg,cost,start,key,targets=target,event_mask=np.ones(len(frame),bool))
                save_account(OUT/period/cost_id,key,ledger,decisions);accounts[key]=ledger
            if period=='evaluation':
                old=ROOT/'reports/research/510300_simple_price_entry_exit_v1/evaluation'/cost_id/'BUY_HOLD_ledger.parquet'
                benchmark=pd.read_parquet(old)
            else:
                benchmark,bd=simulate_event_account(frame,div,cfg,cost,start,'BUY_HOLD',event_mask=np.ones(len(frame),bool))
                save_account(OUT/period/cost_id,'BUY_HOLD',benchmark,bd)
            base=summarize(benchmark,cfg);accounts['BUY_HOLD']=benchmark
            for key,ledger in accounts.items():
                m={'cost':cost_id,'model':key,'name':NAMES.get(key,'买入持有'),**summarize(ledger,cfg)}
                m['annualized_return_excess_vs_buy_hold']=m['annualized_return']-base['annualized_return']
                m['meets_point_target']=m['net_sharpe'] is not None and m['net_sharpe']>=1.2;dest.append(m)
                if period=='evaluation':
                    for year,g in ledger.groupby(ledger.date.dt.year):yearly.append({'cost':cost_id,'model':key,'year':int(year),**summarize(g,cfg)})
                    for label,left,right in [('2020—2021','2020-01-01','2021-12-31'),('2022—2023','2022-01-01','2023-12-31'),('2024—终点','2024-01-01',cfg['data_cutoff'])]:
                        g=ledger[(ledger.date>=left)&(ledger.date<=right)];eras.append({'cost':cost_id,'model':key,'era':label,**summarize(g,cfg)})
            pd.DataFrame({'date':benchmark.date,**{k:ledger.net_return.to_numpy() for k,ledger in accounts.items()}}).to_parquet(OUT/f'{period}_{cost_id}_returns.parquet',index=False)
            print(f'{period}／{cost_id}：8种完整组合账户已完成。',flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diagnostics).to_csv(OUT/'earlier_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(eras).to_csv(OUT/'era_metrics.csv',index=False,encoding='utf-8-sig')
    best=max((m for m in metrics if m['cost']=='BASE' and m['model']!='BUY_HOLD'),key=lambda m:m['net_sharpe'] if m['net_sharpe'] is not None else -999)
    result={'study_id':cfg['study_id'],'completed_at':now(),'status':'FIXED_SIGNAL_BLENDS_AND_EARLIER_DIAGNOSTICS_COMPLETE',
            'candidate_configurations':8,'evaluation_accounts':18,'new_accounts_generated':16,'reused_control_accounts':2,
            'earlier_diagnostic_accounts':18,'earlier_expert_accounts':3,'all_metrics':metrics,'earlier_diagnostics':diagnostics,
            'primary':[m for m in metrics if m['model']==cfg['primary']],'post_selected_best_base':best,
            'historical_point_target_met':any(m['meets_point_target'] for m in metrics if m['model']!='BUY_HOLD'),
            'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED_EXPERTS_SELECTED_AFTER_OBSERVED_HISTORY','position_impact':0}
    write_json(OUT/'result.json',result,exclusive=True)
    print(json.dumps({'状态':result['status'],'主方案':result['primary'],'主评价最高':best,'更早历史':diagnostics},ensure_ascii=False),flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['freeze']:freeze()
    elif sys.argv[1:]==['run']:run()
    else:raise SystemExit('请指定 freeze 或 run')
