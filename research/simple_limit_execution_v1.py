"""保持已有信号和退出规则，检验事先挂限价是否改善完整账户。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.limit_entry_account_v1 import simulate_limit_policy
from research.simple_price_entry_exit_v1 import signals, specifications as price_specs
from research.simple_volume_reversal_v1 import make_rules as volume_rules, specifications as volume_specs
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_simple_limit_execution_v1'
CONFIG=ROOT/'config/510300_simple_limit_execution_v1.json'
SIGNALS={'S1_TREND_REBOUND':'趋势突破与震荡反弹切换','R2_Z_CONFIRM':'偏离均值后的首日回升','V1_CLIMAX_RECOVERY':'急跌后放量收强'}


def candidates():
    return [{'id':f'{key}_D{int(discount*10000)}_L{days}','signal':key,'discount':discount,'valid_days':days}
            for key in SIGNALS for discount in [.005,.01] for days in [1,3]]


def source_rules(data):
    all_rules=signals(data)
    all_rules.update(volume_rules(data)[0])
    return {key:all_rules[key] for key in SIGNALS}


def source_folder(key):
    return '510300_simple_volume_reversal_v1' if key.startswith('V') else '510300_simple_price_entry_exit_v1'


def freeze():
    require(not CONFIG.exists(),'限价执行研究已登记，不覆盖')
    old=json.loads((ROOT/'config/510300_simple_price_entry_exit_v1.json').read_text(encoding='utf-8'))
    cfg={k:v for k,v in old.items() if k not in ['candidate_names','candidate_specs','frozen_files']}
    all_specs=price_specs();all_specs.update(volume_specs())
    cfg.update({'study_id':'510300_SIMPLE_LIMIT_EXECUTION_V1','round':27,'registered_at':now(),'primary':'R2_Z_CONFIRM_D50_L1',
                'primary_execution_assumption':'OPEN_ONLY','candidates':candidates(),'candidate_count':12,'signal_names':SIGNALS,
                'signal_specs':{k:all_specs[k] for k in SIGNALS},'assumptions':['OPEN_ONLY','PENETRATION'],
                'rules':'docs/510300_SIMPLE_LIMIT_EXECUTION_V1.md','earlier_start':'2015-01-05','earlier_terminal':'2019-12-31',
                'earlier_diagnostic_selection':'每种成交假设按主评价基础夏普选择最高设置，原参数检验更早历史；不构成独立验证',
                'fill_evidence':'DAILY_BAR_EXECUTION_MODEL_NOT_OBSERVED_FILL'})
    paths=[Path(__file__),ROOT/'research/limit_entry_account_v1.py',ROOT/'research/simple_price_entry_exit_v1.py',
           ROOT/'research/simple_volume_reversal_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',ROOT/'tests/test_limit_entry_account_v1.py',
           ROOT/cfg['rules'],ROOT/cfg['features'],ROOT/cfg['dividends']]
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in paths]
    write_json(CONFIG,cfg,exclusive=True)
    print('第27轮12种限价设置和两种成交假设已登记；保持原信号与退出参数。',flush=True)


def metric_row(ledger,cfg,cost,key,assumption,kind):
    return {'cost':cost,'model':key,'assumption':assumption,'kind':kind,**summarize(ledger,cfg)}


def run():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    for f in cfg['frozen_files']:require(digest(ROOT/f['path'])==f['sha256'],'限价研究的登记文件变化')
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'RUN_STARTED.json',{'started_at':now(),'config_sha256':digest(CONFIG)},exclusive=True)
    data=pd.read_parquet(ROOT/cfg['features']);div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    rules=source_rules(data);metrics=[];yearly=[];eras=[];all_order_statistics=[]
    for assumption in cfg['assumptions']:
        for cost_id,cost in cfg['costs'].items():
            for i,item in enumerate(cfg['candidates'],1):
                key=item['id'];signal=item['signal'];folder=OUT/'evaluation'/assumption/cost_id
                ledger,decisions,cycles,orders=simulate_limit_policy(data,div,cfg,cost,cfg['evaluation_start'],rules[signal],cfg['signal_specs'][signal],item,assumption)
                save_account(folder,key,ledger,decisions)
                cycles.to_csv(folder/f'{key}_cycles.csv',index=False,encoding='utf-8-sig')
                orders.to_csv(folder/f'{key}_orders.csv',index=False,encoding='utf-8-sig')
                m=metric_row(ledger,cfg,cost_id,key,assumption,'LIMIT_POLICY');metrics.append(m)
                stat={'assumption':assumption,'cost':cost_id,'model':key,'orders_created':len(orders),
                      'filled_orders':int((orders.final_status=='SIMULATED_FILL').sum()) if len(orders) else 0,
                      'intraday_conditional_fills':int((ledger.execution_clock=='INTRADAY_TIME_UNOBSERVED_CONDITIONAL_SIMULATION').sum()),
                      'expired_orders':int((orders.final_status=='EXPIRED_WITHOUT_FILL').sum()) if len(orders) else 0,
                      'invalidated_orders':int(orders.final_status.str.startswith('CANCELLED').sum()) if len(orders) else 0,
                      'observed_real_fills':0}
                all_order_statistics.append(stat)
                for year,g in ledger.groupby(ledger.date.dt.year):yearly.append({'year':int(year),**metric_row(g,cfg,cost_id,key,assumption,'LIMIT_POLICY')})
                for label,start,end in [('2020—2021','2020-01-01','2021-12-31'),('2022—2023','2022-01-01','2023-12-31'),('2024—终点','2024-01-01',cfg['data_cutoff'])]:
                    g=ledger[(ledger.date>=start)&(ledger.date<=end)];eras.append({'era':label,**metric_row(g,cfg,cost_id,key,assumption,'LIMIT_POLICY')})
                if i%4==0:print(f'{assumption}／{cost_id}：已完成 {i}/12 个限价完整账户。',flush=True)
    for cost_id in cfg['costs']:
        for signal in list(SIGNALS)+['BUY_HOLD']:
            p=ROOT/'reports/research'/source_folder(signal)/'evaluation'/cost_id/f'{signal}_ledger.parquet'
            ledger=pd.read_parquet(p)
            metrics.append(metric_row(ledger,cfg,cost_id,signal,'LEGACY_OPEN','REUSED_CONTROL'))
    benchmark={m['cost']:m for m in metrics if m['model']=='BUY_HOLD'}
    for m in metrics:
        m['annualized_return_excess_vs_buy_hold']=m['annualized_return']-benchmark[m['cost']]['annualized_return']
        m['meets_point_target']=m['net_sharpe'] is not None and m['net_sharpe']>=1.2
    best_by_assumption={}
    for assumption in cfg['assumptions']:
        eligible=[m for m in metrics if m['assumption']==assumption and m['cost']=='BASE' and m['net_sharpe'] is not None]
        best_by_assumption[assumption]=max(eligible,key=lambda m:m['net_sharpe']) if eligible else None
    earlier=data[data.date<=cfg['earlier_terminal']].copy();early_rules=source_rules(earlier);diagnostics=[]
    for assumption,best in best_by_assumption.items():
        if best is None:continue
        item=next(c for c in cfg['candidates'] if c['id']==best['model']);signal=item['signal']
        for cost_id,cost in cfg['costs'].items():
            ledger,decisions,cycles,orders=simulate_limit_policy(earlier,div,cfg,cost,cfg['earlier_start'],early_rules[signal],cfg['signal_specs'][signal],item,assumption)
            folder=OUT/'earlier_diagnostic'/assumption/cost_id;save_account(folder,item['id'],ledger,decisions)
            cycles.to_csv(folder/f'{item["id"]}_cycles.csv',index=False,encoding='utf-8-sig')
            orders.to_csv(folder/f'{item["id"]}_orders.csv',index=False,encoding='utf-8-sig')
            diagnostics.append(metric_row(ledger,cfg,cost_id,item['id'],assumption,'EARLIER_POST_SELECTION_DIAGNOSTIC'))
    for cost_id,cost in cfg['costs'].items():
        ledger,decisions=simulate_event_account(earlier,div,cfg,cost,cfg['earlier_start'],'BUY_HOLD',event_mask=np.ones(len(earlier),bool))
        save_account(OUT/'earlier_diagnostic'/'BENCHMARK'/cost_id,'BUY_HOLD',ledger,decisions)
        diagnostics.append(metric_row(ledger,cfg,cost_id,'BUY_HOLD','LEGACY_OPEN','EARLIER_BENCHMARK'))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(eras).to_csv(OUT/'era_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(all_order_statistics).to_csv(OUT/'order_statistics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diagnostics).to_csv(OUT/'earlier_diagnostics.csv',index=False,encoding='utf-8-sig')
    result={'study_id':cfg['study_id'],'completed_at':now(),'status':'LIMIT_ENTRY_FULL_ACCOUNTS_AND_EARLIER_DIAGNOSTICS_COMPLETE',
            'candidate_configurations':12,'execution_assumption_scenarios':2,'evaluation_accounts':56,'new_accounts_generated':48,
            'reused_control_accounts':8,'earlier_diagnostic_accounts':len(diagnostics),'all_metrics':metrics,
            'primary':[m for m in metrics if m['model']==cfg['primary'] and m['assumption']==cfg['primary_execution_assumption']],
            'post_selected_best_by_assumption':best_by_assumption,'post_selected_best_base':best_by_assumption['OPEN_ONLY'],
            'historical_point_target_met':any(m['meets_point_target'] for m in metrics if m['kind']=='LIMIT_POLICY'),
            'earlier_diagnostics':diagnostics,'goal_achieved':False,'position_impact':0,
            'independent_validation':'NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY_AND_UNVERIFIED_EXECUTION',
            'intraday_simulation_is_not_observed_fill':True}
    write_json(OUT/'result.json',result,exclusive=True)
    print(json.dumps({'状态':result['status'],'两种假设的最好结果':best_by_assumption,'更早历史':diagnostics},ensure_ascii=False),flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['freeze']:freeze()
    elif sys.argv[1:]==['run']:run()
    else:raise SystemExit('请指定 freeze 或 run')
