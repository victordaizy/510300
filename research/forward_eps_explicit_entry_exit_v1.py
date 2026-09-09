"""按固定入场、每日退出和再入场规则检验前瞻EPS账户。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_guosen_history_v1 import identity
from research.explicit_entry_exit_account_v1 import simulate_entry_exit_account
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import summarize,save_account
from research.intraday_overnight_increment_v1 import normalize_dividends,block_indices,return_metrics,interval

CONFIG=ROOT/'config/510300_forward_eps_explicit_entry_exit_v1.json'
MANIFEST=ROOT/'config/510300_forward_eps_explicit_entry_exit_v1_manifest.json'
OUT=ROOT/'reports/research/510300_forward_eps_explicit_entry_exit_v1'
SOURCE=ROOT/'reports/research/510300_forward_eps_monthly_policy_v1_csi'
PARENT=ROOT/'reports/research/510300_adaptive_allocation_v1'
PRIMARY='X1_FORWARD_EPS_DAILY_EXIT'
BASELINE='X2_FORWARD_EPS_MONTHLY_EXIT_ONLY'
ORIGINAL='E4_EARNINGS_AND_PRICE_REGIME'


def freeze():
    config=read(CONFIG)
    files=[Path(__file__),CONFIG,ROOT/'research/explicit_entry_exit_account_v1.py',ROOT/'tests/test_explicit_entry_exit_account_v1.py',
           ROOT/'docs/510300_FORWARD_EPS_EXPLICIT_ENTRY_EXIT_V1.md',ROOT/'docs/510300_ENTRY_EXIT_AND_NO_GPT_PACKAGE_20260906.md',
           ROOT/'research/event_clock_account_v1.py',ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',
           ROOT/'research/financial_annual_components_v1.py',ROOT/'research/forward_eps_guosen_history_v1.py',
           ROOT/'config/510300_forward_eps_monthly_policy_v1_manifest.json',ROOT/'config/510300_research_authority_v6.json',
           PARENT/'features.parquet',ROOT/config['inputs']['dividends']]
    save(MANIFEST,{'registered_at':now(),'candidate_configurations':2,'planned_evaluation_accounts':6,
                   'original_monthly_and_buy_hold_controls_to_be_reproduced':True,'financial_results_already_observed':True,
                   'csi_round11_result_exists_at_registration':(SOURCE/'result.json').exists(),'files':[identity(path) for path in files]},exclusive=True)
    print('前瞻EPS明确进出场规则及六条账户已登记。',flush=True)


def run():
    config=read(CONFIG)
    for item in read(MANIFEST)['files']:
        if identity(ROOT/item['path'])['sha256']!=item['sha256']:raise ValueError('冻结方法或输入改变')
    if not (SOURCE/'result.json').exists():raise RuntimeError('原沪深300账户尚未完成，继续承接现有程序')
    if (OUT/'RUN_STARTED.json').exists():raise FileExistsError('本研究已开始，禁止重复启动')
    OUT.mkdir(parents=True,exist_ok=True);save(OUT/'RUN_STARTED.json',{'started_at':now(),'manifest':identity(MANIFEST)},exclusive=True)
    direct=[SOURCE/'result.json',SOURCE/'signals.parquet',SOURCE/'source_receipt.json',SOURCE/'monthly_features_and_mature_labels.parquet']
    for cost in config['costs']:
        for model in [ORIGINAL,'BUY_HOLD']:direct.append(SOURCE/'evaluation'/cost/(model+'_ledger.parquet'))
    save(OUT/'input_receipt.json',{'recorded_at':now(),'files':[identity(path) for path in direct]},exclusive=True)
    data=pd.read_parquet(PARENT/'features.parquet');dividends=normalize_dividends(pd.read_csv(ROOT/config['inputs']['dividends']))
    signals=pd.read_parquet(SOURCE/'signals.parquet');assert signals.date.tolist()==data.date.tolist()
    entry=signals[ORIGINAL].to_numpy(float);mask=signals.event_mask.to_numpy(bool)
    assert np.isin(entry[np.isfinite(entry)],[0.,1.]).all()
    metrics=[];yearly=[];eras=[];checks=[];uncertainty={};cycle_counts={};trigger_counts={}
    for cost_name,cost in config['costs'].items():
        accounts={}
        for model in [PRIMARY,BASELINE,'BUY_HOLD']:
            if model==PRIMARY:
                ledger,decisions,cycles=simulate_entry_exit_account(data,dividends,config,cost,config['evaluation_start'],entry,mask)
                cycles.to_csv(OUT/f'{cost_name}_逐次完整进出场.csv',index=False,encoding='utf-8-sig')
                triggers=decisions.loc[decisions.new_exit_trigger]
                triggers.to_csv(OUT/f'{cost_name}_退出首次触发.csv',index=False,encoding='utf-8-sig')
                cycle_counts[cost_name]=len(cycles)
                trigger_counts[cost_name]=triggers.exit_reasons.value_counts().to_dict()
                allowed=set(signals.loc[signals.event_mask].index)
                assert decisions.loc[~decisions.origin_index.isin(allowed),'requested_quantity'].le(0).all()
                assert decisions.loc[decisions.requested_quantity.gt(0),'entry_signal'].eq(1).all()
                assert not ledger.terminal_unliquidated.iloc[-1]
                assert cycles.exit_date.notna().all()
                np.testing.assert_allclose(cycles.cycle_net_profit_cny.sum(),ledger.pnl.sum(),atol=1e-6,rtol=0)
                checks.append({'cost':cost_name,'model':model,'entries_only_at_eligible_month_end':True,'all_cycles_closed':True,'cycle_pnl_matches_complete_account':True})
            else:
                ledger,decisions=simulate_event_account(data,dividends,config,cost,config['evaluation_start'],model,
                                                       targets=entry if model==BASELINE else None,horizon=60,event_mask=mask)
                source_model=ORIGINAL if model==BASELINE else 'BUY_HOLD'
                previous=pd.read_parquet(SOURCE/'evaluation'/cost_name/(source_model+'_ledger.parquet'))
                for field in ['equity','shares','cash','net_return','commission','slippage_cost','dividend_receivable']:
                    np.testing.assert_array_equal(ledger[field],previous[field])
                checks.append({'cost':cost_name,'model':model,'original_control_parity':True})
            save_account(OUT/'evaluation'/cost_name,model,ledger,decisions);accounts[model]=ledger
        benchmark=summarize(accounts['BUY_HOLD'],config)
        for model,ledger in accounts.items():
            row={'cost':cost_name,'model':model,**summarize(ledger,config)}
            row['annualized_return_excess_vs_buy_hold']=row['annualized_return']-benchmark['annualized_return']
            row['meets_point_target']=row['net_sharpe'] is not None and row['net_sharpe']>=1.2;metrics.append(row)
            for year,g in ledger.groupby(ledger.date.dt.year):yearly.append({'cost':cost_name,'model':model,'year':int(year),**summarize(g,config)})
            for label,a,b in [('2020—2021','2020-01-01','2021-12-31'),('2022—2023','2022-01-01','2023-12-31'),('2024—终点','2024-01-01',config['data_cutoff'])]:
                eras.append({'cost':cost_name,'model':model,'era':label,**summarize(ledger.loc[ledger.date.between(a,b)],config)})
        returns=pd.DataFrame({'date':accounts['BUY_HOLD'].date,**{model:ledger.net_return.to_numpy() for model,ledger in accounts.items()}})
        returns.to_parquet(OUT/f'{cost_name}_all_evaluation_returns.parquet',index=False);uncertainty[cost_name]={}
        for block in config['bootstrap_day_blocks']:
            rng=np.random.default_rng(config['random_seed']);samples=[]
            for _ in range(config['bootstrap_repetitions']):
                ix=block_indices(rng,len(returns),block);a=returns[PRIMARY].to_numpy()[ix]
                samples.append({'primary_sharpe':return_metrics(a,config['annual_days'])['net_sharpe'],
                                'increment_vs_monthly_exit':float((a-returns[BASELINE].to_numpy()[ix]).mean()*config['annual_days']),
                                'excess_vs_buy_hold':float((a-returns.BUY_HOLD.to_numpy()[ix]).mean()*config['annual_days'])})
            saved=pd.DataFrame(samples);saved.to_parquet(OUT/f'{cost_name}_block{block}_saved_bootstrap_statistics.parquet',index=False)
            uncertainty[cost_name][str(block)]={key+'_95_interval':interval(saved[key].dropna().tolist()) for key in saved}
        print('明确进出场规则的完整账户完成',cost_name,'三条账户',flush=True)
    frame=pd.DataFrame(metrics);frame.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(eras).to_csv(OUT/'era_metrics.csv',index=False,encoding='utf-8-sig')
    save(OUT/'execution_checks.json',{'rows':checks},exclusive=True);save(OUT/'uncertainty.json',uncertainty,exclusive=True)
    result={'study_id':config['study_id'],'completed_at':now(),'status':'FORWARD_EPS_EXPLICIT_ENTRY_EXIT_COMPLETE',
            'primary':frame.loc[frame.model.eq(PRIMARY)].to_dict('records'),'all_metrics':metrics,'uncertainty':uncertainty,
            'candidate_configurations':2,'evaluation_accounts':6,'new_models_fit':0,'position_cycles':cycle_counts,'new_exit_trigger_counts':trigger_counts,
            'historical_point_target_met':bool(frame.loc[frame.model.eq(PRIMARY),'meets_point_target'].any()),'goal_achieved':False,
            'independent_validation':'NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY','position_impact':0}
    save(OUT/'result.json',result,exclusive=True);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--run',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else run()
