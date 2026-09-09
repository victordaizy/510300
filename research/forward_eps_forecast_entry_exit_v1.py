"""比较两套前瞻EPS预测入场、每日退出及相同预测的月末退出账户。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_corrected_source_pipeline_v2 import empty_cycle_schema
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import summarize,save_account
from research.intraday_overnight_increment_v1 import normalize_dividends,block_indices,return_metrics,interval

CONFIG=ROOT/'config/510300_forward_eps_forecast_entry_exit_v1.json'
MANIFEST=ROOT/'config/510300_forward_eps_forecast_entry_exit_v1_manifest.json'
OUT=ROOT/'reports/research/510300_forward_eps_forecast_entry_exit_v1'
SOURCE=ROOT/'reports/research/510300_forward_eps_monthly_policy_v2_csi'
PARENT=ROOT/'reports/research/510300_adaptive_allocation_v1'
PAIRS=[('Y1_EPS_DAILY_EXIT','Y2_EPS_MONTHLY_EXIT','E3_FORWARD_EPS'),
       ('Y3_PRICE_EPS_DAILY_EXIT','Y4_PRICE_EPS_MONTHLY_EXIT','E1_PRICE_FORWARD_EPS')]


def entry_condition(prediction,trend,mask):
    prediction=np.asarray(prediction,float);trend=np.asarray(trend,float);mask=np.asarray(mask,bool)
    output=np.full(len(prediction),np.nan);valid=mask & np.isfinite(prediction) & np.isfinite(trend)
    output[valid]=((prediction[valid]>0)&(trend[valid]>0)).astype(float)
    return output


def freeze():
    config=read(CONFIG)
    paths=[Path(__file__),CONFIG,ROOT/'docs/510300_FORWARD_EPS_FORECAST_ENTRY_EXIT_V1.md',
           ROOT/'tests/test_forward_eps_forecast_entry_exit_v1.py',ROOT/'research/explicit_entry_exit_account_v1.py',
           ROOT/'research/forward_eps_corrected_source_pipeline_v2.py',ROOT/'research/forward_eps_explicit_entry_exit_v1.py',
           ROOT/'research/forward_eps_residual_policy_v1.py',ROOT/'research/forward_eps_monthly_policy_v1.py',
           ROOT/'research/event_clock_account_v1.py',ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',
           ROOT/'research/financial_annual_components_v1.py',ROOT/'research/forward_eps_guosen_history_v1.py',
           ROOT/'config/510300_forward_eps_corrected_source_pipeline_v2_manifest.json',SOURCE/'signals.parquet',SOURCE/'result.json',
           PARENT/'features.parquet',ROOT/config['inputs']['dividends']]
    for cost in config['costs']:paths.append(SOURCE/'evaluation'/cost/'BUY_HOLD_ledger.parquet')
    save(MANIFEST,{'registered_at':now(),'candidate_configurations':4,'planned_evaluation_accounts':10,
                   'both_input_model_results_already_observed':True,'new_models_fit':0,'source_predictions_not_rewritten':True,
                   'risk_exit_thresholds_unchanged':True,'files':[identity(path) for path in paths]},exclusive=True)
    print('两套预测的入场、每日退出与月末退出对照共十条账户已登记。',flush=True)


def run():
    config=read(CONFIG)
    for row in read(MANIFEST)['files']:
        if identity(ROOT/row['path'])['sha256']!=row['sha256']:raise ValueError('冻结输入或方法变化')
    if (OUT/'RUN_STARTED.json').exists():raise FileExistsError('本轮已启动，先检查原进程，不重复运行')
    OUT.mkdir(parents=True,exist_ok=True);save(OUT/'RUN_STARTED.json',{'started_at':now(),'manifest':identity(MANIFEST)},exclusive=True)
    data=pd.read_parquet(PARENT/'features.parquet');signals=pd.read_parquet(SOURCE/'signals.parquet')
    assert data.date.tolist()==signals.date.tolist();mask=signals.event_mask.to_numpy(bool)
    dividends=normalize_dividends(pd.read_csv(ROOT/config['inputs']['dividends']))
    entries={prediction:entry_condition(signals[prediction],data.sma120,mask) for daily,monthly,prediction in PAIRS}
    pd.DataFrame({'date':data.date,'event_mask':mask,**entries}).to_parquet(OUT/'explicit_entry_signals.parquet',index=False)
    metrics=[];yearly=[];eras=[];uncertainty={};cycle_counts={};checks=[]
    for cost_name,cost in config['costs'].items():
        accounts={}
        for daily,monthly,prediction in PAIRS:
            entry=entries[prediction]
            ledger,decisions,cycles=empty_cycle_schema(data,dividends,config,cost,config['evaluation_start'],entry,mask)
            cycles.to_csv(OUT/f'{cost_name}_{daily}_逐次完整进出场.csv',index=False,encoding='utf-8-sig')
            decisions.loc[decisions.new_exit_trigger].to_csv(OUT/f'{cost_name}_{daily}_退出首次触发.csv',index=False,encoding='utf-8-sig')
            allowed=set(signals.loc[signals.event_mask].index)
            assert decisions.loc[~decisions.origin_index.isin(allowed),'requested_quantity'].le(0).all()
            assert decisions.loc[decisions.requested_quantity.gt(0),'entry_signal'].eq(1).all()
            assert not ledger.terminal_unliquidated.iloc[-1] and cycles.exit_date.notna().all()
            np.testing.assert_allclose(cycles.cycle_net_profit_cny.sum(),ledger.pnl.sum(),atol=1e-6,rtol=0)
            save_account(OUT/'evaluation'/cost_name,daily,ledger,decisions);accounts[daily]=ledger
            cycle_counts[cost_name+'_'+daily]=len(cycles)
            checks.append({'cost':cost_name,'model':daily,'entry_calendar_and_signal_checked':True,'all_cycles_closed':True,'cycle_pnl_matches_account':True})
            ledger,decisions=simulate_event_account(data,dividends,config,cost,config['evaluation_start'],monthly,targets=entry,horizon=60,event_mask=mask)
            save_account(OUT/'evaluation'/cost_name,monthly,ledger,decisions);accounts[monthly]=ledger
        ledger,decisions=simulate_event_account(data,dividends,config,cost,config['evaluation_start'],'BUY_HOLD',horizon=60,event_mask=mask)
        original=pd.read_parquet(SOURCE/'evaluation'/cost_name/'BUY_HOLD_ledger.parquet')
        np.testing.assert_array_equal(ledger.equity,original.equity)
        save_account(OUT/'evaluation'/cost_name,'BUY_HOLD',ledger,decisions);accounts['BUY_HOLD']=ledger
        benchmark=summarize(ledger,config)
        for model,ledger in accounts.items():
            metric={'cost':cost_name,'model':model,**summarize(ledger,config)}
            metric['annualized_return_excess_vs_buy_hold']=metric['annualized_return']-benchmark['annualized_return']
            metric['meets_point_target']=metric['net_sharpe'] is not None and metric['net_sharpe']>=1.2;metrics.append(metric)
            for year,g in ledger.groupby(ledger.date.dt.year):yearly.append({'cost':cost_name,'model':model,'year':int(year),**summarize(g,config)})
            for name,a,b in [('2020—2021','2020-01-01','2021-12-31'),('2022—2023','2022-01-01','2023-12-31'),('2024—终点','2024-01-01',config['data_cutoff'])]:
                eras.append({'cost':cost_name,'model':model,'era':name,**summarize(ledger.loc[ledger.date.between(a,b)],config)})
        returns=pd.DataFrame({'date':accounts['BUY_HOLD'].date,**{model:ledger.net_return.to_numpy() for model,ledger in accounts.items()}})
        returns.to_parquet(OUT/f'{cost_name}_all_evaluation_returns.parquet',index=False);uncertainty[cost_name]={}
        for block in config['bootstrap_day_blocks']:
            rng=np.random.default_rng(config['random_seed']);draws=[]
            for _ in range(config['bootstrap_repetitions']):
                ix=block_indices(rng,len(returns),block);draw={}
                for daily,monthly,prediction in PAIRS:
                    a=returns[daily].to_numpy()[ix]
                    draw[daily+'_sharpe']=return_metrics(a,config['annual_days'])['net_sharpe']
                    draw[daily+'_minus_monthly']=float((a-returns[monthly].to_numpy()[ix]).mean()*config['annual_days'])
                    draw[daily+'_minus_buy_hold']=float((a-returns.BUY_HOLD.to_numpy()[ix]).mean()*config['annual_days'])
                draws.append(draw)
            saved=pd.DataFrame(draws);saved.to_parquet(OUT/f'{cost_name}_block{block}_saved_bootstrap_statistics.parquet',index=False)
            uncertainty[cost_name][str(block)]={key+'_95_interval':interval(saved[key].dropna().tolist()) for key in saved}
        print('预测入场与退出对照已完成',cost_name,'五条完整账户',flush=True)
    frame=pd.DataFrame(metrics);frame.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(eras).to_csv(OUT/'era_metrics.csv',index=False,encoding='utf-8-sig')
    save(OUT/'execution_checks.json',{'rows':checks},exclusive=True);save(OUT/'uncertainty.json',uncertainty,exclusive=True)
    result={'study_id':config['study_id'],'completed_at':now(),'status':'FORWARD_EPS_FORECAST_ENTRY_AND_EXIT_ACCOUNTS_COMPLETE',
            'primary':frame.loc[frame.model.eq(PAIRS[0][0])].to_dict('records'),'all_metrics':metrics,'uncertainty':uncertainty,
            'candidate_configurations':4,'evaluation_accounts':10,'new_models_fit':0,'position_cycles':cycle_counts,
            'historical_point_target_met':bool(frame.loc[frame.model.eq(PAIRS[0][0]),'meets_point_target'].any()),
            'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY','position_impact':0}
    save(OUT/'result.json',result,exclusive=True);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--run',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else run()
