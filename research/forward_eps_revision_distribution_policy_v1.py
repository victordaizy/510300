"""盈利修正分布的逐月预测、完整账户及协调进出场比较。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_revision_distribution_features_v1 import EPS_FEATURES,build_features
from research.forward_eps_monthly_policy_v1 import PRICE,mature_training
from research.event_clock_account_v1 import simulate_event_account
from research.forward_eps_utility_exit_account_v2 import simulate_utility_exit_account
from research.adaptive_allocation_v1 import save_account,summarize
from research.intraday_overnight_increment_v1 import normalize_dividends,block_indices,return_metrics,interval

CONFIG=ROOT/'config/510300_forward_eps_revision_distribution_policy_v1.json'
MANIFEST=ROOT/'config/510300_forward_eps_revision_distribution_policy_v1_manifest.json'
OUT=ROOT/'reports/research/510300_forward_eps_revision_distribution_policy_v1'
SOURCE=ROOT/'reports/research/510300_forward_eps_monthly_policy_v2_csi'
FEATURE_SOURCE=ROOT/'reports/research/510300_forward_eps_revision_diagnostic_v1'
PARENT=ROOT/'reports/research/510300_adaptive_allocation_v1'
MODELS={'D1_EPS_DISTRIBUTION':EPS_FEATURES,'D2_PRICE_EPS_DISTRIBUTION':PRICE+EPS_FEATURES}
COHERENT='D3_EPS_DISTRIBUTION_COHERENT_TREND'
CONTROLS={'C0_ORIGINAL_EPS':'E3_FORWARD_EPS','C1_ORIGINAL_PRICE_EPS':'E1_PRICE_FORWARD_EPS',
          'C2_MATCHED_PRICE':'E2_MATCHED_PRICE','BUY_HOLD':'BUY_HOLD'}
PAIRS={'D1_EPS_DISTRIBUTION':'C0_ORIGINAL_EPS','D2_PRICE_EPS_DISTRIBUTION':'C1_ORIGINAL_PRICE_EPS',
       COHERENT:'D1_EPS_DISTRIBUTION'}


def freeze():
    config=read(ROOT/'config/510300_forward_eps_monthly_policy_v2.json')
    config.update({'study_id':'510300_FORWARD_EPS_REVISION_DISTRIBUTION_POLICY_V1',
                   'primary':'D1_EPS_DISTRIBUTION','models':MODELS,'forecast_validity_trading_days':60,
                   'bootstrap_day_blocks':[20,60],'candidate_configurations':3,
                   'planned_evaluation_accounts':14,'planned_new_accounts':6,'reused_control_accounts':8,
                   'prepare_gpt_numerical_review_package':False,
                   'coherent_trend_entry_and_reentry_required':True,'old_eps_results_already_observed':True})
    save(CONFIG,config,exclusive=True)
    paths=[CONFIG,Path(__file__),ROOT/'docs/510300_FORWARD_EPS_REVISION_DISTRIBUTION_POLICY_V1.md',
           ROOT/'research/forward_eps_revision_distribution_features_v1.py',
           ROOT/'research/forward_eps_utility_exit_account_v2.py',
           ROOT/'tests/test_forward_eps_revision_distribution_features_v1.py',
           ROOT/'tests/test_forward_eps_utility_exit_account_v2.py',
           ROOT/'research/event_clock_account_v1.py',ROOT/'research/adaptive_allocation_v1.py',
           ROOT/'research/intraday_overnight_increment_v1.py',ROOT/'research/forward_eps_monthly_policy_v1.py',
           ROOT/'research/financial_annual_components_v1.py',ROOT/'research/forward_eps_guosen_history_v1.py',
           ROOT/'config/510300_research_authority_v6.json',PARENT/'features.parquet',ROOT/config['inputs']['dividends'],
           FEATURE_SOURCE/'result.json',FEATURE_SOURCE/'monthly_distribution_features.parquet',
           SOURCE/'company_month_feature_evidence.parquet',SOURCE/'monthly_forward_eps_features.parquet',
           SOURCE/'monthly_features_and_mature_labels.parquet',SOURCE/'signals.parquet',SOURCE/'result.json']
    for cost in config['costs']:
        for original in CONTROLS.values():
            paths.extend([SOURCE/'evaluation'/cost/(original+'_ledger.parquet'),
                          SOURCE/'evaluation'/cost/(original+'_decisions.parquet')])
    save(MANIFEST,{'registered_at':now(),'candidate_configurations':3,'planned_evaluation_accounts':14,
                   'planned_new_accounts':6,'reused_control_accounts':8,'new_candidate_returns_read_before_freeze':False,
                   'factor_source_distribution_inspected_before_design':True,'old_model_history_already_observed':True,
                   'files':[identity(p) for p in paths]},exclusive=True)
    print('第十六轮三种候选已登记：生成六条新账户，复用八条对照。',flush=True)


def run():
    config=read(CONFIG)
    for item in read(MANIFEST)['files']:
        if identity(ROOT/item['path'])['sha256']!=item['sha256']:
            raise ValueError('已登记输入或方法变化')
    OUT.mkdir(parents=True,exist_ok=False)
    save(OUT/'RUN_STARTED.json',{'started_at':now(),'manifest':identity(MANIFEST)},exclusive=True)
    data=pd.read_parquet(PARENT/'features.parquet')
    monthly=pd.read_parquet(FEATURE_SOURCE/'monthly_distribution_features.parquet')
    reconstructed=build_features(pd.read_parquet(SOURCE/'company_month_feature_evidence.parquet'),
                                 pd.read_parquet(SOURCE/'monthly_forward_eps_features.parquet'))
    pd.testing.assert_frame_equal(monthly,reconstructed)
    labels=pd.read_parquet(SOURCE/'monthly_features_and_mature_labels.parquet')
    monthly=monthly.merge(labels[['origin','origin_index','Y60','label_exit_date',*PRICE]],on='origin',validate='one_to_one')
    monthly['all_features_valid']=monthly.distribution_features_valid & np.isfinite(monthly[PRICE]).all(axis=1)
    np.testing.assert_array_equal(monthly.all_features_valid,labels.all_features_valid)
    for field in ['forward_eps_growth_median','reported_forward_earnings_yield_median','Y60','origin_index']:
        np.testing.assert_allclose(monthly[field],labels[field],atol=0,rtol=0,equal_nan=True)
    monthly.to_parquet(OUT/'monthly_features_and_mature_labels.parquet',index=False)
    originals=pd.read_parquet(SOURCE/'signals.parquet')
    assert data.date.tolist()==originals.date.tolist()
    mask=originals.event_mask.to_numpy(bool)
    events=monthly.loc[monthly.origin_index.isin(np.flatnonzero(mask))]
    assert len(events)==int(mask.sum())
    predictions={model:np.full(len(data),np.nan) for model in MODELS}
    receipts=[]
    for row in events.itertuples():
        t=int(row.origin_index)
        train=mature_training(monthly,row.origin)
        for key,columns in MODELS.items():
            valid=bool(row.all_features_valid) and len(train)>=config['minimum_train_samples']
            receipt={'model':key,'origin':str(row.origin.date()),'training_months':train.origin.dt.strftime('%Y-%m-%d').tolist(),
                     'training_samples':len(train),'latest_label_exit_date':str(train.label_exit_date.max()) if len(train) else None,
                     'columns':columns,'status':'TRAINED_MONTHLY_MODEL' if valid else 'NO_VIEW_FEATURES_OR_MATURE_MONTHS_INSUFFICIENT',
                     'prediction':None}
            if valid:
                assert train.origin.lt(row.origin).all() and train.label_exit_date.le(row.origin).all()
                model=make_pipeline(StandardScaler(),Ridge(alpha=config['ridge_alpha']))
                model.fit(train[columns],train.Y60)
                sample=pd.DataFrame([{name:getattr(row,name) for name in columns}],columns=columns)
                value=float(model.predict(sample)[0]);predictions[key][t]=value
                path=OUT/'fitted_models'/f'{key}_{row.origin:%Y%m%d}.joblib'
                path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,path)
                receipt.update({'prediction':value,'model_file':identity(path)})
            receipts.append(receipt)
    save(OUT/'training_receipts.json',{'rows':receipts},exclusive=True)
    pd.DataFrame({'date':data.date,'event_mask':mask,**predictions}).to_parquet(OUT/'signals.parquet',index=False)
    print('两种盈利分布模型的逐月训练完成，已保存拟合数量：',sum(r['status']=='TRAINED_MONTHLY_MODEL' for r in receipts),flush=True)
    dividends=normalize_dividends(pd.read_csv(ROOT/config['inputs']['dividends']))
    metrics,yearly,eras,checks,uncertainty,trigger_counts=[],[],[],[],{},{}
    for cost_name,cost in config['costs'].items():
        accounts={}
        for model in [*MODELS,COHERENT]:
            if model==COHERENT:
                ledger,decisions=simulate_utility_exit_account(data,dividends,config,cost,predictions['D1_EPS_DISTRIBUTION'],mask,
                                                               expiry_enabled=True,trend_enabled=True,entry_trend_required=True)
                buys=decisions.loc[decisions.requested_quantity.gt(0),'origin_index'].to_numpy(int)
                assert (data.sma120.iloc[buys]>0).all()
                selected=decisions.loc[decisions.new_exit_trigger]
                selected.to_csv(OUT/f'{cost_name}_一致趋势退出首次触发.csv',index=False,encoding='utf-8-sig')
                trigger_counts[cost_name]=selected.exit_reasons.value_counts().to_dict()
            else:
                ledger,decisions=simulate_event_account(data,dividends,config,cost,config['evaluation_start'],model,
                                                        prediction=predictions[model],horizon=60,event_mask=mask)
            outside=decisions.loc[~decisions.origin_index.isin(np.flatnonzero(mask))]
            assert outside.requested_quantity.le(0).all()
            assert not ledger.terminal_unliquidated.iloc[-1]
            ledger.loc[ledger.filled_quantity.ne(0)].to_csv(OUT/f'{cost_name}_{model}_全部进出场成交.csv',index=False,encoding='utf-8-sig')
            save_account(OUT/'evaluation'/cost_name,model,ledger,decisions);accounts[model]=ledger
            checks.append({'cost':cost_name,'model':model,'new_account':True,'outside_month_end_new_buys':0,
                           'terminal_unliquidated':False,'max_accounting_error':float(ledger.accounting_error.abs().max())})
        for name,original in CONTROLS.items():
            ledger=pd.read_parquet(SOURCE/'evaluation'/cost_name/(original+'_ledger.parquet'))
            decisions=pd.read_parquet(SOURCE/'evaluation'/cost_name/(original+'_decisions.parquet'))
            save_account(OUT/'evaluation'/cost_name,name,ledger,decisions);accounts[name]=ledger
            checks.append({'cost':cost_name,'model':name,'new_account':False,'reused_saved_model':original})
        benchmark=summarize(accounts['BUY_HOLD'],config)
        for model,ledger in accounts.items():
            metric={'cost':cost_name,'model':model,**summarize(ledger,config)}
            metric['annualized_return_excess_vs_buy_hold']=metric['annualized_return']-benchmark['annualized_return']
            metric['meets_point_target']=metric['net_sharpe'] is not None and metric['net_sharpe']>=1.2
            metrics.append(metric)
            for year,g in ledger.groupby(ledger.date.dt.year):
                yearly.append({'cost':cost_name,'model':model,'year':int(year),**summarize(g,config)})
            for label,a,b in [('2020—2021','2020-01-01','2021-12-31'),('2022—2023','2022-01-01','2023-12-31'),
                              ('2024—终点','2024-01-01',config['data_cutoff'])]:
                eras.append({'cost':cost_name,'model':model,'era':label,**summarize(ledger.loc[ledger.date.between(a,b)],config)})
        returns=pd.DataFrame({'date':accounts['BUY_HOLD'].date,**{model:l.net_return.to_numpy() for model,l in accounts.items()}})
        returns.to_parquet(OUT/f'{cost_name}_all_evaluation_returns.parquet',index=False);uncertainty[cost_name]={}
        for block in config['bootstrap_day_blocks']:
            rng=np.random.default_rng(config['random_seed']);draws=[]
            for _ in range(config['bootstrap_repetitions']):
                ix=block_indices(rng,len(returns),block);draw={}
                for model,control in PAIRS.items():
                    values=returns[model].to_numpy()[ix]
                    draw[model+'_sharpe']=return_metrics(values,config['annual_days'])['net_sharpe']
                    draw[model+'_minus_paired_control']=float((values-returns[control].to_numpy()[ix]).mean()*config['annual_days'])
                    draw[model+'_minus_buy_hold']=float((values-returns.BUY_HOLD.to_numpy()[ix]).mean()*config['annual_days'])
                draws.append(draw)
            saved=pd.DataFrame(draws);saved.to_parquet(OUT/f'{cost_name}_block{block}_saved_bootstrap_statistics.parquet',index=False)
            uncertainty[cost_name][str(block)]={column+'_95_interval':interval(saved[column].dropna().tolist()) for column in saved}
        print('盈利分布与一致进出场账户完成：',cost_name,'三条新账户和四条复用对照',flush=True)
    frame=pd.DataFrame(metrics);frame.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(eras).to_csv(OUT/'era_metrics.csv',index=False,encoding='utf-8-sig')
    save(OUT/'execution_checks.json',{'rows':checks,'trigger_counts':trigger_counts},exclusive=True)
    save(OUT/'uncertainty.json',uncertainty,exclusive=True)
    result={'study_id':config['study_id'],'completed_at':now(),'status':'REVISION_DISTRIBUTION_AND_COHERENT_ENTRY_EXIT_COMPLETE',
            'primary':frame.loc[frame.model.eq(config['primary'])].to_dict('records'),'all_metrics':metrics,'uncertainty':uncertainty,
            'candidate_configurations':3,'evaluation_accounts':14,'new_accounts_generated':6,'reused_control_accounts':8,
            'trained_models':sum(r['status']=='TRAINED_MONTHLY_MODEL' for r in receipts),
            'valid_monthly_feature_origins':int(monthly.all_features_valid.sum()),'evaluation_month_end_events':len(events),
            'trigger_counts':trigger_counts,'historical_point_target_met':bool(frame.loc[frame.model.eq(config['primary']),'meets_point_target'].any()),
            'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY','position_impact':0}
    save(OUT/'result.json',result,exclusive=True)
    print(json.dumps({'status':result['status'],'primary':result['primary']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--run',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else run()
