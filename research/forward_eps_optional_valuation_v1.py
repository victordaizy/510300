"""保持当时EPS基线持续更新，只在资料充分时增加估值误差修正。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_guosen_history_v1 import identity
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import summarize,save_account
from research.intraday_overnight_increment_v1 import normalize_dividends,block_indices,return_metrics,interval

CONFIG=ROOT/'config/510300_forward_eps_optional_valuation_v1.json'
MANIFEST=ROOT/'config/510300_forward_eps_optional_valuation_v1_manifest.json'
OUT=ROOT/'reports/research/510300_forward_eps_optional_valuation_v1'
SOURCE=ROOT/'reports/research/510300_forward_eps_monthly_policy_v2_csi'
VALUATION=ROOT/'reports/research/510300_forward_eps_report_valuation_source_v1'
PARENT=ROOT/'reports/research/510300_adaptive_allocation_v1'
PRIMARY='O1_EPS_OPTIONAL_VALUATION'
FEATURE='report_target_midpoint_upside_median'


def select_mature_errors(monthly,origin):
    return monthly.loc[monthly.origin.lt(origin)&monthly.all_valuation_features_valid
                       &monthly.label_exit_date.notna()&monthly.label_exit_date.le(origin)
                       &np.isfinite(monthly.baseline_prediction)&np.isfinite(monthly.Y60)&np.isfinite(monthly[FEATURE])].copy()


def freeze():
    config=read(ROOT/'config/510300_forward_eps_monthly_policy_v2.json')
    config.update({'study_id':'510300_FORWARD_EPS_OPTIONAL_VALUATION_V1','primary':PRIMARY,
                   'minimum_residual_train_samples':12,'ridge_alpha':10.0,'residual_fit_intercept':False,
                   'bootstrap_day_blocks':[20,60],'candidate_configurations':1,'planned_evaluation_accounts':6,
                   'planned_new_accounts':2,'reused_control_accounts':4,'valuation_missing_uses_independent_eps_baseline':True,
                   'prepare_gpt_numerical_review_package':False})
    save(CONFIG,config,exclusive=True)
    files=[CONFIG,Path(__file__),ROOT/'docs/510300_FORWARD_EPS_OPTIONAL_VALUATION_V1.md',
           ROOT/'tests/test_forward_eps_optional_valuation_v1.py',ROOT/'research/event_clock_account_v1.py',
           ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',
           ROOT/'research/financial_annual_components_v1.py',ROOT/'research/forward_eps_guosen_history_v1.py',
           ROOT/'config/510300_research_authority_v6.json',SOURCE/'signals.parquet',
           SOURCE/'monthly_features_and_mature_labels.parquet',SOURCE/'result.json',
           VALUATION/'monthly_valuation_features.parquet',VALUATION/'result.json',PARENT/'features.parquet',
           ROOT/config['inputs']['dividends']]
    for cost in config['costs']:
        for model in ['E3_FORWARD_EPS','BUY_HOLD']:
            files.extend([SOURCE/'evaluation'/cost/(model+'_ledger.parquet'),SOURCE/'evaluation'/cost/(model+'_decisions.parquet')])
    save(MANIFEST,{'registered_at':now(),'candidate_configurations':1,'planned_evaluation_accounts':6,
                   'new_candidate_returns_read_before_freeze':False,'previous_valuation_results_already_observed':True,
                   'files':[identity(p) for p in files]},exclusive=True)
    print('第十八轮已登记：EPS持续更新、估值按成熟误差补充，一个新方法。',flush=True)


def run():
    config=read(CONFIG)
    for row in read(MANIFEST)['files']:
        if identity(ROOT/row['path'])['sha256']!=row['sha256']:raise ValueError('冻结文件变化')
    OUT.mkdir(parents=True,exist_ok=False)
    save(OUT/'RUN_STARTED.json',{'started_at':now(),'manifest':identity(MANIFEST)},exclusive=True)
    data=pd.read_parquet(PARENT/'features.parquet');source=pd.read_parquet(SOURCE/'signals.parquet')
    assert source.date.tolist()==data.date.tolist()
    monthly=pd.read_parquet(VALUATION/'monthly_valuation_features.parquet')
    labels=pd.read_parquet(SOURCE/'monthly_features_and_mature_labels.parquet')
    monthly=monthly.merge(labels[['origin','origin_index','Y60','label_exit_date']],on='origin',validate='one_to_one')
    monthly['baseline_prediction']=source.E3_FORWARD_EPS.to_numpy()[monthly.origin_index.to_numpy(int)]
    monthly['baseline_realized_error']=monthly.Y60-monthly.baseline_prediction
    monthly.to_parquet(OUT/'monthly_source_and_mature_errors.parquet',index=False)
    prediction=np.full(len(data),np.nan);adjustment=np.full(len(data),np.nan)
    mask=source.event_mask.to_numpy(bool);records=[]
    for row in monthly.loc[monthly.origin_index.isin(np.flatnonzero(mask))].itertuples():
        t=int(row.origin_index);base=float(row.baseline_prediction);train=select_mature_errors(monthly,row.origin)
        record={'origin':str(row.origin.date()),'origin_index':t,'baseline_prediction':base,
                'valuation_adjustment':None,'prediction':None,'training_months':train.origin.dt.strftime('%Y-%m-%d').tolist(),
                'training_samples':len(train),'status':'NO_VIEW_EPS_BASELINE_MISSING','model_file':None}
        if np.isfinite(base):
            prediction[t]=base;record.update({'prediction':base,'status':'EPS_BASELINE_ONLY_VALUATION_NOT_AVAILABLE_OR_IMMATURE'})
            if row.all_valuation_features_valid and np.isfinite(getattr(row,FEATURE)) and len(train)>=config['minimum_residual_train_samples']:
                model=make_pipeline(StandardScaler(),Ridge(alpha=config['ridge_alpha'],fit_intercept=False))
                model.fit(train[[FEATURE]],train.baseline_realized_error)
                delta=float(model.predict(pd.DataFrame([{FEATURE:getattr(row,FEATURE)}]))[0])
                adjustment[t]=delta;prediction[t]=base+delta
                path=OUT/'fitted_models'/f'valuation_error_{row.origin:%Y%m%d}.joblib';path.parent.mkdir(parents=True,exist_ok=True)
                joblib.dump(model,path)
                record.update({'valuation_adjustment':delta,'prediction':prediction[t],'status':'EPS_PLUS_VALUATION_ERROR_ADJUSTMENT',
                               'model_file':identity(path)})
        records.append(record)
    np.testing.assert_array_equal(np.isfinite(prediction),np.isfinite(source.E3_FORWARD_EPS.to_numpy()))
    pd.DataFrame({'date':data.date,'event_mask':mask,'eps_baseline_prediction':source.E3_FORWARD_EPS,
                  'valuation_adjustment':adjustment,PRIMARY:prediction}).to_parquet(OUT/'signals.parquet',index=False)
    save(OUT/'training_receipts.json',{'rows':records},exclusive=True)
    dividends=normalize_dividends(pd.read_csv(ROOT/config['inputs']['dividends']))
    metrics=[];yearly=[];eras=[];uncertainty={}
    for cost_name,cost in config['costs'].items():
        ledger,decisions=simulate_event_account(data,dividends,config,cost,config['evaluation_start'],PRIMARY,
                                                prediction=prediction,horizon=60,event_mask=mask)
        assert not ledger.terminal_unliquidated.iloc[-1]
        assert decisions.loc[~decisions.origin_index.isin(np.flatnonzero(mask)),'requested_quantity'].eq(0).all()
        save_account(OUT/'evaluation'/cost_name,PRIMARY,ledger,decisions)
        ledger.loc[ledger.filled_quantity.ne(0)].to_csv(OUT/f'{cost_name}_主方案全部进出场成交.csv',index=False,encoding='utf-8-sig')
        accounts={PRIMARY:ledger}
        for name,old in [('C0_ORIGINAL_EPS','E3_FORWARD_EPS'),('BUY_HOLD','BUY_HOLD')]:
            l=pd.read_parquet(SOURCE/'evaluation'/cost_name/(old+'_ledger.parquet'))
            d=pd.read_parquet(SOURCE/'evaluation'/cost_name/(old+'_decisions.parquet'))
            save_account(OUT/'evaluation'/cost_name,name,l,d);accounts[name]=l
        baseline=summarize(accounts['BUY_HOLD'],config)
        for name,l in accounts.items():
            value={'cost':cost_name,'model':name,**summarize(l,config)}
            value['annualized_return_excess_vs_buy_hold']=value['annualized_return']-baseline['annualized_return']
            value['meets_point_target']=value['net_sharpe'] is not None and value['net_sharpe']>=1.2;metrics.append(value)
            for year,g in l.groupby(l.date.dt.year):yearly.append({'cost':cost_name,'model':name,'year':int(year),**summarize(g,config)})
            for era,a,b in [('2020—2021','2020-01-01','2021-12-31'),('2022—2023','2022-01-01','2023-12-31'),('2024—终点','2024-01-01',config['data_cutoff'])]:
                eras.append({'cost':cost_name,'model':name,'era':era,**summarize(l.loc[l.date.between(a,b)],config)})
        returns=pd.DataFrame({'date':accounts['BUY_HOLD'].date,**{name:l.net_return.to_numpy() for name,l in accounts.items()}})
        returns.to_parquet(OUT/f'{cost_name}_all_evaluation_returns.parquet',index=False);uncertainty[cost_name]={}
        for block in config['bootstrap_day_blocks']:
            rng=np.random.default_rng(config['random_seed']);draws=[]
            for _ in range(config['bootstrap_repetitions']):
                ix=block_indices(rng,len(returns),block);v=returns[PRIMARY].to_numpy()[ix]
                draws.append({'primary_sharpe':return_metrics(v,config['annual_days'])['net_sharpe'],
                              'increment_vs_eps':float((v-returns.C0_ORIGINAL_EPS.to_numpy()[ix]).mean()*config['annual_days']),
                              'excess_vs_buy_hold':float((v-returns.BUY_HOLD.to_numpy()[ix]).mean()*config['annual_days'])})
            saved=pd.DataFrame(draws);saved.to_parquet(OUT/f'{cost_name}_block{block}_saved_bootstrap_statistics.parquet',index=False)
            uncertainty[cost_name][str(block)]={c+'_95_interval':interval(saved[c].dropna().tolist()) for c in saved}
        print('EPS持续更新与估值补充账户完成：',cost_name,flush=True)
    frame=pd.DataFrame(metrics);frame.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(eras).to_csv(OUT/'era_metrics.csv',index=False,encoding='utf-8-sig')
    save(OUT/'uncertainty.json',uncertainty,exclusive=True)
    result={'study_id':config['study_id'],'completed_at':now(),'status':'EPS_BASELINE_CONTINUITY_AND_OPTIONAL_VALUATION_COMPLETE',
            'primary':frame.loc[frame.model.eq(PRIMARY)].to_dict('records'),'all_metrics':metrics,'uncertainty':uncertainty,
            'candidate_configurations':1,'evaluation_accounts':6,'new_accounts_generated':2,'reused_control_accounts':4,
            'trained_valuation_error_models':sum(r['model_file'] is not None for r in records),
            'eps_prediction_months':int(np.isfinite(prediction).sum()),'valuation_adjusted_months':int(np.isfinite(adjustment).sum()),
            'baseline_availability_preserved_exactly':True,'historical_point_target_met':bool(frame.loc[frame.model.eq(PRIMARY),'meets_point_target'].any()),
            'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY','position_impact':0}
    save(OUT/'result.json',result,exclusive=True)
    print('第十八轮已完成：',result['primary'],'估值实际参与月份',result['valuation_adjusted_months'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--run',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else run()
