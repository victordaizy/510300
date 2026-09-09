"""用当时价格预测的已兑现误差训练前瞻EPS修正，比较完整价格基线。"""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_monthly_policy_v1 import PRICE,EPS
from research.adaptive_allocation_v1 import summarize,save_account
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import normalize_dividends,holding_total_return,block_indices,return_metrics,interval

CONFIG=ROOT/'config/510300_forward_eps_residual_policy_v1.json'
MANIFEST=ROOT/'config/510300_forward_eps_residual_policy_v1_manifest.json'
OUT=ROOT/'reports/research/510300_forward_eps_residual_policy_v1'
SOURCE=ROOT/'reports/research/510300_forward_eps_monthly_policy_v1_csi'
PARENT=ROOT/'reports/research/510300_adaptive_allocation_v1'
PRIMARY='S1_PRICE_PLUS_EPS_RESIDUAL'
BASELINE='S2_PRICE_ALL_MATURE_MONTHS'


def mature_rows(monthly,origin,eps=False):
    valid=monthly.price_features_valid & (monthly.origin<origin) & monthly.label_exit_date.notna() & (monthly.label_exit_date<=origin) & np.isfinite(monthly.Y60)
    if eps:valid &= monthly.all_eps_features_valid & np.isfinite(monthly.price_prequential_prediction)
    return monthly.loc[valid].copy()


def combine_prediction(price_prediction,eps_adjustment):
    if not np.isfinite(price_prediction):return np.nan,'NO_PRICE_VIEW_KEEP_EXISTING_SHARES'
    if not np.isfinite(eps_adjustment):return float(price_prediction),'EXPLICIT_PRICE_BASELINE_WITHOUT_EPS_ADJUSTMENT'
    return float(price_prediction+eps_adjustment),'PRICE_PLUS_AVAILABLE_FORWARD_EPS_ADJUSTMENT'


def freeze():
    config=read(CONFIG)
    paths=[Path(__file__),CONFIG,ROOT/'docs/510300_FORWARD_EPS_RESIDUAL_POLICY_V1.md',
           ROOT/'tests/test_forward_eps_residual_policy_v1.py',ROOT/'research/forward_eps_monthly_policy_v1.py',
           ROOT/'config/510300_forward_eps_monthly_policy_v1_manifest.json',ROOT/'config/510300_research_authority_v6.json',
           ROOT/'research/event_clock_account_v1.py',ROOT/'research/adaptive_allocation_v1.py',
           ROOT/'research/intraday_overnight_increment_v1.py',ROOT/'research/financial_annual_components_v1.py',
           ROOT/'research/forward_eps_guosen_history_v1.py',PARENT/'features.parquet',
           PARENT/'evaluation/BASE/BUY_HOLD_ledger.parquet',PARENT/'evaluation/STRESS/BUY_HOLD_ledger.parquet',
           ROOT/config['inputs']['dividends']]
    save(MANIFEST,{'registered_at':now(),'study_id':config['study_id'],'candidate_configurations':2,'planned_evaluation_accounts':6,
                   'financial_round11_result_already_observed':True,'csi_round11_result_exists_at_registration':(SOURCE/'result.json').exists(),
                   'original_round11_method_unchanged':True,'budget_cny':0,'files':[identity(p) for p in paths]},exclusive=True)
    print('前瞻EPS误差修正的两个候选与六条账户规则已登记。',flush=True)


def run():
    config=read(CONFIG)
    for item in read(MANIFEST)['files']:
        if identity(ROOT/item['path'])['sha256']!=item['sha256']:raise ValueError('登记输入或方法发生变化')
    if not (SOURCE/'result.json').exists():raise RuntimeError('等候原沪深300主研究完成，不能启动重复采集')
    if (OUT/'RUN_STARTED.json').exists():raise FileExistsError('本试验已开始；先核对既有进程和状态，不重复运行')
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'RUN_STARTED.json',{'started_at':now(),'manifest':identity(MANIFEST)},exclusive=True)
    source_paths=[SOURCE/'source_receipt.json',SOURCE/'monthly_forward_eps_features.parquet',SOURCE/'company_month_feature_evidence.parquet',SOURCE/'result.json']
    save(OUT/'source_input_receipt.json',{'recorded_at':now(),'files':[identity(p) for p in source_paths]},exclusive=True)
    for item in read(SOURCE/'source_receipt.json')['feature_files']:
        if identity(ROOT/item['path'])['sha256']!=item['sha256']:raise ValueError('已经形成的EPS因子变化')
    data=pd.read_parquet(PARENT/'features.parquet');data['date']=pd.to_datetime(data.date)
    dividends=normalize_dividends(pd.read_csv(ROOT/config['inputs']['dividends']))
    monthly=pd.read_parquet(SOURCE/'monthly_forward_eps_features.parquet')
    monthly=monthly.merge(data[['date',*PRICE]],left_on='origin',right_on='date',validate='one_to_one').drop(columns='date').sort_values('origin').reset_index(drop=True)
    monthly['price_features_valid']=np.isfinite(monthly[PRICE]).all(axis=1)
    monthly['origin_index']=pd.DatetimeIndex(data.date).get_indexer(monthly.origin)
    monthly['Y60']=np.nan;monthly['label_exit_date']=pd.NaT;monthly['price_prequential_prediction']=np.nan
    monthly['eps_adjustment']=np.nan;monthly['combined_prediction']=np.nan;monthly['view_state']='NOT_FITTED_YET'
    for i,row in monthly.iterrows():
        t=int(row.origin_index)
        if row.price_features_valid and t+61<len(data):
            monthly.loc[i,'Y60']=holding_total_return(data,dividends,t+1,t+61)[0]
            monthly.loc[i,'label_exit_date']=data.date.iloc[t+61]
    training=[]
    for i,row in monthly.iterrows():
        origin=row.origin;price_train=mature_rows(monthly,origin);price_prediction=np.nan;adjustment=np.nan
        base_receipt={'model':BASELINE,'origin':str(origin.date()),'training_months':price_train.origin.dt.strftime('%Y-%m-%d').tolist(),
                      'training_samples':len(price_train),'latest_label_exit_date':str(price_train.label_exit_date.max()) if len(price_train) else None,
                      'status':'NO_PRICE_VIEW_INSUFFICIENT_FEATURES_OR_MATURE_MONTHS','prediction':None}
        if row.price_features_valid and len(price_train)>=config['minimum_price_train_samples']:
            estimator=make_pipeline(StandardScaler(),Ridge(alpha=config['ridge_alpha']))
            estimator.fit(price_train[PRICE],price_train.Y60)
            price_prediction=float(estimator.predict(monthly.loc[[i],PRICE])[0])
            path=OUT/'fitted_models'/f'{BASELINE}_{origin:%Y%m%d}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump(estimator,path)
            base_receipt.update({'status':'TRAINED_PRICE_MODEL','prediction':price_prediction,'model_file':identity(path)})
        monthly.loc[i,'price_prequential_prediction']=price_prediction;training.append(base_receipt)
        eps_train=mature_rows(monthly,origin,eps=True)
        residual=eps_train.Y60-eps_train.price_prequential_prediction
        eps_receipt={'model':'EPS_RESIDUAL_COMPONENT','origin':str(origin.date()),'training_months':eps_train.origin.dt.strftime('%Y-%m-%d').tolist(),
                     'training_samples':len(eps_train),'latest_label_exit_date':str(eps_train.label_exit_date.max()) if len(eps_train) else None,
                     'status':'NO_EPS_ADJUSTMENT_INSUFFICIENT_FEATURES_OR_MATURE_ERRORS','prediction':None,
                     'training_targets':[{'origin':str(d.date()),'realized_return':float(y),'then_saved_price_prediction':float(p),'residual':float(e)}
                                         for d,y,p,e in zip(eps_train.origin,eps_train.Y60,eps_train.price_prequential_prediction,residual)]}
        if np.isfinite(price_prediction) and row.all_eps_features_valid and len(eps_train)>=config['minimum_eps_train_samples']:
            estimator=make_pipeline(StandardScaler(),Ridge(alpha=config['ridge_alpha'],fit_intercept=False))
            estimator.fit(eps_train[EPS],residual)
            adjustment=float(estimator.predict(monthly.loc[[i],EPS])[0])
            path=OUT/'fitted_models'/f'EPS_RESIDUAL_COMPONENT_{origin:%Y%m%d}.joblib';joblib.dump(estimator,path)
            eps_receipt.update({'status':'TRAINED_EPS_RESIDUAL_MODEL','prediction':adjustment,'model_file':identity(path)})
        training.append(eps_receipt)
        combined,state=combine_prediction(price_prediction,adjustment)
        monthly.loc[i,['eps_adjustment','combined_prediction','view_state']]=[adjustment,combined,state]
    monthly.to_parquet(OUT/'monthly_features_labels_and_prequential_predictions.parquet',index=False)
    monthly.to_csv(OUT/'每月前瞻EPS修正与完整价格预测.csv',index=False,encoding='utf-8-sig')
    save(OUT/'training_receipts.json',{'rows':training},exclusive=True)
    anchor=int(np.flatnonzero(data.date>=pd.Timestamp(config['evaluation_start']))[0])-1
    events=monthly.loc[(monthly.origin_index>=anchor)&(monthly.origin_index<len(data)-1)]
    if anchor not in set(events.origin_index):raise ValueError('初始评价锚点缺少月末记录')
    mask=np.zeros(len(data),bool);predictions={model:np.full(len(data),np.nan) for model in [PRIMARY,BASELINE]}
    for row in events.itertuples():
        mask[row.origin_index]=True;predictions[PRIMARY][row.origin_index]=row.combined_prediction;predictions[BASELINE][row.origin_index]=row.price_prequential_prediction
    pd.DataFrame({'date':data.date,'event_mask':mask,**predictions}).to_parquet(OUT/'signals.parquet',index=False)
    metrics=[];yearly=[];eras=[];uncertainty={};execution_checks=[]
    for cost_name,cost in config['costs'].items():
        accounts={}
        for model in [PRIMARY,BASELINE,'BUY_HOLD']:
            ledger,decisions=simulate_event_account(data,dividends,config,cost,config['evaluation_start'],model,
                                                   prediction=predictions.get(model),horizon=60,event_mask=mask)
            assert decisions.loc[~decisions.origin_index.isin(events.origin_index),'requested_quantity'].eq(0).all()
            assert decisions.loc[decisions.signal_state.eq('NO_VIEW_KEEP_EXISTING_SHARES'),'requested_quantity'].eq(0).all()
            if model=='BUY_HOLD':
                old=pd.read_parquet(PARENT/'evaluation'/cost_name/'BUY_HOLD_ledger.parquet')
                for field in ['equity','shares','cash','net_return','commission','slippage_cost','dividend_receivable']:
                    np.testing.assert_array_equal(ledger[field],old[field])
            save_account(OUT/'evaluation'/cost_name,model,ledger,decisions);accounts[model]=ledger
            execution_checks.append({'cost':cost_name,'model':model,'no_view_order_count':0,'outside_month_end_order_count':0,'buy_hold_parity_checked':model=='BUY_HOLD'})
        buy_hold=summarize(accounts['BUY_HOLD'],config)
        for model,ledger in accounts.items():
            metric={'cost':cost_name,'model':model,**summarize(ledger,config)}
            metric['annualized_return_excess_vs_buy_hold']=metric['annualized_return']-buy_hold['annualized_return']
            metric['meets_point_target']=metric['net_sharpe'] is not None and metric['net_sharpe']>=1.2;metrics.append(metric)
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
                                'increment_vs_price':float((a-returns[BASELINE].to_numpy()[ix]).mean()*config['annual_days']),
                                'excess_vs_buy_hold':float((a-returns.BUY_HOLD.to_numpy()[ix]).mean()*config['annual_days'])})
            saved=pd.DataFrame(samples);saved.to_parquet(OUT/f'{cost_name}_block{block}_saved_bootstrap_statistics.parquet',index=False)
            uncertainty[cost_name][str(block)]={key+'_95_interval':interval(saved[key].dropna().tolist()) for key in saved}
        print('前瞻EPS误差修正账户完成',cost_name,'三条完整账户',flush=True)
    frame=pd.DataFrame(metrics);frame.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(yearly).to_csv(OUT/'yearly_metrics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(eras).to_csv(OUT/'era_metrics.csv',index=False,encoding='utf-8-sig')
    save(OUT/'execution_checks.json',{'rows':execution_checks},exclusive=True);save(OUT/'uncertainty.json',uncertainty,exclusive=True)
    result={'study_id':config['study_id'],'completed_at':now(),'status':'FORWARD_EPS_RESIDUAL_AND_FULL_PRICE_BASELINE_ACCOUNTS_COMPLETE',
            'primary':frame.loc[frame.model.eq(PRIMARY)].to_dict('records'),'all_metrics':metrics,'uncertainty':uncertainty,
            'candidate_configurations':2,'evaluation_accounts':6,'trained_price_models':sum(r['status']=='TRAINED_PRICE_MODEL' for r in training),
            'trained_eps_residual_models':sum(r['status']=='TRAINED_EPS_RESIDUAL_MODEL' for r in training),
            'evaluation_month_end_events':len(events),'evaluation_eps_adjustment_months':int(events.eps_adjustment.notna().sum()),
            'historical_point_target_met':bool(frame.loc[frame.model.eq(PRIMARY),'meets_point_target'].any()),
            'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY','position_impact':0}
    save(OUT/'result.json',result,exclusive=True);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--run',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else run()
