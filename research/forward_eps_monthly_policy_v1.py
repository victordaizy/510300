"""前瞻年度盈利、预期修正与价格的月度完整账户研究。"""
from __future__ import annotations

import argparse
from collections import Counter
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
from research.forward_eps_guosen_history_v1 import identity,internal_date
from research.financial_annual_components_v1 import read,save,now
from research.adaptive_allocation_v1 import summarize,save_account
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import normalize_dividends,holding_total_return,block_indices,return_metrics,interval

CONFIG=ROOT/'config/510300_forward_eps_monthly_policy_v1.json'
MANIFEST=ROOT/'config/510300_forward_eps_monthly_policy_v1_manifest.json'
PARENT=ROOT/'reports/research/510300_adaptive_allocation_v1'
PRICE=['mom60','sma120','vol60']
EPS=['forward_eps_growth_median','same_year_profit_revision_90_median','reported_forward_earnings_yield_median']
MODELS={'E1_PRICE_FORWARD_EPS':PRICE+EPS,'E2_MATCHED_PRICE':PRICE,'E3_FORWARD_EPS':EPS}
RULE='E4_EARNINGS_AND_PRICE_REGIME'


def scope_paths(scope):
    facts=ROOT/('reports/research/510300_forward_eps_'+scope+'_facts_v1_1')
    originals=ROOT/('reports/research/510300_forward_eps_'+scope+'_originals_v1')
    out=ROOT/('reports/research/510300_forward_eps_monthly_policy_v1_'+scope)
    return facts,originals,out


def symmetric(a,b):
    if a is None or b is None:return np.nan
    a,b=float(a),float(b)
    den=abs(a)+abs(b)
    return 2*(a-b)/den if np.isfinite([a,b]).all() and den>0 else np.nan


def latest(reports,when):
    eligible=[r for r in reports if pd.Timestamp(r['information_date'])<when]
    if not eligible:return None
    key=lambda x:(x['information_date'],x.get('published_sequence',''))
    r=max(eligible,key=key)
    ties=[x for x in eligible if key(x)==key(r)]
    if len(ties)>1 and len({json.dumps(x['facts'],sort_keys=True,ensure_ascii=False) for x in ties})>1:return None
    if (when-pd.Timestamp(r['information_date'])).days>180 or not r['facts']:return None
    return r


def company_features(reports,origin):
    current=latest(reports,origin)
    empty={'eps_growth':np.nan,'profit_revision':np.nan,'reported_earnings_yield':np.nan,'raw_eps_revision_unadjusted':np.nan}
    if current is None:return {**empty,'status':'NO_VIEW_LATEST_REPORT_MISSING_OR_STALE'}
    year=origin.year
    values={int(x['target_fiscal_year']):x for x in current['facts']}
    today,nextyear=values.get(year),values.get(year+1)
    if today is None or nextyear is None:return {**empty,'status':'NO_VIEW_ABSOLUTE_TARGET_YEAR_MISSING'}
    result={**empty,'eps_growth':symmetric(nextyear['eps_value_exact'],today['eps_value_exact']),
            'status':'CURRENT_FORWARD_EPS_AVAILABLE','report_id':current['report_id'],
            'information_date':current['information_date'],'report_age_days':(origin-pd.Timestamp(current['information_date'])).days,
            'target_fiscal_year':year+1}
    pe=nextyear.get('pe_value_exact')
    if pe is not None and np.isfinite(float(pe)) and float(pe)!=0:result['reported_earnings_yield']=1/float(pe)
    prior=latest(reports,origin-pd.Timedelta(days=90))
    if prior is not None:
        before={int(x['target_fiscal_year']):x for x in prior['facts']}.get(year+1)
        if before is not None:
            result['prior_report_id']=prior['report_id'];result['prior_information_date']=prior['information_date']
            result['raw_eps_revision_unadjusted']=symmetric(nextyear['eps_value_exact'],before['eps_value_exact'])
            if nextyear.get('net_profit_source_label')==before.get('net_profit_source_label'):
                result['profit_revision']=symmetric(nextyear.get('net_profit_value_exact'),before.get('net_profit_value_exact'))
    return result


def monthly_origins(dates,cutoff):
    frame=pd.DataFrame({'date':pd.DatetimeIndex(dates)})
    frame=frame.loc[(frame.date>='2015-01-01')&(frame.date<pd.Timestamp(cutoff).to_period('M').start_time)]
    return frame.groupby(frame.date.dt.to_period('M')).date.max().tolist()


def freeze():
    if CONFIG.exists() or MANIFEST.exists():raise FileExistsError('月度研究已经登记')
    parent=read(ROOT/'config/510300_adaptive_allocation_v1.json')
    config={k:v for k,v in parent.items() if k not in ['models','rule_names','meta_names','shadow_start']}
    config.update({'study_id':'510300_FORWARD_EPS_MONTHLY_POLICY_V1','primary':'E1_PRICE_FORWARD_EPS',
                   'primary_scope':'csi','diagnostic_scope':'financial','ridge_alpha':10.0,'horizon':60,
                   'minimum_train_samples':12,'models':MODELS,'rule':RULE,'max_report_age_calendar_days':180,
                   'revision_calendar_days':90,'minimum_companies':{'financial':{'growth_and_pe':5,'revision':3},'csi':{'growth_and_pe':30,'revision':15}},
                   'eps_cross_report_raw_revision_in_primary':False,'annual_eps_not_exact_ntm':True})
    save(CONFIG,config,exclusive=True)
    paths=[CONFIG,Path(__file__),ROOT/'docs/510300_FORWARD_EPS_MONTHLY_POLICY_V1.md',ROOT/'tests/test_forward_eps_monthly_policy_v1.py',
           ROOT/'research/event_clock_account_v1.py',ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',
           ROOT/'research/forward_eps_guosen_history_v1.py',ROOT/'research/forward_eps_guosen_history_v1_1.py',ROOT/'research/financial_annual_components_v1.py',
           ROOT/'config/510300_research_authority_v6.json',PARENT/'features.parquet',PARENT/'input_receipt.json',
           ROOT/'reports/research/510300_forward_eps_csi_directory_v1/historical_membership.parquet']
    paths+=[ROOT/config['inputs'][k] for k in ['dividends','calendar']]
    for scope in ['financial','csi']:
        facts,originals,out=scope_paths(scope)
        paths.extend([originals/'selected_before_originals.parquet',ROOT/('config/510300_forward_eps_'+scope+'_facts_v1_1_manifest.json')])
    save(MANIFEST,{'frozen_at':now(),'new_strategy_returns_read_before_freeze':False,'history_previously_observed':True,
                   'both_scopes_and_same_methods_frozen_before_financial_diagnostic_returns':True,
                   'files':[identity(p) for p in paths]},exclusive=True)
    print('历史成分主研究与金融诊断的同一月度方法已冻结。',identity(MANIFEST)['sha256'],flush=True)


def verify():
    for r in read(MANIFEST)['files']:
        if identity(ROOT/r['path'])['sha256']!=r['sha256']:raise ValueError('月度研究冻结输入变化')
    return read(CONFIG)


def prepare(scope):
    config=verify();facts,originals,out=scope_paths(scope)
    if not (facts/'result.json').exists():raise RuntimeError('固定队列的预测提取尚未完成')
    if (out/'source_receipt.json').exists():raise FileExistsError('月度来源已经完成')
    out.mkdir(parents=True,exist_ok=True)
    queue=pd.read_parquet(originals/'selected_before_originals.parquet').to_dict('records')
    reportmap={}
    for r in queue:
        aid=r['infoCode'];fp=facts/'document_facts'/(aid+'.json');record=read(fp) if fp.exists() else {'facts':[]}
        available=record.get('conservative_information_date')
        if available is None:
            values=[r['publishDate'][:10]]
            p=originals/'document_records'/(aid+'.json')
            if p.exists():
                raw=read(p);meta=raw['provider_metadata']
                values.extend([meta['notice_date'][:10],meta['eitime'][:10]])
                try:values.append(internal_date(read(originals/'page_texts'/(aid+'.json'))['pages'][0]))
                except ValueError:pass
            available=max(values)
        rp=originals/'document_records'/(aid+'.json')
        sequence=read(rp)['provider_metadata']['eitime'] if rp.exists() else available+' 23:59:59'
        reportmap.setdefault(r['ts_code'],[]).append({'report_id':aid,'information_date':available,'published_sequence':sequence,'facts':record.get('facts',[])})
    dates=pd.read_parquet(PARENT/'features.parquet',columns=['date']).date
    origins=monthly_origins(dates,config['data_cutoff'])
    membership=pd.read_parquet(ROOT/'reports/research/510300_forward_eps_csi_directory_v1/historical_membership.parquet')
    membership['membership_date']=pd.to_datetime(membership.membership_date)
    members={d:set(g.symbol) for d,g in membership.groupby('membership_date')}
    rows=[];companyrows=[]
    threshold=config['minimum_companies'][scope]
    for origin in origins:
        items=[]
        for code in sorted(members.get(origin,set()) & set(reportmap)):
            item={'origin':origin,'ts_code':code,**company_features(reportmap[code],origin)}
            items.append(item);companyrows.append(item)
        frame=pd.DataFrame(items)
        row={'origin':origin,'is_monthly_training_origin':True,'actual_index_members':len(members.get(origin,set()))}
        mapping={'eps_growth':'forward_eps_growth_median','profit_revision':'same_year_profit_revision_90_median',
                 'reported_earnings_yield':'reported_forward_earnings_yield_median','raw_eps_revision_unadjusted':'raw_eps_revision_90_unadjusted_median'}
        for source,target in mapping.items():
            values=pd.to_numeric(frame[source],errors='coerce') if len(frame) else pd.Series(dtype=float)
            values=values[np.isfinite(values)]
            row[target]=float(values.median()) if len(values) else np.nan
            row[source+'_company_count']=len(values)
        row['all_eps_features_valid']=(row['actual_index_members']==300 and row['eps_growth_company_count']>=threshold['growth_and_pe']
                                       and row['reported_earnings_yield_company_count']>=threshold['growth_and_pe']
                                       and row['profit_revision_company_count']>=threshold['revision'] and np.isfinite([row[x] for x in EPS]).all())
        row['eps_company_coverage_of_index']=row['eps_growth_company_count']/300
        rows.append(row)
    result=pd.DataFrame(rows);result.to_parquet(out/'monthly_forward_eps_features.parquet',index=False)
    pd.DataFrame(companyrows).to_parquet(out/'company_month_feature_evidence.parquet',index=False)
    result.to_csv(out/'每月前瞻盈利因子与覆盖.csv',index=False,encoding='utf-8-sig')
    receipt={'prepared_at':now(),'scope':scope,'monthly_origins':len(result),'valid_monthly_origins':int(result.all_eps_features_valid.sum()),
             'first_valid_month':str(result.loc[result.all_eps_features_valid,'origin'].min()),
             'company_month_rows':len(companyrows),'rules_frozen_before_source_features':True,'new_labels_computed':False,
             'source_files':[identity(facts/'result.json'),identity(facts/'annual_eps_forecast_vintages.parquet'),identity(facts/'document_outcomes.json')],
             'feature_files':[identity(out/'monthly_forward_eps_features.parquet'),identity(out/'company_month_feature_evidence.parquet')]}
    save(out/'source_receipt.json',receipt,exclusive=True);print(json.dumps(receipt,ensure_ascii=False),flush=True)


def mature_training(frame,origin):
    return frame.loc[frame.all_features_valid&(frame.origin<origin)&frame.label_exit_date.notna()&(frame.label_exit_date<=origin)&np.isfinite(frame.Y60)]


def run(scope):
    config=verify();facts,originals,out=scope_paths(scope)
    receipt=read(out/'source_receipt.json')
    for x in receipt['feature_files']+receipt['source_files']:
        if identity(ROOT/x['path'])['sha256']!=x['sha256']:raise ValueError('前瞻来源或月度特征变化')
    save(out/'RUN_STARTED.json',{'started_at':now(),'scope':scope,'manifest_sha256':identity(MANIFEST)['sha256']},exclusive=True)
    data=pd.read_parquet(PARENT/'features.parquet');data['date']=pd.to_datetime(data.date)
    dividends=normalize_dividends(pd.read_csv(ROOT/config['inputs']['dividends']))
    monthly=pd.read_parquet(out/'monthly_forward_eps_features.parquet')
    monthly=monthly.merge(data[['date',*PRICE]],left_on='origin',right_on='date',validate='one_to_one').drop(columns='date')
    monthly['all_features_valid']=monthly.all_eps_features_valid & np.isfinite(monthly[PRICE]).all(axis=1)
    dates=pd.DatetimeIndex(data.date);monthly['origin_index']=dates.get_indexer(monthly.origin)
    monthly['Y60']=np.nan;monthly['label_exit_date']=pd.NaT
    for i,row in monthly.iterrows():
        t=int(row.origin_index)
        if row.all_features_valid and t+61<len(data):
            monthly.loc[i,'Y60']=holding_total_return(data,dividends,t+1,t+61)[0]
            monthly.loc[i,'label_exit_date']=data.date.iloc[t+61]
    monthly.to_parquet(out/'monthly_features_and_mature_labels.parquet',index=False)
    anchor=int(np.flatnonzero(data.date>=pd.Timestamp(config['evaluation_start']))[0])-1
    events=monthly.loc[(monthly.origin_index>=anchor)&(monthly.origin_index<len(data)-1)]
    if anchor not in set(events.origin_index):raise ValueError('初始月末时钟未包含评价锚点')
    predictions={k:np.full(len(data),np.nan) for k in MODELS};targets=np.full(len(data),np.nan);mask=np.zeros(len(data),bool);training=[]
    for _,row in events.iterrows():
        t=int(row.origin_index);mask[t]=True;train=mature_training(monthly,row.origin)
        if row.all_features_valid:targets[t]=float(row.forward_eps_growth_median>0 and row.same_year_profit_revision_90_median>0 and row.sma120>0)
        for key,columns in MODELS.items():
            valid=bool(row.all_features_valid) and len(train)>=config['minimum_train_samples']
            record={'scope':scope,'model':key,'origin':str(row.origin.date()),'training_months':train.origin.dt.strftime('%Y-%m-%d').tolist(),
                    'training_samples':len(train),'latest_label_exit_date':str(train.label_exit_date.max()) if len(train) else None,
                    'status':'TRAINED_MONTHLY_MODEL' if valid else 'NO_VIEW_FEATURES_OR_MATURE_MONTHS_INSUFFICIENT','prediction':None}
            if valid:
                model=make_pipeline(StandardScaler(),Ridge(alpha=config['ridge_alpha']))
                model.fit(train[columns],train.Y60)
                value=float(model.predict(pd.DataFrame([row[columns].to_dict()],columns=columns))[0]);predictions[key][t]=value
                p=out/'fitted_models'/f'{key}_{row.origin:%Y%m%d}.joblib';p.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,p)
                record.update({'prediction':value,'columns':columns,'model_file':identity(p)})
            training.append(record)
    save(out/'training_receipts.json',{'rows':training},exclusive=True)
    pd.DataFrame({'date':data.date,'event_mask':mask,**predictions,RULE:targets}).to_parquet(out/'signals.parquet',index=False)
    metrics=[];yearly=[];eras=[];uncertainty={};checks=[]
    for cid,cost in config['costs'].items():
        accounts={}
        for key in list(MODELS)+[RULE,'BUY_HOLD']:
            ledger,decisions=simulate_event_account(data,dividends,config,cost,config['evaluation_start'],key,
                prediction=predictions.get(key),targets=targets if key==RULE else None,horizon=60,event_mask=mask)
            no=decisions.signal_state.eq('NO_VIEW_KEEP_EXISTING_SHARES')
            if decisions.loc[no,'requested_quantity'].ne(0).any():raise ValueError('无观点被转换为交易')
            allowed=decisions.origin_index.isin(events.origin_index)
            if decisions.loc[~allowed,'requested_quantity'].ne(0).any():raise ValueError('月末之外发生主动换仓')
            if key=='BUY_HOLD':
                old=pd.read_parquet(PARENT/'evaluation'/cid/'BUY_HOLD_ledger.parquet')
                for col in ['equity','shares','cash','net_return','commission','slippage_cost','dividend_receivable']:np.testing.assert_array_equal(ledger[col],old[col])
            checks.append({'cost':cid,'model':key,'no_view_order_count':int(decisions.loc[no,'requested_quantity'].ne(0).sum()),'outside_month_end_order_count':int(decisions.loc[~allowed,'requested_quantity'].ne(0).sum()),'buy_hold_parity':key=='BUY_HOLD'})
            save_account(out/'evaluation'/cid,key,ledger,decisions);accounts[key]=ledger
        baseline=summarize(accounts['BUY_HOLD'],config)
        for key,ledger in accounts.items():
            r={'cost':cid,'model':key,**summarize(ledger,config)};r['annualized_return_excess_vs_buy_hold']=r['annualized_return']-baseline['annualized_return']
            r['meets_point_target']=r['net_sharpe'] is not None and r['net_sharpe']>=1.2;metrics.append(r)
            for year,g in ledger.groupby(ledger.date.dt.year):yearly.append({'cost':cid,'model':key,'year':int(year),**summarize(g,config)})
            for name,a,b in [('2020—2021','2020-01-01','2021-12-31'),('2022—2023','2022-01-01','2023-12-31'),('2024—终点','2024-01-01',config['data_cutoff'])]:
                eras.append({'cost':cid,'model':key,'era':name,**summarize(ledger.loc[ledger.date.between(a,b)],config)})
        rr=pd.DataFrame({'date':accounts['BUY_HOLD'].date,**{k:v.net_return.to_numpy() for k,v in accounts.items()}})
        rr.to_parquet(out/f'{cid}_all_evaluation_returns.parquet',index=False)
        rng=np.random.default_rng(config['random_seed']);samples={'primary_sharpe':[],'increment_vs_price':[],'excess_vs_buy_hold':[]}
        for _ in range(config['bootstrap_repetitions']):
            ix=block_indices(rng,len(rr),config['bootstrap_day_block']);a=rr[config['primary']].to_numpy()[ix]
            s=return_metrics(a,config['annual_days'])['net_sharpe'];samples['primary_sharpe'].append(np.nan if s is None else s)
            samples['increment_vs_price'].append(float((a-rr.E2_MATCHED_PRICE.to_numpy()[ix]).mean()*config['annual_days']))
            samples['excess_vs_buy_hold'].append(float((a-rr.BUY_HOLD.to_numpy()[ix]).mean()*config['annual_days']))
        pd.DataFrame(samples).to_parquet(out/f'{cid}_saved_bootstrap_statistics.parquet',index=False)
        uncertainty[cid]={k+'_95_interval':interval(v) for k,v in samples.items()}
        print('前瞻月度账户完成',scope,cid,'五条完整账户',flush=True)
    frame=pd.DataFrame(metrics);frame.to_csv(out/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(yearly).to_csv(out/'yearly_metrics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(eras).to_csv(out/'era_metrics.csv',index=False,encoding='utf-8-sig')
    save(out/'execution_checks.json',{'rows':checks},exclusive=True);save(out/'uncertainty.json',uncertainty,exclusive=True)
    primary=frame.loc[frame.model==config['primary']]
    result={'study_id':config['study_id'],'scope':scope,'completed_at':now(),
            'status':'FINANCIAL_DIAGNOSTIC_COMPLETE_CSI_PRIMARY_SEPARATE' if scope=='financial' else 'CSI_FORWARD_EPS_PRIMARY_HISTORICAL_EVALUATION_COMPLETE',
            'primary':primary.to_dict('records'),'all_metrics':metrics,'candidate_configurations':4,'evaluation_accounts':10,
            'trained_model_count':sum(x['status']=='TRAINED_MONTHLY_MODEL' for x in training),'evaluation_month_end_events':len(events),
            'valid_source_months':int(monthly.all_features_valid.sum()),'uncertainty':uncertainty,
            'historical_point_target_met':bool(primary.meets_point_target.any()),'goal_achieved':False,
            'independent_validation':'NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY','market_consensus':False,'position_impact':0}
    save(out/'result.json',result,exclusive=True);print(json.dumps(result,ensure_ascii=False,default=str),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['financial','csi'],default='financial')
    g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.freeze:freeze()
    elif a.prepare:prepare(a.scope)
    else:run(a.scope)
