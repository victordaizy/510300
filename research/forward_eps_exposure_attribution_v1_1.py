"""从保存账户精确分解前瞻盈利的规模、持仓变化、分红和费用关联。"""
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

MANIFEST=ROOT/'config/510300_forward_eps_exposure_attribution_v1_1_manifest.json'
MODELS=['E1_PRICE_FORWARD_EPS','E2_MATCHED_PRICE','E3_FORWARD_EPS','E4_EARNINGS_AND_PRICE_REGIME','BUY_HOLD']
PAIRS=[('E1_PRICE_FORWARD_EPS','E2_MATCHED_PRICE'),('E1_PRICE_FORWARD_EPS','E3_FORWARD_EPS'),('E1_PRICE_FORWARD_EPS','BUY_HOLD')]


def components(w_on,w_day,r_on,r_day,dividend_residual,cost):
    w_on,w_day,r_on,r_day,dividend_residual,cost=[np.asarray(x,dtype=float) for x in [w_on,w_day,r_on,r_day,dividend_residual,cost]]
    static=w_on.mean()*r_on+w_day.mean()*r_day
    varying=(w_on-w_on.mean())*r_on+(w_day-w_day.mean())*r_day
    net=w_on*r_on+w_day*r_day+dividend_residual-cost
    scale=np.maximum(1.,np.abs(net)+np.abs(static)+np.abs(varying)+np.abs(dividend_residual)+np.abs(cost))
    if np.any(np.abs(net-static-varying-dividend_residual+cost)>32*np.finfo(float).eps*scale):
        raise ValueError('收益或人民币损益分解未通过浮点尺度恒等式核对')
    return {'mean_size':static,'time_variation':varying,'dividend_entitlement':dividend_residual,'cost':-cost,'net':net}


def market_and_exposures(ledger,price,initial_capital):
    g=price.set_index('date').loc[ledger.date]
    all_dates=pd.DatetimeIndex(price.date);positions=all_dates.get_indexer(ledger.date)
    if min(positions)<=0:raise ValueError('缺少首日之前的价格')
    prior_mark=price.close.iloc[positions-1].to_numpy(float)
    previous_nav=ledger.equity.shift(1).fillna(initial_capital).to_numpy(float)
    old=ledger.shares_before.to_numpy(float);current=ledger.shares.to_numpy(float)
    recorded=ledger.shares_after.notna()
    np.testing.assert_allclose(ledger.loc[recorded,'shares_after'],ledger.loc[recorded,'shares'],atol=0,rtol=0)
    opening=ledger.open.to_numpy(float);mark=ledger.mark.to_numpy(float)
    distribution=g.dividend.to_numpy(float)
    ron=(opening-prior_mark+distribution)/prior_mark
    rday=(mark-opening)/opening
    won=old*prior_mark/previous_nav;wday=current*opening/previous_nav
    divdiff=ledger.dividend_recognized.to_numpy(float)-old*distribution
    cost=ledger.commission.to_numpy(float)+ledger.slippage_cost.to_numpy(float)
    reconstructed=won*ron+wday*rday+divdiff/previous_nav-cost/previous_nav
    np.testing.assert_allclose(reconstructed,ledger.net_return,atol=1e-12,rtol=0)
    pnl=old*(opening-prior_mark+distribution)+current*(mark-opening)+divdiff-cost
    np.testing.assert_allclose(pnl,ledger.pnl,atol=1e-6,rtol=0)
    return pd.DataFrame({'date':ledger.date,'previous_nav':previous_nav,'r_on':ron,'r_day':rday,'w_on':won,'w_day':wday,
                         'dividend_residual_rate':divdiff/previous_nav,'cost_rate':cost/previous_nav,'net_return':reconstructed,
                         'q_on':old,'q_day':current,'unit_gain_on':opening-prior_mark+distribution,'unit_gain_day':mark-opening,
                         'dividend_residual_cny':divdiff,'cost_cny':cost,'pnl_cny':pnl})


def pair_contributions(a,b,take=None):
    if take is not None:a,b=a.iloc[take],b.iloc[take]
    ron=a.r_on.to_numpy();rd=a.r_day.to_numpy()
    won=a.w_on.to_numpy()-b.w_on.to_numpy();wd=a.w_day.to_numpy()-b.w_day.to_numpy()
    div=a.dividend_residual_rate.to_numpy()-b.dividend_residual_rate.to_numpy()
    fee=a.cost_rate.to_numpy()-b.cost_rate.to_numpy()
    c=components(won,wd,ron,rd,div,fee)
    actual=a.net_return.to_numpy()-b.net_return.to_numpy()
    np.testing.assert_allclose(c['net'],actual,atol=1e-12,rtol=0)
    return {k:float(v.mean()*242) for k,v in c.items()}


def paired_bootstrap(a,b,block,repetitions,seed):
    if not np.array_equal(a.date.to_numpy(),b.date.to_numpy()):
        raise ValueError('两个保存账户日期未对齐')
    columns=['w_on','w_day','r_on','r_day','dividend_residual_rate','cost_rate']
    aa=a[columns].to_numpy(float);bb=b[columns].to_numpy(float)
    np.testing.assert_allclose(aa[:,2:4],bb[:,2:4],atol=0,rtol=0)
    values=aa.copy();values[:,[0,1,4,5]]-=bb[:,[0,1,4,5]]
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(repetitions):
        begins=rng.integers(0,len(a),size=(len(a)+block-1)//block)
        take=((begins[:,None]+np.arange(block))%len(a)).ravel()[:len(a)]
        c=components(*values[take].T)
        samples.append({key:float(value.mean()*242) for key,value in c.items()})
    return pd.DataFrame(samples)


def freeze():
    paths=[Path(__file__),ROOT/'docs/510300_FORWARD_EPS_EXPOSURE_ATTRIBUTION_V1.md',
           ROOT/'tests/test_forward_eps_exposure_attribution_v1.py',ROOT/'tests/test_forward_eps_exposure_attribution_v1_1.py',
           ROOT/'docs/510300_FORWARD_EPS_EXPOSURE_ATTRIBUTION_V1_1.md',
           ROOT/'research/forward_eps_exposure_attribution_v1.py',ROOT/'config/510300_forward_eps_exposure_attribution_v1_manifest.json',
           ROOT/'research/financial_annual_components_v1.py',ROOT/'research/forward_eps_guosen_history_v1.py',
           ROOT/'config/510300_forward_eps_monthly_policy_v1_manifest.json']
    save(MANIFEST,{'registered_at':now(),'scopes':['financial','csi'],'models':MODELS,'pairs':PAIRS,
                   'periods':['FULL_EVALUATION','AFTER_FIRST_COMMON_AVAILABLE_MODEL'],
                   'block_lengths':[20,60],'bootstrap_repetitions':2000,'random_seed':20260906,
                   'financial_headline_returns_already_observed':True,'not_a_new_tradable_strategy':True,
                   'new_accounts_authorized_by_this_diagnostic':0,'files':[identity(p) for p in paths]},exclusive=True)
    print('两范围的持仓规模与变化分解方法已登记。',flush=True)


def run(scope):
    manifest=read(MANIFEST)
    for r in manifest['files']:
        if identity(ROOT/r['path'])['sha256']!=r['sha256']:raise ValueError('分解方法登记文件变化')
    source=ROOT/('reports/research/510300_forward_eps_monthly_policy_v1_'+scope)
    out=ROOT/('reports/research/510300_forward_eps_exposure_attribution_v1_1_'+scope)
    if not (source/'result.json').exists():raise RuntimeError('对应范围的原完整账户尚未完成')
    if (out/'result.json').exists():raise FileExistsError('该范围的分解已完成，不重复选取结果')
    out.mkdir(parents=True,exist_ok=True)
    paths=[source/'result.json',source/'training_receipts.json',source/'monthly_features_and_mature_labels.parquet',
           ROOT/'reports/research/510300_adaptive_allocation_v1/features.parquet']
    for cost in ['BASE','STRESS']:
        for model in MODELS:
            paths.extend([source/'evaluation'/cost/(model+'_ledger.parquet'),source/'evaluation'/cost/(model+'_decisions.parquet')])
    save(out/'input_receipt.json',{'recorded_at':now(),'files':[identity(p) for p in paths],'new_models_fit':0,'new_accounts_generated':0},exclusive=True)
    price=pd.read_parquet(paths[3]);price['date']=pd.to_datetime(price.date)
    records=read(source/'training_receipts.json')['rows'];labels=pd.read_parquet(source/'monthly_features_and_mature_labels.parquet')
    starts=[]
    for model in MODELS[:3]:
        dates=[pd.Timestamp(x['origin']) for x in records if x['model']==model and x['status']=='TRAINED_MONTHLY_MODEL']
        if dates:starts.append(min(dates))
    common_origin=max(starts) if len(starts)==3 else None
    common_start=price.loc[price.date>common_origin,'date'].iloc[0] if common_origin is not None else None
    training=[]
    for r in records:
        eligible=labels.loc[labels.origin.dt.strftime('%Y-%m-%d').isin(r['training_months'])]
        values=eligible.Y60.dropna().to_numpy(float)
        training.append({'model':r['model'],'origin':r['origin'],'status':r['status'],'samples':len(values),
                         'positive_label_fraction':float((values>0).mean()) if len(values) else None,
                         'mean_training_return':float(values.mean()) if len(values) else None,
                         'minimum_training_return':float(values.min()) if len(values) else None,
                         'maximum_training_return':float(values.max()) if len(values) else None,
                         'training_years':sorted(eligible.origin.dt.year.unique().astype(int).tolist()),
                         'prediction':r['prediction'],'prediction_positive':bool(r['prediction']>0) if r['prediction'] is not None else None})
    pd.DataFrame(training).to_csv(out/'训练样本收益分布与预测.csv',index=False,encoding='utf-8-sig')
    summaries=[];pairs=[];intervals=[];first_trades=[];max_error=0
    for cost in ['BASE','STRESS']:
        raw={};source_ledgers={}
        for model in MODELS:
            ledger=pd.read_parquet(source/'evaluation'/cost/(model+'_ledger.parquet'))
            raw[model]=market_and_exposures(ledger,price,200000);source_ledgers[model]=ledger
            buys=ledger.loc[ledger.filled_quantity>0,'date']
            first_trades.append({'cost':cost,'model':model,'first_buy_date':str(buys.min().date()) if len(buys) else None,
                                 'first_common_model_origin':str(common_origin.date()) if common_origin is not None else None})
        periods=[('FULL_EVALUATION',raw[MODELS[0]].date.min())]
        if common_start is not None:periods.append(('AFTER_FIRST_COMMON_AVAILABLE_MODEL',common_start))
        for period,start in periods:
            frames={model:df.loc[df.date>=start].reset_index(drop=True) for model,df in raw.items()}
            if len(frames[MODELS[0]])<2:continue
            for model,df in frames.items():
                c=components(df.w_on,df.w_day,df.r_on,df.r_day,df.dividend_residual_rate,df.cost_rate)
                cash=components(df.q_on,df.q_day,df.unit_gain_on,df.unit_gain_day,df.dividend_residual_cny,df.cost_cny)
                np.testing.assert_allclose(c['net'],df.net_return,atol=1e-12,rtol=0)
                np.testing.assert_allclose(cash['net'],df.pnl_cny,atol=1e-6,rtol=0)
                for key,value in c.items():df[key+'_return_component']=value
                for key,value in cash.items():df[key+'_cny_component']=value
                error=float(np.max(np.abs(sum(c[k] for k in ['mean_size','time_variation','dividend_entitlement','cost'])-df.net_return)))
                max_error=max(max_error,error)
                path=out/'decomposition_ledgers'/f'{cost}_{period}_{model}.parquet';path.parent.mkdir(exist_ok=True);df.to_parquet(path,index=False)
                summaries.append({'scope':scope,'cost':cost,'period':period,'model':model,'start':str(df.date.min().date()),
                                  'end':str(df.date.max().date()),'days':len(df),'mean_overnight_exposure_previous_nav':float(df.w_on.mean()),
                                  'mean_intraday_exposure_previous_nav':float(df.w_day.mean()),'mean_overnight_shares':float(df.q_on.mean()),
                                  'mean_intraday_shares':float(df.q_day.mean()),
                                  **{k+'_annual_arithmetic':float(v.mean()*242) for k,v in c.items()},
                                  **{k+'_cumulative_cny':float(v.sum()) for k,v in cash.items()}})
            for left,right in PAIRS:
                a,b=frames[left],frames[right];point=pair_contributions(a,b)
                pairs.append({'scope':scope,'cost':cost,'period':period,'left':left,'right':right,'days':len(a),**point})
                for block in manifest['block_lengths']:
                    frame=paired_bootstrap(a,b,block,manifest['bootstrap_repetitions'],manifest['random_seed'])
                    target=out/'bootstrap_statistics'/f'{cost}_{period}_{left}_minus_{right}_block{block}.parquet';target.parent.mkdir(exist_ok=True);frame.to_parquet(target,index=False)
                    intervals.append({'cost':cost,'period':period,'left':left,'right':right,'block_days':block,
                                      'samples':len(frame),'intervals_95':{k:np.quantile(frame[k],[.025,.975]).tolist() for k in frame}})
            print('保存账户分解完成',scope,cost,period,flush=True)
    pd.DataFrame(summaries).to_csv(out/'持仓规模与变化分解_全部账户.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pairs).to_csv(out/'模型增量的规模与变化分解.csv',index=False,encoding='utf-8-sig')
    save(out/'uncertainty.json',{'rows':intervals},exclusive=True)
    result={'completed_at':now(),'status':'EXACT_SAVED_ACCOUNT_ATTRIBUTION_COMPLETED_NO_CAUSAL_ALPHA_CLAIM','scope':scope,
            'common_first_model_origin':str(common_origin.date()) if common_origin is not None else None,
            'common_window_start':str(common_start.date()) if common_start is not None else None,
            'first_buys':first_trades,'account_period_decompositions':len(summaries),'paired_period_decompositions':len(pairs),
            'pairs':pairs,'uncertainty':intervals,'maximum_return_reconciliation_error':max_error,
            'new_models_fit':0,'new_strategy_accounts_generated':0,'period_means_use_full_observed_sample':True,
            'not_a_tradable_static_policy_or_causal_effect':True,'goal_achieved':False}
    save(out/'result.json',result,exclusive=True)
    print(json.dumps({'status':result['status'],'scope':scope,'primary_minus_price': [x for x in pairs if x['right']=='E2_MATCHED_PRICE'],
                      'maximum_return_reconciliation_error':max_error},ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['financial','csi'],default='financial')
    g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');a=ap.parse_args()
    freeze() if a.freeze else run(a.scope)
