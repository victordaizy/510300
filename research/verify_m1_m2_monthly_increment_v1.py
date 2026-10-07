"""离线核对保存来源、模型算术、区块索引及既有风险账户；不重新拟合。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.m1_m2_release_sources_v1 import ROOT, OUT, now, sha, write_json, parse_report
from research.m1_m2_monthly_increment_v1 import read, check_protocol, make_monthly


def verify_macro(destination):
    cfg=check_protocol()
    releases=pd.read_csv(OUT/'admitted_releases.csv')
    for row in releases.to_dict('records'):
        receipt={'url':row['source_url'],'retrieved_at':row['retrieved_at'],'status':'FETCHED',
                 'raw_path':row['raw_path'],'sha256':row['source_sha256']}
        parsed=parse_report(receipt,row['stat_month'],row['title'])
        assert parsed['published_at']==row['published_at']
        np.testing.assert_allclose([parsed['m1_yoy_pp'],parsed['m2_yoy_pp']],
                                   [row['m1_yoy_pp'],row['m2_yoy_pp']],atol=1e-14,rtol=0)
    features=pd.read_csv(OUT/'release_features.csv')
    available=features[features.feature_status.eq('AVAILABLE')]
    assert available.definition_version.eq(available.lag_definition_version).all()
    np.testing.assert_allclose(available.delta3_spread_pp,available.delta3_m1_pp-available.delta3_m2_pp,atol=1e-12)
    data=pd.read_parquet(ROOT/cfg['market_features'])
    data['date']=pd.to_datetime(data.date)
    data['logvol20']=np.log(data.vol20.where(data.vol20>0))
    rebuilt=make_monthly(data,features,cfg)
    samples=pd.read_csv(OUT/'monthly_samples.csv',parse_dates=['origin','mature_date'])
    assert len(samples)==len(rebuilt)
    assert samples.source_status.tolist()==rebuilt.source_status.tolist()
    assert samples.stat_month.fillna('').tolist()==rebuilt.stat_month.fillna('').tolist()
    for col in ['label','delta3_spread_pp','delta3_m1_pp','delta3_m2_pp','mom20','logvol20']:
        np.testing.assert_allclose(samples[col],rebuilt[col],atol=1e-12,rtol=0,equal_nan=True)
    forecasts=pd.read_csv(OUT/'monthly_forecasts.csv',parse_dates=['origin','training_last_maturity'])
    members=pd.read_csv(OUT/'training_members.csv',parse_dates=['fit_origin','sample_origin','mature_date'])
    assert members.sample_origin.lt(members.fit_origin).all() and members.mature_date.le(members.fit_origin).all()
    ready=forecasts[forecasts.fit_status.eq('FIT_READY')]
    assert ready.training_last_maturity.le(ready.origin).all()
    models=read(OUT/'saved_models.json')
    indexed=samples.set_index(samples.origin.astype(str))
    pcfg=cfg['prediction']
    max_equation_error=0.
    for model in models:
        selected=indexed.loc[model['training_origins']]
        assert selected.training_regime.eq(model['training_regime']).all()
        x=selected[model['features']].to_numpy(float)
        y=selected.label.to_numpy(float)
        mean=x.mean(axis=0)
        scale=x.std(axis=0,ddof=1)
        scale=np.where(np.isfinite(scale)&(scale>1e-12),scale,1.)
        np.testing.assert_allclose(mean,model['mean'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(scale,model['scale'],atol=1e-12,rtol=0)
        z=np.clip((x-mean)/scale,-pcfg['standardized_clip'],pcfg['standardized_clip'])
        beta=np.asarray(model['coef'])
        residual=y-(model['intercept']+z@beta)
        equation=(z-z.mean(axis=0)).T@residual-len(x)*pcfg['ridge_mean_loss_penalty']*beta
        error=float(np.max(np.abs(equation)))
        max_equation_error=max(max_equation_error,error)
        assert error<1e-10 and abs(float(residual.mean()))<1e-12
        origin=pd.Timestamp(model['origin'])
        current=samples[samples.origin.eq(origin)].iloc[0]
        current_z=np.clip((current[model['features']].to_numpy(float)-mean)/scale,-pcfg['standardized_clip'],pcfg['standardized_clip'])
        prediction=float(model['intercept']+current_z@beta)
        stored=forecasts[forecasts.origin.eq(origin)].iloc[0][model['model']+'_prediction']
        np.testing.assert_allclose(prediction,stored,atol=1e-12,rtol=0)
        selected_members=members[members.fit_origin.eq(origin)]
        assert selected_members.sample_origin.astype(str).tolist()==selected.origin.astype(str).tolist()
    result=read(OUT/'prediction_result.json')
    draw_count=0
    for outcome in result['outcomes']:
        if not outcome['paired_months']:
            continue
        regime=outcome['regime']
        paired=pd.read_csv(OUT/f'paired_{regime}.csv')
        distributions=pd.read_csv(OUT/f'bootstrap_{regime}.csv')
        with np.load(OUT/f'bootstrap_indices_{regime}.npz') as f:
            ix=f['indices']
        gain=(paired.label-paired.baseline_prediction).to_numpy()**2-(paired.label-paired.increment_prediction).to_numpy()**2
        np.testing.assert_allclose(gain,paired.loss_gain,atol=1e-12,rtol=0)
        np.testing.assert_allclose(gain[ix].mean(axis=1),distributions.mse_gain,atol=1e-12,rtol=0)
        np.testing.assert_allclose(gain.mean(),outcome['mse_gain'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.quantile(distributions.mse_gain,.05),outcome['mse_gain_lower_5pct'],atol=1e-12,rtol=0)
        state=paired.delta3_spread_pp.to_numpy()>0
        count=state[ix].sum(axis=1)
        y=paired.label.to_numpy()
        group=np.divide((y[ix]*state[ix]).sum(axis=1),count,out=np.full(len(ix),np.nan),where=count>0)-np.divide(
            (y[ix]*(~state[ix])).sum(axis=1),len(paired)-count,out=np.full(len(ix),np.nan),where=count<len(paired))
        np.testing.assert_allclose(group,distributions.conditional_return_spread,atol=1e-12,rtol=0,equal_nan=True)
        assert outcome['pass']==all(g['pass'] for g in outcome['gates'])
        draw_count+=len(ix)
    receipt={'verified_at':now(),'status':'PASS_SOURCES_MONTHLY_CLOCK_SAVED_MODELS_AND_BOOTSTRAP_ARITHMETIC',
             'official_reports':len(releases),'monthly_origins':len(samples),'saved_models':len(models),
             'training_members':len(members),'saved_bootstrap_rows':draw_count,
             'max_ridge_equation_error':max_equation_error,'new_model_fits':0,'new_random_draws':0,
             'new_accounts':0,'network_requests':0,'independent_validation':False}
    write_json(destination/'macro_saved_verification.json',receipt)
    return receipt


def verify_risk(destination):
    import research.monthly_downside_forecast_v1 as risk
    import research.monthly_downside_accounts_v1 as accounts
    captured=[]
    def save_receipt(path,value,**kwargs):
        target=destination/('risk_'+Path(path).name)
        write_json(target,value)
        captured.append(value)
    risk.write_json=save_receipt
    accounts.write_json=save_receipt
    risk.verify()
    accounts.verify()
    rdir=ROOT/'reports/research/510300_monthly_downside_forecast_v1'
    cfg=read(rdir/'protocol.json')
    forecasts=pd.read_csv(rdir/'forecasts.csv')
    samples=pd.read_csv(rdir/'monthly_samples.csv',parse_dates=['origin','target_end'])
    for row in forecasts[forecasts.fit_status.eq('FIT_READY')].itertuples():
        origin=pd.Timestamp(row.origin)
        train=samples[samples.target_end.le(origin) & np.isfinite(samples[['feature','label']]).all(axis=1)].tail(cfg['training_months'])
        z=np.clip((train.feature.to_numpy()-row.training_mean)/row.training_scale,-3,3)
        log_label=np.log(np.maximum(train.label.to_numpy(),cfg['positive_risk_floor']))
        smearing=float(np.exp(log_label-row.intercept-row.coefficient*z).mean())
        prediction=float(np.exp(row.intercept+row.coefficient*row.current_standardized)*smearing)
        np.testing.assert_allclose([smearing,prediction],[row.smearing,row.LOG_RISK_RIDGE],atol=1e-12,rtol=0)
    result=read(rdir/'result.json')
    report=[]
    for item in result['metrics']:
        source=ROOT/item['source'] if item['reused'] else rdir/'accounts'/item['period']/item['cost']/item['model']/'ledger.parquet'
        ledger=pd.read_parquet(source)
        previous=np.r_[cfg['initial_capital'],ledger.equity.to_numpy()[:-1]]
        turnover=float((ledger.filled_quantity.abs()*ledger.fill_price.fillna(0)/previous).sum()/len(ledger)*242)
        report.append({**item,'annual_one_way_turnover':turnover,'source_ledger':source.relative_to(ROOT).as_posix(),
                       'source_ledger_sha256':sha(source),'recomputed_this_round':True,'new_account':False})
    pd.DataFrame(report).to_csv(destination/'risk_reused_account_comparison.csv',index=False,encoding='utf-8-sig')
    receipt={'verified_at':now(),'status':'PASS_REUSED_RISK_PREDICTIONS_AND_20_ACCOUNTS',
             'existing_study':cfg['study_id'],'model_arithmetic_rows':len(forecasts),
             'account_count':len(report),'ledger_rows':sum(r['days'] for r in report),
             'original_receipts_untouched':True,'new_fits':0,'new_accounts':0,'new_downloads':0,
             'risk_prediction_gate_pass':result['prediction_gate_pass'],
             'account_performance_goal_achieved':result['all_four_historical_point_targets_pass']}
    write_json(destination/'risk_reuse_verification.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description='保存产物离线算术核对，不拟合、不新建账户')
    parser.add_argument('--output',type=Path,default=OUT/'verification')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result={'macro':verify_macro(args.output),'risk':verify_risk(args.output)}
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
