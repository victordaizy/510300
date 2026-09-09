"""只读核对保存的前瞻盈利账户分解，并明确区分费用情景和费用贡献。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


MODELS=['E1_PRICE_FORWARD_EPS','E2_MATCHED_PRICE','E3_FORWARD_EPS','E4_EARNINGS_AND_PRICE_REGIME','BUY_HOLD']
PAIRS=[(MODELS[0],MODELS[1]),(MODELS[0],MODELS[2]),(MODELS[0],MODELS[4])]
KEYS=['mean_size','time_variation','dividend_entitlement','cost','net']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(a,b,tolerance=1e-12):
    np.testing.assert_allclose(a,b,atol=tolerance,rtol=0)


def verify(root,scope):
    out=root/f'reports/research/510300_forward_eps_exposure_attribution_v1_2_{scope}'
    accounts=root/f'reports/research/510300_forward_eps_monthly_policy_v1_{scope}'
    result=read(out/'result.json');summary=pd.read_csv(out/'持仓规模与变化分解_全部账户.csv')
    intervals=read(out/'uncertainty.json')['rows']
    for item in read(out/'input_receipt.json')['files']:
        path=root/item['path']
        assert path.stat().st_size==item['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    raw_price=pd.read_parquet(root/'reports/research/510300_adaptive_allocation_v1/features.parquet').set_index('date')
    periods=['FULL_EVALUATION']
    if result['common_window_start'] is not None:periods.append('AFTER_FIRST_COMMON_AVAILABLE_MODEL')
    verified={};count=0;max_rate=0.;max_cny=0.
    for cost in ['BASE','STRESS']:
        for period in periods:
            for model in MODELS:
                path=out/'decomposition_ledgers'/f'{cost}_{period}_{model}.parquet'
                frame=pd.read_parquet(path)
                original=pd.read_parquet(accounts/'evaluation'/cost/(model+'_ledger.parquet')).set_index('date')
                expected_start=original.index.min() if period=='FULL_EVALUATION' else pd.Timestamp(result['common_window_start'])
                assert frame.date.min()==expected_start
                assert frame.date.tolist()==original.loc[expected_start:].index.tolist()
                ledger=original.loc[frame.date]
                close(frame.q_on,ledger.shares_before,0);close(frame.q_day,ledger.shares,0)
                previous=original.equity.shift(1).fillna(200000).loc[frame.date]
                close(frame.previous_nav,previous,1e-6)
                prior_price=raw_price.close.shift(1).loc[frame.date].to_numpy(float)
                opening=ledger.open.to_numpy(float);mark=ledger.mark.to_numpy(float)
                distribution=raw_price.dividend.loc[frame.date].to_numpy(float)
                close(frame.r_on,(opening-prior_price+distribution)/prior_price)
                close(frame.r_day,(mark-opening)/opening)
                close(frame.w_on,frame.q_on.to_numpy()*prior_price/previous.to_numpy())
                close(frame.w_day,frame.q_day.to_numpy()*opening/previous.to_numpy())
                close(frame.dividend_residual_cny,ledger.dividend_recognized.to_numpy()-frame.q_on.to_numpy()*distribution,1e-6)
                close(frame.cost_cny,ledger.commission.to_numpy()+ledger.slippage_cost.to_numpy(),1e-6)
                for suffix,overnight,day,gain_on,gain_day,dividend,fee,tol in [
                    ('return_component','w_on','w_day','r_on','r_day','dividend_residual_rate','cost_rate',1e-12),
                    ('cny_component','q_on','q_day','unit_gain_on','unit_gain_day','dividend_residual_cny','cost_cny',1e-6)]:
                    a,b,x,y,d,c=[frame[k].to_numpy(float) for k in [overnight,day,gain_on,gain_day,dividend,fee]]
                    expected={'mean_size':a.mean()*x+b.mean()*y,
                              'time_variation':(a-a.mean())*x+(b-b.mean())*y,
                              'dividend_entitlement':d,'cost':-c,'net':a*x+b*y+d-c}
                    for key,value in expected.items():close(frame[key+'_'+suffix],value,tol)
                    reconstruction=sum(frame[k+'_'+suffix].to_numpy() for k in KEYS[:-1])
                    target=ledger.net_return.to_numpy() if suffix=='return_component' else ledger.pnl.to_numpy()
                    close(reconstruction,target,tol)
                    error=float(np.max(np.abs(reconstruction-target)))
                    if suffix=='return_component':max_rate=max(max_rate,error)
                    else:max_cny=max(max_cny,error)
                row=summary.loc[summary.cost.eq(cost)&summary.period.eq(period)&summary.model.eq(model)]
                assert len(row)==1;row=row.iloc[0]
                for key in KEYS:
                    close(row[key+'_annual_arithmetic'],frame[key+'_return_component'].mean()*242)
                    close(row[key+'_cumulative_cny'],frame[key+'_cny_component'].sum(),1e-6)
                verified[(cost,period,model)]=frame;count+=1
    pairs=[];raw_index=0
    for cost in ['BASE','STRESS']:
        for period in periods:
            for left,right in PAIRS:
                a=verified[(cost,period,left)];b=verified[(cost,period,right)]
                values={k:float((a[k+'_return_component']-b[k+'_return_component']).mean()*242) for k in KEYS}
                old=result['pairs'][raw_index]
                assert old['left']==left and old['right']==right and old['period']==period and old['days']==len(a)
                for key in KEYS:close(values[key],old[key])
                pairs.append({'scope':scope,'cost_scenario':cost,'period':period,'left':left,'right':right,'days':len(a),
                              **{('cost_effect' if key=='cost' else key):value for key,value in values.items()}})
                raw_index+=1
    assert raw_index==len(result['pairs'])
    for row in intervals:
        frame=pd.read_parquet(out/'bootstrap_statistics'/f"{row['cost']}_{row['period']}_{row['left']}_minus_{row['right']}_block{row['block_days']}.parquet")
        assert len(frame)==row['samples']==2000
        close(frame[KEYS[:-1]].sum(axis=1),frame.net)
        for key,limits in row['intervals_95'].items():close(np.quantile(frame[key],[.025,.975]),limits)
    return {'status':'PASS_SAVED_ATTRIBUTION_ACCOUNT_AND_BOOTSTRAP_RECOMPUTATION','scope':scope,
            'account_periods_checked':count,'paired_decompositions_checked':len(pairs),'saved_bootstrap_files_checked':len(intervals),
            'maximum_daily_return_error':max_rate,'maximum_daily_cny_error':max_cny,
            'cost_label_note':'原始配对汇总的cost字段保存费用贡献；本复核根据冻结循环顺序和逐账户结果恢复费用情景，并另列cost_effect，数值不变。',
            'pairs':pairs,'new_models_fit':0,'new_accounts_generated':0,'new_downloads':0,'random_samples_regenerated':0,
            'security_audit_performed':False,'external_review_performed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--scope',choices=['financial','csi'],required=True);parser.add_argument('--save',action='store_true')
    args=parser.parse_args();result=verify(args.root,args.scope)
    if args.save:
        out=args.root/f'reports/research/510300_forward_eps_exposure_attribution_v1_2_{args.scope}'
        with (out/'saved_numerical_verification.json').open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
        target=out/'模型增量分解_含明确费用情景.csv'
        if target.exists():raise FileExistsError('费用标签已明确，不重复覆盖')
        pd.DataFrame(result['pairs']).to_csv(target,index=False,encoding='utf-8-sig')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},ensure_ascii=False),flush=True)
