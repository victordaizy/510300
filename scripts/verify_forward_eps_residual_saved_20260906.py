"""只读复核价格先行预测、已兑现误差、EPS修正和六条完整账户。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PRICE=['mom60','sma120','vol60']
EPS=['forward_eps_growth_median','same_year_profit_revision_90_median','reported_forward_earnings_yield_median']


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def close(a,b,tolerance=1e-10):
    np.testing.assert_allclose(a,b,atol=tolerance,rtol=0,equal_nan=True)


def verify(root,version='v1'):
    out=root/f'reports/research/510300_forward_eps_residual_policy_{version}'
    config=read(root/f'config/510300_forward_eps_residual_policy_{version}.json');result=read(out/'result.json')
    frame=pd.read_parquet(out/'monthly_features_labels_and_prequential_predictions.parquet')
    original=pd.read_parquet(root/f'reports/research/510300_forward_eps_monthly_policy_{version}_csi/monthly_forward_eps_features.parquet').set_index('origin')
    for col in EPS:close(frame[col],original.loc[frame.origin,col])
    assert frame.origin.is_unique and frame.origin.is_monotonic_increasing
    assert frame.all_eps_features_valid.tolist()==original.loc[frame.origin,'all_eps_features_valid'].tolist()
    price=pd.read_parquet(root/'reports/research/510300_adaptive_allocation_v1/features.parquet')
    dividends=pd.read_csv(root/config['inputs']['dividends'])
    for col in ['record_date','ex_date','payment_date']:dividends[col]=pd.to_datetime(dividends[col])
    labels=0
    for row in frame.loc[frame.Y60.notna()].itertuples():
        a,b=int(row.origin_index)+1,int(row.origin_index)+61
        first,last=price.date.iloc[a],price.date.iloc[b]
        cash=dividends.loc[(dividends.record_date>=first)&(dividends.record_date<last)&(dividends.ex_date<=last),'cash_dividend_per_share'].sum()
        close(row.Y60,(price.open.iloc[b]+cash)/price.open.iloc[a]-1)
        assert row.label_exit_date==last;labels+=1
    price_fits=0;eps_fits=0;residual_rows=0
    for receipt in read(out/'training_receipts.json')['rows']:
        origin=pd.Timestamp(receipt['origin']);is_eps=receipt['model']=='EPS_RESIDUAL_COMPONENT'
        mask=frame.price_features_valid & (frame.origin<origin) & frame.label_exit_date.notna() & (frame.label_exit_date<=origin) & frame.Y60.notna()
        if is_eps:mask &= frame.all_eps_features_valid & frame.price_prequential_prediction.notna()
        train=frame.loc[mask];current=frame.loc[frame.origin.eq(origin)]
        assert len(current)==1
        assert receipt['training_months']==train.origin.dt.strftime('%Y-%m-%d').tolist()
        assert receipt['training_samples']==len(train)
        if is_eps:
            assert len(receipt['training_targets'])==len(train)
            for saved,row in zip(receipt['training_targets'],train.itertuples()):
                assert saved['origin']==str(row.origin.date())
                close(saved['realized_return'],row.Y60);close(saved['then_saved_price_prediction'],row.price_prequential_prediction)
                close(saved['residual'],row.Y60-row.price_prequential_prediction);residual_rows+=1
        if receipt['status'].startswith('TRAINED_'):
            path=root/receipt['model_file']['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['model_file']['sha256']
            model=joblib.load(path);columns=EPS if is_eps else PRICE
            assert model.named_steps['ridge'].alpha==10
            assert model.named_steps['ridge'].fit_intercept==(not is_eps)
            close(model.named_steps['standardscaler'].mean_,train[columns].mean().to_numpy())
            close(float(model.predict(current[columns])[0]),receipt['prediction'])
            close(receipt['prediction'],current.iloc[0].eps_adjustment if is_eps else current.iloc[0].price_prequential_prediction)
            assert len(train)>=(12 if is_eps else 24)
            if is_eps:eps_fits+=1
            else:price_fits+=1
    for row in frame.itertuples():
        if pd.isna(row.price_prequential_prediction):
            assert pd.isna(row.combined_prediction) and row.view_state=='NO_PRICE_VIEW_KEEP_EXISTING_SHARES'
        elif pd.isna(row.eps_adjustment):
            close(row.combined_prediction,row.price_prequential_prediction)
            assert row.view_state=='EXPLICIT_PRICE_BASELINE_WITHOUT_EPS_ADJUSTMENT'
        else:
            close(row.combined_prediction,row.price_prequential_prediction+row.eps_adjustment)
            assert row.view_state=='PRICE_PLUS_AVAILABLE_FORWARD_EPS_ADJUSTMENT'
    assert price_fits==result['trained_price_models'] and eps_fits==result['trained_eps_residual_models']
    accounts=0;allowed=set(frame.origin_index)
    for saved in result['all_metrics']:
        ledger=pd.read_parquet(out/'evaluation'/saved['cost']/(saved['model']+'_ledger.parquet'))
        decisions=pd.read_parquet(out/'evaluation'/saved['cost']/(saved['model']+'_decisions.parquet'))
        returns=ledger.net_return.to_numpy(float);wealth=np.r_[1,np.cumprod(1+returns)]
        mean=returns.mean()*242;vol=returns.std(ddof=1)*np.sqrt(242)
        calculated={'cumulative_return':wealth[-1]-1,'annualized_return':wealth[-1]**(242/len(returns))-1,
                    'annualized_arithmetic_mean':mean,'annualized_volatility':vol,'net_sharpe':mean/vol if vol>1e-15 else np.nan,
                    'max_drawdown':(wealth/np.maximum.accumulate(wealth)-1).min(),'commission':ledger.commission.sum(),'slippage_cost':ledger.slippage_cost.sum()}
        for key,value in calculated.items():close(value,np.nan if saved[key] is None else saved[key])
        close(ledger.equity,ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable,1e-6)
        close(ledger.equity,200000*wealth[1:],1e-6)
        close(ledger.pnl,ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost,1e-6)
        assert not ledger.terminal_unliquidated.iloc[-1]
        assert decisions.loc[~decisions.origin_index.isin(allowed),'requested_quantity'].eq(0).all()
        assert decisions.loc[decisions.signal_state.eq('NO_VIEW_KEEP_EXISTING_SHARES'),'requested_quantity'].eq(0).all()
        if saved['model']=='BUY_HOLD':
            baseline=pd.read_parquet(root/'reports/research/510300_adaptive_allocation_v1/evaluation'/saved['cost']/'BUY_HOLD_ledger.parquet')
            close(ledger.equity,baseline.equity,0)
        accounts+=1
    assert accounts==result['evaluation_accounts']==6
    for cost in config['costs']:
        for block in config['bootstrap_day_blocks']:
            samples=pd.read_parquet(out/f'{cost}_block{block}_saved_bootstrap_statistics.parquet')
            assert len(samples)==2000
            for key in samples:
                values=samples[key].dropna().to_numpy();expected=np.quantile(values,[.025,.975]) if len(values) else [np.nan,np.nan]
                saved=result['uncertainty'][cost][str(block)][key+'_95_interval']
                close(expected,[np.nan if value is None else value for value in saved])
    return {'status':'PASS_SAVED_PREQUENTIAL_PRICE_EPS_RESIDUAL_AND_ACCOUNT_VERIFICATION','mature_monthly_labels_checked':labels,
            'saved_price_models_checked':price_fits,'saved_eps_residual_models_checked':eps_fits,'mature_residual_training_rows_checked':residual_rows,
            'complete_accounts_checked':accounts,'bootstrap_files_checked':4,'new_models_fit':0,'new_accounts_generated':0,'new_downloads':0,
            'random_samples_regenerated':0,'security_audit_performed':False,'external_review_performed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--save',action='store_true');parser.add_argument('--version',default='v1')
    args=parser.parse_args();result=verify(args.root,args.version)
    if args.save:
        path=args.root/f'reports/research/510300_forward_eps_residual_policy_{args.version}/saved_numerical_verification.json'
        with path.open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)
