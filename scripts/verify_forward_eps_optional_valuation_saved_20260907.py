"""只读核对EPS连续性、估值成熟误差和完整账户，不重训重跑。"""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_optional_valuation_v1 import PRIMARY,FEATURE,select_mature_errors
from research.adaptive_allocation_v1 import summarize
from research.intraday_overnight_increment_v1 import interval

OUT=ROOT/'reports/research/510300_forward_eps_optional_valuation_v1'
SOURCE=ROOT/'reports/research/510300_forward_eps_monthly_policy_v2_csi'
config=read(ROOT/'config/510300_forward_eps_optional_valuation_v1.json')
result=read(OUT/'result.json')
monthly=pd.read_parquet(OUT/'monthly_source_and_mature_errors.parquet')
signals=pd.read_parquet(OUT/'signals.parquet');old=pd.read_parquet(SOURCE/'signals.parquet')
labels=pd.read_parquet(SOURCE/'monthly_features_and_mature_labels.parquet')
assert signals.date.tolist()==old.date.tolist()
np.testing.assert_array_equal(np.isfinite(signals[PRIMARY]),np.isfinite(old.E3_FORWARD_EPS))
np.testing.assert_allclose(signals.eps_baseline_prediction,old.E3_FORWARD_EPS,atol=0,rtol=0,equal_nan=True)
np.testing.assert_allclose(monthly.Y60,labels.Y60,atol=0,rtol=0,equal_nan=True)
np.testing.assert_allclose(monthly.baseline_prediction,old.E3_FORWARD_EPS.iloc[monthly.origin_index].to_numpy(),atol=0,rtol=0,equal_nan=True)
np.testing.assert_allclose(monthly.baseline_realized_error,monthly.Y60-monthly.baseline_prediction,atol=0,rtol=0,equal_nan=True)
fitted=0;fallback=0
for record in read(OUT/'training_receipts.json')['rows']:
    origin=pd.Timestamp(record['origin']);index=int(record['origin_index'])
    train=select_mature_errors(monthly,origin)
    assert record['training_months']==train.origin.dt.strftime('%Y-%m-%d').tolist()
    assert len(train)==record['training_samples']
    if record['model_file'] is None:
        assert not np.isfinite(signals.valuation_adjustment.iloc[index])
        np.testing.assert_allclose(signals[PRIMARY].iloc[index],old.E3_FORWARD_EPS.iloc[index],atol=0,rtol=0,equal_nan=True)
        if np.isfinite(old.E3_FORWARD_EPS.iloc[index]):fallback+=1
    else:
        fitted+=1
        assert len(train)>=12 and train.label_exit_date.le(origin).all() and train.origin.lt(origin).all()
        model=joblib.load(ROOT/record['model_file']['path'])
        scaler=model.named_steps['standardscaler'];assert scaler.n_samples_seen_==len(train)
        np.testing.assert_allclose(scaler.mean_,train[[FEATURE]].mean().to_numpy(),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(scaler.var_,train[[FEATURE]].var(ddof=0).to_numpy(),atol=1e-12,rtol=1e-12)
        current=monthly.loc[monthly.origin.eq(origin)]
        assert len(current)==1 and current.all_valuation_features_valid.iloc[0]
        delta=float(model.predict(current[[FEATURE]])[0])
        np.testing.assert_allclose(delta,signals.valuation_adjustment.iloc[index],atol=1e-12,rtol=0)
        np.testing.assert_allclose(old.E3_FORWARD_EPS.iloc[index]+delta,signals[PRIMARY].iloc[index],atol=1e-12,rtol=0)
for metric in result['all_metrics']:
    name=metric['model'];folder=OUT/'evaluation'/metric['cost']
    ledger=pd.read_parquet(folder/(name+'_ledger.parquet'))
    for key,value in summarize(ledger,config).items():
        if isinstance(value,(float,int,np.number)) and not isinstance(value,bool):
            np.testing.assert_allclose(value,metric[key],atol=1e-10,rtol=1e-10,equal_nan=True)
        else:assert value==metric[key]
    np.testing.assert_allclose(ledger.equity,ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable,atol=1e-7,rtol=0)
    np.testing.assert_allclose(ledger.pnl,ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost,atol=1e-7,rtol=0)
    previous=np.r_[config['initial_capital'],ledger.equity.to_numpy()[:-1]]
    np.testing.assert_allclose(ledger.net_return,ledger.equity.to_numpy()/previous-1,atol=1e-12,rtol=0)
    if name!=PRIMARY:
        source='E3_FORWARD_EPS' if name=='C0_ORIGINAL_EPS' else 'BUY_HOLD'
        original=pd.read_parquet(SOURCE/'evaluation'/metric['cost']/(source+'_ledger.parquet'))
        pd.testing.assert_frame_equal(ledger,original)
    else:
        decisions=pd.read_parquet(folder/(name+'_decisions.parquet'))
        outside=~decisions.origin_index.isin(np.flatnonzero(signals.event_mask))
        assert decisions.loc[outside,'requested_quantity'].eq(0).all()
for cost in config['costs']:
    for block in config['bootstrap_day_blocks']:
        samples=pd.read_parquet(OUT/f'{cost}_block{block}_saved_bootstrap_statistics.parquet')
        assert len(samples)==config['bootstrap_repetitions']
        for column in samples:
            np.testing.assert_allclose(interval(samples[column].dropna().tolist()),
                                       result['uncertainty'][cost][str(block)][column+'_95_interval'],atol=1e-12,rtol=0)
verification={'status':'PASS_SAVED_EPS_CONTINUITY_OPTIONAL_VALUATION_AND_ACCOUNTS','checked_at':now(),
              'saved_valuation_error_models_checked':fitted,'independent_eps_baseline_only_months':fallback,
              'eps_forecast_availability_preserved':True,'complete_accounts_checked':len(result['all_metrics']),
              'reused_control_accounts_checked':4,'saved_bootstrap_files_checked':4,
              'new_models_fit':0,'new_accounts_generated':0,'new_downloads':0,'random_samples_regenerated':0,
              'security_audit_performed':False}
save(OUT/'saved_numerical_verification.json',verification,exclusive=True)
print('EPS可用时钟、两次估值修正、六账户和保存区间核对通过。',flush=True)
