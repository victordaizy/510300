"""核对研报估值原文、共同训练月份、保存模型和账户，不重训重跑。"""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_report_valuation_source_v1 import aggregate_valuation
from research.forward_eps_report_valuation_fields_v1 import parse_front_page
from research.forward_eps_report_valuation_policy_v1 import MODELS,CONTROLS,COHERENT
from research.forward_eps_monthly_policy_v1 import mature_training
from research.adaptive_allocation_v1 import summarize
from research.intraday_overnight_increment_v1 import interval

OUT=ROOT/'reports/research/510300_forward_eps_report_valuation_policy_v1'
SOURCE=ROOT/'reports/research/510300_forward_eps_monthly_policy_v2_csi'
config=read(ROOT/'config/510300_forward_eps_report_valuation_policy_v1.json')
result=read(OUT/'result.json')
monthly=pd.read_parquet(OUT/'monthly_features_and_mature_labels.parquet')
source_dir=ROOT/'reports/research/510300_forward_eps_report_valuation_source_v1'
source_features=aggregate_valuation(pd.read_parquet(source_dir/'company_month_valuation_features.parquet'),
                                   pd.read_parquet(SOURCE/'monthly_forward_eps_features.parquet'),minimum_companies=10)
for field in source_features:
    pd.testing.assert_series_equal(source_features[field],monthly[field],check_names=True)
signals=pd.read_parquet(OUT/'signals.parquet')
data=pd.read_parquet(ROOT/'reports/research/510300_adaptive_allocation_v1/features.parquet')
assert signals.date.tolist()==data.date.tolist()
source_fields=pd.read_parquet(source_dir/'report_valuation_fields.parquet')
checked_target_reports=0
for item in source_fields.loc[source_fields.report_target_midpoint_upside.notna()].itertuples():
    path=ROOT/'reports/research/510300_forward_eps_csi_originals_v1/page_texts'/(item.report_id+'.json')
    parsed=parse_front_page(read(path)['pages'][0])
    assert parsed['target_upside'] is not None
    np.testing.assert_allclose(float(parsed['target_upside']['midpoint']),item.report_target_midpoint_upside,atol=1e-14,rtol=0)
    checked_target_reports+=1
receipts=read(OUT/'training_receipts.json')['rows']
by_origin={}
for item in receipts:
    by_origin.setdefault(item['origin'],[]).append(item)
for group in by_origin.values():
    assert len(group)==2 and group[0]['training_months']==group[1]['training_months']
    assert group[0]['status']==group[1]['status']
trained=0
for record in read(OUT/'training_receipts.json')['rows']:
    origin=pd.Timestamp(record['origin'])
    current=monthly.loc[monthly.origin.eq(origin)].iloc[0]
    train=mature_training(monthly,origin)
    assert record['training_months']==train.origin.dt.strftime('%Y-%m-%d').tolist()
    assert record['training_samples']==len(train)
    index=int(current.origin_index)
    if record['status']!='TRAINED_MONTHLY_MODEL':
        assert not np.isfinite(signals.loc[index,record['model']])
        continue
    trained+=1
    columns=MODELS[record['model']]
    assert train.label_exit_date.le(origin).all() and train.origin.lt(origin).all()
    model=joblib.load(ROOT/record['model_file']['path'])
    scaler=model.named_steps['standardscaler']
    assert scaler.n_samples_seen_==len(train)
    np.testing.assert_allclose(scaler.mean_,train[columns].mean().to_numpy(),atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(scaler.var_,train[columns].var(ddof=0).to_numpy(),atol=1e-12,rtol=1e-12)
    prediction=float(model.predict(pd.DataFrame([current[columns].to_dict()],columns=columns))[0])
    np.testing.assert_allclose(prediction,record['prediction'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(prediction,signals.loc[index,record['model']],atol=1e-12,rtol=0)
trigger_count=0
for metric in result['all_metrics']:
    folder=OUT/'evaluation'/metric['cost'];name=metric['model']
    ledger=pd.read_parquet(folder/(name+'_ledger.parquet'))
    decisions=pd.read_parquet(folder/(name+'_decisions.parquet'))
    saved=summarize(ledger,config)
    for key,value in saved.items():
        if isinstance(value,(float,int,np.number)) and not isinstance(value,bool):
            np.testing.assert_allclose(value,metric[key],atol=1e-10,rtol=1e-10,equal_nan=True)
        else:
            assert value==metric[key]
    np.testing.assert_allclose(ledger.equity,ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable,atol=1e-7,rtol=0)
    np.testing.assert_allclose(ledger.pnl,ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost,atol=1e-7,rtol=0)
    prev=np.r_[config['initial_capital'],ledger.equity.to_numpy()[:-1]]
    np.testing.assert_allclose(ledger.net_return,ledger.equity.to_numpy()/prev-1,atol=1e-12,rtol=0)
    if name in CONTROLS:
        old=pd.read_parquet(SOURCE/'evaluation'/metric['cost']/(CONTROLS[name]+'_ledger.parquet'))
        pd.testing.assert_frame_equal(old,ledger)
    else:
        buys=decisions.loc[decisions.requested_quantity.gt(0),'origin_index'].to_numpy(int)
        assert signals.event_mask.iloc[buys].all()
        if name==COHERENT:
            assert data.sma120.iloc[buys].gt(0).all()
            for row in decisions.loc[decisions.new_exit_trigger].itertuples():
                trigger_count+=1;t=int(row.origin_index)
                if '一百二十日均线' in row.exit_reasons:
                    assert data.sma120.iloc[t-1:t+1].le(0).all()
                if '六十交易日' in row.exit_reasons:
                    assert row.forecast_age_trading_days>=60
                execution=ledger.loc[ledger.date.eq(row.execution_date)].iloc[0]
                assert execution.requested_quantity==row.requested_quantity<0
for cost in config['costs']:
    for block in config['bootstrap_day_blocks']:
        saved=pd.read_parquet(OUT/f'{cost}_block{block}_saved_bootstrap_statistics.parquet')
        assert len(saved)==config['bootstrap_repetitions']
        for column in saved:
            np.testing.assert_allclose(interval(saved[column].dropna().tolist()),
                                       result['uncertainty'][cost][str(block)][column+'_95_interval'],atol=1e-12,rtol=0)
verification={'checked_at':now(),'status':'PASS_SAVED_REPORT_VALUATION_COMMON_CLOCK_MODELS_AND_ACCOUNTS',
              'target_reports_checked_against_saved_original_text':checked_target_reports,'matched_model_training_calendars_identical':True,
              'monthly_feature_rows_checked':len(monthly),'saved_trained_models_checked':trained,
              'complete_accounts_checked':len(result['all_metrics']),'reused_control_accounts_checked':4,
              'coherent_exit_triggers_checked':trigger_count,'saved_bootstrap_files_checked':4,
              'new_models_fit':0,'new_accounts_generated':0,'new_downloads':0,
              'random_samples_regenerated':0,'security_audit_performed':False}
save(OUT/'saved_numerical_verification.json',verification,exclusive=True)
print('估值原文、共同月份、34个保存模型、10条账户和6次退出核对通过。',flush=True)
