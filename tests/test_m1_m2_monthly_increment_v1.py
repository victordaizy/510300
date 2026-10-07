"""货币增量研究的关键边界测试，全部使用人工构造数据。"""
import numpy as np
import pandas as pd

from research.m1_m2_release_sources_v1 import report_month
from research.m1_m2_monthly_increment_v1 import ridge_fit, bootstrap_indices, make_monthly


def test_report_month_does_not_admit_local_or_credit_reports():
    assert report_month('2025年1月金融统计数据报告') == '2025-01'
    assert report_month('2024年前三季度金融统计数据报告') == '2024-09'
    assert report_month('2024年金融统计数据报告') == '2024-12'
    assert report_month('2025年1月深圳市金融统计数据报告') is None
    assert report_month('2024年一季度金融机构贷款投向统计报告') is None


def test_after_close_release_and_duplicate_month_use_actual_join(monkeypatch):
    import research.monthly_single_factor_walkforward_v1 as monthly
    import research.adaptive_allocation_v1 as accounts
    monkeypatch.setattr(monthly,'monthly_samples',lambda data,dividends: pd.DataFrame({
        'origin':pd.to_datetime(['2024-05-31','2024-06-28','2024-07-31'])}))
    monkeypatch.setattr(accounts,'normalize_dividends',lambda frame: frame)
    monkeypatch.setattr(pd,'read_csv',lambda path: pd.DataFrame())
    releases=pd.DataFrame([
        {'stat_month':'2024-04','available_at_upper_bound':'2024-05-10T17:00:00+08:00','feature_status':'AVAILABLE'},
        {'stat_month':'2024-05','available_at_upper_bound':'2024-05-31T17:00:00+08:00','feature_status':'AVAILABLE'}])
    cfg={'dividends':'人工数据.csv','source_start_month':'2024-01','source_contract':{'publication_age_limit_days':45}}
    result=make_monthly(pd.DataFrame(),releases,cfg)
    assert result.stat_month.tolist()==['2024-04','2024-05','2024-05']
    assert result.source_status.tolist()==['AVAILABLE','AVAILABLE','NO_VIEW_REPEATED_RELEASE']


def test_next_open_label_is_not_mature_at_previous_month_end():
    from research.monthly_single_factor_walkforward_v1 import monthly_samples
    data=pd.DataFrame({'date':pd.bdate_range('2024-04-01','2024-08-02')})
    for col in ['mom20','z20','logvol20']:
        data[col]=0.
    data['open']=4.
    dividends=pd.DataFrame({k:pd.to_datetime([]) for k in ['record_date','ex_date']})
    dividends['cash_dividend_per_share']=pd.Series(dtype=float)
    samples=monthly_samples(data,dividends)
    april=samples[samples.origin.eq(pd.Timestamp('2024-04-30'))].iloc[0]
    assert april.mature_date==pd.Timestamp('2024-06-03')
    assert april.mature_date>pd.Timestamp('2024-05-31')
    assert april.label==0.


def test_standardization_is_independent_of_future_rows():
    cfg={'standardized_clip':3.,'ridge_mean_loss_penalty':1.}
    x=np.arange(80,dtype=float).reshape(40,2)/100
    y=np.sin(np.arange(40))/30
    before=ridge_fit(x,y,[.4,.2],cfg)
    future=np.vstack([x,[1e12,-1e12]])
    after=ridge_fit(future[:40],y,[.4,.2],cfg)
    assert before==after
    np.testing.assert_allclose(before['mean'],x.mean(axis=0))


def test_ridge_new_constant_information_has_no_gain():
    cfg={'standardized_clip':3.,'ridge_mean_loss_penalty':1.}
    x=np.arange(50,dtype=float)[:,None]/100
    y=np.sin(np.arange(50))/25
    a=ridge_fit(x,y,[.2],cfg)
    b=ridge_fit(np.c_[x,np.ones(50)],y,[.2,1.],cfg)
    np.testing.assert_allclose(a['prediction'],b['prediction'],atol=1e-15)
    assert b['coef'][-1]==0.


def test_block_draws_preserve_contiguous_months_and_fixed_seed():
    ix=bootstrap_indices(37,50,6,20260917)
    assert ix.shape==(50,37)
    assert ix.min()>=0 and ix.max()<37
    np.testing.assert_array_equal(ix,bootstrap_indices(37,50,6,20260917))
    for start in range(0,36,6):
        np.testing.assert_array_equal((np.diff(ix[:,start:start+6],axis=1))%37,np.ones((50,5),int))
