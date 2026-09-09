"""验证前瞻月度特征的时间和年度配对，以及标签成熟边界。"""
import numpy as np
import pandas as pd
from research.forward_eps_monthly_policy_v1 import company_features,latest,monthly_origins,mature_training,symmetric


def report(identifier,date,year,eps,profit):
    return {'report_id':identifier,'information_date':date,'facts':[
        {'target_fiscal_year':year,'eps_value_exact':'1.0','pe_value_exact':'10','net_profit_value_exact':'100','net_profit_source_label':'净利润'},
        {'target_fiscal_year':year+1,'eps_value_exact':str(eps),'pe_value_exact':'8','net_profit_value_exact':str(profit),'net_profit_source_label':'净利润'}]}


def test_same_target_year_across_calendar_roll():
    before=report('a','2023-09-01',2023,1.1,110)
    before['facts'].append({'target_fiscal_year':2025,'eps_value_exact':'1.2','pe_value_exact':'7','net_profit_value_exact':'120','net_profit_source_label':'净利润'})
    after=report('b','2024-01-15',2024,1.3,130)
    f=company_features([before,after],pd.Timestamp('2024-01-31'))
    assert f['target_fiscal_year']==2025
    assert np.isclose(f['profit_revision'],symmetric(130,120))
    assert np.isclose(f['raw_eps_revision_unadjusted'],symmetric(1.3,1.2))


def test_publication_same_day_excluded():
    r=report('a','2024-01-31',2024,2,200)
    assert latest([r],pd.Timestamp('2024-01-31')) is None


def test_unparsed_new_report_invalidates_old_view():
    r=report('a','2024-01-01',2024,2,200)
    missing={'report_id':'b','information_date':'2024-01-15','facts':[]}
    assert latest([r,missing],pd.Timestamp('2024-01-31')) is None


def test_stale_forecast_not_carried_indefinitely():
    r=report('a','2023-01-01',2023,2,200)
    assert latest([r],pd.Timestamp('2023-07-31')) is None


def test_partial_cutoff_month_not_added_as_month_end():
    dates=pd.bdate_range('2026-06-01','2026-08-14')
    got=monthly_origins(dates,'2026-08-14')
    assert got==[pd.Timestamp('2026-06-30'),pd.Timestamp('2026-07-31')]


def test_missing_or_zero_over_zero_is_not_neutral_growth():
    assert np.isnan(symmetric(None,2))
    assert np.isnan(symmetric(0,0))
    assert symmetric(1,-1)==2


def test_changed_profit_definition_does_not_pair():
    a=report('a','2023-09-01',2023,2,200);b=report('b','2023-12-01',2023,3,300)
    b['facts'][1]['net_profit_source_label']='归母净利润'
    f=company_features([a,b],pd.Timestamp('2023-12-29'))
    assert np.isnan(f['profit_revision'])


def test_only_mature_distinct_months_train():
    d=pd.DataFrame({'origin':pd.to_datetime(['2023-01-31','2023-02-28','2023-03-31']),
                    'all_features_valid':[True,True,True],'label_exit_date':pd.to_datetime(['2023-05-03','2023-05-31','2023-06-30']),'Y60':[.1,.2,.3]})
    assert mature_training(d,pd.Timestamp('2023-05-31')).origin.tolist()==[pd.Timestamp('2023-01-31'),pd.Timestamp('2023-02-28')]


def test_ambiguous_same_day_conflicting_forecasts_not_ordered_by_identifier():
    a=report('a','2024-01-15',2024,2,200);b=report('b','2024-01-15',2024,3,300)
    assert latest([a,b],pd.Timestamp('2024-01-31')) is None
