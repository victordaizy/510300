"""估值原文缺失不补零，最新报告没有目标时不得沿用旧目标。"""
import numpy as np
import pandas as pd
from research.forward_eps_report_valuation_source_v1 import aggregate_valuation


def test_missing_target_is_excluded_and_coverage_gate_preserved():
    origin=pd.Timestamp('2025-06-30')
    company=pd.DataFrame({'origin':[origin]*3,'ts_code':['a','b','c'],'eps_growth':[.1,.2,.3],
                          'report_target_midpoint_upside':[.2,-.1,np.nan],'report_age_days':[10,20,30]})
    monthly=pd.DataFrame({'origin':[origin],'actual_index_members':[300],'all_eps_features_valid':[True]})
    result=aggregate_valuation(company,monthly,minimum_companies=3)
    assert result.valuation_company_count.iloc[0]==2
    assert np.isnan(result.report_target_midpoint_upside_median.iloc[0])
    assert not result.all_valuation_features_valid.iloc[0]


def test_latest_missing_target_does_not_get_previous_month_target():
    origins=pd.to_datetime(['2025-05-30','2025-06-30'])
    company=pd.DataFrame({'origin':origins,'ts_code':['a','a'],'eps_growth':[.1,.1],
                          'report_target_midpoint_upside':[.2,np.nan],'report_age_days':[15,5]})
    monthly=pd.DataFrame({'origin':origins,'actual_index_members':[300,300],'all_eps_features_valid':[True,True]})
    result=aggregate_valuation(company,monthly,minimum_companies=1)
    assert result.all_valuation_features_valid.tolist()==[True,False]
    assert np.isnan(result.report_target_midpoint_upside_median.iloc[1])
