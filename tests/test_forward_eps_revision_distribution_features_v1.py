"""覆盖中位数掩盖修正、报告更新不同于数值更新，以及缺失不填零。"""
import numpy as np
import pandas as pd
import pytest
from research.forward_eps_revision_distribution_features_v1 import revision_statistics


def test_unchanged_majority_does_not_hide_breadth_or_mean():
    g=pd.DataFrame({'profit_revision':[0,0,0,.2,-.1,-.2],
                    'report_id':['a','b','c2','d2','e2','f2'],
                    'prior_report_id':['a','b','c','d','e','f']})
    r=revision_statistics(g)
    assert r['median_revision']==0
    assert r['same_report_count']==2 and r['new_report_unchanged_count']==1
    assert np.isclose(r['profit_revision_breadth'],-1/6)
    assert np.isclose(r['profit_revision_mean'],-.1/6)
    assert np.isclose(r['profit_report_renewal_fraction'],4/6)


def test_missing_rows_are_excluded_from_denominator_not_filled_zero():
    g=pd.DataFrame({'profit_revision':[.2,np.nan], 'report_id':['b',None], 'prior_report_id':['a',None]})
    r=revision_statistics(g)
    assert r['comparable_company_count']==1 and r['profit_revision_breadth']==1
    assert r['profit_revision_mean']==.2
    missing=revision_statistics(g.iloc[1:])
    assert np.isnan(missing['profit_revision_breadth']) and np.isnan(missing['profit_report_renewal_fraction'])


def test_same_report_cannot_have_changed_same_year_profit():
    g=pd.DataFrame({'profit_revision':[.1],'report_id':['a'],'prior_report_id':['a']})
    with pytest.raises(ValueError,match='同一报告'):
        revision_statistics(g)
