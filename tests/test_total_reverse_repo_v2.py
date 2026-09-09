"""验证新增来源口径与可用时钟，防止重复计量和提前使用。"""
import numpy as np
import pandas as pd
from research.total_reverse_repo_v2 import amount_from_row, parse_amount, inverse_only_rows, disclosed_features

def test_bid_and_award_are_not_added():
    assert amount_from_row(["7天","1.4%","930亿元","930亿元"],["期限","操作利率","投标量","中标量"])==930
    assert amount_from_row(["14天","600亿元","580亿元"],["期限","投标量","中标量"])==580

def test_amount_units():
    assert parse_amount("1.2万亿元")==12000
    assert parse_amount("10,000万元")==1

def test_mlf_table_does_not_enter_reverse_repo():
    h='<div id="zoom"><p>逆回购操作情况</p><table><tr><td>期限</td><td>交易量</td></tr><tr><td>28天</td><td>400亿元</td></tr></table><p>MLF操作情况</p><table><tr><td>期限</td><td>操作量</td></tr><tr><td>1年</td><td>1895亿元</td></tr></table></div>'
    rows=inverse_only_rows(h.encode('utf-8'))
    assert len(rows)==1 and rows[0]['cells']==['28天','400亿元']

def test_late_monthly_and_future_plan_are_not_backfilled():
    d=pd.DataFrame({'date':pd.date_range('2025-05-28','2025-06-07')})
    r=pd.DataFrame({'published_at':pd.to_datetime(['2025-05-28 09:20']),'amount_100m':[100.],'seven_amount_100m':[0.],'tenor_days':[14]})
    b=pd.DataFrame({'published_at':pd.to_datetime(['2025-05-30 18:10','2025-06-05 17:00']),
                    'amount_100m':[7000.,10000.],'kind':['BUYOUT_MONTHLY_ACTUAL_DISCLOSURE','BUYOUT_PLANNED']})
    x=disclosed_features(d,r,b).set_index('date')
    assert x.loc['2025-05-30','buyout_actual_month_report_log20']==0
    assert np.isclose(x.loc['2025-05-31','buyout_actual_month_report_log20'],np.log1p(7000))
    assert x.loc['2025-06-05','buyout_planned_log20']==0
    assert np.isclose(x.loc['2025-06-06','buyout_planned_log20'],np.log1p(10000))
    assert x.loc['2025-05-30','seven_amount20']==0
    assert x.loc['2025-05-30','regular_amount20']==100

def test_future_source_append_cannot_change_past_features():
    d=pd.DataFrame({'date':pd.date_range('2024-01-01','2024-05-01')})
    r=pd.DataFrame({'published_at':pd.to_datetime(['2024-01-01 09:20']),'amount_100m':[100.],'seven_amount_100m':[100.],'tenor_days':[7]})
    b=pd.DataFrame({'published_at':pd.to_datetime(['2024-04-30 17:00']),'amount_100m':[5000.],'kind':['BUYOUT_PLANNED']})
    a=disclosed_features(d,r,b)
    extra=pd.DataFrame({'published_at':pd.to_datetime(['2024-06-01 09:20']),'amount_100m':[999999.],'seven_amount_100m':[999999.],'tenor_days':[7]})
    z=disclosed_features(d,pd.concat([r,extra],ignore_index=True),b)
    pd.testing.assert_frame_equal(a,z)
