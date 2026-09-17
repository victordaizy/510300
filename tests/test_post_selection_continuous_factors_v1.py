"""连续目标保留旧计算、未知语义与末日计划份额。"""
import numpy as np
import pandas as pd

from research.post_selection_continuous_factors_v1 import planned_weights,ordinary_multiplier,continuous_minimum_variance_budget
from research.three_source_order_intent_mix_inputs_v1 import source_plan
from research.two_policy_min_variance_inputs_v1 import budget_frame


def test_planned_shares_match_original_and_keep_zero_wait():
    dates=pd.bdate_range('2026-08-03',periods=5)
    data=pd.DataFrame({'date':dates,'close':[4.,4.,4.,4.,4.]})
    ledger=pd.DataFrame({'date':dates[1:],'equity':[200000.]*4,'shares':[10000,10000,10000,0]})
    decisions=pd.DataFrame({'origin_index':np.arange(5),'origin':dates,
        'execution_date':dates[1:].append(pd.DatetimeIndex(['2026-08-10'])),
        'decision_time':dates+pd.Timedelta(hours=15,minutes=5),
        'requested_quantity':[10000,0,0,-10000,0],'reference_weight':[.2,0.,np.nan,0.,0.]})
    cfg={'initial_capital':200000.,'lot':100}
    old=source_plan(data,decisions.iloc[:-1],ledger,cfg,1)
    new=planned_weights(data,decisions,ledger,cfg,1)
    np.testing.assert_array_equal(new[:-1],old.planned_weight)
    assert new[1]==.2 and np.isnan(new[2]) and new[-1]==0.


def test_ordinary_multiplier_carries_only_last_explicit_value():
    values=ordinary_multiplier(pd.DataFrame({'vol20':[np.nan,0.,.2,np.nan,0.,.05]}))
    np.testing.assert_array_equal(values,[1.,1.,.5,.5,.5,1.])


def test_monthly_budget_prefix_and_last_real_close():
    dates=pd.bdate_range('2020-01-02',periods=520)
    values=np.column_stack([np.sin(np.arange(520))*.01,np.cos(np.arange(520)/2)*.02])
    states=np.column_stack([np.ones(520),np.zeros(520)])
    old=budget_frame(dates,values,states,1)
    new=continuous_minimum_variance_budget(dates,values,states,1)
    pd.testing.assert_frame_equal(old.iloc[:-1],new.iloc[:-1],check_exact=True)
    assert np.isfinite(new.target.iloc[-1]) and np.isnan(old.target.iloc[-1])
