"""误差训练必须来自当时真实预测和已经兑现的标签。"""
import numpy as np
import pandas as pd
from research.forward_eps_optional_valuation_v1 import select_mature_errors,FEATURE


def test_error_selection_excludes_future_label_missing_baseline_and_missing_valuation():
    frame=pd.DataFrame({'origin':pd.to_datetime(['2024-01-31','2024-02-29','2024-03-29','2024-04-30']),
                        'label_exit_date':pd.to_datetime(['2024-04-30','2024-05-30','2024-06-30','2024-07-31']),
                        'all_valuation_features_valid':[True,True,False,True],
                        'baseline_prediction':[.1,np.nan,.1,.1],'Y60':[.2,.2,.2,.2],FEATURE:[.3,.3,.3,.3]})
    selected=select_mature_errors(frame,pd.Timestamp('2024-06-28'))
    assert selected.index.tolist()==[0]


def test_maturity_on_origin_date_is_allowed_but_origin_itself_is_not_training():
    date=pd.Timestamp('2024-06-28')
    frame=pd.DataFrame({'origin':[pd.Timestamp('2024-03-29'),date],'label_exit_date':[date,date],
                        'all_valuation_features_valid':[True,True],'baseline_prediction':[.1,.1],
                        'Y60':[.2,.2],FEATURE:[.3,.3]})
    assert select_mature_errors(frame,date).index.tolist()==[0]
