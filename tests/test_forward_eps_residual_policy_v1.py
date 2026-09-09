"""核对EPS缺失时的独立价格观点，以及历史误差样本的成熟时钟。"""
import numpy as np
import pandas as pd
from research.forward_eps_residual_policy_v1 import mature_rows,combine_prediction


def test_missing_eps_uses_explicit_price_view_without_fabricating_eps():
    value,state=combine_prediction(.07,np.nan)
    assert value==.07 and state=='EXPLICIT_PRICE_BASELINE_WITHOUT_EPS_ADJUSTMENT'


def test_missing_price_cannot_be_replaced_by_standalone_eps_adjustment():
    value,state=combine_prediction(np.nan,.08)
    assert np.isnan(value) and state=='NO_PRICE_VIEW_KEEP_EXISTING_SHARES'


def test_available_negative_adjustment_reduces_price_forecast():
    value,state=combine_prediction(.07,-.03)
    assert np.isclose(value,.04) and state=='PRICE_PLUS_AVAILABLE_FORWARD_EPS_ADJUSTMENT'


def observations():
    return pd.DataFrame({'origin':pd.to_datetime(['2024-01-31','2024-02-29','2024-03-29','2024-04-30']),
                         'label_exit_date':pd.to_datetime(['2024-05-01','2024-05-31','2024-06-28','2024-07-31']),
                         'price_features_valid':[True]*4,'all_eps_features_valid':[False,True,True,True],
                         'Y60':[.1,.2,.3,.4],'price_prequential_prediction':[.03,np.nan,.05,.06]})


def test_price_training_does_not_require_eps_and_excludes_unmatured_labels():
    out=mature_rows(observations(),pd.Timestamp('2024-05-31'))
    assert out.index.tolist()==[0,1]


def test_eps_training_requires_then_available_price_forecast_and_mature_exit():
    out=mature_rows(observations(),pd.Timestamp('2024-06-28'),eps=True)
    assert out.index.tolist()==[2]
