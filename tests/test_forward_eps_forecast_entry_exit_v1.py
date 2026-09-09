"""核对预测、趋势与月末条件，缺失预测不成为零收益信号。"""
import numpy as np
from research.forward_eps_forecast_entry_exit_v1 import entry_condition


def test_positive_forecast_and_trend_required():
    actual=entry_condition([.1,.1,-.1,0],[.1,-.1,.1,.1],[True]*4)
    np.testing.assert_array_equal(actual,[1,0,0,0])


def test_no_prediction_or_non_event_retains_no_new_signal():
    actual=entry_condition([np.nan,.1,.1],[.1,np.nan,.1],[True,True,False])
    assert np.isnan(actual).all()
