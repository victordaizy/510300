"""必要数学测试：周期等权恒等与未知行保留。"""
import numpy as np
import pandas as pd

from research.point_exit_error_components_v1 import decompose


def test_hand_calculated_bias_and_centered_error_are_cycle_equal():
    frame=pd.DataFrame({"cycle_id":[1,1,2],"origin_index":[10,11,20],"entry_year":[2020,2020,2021],
                        "prediction":[0.,0.,0.],"target":[1.,3.,-1.]})
    cycles,summary=decompose(frame)
    assert len(cycles)==2
    assert summary["cycle_equal_total_mse"]==3.
    assert summary["cycle_equal_bias_square"]==2.5
    assert summary["cycle_equal_centered_error_mean_square"]==.5
    assert np.isclose(summary["bias_share_of_mse"],5/6)
    assert summary["max_identity_error"]==0.


def test_missing_predictions_and_targets_stay_unknown():
    frame=pd.DataFrame({"cycle_id":[1,2,3],"origin_index":[10,20,30],"entry_year":[2020,2021,2022],
                        "prediction":[np.nan,1.,1.],"target":[1.,np.nan,1.]})
    cycles,summary=decompose(frame)
    assert len(cycles)==1 and summary["unknown_rows"]==2
    assert summary["cycle_equal_total_mse"]==0.
    assert summary["bias_share_of_mse"] is None
