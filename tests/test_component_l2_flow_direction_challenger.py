"""成分股L2资金流方向模型数据测试。"""

from __future__ import annotations

import pandas as pd

from research.component_l2_flow_direction_challenger import L2_FEATURES, prepare_l2_model_data


def test_l2_features_merge_on_exact_signal_date() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    base = pd.DataFrame(
        {"date": dates, "exec_total_return_20d_net": [0.01, -0.01, None]}
    )
    flow = pd.DataFrame({"date": dates, **{column: 0.1 for column in L2_FEATURES}})
    result = prepare_l2_model_data(base, flow)
    assert len(result) == 3
    assert result.loc[0, "direction_label"]
    assert not result.loc[1, "direction_label"]
    assert pd.isna(result.loc[2, "direction_label"])
