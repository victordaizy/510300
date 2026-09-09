"""HAR式未来波动模型测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.har_volatility_challenger import HAR_FEATURES, walk_forward_har


def test_har_training_labels_are_realized_before_fit() -> None:
    rows = 180
    dates = pd.bdate_range("2023-01-02", periods=rows)
    trend = np.arange(rows, dtype=float)
    data = pd.DataFrame({"date": dates, "signal_rv_20": 0.2})
    for index, feature in enumerate(HAR_FEATURES):
        data[feature] = np.log(0.15 + index * 0.01 + np.sin(trend / 10.0) * 0.01)
    data["future_realized_volatility_20d"] = 0.18 + np.sin(trend / 12.0) * 0.02
    data["volatility_label_end_date_20d"] = pd.Series(dates).shift(-20)
    forecasts = walk_forward_har(
        data, horizon=20, minimum_training_samples=60, refit_interval=5
    )
    assert not forecasts.empty
    assert (
        forecasts["latest_training_label_end_date"] <= forecasts["model_fit_date"]
    ).all()
    assert forecasts["predicted_realized_volatility_20d"].gt(0).all()
