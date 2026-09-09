"""短周期方向概率与未来波动模型测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.short_horizon_direction_volatility import (
    DIRECTION_FEATURES,
    VOLATILITY_FEATURES,
    add_future_realized_volatility,
    walk_forward_models,
)


def test_future_volatility_uses_next_twenty_returns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=40)
    returns = np.linspace(-0.01, 0.01, len(dates))
    close = 4.0 * np.exp(np.cumsum(returns))
    result = add_future_realized_volatility(
        pd.DataFrame({"date": dates, "etf_close": close}), horizon=20
    )
    expected = np.std(returns[1:21], ddof=1) * np.sqrt(242)
    assert np.isclose(result.loc[0, "future_realized_volatility_20d"], expected)
    assert result.loc[0, "volatility_label_end_date_20d"] == dates[20]


def test_walk_forward_training_labels_end_before_model_fit_date() -> None:
    rows = 180
    horizon = 20
    dates = pd.bdate_range("2023-01-02", periods=rows)
    trend = np.arange(rows, dtype=float)
    data = pd.DataFrame({"date": dates, "etf_close": 4.0 + trend * 0.002})
    for index, column in enumerate(DIRECTION_FEATURES + VOLATILITY_FEATURES):
        data[column] = np.sin(trend / (7.0 + index)) + index * 0.01
    data["signal_rv_20"] = 0.18 + 0.01 * np.sin(trend / 10.0)
    data["direction_label"] = (np.sin(trend / 5.0) > 0).astype(float)
    data["exec_total_return_20d_net"] = np.where(data["direction_label"].eq(1), 0.02, -0.02)
    data["label_end_date_20d"] = data["date"].shift(-horizon)
    data["future_realized_volatility_20d"] = 0.20 + 0.02 * np.sin(trend / 8.0)
    forecasts = walk_forward_models(
        data,
        horizon=horizon,
        minimum_training_samples=60,
        refit_interval=5,
    )
    assert not forecasts.empty
    assert (
        forecasts["latest_training_label_end_date"] <= forecasts["model_fit_date"]
    ).all()
    assert forecasts["direction_probability_positive_20d_net"].between(0, 1).all()
    assert forecasts["predicted_realized_volatility_20d"].gt(0).all()
