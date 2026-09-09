"""一级市场资金流方向挑战模型测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.primary_flow_direction_challenger import (
    FLOW_FEATURES,
    prepare_flow_features,
    walk_forward_flow_direction,
)


def test_share_snapshots_only_flow_forward_from_disclosure_date() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    base = pd.DataFrame(
        {
            "date": dates,
            "index_close": 3000.0 + np.arange(100),
            "exec_total_return_20d_net": np.sin(np.arange(100) / 5.0) / 20.0,
            "label_end_date_20d": pd.Series(dates).shift(-20),
        }
    )
    futures = pd.DataFrame({"date": dates, "close": base["index_close"] * 1.001})
    nav = pd.DataFrame({"date": dates, "close_premium_bps": np.arange(100) / 10.0})
    shares = pd.DataFrame(
        {
            "date": [dates[0], dates[30], dates[60]],
            "share_change_pct": [0.01, 0.02, -0.01],
        }
    )
    result = prepare_flow_features(base, futures, nav, shares)
    assert result.loc[29, "latest_monthly_share_change_pct"] == 0.01
    assert result.loc[30, "latest_monthly_share_change_pct"] == 0.02
    assert result.loc[30, "share_snapshot_date"] == dates[30]


def test_flow_walk_forward_has_no_unrealized_training_labels() -> None:
    rows = 160
    dates = pd.bdate_range("2023-01-02", periods=rows)
    trend = np.arange(rows, dtype=float)
    data = pd.DataFrame({"date": dates})
    for index, column in enumerate(FLOW_FEATURES):
        data[column] = np.sin(trend / (5.0 + index))
    data["direction_label"] = (np.sin(trend / 6.0) > 0).astype(float)
    data["exec_total_return_20d_net"] = np.where(data["direction_label"].eq(1), 0.02, -0.02)
    data["label_end_date_20d"] = pd.Series(dates).shift(-20)
    data["share_snapshot_date"] = dates[0]
    data["share_snapshot_age_calendar_days"] = (data["date"] - dates[0]).dt.days
    forecasts = walk_forward_flow_direction(
        data, horizon=20, minimum_training_samples=60, refit_interval=5
    )
    assert not forecasts.empty
    assert (
        forecasts["latest_training_label_end_date"] <= forecasts["model_fit_date"]
    ).all()
