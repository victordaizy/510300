"""测试收盘压力两日方向挑战者的执行标签与防前视约束。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.intraday_close_pressure_direction_challenger import (
    attach_two_day_outcomes,
    walk_forward_forecasts,
)


TARGET = {
    "initial_cash_cny": 20000.0,
    "lot_size": 100,
    "commission_rate": 0.0003,
    "minimum_commission_cny": 5.0,
    "slippage_bps_per_leg": 5.0,
    "stamp_duty_rate": 0.0,
}


def test_two_day_label_uses_next_open_and_following_close() -> None:
    dates = pd.bdate_range("2025-01-02", periods=5)
    features = pd.DataFrame({"date": dates, "close_pressure_30m_60d": np.arange(5, dtype=float)})
    daily = pd.DataFrame(
        {"date": dates, "open": [10, 11, 12, 13, 14], "close": [10.5, 11.5, 12.5, 13.5, 14.5]}
    )
    dividends = pd.DataFrame(
        {"record_date": [dates[1]], "cash_dividend_per_share": [0.1]}
    )
    result = attach_two_day_outcomes(features, daily, dividends, TARGET)
    assert result.loc[0, "entry_date"] == dates[1]
    assert result.loc[0, "exit_date"] == dates[2]
    assert result.loc[0, "entry_open"] == 11
    assert result.loc[0, "exit_close"] == 12.5
    assert result.loc[0, "holding_dividend_per_share"] == pytest.approx(0.1)
    assert pd.isna(result.loc[3, "exec_total_return_2d_net"])


def test_walk_forward_only_uses_realized_labels() -> None:
    generator = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-04", periods=80)
    feature = generator.normal(size=len(dates))
    net_return = 0.003 * feature + generator.normal(scale=0.01, size=len(dates))
    data = pd.DataFrame(
        {
            "date": dates,
            "close_pressure_30m_60d": feature,
            "exec_total_return_2d_net": net_return,
            "direction_label": net_return > 0,
            "label_end_date": pd.Series(dates).shift(-2),
        }
    )
    forecasts = walk_forward_forecasts(data, minimum_training_samples=20, refit_interval=5)
    assert not forecasts.empty
    assert (forecasts["latest_training_label_end_date"] <= forecasts["signal_date"]).all()
    assert forecasts["training_sample_count"].min() >= 20
