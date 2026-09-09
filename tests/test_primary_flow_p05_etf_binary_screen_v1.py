"""主流量概率0.5二元映射的时点与状态测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from research.primary_flow_p05_etf_binary_screen_v1 import (
    audit_and_build_daily_states,
    load_config,
)


def _market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-02", periods=6, freq="B"),
            "open": 4.0,
            "close": 4.0,
        }
    )


def _forecasts() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    return pd.DataFrame(
        {
            "signal_date": dates,
            "model_fit_date": dates,
            "latest_training_label_end_date": dates,
            "training_sample_count": [504, 505, 506, 507],
            "direction_probability_positive_20d_net": [0.49, 0.50, 0.80, 0.20],
            "share_snapshot_date": dates,
            "share_snapshot_age_calendar_days": [0, 0, 0, 0],
        }
    )


def test_exact_half_probability_enters_full_state_and_state_persists() -> None:
    daily, states, audit = audit_and_build_daily_states(
        _market(), _forecasts(), threshold=0.50
    )
    assert states.tolist() == [0, 1, 1, 0, 0, 0]
    assert daily["desired_state_at_close"].tolist() == states.tolist()
    assert audit["signal_full_share"] == 0.5
    assert audit["future_outcome_columns_read"] == 0


def test_training_label_after_fit_is_rejected() -> None:
    forecasts = _forecasts()
    forecasts.loc[0, "latest_training_label_end_date"] = pd.Timestamp("2024-01-03")
    with pytest.raises(ValueError, match="拟合日之后"):
        audit_and_build_daily_states(_market(), forecasts, threshold=0.50)


def test_protocol_reads_no_realized_outcome_columns() -> None:
    config = load_config()
    frozen = config["data_contracts"]["frozen_forecasts"]
    permitted = set(frozen["permitted_columns_to_read"])
    forbidden = set(frozen["forbidden_columns_as_signal"])
    assert permitted.isdisjoint(forbidden)
    assert float(config["state_rule"]["threshold"]) == 0.50
    assert config["state_rule"]["threshold_search"] == "FORBIDDEN"
