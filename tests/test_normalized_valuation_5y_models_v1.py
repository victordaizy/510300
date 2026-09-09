from __future__ import annotations

import numpy as np
import pandas as pd

from research.normalized_valuation_5y_models_v1 import (
    apply_frozen_percentile,
    project_model_signals,
)


def _synthetic_common_panel() -> pd.DataFrame:
    dates = pd.date_range("2016-08-31", periods=60, freq="ME")
    values = np.arange(1, 61, dtype=float) / 1000.0
    return pd.DataFrame(
        {
            "date": dates,
            "signal_observation_date": dates,
            "earliest_execution_date": dates + pd.offsets.BDay(1),
            "signal_available_after": "SNAPSHOT_DATE_CLOSE",
            "earliest_execution_time": "NEXT_INDEX_TRADING_DAY_OPEN",
            "constituent_count": 300,
            "weight_sum": 1.0,
            "price_weight_coverage": 1.0,
            "normalized_earnings_weight_coverage": 1.0,
            "normalized_metric_weight_coverage": 1.0,
            "normalization_company_count": 300,
            "known_financial_event_count": 1000,
            "historical_evidence_label": "HISTORICALLY_CONTAMINATED_NOT_STRICT_OOS",
            "weighted_normalized_earnings_yield": values,
            "val01_input_status": "PASS",
            "norm_ey_percentile_60m_window_observations": np.arange(1, 61),
            "norm_ey_percentile_60m": [np.nan] * 59 + [59.5 / 60.0],
            "norm_ey_percentile_60m_ready": [False] * 59 + [True],
            "cgb_10y": 2.5,
            "cgb_10y_decimal": 0.025,
            "cgb_10y_available_exact_date": True,
            "normalized_ey_spread": values - 0.025,
            "val02_input_status": "PASS",
            "norm_ey_spread_percentile_60m_window_observations": np.arange(1, 61),
            "norm_ey_spread_percentile_60m": [np.nan] * 59 + [59.5 / 60.0],
            "norm_ey_spread_percentile_60m_ready": [False] * 59 + [True],
        }
    )


def test_midrank_requires_full_continuous_window() -> None:
    panel = pd.DataFrame(
        {
            "value": np.arange(1, 61, dtype=float),
            "status": ["PASS"] * 60,
        }
    )
    result = apply_frozen_percentile(panel, "value", "status", "q", 60)
    assert result["q_ready"].sum() == 1
    assert result.loc[59, "q"] == 59.5 / 60.0


def test_midrank_is_blocked_by_one_failed_month() -> None:
    panel = pd.DataFrame(
        {
            "value": np.arange(1, 61, dtype=float),
            "status": ["PASS"] * 59 + ["BLOCKED"],
        }
    )
    result = apply_frozen_percentile(panel, "value", "status", "q", 60)
    assert not result["q_ready"].any()
    assert pd.isna(result.loc[59, "q"])


def test_val01_projection_excludes_bond_and_spread_columns() -> None:
    config = {"models": {}}
    signals = project_model_signals(
        _synthetic_common_panel(), "VAL01_NORM_EY_5Y_V1", config
    )
    assert "cgb_10y" not in signals.columns
    assert "normalized_ey_spread" not in signals.columns
    assert "norm_ey_percentile_60m" in signals.columns


def test_val02_projection_contains_exact_date_bond_and_spread() -> None:
    config = {"models": {}}
    signals = project_model_signals(
        _synthetic_common_panel(), "VAL02_NORM_EY_SPREAD_5Y_V1", config
    )
    assert signals.loc[0, "cgb_10y_decimal"] == 0.025
    assert signals.loc[0, "normalized_ey_spread"] == 0.001 - 0.025
    assert "norm_ey_spread_percentile_60m" in signals.columns


def test_signal_projections_contain_no_return_position_or_order_fields() -> None:
    panel = _synthetic_common_panel()
    for model_id in (
        "VAL01_NORM_EY_5Y_V1",
        "VAL02_NORM_EY_SPREAD_5Y_V1",
    ):
        signals = project_model_signals(panel, model_id, {"models": {}})
        lowered = " ".join(signals.columns).lower()
        assert "return" not in lowered
        assert "position" not in lowered
        assert "order" not in lowered
