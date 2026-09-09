from __future__ import annotations

import pandas as pd
import pytest

from research.normalized_valuation_5y_predictive_screen_v1 import (
    compute_model_diagnostics,
    holm_bonferroni,
)
from research.val04_pb_roe_residual_predictive_screen_v1 import MODEL_ID


def _config() -> dict:
    return {
        "models": {
            MODEL_ID: {
                "predictor_column": "pb_roe_residual_percentile_60m",
                "ready_column": "pb_roe_residual_percentile_60m_ready",
            }
        },
        "label_contract": {"horizons_trading_days": [242]},
        "diagnostic_definition": {"hac_monthly_max_lags": {242: 12}},
    }


def test_single_registered_candidate_holm_is_identity() -> None:
    result = holm_bonferroni({MODEL_ID: 0.03})
    assert result["adjusted_p_values"][MODEL_ID] == 0.03


def test_predictive_diagnostics_reject_signal_label_date_mismatch() -> None:
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-07-30", "2021-08-31"]),
            "pb_roe_residual_percentile_60m": [0.2, 0.8],
            "pb_roe_residual_percentile_60m_ready": [True, True],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_observation_date": pd.to_datetime(["2021-07-30"]),
            "horizon_trading_days": [242],
            "label_status": ["MATURED"],
            "etf_total_return": [0.1],
            "h00300_total_return": [0.1],
        }
    )
    with pytest.raises(ValueError, match="信号日与冻结标签日不一致"):
        compute_model_diagnostics(signals, labels, MODEL_ID, _config())


def test_positive_ordering_produces_positive_spearman() -> None:
    dates = pd.date_range("2021-01-31", periods=8, freq="ME")
    percentile = pd.Series([0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95])
    signals = pd.DataFrame(
        {
            "date": dates,
            "pb_roe_residual_percentile_60m": percentile,
            "pb_roe_residual_percentile_60m_ready": True,
        }
    )
    labels = pd.DataFrame(
        {
            "signal_observation_date": dates,
            "horizon_trading_days": 242,
            "label_status": "MATURED",
            "etf_total_return": percentile * 0.2,
            "h00300_total_return": percentile * 0.18,
        }
    )
    result = compute_model_diagnostics(signals, labels, MODEL_ID, _config())
    assert result["242"]["etf_total_return"]["spearman_ic"] == 1.0
    assert result["242"]["h00300_total_return"]["spearman_ic"] == 1.0
