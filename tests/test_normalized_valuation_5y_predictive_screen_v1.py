from __future__ import annotations

import pandas as pd

from research.normalized_valuation_5y_predictive_screen_v1 import (
    audit_frozen_labels,
    frozen_bucket,
    holm_bonferroni,
)


def test_frozen_bucket_boundaries() -> None:
    assert frozen_bucket(0.0) == "B1_EXPENSIVE"
    assert frozen_bucket(0.199999) == "B1_EXPENSIVE"
    assert frozen_bucket(0.20) == "B2"
    assert frozen_bucket(0.40) == "B3"
    assert frozen_bucket(0.60) == "B4_CHEAP"
    assert frozen_bucket(1.0) == "B4_CHEAP"


def test_holm_two_model_adjustment() -> None:
    result = holm_bonferroni({"VAL01": 0.02, "VAL02": 0.08})
    assert result["adjusted_p_values"]["VAL01"] == 0.04
    assert result["adjusted_p_values"]["VAL02"] == 0.08


def test_holm_smallest_failure_blocks_it_at_family_alpha() -> None:
    result = holm_bonferroni({"VAL01": 0.06, "VAL02": 0.08})
    assert result["adjusted_p_values"]["VAL01"] == 0.12
    assert result["adjusted_p_values"]["VAL02"] == 0.12


def test_label_audit_rejects_signal_columns() -> None:
    labels = pd.DataFrame(
        {
            "signal_observation_date": ["2021-07-30"],
            "horizon_trading_days": [60],
            "label_status": ["MATURED"],
            "etf_total_return": [0.1],
            "h00300_total_return": [0.1],
            "norm_ey_percentile_60m": [0.5],
        }
    )
    config = {
        "label_contract": {
            "horizons_trading_days": [60],
            "expected_matured_counts": {60: 1},
            "expected_right_censored_counts": {60: 0},
            "expected_label_rows": 1,
            "expected_signal_dates": 1,
        }
    }
    audit = audit_frozen_labels(labels, config)
    assert audit["status"] == "BLOCKED"
    assert audit["forbidden_signal_columns"] == ["norm_ey_percentile_60m"]
