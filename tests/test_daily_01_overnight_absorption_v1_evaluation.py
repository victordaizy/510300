from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.daily_01_overnight_absorption_v1 import load_config
from research.daily_01_overnight_absorption_v1_evaluation import (
    build_forward_labels,
    circular_block_bootstrap_difference,
    load_evaluation_config,
    newey_west_regression,
    verify_parent_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = load_config()
EVALUATION = load_evaluation_config()


def _features(rows: int = 25) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.full(rows, 4.0),
            "close": 4.0 + np.arange(rows) * 0.01,
        }
    )


def test_d20_label_starts_next_open_and_ends_t_plus_20_close() -> None:
    features = _features()
    dividends = pd.DataFrame(
        {
            "record_date": [features.loc[10, "date"]],
            "ex_date": [features.loc[11, "date"]],
            "payment_date": [features.loc[15, "date"]],
            "cash_dividend_per_share": [1.0],
        }
    )

    labeled = build_forward_labels(features, dividends, PARENT, EVALUATION)

    assert labeled.loc[0, "target_d20_entry_date"] == features.loc[1, "date"]
    assert labeled.loc[0, "target_d20_end_date"] == features.loc[20, "date"]
    assert np.isfinite(labeled.loc[0, "target_d20_net_excess"])
    assert labeled.loc[5, "target_d20_end_date"] is pd.NaT or pd.isna(labeled.loc[5, "target_d20_end_date"])


def test_dividend_before_entry_is_not_granted_to_new_buyer() -> None:
    features = _features()
    dividends = pd.DataFrame(
        {
            "record_date": [features.loc[0, "date"]],
            "ex_date": [features.loc[1, "date"]],
            "payment_date": [features.loc[5, "date"]],
            "cash_dividend_per_share": [10.0],
        }
    )
    with_dividend = build_forward_labels(features, dividends, PARENT, EVALUATION)
    without_dividend = build_forward_labels(
        features,
        dividends.iloc[0:0].copy(),
        PARENT,
        EVALUATION,
    )

    assert with_dividend.loc[0, "target_d20_net_excess"] == pytest.approx(
        without_dividend.loc[0, "target_d20_net_excess"]
    )


def test_newey_west_recovers_positive_incremental_coefficient() -> None:
    x1 = np.linspace(-1.0, 1.0, 200)
    x2 = np.sin(np.linspace(0.0, 8.0, 200))
    design = np.column_stack([np.ones(200), x1, x2])
    y = 0.01 - 0.2 * x1 + 0.5 * x2

    result = newey_west_regression(y, design, lag=20)

    assert result["coefficients"][2] == pytest.approx(0.5)
    assert result["t_values"][2] > 0


def test_circular_block_bootstrap_preserves_positive_group_difference() -> None:
    values = np.array([0.05, 0.00, -0.03] * 100)
    groups = np.array([1, 0, -1] * 100)

    result = circular_block_bootstrap_difference(
        values,
        groups,
        block_length=20,
        repetitions=200,
        random_seed=20260819,
    )

    assert result["lower_95"] > 0
    assert result["valid_repetitions"] == 200


def test_parent_protocol_is_still_frozen_and_unchanged() -> None:
    manifest = verify_parent_protocol(ROOT, EVALUATION)

    assert manifest["predictive_outcomes_computed"] is False
    assert manifest["strategy_backtest_computed"] is False
    assert EVALUATION["evaluation_protocol"]["strategy_backtest_authorized"] is False
