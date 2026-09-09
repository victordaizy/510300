"""IVS迁移V1的2024严格验证器测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.option_ivs_transfer_v1_validation_2024 import validation_predictions


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def formula_config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "510300_option_ivs_transfer_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def validation_config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "510300_option_ivs_transfer_v1_validation_2024.yaml").read_text(
            encoding="utf-8"
        )
    )


def synthetic_monthly() -> pd.DataFrame:
    signals = pd.date_range("2019-12-31", "2024-11-30", freq="ME")
    entries = signals + pd.offsets.BDay(1)
    exits = signals + pd.offsets.MonthEnd(1)
    return pd.DataFrame(
        {
            "month": signals.to_period("M").astype(str),
            "signal_date": signals,
            "entry_date": entries,
            "exit_date": exits,
            "ivs_date": signals,
            "ivs_staleness_calendar_days": 0,
            "ivs": [-0.04 + (index % 12) * 0.002 for index in range(len(signals))],
            "forward_return": [0.01 if index % 3 else -0.02 for index in range(len(signals))],
        }
    )


def test_validation_only_contains_2024_entry_and_exit(
    formula_config: dict, validation_config: dict
) -> None:
    result = validation_predictions(synthetic_monthly(), formula_config, validation_config)
    assert not result.empty
    assert result["entry_date"].ge(pd.Timestamp("2024-01-01")).all()
    assert result["exit_date"].le(pd.Timestamp("2024-12-31")).all()
    assert result.iloc[0]["signal_date"].year == 2023


def test_validation_beta_is_nonpositive_and_training_is_mature(
    formula_config: dict, validation_config: dict
) -> None:
    result = validation_predictions(synthetic_monthly(), formula_config, validation_config)
    ready = result.loc[result["status"].eq("MODEL_READY")]
    assert not ready.empty
    assert ready["beta"].le(0).all()
    assert ready["latest_training_exit"].le(ready["signal_date"]).all()


def test_holdout_is_forbidden_and_threshold_unchanged(
    formula_config: dict, validation_config: dict
) -> None:
    assert validation_config["final_holdout"]["access_authorized"] is False
    assert validation_config["validation"]["annualized_excess_minimum"] == pytest.approx(
        formula_config["evaluation"]["annualized_excess_minimum"]
    )
    assert validation_config["protocol"]["formula_or_threshold_change_allowed"] is False
