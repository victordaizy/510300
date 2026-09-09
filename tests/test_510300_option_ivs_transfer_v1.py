"""510300期权IVS外部规则迁移V1测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.option_ivs_transfer_v1 import (
    build_daily_ivs,
    build_monthly_rows,
    expanding_predictions,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "510300_option_ivs_transfer_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_open_interest_weighted_paired_ivs_formula(config: dict) -> None:
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2023-01-03"] * 4),
            "expiry_date": pd.to_datetime(["2023-02-22"] * 4),
            "strike": [4.0, 4.0, 4.1, 4.1],
            "option_type": ["C", "P", "C", "P"],
            "is_adjusted": [False] * 4,
            "dte": [50] * 4,
            "open_interest": [100, 300, 100, 100],
            "implied_volatility": [0.20, 0.24, 0.25, 0.26],
        }
    )
    daily = build_daily_ivs(panel, config)
    expected = ((0.20 - 0.24) * 400 + (0.25 - 0.26) * 200) / 600
    assert daily.iloc[0]["ivs"] == pytest.approx(expected)
    assert daily.iloc[0]["pair_count"] == 2


def test_monthly_target_is_next_entry_to_next_month_end(config: dict) -> None:
    dates = pd.bdate_range("2020-01-01", "2020-04-30")
    benchmark = pd.DataFrame({"date": dates, "close": range(100, 100 + len(dates))})
    daily_ivs = pd.DataFrame({"date": dates, "ivs": [-0.02] * len(dates)})
    monthly = build_monthly_rows(benchmark, daily_ivs, config)
    first = monthly.iloc[0]
    expected_entry = dates[dates > first["signal_date"]][0]
    expected_exit = dates[dates.to_period("M") == pd.Period("2020-02")][-1]
    assert first["entry_date"] == expected_entry
    assert first["exit_date"] == expected_exit


def test_expanding_regression_uses_only_mature_months_and_nonpositive_beta(config: dict) -> None:
    dates = pd.bdate_range("2019-12-02", "2023-12-29")
    benchmark = pd.DataFrame({"date": dates, "close": 4000 + pd.Series(range(len(dates))) * 0.5})
    daily_ivs = pd.DataFrame(
        {"date": dates, "ivs": [(-0.03 + (index % 20) / 1000) for index in range(len(dates))]}
    )
    monthly = build_monthly_rows(benchmark, daily_ivs, config)
    predictions = expanding_predictions(monthly, config)
    ready = predictions.loc[predictions["status"].eq("MODEL_READY")]
    assert not ready.empty
    assert ready["beta"].le(0).all()
    assert ready["latest_training_exit"].le(ready["signal_date"]).all()


def test_external_rule_has_one_factor_and_no_direct_generalization(config: dict) -> None:
    assert config["protocol"]["factor_count"] == 1
    assert config["protocol"]["factor_count"] <= 10
    assert config["research_basis"]["original_underlying"] == "SSE50 ETF"
    assert config["research_basis"]["direct_generalization_claim"] is False
