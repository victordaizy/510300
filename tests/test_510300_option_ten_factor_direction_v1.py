"""510300期权十因子方向模型V1测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.option_ten_factor_direction_v1 import (
    build_factor_frame,
    mature_training_rows,
    model_from_config,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "510300_option_ten_factor_direction_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def synthetic_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", periods=180)
    close = 4000.0 * np.exp(np.arange(len(dates)) * 0.0005)
    benchmark = pd.DataFrame({"date": dates, "close": close})
    etf = pd.DataFrame(
        {"date": dates, "high": 4.0 + np.arange(len(dates)) / 10000, "low": 3.96 + np.arange(len(dates)) / 10000}
    )
    stats = pd.DataFrame(
        {"trade_date": dates, "call_volume": 1000 + np.arange(len(dates)), "put_volume": 900 + np.arange(len(dates))}
    )
    option_rows = []
    for date in dates:
        for code, option_type, delta, iv in (
            ("C25.SH", "C", 0.25, 0.20),
            ("P25.SH", "P", -0.25, 0.24),
            ("CATM.SH", "C", 0.50, 0.22),
            ("PATM.SH", "P", -0.50, 0.23),
        ):
            option_rows.append(
                {
                    "trade_date": date,
                    "expiry_date": date + pd.Timedelta(days=30),
                    "contract_code": code,
                    "option_type": option_type,
                    "is_adjusted": False,
                    "dte": 30,
                    "volume": 100,
                    "implied_volatility": iv,
                    "delta": delta,
                }
            )
    return pd.DataFrame(option_rows), benchmark, stats, etf


def test_factor_frame_contains_exactly_ten_frozen_factors(config: dict) -> None:
    panel, benchmark, stats, etf = synthetic_sources()
    factors = build_factor_frame(panel, benchmark, stats, etf, config)
    columns = config["factors"]["columns"]
    assert len(columns) == 10
    assert all(column in factors.columns for column in columns)
    last = factors.iloc[-20]
    assert last["f09_25delta_put_call_iv_skew"] == pytest.approx(0.04)
    assert pd.notna(last["f10_atm_iv_minus_realized_volatility"])


def test_target_matches_entry_t_plus_one_to_exit_t_plus_eleven(config: dict) -> None:
    panel, benchmark, stats, etf = synthetic_sources()
    factors = build_factor_frame(panel, benchmark, stats, etf, config)
    row = factors.iloc[130]
    expected = benchmark.iloc[141]["close"] / benchmark.iloc[131]["close"] - 1.0
    assert row["forward_entry_exit_return"] == pytest.approx(expected)
    assert row["target_start_date"] == benchmark.iloc[131]["date"]
    assert row["target_end_date"] == benchmark.iloc[141]["date"]


def test_training_uses_only_labels_matured_by_signal_date(config: dict) -> None:
    panel, benchmark, stats, etf = synthetic_sources()
    factors = build_factor_frame(panel, benchmark, stats, etf, config)
    signal_date = benchmark.iloc[150]["date"]
    training = mature_training_rows(factors, signal_date, config)
    assert not training.empty
    assert training["target_end_date"].le(signal_date).all()


def test_model_hyperparameters_are_fixed(config: dict) -> None:
    estimator = model_from_config(config)
    params = estimator.get_params()
    assert params["learning_rate"] == pytest.approx(0.03)
    assert params["max_leaf_nodes"] == 7
    assert params["early_stopping"] is False
    assert len(config["factors"]["columns"]) <= 10
