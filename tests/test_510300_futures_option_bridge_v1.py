"""七因子期货—期权—现货桥梁测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.futures_option_bridge_v1 import build_features, fit_predict_probability


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "510300_futures_option_bridge_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def synthetic_inputs(periods: int = 220) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2021-01-01", periods=periods)
    trend = np.arange(periods, dtype=float)
    futures = pd.DataFrame(
        {
            "date": dates,
            "close": 4000.0 + trend,
            "volume": 10000.0 + trend,
            "open_interest": 20000.0 + trend,
        }
    )
    spot = pd.DataFrame({"date": dates, "close": 3990.0 + trend * 0.9})
    stats = pd.DataFrame(
        {
            "trade_date": dates,
            "call_volume": 1000.0 + trend,
            "put_volume": 900.0 + trend,
            "call_open_interest": 2000.0 + trend,
            "put_open_interest": 1800.0 + trend,
        }
    )
    benchmark = pd.DataFrame({"date": dates, "close": 1000.0 + trend})
    return futures, spot, stats, benchmark


def test_feature_values_before_cut_are_unchanged_by_future_rows(config: dict) -> None:
    futures, spot, stats, benchmark = synthetic_inputs()
    short = build_features(
        futures.iloc[:180], spot.iloc[:180], stats.iloc[:180], benchmark.iloc[:180], config
    )
    full = build_features(futures, spot, stats, benchmark, config)
    columns = ["date", *config["factors"]]
    pd.testing.assert_frame_equal(
        short[columns].reset_index(drop=True),
        full.loc[full["date"].isin(short["date"]), columns].reset_index(drop=True),
    )


def test_target_starts_at_t_plus_one_and_exits_after_20_days(config: dict) -> None:
    futures, spot, stats, benchmark = synthetic_inputs()
    features = build_features(futures, spot, stats, benchmark, config)
    first = features.iloc[0]
    expected = benchmark.iloc[21]["close"] / benchmark.iloc[1]["close"] - 1.0
    assert first["target_up"] == float(expected >= 0)
    assert first["target_exit_date"] == benchmark.iloc[21]["date"]


def test_prediction_refuses_immature_training_set(config: dict) -> None:
    rows = int(config["model"]["minimum_mature_training_rows"]) - 1
    frame = pd.DataFrame({name: np.arange(rows, dtype=float) for name in config["factors"]})
    frame["target_up"] = np.arange(rows) % 2
    assert fit_predict_probability(frame, frame.iloc[-1], config) is None


def test_factor_count_is_within_user_limit(config: dict) -> None:
    assert len(config["factors"]) == config["protocol"]["factor_count"]
    assert len(config["factors"]) <= 10
