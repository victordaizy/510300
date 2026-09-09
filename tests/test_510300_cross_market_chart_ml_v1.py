from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.cross_market_chart_ml_v1 import (
    FACTOR_COLUMNS,
    add_forward_label,
    build_chart_features,
    fit_model,
    load_config,
    probability_to_exposure,
    simulate_target,
)


CONFIG = load_config()


def _market(periods: int = 420) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    increments = 0.0003 + 0.006 * np.sin(np.arange(periods) / 11.0)
    close = 100.0 * np.exp(np.cumsum(increments))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * (1.0 - 0.001),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
        }
    )


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype=str),
            "record_date": pd.Series(dtype="datetime64[ns]"),
            "ex_date": pd.Series(dtype="datetime64[ns]"),
            "payment_date": pd.Series(dtype="datetime64[ns]"),
            "cash_dividend_per_share": pd.Series(dtype=float),
            "source": pd.Series(dtype=str),
        }
    )


def test_exactly_ten_factors_and_target_is_excluded_from_training() -> None:
    assert len(FACTOR_COLUMNS) == 10
    assert [item["id"] for item in CONFIG["factors"]["definitions"]] == FACTOR_COLUMNS
    assert CONFIG["governance"]["target_labels_used_for_training"] is False
    assert CONFIG["governance"]["target_features_used_for_training"] is False
    assert CONFIG["protocol"]["parameter_rescue_after_results"] is False


def test_features_at_t_do_not_change_when_future_prices_change() -> None:
    market = _market()
    original = build_chart_features(market, CONFIG)
    changed = market.copy()
    cutoff = 330
    changed.loc[cutoff + 1 :, ["open", "high", "low", "close", "adj_close"]] *= 1.75
    modified = build_chart_features(changed, CONFIG)
    np.testing.assert_allclose(
        original.loc[cutoff, FACTOR_COLUMNS].to_numpy(float),
        modified.loc[cutoff, FACTOR_COLUMNS].to_numpy(float),
        rtol=0.0,
        atol=0.0,
    )


def test_label_uses_next_open_and_twenty_bar_holding() -> None:
    featured = build_chart_features(_market(), CONFIG)
    labeled = add_forward_label(featured, CONFIG)
    index = 300
    expected = np.log(featured.loc[index + 21, "adj_open"] / featured.loc[index + 1, "adj_open"]) - 0.0016
    assert labeled.loc[index, "forward_log_return_net"] == pytest.approx(expected)
    assert labeled.loc[index, "entry_date"] == featured.loc[index + 1, "date"]
    assert labeled.loc[index, "exit_date"] == featured.loc[index + 21, "date"]


def test_probability_mapping_is_fixed_relative_to_training_base_rate() -> None:
    probability = np.array([0.50, 0.525, 0.55, 0.575])
    exposure = probability_to_exposure(probability, 0.55, CONFIG)
    assert exposure.tolist() == pytest.approx([0.0, 0.325, 0.65, 0.975])


def test_model_parameters_match_frozen_config() -> None:
    rng = np.random.default_rng(7)
    sample = pd.DataFrame(rng.normal(size=(500, 10)), columns=FACTOR_COLUMNS)
    sample["label_positive"] = (sample["MOM20_VOL"] + sample["VOL_RATIO_5_60"] > 0.0).astype(int)
    sample["symbol"] = "TEST"
    bundle = fit_model(sample, CONFIG)
    model = bundle["model"]
    assert model.max_iter == 300
    assert model.max_leaf_nodes == 15
    assert model.max_depth == 4
    assert model.min_samples_leaf == 100
    assert model.early_stopping is False


def test_target_signal_executes_only_at_next_open_and_blocks_small_adjustment() -> None:
    config = deepcopy(CONFIG)
    config["target_execution"]["initial_capital_cny"] = 12000.0
    dates = pd.bdate_range("2026-01-05", periods=5)
    features = pd.DataFrame(
        {
            "date": dates,
            "open": [4.0, 4.0, 4.0, 4.0, 4.0],
            "close": [4.0, 4.0, 4.0, 4.0, 4.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "date": [dates[0], dates[2]],
            "probability_positive": [0.70, 0.56],
            "target_exposure": [0.975, 0.65],
        }
    )
    result = simulate_target(
        features,
        predictions,
        _empty_dividends(),
        config,
        slippage_bps=5.0,
    )
    trades = result["trades"]
    assert len(trades) == 1
    assert trades.iloc[0]["signal_date"] == dates[0]
    assert trades.iloc[0]["execution_date"] == dates[1]
    assert trades.iloc[0]["raw_notional"] >= 5000.0
    assert len(result["blocked"]) == 1
    assert result["blocked"].iloc[0]["execution_date"] == dates[3]


def test_source_files_do_not_contain_noncausal_backfill_switches() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "research" / "cross_market_chart_ml_v1.py").read_text(encoding="utf-8")
    assert "bfill()" in source  # 仅H00300日期对齐，不用于任何模型特征
    feature_block = source[source.index("def build_chart_features") : source.index("def add_forward_label")]
    assert "shift(-" not in feature_block
    assert "center=True" not in feature_block
