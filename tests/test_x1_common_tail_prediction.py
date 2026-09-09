"""X1共同尾部信号、未来目标与走步概率的防前视测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from research.x1_common_tail_prediction import (
    X1Rules,
    build_return_tail_outcomes,
    build_x1_common_tail_signal,
    evaluate_x1_prediction,
    walk_forward_x1_probabilities,
)


R6_FEATURES = (
    "earnings_yield",
    "book_yield",
    "trend20",
    "trend120",
    "rv20",
    "rv60",
)


def _rules(**overrides: object) -> X1Rules:
    values: dict[str, object] = {
        "data_cutoff": pd.Timestamp("2026-12-31"),
        "component_lookback": 2,
        "component_tail_quantile": 0.05,
        "component_minimum_history": 3,
        "minimum_signal_weight_coverage": 0.90,
        "risk_minimum_history": 2,
        "top_risk_percentile": 0.90,
        "industry_shock_percentile": 0.90,
        "cash_annual_rate": 0.015,
        "trading_days_per_year": 242,
        "primary_horizon": 2,
        "primary_bad_threshold": -0.05,
        "primary_tail_threshold": -0.08,
        "robust_horizon": 3,
        "robust_bad_threshold": -0.08,
        "robust_tail_threshold": -0.12,
        "minimum_probability_training": 2,
        "probability_clip": (0.000001, 0.999999),
        "logistic_c": 1.0,
        "r6_features": R6_FEATURES,
        "augmented_features": (*R6_FEATURES, "x1_risk_percentile"),
        "pseudo_oos_start": pd.Timestamp("2025-03-31"),
        "bootstrap_repetitions": 100,
        "bootstrap_block_length": 2,
        "random_seed": 20260818,
        "minimum_bad20_lift": 2.0,
        "bootstrap_lower_bound": 1.0,
        "minimum_brier_skill": 0.0,
        "minimum_delay_retention": 0.70,
        "split_lift_lower_bound": 1.0,
    }
    values.update(overrides)
    return X1Rules(**values)


def _signal_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=12)
    symbols = [f"S{index:03d}" for index in range(300)]
    members = pd.DataFrame(
        [
            {"date": date, "con_code": symbol, "is_index_member": True}
            for date in dates
            for symbol in symbols
        ]
    )
    history = pd.DataFrame(
        [
            {
                "date": date,
                "con_code": symbol,
                "total_return_close": 100.0
                * (1.0 + 0.001 * date_index)
                * (1.0 + 0.00001 * symbol_index),
            }
            for date_index, date in enumerate(dates)
            for symbol_index, symbol in enumerate(symbols)
        ]
    )
    weights = pd.DataFrame(
        {
            "trade_date": dates[0],
            "con_code": symbols,
            "weight": 100.0 / len(symbols),
        }
    )
    industries = pd.DataFrame(
        {
            "con_code": symbols,
            "industry_l1": [f"行业{index % 10}" for index in range(300)],
            "in_date": pd.Timestamp("2020-01-01"),
            "out_date": pd.NaT,
        }
    )
    return members, history, weights, industries


def test_signal_uses_strict_prior_history_and_future_price_cannot_rewrite_past() -> None:
    inputs = _signal_inputs()
    first = build_x1_common_tail_signal(*inputs, _rules())
    changed_history = inputs[1].copy()
    changed_history.loc[
        changed_history["date"].eq(changed_history["date"].max()),
        "total_return_close",
    ] *= 0.50
    second = build_x1_common_tail_signal(
        inputs[0], changed_history, inputs[2], inputs[3], _rules()
    )
    cutoff = first["date"].max() - pd.Timedelta(days=1)
    left = first.loc[first["date"].le(cutoff), "x1_raw_common_tail_weight"]
    right = second.loc[second["date"].le(cutoff), "x1_raw_common_tail_weight"]
    assert np.allclose(left, right, equal_nan=True)
    assert first["member_count"].eq(300).all()
    assert first.loc[first["signal_output"].eq("SIGNAL_READY"), "tail_signal_weight_coverage"].ge(0.90).all()


def test_outcome_starts_next_open_and_delay_restarts_at_t_plus_two() -> None:
    dates = pd.bdate_range("2025-01-02", periods=6)
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 10.0, 12.0, 12.0, 12.0, 12.0],
            "low": [10.0, 9.0, 11.0, 12.0, 12.0, 12.0],
            "close": [10.0, 11.0, 12.0, 12.0, 12.0, 12.0],
        }
    )
    dividends = pd.DataFrame(
        {
            "symbol": ["510300.SH"],
            "ex_date": [dates[2]],
            "cash_dividend_per_share": [1.0],
        }
    )
    result = build_return_tail_outcomes(market, dividends, [dates[0]], _rules())
    row = result.iloc[0]
    cash2 = (1.0 + 0.015) ** (2.0 / 242.0)
    assert row["entry_date20"] == dates[1]
    assert row["entry_date20_delay1"] == dates[2]
    assert np.isclose(row["x20"], (11.0 / 10.0) * (13.0 / 11.0) - cash2)
    assert np.isclose(row["x20_delay1"], 12.0 / 12.0 - cash2)


def _walk_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(
        ["2025-01-02", "2025-02-03", "2025-03-31", "2025-04-30", "2025-05-30", "2025-06-30"]
    )
    signal = pd.DataFrame(
        {
            "date": dates,
            "signal_output": "SIGNAL_READY",
            "x1_raw_common_tail_weight": np.linspace(0.01, 0.12, len(dates)),
            "x1_risk_percentile": np.linspace(0.20, 0.95, len(dates)),
        }
    )
    outcomes = pd.DataFrame(
        {
            "signal_date": dates,
            "maturity_date20": dates + pd.Timedelta(days=10),
            "bad20": [False, True, False, True, False, True],
            "tail20": [False, True, False, False, False, True],
            "bad20_delay1": [False, True, False, True, False, True],
            "x20": [0.02, -0.08, 0.01, -0.06, 0.03, -0.07],
        }
    )
    features = pd.DataFrame({"date": dates})
    for index, name in enumerate(R6_FEATURES):
        features[name] = np.linspace(0.01 + index, 0.06 + index, len(dates))
    return signal, outcomes, features


def test_walk_forward_probability_ignores_current_and_future_labels() -> None:
    signal, outcomes, features = _walk_inputs()
    first_analysis, first = walk_forward_x1_probabilities(
        signal, outcomes, features, _rules()
    )
    changed = outcomes.copy()
    changed.loc[changed["signal_date"].ge(pd.Timestamp("2025-03-31")), "bad20"] = True
    _, second = walk_forward_x1_probabilities(signal, changed, features, _rules())
    assert not first_analysis.empty
    assert first.iloc[0]["date"] == pd.Timestamp("2025-03-31")
    probability_columns = [
        "unconditional_probability",
        "x1_probability",
        "r6_probability",
        "r6_plus_x1_probability",
    ]
    assert np.allclose(
        first.iloc[0][probability_columns].astype(float),
        second.iloc[0][probability_columns].astype(float),
    )
    assert first.iloc[0]["training_mature_observations"] == 2


def test_evaluation_never_builds_position_policy() -> None:
    dates = pd.bdate_range("2025-01-02", periods=40)
    bad = np.array([index % 10 == 0 for index in range(40)])
    risk = np.linspace(0.01, 1.0, len(dates))
    analysis = pd.DataFrame(
        {
            "date": dates,
            "x1_risk_percentile": risk,
            "bad20": bad,
            "tail20": bad,
            "bad20_delay1": bad,
            "bad60": bad,
            "tail60": bad,
        }
    )
    forecasts = pd.DataFrame(
        {
            "date": dates,
            "bad20": bad,
            "x20": np.where(bad, -0.06, 0.02),
            "unconditional_probability": 0.10,
            "x1_probability": np.linspace(0.05, 0.15, len(dates)),
            "r6_probability": np.linspace(0.04, 0.14, len(dates)),
            "r6_plus_x1_probability": np.linspace(0.05, 0.15, len(dates)),
        }
    )
    rules = replace(_rules(), pseudo_oos_start=dates[0])
    result = evaluate_x1_prediction(analysis, forecasts, rules)
    assert result["candidate_id"] == "X1"
    assert result["safety"] == {
        "forecast_eligible_emitted": False,
        "combination_model_built": False,
        "position_policy_built": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
