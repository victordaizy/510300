from __future__ import annotations

import numpy as np
import pandas as pd

from research.structural_prediction_v1 import (
    attach_price_features_and_targets,
    build_increment_results,
    build_prediction_state_panel,
    evaluate_predictions,
    validation_summary,
    walk_forward_predictions,
)


def _condition_state(origins: pd.DatetimeIndex) -> pd.DataFrame:
    count = len(origins)
    values = np.linspace(-1.0, 1.0, count)
    return pd.DataFrame(
        {
            "origin": origins,
            "condition_quality_status": ["PASS_FROZEN_PARENT_STATES"] * count,
            "cf_level": values,
            "cf_news": np.gradient(values),
            "cf_breadth": np.linspace(0.4, 0.6, count),
            "cf_concentration": np.linspace(0.2, 0.1, count),
            "cash_flow_expectation_factor": values * 0.5,
            "cross_sectional_expected_return_factor": -values,
            "equity_risk_premium_expanding_z": values * 0.2,
            "discount_rate_easing_news": -np.gradient(values * 0.2),
            "valuation_concentration": np.linspace(0.1, 0.2, count),
            "duration_compression": np.linspace(-0.2, 0.2, count),
            "risk_bearing_capacity_factor": values * 0.3,
            "risk_capacity_news": np.gradient(values * 0.3),
            "member_positive_breadth_60d": np.linspace(0.3, 0.7, count),
            "large_weight_minus_median_member_return_60d": values * 0.01,
            "industry_negative_breadth_20d": np.linspace(0.7, 0.3, count),
        }
    )


def test_prediction_state_is_asof_and_builds_fixed_interactions() -> None:
    origins = pd.to_datetime(["2021-01-29", "2021-02-26", "2021-03-31"])
    condition = _condition_state(origins)
    present_value = pd.DataFrame(
        {
            "origin": origins,
            "cap_weighted_positive_earnings_yield": [0.05, 0.06, 0.07],
            "cap_weighted_book_to_price": [0.5, 0.6, 0.7],
            "cap_weighted_sales_to_price": [0.8, 0.9, 1.0],
        }
    )
    old = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-28", "2021-02-25", "2021-03-30"]),
            "index_close": [5000.0, 5100.0, 5200.0],
            "pe_ttm": [15.0, 16.0, 17.0],
            "pb": [1.5, 1.6, 1.7],
            "fair_mid_12m": [5500.0, 5400.0, 5300.0],
            "target_position": [1.0, 0.5, 0.5],
            "valuation_state": ["满仓", "半仓", "半仓"],
        }
    )
    result = build_prediction_state_panel(
        condition,
        present_value,
        old,
        old_r5_asof_tolerance_calendar_days=10,
    )

    assert (result["old_r5_asof_date"] <= result["origin"]).all()
    assert np.isclose(result.loc[0, "old_r5_valuation_gap_mid_12m"], 0.10)
    assert np.isclose(
        result.loc[1, "cf_news_x_dr_easing_news"],
        result.loc[1, "cf_news"] * result.loc[1, "discount_rate_easing_news"],
    )
    assert not result["h00300_values_read"].any()


def test_price_features_use_only_past_and_targets_use_exact_future_offsets() -> None:
    dates = pd.bdate_range("2020-01-01", periods=230)
    closes = np.linspace(100.0, 200.0, 230)
    closes[181] = 150.0
    total_return = pd.DataFrame({"date": dates, "close": closes})
    state = pd.DataFrame({"origin": [dates[160], dates[210]]})
    result = attach_price_features_and_targets(
        state,
        total_return,
        observation_cutoff=dates[-1],
        return_horizons_market_days=[60, 120],
        drawdown_horizon_market_days=20,
        drawdown_threshold=-0.10,
    )

    assert np.isclose(
        result.loc[0, "trailing_total_return_60d"],
        closes[160] / closes[100] - 1.0,
    )
    assert np.isclose(
        result.loc[0, "target_return_60d"],
        closes[220] / closes[160] - 1.0,
    )
    expected_minimum = np.min(closes[161:181] / closes[160] - 1.0)
    assert np.isclose(
        result.loc[0, "target_drawdown_20d_minimum_return"], expected_minimum
    )
    assert pd.isna(result.loc[1, "target_return_60d"])
    assert result["h00300_values_read"].all()


def test_walk_forward_training_uses_only_strictly_matured_labels() -> None:
    origins = pd.date_range("2016-01-31", periods=60, freq="ME")
    frame = pd.DataFrame(
        {
            "origin": origins,
            "condition_quality_status": ["PASS_FROZEN_PARENT_STATES"] * 60,
            "feature_a": np.linspace(-1.0, 1.0, 60),
            "target_return_60d": np.linspace(-0.05, 0.10, 60),
            "target_return_60d_maturity_date": origins + pd.Timedelta(days=75),
        }
    )
    target_specs = {
        "EXPECTED_60D_TOTAL_RETURN": {
            "target_type": "CONTINUOUS_RETURN",
            "horizon_market_days": 60,
        }
    }
    model_specs = {
        "HISTORICAL_MEAN": {"feature_groups": []},
        "RIDGE_TEST": {"feature_groups": ["group_a"]},
    }
    walk = {
        "minimum_matured_training_observations": 12,
        "continuous_model": {"alpha": 10.0},
        "binary_model": {
            "c": 0.1,
            "maximum_iterations": 1000,
            "minimum_matured_positive_events": 2,
            "minimum_matured_negative_events": 5,
        },
    }
    predictions = walk_forward_predictions(
        frame,
        target_specs=target_specs,
        model_specs=model_specs,
        feature_groups={"group_a": ["feature_a"]},
        walk_forward=walk,
    )
    generated = predictions.loc[predictions["prediction"].notna()]

    assert not generated.empty
    assert (
        pd.to_datetime(generated["training_max_maturity_date"])
        < pd.to_datetime(generated["origin"])
    ).all()
    assert generated["training_observation_count"].ge(12).all()


def test_binary_model_preserves_class_training_gate() -> None:
    origins = pd.date_range("2016-01-31", periods=60, freq="ME")
    events = np.zeros(60)
    events[[5, 15, 25, 35, 45, 55]] = 1.0
    frame = pd.DataFrame(
        {
            "origin": origins,
            "condition_quality_status": ["PASS_FROZEN_PARENT_STATES"] * 60,
            "feature_a": np.linspace(-1.0, 1.0, 60),
            "target_drawdown_20d_event": events,
            "target_drawdown_20d_maturity_date": origins + pd.Timedelta(days=30),
        }
    )
    predictions = walk_forward_predictions(
        frame,
        target_specs={
            "20D_DRAWDOWN_PROBABILITY": {
                "target_type": "BINARY_PROBABILITY",
                "horizon_market_days": 20,
            }
        },
        model_specs={
            "HISTORICAL_MEAN": {"feature_groups": []},
            "LOGISTIC_TEST": {"feature_groups": ["group_a"]},
        },
        feature_groups={"group_a": ["feature_a"]},
        walk_forward={
            "minimum_matured_training_observations": 12,
            "continuous_model": {"alpha": 10.0},
            "binary_model": {
                "c": 0.1,
                "maximum_iterations": 1000,
                "minimum_matured_positive_events": 3,
                "minimum_matured_negative_events": 10,
            },
        },
    )

    logistic = predictions.loc[predictions["model_id"].eq("LOGISTIC_TEST")]
    assert logistic["prediction_status"].eq("NO_VIEW_BINARY_CLASS_TRAINING_GATE").any()
    generated = logistic.loc[logistic["prediction"].notna(), "prediction"]
    assert not generated.empty
    assert generated.between(0.0, 1.0).all()


def test_common_origin_metrics_and_increment_gate() -> None:
    origins = pd.date_range("2021-01-31", periods=36, freq="ME")
    actual = np.linspace(-0.10, 0.15, 36)
    rows: list[dict[str, object]] = []
    for model_id, prediction in {
        "HISTORICAL_MEAN": np.repeat(0.02, 36),
        "CANDIDATE": actual * 0.9,
    }.items():
        for origin, predicted, observed in zip(
            origins, prediction, actual, strict=True
        ):
            rows.append(
                {
                    "origin": origin,
                    "target_id": "EXPECTED_60D_TOTAL_RETURN",
                    "target_type": "CONTINUOUS_RETURN",
                    "model_id": model_id,
                    "prediction": predicted,
                    "actual": observed,
                }
            )
    predictions = pd.DataFrame(rows)
    target_specs = {
        "EXPECTED_60D_TOTAL_RETURN": {
            "target_type": "CONTINUOUS_RETURN",
            "horizon_market_days": 60,
        }
    }
    metrics = evaluate_predictions(
        predictions,
        target_specs=target_specs,
        model_ids=["HISTORICAL_MEAN", "CANDIDATE"],
        sample_scopes={
            "PRIMARY": {"start": "2021-01-01", "end": "2023-12-31"}
        },
        reliability_gate={
            "minimum_observed_predictions": 24,
            "minimum_distinct_calendar_years": 3,
            "binary_minimum_positive_events": 5,
            "binary_minimum_negative_events": 20,
        },
    )
    increments = build_increment_results(
        metrics,
        target_specs=target_specs,
        primary_scope="PRIMARY",
        module_comparisons={
            "TEST_INCREMENT": {
                "candidate": "CANDIDATE",
                "reference": "HISTORICAL_MEAN",
            }
        },
        final_model="CANDIDATE",
        final_baselines=["HISTORICAL_MEAN"],
        minimum_loss_improvement=0.02,
        continuous_auxiliary_gate={
            "minimum_spearman": 0.0,
            "minimum_sign_accuracy": 0.5,
        },
        binary_auxiliary_gate={"minimum_roc_auc": 0.5},
    )

    assert increments["result"].eq("PASS_INCREMENTAL_INFORMATION").all()
    validation = validation_summary(
        increments,
        module_comparison_ids=["TEST_INCREMENT"],
        final_model="CANDIDATE",
        final_baselines=["HISTORICAL_MEAN"],
        target_ids=["EXPECTED_60D_TOTAL_RETURN"],
    )
    assert validation["structural_prediction_validated"]
