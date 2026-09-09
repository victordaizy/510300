"""板块未来贡献预测的目标时点、走步训练与安全测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research.sector_future_return_prediction import (
    PredictionRules,
    build_sector_forward_targets,
    evaluate_sector_forecasts,
    walk_forward_sector_forecasts,
)


FEATURES = (
    "sector_weight",
    "weighted_earnings_yield",
    "weighted_book_yield",
    "weighted_ttm_roe",
    "weighted_ttm_profit_growth_yoy",
    "weighted_ttm_revenue_growth_yoy",
)


def _rules(**overrides: object) -> PredictionRules:
    values: dict[str, object] = {
        "data_cutoff": pd.Timestamp("2026-12-31"),
        "horizon_days": 2,
        "minimum_endpoint_coverage": 0.95,
        "cash_annual_rate": 0.015,
        "trading_days_per_year": 242,
        "alpha": 10.0,
        "minimum_training_dates": 2,
        "numeric_features": FEATURES,
        "interaction_features": FEATURES[1:],
        "minimum_calibration_errors": 1,
        "calibration_quantiles": (0.10, 0.50, 0.90),
        "pseudo_oos_start": pd.Timestamp("2025-03-31"),
        "minimum_oos_predictions": 4,
        "minimum_spearman": 0.20,
        "bootstrap_repetitions": 100,
        "bootstrap_block_length": 2,
        "random_seed": 20260818,
        "bootstrap_lower_bound": 0.0,
        "minimum_mae_improvement_aggregate": 0.10,
        "minimum_mae_improvement_mean": 0.10,
        "minimum_direction_accuracy": 0.55,
        "split_minimum_spearman": 0.0,
        "etf_minimum_spearman": 0.0,
        "minimum_calibrated_predictions": 1,
        "minimum_brier_skill": 0.0,
        "interval_coverage_minimum": 0.65,
        "interval_coverage_maximum": 0.95,
    }
    values.update(overrides)
    return PredictionRules(**values)


def _target_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal = pd.Timestamp("2025-01-31")
    panel = pd.DataFrame(
        {
            "date": [signal, signal],
            "industry_l1": ["行业A", "行业B"],
            "sector_weight": [0.60, 0.40],
            "eligible_for_target_freeze": [True, True],
        }
    )
    weights = pd.DataFrame(
        {
            "trade_date": [signal, signal],
            "con_code": ["A", "B"],
            "weight": [60.0, 40.0],
        }
    )
    dates = pd.to_datetime(["2025-01-31", "2025-02-03", "2025-02-04"])
    daily = pd.DataFrame(
        [
            {"date": date, "con_code": code, "total_return_open": open_, "total_return_close": close}
            for date, code, open_, close in [
                (dates[0], "A", 1.0, 9.0),
                (dates[0], "B", 1.0, 9.0),
                (dates[1], "A", 10.0, 10.5),
                (dates[1], "B", 20.0, 19.0),
                (dates[2], "A", 11.0, 12.0),
                (dates[2], "B", 19.0, 18.0),
            ]
        ]
    )
    industries = pd.DataFrame(
        [
            {"con_code": "A", "industry_l1": "行业A", "in_date": "2020-01-01", "out_date": None},
            {"con_code": "B", "industry_l1": "行业B", "in_date": "2020-01-01", "out_date": "2025-02-04"},
            {"con_code": "B", "industry_l1": "未来行业", "in_date": "2025-02-04", "out_date": None},
        ]
    )
    industries["classification_usage"] = "POINT_IN_TIME_INTERVAL_CITIC"
    industries["source"] = "测试点时行业"
    return panel, weights, daily, industries


def test_target_uses_next_open_fixed_weight_and_signal_date_industry() -> None:
    result = build_sector_forward_targets(*_target_inputs(), _rules())
    by_industry = result.set_index("industry_l1")
    assert np.isclose(by_industry.at["行业A", "sector_contribution_60d"], 0.60 * (12.0 / 10.0 - 1.0))
    assert np.isclose(by_industry.at["行业B", "sector_contribution_60d"], 0.40 * (18.0 / 20.0 - 1.0))
    assert "未来行业" not in by_industry.index
    assert result["entry_date"].eq(pd.Timestamp("2025-02-03")).all()
    assert result["maturity_date"].eq(pd.Timestamp("2025-02-04")).all()
    assert result["target_output"].eq("TARGET_READY").all()


def test_missing_endpoint_weight_is_not_reweighted() -> None:
    panel, weights, daily, industries = _target_inputs()
    daily.loc[
        daily["date"].eq(pd.Timestamp("2025-02-04")) & daily["con_code"].eq("B"),
        "total_return_close",
    ] = np.nan
    result = build_sector_forward_targets(panel, weights, daily, industries, _rules())
    assert result["target_output"].eq("NO_VIEW").all()
    assert result["failure_category"].eq("ENDPOINT_WEIGHT_COVERAGE_BELOW_GATE").all()
    assert np.isclose(result["endpoint_weight_coverage"].iloc[0], 0.60)
    assert np.isclose(result["snapshot_index_return_60d"].iloc[0], 0.60 * (12.0 / 10.0 - 1.0))


def _walk_forward_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(
        ["2025-01-02", "2025-02-03", "2025-03-31", "2025-05-30", "2025-07-31"]
    )
    rows: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        maturity = date + pd.Timedelta(days=10)
        actual = 0.01 * (date_index + 1)
        for industry_index, industry in enumerate(("行业A", "行业B")):
            weight = 0.60 if industry_index == 0 else 0.40
            rows.append(
                {
                    "date": date,
                    "industry_l1": industry,
                    "sector_weight": weight,
                    "eligible_for_target_freeze": True,
                    "weighted_earnings_yield": 0.04 + 0.005 * date_index + 0.002 * industry_index,
                    "weighted_book_yield": 0.30 + 0.01 * date_index + 0.01 * industry_index,
                    "weighted_ttm_roe": 0.10 + 0.01 * date_index + 0.005 * industry_index,
                    "weighted_ttm_profit_growth_yoy": -0.05 + 0.03 * date_index + 0.01 * industry_index,
                    "weighted_ttm_revenue_growth_yoy": 0.02 + 0.02 * date_index + 0.01 * industry_index,
                }
            )
            targets.append(
                {
                    "date": date,
                    "entry_date": date + pd.Timedelta(days=1),
                    "maturity_date": maturity,
                    "industry_l1": industry,
                    "sector_weight": weight,
                    "sector_contribution_60d": actual * weight,
                    "snapshot_index_return_60d": actual,
                    "endpoint_weight_coverage": 1.0,
                    "target_output": "TARGET_READY",
                    "failure_category": "PASS",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(targets)


def test_walk_forward_prediction_does_not_use_current_or_future_label() -> None:
    panel, targets = _walk_forward_inputs()
    rules = _rules()
    first = walk_forward_sector_forecasts(panel, targets, pd.DataFrame(), rules)
    changed = targets.copy()
    changed.loc[changed["date"].ge(pd.Timestamp("2025-03-31")), "sector_contribution_60d"] += 10.0
    changed.loc[changed["date"].ge(pd.Timestamp("2025-03-31")), "snapshot_index_return_60d"] += 20.0
    second = walk_forward_sector_forecasts(panel, changed, pd.DataFrame(), rules)
    assert first["date"].tolist() == second["date"].tolist()
    assert np.isclose(
        first.iloc[0]["predicted_index_return_60d"],
        second.iloc[0]["predicted_index_return_60d"],
    )
    assert first["training_snapshot_dates"].tolist() == [2, 3, 4]
    assert not np.allclose(
        first["actual_index_return_60d"], second["actual_index_return_60d"]
    )


def test_training_window_all_missing_feature_is_rejected() -> None:
    panel, targets = _walk_forward_inputs()
    panel["weighted_ttm_roe"] = np.nan
    with pytest.raises(ValueError, match="训练窗口特征全缺失"):
        walk_forward_sector_forecasts(panel, targets, pd.DataFrame(), _rules())


def test_evaluation_is_deterministic_and_never_authorizes_trading() -> None:
    dates = pd.date_range("2025-01-31", periods=8, freq="ME")
    actual = np.linspace(-0.04, 0.10, len(dates))
    forecasts = pd.DataFrame(
        {
            "date": dates,
            "maturity_date": dates + pd.Timedelta(days=70),
            "predicted_index_return_60d": actual * 0.90,
            "aggregate_baseline_prediction_60d": np.zeros(len(dates)),
            "expanding_mean_prediction_60d": np.full(len(dates), 0.02),
            "actual_index_return_60d": actual,
            "etf_x60": actual * 0.95,
            "predicted_positive_probability": np.where(actual > 0, 0.80, 0.20),
            "baseline_positive_probability": np.full(len(dates), 0.50),
            "predicted_return_q10": actual - 0.03,
            "predicted_return_q90": actual + 0.03,
        }
    )
    rules = replace(_rules(), data_cutoff=pd.Timestamp("2026-12-31"))
    first = evaluate_sector_forecasts(forecasts, rules)
    second = evaluate_sector_forecasts(forecasts, rules)
    assert first["bootstrap_spearman"] == second["bootstrap_spearman"]
    assert first["candidate_id"] == "S1_SECTOR_FUNDAMENTAL_RIDGE_V1"
    assert first["safety"] == {
        "forecast_eligible_emitted": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }


def test_degenerate_bootstrap_returns_rejection_instead_of_crashing() -> None:
    dates = pd.date_range("2025-01-31", periods=8, freq="ME")
    actual = np.linspace(-0.04, 0.10, len(dates))
    forecasts = pd.DataFrame(
        {
            "date": dates,
            "maturity_date": dates + pd.Timedelta(days=70),
            "predicted_index_return_60d": np.zeros(len(dates)),
            "aggregate_baseline_prediction_60d": np.full(len(dates), 0.01),
            "expanding_mean_prediction_60d": np.full(len(dates), 0.02),
            "actual_index_return_60d": actual,
            "etf_x60": actual,
            "predicted_positive_probability": np.nan,
            "baseline_positive_probability": 0.50,
            "predicted_return_q10": np.nan,
            "predicted_return_q90": np.nan,
        }
    )
    result = evaluate_sector_forecasts(forecasts, _rules())
    assert result["status"] == "HISTORICAL_REJECTED_FROZEN"
    assert result["bootstrap_spearman"]["interval_95pct"] is None
    assert result["gates"]["bootstrap_spearman_lower_bound"] is False
