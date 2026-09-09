"""510300横截面收益离散度波动预算V1的冻结单元测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from csi300_return_dispersion_volatility_v1 import (  # noqa: E402
    ContractError,
    _dm_t_stat,
    _newey_west_regression,
    _validate_factor_active_member_counts,
    build_portfolio_signals,
    expanding_forecasts,
    load_config,
    monthly_realized_variance,
    validate_config,
)
from build_510300_csi300_return_dispersion_inputs_v1 import (  # noqa: E402
    compute_monthly_dispersion_segment,
)


def test_config_locks_single_factor_scope_costs_and_dates() -> None:
    config = load_config()
    assert config["protocol"]["project_id"] == (
        "510300_CSI300_RETURN_DISPERSION_VOLATILITY_V1R1"
    )
    assert config["protocol"]["revision_kind"] == (
        "PRE_RESULT_INPUT_ASSERTION_CORRECTION_ONLY"
    )
    assert config["protocol"]["candidate_factor_values_read_before_freeze"] is True
    assert config["protocol"]["candidate_future_volatility_outcomes_read_before_freeze"] is False
    assert config["protocol"]["candidate_portfolio_returns_read_before_freeze"] is False
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["dates"]["evaluation_start"] == "2015-01-05"
    assert config["dates"]["evaluation_end"] == "2026-08-14"
    assert config["factor"]["active_factor_count"] == 1
    assert config["factor"]["threshold_selection"] == "none"
    assert config["factor"]["winsorization"] == "none"
    assert config["allocation"]["baseline_only_allocation"] == "forbidden"
    assert config["portfolio"]["initial_capital_cny"] == 20000.0
    assert config["portfolio"]["lot_size_shares"] == 100
    assert config["portfolio"]["base_costs"]["minimum_commission_cny_per_leg"] == 5.0
    assert config["portfolio"]["base_costs"]["slippage_bps_per_leg"] == 5.0
    assert config["portfolio"]["stress_costs"]["slippage_bps_per_leg"] == 10.0


def test_config_rejects_factor_or_baseline_rescue() -> None:
    config = load_config()
    modified = deepcopy(config)
    modified["factor"]["active_factor_count"] = 2
    with pytest.raises(ContractError, match="活跃因子数量"):
        validate_config(modified)
    modified = deepcopy(config)
    modified["allocation"]["baseline_only_allocation"] = "allowed"
    with pytest.raises(ContractError, match="基线单独"):
        validate_config(modified)


def test_member_count_contract_accepts_early_301_but_rejects_late_301() -> None:
    config = load_config()
    valid = pd.DataFrame(
        {
            "factor_source_segment": [
                "EARLY_SINA_RECONSTRUCTION",
                "EARLY_SINA_RECONSTRUCTION",
                "EXISTING_SINA_EXTERNAL_PANEL",
                "EXISTING_SINA_BACKUP_PANEL",
            ],
            "active_member_count": [300, 301, 300, 300],
        }
    )
    _validate_factor_active_member_counts(valid, config)
    invalid = valid.copy()
    invalid.loc[2, "active_member_count"] = 301
    with pytest.raises(ContractError, match="2015年后"):
        _validate_factor_active_member_counts(invalid, config)


def test_monthly_dispersion_is_equal_weight_sample_standard_deviation() -> None:
    membership = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "opt_in": pd.to_datetime(["2020-01-01"] * 3),
            "opt_out": [pd.NaT, pd.NaT, pd.NaT],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-28",
                    "2020-01-31",
                    "2020-02-28",
                    "2020-01-31",
                    "2020-02-28",
                ]
            ),
            "con_code": ["A", "A", "B", "B", "C", "C"],
            "total_return_close": [100.0, 110.0, 100.0, 100.0, 100.0, 90.0],
            "is_index_member": [True] * 6,
        }
    )
    factor = compute_monthly_dispersion_segment(
        prices,
        membership,
        {"2020-01": pd.Timestamp("2020-01-31"), "2020-02": pd.Timestamp("2020-02-28")},
        first_month="2020-02",
        last_month="2020-02",
        source_segment="TEST",
        member_source="HISTORICAL_INTERVALS",
        minimum_active_members=3,
        maximum_active_members=3,
        minimum_valid_count=3,
        minimum_coverage=1.0,
    )
    assert len(factor) == 1
    assert factor.loc[0, "equal_weight_mean_monthly_return"] == pytest.approx(0.0)
    assert factor.loc[0, "cross_sectional_return_dispersion"] == pytest.approx(0.1)
    assert factor.loc[0, "cross_sectional_return_dispersion_squared"] == pytest.approx(
        0.01
    )


def test_panel_membership_source_uses_month_end_flags_instead_of_intervals() -> None:
    stale_intervals = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "opt_in": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "opt_out": [pd.NaT, pd.NaT],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-28",
                    "2020-01-31",
                    "2020-02-28",
                    "2020-01-31",
                    "2020-02-28",
                ]
            ),
            "con_code": ["A", "A", "B", "B", "C", "C"],
            "total_return_close": [100.0, 110.0, 100.0, 100.0, 100.0, 90.0],
            "is_index_member": [True, True, True, False, False, True],
        }
    )
    factor = compute_monthly_dispersion_segment(
        prices,
        stale_intervals,
        {"2020-01": pd.Timestamp("2020-01-31"), "2020-02": pd.Timestamp("2020-02-28")},
        first_month="2020-02",
        last_month="2020-02",
        source_segment="TEST_PANEL",
        member_source="PANEL_IS_INDEX_MEMBER",
        minimum_active_members=2,
        maximum_active_members=2,
        minimum_valid_count=2,
        minimum_coverage=1.0,
    )
    assert factor.loc[0, "active_member_count"] == 2
    assert factor.loc[0, "equal_weight_mean_monthly_return"] == pytest.approx(0.0)
    assert factor.loc[0, "cross_sectional_return_dispersion"] == pytest.approx(
        np.sqrt(0.02)
    )


def test_monthly_realized_variance_assigns_close_to_close_return_to_current_month() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"]
            ),
            "close": [100.0, 101.0, 99.0, 102.0],
        }
    )
    result = monthly_realized_variance(daily).set_index("month")
    assert result.loc["2020-01", "realized_variance"] == pytest.approx(
        np.log(101.0 / 100.0) ** 2
    )
    assert result.loc["2020-02", "realized_variance"] == pytest.approx(
        np.log(99.0 / 101.0) ** 2 + np.log(102.0 / 99.0) ** 2
    )


def test_newey_west_regression_recovers_positive_dispersion_coefficient() -> None:
    rng = np.random.default_rng(7)
    current = rng.normal(size=200)
    dispersion = rng.normal(size=200)
    target = 0.2 + 0.4 * current + 0.8 * dispersion + rng.normal(scale=0.05, size=200)
    design = np.column_stack([np.ones(200), current, dispersion])
    result = _newey_west_regression(
        target,
        design,
        max_lag=3,
        coefficient_names=["intercept", "current", "dispersion"],
    )
    assert result["coefficients"]["dispersion"]["coefficient"] == pytest.approx(
        0.8, abs=0.02
    )
    assert result["coefficients"]["dispersion"]["t_stat"] > 10.0


def _synthetic_mechanism_panel() -> pd.DataFrame:
    periods = pd.period_range("2005-05", "2015-02", freq="M")
    index = np.arange(len(periods), dtype=float)
    log_current = -5.0 + 0.1 * np.sin(index / 3.0)
    log_dispersion = -4.0 + 0.2 * np.cos(index / 5.0)
    log_next = -1.0 + 0.5 * log_current + 0.7 * log_dispersion
    return pd.DataFrame(
        {
            "factor_month": periods.astype(str),
            "decision_date": periods.to_timestamp("M"),
            "target_month": (periods + 1).astype(str),
            "target_month_end_date": (periods + 1).to_timestamp("M"),
            "current_realized_variance": np.exp(log_current),
            "current_trading_return_count": 20,
            "cross_sectional_return_dispersion": np.exp(log_dispersion / 2.0),
            "cross_sectional_return_dispersion_squared": np.exp(log_dispersion),
            "next_month_realized_variance": np.exp(log_next),
            "next_month_trading_return_count": 20,
            "log_current_realized_variance": log_current,
            "log_dispersion_squared": log_dispersion,
            "log_next_month_realized_variance": log_next,
        }
    )


def test_expanding_forecast_does_not_fit_on_unavailable_next_month_outcome() -> None:
    config = deepcopy(load_config())
    config["dates"]["last_complete_factor_month"] = "2015-01"
    config["mechanism_model"]["minimum_training_pairs"] = 12
    panel = _synthetic_mechanism_panel()
    original = expanding_forecasts(panel, config)
    altered = panel.copy()
    future_row = altered["factor_month"].eq("2014-12")
    altered.loc[future_row, "log_next_month_realized_variance"] += 20.0
    altered.loc[future_row, "next_month_realized_variance"] = np.exp(
        altered.loc[future_row, "log_next_month_realized_variance"]
    )
    changed = expanding_forecasts(altered, config)
    columns = [
        "baseline_variance_forecast",
        "augmented_variance_forecast",
        "augmented_beta_dispersion",
    ]
    assert original.loc[0, columns].tolist() == pytest.approx(
        changed.loc[0, columns].tolist()
    )


def test_dm_stat_is_positive_when_augmented_loss_is_consistently_lower() -> None:
    differences = np.array([0.2, 0.1, 0.25, 0.15, 0.3] * 20, dtype=float)
    statistic = _dm_t_stat(differences, max_lag=3)
    assert statistic is not None
    assert statistic > 1.645


def test_portfolio_signal_uses_next_open_and_clips_weight() -> None:
    config = deepcopy(load_config())
    forecasts = pd.DataFrame(
        {
            "factor_month": ["2014-12", "2015-01"],
            "target_month": ["2015-01", "2015-02"],
            "training_pairs": [100, 101],
            "augmented_variance_forecast": [0.001, 0.01],
        }
    )
    factor = pd.DataFrame(
        {
            "factor_month": ["2014-12", "2015-01"],
            "month_end_date": pd.to_datetime(["2014-12-31", "2015-01-30"]),
        }
    )
    months = pd.period_range("2012-01", "2015-01", freq="M")
    benchmark = pd.DataFrame(
        {
            "date": months.to_timestamp("M"),
            "close": 100.0 * np.power(1.01, np.arange(len(months))),
        }
    )
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2014-12-31", "2015-01-05", "2015-01-30", "2015-02-02"]
            )
        }
    )
    signals = build_portfolio_signals(
        forecasts, factor, benchmark, market, config
    )
    assert signals.loc[0, "execution_date"] == pd.Timestamp("2015-01-05")
    assert signals.loc[1, "execution_date"] == pd.Timestamp("2015-02-02")
    assert signals.loc[0, "target_position"] == pytest.approx(1.0)
    assert 0.0 < signals.loc[1, "target_position"] < 1.0
