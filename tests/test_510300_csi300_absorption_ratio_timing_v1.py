"""510300沪深300PCA吸收率择时V1的冻结单元测试。"""

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

from csi300_absorption_ratio_timing_v1 import (  # noqa: E402
    ContractError,
    _newey_west_regression,
    build_hysteresis_states,
    build_mechanism_panel,
    build_portfolio_signals,
    deflated_sharpe_probability,
    evaluate_mechanism_gate,
    load_config,
    validate_config,
)
from build_510300_csi300_absorption_ratio_inputs_v1 import (  # noqa: E402
    build_daily_membership,
    exponential_covariance_absorption_ratio,
)


def test_config_locks_scope_factor_costs_dates_and_trial_count() -> None:
    config = load_config()
    assert config["protocol"]["project_id"] == (
        "510300_CSI300_ABSORPTION_RATIO_TIMING_V1"
    )
    assert config["protocol"]["one_shot"] is True
    assert config["protocol"]["candidate_factor_values_read_before_freeze"] is True
    assert config["protocol"]["candidate_market_return_outcomes_read_before_freeze"] is False
    assert config["protocol"]["candidate_portfolio_returns_read_before_freeze"] is False
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["dates"]["evaluation_start"] == "2015-01-05"
    assert config["dates"]["evaluation_end"] == "2026-08-14"
    assert config["factor"]["covariance_window_trading_days"] == 500
    assert config["factor"]["exponential_half_life_trading_days"] == 250
    assert config["factor"]["short_average_trading_days"] == 15
    assert config["factor"]["long_average_trading_days"] == 250
    assert config["factor"]["upper_risk_off_threshold"] == 1.0
    assert config["factor"]["lower_risk_on_threshold"] == -1.0
    assert config["factor"]["initial_target_position"] == 0.5
    assert config["portfolio"]["initial_capital_cny"] == 20000.0
    assert config["portfolio"]["lot_size_shares"] == 100
    assert config["portfolio"]["base_costs"]["minimum_commission_cny_per_leg"] == 5.0
    assert config["portfolio"]["base_costs"]["slippage_bps_per_leg"] == 5.0
    assert config["portfolio"]["stress_costs"]["slippage_bps_per_leg"] == 10.0
    assert config["selection_bias"]["expected_prior_manifest_count"] == 333
    assert config["selection_bias"]["expected_total_trial_count_including_current"] == 334
    assert config["data_contract"]["expected_relevant_symbol_count_after_corrections"] == 714
    correction = config["data_contract"]["pre_factor_membership_interval_corrections"][0]
    assert correction["symbol"] == "600357.SH"
    assert correction["corrected_opt_out"] == "2009-12-29"


def test_config_rejects_threshold_window_and_continuous_mapping_rescue() -> None:
    config = load_config()
    modified = deepcopy(config)
    modified["factor"]["upper_risk_off_threshold"] = 0.8
    with pytest.raises(ContractError, match="风险关闭阈值"):
        validate_config(modified)
    modified = deepcopy(config)
    modified["factor"]["covariance_window_trading_days"] = 400
    with pytest.raises(ContractError, match="协方差窗口"):
        validate_config(modified)
    modified = deepcopy(config)
    modified["allocation"]["alternate_continuous_mapping"] = "allowed"
    with pytest.raises(ContractError, match="连续仓位"):
        validate_config(modified)


def test_exponential_covariance_absorption_ratio_matches_manual_eigenvalues() -> None:
    rng = np.random.default_rng(19)
    common = rng.normal(size=(500, 1))
    returns = common @ np.linspace(0.5, 1.5, 10).reshape(1, -1)
    returns += rng.normal(scale=0.2, size=(500, 10))
    observed = exponential_covariance_absorption_ratio(
        returns, half_life=250, eigenvector_count=2
    )
    ages = np.arange(499, -1, -1, dtype=float)
    weights = np.power(0.5, ages / 250.0)
    weights /= weights.sum()
    centered = returns - weights @ returns
    covariance = (centered * np.sqrt(weights)[:, None]).T @ (
        centered * np.sqrt(weights)[:, None]
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    expected = float(eigenvalues[-2:].sum() / np.trace(covariance))
    assert observed == pytest.approx(expected, rel=1e-11, abs=1e-12)
    assert 0.0 < observed < 1.0


def test_hysteresis_uses_exact_thresholds_and_carries_previous_state() -> None:
    config = load_config()
    factor = pd.DataFrame(
        {
            "date": pd.date_range("2014-12-01", periods=8, freq="B"),
            "standardized_absorption_ratio_shift": [
                0.0,
                -1.0,
                -0.2,
                1.0,
                0.1,
                -1.2,
                np.nan,
                0.5,
            ],
        }
    )
    states = build_hysteresis_states(factor, config)
    assert states["target_position"].tolist() == [0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert states["state_changed"].tolist() == [False, True, False, True, False, True, False, False]
    assert states.loc[6, "signal_reason"] == "NO_VIEW_CARRY_PREVIOUS_TARGET"


def test_mechanism_clock_is_prior_close_to_next_open_then_following_open() -> None:
    config = deepcopy(load_config())
    config["dates"]["evaluation_end"] = "2015-01-07"
    config["dates"]["structural_periods"] = {
        "EARLY": {"start": "2015-01-05", "end": "2015-01-07"}
    }
    states = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2015-01-02", "2015-01-05"]),
            "standardized_absorption_ratio_shift": [-1.1, 1.2],
            "target_position": [1.0, 0.0],
            "signal_reason": ["RISK_ON_TRIGGER", "RISK_OFF_TRIGGER"],
        }
    )
    index_daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-01-02", "2015-01-05", "2015-01-06", "2015-01-07"]),
            "open": [99.0, 100.0, 102.0, 101.0],
        }
    )
    panel = build_mechanism_panel(states, index_daily, config)
    assert panel.loc[0, "execution_date"] == pd.Timestamp("2015-01-05")
    assert panel.loc[0, "target_end_date"] == pd.Timestamp("2015-01-06")
    assert panel.loc[0, "target_open_to_open_log_return"] == pytest.approx(
        np.log(102.0 / 100.0)
    )
    assert panel.loc[1, "target_open_to_open_log_return"] == pytest.approx(
        np.log(101.0 / 102.0)
    )


def test_newey_west_regression_recovers_positive_exposure_coefficient() -> None:
    rng = np.random.default_rng(23)
    exposure = np.tile(np.repeat([0.0, 1.0], 50), 30)
    target = 0.0001 + 0.0015 * exposure + rng.normal(scale=0.0005, size=len(exposure))
    result = _newey_west_regression(
        target,
        np.column_stack([np.ones(len(exposure)), exposure]),
        max_lag=20,
        coefficient_names=["intercept", "target_position"],
    )
    estimate = result["coefficients"]["target_position"]
    assert estimate["coefficient"] == pytest.approx(0.0015, abs=0.00008)
    assert estimate["t_stat"] is not None and estimate["t_stat"] > 5.0


def test_mechanism_gate_passes_stable_synthetic_risk_discrimination() -> None:
    config = deepcopy(load_config())
    config["mechanism_model"]["bootstrap_repetitions"] = 100
    config["mechanism_model"]["bootstrap_block_length_trading_days"] = 30
    dates = pd.bdate_range("2015-01-05", "2026-08-14")
    exposure = ((np.arange(len(dates)) // 40) % 2).astype(float)
    noise = 0.00015 * np.sin(np.arange(len(dates)) / 7.0)
    panel = pd.DataFrame(
        {
            "decision_date": dates - pd.offsets.BDay(1),
            "execution_date": dates,
            "target_end_date": dates + pd.offsets.BDay(1),
            "standardized_absorption_ratio_shift": np.where(exposure > 0, -1.1, 1.1),
            "target_position": exposure,
            "signal_reason": np.where(exposure > 0, "RISK_ON_TRIGGER", "RISK_OFF_TRIGGER"),
            "target_open_to_open_log_return": -0.0002 + 0.0012 * exposure + noise,
        }
    )
    panel = panel.loc[panel["target_end_date"] <= pd.Timestamp("2026-08-14")].copy()
    panel["structural_period"] = np.where(
        panel["execution_date"] <= pd.Timestamp("2019-12-31"), "EARLY", "LATE"
    )
    result = evaluate_mechanism_gate(panel, config)
    assert len(panel) >= 2800
    assert result["passed"] is True
    assert all(result["gates"].values())
    assert result["bootstrap"]["confidence_interval"][0] > 0.0


def test_daily_membership_switches_only_at_frozen_segment_boundaries() -> None:
    config = deepcopy(load_config())
    config["data_contract"]["membership_segments"]["HISTORICAL_INTERVALS"][
        "minimum_active_members"
    ] = 2
    config["data_contract"]["membership_segments"]["HISTORICAL_INTERVALS"][
        "maximum_active_members"
    ] = 2
    config["data_contract"]["membership_segments"]["EXTERNAL_POINT_IN_TIME_PANEL"][
        "exact_active_members"
    ] = 2
    config["data_contract"]["membership_segments"]["CURRENT_POINT_IN_TIME_PANEL"][
        "exact_active_members"
    ] = 2
    calendar = pd.DatetimeIndex(
        pd.to_datetime(["2014-12-31", "2015-01-05", "2021-08-12"])
    )
    membership = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "opt_in": pd.to_datetime(["2010-01-01", "2010-01-01"]),
            "opt_out": [pd.NaT, pd.NaT],
        }
    )
    external = {pd.Timestamp("2015-01-05"): ["C", "D"]}
    current = {pd.Timestamp("2021-08-12"): ["E", "F"]}
    members, sources = build_daily_membership(
        calendar, membership, external, current, config
    )
    assert members[pd.Timestamp("2014-12-31")] == ["A", "B"]
    assert members[pd.Timestamp("2015-01-05")] == ["C", "D"]
    assert members[pd.Timestamp("2021-08-12")] == ["E", "F"]
    assert sources[pd.Timestamp("2014-12-31")] == "HISTORICAL_INTERVALS"
    assert sources[pd.Timestamp("2015-01-05")] == "EXTERNAL_POINT_IN_TIME_PANEL"
    assert sources[pd.Timestamp("2021-08-12")] == "CURRENT_POINT_IN_TIME_PANEL"


def test_portfolio_signal_executes_first_frozen_state_at_next_open() -> None:
    config = deepcopy(load_config())
    config["dates"]["evaluation_end"] = "2015-01-06"
    states = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2015-01-02", "2015-01-05"]),
            "standardized_absorption_ratio_shift": [-1.1, 0.0],
            "target_position": [1.0, 1.0],
            "signal_reason": ["RISK_ON_TRIGGER", "BETWEEN_THRESHOLDS_CARRY_PREVIOUS_TARGET"],
            "risk_off_override": [False, False],
        }
    )
    market = pd.DataFrame(
        {"date": pd.to_datetime(["2015-01-02", "2015-01-05", "2015-01-06"])}
    )
    signals = build_portfolio_signals(states, market, config)
    assert signals.loc[0, "decision_date"] == pd.Timestamp("2015-01-02")
    assert signals.loc[0, "execution_date"] == pd.Timestamp("2015-01-05")
    assert signals.loc[0, "target_position"] == 1.0


def test_deflated_sharpe_uses_all_334_trials() -> None:
    returns = pd.Series(np.tile([0.001, -0.0002, 0.0008, 0.0001], 750))
    result = deflated_sharpe_probability(returns, 334)
    assert result["total_trial_count"] == 334
    assert result["expected_maximum_null_sharpe_annualized"] is not None
    assert result["expected_maximum_null_sharpe_annualized"] > 0.0
