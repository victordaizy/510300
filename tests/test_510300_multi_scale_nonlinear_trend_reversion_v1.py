"""510300多尺度非线性趋势延续与反转V1的冻结契约测试。"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from multi_scale_nonlinear_trend_reversion_v1 import (  # noqa: E402
    CONFIG_PATH,
    _circular_block_indices,
    _hac_regression,
    build_portfolio_targets,
    load_config,
    trend_weights,
)


def test_config_locks_scope_costs_and_clock() -> None:
    config = load_config()
    assert config["scope"]["execution_asset"] == "510300.SH"
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["scope"]["leverage_allowed"] is False
    assert config["scope"]["short_selling_allowed"] is False
    assert config["dates"]["evaluation_start"] == "2015-01-05"
    assert config["dates"]["evaluation_end"] == "2026-08-14"
    assert config["trend"]["signal_timing"] == "T日收盘后"
    assert config["trend"]["earliest_execution"] == "T加1交易日开盘"
    assert config["portfolio"]["initial_capital_cny"] == 20000.0
    assert config["portfolio"]["lot_size_shares"] == 100
    assert config["portfolio"]["base_costs"]["commission_rate_per_leg"] == 0.0003
    assert config["portfolio"]["base_costs"]["minimum_commission_cny_per_leg"] == 5.0
    assert config["portfolio"]["base_costs"]["slippage_bps_per_leg"] == 5.0
    assert config["portfolio"]["stress_costs"]["slippage_bps_per_leg"] == 10.0


def test_all_result_rescue_routes_are_forbidden() -> None:
    config = load_config()
    keys = [
        "parameter_rescue_after_result",
        "alternate_scale_rescue_after_result",
        "alternate_weight_rescue_after_result",
        "coefficient_refit_rescue_after_result",
        "direction_reversal_after_result",
        "combination_rescue_after_result",
    ]
    assert all(config["protocol"][key] == "forbidden" for key in keys)


def test_paper_horizons_and_coefficients_are_exactly_locked() -> None:
    config = load_config()
    assert config["trend"]["horizons_trading_days"] == [
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
    ]
    assert config["trend"]["fixed_linear_coefficient"] == 0.0129
    assert config["trend"]["fixed_cubic_coefficient"] == -0.0062
    assert config["trend"]["trend_clip_lower"] == -2.5
    assert config["trend"]["trend_clip_upper"] == 2.5


def test_every_weight_vector_is_l2_normalized_and_captures_99_percent() -> None:
    config = load_config()
    for horizon in config["trend"]["horizons_trading_days"]:
        weights, capture = trend_weights(horizon, config["trend"]["lag_multiplier"])
        assert math.isclose(float(np.square(weights).sum()), 1.0, abs_tol=1e-14)
        assert capture >= 0.99
        assert len(weights) == math.ceil(horizon * 2.25)
    maximum_weights, _ = trend_weights(1024, 2.25)
    assert len(maximum_weights) == 2304


def test_fixed_score_has_weak_trend_persistence_and_extreme_reversion() -> None:
    linear = 0.0129
    cubic = -0.0062

    def score(phi: float) -> float:
        return linear * phi + cubic * phi**3

    assert score(0.5) > 0
    assert score(-0.5) < 0
    assert score(2.0) < 0
    assert score(-2.0) > 0


def test_hac_regression_recovers_linear_and_cubic_coefficients() -> None:
    rng = np.random.default_rng(20260831)
    phi = np.linspace(-2.5, 2.5, 2000)
    noise = rng.normal(0.0, 0.02, size=len(phi))
    outcome = 0.01 + 0.20 * phi - 0.08 * phi**3 + noise
    design = np.column_stack([np.ones(len(phi)), phi, phi**3])
    beta, standard_error, t_stat = _hac_regression(design, outcome, 5)
    assert np.all(np.isfinite(beta))
    assert np.all(standard_error > 0)
    assert beta[1] > 0.19
    assert beta[2] < -0.079
    assert t_stat[1] > 1.645
    assert t_stat[2] < -1.645


def test_circular_block_indices_are_bounded_and_complete() -> None:
    rng = np.random.default_rng(7)
    indices = _circular_block_indices(101, 20, rng)
    assert len(indices) == 101
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 101


def test_position_mapping_is_binary_and_missing_holds_previous() -> None:
    config = load_config()
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="B"),
            "fixed_paper_score": [-0.1, 0.0, 0.1, np.nan],
        }
    )
    targets = build_portfolio_targets(panel, config)
    assert targets["target_position"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert set(targets["target_position"].unique()) <= {0.0, 1.0}


def test_prefreeze_input_audit_explicitly_has_no_candidate_outcome() -> None:
    config = load_config()
    audit_path = ROOT / config["inputs"]["candidate_input_audit"]["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS_PRE_FREEZE_DATA_AND_NORMALIZATION_CONTRACT"
    assert audit["candidate_2015_plus_outcomes_read_or_computed"] is False
    assert audit["candidate_fixed_score_computed"] is False
    assert audit["candidate_portfolio_returns_read_or_computed"] is False
    assert audit["candidate_sharpe_computed"] is False


def test_trial_count_includes_non_manifest_absorption_ratio_failure() -> None:
    config = load_config()
    assert config["selection_bias"]["non_manifest_prefreeze_failed_attempt_count"] == 1
    assert config["selection_bias"]["named_non_manifest_attempts"] == [
        "510300_CSI300_ABSORPTION_RATIO_TIMING_V1"
    ]
    assert config["selection_bias"]["expected_prior_manifest_count_at_freeze"] == 343
    assert config["selection_bias"]["expected_total_trial_count_including_current"] == 345
    assert CONFIG_PATH.exists()

