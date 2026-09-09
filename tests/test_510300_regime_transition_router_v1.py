from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

import regime_transition_router_v1 as study


def _synthetic_market(dates: pd.DatetimeIndex, close: list[float]) -> pd.DataFrame:
    values = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": values,
            "high": values + 0.1,
            "low": values - 0.1,
            "close": values,
            "volume": 1_000_000,
            "amount": 10_000_000,
            "symbol": "510300.SH",
        }
    )


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "record_date",
            "ex_date",
            "payment_date",
            "cash_dividend_per_share",
            "source",
        ]
    )


def test_config_freezes_phase_one_and_execution_boundaries() -> None:
    config = study.load_config()
    assert config["protocol"]["project_id"] == study.PROJECT_ID
    assert config["protocol"]["one_shot"] is True
    assert config["protocol"]["outcome_read_before_freeze"] is False
    assert config["protocol"]["parameter_rescue_after_result"] == "forbidden"
    assert config["scope"]["return_backtest_allowed_in_phase_1"] is False
    assert config["scope"]["position_mapping"] == "disabled"
    assert config["scope"]["paper_signal"] == "disabled"
    assert config["scope"]["shadow_signal"] == "disabled"
    assert config["scope"]["order_generation"] is False
    assert config["scope"]["broker_connection"] is False
    assert config["scope"]["live_trading_authorized"] is False
    assert config["clocks"]["earliest_hypothetical_execution"] == "T+1交易日开盘"
    assert config["phase_2"]["run_only_if_state_gate_passes"] is True


def test_causal_percentile_excludes_current_and_future_values() -> None:
    original = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    changed = original.copy()
    changed.iloc[4:] = [50_000.0, -50_000.0]
    first = study.causal_percentile(
        original, prior_rows=3, minimum_prior_rows=3
    )
    second = study.causal_percentile(
        changed, prior_rows=3, minimum_prior_rows=3
    )
    pd.testing.assert_series_equal(first.iloc[:4], second.iloc[:4])
    assert first.iloc[:3].isna().all()
    assert first.iloc[3] == 1.0


def test_total_return_index_neutralizes_ex_date_cash_drop() -> None:
    dates = pd.bdate_range("2020-01-01", periods=3)
    market = _synthetic_market(dates, [10.0, 9.0, 9.5])
    dividends = pd.DataFrame(
        {
            "symbol": ["510300.SH"],
            "record_date": [dates[0]],
            "ex_date": [dates[1]],
            "payment_date": [dates[2]],
            "cash_dividend_per_share": [1.0],
            "source": ["合成测试"],
        }
    )
    result = study.build_total_return_market(market, dividends)
    assert result.loc[1, "cash_dividend_ex_date"] == 1.0
    assert np.isclose(result.loc[1, "total_return_1d"], 0.0)
    assert np.isclose(result.loc[1, "total_return_index"], 1.0)


def test_outcome_panel_uses_t_plus_one_open_and_record_date_entitlement() -> None:
    config = deepcopy(study.load_config())
    config["outcomes"]["horizons_trading_days"] = [2]
    config["dates"]["outcome_market_cutoff"] = "2020-01-10"
    dates = pd.bdate_range("2020-01-01", periods=5)
    market = _synthetic_market(dates, [10.0] * len(dates))
    states = pd.DataFrame(
        {
            "date": [dates[0], dates[2], dates[-1]],
            "state": [study.NORMAL, study.STRESS, study.UNCERTAIN],
        }
    )
    dividends = pd.DataFrame(
        {
            "symbol": ["510300.SH"],
            "record_date": [dates[2]],
            "ex_date": [dates[3]],
            "payment_date": [dates[4]],
            "cash_dividend_per_share": [1.0],
            "source": ["合成测试"],
        }
    )
    result = study.build_outcome_panel(config, states, market, dividends)
    assert result.loc[0, "entry_date"] == dates[1]
    assert result.loc[0, "future_exit_date_2d"] == dates[2]
    assert np.isclose(result.loc[0, "future_return_2d"], 0.1)
    assert np.isclose(result.loc[0, "future_positive_wealth_fraction_2d"], 0.5)
    assert result.loc[1, "entry_date"] == dates[3]
    assert np.isclose(result.loc[1, "future_return_2d"], 0.0)
    assert bool(result.loc[2, "future_censored_2d"]) is True
    assert pd.isna(result.loc[2, "future_return_2d"])


def test_state_machine_confirms_stress_then_asymmetric_recovery_and_hold() -> None:
    config = deepcopy(study.load_config())
    dates = pd.bdate_range("2020-01-01", periods=45)
    panel = pd.DataFrame(
        {
            "date": dates,
            "liquidity_stress_score": 0.20,
            "downside_vol_ratio_5_20": 0.50,
            "discount_rate_pressure": 0.20,
            "equal_above_ma60_share": 0.60,
            "breadth_change5": 0.01,
            "total_return_index": 2.0,
            "price_ma20_total_return": 1.0,
            "price_log_slope20": 0.01,
        }
    )
    panel.loc[23:24, [
        "liquidity_stress_score",
        "downside_vol_ratio_5_20",
        "discount_rate_pressure",
    ]] = [0.80, 1.20, 0.80]
    panel.loc[25:26, "liquidity_stress_score"] = 0.60
    panel.loc[25:26, "downside_vol_ratio_5_20"] = 0.90
    panel.loc[27:, "downside_vol_ratio_5_20"] = 0.40

    result = study.apply_state_machine(panel, config)
    assert str(result.loc[22, "state"]) == study.NORMAL
    assert str(result.loc[23, "state"]) == study.UNCERTAIN
    assert str(result.loc[24, "state"]) == study.STRESS
    assert str(result.loc[27, "state"]) == study.STRESS
    assert str(result.loc[28, "state"]) == study.RECOVERY
    assert str(result.loc[37, "state"]) == study.RECOVERY
    assert str(result.loc[38, "state"]) == study.NORMAL
    assert not any(
        token in column.lower()
        for column in result.columns
        for token in ["future", "forward", "target_return"]
    )


def test_episode_builder_preserves_left_and_right_censoring() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=6),
            "state": [
                study.NORMAL,
                study.NORMAL,
                study.STRESS,
                study.STRESS,
                study.RECOVERY,
                study.RECOVERY,
            ],
        }
    )
    episodes = study.build_episodes(frame)
    assert episodes["signal_rows"].tolist() == [2, 2, 2]
    assert episodes["left_censored"].tolist() == [True, False, False]
    assert episodes["right_censored"].tolist() == [False, False, True]


def test_fixed_contrast_directions_use_only_state_conditioned_outcomes() -> None:
    frame = pd.DataFrame(
        {
            "state": [
                study.NORMAL,
                study.NORMAL,
                study.STRESS,
                study.STRESS,
                study.RECOVERY,
                study.RECOVERY,
            ],
            "future_return_10d": [0.02, 0.01, -0.08, -0.04, 0.03, 0.04],
            "future_mae_magnitude_10d": [0.01, 0.02, 0.12, 0.08, 0.02, 0.01],
            "future_return_20d": [0.03, 0.02, -0.10, -0.05, 0.12, 0.08],
            "future_positive_wealth_fraction_20d": [
                0.70,
                0.60,
                0.20,
                0.30,
                0.90,
                0.80,
            ],
        }
    )
    contrast = study.compute_contrast(frame, minimum_days=2)
    assert contrast.stress_directions_pass is True
    assert contrast.recovery_directions_pass is True
    assert contrast.all_directions_pass is True


def test_manifest_hashes_validate_when_frozen() -> None:
    if not study.MANIFEST_PATH.exists():
        return
    config = study.load_config()
    manifest = json.loads(study.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["frozen_before_outcome_read"] is True
    assert manifest["outcome_read_before_freeze"] is False
    assert study.validate_manifest(config)["state"] == "FROZEN_BEFORE_FIRST_OUTCOME_READ"
