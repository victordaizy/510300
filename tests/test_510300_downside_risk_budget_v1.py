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

import downside_risk_budget_v1 as study


def test_config_freezes_single_asset_risk_first_contract() -> None:
    config = study.load_config()
    assert config["scope"]["execution_asset"] == "510300.SH"
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["dates"]["evaluation_start"] == "2015-01-05"
    assert config["dates"]["evaluation_end"] == "2026-08-12"
    assert config["protocol"]["one_shot"] is True
    assert config["portfolio"]["enabled_only_if_risk_gate_passes"] is True
    assert config["governance"]["order_generation"] is False
    assert config["governance"]["live_trading_authorized"] is False


def test_input_audit_passed_without_candidate_outcomes() -> None:
    path = ROOT / "reports" / "data_quality" / "510300_downside_risk_inputs_v1.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["candidate_outcomes_computed"] is False
    assert report["portfolio_returns_computed"] is False
    assert report["evaluation_window"]["trading_rows"] == 2821
    assert len(report["canonical_price"]["corrections"]) == 3
    assert report["checks"]["post_correction_tushare_ohlc_exact"] is True


def test_causal_percentile_cannot_see_future_values() -> None:
    original = pd.Series(np.arange(20, dtype=float))
    changed = original.copy()
    changed.iloc[15:] = 1_000_000.0
    first = study.causal_percentile(original, 5)
    second = study.causal_percentile(changed, 5)
    pd.testing.assert_series_equal(first.iloc[:15], second.iloc[:15])
    assert first.iloc[5] == 1.0


def test_cash_dividend_neutralizes_ex_date_price_drop() -> None:
    market = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "open": [10.0, 9.0, 9.5],
            "high": [10.1, 9.1, 9.6],
            "low": [9.9, 8.9, 9.4],
            "close": [10.0, 9.0, 9.5],
        }
    )
    dividends = pd.DataFrame(
        {
            "ex_date": [pd.Timestamp("2020-01-02")],
            "cash_dividend_per_share": [1.0],
        }
    )
    returns, index, cash = study.causal_total_return_series(market, dividends)
    assert returns.iloc[1] == 0.0
    assert cash.iloc[1] == 1.0
    assert index.iloc[1] == 1.0


def test_forward_risk_target_uses_record_date_entitlement() -> None:
    dates = pd.bdate_range("2020-01-01", periods=25)
    market = pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
        }
    )
    features = pd.DataFrame(
        {
            "date": [dates[0]],
            "rv20": [0.20],
        }
    )
    dividends = pd.DataFrame(
        {
            "record_date": [dates[2]],
            "cash_dividend_per_share": [1.0],
        }
    )
    config = study.load_config()
    output = study.add_forward_risk_targets(features, market, dividends, config)
    assert output.loc[0, "future_mae_20d"] == 0.0
    assert output.loc[0, "future_mae_magnitude_20d"] == 0.0
    assert np.isclose(output.loc[0, "future_terminal_return_20d"], 0.1)


def test_portfolio_target_moves_at_most_one_half_step() -> None:
    features = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=4),
            "composite_risk_score": [0.10, 0.10, 0.90, 0.90],
        }
    )
    targets = study.build_portfolio_targets(features, study.load_config())
    assert targets["target_position"].tolist() == [0.5, 1.0, 0.5, 0.0]
    assert targets["target_position"].diff().dropna().abs().max() <= 0.5


def test_ex_date_cash_is_removed_from_gap_measurement() -> None:
    config = deepcopy(study.load_config())
    dates = pd.bdate_range("2010-01-01", periods=800)
    index = np.arange(len(dates), dtype=float)
    close = 10.0 + 0.0005 * index + 0.08 * np.sin(index / 11.0)
    open_price = np.concatenate(([close[0]], close[:-1]))
    market = pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": np.maximum(open_price, close) + 0.05,
            "low": np.minimum(open_price, close) - 0.05,
            "close": close,
            "volume": 1_000_000,
            "amount": 10_000_000,
        }
    )
    ex_index = 700
    previous_close = float(market.loc[ex_index - 1, "close"])
    ex_open = previous_close - 1.0
    market.loc[ex_index, ["open", "high", "low", "close"]] = [
        ex_open,
        ex_open + 0.05,
        ex_open - 0.05,
        ex_open,
    ]
    dividends = pd.DataFrame(
        {
            "ex_date": [dates[ex_index]],
            "cash_dividend_per_share": [1.0],
        }
    )
    config["dates"]["evaluation_start"] = dates[650].date().isoformat()
    config["dates"]["evaluation_end"] = dates[-1].date().isoformat()
    config["data_contract"]["expected_evaluation_rows"] = 150
    config["dates"]["structural_periods"] = {
        "EARLY": {
            "start": dates[650].date().isoformat(),
            "end": dates[699].date().isoformat(),
        },
        "MIDDLE": {
            "start": dates[700].date().isoformat(),
            "end": dates[749].date().isoformat(),
        },
        "LATE": {
            "start": dates[750].date().isoformat(),
            "end": dates[-1].date().isoformat(),
        },
    }
    features = study.build_daily_features(market, dividends, config)
    ex_row = features.loc[features["date"] == dates[ex_index]].iloc[0]
    assert np.isclose(ex_row["absolute_overnight_gap"], 0.0)


def test_bootstrap_is_reproducible_on_synthetic_rows() -> None:
    config = deepcopy(study.load_config())
    config["bootstrap"]["repetitions"] = 100
    score = np.linspace(0.0, 1.0, 400)
    valid = pd.DataFrame(
        {
            "composite_risk_score": score,
            "rv20_risk_percentile": score * 0.8,
            "future_mae_magnitude_20d": score * 0.1,
        }
    )
    first = study.bootstrap_risk_metrics(valid, config)
    second = study.bootstrap_risk_metrics(valid, config)
    assert first == second
    assert first["high_minus_low_mean_mae_magnitude"][0] > 0


def test_bad_synthetic_risk_score_cannot_pass_gate() -> None:
    config = deepcopy(study.load_config())
    config["bootstrap"]["repetitions"] = 100
    dates = pd.date_range("2015-01-05", periods=600, freq="7D")
    score = np.linspace(0.0, 1.0, len(dates))
    target = 0.1 * (1.0 - score)
    frame = pd.DataFrame(
        {
            "date": dates,
            "structural_period": [
                study._structural_period(pd.Timestamp(date), config) for date in dates
            ],
            "composite_risk_score": score,
            "rv20_risk_percentile": score,
            "future_mae_magnitude_20d": target,
            "future_tail_event_20d": (target > 0.07).astype(float),
            "slow_trend_risk_percentile": score,
            "downside_variance_ratio_percentile": score,
            "gap_atr_percentile": score,
            "vol_of_vol_percentile": score,
        }
    )
    result = study.evaluate_risk_gate(frame, config)
    assert result["passed"] is False
    assert result["gates"]["high_minus_low_mean_mae_magnitude_positive"] is False


def test_real_data_features_cover_full_evaluation_without_missing_scores() -> None:
    config = study.load_config()
    market, dividends, _, _ = study.load_inputs(config)
    features = study.build_daily_features(market, dividends, config)
    assert len(features) == 2821
    assert features["date"].min() == pd.Timestamp("2015-01-05")
    assert features["date"].max() == pd.Timestamp("2026-08-12")
    assert features[
        ["composite_risk_score", "rv20_risk_percentile"]
    ].notna().all().all()


def test_manifest_hashes_validate_after_freeze() -> None:
    if not study.MANIFEST_PATH.exists():
        return
    config = study.load_config()
    manifest = json.loads(study.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["config_sha256"] == study.sha256_file(study.CONFIG_PATH)
    assert study.validate_manifest(config)["state"] == "FROZEN_BEFORE_FIRST_RESULT"


def test_formal_result_recomputes_exactly_when_present() -> None:
    config = study.load_config()
    result_path = ROOT / config["paths"]["result_json"]
    if not result_path.exists() or not study.MANIFEST_PATH.exists():
        return
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    recomputed = study.run_study(write=False)
    assert recomputed == stored
