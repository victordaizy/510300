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

import oil_trend_monthly_timing_v1 as study


def _synthetic_oil(months: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2018-01-31", periods=months, freq="ME")
    return pd.DataFrame(
        {
            "date": dates,
            "value": np.linspace(50.0, 90.0, months),
            "availability_date": dates + pd.offsets.BDay(5),
            "robust_availability_date": dates + pd.offsets.BDay(10),
            "series": "RBRTE",
            "units": "$/BBL",
        }
    )


def test_config_freezes_single_asset_one_shot_contract() -> None:
    config = study.load_config()
    assert config["scope"]["execution_asset"] == "510300.SH"
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["dates"]["evaluation_start"] == "2015-01-05"
    assert config["dates"]["evaluation_end"] == "2026-08-12"
    assert config["oil_factor"]["lag_months"] == 12
    assert config["data_contract"]["oil_series"] == "RBRTE"
    assert config["protocol"]["one_shot"] is True
    assert config["governance"]["order_generation"] is False
    assert config["governance"]["live_trading_authorized"] is False


def test_pre_freeze_wti_rejection_is_data_domain_decision() -> None:
    decision = study.load_config()["protocol"]["pre_freeze_data_decision"]
    assert decision["status"] == "CORRECTED_BEFORE_ANY_510300_OUTCOME_WAS_READ"
    assert decision["rejected_input_series"] == "EIA_WTI_RWTC"
    assert decision["admitted_input_series"] == "EIA_BRENT_RBRTE"
    assert "未读取任何510300候选收益" in decision["decision_basis"]


def test_input_audit_passed_without_candidate_outcomes() -> None:
    path = ROOT / "reports" / "data_quality" / "510300_oil_trend_inputs_v1.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["candidate_outcomes_computed"] is False
    assert report["portfolio_returns_computed"] is False
    assert report["oil_daily"]["rows"] == 3706
    assert report["oil_daily"]["nonpositive_values"] == 0
    assert report["checks"]["availability_strictly_after_observation"] is True


def test_oil_factor_matches_frozen_formula() -> None:
    oil = _synthetic_oil(8)
    factor = study.build_monthly_oil_factor(oil, 3)
    prices = oil["value"].to_numpy(dtype=float)
    normalized_2 = prices[0:3].mean() / prices[2]
    normalized_3 = prices[1:4].mean() / prices[3]
    expected_first = np.log(normalized_3) - np.log(normalized_2)
    assert np.isclose(factor.iloc[0]["oil_factor"], expected_first)
    assert factor.iloc[0]["oil_observation_date"] == oil.iloc[3]["date"]
    assert factor.iloc[0]["factor_available_date"] == oil.iloc[3]["availability_date"]


def test_future_oil_change_cannot_change_past_factor() -> None:
    original = _synthetic_oil(24)
    changed = original.copy()
    changed.loc[changed.index >= 18, "value"] *= 100.0
    first = study.build_monthly_oil_factor(original, 6)
    second = study.build_monthly_oil_factor(changed, 6)
    cutoff = original.loc[17, "date"]
    left = first.loc[first["oil_observation_date"] <= cutoff, "oil_factor"].reset_index(drop=True)
    right = second.loc[second["oil_observation_date"] <= cutoff, "oil_factor"].reset_index(drop=True)
    pd.testing.assert_series_equal(left, right)


def test_signal_alignment_never_uses_future_availability() -> None:
    config = study.load_config()
    dates = pd.bdate_range("2018-01-01", "2020-02-14")
    market = pd.DataFrame({"date": dates})
    oil = _synthetic_oil(30)
    factor = study.build_monthly_oil_factor(oil, 3)
    signals = study.build_signal_schedule(market, factor, config)
    valid = signals.loc[signals["oil_factor"].notna()]
    assert not valid.empty
    assert (valid["factor_available_date"] <= valid["decision_date"]).all()
    assert signals["market_month"].max() < dates.max().to_period("M")


def test_positive_factor_maps_to_cash_without_direction_reversal() -> None:
    config = study.load_config()
    dates = pd.bdate_range("2018-01-01", "2020-02-14")
    market = pd.DataFrame({"date": dates})
    factor = study.build_monthly_oil_factor(_synthetic_oil(30), 3)
    signals = study.build_signal_schedule(market, factor, config)
    valid = signals.loc[signals["oil_factor"].notna()]
    assert (valid.loc[valid["oil_factor"] > 0.0, "target_position"] == 0.0).all()
    assert (valid.loc[valid["oil_factor"] <= 0.0, "target_position"] == 1.0).all()


def test_mechanism_target_uses_record_date_entitlement() -> None:
    config = deepcopy(study.load_config())
    config["dates"]["evaluation_start"] = "2020-01-02"
    config["dates"]["evaluation_end"] = "2020-03-31"
    dates = pd.bdate_range("2019-12-31", "2020-03-31")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
        }
    )
    signals = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")],
            "execution_date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-02-03"), pd.Timestamp("2020-03-02")],
            "target_position": [1.0, 1.0, 1.0],
            "oil_factor": [-0.1, -0.1, -0.1],
        }
    )
    dividends = pd.DataFrame(
        {
            "record_date": [pd.Timestamp("2020-01-15"), pd.Timestamp("2020-02-03")],
            "cash_dividend_per_share": [1.0, 5.0],
        }
    )
    targets = study.add_mechanism_targets(signals, market, dividends, config)
    assert len(targets) == 2
    assert targets.iloc[0]["entitled_cash_dividend_per_share"] == 1.0
    assert np.isclose(targets.iloc[0]["future_interval_total_return"], 0.1)


def _synthetic_mechanism(good: bool) -> pd.DataFrame:
    dates = pd.date_range("2015-01-31", periods=138, freq="ME")
    factor = np.where(np.arange(len(dates)) % 2 == 0, 1.0, -1.0)
    direction = -1.0 if good else 1.0
    future_return = direction * 0.02 * factor + 0.001 * np.sin(np.arange(len(dates)))
    return pd.DataFrame(
        {
            "execution_date": dates,
            "calendar_year": dates.year,
            "structural_period": np.where(
                dates <= pd.Timestamp("2019-12-31"),
                "PAPER_SAMPLE_OVERLAP",
                "POST_PUBLISHED_SAMPLE",
            ),
            "oil_factor": factor,
            "future_interval_total_return": future_return,
        }
    )


def test_good_synthetic_relation_passes_mechanism_gate() -> None:
    config = deepcopy(study.load_config())
    config["bootstrap"]["repetitions"] = 200
    result = study.evaluate_mechanism_gate(_synthetic_mechanism(True), config)
    assert result["passed"] is True
    assert result["mean_difference"] < 0.0
    assert result["newey_west_regression"]["t_stat"] < -1.645


def test_wrong_direction_synthetic_relation_cannot_pass() -> None:
    config = deepcopy(study.load_config())
    config["bootstrap"]["repetitions"] = 100
    result = study.evaluate_mechanism_gate(_synthetic_mechanism(False), config)
    assert result["passed"] is False
    assert result["gates"]["full_sample_mean_difference_negative"] is False
    assert result["gates"]["newey_west_slope_negative"] is False


def test_real_inputs_cover_factor_without_reading_candidate_returns() -> None:
    config = study.load_config()
    market, _, _, oil, audit = study.load_inputs(config)
    factor = study.build_monthly_oil_factor(oil, 12)
    signals = study.build_signal_schedule(market, factor, config)
    valid = signals.loc[signals["oil_factor"].notna()]
    assert audit["candidate_outcomes_computed"] is False
    assert len(factor) == 164
    assert len(valid) == 162
    assert (valid["factor_available_date"] <= valid["decision_date"]).all()


def test_synthetic_portfolio_branch_runs_without_real_candidate_outcomes() -> None:
    config = deepcopy(study.load_config())
    dates = pd.bdate_range("2014-12-31", "2026-08-12")
    prices = np.linspace(10.0, 20.0, len(dates))
    market = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices,
            "volume": 1_000_000.0,
            "amount": 10_000_000.0,
        }
    )
    calendar = pd.DataFrame({"date": dates})
    calendar["month"] = calendar["date"].dt.to_period("M")
    decisions = calendar.groupby("month", sort=True).tail(1).copy()
    date_position = {date: index for index, date in enumerate(dates)}
    decisions["execution_date"] = [
        dates[date_position[date] + 1]
        if date_position[date] + 1 < len(dates)
        else pd.NaT
        for date in decisions["date"]
    ]
    decisions = decisions.loc[decisions["execution_date"].notna()].copy()
    signals = pd.DataFrame(
        {
            "decision_date": decisions["date"].to_numpy(),
            "execution_date": decisions["execution_date"].to_numpy(),
            "target_position": 1.0,
            "signal_reason": "合成账户分支测试",
        }
    )
    dividends = pd.DataFrame(
        {
            "ex_date": pd.Series([], dtype="datetime64[ns]"),
            "payment_date": pd.Series([], dtype="datetime64[ns]"),
            "cash_dividend_per_share": pd.Series([], dtype=float),
        }
    )
    benchmark = pd.DataFrame({"date": dates, "close": prices})
    manifest = {
        "selection_bias_control": {"total_trial_count_including_current": 5}
    }
    report, artifacts = study.evaluate_portfolio(
        market,
        dividends,
        benchmark,
        signals,
        signals.copy(),
        config,
        manifest,
    )
    expected_evaluation_rows = int((dates >= pd.Timestamp("2015-01-05")).sum())
    assert report["base"]["observations"] == expected_evaluation_rows
    assert report["base"]["trade_count"] == 1
    assert set(artifacts) == {
        "base_ledger",
        "base_trades",
        "double_cost_ledger",
        "double_cost_trades",
        "buy_hold_ledger",
        "buy_hold_trades",
        "robust_clock_ledger",
        "robust_clock_trades",
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)


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
