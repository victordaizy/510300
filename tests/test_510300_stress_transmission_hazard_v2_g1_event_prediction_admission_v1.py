from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from research.stress_transmission_hazard_v2 import NO_VIEW, UNASSIGNED_SPLIT, VIEW_ALLOWED
from research.stress_transmission_hazard_v2_g1_event_prediction_admission_v1 import (
    B1_FEATURE_COLUMNS,
    build_b1_feature_panel,
    build_daily_common_view,
    build_etf_total_return_series,
    build_g1_admission_artifacts,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1 import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2020-01-02", periods=100)
    returns = 0.0015 * np.sin(np.arange(len(dates)) / 4.0) + 0.0003
    close = 100.0 * np.cumprod(1.0 + returns)
    daily = pd.DataFrame(
        {"date": dates, "symbol": "510300.SH", "close": close}
    )
    ex_date = dates[35]
    dividends = pd.DataFrame(
        {
            "symbol": ["510300.SH"],
            "record_date": [dates[34]],
            "ex_date": [ex_date],
            "payment_date": [dates[40]],
            "cash_dividend_per_share": [0.5],
        }
    )
    return daily, dividends, dates


def _synthetic_mft(dates: pd.DatetimeIndex) -> pd.DataFrame:
    values = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "F": 0.5 + 0.2 * np.sin(values / 8.0),
            "M": 0.5 + 0.2 * np.cos(values / 9.0),
            "T": 0.5 + 0.2 * np.sin(values / 6.0),
            "B2_feature_state": VIEW_ALLOWED,
            "B3_feature_state": VIEW_ALLOWED,
        }
    )


def _synthetic_samples(
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = dates[40:90]
    event_by_position = {
        5: "BAD10_V2_EVENT_0001",
        6: "BAD10_V2_EVENT_0001",
        15: "BAD10_V2_EVENT_0002",
        25: "BAD10_V2_EVENT_0003",
        35: "BAD10_V2_EVENT_0004",
    }
    counts = {"BAD10_V2_EVENT_0001": 2, "BAD10_V2_EVENT_0002": 1,
              "BAD10_V2_EVENT_0003": 1, "BAD10_V2_EVENT_0004": 1}
    rows: list[dict[str, object]] = []
    for position, origin in enumerate(origins):
        event_id = event_by_position.get(position)
        positive = event_id is not None
        sample_group = event_id if positive else f"NON_EVENT_DAY_{origin:%Y%m%d}"
        rows.append(
            {
                "origin_date": origin,
                "entry_date": origin + pd.Timedelta(days=1),
                "horizon_end_date": origin + pd.Timedelta(days=15),
                "bad10": int(positive),
                "event_id": event_id if positive else pd.NA,
                "sample_group_id": sample_group,
                "bootstrap_event_block_id": sample_group,
                "calendar_year_block_id": f"CALENDAR_YEAR_{origin.year}",
                "training_weight": (1.0 / counts[event_id]) if positive else 1.0,
                "split_assignment": UNASSIGNED_SPLIT,
            }
        )
    events = pd.DataFrame({"event_id": sorted(counts)})
    return pd.DataFrame(rows), events


def _synthetic_config(dates: pd.DatetimeIndex) -> dict[str, object]:
    config = deepcopy(load_config(DEFAULT_CONFIG))
    config["program"]["observation_start"] = dates.min().date().isoformat()
    config["program"]["observation_cutoff"] = dates.max().date().isoformat()
    config["g1_gate_contract"]["mechanism_discovery"]["minimum_independent_events"] = 2
    config["g1_gate_contract"]["mechanism_discovery"]["minimum_non_event_risk_days"] = 5
    config["g1_gate_contract"]["full_three_coefficient_model"]["minimum_independent_events"] = 2
    config["g1_gate_contract"]["full_three_coefficient_model"]["minimum_non_event_risk_days"] = 5
    return config


def test_contract_freezes_label_boundary_and_keeps_prediction_closed() -> None:
    config = load_config(DEFAULT_CONFIG)
    validate_config(config)
    assert config["program"]["research_state"] == "DISCOVERY_ONLY"
    assert config["program"]["return_evaluation"] == "NOT_ALLOWED"
    assert config["program"]["actual_label_read_allowed_only_after_manifest_freeze"] is True
    assert config["program"]["actual_future_path_value_read_allowed"] is False
    assert config["program"]["model_training_allowed"] is False
    assert config["program"]["probability_generation_allowed"] is False
    assert config["program"]["portfolio_evaluation_allowed"] is False
    assert config["program"]["position_impact"] == 0
    assert config["g1_gate_contract"]["mechanism_discovery"]["minimum_independent_events"] == 30
    assert config["g1_gate_contract"]["full_three_coefficient_model"]["minimum_independent_events"] == 40


def test_etf_total_return_uses_unadjusted_close_and_ex_date_cash_only() -> None:
    daily, dividends, dates = _synthetic_market()
    result = build_etf_total_return_series(
        etf_daily=daily,
        dividends=dividends,
        symbol="510300.SH",
        observation_start=dates.min().date().isoformat(),
        observation_cutoff=dates.max().date().isoformat(),
    ).set_index("date")
    ex_date = dates[35]
    expected = (daily.loc[35, "close"] + 0.5) / daily.loc[34, "close"] - 1.0
    assert result.loc[ex_date, "cash_dividend_per_share_on_ex_date"] == 0.5
    assert result.loc[ex_date, "daily_total_return"] == expected
    assert result["daily_total_return"].iloc[0] != result["daily_total_return"].iloc[0]
    assert result["daily_total_return"].iloc[1:].notna().all()


def test_b1_causal_features_are_invariant_to_appended_future_prices() -> None:
    daily, dividends, dates = _synthetic_market()
    full_returns = build_etf_total_return_series(
        etf_daily=daily,
        dividends=dividends,
        symbol="510300.SH",
        observation_start=dates.min().date().isoformat(),
        observation_cutoff=dates.max().date().isoformat(),
    )
    prefix_dates = dates[:75]
    prefix_returns = build_etf_total_return_series(
        etf_daily=daily.loc[daily["date"].isin(prefix_dates)],
        dividends=dividends,
        symbol="510300.SH",
        observation_start=prefix_dates.min().date().isoformat(),
        observation_cutoff=prefix_dates.max().date().isoformat(),
    )
    full = build_b1_feature_panel(full_returns).iloc[:75].reset_index(drop=True)
    prefix = build_b1_feature_panel(prefix_returns).reset_index(drop=True)
    pd.testing.assert_frame_equal(full, prefix, check_exact=True)


def test_common_view_is_constructed_without_any_label_input() -> None:
    daily, dividends, dates = _synthetic_market()
    total = build_etf_total_return_series(
        etf_daily=daily,
        dividends=dividends,
        symbol="510300.SH",
        observation_start=dates.min().date().isoformat(),
        observation_cutoff=dates.max().date().isoformat(),
    )
    b1 = build_b1_feature_panel(total)
    mft = _synthetic_mft(dates)
    mft.loc[dates[60] == mft["date"], "B3_feature_state"] = NO_VIEW
    view = build_daily_common_view(b1=b1, mft=mft)
    assert "bad10" not in view.columns
    assert view.loc[view["date"].eq(dates[60]), "b2_vs_b1_eligible"].item()
    assert not view.loc[view["date"].eq(dates[60]), "b3_vs_b2_eligible"].item()
    assert set(B1_FEATURE_COLUMNS).issubset(view.columns)


def test_g1_eligibility_renormalizes_visible_origins_and_never_fits_model() -> None:
    daily, dividends, dates = _synthetic_market()
    mft = _synthetic_mft(dates)
    samples, events = _synthetic_samples(dates)
    first_event_first_origin = dates[45]
    mft.loc[mft["date"].eq(first_event_first_origin), ["B2_feature_state", "B3_feature_state"]] = NO_VIEW
    artifacts = build_g1_admission_artifacts(
        etf_daily=daily,
        dividends=dividends,
        mft=mft,
        samples=samples,
        events=events,
        config=_synthetic_config(dates),
    )
    ledger = artifacts.sample_eligibility_ledger
    event = artifacts.event_eligibility_ledger.set_index("event_id")
    remaining = ledger.loc[
        ledger["event_id"].eq("BAD10_V2_EVENT_0001")
        & ledger["b2_vs_b1_eligible"]
    ]
    assert len(remaining) == 1
    assert remaining["b2_vs_b1_model_weight"].item() == 1.0
    assert event.loc["BAD10_V2_EVENT_0001", "b2_positive_model_weight_sum"] == 1.0
    assert artifacts.gate_result["mechanism_discovery_prerequisite_passed"] is True
    assert artifacts.gate_result["full_three_coefficient_model_prerequisite_passed"] is True
    assert artifacts.gate_result["G1_DATA_AND_EVENTS"] == (
        "PASS_FULL_B2_AND_B3_EVENT_IDENTIFIABILITY"
    )


def test_g1_insufficient_event_count_fails_closed_before_prediction() -> None:
    daily, dividends, dates = _synthetic_market()
    samples, events = _synthetic_samples(dates)
    config = _synthetic_config(dates)
    config["g1_gate_contract"]["mechanism_discovery"]["minimum_independent_events"] = 10
    config["g1_gate_contract"]["full_three_coefficient_model"]["minimum_independent_events"] = 10
    artifacts = build_g1_admission_artifacts(
        etf_daily=daily,
        dividends=dividends,
        mft=_synthetic_mft(dates),
        samples=samples,
        events=events,
        config=config,
    )
    assert artifacts.gate_result["G1_DATA_AND_EVENTS"] == (
        "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY"
    )
    assert artifacts.gate_result["next_allowed_step"] == (
        "STOP_V2_NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY"
    )


def test_frozen_manifest_verifies_when_present() -> None:
    config = load_config(DEFAULT_CONFIG)
    manifest_path = ROOT / config["freeze_contract"]["manifest_output"]
    if not manifest_path.exists():
        return
    manifest = verify_frozen_manifest(DEFAULT_CONFIG)
    assert manifest["status"] == "FROZEN_BEFORE_FIRST_G1_ACTUAL_LABEL_READ"
    assert manifest["actual_label_artifact_parsed"] is False
    assert manifest["model_trained"] is False
    assert manifest["performance_artifact_read"] is False
    assert manifest["position_impact"] == 0
