from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.asymmetric_stress_hazard_g2_mechanism_v1 import (
    G2_FAIL_STATUS,
    G2_PASS_STATUS,
    Subperiod,
    _publication_asof,
    assign_effective_industry,
    build_constituent_return_index,
    build_independent_units,
    causal_midrank,
    evaluate_g2,
    required_median,
)
from scripts.freeze_510300_asymmetric_stress_hazard_v1_g2_mechanism import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_causal_midrank_excludes_current_and_uses_midrank_for_ties() -> None:
    values = pd.Series([1.0, 2.0, 2.0, 0.0, np.nan, 3.0])
    result = causal_midrank(values)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 1.0
    assert result.iloc[2] == 0.75
    assert result.iloc[3] == 0.0
    assert np.isnan(result.iloc[4])
    assert result.iloc[5] == 1.0


def test_required_median_never_substitutes_two_of_three_channels() -> None:
    frame = pd.DataFrame(
        {
            "a": [0.1, 0.1],
            "b": [0.2, np.nan],
            "c": [0.9, 0.9],
        }
    )
    result = required_median(frame, ["a", "b", "c"])
    assert result.iloc[0] == 0.2
    assert np.isnan(result.iloc[1])


def test_industry_assignment_uses_effective_and_out_dates() -> None:
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-06-01", "2021-01-04"]),
            "symbol": ["000001.SZ"] * 3,
        }
    )
    history = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "effective_date": ["2010-01-01", "2020-06-01"],
            "out_date": ["2020-06-01", None],
            "industry_l1_code": ["44", "48"],
        }
    )
    result = assign_effective_industry(membership, history)
    assert result["industry_l1_code"].astype(str).tolist() == ["44", "48", "48"]


def test_industry_assignment_normalizes_mixed_datetime_precision() -> None:
    membership = pd.DataFrame(
        {
            "date": pd.Series(
                np.array(["2020-01-02", "2020-06-01"], dtype="datetime64[s]")
            ),
            "symbol": ["000001.SZ", "000001.SZ"],
        }
    )
    history = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "effective_date": pd.to_datetime(["2010-01-01", "2020-06-01"]),
            "out_date": pd.to_datetime(["2020-06-01", None]),
            "industry_l1_code": ["44", "48"],
        }
    )

    result = assign_effective_industry(membership, history)

    assert str(result["date"].dtype) == "datetime64[ns]"
    assert result["industry_l1_code"].astype(str).tolist() == ["44", "48"]


def test_publication_asof_normalizes_mixed_utc_datetime_precision() -> None:
    market_dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    available_at = pd.Series(
        np.array(
            ["2020-01-02T06:30:00.000000", "2020-01-03T06:30:00.000000"],
            dtype="datetime64[us]",
        )
    ).dt.tz_localize("UTC")

    result = _publication_asof(
        market_dates,
        available_at,
        pd.Series([1.0, 2.0]),
    )

    assert result.tolist() == [1.0, 2.0]


def test_constituent_return_index_switches_without_level_splice() -> None:
    dates = pd.bdate_range("2020-01-29", periods=5)
    historical = pd.DataFrame(
        {
            "date": dates[:3],
            "con_code": ["000001.SZ"] * 3,
            "pre_close": [10.0, 11.0, 12.0],
            "raw_close": [11.0, 12.0, 13.0],
        }
    )
    current = pd.DataFrame(
        {
            "date": dates,
            "con_code": ["000001.SZ"] * 5,
            "total_return_close": [100.0, 110.0, 121.0, 133.1, 146.41],
        }
    )
    result = build_constituent_return_index(
        historical,
        current,
        market_dates=dates,
        cutover_date=str(dates[3].date()),
    )
    expected = np.cumprod([1.1, 12.0 / 11.0, 13.0 / 12.0, 1.1, 1.1])
    assert np.allclose(result["000001.SZ"].to_numpy(), expected)


def _features_and_units() -> tuple[pd.DataFrame, pd.DataFrame, list[Subperiod]]:
    dates = pd.bdate_range("2015-01-05", periods=80)
    within_period = np.linspace(0.1, 0.9, 20)
    internal = np.concatenate(
        [(period + within_period**2) / 4.0 for period in range(4)]
    )
    features = pd.DataFrame(
        {
            "date": dates,
            "M": np.tile(within_period, 4),
            "F": 0.5,
            "T": 0.5,
            "internal_score": internal,
            "price_baseline": np.tile([0.45, 0.55], 40),
            "full_mechanism_score": 0.5,
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [f"E{i}" for i in range(30)],
            "event_origin_start": dates[-30:],
        }
    )
    non_events = pd.DataFrame(
        {
            "block_id": [f"N{i}" for i in range(50)],
            "origin_date": dates[:50],
        }
    )
    units = build_independent_units(
        events=events,
        non_events=non_events,
        features=features,
    )
    periods = [
        Subperiod(f"P{i + 1}", dates[i * 20], dates[i * 20 + 19])
        for i in range(4)
    ]
    return features, units, periods


def test_g2_requires_both_macro_direction_and_internal_auc() -> None:
    features, units, periods = _features_and_units()
    result = evaluate_g2(
        features=features,
        units=units,
        subperiods=periods,
        lead_days=1,
        minimum_events=30,
        minimum_non_events=50,
        required_positive_subperiods=3,
    )
    assert result["status"] == G2_PASS_STATUS
    assert result["passed"] is True
    assert result["macro_lead_gate"]["positive_subperiod_count"] == 4
    assert result["internal_bad10_gate"]["internal_score_roc_auc"] > result[
        "internal_bad10_gate"
    ]["price_baseline_roc_auc"]

    failed = evaluate_g2(
        features=features.assign(M=-features["M"]),
        units=units,
        subperiods=periods,
        lead_days=1,
        minimum_events=30,
        minimum_non_events=50,
        required_positive_subperiods=3,
    )
    assert failed["status"] == G2_FAIL_STATUS
    assert failed["passed"] is False
    assert failed["g3_allowed"] is False


def test_original_route_g2_config_is_fixed() -> None:
    config = load_config(DEFAULT_CONFIG)
    validate_config(config)
    assert config["program"]["extra_temporal_preflight_is_active_gate"] is False
    assert config["feature_contract"]["constituent_return_lookback_market_days"] == 20
    assert config["feature_contract"]["transmission_change_market_days"] == 5
    assert config["g2_gate"]["minimum_scoreable_independent_events"] == 30
    assert config["g2_gate"]["minimum_scoreable_independent_non_event_blocks"] == 120
    from scripts import run_510300_asymmetric_stress_hazard_v1_g2_mechanism as runner

    assert callable(runner.prepare_membership)


def test_frozen_manifest_verifies_when_present() -> None:
    config = load_config(DEFAULT_CONFIG)
    manifest = ROOT / config["outputs"]["manifest"]
    if not manifest.exists():
        pytest.skip("G2 manifest 尚未生成")
    verified_config, payload = verify_frozen_contract(DEFAULT_CONFIG)
    assert verified_config["program"]["stage_id"] == "G2_MECHANISM_CHAIN"
    assert payload["result_values_read"] is False
