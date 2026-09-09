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

import trend_lifecycle_atlas_v1 as atlas


def test_protocol_is_discovery_only_and_preserves_no_trade_boundaries() -> None:
    config = atlas.load_config()
    assert config["protocol"]["project_id"] == atlas.PROJECT_ID
    assert config["protocol"]["strategy_backtest_allowed"] is False
    assert config["scope"]["return_backtest_allowed"] is False
    assert config["scope"]["range_t_backtest_allowed"] is False
    assert config["scope"]["range_default_action"] == "NO_NEW_TRADE"
    assert config["scope"]["portfolio_mapping"] == "disabled"
    assert config["scope"]["paper_signal"] == "disabled"
    assert config["scope"]["shadow_signal"] == "disabled"
    assert config["scope"]["order_generation"] is False
    assert config["scope"]["broker_connection"] is False
    assert config["scope"]["live_trading_authorized"] is False
    assert config["data_gates"]["source_provenance_gate_for_phase_2"] is False
    assert config["adjudication"]["net_sharpe"] == "NOT_COMPUTED"


def test_causal_percentile_excludes_current_and_future_values() -> None:
    original = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    changed = original.copy()
    changed.iloc[4:] = [50_000.0, -50_000.0]
    first = atlas.causal_percentile(original, prior_rows=3, minimum_prior_rows=3)
    second = atlas.causal_percentile(changed, prior_rows=3, minimum_prior_rows=3)
    pd.testing.assert_series_equal(first.iloc[:4], second.iloc[:4])
    assert first.iloc[:3].isna().all()
    assert first.iloc[3] == 1.0


def test_label_a_records_pivot_and_confirmation_separately() -> None:
    config = deepcopy(atlas.load_config())
    config["dates"]["evaluation_start"] = "2020-01-01"
    dates = pd.bdate_range("2020-01-01", periods=240)
    segments = np.concatenate(
        [
            np.linspace(100.0, 101.0, 70),
            np.linspace(101.0, 125.0, 50),
            np.linspace(125.0, 88.0, 60),
            np.linspace(88.0, 118.0, 60),
        ]
    )
    market = pd.DataFrame(
        {
            "date": dates,
            "symbol": "H00300",
            "name": "合成全收益指数",
            "close": segments,
            "source": "合成测试",
        }
    )
    events = atlas.build_label_a_events(config, market)
    assert {"UP_START", "UP_END", "DOWN_START", "DOWN_END"}.issubset(
        set(events["event_type"])
    )
    assert (events["confirmation_date"] >= events["event_date"]).all()
    assert (events["confirmation_delay_rows"] >= 0).all()
    assert events["event_id"].is_unique


def test_future_path_label_is_separate_and_right_censored() -> None:
    config = deepcopy(atlas.load_config())
    dates = pd.bdate_range("2020-01-01", periods=180)
    close = np.exp(np.linspace(np.log(100.0), np.log(180.0), len(dates)))
    market = pd.DataFrame(
        {
            "date": dates,
            "symbol": "H00300",
            "name": "合成全收益指数",
            "close": close,
            "source": "合成测试",
        }
    )
    labels = atlas.build_outcome_label_panel(config, market)
    assert labels["outcome_complete_60d"].sum() == len(labels) - 60
    assert labels.tail(60)["future_path_label"].eq("CENSORED_FUTURE_60D").all()
    assert labels.iloc[80]["future_path_label"] == "SUSTAINED_UP"
    assert all(column.startswith(("date", "future_", "outcome_")) for column in labels)


def test_label_b_bridges_only_short_gaps_and_enforces_episode_length() -> None:
    config = deepcopy(atlas.load_config())
    dates = pd.bdate_range("2020-06-23", periods=60)
    labels = ["MIXED_TRANSITION"] * 60
    for position in [5, 6, 7, 9, 10, 11, 12]:
        labels[position] = "SUSTAINED_UP"
    for position in [35, 36, 37, 38]:
        labels[position] = "SUSTAINED_DOWN"
    panel = pd.DataFrame({"date": dates, "future_path_label": labels})
    events = atlas.build_label_b_events(config, panel)
    up = events.loc[events["path_label"].eq("SUSTAINED_UP")]
    assert set(up["event_type"]) == {"UP_START", "UP_END"}
    assert up["episode_rows"].eq(8).all()
    assert not events["path_label"].eq("SUSTAINED_DOWN").any()


def test_average_pairwise_correlation_uses_only_complete_past_window() -> None:
    dates = pd.bdate_range("2020-01-01", periods=8)
    base = np.arange(8, dtype=float)
    returns = pd.DataFrame(
        {"A": base, "B": base * 2.0, "C": base * 3.0}, index=dates
    )
    result = atlas._rolling_average_pairwise_correlation(
        returns,
        window=5,
        minimum_complete_rows=5,
        minimum_assets=3,
    )
    assert result.iloc[:4].isna().all()
    np.testing.assert_allclose(result.iloc[4:], 1.0, atol=1e-12)


def _synthetic_trajectories(config: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for label in [atlas.LABEL_A, atlas.LABEL_B]:
        for event_number in range(8):
            era = "EARLY" if event_number < 4 else "LATE"
            for relative_day in range(-120, 61):
                if -15 <= relative_day <= -5:
                    z_value = 1.0
                elif -20 <= relative_day <= -16:
                    z_value = 0.7
                else:
                    z_value = 0.0
                rows.append(
                    {
                        "event_id": f"{label}_{event_number}",
                        "label_system": label,
                        "event_type": "UP_START",
                        "event_date": pd.Timestamp("2021-01-01"),
                        "era": era,
                        "relative_trading_day": relative_day,
                        "feature": "price_trend_z20",
                        "module": "PRICE_ACCEPTANCE",
                        "raw_value": z_value,
                        "baseline_median": 0.0,
                        "baseline_scale": 1.0,
                        "event_z": z_value,
                        "feature_available": True,
                    }
                )
    return pd.DataFrame(rows)


def test_leading_gate_is_event_weighted_and_dual_label() -> None:
    config = atlas.load_config()
    trajectories = _synthetic_trajectories(config)
    leading = atlas.evaluate_leading_results(config, trajectories)
    assert len(leading) == 2
    assert leading["passed"].all()
    assert leading["event_observations"].eq(8).all()
    assert leading["median_onset_relative_day"].eq(-20.0).all()
    candidates, gate = atlas.build_dual_label_candidates(config, leading)
    row = candidates.loc[
        candidates["event_type"].eq("UP_START")
        & candidates["feature"].eq("price_trend_z20")
    ].iloc[0]
    assert bool(row["dual_label_candidate_passed"]) is True
    assert gate["all_required_transitions_passed"] is False


def test_reference_state_priority_keeps_down_state_over_up_state() -> None:
    config = deepcopy(atlas.load_config())
    dates = pd.bdate_range("2020-01-01", periods=220)
    feature_panel = pd.DataFrame(
        {
            "date": dates,
            "valid_for_atlas": True,
            "data_status": "PASS_ATLAS_FEATURE_ROW",
            "contribution_hhi20": np.linspace(0.1, 0.9, len(dates)),
            "average_pairwise_correlation20": np.linspace(0.1, 0.9, len(dates)),
            "downside_vol_ratio5_20": np.linspace(0.1, 1.5, len(dates)),
            "extreme_down_weight": np.linspace(0.0, 0.5, len(dates)),
            "liquidity_impact_pressure": np.linspace(0.0, 1.0, len(dates)),
            "cgb_10y_change63_pressure": np.linspace(0.0, 1.0, len(dates)),
            "breadth_above_ma60_snapshot_weighted": 0.8,
        }
    )
    outcome = pd.DataFrame(
        {"date": dates, "future_path_label": "SUSTAINED_UP"}
    )
    event_date = dates[170]
    events = pd.DataFrame(
        {
            "event_id": ["A_DOWN_START"],
            "label_system": [atlas.LABEL_A],
            "event_type": ["DOWN_START"],
            "event_date": [event_date],
        }
    )
    result = atlas.build_lifecycle_reference_panel(
        config, feature_panel, outcome, events
    )
    assert result.loc[170, "reference_state"] == "D0_EARLY_DOWN"
    assert result.loc[170, "reference_state"] != "U1_BROAD_UPTREND"
    assert result["reference_semantics"].eq(
        "RETROSPECTIVE_REFERENCE_ONLY_NOT_A_POINT_IN_TIME_SIGNAL"
    ).all()


def test_manifest_integrity_when_frozen() -> None:
    if not atlas.MANIFEST_PATH.exists():
        return
    config = atlas.load_config()
    manifest = json.loads(atlas.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["frozen_before_new_lifecycle_outcome_read"] is True
    assert manifest["new_lifecycle_outcome_read_before_freeze"] is False
    assert atlas.validate_manifest(config)["state"] == (
        "FROZEN_BEFORE_FIRST_NEW_LIFECYCLE_OUTCOME_READ"
    )
