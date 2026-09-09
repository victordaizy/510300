from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

import macd_breadth_downside_preflight_v1 as preflight


def test_config_freezes_510300_only_and_disables_execution() -> None:
    config = preflight.load_config()
    assert config["scope"]["execution_asset"] == "510300.SH"
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["scope"]["leverage_allowed"] is False
    assert config["scope"]["short_selling_allowed"] is False
    assert config["scope"]["derivatives_execution_allowed"] is False
    assert config["governance"]["order_generation"] is False
    assert config["governance"]["broker_connection"] is False
    assert config["governance"]["position_change"] is False
    assert config["governance"]["live_trading_authorized"] is False


def test_config_freezes_five_year_window_and_no_rescue() -> None:
    config = preflight.load_config()
    assert config["dates"]["evaluation_start"] == "2021-08-12"
    assert config["dates"]["evaluation_end"] == "2026-08-12"
    assert config["protocol"]["one_shot"] is True
    assert config["protocol"]["parameter_rescue_after_result"] == "forbidden"
    assert config["protocol"]["reverse_direction_after_result"] == "forbidden"
    assert config["event_sampling"]["minimum_gap_within_class_trading_days"] == 10
    assert config["bootstrap"]["repetitions"] == 5000


def test_current_input_files_have_required_identity_and_coverage() -> None:
    config = preflight.load_config()
    market, dividends, breadth = preflight.load_inputs(config)
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    breadth["date"] = pd.to_datetime(breadth["date"]).dt.normalize()
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    evaluation_market = market.loc[market["date"].between(start, end)]
    evaluation_breadth = breadth.loc[breadth["date"].between(start, end)]
    assert len(evaluation_market) == 1211
    assert len(evaluation_breadth) == 1211
    assert evaluation_market["symbol"].drop_duplicates().tolist() == ["510300.SH"]
    assert (evaluation_breadth["component_count"] == 300).all()
    assert (evaluation_breadth["weight_snapshot_date"] <= evaluation_breadth["date"]).all()
    assert dividends.loc[dividends["symbol"] == "510300.SH"].shape[0] >= 14


def test_forward_outcome_uses_record_date_entitlement_not_ex_date_backfill() -> None:
    config = preflight.load_config()
    dates = pd.bdate_range("2026-01-01", periods=30)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(len(dates), 100.0),
            "close": np.full(len(dates), 100.0),
            "record_dividend_cumulative": np.r_[0.0, np.full(len(dates) - 1, 2.0)],
        }
    )
    output = preflight.add_forward_outcomes(frame, config)
    assert np.isclose(output.loc[0, "forward_return_5d"], 0.02)
    assert np.isclose(output.loc[1, "forward_return_5d"], 0.0)
    assert np.isclose(output.loc[0, "forward_mae_5d"], 0.02)


def test_event_selection_enforces_ten_day_gap_within_each_class() -> None:
    config = preflight.load_config()
    rows = 50
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=rows),
            "macd_transition": [
                "CONVERGING" if index % 2 == 0 else "EXPANDING"
                for index in range(rows)
            ],
            "breadth_state": ["IMPROVING"] * rows,
            "downside_state": ["COOLING"] * rows,
        }
    )
    selected = preflight.select_events(frame, config)
    source_positions = {date: index for index, date in enumerate(frame["date"])}
    for _, group in selected.groupby("primary_class"):
        positions = [source_positions[date] for date in group["date"]]
        assert all(right - left >= 10 for left, right in zip(positions, positions[1:]))


def test_block_bootstrap_is_deterministic_and_preserves_direction() -> None:
    config = preflight.load_config()
    left = np.linspace(0.01, 0.03, 40)
    right = np.linspace(-0.02, 0.0, 40)
    first = preflight.bootstrap_mean_difference(left, right, config, seed_offset=7)
    second = preflight.bootstrap_mean_difference(left, right, config, seed_offset=7)
    assert first == second
    assert first[0] is not None and first[0] > 0
    assert first[1] is not None and first[1] > first[0]


def test_manifest_hashes_validate_after_freeze() -> None:
    if not preflight.MANIFEST_PATH.exists():
        return
    config = preflight.load_config()
    manifest = json.loads(preflight.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["config_sha256"] == preflight.sha256_file(preflight.CONFIG_PATH)
    validated = preflight.validate_manifest(config)
    assert validated["hash_mismatches"] == {}
