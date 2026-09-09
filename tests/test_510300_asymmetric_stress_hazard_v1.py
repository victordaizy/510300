from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.asymmetric_stress_hazard_v1 import (
    adjudicate_g1,
    build_bad10_origin_panel,
    build_census_summary,
    merge_bad10_events,
    prepare_cash_dividends,
    prepare_etf_prices,
    select_independent_non_event_blocks,
)
from research.project_evidence_contract_v1 import (
    read_json_strict,
    validate_authority_state,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_asymmetric_stress_hazard_v1.yaml"
AUTHORITY = ROOT / "config/510300_research_authority_v2.json"


def _prices(count: int = 20) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=count)
    close = np.full(count, 100.0)
    close[2] = 95.5
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.full(count, 100.0),
            "close": close,
            "symbol": "510300.SH",
        }
    )


def _dividends() -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=4)
    return pd.DataFrame(
        {
            "symbol": ["510300.SH"],
            "record_date": [dates[1]],
            "ex_date": [dates[2]],
            "payment_date": [dates[3]],
            "cash_dividend_per_share": [0.5],
        }
    )


def test_bad10_uses_next_open_and_recognizes_entitled_dividend_receivable() -> None:
    prices = prepare_etf_prices(
        _prices(),
        expected_symbol="510300.SH",
        start_date="2021-01-04",
        observation_cutoff="2021-02-28",
    )
    dividends = prepare_cash_dividends(
        _dividends(), expected_symbol="510300.SH"
    )
    panel = build_bad10_origin_panel(
        prices,
        dividends,
        horizon_market_days=10,
        loss_threshold=-0.04,
    )
    first = panel.iloc[0]
    assert first["entry_date"] == pd.Timestamp("2021-01-05")
    assert first["first_breach_date"] == pd.Timestamp("2021-01-06")
    assert np.isclose(first["minimum_path_return"], -0.04)
    assert first["bad10"] == 1
    assert np.isclose(first["cash_dividend_per_share_in_horizon"], 0.5)

    changed_origin_close = prices.copy()
    changed_origin_close.loc[0, "close"] = 1000.0
    changed = build_bad10_origin_panel(
        changed_origin_close,
        dividends,
        horizon_market_days=10,
        loss_threshold=-0.04,
    )
    pd.testing.assert_series_equal(panel.iloc[0], changed.iloc[0])


def test_future_append_cannot_rewrite_already_complete_bad10_origins() -> None:
    full_prices = prepare_etf_prices(
        _prices(25),
        expected_symbol="510300.SH",
        start_date="2021-01-04",
        observation_cutoff="2021-03-31",
    )
    dividends = prepare_cash_dividends(
        _dividends(), expected_symbol="510300.SH"
    )
    short_panel = build_bad10_origin_panel(
        full_prices.iloc[:20].copy(),
        dividends,
        horizon_market_days=10,
        loss_threshold=-0.04,
    )
    full_panel = build_bad10_origin_panel(
        full_prices,
        dividends,
        horizon_market_days=10,
        loss_threshold=-0.04,
    )
    pd.testing.assert_frame_equal(
        short_panel,
        full_panel.loc[
            full_panel["origin_date"].isin(short_panel["origin_date"])
        ].reset_index(drop=True),
    )


def _origin_row(
    origin: str,
    entry: str,
    end: str,
    bad10: int,
    minimum: float,
) -> dict[str, object]:
    return {
        "origin_date": pd.Timestamp(origin),
        "entry_date": pd.Timestamp(entry),
        "horizon_end_date": pd.Timestamp(end),
        "bad10": bad10,
        "minimum_path_return": minimum,
        "first_breach_date": pd.Timestamp(entry) if bad10 else pd.NaT,
        "cash_dividend_per_share_in_horizon": 0.0,
    }


def test_overlapping_bad10_windows_merge_by_transitive_closure() -> None:
    panel = pd.DataFrame(
        [
            _origin_row("2021-01-01", "2021-01-04", "2021-01-15", 1, -0.05),
            _origin_row("2021-01-08", "2021-01-11", "2021-01-22", 1, -0.06),
            _origin_row("2021-01-20", "2021-01-21", "2021-02-03", 1, -0.07),
            _origin_row("2021-02-10", "2021-02-11", "2021-02-24", 1, -0.08),
        ]
    )
    events = merge_bad10_events(panel)
    assert list(events["event_id"]) == ["BAD10_EVENT_0001", "BAD10_EVENT_0002"]
    assert list(events["bad10_origin_count"]) == [3, 1]
    assert events.iloc[0]["event_horizon_end"] == pd.Timestamp("2021-02-03")
    assert np.isclose(events.iloc[0]["worst_minimum_path_return"], -0.07)


def test_non_event_blocks_are_greedy_nonoverlapping_and_avoid_events() -> None:
    panel = pd.DataFrame(
        [
            _origin_row("2021-01-01", "2021-01-04", "2021-01-15", 0, -0.01),
            _origin_row("2021-01-08", "2021-01-11", "2021-01-22", 0, -0.02),
            _origin_row("2021-01-25", "2021-01-26", "2021-02-08", 1, -0.05),
            _origin_row("2021-02-09", "2021-02-10", "2021-02-23", 0, -0.01),
            _origin_row("2021-02-24", "2021-02-25", "2021-03-10", 0, -0.02),
        ]
    )
    events = merge_bad10_events(panel)
    blocks = select_independent_non_event_blocks(panel, events)
    assert list(blocks["origin_date"]) == [
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-02-09"),
        pd.Timestamp("2021-02-24"),
    ]
    assert (blocks["entry_date"].iloc[1:].to_numpy() > blocks["horizon_end_date"].iloc[:-1].to_numpy()).all()


def test_g1_requires_both_event_and_non_event_counts_without_rescue() -> None:
    failed = adjudicate_g1(
        independent_event_count=29,
        independent_non_event_block_count=500,
        minimum_independent_events=30,
        minimum_independent_non_event_blocks=120,
    )
    assert failed["passed"] is False
    assert failed["status"] == (
        "REJECTED_FROZEN_G1_LABEL_IDENTIFIABILITY_FAILED_NO_RESCUE"
    )
    assert failed["rescue_allowed"] is False
    passed = adjudicate_g1(
        independent_event_count=30,
        independent_non_event_block_count=120,
        minimum_independent_events=30,
        minimum_independent_non_event_blocks=120,
    )
    assert passed["passed"] is True


def test_census_summary_contains_no_portfolio_result() -> None:
    prices = prepare_etf_prices(
        _prices(),
        expected_symbol="510300.SH",
        start_date="2021-01-04",
        observation_cutoff="2021-02-28",
    )
    dividends = prepare_cash_dividends(
        _dividends(), expected_symbol="510300.SH"
    )
    panel = build_bad10_origin_panel(
        prices,
        dividends,
        horizon_market_days=10,
        loss_threshold=-0.04,
    )
    events = merge_bad10_events(panel)
    blocks = select_independent_non_event_blocks(panel, events)
    gate = adjudicate_g1(
        independent_event_count=len(events),
        independent_non_event_block_count=len(blocks),
        minimum_independent_events=30,
        minimum_independent_non_event_blocks=120,
    )
    summary = build_census_summary(
        panel,
        events,
        blocks,
        horizon_market_days=10,
        loss_threshold=-0.04,
        gate=gate,
    )
    serialized = str(summary).casefold()
    for forbidden in ("sharpe", "annual_return", "position_target", "order"):
        assert forbidden not in serialized


def test_protocol_locks_label_dr007_gates_and_no_trade_authority() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    label = config["bad10_label"]
    assert label["horizon_market_days"] == 10
    assert label["loss_threshold"] == -0.04
    assert label["execution_clock"] == "NEXT_TRADING_DAY_OPEN"
    assert label["index_proxy_allowed"] is False
    funding = config["feature_charter"]["macro_vulnerability_M"]["channels"][
        "M2_FUNDING_STRESS"
    ]
    assert funding["selected_rate"] == "DR007"
    assert funding["alternative_rates_forbidden"] == ["FDR007", "R007"]
    g1 = config["gates"]["G1_LABEL_IDENTIFIABILITY"]
    assert g1["minimum_independent_bad10_events"] == 30
    assert g1["minimum_independent_non_event_10d_blocks"] == 120
    assert config["program"]["feature_construction_allowed_before_g1"] is False
    assert config["program"]["portfolio_evaluation_allowed_before_g3"] is False

    authority = read_json_strict(AUTHORITY)
    validate_authority_state(authority)
    assert authority["model_position_target"] == "UNSET"
    assert authority["order_authorization"] == "NOT_AUTHORIZED"
    assert authority["position_impact"] == 0
