from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from research.a_share_hs_cash_option_floor_alpha_v1 import (
    CashOptionFloorProtocolError,
    ENTRY_FOUND,
    NO_ENTRY,
    NO_VIEW_MARKET,
    NOT_QUALIFIED,
    QUALIFIED,
    best_case_csi300_return,
    evaluate_study,
    failed_or_withdrawn_outcome_metrics,
    screen_floor_metrics,
    select_entry_row,
    successful_cash_outcome_metrics,
    validate_protocol_config,
    wilson_lower_bound,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/a_share_hs_official_cash_option_floor_alpha_v1.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def event() -> dict:
    return {
        "event_id": "E1",
        "ts_code": "600001.SH",
        "effective_announcement_date": "2025-01-02",
        "rights_registration_date": "2025-01-06",
        "application_end_date": "2025-01-08",
        "cash_option_price": 12.0,
    }


def test_frozen_config_passes_exact_validation() -> None:
    validate_protocol_config(config())


def test_price_threshold_cannot_drift() -> None:
    mutated = copy.deepcopy(config())
    mutated["screening"]["minimum_net_conditional_floor_return"] = 0.079
    with pytest.raises(CashOptionFloorProtocolError, match="minimum_net_conditional_floor_return"):
        validate_protocol_config(mutated)


def test_entry_skips_one_price_up_and_uses_next_day_high() -> None:
    daily = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "date": "2025-01-03",
                "pre_close": 10.0,
                "raw_open": 11.0,
                "raw_high": 11.0,
                "raw_low": 11.0,
                "raw_close": 11.0,
                "volume": 100.0,
                "price_source": "测试源",
            },
            {
                "ts_code": "600001.SH",
                "date": "2025-01-06",
                "pre_close": 11.0,
                "raw_open": 10.6,
                "raw_high": 10.8,
                "raw_low": 10.4,
                "raw_close": 10.7,
                "volume": 1.0,
                "price_source": "测试源",
            },
        ]
    )
    selected = select_entry_row(event(), daily, config())
    assert selected["entry_status"] == ENTRY_FOUND
    assert selected["entry_date"] == "2025-01-06"
    assert selected["entry_high"] == pytest.approx(10.8)
    assert selected["one_price_up_days_skipped"] == 1


def test_positive_volume_invalid_ohlc_is_no_view_not_silently_skipped() -> None:
    daily = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "date": "2025-01-03",
                "pre_close": 10.0,
                "raw_open": 10.0,
                "raw_high": 9.8,
                "raw_low": 9.7,
                "raw_close": 10.1,
                "volume": 1.0,
            }
        ]
    )
    selected = select_entry_row(event(), daily, config())
    assert selected["entry_status"] == NO_VIEW_MARKET
    assert selected["reason_code"] == "INVALID_POSITIVE_VOLUME_OHLC_OR_PRE_CLOSE"


def test_empty_window_is_no_replicable_entry_not_zero_return() -> None:
    daily = pd.DataFrame(
        columns=[
            "ts_code",
            "date",
            "pre_close",
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "volume",
        ]
    )
    selected = select_entry_row(event(), daily, config())
    assert selected["entry_status"] == NO_ENTRY


def test_screen_uses_board_lot_minimum_fees_and_funding() -> None:
    metrics = screen_floor_metrics(
        entry_high=10.0,
        cash_option_price=12.0,
        entry_date="2025-01-03",
        application_end_date="2025-01-08",
        config=config(),
    )
    assert metrics["entry_notional"] == pytest.approx(1000.0)
    assert metrics["buy_fee"] == pytest.approx(5.0)
    assert metrics["cash_exercise_and_settlement_fee"] == pytest.approx(12.0)
    assert metrics["funding_days"] == 16
    assert metrics["price_disposition"] == QUALIFIED
    assert metrics["net_conditional_floor_return"] >= 0.08


def test_actual_success_and_failure_paths_use_different_terminal_legs() -> None:
    frozen = config()
    success = successful_cash_outcome_metrics(
        entry_high=10.0,
        entry_date="2025-01-03",
        final_cash_option_price=12.0,
        cash_availability_date="2025-01-20",
        entitled_cash_distribution_per_original_share=0.10,
        config=frozen,
    )
    failure = failed_or_withdrawn_outcome_metrics(
        entry_high=10.0,
        entry_date="2025-01-03",
        exit_open=8.0,
        exit_date="2025-01-20",
        entitled_cash_distribution_per_original_share=0.10,
        config=frozen,
    )
    assert success["outcome_type"] == "SUCCESS_CASH_EXERCISED_AND_SETTLED"
    assert failure["outcome_type"] == "FAILED_WITHDRAWN_OR_TERMINATED_MARKET_EXIT"
    assert success["realized_net_return"] > 0.08
    assert failure["realized_net_return"] < 0.0


def test_benchmark_is_best_case_low_to_high_envelope() -> None:
    assert best_case_csi300_return(100.0, 110.0) == pytest.approx(0.10)


def make_price_screen(qualified_count: int, no_view_count: int = 0) -> pd.DataFrame:
    dates = [
        "2008-01-02",
        "2010-01-02",
        "2015-01-02",
        "2021-01-02",
        "2024-01-02",
    ]
    rows = []
    for index in range(21):
        if index < no_view_count:
            disposition = NO_VIEW_MARKET
        elif index < no_view_count + qualified_count:
            disposition = QUALIFIED
        else:
            disposition = NOT_QUALIFIED
        rows.append(
            {
                "event_id": f"E{index:02d}",
                "price_disposition": disposition,
                "effective_announcement_date": dates[index % len(dates)],
            }
        )
    return pd.DataFrame(rows)


def make_passing_outcomes() -> pd.DataFrame:
    dates = ["2008-01-02", "2010-01-02", "2015-01-02", "2021-01-02", "2024-01-02"]
    return pd.DataFrame(
        [
            {
                "event_id": f"E{index:02d}",
                "effective_announcement_date": announcement_date,
                "terminal_date": pd.Timestamp(announcement_date) + pd.Timedelta(days=20),
                "lifecycle_complete": True,
                "benchmark_complete": True,
                "transaction_completed": True,
                "realized_net_return": 0.20,
                "benchmark_best_case_return": 0.02,
            }
            for index, announcement_date in enumerate(dates)
        ]
    )


def test_any_market_no_view_stops_before_visible_subset_evaluation() -> None:
    result = evaluate_study(make_price_screen(5, no_view_count=1), pd.DataFrame(), config())
    assert result["status"] == "NO_VIEW_INCOMPLETE_CASH_OPTION_PRICE_UNIVERSE"
    assert result["gates_evaluated"] is False


def test_one_to_four_qualified_events_remain_insufficient() -> None:
    result = evaluate_study(make_price_screen(4), pd.DataFrame(), config())
    assert result["status"] == "NO_VIEW_INSUFFICIENT_REPEATABLE_HIGH_FLOOR_EVENTS"
    assert result["gates_evaluated"] is False


def test_five_diversified_all_positive_events_pass_all_frozen_gates() -> None:
    result = evaluate_study(make_price_screen(5), make_passing_outcomes(), config())
    assert result["status"] == (
        "HISTORICAL_HIGH_ODDS_HIGH_PAYOFF_CASH_OPTION_FLOOR_ALPHA_CANDIDATE_PASS_SHADOW_REQUIRED"
    )
    assert result["gates_evaluated"] is True
    assert result["failed_gates"] == []
    assert all(result["gates"].values())
    assert wilson_lower_bound(5, 5) > 0.50
