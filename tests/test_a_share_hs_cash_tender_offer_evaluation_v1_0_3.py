from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from research.a_share_hs_cash_tender_offer_evaluation_v1_0_3 import (
    _corporate_action_proceeds,
    build_event_clock,
    evaluate_event_ledger,
    odds_metrics,
    one_price_upward_lock,
    validate_static_config,
    wilson_lower_bound,
)
from scripts.run_a_share_hs_cash_tender_offer_evaluation_v1_0_3 import (
    extract_market_inputs,
    validate_h00300_completion,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "config/a_share_hs_official_unconditional_full_cash_tender_spread_v1_"
    "evaluation_v1_0_3.json"
)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_static_contract_keeps_high_odds_rules_and_no_liquidity_filter() -> None:
    result = validate_static_config(load_config())
    assert result["event_count"] == 34
    assert result["spread_threshold"] == 0.08
    assert result["stress_total_cost_bps"] == 200
    assert result["tender_event_market_price_read"] is False


def test_real_frozen_dates_build_exact_34_event_clock_without_market_prices() -> None:
    config = load_config()
    inputs = config["inputs"]
    clock = build_event_clock(
        pd.read_parquet(ROOT / inputs["final_event_outcomes"]),
        pd.read_parquet(ROOT / inputs["final_corporate_actions"]),
        pd.read_parquet(ROOT / inputs["lifecycle_review_archive"]),
        pd.read_parquet(ROOT / inputs["trading_calendar"]),
        pd.read_parquet(ROOT / inputs["security_master"]),
        pd.read_parquet(ROOT / inputs["security_status_intervals"]),
        config,
    )
    assert len(clock) == 34
    assert clock["event_id"].nunique() == 34
    assert clock["clock_status"].eq(
        "READY_FOR_FROZEN_MARKET_EVALUATION"
    ).all()
    jichuan = clock.loc[
        clock["event_id"].eq("600566.SH_2025-06-18_1223880087")
    ].iloc[0]
    assert jichuan["observed_offer_price_cny"] == 26.93
    assert jichuan["final_effective_offer_price_cny"] == 24.85


def test_one_price_lock_requires_same_ohlc_and_threshold() -> None:
    locked = {
        "raw_open": 10.0,
        "raw_high": 10.0,
        "raw_low": 10.0,
        "raw_close": 10.0,
        "pct_chg": 5.0,
    }
    assert one_price_upward_lock(
        locked, relative_tolerance=1e-8, pct_threshold=4.8
    )
    unlocked = {**locked, "raw_low": 9.99}
    assert not one_price_upward_lock(
        unlocked, relative_tolerance=1e-8, pct_threshold=4.8
    )


def synthetic_clock() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "TEST_EVENT",
                "ts_code": "000001.SZ",
                "formal_report_pdf_date": "2020-01-01",
                "formal_announcement_id": "FORMAL",
                "reported_offer_price_cny": 10.8,
                "offer_end_date": "2020-01-20",
                "final_effective_offer_price_cny": 10.8,
                "outcome_status": "SUCCESS_SETTLEMENT_CONFIRMED",
                "result_announcement_id": "RESULT",
                "result_pdf_date": "2020-01-20",
                "exchange": "SZSE",
                "report_market_date": "2020-01-02",
                "entry_date": "2020-01-03",
                "cash_availability_date": "2020-01-06",
                "observed_offer_price_cny": 10.8,
                "entry_limit_price_cny": 10.0,
                "open_dates_strictly_after_entry_through_offer_end": 10,
                "benchmark_after_entry_open_date_count": 1,
                "benchmark_after_entry_dates_json": '["2020-01-06"]',
                "entry_security_active": True,
                "entry_security_type": "ORDINARY_A_SHARE",
                "entry_security_status": "L",
                "clock_status": "READY_FOR_FROZEN_MARKET_EVALUATION",
            }
        ]
    )


def synthetic_market(entry_open: float = 9.8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "TEST_EVENT",
                "market_role": "REPORT",
                "raw_open": 10.0,
                "raw_high": 10.1,
                "raw_low": 9.9,
                "raw_close": 10.0,
                "pct_chg": 0.0,
                "amount": 1.0,
                "is_suspended": False,
                "observed_traded_row": True,
            },
            {
                "event_id": "TEST_EVENT",
                "market_role": "ENTRY",
                "raw_open": entry_open,
                "raw_high": entry_open + 0.1,
                "raw_low": entry_open - 0.1,
                "raw_close": entry_open + 0.05,
                "pct_chg": 0.0,
                "amount": 1.0,
                "is_suspended": False,
                "observed_traded_row": True,
            },
        ]
    )


def test_exact_eight_percent_signal_fills_only_at_frozen_next_open() -> None:
    ledger = evaluate_event_ledger(
        synthetic_clock(),
        synthetic_market(),
        pd.DataFrame(
            [
                {
                    "event_id": "TEST_EVENT",
                    "industry_l1_code": "I1",
                    "entry_intraday_peer_count": 10,
                    "entry_intraday_peer_log_return": 0.0,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "event_id": "TEST_EVENT",
                    "date": "2020-01-06",
                    "industry_peer_count": 10,
                    "industry_peer_log_return": 0.0,
                }
            ]
        ),
        pd.DataFrame(
            [
                {"date": "2020-01-03", "open": 100.0, "close": 100.0},
                {"date": "2020-01-06", "open": 100.0, "close": 100.0},
            ]
        ),
        pd.DataFrame(
            columns=[
                "event_id",
                "announcement_id",
                "record_date",
                "gross_cash_per_share_cny",
            ]
        ),
        load_config(),
    )
    row = ledger.iloc[0]
    assert row["execution_status"] == "EXECUTED_RESOLVED_ALL_FROZEN_TARGETS"
    assert abs(row["gross_cash_spread_at_observation"] - 0.08) < 1e-12
    assert bool(row["entry_fill"])
    assert row["stress_stock_growth"] > 1.0
    assert row["primary_stress_net_industry_excess_return"] > 0.0


def test_open_above_limit_is_skipped_without_later_refill() -> None:
    ledger = evaluate_event_ledger(
        synthetic_clock(),
        synthetic_market(entry_open=10.01),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(
            [
                {"date": "2020-01-03", "open": 100.0, "close": 100.0},
                {"date": "2020-01-06", "open": 100.0, "close": 100.0},
            ]
        ),
        pd.DataFrame(),
        load_config(),
    )
    assert ledger.iloc[0]["execution_status"] == (
        "SKIPPED_ENTRY_OPEN_ABOVE_FROZEN_LIMIT"
    )


def test_jichuan_cash_distribution_and_adjusted_offer_price_are_both_counted() -> None:
    event = pd.Series(
        {
            "entry_date": "2025-06-20",
            "offer_end_date": "2025-07-17",
            "result_pdf_date": "2025-07-22",
        }
    )
    actions = pd.DataFrame(
        [
            {
                "announcement_id": "1223956106",
                "record_date": "2025-06-27",
                "gross_cash_per_share_cny": 2.09,
            }
        ]
    )
    result = _corporate_action_proceeds(event, actions, load_config())
    assert result["corporate_action_accounting_resolved"]
    assert result["entitled_gross_cash_distribution_cny"] == 2.09
    assert 24.85 + result["entitled_gross_cash_distribution_cny"] == 26.94


def test_odds_metrics_and_wilson_are_not_replaced_by_raw_win_rate() -> None:
    values = pd.Series([0.10] * 6 + [-0.02] * 2).to_numpy()
    metrics = odds_metrics(values)
    assert metrics["win_rate"] == 0.75
    assert abs(metrics["payoff_ratio"] - 5.0) < 1e-12
    assert abs(metrics["profit_factor"] - 15.0) < 1e-12
    assert wilson_lower_bound(6, 8) < 0.50


def test_duckdb_extraction_builds_subject_excluded_entry_open_bridge(
    tmp_path: Path,
) -> None:
    market_path = tmp_path / "market.parquet"
    peer_path = tmp_path / "peer.parquet"
    market_rows = []
    for code, date, tr_open, tr_close in (
        ("000001.SZ", "2020-01-02", 9.9, 10.0),
        ("000001.SZ", "2020-01-03", 10.0, 10.1),
        ("000002.SZ", "2020-01-03", 100.0, 101.0),
        ("000003.SZ", "2020-01-03", 100.0, 99.0),
    ):
        market_rows.append(
            {
                "con_code": code,
                "date": date,
                "raw_open": tr_open,
                "raw_high": max(tr_open, tr_close),
                "raw_low": min(tr_open, tr_close),
                "raw_close": tr_close,
                "pct_chg": 0.0,
                "amount": 1.0,
                "total_return_open": tr_open,
                "total_return_close": tr_close,
                "is_suspended": False,
                "observed_traded_row": True,
            }
        )
    pd.DataFrame(market_rows).to_parquet(market_path, index=False)
    pd.DataFrame(
        [
            {
                "con_code": "000001.SZ",
                "date": "2020-01-03",
                "industry_l1_code": "I1",
                "industry_peer_count": 2,
                "industry_peer_log_return": 0.0,
            },
            {
                "con_code": "000002.SZ",
                "date": "2020-01-03",
                "industry_l1_code": "I1",
                "industry_peer_count": 2,
                "industry_peer_log_return": 0.0,
            },
            {
                "con_code": "000003.SZ",
                "date": "2020-01-03",
                "industry_l1_code": "I1",
                "industry_peer_count": 2,
                "industry_peer_log_return": 0.0,
            },
            {
                "con_code": "000001.SZ",
                "date": "2020-01-06",
                "industry_l1_code": "I1",
                "industry_peer_count": 2,
                "industry_peer_log_return": 0.001,
            },
        ]
    ).to_parquet(peer_path, index=False)
    config = load_config()
    config["inputs"]["unified_daily_market"] = str(market_path)
    config["inputs"]["industry_peer_return_panel"] = str(peer_path)
    clock = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "ts_code": "000001.SZ",
                "report_market_date": "2020-01-02",
                "entry_date": "2020-01-03",
                "cash_availability_date": "2020-01-06",
            }
        ]
    )
    event_rows, entry_stats, daily_rows, receipt = extract_market_inputs(
        config, clock
    )
    assert len(event_rows) == 2
    assert entry_stats.iloc[0]["entry_intraday_peer_count"] == 2
    expected = (math.log(1.01) + math.log(0.99)) / 2.0
    assert abs(entry_stats.iloc[0]["entry_intraday_peer_log_return"] - expected) < 1e-12
    assert len(daily_rows) == 1
    assert receipt["market_rows"] == 4


def test_h00300_open_bridge_preserves_total_return_close_and_uses_intraday_ratio(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "h00300_close.parquet"
    price_path = tmp_path / "000300_price.parquet"
    pd.DataFrame(
        [
            {"date": "2020-01-02", "close": 100.0},
            {"date": "2020-01-03", "close": 110.0},
        ]
    ).to_parquet(reference_path, index=False)
    pd.DataFrame(
        [
            {"date": "2020-01-02", "open": 10.0, "close": 10.5},
            {"date": "2020-01-03", "open": 11.0, "close": 10.0},
        ]
    ).to_parquet(price_path, index=False)
    config = load_config()
    config["inputs"]["h00300_close_reference"] = str(reference_path)
    config["inputs"]["csi300_price_index_ohlc_bridge"] = str(price_path)
    bridged, receipt = validate_h00300_completion(config)
    assert (
        receipt["status"]
        == "PASS_H00300_EXCHANGE_OPEN_DATE_PRICE_INDEX_BRIDGE_VALIDATED"
    )
    assert bridged["close"].tolist() == [100.0, 110.0]
    assert abs(bridged.iloc[0]["open"] - 100.0 * 10.0 / 10.5) < 1e-12
    assert abs(bridged.iloc[1]["open"] - 121.0) < 1e-12


def test_h00300_open_bridge_excludes_non_open_reference_without_imputation(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "h00300_close.parquet"
    price_path = tmp_path / "000300_price.parquet"
    calendar_path = tmp_path / "calendar.parquet"
    pd.DataFrame(
        [
            {"date": "2020-01-03", "close": 100.0},
            {"date": "2020-01-04", "close": 100.5},
            {"date": "2020-01-06", "close": 101.0},
        ]
    ).to_parquet(reference_path, index=False)
    pd.DataFrame(
        [
            {"date": "2020-01-03", "open": 10.0, "close": 10.5},
            {"date": "2020-01-06", "open": 10.5, "close": 10.4},
        ]
    ).to_parquet(price_path, index=False)
    pd.DataFrame(
        [
            {"date": "2020-01-03", "is_open": True},
            {"date": "2020-01-04", "is_open": False},
            {"date": "2020-01-06", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)
    event_clock = pd.DataFrame(
        [
            {
                "entry_date": "2020-01-03",
                "cash_availability_date": "2020-01-06",
            }
        ]
    )
    config = load_config()
    config["inputs"]["h00300_close_reference"] = str(reference_path)
    config["inputs"]["csi300_price_index_ohlc_bridge"] = str(price_path)
    config["inputs"]["trading_calendar"] = str(calendar_path)
    bridged, receipt = validate_h00300_completion(config, event_clock)
    assert bridged["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-03",
        "2020-01-06",
    ]
    assert receipt["excluded_non_open_reference_rows"] == 1
    assert receipt["excluded_non_open_reference_dates"] == ["2020-01-04"]
    assert receipt["required_event_date_count"] == 2
    assert receipt["missing_required_event_dates"] == []
