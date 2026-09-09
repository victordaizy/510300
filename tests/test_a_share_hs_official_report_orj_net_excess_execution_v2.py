"""ORJ V2 历史执行验证的合成数据测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_hs_official_report_orj_net_excess_execution_v2 import (
    apply_same_security_overlap,
    at_price_limit,
    attach_benchmark_returns,
    build_decision_date_metrics,
    calculate_filled_trade_economics,
    classify_independent_execution,
    evaluate_gate,
    load_config,
    price_limit_fraction,
    rounded_limit_price,
    select_high_orj,
)


def test_selection_uses_signal_eligibility_and_never_future_label() -> None:
    config = load_config()
    first_date = pd.Timestamp("2021-04-30")
    second_date = pd.Timestamp("2021-08-31")
    records: list[dict[str, object]] = []
    for index in range(50):
        records.append(
            {
                "ts_code": f"{600000 + index:06d}.SH",
                "event_market_date": first_date,
                "event_market_day_index": 100,
                "orj": float(index),
                "signal_eligible": True,
                "analysis_eligible": index < 40,
            }
        )
    records.append(
        {
            "ts_code": "999999.SH",
            "event_market_date": first_date,
            "event_market_day_index": 100,
            "orj": 10_000.0,
            "signal_eligible": False,
            "analysis_eligible": True,
        }
    )
    for index in range(49):
        records.append(
            {
                "ts_code": f"{300000 + index:06d}.SZ",
                "event_market_date": second_date,
                "event_market_day_index": 180,
                "orj": float(index),
                "signal_eligible": True,
                "analysis_eligible": True,
            }
        )
    selected = select_high_orj(pd.DataFrame.from_records(records), config)
    assert len(selected) == 10
    assert selected["event_market_date"].nunique() == 1
    assert selected["event_date_candidate_count"].eq(50).all()
    assert selected["selected_tail_count"].eq(10).all()
    assert selected["orj"].min() == 40.0
    assert not selected["analysis_eligible"].any()
    assert "999999.SH" not in set(selected["ts_code"])


def test_historical_price_limit_rules_and_round_half_up() -> None:
    config = load_config()
    assert price_limit_fraction("600001.SH", "2021-01-04", "NORMAL", config) == 0.10
    assert price_limit_fraction("600001.SH", "2021-01-04", "ST", config) == 0.05
    assert price_limit_fraction("300001.SZ", "2020-08-21", "NORMAL", config) == 0.10
    assert price_limit_fraction("300001.SZ", "2020-08-24", "ST", config) == 0.20
    assert price_limit_fraction("688001.SH", "2021-01-04", "ST", config) == 0.20
    assert rounded_limit_price(10.05, 0.10, upper=True) == 11.06
    assert rounded_limit_price(10.05, 0.10, upper=False) == 9.05
    assert at_price_limit(11.06, 10.05, 0.10, upper=True)
    assert not at_price_limit(11.05, 10.05, 0.10, upper=True)


def test_one_lot_costs_include_minimum_commission_transfer_stamp_and_slippage() -> None:
    config = load_config()
    base = calculate_filled_trade_economics(
        entry_raw_open=10.0,
        entry_total_return_open=100.0,
        exit_total_return_close=110.0,
        entry_date="2021-01-04",
        exit_date="2021-02-01",
        slippage_fraction_each_side=0.001,
        config=config,
    )
    stress = calculate_filled_trade_economics(
        entry_raw_open=10.0,
        entry_total_return_open=100.0,
        exit_total_return_close=110.0,
        entry_date="2024-01-04",
        exit_date="2024-02-01",
        slippage_fraction_each_side=0.005,
        config=config,
    )
    assert base["buy_commission"] == 5.0
    assert base["sell_commission"] == 5.0
    assert np.isclose(base["buy_transfer_fee"], 1001.0 * 0.00002)
    assert np.isclose(base["sell_stamp_duty"], 1098.9 * 0.001)
    assert np.isclose(stress["buy_transfer_fee"], 1005.0 * 0.00001)
    assert np.isclose(stress["sell_stamp_duty"], 1094.5 * 0.0005)
    assert base["net_return"] > stress["net_return"]
    assert base["total_explicit_and_slippage_cost"] > 0.0


def test_upper_limit_is_cash_and_lower_limit_exit_is_unresolved() -> None:
    config = load_config()
    common = {
        "ts_code": "600001.SH",
        "entry_date": pd.Timestamp("2021-01-04"),
        "exit_date": pd.Timestamp("2021-02-01"),
        "entry_row_present": True,
        "entry_security_status": "NORMAL",
        "entry_is_suspended": False,
        "entry_observed_traded_row": True,
        "entry_pre_close": 10.0,
        "entry_total_return_open": 11.0,
        "exit_row_present": True,
        "exit_security_status": "NORMAL",
        "exit_is_suspended": False,
        "exit_observed_traded_row": True,
        "exit_pre_close": 10.0,
        "exit_total_return_close": 9.0,
    }
    upper = classify_independent_execution(
        {
            **common,
            "entry_raw_open": 11.0,
            "exit_raw_close": 10.0,
        },
        config,
    )
    assert upper["independent_execution_status"] == "NO_FILL_ENTRY_UPPER_LIMIT_CASH"
    assert upper["execution_resolved_before_overlap"] is True
    lower = classify_independent_execution(
        {
            **common,
            "entry_raw_open": 10.5,
            "entry_total_return_open": 10.5,
            "exit_raw_close": 9.0,
        },
        config,
    )
    assert lower["independent_execution_status"] == "FILLED_UNRESOLVED_EXIT_LOWER_LIMIT"
    assert lower["entry_filled"] is True


def test_same_security_overlap_skips_active_and_blocks_after_unresolved() -> None:
    frame = pd.DataFrame(
        {
            "execution_event_id": ["A1", "A2", "A3", "B1", "B2"],
            "ts_code": ["600001.SH"] * 3 + ["000001.SZ"] * 2,
            "entry_market_day_index": [101, 110, 121, 201, 230],
            "exit_market_day_index": [120, 129, 140, 220, 249],
            "independent_execution_status": [
                "FILLED_RESOLVED",
                "FILLED_RESOLVED",
                "NO_FILL_ENTRY_UPPER_LIMIT_CASH",
                "FILLED_UNRESOLVED_EXIT_SUSPENDED",
                "FILLED_RESOLVED",
            ],
        }
    )
    result = apply_same_security_overlap(frame).set_index("execution_event_id")
    assert result.loc["A1", "execution_status"] == "FILLED_RESOLVED"
    assert result.loc["A2", "execution_status"] == "SKIPPED_SAME_SECURITY_ACTIVE_POSITION"
    assert not bool(result.loc["A2", "allocation_included"])
    assert result.loc["A3", "execution_status"] == "NO_FILL_ENTRY_UPPER_LIMIT_CASH"
    assert result.loc["B2", "execution_status"] == (
        "SKIPPED_AFTER_UNRESOLVED_SAME_SECURITY_STATE"
    )


def test_benchmarks_are_aligned_from_entry_open() -> None:
    config = load_config()
    frame = pd.DataFrame(
        {
            "industry_valid_path_rows": [20],
            "industry_minimum_peer_count": [15],
            "industry_log_return_sum_t1_t20": [0.10],
            "industry_peer_log_return_t1_close_to_close": [0.02],
            "industry_peer_log_return_t1_open_to_close": [0.01],
            "industry_intraday_peer_count": [14],
            "h00300_entry_close": [100.0],
            "h00300_exit_close": [110.0],
            "csi300_entry_open": [98.0],
            "csi300_entry_close": [100.0],
        }
    )
    result = attach_benchmark_returns(frame, config).iloc[0]
    assert np.isclose(result["industry_return"], np.expm1(0.09))
    assert np.isclose(result["broad_market_return"], 1.10 * 100.0 / 98.0 - 1.0)
    assert str(result["industry_benchmark_status"]).startswith("PASS_")
    assert str(result["broad_market_benchmark_status"]).startswith("PASS_")


def test_decision_date_weights_one_lot_capital_and_keeps_no_fill_cash() -> None:
    config = load_config()
    frame = pd.DataFrame(
        {
            "event_market_date": [pd.Timestamp("2021-01-04")] * 2,
            "evaluation_period_v2": ["PRIMARY_2021_2025"] * 2,
            "allocation_included": [True, True],
            "execution_resolved": [True, True],
            "execution_status": [
                "FILLED_RESOLVED",
                "NO_FILL_ENTRY_UPPER_LIMIT_CASH",
            ],
            "industry_return": [0.05, 0.05],
            "broad_market_return": [0.05, 0.05],
            "base_committed_capital": [100.0, 100.0],
            "base_net_pnl": [10.0, 0.0],
            "base_net_return": [0.10, 0.0],
            "stress_committed_capital": [100.0, 100.0],
            "stress_net_pnl": [10.0, 0.0],
            "stress_net_return": [0.10, 0.0],
        }
    )
    metrics = build_decision_date_metrics(frame, config)
    assert len(metrics) == 4
    assert metrics["resolution_fraction"].eq(1.0).all()
    assert metrics["portfolio_net_return"].eq(0.05).all()
    assert np.allclose(metrics["net_excess_return"], 0.0)
    assert metrics["no_fill_cash_count"].eq(1).all()


def test_gate_can_pass_alpha_without_win_rate_or_payoff_gate() -> None:
    config = deepcopy(load_config())
    config["inference"]["bootstrap_repetitions"] = 400
    config["inference"]["bootstrap_block_length_dates"] = 5
    rows: list[dict[str, object]] = []
    for year in config["periods"]["primary_complete_years"]:
        for date in pd.bdate_range(f"{year}-01-04", periods=20):
            for scenario in ("base", "stress"):
                for benchmark in ("industry", "broad_market"):
                    excess = 0.01 if benchmark == "industry" else -0.01
                    rows.append(
                        {
                            "event_market_date": date,
                            "year": year,
                            "evaluation_period_v2": "PRIMARY_2021_2025",
                            "cost_scenario": scenario,
                            "benchmark": benchmark,
                            "allocation_count": 10,
                            "resolved_allocation_count": 10,
                            "decision_date_eligible": True,
                            "net_excess_return": excess,
                        }
                    )
    gate = evaluate_gate(pd.DataFrame.from_records(rows), config)
    assert gate["historical_status"] == "HISTORICAL_ALPHA_CANDIDATE_NOT_BLIND"
    assert gate["historical_alpha_candidate"] is True
    assert gate["historical_strong_beta_candidate"] is False
    assert "win_rate" in gate["not_acceptance_gates"]
    assert gate["stress_industry_path"]["bootstrap"]["ci_lower"] > 0.0
