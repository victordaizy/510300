from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from backtest.engine import (  # noqa: E402
    BacktestCosts,
    run_long_cash_backtest,
    summarize_backtest,
)
from oracle_information_budget_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    STATE_BEAR,
    STATE_BULL,
    STATE_RANGE,
    build_timing_signal,
    load_config,
    prepare_engine_inputs,
    sample_exact_predictions,
    simulate_dense_signals,
    validate_manifest,
)


def test_config_freezes_asymmetric_up20_budget() -> None:
    config = load_config()
    assert config["research_question"]["prediction_target"] == "UP20"
    assert config["source_oracle"]["expected_state_counts"] == {
        STATE_BULL: 23,
        STATE_BEAR: 21,
        STATE_RANGE: 97,
    }
    assert config["random_error_budget"]["repetitions_per_cell"] == 500
    assert config["random_error_budget"]["joint_grid"]["false_bear_counts"] == [
        0,
        1,
        2,
        4,
        8,
        21,
    ]
    assert config["robustness_annex"]["block_phase_offsets"] == list(range(20))


def test_exact_count_sampling_preserves_each_state_count() -> None:
    predictions = sample_exact_predictions(
        block_count=12,
        bull_positions=np.array([0, 1, 2]),
        range_positions=np.array([3, 4, 5, 6, 7]),
        bear_positions=np.array([8, 9, 10, 11]),
        captured_bulls=2,
        false_ranges=3,
        false_bears=1,
        repetitions=50,
        seed_sequence=np.random.SeedSequence([1234]),
    )
    assert predictions.shape == (50, 12)
    assert np.all(predictions[:, [0, 1, 2]].sum(axis=1) == 2)
    assert np.all(predictions[:, [3, 4, 5, 6, 7]].sum(axis=1) == 3)
    assert np.all(predictions[:, [8, 9, 10, 11]].sum(axis=1) == 1)


def test_batch_engine_matches_reference_engine_with_sparse_signals_and_dividend() -> None:
    dates = pd.bdate_range("2020-01-02", periods=12)
    prices = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(4.00, 4.44, len(dates)),
            "high": np.linspace(4.04, 4.48, len(dates)),
            "low": np.linspace(3.96, 4.40, len(dates)),
            "close": np.linspace(4.02, 4.46, len(dates)),
        }
    )
    dividends = pd.DataFrame(
        {
            "ex_date": [dates[4]],
            "payment_date": [dates[7]],
            "cash_dividend_per_share": [0.05],
        }
    )
    signal_positions = [0, 5, 7, 9]
    signal_values = [1, 0, 1, 0]
    targets = pd.DataFrame(
        {
            "date": dates[signal_positions],
            "target_position": signal_values,
            "trade_allowed": True,
            "risk_off_override": [False, True, False, True],
        }
    )
    costs = BacktestCosts(
        commission_rate=0.0003,
        minimum_commission_cny=5.0,
        stamp_duty_rate=0.0,
        slippage_bps=10.0,
        lot_size=100,
        cash_annual_rate=0.0,
    )
    ledger, trades = run_long_cash_backtest(
        prices,
        dividends,
        targets,
        20000.0,
        costs,
        dates[0],
        dates[-1],
    )
    reference = summarize_backtest(ledger, trades, 20000.0)
    dense = np.full((1, len(dates)), -1, dtype=np.int8)
    dense[0, signal_positions] = signal_values
    engine = prepare_engine_inputs(prices, dividends, dates[0], dates[-1])
    fast = simulate_dense_signals(dense, engine, 20000.0, costs).iloc[0]
    assert math.isclose(
        float(fast["stress_total_return"]),
        float(reference["total_return"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(fast["stress_net_sharpe"]),
        float(reference["sharpe_zero_cash_rate"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(fast["stress_max_drawdown"]),
        float(reference["max_drawdown"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert int(fast["trade_count"]) == int(reference["trade_count"])


def test_timing_signal_delays_only_event_entry() -> None:
    class DummyContext:
        pass

    context = DummyContext()
    context.engine = type("Engine", (), {"dates": pd.bdate_range("2020-01-01", periods=50)})()
    context.anchor_positions = np.array([0, 20, 40])
    context.states = np.array([STATE_BULL, STATE_RANGE, "CENSORED"])
    events = [
        {
            "start_anchor_position": 0,
            "transition_anchor_position": 20,
        }
    ]
    delayed = build_timing_signal(context, events, "ENTRY_DELAY", 3)
    assert delayed[0] == 0
    assert delayed[3] == 1
    assert delayed[20] == 0


def test_governance_never_promotes_budget_to_strategy() -> None:
    config = load_config()
    assert config["governance"]["prediction_model_trained"] is False
    assert config["governance"]["historical_strategy_target_can_pass"] is False
    assert config["governance"]["verified_forward_target_can_pass"] is False
    for key in (
        "paper_signal_allowed",
        "shadow_signal_allowed",
        "position_mapping_enabled",
        "order_generation",
        "broker_connection",
        "position_change",
        "live_trading_authorized",
    ):
        assert config["governance"][key] is False


def test_manifest_integrity_when_frozen() -> None:
    if not MANIFEST_PATH.exists():
        pytest.skip("信息预算首次冻结前尚无清单")
    manifest = validate_manifest(load_config(CONFIG_PATH))
    assert manifest["state"] == "FROZEN_BEFORE_INFORMATION_BUDGET_CALCULATION"
    assert manifest["source_oracle_outcomes_already_known"] is True
    assert manifest["information_budget_outputs_computed_before_freeze"] is False


def test_result_contract_when_written() -> None:
    result_path = ROOT / "reports" / "research" / "510300_oracle_information_budget_v1_result.json"
    if not result_path.exists():
        pytest.skip("信息预算正式结果尚未写入")
    import json

    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["source_oracle_boundary"]["source_result_unchanged"] is True
    assert report["source_oracle_boundary"]["prediction_model_evaluated"] is False
    assert report["adjudication"]["realistic_up20_forecast"] == "NOT_EVALUATED"
    assert report["adjudication"]["historical_tradable_strategy_target_achieved"] is False
    assert report["adjudication"]["goal_achieved"] is False
