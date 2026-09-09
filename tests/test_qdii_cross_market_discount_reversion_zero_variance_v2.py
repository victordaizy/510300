"""QDII跨市场折价回归零方差评估修正V2测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.qdii_cross_market_discount_reversion_v1 import (
    build_discount_signals,
    load_contract as load_v1_contract,
    load_research_inputs,
    run_portfolio_backtest,
)
from research.qdii_cross_market_discount_reversion_zero_variance_v2 import (
    CONFIG,
    _paired_block_bootstrap_conservative,
    evaluate_historical_returns_zero_variance_v2,
    load_correction_contract,
    safe_annualized_sharpe,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_correction_contract_keeps_current_hard_requirements() -> None:
    contract = load_correction_contract(CONFIG)
    assert contract["objective"] == {
        "initial_capital_cny": 500000.0,
        "user_transaction_fee_rate_per_leg": 0.0001,
        "benchmark": "H00300_TOTAL_RETURN",
        "minimum_annualized_net_excess": 0.40,
        "minimum_strategy_net_sharpe": 1.50,
        "base_and_stress_must_both_pass": True,
    }
    assert not contract["protocol"][
        "economic_signal_execution_or_cost_parameter_changed"
    ]
    assert not contract["protocol"]["versioned_correction_is_new_holdout"]
    assert contract["evaluation_correction"]["zero_variance_sharpe_value"] is None


def test_v1_failure_artifacts_remain_byte_identical() -> None:
    contract = load_correction_contract(CONFIG)
    source = contract["source_freeze"]
    manifest = ROOT / source["v1_manifest"]
    report_path = ROOT / source["v1_visible_failure_report_json"]
    markdown = ROOT / source["v1_visible_failure_report_markdown"]
    assert _sha256(manifest) == source["expected_v1_manifest_sha256"]
    assert _sha256(report_path) == source["expected_v1_visible_failure_report_sha256"]
    assert _sha256(markdown) == source["expected_v1_visible_failure_markdown_sha256"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == source["expected_v1_failure_status"]
    assert report["error_type"] == source["expected_v1_error_type"]
    assert report["error"] == source["expected_v1_error"]
    assert report["performance_metrics_available"] is False
    assert report["decision"]["sealed_replication_open"] is False


def test_safe_sharpe_returns_null_for_zero_variance() -> None:
    daily_cash = (1.0 + 0.015) ** (1.0 / 242) - 1.0
    result = safe_annualized_sharpe(
        pd.Series(np.full(242, daily_cash)),
        trading_days_per_year=242,
        cash_annual_rate=0.015,
        zero_variance_epsilon=1e-15,
    )
    assert result["value"] is None
    assert result["status"] == "UNDEFINED_ZERO_VARIANCE"
    json.dumps(result, allow_nan=False)


def test_safe_sharpe_remains_defined_for_variable_returns() -> None:
    result = safe_annualized_sharpe(
        pd.Series([0.01, -0.005, 0.003, 0.002]),
        trading_days_per_year=242,
        cash_annual_rate=0.015,
        zero_variance_epsilon=1e-15,
    )
    assert result["value"] is not None
    assert result["status"] == "DEFINED"
    assert np.isfinite(float(result["value"]))


def test_bootstrap_records_mixed_zero_variance_samples_without_raising() -> None:
    daily_cash = (1.0 + 0.015) ** (1.0 / 242) - 1.0
    strategy = np.full(40, daily_cash)
    strategy[20] += 0.01
    result = _paired_block_bootstrap_conservative(
        pd.Series(strategy),
        pd.Series(np.zeros(40)),
        repetitions=100,
        block_length=10,
        trading_days_per_year=242,
        cash_annual_rate=0.015,
        zero_variance_epsilon=1e-15,
        random_seed=20260827,
        lower_quantile=0.025,
        upper_quantile=0.975,
    )
    undefined = result["strategy_sharpe_undefined_zero_variance_sample_count"]
    assert 0 < undefined < 100
    assert result["strategy_sharpe_status"] == "PARTIAL_UNDEFINED_ZERO_VARIANCE"
    assert result["strategy_sharpe_all_samples_defined"] is False
    json.dumps(result, allow_nan=False)


def test_full_evaluator_conservatively_rejects_cash_only_path() -> None:
    v1_contract = load_v1_contract(
        ROOT / "config" / "qdii_cross_market_discount_reversion_v1.yaml"
    )
    correction = load_correction_contract(CONFIG)
    dates = pd.bdate_range("2020-01-02", periods=484)
    daily_cash = (1.0 + 0.015) ** (1.0 / 242) - 1.0
    daily = pd.DataFrame(
        {
            "trade_date": dates,
            "strategy_base_net_return": daily_cash,
            "strategy_stress_net_return": daily_cash,
            "benchmark_total_return": 0.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "quality_complete": True,
            "capacity_pass": True,
        }
    )
    result = evaluate_historical_returns_zero_variance_v2(
        daily,
        v1_contract,
        correction,
        bootstrap_repetitions_override=25,
    )
    assert result["metrics"]["base_strategy_net_sharpe"] is None
    assert result["metrics"]["base_strategy_net_sharpe_status"] == (
        "UNDEFINED_ZERO_VARIANCE"
    )
    assert not result["gates"]["base_strategy_sharpe_defined"]
    assert not result["gates"]["base_rolling_sharpe_all_windows_defined"]
    assert not result["gates"]["base_bootstrap_sharpe_all_samples_defined"]
    assert result["all_visible_gates_pass"] is False
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_real_v1_path_completes_v2_evaluation_without_changing_execution() -> None:
    v1_contract = load_v1_contract(
        ROOT / "config" / "qdii_cross_market_discount_reversion_v1.yaml"
    )
    correction = load_correction_contract(CONFIG)
    partition = v1_contract["historical_partition"]
    start = pd.Timestamp(partition["visible_start"])
    end = pd.Timestamp(partition["visible_end"])
    panel, master, benchmark, fx, indices = load_research_inputs(v1_contract)
    signals, market, return_wide, _ = build_discount_signals(
        panel,
        master,
        benchmark,
        fx,
        indices,
        v1_contract,
        start=start,
        end=end,
    )
    daily, trades, portfolio_audit = run_portfolio_backtest(
        signals,
        market,
        benchmark,
        return_wide,
        v1_contract,
        start=start,
        end=end,
    )
    result = evaluate_historical_returns_zero_variance_v2(
        daily,
        v1_contract,
        correction,
        bootstrap_repetitions_override=20,
    )
    assert portfolio_audit["entry_transaction_count"] == 16
    assert portfolio_audit["exit_transaction_count"] == 16
    assert len(trades) == 32
    assert result["metrics"]["undefined_zero_variance_year_block_count"] >= 1
    assert result["evaluation_semantics"] == (
        "VERSIONED_EVALUATION_CORRECTION_NOT_NEW_HOLDOUT"
    )
    json.dumps(result, ensure_ascii=False, allow_nan=False)
