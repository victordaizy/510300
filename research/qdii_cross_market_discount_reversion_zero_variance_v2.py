"""QDII跨市场折价回归V2：仅修正零方差夏普评估语义。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.multi_asset_annual_excess_40pct_high_sharpe_v1 import (
    annualized_return,
    maximum_drawdown,
)
from research.qdii_cross_market_discount_reversion_v1 import (
    build_discount_signals,
    load_contract as load_v1_contract,
    load_research_inputs,
    run_portfolio_backtest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "qdii_cross_market_discount_reversion_zero_variance_v2.yaml"


def load_correction_contract(path: Path = CONFIG) -> dict[str, Any]:
    """读取并校验评估修正合同。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("QDII零方差评估修正合同必须是YAML对象")
    validate_correction_contract(payload)
    return payload


def validate_correction_contract(contract: dict[str, Any]) -> None:
    """保证V2只改变零方差评估语义，不弱化目标或交易边界。"""

    protocol = contract.get("protocol", {})
    objective = contract.get("objective", {})
    source = contract.get("source_freeze", {})
    correction = contract.get("evaluation_correction", {})
    outputs = contract.get("outputs", {})
    safety = contract.get("safety", {})
    failures: list[str] = []

    expected_text = {
        "candidate_id": (
            protocol,
            "QDII_CROSS_MARKET_DISCOUNT_REVERSION_ZERO_VARIANCE_V2",
        ),
        "parent_candidate_id": (
            protocol,
            "QDII_CROSS_MARKET_DISCOUNT_REVERSION_V1",
        ),
        "parent_protocol_id": (
            protocol,
            "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V2",
        ),
        "lane": (protocol, "CROSS_BORDER_QDII_RELATIVE_VALUE"),
        "correction_class": (
            protocol,
            "EVALUATION_ONLY_ZERO_VARIANCE_SHARPE_SEMANTICS",
        ),
        "benchmark": (objective, "H00300_TOTAL_RETURN"),
        "zero_variance_status": (correction, "UNDEFINED_ZERO_VARIANCE"),
        "visible_result_label": (
            correction,
            "VERSIONED_EVALUATION_CORRECTION_NOT_NEW_HOLDOUT",
        ),
        "expected_v1_failure_status": (
            source,
            "FAILED_DATA_OR_EXECUTION_CONTRACT",
        ),
        "expected_v1_error_type": (source, "ValueError"),
        "expected_v1_error": (source, "收益波动为零，夏普不可识别"),
    }
    for name, (section, expected) in expected_text.items():
        if section.get(name) != expected:
            failures.append(name)

    expected_numbers = {
        "initial_capital_cny": (objective, 500_000.0),
        "user_transaction_fee_rate_per_leg": (objective, 0.0001),
        "minimum_annualized_net_excess": (objective, 0.40),
        "minimum_strategy_net_sharpe": (objective, 1.50),
        "annualization_trading_days": (correction, 242.0),
        "zero_variance_epsilon": (correction, 1e-15),
    }
    for name, (section, expected) in expected_numbers.items():
        try:
            actual = float(section.get(name, np.nan))
        except (TypeError, ValueError):
            actual = np.nan
        if actual != expected:
            failures.append(name)

    required_true = {
        "base_and_stress_must_both_pass": objective,
        "same_visible_path_reused_for_versioned_correction": protocol,
        "v1_outcome_partly_revealed_before_v2_freeze": protocol,
        "exact_v1_economic_execution_reuse_required": source,
        "zero_variance_target_qualified": correction,
        "full_sample_sharpe_must_be_defined": correction,
        "rolling_sharpe_gate_requires_all_windows_defined": correction,
        "bootstrap_sharpe_gate_requires_all_samples_defined": correction,
        "json_non_finite_values_allowed": correction,
        "corrected_visible_may_authorize_sealed_only_if_all_original_gates_pass": correction,
    }
    inverted_true = {
        "zero_variance_target_qualified",
        "json_non_finite_values_allowed",
    }
    for name, section in required_true.items():
        actual = bool(section.get(name))
        expected = name not in inverted_true
        if actual is not expected:
            failures.append(name)

    required_false = {
        "economic_signal_execution_or_cost_parameter_changed": protocol,
        "versioned_correction_is_new_holdout": protocol,
        "outcome_used_to_change_economic_parameters": protocol,
        "non_overlapping_block_with_undefined_sharpe_is_target_qualified": correction,
    }
    for name, section in required_false.items():
        if bool(section.get(name, True)):
            failures.append(name)

    if correction.get("zero_variance_sharpe_value", "NOT_NULL") is not None:
        failures.append("zero_variance_sharpe_value")
    if not outputs or len(set(outputs.values())) != len(outputs):
        failures.append("outputs")
    if any(
        bool(safety.get(name, True))
        for name in (
            "paper_position_generation",
            "shadow_signal_generation",
            "order_generation",
            "broker_connection",
            "live_trading_authorized",
        )
    ):
        failures.append("safety")
    if failures:
        raise ValueError(f"QDII零方差评估修正合同被弱化或损坏：{sorted(set(failures))}")


def safe_annualized_sharpe(
    returns: pd.Series | np.ndarray,
    *,
    trading_days_per_year: int,
    cash_annual_rate: float,
    zero_variance_epsilon: float,
) -> dict[str, Any]:
    """计算可审计夏普；零方差返回null和显式状态，而不是抛错。"""

    values = np.asarray(pd.to_numeric(pd.Series(returns), errors="coerce"), dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("夏普收益序列不足或包含非法值")
    daily_cash = (1.0 + cash_annual_rate) ** (1.0 / trading_days_per_year) - 1.0
    excess = values - daily_cash
    volatility = float(np.std(excess, ddof=1))
    if volatility <= zero_variance_epsilon:
        return {
            "value": None,
            "status": "UNDEFINED_ZERO_VARIANCE",
            "observation_count": int(len(values)),
            "daily_excess_volatility": volatility,
        }
    return {
        "value": float(np.mean(excess) / volatility * np.sqrt(trading_days_per_year)),
        "status": "DEFINED",
        "observation_count": int(len(values)),
        "daily_excess_volatility": volatility,
    }


def _rolling_metrics_conservative(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    window: int,
    trading_days_per_year: int,
    cash_annual_rate: float,
    zero_variance_epsilon: float,
) -> dict[str, Any]:
    """报告所有滚动窗口，并让任一不可识别夏普保守失败。"""

    strategy_values = strategy.astype(float).reset_index(drop=True)
    benchmark_values = benchmark.astype(float).reset_index(drop=True)
    if len(strategy_values) != len(benchmark_values):
        raise ValueError("策略与基准滚动评价长度不同")
    total_windows = max(0, len(strategy_values) - window + 1)
    if total_windows == 0:
        return {
            "window_trading_days": int(window),
            "window_count": 0,
            "annualized_excess_median": None,
            "sharpe_median_defined_windows": None,
            "sharpe_defined_window_count": 0,
            "sharpe_undefined_zero_variance_window_count": 0,
            "sharpe_all_windows_defined": False,
            "sharpe_status": "NO_COMPLETE_WINDOW",
        }

    strategy_log = np.log1p(strategy_values.to_numpy(dtype=float))
    benchmark_log = np.log1p(benchmark_values.to_numpy(dtype=float))
    excess_values: list[float] = []
    sharpe_values: list[float] = []
    undefined_count = 0
    for start in range(total_windows):
        stop = start + window
        strategy_slice = strategy_values.iloc[start:stop]
        strategy_annual = float(np.expm1(strategy_log[start:stop].mean() * trading_days_per_year))
        benchmark_annual = float(np.expm1(benchmark_log[start:stop].mean() * trading_days_per_year))
        excess_values.append(strategy_annual - benchmark_annual)
        sharpe = safe_annualized_sharpe(
            strategy_slice,
            trading_days_per_year=trading_days_per_year,
            cash_annual_rate=cash_annual_rate,
            zero_variance_epsilon=zero_variance_epsilon,
        )
        if sharpe["value"] is None:
            undefined_count += 1
        else:
            sharpe_values.append(float(sharpe["value"]))
    if undefined_count == 0:
        status = "ALL_WINDOWS_DEFINED"
    elif not sharpe_values:
        status = "ALL_WINDOWS_UNDEFINED_ZERO_VARIANCE"
    else:
        status = "PARTIAL_UNDEFINED_ZERO_VARIANCE"
    return {
        "window_trading_days": int(window),
        "window_count": int(total_windows),
        "annualized_excess_median": float(np.median(excess_values)),
        "sharpe_median_defined_windows": (
            None if not sharpe_values else float(np.median(sharpe_values))
        ),
        "sharpe_defined_window_count": int(len(sharpe_values)),
        "sharpe_undefined_zero_variance_window_count": int(undefined_count),
        "sharpe_all_windows_defined": bool(undefined_count == 0),
        "sharpe_status": status,
    }


def _paired_block_bootstrap_conservative(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    repetitions: int,
    block_length: int,
    trading_days_per_year: int,
    cash_annual_rate: float,
    zero_variance_epsilon: float,
    random_seed: int,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, Any]:
    """成对移动块Bootstrap；零方差样本记账且使夏普下界门失败。"""

    left = strategy.to_numpy(dtype=float)
    right = benchmark.to_numpy(dtype=float)
    n = len(left)
    if n != len(right):
        raise ValueError("策略与基准Bootstrap长度不同")
    if n < block_length:
        raise ValueError("历史收益不足一个Bootstrap块")
    if repetitions <= 0:
        raise ValueError("Bootstrap重复次数必须为正")
    starts = np.arange(n - block_length + 1)
    block_count = int(np.ceil(n / block_length))
    rng = np.random.default_rng(random_seed)
    excess_values = np.empty(repetitions, dtype=float)
    sharpe_values: list[float] = []
    undefined_count = 0
    for index in range(repetitions):
        chosen = rng.choice(starts, size=block_count, replace=True)
        indices = np.concatenate(
            [np.arange(start, start + block_length) for start in chosen]
        )[:n]
        sampled_strategy = left[indices]
        sampled_benchmark = right[indices]
        strategy_annual = float(
            np.expm1(np.log1p(sampled_strategy).mean() * trading_days_per_year)
        )
        benchmark_annual = float(
            np.expm1(np.log1p(sampled_benchmark).mean() * trading_days_per_year)
        )
        excess_values[index] = strategy_annual - benchmark_annual
        sharpe = safe_annualized_sharpe(
            sampled_strategy,
            trading_days_per_year=trading_days_per_year,
            cash_annual_rate=cash_annual_rate,
            zero_variance_epsilon=zero_variance_epsilon,
        )
        if sharpe["value"] is None:
            undefined_count += 1
        else:
            sharpe_values.append(float(sharpe["value"]))
    if undefined_count == 0:
        status = "ALL_SAMPLES_DEFINED"
    elif not sharpe_values:
        status = "ALL_SAMPLES_UNDEFINED_ZERO_VARIANCE"
    else:
        status = "PARTIAL_UNDEFINED_ZERO_VARIANCE"
    sharpe_interval: list[float | None]
    if sharpe_values:
        sharpe_array = np.asarray(sharpe_values, dtype=float)
        sharpe_interval = [
            float(np.quantile(sharpe_array, lower_quantile)),
            float(np.quantile(sharpe_array, upper_quantile)),
        ]
        sharpe_median: float | None = float(np.median(sharpe_array))
    else:
        sharpe_interval = [None, None]
        sharpe_median = None
    return {
        "method": "PAIRED_MOVING_BLOCK_CONSERVATIVE_ZERO_VARIANCE",
        "repetitions": int(repetitions),
        "block_length_trading_days": int(block_length),
        "annualized_excess_median": float(np.median(excess_values)),
        "annualized_excess_interval_95pct": [
            float(np.quantile(excess_values, lower_quantile)),
            float(np.quantile(excess_values, upper_quantile)),
        ],
        "strategy_sharpe_median_defined_samples": sharpe_median,
        "strategy_sharpe_interval_95pct_defined_samples": sharpe_interval,
        "strategy_sharpe_defined_sample_count": int(len(sharpe_values)),
        "strategy_sharpe_undefined_zero_variance_sample_count": int(undefined_count),
        "strategy_sharpe_all_samples_defined": bool(undefined_count == 0),
        "strategy_sharpe_status": status,
    }


def _prepare_daily(daily: pd.DataFrame, v1_contract: dict[str, Any]) -> pd.DataFrame:
    required = {
        "trade_date",
        "strategy_base_net_return",
        "strategy_stress_net_return",
        "benchmark_total_return",
        "gross_exposure",
        "net_exposure",
        "quality_complete",
        "capacity_pass",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise DataContractError(f"历史收益缺少字段：{sorted(missing)}")
    data = daily.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data.sort_values("trade_date", inplace=True)
    data.reset_index(drop=True, inplace=True)
    if data.empty or data["trade_date"].isna().any() or data["trade_date"].duplicated().any():
        raise DataContractError("历史收益为空、日期无效或日期重复")
    numeric = [
        "strategy_base_net_return",
        "strategy_stress_net_return",
        "benchmark_total_return",
        "gross_exposure",
        "net_exposure",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[numeric].isna().any().any() or not np.isfinite(
        data[numeric].to_numpy(dtype=float)
    ).all():
        raise DataContractError("历史收益或敞口含缺失或非有限值")
    returns = [
        "strategy_base_net_return",
        "strategy_stress_net_return",
        "benchmark_total_return",
    ]
    if (data[returns] <= -1.0).any().any():
        raise DataContractError("历史日收益不得小于等于-100%")
    gates = v1_contract["visible_gates"]
    if bool(gates["data_quality_complete_required"]) and not data[
        "quality_complete"
    ].astype(bool).all():
        raise DataContractError("历史路径存在不完整数据")
    if bool(gates["capacity_pass_required"]) and not data["capacity_pass"].astype(bool).all():
        raise DataContractError("历史路径容量失败")
    risk = v1_contract["risk"]
    if (data["gross_exposure"] < -1e-12).any() or (
        data["gross_exposure"] > float(risk["maximum_gross_exposure"]) + 1e-12
    ).any():
        raise DataContractError("历史路径毛敞口超过冻结上限")
    if (
        data["net_exposure"].abs()
        > float(risk["maximum_absolute_net_exposure"]) + 1e-12
    ).any():
        raise DataContractError("历史路径净敞口超过冻结上限")
    return data


def evaluate_historical_returns_zero_variance_v2(
    daily: pd.DataFrame,
    v1_contract: dict[str, Any],
    correction_contract: dict[str, Any],
    *,
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    """按原40个百分点和1.50夏普硬门完成保守的V2历史评价。"""

    validate_correction_contract(correction_contract)
    data = _prepare_daily(daily, v1_contract)
    gates_config = v1_contract["visible_gates"]
    correction = correction_contract["evaluation_correction"]
    trading_days = int(gates_config["annualization_trading_days"])
    cash_rate = float(v1_contract["account"]["cash_annual_rate"])
    epsilon = float(correction["zero_variance_epsilon"])
    benchmark = data["benchmark_total_return"].astype(float)
    base = data["strategy_base_net_return"].astype(float)
    stress = data["strategy_stress_net_return"].astype(float)

    benchmark_cagr = annualized_return(benchmark, trading_days)
    base_cagr = annualized_return(base, trading_days)
    stress_cagr = annualized_return(stress, trading_days)
    base_excess = base_cagr - benchmark_cagr
    stress_excess = stress_cagr - benchmark_cagr
    base_sharpe = safe_annualized_sharpe(
        base,
        trading_days_per_year=trading_days,
        cash_annual_rate=cash_rate,
        zero_variance_epsilon=epsilon,
    )
    stress_sharpe = safe_annualized_sharpe(
        stress,
        trading_days_per_year=trading_days,
        cash_annual_rate=cash_rate,
        zero_variance_epsilon=epsilon,
    )

    window = int(gates_config["rolling_window_trading_days"])
    base_rolling = _rolling_metrics_conservative(
        base,
        benchmark,
        window=window,
        trading_days_per_year=trading_days,
        cash_annual_rate=cash_rate,
        zero_variance_epsilon=epsilon,
    )
    stress_rolling = _rolling_metrics_conservative(
        stress,
        benchmark,
        window=window,
        trading_days_per_year=trading_days,
        cash_annual_rate=cash_rate,
        zero_variance_epsilon=epsilon,
    )

    bootstrap_config = gates_config["bootstrap"]
    repetitions = (
        int(bootstrap_repetitions_override)
        if bootstrap_repetitions_override is not None
        else int(bootstrap_config["repetitions"])
    )
    common_bootstrap = {
        "repetitions": repetitions,
        "block_length": int(bootstrap_config["block_length_trading_days"]),
        "trading_days_per_year": trading_days,
        "cash_annual_rate": cash_rate,
        "zero_variance_epsilon": epsilon,
        "lower_quantile": float(bootstrap_config["lower_quantile"]),
        "upper_quantile": 1.0 - float(bootstrap_config["lower_quantile"]),
    }
    base_bootstrap = _paired_block_bootstrap_conservative(
        base,
        benchmark,
        random_seed=int(bootstrap_config["random_seed"]),
        **common_bootstrap,
    )
    stress_bootstrap = _paired_block_bootstrap_conservative(
        stress,
        benchmark,
        random_seed=int(bootstrap_config["random_seed"]) + 1,
        **common_bootstrap,
    )

    target = float(gates_config["minimum_annualized_net_excess"])
    sharpe_target = float(gates_config["minimum_strategy_net_sharpe"])
    year_blocks: list[dict[str, Any]] = []
    for start in range(0, len(data) - window + 1, window):
        stop = start + window
        block_benchmark = benchmark.iloc[start:stop]
        block_base = base.iloc[start:stop]
        block_stress = stress.iloc[start:stop]
        benchmark_return = annualized_return(block_benchmark, trading_days)
        base_return = annualized_return(block_base, trading_days)
        stress_return = annualized_return(block_stress, trading_days)
        base_block_sharpe = safe_annualized_sharpe(
            block_base,
            trading_days_per_year=trading_days,
            cash_annual_rate=cash_rate,
            zero_variance_epsilon=epsilon,
        )
        stress_block_sharpe = safe_annualized_sharpe(
            block_stress,
            trading_days_per_year=trading_days,
            cash_annual_rate=cash_rate,
            zero_variance_epsilon=epsilon,
        )
        qualified = bool(
            base_return - benchmark_return >= target
            and stress_return - benchmark_return >= target
            and base_block_sharpe["value"] is not None
            and stress_block_sharpe["value"] is not None
            and float(base_block_sharpe["value"]) >= sharpe_target
            and float(stress_block_sharpe["value"]) >= sharpe_target
        )
        year_blocks.append(
            {
                "start": data["trade_date"].iloc[start].date().isoformat(),
                "end": data["trade_date"].iloc[stop - 1].date().isoformat(),
                "base_annualized_excess": base_return - benchmark_return,
                "stress_annualized_excess": stress_return - benchmark_return,
                "base_sharpe": base_block_sharpe["value"],
                "stress_sharpe": stress_block_sharpe["value"],
                "base_sharpe_status": base_block_sharpe["status"],
                "stress_sharpe_status": stress_block_sharpe["status"],
                "target_qualified": qualified,
            }
        )
    qualified_blocks = sum(bool(item["target_qualified"]) for item in year_blocks)
    undefined_blocks = sum(
        item["base_sharpe_status"] != "DEFINED"
        or item["stress_sharpe_status"] != "DEFINED"
        for item in year_blocks
    )

    base_sharpe_value = base_sharpe["value"]
    stress_sharpe_value = stress_sharpe["value"]
    base_rolling_sharpe = base_rolling["sharpe_median_defined_windows"]
    stress_rolling_sharpe = stress_rolling["sharpe_median_defined_windows"]
    excess_floor = float(bootstrap_config["minimum_excess_lower_bound"])
    bootstrap_sharpe_floor = float(bootstrap_config["minimum_sharpe_lower_bound"])
    base_bootstrap_lower = base_bootstrap[
        "strategy_sharpe_interval_95pct_defined_samples"
    ][0]
    stress_bootstrap_lower = stress_bootstrap[
        "strategy_sharpe_interval_95pct_defined_samples"
    ][0]
    base_rolling_defined_gate = bool(
        base_rolling["window_count"] > 0 and base_rolling["sharpe_all_windows_defined"]
    )
    stress_rolling_defined_gate = bool(
        stress_rolling["window_count"] > 0
        and stress_rolling["sharpe_all_windows_defined"]
    )
    base_bootstrap_defined_gate = bool(
        base_bootstrap["strategy_sharpe_all_samples_defined"]
    )
    stress_bootstrap_defined_gate = bool(
        stress_bootstrap["strategy_sharpe_all_samples_defined"]
    )
    gates = {
        "base_annualized_excess_at_least_40pct": bool(base_excess >= target),
        "stress_annualized_excess_at_least_40pct": bool(stress_excess >= target),
        "base_strategy_sharpe_defined": bool(base_sharpe_value is not None),
        "stress_strategy_sharpe_defined": bool(stress_sharpe_value is not None),
        "base_strategy_sharpe_at_least_1_5": bool(
            base_sharpe_value is not None and float(base_sharpe_value) >= sharpe_target
        ),
        "stress_strategy_sharpe_at_least_1_5": bool(
            stress_sharpe_value is not None and float(stress_sharpe_value) >= sharpe_target
        ),
        "base_rolling_excess_median_at_least_40pct": bool(
            base_rolling["annualized_excess_median"] is not None
            and float(base_rolling["annualized_excess_median"])
            >= float(gates_config["minimum_rolling_excess_median"])
        ),
        "stress_rolling_excess_median_at_least_40pct": bool(
            stress_rolling["annualized_excess_median"] is not None
            and float(stress_rolling["annualized_excess_median"])
            >= float(gates_config["minimum_rolling_excess_median"])
        ),
        "base_rolling_sharpe_all_windows_defined": base_rolling_defined_gate,
        "stress_rolling_sharpe_all_windows_defined": stress_rolling_defined_gate,
        "base_rolling_sharpe_median_at_least_1_5": bool(
            base_rolling_defined_gate
            and base_rolling_sharpe is not None
            and float(base_rolling_sharpe)
            >= float(gates_config["minimum_rolling_sharpe_median"])
        ),
        "stress_rolling_sharpe_median_at_least_1_5": bool(
            stress_rolling_defined_gate
            and stress_rolling_sharpe is not None
            and float(stress_rolling_sharpe)
            >= float(gates_config["minimum_rolling_sharpe_median"])
        ),
        "base_bootstrap_excess_lower_bound_positive": bool(
            base_bootstrap["annualized_excess_interval_95pct"][0] > excess_floor
        ),
        "stress_bootstrap_excess_lower_bound_positive": bool(
            stress_bootstrap["annualized_excess_interval_95pct"][0] > excess_floor
        ),
        "base_bootstrap_sharpe_all_samples_defined": base_bootstrap_defined_gate,
        "stress_bootstrap_sharpe_all_samples_defined": stress_bootstrap_defined_gate,
        "base_bootstrap_sharpe_lower_bound_at_least_1": bool(
            base_bootstrap_defined_gate
            and base_bootstrap_lower is not None
            and float(base_bootstrap_lower) >= bootstrap_sharpe_floor
        ),
        "stress_bootstrap_sharpe_lower_bound_at_least_1": bool(
            stress_bootstrap_defined_gate
            and stress_bootstrap_lower is not None
            and float(stress_bootstrap_lower) >= bootstrap_sharpe_floor
        ),
        "minimum_non_overlapping_year_blocks": bool(
            len(year_blocks)
            >= int(gates_config["minimum_non_overlapping_242d_blocks"])
        ),
        "minimum_target_qualified_year_blocks": bool(
            qualified_blocks >= int(gates_config["minimum_target_qualified_blocks"])
        ),
        "data_quality_complete": bool(data["quality_complete"].astype(bool).all()),
        "capacity_pass": bool(data["capacity_pass"].astype(bool).all()),
    }
    result = {
        "eligible_historical_day_count": int(len(data)),
        "first_trade_date": data["trade_date"].iloc[0].date().isoformat(),
        "last_trade_date": data["trade_date"].iloc[-1].date().isoformat(),
        "evaluation_semantics": correction["visible_result_label"],
        "metrics": {
            "benchmark_total_return_cagr": benchmark_cagr,
            "strategy_base_net_cagr": base_cagr,
            "strategy_stress_net_cagr": stress_cagr,
            "base_annualized_excess": base_excess,
            "stress_annualized_excess": stress_excess,
            "base_strategy_net_sharpe": base_sharpe_value,
            "stress_strategy_net_sharpe": stress_sharpe_value,
            "base_strategy_net_sharpe_status": base_sharpe["status"],
            "stress_strategy_net_sharpe_status": stress_sharpe["status"],
            "base_maximum_drawdown": maximum_drawdown(base),
            "stress_maximum_drawdown": maximum_drawdown(stress),
            "base_rolling_242d_excess_median": base_rolling[
                "annualized_excess_median"
            ],
            "stress_rolling_242d_excess_median": stress_rolling[
                "annualized_excess_median"
            ],
            "base_rolling_242d_sharpe_median_defined_windows": base_rolling_sharpe,
            "stress_rolling_242d_sharpe_median_defined_windows": stress_rolling_sharpe,
            "base_rolling": base_rolling,
            "stress_rolling": stress_rolling,
            "base_bootstrap": base_bootstrap,
            "stress_bootstrap": stress_bootstrap,
            "non_overlapping_years": year_blocks,
            "non_overlapping_year_block_count": int(len(year_blocks)),
            "undefined_zero_variance_year_block_count": int(undefined_blocks),
            "target_qualified_year_block_count": int(qualified_blocks),
        },
        "gates": gates,
        "all_visible_gates_pass": bool(all(gates.values())),
    }
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写入严格JSON，拒绝NaN和无穷值。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    """原子写入UTF-8文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _format_percent(value: float | None) -> str:
    return "不可识别" if value is None else f"{value:.2%}"


def _format_float(value: float | None) -> str:
    return "不可识别" if value is None else f"{value:.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    """渲染只包含结论相关证据的可见期报告。"""

    evaluation = report["evaluation"]
    metrics = evaluation["metrics"]
    portfolio = report["data_audit"]["portfolio"]
    gate_lines = [
        f"- `{name}`：{value}。" for name, value in evaluation["gates"].items()
    ]
    block_lines = []
    for item in metrics["non_overlapping_years"]:
        block_lines.append(
            "- {start} 至 {end}：基础/压力超额 {base_excess}/{stress_excess}，"
            "基础/压力夏普 {base_sharpe}/{stress_sharpe}，状态 "
            "{base_status}/{stress_status}，合格={qualified}。".format(
                start=item["start"],
                end=item["end"],
                base_excess=_format_percent(item["base_annualized_excess"]),
                stress_excess=_format_percent(item["stress_annualized_excess"]),
                base_sharpe=_format_float(item["base_sharpe"]),
                stress_sharpe=_format_float(item["stress_sharpe"]),
                base_status=item["base_sharpe_status"],
                stress_status=item["stress_sharpe_status"],
                qualified=item["target_qualified"],
            )
        )
    return "\n".join(
        [
            "# QDII跨市场折价回归零方差评估修正V2可见期结果",
            "",
            f"状态：`{report['status']}`",
            "",
            f"证据身份：`{evaluation['evaluation_semantics']}`。同一可见路径仅作版本化评估修正，不是新未见样本。",
            "",
            "## 核心结果",
            "",
            f"- 可见期：{report['period']['start']} 至 {report['period']['end']}。",
            f"- H00300全收益CAGR：{_format_percent(metrics['benchmark_total_return_cagr'])}。",
            f"- 基础/压力策略净CAGR：{_format_percent(metrics['strategy_base_net_cagr'])}/{_format_percent(metrics['strategy_stress_net_cagr'])}。",
            f"- 基础/压力年化净超额：{_format_percent(metrics['base_annualized_excess'])}/{_format_percent(metrics['stress_annualized_excess'])}。",
            f"- 基础/压力净夏普：{_format_float(metrics['base_strategy_net_sharpe'])}/{_format_float(metrics['stress_strategy_net_sharpe'])}。",
            f"- 基础/压力最大回撤：{_format_percent(metrics['base_maximum_drawdown'])}/{_format_percent(metrics['stress_maximum_drawdown'])}。",
            f"- 买入/卖出交易数：{portfolio['entry_transaction_count']}/{portfolio['exit_transaction_count']}。",
            f"- 最终基础/压力净值：{portfolio['final_base_nav_cny']:.2f}/{portfolio['final_stress_nav_cny']:.2f}元。",
            "",
            "## 零方差处理",
            "",
            f"- 基础/压力滚动242日不可识别窗口：{metrics['base_rolling']['sharpe_undefined_zero_variance_window_count']}/{metrics['stress_rolling']['sharpe_undefined_zero_variance_window_count']}。",
            f"- 基础/压力Bootstrap不可识别样本：{metrics['base_bootstrap']['strategy_sharpe_undefined_zero_variance_sample_count']}/{metrics['stress_bootstrap']['strategy_sharpe_undefined_zero_variance_sample_count']}。",
            f"- 互不重叠242日块总数/零方差块数/合格块数：{metrics['non_overlapping_year_block_count']}/{metrics['undefined_zero_variance_year_block_count']}/{metrics['target_qualified_year_block_count']}。",
            "",
            "## 互不重叠242日块",
            "",
            *block_lines,
            "",
            "## 全部硬门",
            "",
            *gate_lines,
            "",
            f"全部可见期门通过：{evaluation['all_visible_gates_pass']}。",
            "",
            "## 决策",
            "",
            report["decision"]["next_step"],
            "",
            "V1失败原样保留；封存复验、Paper、Shadow、订单、券商连接和实盘均未打开。",
            "",
        ]
    )


def run_visible(
    correction_contract: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    """复用V1经济路径，运行一次V2可见期评估修正。"""

    validate_correction_contract(correction_contract)
    v1_config = ROOT / correction_contract["source_freeze"]["v1_config"]
    v1_contract = load_v1_contract(v1_config)
    partition = v1_contract["historical_partition"]
    start = pd.Timestamp(partition["visible_start"])
    end = pd.Timestamp(partition["visible_end"])
    panel, master, benchmark, fx, indices = load_research_inputs(v1_contract)
    signals, market, return_wide, signal_audit = build_discount_signals(
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
    evaluation = evaluate_historical_returns_zero_variance_v2(
        daily,
        v1_contract,
        correction_contract,
        bootstrap_repetitions_override=bootstrap_repetitions_override,
    )
    passed = bool(evaluation["all_visible_gates_pass"])
    status = (
        "VISIBLE_PASS_SEALED_REPLICATION_AUTHORIZED_NOT_OPENED"
        if passed
        else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
    )
    source = correction_contract["source_freeze"]
    report = {
        "schema_version": "2.0.0",
        "report_id": "QDII_CROSS_MARKET_DISCOUNT_REVERSION_ZERO_VARIANCE_V2_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": status,
        "goal_achieved": False,
        "candidate_id": correction_contract["protocol"]["candidate_id"],
        "economic_candidate_id": v1_contract["protocol"]["candidate_id"],
        "parent_protocol_id": v1_contract["protocol"]["parent_protocol_id"],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "objective": correction_contract["objective"],
        "correction_scope": {
            "class": correction_contract["protocol"]["correction_class"],
            "economic_signal_execution_or_cost_parameter_changed": False,
            "same_visible_path_reused": True,
            "new_holdout": False,
            "partly_revealed_before_v2_freeze": True,
        },
        "v1_failure_preservation": {
            "status": source["expected_v1_failure_status"],
            "error_type": source["expected_v1_error_type"],
            "error": source["expected_v1_error"],
            "manifest_sha256": source["expected_v1_manifest_sha256"],
            "visible_failure_report_sha256": source[
                "expected_v1_visible_failure_report_sha256"
            ],
            "visible_failure_markdown_sha256": source[
                "expected_v1_visible_failure_markdown_sha256"
            ],
        },
        "manifest_verification": manifest_verification,
        "data_audit": {
            "signal": signal_audit,
            "portfolio": portfolio_audit,
            "foreign_index_ranges": {
                group_id: {
                    "first_date": frame["date"].min().date().isoformat(),
                    "last_date": frame["date"].max().date().isoformat(),
                    "rows": int(len(frame)),
                }
                for group_id, frame in indices.items()
            },
            "fx_range": {
                "first_date": fx["date"].min().date().isoformat(),
                "last_date": fx["date"].max().date().isoformat(),
                "rows": int(len(fx)),
            },
        },
        "evaluation": evaluation,
        "historical_evidence_limits": v1_contract["historical_evidence_limits"],
        "decision": {
            "sealed_replication_authorized": passed,
            "sealed_replication_open": False,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "next_step": (
                "可见期原始硬门和保守零方差门全部通过；仅授权另行验证后打开原封存复验，本次未打开"
                if passed
                else "冻结拒绝本候选；不打开封存期，不改变经济参数救回，转向新的独立机制"
            ),
        },
        "outputs": correction_contract["outputs"],
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": correction_contract["safety"],
    }
    outputs = correction_contract["outputs"]
    daily_path = ROOT / outputs["visible_daily_returns"]
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(daily_path, index=False)
    signals.to_parquet(ROOT / outputs["visible_signals"], index=False)
    trades.to_parquet(ROOT / outputs["visible_trades"], index=False)
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
