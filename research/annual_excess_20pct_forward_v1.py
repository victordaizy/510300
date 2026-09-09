"""510300 年化净超额 20 个百分点前瞻目标的计算核心。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REQUIRED_RETURN_COLUMNS = {
    "trade_date",
    "strategy_base_net_return",
    "strategy_stress_net_return",
    "benchmark_total_return",
    "quality_complete",
    "forward_observed",
}


def load_contract(path: Path) -> dict[str, Any]:
    """读取并校验目标合同。"""

    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    """确保用户给定的 20%目标没有被弱化。"""

    protocol = contract.get("protocol", {})
    objective = contract.get("objective", {})
    scope = contract.get("scope", {})
    governance = contract.get("governance", {})
    maturity = contract.get("maturity", {})

    failures: list[str] = []
    if protocol.get("protocol_id") != "510300_ANNUAL_EXCESS_20PCT_FORWARD_V1":
        failures.append("protocol_id")
    if float(objective.get("minimum_annualized_excess", -1.0)) != 0.20:
        failures.append("minimum_annualized_excess")
    if float(objective.get("minimum_rolling_242d_excess_median", -1.0)) != 0.20:
        failures.append("minimum_rolling_242d_excess_median")
    if not bool(objective.get("base_cost_required")):
        failures.append("base_cost_required")
    if not bool(objective.get("stress_cost_required")):
        failures.append("stress_cost_required")
    if objective.get("comparator") != "GREATER_THAN_OR_EQUAL":
        failures.append("comparator")
    if objective.get("primary_metric") != "STRATEGY_NET_CAGR_MINUS_H00300_TOTAL_RETURN_CAGR":
        failures.append("primary_metric")
    if int(scope.get("maximum_active_research_streams", -1)) != 2:
        failures.append("maximum_active_research_streams")
    if len(scope.get("active_research_streams", [])) != 2:
        failures.append("active_research_streams")
    if bool(scope.get("leverage_allowed")) or bool(scope.get("short_selling_allowed")):
        failures.append("long_only_unlevered_scope")
    if float(scope.get("initial_capital_cny", -1.0)) != 500_000.0:
        failures.append("initial_capital_cny")
    costs = contract.get("costs", {})
    if float(costs.get("commission_rate_per_leg", -1.0)) != 0.0001:
        failures.append("commission_rate_per_leg")
    if float(costs.get("minimum_commission_cny_per_leg", -1.0)) != 0.0:
        failures.append("minimum_commission_cny_per_leg")
    if float(costs.get("maximum_single_round_trip_capital_fraction", -1.0)) != 0.50:
        failures.append("maximum_single_round_trip_capital_fraction")
    if bool(governance.get("target_may_be_lowered_after_results", True)):
        failures.append("target_may_be_lowered_after_results")
    if bool(governance.get("failed_candidate_parameter_rescue_allowed", True)):
        failures.append("failed_candidate_parameter_rescue_allowed")
    if bool(governance.get("historical_backfill_allowed", True)):
        failures.append("historical_backfill_allowed")
    if int(maturity.get("verified_claim_days", -1)) < 726:
        failures.append("verified_claim_days")

    required_columns = set(contract.get("evaluation_input", {}).get("required_columns", []))
    if required_columns != REQUIRED_RETURN_COLUMNS:
        failures.append("required_columns")
    if failures:
        raise ValueError(f"20%年化净超额合同无效：{sorted(failures)}")


def round_trip_cost_bps(
    notional_cny: float,
    *,
    commission_rate_per_leg: float,
    minimum_commission_cny_per_leg: float,
    slippage_bps_per_leg: float,
    stamp_duty_sell_rate: float = 0.0,
) -> dict[str, float]:
    """计算同一名义金额一次买卖往返的显式成本与滑点。"""

    if notional_cny <= 0:
        raise ValueError("名义金额必须为正")
    if min(
        commission_rate_per_leg,
        minimum_commission_cny_per_leg,
        slippage_bps_per_leg,
        stamp_duty_sell_rate,
    ) < 0:
        raise ValueError("成本参数不得为负")

    commission_per_leg = max(notional_cny * commission_rate_per_leg, minimum_commission_cny_per_leg)
    commission_cny = commission_per_leg * 2.0
    slippage_cny = notional_cny * slippage_bps_per_leg / 10_000.0 * 2.0
    stamp_duty_cny = notional_cny * stamp_duty_sell_rate
    total_cny = commission_cny + slippage_cny + stamp_duty_cny
    return {
        "notional_cny": float(notional_cny),
        "commission_cny": float(commission_cny),
        "slippage_cny": float(slippage_cny),
        "stamp_duty_cny": float(stamp_duty_cny),
        "total_cost_cny": float(total_cny),
        "total_cost_bps": float(total_cny / notional_cny * 10_000.0),
    }


def required_edge_budget(
    *,
    initial_capital_cny: float,
    annual_excess_target: float,
    event_notional_cny: float,
    events_per_year: int,
    round_trip_cost: dict[str, float],
) -> dict[str, float | int]:
    """计算达到年度超额目标时，每次事件需要贡献的净边际与毛边际。"""

    if initial_capital_cny <= 0 or annual_excess_target < 0:
        raise ValueError("本金必须为正且目标不得为负")
    if event_notional_cny <= 0 or events_per_year <= 0:
        raise ValueError("事件金额和年度事件数必须为正")

    target_incremental_cny = initial_capital_cny * annual_excess_target
    required_net_edge_bps = target_incremental_cny / (event_notional_cny * events_per_year) * 10_000.0
    required_gross_edge_bps = required_net_edge_bps + float(round_trip_cost["total_cost_bps"])
    return {
        "initial_capital_cny": float(initial_capital_cny),
        "annual_excess_target": float(annual_excess_target),
        "target_incremental_cny_first_year_approximation": float(target_incremental_cny),
        "event_notional_cny": float(event_notional_cny),
        "events_per_year": int(events_per_year),
        "required_net_edge_bps_per_event": float(required_net_edge_bps),
        "round_trip_cost_bps": float(round_trip_cost["total_cost_bps"]),
        "required_gross_edge_bps_per_event": float(required_gross_edge_bps),
    }


def summarize_iopv_gap_sample(frame: pd.DataFrame) -> dict[str, Any]:
    """汇总冻结前 IOPV 样本；该汇总只用于容量预算。"""

    required = {"exchange_timestamp", "trade_date", "premium_discount_bps"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"IOPV样本缺少字段：{sorted(missing)}")

    data = frame.copy()
    data["exchange_timestamp"] = pd.to_datetime(data["exchange_timestamp"], errors="coerce")
    data["premium_discount_bps"] = pd.to_numeric(data["premium_discount_bps"], errors="coerce")
    data.dropna(subset=["exchange_timestamp", "premium_discount_bps"], inplace=True)
    clock = data["exchange_timestamp"].dt.time
    morning = (clock >= pd.Timestamp("09:30:00").time()) & (clock <= pd.Timestamp("11:30:00").time())
    afternoon = (clock >= pd.Timestamp("13:00:00").time()) & (clock <= pd.Timestamp("15:00:00").time())
    data = data.loc[morning | afternoon].copy()
    if data.empty:
        return {
            "status": "NO_IN_SESSION_SAMPLE",
            "row_count": 0,
            "trade_date_count": 0,
            "discovery_only": True,
        }

    gap = data["premium_discount_bps"].astype(float)
    absolute = gap.abs()
    return {
        "status": "DISCOVERY_ONLY_NOT_PERFORMANCE_EVIDENCE",
        "row_count": int(len(data)),
        "trade_date_count": int(data["trade_date"].nunique()),
        "first_trade_date": str(data["trade_date"].min()),
        "last_trade_date": str(data["trade_date"].max()),
        "minimum_bps": float(gap.min()),
        "maximum_bps": float(gap.max()),
        "absolute_median_bps": float(absolute.median()),
        "absolute_p95_bps": float(absolute.quantile(0.95)),
        "absolute_p99_bps": float(absolute.quantile(0.99)),
        "absolute_maximum_bps": float(absolute.max()),
        "discovery_only": True,
    }


def geometric_annualized_return(returns: pd.Series, trading_days_per_year: int) -> float:
    """按交易日对几何收益进行年化。"""

    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("收益序列为空或包含非有限值")
    if (values <= -1.0).any():
        raise ValueError("收益率不得小于等于-100%")
    return float(np.expm1(np.log1p(values).mean() * trading_days_per_year))


def _rolling_excess(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    window: int,
    trading_days_per_year: int,
) -> pd.Series:
    strategy_log = np.log1p(strategy.astype(float))
    benchmark_log = np.log1p(benchmark.astype(float))
    strategy_annual = np.expm1(strategy_log.rolling(window).mean() * trading_days_per_year)
    benchmark_annual = np.expm1(benchmark_log.rolling(window).mean() * trading_days_per_year)
    return strategy_annual - benchmark_annual


def _paired_block_bootstrap(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    repetitions: int,
    block_length: int,
    trading_days_per_year: int,
    random_seed: int,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, Any]:
    left = strategy.to_numpy(dtype=float)
    right = benchmark.to_numpy(dtype=float)
    n = len(left)
    if n < block_length:
        raise ValueError("收益序列不足一个Bootstrap块")

    starts = np.arange(n - block_length + 1)
    block_count = int(np.ceil(n / block_length))
    rng = np.random.default_rng(random_seed)
    values = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        chosen = rng.choice(starts, size=block_count, replace=True)
        sample_indices = np.concatenate(
            [np.arange(start, start + block_length) for start in chosen]
        )[:n]
        strategy_cagr = float(np.expm1(np.log1p(left[sample_indices]).mean() * trading_days_per_year))
        benchmark_cagr = float(np.expm1(np.log1p(right[sample_indices]).mean() * trading_days_per_year))
        values[index] = strategy_cagr - benchmark_cagr
    return {
        "repetitions": int(repetitions),
        "block_length_trading_days": int(block_length),
        "median": float(np.median(values)),
        "interval_95pct": [
            float(np.quantile(values, lower_quantile)),
            float(np.quantile(values, upper_quantile)),
        ],
    }


def _prepare_return_frame(frame: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    missing = REQUIRED_RETURN_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"前瞻收益台账缺少字段：{sorted(missing)}")

    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    if data["trade_date"].isna().any():
        raise ValueError("trade_date包含无效日期")
    if data["trade_date"].duplicated().any():
        raise ValueError("trade_date必须唯一")
    data.sort_values("trade_date", inplace=True)

    effective_from = pd.Timestamp(contract["protocol"]["effective_from"])
    if bool(contract["evaluation_input"]["pre_effective_rows_allowed"]) is False:
        if (data["trade_date"] < effective_from).any():
            raise ValueError("前瞻收益台账包含合同生效日前记录")

    quality = data["quality_complete"].astype(bool)
    forward = data["forward_observed"].astype(bool)
    eligible = data.loc[quality & forward].copy()
    return_columns = [
        "strategy_base_net_return",
        "strategy_stress_net_return",
        "benchmark_total_return",
    ]
    for column in return_columns:
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
    if eligible[return_columns].isna().any().any():
        raise ValueError("合格前瞻记录包含缺失收益")
    values = eligible[return_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("合格前瞻记录包含非法收益")
    return eligible.reset_index(drop=True)


def evaluate_forward_returns(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    *,
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    """按冻结门槛评价前瞻日收益台账。"""

    validate_contract(contract)
    data = _prepare_return_frame(frame, contract)
    maturity = contract["maturity"]
    objective = contract["objective"]
    n = len(data)
    first_unseen = int(maturity["pcf_first_unseen_evaluation_days"])
    replication = int(maturity["pcf_replication_days"])
    full_year = int(maturity["first_full_year_estimate_days"])
    verified_days = int(maturity["verified_claim_days"])

    base_result: dict[str, Any] = {
        "eligible_forward_day_count": int(n),
        "first_trade_date": None if data.empty else data["trade_date"].iloc[0].date().isoformat(),
        "last_trade_date": None if data.empty else data["trade_date"].iloc[-1].date().isoformat(),
        "target_annualized_excess": float(objective["minimum_annualized_excess"]),
        "goal_achieved": False,
        "metrics": None,
        "gates": None,
    }
    if n < first_unseen:
        base_result["status"] = "PENDING_FIRST_UNSEEN_GATE"
        base_result["next_required_day_count"] = first_unseen
        return base_result

    trading_days_per_year = int(objective["annualization_trading_days"])
    benchmark = data["benchmark_total_return"]
    base = data["strategy_base_net_return"]
    stress = data["strategy_stress_net_return"]
    benchmark_cagr = geometric_annualized_return(benchmark, trading_days_per_year)
    base_cagr = geometric_annualized_return(base, trading_days_per_year)
    stress_cagr = geometric_annualized_return(stress, trading_days_per_year)
    base_excess = base_cagr - benchmark_cagr
    stress_excess = stress_cagr - benchmark_cagr

    rolling_window = int(objective["rolling_window_trading_days"])
    base_rolling = _rolling_excess(
        base,
        benchmark,
        window=rolling_window,
        trading_days_per_year=trading_days_per_year,
    ).dropna()
    stress_rolling = _rolling_excess(
        stress,
        benchmark,
        window=rolling_window,
        trading_days_per_year=trading_days_per_year,
    ).dropna()

    bootstrap_contract = deepcopy(objective["bootstrap"])
    repetitions = (
        int(bootstrap_repetitions_override)
        if bootstrap_repetitions_override is not None
        else int(bootstrap_contract["repetitions"])
    )
    base_bootstrap = _paired_block_bootstrap(
        base,
        benchmark,
        repetitions=repetitions,
        block_length=int(bootstrap_contract["block_length_trading_days"]),
        trading_days_per_year=trading_days_per_year,
        random_seed=int(bootstrap_contract["random_seed"]),
        lower_quantile=float(bootstrap_contract["lower_quantile"]),
        upper_quantile=float(bootstrap_contract["upper_quantile"]),
    )
    stress_bootstrap = _paired_block_bootstrap(
        stress,
        benchmark,
        repetitions=repetitions,
        block_length=int(bootstrap_contract["block_length_trading_days"]),
        trading_days_per_year=trading_days_per_year,
        random_seed=int(bootstrap_contract["random_seed"]) + 1,
        lower_quantile=float(bootstrap_contract["lower_quantile"]),
        upper_quantile=float(bootstrap_contract["upper_quantile"]),
    )

    year_block = int(objective["non_overlapping_years"]["block_length_trading_days"])
    non_overlapping_years: list[dict[str, Any]] = []
    for start in range(0, n - year_block + 1, year_block):
        stop = start + year_block
        block_base = geometric_annualized_return(base.iloc[start:stop], trading_days_per_year)
        block_stress = geometric_annualized_return(stress.iloc[start:stop], trading_days_per_year)
        block_benchmark = geometric_annualized_return(benchmark.iloc[start:stop], trading_days_per_year)
        non_overlapping_years.append(
            {
                "start": data["trade_date"].iloc[start].date().isoformat(),
                "end": data["trade_date"].iloc[stop - 1].date().isoformat(),
                "base_annualized_excess": float(block_base - block_benchmark),
                "stress_annualized_excess": float(block_stress - block_benchmark),
            }
        )

    target = float(objective["minimum_annualized_excess"])
    rolling_target = float(objective["minimum_rolling_242d_excess_median"])
    bootstrap_floor = float(bootstrap_contract["minimum_lower_bound"])
    minimum_year_blocks = int(objective["non_overlapping_years"]["minimum_blocks_for_verified_claim"])
    minimum_positive_years = int(objective["non_overlapping_years"]["minimum_positive_excess_blocks"])
    positive_years = sum(
        item["base_annualized_excess"] > 0.0 and item["stress_annualized_excess"] > 0.0
        for item in non_overlapping_years
    )
    gates = {
        "base_annualized_excess_at_least_20pct": base_excess >= target,
        "stress_annualized_excess_at_least_20pct": stress_excess >= target,
        "base_rolling_242d_median_at_least_20pct": bool(
            not base_rolling.empty and float(base_rolling.median()) >= rolling_target
        ),
        "stress_rolling_242d_median_at_least_20pct": bool(
            not stress_rolling.empty and float(stress_rolling.median()) >= rolling_target
        ),
        "base_bootstrap_lower_bound_positive": base_bootstrap["interval_95pct"][0] > bootstrap_floor,
        "stress_bootstrap_lower_bound_positive": stress_bootstrap["interval_95pct"][0] > bootstrap_floor,
        "minimum_non_overlapping_year_blocks": len(non_overlapping_years) >= minimum_year_blocks,
        "minimum_positive_non_overlapping_year_blocks": positive_years >= minimum_positive_years,
        "verified_forward_day_count": n >= verified_days,
    }
    metrics = {
        "strategy_base_net_cagr": float(base_cagr),
        "strategy_stress_net_cagr": float(stress_cagr),
        "benchmark_total_return_cagr": float(benchmark_cagr),
        "base_annualized_excess": float(base_excess),
        "stress_annualized_excess": float(stress_excess),
        "base_rolling_242d_excess_median": None if base_rolling.empty else float(base_rolling.median()),
        "stress_rolling_242d_excess_median": None if stress_rolling.empty else float(stress_rolling.median()),
        "base_bootstrap": base_bootstrap,
        "stress_bootstrap": stress_bootstrap,
        "non_overlapping_years": non_overlapping_years,
        "positive_non_overlapping_year_count": int(positive_years),
    }
    base_result["metrics"] = metrics
    base_result["gates"] = gates

    if n < replication:
        status = "PROVISIONAL_FIRST_UNSEEN_NOT_REPLICATED"
        next_required = replication
    elif n < full_year:
        status = "PROVISIONAL_REPLICATION_NOT_FULL_YEAR"
        next_required = full_year
    elif n < verified_days:
        status = "ONE_YEAR_TARGET_CANDIDATE_NOT_MULTIYEAR" if all(
            gates[key]
            for key in (
                "base_annualized_excess_at_least_20pct",
                "stress_annualized_excess_at_least_20pct",
                "base_rolling_242d_median_at_least_20pct",
                "stress_rolling_242d_median_at_least_20pct",
                "base_bootstrap_lower_bound_positive",
                "stress_bootstrap_lower_bound_positive",
            )
        ) else "CURRENT_POINT_ESTIMATE_BELOW_20PCT_TARGET"
        next_required = verified_days
    elif all(gates.values()):
        status = "VERIFIED_ANNUAL_EXCESS_20PCT"
        next_required = None
        base_result["goal_achieved"] = True
    else:
        status = "REJECTED_ANNUAL_EXCESS_20PCT"
        next_required = None

    base_result["status"] = status
    base_result["next_required_day_count"] = next_required
    return base_result
