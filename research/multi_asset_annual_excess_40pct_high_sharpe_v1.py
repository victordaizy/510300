"""多资产年化净超额40个百分点与高夏普目标的前瞻评估核心。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "trade_date",
    "strategy_base_net_return",
    "strategy_stress_net_return",
    "benchmark_total_return",
    "gross_exposure",
    "net_exposure",
    "quality_complete",
    "capacity_pass",
    "forward_observed",
}


def load_contract(path: Path) -> dict[str, Any]:
    """读取并校验多资产40%高夏普合同。"""

    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    """确保用户指定目标、账户、费用和研究边界没有被弱化。"""

    protocol = contract.get("protocol", {})
    scope = contract.get("scope", {})
    objective = contract.get("objective", {})
    costs = contract.get("costs", {})
    maturity = contract.get("maturity", {})
    governance = contract.get("candidate_governance", {})
    safety = contract.get("safety", {})
    failures: list[str] = []

    if protocol.get("protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V1":
        failures.append("protocol_id")
    if float(scope.get("initial_capital_cny", -1.0)) != 500_000.0:
        failures.append("initial_capital_cny")
    if int(scope.get("maximum_active_research_lanes", -1)) != 2:
        failures.append("maximum_active_research_lanes")
    if len(scope.get("active_research_lanes", [])) != 2:
        failures.append("active_research_lanes")
    risk = scope.get("risk_envelope", {})
    if float(risk.get("maximum_gross_exposure", -1.0)) != 2.0:
        failures.append("maximum_gross_exposure")
    if float(risk.get("maximum_absolute_net_exposure", -1.0)) != 1.0:
        failures.append("maximum_absolute_net_exposure")
    if bool(risk.get("naked_short_option_allowed", True)):
        failures.append("naked_short_option_allowed")
    if float(objective.get("minimum_annualized_excess", -1.0)) != 0.40:
        failures.append("minimum_annualized_excess")
    if float(objective.get("minimum_strategy_net_sharpe", -1.0)) != 1.50:
        failures.append("minimum_strategy_net_sharpe")
    if float(objective.get("minimum_rolling_242d_excess_median", -1.0)) != 0.40:
        failures.append("minimum_rolling_242d_excess_median")
    if float(objective.get("minimum_rolling_242d_sharpe_median", -1.0)) != 1.50:
        failures.append("minimum_rolling_242d_sharpe_median")
    if not bool(objective.get("base_cost_required")) or not bool(objective.get("stress_cost_required")):
        failures.append("base_and_stress_cost_required")
    if float(costs.get("user_transaction_fee_rate_per_leg", -1.0)) != 0.0001:
        failures.append("user_transaction_fee_rate_per_leg")
    if float(costs.get("minimum_transaction_fee_cny_per_leg", -1.0)) != 0.0:
        failures.append("minimum_transaction_fee_cny_per_leg")
    if not bool(costs.get("mandatory_instrument_specific_taxes_required")):
        failures.append("mandatory_instrument_specific_taxes_required")
    if not bool(costs.get("borrow_funding_roll_and_margin_cost_required_when_applicable")):
        failures.append("borrow_funding_roll_and_margin_cost_required")
    if bool(governance.get("prior_failed_formula_reuse_allowed", True)):
        failures.append("prior_failed_formula_reuse_allowed")
    if bool(governance.get("prior_failed_family_parameter_rescue_allowed", True)):
        failures.append("prior_failed_family_parameter_rescue_allowed")
    if bool(governance.get("historical_backfill_allowed", True)):
        failures.append("historical_backfill_allowed")
    if bool(governance.get("target_may_be_lowered_after_results", True)):
        failures.append("target_may_be_lowered_after_results")
    if bool(governance.get("sharpe_threshold_may_be_lowered_after_results", True)):
        failures.append("sharpe_threshold_may_be_lowered_after_results")
    if int(maturity.get("verified_claim_days", -1)) < 726:
        failures.append("verified_claim_days")
    if set(contract.get("evaluation_input", {}).get("required_columns", [])) != REQUIRED_COLUMNS:
        failures.append("required_columns")
    if any(
        bool(safety.get(field, True))
        for field in (
            "paper_position_generation",
            "order_generation",
            "broker_connection",
            "live_trading_authorized",
        )
    ):
        failures.append("safety_switches")
    if failures:
        raise ValueError(f"多资产40%高夏普合同无效：{sorted(failures)}")


def annualized_return(returns: pd.Series, trading_days_per_year: int) -> float:
    """计算日收益序列的几何年化收益。"""

    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("收益序列为空或包含非法值")
    return float(np.expm1(np.log1p(values).mean() * trading_days_per_year))


def annualized_sharpe(
    returns: pd.Series,
    *,
    trading_days_per_year: int,
    cash_annual_rate: float,
) -> float:
    """使用年化现金收益换算的日无风险收益计算年化夏普。"""

    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("夏普收益序列不足或包含非法值")
    daily_cash = (1.0 + cash_annual_rate) ** (1.0 / trading_days_per_year) - 1.0
    excess = values - daily_cash
    volatility = float(np.std(excess, ddof=1))
    if volatility <= 1e-15:
        raise ValueError("收益波动为零，夏普不可识别")
    return float(np.mean(excess) / volatility * np.sqrt(trading_days_per_year))


def maximum_drawdown(returns: pd.Series) -> float:
    """计算净值曲线最大回撤。"""

    wealth = (1.0 + returns.astype(float)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def _prepare_frame(frame: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"前瞻台账缺少字段：{sorted(missing)}")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    if data["trade_date"].isna().any() or data["trade_date"].duplicated().any():
        raise ValueError("trade_date包含无效值或重复值")
    data.sort_values("trade_date", inplace=True)
    if not bool(contract["evaluation_input"]["pre_effective_rows_allowed"]):
        effective_from = pd.Timestamp(contract["protocol"]["effective_from"])
        if (data["trade_date"] < effective_from).any():
            raise ValueError("前瞻台账包含合同生效日前记录")

    numeric = [
        "strategy_base_net_return",
        "strategy_stress_net_return",
        "benchmark_total_return",
        "gross_exposure",
        "net_exposure",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[numeric].isna().any().any() or not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("前瞻台账包含缺失或非有限数值")
    if (data[["strategy_base_net_return", "strategy_stress_net_return", "benchmark_total_return"]] <= -1.0).any().any():
        raise ValueError("日收益不得小于等于-100%")

    risk = contract["scope"]["risk_envelope"]
    if (data["gross_exposure"] < 0.0).any() or (
        data["gross_exposure"] > float(risk["maximum_gross_exposure"]) + 1e-12
    ).any():
        raise ValueError("gross_exposure超出冻结上限")
    if (data["net_exposure"].abs() > float(risk["maximum_absolute_net_exposure"]) + 1e-12).any():
        raise ValueError("net_exposure超出冻结上限")

    eligible = (
        data["quality_complete"].astype(bool)
        & data["capacity_pass"].astype(bool)
        & data["forward_observed"].astype(bool)
    )
    return data.loc[eligible].reset_index(drop=True)


def _rolling_metrics(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    window: int,
    trading_days_per_year: int,
    cash_annual_rate: float,
) -> tuple[pd.Series, pd.Series]:
    strategy_log = np.log1p(strategy.astype(float))
    benchmark_log = np.log1p(benchmark.astype(float))
    strategy_annual = np.expm1(strategy_log.rolling(window).mean() * trading_days_per_year)
    benchmark_annual = np.expm1(benchmark_log.rolling(window).mean() * trading_days_per_year)
    rolling_excess = (strategy_annual - benchmark_annual).dropna()
    daily_cash = (1.0 + cash_annual_rate) ** (1.0 / trading_days_per_year) - 1.0
    active = strategy.astype(float) - daily_cash
    rolling_sharpe = (
        active.rolling(window).mean()
        / active.rolling(window).std(ddof=1)
        * np.sqrt(trading_days_per_year)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    return rolling_excess, rolling_sharpe


def _paired_block_bootstrap(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    repetitions: int,
    block_length: int,
    trading_days_per_year: int,
    cash_annual_rate: float,
    random_seed: int,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, Any]:
    left = strategy.to_numpy(dtype=float)
    right = benchmark.to_numpy(dtype=float)
    n = len(left)
    if n < block_length:
        raise ValueError("前瞻收益不足一个Bootstrap块")
    starts = np.arange(n - block_length + 1)
    block_count = int(np.ceil(n / block_length))
    rng = np.random.default_rng(random_seed)
    excess_values = np.empty(repetitions, dtype=float)
    sharpe_values = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        chosen = rng.choice(starts, size=block_count, replace=True)
        indices = np.concatenate([np.arange(start, start + block_length) for start in chosen])[:n]
        sampled_strategy = pd.Series(left[indices])
        sampled_benchmark = pd.Series(right[indices])
        excess_values[index] = annualized_return(
            sampled_strategy, trading_days_per_year
        ) - annualized_return(sampled_benchmark, trading_days_per_year)
        sharpe_values[index] = annualized_sharpe(
            sampled_strategy,
            trading_days_per_year=trading_days_per_year,
            cash_annual_rate=cash_annual_rate,
        )
    return {
        "repetitions": repetitions,
        "block_length_trading_days": block_length,
        "annualized_excess_median": float(np.median(excess_values)),
        "annualized_excess_interval_95pct": [
            float(np.quantile(excess_values, lower_quantile)),
            float(np.quantile(excess_values, upper_quantile)),
        ],
        "strategy_sharpe_median": float(np.median(sharpe_values)),
        "strategy_sharpe_interval_95pct": [
            float(np.quantile(sharpe_values, lower_quantile)),
            float(np.quantile(sharpe_values, upper_quantile)),
        ],
    }


def evaluate_forward_returns(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    *,
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    """按冻结40%超额与高夏普门槛评价前瞻收益。"""

    validate_contract(contract)
    data = _prepare_frame(frame, contract)
    maturity = contract["maturity"]
    objective = contract["objective"]
    n = len(data)
    result: dict[str, Any] = {
        "eligible_forward_day_count": n,
        "first_trade_date": None if data.empty else data["trade_date"].iloc[0].date().isoformat(),
        "last_trade_date": None if data.empty else data["trade_date"].iloc[-1].date().isoformat(),
        "goal_achieved": False,
        "metrics": None,
        "gates": None,
    }
    first_unseen = int(maturity["first_unseen_estimate_days"])
    if n < first_unseen:
        result["status"] = "PENDING_FIRST_UNSEEN_GATE"
        result["next_required_day_count"] = first_unseen
        return result

    trading_days = int(objective["annualization_trading_days"])
    cash_rate = float(objective["sharpe_cash_annual_rate"])
    benchmark = data["benchmark_total_return"]
    base = data["strategy_base_net_return"]
    stress = data["strategy_stress_net_return"]
    benchmark_cagr = annualized_return(benchmark, trading_days)
    base_cagr = annualized_return(base, trading_days)
    stress_cagr = annualized_return(stress, trading_days)
    base_excess = base_cagr - benchmark_cagr
    stress_excess = stress_cagr - benchmark_cagr
    base_sharpe = annualized_sharpe(
        base, trading_days_per_year=trading_days, cash_annual_rate=cash_rate
    )
    stress_sharpe = annualized_sharpe(
        stress, trading_days_per_year=trading_days, cash_annual_rate=cash_rate
    )
    rolling_window = int(objective["rolling_window_trading_days"])
    base_rolling_excess, base_rolling_sharpe = _rolling_metrics(
        base,
        benchmark,
        window=rolling_window,
        trading_days_per_year=trading_days,
        cash_annual_rate=cash_rate,
    )
    stress_rolling_excess, stress_rolling_sharpe = _rolling_metrics(
        stress,
        benchmark,
        window=rolling_window,
        trading_days_per_year=trading_days,
        cash_annual_rate=cash_rate,
    )

    bootstrap = deepcopy(objective["bootstrap"])
    repetitions = (
        int(bootstrap_repetitions_override)
        if bootstrap_repetitions_override is not None
        else int(bootstrap["repetitions"])
    )
    common_bootstrap = {
        "repetitions": repetitions,
        "block_length": int(bootstrap["block_length_trading_days"]),
        "trading_days_per_year": trading_days,
        "cash_annual_rate": cash_rate,
        "lower_quantile": float(bootstrap["lower_quantile"]),
        "upper_quantile": float(bootstrap["upper_quantile"]),
    }
    base_bootstrap = _paired_block_bootstrap(
        base,
        benchmark,
        random_seed=int(bootstrap["random_seed"]),
        **common_bootstrap,
    )
    stress_bootstrap = _paired_block_bootstrap(
        stress,
        benchmark,
        random_seed=int(bootstrap["random_seed"]) + 1,
        **common_bootstrap,
    )

    year_length = int(objective["non_overlapping_years"]["block_length_trading_days"])
    target = float(objective["minimum_annualized_excess"])
    sharpe_target = float(objective["minimum_strategy_net_sharpe"])
    year_blocks: list[dict[str, Any]] = []
    for start in range(0, n - year_length + 1, year_length):
        stop = start + year_length
        block_benchmark = annualized_return(benchmark.iloc[start:stop], trading_days)
        block_base = annualized_return(base.iloc[start:stop], trading_days)
        block_stress = annualized_return(stress.iloc[start:stop], trading_days)
        block_base_sharpe = annualized_sharpe(
            base.iloc[start:stop], trading_days_per_year=trading_days, cash_annual_rate=cash_rate
        )
        block_stress_sharpe = annualized_sharpe(
            stress.iloc[start:stop], trading_days_per_year=trading_days, cash_annual_rate=cash_rate
        )
        qualified = (
            block_base - block_benchmark >= target
            and block_stress - block_benchmark >= target
            and block_base_sharpe >= sharpe_target
            and block_stress_sharpe >= sharpe_target
        )
        year_blocks.append(
            {
                "start": data["trade_date"].iloc[start].date().isoformat(),
                "end": data["trade_date"].iloc[stop - 1].date().isoformat(),
                "base_annualized_excess": block_base - block_benchmark,
                "stress_annualized_excess": block_stress - block_benchmark,
                "base_sharpe": block_base_sharpe,
                "stress_sharpe": block_stress_sharpe,
                "target_qualified": qualified,
            }
        )
    qualified_blocks = sum(bool(item["target_qualified"]) for item in year_blocks)
    minimum_blocks = int(objective["non_overlapping_years"]["minimum_blocks_for_verified_claim"])
    minimum_qualified = int(objective["non_overlapping_years"]["minimum_target_qualified_blocks"])
    excess_floor = float(bootstrap["minimum_excess_lower_bound"])
    sharpe_floor = float(bootstrap["minimum_sharpe_lower_bound"])
    rolling_excess_target = float(objective["minimum_rolling_242d_excess_median"])
    rolling_sharpe_target = float(objective["minimum_rolling_242d_sharpe_median"])
    gates = {
        "base_annualized_excess_at_least_40pct": base_excess >= target,
        "stress_annualized_excess_at_least_40pct": stress_excess >= target,
        "base_strategy_sharpe_at_least_1_5": base_sharpe >= sharpe_target,
        "stress_strategy_sharpe_at_least_1_5": stress_sharpe >= sharpe_target,
        "base_rolling_excess_median_at_least_40pct": bool(
            not base_rolling_excess.empty and float(base_rolling_excess.median()) >= rolling_excess_target
        ),
        "stress_rolling_excess_median_at_least_40pct": bool(
            not stress_rolling_excess.empty
            and float(stress_rolling_excess.median()) >= rolling_excess_target
        ),
        "base_rolling_sharpe_median_at_least_1_5": bool(
            not base_rolling_sharpe.empty and float(base_rolling_sharpe.median()) >= rolling_sharpe_target
        ),
        "stress_rolling_sharpe_median_at_least_1_5": bool(
            not stress_rolling_sharpe.empty
            and float(stress_rolling_sharpe.median()) >= rolling_sharpe_target
        ),
        "base_bootstrap_excess_lower_bound_positive": base_bootstrap[
            "annualized_excess_interval_95pct"
        ][0]
        > excess_floor,
        "stress_bootstrap_excess_lower_bound_positive": stress_bootstrap[
            "annualized_excess_interval_95pct"
        ][0]
        > excess_floor,
        "base_bootstrap_sharpe_lower_bound_at_least_1": base_bootstrap[
            "strategy_sharpe_interval_95pct"
        ][0]
        >= sharpe_floor,
        "stress_bootstrap_sharpe_lower_bound_at_least_1": stress_bootstrap[
            "strategy_sharpe_interval_95pct"
        ][0]
        >= sharpe_floor,
        "minimum_non_overlapping_year_blocks": len(year_blocks) >= minimum_blocks,
        "minimum_target_qualified_year_blocks": qualified_blocks >= minimum_qualified,
        "verified_forward_day_count": n >= int(maturity["verified_claim_days"]),
    }
    result["metrics"] = {
        "benchmark_total_return_cagr": benchmark_cagr,
        "strategy_base_net_cagr": base_cagr,
        "strategy_stress_net_cagr": stress_cagr,
        "base_annualized_excess": base_excess,
        "stress_annualized_excess": stress_excess,
        "base_strategy_net_sharpe": base_sharpe,
        "stress_strategy_net_sharpe": stress_sharpe,
        "base_maximum_drawdown": maximum_drawdown(base),
        "stress_maximum_drawdown": maximum_drawdown(stress),
        "base_rolling_242d_excess_median": None
        if base_rolling_excess.empty
        else float(base_rolling_excess.median()),
        "stress_rolling_242d_excess_median": None
        if stress_rolling_excess.empty
        else float(stress_rolling_excess.median()),
        "base_rolling_242d_sharpe_median": None
        if base_rolling_sharpe.empty
        else float(base_rolling_sharpe.median()),
        "stress_rolling_242d_sharpe_median": None
        if stress_rolling_sharpe.empty
        else float(stress_rolling_sharpe.median()),
        "base_bootstrap": base_bootstrap,
        "stress_bootstrap": stress_bootstrap,
        "non_overlapping_years": year_blocks,
        "target_qualified_year_block_count": qualified_blocks,
    }
    result["gates"] = gates

    full_year = int(maturity["first_full_year_days"])
    verified_days = int(maturity["verified_claim_days"])
    if n < full_year:
        result["status"] = "PROVISIONAL_FIRST_UNSEEN_NOT_FULL_YEAR"
        result["next_required_day_count"] = full_year
    elif n < verified_days:
        result["status"] = (
            "ONE_YEAR_40PCT_HIGH_SHARPE_CANDIDATE_NOT_VERIFIED"
            if all(
                gates[key]
                for key in (
                    "base_annualized_excess_at_least_40pct",
                    "stress_annualized_excess_at_least_40pct",
                    "base_strategy_sharpe_at_least_1_5",
                    "stress_strategy_sharpe_at_least_1_5",
                    "base_rolling_excess_median_at_least_40pct",
                    "stress_rolling_excess_median_at_least_40pct",
                    "base_rolling_sharpe_median_at_least_1_5",
                    "stress_rolling_sharpe_median_at_least_1_5",
                )
            )
            else "CURRENT_ESTIMATE_BELOW_40PCT_OR_HIGH_SHARPE_GATE"
        )
        result["next_required_day_count"] = verified_days
    elif all(gates.values()):
        result["status"] = "VERIFIED_40PCT_EXCESS_HIGH_SHARPE"
        result["goal_achieved"] = True
        result["next_required_day_count"] = None
    else:
        result["status"] = "REJECTED_40PCT_EXCESS_HIGH_SHARPE"
        result["next_required_day_count"] = None
    return result
