"""以沪深300全收益指数为主基准，计算策略的净超额表现。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _compound(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def evaluate_relative_ledger(
    ledger: pd.DataFrame,
    benchmark: pd.DataFrame,
    initial_cash: float,
    trading_days_per_year: int = 242,
    rolling_window: int = 242,
) -> dict[str, Any]:
    """计算绝对收益、主动收益、信息比率与相对净值回撤。"""

    required_ledger = {"date", "equity"}
    required_benchmark = {"date", "close"}
    if missing := required_ledger.difference(ledger.columns):
        raise ValueError(f"策略账本缺少字段：{sorted(missing)}")
    if missing := required_benchmark.difference(benchmark.columns):
        raise ValueError(f"基准数据缺少字段：{sorted(missing)}")
    if initial_cash <= 0 or trading_days_per_year <= 0 or rolling_window < 2:
        raise ValueError("初始资金、年交易日和滚动窗口必须为正")

    strategy = ledger[["date", "equity"]].copy()
    strategy["date"] = pd.to_datetime(strategy["date"]).dt.normalize()
    strategy["equity"] = pd.to_numeric(strategy["equity"], errors="coerce")
    market = benchmark[["date", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    data = strategy.merge(market, on="date", how="inner", validate="one_to_one")
    data = data.sort_values("date").reset_index(drop=True)
    if len(data) < 2 or data[["equity", "close"]].isna().any().any():
        raise ValueError("策略与基准没有足够的有效重叠数据")
    if (data[["equity", "close"]] <= 0).any().any():
        raise ValueError("策略权益和基准收盘必须为正")

    data["strategy_return"] = data["equity"].pct_change(fill_method=None).fillna(0.0)
    data["benchmark_return"] = data["close"].pct_change(fill_method=None).fillna(0.0)
    data["active_return"] = data["strategy_return"] - data["benchmark_return"]
    data["relative_wealth"] = (
        data["equity"] / initial_cash
    ) / (data["close"] / data["close"].iloc[0])
    data["relative_peak"] = data["relative_wealth"].cummax()
    data["relative_drawdown"] = data["relative_wealth"] / data["relative_peak"] - 1.0

    elapsed_days = max(int((data["date"].iloc[-1] - data["date"].iloc[0]).days), 1)
    elapsed_years = elapsed_days / 365.25
    strategy_total = float(data["equity"].iloc[-1] / initial_cash - 1.0)
    benchmark_total = float(data["close"].iloc[-1] / data["close"].iloc[0] - 1.0)
    strategy_cagr = float((1.0 + strategy_total) ** (1.0 / elapsed_years) - 1.0)
    benchmark_cagr = float((1.0 + benchmark_total) ** (1.0 / elapsed_years) - 1.0)
    tracking_error = float(data["active_return"].std(ddof=1) * np.sqrt(trading_days_per_year))
    information_ratio = (
        float(data["active_return"].mean() * trading_days_per_year / tracking_error)
        if tracking_error > 0
        else None
    )

    strategy_rolling = (
        (1.0 + data["strategy_return"])
        .rolling(rolling_window, min_periods=rolling_window)
        .apply(np.prod, raw=True)
        - 1.0
    )
    benchmark_rolling = (
        (1.0 + data["benchmark_return"])
        .rolling(rolling_window, min_periods=rolling_window)
        .apply(np.prod, raw=True)
        - 1.0
    )
    rolling_excess = (strategy_rolling - benchmark_rolling).dropna()

    halves: list[dict[str, Any]] = []
    midpoint = len(data) // 2
    for name, frame in [("前半段", data.iloc[:midpoint]), ("后半段", data.iloc[midpoint:])]:
        strategy_return = _compound(frame["strategy_return"])
        benchmark_return = _compound(frame["benchmark_return"])
        halves.append(
            {
                "period": name,
                "start_date": str(frame["date"].iloc[0].date()),
                "end_date": str(frame["date"].iloc[-1].date()),
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": strategy_return - benchmark_return,
            }
        )

    annual: list[dict[str, Any]] = []
    for year, frame in data.groupby(data["date"].dt.year, sort=True):
        strategy_return = _compound(frame["strategy_return"])
        benchmark_return = _compound(frame["benchmark_return"])
        annual.append(
            {
                "year": int(year),
                "observations": int(len(frame)),
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": strategy_return - benchmark_return,
            }
        )

    return {
        "start_date": str(data["date"].iloc[0].date()),
        "end_date": str(data["date"].iloc[-1].date()),
        "observations": int(len(data)),
        "elapsed_years": float(elapsed_years),
        "ending_equity_cny": float(data["equity"].iloc[-1]),
        "strategy_total_return": strategy_total,
        "benchmark_total_return": benchmark_total,
        "total_excess_return": strategy_total - benchmark_total,
        "strategy_cagr": strategy_cagr,
        "benchmark_cagr": benchmark_cagr,
        "annualized_excess": strategy_cagr - benchmark_cagr,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "maximum_relative_drawdown": float(data["relative_drawdown"].min()),
        "rolling_excess": {
            "window_trading_days": int(rolling_window),
            "observations": int(len(rolling_excess)),
            "median": float(rolling_excess.median()),
            "positive_ratio": float(rolling_excess.gt(0.0).mean()),
            "minimum": float(rolling_excess.min()),
            "maximum": float(rolling_excess.max()),
        },
        "chronological_halves": halves,
        "annual_returns": annual,
    }


def evaluate_objective_gates(
    base: dict[str, Any],
    stress: dict[str, Any] | None,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """按照冻结门槛判断历史结果，缺少压力账本时不得通过。"""

    checks = {
        "minimum_evaluation_years": base["elapsed_years"]
        >= float(gates["minimum_evaluation_years"]),
        "base_annualized_excess_positive": base["annualized_excess"]
        > float(gates["minimum_base_annualized_excess"]),
        "stress_annualized_excess_positive": stress is not None
        and stress["annualized_excess"] > float(gates["minimum_stress_annualized_excess"]),
        "both_chronological_halves_positive": all(
            item["excess_return"] > 0.0 for item in base["chronological_halves"]
        ),
        "rolling_excess_median_positive": base["rolling_excess"]["median"]
        > float(gates["minimum_rolling_excess_median"]),
        "rolling_positive_ratio": base["rolling_excess"]["positive_ratio"]
        >= float(gates["minimum_rolling_positive_ratio"]),
        "information_ratio_positive": base["information_ratio"] is not None
        and base["information_ratio"] > float(gates["minimum_information_ratio"]),
        "relative_drawdown_within_limit": base["maximum_relative_drawdown"]
        >= float(gates["maximum_relative_drawdown"]),
    }
    return {
        "status": (
            "RETROSPECTIVE_PASS_FORWARD_ONLY"
            if all(checks.values())
            else "REJECTED_OR_INCOMPLETE"
        ),
        "checks": checks,
        "passed_count": int(sum(checks.values())),
        "total_count": int(len(checks)),
    }
