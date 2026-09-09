"""M2增速加速度 × 510300长期趋势：冻结发现回测。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
import yaml

from backtest.engine import BacktestCosts, run_long_cash_backtest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "macro_01_m2_accel_trend_v1.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_total_return_series(market: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    result = market.copy().sort_values("date").reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["close"] = pd.to_numeric(result["close"], errors="raise")
    events = dividends.copy()
    events["ex_date"] = pd.to_datetime(events["ex_date"], errors="raise")
    events["cash_dividend_per_share"] = pd.to_numeric(events["cash_dividend_per_share"], errors="raise")
    cash_by_date = events.groupby("ex_date")["cash_dividend_per_share"].sum()
    result["cash_dividend_per_share"] = result["date"].map(cash_by_date).fillna(0.0)
    previous_close = result["close"].shift(1)
    result["total_return"] = ((result["close"] + result["cash_dividend_per_share"]) / previous_close - 1.0).fillna(0.0)
    result["total_return_index"] = (1.0 + result["total_return"]).cumprod()
    return result


def add_exact_m2_acceleration(money: pd.DataFrame, lag_months: int) -> pd.DataFrame:
    result = money.copy().sort_values("month").reset_index(drop=True)
    result["month"] = pd.to_datetime(result["month"], errors="raise")
    result["period"] = result["month"].dt.to_period("M")
    lag = result[["period", "m2_yoy_pct"]].copy()
    lag["period"] = lag["period"] + lag_months
    lag = lag.rename(columns={"m2_yoy_pct": "m2_yoy_lag_pct"})
    result = result.merge(lag, on="period", how="left", validate="one_to_one")
    result["m2_accel_pct_point"] = result["m2_yoy_pct"] - result["m2_yoy_lag_pct"]
    return result


def build_signal_table(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    money: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    factor = config["factor"]
    protocol = config["protocol"]
    execution = config["account_and_execution"]
    total_return = build_total_return_series(market, dividends)
    window = int(factor["trend_window_trading_days"])
    total_return["trend_sma"] = total_return["total_return_index"].rolling(window, min_periods=window).mean()
    total_return["trend_on"] = total_return["total_return_index"] > total_return["trend_sma"]
    accelerated = add_exact_m2_acceleration(money, int(factor["m2_lag_months"]))
    accelerated["availability_date"] = accelerated["period"].map(
        lambda period: pd.Timestamp(year=period.year, month=period.month, day=1)
        + pd.DateOffset(months=1, days=19)
    )

    dates = pd.DatetimeIndex(total_return["date"])
    rows: list[dict[str, Any]] = []
    for row in accelerated.dropna(subset=["m2_accel_pct_point"]).itertuples(index=False):
        decision_position = int(dates.searchsorted(pd.Timestamp(row.availability_date), side="left"))
        if decision_position >= len(dates) - 1:
            continue
        decision = total_return.iloc[decision_position]
        entry = total_return.iloc[decision_position + 1]
        rows.append(
            {
                "m2_period": str(row.period),
                "m2_month": pd.Timestamp(row.month),
                "m2_yoy_pct": float(row.m2_yoy_pct),
                "m2_yoy_lag_pct": float(row.m2_yoy_lag_pct),
                "m2_accel_pct_point": float(row.m2_accel_pct_point),
                "availability_date": pd.Timestamp(row.availability_date),
                "decision_date": pd.Timestamp(decision["date"]),
                "entry_date": pd.Timestamp(entry["date"]),
                "total_return_index": float(decision["total_return_index"]),
                "trend_sma": float(decision["trend_sma"]) if pd.notna(decision["trend_sma"]) else np.nan,
                "trend_on": bool(decision["trend_on"]),
            }
        )
    signals = pd.DataFrame(rows)
    start = pd.Timestamp(protocol["historical_evaluation_start"])
    end = pd.Timestamp(protocol["historical_evaluation_end"])
    signals = signals.loc[signals["decision_date"].between(start, end)].copy().reset_index(drop=True)
    if signals.empty or signals["decision_date"].duplicated().any():
        raise ValueError("宏观信号为空或决策日期重复")
    threshold = float(factor["m2_risk_on_threshold_pct_point"])
    signals["m2_on"] = signals["m2_accel_pct_point"] > threshold
    signals["combined_on"] = signals["m2_on"] & signals["trend_on"]
    high = float(execution["risk_on_target_exposure"])
    low = float(execution["risk_off_target_exposure"])
    signals["combined_target"] = np.where(signals["combined_on"], high, low)
    signals["trend_only_target"] = np.where(signals["trend_on"], high, low)
    signals["m2_only_target"] = np.where(signals["m2_on"], high, low)
    return signals


def build_targets(signals: pd.DataFrame, target_column: str) -> pd.DataFrame:
    result = signals[["decision_date", target_column]].rename(
        columns={"decision_date": "date", target_column: "target_position"}
    )
    result["signal_reason"] = target_column
    return result


def build_static_target(signals: pd.DataFrame, exposure: float, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": signals["decision_date"].iloc[0], "target_position": exposure, "signal_reason": label}]
    )


def build_costs(config: dict[str, Any], slippage_bps: float) -> BacktestCosts:
    execution = config["account_and_execution"]
    return BacktestCosts(
        commission_rate=float(execution["commission_rate"]),
        minimum_commission_cny=float(execution["minimum_commission_cny"]),
        stamp_duty_rate=float(execution["stamp_duty_rate"]),
        slippage_bps=float(slippage_bps),
        lot_size=int(execution["lot_size_shares"]),
        cash_annual_rate=float(execution["cash_annual_rate"]),
    )


def run_account_path(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    slippage_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    execution = config["account_and_execution"]
    protocol = config["protocol"]
    return run_long_cash_backtest(
        prices=market,
        dividends=dividends,
        targets=targets,
        initial_cash=float(execution["initial_capital_cny"]),
        costs=build_costs(config, slippage_bps),
        start_date=protocol["historical_evaluation_start"],
        end_date=protocol["historical_evaluation_end"],
        minimum_trade_notional_cny=float(execution["minimum_trade_notional_cny"]),
        minimum_trade_shares=int(execution["minimum_trade_shares"]),
    )


def summarize_path(ledger: pd.DataFrame, trades: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    execution = config["account_and_execution"]
    initial = float(execution["initial_capital_cny"])
    elapsed = max((ledger["date"].iloc[-1] - ledger["date"].iloc[0]).days, 1)
    total_return = float(ledger["equity"].iloc[-1] / initial - 1.0)
    cagr = float((1.0 + total_return) ** (365.25 / elapsed) - 1.0) if total_return > -1 else -1.0
    cash_daily = float(execution["cash_annual_rate"]) / float(execution["trading_days_per_year"])
    excess = ledger["daily_return"].astype(float) - cash_daily
    volatility = float(ledger["daily_return"].std(ddof=1) * np.sqrt(float(execution["trading_days_per_year"])))
    excess_sharpe = (
        float(excess.mean() * float(execution["trading_days_per_year"]) / volatility)
        if volatility > 0
        else None
    )
    return {
        "start_date": str(ledger["date"].iloc[0].date()),
        "end_date": str(ledger["date"].iloc[-1].date()),
        "observations": int(len(ledger)),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_excess_cash": excess_sharpe,
        "max_drawdown": float(ledger["drawdown"].min()),
        "average_exposure": float(ledger["actual_position"].mean()),
        "trade_count": int(len(trades)),
        "total_explicit_cost_cny": float(ledger["daily_explicit_cost_cny"].sum()),
        "total_slippage_cost_cny": float(ledger["daily_slippage_cost_cny"].sum()),
        "ending_equity": float(ledger["equity"].iloc[-1]),
    }


def build_signal_intervals(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    signals: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    prices = market.copy().sort_values("date").reset_index(drop=True)
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    opens = prices.set_index("date")["open"].astype(float)
    date_positions = {date: index for index, date in enumerate(prices["date"])}
    events = dividends.copy()
    events["ex_date"] = pd.to_datetime(events["ex_date"], errors="raise")
    cash_rate = float(config["account_and_execution"]["cash_annual_rate"])
    trading_days = float(config["account_and_execution"]["trading_days_per_year"])
    rows: list[dict[str, Any]] = []
    ordered = signals.sort_values("entry_date").reset_index(drop=True)
    for index in range(len(ordered) - 1):
        current = ordered.iloc[index]
        following = ordered.iloc[index + 1]
        start = pd.Timestamp(current["entry_date"])
        end = pd.Timestamp(following["entry_date"])
        dividend_cash = float(
            events.loc[events["ex_date"].gt(start) & events["ex_date"].le(end), "cash_dividend_per_share"].sum()
        )
        gross_return = float((opens.loc[end] + dividend_cash) / opens.loc[start] - 1.0)
        holding_days = int(date_positions[end] - date_positions[start])
        cash_return = float((1.0 + cash_rate / trading_days) ** holding_days - 1.0)
        rows.append(
            {
                **current.to_dict(),
                "exit_entry_date": end,
                "holding_trading_days": holding_days,
                "gross_etf_return": gross_return,
                "cash_return": cash_return,
                "future_excess_cash_return": gross_return - cash_return,
            }
        )
    return pd.DataFrame(rows)


def newey_west_regression(intervals: pd.DataFrame, lag: int) -> dict[str, Any]:
    frame = intervals.dropna(subset=["future_excess_cash_return", "m2_accel_pct_point", "trend_on"]).copy()
    y = frame["future_excess_cash_return"].to_numpy(dtype=float)
    x = np.column_stack(
        [np.ones(len(frame)), frame["trend_on"].astype(float), frame["m2_accel_pct_point"].astype(float)]
    )
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residuals = y - x @ beta
    scores = x * residuals[:, None]
    meat = scores.T @ scores
    for step in range(1, min(lag, len(frame) - 1) + 1):
        weight = 1.0 - step / (lag + 1.0)
        gamma = scores[step:].T @ scores[:-step]
        meat += weight * (gamma + gamma.T)
    covariance = inverse @ meat @ inverse
    standard_error = float(np.sqrt(max(covariance[2, 2], 0.0)))
    coefficient = float(beta[2])
    t_stat = coefficient / standard_error if standard_error > 0 else np.nan
    p_value = float(1.0 - norm.cdf(t_stat)) if np.isfinite(t_stat) else 1.0
    return {
        "observations": int(len(frame)),
        "coefficient_per_m2_pct_point": coefficient,
        "standard_error_hac": standard_error,
        "t_statistic": float(t_stat) if np.isfinite(t_stat) else None,
        "one_sided_p_value": p_value,
        "hac_lag_months": int(lag),
    }


def conditional_spread(frame: pd.DataFrame) -> dict[str, Any]:
    conditional = frame.loc[frame["trend_on"]].copy()
    high = conditional.loc[conditional["m2_on"], "future_excess_cash_return"].astype(float)
    low = conditional.loc[~conditional["m2_on"], "future_excess_cash_return"].astype(float)
    spread = float(high.mean() - low.mean()) if len(high) and len(low) else np.nan
    return {
        "trend_on_observations": int(len(conditional)),
        "m2_on_observations": int(len(high)),
        "m2_off_observations": int(len(low)),
        "m2_on_mean_future_excess": float(high.mean()) if len(high) else None,
        "m2_off_mean_future_excess": float(low.mean()) if len(low) else None,
        "spread": spread if np.isfinite(spread) else None,
    }


def circular_block_bootstrap_conditional_spread(
    intervals: pd.DataFrame,
    block_length: int,
    repetitions: int,
    random_seed: int,
) -> dict[str, Any]:
    frame = intervals.loc[intervals["trend_on"]].reset_index(drop=True)
    if len(frame) < block_length or frame["m2_on"].nunique() < 2:
        return {"valid_repetitions": 0, "lower_95": None, "median": None, "upper_95": None}
    rng = np.random.default_rng(random_seed)
    n = len(frame)
    estimates: list[float] = []
    blocks_needed = int(np.ceil(n / block_length))
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate([(start + np.arange(block_length)) % n for start in starts])[:n]
        sample = frame.iloc[indices]
        high = sample.loc[sample["m2_on"], "future_excess_cash_return"]
        low = sample.loc[~sample["m2_on"], "future_excess_cash_return"]
        if len(high) and len(low):
            estimates.append(float(high.mean() - low.mean()))
    if not estimates:
        return {"valid_repetitions": 0, "lower_95": None, "median": None, "upper_95": None}
    values = np.asarray(estimates, dtype=float)
    return {
        "valid_repetitions": int(len(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
        "block_length_months": int(block_length),
    }


def chronological_half_spreads(intervals: pd.DataFrame) -> list[dict[str, Any]]:
    midpoint = len(intervals) // 2
    results: list[dict[str, Any]] = []
    for label, frame in (("H1", intervals.iloc[:midpoint]), ("H2", intervals.iloc[midpoint:])):
        stats = conditional_spread(frame)
        results.append({"half": label, **stats, "positive": stats["spread"] is not None and stats["spread"] > 0})
    return results


def monthly_return_series(ledger: pd.DataFrame) -> pd.Series:
    frame = ledger[["date", "equity"]].copy()
    frame["period"] = pd.to_datetime(frame["date"]).dt.to_period("M")
    month_end = frame.groupby("period", sort=True)["equity"].last()
    return month_end.pct_change().dropna()


def profit_factor(returns: pd.Series) -> float | None:
    values = pd.Series(returns, dtype=float).dropna()
    positive = float(values.loc[values > 0].sum())
    negative = float(-values.loc[values < 0].sum())
    if negative <= 0:
        return None
    return positive / negative


def subperiod_comparison(
    combined: pd.DataFrame,
    static: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for period in config["evaluation"]["subperiods"]:
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        combined_slice = combined.loc[combined["date"].between(start, end)]
        static_slice = static.loc[static["date"].between(start, end)]
        if len(combined_slice) < 2 or len(static_slice) < 2:
            raise ValueError(f"子期数据不足：{period['id']}")
        combined_return = float(combined_slice["equity"].iloc[-1] / combined_slice["equity"].iloc[0] - 1.0)
        static_return = float(static_slice["equity"].iloc[-1] / static_slice["equity"].iloc[0] - 1.0)
        results.append(
            {
                "id": period["id"],
                "start": period["start"],
                "end": period["end"],
                "combined_return": combined_return,
                "static_60_return": static_return,
                "difference": combined_return - static_return,
                "positive": combined_return > static_return,
            }
        )
    return results


def maximum_year_positive_contribution(monthly_excess: pd.Series) -> tuple[float | None, dict[str, float]]:
    annual = monthly_excess.groupby(monthly_excess.index.year).sum()
    positive = annual.clip(lower=0.0)
    total = float(positive.sum())
    share = float(positive.max() / total) if total > 0 else None
    return share, {str(int(year)): float(value) for year, value in annual.items()}


def audit_and_load_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contracts = config["data_contracts"]
    for contract in contracts.values():
        path = ROOT / contract["file"]
        if not path.exists():
            raise FileNotFoundError(f"冻结输入缺失：{path}")
        actual_hash = sha256_file(path)
        if actual_hash != contract["sha256"]:
            raise ValueError(f"冻结输入哈希不匹配：{path}，实际{actual_hash}")

    market = pd.read_parquet(ROOT / contracts["etf_market"]["file"])
    dividends = pd.read_csv(ROOT / contracts["distributions"]["file"])
    money = pd.read_parquet(ROOT / contracts["money_supply"]["file"])
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    dividends["ex_date"] = pd.to_datetime(dividends["ex_date"], errors="raise")
    dividends["payment_date"] = pd.to_datetime(dividends["payment_date"], errors="raise")
    money["month"] = pd.to_datetime(money["month"], errors="raise")
    if len(market) != int(contracts["etf_market"]["required_rows"]):
        raise ValueError("510300行情行数不符合冻结合同")
    if str(market["date"].min().date()) != contracts["etf_market"]["required_first_date"]:
        raise ValueError("510300行情首日不符合冻结合同")
    if str(market["date"].max().date()) != contracts["etf_market"]["required_last_date"]:
        raise ValueError("510300行情末日不符合冻结合同")
    if len(dividends) != int(contracts["distributions"]["required_event_count"]):
        raise ValueError("分红事件数不符合冻结合同")
    if len(money) != int(contracts["money_supply"]["required_rows"]):
        raise ValueError("M2月度行数不符合冻结合同")
    cross = json.loads((ROOT / contracts["etf_market_cross_check"]["file"]).read_text(encoding="utf-8"))
    coverage = json.loads((ROOT / contracts["distribution_coverage"]["file"]).read_text(encoding="utf-8"))
    money_quality = json.loads((ROOT / contracts["money_supply_quality"]["file"]).read_text(encoding="utf-8"))
    if cross["status"] != contracts["etf_market_cross_check"]["required_status"]:
        raise ValueError("510300交叉源状态不通过")
    if not coverage["complete_history_confirmed"] or coverage["coverage_end"] < contracts["distribution_coverage"]["required_coverage_end"]:
        raise ValueError("510300分红完整性不通过")
    if money_quality["status"] != contracts["money_supply_quality"]["required_status"]:
        raise ValueError("M2数据质量状态不通过")
    checkpoint_passes = sum(bool(item["passed"]) for item in money_quality["official_checkpoint_results"])
    if checkpoint_passes != int(contracts["money_supply_quality"]["required_official_checkpoints_passed"]):
        raise ValueError("M2官方抽查数量不符合冻结合同")
    if bool(money_quality["formal_point_in_time_evidence"]) is not bool(
        contracts["money_supply_quality"]["required_formal_point_in_time_evidence"]
    ):
        raise ValueError("M2点时证据状态不符合冻结合同")
    audit = {
        "market_rows": int(len(market)),
        "market_first_date": str(market["date"].min().date()),
        "market_last_date": str(market["date"].max().date()),
        "market_cross_check_status": cross["status"],
        "dividend_events": int(len(dividends)),
        "dividend_coverage_end": coverage["coverage_end"],
        "money_rows": int(len(money)),
        "money_first_month": str(money["month"].min().date()),
        "money_last_month": str(money["month"].max().date()),
        "money_quality_status": money_quality["status"],
        "official_checkpoint_passes": checkpoint_passes,
        "formal_point_in_time_evidence": bool(money_quality["formal_point_in_time_evidence"]),
        "vintage_limitation": money_quality["known_methodology"]["vintage_limitation"],
    }
    return audit, market, dividends, money


def evaluate_protocol(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    audit, market, dividends, money = audit_and_load_inputs(config)
    signals = build_signal_table(market, dividends, money, config)
    intervals = build_signal_intervals(market, dividends, signals, config)
    execution = config["account_and_execution"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    base_slippage = float(execution["base_slippage_bps"])
    stress_slippage = float(execution["stress_slippage_bps"])

    target_sets = {
        "combined": build_targets(signals, "combined_target"),
        "trend_only": build_targets(signals, "trend_only_target"),
        "m2_only": build_targets(signals, "m2_only_target"),
        "static_60": build_static_target(signals, float(execution["static_60_exposure"]), "static_60"),
        "buy_hold": build_static_target(signals, float(execution["buy_hold_exposure"]), "buy_hold"),
    }
    paths: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, targets in target_sets.items():
        paths[f"base_{name}"] = run_account_path(market, dividends, targets, config, base_slippage)
        paths[f"stress_{name}"] = run_account_path(market, dividends, targets, config, stress_slippage)
    summaries = {name: summarize_path(ledger, trades, config) for name, (ledger, trades) in paths.items()}

    conditional = conditional_spread(intervals)
    hac = newey_west_regression(intervals, int(evaluation["hac_lag_months"]))
    bootstrap = circular_block_bootstrap_conditional_spread(
        intervals,
        int(evaluation["bootstrap_block_length_months"]),
        int(evaluation["bootstrap_repetitions"]),
        int(evaluation["random_seed"]),
    )
    halves = chronological_half_spreads(intervals)
    base_monthly_excess = monthly_return_series(paths["base_combined"][0]).sub(
        monthly_return_series(paths["base_static_60"][0]), fill_value=np.nan
    ).dropna()
    stress_monthly_excess = monthly_return_series(paths["stress_combined"][0]).sub(
        monthly_return_series(paths["stress_static_60"][0]), fill_value=np.nan
    ).dropna()
    base_pf = profit_factor(base_monthly_excess)
    stress_pf = profit_factor(stress_monthly_excess)
    if base_pf is None or stress_pf is None:
        raise RuntimeError("月度超额盈利因子无法计算")
    subperiods = subperiod_comparison(paths["base_combined"][0], paths["base_static_60"][0], config)
    maximum_year_share, annual_contributions = maximum_year_positive_contribution(base_monthly_excess)
    base_combined = summaries["base_combined"]
    stress_combined = summaries["stress_combined"]
    base_trend = summaries["base_trend_only"]
    base_static = summaries["base_static_60"]
    stress_static = summaries["stress_static_60"]
    sharpe_delta_trend = float(base_combined["sharpe_excess_cash"] - base_trend["sharpe_excess_cash"])
    base_cagr_delta_static = float(base_combined["cagr"] - base_static["cagr"])
    stress_cagr_delta_static = float(stress_combined["cagr"] - stress_static["cagr"])

    group_minimum = int(evaluation["minimum_conditional_group_observations"])
    gate_checks = {
        "complete_signal_intervals": len(intervals) >= int(evaluation["minimum_complete_signal_intervals"]),
        "conditional_group_support": conditional["m2_on_observations"] >= group_minimum and conditional["m2_off_observations"] >= group_minimum,
        "conditional_spread_positive": conditional["spread"] is not None and conditional["spread"] > float(gates["conditional_spread_minimum"]),
        "conditional_bootstrap_lower_positive": bootstrap["lower_95"] is not None and bootstrap["lower_95"] > float(gates["conditional_bootstrap_lower_95_minimum"]),
        "hac_m2_effect": hac["coefficient_per_m2_pct_point"] > 0 and hac["one_sided_p_value"] <= float(gates["hac_one_sided_p_maximum"]),
        "chronological_halves": sum(bool(item["positive"]) for item in halves) >= int(gates["chronological_halves_positive_minimum"]),
        "base_strategy_sharpe": base_combined["sharpe_excess_cash"] >= float(gates["base_strategy_sharpe_minimum"]),
        "stress_strategy_sharpe": stress_combined["sharpe_excess_cash"] >= float(gates["stress_strategy_sharpe_minimum"]),
        "base_monthly_excess_profit_factor": base_pf >= float(gates["base_monthly_excess_profit_factor_minimum"]),
        "stress_monthly_excess_profit_factor": stress_pf >= float(gates["stress_monthly_excess_profit_factor_minimum"]),
        "sharpe_improvement_vs_trend_only": sharpe_delta_trend >= float(gates["sharpe_improvement_vs_trend_only_minimum"]),
        "base_cagr_vs_static_60": base_cagr_delta_static > float(gates["base_cagr_difference_vs_static_60_minimum"]),
        "stress_cagr_vs_static_60": stress_cagr_delta_static > float(gates["stress_cagr_difference_vs_static_60_minimum"]),
        "positive_subperiods": sum(bool(item["positive"]) for item in subperiods) >= int(gates["positive_subperiods_minimum"]),
        "year_contribution_concentration": maximum_year_share is not None and maximum_year_share <= float(gates["maximum_single_year_positive_contribution_share"]),
    }
    passed = all(gate_checks.values())
    failed_gates = [name for name, value in gate_checks.items() if not value]
    decision = "DISCOVERY_PASS_REQUIRES_VINTAGE_RECONSTRUCTION" if passed else "REJECT_DISCOVERY_STOP_NO_RESCUE"
    report = {
        "project_id": config["protocol"]["project_id"],
        "decision": decision,
        "decision_explanation": (
            "全部冻结门槛通过，但M2是当前历史版本；只允许重建逐月公告快照，不授权Paper或实盘。"
            if passed
            else f"冻结门槛失败：{', '.join(failed_gates)}。本版本停止，禁止改窗口、均线、暴露或样本起点补救。"
        ),
        "failed_gates": failed_gates,
        "data_audit": audit,
        "signal_summary": {
            "signals": int(len(signals)),
            "complete_intervals": int(len(intervals)),
            "first_decision_date": str(signals["decision_date"].min().date()),
            "last_decision_date": str(signals["decision_date"].max().date()),
            "latest_m2_period_used": str(signals["m2_period"].iloc[-1]),
            "combined_risk_on_signals": int(signals["combined_on"].sum()),
            "combined_risk_off_signals": int((~signals["combined_on"]).sum()),
        },
        "predictive_statistics": {
            "conditional_on_trend": conditional,
            "hac_regression": hac,
            "block_bootstrap": bootstrap,
            "chronological_halves": halves,
        },
        "account_paths": summaries,
        "comparisons": {
            "base_monthly_excess_profit_factor_vs_static_60": float(base_pf),
            "stress_monthly_excess_profit_factor_vs_static_60": float(stress_pf),
            "base_sharpe_improvement_vs_trend_only": sharpe_delta_trend,
            "base_cagr_difference_vs_static_60": base_cagr_delta_static,
            "stress_cagr_difference_vs_static_60": stress_cagr_delta_static,
            "subperiods": subperiods,
            "positive_subperiod_count": int(sum(bool(item["positive"]) for item in subperiods)),
            "annual_monthly_excess_contributions": annual_contributions,
            "maximum_single_year_positive_contribution_share": maximum_year_share,
        },
        "gates": gate_checks,
        "governance": {
            "external_macro_signal_used": True,
            "isolated_expansion_research": True,
            "formal_point_in_time_evidence": False,
            "minute_data_used": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }
    artifacts = {
        "base_ledger": paths["base_combined"][0],
        "base_trades": paths["base_combined"][1],
        "stress_ledger": paths["stress_combined"][0],
        "stress_trades": paths["stress_combined"][1],
        "signals": signals,
        "intervals": intervals,
    }
    return report, artifacts


def markdown_report(report: dict[str, Any]) -> str:
    paths = report["account_paths"]
    comparisons = report["comparisons"]
    predictive = report["predictive_statistics"]
    conditional = predictive["conditional_on_trend"]
    lines = [
        "# 510300 MACRO-01：M2增速加速度 × 长期趋势确认",
        "",
        f"- 决策：`{report['decision']}`",
        f"- 说明：{report['decision_explanation']}",
        f"- 信号数/完整区间：{report['signal_summary']['signals']}/{report['signal_summary']['complete_intervals']}",
        f"- 使用M2月份：截至`{report['signal_summary']['latest_m2_period_used']}`；执行日线，无分钟线。",
        "",
        "## 数据边界",
        "",
        f"- 510300行情：{report['data_audit']['market_first_date']}至{report['data_audit']['market_last_date']}，交叉源`{report['data_audit']['market_cross_check_status']}`。",
        f"- M2：{report['data_audit']['money_first_month']}至{report['data_audit']['money_last_month']}，官方抽查通过{report['data_audit']['official_checkpoint_passes']}项。",
        f"- 点时限制：`formal_point_in_time_evidence={str(report['data_audit']['formal_point_in_time_evidence']).lower()}`。{report['data_audit']['vintage_limitation']}",
        "",
        "## 预测层",
        "",
        f"- 趋势开启条件下，M2加速/未加速样本：{conditional['m2_on_observations']}/{conditional['m2_off_observations']}。",
        f"- 下一完整区间超现金收益均值差：{conditional['spread']:.4%}。" if conditional["spread"] is not None else "- 下一完整区间均值差：无法计算。",
        f"- 6个月区块Bootstrap 95%区间：[{predictive['block_bootstrap']['lower_95']:.4%}, {predictive['block_bootstrap']['upper_95']:.4%}]。" if predictive["block_bootstrap"]["lower_95"] is not None else "- Bootstrap：无法计算。",
        f"- HAC中M2系数：{predictive['hac_regression']['coefficient_per_m2_pct_point']:.6f}，单侧p={predictive['hac_regression']['one_sided_p_value']:.4f}。",
        "",
        "## 账户回测",
        "",
        "| 路径 | CAGR | 超现金夏普 | 最大回撤 | 交易数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("base_combined", "组合-基础成本"),
        ("stress_combined", "组合-压力成本"),
        ("base_trend_only", "仅趋势-基础成本"),
        ("base_m2_only", "仅M2-基础成本"),
        ("base_static_60", "静态60%-基础成本"),
        ("base_buy_hold", "98.5%持有-基础成本"),
    ):
        item = paths[key]
        lines.append(
            f"| {label} | {item['cagr']:.2%} | {item['sharpe_excess_cash']:.3f} | {item['max_drawdown']:.2%} | {item['trade_count']} |"
        )
    lines.extend(
        [
            "",
            f"- 基础/压力月度超额盈利因子（相对静态60%）：{comparisons['base_monthly_excess_profit_factor_vs_static_60']:.3f}/{comparisons['stress_monthly_excess_profit_factor_vs_static_60']:.3f}。",
            f"- 基础夏普相对仅趋势增量：{comparisons['base_sharpe_improvement_vs_trend_only']:.3f}。",
            f"- 基础/压力CAGR相对静态60%：{comparisons['base_cagr_difference_vs_static_60']:.2%}/{comparisons['stress_cagr_difference_vs_static_60']:.2%}。",
            "",
            "## 门槛",
            "",
        ]
    )
    for name, passed in report["gates"].items():
        lines.append(f"- {'通过' if passed else '失败'}：`{name}`")
    lines.extend(
        [
            "",
            "## 治理结论",
            "",
            "本报告不修改现行510300-only系统，不启用仓位映射，不连接券商，不生成订单。通过也只能进入逐月历史公告快照重建；失败则本版本冻结停止。",
            "",
        ]
    )
    return "\n".join(lines)


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法序列化：{type(value)!r}")
