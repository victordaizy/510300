"""AAA信用利差压缩 × 510300长期趋势：冻结发现回测。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
import yaml

from research.daily_macro_01_m2_accel_trend_v1 import (
    build_signal_intervals,
    build_static_target,
    build_targets,
    build_total_return_series,
    maximum_year_positive_contribution,
    monthly_return_series,
    profit_factor,
    run_account_path,
    subperiod_comparison,
    summarize_path,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "macro_02_credit_spread_trend_v1.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def add_credit_compression(
    credit: pd.DataFrame, lookback_observations: int
) -> pd.DataFrame:
    """按曲线观测序列计算精确63期信用利差压缩量。"""

    result = credit.copy().sort_values("date").reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["credit_spread_3y_bp"] = pd.to_numeric(
        result["credit_spread_3y_bp"], errors="raise"
    )
    result["credit_spread_lag_bp"] = result["credit_spread_3y_bp"].shift(
        lookback_observations
    )
    result["compression_impulse_bp"] = (
        result["credit_spread_lag_bp"] - result["credit_spread_3y_bp"]
    )
    return result


def build_signal_table(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    credit: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """构造月末17:30后信号，并绑定下一510300交易日开盘。"""

    factor = config["factor"]
    protocol = config["protocol"]
    execution = config["account_and_execution"]
    total_return = build_total_return_series(market, dividends)
    trend_window = int(factor["trend_window_trading_days"])
    total_return["trend_sma"] = total_return["total_return_index"].rolling(
        trend_window, min_periods=trend_window
    ).mean()
    total_return["trend_on"] = total_return["total_return_index"] > total_return["trend_sma"]

    compressed = add_credit_compression(
        credit, int(factor["compression_lookback_curve_observations"])
    )
    joined = total_return.merge(
        compressed[
            [
                "date",
                "cgb_3y",
                "cpnote_aaa_3y",
                "credit_spread_3y_bp",
                "credit_spread_lag_bp",
                "compression_impulse_bp",
            ]
        ],
        on="date",
        how="inner",
        validate="one_to_one",
    ).sort_values("date")
    joined["calendar_month"] = joined["date"].dt.to_period("M")
    month_end = joined.groupby("calendar_month", sort=True, as_index=False).tail(1).copy()
    month_end = month_end.dropna(subset=["trend_sma", "compression_impulse_bp"])

    market_dates = pd.DatetimeIndex(pd.to_datetime(market["date"], errors="raise").sort_values())
    entry_dates: list[pd.Timestamp] = []
    for decision_date in month_end["date"]:
        entry_position = int(market_dates.searchsorted(pd.Timestamp(decision_date), side="right"))
        entry_dates.append(
            market_dates[entry_position] if entry_position < len(market_dates) else pd.NaT
        )
    month_end["decision_date"] = month_end["date"]
    month_end["entry_date"] = entry_dates
    month_end = month_end.dropna(subset=["entry_date"])

    start = pd.Timestamp(protocol["historical_evaluation_start"])
    end = pd.Timestamp(protocol["historical_evaluation_end"])
    signals = month_end.loc[month_end["decision_date"].between(start, end)].copy()
    signals = signals[
        [
            "calendar_month",
            "decision_date",
            "entry_date",
            "cgb_3y",
            "cpnote_aaa_3y",
            "credit_spread_3y_bp",
            "credit_spread_lag_bp",
            "compression_impulse_bp",
            "total_return_index",
            "trend_sma",
            "trend_on",
        ]
    ].reset_index(drop=True)
    if signals.empty or signals["decision_date"].duplicated().any():
        raise ValueError("信用利差月度信号为空或决策日期重复")
    if not signals["calendar_month"].is_unique:
        raise ValueError("同一自然月出现多个信用利差信号")

    threshold = float(factor["compression_risk_on_threshold_bp"])
    signals["compression_on"] = signals["compression_impulse_bp"] > threshold
    signals["combined_on"] = signals["compression_on"] & signals["trend_on"]
    high = float(execution["risk_on_target_exposure"])
    low = float(execution["risk_off_target_exposure"])
    signals["combined_target"] = np.where(signals["combined_on"], high, low)
    signals["trend_only_target"] = np.where(signals["trend_on"], high, low)
    signals["compression_only_target"] = np.where(signals["compression_on"], high, low)
    return signals


def newey_west_regression(intervals: pd.DataFrame, lag: int) -> dict[str, Any]:
    """回归下一月超现金收益对趋势和每1bp信用压缩的敏感度。"""

    frame = intervals.dropna(
        subset=["future_excess_cash_return", "compression_impulse_bp", "trend_on"]
    ).copy()
    y = frame["future_excess_cash_return"].to_numpy(dtype=float)
    x = np.column_stack(
        [
            np.ones(len(frame)),
            frame["trend_on"].astype(float),
            frame["compression_impulse_bp"].astype(float),
        ]
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
        "coefficient_per_compression_bp": coefficient,
        "standard_error_hac": standard_error,
        "t_statistic": float(t_stat) if np.isfinite(t_stat) else None,
        "one_sided_p_value": p_value,
        "hac_lag_months": int(lag),
    }


def conditional_spread(frame: pd.DataFrame) -> dict[str, Any]:
    conditional = frame.loc[frame["trend_on"]].copy()
    high = conditional.loc[
        conditional["compression_on"], "future_excess_cash_return"
    ].astype(float)
    low = conditional.loc[
        ~conditional["compression_on"], "future_excess_cash_return"
    ].astype(float)
    spread = float(high.mean() - low.mean()) if len(high) and len(low) else np.nan
    return {
        "trend_on_observations": int(len(conditional)),
        "compression_on_observations": int(len(high)),
        "compression_off_observations": int(len(low)),
        "compression_on_mean_future_excess": float(high.mean()) if len(high) else None,
        "compression_off_mean_future_excess": float(low.mean()) if len(low) else None,
        "spread": spread if np.isfinite(spread) else None,
    }


def circular_block_bootstrap_conditional_spread(
    intervals: pd.DataFrame,
    block_length: int,
    repetitions: int,
    random_seed: int,
) -> dict[str, Any]:
    frame = intervals.loc[intervals["trend_on"]].reset_index(drop=True)
    if len(frame) < block_length or frame["compression_on"].nunique() < 2:
        return {"valid_repetitions": 0, "lower_95": None, "median": None, "upper_95": None}
    rng = np.random.default_rng(random_seed)
    n = len(frame)
    blocks_needed = int(np.ceil(n / block_length))
    estimates: list[float] = []
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate(
            [(start + np.arange(block_length)) % n for start in starts]
        )[:n]
        sample = frame.iloc[indices]
        high = sample.loc[
            sample["compression_on"], "future_excess_cash_return"
        ]
        low = sample.loc[
            ~sample["compression_on"], "future_excess_cash_return"
        ]
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
        statistics = conditional_spread(frame)
        results.append(
            {
                "half": label,
                **statistics,
                "positive": statistics["spread"] is not None and statistics["spread"] > 0,
            }
        )
    return results


def audit_and_load_inputs(
    config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    credit = pd.read_parquet(ROOT / contracts["credit_curve"]["file"])
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    dividends["ex_date"] = pd.to_datetime(dividends["ex_date"], errors="raise")
    dividends["payment_date"] = pd.to_datetime(dividends["payment_date"], errors="raise")
    credit["date"] = pd.to_datetime(credit["date"], errors="raise")

    market_contract = contracts["etf_market"]
    credit_contract = contracts["credit_curve"]
    if len(market) != int(market_contract["required_rows"]):
        raise ValueError("510300行情行数不符合冻结合同")
    if str(market["date"].min().date()) != market_contract["required_first_date"]:
        raise ValueError("510300行情首日不符合冻结合同")
    if str(market["date"].max().date()) != market_contract["required_last_date"]:
        raise ValueError("510300行情末日不符合冻结合同")
    if len(dividends) != int(contracts["distributions"]["required_event_count"]):
        raise ValueError("分红事件数不符合冻结合同")
    if len(credit) != int(credit_contract["required_rows"]):
        raise ValueError("信用曲线行数不符合冻结合同")
    if str(credit["date"].min().date()) != credit_contract["required_first_date"]:
        raise ValueError("信用曲线首日不符合冻结合同")
    if str(credit["date"].max().date()) != credit_contract["required_last_date"]:
        raise ValueError("信用曲线末日不符合冻结合同")

    cross = json.loads(
        (ROOT / contracts["etf_market_cross_check"]["file"]).read_text(encoding="utf-8")
    )
    coverage = json.loads(
        (ROOT / contracts["distribution_coverage"]["file"]).read_text(encoding="utf-8")
    )
    quality = json.loads(
        (ROOT / contracts["credit_curve_quality"]["file"]).read_text(encoding="utf-8")
    )
    if cross["status"] != contracts["etf_market_cross_check"]["required_status"]:
        raise ValueError("510300交叉源状态不通过")
    if int(cross["overlap_rows"]) != int(
        contracts["etf_market_cross_check"]["required_overlap_rows"]
    ):
        raise ValueError("510300交叉源重叠行数不符合冻结合同")
    if not coverage["complete_history_confirmed"] or coverage["coverage_end"] < contracts[
        "distribution_coverage"
    ]["required_coverage_end"]:
        raise ValueError("510300分红完整性不通过")
    quality_contract = contracts["credit_curve_quality"]
    if quality["status"] != quality_contract["required_status"]:
        raise ValueError("信用曲线质量状态不通过")
    checkpoint_passes = sum(bool(item["passed"]) for item in quality["official_checkpoint_results"])
    if checkpoint_passes != int(quality_contract["required_official_checkpoints_passed"]):
        raise ValueError("信用曲线官方检查点数量不符合冻结合同")
    if bool(quality["existing_government_batch_cross_check"]["passed"]) is not bool(
        quality_contract["required_existing_government_cross_check_passed"]
    ):
        raise ValueError("既有国债批次一致性核对不符合冻结合同")

    audit = {
        "market_rows": int(len(market)),
        "market_first_date": str(market["date"].min().date()),
        "market_last_date": str(market["date"].max().date()),
        "market_cross_check_status": cross["status"],
        "dividend_events": int(len(dividends)),
        "dividend_coverage_end": coverage["coverage_end"],
        "credit_curve_rows": int(len(credit)),
        "credit_curve_first_date": str(credit["date"].min().date()),
        "credit_curve_last_date": str(credit["date"].max().date()),
        "credit_spread_min_bp": float(credit["credit_spread_3y_bp"].min()),
        "credit_spread_max_bp": float(credit["credit_spread_3y_bp"].max()),
        "credit_quality_status": quality["status"],
        "official_checkpoint_passes": checkpoint_passes,
        "existing_government_overlap_rows": int(
            quality["existing_government_batch_cross_check"]["overlap_rows"]
        ),
        "tls_verified_rows": int(quality["tls_verified_rows"]),
        "tls_unverified_rows": int(quality["tls_unverified_rows"]),
        "independent_supplier_validation": False,
        "transport_limitation": quality["transport_limitation"],
    }
    return audit, market, dividends, credit


def evaluate_protocol(
    config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    audit, market, dividends, credit = audit_and_load_inputs(config)
    signals = build_signal_table(market, dividends, credit, config)
    intervals = build_signal_intervals(market, dividends, signals, config)
    execution = config["account_and_execution"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    base_slippage = float(execution["base_slippage_bps"])
    stress_slippage = float(execution["stress_slippage_bps"])

    target_sets = {
        "combined": build_targets(signals, "combined_target"),
        "trend_only": build_targets(signals, "trend_only_target"),
        "compression_only": build_targets(signals, "compression_only_target"),
        "static_60": build_static_target(
            signals, float(execution["static_60_exposure"]), "static_60"
        ),
        "buy_hold": build_static_target(
            signals, float(execution["buy_hold_exposure"]), "buy_hold"
        ),
    }
    paths: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, targets in target_sets.items():
        paths[f"base_{name}"] = run_account_path(
            market, dividends, targets, config, base_slippage
        )
        paths[f"stress_{name}"] = run_account_path(
            market, dividends, targets, config, stress_slippage
        )
    summaries = {
        name: summarize_path(ledger, trades, config)
        for name, (ledger, trades) in paths.items()
    }

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
    subperiods = subperiod_comparison(
        paths["base_combined"][0], paths["base_static_60"][0], config
    )
    maximum_year_share, annual_contributions = maximum_year_positive_contribution(
        base_monthly_excess
    )
    base_combined = summaries["base_combined"]
    stress_combined = summaries["stress_combined"]
    base_trend = summaries["base_trend_only"]
    base_static = summaries["base_static_60"]
    stress_static = summaries["stress_static_60"]
    sharpe_delta_trend = float(
        base_combined["sharpe_excess_cash"] - base_trend["sharpe_excess_cash"]
    )
    base_cagr_delta_static = float(base_combined["cagr"] - base_static["cagr"])
    stress_cagr_delta_static = float(stress_combined["cagr"] - stress_static["cagr"])

    group_minimum = int(evaluation["minimum_conditional_group_observations"])
    gate_checks = {
        "complete_signal_intervals": len(intervals)
        >= int(evaluation["minimum_complete_signal_intervals"]),
        "conditional_group_support": conditional["compression_on_observations"]
        >= group_minimum
        and conditional["compression_off_observations"] >= group_minimum,
        "conditional_spread_positive": conditional["spread"] is not None
        and conditional["spread"] > float(gates["conditional_spread_minimum"]),
        "conditional_bootstrap_lower_positive": bootstrap["lower_95"] is not None
        and bootstrap["lower_95"]
        > float(gates["conditional_bootstrap_lower_95_minimum"]),
        "hac_compression_effect": hac["coefficient_per_compression_bp"] > 0
        and hac["one_sided_p_value"] <= float(gates["hac_one_sided_p_maximum"]),
        "chronological_halves": sum(bool(item["positive"]) for item in halves)
        >= int(gates["chronological_halves_positive_minimum"]),
        "base_strategy_sharpe": base_combined["sharpe_excess_cash"]
        >= float(gates["base_strategy_sharpe_minimum"]),
        "stress_strategy_sharpe": stress_combined["sharpe_excess_cash"]
        >= float(gates["stress_strategy_sharpe_minimum"]),
        "base_monthly_excess_profit_factor": base_pf
        >= float(gates["base_monthly_excess_profit_factor_minimum"]),
        "stress_monthly_excess_profit_factor": stress_pf
        >= float(gates["stress_monthly_excess_profit_factor_minimum"]),
        "sharpe_improvement_vs_trend_only": sharpe_delta_trend
        >= float(gates["sharpe_improvement_vs_trend_only_minimum"]),
        "base_cagr_vs_static_60": base_cagr_delta_static
        > float(gates["base_cagr_difference_vs_static_60_minimum"]),
        "stress_cagr_vs_static_60": stress_cagr_delta_static
        > float(gates["stress_cagr_difference_vs_static_60_minimum"]),
        "positive_subperiods": sum(bool(item["positive"]) for item in subperiods)
        >= int(gates["positive_subperiods_minimum"]),
        "year_contribution_concentration": maximum_year_share is not None
        and maximum_year_share
        <= float(gates["maximum_single_year_positive_contribution_share"]),
    }
    passed = all(gate_checks.values())
    failed_gates = [name for name, value in gate_checks.items() if not value]
    decision = (
        "DISCOVERY_PASS_REQUIRES_INDEPENDENT_SOURCE_VALIDATION"
        if passed
        else "REJECT_DISCOVERY_STOP_NO_RESCUE"
    )
    report = {
        "project_id": config["protocol"]["project_id"],
        "decision": decision,
        "decision_explanation": (
            "全部冻结门槛通过；只允许补做独立数据源核对和前向Shadow协议，不授权Paper仓位或实盘。"
            if passed
            else f"冻结门槛失败：{', '.join(failed_gates)}。本版本停止，禁止改期限、窗口、均线、暴露或样本起点补救。"
        ),
        "failed_gates": failed_gates,
        "data_audit": audit,
        "signal_summary": {
            "signals": int(len(signals)),
            "complete_intervals": int(len(intervals)),
            "first_decision_date": str(signals["decision_date"].min().date()),
            "last_decision_date": str(signals["decision_date"].max().date()),
            "latest_calendar_month_used": str(signals["calendar_month"].iloc[-1]),
            "compression_on_signals": int(signals["compression_on"].sum()),
            "compression_off_signals": int((~signals["compression_on"]).sum()),
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
            "positive_subperiod_count": int(
                sum(bool(item["positive"]) for item in subperiods)
            ),
            "annual_monthly_excess_contributions": annual_contributions,
            "maximum_single_year_positive_contribution_share": maximum_year_share,
        },
        "gates": gate_checks,
        "governance": {
            "external_macro_signal_used": True,
            "isolated_expansion_research": True,
            "independent_supplier_validation": False,
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
        "# 510300 MACRO-02：AAA信用利差压缩 × 长期趋势确认",
        "",
        f"- 决策：`{report['decision']}`",
        f"- 说明：{report['decision_explanation']}",
        f"- 信号数/完整区间：{report['signal_summary']['signals']}/{report['signal_summary']['complete_intervals']}",
        f"- 决策范围：{report['signal_summary']['first_decision_date']}至{report['signal_summary']['last_decision_date']}；只用日线。",
        "",
        "## 数据边界",
        "",
        f"- 510300行情：{report['data_audit']['market_first_date']}至{report['data_audit']['market_last_date']}，交叉源`{report['data_audit']['market_cross_check_status']}`。",
        f"- 3年信用利差：{report['data_audit']['credit_curve_first_date']}至{report['data_audit']['credit_curve_last_date']}，范围{report['data_audit']['credit_spread_min_bp']:.2f}至{report['data_audit']['credit_spread_max_bp']:.2f}bp。",
        f"- 官方固定日期核对通过{report['data_audit']['official_checkpoint_passes']}项；既有国债批次重叠{report['data_audit']['existing_government_overlap_rows']}日。",
        f"- TLS验证/未验证行：{report['data_audit']['tls_verified_rows']}/{report['data_audit']['tls_unverified_rows']}；独立供应商验证=`false`。",
        "",
        "## 预测层",
        "",
        f"- 趋势开启条件下，信用压缩/未压缩样本：{conditional['compression_on_observations']}/{conditional['compression_off_observations']}。",
        (
            f"- 下一完整月度区间超现金收益均值差：{conditional['spread']:.4%}。"
            if conditional["spread"] is not None
            else "- 下一完整月度区间均值差：无法计算。"
        ),
        (
            f"- 6个月区块Bootstrap 95%区间：[{predictive['block_bootstrap']['lower_95']:.4%}, {predictive['block_bootstrap']['upper_95']:.4%}]。"
            if predictive["block_bootstrap"]["lower_95"] is not None
            else "- Bootstrap：无法计算。"
        ),
        f"- HAC每1bp压缩系数：{predictive['hac_regression']['coefficient_per_compression_bp']:.6f}，单侧p={predictive['hac_regression']['one_sided_p_value']:.4f}。",
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
        ("base_compression_only", "仅信用压缩-基础成本"),
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
            "本报告不修改现行510300-only系统，不启用仓位映射，不连接券商，不生成订单。通过也只能补做独立数据源验证与前向Shadow协议；失败则本版本冻结停止。",
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
