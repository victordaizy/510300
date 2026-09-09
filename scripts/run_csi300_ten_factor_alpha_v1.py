"""运行冻结的沪深300十因子非线性小账户历史伪样本外检验。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_final_alpha_strategy import (  # noqa: E402
    _benchmark_series,
    _return_metrics,
    summarize,
)
from research.csi300_ten_factor_alpha_v1 import (  # noqa: E402
    TenFactorRules,
    build_forward_relative_outcomes,
    build_ten_factor_panel,
    load_research_history,
    walk_forward_predictions,
)
from research.small_account_cross_sectional import (  # noqa: E402
    SmallAccountCosts,
    run_small_account_open_backtest,
)
from scripts.freeze_csi300_ten_factor_alpha_v1 import (  # noqa: E402
    CONFIG_FILE,
    FROZEN_FILES,
    MANIFEST_FILE,
    sha256,
    tree_sha256,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NaT:
        return None
    return value


def _verify_manifest(contract: dict) -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError("十因子协议尚未冻结，禁止读取未来收益")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_OUTCOME_READ":
        raise RuntimeError("十因子冻结清单状态错误")
    changed: list[str] = []
    for relative, expected in manifest["frozen_files"].items():
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected:
            changed.append(relative)
    if set(manifest["frozen_files"]) != set(FROZEN_FILES):
        changed.append("FROZEN_FILE_SET")
    for relative, expected in manifest["input_files"].items():
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected:
            changed.append(relative)
    history_dir = ROOT / contract["inputs"]["component_history_cache"]
    if tree_sha256(history_dir) != manifest["component_history_cache"]:
        changed.append(contract["inputs"]["component_history_cache"])
    if changed:
        raise RuntimeError(f"冻结后文件变化：{sorted(set(changed))}")
    return manifest


def _costs(contract: dict, slippage_key: str) -> SmallAccountCosts:
    account = contract["account"]
    costs = contract["costs"]
    return SmallAccountCosts(
        commission_rate=float(costs["commission_rate"]),
        minimum_commission_cny=float(costs["minimum_commission_cny"]),
        stamp_duty_sell_rate=float(costs["stamp_duty_sell_rate"]),
        stamp_duty_sell_rate_before_reduction=float(costs["stamp_duty_sell_rate_before_reduction"]),
        stamp_duty_reduction_effective_date=str(costs["stamp_duty_reduction_effective_date"]),
        slippage_bps_per_leg=float(costs[slippage_key]),
        cash_annual_rate=float(account["cash_annual_rate"]),
        lot_size=int(account["lot_size"]),
        minimum_trade_notional_cny=float(account["minimum_trade_notional_cny"]),
        maximum_positions=int(account["maximum_positions"]),
    )


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _subperiod_summary(
    ledger: pd.DataFrame,
    benchmark: pd.DataFrame,
    start: pd.Timestamp,
) -> dict:
    mask = ledger["date"].ge(start)
    strategy = _return_metrics(ledger.loc[mask, "daily_return"], ledger.loc[mask, "date"])
    benchmark_metrics = _return_metrics(
        benchmark.loc[mask, "daily_return"], benchmark.loc[mask, "date"]
    )
    return {
        "start": str(start.date()),
        "end": str(pd.Timestamp(ledger.loc[mask, "date"].max()).date()),
        "strategy": strategy,
        "benchmark": benchmark_metrics,
        "annualized_excess": strategy["cagr"] - benchmark_metrics["cagr"],
    }


def _paired_block_bootstrap_excess(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    repetitions: int,
    block_length: int,
    seed: int,
) -> dict:
    strategy = strategy_returns.to_numpy(dtype=float)
    benchmark = benchmark_returns.to_numpy(dtype=float)
    if len(strategy) != len(benchmark) or len(strategy) < block_length:
        raise ValueError("Bootstrap收益序列长度无效")
    rng = np.random.default_rng(seed)
    n = len(strategy)
    starts = np.arange(0, n - block_length + 1)
    samples = np.empty(repetitions, dtype=float)
    block_count = int(np.ceil(n / block_length))
    for repetition in range(repetitions):
        chosen = rng.choice(starts, size=block_count, replace=True)
        indices = np.concatenate(
            [np.arange(start, start + block_length) for start in chosen]
        )[:n]
        strategy_cagr = np.expm1(np.log1p(strategy[indices]).mean() * 242.0)
        benchmark_cagr = np.expm1(np.log1p(benchmark[indices]).mean() * 242.0)
        samples[repetition] = strategy_cagr - benchmark_cagr
    return {
        "repetitions": repetitions,
        "block_length_trading_days": block_length,
        "median": float(np.median(samples)),
        "interval_95pct": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def _prediction_diagnostics(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> dict:
    evaluated = predictions.loc[
        predictions["model_output"].eq("PREDICTION_AUDIT_ONLY")
    ].merge(
        outcomes[["date", "con_code", "future_excess_log_return_10d"]],
        on=["date", "con_code"],
        how="left",
        validate="one_to_one",
    )
    matured = evaluated.dropna(
        subset=["predicted_excess_log_return_10d", "future_excess_log_return_10d"]
    )
    information_coefficients = matured.groupby("date", sort=True).apply(
        lambda frame: frame["predicted_excess_log_return_10d"].corr(
            frame["future_excess_log_return_10d"], method="spearman"
        ),
        include_groups=False,
    )
    top = matured.loc[matured["prediction_rank"].le(3)]
    return {
        "prediction_rows": int(len(predictions)),
        "model_ready_rows": int(predictions["model_output"].eq("PREDICTION_AUDIT_ONLY").sum()),
        "model_signal_dates": int(
            predictions.loc[
                predictions["model_output"].eq("PREDICTION_AUDIT_ONLY"), "date"
            ].nunique()
        ),
        "matured_evaluation_rows": int(len(matured)),
        "mean_daily_spearman_ic": float(information_coefficients.mean()),
        "median_daily_spearman_ic": float(information_coefficients.median()),
        "positive_daily_ic_ratio": float(information_coefficients.gt(0).mean()),
        "top3_mean_future_excess_log_return_10d": float(top["future_excess_log_return_10d"].mean()),
        "top3_positive_excess_ratio": float(top["future_excess_log_return_10d"].gt(0).mean()),
    }


def _render(report: dict) -> str:
    base = report["base_cost"]
    stress = report["stress_cost"]
    sub_base = report["evaluation_subperiod"]["base_cost"]
    sub_stress = report["evaluation_subperiod"]["stress_cost"]
    lines = [
        "# 沪深300十因子非线性小账户 Alpha V1 历史结果",
        "",
        f"- 总状态：`{report['status']}`",
        f"- 证据标签：`{report['evidence_label']}`",
        "- 严格未来样本外：`NOT_AVAILABLE`",
        f"- 因子数量：{report['factor_budget']['used']} / {report['factor_budget']['maximum']}",
        "",
        "|区间/成本|策略年化|基准年化|年化超额|最大回撤|",
        "|---|---:|---:|---:|---:|",
        f"|全伪样本外/5bp|{base['strategy']['cagr']:.2%}|{base['benchmark']['cagr']:.2%}|{base['annualized_excess']:.2%}|{base['strategy']['maximum_drawdown']:.2%}|",
        f"|全伪样本外/15bp|{stress['strategy']['cagr']:.2%}|{stress['benchmark']['cagr']:.2%}|{stress['annualized_excess']:.2%}|{stress['strategy']['maximum_drawdown']:.2%}|",
        f"|2025后/5bp|{sub_base['strategy']['cagr']:.2%}|{sub_base['benchmark']['cagr']:.2%}|{sub_base['annualized_excess']:.2%}|{sub_base['strategy']['maximum_drawdown']:.2%}|",
        f"|2025后/15bp|{sub_stress['strategy']['cagr']:.2%}|{sub_stress['benchmark']['cagr']:.2%}|{sub_stress['annualized_excess']:.2%}|{sub_stress['strategy']['maximum_drawdown']:.2%}|",
        "",
        f"- 20日块Bootstrap年化超额95%区间：{report['bootstrap_annualized_excess']['interval_95pct']}",
        f"- 日度Spearman IC均值：{report['prediction_diagnostics']['mean_daily_spearman_ic']:.4f}",
        f"- Top3未来10日平均对数超额：{report['prediction_diagnostics']['top3_mean_future_excess_log_return_10d']:.4%}",
        f"- 最小实际成交：{report['execution_audit']['minimum_trade_notional_cny']:.2f}元",
        "",
        "## 门槛",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in report["gates"].items()
    )
    lines += [
        "",
        "## 边界",
        "",
        "模型失败后不得在同一历史上修改十因子、模型参数、持股数或周期补救。即使全部历史门槛通过，也不能替代冻结日之后的真正前向证据，且不生成仓位或订单。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest = _verify_manifest(contract)
    rules = TenFactorRules.from_contract(contract)
    inputs = contract["inputs"]
    history_dir = ROOT / inputs["component_history_cache"]
    history = load_research_history(sorted(history_dir.glob("*.parquet")))
    members = pd.read_parquet(ROOT / inputs["constituent_member_panel"])
    moneyflow = pd.read_parquet(ROOT / inputs["component_moneyflow"])
    financials = pd.read_parquet(ROOT / inputs["point_in_time_financials"])
    index = pd.read_parquet(ROOT / inputs["index_daily"])
    benchmark_raw = pd.read_parquet(ROOT / inputs["benchmark_total_return"])
    breadth = pd.read_parquet(ROOT / inputs["breadth"])
    calendar = pd.DatetimeIndex(
        sorted(
            pd.to_datetime(
                members.loc[members["date"].le(rules.end_date), "date"]
            ).unique()
        )
    )

    factors = build_ten_factor_panel(
        history, members, moneyflow, financials, index, breadth, rules
    )
    outcomes = build_forward_relative_outcomes(
        factors, history, benchmark_raw, calendar, rules
    )
    predictions = walk_forward_predictions(factors, outcomes, rules)
    ready = predictions.loc[
        predictions["model_output"].eq("PREDICTION_AUDIT_ONLY")
        & predictions["prediction_rank"].le(int(contract["selection"]["holdings"]))
    ].copy()
    if ready.empty:
        raise RuntimeError("十因子走步模型没有产生可评价目标")
    targets = ready[
        ["date", "con_code", "prediction_rank", "predicted_excess_log_return_10d"]
    ].rename(columns={"date": "signal_date", "prediction_rank": "selection_rank"})
    targets["regime"] = "TEN_FACTOR_NONLINEAR"

    execution_history = history[
        ["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]
    ].copy()
    execution_history["is_suspended"] = execution_history["volume"].fillna(0.0).le(0.0)
    first_signal = pd.Timestamp(targets["signal_date"].min())
    trade_calendar = calendar[(calendar >= first_signal) & (calendar <= rules.end_date)]
    initial_cash = float(contract["account"]["initial_cash_cny"])
    gap = float(contract["selection"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(
        execution_history,
        targets,
        trade_calendar,
        initial_cash,
        _costs(contract, "base_slippage_bps_per_leg"),
        gap,
    )
    stress_ledger, stress_trades = run_small_account_open_backtest(
        execution_history,
        targets,
        trade_calendar,
        initial_cash,
        _costs(contract, "stress_slippage_bps_per_leg"),
        gap,
    )
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial_cash)
    base = summarize(base_ledger, base_trades, benchmark, initial_cash)
    stress = summarize(stress_ledger, stress_trades, benchmark, initial_cash)
    subperiod_start = pd.Timestamp(contract["evaluation"]["evaluation_subperiod_start"])
    sub_base = _subperiod_summary(base_ledger, benchmark, subperiod_start)
    sub_stress = _subperiod_summary(stress_ledger, benchmark, subperiod_start)
    bootstrap = _paired_block_bootstrap_excess(
        base_ledger["daily_return"],
        benchmark["daily_return"],
        int(contract["evaluation"]["bootstrap_repetitions"]),
        int(contract["evaluation"]["bootstrap_block_length_trading_days"]),
        int(contract["evaluation"]["random_seed"]),
    )
    diagnostics = _prediction_diagnostics(predictions, outcomes)
    threshold = float(contract["evaluation"]["annualized_excess_minimum"])
    stress_threshold = float(contract["evaluation"]["stress_annualized_excess_minimum"])
    sub_threshold = float(contract["evaluation"]["subperiod_annualized_excess_minimum"])
    gates = {
        "full_base_annualized_excess_20pct": base["annualized_excess"] >= threshold,
        "full_stress_annualized_excess_20pct": stress["annualized_excess"] >= stress_threshold,
        "subperiod_base_annualized_excess_20pct": sub_base["annualized_excess"] >= sub_threshold,
        "subperiod_stress_annualized_excess_20pct": sub_stress["annualized_excess"] >= sub_threshold,
        "bootstrap_lower_bound_positive": bootstrap["interval_95pct"][0]
        > float(contract["evaluation"]["bootstrap_excess_lower_bound_strictly_above"]),
        "all_trades_at_least_5000": bool(
            base_trades.empty
            or base_trades["notional"].ge(float(contract["account"]["minimum_trade_notional_cny"]) - 1e-8).all()
        ),
    }
    status = "HISTORICAL_FORMULA_PASS_AWAITING_TRUE_FORWARD" if all(gates.values()) else "HISTORICAL_FORMULA_REJECTED_FROZEN"
    minimum_trade = float(base_trades["notional"].min()) if not base_trades.empty else None
    report = _json_safe(
        {
            "project_id": contract["protocol"]["project_id"],
            "version": contract["protocol"]["version"],
            "status": status,
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "evidence_label": contract["protocol"]["evidence_label"],
            "manifest_sha256": sha256(MANIFEST_FILE),
            "frozen_at": manifest["frozen_at"],
            "factor_budget": contract["factor_budget"],
            "model": contract["model"],
            "row_counts": {
                "factor_rows": int(len(factors)),
                "ready_factor_rows": int(factors["signal_output"].eq("SIGNAL_READY").sum()),
                "outcome_rows": int(outcomes["future_excess_log_return_10d"].notna().sum()),
                "prediction_rows": int(len(predictions)),
                "target_signal_dates": int(targets["signal_date"].nunique()),
            },
            "base_cost": base,
            "stress_cost": stress,
            "evaluation_subperiod": {"base_cost": sub_base, "stress_cost": sub_stress},
            "bootstrap_annualized_excess": bootstrap,
            "prediction_diagnostics": diagnostics,
            "execution_audit": {
                "minimum_trade_notional_cny": minimum_trade,
                "maximum_position_count": int(base_ledger["position_count"].max()),
                "blocked_small_sell_days": int(base_ledger["blocked_small_sell_count"].gt(0).sum()),
                "signal_to_execution": "NEXT_TRADING_DAY_OPEN",
                "t_plus_one_enforced": True,
            },
            "gates": gates,
            "safety": {
                "strict_true_forward": "NOT_AVAILABLE",
                "paper_signal": "DISABLED",
                "position_mapping": "DISABLED",
                "order_generation": "DISABLED",
                "broker_connection": "DISABLED",
                "live_trading": "NOT_AUTHORIZED",
            },
        }
    )
    outputs = contract["outputs"]
    for frame, key in (
        (factors, "factor_panel"),
        (predictions, "predictions"),
        (base_ledger, "base_ledger"),
        (base_trades, "base_trades"),
        (stress_ledger, "stress_ledger"),
        (stress_trades, "stress_trades"),
    ):
        _atomic_parquet(frame, ROOT / outputs[key])
    _atomic_text(json.dumps(report, ensure_ascii=False, indent=2), ROOT / outputs["result_json"])
    _atomic_text(_render(report), ROOT / outputs["result_markdown"])
    print(
        json.dumps(
            {
                "状态": status,
                "基础年化超额": base["annualized_excess"],
                "压力年化超额": stress["annualized_excess"],
                "2025后基础年化超额": sub_base["annualized_excess"],
                "Bootstrap下界": bootstrap["interval_95pct"][0],
                "平均IC": diagnostics["mean_daily_spearman_ic"],
                "严格未来样本外": "NOT_AVAILABLE",
                "结果": str(ROOT / outputs["result_json"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

