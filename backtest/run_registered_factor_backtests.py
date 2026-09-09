"""按预注册30/70滞回规则运行单因子策略回测。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from backtest.engine import BacktestCosts, run_long_cash_backtest, summarize_backtest
from research.registered_factors import build_hysteresis_target


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry.yaml"
FEATURE_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
RESEARCH_FILE = ROOT / "reports" / "research" / "registered_factor_research.json"
ETF_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
DIVIDEND_FILE = ROOT / "data" / "reference" / "510300_dividends.csv"
JSON_FILE = ROOT / "reports" / "backtest" / "registered_factor_backtests.json"
MARKDOWN_FILE = ROOT / "reports" / "backtest" / "registered_factor_backtests.md"
LEDGER_ROOT = ROOT / "data" / "processed" / "registered_factor_backtests"


def _metric_improves(strategy: dict, benchmark: dict) -> bool:
    drawdown = abs(strategy["max_drawdown"]) < abs(benchmark["max_drawdown"])
    sharpe = (
        strategy["sharpe_zero_cash_rate"] is not None
        and benchmark["sharpe_zero_cash_rate"] is not None
        and strategy["sharpe_zero_cash_rate"] > benchmark["sharpe_zero_cash_rate"]
    )
    calmar = (
        strategy["calmar"] is not None
        and benchmark["calmar"] is not None
        and strategy["calmar"] > benchmark["calmar"]
    )
    return bool(drawdown or sharpe or calmar)


def _rolling_excess_summary(strategy_ledger: pd.DataFrame, benchmark_ledger: pd.DataFrame, window: int = 242) -> dict:
    merged = strategy_ledger[["date", "equity"]].merge(
        benchmark_ledger[["date", "equity"]], on="date", suffixes=("_strategy", "_benchmark"), validate="one_to_one"
    )
    strategy_return = merged["equity_strategy"] / merged["equity_strategy"].shift(window) - 1.0
    benchmark_return = merged["equity_benchmark"] / merged["equity_benchmark"].shift(window) - 1.0
    excess = (strategy_return - benchmark_return).dropna()
    if excess.empty:
        return {
            "window_trading_days": window, "observations": 0, "mean": None, "median": None,
            "minimum": None, "maximum": None, "positive_ratio": None, "ge_20pct_ratio": None,
        }
    return {
        "window_trading_days": window,
        "observations": int(len(excess)),
        "mean": float(excess.mean()),
        "median": float(excess.median()),
        "minimum": float(excess.min()),
        "maximum": float(excess.max()),
        "positive_ratio": float((excess > 0).mean()),
        "ge_20pct_ratio": float((excess >= 0.20).mean()),
    }


def _run_period(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    buy_hold_targets: pd.DataFrame,
    costs: BacktestCosts,
    initial_cash: float,
    start: str,
    end: str,
) -> dict:
    strategy_ledger, strategy_trades = run_long_cash_backtest(
        prices, dividends, targets, initial_cash, costs, start, end
    )
    benchmark_ledger, benchmark_trades = run_long_cash_backtest(
        prices, dividends, buy_hold_targets, initial_cash, costs, start, end
    )
    strategy = summarize_backtest(strategy_ledger, strategy_trades, initial_cash)
    benchmark = summarize_backtest(benchmark_ledger, benchmark_trades, initial_cash)
    return {
        "strategy": strategy,
        "realistic_buy_hold": benchmark,
        "excess_vs_realistic_buy_hold": strategy["total_return"] - benchmark["total_return"],
        "risk_adjusted_or_drawdown_improves": _metric_improves(strategy, benchmark),
        "rolling_12m_excess": _rolling_excess_summary(strategy_ledger, benchmark_ledger),
        "ledger": strategy_ledger,
        "trades": strategy_trades,
    }


def _without_frames(item: dict) -> dict:
    return {key: value for key, value in item.items() if key not in {"ledger", "trades"}}


def _render_markdown(report: dict) -> str:
    lines = [
        "# 510300预注册单因子策略回测", "",
        "> 第一轮历史快照：其中价量候选已在第二轮机制复核中退役；当前状态以`round2_factor_selection_decision.md`为准。", "",
        "## 治理口径", "",
        "- 所有因子使用同一套30%/70%点时滞回规则，没有逐因子调阈值。",
        "- 信号在收盘后形成，下一交易日开盘执行；执行T+1、100份整手、佣金、最低佣金、分红和滑点。",
        "- `pseudo_oos`只使用2024-08-01至2025-06-30顺序验证段。",
        "- 2025-08-01以后的受污染区间仅作诊断，不参与候选晋级。",
        "- 本报告仍是历史回顾研究，不代表真正样本外或实盘批准。", "",
        "## D20单因子结果", "",
        "|因子|证据|开发超额|pseudo-OOS超额|pseudo-OOS最大回撤|15bp压力超额|交易次数|全样本滚动12月超额>0|候选状态|",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["factors"]:
        development = item["periods"]["development"]
        pseudo = item["periods"]["pseudo_oos"]
        stress = item["pseudo_oos_stress_15bps"]
        rolling = item["periods"]["full_retrospective"]["rolling_12m_excess"]["positive_ratio"]
        lines.append(
            f"|{item['factor_name_cn']}|{item['d20_evidence_rating']}（{item['d20_evidence_score']}/7）|"
            f"{development['excess_vs_realistic_buy_hold']:.2%}|{pseudo['excess_vs_realistic_buy_hold']:.2%}|"
            f"{pseudo['strategy']['max_drawdown']:.2%}|{stress['excess_vs_realistic_buy_hold']:.2%}|"
            f"{pseudo['strategy']['trade_count']}|{'' if rolling is None else f'{rolling:.2%}'}|{item['candidate_status']}|"
        )
    lines += ["", "## 候选结论", ""]
    selected = [item for item in report["factors"] if item["candidate_status"] == "ELIGIBLE_FOR_COMBINATION_DISCUSSION"]
    if selected:
        for item in selected:
            lines.append(f"- `{item['factor_id']}`：{item['factor_name_cn']}可以进入组合讨论，但仍需真正前向验证。")
    else:
        lines.append("- 本轮没有因子同时通过证据、pseudo-OOS超额、风险改善和15bp成本压力门槛；不得强制组合。")
    lines += ["", "完整开发、顺序验证、受污染诊断和全样本指标保存在同名JSON报告中。", ""]
    return "\n".join(lines)


def main() -> int:
    for path in (FEATURE_FILE, RESEARCH_FILE, ETF_FILE, DIVIDEND_FILE):
        if not path.exists():
            raise FileNotFoundError(f"回测输入缺失：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    research = json.loads(RESEARCH_FILE.read_text(encoding="utf-8"))
    d20_evidence = {
        item["factor_id"]: item["evidence"]
        for item in research["results"]
        if item["target"] == registry["targets"]["primary"]
    }
    features = pd.read_parquet(FEATURE_FILE)
    features["date"] = pd.to_datetime(features["date"])
    prices = pd.read_parquet(ETF_FILE)
    prices["date"] = pd.to_datetime(prices["date"])
    dividends = pd.read_csv(DIVIDEND_FILE, parse_dates=["record_date", "ex_date", "payment_date"])
    costs_config = settings["backtest"]
    base_costs = BacktestCosts(
        commission_rate=float(costs_config["commission_rate"]),
        minimum_commission_cny=float(costs_config["minimum_commission_cny"]),
        stamp_duty_rate=float(costs_config["stamp_duty_rate"]),
        slippage_bps=float(costs_config["slippage_bps_base"]),
        lot_size=int(costs_config["lot_size"]),
        cash_annual_rate=float(costs_config["cash_annual_rate"]),
    )
    stress_costs = BacktestCosts(
        commission_rate=base_costs.commission_rate,
        minimum_commission_cny=base_costs.minimum_commission_cny,
        stamp_duty_rate=base_costs.stamp_duty_rate,
        slippage_bps=float(costs_config["slippage_bps_stress"]),
        lot_size=base_costs.lot_size,
        cash_annual_rate=base_costs.cash_annual_rate,
    )
    initial_cash = float(costs_config["initial_cash"])
    rule = registry["strategy_rule"]
    split = settings["research_split"]
    periods = {
        "development": (split["development_start"], split["development_end"]),
        "pseudo_oos": (split["validation_start"], split["validation_end"]),
        "contaminated_retrospective": (split["final_holdout_start"], split["final_holdout_end"]),
        "full_retrospective": (settings["project"]["start_date"], settings["project"]["end_date"]),
    }
    buy_hold_targets = features[["date"]].assign(target_position=1.0)
    factor_results: list[dict] = []
    LEDGER_ROOT.mkdir(parents=True, exist_ok=True)
    for definition in registry["factor_definitions"]:
        factor_id = definition["factor_id"]
        targets = build_hysteresis_target(
            features,
            factor_column=definition["column"],
            expected_sign=definition["expected_sign"],
            threshold_mode=definition["threshold_mode"],
            percentile_window=int(rule["percentile_window"]),
            minimum_history=int(rule["minimum_history"]),
            entry_percentile=float(rule["entry_percentile"]),
            exit_percentile=float(rule["exit_percentile"]),
            initial_position=float(rule["initial_position"]),
        )
        item_periods: dict[str, dict] = {}
        raw_periods: dict[str, dict] = {}
        for name, (start, end) in periods.items():
            raw = _run_period(prices, dividends, targets, buy_hold_targets, base_costs, initial_cash, start, end)
            raw_periods[name] = raw
            item_periods[name] = _without_frames(raw)
            factor_dir = LEDGER_ROOT / factor_id
            factor_dir.mkdir(parents=True, exist_ok=True)
            raw["ledger"].to_parquet(factor_dir / f"{name}_ledger.parquet", index=False)
            raw["trades"].to_parquet(factor_dir / f"{name}_trades.parquet", index=False)
        stress_raw = _run_period(
            prices, dividends, targets, buy_hold_targets, stress_costs, initial_cash,
            split["validation_start"], split["validation_end"],
        )
        stress = _without_frames(stress_raw)
        evidence = d20_evidence[factor_id]
        pseudo = item_periods["pseudo_oos"]
        eligible = (
            evidence["rating"] in {"WEAK", "STRONG"}
            and pseudo["excess_vs_realistic_buy_hold"] > 0
            and pseudo["risk_adjusted_or_drawdown_improves"]
            and stress["excess_vs_realistic_buy_hold"] > 0
            and pseudo["strategy"]["trade_count"] >= 6
        )
        factor_results.append(
            {
                "factor_id": factor_id,
                "factor_name_cn": definition["name_cn"],
                "expected_sign": definition["expected_sign"],
                "d20_evidence_rating": evidence["rating"],
                "d20_evidence_score": evidence["score"],
                "periods": item_periods,
                "pseudo_oos_stress_15bps": stress,
                "candidate_status": "ELIGIBLE_FOR_COMBINATION_DISCUSSION" if eligible else "NOT_ELIGIBLE",
                "posthoc_sample_sufficiency_guard": {
                    "minimum_pseudo_oos_trade_count": 6,
                    "observed_trade_count": pseudo["strategy"]["trade_count"],
                    "passes": pseudo["strategy"]["trade_count"] >= 6,
                    "governance": "ADDED_AFTER_FIRST_BACKTEST_AND_APPLIED_UNIFORMLY",
                },
            }
        )
    report = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "governance": registry["research_status"],
        "rule": rule,
        "costs": base_costs.__dict__,
        "stress_slippage_bps": stress_costs.slippage_bps,
        "factor_count": len(factor_results),
        "factors": factor_results,
        "decision_scope": "ROUND1_HISTORICAL_SNAPSHOT",
        "superseded_by": "reports/research/round2_factor_selection_decision.json",
    }
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    MARKDOWN_FILE.write_text(_render_markdown(report), encoding="utf-8")
    print(f"单因子回测报告：{MARKDOWN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
