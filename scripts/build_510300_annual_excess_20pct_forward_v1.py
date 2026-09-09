"""生成 510300 年化净超额 20%目标的决策型当前状态。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.annual_excess_20pct_forward_v1 import (  # noqa: E402
    load_contract,
    required_edge_budget,
    round_trip_cost_bps,
    summarize_iopv_gap_sample,
)
import backtest.run_round5_defensive_valuation_timing as round5  # noqa: E402
from backtest.engine import BacktestCosts, run_long_cash_backtest  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "510300_annual_excess_20pct_forward_v1.yaml"
PCF_READINESS = ROOT / "reports" / "data_quality" / "510300_primary_market_readiness_v1_2.json"
INDUSTRY_STATUS = (
    ROOT
    / "reports"
    / "forward"
    / "industry_expectation_gap_v1_evaluation"
    / "operations_status_v1_2.json"
)
LEGACY_IOPV = ROOT / "data" / "raw" / "primary_market" / "510300_iopv_snapshots.parquet"
ROUND5_RESULT = ROOT / "reports" / "backtest" / "round5_three_strategy_comparison.json"
HASH_HOLDOUT_RESULT = ROOT / "reports" / "backtest" / "a_share_hash_holdout_alpha_v2.json"
DAILY_510300 = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
DIVIDENDS_510300 = ROOT / "data" / "reference" / "510300_dividends.csv"
H00300_TOTAL_RETURN = ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
ROUND5_TARGET_LEDGER = (
    ROOT / "data" / "processed" / "round5_defensive_valuation_timing" / "enhanced_ledger.parquet"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _cost_budgets(contract: dict[str, Any]) -> list[dict[str, Any]]:
    costs = contract["costs"]
    scope = contract["scope"]
    objective = contract["objective"]
    rows: list[dict[str, Any]] = []
    for notional in (50_000.0, 125_000.0, 250_000.0):
        base_cost = round_trip_cost_bps(
            notional,
            commission_rate_per_leg=float(costs["commission_rate_per_leg"]),
            minimum_commission_cny_per_leg=float(costs["minimum_commission_cny_per_leg"]),
            slippage_bps_per_leg=float(costs["base_slippage_bps_per_leg"]),
            stamp_duty_sell_rate=float(costs["etf_stamp_duty_sell_rate"]),
        )
        stress_cost = round_trip_cost_bps(
            notional,
            commission_rate_per_leg=float(costs["commission_rate_per_leg"]),
            minimum_commission_cny_per_leg=float(costs["minimum_commission_cny_per_leg"]),
            slippage_bps_per_leg=float(costs["stress_slippage_bps_per_leg"]),
            stamp_duty_sell_rate=float(costs["etf_stamp_duty_sell_rate"]),
        )
        rows.append(
            {
                "event_notional_cny": notional,
                "events_per_year": int(objective["annualization_trading_days"]),
                "base": required_edge_budget(
                    initial_capital_cny=float(scope["initial_capital_cny"]),
                    annual_excess_target=float(objective["minimum_annualized_excess"]),
                    event_notional_cny=notional,
                    events_per_year=int(objective["annualization_trading_days"]),
                    round_trip_cost=base_cost,
                ),
                "stress": required_edge_budget(
                    initial_capital_cny=float(scope["initial_capital_cny"]),
                    annual_excess_target=float(objective["minimum_annualized_excess"]),
                    event_notional_cny=notional,
                    events_per_year=int(objective["annualization_trading_days"]),
                    round_trip_cost=stress_cost,
                ),
            }
        )
    return rows


def _equity_metrics(frame: pd.DataFrame, initial_capital_cny: float) -> dict[str, float]:
    elapsed_days = max((pd.Timestamp(frame["date"].iloc[-1]) - pd.Timestamp(frame["date"].iloc[0])).days, 1)
    total_return = float(frame["equity"].iloc[-1] / initial_capital_cny - 1.0)
    return {
        "total_return": total_return,
        "cagr": float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0),
        "maximum_drawdown": float((frame["equity"] / frame["equity"].cummax() - 1.0).min()),
        "ending_equity_cny": float(frame["equity"].iloc[-1]),
    }


def _retrospective_500000_reference(contract: dict[str, Any]) -> dict[str, Any]:
    """用旧冻结目标仓位重算新资金和费用，只作容量基线。"""

    initial_capital = float(contract["scope"]["initial_capital_cny"])
    costs = contract["costs"]
    daily = pd.read_parquet(DAILY_510300)
    dividends = pd.read_csv(DIVIDENDS_510300)
    old_ledger = pd.read_parquet(ROUND5_TARGET_LEDGER)
    targets = old_ledger[
        ["date", "target_position", "trade_allowed", "risk_off_override", "signal_reason"]
    ].copy()
    start = pd.Timestamp(old_ledger["date"].min())
    end = pd.Timestamp(old_ledger["date"].max())
    benchmark_raw = pd.read_parquet(H00300_TOTAL_RETURN)
    benchmark = round5._benchmark_ledger(benchmark_raw, old_ledger["date"], initial_capital)
    benchmark_metrics = _equity_metrics(benchmark, initial_capital)

    results: dict[str, Any] = {}
    for name, slippage in (
        ("base", float(costs["base_slippage_bps_per_leg"])),
        ("stress", float(costs["stress_slippage_bps_per_leg"])),
    ):
        backtest_costs = BacktestCosts(
            commission_rate=float(costs["commission_rate_per_leg"]),
            minimum_commission_cny=float(costs["minimum_commission_cny_per_leg"]),
            stamp_duty_rate=float(costs["etf_stamp_duty_sell_rate"]),
            slippage_bps=slippage,
            lot_size=int(costs["lot_size_shares"]),
            cash_annual_rate=float(costs["cash_annual_rate"]),
        )
        ledger, trades = run_long_cash_backtest(
            daily,
            dividends,
            targets,
            initial_capital,
            backtest_costs,
            start,
            end,
        )
        metrics = _equity_metrics(ledger, initial_capital)
        metrics["annualized_excess_vs_h00300"] = float(
            metrics["cagr"] - benchmark_metrics["cagr"]
        )
        metrics["rolling_242d_excess_vs_h00300"] = round5._rolling_excess(ledger, benchmark)
        metrics["trade_count"] = int(len(trades))
        metrics["commission_cny"] = float(trades["commission"].sum()) if not trades.empty else 0.0
        results[name] = metrics

    target = float(contract["objective"]["minimum_annualized_excess"])
    rolling_target = float(contract["objective"]["minimum_rolling_242d_excess_median"])
    return {
        "status": "RETROSPECTIVE_ONLY_NOT_FORWARD_EVIDENCE",
        "account_capital_cny": initial_capital,
        "fee_rate_per_leg": float(costs["commission_rate_per_leg"]),
        "base": results["base"],
        "stress": results["stress"],
        "gates": {
            "base_annualized_excess_at_least_20pct": results["base"]["annualized_excess_vs_h00300"] >= target,
            "stress_annualized_excess_at_least_20pct": results["stress"]["annualized_excess_vs_h00300"] >= target,
            "base_rolling_median_at_least_20pct": results["base"]["rolling_242d_excess_vs_h00300"]["median"] >= rolling_target,
            "stress_rolling_median_at_least_20pct": results["stress"]["rolling_242d_excess_vs_h00300"]["median"] >= rolling_target,
        },
        "interpretation": "只改变用户明确给定的本金和费率，沿用已冻结仓位路径；历史已被查看，不能作为前瞻通过。",
    }


def build_status(contract: dict[str, Any]) -> dict[str, Any]:
    pcf = read_json(PCF_READINESS)
    industry = read_json(INDUSTRY_STATUS)
    round5 = read_json(ROUND5_RESULT)
    hash_holdout = read_json(HASH_HOLDOUT_RESULT)
    gap_sample = summarize_iopv_gap_sample(pd.read_parquet(LEGACY_IOPV))
    cost_budgets = _cost_budgets(contract)
    half_capital = next(row for row in cost_budgets if row["event_notional_cny"] == 250_000.0)
    retrospective_500000 = _retrospective_500000_reference(contract)

    round5_swing_excess = float(round5["summaries"]["swing"]["annualized_excess_vs_h00300"])
    target = float(contract["objective"]["minimum_annualized_excess"])
    hash_base = float(hash_holdout["base_cost"]["annualized_excess"])
    hash_stress = float(hash_holdout["stress_cost"]["annualized_excess"])
    pcf_state = {
        "status": pcf["status"],
        "observed_trading_days": int(pcf["observed_trading_days"]),
        "full_quality_days": int(pcf["full_coverage_days"]),
        "first_unseen_gate_days": int(pcf["first_unseen_evaluation_full_coverage_days"]),
        "replication_gate_days": int(pcf["replication_full_coverage_days"]),
        "performance_eligible": bool(pcf["eligible_for_first_unseen_evaluation"]),
    }
    industry_maturity = industry["maturity"]
    industry_state = {
        "status": industry["status"],
        "view_status": industry["view_status"],
        "origin_cluster_count": int(industry_maturity["origin_cluster_count"]),
        "mature_origin_cluster_count": int(industry_maturity["mature_origin_cluster_count"]),
        "non_overlapping_60d_block_count": int(industry_maturity["non_overlapping_60d_block_count"]),
        "calibration_eligible": bool(industry_maturity["calibration"]["eligible"]),
        "model_comparison_eligible": bool(industry_maturity["model_comparison"]["eligible"]),
    }

    return {
        "schema_version": "1.0.0",
        "report_id": "510300_ANNUAL_EXCESS_20PCT_FORWARD_V1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "TARGET_NOT_YET_MET",
        "goal_achieved": False,
        "objective": {
            "metric": contract["objective"]["primary_metric"],
            "minimum_annualized_excess": target,
            "benchmark": contract["benchmark"]["primary_id"],
            "base_and_stress_cost_required": True,
        },
        "current_empirical_evidence": {
            "legacy_20000_round5_510300_swing_annualized_excess": round5_swing_excess,
            "legacy_20000_round5_gap_to_20pct": target - round5_swing_excess,
            "legacy_20000_hash_holdout_base_annualized_excess": hash_base,
            "legacy_20000_hash_holdout_stress_annualized_excess": hash_stress,
            "legacy_20000_hash_holdout_status": hash_holdout["status"],
            "legacy_results_directly_comparable_to_500000_contract": False,
            "retrospective_500000_round5_reference": retrospective_500000,
            "current_500000_forward_candidate_evaluation_status": "NOT_YET_ELIGIBLE",
            "no_verified_500000_candidate_meets_target": True,
        },
        "active_streams": {
            "PRIMARY_MARKET_PCF_IOPV": pcf_state,
            "INDUSTRY_EXPECTATION_GAP": industry_state,
        },
        "pcf_iopv_feasibility": {
            "gap_sample": gap_sample,
            "account_capital_cny": float(contract["scope"]["initial_capital_cny"]),
            "maximum_single_round_trip_notional_cny": 250_000.0,
            "half_capital_base_round_trip_cost_bps": half_capital["base"]["round_trip_cost_bps"],
            "half_capital_stress_round_trip_cost_bps": half_capital["stress"]["round_trip_cost_bps"],
            "required_base_gross_edge_bps_for_20pct_at_242_events": half_capital["base"]["required_gross_edge_bps_per_event"],
            "required_stress_gross_edge_bps_for_20pct_at_242_events": half_capital["stress"]["required_gross_edge_bps_per_event"],
            "current_interpretation": "EXECUTION_FILTER_NOT_YET_PRIMARY_ALPHA_SOURCE",
            "reason": "冻结前折溢价样本绝对极值低于小账户往返成本；样本仅用于容量预算，不构成终局否证。",
        },
        "cost_capacity_scenarios": cost_budgets,
        "decision": {
            "primary_market_role": "继续收集；20个完整质量日后先做成本容量门，不直接优化交易阈值",
            "industry_role": "继续独立原点；成熟前不读取部分收益或映射仓位",
            "stop_rule": "若两条流在各自冻结成熟门后都不能支持20%净超额，替换机制而不降低目标或费用",
            "next_evidence": "2026-08-27 V1.8 首个合格 PCF/IOPV 窗口",
        },
        "inputs": {
            "contract": {"path": str(DEFAULT_CONFIG.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(DEFAULT_CONFIG)},
            "pcf_readiness": {"path": str(PCF_READINESS.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(PCF_READINESS)},
            "industry_status": {"path": str(INDUSTRY_STATUS.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(INDUSTRY_STATUS)},
            "legacy_iopv": {"path": str(LEGACY_IOPV.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(LEGACY_IOPV)},
            "round5_result": {"path": str(ROUND5_RESULT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(ROUND5_RESULT)},
            "hash_holdout_result": {"path": str(HASH_HOLDOUT_RESULT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(HASH_HOLDOUT_RESULT)},
            "daily_510300": {"path": str(DAILY_510300.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(DAILY_510300)},
            "dividends_510300": {"path": str(DIVIDENDS_510300.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(DIVIDENDS_510300)},
            "h00300_total_return": {"path": str(H00300_TOTAL_RETURN.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(H00300_TOTAL_RETURN)},
            "round5_target_ledger": {"path": str(ROUND5_TARGET_LEDGER.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(ROUND5_TARGET_LEDGER)},
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    empirical = report["current_empirical_evidence"]
    pcf = report["active_streams"]["PRIMARY_MARKET_PCF_IOPV"]
    industry = report["active_streams"]["INDUSTRY_EXPECTATION_GAP"]
    feasibility = report["pcf_iopv_feasibility"]
    gap = feasibility["gap_sample"]
    reference_500000 = empirical["retrospective_500000_round5_reference"]
    lines = [
        "# 510300 年化净超额 20 个百分点：当前研究状态",
        "",
        f"- 状态：`{report['status']}`",
        "- 目标：基础与压力成本后，策略净 CAGR 相对 H00300 全收益 CAGR 均不低于 20 个百分点。",
        "- 当前结论：没有现有候选达到目标，不能生成仓位或订单。",
        "",
        "## 已知差距",
        "",
        f"- 旧 2 万元 510300 第五轮波段候选年化超额：{empirical['legacy_20000_round5_510300_swing_annualized_excess']:.2%}；距离目标 {empirical['legacy_20000_round5_gap_to_20pct']:.2%}。",
        f"- 旧 2 万元全 A 股哈希盲测：基础 {empirical['legacy_20000_hash_holdout_base_annualized_excess']:.2%}，压力 {empirical['legacy_20000_hash_holdout_stress_annualized_excess']:.2%}，已冻结拒绝。",
        "- 上述旧结果与 50 万元、每腿 1bp 的新合同不可直接比较；下面仅按原冻结仓位路径重算资金与费用影响。",
        f"- 50 万元同仓位路径回顾性重算：基础年化超额 {reference_500000['base']['annualized_excess_vs_h00300']:.2%}，压力年化超额 {reference_500000['stress']['annualized_excess_vs_h00300']:.2%}，不满足20%目标。",
        "",
        "## 两条活动研究流",
        "",
        f"- PCF/IOPV：完整质量日 {pcf['full_quality_days']}；首次未见评价门 {pcf['first_unseen_gate_days']}，复制门 {pcf['replication_gate_days']}。",
        f"- 行业预期差：独立原点 {industry['origin_cluster_count']}，成熟原点 {industry['mature_origin_cluster_count']}，非重叠60日块 {industry['non_overlapping_60d_block_count']}。",
        "",
        "## PCF/IOPV 经济容量",
        "",
        f"- 冻结前样本：{gap['trade_date_count']}日、{gap['row_count']}条盘中记录；折溢价范围 {gap['minimum_bps']:.2f}bp 至 {gap['maximum_bps']:.2f}bp。",
        f"- 250,000元单次名义金额往返成本：基础 {feasibility['half_capital_base_round_trip_cost_bps']:.2f}bp，压力 {feasibility['half_capital_stress_round_trip_cost_bps']:.2f}bp。",
        f"- 若每年有242次机会，为贡献20%增量收益，每次所需毛边际：基础 {feasibility['required_base_gross_edge_bps_for_20pct_at_242_events']:.2f}bp，压力 {feasibility['required_stress_gross_edge_bps_for_20pct_at_242_events']:.2f}bp。",
        "- 当前定位：IOPV先作为执行过滤器；20个完整质量日前不优化阈值，也不把旧样本写成收益证据。",
        "",
        "## 下一步",
        "",
        "1. 从2026-08-27首个V1.8窗口继续真实采集。",
        "2. PCF达到20个完整质量日后先检验成本容量；容量不够则终止其主要Alpha角色。",
        "3. 行业原点按冻结60/120日窗口成熟；未成熟不看部分收益。",
        "4. 两条流均不支持20%时，在两流上限内替换经济机制，不降低目标。",
        "",
        "本文件是研究决策状态，不是收益承诺、仓位建议或交易授权。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成510300年化净超额20%前瞻目标状态")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    contract = load_contract(config_path)
    report = build_status(contract)
    if not args.no_write:
        json_path = ROOT / contract["outputs"]["current_status_json"]
        markdown_path = ROOT / contract["outputs"]["current_status_markdown"]
        atomic_text(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        atomic_text(markdown_path, render_markdown(report))
    print(
        json.dumps(
            {
                "状态": report["status"],
                "目标已达成": report["goal_achieved"],
                "目标年化净超额": report["objective"]["minimum_annualized_excess"],
                "50万元回顾性基础超额": report["current_empirical_evidence"]["retrospective_500000_round5_reference"]["base"]["annualized_excess_vs_h00300"],
                "50万元回顾性压力超额": report["current_empirical_evidence"]["retrospective_500000_round5_reference"]["stress"]["annualized_excess_vs_h00300"],
                "50万元前瞻候选评价状态": report["current_empirical_evidence"]["current_500000_forward_candidate_evaluation_status"],
                "PCF完整质量日": report["active_streams"]["PRIMARY_MARKET_PCF_IOPV"]["full_quality_days"],
                "行业成熟原点": report["active_streams"]["INDUSTRY_EXPECTATION_GAP"]["mature_origin_cluster_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
