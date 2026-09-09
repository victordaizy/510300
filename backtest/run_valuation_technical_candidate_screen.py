"""在估值目标仓位之上统一重筛既有与标准技术时点候选。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from backtest.intraday_entry_overlay_engine import (
    OverlayCosts,
    prepare_executable_targets,
    run_valuation_entry_policy,
    summarize_policy,
)
from backtest.prototype_lc_fvg_pv.logic import prepare_features, simulate_trades
from research.technical_timing_candidates import (
    build_standard_candidate_events,
    events_from_close_pressure_forecasts,
    events_from_trade_records,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "valuation_technical_candidate_screen.yaml"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _relative_diagnostics(
    baseline: pd.DataFrame,
    overlay: pd.DataFrame,
) -> dict[str, Any]:
    merged = baseline[["date", "daily_return", "equity"]].merge(
        overlay[["date", "daily_return", "equity"]],
        on="date",
        suffixes=("_baseline", "_overlay"),
        validate="one_to_one",
    )
    midpoint = len(merged) // 2
    halves = []
    for label, frame in [("前半段", merged.iloc[:midpoint]), ("后半段", merged.iloc[midpoint:])]:
        overlay_return = float((1.0 + frame["daily_return_overlay"]).prod() - 1.0)
        baseline_return = float((1.0 + frame["daily_return_baseline"]).prod() - 1.0)
        halves.append(
            {
                "period": label,
                "start_date": str(frame["date"].iloc[0].date()),
                "end_date": str(frame["date"].iloc[-1].date()),
                "relative_return": overlay_return - baseline_return,
            }
        )
    return {
        "ending_equity_difference_cny": float(
            merged["equity_overlay"].iloc[-1] - merged["equity_baseline"].iloc[-1]
        ),
        "chronological_halves": halves,
    }


def _candidate_decision(
    config: dict[str, Any],
    baseline_base: dict[str, Any],
    baseline_stress: dict[str, Any],
    overlay_base: dict[str, Any],
    overlay_stress: dict[str, Any],
    diagnostics: dict[str, Any],
    relative: dict[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    evidence = {
        "valuation_buy_opportunities": diagnostics["valuation_buy_opportunities"]
        >= int(gates["minimum_valuation_buy_opportunities"]),
        "valuation_sell_opportunities": diagnostics["valuation_sell_opportunities"]
        >= int(gates["minimum_valuation_sell_opportunities"]),
        "technical_confirmed_buys": diagnostics["technical_confirmed_buys"]
        >= int(gates["minimum_technical_confirmed_buys"]),
        "technical_confirmed_sells": diagnostics["technical_confirmed_sells"]
        >= int(gates["minimum_technical_confirmed_sells"]),
        "evaluation_years": overlay_base["elapsed_years"]
        >= float(gates["minimum_evaluation_years"]),
    }
    performance = {
        "base_total_return_improved": overlay_base["total_return"] > baseline_base["total_return"],
        "base_max_drawdown_no_worse": overlay_base["max_drawdown"] >= baseline_base["max_drawdown"],
        "stress_total_return_improved": overlay_stress["total_return"] > baseline_stress["total_return"],
        "both_chronological_halves_positive_relative_return": all(
            item["relative_return"] > 0 for item in relative["chronological_halves"]
        ),
    }
    performance_pass_count = sum(performance.values())
    if all(evidence.values()) and all(performance.values()):
        status = "RETROSPECTIVE_SCREEN_PASS_FORWARD_ONLY"
    elif (
        all(evidence.values())
        and performance_pass_count >= int(gates["near_miss_minimum_performance_checks"])
    ):
        status = "NEAR_MISS_FORWARD_OBSERVATION_ONLY"
    elif not all(evidence.values()):
        status = "INSUFFICIENT_EVIDENCE"
    else:
        status = "REJECTED_RETROSPECTIVE"
    return {
        "status": status,
        "evidence_checks": evidence,
        "performance_checks": performance,
        "performance_pass_count": performance_pass_count,
    }


def _load_candidate_family(
    config: dict[str, Any],
    daily: pd.DataFrame,
) -> dict[str, tuple[dict, dict, str]]:
    paths = {key: ROOT / value for key, value in config["data"].items()}
    factors = pd.read_parquet(paths["intraday_factors_file"])
    indicator = pd.read_parquet(paths["intraday_indicator_file"])
    candidates = build_standard_candidate_events(factors, indicator)

    vwap_trades = pd.read_parquet(paths["vwap_v1_trades_file"])
    vwap_trades = vwap_trades.loc[
        vwap_trades["scenario"].eq("BASE_COST") & vwap_trades["period"].eq("full_period")
    ]
    candidates["VWAP_REVERSION_V1"] = (
        *events_from_trade_records(vwap_trades, "VWAP_REVERSION_V1"),
        "既有预注册累计VWAP偏离后反转信号",
    )

    anchor_trades = pd.read_csv(
        paths["previous_close_anchor_trades_file"], encoding="utf-8-sig"
    )
    candidates["PREVIOUS_CLOSE_ANCHOR_V2"] = (
        *events_from_trade_records(anchor_trades, "PREVIOUS_CLOSE_ANCHOR_V2"),
        "既有前一交易日收盘固定锚反转信号",
    )

    forecasts = pd.read_parquet(paths["close_pressure_forecasts_file"])
    candidates["CLOSE_PRESSURE_30M_MODEL"] = (
        *events_from_close_pressure_forecasts(forecasts, daily),
        "收盘前30分钟价量压力模型概率相对当时扩展基准率",
    )

    lc_config = yaml.safe_load(paths["lc_fvg_pv_config_file"].read_text(encoding="utf-8"))
    minute = pd.read_parquet(paths["minute_file"])
    lc_features = prepare_features(minute, lc_config)
    lc_trades = simulate_trades(lc_features, lc_config)
    lc_records = lc_trades.rename(
        columns={"entry_time": "entry_execution_time", "entry_raw_price": "entry_raw_open_cny"}
    )
    candidates["LC_FVG_PV_V1"] = (
        *events_from_trade_records(lc_records, "LC_FVG_PV_V1"),
        "既有流动性扫高低、分钟FVG、相对成交量与VWAP确认",
    )
    return candidates


def _render_markdown(payload: dict[str, Any]) -> str:
    def pct(value: float) -> str:
        return f"{value:.2%}"

    baseline = payload["baseline"]["base"]
    lines = [
        "# 510300估值仓位与技术指标统一重筛",
        "",
        f"> 本轮一次性比较{payload['candidate_count']}条技术轨道。估值模型决定仓位，技术指标只优化普通加减仓时点；任何历史赢家都不自动授权交易。",
        "",
        "## 数据纠正",
        "",
        "- 正式输入是2021-08-12至2026-08-12的五年Tushare代理1分钟K线聚合15分钟OHLCVA，共1211个完整交易日。",
        "- 日OHLCVA与独立日线1211日逐日完全一致；来源仍有第三方代理和时间戳语义警告。",
        "- 旧的“两年High/Low限制”只属于通达信原生15分钟主文件，不代表仓库缺少五年可审计分钟OHLCV。",
        "",
        "## 估值基线",
        "",
        f"- 5bp总收益{pct(baseline['total_return'])}，CAGR {pct(baseline['cagr'])}，最大回撤{pct(baseline['max_drawdown'])}，期末权益{baseline['ending_equity']:.2f}元。",
        "",
        "## 候选排名",
        "",
        "| 排名 | 技术轨道 | 5bp总收益 | 相对基线权益 | 最大回撤 | 15bp总收益 | 买/卖确认 | 分段 | 结论 |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, item in enumerate(payload["ranking"], start=1):
        halves = item["relative_diagnostics"]["chronological_halves"]
        half_text = "/".join(f"{part['relative_return']:+.2%}" for part in halves)
        diagnostics = item["diagnostics"]
        lines.append(
            f"| {rank} | {item['candidate_id']} | {pct(item['base']['total_return'])} | "
            f"{item['relative_diagnostics']['ending_equity_difference_cny']:+.2f}元 | "
            f"{pct(item['base']['max_drawdown'])} | {pct(item['stress']['total_return'])} | "
            f"{diagnostics['technical_confirmed_buys']}/{diagnostics['technical_confirmed_sells']} | "
            f"{half_text} | `{item['decision']['status']}` |"
        )
    lines.extend(
        [
            "",
            "## 治理结论",
            "",
            f"- 历史筛选通过：{payload['selection_summary']['retrospective_pass_count']}条；近似可用：{payload['selection_summary']['near_miss_count']}条。",
            f"- 推荐纸面前向观察：`{payload['selection_summary']['forward_observation_candidate'] or '无'}`。",
            "- 本轮同时比较多条候选，存在多重检验与赢家诅咒；即便历史门槛全过，也只能冻结后向前观察。",
            "- 两万元正式仓位生成器保持不变。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    paths = {key: ROOT / value for key, value in config["data"].items()}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"统一技术重筛缺少输入：{path}")
    valuation = pd.read_parquet(paths["valuation_signals_file"])
    daily = pd.read_parquet(paths["daily_file"])
    dividends = pd.read_csv(paths["dividend_file"])
    start = pd.Timestamp(config["evaluation"]["start_date"])
    end = pd.Timestamp(config["evaluation"]["end_date"])
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.loc[daily["date"].between(start, end)].copy()
    targets = prepare_executable_targets(
        valuation, daily["date"], float(config["evaluation"]["position_grid_step"])
    )
    candidates = _load_candidate_family(config, daily)

    def make_costs(slippage: float) -> OverlayCosts:
        if float(config["costs"]["market_impact_bps"]) != 0.0:
            raise ValueError("两万元账户市场冲击成本必须为0")
        return OverlayCosts(
            commission_rate=float(config["costs"]["commission_rate"]),
            minimum_commission_cny=float(config["costs"]["minimum_commission_cny"]),
            slippage_bps_per_leg=slippage,
            lot_size=int(config["account"]["lot_size"]),
            minimum_trade_shares=int(config["account"]["minimum_normal_trade_shares"]),
            cash_annual_rate=float(config["account"]["cash_annual_rate"]),
        )

    initial_cash = float(config["evaluation"]["initial_cash_cny"])
    maximum_wait = int(config["evaluation"]["maximum_timing_wait_trading_days"])
    baseline_runs = {}
    for suffix, slippage in [
        ("base", float(config["costs"]["base_slippage_bps_per_leg"])),
        ("stress", float(config["costs"]["stress_slippage_bps_per_leg"])),
    ]:
        baseline_runs[suffix] = run_valuation_entry_policy(
            daily, targets, dividends, {}, {}, make_costs(slippage), initial_cash,
            "VALUATION_OPEN_BASELINE", maximum_wait,
        )
    baseline_summary = {
        key: summarize_policy(value[0], value[1], initial_cash)
        for key, value in baseline_runs.items()
    }

    output_dir = ROOT / config["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate_id, (buy_events, sell_events, description) in candidates.items():
        runs = {}
        for suffix, slippage in [
            ("base", float(config["costs"]["base_slippage_bps_per_leg"])),
            ("stress", float(config["costs"]["stress_slippage_bps_per_leg"])),
        ]:
            runs[suffix] = run_valuation_entry_policy(
                daily, targets, dividends, buy_events, sell_events,
                make_costs(slippage), initial_cash, "VALUATION_TECHNICAL_OVERLAY",
                maximum_wait,
            )
        summaries = {
            key: summarize_policy(value[0], value[1], initial_cash)
            for key, value in runs.items()
        }
        relative = _relative_diagnostics(baseline_runs["base"][0], runs["base"][0])
        decision = _candidate_decision(
            config, baseline_summary["base"], baseline_summary["stress"],
            summaries["base"], summaries["stress"], runs["base"][2], relative,
        )
        results.append(
            {
                "candidate_id": candidate_id,
                "description": description,
                "raw_buy_event_days": len(buy_events),
                "raw_sell_event_days": len(sell_events),
                "base": summaries["base"],
                "stress": summaries["stress"],
                "diagnostics": runs["base"][2],
                "relative_diagnostics": relative,
                "decision": decision,
            }
        )
        safe_name = candidate_id.lower()
        runs["base"][0].to_parquet(output_dir / f"{safe_name}_ledger.parquet", index=False)
        runs["base"][1].to_csv(output_dir / f"{safe_name}_trades.csv", index=False, encoding="utf-8-sig")

    ranking = sorted(
        results,
        key=lambda item: item["relative_diagnostics"]["ending_equity_difference_cny"],
        reverse=True,
    )
    passes = [item for item in ranking if item["decision"]["status"] == "RETROSPECTIVE_SCREEN_PASS_FORWARD_ONLY"]
    near = [item for item in ranking if item["decision"]["status"] == "NEAR_MISS_FORWARD_OBSERVATION_ONLY"]
    forward_candidate = (passes or near or [None])[0]
    payload = {
        "status": "PASS_SCREEN_COMPLETED",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")),
        "research": config["research"],
        "candidate_count": len(ranking),
        "baseline": baseline_summary,
        "ranking": ranking,
        "selection_summary": {
            "retrospective_pass_count": len(passes),
            "near_miss_count": len(near),
            "forward_observation_candidate": (
                forward_candidate["candidate_id"] if forward_candidate is not None else None
            ),
            "live_or_paper_position_integration_authorized": False,
        },
        "governance": config["governance"],
    }
    report_json = ROOT / config["output"]["report_json"]
    report_markdown = ROOT / config["output"]["report_markdown"]
    report_json.parent.mkdir(parents=True, exist_ok=True)
    ready = _json_ready(payload)
    report_json.write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding="utf-8")
    report_markdown.write_text(_render_markdown(ready), encoding="utf-8")
    print(json.dumps(ready, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
