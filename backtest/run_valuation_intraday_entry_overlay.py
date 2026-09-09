"""运行估值仓位+十五分钟技术进场叠加审计。"""

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
    prepare_intraday_events,
    run_valuation_entry_policy,
    summarize_policy,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "valuation_intraday_entry_overlay.yaml"


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
    for name, frame in [("前半段", merged.iloc[:midpoint]), ("后半段", merged.iloc[midpoint:])]:
        overlay_return = float((1.0 + frame["daily_return_overlay"]).prod() - 1.0)
        baseline_return = float((1.0 + frame["daily_return_baseline"]).prod() - 1.0)
        halves.append(
            {
                "period": name,
                "start_date": str(frame["date"].iloc[0].date()),
                "end_date": str(frame["date"].iloc[-1].date()),
                "overlay_return": overlay_return,
                "baseline_return": baseline_return,
                "relative_return": overlay_return - baseline_return,
            }
        )
    return {
        "ending_equity_difference_cny": float(
            merged["equity_overlay"].iloc[-1] - merged["equity_baseline"].iloc[-1]
        ),
        "chronological_halves": halves,
    }


def _gate_result(
    config: dict[str, Any],
    baseline: dict[str, Any],
    overlay: dict[str, Any],
    baseline_stress: dict[str, Any],
    overlay_stress: dict[str, Any],
    diagnostics: dict[str, Any],
    relative: dict[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    checks = {
        "valuation_buy_opportunities": diagnostics["valuation_buy_opportunities"]
        >= int(gates["minimum_valuation_buy_opportunities"]),
        "technical_confirmed_entries": diagnostics["technical_confirmed_buys"]
        >= int(gates["minimum_confirmed_technical_entries"]),
        "valuation_sell_opportunities": diagnostics["valuation_sell_opportunities"]
        >= int(gates["minimum_valuation_sell_opportunities"]),
        "technical_confirmed_exits": diagnostics["technical_confirmed_sells"]
        >= int(gates["minimum_confirmed_technical_exits"]),
        "evaluation_years": overlay["elapsed_years"]
        >= float(gates["minimum_evaluation_years"]),
        "base_total_return_improved": overlay["total_return"] > baseline["total_return"],
        "base_max_drawdown_no_worse": overlay["max_drawdown"] >= baseline["max_drawdown"],
        "stress_total_return_improved": overlay_stress["total_return"]
        > baseline_stress["total_return"],
        "both_halves_positive_relative_return": all(
            item["relative_return"] > 0
            for item in relative["chronological_halves"]
        ),
    }
    evidence_checks = [
        checks["valuation_buy_opportunities"],
        checks["technical_confirmed_entries"],
        checks["valuation_sell_opportunities"],
        checks["technical_confirmed_exits"],
        checks["evaluation_years"],
    ]
    performance_checks = [
        checks["base_total_return_improved"],
        checks["base_max_drawdown_no_worse"],
        checks["stress_total_return_improved"],
        checks["both_halves_positive_relative_return"],
    ]
    if not all(evidence_checks) and not all(performance_checks):
        status = "RETROSPECTIVE_FAIL_AND_INSUFFICIENT_EVIDENCE_DO_NOT_INTEGRATE"
    elif not all(evidence_checks):
        status = "INSUFFICIENT_EVIDENCE_DO_NOT_INTEGRATE"
    elif all(performance_checks):
        status = "RETROSPECTIVE_PASS_FORWARD_VALIDATION_REQUIRED"
    else:
        status = "REJECTED_DO_NOT_INTEGRATE"
    return {"status": status, "checks": checks, "all_checks_passed": all(checks.values())}


def _markdown(payload: dict[str, Any]) -> str:
    def pct(value: float) -> str:
        return f"{value:.2%}"

    base = payload["results"]["baseline_base"]
    overlay = payload["results"]["overlay_base"]
    base_stress = payload["results"]["baseline_stress"]
    overlay_stress = payload["results"]["overlay_stress"]
    diagnostics = payload["diagnostics"]["overlay_base"]
    lines = [
        "# 510300估值仓位与十五分钟加减仓时点叠加审计",
        "",
        f"> 结论状态：`{payload['decision']['status']}`。本结果是回顾性审计，不是严格样本外证明。",
        "",
        "## 组合口径",
        "",
        "- 估值模型独占目标仓位决定权；技术指标不得提高或降低目标仓位。",
        "- 普通加仓和减仓都最多等待5个完整交易日；危机强制减仓下一交易日开盘直接执行。",
        "- 五日内出现对应十五分钟进场/退出事件，则在信号后下一根K线开盘交易；否则下一交易日开盘回退执行。",
        "- 两万元、25%仓位档、100份整手、普通调仓至少1000份、最低佣金5元；市场冲击成本为0。",
        "",
        "## 回测结果",
        "",
        "| 场景 | 总收益 | CAGR | 最大回撤 | 成交数 | 期末权益 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 估值开盘基线（5bp） | {pct(base['total_return'])} | {pct(base['cagr'])} | {pct(base['max_drawdown'])} | {base['trade_count']} | {base['ending_equity']:.2f}元 |",
        f"| 估值+技术进场（5bp） | {pct(overlay['total_return'])} | {pct(overlay['cagr'])} | {pct(overlay['max_drawdown'])} | {overlay['trade_count']} | {overlay['ending_equity']:.2f}元 |",
        f"| 估值开盘基线（15bp） | {pct(base_stress['total_return'])} | {pct(base_stress['cagr'])} | {pct(base_stress['max_drawdown'])} | {base_stress['trade_count']} | {base_stress['ending_equity']:.2f}元 |",
        f"| 估值+技术进场（15bp） | {pct(overlay_stress['total_return'])} | {pct(overlay_stress['cagr'])} | {pct(overlay_stress['max_drawdown'])} | {overlay_stress['trade_count']} | {overlay_stress['ending_equity']:.2f}元 |",
        "",
        "## 样本与执行",
        "",
        f"- 估值加仓机会：{diagnostics['valuation_buy_opportunities']}次。",
        f"- 技术确认后加仓：{diagnostics['technical_confirmed_buys']}次；等待超时回退：{diagnostics['fallback_buys']}次；挂起方向取消/反转：{diagnostics['cancelled_pending_orders']}次。",
        f"- 技术确认后减仓：{diagnostics['technical_confirmed_sells']}次；等待超时回退：{diagnostics['fallback_sells']}次；危机立即减仓：{diagnostics['risk_off_immediate_sells']}次。",
        f"- 平均等待：{diagnostics['average_technical_wait_days'] if diagnostics['average_technical_wait_days'] is not None else '无'}个交易日。",
        "",
        "## 验收",
        "",
    ]
    for name, passed in payload["decision"]["checks"].items():
        lines.append(f"- {'通过' if passed else '未通过'}：`{name}`。")
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "- 原生十五分钟OHLCV只从2024年7月底开始，无法覆盖估值模型的完整五年历史。",
            "- 当前阈值虽在本次收益计算前冻结，但指标设计已使用既有历史背景，因此仍属于回顾性证据。",
            "- 震荡低吸事件极少，不能与趋势突破合并后宣称两条轨道都有效。",
            "- 未通过全部门槛前，不修改两万元纸面仓位生成器。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    paths = {key: ROOT / value for key, value in config["data"].items()}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"回测缺少输入：{path}")

    valuation = pd.read_parquet(paths["valuation_signals_file"])
    indicator = pd.read_parquet(paths["intraday_indicator_file"])
    daily = pd.read_parquet(paths["daily_etf_file"])
    dividends = pd.read_csv(paths["dividend_file"])
    start = pd.Timestamp(config["evaluation"]["start_date"])
    end = pd.Timestamp(config["evaluation"]["end_date"])
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.loc[daily["date"].between(start, end)].copy()
    targets = prepare_executable_targets(
        valuation,
        daily["date"],
        float(config["evaluation"]["position_grid_step"]),
    )
    buy_events = prepare_intraday_events(
        indicator,
        str(config["evaluation"]["entry_event_column"]),
        "entry_signal",
        set(config["evaluation"]["accepted_entry_signals"]),
    )
    sell_events = prepare_intraday_events(
        indicator,
        str(config["evaluation"]["exit_event_column"]),
        "exit_signal",
        set(config["evaluation"]["accepted_exit_signals"]),
    )

    def costs(slippage: float) -> OverlayCosts:
        account = config["account"]
        cost = config["costs"]
        if float(cost["market_impact_bps"]) != 0.0:
            raise ValueError("两万元510300审计明确要求市场冲击成本为0")
        return OverlayCosts(
            commission_rate=float(cost["commission_rate"]),
            minimum_commission_cny=float(cost["minimum_commission_cny"]),
            slippage_bps_per_leg=slippage,
            lot_size=int(account["lot_size"]),
            minimum_trade_shares=int(account["minimum_normal_trade_shares"]),
            cash_annual_rate=float(account["cash_annual_rate"]),
        )

    initial_cash = float(config["evaluation"]["initial_cash_cny"])
    maximum_wait = int(config["evaluation"]["maximum_timing_wait_trading_days"])
    scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    for suffix, slippage in [
        ("base", float(config["costs"]["base_slippage_bps_per_leg"])),
        ("stress", float(config["costs"]["stress_slippage_bps_per_leg"])),
    ]:
        for prefix, policy in [
            ("baseline", "VALUATION_OPEN_BASELINE"),
            ("overlay", "VALUATION_TECHNICAL_OVERLAY"),
        ]:
            scenarios[f"{prefix}_{suffix}"] = run_valuation_entry_policy(
                daily,
                targets,
                dividends,
                buy_events,
                sell_events,
                costs(slippage),
                initial_cash,
                policy,
                maximum_wait,
            )

    summaries = {
        name: summarize_policy(ledger, trades, initial_cash)
        for name, (ledger, trades, _) in scenarios.items()
    }
    relative = _relative_diagnostics(
        scenarios["baseline_base"][0], scenarios["overlay_base"][0]
    )
    decision = _gate_result(
        config,
        summaries["baseline_base"],
        summaries["overlay_base"],
        summaries["baseline_stress"],
        summaries["overlay_stress"],
        scenarios["overlay_base"][2],
        relative,
    )
    payload = {
        "status": "PASS_BACKTEST_COMPLETED",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")),
        "strategy": config["strategy"],
        "assumptions": {
            "valuation_controls_position": True,
            "technical_indicator_controls_buy_and_normal_sell_timing": True,
            "risk_off_sell_waits_for_technical_signal": False,
            "market_impact_bps": float(config["costs"]["market_impact_bps"]),
            "maximum_timing_wait_trading_days": maximum_wait,
        },
        "results": summaries,
        "relative_diagnostics": relative,
        "diagnostics": {name: data[2] for name, data in scenarios.items()},
        "decision": decision,
        "governance": config["governance"],
    }
    output = config["output"]
    output_dir = ROOT / output["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (ledger, trades, _) in scenarios.items():
        ledger.to_parquet(output_dir / f"{name}_ledger.parquet", index=False)
        trades.to_csv(output_dir / f"{name}_trades.csv", index=False, encoding="utf-8-sig")
    report_json = ROOT / output["report_json"]
    report_markdown = ROOT / output["report_markdown"]
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_markdown.write_text(_markdown(_json_ready(payload)), encoding="utf-8")
    print(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
