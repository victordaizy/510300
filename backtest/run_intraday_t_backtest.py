"""运行510300底仓-VWAP做T V1预注册回测并生成拒绝/前向验证决策。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.intraday_t_engine import (
    SimulationResult,
    build_intraday_features,
    calendar_year_metrics,
    load_strategy_config,
    performance_metrics,
    simulate_period,
    trade_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "intraday_t_strategy.yaml"
AUDIT_FILE = (
    PROJECT_ROOT / "reports" / "data_quality" / "510300_1m_t_strategy_readiness.json"
)
RESULT_FILE = PROJECT_ROOT / "reports" / "research" / "intraday_t_v1_results.json"
DECISION_FILE = PROJECT_ROOT / "reports" / "research" / "intraday_t_v1_decision.md"
TRADES_FILE = PROJECT_ROOT / "data" / "backtests" / "intraday_t_v1_trades.parquet"
LEDGER_FILE = PROJECT_ROOT / "data" / "backtests" / "intraday_t_v1_daily_ledger.parquet"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _write_parquet_atomic(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    data.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _summarize_result(
    result: SimulationResult,
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    initial_cash = float(config["account"]["initial_cash_cny"])
    annual_days = int(evaluation["annual_trading_days"])
    complete_days = int(evaluation["complete_calendar_year_minimum_trading_days"])
    ledger = result.daily_ledger
    trades = result.trades
    strategy = performance_metrics(ledger["strategy_nav_cny"], initial_cash, annual_days)
    static_core = performance_metrics(ledger["static_core_nav_cny"], initial_cash, annual_days)
    full_buyhold = performance_metrics(ledger["full_buyhold_nav_cny"], initial_cash, annual_days)
    trade_summary = trade_metrics(trades)
    years = calendar_year_metrics(ledger, trades, initial_cash, complete_days)
    t_net_pnl = float(trade_summary["net_pnl_cny"])
    return {
        "scenario": result.scenario,
        "period": result.period,
        "requested_start": result.start_date,
        "requested_end": result.end_date,
        "actual_start": ledger["date"].min(),
        "actual_end": ledger["date"].max(),
        "strategy": strategy,
        "static_2500_core": static_core,
        "full_capital_buy_and_hold": full_buyhold,
        "intraday_t": trade_summary,
        "calendar_years": years,
        "pnl_attribution": {
            "static_core_total_pnl_cny": float(static_core["total_pnl_cny"]),
            "intraday_t_net_pnl_cny": t_net_pnl,
            "strategy_total_pnl_cny": float(strategy["total_pnl_cny"]),
            "static_plus_t_pnl_cny": float(static_core["total_pnl_cny"]) + t_net_pnl,
            "attribution_error_cny": float(
                strategy["total_pnl_cny"] - static_core["total_pnl_cny"] - t_net_pnl
            ),
        },
        "execution_metadata": result.metadata,
    }


def _build_gate_decision(
    summaries: dict[str, dict[str, dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    primary = evaluation["primary_gate"]
    edge = evaluation["edge_gate"]
    full = summaries["BASE_COST"]["full_period"]
    pseudo = summaries["BASE_COST"]["pseudo_out_of_sample"]
    pseudo_stress = summaries["STRESS_SLIPPAGE"]["pseudo_out_of_sample"]
    complete_years = [
        item for item in full["calendar_years"] if item["is_complete_calendar_year"]
    ]
    profitable_complete_year_ratio = (
        sum(item["t_net_pnl_cny"] > 0.0 for item in complete_years) / len(complete_years)
        if complete_years
        else 0.0
    )
    complete_year_trade_counts = [item["round_trip_count"] for item in complete_years]
    minimum_year_trades = min(complete_year_trade_counts) if complete_year_trade_counts else 0
    maximum_year_trades = max(complete_year_trade_counts) if complete_year_trade_counts else 0

    checks = {
        "full_period_cagr": {
            "actual": full["strategy"]["cagr"],
            "requirement": f">={float(primary['full_period_cagr_minimum']):.2%}",
            "passed": full["strategy"]["cagr"] >= float(primary["full_period_cagr_minimum"]),
        },
        "rolling_242_day_median_return": {
            "actual": full["strategy"]["rolling_242_day_median_return"],
            "requirement": f">={float(primary['rolling_242_day_median_return_minimum']):.2%}",
            "passed": full["strategy"]["rolling_242_day_median_return"]
            >= float(primary["rolling_242_day_median_return_minimum"]),
        },
        "rolling_242_day_above_20pct_ratio": {
            "actual": full["strategy"]["rolling_242_day_return_above_20pct_ratio"],
            "requirement": f">={float(primary['rolling_242_day_return_above_20pct_ratio_minimum']):.2%}",
            "passed": full["strategy"]["rolling_242_day_return_above_20pct_ratio"]
            >= float(primary["rolling_242_day_return_above_20pct_ratio_minimum"]),
        },
        "maximum_drawdown": {
            "actual": full["strategy"]["maximum_drawdown"],
            "requirement": f"<={float(primary['maximum_drawdown_ceiling']):.2%}",
            "passed": full["strategy"]["maximum_drawdown"]
            <= float(primary["maximum_drawdown_ceiling"]),
        },
        "pseudo_out_of_sample_t_net_pnl": {
            "actual": pseudo["intraday_t"]["net_pnl_cny"],
            "requirement": f">{float(edge['pseudo_out_of_sample_t_net_pnl_minimum_cny']):.2f}元",
            "passed": pseudo["intraday_t"]["net_pnl_cny"]
            > float(edge["pseudo_out_of_sample_t_net_pnl_minimum_cny"]),
        },
        "pseudo_out_of_sample_stress_t_net_pnl": {
            "actual": pseudo_stress["intraday_t"]["net_pnl_cny"],
            "requirement": f">{float(edge['pseudo_out_of_sample_stress_t_net_pnl_minimum_cny']):.2f}元",
            "passed": pseudo_stress["intraday_t"]["net_pnl_cny"]
            > float(edge["pseudo_out_of_sample_stress_t_net_pnl_minimum_cny"]),
        },
        "profitable_complete_calendar_year_ratio": {
            "actual": profitable_complete_year_ratio,
            "requirement": f">={float(edge['profitable_complete_calendar_year_ratio_minimum']):.2%}",
            "passed": profitable_complete_year_ratio
            >= float(edge["profitable_complete_calendar_year_ratio_minimum"]),
        },
        "full_period_profit_factor": {
            "actual": full["intraday_t"]["profit_factor"],
            "requirement": f">={float(edge['profit_factor_minimum']):.2f}",
            "passed": full["intraday_t"]["profit_factor"] >= float(edge["profit_factor_minimum"]),
        },
        "minimum_round_trips_each_complete_year": {
            "actual": minimum_year_trades,
            "requirement": f">={int(edge['minimum_round_trips_per_complete_year'])}",
            "passed": bool(complete_years)
            and minimum_year_trades >= int(edge["minimum_round_trips_per_complete_year"]),
        },
        "maximum_round_trips_each_complete_year": {
            "actual": maximum_year_trades,
            "requirement": f"<={int(edge['maximum_round_trips_per_complete_year'])}",
            "passed": bool(complete_years)
            and maximum_year_trades <= int(edge["maximum_round_trips_per_complete_year"]),
        },
    }
    all_passed = all(item["passed"] for item in checks.values())
    return {
        "decision": "QUALIFIED_FOR_FORWARD_PAPER_TEST" if all_passed else "REJECTED_V1",
        "all_gates_passed": all_passed,
        "strict_holdout_available": False,
        "complete_calendar_years": [item["year"] for item in complete_years],
        "checks": checks,
        "next_action": (
            "冻结V1并从下一个交易日起做完全前向模拟，不投入实盘资金"
            if all_passed
            else "保留V1拒绝记录，不在同一样本调参；先做机制归因，再预注册结构不同的V2"
        ),
    }


def _format_percentage(value: float | None) -> str:
    return "不可用" if value is None or not math.isfinite(float(value)) else f"{float(value):.2%}"


def _format_number(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "不可用"
    number = float(value)
    if math.isinf(number):
        return "∞"
    if math.isnan(number):
        return "不可用"
    return f"{number:,.{decimals}f}"


def _render_markdown(report: dict[str, Any]) -> str:
    summaries = report["summaries"]
    decision = report["decision"]
    full = summaries["BASE_COST"]["full_period"]
    stress_full = summaries["STRESS_SLIPPAGE"]["full_period"]
    period_names = {
        "full_period": "全历史",
        "development": "开发期",
        "pseudo_out_of_sample": "伪样本外",
        "contaminated_recent_period": "已污染近期段",
    }
    scenario_names = {"BASE_COST": "基础成本", "STRESS_SLIPPAGE": "压力滑点"}
    rows: list[str] = []
    for scenario, period_map in summaries.items():
        for period, item in period_map.items():
            rows.append(
                "| {scenario} | {period} | {cagr} | {static} | {buyhold} | {t_pnl} | {trades} | {pf} | {drawdown} |".format(
                    scenario=scenario_names[scenario],
                    period=period_names[period],
                    cagr=_format_percentage(item["strategy"]["cagr"]),
                    static=_format_percentage(item["static_2500_core"]["cagr"]),
                    buyhold=_format_percentage(item["full_capital_buy_and_hold"]["cagr"]),
                    t_pnl=_format_number(item["intraday_t"]["net_pnl_cny"]),
                    trades=item["intraday_t"]["round_trip_count"],
                    pf=_format_number(item["intraday_t"]["profit_factor"]),
                    drawdown=_format_percentage(item["strategy"]["maximum_drawdown"]),
                )
            )
    gate_names = {
        "full_period_cagr": "全历史年化收益",
        "rolling_242_day_median_return": "滚动242日收益中位数",
        "rolling_242_day_above_20pct_ratio": "滚动242日收益超过20%的比例",
        "maximum_drawdown": "最大回撤",
        "pseudo_out_of_sample_t_net_pnl": "伪样本外做T净收益",
        "pseudo_out_of_sample_stress_t_net_pnl": "压力成本伪样本外做T净收益",
        "profitable_complete_calendar_year_ratio": "做T盈利完整年度比例",
        "full_period_profit_factor": "全历史做T利润因子",
        "minimum_round_trips_each_complete_year": "完整年度最少交易次数",
        "maximum_round_trips_each_complete_year": "完整年度最多交易次数",
    }
    percentage_gates = {
        "full_period_cagr",
        "rolling_242_day_median_return",
        "rolling_242_day_above_20pct_ratio",
        "maximum_drawdown",
        "profitable_complete_calendar_year_ratio",
    }
    gate_rows: list[str] = []
    for key, item in decision["checks"].items():
        actual = (
            _format_percentage(item["actual"])
            if key in percentage_gates
            else _format_number(item["actual"])
        )
        gate_rows.append(
            f"| {gate_names[key]} | {actual} | {item['requirement']} | {'通过' if item['passed'] else '未通过'} |"
        )
    year_rows = [
        "| {year} | {strategy} | {static} | {t_pnl} | {trades} | {complete} |".format(
            year=item["year"],
            strategy=_format_percentage(item["strategy_return"]),
            static=_format_percentage(item["static_core_return"]),
            t_pnl=_format_number(item["t_net_pnl_cny"]),
            trades=item["round_trip_count"],
            complete="是" if item["is_complete_calendar_year"] else "否",
        )
        for item in full["calendar_years"]
    ]
    full_trade = full["intraday_t"]
    stress_trade = stress_full["intraday_t"]
    return f"""# 510300底仓-VWAP做T V1回测决策

结论：**{decision['decision']}**。V1 的预注册门槛{'全部通过' if decision['all_gates_passed'] else '未能全部通过'}；这是一项历史研究结论，不是收益承诺。

最重要的拆分是：基础成本下，2500份静态底仓贡献 **{_format_number(full['pnl_attribution']['static_core_total_pnl_cny'])} 元**，做T贡献 **{_format_number(full['pnl_attribution']['intraday_t_net_pnl_cny'])} 元**，账户合计损益 **{_format_number(full['pnl_attribution']['strategy_total_pnl_cny'])} 元**。因此不能把指数本身上涨误算成做T能力。

## 各区间与成本情景

| 情景 | 区间 | 策略年化 | 2500份静态底仓年化 | 满仓持有年化 | 做T净收益（元） | 回合数 | 利润因子 | 最大回撤 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 预注册门槛

| 门槛 | 实际值 | 要求 | 结论 |
|---|---:|---:|---:|
{chr(10).join(gate_rows)}

## 自然年度稳定性

| 年度 | 策略收益 | 静态底仓收益 | 做T净收益（元） | 回合数 | 完整年度 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(year_rows)}

## 成本与执行归因

- 基础成本：原始开盘价之间的毛收益 {_format_number(full_trade['gross_raw_pnl_cny'])} 元，滑点与最小价位损耗 {_format_number(full_trade['slippage_and_tick_cost_cny'])} 元，佣金 {_format_number(full_trade['commission_cny'])} 元，最终做T净收益 {_format_number(full_trade['net_pnl_cny'])} 元。
- 压力滑点：滑点与最小价位损耗 {_format_number(stress_trade['slippage_and_tick_cost_cny'])} 元，佣金 {_format_number(stress_trade['commission_cny'])} 元，最终做T净收益 {_format_number(stress_trade['net_pnl_cny'])} 元。
- 每个信号只使用当前及更早记录；信号后的下一条记录开盘成交。每天最多一个回合，日终必须恢复2500份。
- 历史区间已经被探索过，没有真正未查看的严格样本外。伪样本外和近期段只能提供稳健性证据，不能替代未来验证。

## 下一步

{decision['next_action']}。
"""


def main() -> int:
    config = load_strategy_config(CONFIG_FILE)
    if not AUDIT_FILE.exists():
        raise FileNotFoundError("缺少一分钟做T数据审计报告")
    audit = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("一分钟数据审计未通过，拒绝运行策略回测")

    minute_file = PROJECT_ROOT / config["data"]["minute_file"]
    dividend_file = PROJECT_ROOT / config["data"]["dividend_file"]
    minute = pd.read_parquet(minute_file)
    minute_features = build_intraday_features(minute)
    dividends = pd.read_csv(dividend_file, encoding="utf-8-sig")
    minute_dates = pd.to_datetime(minute["trade_time"]).dt.normalize()
    periods: dict[str, tuple[str, str]] = {
        "full_period": (
            minute_dates.min().date().isoformat(),
            minute_dates.max().date().isoformat(),
        )
    }
    for name, values in config["research_partitions"].items():
        if isinstance(values, dict) and "start" in values and "end" in values:
            periods[name] = (str(values["start"]), str(values["end"]))

    scenarios = {
        "BASE_COST": float(config["costs"]["base_slippage_bps_per_leg"]),
        "STRESS_SLIPPAGE": float(config["costs"]["stress_slippage_bps_per_leg"]),
    }
    result_objects: list[SimulationResult] = []
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario, slippage in scenarios.items():
        summaries[scenario] = {}
        for period, (start, end) in periods.items():
            result = simulate_period(
                minute_features,
                dividends,
                config,
                start,
                end,
                period,
                scenario,
                slippage,
            )
            result_objects.append(result)
            summaries[scenario][period] = _summarize_result(result, config)
            print(
                f"完成：{scenario}/{period}，交易{len(result.trades)}回合",
                flush=True,
            )

    all_trades = pd.concat([result.trades for result in result_objects], ignore_index=True)
    all_ledgers = pd.concat([result.daily_ledger for result in result_objects], ignore_index=True)
    _write_parquet_atomic(all_trades, TRADES_FILE)
    _write_parquet_atomic(all_ledgers, LEDGER_FILE)
    decision = _build_gate_decision(summaries, config)
    report = {
        "strategy_id": config["strategy"]["strategy_id"],
        "strategy_version": config["strategy"]["version"],
        "research_status": decision["decision"],
        "evidence_scope": config["strategy"]["evidence_scope"],
        "inputs": {
            "strategy_config": {
                "path": CONFIG_FILE.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256_file(CONFIG_FILE),
            },
            "data_audit": {
                "path": AUDIT_FILE.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256_file(AUDIT_FILE),
            },
            "minute_data": {
                "path": minute_file.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256_file(minute_file),
            },
            "dividends": {
                "path": dividend_file.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256_file(dividend_file),
            },
        },
        "outputs": {
            "trades": TRADES_FILE.relative_to(PROJECT_ROOT).as_posix(),
            "daily_ledger": LEDGER_FILE.relative_to(PROJECT_ROOT).as_posix(),
        },
        "summaries": summaries,
        "decision": decision,
    }
    DECISION_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_markdown = DECISION_FILE.with_suffix(".md.tmp")
    temporary_markdown.write_text(_render_markdown(report), encoding="utf-8")
    temporary_markdown.replace(DECISION_FILE)
    temporary_json = RESULT_FILE.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(_json_ready(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_json.replace(RESULT_FILE)
    print(json.dumps(_json_ready(decision), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
