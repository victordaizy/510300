"""美国多资产容量感知期末减仓V4。"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.qdii_cross_market_discount_reversion_zero_variance_v2 import (
    evaluate_historical_returns_zero_variance_v2,
    load_correction_contract,
)
from research.us_leveraged_multi_asset_absolute_momentum_v1 import (
    CONSERVATIVE_EVALUATOR_CONFIG,
    atomic_json,
    atomic_parquet,
    atomic_text,
    build_monthly_selections,
    prepare_daily_context,
)
from research.us_liquid_treasury_multi_asset_absolute_momentum_v2 import (
    load_visible_inputs,
)
from research.us_multi_asset_capacity_aware_absolute_momentum_v3 import (
    load_contract as load_v3_contract,
    run_portfolio_backtest,
    validate_contract as validate_v3_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "us_multi_asset_capacity_aware_terminal_winddown_v4.yaml"

ALLOWED_CHANGED_PATHS = {
    "protocol.candidate_id",
    "protocol.parent_protocol_id",
    "protocol.lane",
    "protocol.mechanism",
    "inputs.parent_manifest",
    "portfolio.terminal_liquidation_time",
    "portfolio.terminal_open_positions_policy",
    "portfolio.terminal_winddown_common_trading_days",
    "portfolio.stop_new_entries_at_terminal_winddown_start",
    "portfolio.terminal_winddown_schedule_mode",
    "historical_evidence_limits.performance_screen_label",
    "outputs.input_audit_json",
    "outputs.manifest",
    "outputs.visible_daily_returns",
    "outputs.visible_selections",
    "outputs.visible_trades",
    "outputs.visible_report_json",
    "outputs.visible_report_markdown",
    "outputs.replication_report_json",
}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], child))
    else:
        result[prefix] = value
    return result


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def load_override(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V4期末减仓配置必须是YAML对象")
    return payload


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    override = load_override(path)
    if override.get("base_candidate_config") != (
        "config/us_multi_asset_capacity_aware_absolute_momentum_v3.yaml"
    ):
        raise ValueError("V4基础候选路径发生变化")
    base = load_v3_contract()
    contract = copy.deepcopy(base)
    _deep_update(contract, override["overrides"])
    validate_contract(contract, base=base, override=override)
    return contract


def validate_contract(
    contract: dict[str, Any],
    *,
    base: dict[str, Any] | None = None,
    override: dict[str, Any] | None = None,
) -> None:
    if override is None:
        override = load_override()
    if base is None:
        base = load_v3_contract()
    validate_v3_contract(base)
    left = _flatten(base)
    right = _flatten(contract)
    changed = {
        path
        for path in set(left) | set(right)
        if left.get(path) != right.get(path)
    }
    if changed != ALLOWED_CHANGED_PATHS:
        raise ValueError(
            "V4差异越过V10授权范围："
            f"unexpected={sorted(changed - ALLOWED_CHANGED_PATHS)}, "
            f"missing={sorted(ALLOWED_CHANGED_PATHS - changed)}"
        )
    reconstruction = override.get("execution_reconstitution", {})
    if reconstruction.get("reconstitution_of") != (
        "US_MULTI_ASSET_CAPACITY_AWARE_ABSOLUTE_MOMENTUM_V3"
    ):
        raise ValueError("V4执行重构父候选发生变化")
    if reconstruction.get("no_v3_performance_available") is not True:
        raise ValueError("V4未声明V3无绩效可用")
    if reconstruction.get("selection_products_costs_capacity_and_gates_unchanged") is not True:
        raise ValueError("V4未声明信号、产品、成本、容量和硬门不变")
    if contract["protocol"]["candidate_id"] != (
        "US_MULTI_ASSET_CAPACITY_AWARE_TERMINAL_WINDDOWN_V4"
    ):
        raise ValueError("V4候选编号不正确")
    if contract["protocol"]["parent_protocol_id"] != (
        "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V10"
    ):
        raise ValueError("V4父协议不正确")
    portfolio = contract["portfolio"]
    if int(portfolio.get("terminal_winddown_common_trading_days", 0)) != 20:
        raise ValueError("V4期末减仓窗口必须为20个共同交易日")
    if portfolio.get("stop_new_entries_at_terminal_winddown_start") is not True:
        raise ValueError("V4必须从减仓起点停止新开仓")
    if float(contract["universe"]["maximum_order_fraction_of_prior_median_dollar_turnover"]) != 0.0005:
        raise ValueError("V4容量比例必须保持0.05%")
    if any(bool(value) for value in contract["safety"].values()):
        raise ValueError("V4研究边界被打开")


def changed_paths(contract: dict[str, Any] | None = None) -> list[str]:
    base = load_v3_contract()
    reconstructed = load_contract() if contract is None else contract
    left = _flatten(base)
    right = _flatten(reconstructed)
    return sorted(
        path
        for path in set(left) | set(right)
        if left.get(path) != right.get(path)
    )


def build_terminal_winddown_plan(
    selections: pd.DataFrame,
    schedule: pd.DataFrame,
    features: pd.DataFrame,
    context: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """在最后20个共同交易日的第一日插入无持仓目标的强制退出事件。"""

    days = int(contract["portfolio"]["terminal_winddown_common_trading_days"])
    dates = pd.DatetimeIndex(pd.to_datetime(context["date"])).sort_values().unique()
    if len(dates) < days:
        raise DataContractError("V4日历不足冻结的期末减仓窗口")
    winddown_date = pd.Timestamp(dates[-days])
    execution_selections = selections.loc[
        pd.to_datetime(selections["execution_date"]).lt(winddown_date)
    ].copy()
    execution_schedule = schedule.loc[
        pd.to_datetime(schedule["execution_date"]).lt(winddown_date)
    ].copy()
    winddown_row = pd.DataFrame(
        {"signal_date": [winddown_date], "execution_date": [winddown_date]}
    )
    execution_schedule = pd.concat(
        [execution_schedule, winddown_row],
        ignore_index=True,
    ).sort_values("execution_date").reset_index(drop=True)
    execution_features = features.loc[
        pd.to_datetime(features["execution_date"]).lt(winddown_date)
    ].copy()
    audit = {
        "terminal_winddown_common_trading_days": days,
        "winddown_start_date": winddown_date.date().isoformat(),
        "visible_end_date": pd.Timestamp(dates[-1]).date().isoformat(),
        "original_selection_rows": int(len(selections)),
        "executed_selection_rows": int(len(execution_selections)),
        "original_schedule_rows": int(len(schedule)),
        "execution_schedule_rows_including_winddown": int(len(execution_schedule)),
        "new_entries_after_winddown": 0,
        "market_return_or_future_open_read": False,
    }
    return execution_selections, execution_schedule, execution_features, audit


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["evaluation"]["metrics"]
    gates = report["evaluation"]["gates"]
    portfolio = report["data_audit"]["portfolio"]
    winddown = report["data_audit"]["terminal_winddown"]
    return "\n".join(
        [
            "# 美国多资产容量感知期末减仓V4可见期结果",
            "",
            f"状态：`{report['status']}`",
            "",
            f"- 期末减仓起点：{winddown['winddown_start_date']}。",
            f"- H00300全收益CAGR：{metrics['benchmark_total_return_cagr']:.2%}。",
            f"- 基础/压力策略净CAGR：{metrics['strategy_base_net_cagr']:.2%}/{metrics['strategy_stress_net_cagr']:.2%}。",
            f"- 基础/压力年化净超额：{metrics['base_annualized_excess']:.2%}/{metrics['stress_annualized_excess']:.2%}。",
            f"- 基础/压力净夏普：{metrics['base_strategy_net_sharpe']:.3f}/{metrics['stress_strategy_net_sharpe']:.3f}。",
            f"- 最大实际订单/历史中位成交额：{portfolio['maximum_observed_order_capacity_fraction']:.4%}。",
            f"- 基础/压力40个百分点门：{gates['base_annualized_excess_at_least_40pct']}/{gates['stress_annualized_excess_at_least_40pct']}。",
            f"- 基础/压力1.50夏普门：{gates['base_strategy_sharpe_at_least_1_5']}/{gates['stress_strategy_sharpe_at_least_1_5']}。",
            f"- 全部门通过：{report['evaluation']['all_visible_gates_pass']}。",
            "",
            f"{report['decision']['next_step']}。",
            "",
            "封存复验、Paper、Shadow、订单、经纪商连接和实盘均未打开。",
            "",
        ]
    )


def run_visible(
    contract: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    validate_contract(contract)
    partition = contract["historical_partition"]
    start = pd.Timestamp(partition["visible_start"])
    end = pd.Timestamp(partition["visible_end"])
    panel, master, fx, benchmark, input_audit = load_visible_inputs(
        contract,
        signal_only=False,
    )
    selections, schedule, features, signal_audit = build_monthly_selections(
        panel,
        contract,
        start=start,
        end=end,
    )
    context = prepare_daily_context(
        panel,
        fx,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    execution_selections, execution_schedule, execution_features, winddown_audit = (
        build_terminal_winddown_plan(
            selections,
            schedule,
            features,
            context,
            contract,
        )
    )
    daily, trades, portfolio_audit = run_portfolio_backtest(
        execution_selections,
        execution_schedule,
        execution_features,
        panel,
        context,
        contract,
        start=start,
        end=end,
    )
    correction = load_correction_contract(CONSERVATIVE_EVALUATOR_CONFIG)
    evaluation = evaluate_historical_returns_zero_variance_v2(
        daily,
        contract,
        correction,
        bootstrap_repetitions_override=bootstrap_repetitions_override,
    )
    passed = bool(evaluation["all_visible_gates_pass"])
    status = (
        "VISIBLE_PASS_FORMULA_FAMILY_REPLICATION_AUTHORIZED_NOT_OPENED"
        if passed
        else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
    )
    report = {
        "schema_version": "1.0.0",
        "report_id": "US_MULTI_ASSET_CAPACITY_AWARE_TERMINAL_WINDDOWN_V4_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": status,
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "objective": contract["objective"],
        "manifest_verification": manifest_verification,
        "execution_reconstitution_changed_paths": changed_paths(contract),
        "data_audit": {
            "inputs": input_audit,
            "product_master_rows": int(len(master)),
            "signal": signal_audit,
            "terminal_winddown": winddown_audit,
            "portfolio": portfolio_audit,
            "daily_context": {
                "rows": int(len(context)),
                "maximum_fx_age_calendar_days": int(context["fx_age_calendar_days"].max()),
                "strictly_lagged_fx": bool((context["fx_source_date"] < context["date"]).all()),
            },
        },
        "evaluation": evaluation,
        "historical_evidence_limits": contract["historical_evidence_limits"],
        "decision": {
            "formula_family_replication_conditionally_eligible": passed,
            "sealed_replication_authorized": passed,
            "sealed_replication_open": False,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "paper_or_live_authorized": False,
            "next_step": (
                "可见期全部通过；另行验证清单后才可打开封存公式族复验，当前仍未打开"
                if passed
                else "冻结拒绝V4，不打开封存期，不修改减仓窗口、容量、产品或信号救回"
            ),
        },
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    outputs = contract["outputs"]
    atomic_parquet(ROOT / outputs["visible_daily_returns"], daily)
    atomic_parquet(ROOT / outputs["visible_selections"], execution_selections)
    atomic_parquet(ROOT / outputs["visible_trades"], trades)
    atomic_json(ROOT / outputs["input_audit_json"], report["data_audit"])
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
