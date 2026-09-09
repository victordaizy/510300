"""美国杠杆多资产V1R：只修正官方汇率来源与法定长假陈旧上限。"""

from __future__ import annotations

import copy
import json
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
    run_portfolio_backtest,
    validate_contract as validate_v1_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "us_leveraged_multi_asset_absolute_momentum_v1r_fx_source_correction.yaml"

ALLOWED_CHANGED_PATHS = {
    "protocol.candidate_id",
    "protocol.parent_protocol_id",
    "protocol.status",
    "inputs.source_status",
    "inputs.fx_full",
    "inputs.visible_fx",
    "inputs.sealed_replication_fx",
    "inputs.parent_manifest",
    "currency.fx_series",
    "currency.daily_fx_mapping",
    "currency.maximum_fx_staleness_calendar_days",
    "data_sources.fx",
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
        raise ValueError("V1R汇率修正配置必须是YAML对象")
    return payload


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    """合成完整V1R合同，并证明差异严格限于V6授权字段。"""

    override = load_override(path)
    base_path = ROOT / override["base_candidate_config"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("V1基础合同必须是YAML对象")
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
    """验证V1R没有夹带任何信号、日期、成本、组合或目标变化。"""

    if override is None:
        override = load_override()
    if base is None:
        base = yaml.safe_load((ROOT / override["base_candidate_config"]).read_text(encoding="utf-8"))
    flat_base = _flatten(base)
    flat_contract = _flatten(contract)
    changed = {
        path
        for path in set(flat_base) | set(flat_contract)
        if flat_base.get(path) != flat_contract.get(path)
    }
    if changed != ALLOWED_CHANGED_PATHS:
        raise ValueError(
            "V1R差异越过V6授权范围："
            f"unexpected={sorted(changed - ALLOWED_CHANGED_PATHS)}, "
            f"missing={sorted(ALLOWED_CHANGED_PATHS - changed)}"
        )
    normalized = copy.deepcopy(contract)
    for path in ALLOWED_CHANGED_PATHS:
        keys = path.split(".")
        target = normalized
        source = base
        for key in keys[:-1]:
            target = target[key]
            source = source[key]
        target[keys[-1]] = copy.deepcopy(source[keys[-1]])
    validate_v1_contract(normalized)

    correction = override["correction"]
    if correction["correction_of"] != "US_LEVERAGED_MULTI_ASSET_ABSOLUTE_MOMENTUM_V1":
        raise ValueError("V1R修正父候选发生变化")
    if correction["all_strategy_dates_costs_portfolio_and_gates_unchanged"] is not True:
        raise ValueError("V1R未声明全部策略与评价字段保持不变")
    if contract["protocol"]["candidate_id"] != override["protocol"]["candidate_id"]:
        raise ValueError("V1R候选编号不一致")
    if contract["protocol"]["parent_protocol_id"] != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V6":
        raise ValueError("V1R父协议不正确")
    if contract["currency"]["maximum_fx_staleness_calendar_days"] != 14:
        raise ValueError("V1R最长汇率陈旧期必须为14天")
    if contract["data_sources"]["fx"] != "CHINAMONEY_CFETS_CC_PR_HISTORICAL_USD_CNY":
        raise ValueError("V1R未使用中国外汇交易中心官方中间价")


def changed_paths(contract: dict[str, Any] | None = None) -> list[str]:
    """返回V1R相对V1的实际差异路径。"""

    override = load_override()
    base = yaml.safe_load((ROOT / override["base_candidate_config"]).read_text(encoding="utf-8"))
    corrected = load_contract() if contract is None else contract
    left = _flatten(base)
    right = _flatten(corrected)
    return sorted(
        path
        for path in set(left) | set(right)
        if left.get(path) != right.get(path)
    )


def _normalize_date(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = (
        pd.to_datetime(result["date"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    return result


def load_visible_inputs(
    contract: dict[str, Any],
    *,
    signal_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """只读V1继承的可见证券输入与V1R官方可见汇率。"""

    inputs = contract["inputs"]
    status = json.loads((ROOT / inputs["source_status"]).read_text(encoding="utf-8"))
    if status.get("status") != "PASS_FX_SOURCE_CORRECTION_INPUT_ACQUISITION_ONLY":
        raise DataContractError(f"V1R汇率采集状态未通过：{status.get('status')}")
    if status.get("strategy_total_return_or_rank_computed") is not False:
        raise DataContractError("V1R采集阶段越权计算了策略收益或排名")
    if status.get("security_price_inputs_reused_without_change") is not True:
        raise DataContractError("V1R未确认证券输入原样复用")

    master = pd.read_parquet(ROOT / inputs["product_master"])
    signal_columns = [
        "ticker",
        "date",
        "raw_close",
        "raw_volume",
        "raw_dollar_turnover_usd",
        "signal_total_return_index",
    ]
    full_columns = [
        "ticker",
        "date",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_volume",
        "raw_dollar_turnover_usd",
        "qfq_factor",
        "adjust",
        "split_ratio_at_open",
        "cash_distribution_per_post_event_share_usd",
        "signal_total_return_index",
    ]
    panel_columns = signal_columns if signal_only else full_columns
    panel = _normalize_date(
        pd.read_parquet(ROOT / inputs["visible_panel"], columns=panel_columns)
    )
    visible_end = pd.Timestamp(contract["historical_partition"]["visible_end"])
    if panel.empty or panel["date"].max() > visible_end:
        raise DataContractError("V1R继承的可见证券面板为空或越界")
    if signal_only:
        fx = pd.DataFrame(columns=["date", "cny_per_usd"])
        benchmark = pd.DataFrame(columns=["date", "close"])
        fx_columns: list[str] = []
        benchmark_columns: list[str] = []
    else:
        fx = _normalize_date(pd.read_parquet(ROOT / inputs["visible_fx"]))
        benchmark = _normalize_date(pd.read_parquet(ROOT / inputs["visible_benchmark"]))
        fx_columns = list(fx.columns)
        benchmark_columns = list(benchmark.columns)
        if fx.empty or benchmark.empty:
            raise DataContractError("V1R可见官方汇率或基准为空")
        if fx["date"].max() > visible_end or benchmark["date"].max() > visible_end:
            raise DataContractError("V1R可见官方汇率或基准越过可见期末")
    audit = {
        "source_status": status["status"],
        "panel_columns_read": panel_columns,
        "fx_columns_read": fx_columns,
        "benchmark_columns_read": benchmark_columns,
        "visible_panel_path": inputs["visible_panel"],
        "visible_fx_path": inputs["visible_fx"],
        "sealed_panel_or_benchmark_read": False,
        "future_open_or_strategy_return_read": False if signal_only else True,
        "signal_only": signal_only,
    }
    return panel, master, fx, benchmark, audit


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["evaluation"]["metrics"]
    gates = report["evaluation"]["gates"]
    portfolio = report["data_audit"]["portfolio"]
    return "\n".join(
        [
            "# 美国杠杆多资产绝对动量V1R汇率修正可见期结果",
            "",
            f"状态：`{report['status']}`",
            "",
            "## 核心结果",
            "",
            f"- H00300全收益CAGR：{metrics['benchmark_total_return_cagr']:.2%}。",
            f"- 基础/压力策略净CAGR：{metrics['strategy_base_net_cagr']:.2%}/{metrics['strategy_stress_net_cagr']:.2%}。",
            f"- 基础/压力年化净超额：{metrics['base_annualized_excess']:.2%}/{metrics['stress_annualized_excess']:.2%}。",
            f"- 基础/压力净夏普：{metrics['base_strategy_net_sharpe']:.3f}/{metrics['stress_strategy_net_sharpe']:.3f}。",
            f"- 基础/压力最大回撤：{metrics['base_maximum_drawdown']:.2%}/{metrics['stress_maximum_drawdown']:.2%}。",
            f"- 买入/卖出交易数：{portfolio['entry_transaction_count']}/{portfolio['exit_transaction_count']}。",
            "",
            "## 硬门",
            "",
            f"- 基础/压力净超额至少40个百分点：{gates['base_annualized_excess_at_least_40pct']}/{gates['stress_annualized_excess_at_least_40pct']}。",
            f"- 基础/压力净夏普至少1.50：{gates['base_strategy_sharpe_at_least_1_5']}/{gates['stress_strategy_sharpe_at_least_1_5']}。",
            f"- 全部门通过：{report['evaluation']['all_visible_gates_pass']}。",
            "",
            "## 决策",
            "",
            f"{report['decision']['next_step']}。",
            "",
            "官方中间价不是可成交即期价；封存复验、Paper、Shadow、订单和实盘均未打开。",
            "",
        ]
    )


def run_visible(
    contract: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    """运行唯一一次V1R可见期；不读取封存区。"""

    validate_contract(contract)
    partition = contract["historical_partition"]
    start = pd.Timestamp(partition["visible_start"])
    end = pd.Timestamp(partition["visible_end"])
    panel, master, fx, benchmark, input_audit = load_visible_inputs(
        contract, signal_only=False
    )
    selections, schedule, features, signal_audit = build_monthly_selections(
        panel, contract, start=start, end=end
    )
    context = prepare_daily_context(
        panel, fx, benchmark, contract, start=start, end=end
    )
    daily, trades, portfolio_audit = run_portfolio_backtest(
        selections,
        schedule,
        features,
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
        "report_id": "US_LEVERAGED_MULTI_ASSET_ABSOLUTE_MOMENTUM_V1R_FX_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": status,
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "objective": contract["objective"],
        "manifest_verification": manifest_verification,
        "correction_changed_paths": changed_paths(contract),
        "data_audit": {
            "inputs": input_audit,
            "product_master_rows": int(len(master)),
            "signal": signal_audit,
            "portfolio": portfolio_audit,
            "daily_context": {
                "rows": int(len(context)),
                "first_date": context["date"].min().date().isoformat(),
                "last_date": context["date"].max().date().isoformat(),
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
                else "冻结拒绝V1R，不打开封存期，不再修改汇率或策略参数救回"
            ),
        },
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    outputs = contract["outputs"]
    atomic_parquet(ROOT / outputs["visible_daily_returns"], daily)
    atomic_parquet(ROOT / outputs["visible_selections"], selections)
    atomic_parquet(ROOT / outputs["visible_trades"], trades)
    atomic_json(ROOT / outputs["input_audit_json"], report["data_audit"])
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
