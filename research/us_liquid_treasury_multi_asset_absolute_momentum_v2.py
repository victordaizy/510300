"""美国高流动性美债多资产绝对动量V2。"""

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
    fx_spread_rate,
    prepare_daily_context,
    run_portfolio_backtest,
    security_cost_rate,
    validate_contract as validate_v1_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "us_liquid_treasury_multi_asset_absolute_momentum_v2.yaml"

ALLOWED_CHANGED_PATHS = {
    "protocol.candidate_id",
    "protocol.parent_protocol_id",
    "protocol.lane",
    "protocol.mechanism",
    "universe.fixed_tickers",
    "universe.products.TMF.risk_source",
    "universe.products.TMF.stated_daily_multiple",
    "universe.products.TMF.issuer",
    "universe.products.TMF.inception_date",
    "universe.products.TLT.risk_source",
    "universe.products.TLT.stated_daily_multiple",
    "universe.products.TLT.issuer",
    "universe.products.TLT.inception_date",
    "inputs.source_status",
    "inputs.tlt_raw_receipt",
    "inputs.product_master",
    "inputs.daily_panel",
    "inputs.visible_panel",
    "inputs.sealed_replication_panel",
    "inputs.fx_full",
    "inputs.visible_fx",
    "inputs.sealed_replication_fx",
    "inputs.parent_manifest",
    "currency.fx_series",
    "currency.daily_fx_mapping",
    "currency.maximum_fx_staleness_calendar_days",
    "data_sources.fx",
    "data_sources.us_daily_raw",
    "data_sources.us_qfq_factor",
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
        raise ValueError("V2流动性美债重构配置必须是YAML对象")
    return payload


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    """合成完整V2合同，并证明差异只包含授权的产品与数据重构。"""

    override = load_override(path)
    base_path = ROOT / override["base_candidate_config"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("V1基础合同必须是YAML对象")
    contract = copy.deepcopy(base)
    _deep_update(contract, override["overrides"])
    contract["universe"]["products"] = copy.deepcopy(
        override["overrides"]["universe"]["products"]
    )
    validate_contract(contract, base=base, override=override)
    return contract


def validate_contract(
    contract: dict[str, Any],
    *,
    base: dict[str, Any] | None = None,
    override: dict[str, Any] | None = None,
) -> None:
    """禁止借产品重构改变公式、成本、日期、资金或目标。"""

    if override is None:
        override = load_override()
    if base is None:
        base = yaml.safe_load(
            (ROOT / override["base_candidate_config"]).read_text(encoding="utf-8")
        )
    if not isinstance(base, dict):
        raise ValueError("V1基础合同必须是YAML对象")
    validate_v1_contract(base)
    left = _flatten(base)
    right = _flatten(contract)
    changed = {
        path
        for path in set(left) | set(right)
        if left.get(path) != right.get(path)
    }
    if changed != ALLOWED_CHANGED_PATHS:
        raise ValueError(
            "V2差异越过V8授权范围："
            f"unexpected={sorted(changed - ALLOWED_CHANGED_PATHS)}, "
            f"missing={sorted(ALLOWED_CHANGED_PATHS - changed)}"
        )
    reconstruction = override.get("reconstitution", {})
    if reconstruction.get("reconstitution_of") != (
        "US_LEVERAGED_MULTI_ASSET_ABSOLUTE_MOMENTUM_V1R_FX_SOURCE_CORRECTION"
    ):
        raise ValueError("V2重构父候选发生变化")
    if reconstruction.get("failed_product") != "TMF" or reconstruction.get(
        "replacement_product"
    ) != "TLT":
        raise ValueError("V2产品替换发生变化")
    if reconstruction.get("no_prior_candidate_performance_available") is not True:
        raise ValueError("V2未声明旧候选无绩效可用")
    if reconstruction.get(
        "all_signal_dates_windows_ranking_weights_costs_and_gates_unchanged"
    ) is not True:
        raise ValueError("V2未声明公式、日期、成本与硬门保持不变")
    if contract["protocol"]["candidate_id"] != (
        "US_LIQUID_TREASURY_MULTI_ASSET_ABSOLUTE_MOMENTUM_V2"
    ):
        raise ValueError("V2候选编号不正确")
    if contract["protocol"]["parent_protocol_id"] != (
        "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V8"
    ):
        raise ValueError("V2父协议不正确")
    if contract["universe"]["fixed_tickers"] != ["UPRO", "TQQQ", "TLT", "UGL"]:
        raise ValueError("V2固定产品发生变化")
    products = contract["universe"]["products"]
    if set(products) != {"UPRO", "TQQQ", "TLT", "UGL"}:
        raise ValueError("V2产品身份表发生变化")
    tlt = products["TLT"]
    if tlt != {
        "risk_source": "ICE_US_TREASURY_20_PLUS_YEAR",
        "stated_daily_multiple": 1.0,
        "issuer": "BLACKROCK_ISHARES",
        "inception_date": "2002-07-22",
    }:
        raise ValueError("TLT身份定义发生变化")
    if contract["currency"]["maximum_fx_staleness_calendar_days"] != 14:
        raise ValueError("V2官方汇率最长陈旧期必须为14天")
    if contract["data_sources"]["fx"] != "CHINAMONEY_CFETS_CC_PR_HISTORICAL_USD_CNY":
        raise ValueError("V2未使用中国外汇交易中心官方中间价")
    if any(bool(value) for value in contract["safety"].values()):
        raise ValueError("V2研究边界被打开")


def changed_paths(contract: dict[str, Any] | None = None) -> list[str]:
    override = load_override()
    base = yaml.safe_load(
        (ROOT / override["base_candidate_config"]).read_text(encoding="utf-8")
    )
    reconstructed = load_contract() if contract is None else contract
    left = _flatten(base)
    right = _flatten(reconstructed)
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
    """只读重构后的可见证券输入、官方可见汇率与可见基准。"""

    inputs = contract["inputs"]
    status = json.loads((ROOT / inputs["source_status"]).read_text(encoding="utf-8"))
    if status.get("status") != "PASS_INPUT_ACQUISITION_ONLY":
        raise DataContractError(f"V2采集状态未通过：{status.get('status')}")
    if status.get("strategy_total_return_or_rank_computed") is not False:
        raise DataContractError("V2采集阶段越权计算了策略收益或排名")
    if status.get("portfolio_nav_or_target_gate_computed") is not False:
        raise DataContractError("V2采集阶段越权计算了组合净值或目标门")
    if status.get("security_price_inputs_reconstituted_before_outcome") is not True:
        raise DataContractError("V2未确认证券输入在结果前完成重构")

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
    tickers = set(contract["universe"]["fixed_tickers"])
    if panel.empty or panel["date"].max() > visible_end:
        raise DataContractError("V2可见证券面板为空或越界")
    if panel[["ticker", "date"]].duplicated().any():
        raise DataContractError("V2可见证券面板存在重复主键")
    if set(panel["ticker"].unique()) != tickers:
        raise DataContractError("V2可见证券面板未覆盖固定产品")
    if set(master["ticker"].astype(str)) != tickers:
        raise DataContractError("V2产品身份表未覆盖固定产品")
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
            raise DataContractError("V2可见官方汇率或基准为空")
        if fx["date"].max() > visible_end or benchmark["date"].max() > visible_end:
            raise DataContractError("V2可见官方汇率或基准越过可见期末")
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
            "# 美国高流动性美债多资产绝对动量V2可见期结果",
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
    """运行唯一一次V2可见期，不读取封存区。"""

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
        "report_id": "US_LIQUID_TREASURY_MULTI_ASSET_ABSOLUTE_MOMENTUM_V2_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": status,
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "objective": contract["objective"],
        "manifest_verification": manifest_verification,
        "reconstitution_changed_paths": changed_paths(contract),
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
                else "冻结拒绝V2，不打开封存期，不再替换产品或调整策略参数救回"
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
