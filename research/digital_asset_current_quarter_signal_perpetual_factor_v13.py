"""以当季期货基差信号交易永续合约的两资产高低组合V13。"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.digital_asset_cross_sectional_hourly_reversal_v8 import (
    _exact_symbol_grid,
    _normalize_time,
    _sha256,
)
from research.digital_asset_cross_sectional_perpetual_basis_v9 import (
    run_portfolio_backtest as _run_perpetual_portfolio_backtest,
)
from research.digital_asset_current_quarter_basis_factor_v11 import (
    build_daily_targets as _build_current_quarter_daily_targets,
)
from research.digital_asset_spot_volatility_scaled_trend_v1 import (
    CONSERVATIVE_EVALUATOR_CONFIG,
    atomic_json,
    atomic_parquet,
    atomic_text,
)
from research.qdii_cross_market_discount_reversion_zero_variance_v2 import (
    evaluate_historical_returns_zero_variance_v2,
    load_correction_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "config" / "digital_asset_current_quarter_signal_perpetual_factor_v13.yaml"
)
FIXED_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("当季基差信号永续因子V13配置必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    failures: list[str] = []
    protocol = contract.get("protocol", {})
    if (
        protocol.get("candidate_id")
        != "DIGITAL_ASSET_CURRENT_QUARTER_SIGNAL_PERPETUAL_FACTOR_V13"
    ):
        failures.append("candidate_id")
    if (
        protocol.get("parent_protocol_id")
        != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V23"
    ):
        failures.append("parent_protocol_id")
    account = contract.get("account", {})
    if float(account.get("initial_capital_cny", math.nan)) != 500000.0:
        failures.append("initial_capital_cny")
    if float(account.get("user_transaction_fee_rate_per_leg", math.nan)) != 0.0001:
        failures.append("user_transaction_fee_rate_per_leg")
    history = contract.get("historical_partition", {})
    if (
        history.get("acquisition_start") != "2021-08-25 00:00:00"
        or history.get("visible_start") != "2021-09-01"
        or history.get("visible_end") != "2026-08-14"
    ):
        failures.append("historical_partition")
    universe = contract.get("universe", {})
    if universe.get("fixed_symbols") != FIXED_SYMBOLS:
        failures.append("fixed_symbols")
    if int(universe.get("turnover_lookback_hours", 0)) != 168:
        failures.append("turnover_lookback_hours")
    if int(universe.get("turnover_lookback_observed_venue_hours", 0)) != 168:
        failures.append("turnover_lookback_observed_venue_hours")
    if (
        float(
            universe.get(
                "maximum_order_fraction_of_prior_median_quote_volume", math.nan
            )
        )
        != 0.001
    ):
        failures.append("maximum_order_fraction")
    signal = contract.get("signal", {})
    fixed_signal: dict[str, Any] = {
        "daily_basis_formula": "LOG_SPOT_23UTC_CLOSE_DIV_CURRENT_QUARTER_23UTC_CLOSE",
        "basis_lookback_calendar_days": 5,
        "basis_aggregation": "ARITHMETIC_MEAN",
        "ranking": "DESCENDING_MEAN_BASIS_THEN_SYMBOL_ASCENDING",
        "long_count": 1,
        "short_count": 1,
        "long_side_target_gross": 0.45,
        "short_side_target_gross": 0.45,
        "total_target_gross": 0.90,
        "target_net": 0.0,
        "execution_instrument": "SAME_SYMBOL_USDS_MARGINED_PERPETUAL",
        "spot_position_allowed": False,
        "current_quarter_position_allowed": False,
        "parameter_grid_search_allowed": False,
    }
    for key, expected in fixed_signal.items():
        value = signal.get(key)
        if isinstance(expected, bool):
            if value is not expected:
                failures.append(key)
        elif isinstance(expected, int):
            if int(value or 0) != expected:
                failures.append(key)
        elif isinstance(expected, float):
            if float(value if value is not None else math.nan) != expected:
                failures.append(key)
        elif value != expected:
            failures.append(key)
    delivery = contract.get("delivery", {})
    if (
        delivery.get("delivery_calendar")
        != "LAST_FRIDAY_OF_MARCH_JUNE_SEPTEMBER_DECEMBER"
        or delivery.get("delivery_day_position_policy")
        != "ZERO_FROM_00UTC_THROUGH_23UTC"
    ):
        failures.append("delivery")
    portfolio = contract.get("portfolio", {})
    if portfolio.get("quantity_increment") != {
        "BTCUSDT": 0.001,
        "ETHUSDT": 0.001,
    }:
        failures.append("quantity_increment")
    for key, expected in {
        "target_long_gross": 0.45,
        "target_short_gross": 0.45,
        "target_total_gross": 0.90,
        "stress_cost_reserve_multiplier": 0.98,
        "intraday_risk_deleveraging_gross_trigger": 1.029,
        "intraday_risk_deleveraging_absolute_net_trigger": 0.098,
    }.items():
        if float(portfolio.get(key, math.nan)) != expected:
            failures.append(key)
    if portfolio.get("terminal_winddown_start") != "2026-08-08 00:00:00":
        failures.append("terminal_winddown_start")
    if int(portfolio.get("terminal_winddown_calendar_days", 0)) != 7:
        failures.append("terminal_winddown_calendar_days")
    if portfolio.get("new_entries_after_winddown_start_allowed") is not False:
        failures.append("new_entries_after_winddown_start_allowed")
    costs = contract.get("costs", {})
    fixed_costs = {
        "user_transaction_fee_rate_per_leg": 0.0001,
        "base_total_perpetual_cost_bps_per_leg": 4.0,
        "stress_total_perpetual_cost_bps_per_leg": 16.0,
        "base_funding_receipt_multiplier": 1.0,
        "base_funding_payment_multiplier": 1.0,
        "stress_funding_receipt_multiplier": 0.75,
        "stress_funding_payment_multiplier": 1.25,
        "base_fx_conversion_spread_bps_per_conversion": 10.0,
        "stress_fx_conversion_spread_bps_per_conversion": 30.0,
    }
    for key, expected in fixed_costs.items():
        if float(costs.get(key, math.nan)) != expected:
            failures.append(key)
    base_components = (
        float(costs.get("user_transaction_fee_rate_per_leg", math.nan)) * 10000.0
        + float(costs.get("base_slippage_bps_per_leg", math.nan))
        + float(costs.get("base_market_impact_bps_per_leg", math.nan))
    )
    stress_components = (
        float(costs.get("user_transaction_fee_rate_per_leg", math.nan)) * 10000.0
        + float(costs.get("stress_slippage_bps_per_leg", math.nan))
        + float(costs.get("stress_market_impact_bps_per_leg", math.nan))
    )
    if not math.isclose(base_components, 4.0):
        failures.append("base_cost_components")
    if not math.isclose(stress_components, 16.0):
        failures.append("stress_cost_components")
    risk = contract.get("risk", {})
    if float(risk.get("maximum_gross_exposure", math.nan)) != 1.05:
        failures.append("maximum_gross_exposure")
    if float(risk.get("maximum_absolute_net_exposure", math.nan)) != 0.10:
        failures.append("maximum_absolute_net_exposure")
    if float(risk.get("intraday_risk_deleveraging_buffer_multiplier", math.nan)) != 0.98:
        failures.append("risk_buffer")
    if float(risk.get("minimum_combined_equity_fraction_of_gross_notional", math.nan)) != 0.05:
        failures.append("maintenance_fraction")
    if not math.isclose(
        float(portfolio.get("intraday_risk_deleveraging_gross_trigger", math.nan)),
        float(risk.get("maximum_gross_exposure", math.nan))
        * float(risk.get("intraday_risk_deleveraging_buffer_multiplier", math.nan)),
    ):
        failures.append("gross_trigger_buffer")
    if not math.isclose(
        float(
            portfolio.get(
                "intraday_risk_deleveraging_absolute_net_trigger", math.nan
            )
        ),
        float(risk.get("maximum_absolute_net_exposure", math.nan))
        * float(risk.get("intraday_risk_deleveraging_buffer_multiplier", math.nan)),
    ):
        failures.append("net_trigger_buffer")
    gates = contract.get("visible_gates", {})
    if float(gates.get("minimum_annualized_net_excess", math.nan)) != 0.40:
        failures.append("minimum_annualized_net_excess")
    if float(gates.get("minimum_strategy_net_sharpe", math.nan)) != 1.50:
        failures.append("minimum_strategy_net_sharpe")
    if int(gates.get("annualization_trading_days", 0)) != 365:
        failures.append("annualization_trading_days")
    if gates.get("base_and_stress_must_both_pass") is not True:
        failures.append("base_and_stress_must_both_pass")
    if any(bool(value) for value in contract.get("safety", {}).values()):
        failures.append("safety")
    if failures:
        raise ValueError(
            f"当季基差信号永续因子V13配置被弱化或损坏：{sorted(set(failures))}"
        )


def load_visible_inputs(
    contract: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    inputs = contract["inputs"]
    status_path = ROOT / inputs["source_status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    forbidden = (
        "basis_value_or_rank_computed",
        "long_or_short_direction_computed",
        "signal_or_target_computed",
        "strategy_or_benchmark_return_computed",
        "account_or_authenticated_endpoint_used",
    )
    if (
        status.get("status") != "PASS_INPUT_ACQUISITION_ONLY"
        or status.get("candidate_id") != contract["protocol"]["candidate_id"]
        or any(bool(status.get(key)) for key in forbidden)
    ):
        raise DataContractError("V13官方输入采集状态不合格")
    paths = {
        "spot_1h": ROOT / inputs["spot_1h"],
        "current_quarter_1h": ROOT / inputs["current_quarter_1h"],
        "perpetual_1h": ROOT / inputs["perpetual_1h"],
        "funding_rates": ROOT / inputs["funding_rates"],
        "fx": ROOT / inputs["visible_fx"],
        "benchmark": ROOT / inputs["visible_benchmark"],
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少V13输入：{key}={path}")
        receipt = status.get("outputs", {}).get(key, {})
        if (
            receipt.get("path") != path.relative_to(ROOT).as_posix()
            or receipt.get("sha256") != _sha256(path)
        ):
            raise DataContractError(f"V13输入与采集回执不一致：{key}")
    spot = _normalize_time(pd.read_parquet(paths["spot_1h"]), "open_time")
    current_quarter = _normalize_time(
        pd.read_parquet(paths["current_quarter_1h"]), "open_time"
    )
    perpetual = _normalize_time(
        pd.read_parquet(paths["perpetual_1h"]), "open_time"
    )
    funding = _normalize_time(
        pd.read_parquet(paths["funding_rates"]), "funding_time"
    )
    fx = pd.read_parquet(paths["fx"])
    benchmark = pd.read_parquet(paths["benchmark"])
    fx["date"] = (
        pd.to_datetime(fx["date"], errors="coerce")
        .astype("datetime64[ns]")
        .dt.normalize()
    )
    benchmark["date"] = (
        pd.to_datetime(benchmark["date"], errors="coerce")
        .astype("datetime64[ns]")
        .dt.normalize()
    )
    symbols = list(contract["universe"]["fixed_symbols"])
    gate = contract["prefreeze_coverage_gate"]
    expected_hourly = pd.date_range(
        pd.Timestamp(gate["required_grid_start"]),
        pd.Timestamp(gate["required_grid_end"]),
        freq="1h",
    )
    expected_funding = pd.date_range(
        pd.Timestamp(gate["required_grid_start"]),
        pd.Timestamp(gate["required_funding_end"]),
        freq="8h",
    )
    _exact_symbol_grid(
        current_quarter,
        time_column="open_time",
        symbols=symbols,
        expected=expected_hourly,
        label="V13当季期货小时K线",
    )
    _exact_symbol_grid(
        perpetual,
        time_column="open_time",
        symbols=symbols,
        expected=expected_hourly,
        label="V13永续小时K线",
    )
    _exact_symbol_grid(
        funding,
        time_column="funding_time",
        symbols=symbols,
        expected=expected_funding,
        label="V13资金费率",
    )
    if sorted(spot["symbol"].astype(str).unique()) != sorted(symbols):
        raise DataContractError("V13现货产品集合不正确")
    spot_grids: list[pd.DatetimeIndex] = []
    required_spot_rows = int(gate["required_spot_hourly_rows_per_symbol"])
    for symbol in symbols:
        subset = spot.loc[spot["symbol"].eq(symbol), "open_time"]
        if subset.duplicated().any() or len(subset) != required_spot_rows:
            raise DataContractError(f"V13现货{symbol}覆盖或重复不正确")
        spot_grids.append(pd.DatetimeIndex(subset).sort_values().unique())
    if not spot_grids[0].equals(spot_grids[1]):
        raise DataContractError("V13现货两产品小时网格不同")
    spot_missing = expected_hourly.difference(spot_grids[0])
    if (
        len(spot_missing) != int(gate["required_common_spot_missing_hours"])
        or any(timestamp.hour in (0, 23) for timestamp in spot_missing)
    ):
        raise DataContractError("V13现货缺口数量变化或触及决策时点")
    for label, frame in (
        ("现货", spot),
        ("当季期货", current_quarter),
        ("永续", perpetual),
    ):
        numeric_columns = ["open", "high", "low", "close", "quote_volume"]
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all():
            raise DataContractError(f"V13{label}价格或成交额含缺失或非有限值")
        if not (numeric[["open", "high", "low", "close"]] > 0.0).all().all():
            raise DataContractError(f"V13{label}价格非正")
        if not (numeric["quote_volume"] >= 0.0).all():
            raise DataContractError(f"V13{label}成交额为负")
    rates = pd.to_numeric(funding["funding_rate"], errors="coerce")
    marks = pd.to_numeric(funding["mark_price"], errors="coerce")
    offsets = pd.to_numeric(
        funding["funding_time_offset_milliseconds"], errors="coerce"
    )
    if (
        rates.isna().any()
        or not np.isfinite(rates.to_numpy(dtype=float)).all()
        or marks.isna().any()
        or not (marks > 0.0).all()
        or offsets.isna().any()
        or float(offsets.abs().max()) > 1000.0
    ):
        raise DataContractError("V13实际资金费率、标记价格或结算时点不完整")
    for label, frame, column in (
        ("汇率", fx, "cny_per_usd"),
        ("基准", benchmark, "close"),
    ):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if (
            frame["date"].isna().any()
            or numeric.isna().any()
            or not (numeric > 0.0).all()
        ):
            raise DataContractError(f"V13{label}上下文不完整或非正")
    audit = {
        "source_status": status["status"],
        "spot_rows": int(len(spot)),
        "current_quarter_rows": int(len(current_quarter)),
        "perpetual_rows": int(len(perpetual)),
        "funding_rows": int(len(funding)),
        "fx_rows": int(len(fx)),
        "benchmark_rows": int(len(benchmark)),
        "symbol_count": len(symbols),
        "spot_common_missing_hour_count": int(len(spot_missing)),
        "missing_signal_or_execution_hour_count": 0,
        "perpetual_hourly_grid_complete": True,
        "funding_grid_complete": True,
        "maximum_funding_time_offset_milliseconds": float(offsets.abs().max()),
        "missing_values_backfilled": False,
        "input_sha256": {key: _sha256(path) for key, path in paths.items()},
        "basis_rank_or_strategy_return_computed_during_collection": False,
        "authenticated_endpoint_used": False,
    }
    return spot, current_quarter, perpetual, funding, fx, benchmark, audit


def build_daily_targets(
    spot: pd.DataFrame,
    current_quarter: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    targets, features, audit = _build_current_quarter_daily_targets(
        spot, current_quarter, contract
    )
    audit = {
        **audit,
        "signal_instrument": "CURRENT_QUARTER_CONTINUOUS",
        "execution_instrument": "USDS_MARGINED_PERPETUAL",
        "current_quarter_position_generated": False,
    }
    return targets, features, audit


def run_portfolio_backtest(
    targets: pd.DataFrame,
    perpetual: pd.DataFrame,
    funding: pd.DataFrame,
    fx: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    daily, trades, portfolio_audit, context_audit = (
        _run_perpetual_portfolio_backtest(
            targets, perpetual, funding, fx, benchmark, contract
        )
    )
    portfolio_audit = {
        **portfolio_audit,
        "signal_market": "CURRENT_QUARTER_CONTINUOUS",
        "trading_market": "USDS_MARGINED_PERPETUAL",
        "actual_funding_history_applied": True,
        "current_quarter_position_count": 0,
    }
    return daily, trades, portfolio_audit, context_audit


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["evaluation"]["metrics"]
    gates = report["evaluation"]["gates"]
    return "\n".join(
        [
            "# 数字资产当季基差信号、永续执行因子V13可见期结果",
            "",
            f"状态：`{report['status']}`",
            "",
            f"- H00300全收益CAGR：{metrics['benchmark_total_return_cagr']:.2%}。",
            f"- 基础/压力策略净CAGR：{metrics['strategy_base_net_cagr']:.2%}/{metrics['strategy_stress_net_cagr']:.2%}。",
            f"- 基础/压力年化净超额：{metrics['base_annualized_excess']:.2%}/{metrics['stress_annualized_excess']:.2%}。",
            f"- 基础/压力净夏普：{metrics['base_strategy_net_sharpe']}/{metrics['stress_strategy_net_sharpe']}。",
            f"- 基础/压力40个百分点门：{gates['base_annualized_excess_at_least_40pct']}/{gates['stress_annualized_excess_at_least_40pct']}。",
            f"- 基础/压力1.50夏普门：{gates['base_strategy_sharpe_at_least_1_5']}/{gates['stress_strategy_sharpe_at_least_1_5']}。",
            f"- 全部门通过：{report['evaluation']['all_visible_gates_pass']}。",
            "",
            f"{report['decision']['next_step']}。",
            "",
            "Paper、Shadow、订单、交易所账户连接和实盘均未打开。",
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
    (
        spot,
        current_quarter,
        perpetual,
        funding,
        fx,
        benchmark,
        input_audit,
    ) = load_visible_inputs(contract)
    targets, features, signal_audit = build_daily_targets(
        spot, current_quarter, contract
    )
    daily, trades, portfolio_audit, context_audit = run_portfolio_backtest(
        targets, perpetual, funding, fx, benchmark, contract
    )
    correction = load_correction_contract(CONSERVATIVE_EVALUATOR_CONFIG)
    evaluation = evaluate_historical_returns_zero_variance_v2(
        daily,
        contract,
        correction,
        bootstrap_repetitions_override=bootstrap_repetitions_override,
    )
    passed = bool(evaluation["all_visible_gates_pass"])
    report = {
        "schema_version": "1.0.0",
        "report_id": "DIGITAL_ASSET_CURRENT_QUARTER_SIGNAL_PERPETUAL_FACTOR_V13_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": (
            "VISIBLE_PASS_40PCT_HIGH_SHARPE_RESEARCH_SCREEN_ONLY"
            if passed
            else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
        ),
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {
            "start": contract["historical_partition"]["visible_start"],
            "end": contract["historical_partition"]["visible_end"],
        },
        "objective": {
            "initial_capital_cny": contract["account"]["initial_capital_cny"],
            "minimum_annualized_net_excess": contract["visible_gates"][
                "minimum_annualized_net_excess"
            ],
            "minimum_strategy_net_sharpe": contract["visible_gates"][
                "minimum_strategy_net_sharpe"
            ],
            "user_transaction_fee_rate_per_leg": contract["account"][
                "user_transaction_fee_rate_per_leg"
            ],
        },
        "manifest_verification": manifest_verification,
        "data_audit": {
            "inputs": input_audit,
            "signal": signal_audit,
            "portfolio": portfolio_audit,
            "daily_context": context_audit,
            "selection_feature_rows": int(len(features)),
        },
        "evaluation": evaluation,
        "historical_evidence_limits": contract["historical_evidence_limits"],
        "decision": {
            "external_forward_validation_required": passed,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "paper_or_live_authorized": False,
            "next_step": (
                "可见期全部通过；锁定V13并另设真正前瞻复验，当前不生成任何交易信号"
                if passed
                else "冻结拒绝V13，不修改5日当季基差、两资产、多空方向、0.45/0.45、资金费、成本、交割日、容量或风险门救回"
            ),
        },
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    outputs = contract["outputs"]
    atomic_parquet(ROOT / outputs["visible_daily_returns"], daily)
    atomic_parquet(ROOT / outputs["visible_targets"], targets)
    atomic_parquet(ROOT / outputs["visible_trades"], trades)
    atomic_json(ROOT / outputs["input_audit_json"], report["data_audit"])
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
