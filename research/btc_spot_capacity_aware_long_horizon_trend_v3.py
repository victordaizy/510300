"""BTC现货容量感知长周期趋势V3。"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.btc_spot_long_horizon_trend_v2 import (
    build_weekly_targets,
    load_contract as load_v2_contract,
    load_visible_inputs,
)
from research.digital_asset_spot_volatility_scaled_trend_v1 import (
    CONSERVATIVE_EVALUATOR_CONFIG,
    _floor_quantity,
    _market_row,
    _normalize_date,
    _valid_execution,
    atomic_json,
    atomic_parquet,
    atomic_text,
    fx_spread_rate,
    prepare_daily_context,
    spot_cost_rate,
)
from research.qdii_cross_market_discount_reversion_zero_variance_v2 import (
    evaluate_historical_returns_zero_variance_v2,
    load_correction_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "btc_spot_capacity_aware_long_horizon_trend_v3.yaml"

ALLOWED_CHANGED_PATHS = {
    "protocol.candidate_id",
    "protocol.parent_protocol_id",
    "protocol.lane",
    "protocol.mechanism",
    "inputs.parent_manifest",
    "portfolio.target_notional_basis",
    "portfolio.blocked_entry_policy",
    "portfolio.blocked_exit_policy",
    "portfolio.terminal_open_positions_policy",
    "portfolio.capacity_aware_targeting_enabled",
    "portfolio.capacity_target_formula",
    "portfolio.capacity_slice_rounding",
    "portfolio.pending_resize_policy",
    "portfolio.terminal_winddown_calendar_days",
    "portfolio.terminal_liquidation_time",
    "portfolio.stop_new_entries_at_terminal_winddown_start",
    "portfolio.terminal_winddown_schedule_mode",
    "historical_evidence_limits.performance_screen_label",
    "outputs.input_audit_json",
    "outputs.manifest",
    "outputs.visible_daily_returns",
    "outputs.visible_targets",
    "outputs.visible_trades",
    "outputs.visible_report_json",
    "outputs.visible_report_markdown",
    "outputs.replication_report_json",
}


@dataclass
class CapacitySpotPosition:
    product_id: str
    quantity: float
    entry_date: pd.Timestamp
    last_close: float
    latest_prior_median_turnover_usd: float
    pending_exit: bool = False
    exit_signal_date: pd.Timestamp | None = None
    delayed_exit_days: int = 0
    pending_resize_target_quantity: float | None = None


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
        raise ValueError("BTC容量感知配置必须是YAML对象")
    return payload


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    override = load_override(path)
    if override.get("base_candidate_config") != "config/btc_spot_long_horizon_trend_v2.yaml":
        raise ValueError("BTC V3基础候选路径发生变化")
    base = load_v2_contract()
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
    if base is None:
        base = load_v2_contract()
    if override is None:
        override = load_override()
    left = _flatten(base)
    right = _flatten(contract)
    changed = {path for path in set(left) | set(right) if left.get(path) != right.get(path)}
    if changed != ALLOWED_CHANGED_PATHS:
        raise ValueError(
            "BTC V3差异越过V12授权范围："
            f"unexpected={sorted(changed - ALLOWED_CHANGED_PATHS)}, "
            f"missing={sorted(ALLOWED_CHANGED_PATHS - changed)}"
        )
    reconstruction = override.get("execution_reconstitution", {})
    if reconstruction.get("reconstitution_of") != "BTC_SPOT_LONG_HORIZON_TREND_V2":
        raise ValueError("BTC V3执行重构父候选发生变化")
    if reconstruction.get("no_v2_performance_available") is not True:
        raise ValueError("BTC V3未声明V2无绩效可用")
    if reconstruction.get(
        "product_signal_windows_target_volatility_costs_and_gates_unchanged"
    ) is not True:
        raise ValueError("BTC V3未声明产品、信号、成本和硬门不变")
    if reconstruction.get("maximum_capacity_fraction_unchanged") is not True:
        raise ValueError("BTC V3未声明容量比例不变")
    if contract["protocol"]["candidate_id"] != "BTC_SPOT_CAPACITY_AWARE_LONG_HORIZON_TREND_V3":
        raise ValueError("BTC V3候选编号不正确")
    if contract["protocol"]["parent_protocol_id"] != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V12":
        raise ValueError("BTC V3父协议不正确")
    if int(contract["portfolio"]["terminal_winddown_calendar_days"]) != 30:
        raise ValueError("BTC V3末端减仓窗口必须为30日")
    if float(contract["universe"]["maximum_order_fraction_of_prior_median_dollar_turnover"]) != 0.003:
        raise ValueError("BTC V3容量比例必须保持0.3%")
    if any(bool(value) for value in contract["safety"].values()):
        raise ValueError("BTC V3研究边界被打开")


def changed_paths(contract: dict[str, Any] | None = None) -> list[str]:
    base = load_v2_contract()
    reconstructed = load_contract() if contract is None else contract
    left = _flatten(base)
    right = _flatten(reconstructed)
    return sorted(path for path in set(left) | set(right) if left.get(path) != right.get(path))


def build_terminal_winddown_plan(
    targets: pd.DataFrame,
    schedule: pd.DataFrame,
    features: pd.DataFrame,
    context: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    days = int(contract["portfolio"]["terminal_winddown_calendar_days"])
    dates = pd.DatetimeIndex(pd.to_datetime(context["date"])).sort_values().unique()
    if len(dates) < days:
        raise DataContractError("BTC V3日历不足冻结的末端减仓窗口")
    winddown_date = pd.Timestamp(dates[-days])
    execution_targets = targets.loc[
        pd.to_datetime(targets["execution_date"]).lt(winddown_date)
    ].copy()
    execution_schedule = schedule.loc[
        pd.to_datetime(schedule["execution_date"]).lt(winddown_date)
    ].copy()
    execution_schedule = pd.concat(
        [
            execution_schedule,
            pd.DataFrame(
                {"signal_date": [winddown_date], "execution_date": [winddown_date]}
            ),
        ],
        ignore_index=True,
    ).sort_values("execution_date").reset_index(drop=True)
    execution_features = features.loc[
        pd.to_datetime(features["execution_date"]).lt(winddown_date)
    ].copy()
    audit = {
        "terminal_winddown_calendar_days": days,
        "winddown_start_date": winddown_date.date().isoformat(),
        "visible_end_date": pd.Timestamp(dates[-1]).date().isoformat(),
        "original_target_rows": int(len(targets)),
        "executed_target_rows": int(len(execution_targets)),
        "original_schedule_rows": int(len(schedule)),
        "execution_schedule_rows_including_winddown": int(len(execution_schedule)),
        "new_entries_after_winddown": 0,
        "market_return_or_future_open_read": False,
    }
    return execution_targets, execution_schedule, execution_features, audit


def run_portfolio_backtest(
    targets: pd.DataFrame,
    schedule: pd.DataFrame,
    signal_features: pd.DataFrame,
    panel: pd.DataFrame,
    context: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if context.empty or pd.Timestamp(context.iloc[0]["date"]) != start_ts or pd.Timestamp(context.iloc[-1]["date"]) != end_ts:
        raise DataContractError("BTC V3日度上下文未精确覆盖评价首尾")
    market = _normalize_date(panel).set_index(["date", "product_id"]).sort_index()
    targets_by_execution = {
        pd.Timestamp(date): group.sort_values("product_id")
        for date, group in targets.groupby("execution_date", observed=True)
    }
    schedule_by_execution = {
        pd.Timestamp(row.execution_date): pd.Timestamp(row.signal_date)
        for row in schedule.itertuples(index=False)
    }
    features_by_key = {
        (pd.Timestamp(row.execution_date), str(row.product_id)): row
        for row in signal_features.itertuples(index=False)
    }
    costs = {scenario: spot_cost_rate(contract, scenario) for scenario in ("base", "stress")}
    fx_spreads = {scenario: fx_spread_rate(contract, scenario) for scenario in ("base", "stress")}
    initial_capital = float(contract["account"]["initial_capital_cny"])
    first_fx = float(context.iloc[0]["cny_per_usd"])
    cash = {
        scenario: initial_capital / first_fx * (1.0 - fx_spreads[scenario])
        for scenario in ("base", "stress")
    }
    prior_nav_cny = {"base": initial_capital, "stress": initial_capital}
    daily_cash_rate = (1.0 + float(contract["account"]["cash_annual_rate"])) ** (
        1.0 / int(contract["visible_gates"]["annualization_trading_days"])
    ) - 1.0
    increment = float(contract["portfolio"]["quantity_increment"])
    minimum_trade = float(contract["portfolio"]["minimum_trade_notional_usd"])
    maximum_capacity = float(
        contract["universe"]["maximum_order_fraction_of_prior_median_dollar_turnover"]
    )
    positions: dict[str, CapacitySpotPosition] = {}
    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    entry_count = exit_count = resize_sell_count = 0
    capacity_target_capped_count = capacity_sliced_exit_count = 0
    capacity_sliced_resize_count = blocked_entry_count = blocked_exit_count = 0
    stale_mark_count = 0
    maximum_capacity_fraction = maximum_gross_exposure = 0.0
    maximum_position_count = 0

    def capacity_quantity(open_price: float, median: float) -> float:
        if not np.isfinite(open_price) or open_price <= 0.0 or not np.isfinite(median) or median <= 0.0:
            return 0.0
        return _floor_quantity(maximum_capacity * median / open_price, increment)

    def execute(
        *,
        date: pd.Timestamp,
        signal_date: pd.Timestamp,
        product_id: str,
        side: str,
        reason: str,
        quantity: float,
        open_price: float,
        median: float,
        delayed_exit_days: int = 0,
    ) -> None:
        nonlocal maximum_capacity_fraction
        gross = quantity * open_price
        fraction = gross / median
        if fraction > maximum_capacity + 1e-12:
            raise DataContractError(f"{date.date()} {product_id}订单越过冻结容量")
        maximum_capacity_fraction = max(maximum_capacity_fraction, fraction)
        if side == "BUY":
            for scenario in ("base", "stress"):
                cash[scenario] -= gross * (1.0 + costs[scenario])
        else:
            for scenario in ("base", "stress"):
                cash[scenario] += gross * (1.0 - costs[scenario])
        if min(cash.values()) < -1e-7:
            raise DataContractError(f"{date.date()}BTC V3交易后现金为负")
        trade_rows.append(
            {
                "trade_date": date,
                "signal_date": signal_date,
                "product_id": product_id,
                "side": side,
                "reason": reason,
                "quantity": quantity,
                "open_usd": open_price,
                "gross_notional_usd": gross,
                "base_cost_usd": gross * costs["base"],
                "stress_cost_usd": gross * costs["stress"],
                "prior_median_dollar_turnover_20_usd": median,
                "capacity_fraction": fraction,
                "delayed_exit_days": delayed_exit_days,
            }
        )

    for day_index, context_row in enumerate(context.itertuples(index=False)):
        date = pd.Timestamp(context_row.date)
        fx_today = float(context_row.cny_per_usd)
        if day_index > 0:
            for scenario in ("base", "stress"):
                cash[scenario] *= 1.0 + daily_cash_rate
        signal_date = schedule_by_execution.get(date)
        todays_targets = targets_by_execution.get(
            date,
            pd.DataFrame(columns=["product_id", "target_weight", "prior_median_dollar_turnover_20"]),
        )
        target_map = {
            str(row.product_id): row for row in todays_targets.itertuples(index=False)
        }
        if signal_date is not None:
            for product_id, position in positions.items():
                feature = features_by_key.get((date, product_id))
                if feature is not None:
                    position.latest_prior_median_turnover_usd = float(
                        feature.prior_median_dollar_turnover_20
                    )
                if product_id not in target_map and not position.pending_exit:
                    position.pending_exit = True
                    position.exit_signal_date = signal_date
                    position.delayed_exit_days = 0
                    position.pending_resize_target_quantity = None
        if date == end_ts:
            for position in positions.values():
                if not position.pending_exit:
                    position.pending_exit = True
                    position.exit_signal_date = date
                    position.delayed_exit_days = 0
                    position.pending_resize_target_quantity = None

        exited_today: set[str] = set()
        for product_id in list(positions):
            position = positions[product_id]
            if not position.pending_exit:
                continue
            row = _market_row(market, date, product_id)
            if not _valid_execution(row):
                position.delayed_exit_days += 1
                blocked_exit_count += 1
                continue
            open_price = float(row["open"])
            cap_quantity = capacity_quantity(
                open_price, position.latest_prior_median_turnover_usd
            )
            sell_quantity = min(position.quantity, cap_quantity)
            if sell_quantity <= increment / 2:
                position.delayed_exit_days += 1
                blocked_exit_count += 1
                continue
            if date == end_ts and sell_quantity + increment / 2 < position.quantity:
                raise DataContractError("BTC V3评价期末容量内仍无法完整退出")
            partial = sell_quantity + increment / 2 < position.quantity
            execute(
                date=date,
                signal_date=position.exit_signal_date or date,
                product_id=product_id,
                side="SELL",
                reason="TERMINAL_EXIT" if date == end_ts else (
                    "CAPACITY_SLICED_TREND_EXIT" if partial else "TREND_EXIT"
                ),
                quantity=sell_quantity,
                open_price=open_price,
                median=position.latest_prior_median_turnover_usd,
                delayed_exit_days=position.delayed_exit_days,
            )
            exit_count += 1
            position.quantity = _floor_quantity(position.quantity - sell_quantity, increment)
            if partial:
                capacity_sliced_exit_count += 1
                position.delayed_exit_days += 1
            if position.quantity <= increment / 2:
                exited_today.add(product_id)
                del positions[product_id]

        desired_quantities: dict[str, tuple[float, Any]] = {}
        if signal_date is not None and date != end_ts:
            reference_nav_usd = min(prior_nav_cny.values()) / fx_today
            for product_id, target in target_map.items():
                row = _market_row(market, date, product_id)
                if not _valid_execution(row):
                    blocked_entry_count += 1
                    continue
                median = float(target.prior_median_dollar_turnover_20)
                weight_notional = (
                    reference_nav_usd * float(target.target_weight) / (1.0 + costs["stress"])
                )
                capacity_notional = maximum_capacity * median
                target_notional = min(weight_notional, capacity_notional)
                if capacity_notional < weight_notional - 1e-8:
                    capacity_target_capped_count += 1
                target_quantity = _floor_quantity(target_notional / float(row["open"]), increment)
                desired_quantities[product_id] = (target_quantity, target)
                if product_id in positions and not positions[product_id].pending_exit:
                    position = positions[product_id]
                    position.latest_prior_median_turnover_usd = median
                    position.pending_resize_target_quantity = (
                        target_quantity if position.quantity > target_quantity else None
                    )

        for product_id in list(positions):
            position = positions[product_id]
            target_quantity = position.pending_resize_target_quantity
            if position.pending_exit or target_quantity is None:
                continue
            excess = _floor_quantity(max(0.0, position.quantity - target_quantity), increment)
            if excess <= increment / 2:
                position.pending_resize_target_quantity = None
                continue
            row = _market_row(market, date, product_id)
            if not _valid_execution(row):
                continue
            open_price = float(row["open"])
            cap_quantity = capacity_quantity(
                open_price, position.latest_prior_median_turnover_usd
            )
            sell_quantity = min(excess, cap_quantity)
            if sell_quantity <= increment / 2:
                continue
            partial = sell_quantity + increment / 2 < excess
            execute(
                date=date,
                signal_date=signal_date or date,
                product_id=product_id,
                side="SELL",
                reason="CAPACITY_SLICED_WEEKLY_RESIZE_SELL" if partial else "WEEKLY_RESIZE_SELL",
                quantity=sell_quantity,
                open_price=open_price,
                median=position.latest_prior_median_turnover_usd,
            )
            resize_sell_count += 1
            if partial:
                capacity_sliced_resize_count += 1
            position.quantity = _floor_quantity(position.quantity - sell_quantity, increment)
            if position.quantity <= target_quantity + increment / 2:
                position.pending_resize_target_quantity = None
            if position.quantity <= increment / 2:
                del positions[product_id]

        if signal_date is not None and date != end_ts:
            for product_id, (target_quantity, target) in sorted(desired_quantities.items()):
                if product_id in exited_today:
                    continue
                if product_id in positions and positions[product_id].pending_exit:
                    continue
                current = positions[product_id].quantity if product_id in positions else 0.0
                buy_quantity = _floor_quantity(max(0.0, target_quantity - current), increment)
                row = _market_row(market, date, product_id)
                if buy_quantity <= increment / 2 or not _valid_execution(row):
                    continue
                open_price = float(row["open"])
                affordable = min(
                    _floor_quantity(
                        cash[scenario] / (open_price * (1.0 + costs[scenario])), increment
                    )
                    for scenario in ("base", "stress")
                )
                median = float(target.prior_median_dollar_turnover_20)
                buy_quantity = min(
                    buy_quantity,
                    affordable,
                    capacity_quantity(open_price, median),
                )
                gross = buy_quantity * open_price
                if buy_quantity <= increment / 2 or gross < minimum_trade:
                    blocked_entry_count += 1
                    continue
                execute(
                    date=date,
                    signal_date=signal_date,
                    product_id=product_id,
                    side="BUY",
                    reason="CAPACITY_AWARE_WEEKLY_ENTRY_OR_RESIZE_BUY",
                    quantity=buy_quantity,
                    open_price=open_price,
                    median=median,
                )
                if product_id in positions:
                    positions[product_id].quantity = _floor_quantity(
                        positions[product_id].quantity + buy_quantity, increment
                    )
                    positions[product_id].latest_prior_median_turnover_usd = median
                else:
                    positions[product_id] = CapacitySpotPosition(
                        product_id=product_id,
                        quantity=buy_quantity,
                        entry_date=date,
                        last_close=float(row["close"]),
                        latest_prior_median_turnover_usd=median,
                    )
                entry_count += 1

        position_value_usd = 0.0
        quality_complete_today = True
        for product_id, position in positions.items():
            row = _market_row(market, date, product_id)
            close = np.nan if row is None else float(row.get("close", np.nan))
            if not np.isfinite(close) or close <= 0.0:
                close = position.last_close
                quality_complete_today = False
                stale_mark_count += 1
            else:
                position.last_close = close
            position_value_usd += position.quantity * close
        if date == end_ts and positions:
            raise DataContractError(f"BTC V3评价期末仍有持仓：{sorted(positions)}")
        nav_usd = {scenario: cash[scenario] + position_value_usd for scenario in ("base", "stress")}
        if min(nav_usd.values()) <= 0.0:
            raise DataContractError(f"{date.date()}BTC V3组合净值非正")
        reference_close_nav = min(nav_usd.values())
        gross_exposure = position_value_usd / reference_close_nav
        if gross_exposure > float(contract["risk"]["maximum_gross_exposure"]) + 1e-10:
            raise DataContractError(f"{date.date()}BTC V3毛敞口超过100%")
        maximum_gross_exposure = max(maximum_gross_exposure, gross_exposure)
        maximum_position_count = max(maximum_position_count, len(positions))
        nav_cny = (
            {
                scenario: cash[scenario] * fx_today * (1.0 - fx_spreads[scenario])
                for scenario in ("base", "stress")
            }
            if date == end_ts
            else {scenario: nav_usd[scenario] * fx_today for scenario in ("base", "stress")}
        )
        daily_rows.append(
            {
                "trade_date": date,
                "strategy_base_net_return": nav_cny["base"] / prior_nav_cny["base"] - 1.0,
                "strategy_stress_net_return": nav_cny["stress"] / prior_nav_cny["stress"] - 1.0,
                "benchmark_total_return": float(context_row.benchmark_total_return),
                "base_nav_cny": nav_cny["base"],
                "stress_nav_cny": nav_cny["stress"],
                "cny_per_usd_lagged": fx_today,
                "gross_exposure": gross_exposure,
                "net_exposure": gross_exposure,
                "position_count": len(positions),
                "quality_complete": quality_complete_today,
                "capacity_pass": True,
            }
        )
        prior_nav_cny = nav_cny

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    audit = {
        "initial_capital_cny": initial_capital,
        "final_base_nav_cny": float(daily["base_nav_cny"].iloc[-1]),
        "final_stress_nav_cny": float(daily["stress_nav_cny"].iloc[-1]),
        "entry_transaction_count": int(entry_count),
        "exit_transaction_count": int(exit_count),
        "resize_sell_transaction_count": int(resize_sell_count),
        "blocked_entry_count": int(blocked_entry_count),
        "blocked_exit_attempt_count": int(blocked_exit_count),
        "capacity_block_count": 0,
        "capacity_target_capped_count": int(capacity_target_capped_count),
        "capacity_sliced_exit_transaction_count": int(capacity_sliced_exit_count),
        "capacity_sliced_resize_transaction_count": int(capacity_sliced_resize_count),
        "stale_mark_count": int(stale_mark_count),
        "maximum_order_capacity_fraction": float(maximum_capacity_fraction),
        "maximum_gross_exposure": float(maximum_gross_exposure),
        "maximum_net_exposure": float(maximum_gross_exposure),
        "maximum_position_count": int(maximum_position_count),
        "base_spot_cost_rate_per_leg": costs["base"],
        "stress_spot_cost_rate_per_leg": costs["stress"],
        "base_fx_spread_rate_per_conversion": fx_spreads["base"],
        "stress_fx_spread_rate_per_conversion": fx_spreads["stress"],
        "terminal_position_count": int(daily.iloc[-1]["position_count"]),
        "sealed_replication_read": False,
    }
    return daily, trades, audit


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["evaluation"]["metrics"]
    gates = report["evaluation"]["gates"]
    winddown = report["data_audit"]["terminal_winddown"]
    portfolio = report["data_audit"]["portfolio"]
    return "\n".join(
        [
            "# BTC现货容量感知长周期趋势V3可见期结果", "",
            f"状态：`{report['status']}`", "",
            f"- 末端减仓起点：{winddown['winddown_start_date']}。",
            f"- H00300全收益CAGR：{metrics['benchmark_total_return_cagr']:.2%}。",
            f"- 基础/压力策略净CAGR：{metrics['strategy_base_net_cagr']:.2%}/{metrics['strategy_stress_net_cagr']:.2%}。",
            f"- 基础/压力年化净超额：{metrics['base_annualized_excess']:.2%}/{metrics['stress_annualized_excess']:.2%}。",
            f"- 基础/压力净夏普：{metrics['base_strategy_net_sharpe']:.3f}/{metrics['stress_strategy_net_sharpe']:.3f}。",
            f"- 最大订单容量比例：{portfolio['maximum_order_capacity_fraction']:.4%}。",
            f"- 基础/压力40个百分点门：{gates['base_annualized_excess_at_least_40pct']}/{gates['stress_annualized_excess_at_least_40pct']}。",
            f"- 基础/压力1.50夏普门：{gates['base_strategy_sharpe_at_least_1_5']}/{gates['stress_strategy_sharpe_at_least_1_5']}。",
            f"- 全部门通过：{report['evaluation']['all_visible_gates_pass']}。", "",
            f"{report['decision']['next_step']}。", "",
            "封存复验、Paper、Shadow、订单、交易所连接和实盘均未打开。", "",
        ]
    )


def run_visible(
    contract: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    validate_contract(contract)
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    end = pd.Timestamp(contract["historical_partition"]["visible_end"])
    panel, master, fx, benchmark, input_audit = load_visible_inputs(contract)
    targets, schedule, features, signal_audit = build_weekly_targets(
        panel, contract, start=start, end=end
    )
    context = prepare_daily_context(panel, fx, benchmark, contract, start=start, end=end)
    execution_targets, execution_schedule, execution_features, winddown_audit = (
        build_terminal_winddown_plan(targets, schedule, features, context, contract)
    )
    daily, trades, portfolio_audit = run_portfolio_backtest(
        execution_targets, execution_schedule, execution_features,
        panel, context, contract, start=start, end=end,
    )
    correction = load_correction_contract(CONSERVATIVE_EVALUATOR_CONFIG)
    evaluation = evaluate_historical_returns_zero_variance_v2(
        daily, contract, correction,
        bootstrap_repetitions_override=bootstrap_repetitions_override,
    )
    passed = bool(evaluation["all_visible_gates_pass"])
    report = {
        "schema_version": "1.0.0",
        "report_id": "BTC_SPOT_CAPACITY_AWARE_LONG_HORIZON_TREND_V3_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": (
            "VISIBLE_PASS_FORMULA_FAMILY_REPLICATION_AUTHORIZED_NOT_OPENED"
            if passed else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
        ),
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "objective": {
            "initial_capital_cny": contract["account"]["initial_capital_cny"],
            "minimum_annualized_net_excess": contract["visible_gates"]["minimum_annualized_net_excess"],
            "minimum_strategy_net_sharpe": contract["visible_gates"]["minimum_strategy_net_sharpe"],
            "user_transaction_fee_rate_per_leg": contract["account"]["user_transaction_fee_rate_per_leg"],
        },
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
                if passed else "冻结拒绝V3，不打开封存期，不修改30日窗口、容量、产品、信号、波动目标或成本救回"
            ),
        },
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    outputs = contract["outputs"]
    atomic_parquet(ROOT / outputs["visible_daily_returns"], daily)
    atomic_parquet(ROOT / outputs["visible_targets"], execution_targets)
    atomic_parquet(ROOT / outputs["visible_trades"], trades)
    atomic_json(ROOT / outputs["input_audit_json"], report["data_audit"])
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
