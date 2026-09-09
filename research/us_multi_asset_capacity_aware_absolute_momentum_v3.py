"""美国多资产容量感知绝对动量V3。"""

from __future__ import annotations

import copy
import json
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
from research.qdii_cross_market_discount_reversion_zero_variance_v2 import (
    evaluate_historical_returns_zero_variance_v2,
    load_correction_contract,
)
from research.us_leveraged_multi_asset_absolute_momentum_v1 import (
    CONSERVATIVE_EVALUATOR_CONFIG,
    _market_row,
    _normalize_date_column,
    _valid_execution_row,
    atomic_json,
    atomic_parquet,
    atomic_text,
    build_monthly_selections,
    fx_spread_rate,
    prepare_daily_context,
    security_cost_rate,
)
from research.us_liquid_treasury_multi_asset_absolute_momentum_v2 import (
    load_contract as load_v2_contract,
    load_visible_inputs,
    validate_contract as validate_v2_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "us_multi_asset_capacity_aware_absolute_momentum_v3.yaml"

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
    "portfolio.capacity_target_includes_stress_cost_reserve",
    "portfolio.capacity_slice_rounding",
    "portfolio.pending_resize_policy",
    "portfolio.final_mandatory_exit_may_ignore_minimum_trade_notional",
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


@dataclass
class CapacityAwarePosition:
    """共享整数股持仓及待退出、待减仓状态。"""

    ticker: str
    shares: int
    entry_date: pd.Timestamp
    last_raw_close: float
    latest_prior_median_turnover_usd: float
    pending_exit: bool = False
    exit_signal_date: pd.Timestamp | None = None
    delayed_exit_days: int = 0
    pending_resize_target_shares: int | None = None


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
        raise ValueError("V3容量感知配置必须是YAML对象")
    return payload


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    override = load_override(path)
    if override.get("base_candidate_config") != (
        "config/us_liquid_treasury_multi_asset_absolute_momentum_v2.yaml"
    ):
        raise ValueError("V3基础候选路径发生变化")
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
    """证明V3只改变容量感知执行，不弱化公式、成本或硬门。"""

    if override is None:
        override = load_override()
    if base is None:
        base = load_v2_contract()
    validate_v2_contract(base)
    left = _flatten(base)
    right = _flatten(contract)
    changed = {
        path
        for path in set(left) | set(right)
        if left.get(path) != right.get(path)
    }
    if changed != ALLOWED_CHANGED_PATHS:
        raise ValueError(
            "V3差异越过V9授权范围："
            f"unexpected={sorted(changed - ALLOWED_CHANGED_PATHS)}, "
            f"missing={sorted(ALLOWED_CHANGED_PATHS - changed)}"
        )
    reconstruction = override.get("execution_reconstitution", {})
    if reconstruction.get("reconstitution_of") != (
        "US_LIQUID_TREASURY_MULTI_ASSET_ABSOLUTE_MOMENTUM_V2"
    ):
        raise ValueError("V3执行重构父候选发生变化")
    if reconstruction.get("no_v2_performance_available") is not True:
        raise ValueError("V3未声明V2无绩效可用")
    if reconstruction.get(
        "selection_dates_windows_ranking_products_costs_and_gates_unchanged"
    ) is not True:
        raise ValueError("V3未声明信号、产品、成本和硬门保持不变")
    if reconstruction.get("maximum_capacity_fraction_unchanged") is not True:
        raise ValueError("V3未声明容量比例保持不变")
    if contract["protocol"]["candidate_id"] != (
        "US_MULTI_ASSET_CAPACITY_AWARE_ABSOLUTE_MOMENTUM_V3"
    ):
        raise ValueError("V3候选编号不正确")
    if contract["protocol"]["parent_protocol_id"] != (
        "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V9"
    ):
        raise ValueError("V3父协议不正确")
    portfolio = contract["portfolio"]
    required = {
        "capacity_aware_targeting_enabled": True,
        "capacity_target_formula": "MIN_WEIGHT_CEILING_NOTIONAL_AND_PRIOR_MEDIAN_TURNOVER_TIMES_MAXIMUM_ORDER_FRACTION",
        "capacity_target_includes_stress_cost_reserve": True,
        "capacity_slice_rounding": "FLOOR_TO_WHOLE_SHARE",
        "pending_resize_policy": "CONTINUE_DAILY_SELL_SLICES_UNTIL_TARGET_OR_NEXT_REBALANCE",
        "final_mandatory_exit_may_ignore_minimum_trade_notional": True,
    }
    for key, value in required.items():
        if portfolio.get(key) != value:
            raise ValueError(f"V3容量感知字段发生变化：{key}")
    if float(contract["universe"]["maximum_order_fraction_of_prior_median_dollar_turnover"]) != 0.0005:
        raise ValueError("V3容量比例必须保持0.05%")
    if float(portfolio["target_weight_per_selected_asset"]) != 0.50:
        raise ValueError("V3每资产权重上限必须保持50%")
    if any(bool(value) for value in contract["safety"].values()):
        raise ValueError("V3研究边界被打开")


def changed_paths(contract: dict[str, Any] | None = None) -> list[str]:
    base = load_v2_contract()
    reconstructed = load_contract() if contract is None else contract
    left = _flatten(base)
    right = _flatten(reconstructed)
    return sorted(
        path
        for path in set(left) | set(right)
        if left.get(path) != right.get(path)
    )


def run_portfolio_backtest(
    selections: pd.DataFrame,
    schedule: pd.DataFrame,
    signal_features: pd.DataFrame,
    panel: pd.DataFrame,
    context: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """运行容量约束目标、分日卖单、整数股与双成本账本。"""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    calendar = pd.DatetimeIndex(context["date"])
    if calendar.empty or calendar[0] != start_ts or calendar[-1] != end_ts:
        raise DataContractError("V3日度上下文未精确覆盖冻结评价首尾")
    if context["date"].duplicated().any():
        raise DataContractError("V3日度上下文日期重复")
    market_frame = _normalize_date_column(panel)
    market = market_frame.set_index(["date", "ticker"]).sort_index()
    selected_by_execution = {
        pd.Timestamp(date): group.sort_values(["selection_rank", "ticker"])
        for date, group in selections.groupby("execution_date", observed=True)
    }
    schedule_by_execution = {
        pd.Timestamp(row.execution_date): pd.Timestamp(row.signal_date)
        for row in schedule.itertuples(index=False)
    }
    features_by_key = {
        (pd.Timestamp(row.execution_date), str(row.ticker)): row
        for row in signal_features.itertuples(index=False)
    }
    costs = {
        scenario: security_cost_rate(contract, scenario)
        for scenario in ("base", "stress")
    }
    fx_spreads = {
        scenario: fx_spread_rate(contract, scenario)
        for scenario in ("base", "stress")
    }
    initial_capital = float(contract["account"]["initial_capital_cny"])
    first_fx = float(context.iloc[0]["cny_per_usd"])
    cash = {
        scenario: initial_capital / first_fx * (1.0 - fx_spreads[scenario])
        for scenario in ("base", "stress")
    }
    positions: dict[str, CapacityAwarePosition] = {}
    prior_nav_cny = {"base": initial_capital, "stress": initial_capital}
    daily_cash_rate = (1.0 + float(contract["account"]["cash_annual_rate"])) ** (
        1.0 / int(contract["visible_gates"]["annualization_trading_days"])
    ) - 1.0
    weight_ceiling = float(contract["portfolio"]["target_weight_per_selected_asset"])
    minimum_trade = float(contract["portfolio"]["minimum_trade_notional_usd"])
    maximum_capacity = float(
        contract["universe"]["maximum_order_fraction_of_prior_median_dollar_turnover"]
    )
    withholding = float(
        contract["corporate_actions"]["cash_distribution_withholding_rate"]
    )
    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    entry_transaction_count = 0
    exit_transaction_count = 0
    resize_sell_transaction_count = 0
    blocked_entry_count = 0
    blocked_exit_attempt_count = 0
    blocked_resize_attempt_count = 0
    stale_mark_count = 0
    split_event_count = 0
    fractional_cash_in_lieu_count = 0
    distribution_event_count = 0
    capacity_target_capped_count = 0
    capacity_sliced_exit_transaction_count = 0
    capacity_sliced_resize_transaction_count = 0
    maximum_observed_order_capacity_fraction = 0.0
    maximum_position_count = 0
    maximum_gross_exposure = 0.0
    maximum_net_exposure = 0.0

    def capacity_fraction(gross: float, median_turnover: float) -> float:
        if not np.isfinite(median_turnover) or median_turnover <= 0.0:
            return float("inf")
        return gross / median_turnover

    def maximum_capacity_shares(open_price: float, median_turnover: float) -> int:
        if (
            not np.isfinite(open_price)
            or open_price <= 0.0
            or not np.isfinite(median_turnover)
            or median_turnover <= 0.0
        ):
            return 0
        return max(0, int(math.floor(maximum_capacity * median_turnover / open_price)))

    def record_trade(
        *,
        date: pd.Timestamp,
        signal_date: pd.Timestamp,
        ticker: str,
        side: str,
        reason: str,
        shares: int,
        open_price: float,
        prior_median: float,
        delayed_exit_days: int = 0,
    ) -> None:
        nonlocal maximum_observed_order_capacity_fraction
        gross = float(shares) * open_price
        cap_fraction = capacity_fraction(gross, prior_median)
        if cap_fraction > maximum_capacity + 1e-15:
            raise DataContractError(
                f"{date.date()} {ticker}容量感知订单仍越过冻结上限"
            )
        maximum_observed_order_capacity_fraction = max(
            maximum_observed_order_capacity_fraction,
            cap_fraction,
        )
        if side == "BUY":
            for scenario in ("base", "stress"):
                cash[scenario] -= gross * (1.0 + costs[scenario])
        elif side == "SELL":
            for scenario in ("base", "stress"):
                cash[scenario] += gross * (1.0 - costs[scenario])
        else:
            raise ValueError(f"未知交易方向：{side}")
        if min(cash.values()) < -1e-8:
            raise DataContractError(f"{date.date()}容量感知交易后现金为负")
        trade_rows.append(
            {
                "trade_date": date,
                "signal_date": signal_date,
                "ticker": ticker,
                "side": side,
                "reason": reason,
                "shares": int(shares),
                "raw_open_usd": open_price,
                "gross_notional_usd": gross,
                "base_cost_usd": gross * costs["base"],
                "stress_cost_usd": gross * costs["stress"],
                "prior_median_dollar_turnover_20_usd": prior_median,
                "capacity_fraction": cap_fraction,
                "delayed_exit_days": int(delayed_exit_days),
            }
        )

    for day_index, context_row in enumerate(context.itertuples(index=False)):
        date = pd.Timestamp(context_row.date)
        fx_today = float(context_row.cny_per_usd)
        if day_index > 0:
            for scenario in ("base", "stress"):
                cash[scenario] *= 1.0 + daily_cash_rate

        entitled_distributions: list[tuple[int, float]] = []
        for ticker in list(positions):
            position = positions[ticker]
            row = _market_row(market, date, ticker)
            if row is None:
                continue
            ratio = float(row.get("split_ratio_at_open", 1.0))
            if not np.isfinite(ratio) or ratio <= 0.0:
                raise DataContractError(f"{date.date()} {ticker}拆分比率无效")
            if not np.isclose(ratio, 1.0, rtol=0.0, atol=1e-12):
                exact_shares = position.shares * ratio
                whole_shares = int(math.floor(exact_shares + 1e-10))
                fractional = max(0.0, exact_shares - whole_shares)
                if position.pending_resize_target_shares is not None:
                    position.pending_resize_target_shares = int(
                        math.floor(position.pending_resize_target_shares * ratio + 1e-10)
                    )
                if fractional > 1e-10:
                    open_price = float(row.get("raw_open", np.nan))
                    if not np.isfinite(open_price) or open_price <= 0.0:
                        raise DataContractError(
                            f"{date.date()} {ticker}零股现金替代缺少有效开盘"
                        )
                    for scenario in ("base", "stress"):
                        cash[scenario] += fractional * open_price
                    fractional_cash_in_lieu_count += 1
                position.shares = whole_shares
                split_event_count += 1
                if position.shares == 0:
                    del positions[ticker]
                    continue
            distribution = float(
                row.get("cash_distribution_per_post_event_share_usd", 0.0)
            )
            if not np.isfinite(distribution) or distribution < -1e-12:
                raise DataContractError(f"{date.date()} {ticker}现金分配无效")
            if distribution > 0.0 and position.shares > 0:
                entitled_distributions.append((position.shares, distribution))
                distribution_event_count += 1

        execution_signal = schedule_by_execution.get(date)
        selected_today = selected_by_execution.get(
            date,
            pd.DataFrame(
                columns=[
                    "signal_date",
                    "execution_date",
                    "ticker",
                    "selection_rank",
                    "prior_median_dollar_turnover_20",
                ]
            ),
        )
        if execution_signal is not None:
            selected_set = set(selected_today["ticker"].astype(str))
            for ticker, position in positions.items():
                feature = features_by_key.get((date, ticker))
                if feature is not None:
                    position.latest_prior_median_turnover_usd = float(
                        feature.prior_median_dollar_turnover_20
                    )
                if ticker not in selected_set and not position.pending_exit:
                    position.pending_exit = True
                    position.exit_signal_date = execution_signal
                    position.delayed_exit_days = 0
                    position.pending_resize_target_shares = None
        if date == end_ts:
            for position in positions.values():
                if not position.pending_exit:
                    position.pending_exit = True
                    position.exit_signal_date = date
                    position.delayed_exit_days = 0
                    position.pending_resize_target_shares = None

        exited_tickers_today: set[str] = set()
        for ticker in list(positions):
            position = positions[ticker]
            if not position.pending_exit:
                continue
            row = _market_row(market, date, ticker)
            if not _valid_execution_row(row):
                position.delayed_exit_days += 1
                blocked_exit_attempt_count += 1
                continue
            open_price = float(row["raw_open"])
            maximum_shares = maximum_capacity_shares(
                open_price,
                position.latest_prior_median_turnover_usd,
            )
            sell_shares = min(position.shares, maximum_shares)
            if sell_shares <= 0:
                position.delayed_exit_days += 1
                blocked_exit_attempt_count += 1
                continue
            if date == end_ts and sell_shares < position.shares:
                raise DataContractError(
                    f"评价期末容量内无法完整退出美国ETF持仓：{ticker}"
                )
            partial = sell_shares < position.shares
            record_trade(
                date=date,
                signal_date=position.exit_signal_date or date,
                ticker=ticker,
                side="SELL",
                reason=(
                    "TERMINAL_EXIT"
                    if date == end_ts
                    else "CAPACITY_SLICED_SELECTION_EXIT" if partial else "SELECTION_EXIT"
                ),
                shares=sell_shares,
                open_price=open_price,
                prior_median=position.latest_prior_median_turnover_usd,
                delayed_exit_days=position.delayed_exit_days,
            )
            exit_transaction_count += 1
            position.shares -= sell_shares
            if partial:
                capacity_sliced_exit_transaction_count += 1
                position.delayed_exit_days += 1
            if position.shares == 0:
                exited_tickers_today.add(ticker)
                del positions[ticker]

        desired_targets: dict[str, tuple[int, Any]] = {}
        if execution_signal is not None and date != end_ts:
            reference_nav_usd = min(prior_nav_cny.values()) / fx_today
            weight_notional = (
                reference_nav_usd * weight_ceiling / (1.0 + costs["stress"])
            )
            for selection in selected_today.itertuples(index=False):
                ticker = str(selection.ticker)
                row = _market_row(market, date, ticker)
                if not _valid_execution_row(row):
                    blocked_entry_count += 1
                    continue
                prior_median = float(selection.prior_median_dollar_turnover_20)
                capacity_notional = maximum_capacity * prior_median
                target_notional = min(weight_notional, capacity_notional)
                if capacity_notional < weight_notional - 1e-9:
                    capacity_target_capped_count += 1
                open_price = float(row["raw_open"])
                target_shares = max(0, int(math.floor(target_notional / open_price)))
                desired_targets[ticker] = (target_shares, selection)
                if ticker in positions and not positions[ticker].pending_exit:
                    position = positions[ticker]
                    position.latest_prior_median_turnover_usd = prior_median
                    position.pending_resize_target_shares = (
                        target_shares if position.shares > target_shares else None
                    )

        for ticker in list(positions):
            position = positions[ticker]
            target_shares = position.pending_resize_target_shares
            if position.pending_exit or target_shares is None:
                continue
            excess_shares = max(0, position.shares - target_shares)
            if excess_shares <= 0:
                position.pending_resize_target_shares = None
                continue
            row = _market_row(market, date, ticker)
            if not _valid_execution_row(row):
                blocked_resize_attempt_count += 1
                continue
            open_price = float(row["raw_open"])
            maximum_shares = maximum_capacity_shares(
                open_price,
                position.latest_prior_median_turnover_usd,
            )
            sell_shares = min(excess_shares, maximum_shares)
            gross = sell_shares * open_price
            if sell_shares <= 0 or gross < minimum_trade:
                blocked_resize_attempt_count += 1
                if sell_shares == excess_shares:
                    position.pending_resize_target_shares = None
                continue
            partial = sell_shares < excess_shares
            record_trade(
                date=date,
                signal_date=execution_signal or date,
                ticker=ticker,
                side="SELL",
                reason="CAPACITY_SLICED_MONTHLY_RESIZE_SELL" if partial else "MONTHLY_RESIZE_SELL",
                shares=sell_shares,
                open_price=open_price,
                prior_median=position.latest_prior_median_turnover_usd,
            )
            position.shares -= sell_shares
            resize_sell_transaction_count += 1
            if partial:
                capacity_sliced_resize_transaction_count += 1
            if position.shares <= target_shares:
                position.pending_resize_target_shares = None
            if position.shares == 0:
                del positions[ticker]

        if execution_signal is not None and date != end_ts:
            for ticker, (target_shares, selection) in sorted(
                desired_targets.items(),
                key=lambda item: (item[1][1].selection_rank, item[0]),
            ):
                if ticker in exited_tickers_today:
                    continue
                if ticker in positions and positions[ticker].pending_exit:
                    continue
                current_shares = positions[ticker].shares if ticker in positions else 0
                desired_shares = max(0, target_shares - current_shares)
                if desired_shares <= 0:
                    continue
                row = _market_row(market, date, ticker)
                if not _valid_execution_row(row):
                    blocked_entry_count += 1
                    continue
                open_price = float(row["raw_open"])
                affordable = min(
                    int(
                        math.floor(
                            cash[scenario] / (open_price * (1.0 + costs[scenario]))
                        )
                    )
                    for scenario in ("base", "stress")
                )
                prior_median = float(selection.prior_median_dollar_turnover_20)
                capacity_shares = maximum_capacity_shares(open_price, prior_median)
                buy_shares = min(desired_shares, max(0, affordable), capacity_shares)
                gross = buy_shares * open_price
                if buy_shares <= 0 or gross < minimum_trade:
                    blocked_entry_count += 1
                    continue
                record_trade(
                    date=date,
                    signal_date=execution_signal,
                    ticker=ticker,
                    side="BUY",
                    reason="CAPACITY_AWARE_MONTHLY_ENTRY_OR_RESIZE_BUY",
                    shares=buy_shares,
                    open_price=open_price,
                    prior_median=prior_median,
                )
                if ticker in positions:
                    positions[ticker].shares += buy_shares
                    positions[ticker].latest_prior_median_turnover_usd = prior_median
                else:
                    positions[ticker] = CapacityAwarePosition(
                        ticker=ticker,
                        shares=buy_shares,
                        entry_date=date,
                        last_raw_close=float(row["raw_close"]),
                        latest_prior_median_turnover_usd=prior_median,
                    )
                entry_transaction_count += 1

        for shares, distribution in entitled_distributions:
            net_cash = shares * distribution * (1.0 - withholding)
            for scenario in ("base", "stress"):
                cash[scenario] += net_cash

        position_value_usd = 0.0
        quality_complete_today = True
        for ticker, position in positions.items():
            row = _market_row(market, date, ticker)
            close_price = np.nan if row is None else float(row.get("raw_close", np.nan))
            if not np.isfinite(close_price) or close_price <= 0.0:
                close_price = position.last_raw_close
                quality_complete_today = False
                stale_mark_count += 1
            else:
                position.last_raw_close = close_price
            position_value_usd += position.shares * close_price
        if date == end_ts and positions:
            raise DataContractError(
                f"评价期末仍有无法退出的美国ETF持仓：{sorted(positions)}"
            )
        nav_usd = {
            scenario: cash[scenario] + position_value_usd
            for scenario in ("base", "stress")
        }
        if min(nav_usd.values()) <= 0.0:
            raise DataContractError(f"{date.date()}V3组合净值非正")
        reference_nav_usd_close = min(nav_usd.values())
        gross_exposure = position_value_usd / reference_nav_usd_close
        net_exposure = gross_exposure
        maximum_position_count = max(maximum_position_count, len(positions))
        maximum_gross_exposure = max(maximum_gross_exposure, gross_exposure)
        maximum_net_exposure = max(maximum_net_exposure, abs(net_exposure))
        if gross_exposure > float(contract["risk"]["maximum_gross_exposure"]) + 1e-12:
            raise DataContractError(f"{date.date()}V3毛敞口超过冻结上限")
        if abs(net_exposure) > float(contract["risk"]["maximum_absolute_net_exposure"]) + 1e-12:
            raise DataContractError(f"{date.date()}V3净敞口超过冻结上限")
        if date == end_ts:
            nav_cny = {
                scenario: cash[scenario] * fx_today * (1.0 - fx_spreads[scenario])
                for scenario in ("base", "stress")
            }
        else:
            nav_cny = {
                scenario: nav_usd[scenario] * fx_today
                for scenario in ("base", "stress")
            }
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
                "net_exposure": net_exposure,
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
        "entry_transaction_count": int(entry_transaction_count),
        "exit_transaction_count": int(exit_transaction_count),
        "resize_sell_transaction_count": int(resize_sell_transaction_count),
        "blocked_entry_count": int(blocked_entry_count),
        "blocked_exit_attempt_count": int(blocked_exit_attempt_count),
        "blocked_resize_attempt_count": int(blocked_resize_attempt_count),
        "capacity_block_count": 0,
        "capacity_target_capped_count": int(capacity_target_capped_count),
        "capacity_sliced_exit_transaction_count": int(
            capacity_sliced_exit_transaction_count
        ),
        "capacity_sliced_resize_transaction_count": int(
            capacity_sliced_resize_transaction_count
        ),
        "stale_mark_count": int(stale_mark_count),
        "split_event_count_for_held_positions": int(split_event_count),
        "fractional_cash_in_lieu_count": int(fractional_cash_in_lieu_count),
        "distribution_event_count_for_held_positions": int(distribution_event_count),
        "maximum_observed_order_capacity_fraction": float(
            maximum_observed_order_capacity_fraction
        ),
        "maximum_gross_exposure": float(maximum_gross_exposure),
        "maximum_net_exposure": float(maximum_net_exposure),
        "maximum_position_count": int(maximum_position_count),
        "base_security_cost_rate_per_leg": costs["base"],
        "stress_security_cost_rate_per_leg": costs["stress"],
        "base_fx_spread_rate_per_conversion": fx_spreads["base"],
        "stress_fx_spread_rate_per_conversion": fx_spreads["stress"],
        "terminal_position_count": int(daily.iloc[-1]["position_count"]),
        "sealed_replication_read": False,
    }
    return daily, trades, audit


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["evaluation"]["metrics"]
    gates = report["evaluation"]["gates"]
    portfolio = report["data_audit"]["portfolio"]
    return "\n".join(
        [
            "# 美国多资产容量感知绝对动量V3可见期结果",
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
            f"- 容量封顶目标次数：{portfolio['capacity_target_capped_count']}。",
            f"- 最大实际订单/历史中位成交额：{portfolio['maximum_observed_order_capacity_fraction']:.4%}。",
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
    """运行唯一一次V3可见期，不读取封存区。"""

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
        "report_id": "US_MULTI_ASSET_CAPACITY_AWARE_ABSOLUTE_MOMENTUM_V3_VISIBLE",
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
                else "冻结拒绝V3，不打开封存期，不修改容量、权重、产品或信号救回"
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
