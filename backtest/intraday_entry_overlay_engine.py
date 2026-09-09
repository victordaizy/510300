"""估值目标仓位与十五分钟进场指标的组合回测引擎。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtest.small_account_execution import quantize_position


@dataclass(frozen=True)
class OverlayCosts:
    commission_rate: float
    minimum_commission_cny: float
    slippage_bps_per_leg: float
    lot_size: int
    minimum_trade_shares: int
    cash_annual_rate: float


def _commission(notional: float, costs: OverlayCosts) -> float:
    if notional <= 0:
        return 0.0
    return max(notional * costs.commission_rate, costs.minimum_commission_cny)


def prepare_executable_targets(
    valuation_signals: pd.DataFrame,
    trading_dates: pd.Series,
    position_step: float,
) -> pd.DataFrame:
    """把收盘后估值信号转换成下一交易日可执行的五档仓位。"""

    required = {"date", "target_position"}
    missing = required.difference(valuation_signals.columns)
    if missing:
        raise ValueError(f"估值信号缺少字段：{sorted(missing)}")
    columns = ["date", "target_position"]
    if "risk_off_override" in valuation_signals.columns:
        columns.append("risk_off_override")
    signals = valuation_signals[columns].copy()
    if "risk_off_override" not in signals.columns:
        signals["risk_off_override"] = False
    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()
    signals = signals.sort_values("date").drop_duplicates("date", keep="last")
    signals["target_position_grid"] = signals["target_position"].map(
        lambda value: quantize_position(float(value), position_step)
    )
    calendar = pd.DataFrame({"date": pd.to_datetime(trading_dates).dt.normalize()})
    calendar = calendar.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    merged = calendar.merge(
        signals[["date", "target_position_grid", "risk_off_override"]], on="date", how="left"
    )
    merged["signal_target_position"] = merged["target_position_grid"].ffill()
    merged["executable_target_position"] = merged["signal_target_position"].shift(1)
    merged["executable_target_position"] = merged[
        "executable_target_position"
    ].fillna(merged["signal_target_position"])
    merged["executable_risk_off_override"] = (
        merged["risk_off_override"].fillna(False).astype(bool).shift(1, fill_value=False)
    )
    if merged["executable_target_position"].isna().any():
        raise ValueError("回测首日以前没有可用估值目标仓位")
    return merged[["date", "executable_target_position", "executable_risk_off_override"]]


def prepare_intraday_events(
    indicator: pd.DataFrame,
    event_column: str,
    signal_column: str,
    accepted_signals: set[str],
) -> dict[pd.Timestamp, dict[str, Any]]:
    """提取每天第一条合格信号及其下一根K线开盘价。"""

    required = {
        "signal_asof", "trade_date", "bar_slot", "open", signal_column, event_column
    }
    missing = required.difference(indicator.columns)
    if missing:
        raise ValueError(f"十五分钟指标缺少字段：{sorted(missing)}")
    data = indicator.copy().sort_values("signal_asof").reset_index(drop=True)
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    data["next_open"] = pd.to_numeric(data["open"], errors="coerce").shift(-1)
    data["next_trade_date"] = data["trade_date"].shift(-1)
    eligible = (
        data[event_column].fillna(False).astype(bool)
        & data[signal_column].isin(accepted_signals)
        & data["bar_slot"].lt(16)
        & data["next_trade_date"].eq(data["trade_date"])
        & data["next_open"].gt(0)
    )
    events: dict[pd.Timestamp, dict[str, Any]] = {}
    for row in data.loc[eligible].itertuples(index=False):
        date = pd.Timestamp(row.trade_date).normalize()
        events.setdefault(
            date,
            {
                "signal_asof": pd.Timestamp(row.signal_asof),
                "technical_signal": str(getattr(row, signal_column)),
                "raw_execution_price": float(row.next_open),
            },
        )
    return events


def _desired_shares(
    target_position: float,
    equity: float,
    raw_price: float,
    side: str,
    costs: OverlayCosts,
) -> int:
    slip = costs.slippage_bps_per_leg / 10_000.0
    execution_price = raw_price * (1.0 + slip if side == "BUY" else 1.0 - slip)
    return max(
        int(np.floor(target_position * equity / execution_price / costs.lot_size))
        * costs.lot_size,
        0,
    )


def run_valuation_entry_policy(
    daily: pd.DataFrame,
    targets: pd.DataFrame,
    dividends: pd.DataFrame,
    buy_events_by_date: dict[pd.Timestamp, dict[str, Any]],
    sell_events_by_date: dict[pd.Timestamp, dict[str, Any]],
    costs: OverlayCosts,
    initial_cash: float,
    policy: str,
    maximum_wait_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """运行估值开盘基线或技术信号延迟加仓策略。"""

    if policy not in {"VALUATION_OPEN_BASELINE", "VALUATION_TECHNICAL_OVERLAY"}:
        raise ValueError("未知回测策略")
    if maximum_wait_days < 1:
        raise ValueError("最大等待交易日必须为正整数")
    required_daily = {"date", "open", "close"}
    missing = required_daily.difference(daily.columns)
    if missing:
        raise ValueError(f"ETF日线缺少字段：{sorted(missing)}")

    market = daily.copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    target_frame = targets.copy()
    target_frame["date"] = pd.to_datetime(target_frame["date"]).dt.normalize()
    if "executable_risk_off_override" not in target_frame.columns:
        target_frame["executable_risk_off_override"] = False
    market = market.merge(target_frame, on="date", how="inner", validate="one_to_one")
    if market.empty:
        raise ValueError("日线与估值目标没有重叠日期")

    dividend_events = dividends.copy()
    for column in ["ex_date", "payment_date"]:
        dividend_events[column] = pd.to_datetime(dividend_events[column]).dt.normalize()
    ex_by_date = {
        date: group.to_dict("records")
        for date, group in dividend_events.groupby("ex_date", sort=False)
    }

    cash = float(initial_cash)
    shares = 0
    receivable = 0.0
    scheduled_payments: dict[pd.Timestamp, float] = {}
    pending_age: int | None = None
    pending_side: str | None = None
    pending_started: pd.Timestamp | None = None
    ledger_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    diagnostics = {
        "valuation_buy_opportunities": 0,
        "technical_confirmed_buys": 0,
        "fallback_buys": 0,
        "valuation_sell_opportunities": 0,
        "technical_confirmed_sells": 0,
        "fallback_sells": 0,
        "risk_off_immediate_sells": 0,
        "cancelled_pending_orders": 0,
        "technical_wait_days": [],
    }

    def execute(
        *,
        date: pd.Timestamp,
        raw_price: float,
        target_position: float,
        reason: str,
        signal_asof: pd.Timestamp | None = None,
        entry_signal: str | None = None,
        force_small_initial_trade: bool = False,
    ) -> None:
        nonlocal cash, shares
        equity = cash + shares * raw_price + receivable
        current_notional = shares * raw_price
        side = "BUY" if target_position * equity > current_notional else "SELL"
        desired = _desired_shares(target_position, equity, raw_price, side, costs)
        signed_quantity = desired - shares
        if signed_quantity == 0:
            return
        if (
            abs(signed_quantity) < costs.minimum_trade_shares
            and not force_small_initial_trade
        ):
            return
        slip = costs.slippage_bps_per_leg / 10_000.0
        execution_price = raw_price * (
            1.0 + slip if signed_quantity > 0 else 1.0 - slip
        )
        if signed_quantity > 0:
            quantity = signed_quantity
            while quantity >= costs.lot_size:
                notional = quantity * execution_price
                commission = _commission(notional, costs)
                if notional + commission <= cash + 1e-9:
                    break
                quantity -= costs.lot_size
            if quantity < costs.lot_size:
                return
            signed_quantity = quantity
            notional = quantity * execution_price
            commission = _commission(notional, costs)
            cash -= notional + commission
            shares += quantity
        else:
            quantity = min(-signed_quantity, shares)
            quantity = quantity // costs.lot_size * costs.lot_size
            if quantity <= 0:
                return
            notional = quantity * execution_price
            commission = _commission(notional, costs)
            cash += notional - commission
            shares -= quantity
            signed_quantity = -quantity
        trade_rows.append(
            {
                "date": date,
                "side": "买入" if signed_quantity > 0 else "卖出",
                "quantity": abs(int(signed_quantity)),
                "raw_price": float(raw_price),
                "execution_price": float(execution_price),
                "notional_cny": float(abs(signed_quantity) * execution_price),
                "commission_cny": float(commission),
                "target_position": float(target_position),
                "reason": reason,
                "signal_asof": signal_asof,
                "entry_signal": entry_signal,
            }
        )

    for day_number, row in enumerate(market.itertuples(index=False)):
        date = pd.Timestamp(row.date).normalize()
        raw_open = float(row.open)
        raw_close = float(row.close)
        target = float(row.executable_target_position)
        risk_off_override = bool(row.executable_risk_off_override)
        shares_at_start = shares
        cash += max(cash, 0.0) * costs.cash_annual_rate / 242.0

        for event in ex_by_date.get(date, []):
            entitlement = shares_at_start * float(event["cash_dividend_per_share"])
            receivable += entitlement
            payment_date = pd.Timestamp(event["payment_date"]).normalize()
            scheduled_payments[payment_date] = (
                scheduled_payments.get(payment_date, 0.0) + entitlement
            )

        if day_number == 0:
            execute(
                date=date,
                raw_price=raw_open,
                target_position=target,
                reason="共同初始估值仓位",
                force_small_initial_trade=True,
            )
        else:
            open_equity = cash + shares * raw_open + receivable
            desired_at_open = _desired_shares(
                target,
                open_equity,
                raw_open,
                "BUY" if target * open_equity > shares * raw_open else "SELL",
                costs,
            )
            delta_at_open = desired_at_open - shares
            direction = (
                "BUY" if delta_at_open >= costs.minimum_trade_shares
                else "SELL" if delta_at_open <= -costs.minimum_trade_shares
                else None
            )
            if direction is None:
                if pending_age is not None:
                    diagnostics["cancelled_pending_orders"] += 1
                pending_age = None
                pending_side = None
                pending_started = None
            elif policy == "VALUATION_OPEN_BASELINE":
                key = "valuation_buy_opportunities" if direction == "BUY" else "valuation_sell_opportunities"
                diagnostics[key] += 1
                execute(
                    date=date,
                    raw_price=raw_open,
                    target_position=target,
                    reason=f"估值目标{'上升加仓' if direction == 'BUY' else '下降减仓'}，下一交易日开盘执行",
                )
            elif direction == "SELL" and risk_off_override:
                if pending_age is not None:
                    diagnostics["cancelled_pending_orders"] += 1
                diagnostics["valuation_sell_opportunities"] += 1
                diagnostics["risk_off_immediate_sells"] += 1
                execute(
                    date=date,
                    raw_price=raw_open,
                    target_position=target,
                    reason="危机强制减仓，开盘立即执行",
                )
                pending_age = None
                pending_side = None
                pending_started = None
            else:
                if pending_side != direction:
                    if pending_age is not None:
                        diagnostics["cancelled_pending_orders"] += 1
                    key = "valuation_buy_opportunities" if direction == "BUY" else "valuation_sell_opportunities"
                    diagnostics[key] += 1
                    pending_age = 0
                    pending_side = direction
                    pending_started = date
                if pending_age >= maximum_wait_days:
                    execute(
                        date=date,
                        raw_price=raw_open,
                        target_position=target,
                        reason=f"技术信号等待超时，开盘回退{'加仓' if direction == 'BUY' else '减仓'}",
                    )
                    diagnostics["fallback_buys" if direction == "BUY" else "fallback_sells"] += 1
                    diagnostics["technical_wait_days"].append(pending_age)
                    pending_age = None
                    pending_side = None
                    pending_started = None
                else:
                    technical = (
                        buy_events_by_date.get(date)
                        if direction == "BUY"
                        else sell_events_by_date.get(date)
                    )
                    if technical is not None:
                        execute(
                            date=date,
                            raw_price=float(technical["raw_execution_price"]),
                            target_position=target,
                            reason=f"估值要求{'加仓' if direction == 'BUY' else '减仓'}，十五分钟技术信号确认",
                            signal_asof=pd.Timestamp(technical["signal_asof"]),
                            entry_signal=str(technical["technical_signal"]),
                        )
                        diagnostics[
                            "technical_confirmed_buys" if direction == "BUY" else "technical_confirmed_sells"
                        ] += 1
                        diagnostics["technical_wait_days"].append(pending_age)
                        pending_age = None
                        pending_side = None
                        pending_started = None
                    else:
                        pending_age += 1

        payment = float(scheduled_payments.pop(date, 0.0))
        if payment:
            cash += payment
            receivable -= payment
        equity = cash + shares * raw_close + receivable
        ledger_rows.append(
            {
                "date": date,
                "target_position": target,
                "shares": int(shares),
                "cash": float(cash),
                "dividend_receivable": float(receivable),
                "close": raw_close,
                "equity": float(equity),
                "actual_position": float(shares * raw_close / equity) if equity else 0.0,
                "pending_buy_age": pending_age,
                "pending_buy_started": pending_started,
                "pending_side": pending_side,
            }
        )

    ledger = pd.DataFrame(ledger_rows)
    ledger["daily_return"] = ledger["equity"].pct_change().fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    trades = pd.DataFrame(trade_rows)
    diagnostics["unresolved_pending_order_at_end"] = pending_age is not None
    diagnostics["average_technical_wait_days"] = (
        float(np.mean(diagnostics["technical_wait_days"]))
        if diagnostics["technical_wait_days"]
        else None
    )
    return ledger, trades, diagnostics


def summarize_policy(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
) -> dict[str, Any]:
    elapsed_days = max((ledger["date"].iloc[-1] - ledger["date"].iloc[0]).days, 1)
    elapsed_years = elapsed_days / 365.25
    total_return = float(ledger["equity"].iloc[-1] / initial_cash - 1.0)
    cagr = float((1.0 + total_return) ** (1.0 / elapsed_years) - 1.0)
    volatility = float(ledger["daily_return"].std(ddof=1) * np.sqrt(242.0))
    annual_mean = float(ledger["daily_return"].mean() * 242.0)
    return {
        "start_date": str(ledger["date"].iloc[0].date()),
        "end_date": str(ledger["date"].iloc[-1].date()),
        "elapsed_years": elapsed_years,
        "observations": int(len(ledger)),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": annual_mean / volatility if volatility > 0 else None,
        "max_drawdown": float(ledger["drawdown"].min()),
        "average_position": float(ledger["actual_position"].mean()),
        "trade_count": int(len(trades)),
        "buy_count": int(trades["side"].eq("买入").sum()) if not trades.empty else 0,
        "sell_count": int(trades["side"].eq("卖出").sum()) if not trades.empty else 0,
        "commission_cny": float(trades["commission_cny"].sum()) if not trades.empty else 0.0,
        "ending_equity": float(ledger["equity"].iloc[-1]),
    }
