"""510300 日线多头/现金回测引擎。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestCosts:
    commission_rate: float = 0.0003
    minimum_commission_cny: float = 5.0
    stamp_duty_rate: float = 0.0
    slippage_bps: float = 5.0
    lot_size: int = 100
    cash_annual_rate: float = 0.0


def _commission(notional: float, costs: BacktestCosts) -> float:
    if notional <= 0:
        return 0.0
    return max(costs.minimum_commission_cny, notional * costs.commission_rate)


def _validate_inputs(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_prices = {"date", "open", "high", "low", "close"}
    required_dividends = {"ex_date", "payment_date", "cash_dividend_per_share"}
    required_targets = {"date", "target_position"}
    if missing := required_prices.difference(prices.columns):
        raise ValueError(f"行情缺少字段：{sorted(missing)}")
    if missing := required_dividends.difference(dividends.columns):
        raise ValueError(f"分红缺少字段：{sorted(missing)}")
    if missing := required_targets.difference(targets.columns):
        raise ValueError(f"目标仓位缺少字段：{sorted(missing)}")

    market = prices.copy()
    events = dividends.copy()
    signals = targets.copy()
    market["date"] = pd.to_datetime(market["date"])
    events["ex_date"] = pd.to_datetime(events["ex_date"])
    events["payment_date"] = pd.to_datetime(events["payment_date"])
    signals["date"] = pd.to_datetime(signals["date"])
    market = market.sort_values("date").reset_index(drop=True)
    signals = signals.sort_values("date").reset_index(drop=True)
    if market["date"].duplicated().any() or signals["date"].duplicated().any():
        raise ValueError("行情或目标仓位日期重复")
    numeric_prices = market[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if numeric_prices.isna().any().any() or (numeric_prices <= 0).any().any():
        raise ValueError("行情价格存在空值、零值或负值")
    market[["open", "high", "low", "close"]] = numeric_prices
    signals["target_position"] = pd.to_numeric(signals["target_position"], errors="coerce")
    if signals["target_position"].isna().any() or not signals["target_position"].between(0, 1).all():
        raise ValueError("目标仓位必须位于0到1之间")
    events["cash_dividend_per_share"] = pd.to_numeric(
        events["cash_dividend_per_share"], errors="coerce"
    )
    if events["cash_dividend_per_share"].isna().any() or (events["cash_dividend_per_share"] < 0).any():
        raise ValueError("分红金额存在空值或负值")
    return market, events, signals


def run_long_cash_backtest(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    initial_cash: float,
    costs: BacktestCosts,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    minimum_trade_notional_cny: float = 0.0,
    minimum_trade_shares: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """执行日线回测。

    `targets.date` 是收盘后形成信号的日期，目标仓位只会在下一交易日开盘执行。
    分红在除息日确认应收，在发放日转为可用现金；净值始终包含应收股利。
    """

    market, events, signals = _validate_inputs(prices, dividends, targets)
    if minimum_trade_notional_cny < 0:
        raise ValueError("最小成交金额不能为负数")
    if minimum_trade_shares < 0 or minimum_trade_shares % costs.lot_size != 0:
        raise ValueError("最小成交份额必须是非负整手数")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    market = market.loc[market["date"].between(start, end)].copy().reset_index(drop=True)
    if len(market) < 2:
        raise ValueError("回测区间至少需要两个交易日")
    signals = signals.loc[signals["date"].between(start, end)].copy()
    signal_by_date = signals.set_index("date").to_dict("index")
    has_execution_control = "trade_allowed" in signals.columns
    events_by_ex_date = {
        date: frame.to_dict("records") for date, frame in events.groupby("ex_date", sort=False)
    }

    cash = float(initial_cash)
    shares = 0
    dividend_receivable = 0.0
    scheduled_payments: dict[pd.Timestamp, float] = {}
    ledger_rows: list[dict] = []
    trade_rows: list[dict] = []
    previous_date: pd.Timestamp | None = None
    previous_target = 0.0

    for row in market.itertuples(index=False):
        date = pd.Timestamp(row.date)
        shares_at_start = shares
        if ledger_rows and costs.cash_annual_rate != 0:
            cash *= 1.0 + costs.cash_annual_rate / 242.0

        entitlement_today = 0.0
        for event in events_by_ex_date.get(date, []):
            entitlement = shares_at_start * float(event["cash_dividend_per_share"])
            entitlement_today += entitlement
            payment_date = pd.Timestamp(event["payment_date"])
            scheduled_payments[payment_date] = scheduled_payments.get(payment_date, 0.0) + entitlement
        dividend_receivable += entitlement_today

        signal_date = previous_date
        signal = signal_by_date.get(signal_date, {})
        if signal:
            previous_target = float(signal["target_position"])
        target = previous_target
        trade_allowed = bool(signal.get("trade_allowed", True)) if signal else not has_execution_control
        risk_off_override = bool(signal.get("risk_off_override", False)) if signal else False
        signal_reason = str(signal.get("signal_reason", "常规目标调整")) if signal else "目标维持"
        open_equity = cash + shares * float(row.open) + dividend_receivable
        buy_price = float(row.open) * (1.0 + costs.slippage_bps / 10000.0)
        sell_price = float(row.open) * (1.0 - costs.slippage_bps / 10000.0)
        reference_price = buy_price if target * open_equity >= shares * float(row.open) else sell_price
        desired_shares = int(np.floor((target * open_equity) / reference_price / costs.lot_size)) * costs.lot_size
        desired_shares = max(desired_shares, 0)

        quantity = desired_shares - shares
        side = "无交易"
        execution_price = np.nan
        commission = 0.0
        stamp_duty = 0.0
        if not trade_allowed:
            quantity = 0
        if quantity > 0:
            while quantity > 0:
                notional = quantity * buy_price
                commission = _commission(notional, costs)
                if notional + commission <= cash + 1e-9:
                    break
                quantity -= costs.lot_size
            if quantity > 0 and (
                quantity * buy_price + 1e-9 < minimum_trade_notional_cny
                or quantity < minimum_trade_shares
            ):
                quantity = 0
                commission = 0.0
            if quantity > 0:
                notional = quantity * buy_price
                commission = _commission(notional, costs)
                cash -= notional + commission
                shares += quantity
                side = "买入"
                execution_price = buy_price
            else:
                commission = 0.0
        elif quantity < 0:
            sell_quantity = min(-quantity, shares_at_start)
            sell_quantity = sell_quantity // costs.lot_size * costs.lot_size
            if (
                (
                    sell_quantity * sell_price + 1e-9 < minimum_trade_notional_cny
                    or sell_quantity < minimum_trade_shares
                )
                and not risk_off_override
            ):
                sell_quantity = 0
            if sell_quantity > 0:
                notional = sell_quantity * sell_price
                commission = _commission(notional, costs)
                stamp_duty = notional * costs.stamp_duty_rate
                cash += notional - commission - stamp_duty
                shares -= sell_quantity
                quantity = -sell_quantity
                side = "卖出"
                execution_price = sell_price
            else:
                quantity = 0

        if side != "无交易":
            trade_rows.append(
                {
                    "date": date,
                    "signal_date": signal_date,
                    "side": side,
                    "quantity": int(abs(quantity)),
                    "signed_quantity": int(quantity),
                    "open_price": float(row.open),
                    "execution_price": float(execution_price),
                    "commission": float(commission),
                    "stamp_duty": float(stamp_duty),
                    "target_position": target,
                    "t_plus_one_sellable_before_trade": int(shares_at_start),
                    "risk_off_override": risk_off_override,
                    "signal_reason": signal_reason,
                }
            )

        open_equity_after_trade = cash + shares * float(row.open) + dividend_receivable
        open_actual_position_after_trade = (
            float(shares * float(row.open) / open_equity_after_trade)
            if open_equity_after_trade
            else 0.0
        )
        daily_explicit_cost = float(commission + stamp_duty)
        daily_slippage_cost = (
            float(abs(float(execution_price) - float(row.open)) * abs(quantity))
            if side != "无交易"
            else 0.0
        )

        payment_today = float(scheduled_payments.pop(date, 0.0))
        if payment_today:
            cash += payment_today
            dividend_receivable -= payment_today
            if abs(dividend_receivable) < 1e-9:
                dividend_receivable = 0.0
        close_equity = cash + shares * float(row.close) + dividend_receivable
        ledger_rows.append(
            {
                "date": date,
                "signal_date_used": signal_date,
                "target_position": target,
                "shares": int(shares),
                "cash": float(cash),
                "dividend_receivable": float(dividend_receivable),
                "dividend_entitlement_today": float(entitlement_today),
                "dividend_payment_today": payment_today,
                "close": float(row.close),
                "equity": float(close_equity),
                "open_equity_before_trade": float(open_equity),
                "open_equity_after_trade": float(open_equity_after_trade),
                "open_actual_position_after_trade": open_actual_position_after_trade,
                "daily_explicit_cost_cny": daily_explicit_cost,
                "daily_slippage_cost_cny": daily_slippage_cost,
                "daily_total_execution_cost_cny": daily_explicit_cost
                + daily_slippage_cost,
                "actual_position": float(shares * float(row.close) / close_equity) if close_equity else 0.0,
                "trade_allowed": trade_allowed,
                "risk_off_override": risk_off_override,
                "signal_reason": signal_reason,
            }
        )
        previous_date = date

    ledger = pd.DataFrame(ledger_rows)
    ledger["daily_return"] = ledger["equity"].pct_change().fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    trades = pd.DataFrame(trade_rows)
    return ledger, trades


def summarize_backtest(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
) -> dict:
    if ledger.empty:
        raise ValueError("净值表为空")
    elapsed_days = max((ledger["date"].iloc[-1] - ledger["date"].iloc[0]).days, 1)
    total_return = float(ledger["equity"].iloc[-1] / initial_cash - 1.0)
    cagr = float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0) if total_return > -1 else -1.0
    volatility = float(ledger["daily_return"].std(ddof=1) * np.sqrt(242))
    mean_return = float(ledger["daily_return"].mean() * 242)
    sharpe = mean_return / volatility if volatility > 0 else None
    max_drawdown = float(ledger["drawdown"].min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else None
    if trades.empty:
        total_cost = 0.0
        turnover_notional = 0.0
    else:
        total_cost = float((trades["commission"] + trades["stamp_duty"]).sum())
        turnover_notional = float((trades["quantity"] * trades["execution_price"]).sum())
    average_equity = float(ledger["equity"].mean())
    return {
        "start_date": str(ledger["date"].iloc[0].date()),
        "end_date": str(ledger["date"].iloc[-1].date()),
        "observations": int(len(ledger)),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "average_exposure": float(ledger["actual_position"].mean()),
        "trade_count": int(len(trades)),
        "total_explicit_cost_cny": total_cost,
        "turnover_over_average_equity": turnover_notional / average_equity if average_equity else None,
        "ending_equity": float(ledger["equity"].iloc[-1]),
    }
