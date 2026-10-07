"""510300 波动分层的固定价格代理实验，仅使用本地资料，不发送订单。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def session_of(clock: str) -> str | None:
    """采用全历史统一的连续时段，避开午休与尾盘竞价。"""
    if "09:30" <= clock <= "11:29":
        return "AM"
    if "13:01" <= clock <= "14:56":
        return "PM"
    return None


def build_features(bars: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """今日状态与阈值只用昨天及以前数据，返回状态、价格矩阵和阈值。"""
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["date"] = frame.timestamp.dt.strftime("%Y-%m-%d")
    frame["clock"] = frame.timestamp.dt.strftime("%H:%M")
    if frame.duplicated(["date", "clock"]).any():
        raise ValueError("分钟数据存在重复日期与时刻，停止计算。")
    if (frame.close <= 0).any() or frame.close.isna().any():
        raise ValueError("分钟价格缺失或非正，停止计算。")
    prices = frame.pivot(index="date", columns="clock", values="close").sort_index().sort_index(axis=1)
    if prices.isna().any().any():
        raise ValueError("各交易日分钟网格不一致，停止计算。")
    logp = np.log(prices)
    increments = []
    shock = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    clocks = list(prices.columns)
    locations = {clock: i for i, clock in enumerate(clocks)}
    for clock in clocks:
        session = session_of(clock)
        if session is None:
            continue
        at = pd.Timestamp("2000-01-01 " + clock)
        previous = (at - pd.Timedelta(minutes=1)).strftime("%H:%M")
        if previous in locations and session_of(previous) == session:
            increments.append((logp[clock] - logp[previous]) ** 2)
        earlier = (at - pd.Timedelta(minutes=config["signals"]["shock_minutes"])).strftime("%H:%M")
        if earlier in locations and session_of(earlier) == session:
            shock[clock] = logp[clock] - logp[earlier]
    variance = pd.concat(increments, axis=1).sum(axis=1)
    window = config["volatility"]["recent_days"]
    reference = config["volatility"]["reference_days"]
    rv = np.sqrt(variance.rolling(window, min_periods=window).mean().shift(1) * config["annual_trading_days"])
    ranks = []
    for i, current in enumerate(rv):
        past = rv.iloc[max(0, i - reference):i]
        ranks.append(float((past <= current).mean()) if len(past) == reference and past.notna().all() and pd.notna(current) else np.nan)
    states = pd.DataFrame({"date": prices.index, "prior_rv20_annualized": rv.to_numpy(), "prior_rank": ranks})
    states["regime"] = "WARMUP_NO_VIEW"
    v = config["volatility"]
    for i, row in states.iterrows():
        rank = row.prior_rank
        if pd.isna(rank):
            continue
        states.at[i, "regime"] = "LOW" if rank <= v["low_rank_at_most"] else "MEDIUM" if rank <= v["medium_rank_at_most"] else "HIGH" if rank <= v["high_rank_at_most"] else "EXTREME"
    states["last_feature_date"] = states.date.shift(1)
    states["rv_input_days"] = window
    states["rank_reference_days"] = reference
    sigma = shock.rolling(config["signals"]["same_clock_sigma_days"], min_periods=config["signals"]["same_clock_sigma_days"]).std(ddof=1).shift(1)
    threshold = (sigma * config["signals"]["sigma_multiple"]).clip(lower=config["signals"]["minimum_shock_bps"] / 10000.0)
    return states, prices, threshold


def detect_episodes(states: pd.DataFrame, prices: pd.DataFrame, threshold: pd.DataFrame, config: dict) -> pd.DataFrame:
    """保留每次合并后的正负冲击，确认失败与时段截断也进入事件表。"""
    records = []
    signal = config["signals"]
    clocks = list(prices.columns)
    clock_set = set(clocks)
    state_map = states.set_index("date")
    for date, row in prices.iterrows():
        state = state_map.loc[date]
        if state.regime == "WARMUP_NO_VIEW":
            continue
        for direction in ("RECOVERY", "CONTINUATION"):
            armed, quiet, previous_session = True, 0, None
            for clock in clocks:
                session = session_of(clock)
                if session is None:
                    continue
                if session != previous_session:
                    armed, quiet, previous_session = True, 0, session
                at = pd.Timestamp(date + " " + clock)
                earlier = (at - pd.Timedelta(minutes=signal["shock_minutes"])).strftime("%H:%M")
                boundary = threshold.at[date, clock]
                if earlier not in clock_set or session_of(earlier) != session or pd.isna(boundary):
                    continue
                shock_return = math.log(row[clock] / row[earlier])
                triggered = shock_return <= -boundary if direction == "RECOVERY" else shock_return >= boundary
                if not triggered:
                    quiet += 1
                    if quiet >= signal["reset_quiet_minutes"]:
                        armed = True
                    continue
                quiet = 0
                if not armed:
                    continue
                armed = False
                observe_clock = (at + pd.Timedelta(minutes=signal["observation_minutes"])).strftime("%H:%M")
                entry_clock = (at + pd.Timedelta(minutes=signal["observation_minutes"] + 1)).strftime("%H:%M")
                observable = observe_clock in clock_set and session_of(observe_clock) == session
                entry_exists = entry_clock in clock_set and session_of(entry_clock) == session
                observation_return = math.log(row[observe_clock] / row[clock]) if observable else np.nan
                confirmed = bool(observable and row[observe_clock] > row[clock] and (direction == "CONTINUATION" or row[observe_clock] < row[earlier]))
                status = "CONFIRMED" if confirmed and entry_exists else "UNCONFIRMED" if observable and entry_exists else "SESSION_ENDPOINT_UNAVAILABLE"
                records.append({
                    "event_id": f"{date}_{clock.replace(':', '')}_{direction}", "date": date,
                    "direction": direction, "regime": state.regime, "prior_rank": state.prior_rank,
                    "last_feature_date": state.last_feature_date, "shock_clock": clock,
                    "observation_end_clock": observe_clock, "entry_clock": entry_clock,
                    "threshold_bps": boundary * 10000, "shock_return_bps": shock_return * 10000,
                    "observation_return_bps": observation_return * 10000, "status": status,
                    "entry_raw_proxy": row[entry_clock] if entry_exists else np.nan,
                    "actual_fill": False,
                })
    columns = ["event_id", "date", "direction", "regime", "prior_rank", "last_feature_date", "shock_clock", "observation_end_clock", "entry_clock", "threshold_bps", "shock_return_bps", "observation_return_bps", "status", "entry_raw_proxy", "actual_fill"]
    return pd.DataFrame(records, columns=columns).sort_values(["date", "entry_clock", "direction"]).reset_index(drop=True)


def commission(value: float, cost: dict) -> float:
    return max(cost["minimum_commission_cny"], value * cost["commission_rate"]) if value > 0 else 0.0


def affordable_quantity(cash: float, price: float, cost: dict, lot: int) -> int:
    quantity = math.floor(cash / (price * lot)) * lot
    while quantity > 0 and quantity * price + commission(quantity * price, cost) > cash + 1e-9:
        quantity -= lot
    return max(0, quantity)


def simulate_account(states: pd.DataFrame, prices: pd.DataFrame, episodes: pd.DataFrame, dividends: pd.DataFrame, config: dict, policy: str, initial_cash: float, cost_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全交易日价格代理账本。应收分红与现金分开，日内不卖出新买份额。"""
    eligible = states.loc[states.regime != "WARMUP_NO_VIEW"].copy()
    cost = config["cost_scenarios"][cost_name]
    friction = cost["friction_bps_each_side"] / 10000
    dates = list(eligible.date)
    state_map = eligible.set_index("date")
    choices = {date: group for date, group in episodes.loc[episodes.status == "CONFIRMED"].groupby("date", sort=False)}
    distributions = dividends.to_dict("records")
    cash, position, pending = float(initial_cash), None, []
    daily_records, trade_records = [], []
    previous_equity = float(initial_cash)
    for i, date in enumerate(dates):
        state = state_map.loc[date]
        paid = 0.0
        for item in pending:
            if not item["paid"] and item["payment_date"] <= date:
                cash += item["amount"]
                paid += item["amount"]
                item["paid"] = True
        cash_before_orders = cash
        sold, bought, day_fees, day_friction = 0, 0, 0.0, 0.0
        if position is not None:
            raw_exit = float(prices.at[date, "09:36"])
            exit_price = raw_exit * (1 - friction)
            exit_value = position["quantity"] * exit_price
            exit_fee = commission(exit_value, cost)
            cash += exit_value - exit_fee
            gross = position["quantity"] * (raw_exit - position["entry_raw_proxy"]) + position["dividend_entitlement"]
            slippage = position["quantity"] * (position["entry_price_proxy"] - position["entry_raw_proxy"] + raw_exit - exit_price)
            net = gross - slippage - position["entry_fee"] - exit_fee
            trade_records.append({**position, "exit_date": date, "exit_clock": "09:36", "exit_raw_proxy": raw_exit,
                                  "exit_price_proxy": exit_price, "exit_fee": exit_fee,
                                  "gross_pnl_cny": gross, "friction_cny": slippage,
                                  "total_fee_cny": position["entry_fee"] + exit_fee, "net_pnl_cny": net,
                                  "net_trade_return_bps": net / position["entry_debit"] * 10000,
                                  "actual_fill": False})
            sold = position["quantity"]
            day_fees += exit_fee
            day_friction += position["quantity"] * (raw_exit - exit_price)
            position = None
        mode = config["policies"]["SWITCH"][state.regime] if policy == "SWITCH" else "RECOVERY" if policy == "FIXED_RECOVERY" else "CONTINUATION" if policy == "FIXED_CONTINUATION" else "CASH"
        reason = "WAIT_REGIME" if mode == "CASH" else "NO_CONFIRMED_EVENT"
        selected_id = None
        if i == len(dates) - 1:
            reason = "TERMINAL_DAY_NO_NEW_ENTRY"
        elif mode != "CASH" and date in choices:
            candidates = choices[date]
            candidates = candidates.loc[(candidates.direction == mode) & (candidates.entry_clock >= config["hypothetical_account"]["entry_not_before"])].sort_values(["entry_clock", "event_id"])
            if not candidates.empty:
                candidate = candidates.iloc[0]
                entry_price = float(candidate.entry_raw_proxy) * (1 + friction)
                quantity = affordable_quantity(cash, entry_price, cost, config["hypothetical_account"]["lot_size"])
                if quantity > 0:
                    entry_fee = commission(quantity * entry_price, cost)
                    entry_debit = quantity * entry_price + entry_fee
                    cash -= entry_debit
                    position = {"event_id": candidate.event_id, "entry_date": date, "entry_clock": candidate.entry_clock,
                                "direction": mode, "regime": state.regime, "quantity": quantity,
                                "entry_raw_proxy": float(candidate.entry_raw_proxy), "entry_price_proxy": entry_price,
                                "entry_fee": entry_fee, "entry_debit": entry_debit, "dividend_entitlement": 0.0}
                    bought = quantity
                    selected_id = candidate.event_id
                    day_fees += entry_fee
                    day_friction += quantity * (entry_price - float(candidate.entry_raw_proxy))
                    reason = "HYPOTHETICAL_ENTRY"
                else:
                    reason = "INSUFFICIENT_CASH_FOR_ONE_LOT"
        if position is not None:
            for dividend in distributions:
                if str(dividend["record_date"]) == date:
                    amount = position["quantity"] * float(dividend["cash_dividend_per_share"])
                    pending.append({"ex_date": str(dividend["ex_date"]), "payment_date": str(dividend["payment_date"]), "amount": amount, "paid": False})
                    position["dividend_entitlement"] += amount
        receivable = sum(item["amount"] for item in pending if not item["paid"] and item["ex_date"] <= date)
        market_value = position["quantity"] * float(prices.at[date, "15:00"]) if position else 0.0
        equity = cash + receivable + market_value
        if cash < -1e-7 or equity <= 0:
            raise ValueError("账户现金为负或净值非正，停止计算。")
        daily_records.append({"date": date, "regime": state.regime, "policy": policy, "cost": cost_name,
                              "initial_cash_cny": initial_cash, "cash_before_orders_cny": cash_before_orders,
                              "cash_cny": cash, "receivable_cny": receivable, "market_value_cny": market_value,
                              "equity_cny": equity, "daily_return": equity / previous_equity - 1,
                              "bought_quantity": bought, "sold_quantity": sold,
                              "end_quantity": position["quantity"] if position else 0,
                              "no_order_cash_day": bought == 0 and sold == 0,
                              "entry_event_id": selected_id, "reason": reason,
                              "commission_cny": day_fees, "friction_cny": day_friction,
                              "dividend_paid_cny": paid, "actual_orders": False})
        previous_equity = equity
    if position is not None:
        raise ValueError("区间末尾存在未退出仓位。")
    return pd.DataFrame(daily_records), pd.DataFrame(trade_records)


def metrics(daily: pd.DataFrame, trades: pd.DataFrame, annual_days: int, initial: float | None = None) -> dict:
    """每个现金日均进入分母；包括首日相对于期初资金的收益和回撤。"""
    returns = daily.daily_return.to_numpy(dtype=float)
    growth = np.cumprod(1 + returns)
    running = np.maximum.accumulate(np.r_[1.0, growth])[1:]
    standard = returns.std(ddof=1) if len(returns) > 1 else 0.0
    net = trades.net_pnl_cny.to_numpy(dtype=float) if not trades.empty else np.array([])
    wins, losses = net[net > 0], net[net < 0]
    return {
        "days": len(daily), "start": str(daily.date.iloc[0]), "end": str(daily.date.iloc[-1]),
        "total_return": float(growth[-1] - 1), "cagr": float(growth[-1] ** (annual_days / len(daily)) - 1),
        "sharpe": float(returns.mean() / standard * math.sqrt(annual_days)) if standard > 0 else None,
        "maximum_drawdown": float(np.min(growth / running - 1)), "trade_count": len(net),
        "entry_days": int((daily.bought_quantity > 0).sum()), "no_entry_days": int((daily.bought_quantity == 0).sum()),
        "no_order_cash_days": int(daily.no_order_cash_day.sum()),
        "close_position_days": int((daily.end_quantity > 0).sum()),
        "win_rate": float(len(wins) / len(net)) if len(net) else None,
        "mean_net_pnl_cny": float(net.mean()) if len(net) else None,
        "mean_net_trade_bps": float(trades.net_trade_return_bps.mean()) if len(net) else None,
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else None,
        "gross_pnl_cny": float(trades.gross_pnl_cny.sum()) if len(net) else 0.0,
        "friction_cny": float(daily.friction_cny.sum()), "commission_cny": float(daily.commission_cny.sum()),
        "ending_equity_cny": float(daily.equity_cny.iloc[-1]),
        "ending_receivable_cny": float(daily.receivable_cny.iloc[-1]),
        "unconditional_daily_mean_bps": float(returns.mean() * 10000),
        "initial_cash_cny": initial,
    }


def block_mean_interval(values: np.ndarray, block: int, draws: int, seed: int) -> dict:
    """固定20日循环分块的描述性区间，不宣称独立样本或经过选择偏差校正。"""
    rng = np.random.default_rng(seed)
    n = len(values)
    count = math.ceil(n / block)
    means = np.empty(draws)
    offsets = np.arange(block)
    for i in range(draws):
        starts = rng.integers(0, n, size=count)
        indices = ((starts[:, None] + offsets) % n).ravel()[:n]
        means[i] = values[indices].mean() * 10000
    return {"mean_daily_bps": float(values.mean() * 10000), "lower_95_daily_bps": float(np.quantile(means, .025)),
            "upper_95_daily_bps": float(np.quantile(means, .975)), "block_days": block, "draws": draws,
            "independent_validation": False}


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
