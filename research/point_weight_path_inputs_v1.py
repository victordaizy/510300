"""分解已保存账户的实际持仓路径；收盘清算标记只作事后解释。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.point_account_nr7_inputs_v1 import fill, fee


def require(ok, message):
    if not ok:
        raise ValueError(message)


def cycle_path(prices, dividends, orders, trade, cost, last_date):
    """只按真实订单重建现金流、库存、登记权益与逐日损益，不生成交易指令。"""
    p = prices.set_index("date").sort_index()
    require(p.index.is_unique, "价格日期重复。")
    entry = pd.Timestamp(trade["entry_date"])
    complete = trade["status"] == "COMPLETE"
    exit_date = pd.Timestamp(trade["exit_date"]) if complete else pd.Timestamp(last_date)
    order = orders.loc[orders.cycle_id.eq(trade["cycle_id"])].copy()
    require(len(order) > 0 and order.date.min() == entry, "周期没有匹配的首次订单。")
    require(order.date.between(entry, exit_date).all(), "周期订单超出实际持有边界。")
    require(order.side.isin(["BUY", "SELL"]).all(), "存在未知成交方向。")
    require(order.quantity.gt(0).all(), "实际成交数量必须为正。")
    require(order.origin.lt(order.date).all(), "实际订单不是事前决定。")
    order["signed_quantity"] = np.where(order.side.eq("BUY"), order.quantity, -order.quantity)
    order["cash_flow"] = -order.signed_quantity * order.fill_price - order.commission
    require(order.date.isin(p.index).all(), "价格缺少实际成交日期。")
    inventory = order.groupby("date").signed_quantity.sum().cumsum()
    require(inventory.ge(0).all(), "周期出现负库存。")
    if complete:
        require(inventory.iloc[-1] == 0, "完整周期没有自然清仓。")
    entitlements = []
    for event in dividends.itertuples():
        if event.record_date < entry or event.record_date > exit_date:
            continue
        eligible = inventory.loc[inventory.index <= event.record_date]
        quantity = int(eligible.iloc[-1]) if len(eligible) else 0
        if quantity and event.ex_date <= pd.Timestamp(last_date):
            require(event.ex_date >= event.record_date, "分红除息先于登记。")
            entitlements.append((pd.Timestamp(event.ex_date), quantity * float(event.cash_dividend_per_share)))
    recognition_end = max([exit_date] + [date for date, _ in entitlements])
    require(recognition_end <= pd.Timestamp(last_date), "读取了样本末端后的权益。")
    local = p.loc[entry:recognition_end, ["open", "close"]].copy()
    require(len(local) > 0 and entry in local.index and recognition_end in local.index, "持有路径日期不完整。")
    previous_close = p.close.shift().reindex(local.index)
    require(previous_close.notna().all(), "缺少路径前一交易日收盘。")
    grouped = order.groupby("date")[["signed_quantity", "cash_flow", "commission", "slippage"]].sum()
    cash_flow = grouped.cash_flow.reindex(local.index, fill_value=0.)
    local["shares"] = grouped.signed_quantity.reindex(local.index, fill_value=0).cumsum()
    local["shares_before"] = local.shares.shift(fill_value=0)
    local["commission"] = grouped.commission.reindex(local.index, fill_value=0.)
    local["slippage"] = grouped.slippage.reindex(local.index, fill_value=0.)
    accrual = pd.Series(0., index=local.index)
    for date, amount in entitlements:
        require(date in accrual.index, "除息日缺少交易日记录。")
        accrual.loc[date] += amount
    local["dividend_accrual"] = accrual
    local["overnight_price_pnl"] = local.shares_before * (local.open - previous_close)
    local["intraday_price_pnl"] = local.shares * (local.close - local.open)
    local["net_increment"] = local.overnight_price_pnl + local.intraday_price_pnl + accrual - local.commission - local.slippage
    local["net_mark"] = cash_flow.cumsum() + accrual.cumsum() + local.shares * local.close
    require(np.allclose(local.net_increment.cumsum(), local.net_mark, atol=1e-6, rtol=0), "周期价格与现金流分解不一致。")
    reserve = []
    for row in local.itertuples():
        if row.shares:
            px = fill(float(row.close), -1, cost)
            reserve.append(row.shares * (row.close - px) + fee(row.shares * px, cost))
        else:
            reserve.append(0.)
    local["hypothetical_close_exit_cost"] = reserve
    local["hypothetical_close_liquidation_pnl"] = local.net_mark - local.hypothetical_close_exit_cost
    local["cycle_id"] = int(trade["cycle_id"])
    local["status"] = trade["status"]
    local["phase"] = np.where(local.index == entry, "ENTRY_DAY", np.where(local.index < exit_date, "HOLDING_INTERIOR", np.where(complete & (local.index == exit_date), "EXIT_DAY", "POST_EXIT_DIVIDEND")))
    if not complete:
        local.loc[local.index > entry, "phase"] = "HOLDING_INTERIOR"
    require(abs(float(accrual.sum()) - float(trade["dividend_cny"])) < 1e-6, "周期登记权益与保存结果不一致。")
    if complete:
        require(abs(float(local.net_mark.iloc[-1]) - float(trade["net_pnl"])) < 1e-6, "周期最终净损益无法复算。")
    return local.reset_index()


def describe_cycle(path, trade, decisions):
    complete = trade["status"] == "COMPLETE"
    held = path.loc[path.shares.gt(0)]
    peak = held.hypothetical_close_liquidation_pnl.max() if len(held) else np.nan
    trough = held.hypothetical_close_liquidation_pnl.min() if len(held) else np.nan
    net = float(trade["net_pnl"]) if complete else np.nan
    category = "UNFINISHED"
    if complete:
        category = "WIN" if net > 0 else "FLAT" if net == 0 else "LOSS_AFTER_POSITIVE_CLOSE_MARK" if peak > 0 else "LOSS_WITHOUT_POSITIVE_CLOSE_MARK"
    row = {"cycle_id": int(trade["cycle_id"]), "entry_date": trade["entry_date"], "exit_date": trade["exit_date"],
           "status": trade["status"], "net_pnl": net, "net_return": trade["net_return"],
           "entry_equity": float(trade["entry_equity"]), "buy_debit": float(trade["buy_debit"]),
           "held_closes": int(len(held)), "peak_close_liquidation_pnl": float(peak),
           "trough_close_liquidation_pnl": float(trough), "outcome_path": category,
           "peak_to_exit_difference": float(max(0., peak - net)) if complete and np.isfinite(peak) else np.nan,
           "terminal_net_mark": float(path.net_mark.iloc[-1]),
           "exit_reason": trade.get("exit_reason", "UNFINISHED"),
           "final_open_vs_prior_close_mark": np.nan, "exit_intent_to_execution_sessions": np.nan,
           "first_exit_intent_origin": pd.NaT, "first_exit_intent_to_actual_pnl": np.nan}
    if len(held):
        row["peak_close_date"] = held.loc[held.hypothetical_close_liquidation_pnl.idxmax(), "date"]
        row["entry_close_liquidation_pnl"] = float(held.hypothetical_close_liquidation_pnl.iloc[0])
    for phase in ("ENTRY_DAY", "HOLDING_INTERIOR", "EXIT_DAY", "POST_EXIT_DIVIDEND"):
        row[phase.lower() + "_net_pnl"] = float(path.loc[path.phase.eq(phase), "net_increment"].sum())
    for column in ("overnight_price_pnl", "intraday_price_pnl", "dividend_accrual", "commission", "slippage"):
        row[column] = float(path[column].sum())
    if complete:
        before = path.loc[path.date.lt(trade["exit_date"])]
        if len(before):
            row["final_open_vs_prior_close_mark"] = net - float(before.hypothetical_close_liquidation_pnl.iloc[-1])
        intents = decisions.loc[decisions.origin.ge(trade["entry_date"]) & decisions.origin.lt(trade["exit_date"]) & decisions.shares_before.gt(0) & decisions.desired_shares.eq(0)]
        require(len(intents) > 0, "完整清仓缺少事前退出意图。")
        first = intents.sort_values("origin").iloc[0]
        origin = first.origin
        mark = path.loc[path.date.eq(origin), "hypothetical_close_liquidation_pnl"]
        require(len(mark) == 1, "首次退出决定不在已观察路径内。")
        row["first_exit_intent_origin"] = origin
        row["first_exit_intent_to_actual_pnl"] = net - float(mark.iloc[0])
        row["exit_intent_to_execution_sessions"] = int(path.date.gt(origin).where(path.date.le(trade["exit_date"]), False).sum())
    return row


def describe_account(prices, dividends, account, cost):
    paths, cycles = [], []
    daily = account["daily"]
    for trade in account["trades"].to_dict("records"):
        path = cycle_path(prices, dividends, account["orders"], trade, cost, daily.date.iloc[-1])
        paths.append(path)
        cycles.append(describe_cycle(path, trade, account["decisions"]))
    require(len(paths) > 0, "账户无交易周期，不能进行持有路径归因。")
    paths = pd.concat(paths, ignore_index=True)
    components = ["shares", "net_increment", "overnight_price_pnl", "intraday_price_pnl", "dividend_accrual", "commission", "slippage"]
    by_day = paths.groupby("date")[components].sum().reindex(pd.DatetimeIndex(daily.date), fill_value=0).reset_index()
    require(np.array_equal(by_day.shares.to_numpy(), daily.shares.to_numpy()), "周期库存之和不能还原账户库存。")
    for field in ("dividend_accrual", "commission", "slippage"):
        require(np.allclose(by_day[field], daily[field], atol=1e-6, rtol=0), "周期不能还原账户：" + field)
    require(np.allclose(by_day.overnight_price_pnl + by_day.intraday_price_pnl, daily.price_pnl, atol=1e-6, rtol=0), "两类价格损益不能还原账户。")
    expected = daily.equity.diff()
    expected.iloc[0] = daily.equity.iloc[0] - 200000.
    require(np.allclose(by_day.net_increment, expected, atol=1e-6, rtol=0), "全账户权益变动无法逐日还原。")
    by_day["saved_equity"] = daily.equity.to_numpy()
    by_day["reconstructed_equity"] = 200000. + by_day.net_increment.cumsum()
    return pd.DataFrame(cycles), paths, by_day
