"""联合状态决策的完整账户：复用分红、真实成交和逐日财富循环。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import Account, affordable_quantity, execute_order, fill_price, require


def simulate_joint_policy(data, dividends, config, cost, start, controller):
    """空仓和持仓均调用策略；已锁定卖出持续，下一开盘执行。"""
    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])
    last, anchor = len(data) - 1, first - 1
    dates = pd.DatetimeIndex(data.date)
    op, cl = data.open.to_numpy(float), data.close.to_numpy(float)
    previous, distribution = data.previous_close.to_numpy(float), data.dividend.to_numpy(float)
    require(first > 0 and dates.is_monotonic_increasing and not dates.has_duplicates, "账户需要完整递增日历和收盘起点")
    record_events, ex_events, pay_events = {}, {}, {}
    for key, event in enumerate(dividends.itertuples()):
        for field, mapping in [("record_date", record_events), ("ex_date", ex_events)]:
            when = getattr(event, field)
            if when in dates:
                mapping.setdefault(dates.get_loc(when), []).append((key, event.cash_dividend_per_share))
        idx = dates.searchsorted(event.payment_date)
        if idx < len(dates):
            pay_events.setdefault(idx, []).append((key, event.payment_date == dates[idx]))
    account = Account(config["initial_capital"])
    previous_nav, previous_mark = config["initial_capital"], cl[anchor]
    records, decisions, cycles = [], [], []
    cycle, cycle_number = None, 0
    pending_reasons, peak_value, cycle_dividends = [], np.nan, 0.

    def decide(t):
        nonlocal pending_reasons
        mode = 0 if not account.shares else (2 if pending_reasons else 1)
        learned = controller(t, mode)
        desired, qty, reasons = learned.get("policy_action"), 0, []
        if account.shares and (pending_reasons or desired == 0):
            if not pending_reasons:
                pending_reasons = ["联合状态策略选择现金，请求全部退出"]
            qty, reasons = -account.shares, pending_reasons
            action = "请求全部退出，受阻后逐日继续请求"
        elif not account.shares and desired == 1:
            qty = affordable_quantity(account.cash, fill_price(cl[t], 1, cost, config["tick"]), cost, config["lot"])
            action = "联合状态策略选择持仓，下一开盘请求买入"
        elif desired is None:
            action = "缺失模型或市场状态，不发起新交易并保留实际份额"
        elif account.shares:
            action = "保持实际已有份额，不加仓再平衡"
        else:
            action = "联合状态策略选择现金，空仓等待"
        row = {"origin": dates[t], "execution_date": dates[t+1], "origin_index": t,
               "requested_quantity": int(qty), "entry_mode": 1 if qty > 0 or account.shares else 0,
               "exit_reasons": "；".join(reasons), "action": action,
               "reference_weight": 0. if qty < 0 or (not account.shares and not qty) else 1.,
               "simulation_only": True, **learned}
        decisions.append(row)
        return row

    pending = decide(anchor)
    for day in range(first, last + 1):
        old_shares, recognized, paid = account.shares, 0., 0.
        for key, amount in ex_events.get(day, []):
            value = account.entitlements.get(key, 0) * amount
            account.receivables[key] = value
            recognized += value
        if old_shares:
            cycle_dividends += recognized
        for key, same in pay_events.get(day, []):
            if not same:
                value = account.receivables.pop(key, 0.)
                paid += value
                account.cash += value
        terminal = day == last
        quantity = -account.shares if terminal else int(pending["requested_quantity"])
        before = account.value(op[day])
        execution = execute_order(account, quantity, op[day], previous[day], distribution[day], day, cost, config)
        mark = op[day] if terminal else cl[day]
        if old_shares == 0 and account.shares > 0:
            cycle_number += 1
            cycle_dividends, pending_reasons = 0., []
            cycle = {"cycle_id": cycle_number, "mode": int(pending["entry_mode"]), "entry_date": dates[day], "entry_index": day,
                     "entry_origin": pending["origin"], "entry_quantity": account.shares,
                     "entry_cost_cny": execution["notional"] + execution["commission"]}
            peak_value = cycle["entry_cost_cny"]
        elif old_shares > 0 and account.shares == 0:
            proceeds = execution["notional"] - execution["commission"]
            cycle.update({"exit_date": dates[day], "exit_origin": pending["origin"] if not terminal else dates[day],
                          "exit_reasons": pending["exit_reasons"] if not terminal else "研究终点统一开盘退出",
                          "holding_intervals": day - cycle["entry_index"], "dividend_cny": cycle_dividends,
                          "net_profit_cny": proceeds + cycle_dividends - cycle["entry_cost_cny"]})
            cycles.append(cycle)
            cycle, pending_reasons, peak_value, cycle_dividends = None, [], np.nan, 0.
        elif account.shares and execution["filled_quantity"]:
            raise ValueError("本轮仅单次买入及全部退出，出现非预定份额变化")
        if not terminal:
            for key, same in pay_events.get(day, []):
                if same:
                    value = account.receivables.pop(key, 0.)
                    paid += value
                    account.cash += value
            for key, amount in record_events.get(day, []):
                account.entitlements[key] = account.shares
        if account.shares:
            peak_value = max(peak_value, account.shares * mark + cycle_dividends)
        nav = account.value(mark)
        price_pnl = old_shares * (op[day] - previous_mark) + account.shares * (mark - op[day])
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, "价格策略账户财富恒等式不成立")
        account.assert_valid()
        records.append({"date": dates[day], "open": op[day], "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(), "equity": nav,
                        "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav, "price_pnl": price_pnl,
                        "dividend_recognized": recognized, "dividend_paid": paid, "exposure": account.shares * mark / nav,
                        "accounting_error": error, "origin": dates[day - 1], "terminal_unliquidated": bool(terminal and account.shares),
                        "turnover": execution["notional"] / before, "cycle_id": cycle["cycle_id"] if cycle else None,
                        "mode": cycle["mode"] if cycle else 0, "execution_reasons": pending["exit_reasons"] if not terminal else "研究终点统一退出",
                        **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decide(day)
    if cycle:
        cycles.append({**cycle, "exit_date": None, "exit_reasons": "研究终点退出未成交"})
    return pd.DataFrame(records), pd.DataFrame(decisions), pd.DataFrame(cycles)
