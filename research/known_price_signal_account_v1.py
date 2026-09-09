"""保留原现金成交及退出锁定，未知价格信号不授予再次入场资格。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import Account, affordable_quantity, execute_order, fill_price, require


def simulate_known_price_exit(data, dividends, config, cost, start, rule, spec, controller=None):
    """按实际成交状态维护持仓周期、退出待成交和再入场等待期。"""
    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])
    last, anchor = len(data) - 1, first - 1
    dates = pd.DatetimeIndex(data.date)
    op, cl = data.open.to_numpy(float), data.close.to_numpy(float)
    previous, distribution = data.previous_close.to_numpy(float), data.dividend.to_numpy(float)
    require(len(rule["entry"]) == len(data), "信号长度不符")
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
    cycle, last_exit, cycle_number = None, -1000000, 0
    pending_reasons, peak_value, cycle_dividends = [], np.nan, 0.
    entry_rearmed, last_entry_mode = True, 0

    def decide(t):
        nonlocal pending_reasons, entry_rearmed
        entry_value = float(rule["entry"][t])
        entry_known = np.isfinite(entry_value)
        require(entry_known or np.isnan(entry_value), "价格进入信号不能是无穷值")
        require(not entry_known or entry_value in (0., 1.), "本轮价格进入只允许零、一或未知")
        mode = int(entry_value) if entry_known else 0
        if entry_known and not account.shares and (mode == 0 or (last_entry_mode and mode != last_entry_mode)):
            entry_rearmed = True
        reasons, qty, selected_mode = [], 0, 0
        learned = {}
        if account.shares:
            selected_mode = cycle["mode"]
            s = spec["modes"].get(selected_mode, spec["modes"].get(str(selected_mode)))
            current_value = account.shares * cl[t] + cycle_dividends
            cycle_return = current_value / cycle["entry_cost_cny"] - 1
            if controller is not None:
                learned = controller(t, cycle, current_value, peak_value)
            if not pending_reasons:
                exit_value = float(rule["exit"][selected_mode][t])
                require(np.isnan(exit_value) or exit_value in (0., 1.), "价格退出只允许零、一或未知")
                if np.isfinite(exit_value) and exit_value == 1.:
                    reasons.append("本持仓模式的价格退出条件成立")
                if s["loss"] is not None and cycle_return <= -s["loss"]:
                    reasons.append("持仓含分红收益触及固定止损")
                if s["trail"] is not None and current_value / peak_value - 1 <= -s["trail"]:
                    reasons.append("含分红持仓价值从周期高点回落至追踪退出线")
                if s["take"] is not None and cycle_return >= s["take"]:
                    reasons.append("持仓含分红收益达到预定止盈")
                if s["days"] is not None and t - cycle["entry_index"] + 1 >= s["days"]:
                    reasons.append("预定最长持有交易日到期")
                if learned.get("learned_exit_requested", False):
                    reasons.append("连续两个收盘预测继续持有收益为负，学习条件请求退出")
                pending_reasons = reasons
            if pending_reasons:
                qty, reasons = -account.shares, pending_reasons
                action = "请求全部退出，受阻后逐日继续请求"
            else:
                action = "保持已有份额，收盘检查退出"
        elif np.isfinite(rule["exit"][1][t]) and rule["exit"][1][t] == 1.:
            action = "已知价格退出条件优先，空仓不请求新买入"
        elif not entry_known:
            action = "入场信息未知，保留资格状态且不请求新买入"
        elif mode and entry_rearmed and t - last_exit >= spec["cooldown"]:
            qty = affordable_quantity(account.cash, fill_price(cl[t], 1, cost, config["tick"]), cost, config["lot"])
            selected_mode, action = mode, "入场条件成立，下一开盘请求买入"
        elif mode and not entry_rearmed:
            action = "旧入场条件尚未消失，等待新的机会"
        elif t - last_exit < spec["cooldown"]:
            action = "退出后的等待期尚未结束"
        else:
            action = "空仓等待新的入场条件"
        row = {"origin": dates[t], "execution_date": dates[t + 1] if t + 1 < len(dates) else pd.NaT,
               "origin_index": t, "requested_quantity": int(qty), "entry_mode": selected_mode,
               "exit_reasons": "；".join(reasons), "action": action,
               "reference_weight": 0. if qty < 0 or (not account.shares and not qty) else 1.,
               "simulation_only": True}
        row["entry_rearmed"] = entry_rearmed
        row["entry_signal_known"] = bool(entry_known)
        row["entry_signal_value"] = entry_value
        row["exit_signal_value"] = float(rule["exit"][1][t])
        row["exit_signal_known"] = bool(np.isfinite(row["exit_signal_value"]))
        if qty == 0 and (not entry_known or (account.shares and not row["exit_signal_known"])):
            row["reference_weight"] = float("nan")
        row.update(learned)
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
            entry_rearmed, last_entry_mode = False, int(pending["entry_mode"])
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
            cycle, last_exit, pending_reasons, peak_value, cycle_dividends = None, day, [], np.nan, 0.
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
