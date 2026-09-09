"""学习信号只减仓一次，剩余份额按原自然退出，保存完整实际账户。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import Account, affordable_quantity, execute_order, fill_price, require


def simulate_partial_exit(data, dividends, config, cost, start, rule, spec, controller=None, learned_sell_fraction=.5):
    """风险账按剩余份额分摊，实际登记权益不随后续卖出而减少。"""
    require(0 < learned_sell_fraction <= 1, "预定减仓比例必须在零至一之间")
    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])
    last, anchor = len(data) - 1, first - 1
    dates = pd.DatetimeIndex(data.date)
    op, cl = data.open.to_numpy(float), data.close.to_numpy(float)
    previous, distribution = data.previous_close.to_numpy(float), data.dividend.to_numpy(float)
    require(len(rule["entry"]) == len(data), "部分退出信号长度不符")
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
    full_exit_reasons, reduction_target = [], None
    risk_cost, risk_peak, risk_dividends = np.nan, np.nan, 0.
    risk_entitlements, dividend_owners = {}, {}
    entry_rearmed, last_entry_mode = True, 0

    def decide(t):
        nonlocal full_exit_reasons, reduction_target, entry_rearmed
        mode = int(rule["entry"][t])
        if not account.shares and (mode == 0 or (last_entry_mode and mode != last_entry_mode)):
            entry_rearmed = True
        qty, selected_mode, action_kind = 0, 0, "WAIT"
        reasons, learned = [], {}
        current_risk_value = np.nan
        if account.shares:
            selected_mode = cycle["mode"]
            s = spec["modes"].get(selected_mode, spec["modes"].get(str(selected_mode)))
            current_risk_value = account.shares * cl[t] + risk_dividends
            risk_return = current_risk_value / risk_cost - 1
            if not full_exit_reasons:
                if rule["exit"][selected_mode][t]:
                    reasons.append("原持仓模式的价格退出条件成立")
                if s["loss"] is not None and risk_return <= -s["loss"]:
                    reasons.append("剩余持仓含分红收益触及原固定止损")
                if s["trail"] is not None and current_risk_value / risk_peak - 1 <= -s["trail"]:
                    reasons.append("剩余持仓价值从同比例高点回落至原追踪退出线")
                if s["take"] is not None and risk_return >= s["take"]:
                    reasons.append("剩余持仓含分红收益达到原止盈")
                if s["days"] is not None and t - cycle["entry_index"] + 1 >= s["days"]:
                    reasons.append("从原实际买入日起的最长持有时间到期")
                full_exit_reasons = reasons
            if not full_exit_reasons and not cycle["learning_reduction_requested"] and controller is not None:
                # 学习只在第一次减仓请求之前调用，此时仍是完整的原买入份额。
                require(account.shares == cycle["entry_quantity"], "学习减仓前份额已经改变")
                learned = controller(t, cycle, current_risk_value, risk_peak)
                if learned.get("learned_exit_requested", False):
                    cycle["learning_reduction_requested"] = True
                    cycle["reduction_origin"] = dates[t]
                    sell = int(np.floor(account.shares * learned_sell_fraction / config["lot"])) * config["lot"]
                    if sell == 0:
                        sell = account.shares
                        full_exit_reasons = ["原持仓仅一手，预定减半不足一手，学习信号全部退出"]
                    elif sell == account.shares:
                        full_exit_reasons = ["学习信号按验证用全量比例请求全部退出"]
                    else:
                        reduction_target = account.shares - sell
                        cycle["reduction_target_remaining_shares"] = reduction_target
            if full_exit_reasons:
                qty, reasons, action_kind = -account.shares, full_exit_reasons, "FULL_EXIT"
                action = "请求全部退出，受阻后继续保留"
            elif reduction_target is not None and account.shares > reduction_target:
                qty = reduction_target - account.shares
                reasons, action_kind = ["首次连续两个学习预测为负，只执行预定一次减仓"], "LEARNED_REDUCTION"
                action = "请求减至已锁定剩余份额，受阻后只补足未卖数量"
            else:
                action_kind = "HOLD_REMAINDER" if cycle["learning_reduction_requested"] else "HOLD_FULL"
                action = "保持剩余份额，只按原自然退出规则检查" if cycle["learning_reduction_requested"] else "保持原份额，检查原保护及学习条件"
        elif mode and entry_rearmed and t - last_exit >= spec["cooldown"]:
            qty = affordable_quantity(account.cash, fill_price(cl[t], 1, cost, config["tick"]), cost, config["lot"])
            selected_mode, action_kind, action = mode, "ENTRY", "原进入条件成立，下一开盘请求一次买入"
        elif mode and not entry_rearmed:
            action = "旧进入条件尚未消失，等待新的机会"
        elif t - last_exit < spec["cooldown"]:
            action = "最终全部卖出后的等待期尚未结束"
        else:
            action = "空仓等待原进入条件"
        row = {"origin": dates[t], "execution_date": dates[t + 1] if t + 1 < len(dates) else pd.NaT,
               "origin_index": t, "requested_quantity": int(qty), "entry_mode": selected_mode,
               "exit_reasons": "；".join(reasons), "action": action, "action_kind": action_kind,
               "requested_remaining_shares": account.shares + qty if qty <= 0 else qty,
               "remaining_risk_cost": risk_cost, "remaining_risk_dividends": risk_dividends,
               "remaining_risk_value": current_risk_value, "remaining_risk_peak": risk_peak,
               "reduction_target_remaining_shares": reduction_target, "entry_rearmed": entry_rearmed,
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
            owner = dividend_owners.get(key)
            if owner is not None:
                owner["dividend_cny"] += value
                if owner is cycle:
                    risk_dividends += risk_entitlements.get(key, 0.) * amount
        for key, same in pay_events.get(day, []):
            if not same:
                value = account.receivables.pop(key, 0.)
                paid += value
                account.cash += value
        terminal = day == last
        quantity = -account.shares if terminal else int(pending["requested_quantity"])
        before = account.value(op[day])
        execution_cycle_id = cycle["cycle_id"] if cycle else None
        execution = execute_order(account, quantity, op[day], previous[day], distribution[day], day, cost, config)
        mark = op[day] if terminal else cl[day]
        if old_shares == 0 and account.shares > 0:
            cycle_number += 1
            entry_rearmed, last_entry_mode = False, int(pending["entry_mode"])
            full_exit_reasons, reduction_target, risk_dividends, risk_entitlements = [], None, 0., {}
            cycle = {"cycle_id": cycle_number, "mode": int(pending["entry_mode"]), "entry_date": dates[day], "entry_index": day,
                     "entry_origin": pending["origin"], "entry_quantity": account.shares,
                     "entry_cost_cny": execution["notional"] + execution["commission"], "sale_proceeds_cny": 0., "dividend_cny": 0.,
                     "learning_reduction_requested": False, "reduction_origin": None, "reduction_target_remaining_shares": None,
                     "partial_exit_fills": 0, "partial_sold_quantity": 0, "first_partial_date": None, "last_partial_date": None}
            execution_cycle_id = cycle_number
            risk_cost, risk_peak = cycle["entry_cost_cny"], cycle["entry_cost_cny"]
        elif old_shares > 0 and execution["filled_quantity"] < 0:
            cycle["sale_proceeds_cny"] += execution["notional"] - execution["commission"]
            if not terminal and pending["action_kind"] == "LEARNED_REDUCTION":
                cycle["partial_exit_fills"] += 1
                cycle["partial_sold_quantity"] += -execution["filled_quantity"]
                cycle["first_partial_date"] = cycle["first_partial_date"] or dates[day]
                cycle["last_partial_date"] = dates[day]
            if account.shares:
                ratio = account.shares / old_shares
                risk_cost *= ratio
                risk_peak *= ratio
                risk_dividends *= ratio
                risk_entitlements = {key: amount * ratio for key, amount in risk_entitlements.items()}
                if not terminal and pending["action_kind"] == "LEARNED_REDUCTION":
                    require(account.shares >= reduction_target, "减仓超出第一次锁定的份额")
            else:
                cycle.update({"exit_date": dates[day], "exit_index": day,
                              "exit_origin": pending["origin"] if not terminal else dates[day],
                              "exit_reasons": pending["exit_reasons"] if not terminal else "研究终点统一开盘退出",
                              "holding_intervals": day - cycle["entry_index"]})
                cycles.append(cycle)
                cycle, last_exit, full_exit_reasons, reduction_target = None, day, [], None
                risk_cost, risk_peak, risk_dividends, risk_entitlements = np.nan, np.nan, 0., {}
        elif account.shares and execution["filled_quantity"] > 0:
            raise ValueError("部分退出研究不允许持仓期间追加买入")
        if not terminal:
            for key, same in pay_events.get(day, []):
                if same:
                    value = account.receivables.pop(key, 0.)
                    paid += value
                    account.cash += value
            for key, amount in record_events.get(day, []):
                account.entitlements[key] = account.shares
                dividend_owners[key] = cycle
                if cycle is not None:
                    risk_entitlements[key] = float(account.shares)
        if account.shares:
            risk_peak = max(risk_peak, account.shares * mark + risk_dividends)
        nav = account.value(mark)
        price_pnl = old_shares * (op[day] - previous_mark) + account.shares * (mark - op[day])
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, "部分退出账户财富恒等式不成立")
        account.assert_valid()
        records.append({"date": dates[day], "open": op[day], "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(), "equity": nav,
                        "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav, "price_pnl": price_pnl,
                        "dividend_recognized": recognized, "dividend_paid": paid, "exposure": account.shares * mark / nav,
                        "accounting_error": error, "origin": dates[day - 1], "terminal_unliquidated": bool(terminal and account.shares),
                        "turnover": execution["notional"] / before, "cycle_id": cycle["cycle_id"] if cycle else None,
                        "execution_cycle_id": execution_cycle_id, "mode": cycle["mode"] if cycle else 0,
                        "execution_action_kind": pending["action_kind"] if not terminal else "TERMINAL",
                        "execution_reasons": pending["exit_reasons"] if not terminal else "研究终点统一退出", **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decide(day)
    if cycle:
        cycles.append({**cycle, "exit_date": None, "exit_reasons": "研究终点退出未成交"})
    for item in cycles:
        if item["exit_date"] is not None:
            item["net_profit_cny"] = item["sale_proceeds_cny"] + item["dividend_cny"] - item["entry_cost_cny"]
    if not account.shares:
        require(abs(sum(c["net_profit_cny"] for c in cycles) - (previous_nav - config["initial_capital"])) < 1e-6,
                "全部周期损益与实际账户终点资产不一致")
    return pd.DataFrame(records), pd.DataFrame(decisions), pd.DataFrame(cycles)
