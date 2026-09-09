"""组合自身净值回撤控制；实际现金、整手、权益及下一开盘交易。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import Account, execute_order, require, target_request


def cushion_target(equity, high_water, parent_target, floor_fraction=.8, multiplier=5.):
    require(np.isfinite(equity) and equity > 0 and np.isfinite(high_water) and high_water > 0, "自身净值或历史高点不完整")
    high_water = max(float(equity), float(high_water))
    floor = floor_fraction * high_water
    cushion = max(0., equity-floor)
    scale = min(1., max(0., multiplier*cushion/equity))
    if not np.isfinite(parent_target):
        target, state = float("nan"), "NO_VIEW_PARENT_KEEP_EXISTING_SHARES"
    else:
        require(0 <= parent_target <= 1, "原股票目标超出无杠杆范围")
        target = float(parent_target)*scale
        state = "EXPLICIT_FLOOR_EXIT" if scale == 0 else "OWN_ACCOUNT_CUSHION_TARGET"
    return {"high_water": high_water, "floor_value": floor, "cushion_value": cushion, "risk_scale": scale,
        "parent_target": float(parent_target), "target": target, "risk_status": state, "floor_breached_at_origin": equity < floor,
        "origin_equity": float(equity), "origin_drawdown": float(equity/high_water-1)}


def simulate_cushion_account(data, dividends, config, cost, start, model_id, parent_targets):
    """复制现有事件账户的固定成交时钟，仅在自身收盘净值上形成动态目标。"""
    require(len(parent_targets) == len(data), "原意向与价格日历长度不符")
    first = int(np.flatnonzero(data.date.ge(start))[0])
    last, anchor = len(data)-1, first-1
    dates = pd.DatetimeIndex(data.date)
    op, cl = data.open.to_numpy(float), data.close.to_numpy(float)
    previous, distribution = data.previous_close.to_numpy(float), data.dividend.to_numpy(float)
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
    high_water = float(config["initial_capital"])
    records, decisions = [], []

    def decide(t):
        nonlocal high_water
        state = cushion_target(account.value(cl[t]), high_water, float(parent_targets[t]), config["floor_fraction"], config["cushion_multiplier"])
        high_water = state["high_water"]
        if np.isfinite(state["target"]):
            result = target_request(account, cl[t], state["target"], config)
        else:
            result = {"requested_quantity": 0, "reference_weight": float("nan"), "action": "原策略意向缺失，保留实际份额"}
        row = {"origin": dates[t], "execution_date": dates[t+1], "origin_index": t,
            "simulation_only": True, "signal_state": state["risk_status"], **state, **result}
        decisions.append(row)
        return row

    pending = decide(anchor)
    for day in range(first, last+1):
        old_shares, recognized, paid = account.shares, 0., 0.
        for key, amount in ex_events.get(day, []):
            value = account.entitlements.get(key, 0)*amount
            account.receivables[key] = value
            recognized += value
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
        if not terminal:
            for key, same in pay_events.get(day, []):
                if same:
                    value = account.receivables.pop(key, 0.)
                    paid += value
                    account.cash += value
            for key, amount in record_events.get(day, []):
                account.entitlements[key] = account.shares
        nav = account.value(mark)
        price_pnl = old_shares*(op[day]-previous_mark) + account.shares*(mark-op[day])
        error = nav-previous_nav-price_pnl-recognized+execution["commission"]+execution["slippage_cost"]
        require(abs(error) < 1e-6, "自身净值控制账户财富恒等式不成立")
        account.assert_valid()
        records.append({"date": dates[day], "open": op[day], "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
            "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(), "equity": nav,
            "net_return": nav/previous_nav-1, "pnl": nav-previous_nav, "price_pnl": price_pnl,
            "dividend_recognized": recognized, "dividend_paid": paid, "exposure": account.shares*mark/nav,
            "accounting_error": error, "origin": dates[day-1], "terminal_unliquidated": bool(terminal and account.shares),
            "turnover": execution["notional"]/before, "decision_floor_value": pending["floor_value"], "below_previous_decision_floor": nav < pending["floor_value"], **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decide(day)
    return pd.DataFrame(records), pd.DataFrame(decisions)
