"""原收盘策略增加事前止盈限价单；收益阈值、入场和常规退出保持。"""
import math
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import Account, affordable_quantity, execute_order, fill_price, require

ASSUMPTIONS = ["THROUGH_PRICE", "CLOSE_STILL_ABOVE"]


def profit_limit_price(shares, entry_cost, known_dividends, take, tick):
    require(shares > 0 and entry_cost > 0 and np.isfinite(known_dividends), "止盈限价缺少实际旧份额、成本或已知分红")
    raw = (entry_cost*(1+take)-known_dividends)/shares
    require(np.isfinite(raw) and raw > 0, "止盈限价非正或无效")
    return math.ceil(raw/tick-1e-10)*tick


def required_reference_price(limit_price, cost, tick):
    require(0 <= cost["slippage"] < 1 and limit_price > 0, "止盈滑点或限价范围无效")
    value = math.ceil(limit_price/(1-cost["slippage"])/tick-1e-10)*tick
    while fill_price(value, -1, cost, tick) < limit_price-1e-10:
        value += tick
    return value


def standing_execution_reference(limit_price, op, high, close, volume, previous, dividend, shares, cost, cfg, assumption):
    require(assumption in ASSUMPTIONS, "未知止盈限价成交情景")
    threshold = required_reference_price(limit_price, cost, cfg["tick"])
    upper = math.floor((previous-dividend)*(1+cfg["limit_fraction"])/cfg["tick"]+.5+1e-9)*cfg["tick"]
    if limit_price > upper+1e-9:
        return None, "NOT_SUBMITTED_ABOVE_DAILY_UPPER", threshold
    if np.isfinite(op) and op >= threshold-1e-10:
        return float(op), "OPEN_MARKETABLE_PROFIT_LIMIT", threshold
    if not np.isfinite(high) or not np.isfinite(volume) or volume <= 0:
        return None, "NO_VIEW_INTRADAY_EXECUTION_INPUT", threshold
    if shares > cfg["daily_volume_cap"]*volume:
        return None, "UNFILLED_DAILY_VOLUME_CAP", threshold
    # 日内必须穿过足以支付原滑点的参考价至少一个最小价位，触线不假设成交。
    if high < threshold+cfg["tick"]-1e-10:
        return None, "LIMIT_ACTIVE_NOT_THROUGH", threshold
    if assumption == "CLOSE_STILL_ABOVE":
        if not np.isfinite(close):
            return None, "NO_VIEW_CLOSE_EXECUTION_CONFIRMATION", threshold
        if close < threshold+cfg["tick"]-1e-10:
            return None, "UNFILLED_CLOSE_NOT_STILL_ABOVE", threshold
    return threshold, "INTRADAY_CONDITIONAL_PROFIT_LIMIT", threshold


def simulate_profit_limit(data, dividends, config, cost, start, rule, spec, assumption="DISABLED"):
    """按实际成交状态维护持仓周期、退出待成交和再入场等待期。"""
    require(assumption in ["DISABLED"]+ASSUMPTIONS, "止盈执行情景不符")
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

    def decide(t):
        nonlocal pending_reasons
        mode = int(rule["entry"][t])
        reasons, qty, selected_mode = [], 0, 0
        if account.shares:
            selected_mode = cycle["mode"]
            s = spec["modes"].get(selected_mode, spec["modes"].get(str(selected_mode)))
            current_value = account.shares * cl[t] + cycle_dividends
            cycle_return = current_value / cycle["entry_cost_cny"] - 1
            if not pending_reasons:
                if rule["exit"][selected_mode][t]:
                    reasons.append("本持仓模式的价格退出条件成立")
                if s["loss"] is not None and cycle_return <= -s["loss"]:
                    reasons.append("持仓含分红收益触及固定止损")
                if s["trail"] is not None and current_value / peak_value - 1 <= -s["trail"]:
                    reasons.append("含分红持仓价值从周期高点回落至追踪退出线")
                if s["take"] is not None and cycle_return >= s["take"]:
                    reasons.append("持仓含分红收益达到预定止盈")
                if s["days"] is not None and t - cycle["entry_index"] + 1 >= s["days"]:
                    reasons.append("预定最长持有交易日到期")
                pending_reasons = reasons
            if pending_reasons:
                qty, reasons = -account.shares, pending_reasons
                action = "请求全部退出，受阻后逐日继续请求"
            else:
                action = "保持已有份额，收盘检查退出"
        elif mode and t - last_exit >= spec["cooldown"]:
            qty = affordable_quantity(account.cash, fill_price(cl[t], 1, cost, config["tick"]), cost, config["lot"])
            selected_mode, action = mode, "入场条件成立，下一开盘请求买入"
        elif t - last_exit < spec["cooldown"]:
            action = "退出后的等待期尚未结束"
        else:
            action = "空仓等待新的入场条件"
        row = {"origin": dates[t], "execution_date": dates[t + 1] if t + 1 < len(dates) else pd.NaT,
               "origin_index": t, "requested_quantity": int(qty), "entry_mode": selected_mode,
               "exit_reasons": "；".join(reasons), "action": action,
               "reference_weight": 0. if qty < 0 or (not account.shares and not qty) else 1.,
               "simulation_only": True}
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
        entry_cost_before = cycle["entry_cost_cny"] if cycle else np.nan
        known_dividends_before = cycle_dividends if cycle else np.nan
        sellable_before = account.sellable(day)
        limit_price, threshold, standing_quantity = None, None, 0
        limit_status, limit_reference = "NO_ELIGIBLE_STANDING_ORDER", None
        exit_text = pending["exit_reasons"] if not terminal else "研究终点统一开盘退出"
        clock = "OPEN_TERMINAL" if terminal else ("OPEN_SCHEDULED" if quantity else "NONE")
        if old_shares and not terminal and assumption != "DISABLED":
            if quantity < 0:
                limit_status = "SCHEDULED_EXIT_HAS_PRIORITY"
            else:
                protection = spec["modes"].get(cycle["mode"], spec["modes"].get(str(cycle["mode"])))
                require(protection["take"] is not None and sellable_before == old_shares, "止盈限价不能使用当日新买份额")
                limit_price = profit_limit_price(old_shares, cycle["entry_cost_cny"], cycle_dividends, protection["take"], config["tick"])
                standing_quantity = -old_shares
                limit_reference, limit_status, threshold = standing_execution_reference(limit_price, op[day], float(data.high.iloc[day]), cl[day], float(data.volume.iloc[day]), previous[day], distribution[day], old_shares, cost, config, assumption)
        if limit_reference is not None:
            exit_text = "此前已知含分红止盈价的限价卖单符合本情景成交条件"
            clock = limit_status
            execution = execute_order(account, standing_quantity, limit_reference, previous[day], distribution[day], day, cost, config)
            if execution["filled_quantity"]:
                require(execution["fill_price"] >= limit_price-1e-10, "卖出限价单的成交价低于限价")
        else:
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
                          "exit_reasons": exit_text, "exit_clock": clock,
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
        price_pnl = old_shares*(mark-previous_mark)+execution["filled_quantity"]*(mark-execution["open_price"])
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, "价格策略账户财富恒等式不成立")
        account.assert_valid()
        records.append({"date": dates[day], "open": op[day], "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(), "equity": nav,
                        "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav, "price_pnl": price_pnl,
                        "dividend_recognized": recognized, "dividend_paid": paid, "exposure": account.shares * mark / nav,
                        "accounting_error": error, "origin": dates[day - 1], "terminal_unliquidated": bool(terminal and account.shares),
                        "turnover": execution["notional"] / before, "cycle_id": cycle["cycle_id"] if cycle else None,
                        "mode": cycle["mode"] if cycle else 0, "execution_reasons": exit_text, "execution_clock": clock,
                        "standing_limit_price": limit_price, "standing_reference_threshold": threshold, "standing_limit_quantity": standing_quantity,
                        "standing_limit_status": limit_status, "entry_cost_before": entry_cost_before, "known_cycle_dividends_before": known_dividends_before,
                        "sellable_before": sellable_before, "assumption": assumption,
                        **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decide(day)
    if cycle:
        cycles.append({**cycle, "exit_date": None, "exit_reasons": "研究终点退出未成交"})
    return pd.DataFrame(records), pd.DataFrame(decisions), pd.DataFrame(cycles)
