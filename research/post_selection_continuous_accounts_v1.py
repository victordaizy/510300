"""连续收盘结算与可恢复账户；原成交及持仓规则保持。"""
from copy import deepcopy
from dataclasses import asdict
import math
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import Account, affordable_quantity, fill_price, choose_order, execute_order, require, target_request


def pack(value):
    if isinstance(value, dict):
        return {str(k):pack(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)):
        return [pack(v) for v in value]
    if value is pd.NaT:
        return {'__timestamp__':'NaT'}
    if isinstance(value,pd.Timestamp):
        return {'__timestamp__':value.isoformat()}
    if isinstance(value,(float,np.floating)):
        return float(value) if math.isfinite(value) else {'__float__':str(float(value))}
    if isinstance(value,(np.integer,)):
        return int(value)
    if isinstance(value,(np.bool_,)):
        return bool(value)
    return value


def unpack(value):
    if isinstance(value,dict):
        if set(value)=={'__timestamp__'}:
            return pd.Timestamp(value['__timestamp__'])
        if set(value)=={'__float__'}:
            return float(value['__float__'])
        return {k:unpack(v) for k,v in value.items()}
    if isinstance(value,list):
        return [unpack(v) for v in value]
    return value


def restore(snapshot,data,config,cost,kind,identity,controller=None):
    require(snapshot['engine']==kind and snapshot['identity']==identity and not snapshot['terminal_liquidation'], '恢复账户身份或结算方式不同')
    require(snapshot['economics']=={k:config[k] for k in ['initial_capital','lot','tick','limit_fraction']} and snapshot['cost']==cost, '恢复成本或份额规则不同')
    state=unpack(snapshot['state'])
    require(0<=state['asof_index']<len(data) and data.date.iloc[state['asof_index']]==state['asof_date'], '恢复状态日期与原索引不同')
    item=state['account']
    item['receivables']={int(k):v for k,v in item['receivables'].items()}
    item['entitlements']={int(k):v for k,v in item['entitlements'].items()}
    item['purchase_lots']=[tuple(x) for x in item['purchase_lots']]
    state['account']=Account(**item)
    state['account'].assert_valid()
    saved=snapshot.get('controller')
    require((saved is None)==(controller is None), '恢复控制器有无不同')
    if saved:
        require(saved['class']==controller.__class__.__name__, '恢复控制器类型不同')
        controller.__dict__.update(unpack(saved['fields']))
    return state


def checkpoint(context,config,cost,kind,identity,fields,controller=None):
    state={k:deepcopy(context[k]) for k in fields}
    state.update(account=asdict(context['account']),asof_index=int(context['last']),asof_date=context['dates'][context['last']])
    saved=None
    if controller is not None:
        saved={'class':controller.__class__.__name__,'fields':pack({k:v for k,v in controller.__dict__.items() if k not in ['data','models','indexes','fit_indexes']})}
    return dict(version=1,engine=kind,identity=identity,terminal_liquidation=bool(context['terminal_liquidation']),
        economics={k:config[k] for k in ['initial_capital','lot','tick','limit_fraction']},cost=cost,state=pack(state),controller=saved)


def check_next_date(data,last,terminal_liquidation,next_execution_date):
    if last==len(data)-1 and not terminal_liquidation:
        require(next_execution_date is not None and pd.notna(pd.Timestamp(next_execution_date)) and pd.Timestamp(next_execution_date)>data.date.iloc[-1],
            '末日连续决定必须提供明确的下一交易日，不能猜测或使用当前时间')


def simulate_indexed_request_account(data: pd.DataFrame, dividends: pd.DataFrame, config: dict, cost: dict, start: str,
             model_id: str, targets: np.ndarray | None = None, prediction: np.ndarray | None = None,
             horizon: int = 1, event_mask: np.ndarray | None = None, *, request_policy, terminal_liquidation=False, resume=None, stop_index=None, next_execution_date=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    require(event_mask is not None and len(event_mask) == len(data), "事件调整时钟缺失或长度不符")
    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])
    last, anchor = len(data) - 1 if stop_index is None else int(stop_index), first - 1
    check_next_date(data,last,terminal_liquidation,next_execution_date)
    dates = pd.DatetimeIndex(data.date)
    open_prices, close_prices = data.open.to_numpy(float), data.close.to_numpy(float)
    previous_close, ex_amount = data.previous_close.to_numpy(float), data.dividend.to_numpy(float)
    variance = data.variance60.to_numpy(float)
    record_events, ex_events, pay_events = {}, {}, {}
    for k, event in enumerate(dividends.itertuples()):
        for field, mapping in (("record_date", record_events), ("ex_date", ex_events)):
            event_date = getattr(event, field)
            if event_date in dates:
                mapping.setdefault(dates.get_loc(event_date), []).append((k, event.cash_dividend_per_share))
        pay_index = dates.searchsorted(event.payment_date)
        if pay_index < len(dates):
            pay_events.setdefault(pay_index, []).append((k, event.payment_date == dates[pay_index]))
    account = Account(config["initial_capital"])
    records, decisions = [], []
    previous_nav = config["initial_capital"]
    previous_mark = close_prices[anchor]
    reference_target = float("nan")
    signal_state = "NO_MODEL_VIEW_AT_START"

    def decision_at(t: int) -> dict:
        nonlocal reference_target, signal_state
        price = close_prices[t]
        scheduled = t == anchor or bool(event_mask[t])
        if model_id == "BUY_HOLD":
            signal_state = "EXPLICIT_BUY_HOLD_POLICY"
            if t == anchor:
                request = affordable_quantity(account.cash, fill_price(price, 1, cost, config["tick"]), cost, config["lot"])
                result = {"requested_quantity": request, "reference_weight": 1.0, "action": "初始买入持有"}
            else:
                result = {"requested_quantity": 0, "reference_weight": 1.0, "action": "长期持有"}
        elif scheduled:
            value = float(prediction[t]) if prediction is not None else float(targets[t]) if targets is not None else float("nan")
            if np.isfinite(value):
                signal_state = "MODEL_VIEW_AVAILABLE"
                result = choose_order(account, price, value, horizon * variance[t], cost, config) if prediction is not None else request_policy(account, price, value, config, model_id, t)
            else:
                signal_state = "NO_VIEW_KEEP_EXISTING_SHARES"
                result = {"requested_quantity": 0, "reference_weight": float("nan"), "action": "资料不足，无新判断，保留已有份额"}
        else:
            result = {"requested_quantity": 0, "reference_weight": reference_target, "action": "非公告调整时点维持份额"}
        result["signal_state"] = signal_state
        reference_target = float(result["reference_weight"])
        decisions.append({"origin": dates[t], "execution_date": dates[t + 1] if t + 1 < len(dates) else pd.Timestamp(next_execution_date),
                          "origin_index": t, "simulation_only": True, **result})
        return result

    range_start=first
    if resume is not None:
        state=restore(resume,data,config,cost,'target',model_id)
        account=state['account']
        previous_nav,previous_mark=state['previous_nav'],state['previous_mark']
        reference_target,signal_state=state['reference_target'],state['signal_state']
        range_start=state['asof_index']+1
    require(first<=range_start<=last<len(data), '连续账户区间无效')
    pending = state['pending'] if resume is not None else decision_at(anchor)
    for day in range(range_start, last + 1):
        date, op, cl = dates[day], open_prices[day], close_prices[day]
        old_shares, recognized, paid = account.shares, 0.0, 0.0
        for k, amount in ex_events.get(day, []):
            value = account.entitlements.get(k, 0) * amount
            account.receivables[k] = value
            recognized += value
        for k, same_day in pay_events.get(day, []):
            if not same_day:
                value = account.receivables.pop(k, 0.0)
                paid += value
                account.cash += value
        terminal = terminal_liquidation and day == last
        quantity = -account.shares if terminal else int(pending["requested_quantity"])
        pretrade_nav = account.value(op)
        execution = execute_order(account, quantity, op, previous_close[day], ex_amount[day], day, cost, config)
        mark = op if terminal else cl
        if not terminal:
            for k, same_day in pay_events.get(day, []):
                if same_day:
                    value = account.receivables.pop(k, 0.0)
                    paid += value
                    account.cash += value
            for k, amount in record_events.get(day, []):
                account.entitlements[k] = account.shares
        nav = account.value(mark)
        price_pnl = old_shares * (op - previous_mark) + account.shares * (mark - op)
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, f"{model_id} 在 {date} 财富恒等式不成立：{error}")
        account.assert_valid()
        records.append({"date": date, "open": op, "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                        "equity": nav, "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav,
                        "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
                        "exposure": account.shares * mark / nav, "accounting_error": error,
                        "origin": dates[day - 1], "terminal_unliquidated": bool(terminal and account.shares),
                        "turnover": execution["notional"] / pretrade_nav, **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decision_at(day)
    state=checkpoint(locals(),config,cost,'target',model_id,['previous_nav', 'previous_mark', 'reference_target', 'signal_state', 'pending'])
    return pd.DataFrame(records), pd.DataFrame(decisions), state


def simulate_policy(data, dividends, config, cost, start, rule, spec, *, terminal_liquidation=False, resume=None, stop_index=None, next_execution_date=None):
    """按实际成交状态维护持仓周期、退出待成交和再入场等待期。"""
    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])
    last, anchor = len(data) - 1 if stop_index is None else int(stop_index), first - 1
    check_next_date(data,last,terminal_liquidation,next_execution_date)
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
        row = {"origin": dates[t], "execution_date": dates[t + 1] if t + 1 < len(dates) else pd.Timestamp(next_execution_date),
               "origin_index": t, "requested_quantity": int(qty), "entry_mode": selected_mode,
               "exit_reasons": "；".join(reasons), "action": action,
               "reference_weight": 0. if qty < 0 or (not account.shares and not qty) else 1.,
               "simulation_only": True}
        decisions.append(row)
        return row

    range_start=first
    if resume is not None:
        state=restore(resume,data,config,cost,'price','cycle')
        account=state['account']
        previous_nav=state['previous_nav']
        previous_mark=state['previous_mark']
        cycle=state['cycle']
        last_exit=state['last_exit']
        cycle_number=state['cycle_number']
        pending_reasons=state['pending_reasons']
        peak_value=state['peak_value']
        cycle_dividends=state['cycle_dividends']
        range_start=state['asof_index']+1
    require(first<=range_start<=last<len(data), '连续周期账户区间无效')
    pending = state['pending'] if resume is not None else decide(anchor)
    for day in range(range_start, last + 1):
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
        terminal = terminal_liquidation and day == last
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
    state=checkpoint(locals(),config,cost,'price','cycle',['previous_nav', 'previous_mark', 'cycle', 'last_exit', 'cycle_number', 'pending_reasons', 'peak_value', 'cycle_dividends', 'pending'])
    if cycle:
        cycles.append({**cycle, "exit_date": None, "exit_reasons": "研究终点退出未成交" if terminal_liquidation else "截至收盘仍持仓，按原规则继续"})
    return pd.DataFrame(records), pd.DataFrame(decisions), pd.DataFrame(cycles), state


def simulate_rearmed_exit(data, dividends, config, cost, start, rule, spec, controller=None, *, terminal_liquidation=False, resume=None, stop_index=None, next_execution_date=None):
    """按实际成交状态维护持仓周期、退出待成交和再入场等待期。"""
    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])
    last, anchor = len(data) - 1 if stop_index is None else int(stop_index), first - 1
    check_next_date(data,last,terminal_liquidation,next_execution_date)
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
        mode = int(rule["entry"][t])
        if not account.shares and (mode == 0 or (last_entry_mode and mode != last_entry_mode)):
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
                if learned.get("learned_exit_requested", False):
                    reasons.append("连续两个收盘预测继续持有收益为负，学习条件请求退出")
                pending_reasons = reasons
            if pending_reasons:
                qty, reasons = -account.shares, pending_reasons
                action = "请求全部退出，受阻后逐日继续请求"
            else:
                action = "保持已有份额，收盘检查退出"
        elif mode and entry_rearmed and t - last_exit >= spec["cooldown"]:
            qty = affordable_quantity(account.cash, fill_price(cl[t], 1, cost, config["tick"]), cost, config["lot"])
            selected_mode, action = mode, "入场条件成立，下一开盘请求买入"
        elif mode and not entry_rearmed:
            action = "旧入场条件尚未消失，等待新的机会"
        elif t - last_exit < spec["cooldown"]:
            action = "退出后的等待期尚未结束"
        else:
            action = "空仓等待新的入场条件"
        row = {"origin": dates[t], "execution_date": dates[t + 1] if t + 1 < len(dates) else pd.Timestamp(next_execution_date),
               "origin_index": t, "requested_quantity": int(qty), "entry_mode": selected_mode,
               "exit_reasons": "；".join(reasons), "action": action,
               "reference_weight": 0. if qty < 0 or (not account.shares and not qty) else 1.,
               "simulation_only": True}
        row["entry_rearmed"] = entry_rearmed
        row.update(learned)
        decisions.append(row)
        return row

    range_start=first
    if resume is not None:
        state=restore(resume,data,config,cost,'rearmed','cycle',controller)
        account=state['account']
        previous_nav=state['previous_nav']
        previous_mark=state['previous_mark']
        cycle=state['cycle']
        last_exit=state['last_exit']
        cycle_number=state['cycle_number']
        pending_reasons=state['pending_reasons']
        peak_value=state['peak_value']
        cycle_dividends=state['cycle_dividends']
        entry_rearmed=state['entry_rearmed']
        last_entry_mode=state['last_entry_mode']
        range_start=state['asof_index']+1
    require(first<=range_start<=last<len(data), '连续周期账户区间无效')
    pending = state['pending'] if resume is not None else decide(anchor)
    for day in range(range_start, last + 1):
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
        terminal = terminal_liquidation and day == last
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
    state=checkpoint(locals(),config,cost,'rearmed','cycle',['previous_nav', 'previous_mark', 'cycle', 'last_exit', 'cycle_number', 'pending_reasons', 'peak_value', 'cycle_dividends', 'pending', 'entry_rearmed', 'last_entry_mode'],controller)
    if cycle:
        cycles.append({**cycle, "exit_date": None, "exit_reasons": "研究终点退出未成交" if terminal_liquidation else "截至收盘仍持仓，按原规则继续"})
    return pd.DataFrame(records), pd.DataFrame(decisions), pd.DataFrame(cycles), state
