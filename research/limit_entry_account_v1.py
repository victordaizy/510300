"""预先确定限价与有效期的完整研究账户；盘中成交保留假设属性。"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import Account, affordable_quantity, execute_order, fill_price, require


def entry_execution(account, order, data, day, cost, config, assumption):
    """只在执行层读取当日开低价格，不影响前一收盘的订单定义。"""
    row = data.iloc[day]
    op, low, previous, dividend = map(float, [row.open, row.low, row.previous_close, row.dividend])
    limit = float(order["limit_price"])
    tick = config["tick"]
    basis = previous - dividend
    lower = math.floor(basis * (1-config["limit_fraction"])/tick+.5+1e-9)*tick
    upper = math.floor(basis * (1+config["limit_fraction"])/tick+.5+1e-9)*tick
    reason = None
    reference = op
    clock = "NO_FILL"
    if dividend:
        reason = "UNFILLED_EX_DIVIDEND_CANCEL"
    elif not lower-1e-9 <= limit <= upper+1e-9:
        reason = "UNFILLED_LIMIT_OUTSIDE_DAILY_BAND"
    elif op < upper-1e-9 and fill_price(op,1,cost,tick) <= limit+1e-9:
        clock = "OPEN_LIMIT_ELIGIBLE_SIMULATION"
    elif assumption == "PENETRATION":
        reference = math.floor(limit/(1+cost["slippage"])/tick+1e-10)*tick
        if low <= reference-2*tick+1e-9 and lower-1e-9 <= reference < upper-1e-9:
            clock = "INTRADAY_TIME_UNOBSERVED_CONDITIONAL_SIMULATION"
        else:
            reason = "UNFILLED_NO_TWO_TICK_PENETRATION"
    else:
        reason = "UNFILLED_OPEN_EXCEEDS_LIMIT"
    if reason:
        execution = execute_order(account,0,op,previous,dividend,day,cost,config)
        execution.update({"requested_quantity":order["quantity"],"status":reason,"execution_clock":clock,
                          "reference_execution_price":op,"order_limit_price":limit,"fill_evidence":"NO_FILL"})
    else:
        execution = execute_order(account,order["quantity"],reference,previous,dividend,day,cost,config)
        require(execution["filled_quantity"] == 0 or execution["fill_price"] <= limit+1e-9,"模拟成交价超过预定限价")
        execution.update({"execution_clock":clock,"reference_execution_price":reference,"order_limit_price":limit,
                          "fill_evidence":"MODEL_ASSUMPTION_NOT_OBSERVED_EXECUTION"})
    return execution


def simulate_limit_policy(data, dividends, config, cost, start, rule, spec, entry_spec, assumption):
    require(assumption in ["OPEN_ONLY","PENETRATION"],"未知限价成交假设")
    first=int(np.flatnonzero(data.date>=pd.Timestamp(start))[0]);last=len(data)-1;anchor=first-1
    dates=pd.DatetimeIndex(data.date)
    op,cl=data.open.to_numpy(float),data.close.to_numpy(float)
    previous,distribution=data.previous_close.to_numpy(float),data.dividend.to_numpy(float)
    record_events,ex_events,pay_events={},{},{}
    for key,event in enumerate(dividends.itertuples()):
        for field,mapping in [("record_date",record_events),("ex_date",ex_events)]:
            when=getattr(event,field)
            if when in dates:mapping.setdefault(dates.get_loc(when),[]).append((key,event.cash_dividend_per_share))
        idx=dates.searchsorted(event.payment_date)
        if idx<len(dates):pay_events.setdefault(idx,[]).append((key,event.payment_date==dates[idx]))
    account=Account(config["initial_capital"])
    previous_nav,previous_mark=config["initial_capital"],cl[anchor]
    records,decisions,cycles,orders=[],[],[],[]
    cycle,active_order=None,None
    cycle_number,order_number=0,0
    last_exit,last_order_end=-1000000,-1000000
    pending_reasons=[];peak_value=np.nan;cycle_dividends=0.

    def close_order(t,status):
        nonlocal active_order,last_order_end
        active_order.update({"final_status":status,"end_date":dates[t],"end_index":t})
        last_order_end=t
        active_order=None

    def decide(t):
        nonlocal active_order,order_number,pending_reasons
        mode=int(rule["entry"][t]);qty=0;action="空仓等待符合条件的信号";reasons=[]
        if account.shares:
            selected_mode=cycle["mode"]
            s=spec["modes"].get(selected_mode,spec["modes"].get(str(selected_mode)))
            value=account.shares*cl[t]+cycle_dividends
            r=value/cycle["entry_cost_cny"]-1
            if not pending_reasons:
                if rule["exit"][selected_mode][t]:reasons.append("本持仓模式的价格退出条件成立")
                if s["loss"] is not None and r<=-s["loss"]:reasons.append("持仓含分红收益触及固定止损")
                if s["trail"] is not None and value/peak_value-1<=-s["trail"]:reasons.append("持仓价值从周期高点回落至追踪退出线")
                if s["take"] is not None and r>=s["take"]:reasons.append("持仓含分红收益达到预定止盈")
                if s["days"] is not None and t-cycle["entry_index"]+1>=s["days"]:reasons.append("最长持有交易日到期")
                pending_reasons=reasons
            if pending_reasons:
                qty=-account.shares;reasons=pending_reasons;action="下一开盘全部退出，未成交则保留请求"
            else:action="持有已有份额并检查退出"
        else:
            if active_order is not None:
                if t>=active_order["expiry_index"]:
                    close_order(t,"EXPIRED_WITHOUT_FILL");action="限价单到期，保持未成交"
                elif rule["exit"][active_order["mode"]][t]:
                    close_order(t,"CANCELLED_PRICE_EXIT_CONDITION");action="原持仓模式退出条件成立，撤销未成交买单"
            if active_order is None and mode and t-last_exit>=spec["cooldown"] and t-last_order_end>=1:
                limit=math.floor(cl[t]*(1-entry_spec["discount"])/config["tick"]+1e-10)*config["tick"]
                qty=affordable_quantity(account.cash,limit,cost,config["lot"])
                if qty:
                    order_number+=1
                    active_order={"order_id":order_number,"origin":dates[t],"origin_index":t,"mode":mode,
                                  "limit_price":limit,"quantity":qty,"expiry_index":t+entry_spec["valid_days"],
                                  "valid_days":entry_spec["valid_days"],"discount":entry_spec["discount"],
                                  "final_status":"PENDING","end_date":pd.NaT,"end_index":None,
                                  "actual_fill_date":pd.NaT,"fill_evidence":"NO_FILL"}
                    orders.append(active_order)
                    action="按当前收盘预定限价与数量，下一交易日起等待"
            if active_order is not None:
                qty=active_order["quantity"]
                if action=="空仓等待符合条件的信号":action="保留原限价、数量和到期日，不追价"
        record={"origin":dates[t],"execution_date":dates[t+1] if t+1<len(data) else pd.NaT,"origin_index":t,
                "requested_quantity":int(qty),"exit_reasons":"；".join(reasons),"action":action,
                "active_order_id":active_order["order_id"] if active_order else None,
                "limit_price":active_order["limit_price"] if active_order else None,
                "entry_mode":active_order["mode"] if active_order else cycle["mode"] if cycle else 0,
                "reference_weight":1. if qty>0 or (account.shares and qty==0) else 0.,"simulation_only":True}
        decisions.append(record)
        return record

    pending=decide(anchor)
    for day in range(first,last+1):
        old_shares,recognized,paid=account.shares,0.,0.
        for key,amount in ex_events.get(day,[]):
            value=account.entitlements.get(key,0)*amount;account.receivables[key]=value;recognized+=value
        if old_shares:cycle_dividends+=recognized
        for key,same in pay_events.get(day,[]):
            if not same:
                value=account.receivables.pop(key,0.);paid+=value;account.cash+=value
        terminal=day==last;mark=op[day] if terminal else cl[day]
        before=account.value(op[day]);filled_order=None
        actual_order_origin=dates[day-1]
        if terminal or old_shares:
            qty=-account.shares if terminal else int(pending["requested_quantity"])
            execution=execute_order(account,qty,op[day],previous[day],distribution[day],day,cost,config)
            execution.update({"reference_execution_price":op[day],"execution_clock":"OPEN_EXIT" if qty<0 else "NO_FILL", "order_limit_price":None,
                              "fill_evidence":"OPEN_EXECUTION_SIMULATION" if execution["filled_quantity"] else "NO_FILL"})
            if terminal and active_order is not None:close_order(day,"CANCELLED_AT_RESEARCH_TERMINAL")
        elif active_order is not None:
            require(active_order["origin_index"]<day<=active_order["expiry_index"],"限价单在未生效或到期后执行")
            actual_order_origin=active_order["origin"]
            execution=entry_execution(account,active_order,data,day,cost,config,assumption)
            if execution["filled_quantity"]:
                filled_order=active_order.copy()
                active_order.update({"actual_fill_date":dates[day],"fill_price":execution["fill_price"],
                                     "filled_quantity":execution["filled_quantity"],"fill_evidence":execution["fill_evidence"],
                                     "execution_clock":execution["execution_clock"]})
                close_order(day,"SIMULATED_FILL")
            elif distribution[day]:close_order(day,"CANCELLED_ON_EX_DIVIDEND_DATE")
        else:
            execution=execute_order(account,0,op[day],previous[day],distribution[day],day,cost,config)
            execution.update({"reference_execution_price":op[day],"execution_clock":"NO_FILL","order_limit_price":None,"fill_evidence":"NO_FILL"})
        reference=execution["reference_execution_price"]
        if old_shares==0 and account.shares:
            require(filled_order is not None,"新买入缺少事前限价单")
            cycle_number+=1;cycle_dividends=0.;pending_reasons=[]
            cycle={"cycle_id":cycle_number,"mode":filled_order["mode"],"entry_date":dates[day],"entry_index":day,
                   "entry_origin":filled_order["origin"],"entry_order_id":filled_order["order_id"],
                   "entry_quantity":account.shares,"entry_limit":filled_order["limit_price"],"entry_clock":execution["execution_clock"],
                   "entry_cost_cny":execution["notional"]+execution["commission"],"fill_evidence":execution["fill_evidence"]}
            peak_value=cycle["entry_cost_cny"]
        elif old_shares and not account.shares:
            proceeds=execution["notional"]-execution["commission"]
            cycle.update({"exit_date":dates[day],"exit_origin":pending["origin"] if not terminal else dates[day],
                          "exit_reasons":pending["exit_reasons"] if not terminal else "研究终点统一开盘退出",
                          "holding_intervals":day-cycle["entry_index"],"dividend_cny":cycle_dividends,
                          "net_profit_cny":proceeds+cycle_dividends-cycle["entry_cost_cny"]})
            cycles.append(cycle);cycle=None;last_exit=day;pending_reasons=[];peak_value=np.nan;cycle_dividends=0.
        elif account.shares and execution["filled_quantity"]:raise ValueError("限价策略不允许持仓周期中加减仓")
        if not terminal:
            for key,same in pay_events.get(day,[]):
                if same:
                    value=account.receivables.pop(key,0.);paid+=value;account.cash+=value
            for key,amount in record_events.get(day,[]):account.entitlements[key]=account.shares
        if account.shares:peak_value=max(peak_value,account.shares*mark+cycle_dividends)
        nav=account.value(mark)
        price_pnl=old_shares*(mark-previous_mark)+execution["filled_quantity"]*(mark-reference)
        error=nav-previous_nav-price_pnl-recognized+execution["commission"]+execution["slippage_cost"]
        require(abs(error)<1e-6,"限价完整账户财富恒等式不成立")
        account.assert_valid()
        records.append({"date":dates[day],"open":op[day],"mark":mark,"mark_clock":"OPEN_TERMINAL" if terminal else "CLOSE",
                        "cash":account.cash,"shares":account.shares,"dividend_receivable":account.receivable(),"equity":nav,
                        "net_return":nav/previous_nav-1,"pnl":nav-previous_nav,"price_pnl":price_pnl,"dividend_recognized":recognized,"dividend_paid":paid,
                        "exposure":account.shares*mark/nav,"accounting_error":error,"origin":actual_order_origin,
                        "terminal_unliquidated":bool(terminal and account.shares),"turnover":execution["notional"]/before,
                        "cycle_id":cycle["cycle_id"] if cycle else None,"mode":cycle["mode"] if cycle else 0,
                        "execution_reasons":pending["exit_reasons"] if not terminal else "研究终点统一退出",**execution})
        previous_nav,previous_mark=nav,mark
        if not terminal:pending=decide(day)
    if cycle:cycles.append({**cycle,"exit_date":None,"exit_reasons":"终点退出未成交"})
    return pd.DataFrame(records),pd.DataFrame(decisions),pd.DataFrame(cycles),pd.DataFrame(orders)
