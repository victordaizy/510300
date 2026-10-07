"""支持信息与价格接受的隔离现金账户；原资金、费用与风险预算复用。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import support_price_acceptance_inputs_v1 as candidate
from research import daily_supply_test_v1 as budget
from research.point_account_nr7_inputs_v1 import fee, fill, open_blocked

EPS = candidate.EPS


def account(data, dividends, risks, policy, cost, start):
    if policy not in candidate.POLICIES:
        raise ValueError("未知支持与价格接受政策。")
    d = data.reset_index(drop=True)
    first = int(np.flatnonzero(d.date.ge(pd.Timestamp(start)))[0])
    if first < 1:
        raise ValueError("缺少首个开盘前的决定。")
    risk = dict(zip(risks.idx.astype(int), risks.es95))
    cash, q, receivable, peak, previous_nav = 200000., 0, 0., 200000., 200000.
    active, stopped, locked_exit = None, False, False
    entitlements, outstanding, cycles = {}, {}, []
    events = dividends.to_dict("records")
    daily, orders, rejections, decisions = [], [], [], []
    prefix = "price" if policy == candidate.POLICIES[1] else "support"

    def decide(i):
        nonlocal locked_exit
        row = d.iloc[i]
        nav = cash+receivable+q*float(row.close)
        es = risk.get(i, np.nan)
        entry = bool(row[prefix+"_entry_event"])
        desired, reason = q, "KEEP"
        if stopped or locked_exit:
            desired, reason = 0, "ACCOUNT_DRAWDOWN_STOP" if stopped else "LOCKED_EXIT"
        elif active is not None:
            failure = candidate.exit_decision(active, row, i, policy)
            if failure:
                desired, reason = 0, failure
            elif np.isfinite(es):
                desired = budget.cap_quantity(float(row.close), budget.limits(nav, peak, es), q, existing=q)
                reason = "PRIOR_CLOSE_RISK_REDUCTION" if desired < q else "KEEP"
        elif entry:
            if np.isfinite(es) and np.isfinite(row[prefix+"_setup_low"]) and np.isfinite(row.atr14):
                planned = int(.5*nav/float(row.close)//100)*100
                desired = budget.cap_quantity(float(row.close), budget.limits(nav, peak, es), planned)
                reason = prefix.upper()+"_PRICE_ACCEPTED_CLOSE_ENTRY"
            else:
                desired, reason = 0, "NO_VIEW_ENTRY_RISK_OR_ANCHOR"
        else:
            desired, reason = 0, "NO_ENTRY_EVENT"
        if q and desired == 0:
            locked_exit = True
        result = {"origin": row.date, "execution_date": d.date.iloc[i+1] if i+1 < len(d) else pd.NaT,
            "desired_shares": desired, "shares_before": q, "known_es95": es, "reason": reason,
            "entry_type": prefix.upper()+"_ACCEPTED" if entry else "NONE", "entry_event": entry,
            "atr_at_origin": float(row.atr14), "support_at_origin": float(row[prefix+"_setup_low"]),
            "setup_date": row[prefix+"_setup_date"], "setup_source_ids": row[prefix+"_setup_source_ids"],
            "holding_stage": active["holding_stage"] if active else "FLAT",
            "structural_stop": active["structural_stop"] if active else np.nan,
            "reverse_information": bool(row.reverse_operation_arrived) if prefix == "support" else False,
            "complete_week_last": row.complete_week_last}
        decisions.append(result)
        return result

    pending = decide(first-1)
    for i in range(first, len(d)):
        row = d.iloc[i]
        old_q, accrued, paid, commission, slippage = q, 0., 0., 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == row.date and k in entitlements:
                eligible, owner = entitlements[k]
                amount = eligible*event["cash_dividend_per_share"]
                if amount:
                    outstanding[k] = amount
                    receivable += amount
                    accrued += amount
                    owner["dividend_cny"] += amount
        desired = int(pending["desired_shares"])
        side = int(np.sign(desired-q))
        if side and open_blocked(d, i, side):
            rejections.append({"date": row.date, "origin": pending["origin"], "reason": "OPEN_LIMIT_OR_MISSING_PRICE", "entry_type": pending["entry_type"]})
        elif side < 0:
            if active is None or i <= active["entry_idx"]:
                raise AssertionError("卖出违反T+1或缺少原持仓。")
            sold = q-desired
            px = fill(float(row.open), -1, cost)
            charge = fee(sold*px, cost)
            cash += sold*px-charge
            q = desired
            active["sell_net_cny"] += sold*px-charge
            active["sell_fees"] += charge
            commission += charge
            slippage += sold*(float(row.open)-px)
            orders.append({"date": row.date, "origin": pending["origin"], "cycle_id": active["cycle_id"], "source": active["source"],
                "side": "SELL", "quantity": sold, "raw_open": row.open, "fill_price": px, "commission": charge,
                "slippage": sold*(float(row.open)-px), "reason": pending["reason"], "holding_stage": active["holding_stage"]})
            if q == 0:
                active.update(exit_date=row.date, exit_idx=i, exit_reason=pending["reason"], status="COMPLETE", holding_sessions=i-active["entry_idx"])
                active, locked_exit = None, False
        elif side > 0:
            if q:
                raise AssertionError("本用途不允许持仓补买。")
            if float(row.open+row.cash_shift) <= pending["support_at_origin"]+EPS:
                rejections.append({"date": row.date, "origin": pending["origin"], "reason": "OPEN_ALREADY_BELOW_KNOWN_ANCHOR", "entry_type": pending["entry_type"]})
            else:
                px = fill(float(row.open), 1, cost)
                nav_open = cash+receivable
                limits = budget.limits(nav_open, peak, pending["known_es95"])
                extra = budget.cap_quantity(float(row.open), limits, desired)
                while extra:
                    debit = extra*px+fee(extra*px, cost)
                    if debit <= cash+1e-8 and budget.risk_ok(extra, float(row.open), limits, buying=True):
                        break
                    extra -= 100
                if extra:
                    charge = fee(extra*px, cost)
                    debit = extra*px+charge
                    cash -= debit
                    q = extra
                    active = {"cycle_id": len(cycles)+1, "source": prefix.upper()+"_ACCEPTED", "entry_origin": pending["origin"],
                        "entry_date": row.date, "entry_idx": i, "entry_raw": float(row.open), "entry_price": px, "entry_quantity": extra,
                        "entry_equity": nav_open, "buy_debit": debit, "entry_fee": charge, "sell_net_cny": 0., "sell_fees": 0., "dividend_cny": 0.,
                        "exit_date": pd.NaT, "status": "RIGHT_CENSORED", "holding_stage": "EARLY", "promotion_date": pd.NaT,
                        "structural_stop": pending["support_at_origin"], "setup_date": pending["setup_date"], "setup_source_ids": pending["setup_source_ids"],
                        "fixed_stop": float(row.open+row.cash_shift-pending["atr_at_origin"]),
                        "fixed_target": float(row.open+row.cash_shift+2*pending["atr_at_origin"])}
                    cycles.append(active)
                    commission += charge
                    slippage += extra*(px-float(row.open))
                    orders.append({"date": row.date, "origin": pending["origin"], "cycle_id": active["cycle_id"], "source": active["source"],
                        "side": "BUY", "quantity": extra, "raw_open": row.open, "fill_price": px, "commission": charge,
                        "slippage": extra*(px-float(row.open)), "reason": pending["reason"], "known_es95": pending["known_es95"], "planned_quantity": desired,
                        "setup_date": pending["setup_date"], "setup_source_ids": pending["setup_source_ids"]})
                else:
                    rejections.append({"date": row.date, "origin": pending["origin"], "reason": "CASH_OR_RISK_BELOW_ONE_LOT", "entry_type": pending["entry_type"]})
        for k, event in enumerate(events):
            if event["record_date"] == row.date and q:
                entitlements[k] = (q, active)
            if event["payment_date"] <= row.date and k in outstanding:
                amount = outstanding.pop(k)
                cash += amount
                receivable -= amount
                paid += amount
        nav = cash+receivable+q*float(row.close)
        peak = max(peak, nav)
        dd = 1-nav/peak
        stopped = stopped or dd >= .1
        price_pnl = old_q*(float(row.open)-float(d.close.iloc[i-1]))+q*(float(row.close)-float(row.open))
        error = nav-previous_nav-price_pnl-accrued+commission+slippage
        if abs(error) > 1e-6 or cash < -1e-6 or receivable < -1e-6 or q % 100:
            raise AssertionError("支持接受账户现金、库存或财富恒等式失败。")
        daily.append({"date": row.date, "equity": nav, "cash": cash, "shares": q, "close": row.close, "receivable": receivable,
            "net_return": nav/previous_nav-1, "price_pnl": price_pnl, "dividend_accrual": accrued, "dividend_paid": paid,
            "commission": commission, "slippage": slippage, "exposure": q*row.close/nav, "drawdown": dd, "accounting_error": error,
            "risk_stopped": stopped, "source": active["source"] if active else "FLAT", "holding_stage": active["holding_stage"] if active else "FLAT"})
        previous_nav = nav
        pending = decide(i)
    for trade in cycles:
        trade["net_pnl"] = trade["sell_net_cny"]+trade["dividend_cny"]-trade["buy_debit"] if trade["status"] == "COMPLETE" else np.nan
        trade["net_return"] = trade["net_pnl"]/trade["buy_debit"] if trade["status"] == "COMPLETE" else np.nan
    trade_columns = ["cycle_id", "source", "entry_origin", "entry_date", "exit_date", "status", "net_pnl", "net_return"]
    terminal = {"open_shares": q, "stopped": stopped, "unpaid_dividend_cny": receivable, "planned_next_close_request": pending,
        "open_source": active["source"] if active else None,
        "open_pnl_cny": q*float(d.close.iloc[-1])+active["sell_net_cny"]+active["dividend_cny"]-active["buy_debit"] if active else 0.,
        "estimated_future_exit_friction": q*(float(d.close.iloc[-1])-fill(float(d.close.iloc[-1]), -1, cost))+fee(q*fill(float(d.close.iloc[-1]), -1, cost), cost) if q else 0.,
        "artificial_terminal_liquidation": False}
    return {"daily": pd.DataFrame(daily), "orders": pd.DataFrame(orders), "trades": pd.DataFrame(cycles) if cycles else pd.DataFrame(columns=trade_columns),
            "decisions": pd.DataFrame(decisions), "rejections": pd.DataFrame(rejections), "terminal": terminal}
