"""同一风险预算下，比较保存的仓位强弱与二值点位；不重新训练信号。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import daily_supply_test_v1 as risk_base
from research.point_account_nr7_inputs_v1 import fill, fee, open_blocked, PARENT_A


def weight_account(data, dividends, parents, risks, cost, start):
    d = data.reset_index(drop=True)
    parent = parents.set_index("origin")[PARENT_A]
    first = int(np.flatnonzero(d.date.ge(pd.Timestamp(start)))[0])
    if first < 1 or not pd.DatetimeIndex(d.date.iloc[first - 1:-1]).isin(parent.index).all():
        raise ValueError("父权重没有覆盖所有开盘之前的决策日。")
    risk = dict(zip(risks.idx.astype(int), risks.es95))
    cash, q, receivable, peak, previous_nav = 200000., 0, 0., 200000., 200000.
    active, stopped, locked_exit = None, False, False
    entitlements, outstanding, cycles = {}, {}, []
    events = dividends.to_dict("records")
    daily, orders, rejections, decisions = [], [], [], []

    def decide(i):
        nonlocal locked_exit
        raw = float(d.close.iloc[i])
        target = float(parent.get(d.date.iloc[i], np.nan))
        nav = cash + receivable + q * raw
        es = risk.get(i, np.nan)
        reason = "KEEP_WITHIN_ORIGINAL_TEN_POINT_BAND"
        if stopped or locked_exit or target == 0:
            desired = 0
            reason = "ACCOUNT_DRAWDOWN_STOP" if stopped else "LOCKED_EXIT" if locked_exit else "PARENT_ZERO"
        elif not np.isfinite(target) or not np.isfinite(es):
            desired, reason = q, "UNKNOWN_KEEP_EXISTING"
        else:
            if not 0 <= target <= 1:
                raise ValueError("来源目标超出0至1。")
            target = min(.5, target)
            if q == 0 or abs(target - q * raw / nav) >= .1:
                desired = int(target * nav / raw // 100) * 100
                reason = "SAVED_WEIGHT_REBALANCE"
            else:
                desired = q
            cap = risk_base.cap_quantity(raw, risk_base.limits(nav, peak, es), desired, existing=q)
            if cap < desired:
                desired, reason = cap, "PRIOR_CLOSE_RISK_CAP"
        if q and desired == 0:
            locked_exit = True
        row = {"origin": d.date.iloc[i], "execution_date": d.date.iloc[i + 1] if i + 1 < len(d) else pd.NaT,
               "source_weight": target, "shares_before": q, "desired_shares": desired,
               "known_es95": es, "reason": reason}
        decisions.append(row)
        return row

    pending = decide(first - 1)
    for i in range(first, len(d)):
        r = d.iloc[i]
        old_q, accrued, paid, commission, slippage = q, 0., 0., 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == r.date and k in entitlements:
                eligible, owner = entitlements[k]
                amount = eligible * event["cash_dividend_per_share"]
                if amount:
                    outstanding[k] = amount
                    receivable += amount
                    accrued += amount
                    owner["dividend_cny"] += amount
        desired = int(pending["desired_shares"])
        side = int(np.sign(desired - q))
        if side and open_blocked(d, i, side):
            rejections.append({"date": r.date, "origin": pending["origin"], "reason": "OPEN_LIMIT_OR_MISSING_PRICE"})
        elif side < 0:
            sold = q - desired
            px = fill(float(r.open), -1, cost)
            charge = fee(sold * px, cost)
            cash += sold * px - charge
            q = desired
            active["sell_net_cny"] += sold * px - charge
            active["sell_fees"] += charge
            commission += charge
            slippage += sold * (float(r.open) - px)
            orders.append({"date": r.date, "origin": pending["origin"], "cycle_id": active["cycle_id"],
                           "source": "CORE_WEIGHT", "side": "SELL", "quantity": sold,
                           "raw_open": r.open, "fill_price": px, "commission": charge,
                           "slippage": sold * (float(r.open) - px), "reason": pending["reason"]})
            if q == 0:
                active.update(exit_date=r.date, exit_idx=i, exit_reason=pending["reason"], status="COMPLETE",
                              holding_sessions=i - active["entry_idx"])
                active, locked_exit = None, False
        elif side > 0:
            px = fill(float(r.open), 1, cost)
            nav_open = cash + receivable + q * float(r.open)
            budget = risk_base.limits(nav_open, peak, pending["known_es95"])
            # 新开盘只能减少前一晚计划；所有仓位一起接受同一风险上限检查。
            total = risk_base.cap_quantity(float(r.open), budget, desired)
            extra = max(0, total - q)
            while extra:
                debit = extra * px + fee(extra * px, cost)
                if debit <= cash + 1e-8 and risk_base.risk_ok(q + extra, float(r.open), budget, buying=True):
                    break
                extra -= 100
            if extra:
                charge = fee(extra * px, cost)
                debit = extra * px + charge
                cash -= debit
                q += extra
                if active is None:
                    active = {"cycle_id": len(cycles) + 1, "source": "CORE_WEIGHT", "entry_origin": pending["origin"],
                              "entry_date": r.date, "entry_idx": i, "entry_raw": float(r.open),
                              "entry_price": px, "entry_quantity": extra, "entry_equity": nav_open,
                              "buy_debit": 0., "entry_fee": 0., "sell_net_cny": 0., "sell_fees": 0.,
                              "dividend_cny": 0., "exit_date": pd.NaT, "status": "RIGHT_CENSORED"}
                    cycles.append(active)
                active["buy_debit"] += debit
                active["entry_fee"] += charge
                commission += charge
                slippage += extra * (px - float(r.open))
                orders.append({"date": r.date, "origin": pending["origin"], "cycle_id": active["cycle_id"],
                               "source": "CORE_WEIGHT", "side": "BUY", "quantity": extra,
                               "raw_open": r.open, "fill_price": px, "commission": charge,
                               "slippage": extra * (px - float(r.open)), "reason": pending["reason"],
                               "known_es95": pending["known_es95"], "planned_total_quantity": desired})
            else:
                rejections.append({"date": r.date, "origin": pending["origin"], "reason": "CASH_OR_OPEN_RISK_CAP"})
        for k, event in enumerate(events):
            if event["record_date"] == r.date and q:
                entitlements[k] = (q, active)
            if event["payment_date"] <= r.date and k in outstanding:
                amount = outstanding.pop(k)
                cash += amount
                receivable -= amount
                paid += amount
        nav = cash + receivable + q * float(r.close)
        peak = max(peak, nav)
        dd = 1 - nav / peak
        stopped = stopped or dd >= .1
        price_pnl = old_q * (float(r.open) - float(d.close.iloc[i - 1])) + q * (float(r.close) - float(r.open))
        error = nav - previous_nav - price_pnl - accrued + commission + slippage
        if abs(error) > 1e-6 or cash < -1e-6 or receivable < -1e-6 or q % 100:
            raise AssertionError("仓位强弱账户的财富恒等式失败。")
        daily.append({"date": r.date, "equity": nav, "cash": cash, "shares": q, "close": r.close,
                      "receivable": receivable, "net_return": nav / previous_nav - 1, "price_pnl": price_pnl,
                      "dividend_accrual": accrued, "dividend_paid": paid, "commission": commission, "slippage": slippage,
                      "exposure": q * r.close / nav, "drawdown": dd, "accounting_error": error,
                      "risk_stopped": stopped, "source": "CORE_WEIGHT" if q else "FLAT"})
        previous_nav = nav
        pending = decide(i)
    for t in cycles:
        if t["status"] == "COMPLETE":
            t["net_pnl"] = t["sell_net_cny"] + t["dividend_cny"] - t["buy_debit"]
            t["net_return"] = t["net_pnl"] / t["buy_debit"]
        else:
            t["net_pnl"], t["net_return"] = np.nan, np.nan
    trade_columns = ["cycle_id", "source", "entry_origin", "entry_date", "exit_date", "status", "net_pnl", "net_return"]
    return {"daily": pd.DataFrame(daily), "trades": pd.DataFrame(cycles) if cycles else pd.DataFrame(columns=trade_columns), "orders": pd.DataFrame(orders),
            "decisions": pd.DataFrame(decisions), "rejections": pd.DataFrame(rejections),
            "terminal": {"open_shares": q, "stopped": stopped, "unpaid_dividend_cny": receivable,
                         "planned_next_close_request": pending, "artificial_terminal_liquidation": False}}
