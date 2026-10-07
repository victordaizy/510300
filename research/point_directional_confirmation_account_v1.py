"""原5%方向变化定义的新完整交易用途：事前确认入场、反向确认退出。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import daily_supply_test_v1 as risk_base
from research.point_account_nr7_inputs_v1 import PARENT_A, fill, fee, open_blocked

MODES = ("A_CONTROL", "DIRECTIONAL_ONLY", "A_PLUS_DIRECTIONAL")
THRESHOLD = .05


def signals(data: pd.DataFrame) -> pd.DataFrame:
    """极值仅用于当日确认，事件日期是确认日；不将信号追溯到极值。"""
    d = data.reset_index(drop=True)
    if len(d) == 0 or not d.date.is_monotonic_increasing or not d.date.is_unique:
        raise ValueError("方向确认需要唯一递增的完整日线。")
    close, dividend = d.close.to_numpy(float), d.dividend.to_numpy(float)
    if not np.isfinite(close).all() or not np.isfinite(dividend).all() or (close <= 0).any():
        raise ValueError("方向确认价格或股息缺失，不能填零或跳过。")
    growth = np.ones(len(d))
    growth[1:] = (close[1:] + dividend[1:]) / close[:-1]
    if (growth <= 0).any():
        raise ValueError("含息收益不能构成正的财富指数。")
    wealth = np.cumprod(growth)
    mode, lo, hi, last_confirmation = 0, 0, 0, -1
    rows = []
    for i, value in enumerate(wealth):
        turn, extreme = 0, np.nan
        if mode == 0:
            if value < wealth[lo]:
                lo = i
            if value > wealth[hi]:
                hi = i
            if value >= wealth[lo] * (1 + THRESHOLD):
                mode, turn, extreme, hi = 1, 1, lo, i
            elif value <= wealth[hi] * (1 - THRESHOLD):
                mode, turn, extreme, lo = -1, -1, hi, i
        elif mode == 1:
            if value > wealth[hi]:
                hi = i
            if value <= wealth[hi] * (1 - THRESHOLD):
                mode, turn, extreme, lo = -1, -1, hi, i
        else:
            if value < wealth[lo]:
                lo = i
            if value >= wealth[lo] * (1 + THRESHOLD):
                mode, turn, extreme, hi = 1, 1, lo, i
        if turn:
            last_confirmation = i
        row = {"date": d.date.iloc[i], "origin_index": i, "wealth_index": value,
               "direction_state": mode, "entry_event": turn == 1, "exit_event": turn == -1,
               "event_status": "UP_CONFIRMATION" if turn == 1 else "DOWN_CONFIRMATION" if turn == -1 else "NO_NEW_CONFIRMATION",
               "event_id": f"DC_{turn:+d}_{int(extreme)}_{i}" if turn else None,
               "confirmed_extreme_index": extreme, "confirmation_index": i if turn else np.nan,
               "last_confirmation_index": last_confirmation,
               "stop_index": np.nan, "target_index": np.nan}
        for name in ["daily_hist", "weekly_hist", "relative_volume", "rv_ratio"]:
            row[name] = float(d[name].iloc[i]) if name in d else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def account(data, dividends, parents, risks, signal_rows, cost, start, mode):
    """保留原A执行；独立方向事件下一开盘入场、反向确认下一开盘退出。"""
    if mode not in MODES:
        raise ValueError("未知固定账户用途。")
    d = data.reset_index(drop=True)
    sig = signal_rows.reset_index(drop=True)
    if not pd.DatetimeIndex(d.date).equals(pd.DatetimeIndex(sig.date)):
        raise ValueError("方向确认信号与原日线日期不一致。")
    parent = parents.set_index("origin")[PARENT_A]
    first = int(np.flatnonzero(d.date.ge(pd.Timestamp(start)))[0])
    if first < 1 or not pd.DatetimeIndex(d.date.iloc[first - 1:-1]).isin(parent.index).all():
        raise ValueError("原A权重没有覆盖所有开盘前决定。")
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
        source = active["source"] if active else "NONE"
        event = sig.iloc[i]
        reason = "KEEP_WITHIN_ORIGINAL_TEN_POINT_BAND"
        if stopped or locked_exit:
            desired = 0
            reason = "ACCOUNT_DRAWDOWN_STOP" if stopped else "LOCKED_EXIT"
        elif active is not None and active["source"] == "DIRECTIONAL":
            if int(event.direction_state) == -1:
                desired = 0
                reason = "DIRECTIONAL_DOWN_CONFIRMATION"
            elif np.isfinite(es):
                desired = risk_base.cap_quantity(raw, risk_base.limits(nav, peak, es), q, existing=q)
                reason = "DIRECTIONAL_RISK_REDUCTION" if desired < q else "DIRECTIONAL_KEEP_FIXED_STRUCTURE"
            else:
                desired, reason = q, "UNKNOWN_KEEP_EXISTING"
        else:
            native_request = (q == 0 and mode != "A_CONTROL" and bool(event.entry_event)
                              and (mode == "DIRECTIONAL_ONLY" or target == 0))
            core_enabled = mode != "DIRECTIONAL_ONLY"
            if native_request:
                source = "DIRECTIONAL"
                if np.isfinite(es):
                    desired = int(.5 * nav / raw // 100) * 100
                    desired = risk_base.cap_quantity(raw, risk_base.limits(nav, peak, es), desired)
                    reason = "DIRECTIONAL_UP_CONFIRMATION_ENTRY"
                else:
                    desired, reason = 0, "MISSING_PRIOR_RISK_ESTIMATE"
            elif not core_enabled or target == 0:
                desired, reason = 0, "NO_NEW_EVENT" if not core_enabled else "PARENT_ZERO"
            elif not np.isfinite(target) or not np.isfinite(es):
                desired, reason = q, "UNKNOWN_KEEP_EXISTING"
            else:
                if not 0 <= target <= 1:
                    raise ValueError("原A目标超出0至1。")
                source = "CORE_WEIGHT"
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
               "known_es95": es, "reason": reason, "source_request": source,
               "entry_event": bool(event.entry_event), "event_id": event.event_id,
               "stop_index": float(event.stop_index), "target_index": float(event.target_index)}
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
            rejections.append({"date": r.date, "origin": pending["origin"], "reason": "OPEN_LIMIT_OR_MISSING_PRICE",
                               "source_request": pending["source_request"], "event_id": pending["event_id"]})
        elif side < 0:
            if active is None or i <= active["entry_idx"]:
                raise AssertionError("清仓违反T+1或缺少原持仓。")
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
                           "source": active["source"], "side": "SELL", "quantity": sold,
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
                    active = {"cycle_id": len(cycles) + 1, "source": pending["source_request"],
                              "event_id": pending["event_id"] if pending["source_request"] == "DIRECTIONAL" else None,
                              "entry_origin": pending["origin"], "entry_date": r.date, "entry_idx": i,
                              "entry_raw": float(r.open), "entry_price": px, "entry_quantity": extra,
                              "entry_equity": nav_open, "buy_debit": 0., "entry_fee": 0.,
                              "sell_net_cny": 0., "sell_fees": 0., "dividend_cny": 0.,
                              "exit_date": pd.NaT, "status": "RIGHT_CENSORED",
                              "stop_index": pending["stop_index"] if pending["source_request"] == "DIRECTIONAL" else np.nan,
                              "target_index": pending["target_index"] if pending["source_request"] == "DIRECTIONAL" else np.nan,
                              "entry_net_reward_risk": np.nan}
                    cycles.append(active)
                active["buy_debit"] += debit
                active["entry_fee"] += charge
                commission += charge
                slippage += extra * (px - float(r.open))
                orders.append({"date": r.date, "origin": pending["origin"], "cycle_id": active["cycle_id"],
                               "source": active["source"], "side": "BUY", "quantity": extra,
                               "raw_open": r.open, "fill_price": px, "commission": charge,
                               "slippage": extra * (px - float(r.open)), "reason": pending["reason"],
                               "known_es95": pending["known_es95"], "planned_total_quantity": desired})
            else:
                rejections.append({"date": r.date, "origin": pending["origin"], "reason": "CASH_OR_OPEN_RISK_CAP",
                                   "source_request": pending["source_request"], "event_id": pending["event_id"]})
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
            raise AssertionError("方向确认共同账户的资金、库存或财富恒等式失败。")
        daily.append({"date": r.date, "equity": nav, "cash": cash, "shares": q, "close": r.close,
                      "receivable": receivable, "net_return": nav / previous_nav - 1, "price_pnl": price_pnl,
                      "dividend_accrual": accrued, "dividend_paid": paid, "commission": commission, "slippage": slippage,
                      "exposure": q * r.close / nav, "drawdown": dd, "accounting_error": error,
                      "risk_stopped": stopped, "source": active["source"] if q else "FLAT"})
        previous_nav = nav
        pending = decide(i)
    for trade in cycles:
        if trade["status"] == "COMPLETE":
            trade["net_pnl"] = trade["sell_net_cny"] + trade["dividend_cny"] - trade["buy_debit"]
            trade["net_return"] = trade["net_pnl"] / trade["buy_debit"]
        else:
            trade["net_pnl"], trade["net_return"] = np.nan, np.nan
    trade_columns = ["cycle_id", "source", "entry_origin", "entry_date", "exit_date", "status", "net_pnl", "net_return"]
    return {"daily": pd.DataFrame(daily),
            "trades": pd.DataFrame(cycles) if cycles else pd.DataFrame(columns=trade_columns),
            "orders": pd.DataFrame(orders), "decisions": pd.DataFrame(decisions),
            "rejections": pd.DataFrame(rejections),
            "terminal": {"open_shares": q, "stopped": stopped, "unpaid_dividend_cny": receivable,
                         "planned_next_close_request": pending, "artificial_terminal_liquidation": False}}
