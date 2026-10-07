"""累计量先行、价格后确认的隔离账户；固定低点或参考量价共同失效。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from research import daily_supply_test_v1 as risk_base
from research.full_daily_gap_inputs_v1 import quote_tick
from research.point_account_nr7_inputs_v1 import PARENT_A, fill, fee, open_blocked
MODES = ("A_CONTROL", "VOLUME_PRICE_ONLY", "PRICE_ONLY")


def account(data, dividends, parents, risks, signal_rows, cost, start, mode):
    """量先恢复后的价格确认次开进；固定低点或参考量价共同失效次开出。原账户口径保持。"""
    if mode not in MODES:
        raise ValueError("未知固定账户用途。")
    d = data.reset_index(drop=True)
    sig = signal_rows.reset_index(drop=True)
    if not pd.DatetimeIndex(d.date).equals(pd.DatetimeIndex(sig.date)):
        raise ValueError("量价先后确认状态与原日期不一致。")
    parent = parents.set_index("origin")[PARENT_A]
    first = int(np.flatnonzero(d.date.ge(pd.Timestamp(start)))[0])
    if first < 1 or not pd.DatetimeIndex(d.date.iloc[first - 1:-1]).isin(parent.index).all():
        raise ValueError("原A权重没有覆盖所有开盘前决定。")
    risk = dict(zip(risks.idx.astype(int), risks.es95))
    known_cash_shift_ticks = sig.cash_shift_ticks.to_numpy(np.int64)
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
        elif active is not None and active["source"] in ("VOLUME_LEAD_PRICE_CONFIRM", "PRICE_CONFIRMATION"):
            if active["source"] == "VOLUME_LEAD_PRICE_CONFIRM":
                known = bool(event.current_quote_known)
                floor_failed = bool(known and int(event.known_cash_close_ticks) <= active["fixed_volume_floor_ticks"])
                joint_failed = bool(known and bool(event.cumulative_volume_known)
                                    and int(event.cumulative_signed_volume) <= active["fixed_reference_volume"]
                                    and int(event.known_cash_close_ticks) <= active["fixed_reference_high_ticks"])
                invalid = bool(floor_failed or joint_failed)
            else:
                invalid = float(d.ac.iloc[i]) <= float(d.ema20.iloc[i])
            if invalid:
                reason = ("VOLUME_LEAD_PRICE_CONFIRM_FIXED_LOW_FAILED_CLOSE" if floor_failed else "VOLUME_LEAD_PRICE_CONFIRM_REFERENCE_VOLUME_AND_PRICE_FAILED_CLOSE") if active["source"] == "VOLUME_LEAD_PRICE_CONFIRM" else "REPAIR_PRICE_CONFIRMATION_FAILED_CLOSE"
                desired = 0
            elif np.isfinite(es):
                desired = risk_base.cap_quantity(raw, risk_base.limits(nav, peak, es), q, existing=q)
                reason = "REPAIR_ORIGINAL_RISK_REDUCTION" if desired < q else "REPAIR_KEEP_CONFIRMED_PRICE"
            else:
                desired, reason = q, "UNKNOWN_KEEP_EXISTING"
        else:
            native_request = q == 0 and mode != "A_CONTROL" and bool(event.entry_event)
            core_enabled = mode == "A_CONTROL"
            if native_request:
                source = "VOLUME_LEAD_PRICE_CONFIRM" if mode == "VOLUME_PRICE_ONLY" else "PRICE_CONFIRMATION"
                if np.isfinite(es):
                    desired = int(.5 * nav / raw // 100) * 100
                    desired = risk_base.cap_quantity(raw, risk_base.limits(nav, peak, es), desired)
                    reason = "VOLUME_LEAD_PRICE_CONFIRM_ENTRY" if mode == "VOLUME_PRICE_ONLY" else "REPAIR_PRICE_CONFIRMATION_ENTRY"
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
               "stop_index": float(event.stop_index), "target_index": float(event.target_index),
               "atr_at_origin": float(event.atr),
               "entry_known_fit_index": float(event.get("fit_index", np.nan)),
               "predicted_p_times_b": float(event.get("predicted_p_times_b", np.nan)),
               "predicted_net_expectation": float(event.get("predicted_net_expectation", np.nan)),
               "known_under_start": int(event.get("known_under_start", -1)),
               "current_volume_positive_start": int(event.get("current_volume_positive_start", -1)),
               "current_macd_positive_start": int(event.get("current_macd_positive_start", -1)),
               "known_reference_status": event.get("reference_status", "NO_VIEW"),
               "known_reference_anchor_id": None if pd.isna(event.get("reference_anchor_id", None)) else str(event.get("reference_anchor_id")),
               "volume_floor_ticks": int(event.get("fixed_low_ticks", -1)),
               "reference_high_ticks": int(event.get("reference_high_ticks", -1)),
               "reference_cumulative_volume": None if pd.isna(event.get("reference_cumulative_volume", None)) else int(event.reference_cumulative_volume),
               "known_current_volume_floor_ticks": active["fixed_volume_floor_ticks"] if active and active["source"] == "VOLUME_LEAD_PRICE_CONFIRM" else -1,
               "known_current_reference_high_ticks": active["fixed_reference_high_ticks"] if active and active["source"] == "VOLUME_LEAD_PRICE_CONFIRM" else -1,
               "known_current_reference_volume": active["fixed_reference_volume"] if active and active["source"] == "VOLUME_LEAD_PRICE_CONFIRM" else None,
               "known_rule_exit": bool(event.get("rule_exit", False)),
               "known_daily_dif": float(event.get("current_daily_dif", np.nan)),
               "known_previous_week_hist": float(event.get("previous_complete_week_hist", np.nan)),
               "known_previous_week_date": event.get("previous_complete_week_date", pd.NaT)}
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
        elif side > 0 and q == 0 and pending["source_request"] == "VOLUME_LEAD_PRICE_CONFIRM" and (
                known_cash_shift_ticks[i] < 0 or quote_tick(float(r.open)) is None
                or quote_tick(float(r.open))+known_cash_shift_ticks[i] <= pending["volume_floor_ticks"]):
            rejection_reason = "NO_VIEW_OPENING_VOLUME_PRICE_COORDINATE" if known_cash_shift_ticks[i] < 0 or quote_tick(float(r.open)) is None else "OPEN_AT_OR_BELOW_KNOWN_VOLUME_FLOOR_CANCELLED"
            rejections.append({"date": r.date, "origin": pending["origin"], "reason": rejection_reason,
                               "source_request": pending["source_request"], "event_id": pending["event_id"]})
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
                              "event_id": pending["event_id"] if pending["source_request"] in ("VOLUME_LEAD_PRICE_CONFIRM", "PRICE_CONFIRMATION") else None,
                              "entry_origin": pending["origin"], "entry_date": r.date, "entry_idx": i,
                              "entry_raw": float(r.open), "entry_price": px, "entry_quantity": extra,
                              "entry_equity": nav_open, "buy_debit": 0., "entry_fee": 0.,
                              "sell_net_cny": 0., "sell_fees": 0., "dividend_cny": 0.,
                              "exit_date": pd.NaT, "status": "RIGHT_CENSORED",
                              "stop_index": pending["stop_index"] if pending["source_request"] == "VOLUME_LEAD_PRICE_CONFIRM" else np.nan,
                              "entry_stop_index": pending["stop_index"] if pending["source_request"] == "VOLUME_LEAD_PRICE_CONFIRM" else np.nan,
                              "entry_volume_floor_ticks": pending["volume_floor_ticks"] if pending["source_request"] == "VOLUME_LEAD_PRICE_CONFIRM" else -1,
                              "fixed_volume_floor_ticks": pending["volume_floor_ticks"] if pending["source_request"] == "VOLUME_LEAD_PRICE_CONFIRM" else -1,
                              "fixed_reference_high_ticks": pending["reference_high_ticks"] if pending["source_request"] == "VOLUME_LEAD_PRICE_CONFIRM" else -1,
                              "fixed_reference_volume": pending["reference_cumulative_volume"] if pending["source_request"] == "VOLUME_LEAD_PRICE_CONFIRM" else None,
                              "target_index": np.nan,
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
            raise AssertionError("量价先后确认共同账户的资金、库存或财富恒等式失败。")
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
