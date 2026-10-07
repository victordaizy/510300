"""阶段点位及持有政策：只用当前已形成的价量与已公布宏观。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import daily_supply_test_v1 as budget
from research.point_account_nr7_inputs_v1 import fill, fee, open_blocked

POLICIES = ("STAGE_ENTRY_AND_EXIT", "SAME_ENTRY_FIXED_EXIT", "RAW_ENTRY_SAME_EXIT")
NAMES = {POLICIES[0]: "分阶段进入与持有", POLICIES[1]: "同进入条件固定边界退出", POLICIES[2]: "不分阶段进入同持有退出"}
EPS = 1e-10


def observations(frame):
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date).astype("datetime64[ns]")
    if not d.date.is_unique or not d.date.is_monotonic_increasing:
        raise ValueError("阶段政策日历未排序或重复。")
    d["ema20"] = d.ac.ewm(span=20, adjust=False, min_periods=20).mean()
    d["daily_hist"] = d.daily_hist_atr * d.atr14
    d["weekly_hist"] = d.weekly_hist_atr * d.atr14 * np.sqrt(5)
    d["cash_low"] = d.low + d.cash_shift
    d["price_high20"] = d.ac.shift().rolling(20, min_periods=20).max()
    economic = np.log((d.close + d.dividend) / d.close.shift())
    d["volatility20_60"] = economic.rolling(20, min_periods=20).std(ddof=1) / economic.rolling(60, min_periods=60).std(ddof=1)
    periods = d.date.dt.to_period("W-FRI")
    weeks = d.groupby(periods).agg(week_low=("cash_low", "min"), week_last=("date", "last"))
    weeks["available"] = weeks.index.to_timestamp(how="end").normalize() + pd.Timedelta(days=1)
    w = pd.merge_asof(d[["date"]], weeks.reset_index(drop=True).sort_values("available"),
                      left_on="date", right_on="available", direction="backward")
    d["previous_week_low"] = w.week_low
    d["support_week_last_date"] = w.week_last
    d["support_available_at"] = w.available
    valid = d.support_week_last_date.notna()
    if not d.loc[valid, "support_week_last_date"].lt(d.loc[valid, "date"]).all():
        raise ValueError("结构位包含本周尚未完成低价。")
    price_known = np.isfinite(d[["ac", "ema20", "daily_hist", "atr14", "previous_week_low"]].to_numpy(float)).all(axis=1)
    weekly_known = d.weekly_hist.notna() & d.weekly_last_date.lt(d.date)
    d["raw_repair_condition"] = price_known & d.ac.gt(d.ema20 + EPS) & d.daily_hist.gt(0)
    d["raw_repricing_condition"] = (price_known & d.price_high20.notna() & d.ac.gt(d.price_high20 + EPS)
                                     & d.log_relative_volume.ge(np.log(1.5)) & d.daily_hist.gt(0))
    d["raw_pullback_condition"] = (price_known & d.ac.gt(d.ema20 + EPS) & d.ac.shift().le(d.ema20.shift() + EPS)
                                    & d.daily_hist.gt(d.daily_hist.shift() + EPS))
    funding_valid = d.funding_known.eq(True) & d.funding_gap_change5.notna()
    orders_valid = d.orders_known.eq(True) & d.pmi_orders_change.notna()
    d["funding_repair_support"] = funding_valid & d.funding_gap_pp.le(0) & d.funding_gap_change5.lt(0)
    d["orders_repair_support"] = orders_valid & d.pmi_orders_change.gt(0)
    d["repair_macro_support"] = d.funding_repair_support | d.orders_repair_support
    margin_valid = d.margin_known.eq(True) & d.financing_net_change5.notna()
    d["stage_repair_condition"] = (d.raw_repair_condition & weekly_known & d.weekly_hist.le(0)
                                    & d.volatility20_60.le(1) & d.repair_macro_support)
    d["stage_repricing_condition"] = d.raw_repricing_condition
    d["stage_pullback_condition"] = (d.raw_pullback_condition & weekly_known & d.weekly_hist.gt(0)
                                     & margin_valid & d.financing_net_change5.ge(0))
    for prefix in ("raw", "stage"):
        for kind in ("repair", "repricing", "pullback"):
            name = f"{prefix}_{kind}_condition"
            d[f"{prefix}_{kind}_event"] = d[name] & ~d[name].shift(1, fill_value=False)
        d[f"{prefix}_entry_type"] = np.select(
            [d[f"{prefix}_repricing_event"], d[f"{prefix}_pullback_event"], d[f"{prefix}_repair_event"]],
            ["REPRICING", "PULLBACK", "REPAIR"], default="NONE")
    d["trend_confirmation"] = (weekly_known & d.weekly_hist.gt(0) & d.ac.gt(d.ema20 + EPS)
                                & margin_valid & d.financing_net_change5.ge(0))
    d["funding_distribution"] = (funding_valid & d.funding_gap_pp.gt(0) & d.funding_gap_change5.gt(0)
                                  & margin_valid & d.financing_net_change5.lt(0) & d.ac.lt(d.ema20 - EPS))
    return d


def exit_decision(active, row, i, policy):
    """收盘决定退出；持有状态只在实际持仓内更新，不按未来行情回写。"""
    if policy == "SAME_ENTRY_FIXED_EXIT":
        if row.ac <= active["fixed_stop"] + EPS:
            return "FIXED_LOSS_CLOSE"
        if row.ac >= active["fixed_target"] - EPS:
            return "FIXED_PROFIT_CLOSE"
        return "FIXED_TWENTY_CLOSES" if i - active["entry_idx"] + 1 >= 20 else None
    if row.ac <= active["structural_stop"] + EPS:
        return "KNOWN_STRUCTURAL_LOW_FAILED"
    if active["holding_stage"] == "EARLY" and bool(row.trend_confirmation):
        active["holding_stage"] = "TREND"
        active["promotion_date"] = row.date
    if active["holding_stage"] == "TREND":
        if np.isfinite(row.previous_week_low):
            active["structural_stop"] = max(active["structural_stop"], float(row.previous_week_low))
        if row.ac <= active["structural_stop"] + EPS:
            return "TREND_WEEKLY_LOW_FAILED"
        if np.isfinite(row.weekly_hist) and row.weekly_hist <= 0:
            return "COMPLETED_WEEK_TREND_FAILED"
    elif np.isfinite(row.daily_hist) and row.daily_hist < 0 and row.ac < row.ema20 - EPS:
        return "EARLY_PRICE_MOMENTUM_FAILED"
    if bool(row.funding_distribution):
        return "KNOWN_FUNDING_AND_PRICE_DETERIORATION"
    return None


def account(data, dividends, risks, policy, cost, start):
    if policy not in POLICIES:
        raise ValueError("未知阶段政策。")
    d = data.reset_index(drop=True)
    first = int(np.flatnonzero(d.date.ge(pd.Timestamp(start)))[0])
    if first < 1:
        raise ValueError("缺少首个开盘之前的决定。")
    risk = dict(zip(risks.idx.astype(int), risks.es95))
    cash, q, receivable, peak, previous_nav = 200000., 0, 0., 200000., 200000.
    active, stopped, locked_exit = None, False, False
    entitlements, outstanding, cycles = {}, {}, []
    events = dividends.to_dict("records")
    daily, orders, rejections, decisions = [], [], [], []

    def decide(i):
        nonlocal locked_exit
        row = d.iloc[i]
        nav = cash + receivable + q * float(row.close)
        es = risk.get(i, np.nan)
        prefix = "raw" if policy == "RAW_ENTRY_SAME_EXIT" else "stage"
        kind = row[f"{prefix}_entry_type"]
        desired, reason = q, "KEEP"
        if stopped or locked_exit:
            desired, reason = 0, "ACCOUNT_DRAWDOWN_STOP" if stopped else "LOCKED_EXIT"
        elif active is not None:
            failure = exit_decision(active, row, i, policy)
            if failure:
                desired, reason = 0, failure
            elif np.isfinite(es):
                desired = budget.cap_quantity(float(row.close), budget.limits(nav, peak, es), q, existing=q)
                reason = "PRIOR_CLOSE_RISK_REDUCTION" if desired < q else "KEEP"
        elif kind != "NONE":
            if np.isfinite(es) and np.isfinite(row.previous_week_low) and np.isfinite(row.atr14):
                planned = int(.5 * nav / float(row.close) // 100) * 100
                desired = budget.cap_quantity(float(row.close), budget.limits(nav, peak, es), planned)
                reason = f"{kind}_KNOWN_CLOSE_ENTRY"
            else:
                desired, reason = 0, "NO_VIEW_ENTRY_RISK_OR_SUPPORT"
        else:
            desired, reason = 0, "NO_ENTRY_EVENT"
        if q and desired == 0:
            locked_exit = True
        result = {"origin": row.date, "execution_date": d.date.iloc[i+1] if i+1 < len(d) else pd.NaT,
                  "desired_shares": desired, "shares_before": q, "known_es95": es, "reason": reason,
                  "entry_type": kind, "atr_at_origin": float(row.atr14), "support_at_origin": float(row.previous_week_low),
                  "holding_stage": active["holding_stage"] if active else "FLAT",
                  "structural_stop": active["structural_stop"] if active else np.nan,
                  "orders_known": bool(row.orders_known), "funding_known": bool(row.funding_known), "margin_known": bool(row.margin_known)}
        decisions.append(result)
        return result

    pending = decide(first-1)
    for i in range(first, len(d)):
        row = d.iloc[i]
        old_q, accrued, paid, commission, slippage = q, 0., 0., 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == row.date and k in entitlements:
                eligible, owner = entitlements[k]
                amount = eligible * event["cash_dividend_per_share"]
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
                raise AssertionError("阶段政策不能持仓补买。")
            if float(row.open+row.cash_shift) <= pending["support_at_origin"] + EPS:
                rejections.append({"date": row.date, "origin": pending["origin"], "reason": "OPEN_ALREADY_BELOW_KNOWN_SUPPORT", "entry_type": pending["entry_type"]})
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
                    active = {"cycle_id": len(cycles)+1, "source": pending["entry_type"], "entry_origin": pending["origin"],
                              "entry_date": row.date, "entry_idx": i, "entry_raw": float(row.open), "entry_price": px, "entry_quantity": extra,
                              "entry_equity": nav_open, "buy_debit": debit, "entry_fee": charge, "sell_net_cny": 0., "sell_fees": 0., "dividend_cny": 0.,
                              "exit_date": pd.NaT, "status": "RIGHT_CENSORED", "holding_stage": "EARLY", "promotion_date": pd.NaT,
                              "structural_stop": pending["support_at_origin"],
                              "fixed_stop": float(row.open+row.cash_shift-pending["atr_at_origin"]),
                              "fixed_target": float(row.open+row.cash_shift+2*pending["atr_at_origin"])}
                    cycles.append(active)
                    commission += charge
                    slippage += extra*(px-float(row.open))
                    orders.append({"date": row.date, "origin": pending["origin"], "cycle_id": active["cycle_id"], "source": active["source"],
                                   "side": "BUY", "quantity": extra, "raw_open": row.open, "fill_price": px, "commission": charge,
                                   "slippage": extra*(px-float(row.open)), "reason": pending["reason"], "known_es95": pending["known_es95"], "planned_quantity": desired})
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
            raise AssertionError("阶段账户现金、库存或财富恒等式失败。")
        daily.append({"date": row.date, "equity": nav, "cash": cash, "shares": q, "close": row.close, "receivable": receivable,
                      "net_return": nav/previous_nav-1, "price_pnl": price_pnl, "dividend_accrual": accrued, "dividend_paid": paid,
                      "commission": commission, "slippage": slippage, "exposure": q*row.close/nav, "drawdown": dd, "accounting_error": error,
                      "risk_stopped": stopped, "source": active["source"] if active else "FLAT", "holding_stage": active["holding_stage"] if active else "FLAT"})
        previous_nav = nav
        pending = decide(i)
    for trade in cycles:
        trade["net_pnl"] = trade["sell_net_cny"]+trade["dividend_cny"]-trade["buy_debit"] if trade["status"] == "COMPLETE" else np.nan
        trade["net_return"] = trade["net_pnl"]/trade["buy_debit"] if trade["status"] == "COMPLETE" else np.nan
    columns = ["cycle_id", "source", "entry_origin", "entry_date", "exit_date", "status", "net_pnl", "net_return"]
    terminal = {"open_shares": q, "stopped": stopped, "unpaid_dividend_cny": receivable, "planned_next_close_request": pending,
                "open_source": active["source"] if active else None, "open_pnl_cny": q*float(d.close.iloc[-1])+active["sell_net_cny"]+active["dividend_cny"]-active["buy_debit"] if active else 0.,
                "estimated_future_exit_friction": q*(float(d.close.iloc[-1])-fill(float(d.close.iloc[-1]), -1, cost))+fee(q*fill(float(d.close.iloc[-1]), -1, cost), cost) if q else 0.,
                "artificial_terminal_liquidation": False}
    return {"daily": pd.DataFrame(daily), "orders": pd.DataFrame(orders), "trades": pd.DataFrame(cycles) if cycles else pd.DataFrame(columns=columns),
            "decisions": pd.DataFrame(decisions), "rejections": pd.DataFrame(rejections), "terminal": terminal}
