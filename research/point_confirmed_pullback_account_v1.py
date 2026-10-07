"""已确认上涨段缩量回调再转强：固定事件及共同现金账户执行。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import daily_supply_test_v1 as risk_base
from research.point_account_nr7_inputs_v1 import PARENT_A, fill, fee, open_blocked
from research.point_c04_information_intake_v1 import build_field, FIELD

MODES = ("A_CONTROL", "PULLBACK_ONLY", "A_PLUS_PULLBACK")


def signals(data: pd.DataFrame) -> pd.DataFrame:
    """沿用原阶段，首个转强事件只判断一次；成交收缩必须在前日已知。"""
    d = data.reset_index(drop=True)
    stage = build_field(d)
    cash = d.dividend.cumsum().to_numpy(float)
    ac, ah, al = [d[name].to_numpy(float) + cash for name in ["close", "high", "low"]]
    seen = set()
    rows = []
    for i, phase in enumerate(stage.itertuples(index=False)):
        result = {"date": d.date.iloc[i], "origin_index": i, "entry_event": False,
                  "event_id": None, "event_status": "NO_QUALIFIED_PHASE",
                  "phase_start_index": phase.phase_start_index, "low_index": phase.low_index,
                  "high_index": phase.high_index, "stop_index": np.nan, "target_index": np.nan,
                  "known_prior_amount_ratio": np.nan, "raw_planned_reward_risk": np.nan}
        if i and phase.field_status == "FIELD_AVAILABLE" and pd.notna(phase.phase_start_index):
            e, a, b = int(phase.phase_start_index), int(phase.low_index), int(phase.high_index)
            identity = (a, b, e)
            result["event_id"] = f"PULLBACK_{a}_{b}_{e}"
            if i > e and ac[i] > ah[i - 1] + 1e-10:
                if identity in seen:
                    result["event_status"] = "FIRST_TURN_ALREADY_CONSUMED"
                else:
                    seen.add(identity)
                    previous = stage.iloc[i - 1]
                    same = (previous.field_status == "FIELD_AVAILABLE"
                            and previous.phase_start_index == e
                            and previous.low_index == a and previous.high_index == b)
                    ratio = float(previous[FIELD]) if same else np.nan
                    result["known_prior_amount_ratio"] = ratio
                    stop, target = float(al[e:i + 1].min() - .001), float(ac[b])
                    result.update(stop_index=stop, target_index=target)
                    if not same or not np.isfinite(ratio):
                        result["event_status"] = "NO_VIEW_PRIOR_CONTRACTION_PHASE"
                    elif ratio >= 1.:
                        result["event_status"] = "NO_PRIOR_AMOUNT_CONTRACTION"
                    elif not stop < ac[i] < target:
                        result["event_status"] = "NO_POSITIVE_REMAINING_STRUCTURE"
                    else:
                        rr = (target - ac[i]) / (ac[i] - stop)
                        result["raw_planned_reward_risk"] = rr
                        result["entry_event"] = bool(rr >= 2.)
                        result["event_status"] = "STRUCTURAL_ENTRY_EVENT" if rr >= 2. else "REMAINING_ROOM_BELOW_TWO_R"
        for name in ["daily_hist", "weekly_hist", "relative_volume", "rv_ratio"]:
            result[name] = float(d[name].iloc[i]) if name in d else np.nan
        rows.append(result)
    return pd.DataFrame(rows)


def geometry(quantity: int, raw_entry: float, stop: float, target: float, cost: str) -> dict:
    """按当前已知报价和实际计划份额计算含双边费用的计划比，不当作实现收益。"""
    if quantity <= 0 or not 0 < stop < raw_entry < target:
        return {"net_reward_risk": np.nan, "planned_net_reward": np.nan, "planned_net_loss": np.nan}
    buy = quantity * fill(raw_entry, 1, cost)
    debit = buy + fee(buy, cost)
    at_stop = quantity * fill(stop, -1, cost)
    at_target = quantity * fill(target, -1, cost)
    loss = debit - (at_stop - fee(at_stop, cost))
    reward = at_target - fee(at_target, cost) - debit
    return {"net_reward_risk": reward / loss if loss > 0 else np.nan,
            "planned_net_reward": reward, "planned_net_loss": loss}


def account(data, dividends, parents, risks, signal_rows, cost, start, mode):
    """保持原A执行，只在明确空仓机会采用新事件，固定止损/目标、下一开盘。"""
    if mode not in MODES:
        raise ValueError("未知固定账户用途。")
    d = data.reset_index(drop=True)
    sig = signal_rows.reset_index(drop=True)
    if not pd.DatetimeIndex(d.date).equals(pd.DatetimeIndex(sig.date)):
        raise ValueError("回调信号与原日线日期不一致。")
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
        elif active is not None and active["source"] == "PULLBACK":
            adjusted = float(d.ac.iloc[i])
            if adjusted <= active["stop_index"] or adjusted >= active["target_index"]:
                desired = 0
                reason = "STRUCTURAL_STOP_CONFIRMED" if adjusted <= active["stop_index"] else "STRUCTURAL_TARGET_CONFIRMED"
            elif np.isfinite(es):
                desired = risk_base.cap_quantity(raw, risk_base.limits(nav, peak, es), q, existing=q)
                reason = "PULLBACK_RISK_REDUCTION" if desired < q else "PULLBACK_KEEP_FIXED_STRUCTURE"
            else:
                desired, reason = q, "UNKNOWN_KEEP_EXISTING"
        else:
            native_request = (q == 0 and mode != "A_CONTROL" and bool(event.entry_event)
                              and (mode == "PULLBACK_ONLY" or target == 0))
            core_enabled = mode != "PULLBACK_ONLY"
            if native_request:
                source = "PULLBACK"
                if np.isfinite(es):
                    desired = int(.5 * nav / raw // 100) * 100
                    desired = risk_base.cap_quantity(raw, risk_base.limits(nav, peak, es), desired)
                    estimate = geometry(desired, raw, float(event.stop_index - d.cash_shift.iloc[i]),
                                        float(event.target_index - d.cash_shift.iloc[i]), cost)
                    if not np.isfinite(estimate["net_reward_risk"]) or estimate["net_reward_risk"] < 2.:
                        desired, reason = 0, "PRIOR_NET_REWARD_RISK_BELOW_TWO"
                    else:
                        reason = "PULLBACK_FIRST_TURN_ENTRY"
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
            planned = None
            if extra and pending["source_request"] == "PULLBACK":
                planned = geometry(extra, float(r.open), float(pending["stop_index"] - r.cash_shift),
                                   float(pending["target_index"] - r.cash_shift), cost)
                if not np.isfinite(planned["net_reward_risk"]) or planned["net_reward_risk"] < 2.:
                    rejections.append({"date": r.date, "origin": pending["origin"],
                                       "reason": "OPEN_OUTSIDE_NET_TWO_R_GEOMETRY", "source_request": "PULLBACK",
                                       "event_id": pending["event_id"], **planned})
                    extra = 0
            if extra:
                charge = fee(extra * px, cost)
                debit = extra * px + charge
                cash -= debit
                q += extra
                if active is None:
                    active = {"cycle_id": len(cycles) + 1, "source": pending["source_request"],
                              "event_id": pending["event_id"] if pending["source_request"] == "PULLBACK" else None,
                              "entry_origin": pending["origin"], "entry_date": r.date, "entry_idx": i,
                              "entry_raw": float(r.open), "entry_price": px, "entry_quantity": extra,
                              "entry_equity": nav_open, "buy_debit": 0., "entry_fee": 0.,
                              "sell_net_cny": 0., "sell_fees": 0., "dividend_cny": 0.,
                              "exit_date": pd.NaT, "status": "RIGHT_CENSORED",
                              "stop_index": pending["stop_index"] if pending["source_request"] == "PULLBACK" else np.nan,
                              "target_index": pending["target_index"] if pending["source_request"] == "PULLBACK" else np.nan,
                              "entry_net_reward_risk": planned["net_reward_risk"] if planned else np.nan}
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
            elif planned is None:
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
            raise AssertionError("回调共同账户的资金、库存或财富恒等式失败。")
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
