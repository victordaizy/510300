"""原A权重来源和现金执行保持，实际进入后按新增证据改变持有。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import core_actual_acceptance_inputs_v1 as candidate
from research import daily_supply_test_v1 as risk_base
from research.point_account_nr7_inputs_v1 import PARENT_A, fill, fee, open_blocked


def account(data, dividends, parents, risks, policy, cost, start):
    if policy not in (*candidate.POLICIES, candidate.BASELINE):
        raise ValueError("未知原A实际接受账户。")
    d = data.reset_index(drop=True)
    parent = candidate.parent_episodes(parents).set_index("origin")
    first = int(np.flatnonzero(d.date.ge(pd.Timestamp(start)))[0])
    if first < 1 or not pd.DatetimeIndex(d.date.iloc[first-1:-1]).isin(parent.index).all():
        raise ValueError("原保存来源未覆盖全部开盘前决定。")
    risk = dict(zip(risks.idx.astype(int), risks.es95))
    cash, q, receivable, peak, previous_nav = 200000., 0, 0., 200000., 200000.
    active, stopped, locked_exit = None, False, False
    lock_reason, lock_episode, consumed_episode = None, 0, -1
    entitlements, outstanding, cycles = {}, {}, []
    events = dividends.to_dict("records")
    daily, orders, rejections, decisions, holding_evidence = [], [], [], [], []

    def decide(i):
        nonlocal locked_exit, lock_reason, lock_episode
        row = d.iloc[i]
        raw = float(row.close)
        target = float(parent[PARENT_A].get(row.date, np.nan))
        episode = int(parent.parent_episode_id.get(row.date, 0))
        nav = cash + receivable + q*raw
        es = risk.get(i, np.nan)
        source = active["source"] if active else "NONE"
        evidence = None
        if active is not None and policy != candidate.BASELINE:
            evidence = candidate.advance(active, row, policy)
            holding_evidence.append({"origin": row.date, "cycle_id": active["cycle_id"], "entry_date": active["entry_date"], **evidence})
        reason = "KEEP_WITHIN_ORIGINAL_TEN_POINT_BAND"
        if stopped or locked_exit:
            desired = 0
            reason = "ACCOUNT_DRAWDOWN_STOP" if stopped else ("LOCKED_EXIT" if policy == candidate.BASELINE else lock_reason)
        elif evidence is not None and evidence["extra_exit_reason"] != "NONE":
            desired, reason = 0, evidence["extra_exit_reason"]
        elif evidence is not None and evidence["holding_stage"] == "CARRY" and (target == 0 or not np.isfinite(target)):
            if np.isfinite(es):
                desired = risk_base.cap_quantity(raw, risk_base.limits(nav, peak, es), q, existing=q)
                reason = "CARRY_RISK_REDUCTION" if desired < q else "CARRY_INDEPENDENT_OF_OLD_PARENT_ZERO_OR_UNKNOWN"
            else:
                desired, reason = q, "CARRY_UNKNOWN_RISK_KEEP_EXISTING"
        elif target == 0:
            desired, reason = 0, "PARENT_ZERO"
        elif not np.isfinite(target) or not np.isfinite(es):
            desired, reason = q, "UNKNOWN_KEEP_EXISTING"
        elif policy != candidate.BASELINE and q == 0 and episode == consumed_episode:
            desired, reason = 0, "FORCED_EXIT_CONSUMED_SAME_PARENT_EPISODE"
        else:
            if not 0 <= target <= 1:
                raise ValueError("原保存权重超出0至1。")
            source = "CORE_WEIGHT"
            target = min(.5, target)
            if q == 0 or abs(target-q*raw/nav) >= .1:
                desired = int(target*nav/raw//100)*100
                reason = "SAVED_WEIGHT_REBALANCE"
            else:
                desired = q
            cap = risk_base.cap_quantity(raw, risk_base.limits(nav, peak, es), desired, existing=q)
            if cap < desired:
                desired, reason = cap, "PRIOR_CLOSE_RISK_CAP"
        if q and desired == 0 and not locked_exit:
            locked_exit, lock_reason, lock_episode = True, reason, episode
        result = {"origin": row.date, "execution_date": d.date.iloc[i+1] if i+1 < len(d) else pd.NaT,
            "source_weight": target, "shares_before": q, "desired_shares": desired,
            "known_es95": es, "reason": reason, "source_request": source,
            "entry_event": False, "event_id": None, "stop_index": np.nan, "target_index": np.nan,
            "atr_at_origin": np.nan, "entry_known_fit_index": np.nan,
            "predicted_p_times_b": np.nan, "predicted_net_expectation": np.nan}
        if policy != candidate.BASELINE:
            result.update(parent_episode_id=episode, consumed_parent_episode_id=consumed_episode,
                forced_exit_episode_id=lock_episode, holding_stage=active["holding_stage"] if active else "FLAT",
                protection_armed=active["protection_armed"] if active else False,
                structural_stop=active["structural_stop"] if active else np.nan,
                extension_revoked=active["extension_revoked"] if active else False)
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
            rejections.append({"date": row.date, "origin": pending["origin"], "reason": "OPEN_LIMIT_OR_MISSING_PRICE",
                               "source_request": pending["source_request"], "event_id": pending["event_id"]})
        elif side < 0:
            if active is None or i <= active["entry_idx"]:
                raise AssertionError("卖出违反T+1或缺少真实持仓。")
            sold = q-desired
            px = fill(float(row.open), -1, cost)
            charge = fee(sold*px, cost)
            cash += sold*px-charge
            q = desired
            active["sell_net_cny"] += sold*px-charge
            active["sell_fees"] += charge
            commission += charge
            slippage += sold*(float(row.open)-px)
            orders.append({"date": row.date, "origin": pending["origin"], "cycle_id": active["cycle_id"],
                "source": active["source"], "side": "SELL", "quantity": sold, "raw_open": row.open,
                "fill_price": px, "commission": charge, "slippage": sold*(float(row.open)-px), "reason": pending["reason"]})
            if q == 0:
                active.update(exit_date=row.date, exit_idx=i, exit_reason=pending["reason"], status="COMPLETE", holding_sessions=i-active["entry_idx"])
                if policy != candidate.BASELINE and pending["reason"] in {"ACTUAL_ACCEPTANCE_STRUCTURE_FAILED", "CARRY_TRAIL_STRUCTURE_FAILED"}:
                    consumed_episode = int(pending["forced_exit_episode_id"])
                active, locked_exit, lock_reason = None, False, None
        elif side > 0:
            px = fill(float(row.open), 1, cost)
            nav_open = cash+receivable+q*float(row.open)
            limits = risk_base.limits(nav_open, peak, pending["known_es95"])
            total = risk_base.cap_quantity(float(row.open), limits, desired)
            extra = max(0, total-q)
            while extra:
                debit = extra*px+fee(extra*px, cost)
                if debit <= cash+1e-8 and risk_base.risk_ok(q+extra, float(row.open), limits, buying=True):
                    break
                extra -= 100
            if extra:
                charge = fee(extra*px, cost)
                debit = extra*px+charge
                cash -= debit
                q += extra
                if active is None:
                    active = {"cycle_id": len(cycles)+1, "source": pending["source_request"], "event_id": None,
                        "entry_origin": pending["origin"], "entry_date": row.date, "entry_idx": i,
                        "entry_raw": float(row.open), "entry_price": px, "entry_quantity": extra,
                        "entry_equity": nav_open, "buy_debit": 0., "entry_fee": 0., "sell_net_cny": 0.,
                        "sell_fees": 0., "dividend_cny": 0., "exit_date": pd.NaT, "status": "RIGHT_CENSORED",
                        "stop_index": np.nan, "target_index": np.nan, "entry_net_reward_risk": np.nan}
                    if policy != candidate.BASELINE:
                        active.update(entry_day_high_anchor=np.nan, entry_day_low_anchor=np.nan, holding_stage="CORE",
                            protection_armed=False, structural_stop=np.nan, first_price_acceptance=pd.NaT,
                            promotion_date=pd.NaT, extension_revoked=False, first_extension_revocation=pd.NaT)
                    cycles.append(active)
                active["buy_debit"] += debit
                active["entry_fee"] += charge
                commission += charge
                slippage += extra*(px-float(row.open))
                orders.append({"date": row.date, "origin": pending["origin"], "cycle_id": active["cycle_id"],
                    "source": active["source"], "side": "BUY", "quantity": extra, "raw_open": row.open,
                    "fill_price": px, "commission": charge, "slippage": extra*(px-float(row.open)),
                    "reason": pending["reason"], "known_es95": pending["known_es95"], "planned_total_quantity": desired})
            else:
                rejections.append({"date": row.date, "origin": pending["origin"], "reason": "CASH_OR_OPEN_RISK_CAP",
                                   "source_request": pending["source_request"], "event_id": pending["event_id"]})
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
            raise AssertionError("实际接受账户的现金、库存或财富恒等式失败。")
        daily_row = {"date": row.date, "equity": nav, "cash": cash, "shares": q, "close": row.close,
            "receivable": receivable, "net_return": nav/previous_nav-1, "price_pnl": price_pnl,
            "dividend_accrual": accrued, "dividend_paid": paid, "commission": commission, "slippage": slippage,
            "exposure": q*row.close/nav, "drawdown": dd, "accounting_error": error,
            "risk_stopped": stopped, "source": active["source"] if q else "FLAT"}
        previous_nav = nav
        pending = decide(i)
        if policy != candidate.BASELINE:
            daily_row.update(holding_stage=active["holding_stage"] if active else "FLAT",
                             consumed_parent_episode_id=consumed_episode)
        daily.append(daily_row)
    for trade in cycles:
        trade["net_pnl"] = trade["sell_net_cny"]+trade["dividend_cny"]-trade["buy_debit"] if trade["status"] == "COMPLETE" else np.nan
        trade["net_return"] = trade["net_pnl"]/trade["buy_debit"] if trade["status"] == "COMPLETE" else np.nan
    terminal = {"open_shares": q, "stopped": stopped, "unpaid_dividend_cny": receivable,
        "planned_next_close_request": pending, "artificial_terminal_liquidation": False}
    if policy != candidate.BASELINE:
        terminal.update(open_source=active["source"] if active else None, consumed_parent_episode_id=consumed_episode,
            open_pnl_cny=q*float(d.close.iloc[-1])+active["sell_net_cny"]+active["dividend_cny"]-active["buy_debit"] if active else 0.,
            estimated_future_exit_friction=q*(float(d.close.iloc[-1])-fill(float(d.close.iloc[-1]), -1, cost))+fee(q*fill(float(d.close.iloc[-1]), -1, cost), cost) if q else 0.)
    columns = ["cycle_id", "source", "entry_origin", "entry_date", "exit_date", "status", "net_pnl", "net_return"]
    return {"daily": pd.DataFrame(daily), "orders": pd.DataFrame(orders),
        "trades": pd.DataFrame(cycles) if cycles else pd.DataFrame(columns=columns),
        "decisions": pd.DataFrame(decisions), "rejections": pd.DataFrame(rejections),
        "holding_evidence": pd.DataFrame(holding_evidence), "terminal": terminal}
