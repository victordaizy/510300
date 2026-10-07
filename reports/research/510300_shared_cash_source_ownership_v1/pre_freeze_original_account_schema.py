"""统一现金与原风险：CORE权重或补充持仓拥有一个完整交易周期。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import shared_cash_source_ownership_rules_v1 as rules
from research import daily_supply_test_v1 as budget
from research.point_account_nr7_inputs_v1 import PARENT_A, fill, fee, open_blocked

EPS = 1e-10


def account(data, dividends, parents, risks, policy, cost, start):
    prefix = rules.complement_prefix(policy)
    d = data.reset_index(drop=True)
    parent = parents.set_index("origin")
    first = int(np.flatnonzero(d.date.ge(pd.Timestamp(start)))[0])
    if first < 1 or not pd.DatetimeIndex(d.date.iloc[first-1:-1]).isin(parent.index).all():
        raise ValueError("缺少首个前收盘或原CORE完整来源。")
    risk = dict(zip(risks.idx.astype(int),risks.es95))
    cash,q,receivable,peak,previous_nav = 200000.,0,0.,200000.,200000.
    active,stopped,locked_exit = None,False,False
    entitlements,outstanding,cycles = {},{},[]
    daily,orders,rejections,decisions,ownership_evidence = [],[],[],[],[]
    events = dividends.to_dict("records")

    def decide(i):
        nonlocal locked_exit
        row = d.iloc[i]
        target = float(parent[PARENT_A].get(row.date,np.nan)) if policy != rules.BASELINE_SUPPORT else 0.
        raw_target = target
        if np.isfinite(target) and not 0 <= target <= 1:
            raise ValueError("保存原目标超出0至1。")
        raw = float(row.close)
        nav = cash+receivable+q*raw
        es = risk.get(i,np.nan)
        entry = bool(row[prefix+"_entry_event"])
        role = rules.event_role(q,target,stopped,locked_exit,entry,policy)
        source = active["source"] if active else "NONE"
        owner = active["owner"] if active else "FLAT"
        desired,reason = q,"KEEP_WITHIN_ORIGINAL_TEN_POINT_BAND"
        if stopped or locked_exit:
            desired,reason = 0,"ACCOUNT_DRAWDOWN_STOP" if stopped else "LOCKED_EXIT"
        elif active is not None and active["owner"] == "COMPLEMENT":
            failure = rules.support.exit_decision(active,row,i,rules.complement_exit_policy(policy))
            if failure:
                desired,reason = 0,failure
            elif np.isfinite(es):
                desired = budget.cap_quantity(raw,budget.limits(nav,peak,es),q,existing=q)
                reason = "PRIOR_CLOSE_RISK_REDUCTION" if desired < q else "KEEP"
            else:
                desired,reason = q,"KEEP"
        elif target == 0:
            if q:
                desired,reason = 0,"PARENT_ZERO"
            elif role == "CURRENT_ZERO_CORE_COMPLEMENT_ATTEMPT":
                if np.isfinite(es) and np.isfinite([row[prefix+"_setup_low"],row.atr14]).all():
                    planned = int(.5*nav/raw//100)*100
                    desired = budget.cap_quantity(raw,budget.limits(nav,peak,es),planned)
                    source,reason = prefix.upper()+"_ACCEPTED",prefix.upper()+"_PRICE_ACCEPTED_CLOSE_ENTRY"
                else:
                    desired,reason = 0,"NO_VIEW_ENTRY_RISK_OR_ANCHOR"
            else:
                desired,reason = 0,"NO_ENTRY_EVENT" if policy == rules.BASELINE_SUPPORT else "PARENT_ZERO"
        elif not np.isfinite(target) or not np.isfinite(es):
            desired,reason = q,"UNKNOWN_KEEP_EXISTING"
        else:
            source = "CORE_WEIGHT"
            target = min(.5,target)
            if q == 0 or abs(target-q*raw/nav) >= .1:
                desired = int(target*nav/raw//100)*100
                reason = "SAVED_WEIGHT_REBALANCE"
            cap = budget.cap_quantity(raw,budget.limits(nav,peak,es),desired,existing=q)
            if cap < desired:
                desired,reason = cap,"PRIOR_CLOSE_RISK_CAP"
        if q and desired == 0:
            locked_exit = True
        new_complement = q == 0 and source == prefix.upper()+"_ACCEPTED" and desired > 0
        result = {"origin":row.date,"execution_date":d.date.iloc[i+1] if i+1 < len(d) else pd.NaT,
            "source_weight":target,"raw_saved_CORE_weight":raw_target,"shares_before":q,"desired_shares":desired,
            "known_es95":es,"reason":reason,"source_request":source,"owner_before":owner,
            "request_owner":"COMPLEMENT" if new_complement else ("CORE" if source == "CORE_WEIGHT" else owner),
            "entry_event":entry,"event_id":prefix.upper()+"_ACCEPT_"+row.date.strftime("%Y%m%d") if entry else None,
            "stop_index":np.nan,"target_index":np.nan,"atr_at_origin":float(row.atr14) if new_complement else np.nan,
            "support_at_origin":float(row[prefix+"_setup_low"]) if new_complement else np.nan,
            "setup_date":row[prefix+"_setup_date"] if new_complement else pd.NaT,
            "setup_source_ids":row[prefix+"_setup_source_ids"] if new_complement else "",
            "entry_known_fit_index":np.nan,"predicted_p_times_b":np.nan,"predicted_net_expectation":np.nan,
            "entry_type":prefix.upper()+"_ACCEPTED" if entry else "NONE",
            "current_complement_event_role":role,"complement_event_consumed_today":entry,
            "holding_stage":active["holding_stage"] if active else "FLAT"}
        decisions.append(result)
        if entry or owner == "COMPLEMENT":
            ownership_evidence.append({"origin":row.date,"actual_cycle_id":active["cycle_id"] if active else None,
                "actual_owner":owner,"shares_before":q,"original_CORE_target":raw_target,
                "complement_entry_event":entry,"event_id":result["event_id"],"event_role":role,
                "event_consumed_today":entry,"pending_reason":reason,"pending_shares":desired,
                "setup_date":row[prefix+"_setup_date"],"setup_source_ids":row[prefix+"_setup_source_ids"],
                "holding_stage":result["holding_stage"]})
        return result

    pending = decide(first-1)
    for i in range(first,len(d)):
        row = d.iloc[i]
        old_q,accrued,paid,commission,slippage = q,0.,0.,0.,0.
        for k,event in enumerate(events):
            if event["ex_date"] == row.date and k in entitlements:
                eligible,owner = entitlements[k]
                amount = eligible*event["cash_dividend_per_share"]
                if amount:
                    outstanding[k] = amount
                    receivable += amount
                    accrued += amount
                    owner["dividend_cny"] += amount
        desired = int(pending["desired_shares"])
        side = int(np.sign(desired-q))
        if side and open_blocked(d,i,side):
            rejections.append({"date":row.date,"origin":pending["origin"],"reason":"OPEN_LIMIT_OR_MISSING_PRICE",
                "source_request":pending["source_request"],"request_owner":pending["request_owner"],"event_id":pending["event_id"]})
        elif side < 0:
            if active is None or i <= active["entry_idx"]:
                raise AssertionError("卖出违反T+1或缺少持仓身份。")
            sold = q-desired
            px = fill(float(row.open),-1,cost)
            charge = fee(sold*px,cost)
            cash += sold*px-charge
            q = desired
            active["sell_net_cny"] += sold*px-charge
            active["sell_fees"] += charge
            commission += charge
            slippage += sold*(float(row.open)-px)
            orders.append({"date":row.date,"origin":pending["origin"],"cycle_id":active["cycle_id"],
                "source":active["source"],"owner":active["owner"],"side":"SELL","quantity":sold,"raw_open":row.open,
                "fill_price":px,"commission":charge,"slippage":sold*(float(row.open)-px),"reason":pending["reason"],
                "holding_stage":active["holding_stage"]})
            if q == 0:
                active.update(exit_date=row.date,exit_idx=i,exit_reason=pending["reason"],status="COMPLETE",holding_sessions=i-active["entry_idx"])
                active,locked_exit = None,False
        elif side > 0:
            complement = pending["request_owner"] == "COMPLEMENT"
            if complement and q:
                raise AssertionError("补充来源持有中不得补买或转归属。")
            if active is not None and active["owner"] != pending["request_owner"]:
                raise AssertionError("真实持仓归属被另一源改变。")
            if complement and float(row.open+row.cash_shift) <= pending["support_at_origin"]+EPS:
                rejections.append({"date":row.date,"origin":pending["origin"],"reason":"OPEN_ALREADY_BELOW_KNOWN_ANCHOR",
                    "source_request":pending["source_request"],"event_id":pending["event_id"]})
            else:
                px = fill(float(row.open),1,cost)
                nav_open = cash+receivable+q*float(row.open)
                limits = budget.limits(nav_open,peak,pending["known_es95"])
                total = budget.cap_quantity(float(row.open),limits,desired)
                extra = max(0,total-q)
                while extra:
                    debit = extra*px+fee(extra*px,cost)
                    if debit <= cash+1e-8 and budget.risk_ok(q+extra,float(row.open),limits,buying=True):
                        break
                    extra -= 100
                if extra:
                    charge = fee(extra*px,cost)
                    debit = extra*px+charge
                    cash -= debit
                    q += extra
                    if active is None:
                        active = {"cycle_id":len(cycles)+1,"source":pending["source_request"],"owner":pending["request_owner"],
                            "event_id":pending["event_id"] if complement else None,"entry_origin":pending["origin"],
                            "entry_date":row.date,"entry_idx":i,"entry_raw":float(row.open),"entry_price":px,"entry_quantity":extra,
                            "entry_equity":nav_open,"buy_debit":0.,"entry_fee":0.,"sell_net_cny":0.,"sell_fees":0.,
                            "dividend_cny":0.,"exit_date":pd.NaT,"status":"RIGHT_CENSORED","stop_index":np.nan,
                            "target_index":np.nan,"entry_net_reward_risk":np.nan,"holding_stage":"EARLY" if complement else "CORE",
                            "promotion_date":pd.NaT,"structural_stop":pending["support_at_origin"] if complement else np.nan,
                            "setup_date":pending["setup_date"],"setup_source_ids":pending["setup_source_ids"],
                            "fixed_stop":float(row.open+row.cash_shift-pending["atr_at_origin"]) if complement else np.nan,
                            "fixed_target":float(row.open+row.cash_shift+2*pending["atr_at_origin"]) if complement else np.nan}
                        cycles.append(active)
                    active["buy_debit"] += debit
                    active["entry_fee"] += charge
                    commission += charge
                    slippage += extra*(px-float(row.open))
                    orders.append({"date":row.date,"origin":pending["origin"],"cycle_id":active["cycle_id"],
                        "source":active["source"],"owner":active["owner"],"side":"BUY","quantity":extra,"raw_open":row.open,
                        "fill_price":px,"commission":charge,"slippage":extra*(px-float(row.open)),"reason":pending["reason"],
                        "known_es95":pending["known_es95"],"planned_total_quantity":desired,"planned_quantity":desired,
                        "setup_date":pending["setup_date"],"setup_source_ids":pending["setup_source_ids"]})
                else:
                    rejections.append({"date":row.date,"origin":pending["origin"],"reason":"CASH_OR_OPEN_RISK_CAP",
                        "source_request":pending["source_request"],"event_id":pending["event_id"]})
        for k,event in enumerate(events):
            if event["record_date"] == row.date and q:
                entitlements[k] = (q,active)
            if event["payment_date"] <= row.date and k in outstanding:
                amount = outstanding.pop(k)
                cash += amount
                receivable -= amount
                paid += amount
        nav = cash+receivable+q*float(row.close)
        peak = max(peak,nav)
        dd = 1-nav/peak
        stopped = stopped or dd >= .1
        price_pnl = old_q*(float(row.open)-float(d.close.iloc[i-1]))+q*(float(row.close)-float(row.open))
        error = nav-previous_nav-price_pnl-accrued+commission+slippage
        if abs(error) > 1e-6 or cash < -1e-6 or receivable < -1e-6 or q % 100:
            raise AssertionError("共同现金、库存或财富恒等式失败。")
        daily_row = {"date":row.date,"equity":nav,"cash":cash,"shares":q,"close":row.close,"receivable":receivable,
            "net_return":nav/previous_nav-1,"price_pnl":price_pnl,"dividend_accrual":accrued,"dividend_paid":paid,
            "commission":commission,"slippage":slippage,"exposure":q*row.close/nav,"drawdown":dd,"accounting_error":error,
            "risk_stopped":stopped,"source":active["source"] if active else "FLAT","owner":active["owner"] if active else "FLAT",
            "holding_stage":active["holding_stage"] if active else "FLAT"}
        previous_nav = nav
        pending = decide(i)
        daily.append(daily_row)
    for trade in cycles:
        trade["net_pnl"] = trade["sell_net_cny"]+trade["dividend_cny"]-trade["buy_debit"] if trade["status"] == "COMPLETE" else np.nan
        trade["net_return"] = trade["net_pnl"]/trade["buy_debit"] if trade["status"] == "COMPLETE" else np.nan
    terminal = {"open_shares":q,"stopped":stopped,"unpaid_dividend_cny":receivable,"planned_next_close_request":pending,
        "open_source":active["source"] if active else None,"open_owner":active["owner"] if active else None,
        "open_pnl_cny":q*float(d.close.iloc[-1])+active["sell_net_cny"]+active["dividend_cny"]-active["buy_debit"] if active else 0.,
        "estimated_future_exit_friction":q*(float(d.close.iloc[-1])-fill(float(d.close.iloc[-1]),-1,cost))+fee(q*fill(float(d.close.iloc[-1]),-1,cost),cost) if q else 0.,
        "artificial_terminal_liquidation":False}
    columns = ["cycle_id","source","owner","entry_origin","entry_date","exit_date","status","net_pnl","net_return"]
    return {"daily":pd.DataFrame(daily),"orders":pd.DataFrame(orders),"trades":pd.DataFrame(cycles) if cycles else pd.DataFrame(columns=columns),
        "decisions":pd.DataFrame(decisions),"rejections":pd.DataFrame(rejections),"ownership_evidence":pd.DataFrame(ownership_evidence),"terminal":terminal}
