"""描述原支持接受与CORE来源的真实时钟、持仓冲突及一次容量快照。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import daily_supply_test_v1 as budget
from research.point_account_nr7_inputs_v1 import fill, fee, open_blocked

EPS = 1e-10
ACCOUNTS = ("A_SAVED_WEIGHT", "CORE_ACTUAL_ACCEPTANCE_INFORMATION_CARRY", "SUPPORT_PRICE_ACCEPTANCE_CARRY")
NAMES = dict(zip(ACCOUNTS, ("原A", "R236主", "R232支持独立账户")))
DATE_COLUMNS = {"origin", "date", "entry_origin", "entry_date", "exit_date", "execution_date", "support_setup_date", "source_first_decision_date", "original_A_entry_origin", "original_A_actual_entry_date", "support_cycle_entry_origin_at_Aorigin"}
CLOCK_COLUMNS = {"decision_time", "source_available_upper", "next_open_observed_at", "orders_available_at", "source_setup_clock", "background_actual_next_order_observed_at", "identity_observed_at"}
OPTIONAL_NUMERIC = {"next_open_one_shot_quantity_not_combined_position", "next_raw_open", "next_cash_shift",
    "planned_at_prior_close_one_shot_quantity", "one_shot_fill_price", "background_actual_next_BUY_quantity",
    "background_actual_next_SELL_quantity", "background_shares_at_original_A_origin", "background_next_requested_shares"}


def clock(value):
    if pd.isna(value):
        return pd.NaT
    x = pd.Timestamp(value)
    if x.tz is None:
        raise ValueError("有限来源钟缺少明确时区。")
    return x.tz_convert("Asia/Shanghai").as_unit("ns")


def normalized(frame):
    d = frame.copy().reset_index(drop=True)
    for column in DATE_COLUMNS.intersection(d.columns):
        d[column] = pd.to_datetime(d[column]).astype("datetime64[ns]")
    for column in CLOCK_COLUMNS.intersection(d.columns):
        values = [clock(v) for v in d[column]]
        d[column] = pd.to_datetime(pd.Series(values), utc=True).dt.tz_convert("Asia/Shanghai").astype("datetime64[ns, Asia/Shanghai]")
    for column in OPTIONAL_NUMERIC.intersection(d.columns):
        d[column] = pd.to_numeric(d[column]).astype("float64")
    return d


def decision_role(shares, parent, stopped, es, price_known):
    if shares is None or pd.isna(shares):
        return "NO_VIEW_ACTUAL_ACCOUNT"
    if shares > 0:
        return "OWN_POSITION_HELD_EVENT_CONSUMED_NO_SAME_OPEN_REBUY"
    if not np.isfinite(parent):
        return "NO_VIEW_ORIGINAL_CORE_PARENT"
    if parent > 0:
        return "ORIGINAL_CORE_POSITIVE_PRIORITY"
    if parent < 0:
        raise ValueError("原CORE目标不能为负。")
    if stopped:
        return "ACCOUNT_STOPPED"
    if not np.isfinite(es):
        return "NO_VIEW_MATURE_RISK"
    if not price_known:
        return "NO_VIEW_SUPPORT_ANCHOR"
    return "POTENTIAL_COMPLEMENT_ATTEMPT_ONLY_NOT_NEW_STRATEGY_ENTRY"


def source_identity(row, receipts):
    ids = [x for x in str(row.support_setup_source_ids).split("|") if x]
    known = receipts.set_index("node_id")
    if not ids or any(x not in known.index for x in ids):
        raise ValueError("价格锚没有完整原来源身份。")
    setup = pd.Timestamp(row.support_setup_date)
    setup_clock = setup.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
    if not setup < row.date:
        raise ValueError("没有等待支持锚之后的价格接受。")
    rows = known.loc[ids]
    clocks = [clock(x) for x in rows.source_available_upper]
    if any(pd.isna(x) or x > setup_clock for x in clocks) or not rows.first_original_decision_date.le(setup).all():
        raise ValueError("锚使用了以后才公开或首次可知的来源。")
    if not rows.actual_information_role.isin(["RECORDED_SUPPORT_INFORMATION", "OBSERVED_RATE_LOWER"]).all():
        raise ValueError("确认或非支持记录被当作新的支持源。")
    return {"source_node_ids": "|".join(ids), "source_common_ids": "|".join(sorted(set(rows.common_source_id))),
        "source_common_count_not_independent_votes": len(set(rows.common_source_id)),
        "source_available_upper": max(clocks), "source_first_decision_date": rows.first_original_decision_date.max(),
        "source_setup_clock": setup_clock, "source_scope": "RECORDED_POSITIVE_EVIDENCE_NOT_FULL_POLICY_COVERAGE"}


def open_capacity(data, idx, snapshot, es, anchor_low, cost):
    """只用原背景的一次容量说明；不更新组合现金、不生成新收益。"""
    if idx+1 >= len(data):
        return {"next_open_role": "NOT_OBSERVED_YET", "next_open_observed_at": pd.NaT,
                "next_open_one_shot_quantity_not_combined_position": np.nan}
    row = data.iloc[idx+1]
    at = pd.Timestamp(row.date).tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9, minutes=30)
    result = {"next_open_observed_at": at, "next_open_one_shot_quantity_not_combined_position": 0,
        "next_raw_open": float(row.open), "next_cash_shift": float(row.cash_shift)}
    if snapshot["shares"] != 0:
        raise ValueError("单次补充容量只适用于原背景空仓。")
    if open_blocked(data, idx+1, 1):
        return {**result, "next_open_role": "OPEN_LIMIT_OR_MISSING_PRICE"}
    if float(row.open+row.cash_shift) <= anchor_low+EPS:
        return {**result, "next_open_role": "OPEN_ALREADY_BELOW_KNOWN_ANCHOR"}
    prior = data.iloc[idx]
    nav_close = snapshot["cash"]+snapshot["receivable"]
    planned = int(.5*nav_close/float(prior.close)//100)*100
    planned = budget.cap_quantity(float(prior.close), budget.limits(nav_close, snapshot["peak"], es), planned)
    # 原背景的除息权益在开盘前确认；当日支付按原引擎在订单之后，不提前拿来买入。
    nav_open = nav_close+snapshot.get("next_dividend_accrual", 0.)
    limits = budget.limits(nav_open, snapshot["peak"], es)
    quantity = budget.cap_quantity(float(row.open), limits, planned)
    px = fill(float(row.open), 1, cost)
    while quantity:
        if quantity*px+fee(quantity*px, cost) <= snapshot["cash"]+1e-8 and budget.risk_ok(quantity, float(row.open), limits, buying=True):
            break
        quantity -= 100
    return {**result, "next_open_role": "ONE_SHOT_CAPACITY_POSITIVE_NOT_REAL_COMBINED_FILL" if quantity else "ONE_SHOT_CASH_OR_RISK_BELOW_LOT",
        "next_open_one_shot_quantity_not_combined_position": quantity,
        "planned_at_prior_close_one_shot_quantity": planned, "one_shot_fill_price": px}


def describe(data, receipts, accounts, risks, periods):
    d = normalized(data)
    if not d.date.is_unique or not d.date.is_monotonic_increasing:
        raise ValueError("原观察日重复或未排序。")
    risk_map = risks.set_index("idx")
    snapshots = {}
    for key, account in accounts.items():
        daily, decisions = normalized(account["daily"]), normalized(account["decisions"])
        daily["known_peak_at_close"] = np.maximum(200000., daily.equity.cummax())
        snapshots[key] = {"daily":daily.set_index("date"),"decisions":decisions.set_index("origin"),
            "orders":normalized(account["orders"]),"trades":normalized(account["trades"])}
    contexts, points, entry_contexts = [], [], []
    costs = sorted({key[1] for key in accounts})
    for idx in np.flatnonzero(d.support_entry_event.to_numpy(bool)):
        row = d.iloc[idx]
        period = next((p for p,(start,end) in periods.items() if pd.Timestamp(start) <= row.date <= pd.Timestamp(end)), None)
        if period is None:
            continue
        identity = "SUPPORT_ACCEPT_"+row.date.strftime("%Y%m%d")
        source = source_identity(row, normalized(receipts))
        if not row.ac > row.support_setup_high+EPS or not row.daily_hist > 0:
            raise ValueError("保存接受事件的价格条件不一致。")
        es = float(risk_map.es95.loc[idx]) if idx in risk_map.index else np.nan
        if idx in risk_map.index and float(risk_map.latest_label_exit_idx.loc[idx]) > idx:
            raise ValueError("风险估计含未成熟标签。")
        point = {"point_id":identity,"period":period,"origin":row.date,"decision_time":clock(row.decision_time),
            "support_setup_date":row.support_setup_date,"support_setup_low":row.support_setup_low,
            "support_setup_high":row.support_setup_high,"accepted_cash_close":row.ac,"daily_hist":row.daily_hist,
            "known_es95":es,"pmi_original_level":row.pmi_orders_level+50.,"pmi_original_month_change":row.pmi_orders_change,
            "pmi_known":bool(row.orders_known),"orders_available_at":clock(row.orders_available_at), **source}
        points.append(point)
        for cost in costs:
            original = snapshots[(period,cost,ACCOUNTS[0])]
            original_decision = original["decisions"].loc[row.date]
            parent = float(original_decision.source_weight)
            for name in ACCOUNTS:
                own = snapshots[(period,cost,name)]
                actual = own["daily"].loc[row.date]
                request = own["decisions"].loc[row.date]
                role = decision_role(int(actual.shares),parent,bool(actual.risk_stopped),es,np.isfinite([row.support_setup_low,row.support_setup_high,row.atr14]).all())
                eligible = role == "POTENTIAL_COMPLEMENT_ATTEMPT_ONLY_NOT_NEW_STRATEGY_ENTRY"
                event = {**point,"cost":cost,"background_policy":name,"original_CORE_source_weight":parent,
                    "background_close_shares":int(actual.shares),"background_cash":float(actual.cash),
                    "background_receivable":float(actual.receivable),"background_equity":float(actual.equity),
                    "background_next_requested_shares":int(request.desired_shares),"background_pending_reason":str(request.reason),
                    "background_exit_pending_with_held_shares":bool(actual.shares > 0 and request.desired_shares == 0),
                    "decision_role":role,"potential_attempt_at_decision":eligible}
                event.update(next_raw_open=np.nan,next_cash_shift=np.nan,
                    planned_at_prior_close_one_shot_quantity=np.nan,one_shot_fill_price=np.nan)
                if eligible:
                    next_accrual = float(own["daily"].loc[d.date.iloc[idx+1]].dividend_accrual) if idx+1 < len(d) and d.date.iloc[idx+1] in own["daily"].index else 0.
                    snapshot = {"cash":float(actual.cash),"receivable":float(actual.receivable),"shares":int(actual.shares),
                        "peak":float(actual.known_peak_at_close),"next_dividend_accrual":next_accrual}
                    event.update(open_capacity(d.loc[d.date.le(periods[period][1])].reset_index(drop=True),idx,snapshot,es,float(row.support_setup_low),cost))
                else:
                    event.update(next_open_role="NOT_EVALUATED_INELIGIBLE_AT_DECISION",next_open_observed_at=pd.NaT,
                        next_open_one_shot_quantity_not_combined_position=np.nan)
                next_seen = idx+1 < len(d) and d.date.iloc[idx+1] <= pd.Timestamp(periods[period][1])
                next_date = d.date.iloc[idx+1] if next_seen else pd.NaT
                if next_seen:
                    orders = own["orders"]
                    matched = orders.loc[orders.origin.eq(row.date) & orders.date.eq(next_date)] if len(orders) else orders
                    event.update(background_actual_next_BUY_quantity=int(matched.loc[matched.side.eq("BUY"),"quantity"].sum()) if len(matched) else 0,
                        background_actual_next_SELL_quantity=int(matched.loc[matched.side.eq("SELL"),"quantity"].sum()) if len(matched) else 0,
                        background_actual_next_order_observed_at=next_date.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9,minutes=30))
                else:
                    event.update(background_actual_next_BUY_quantity=np.nan,background_actual_next_SELL_quantity=np.nan,
                        background_actual_next_order_observed_at=pd.NaT)
                contexts.append(event)
    for period,(start,end) in periods.items():
        for cost in costs:
            original = snapshots[(period,cost,ACCOUNTS[0])]
            for trade in original["trades"].loc[original["trades"].entry_date.le(d.date.iloc[-1])].itertuples():
                origin = trade.entry_origin
                for name in ACCOUNTS:
                    own = snapshots[(period,cost,name)]
                    actual = own["daily"].loc[origin] if origin in own["daily"].index else None
                    request = own["decisions"].loc[origin] if origin in own["decisions"].index else None
                    support_origin = pd.NaT
                    if actual is not None and actual.shares > 0 and name == ACCOUNTS[2]:
                        existing = own["trades"].loc[own["trades"].entry_date.le(origin)].sort_values("entry_date")
                        if existing.empty:
                            raise ValueError("支持真实持仓缺少先前进入身份。")
                        support_origin = existing.entry_origin.iloc[-1]
                    entry_contexts.append({"period":period,"cost":cost,"background_policy":name,
                        "original_A_cycle_id":trade.cycle_id,"original_A_entry_origin":origin,"original_A_actual_entry_date":trade.entry_date,
                        "identity_observed_at":trade.entry_date.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9,minutes=30),
                        "background_shares_at_original_A_origin":int(actual.shares) if actual is not None else np.nan,
                        "background_next_requested_shares":int(request.desired_shares) if request is not None else np.nan,
                        "support_held_at_original_A_origin":bool(actual is not None and actual.shares > 0 and name==ACCOUNTS[2]),
                        "support_cycle_entry_origin_at_Aorigin":support_origin,
                        "context_role":"OWN_POSITION_AT_ORIGINAL_A_ENTRY_ORIGIN" if actual is not None and actual.shares > 0 else
                            ("FLAT_AT_ORIGINAL_A_ENTRY_ORIGIN" if actual is not None else "NO_VIEW_FIRST_ORIGIN_BEFORE_ACCOUNT_START"),
                        "entry_identity_role":"REAL_SAVED_ENTRY_ONLY_INCLUDED_AFTER_ACTUAL_ENTRY_OPEN"})
    return normalized(pd.DataFrame(points)), normalized(pd.DataFrame(contexts)), normalized(pd.DataFrame(entry_contexts))


def known_context_columns(frame):
    return [x for x in frame.columns if not x.startswith("next_") and not x.startswith("planned_")
            and not x.startswith("one_shot_") and not x.startswith("background_actual_next_")]
