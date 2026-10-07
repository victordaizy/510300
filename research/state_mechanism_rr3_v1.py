"""状态与经济机制的有限比较：事前目标、失效点及压力成本后3:1准入。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "reports/research/510300_state_mechanism_rr3_v1"
STUDY = "510300_STATE_MECHANISM_RR3_V1"
COSTS = {"BASE": (.0002, .0005), "STRESS": (.0004, .001)}
POLICIES = ("TREND_ONLY", "REPAIR_ONLY", "UNION", "PRICE_ROUTE", "PRICE_ROUTE_MATCHED", "JOINT_ROUTE", "CASH", "BUY_HOLD")
KINDS = ("TREND_PULLBACK", "SHOCK_REPAIR")
START, END = "2021-01-04", "2026-08-14"
RR = 3.0
RISK_FRACTION = .01
TICK = .001
SOURCES = {
    "features.parquet": "reports/research/510300_holiday_event_capital_risk_v1/inputs/features.parquet",
    "dividends.csv": "reports/research/510300_holiday_event_capital_risk_v1/inputs/dividends.csv",
    "known_information.parquet": "reports/research/510300_integrated_macro_micro_prediction_v1/results/model_inputs.parquet",
    "previous_price_strategy.py": "research/simple_price_entry_exit_v1.py",
    "previous_context_strategy.py": "research/contextual_expert_tracking_v1.py",
    "previous_regime_strategy.py": "research/sequential_regime_strategy_selection_inputs_v1.py",
    "previous_conditional_mechanism_status.json": "reports/research/510300_conditional_mechanism_map_v1_status.json",
    "current_mandate.json": "config/510300_existing_data_training_mandate_v1.json",
}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fee(value, cost):
    return max(value * COSTS[cost][0], 5.0)


def quote(raw, side, cost):
    adjusted = raw * (1 + COSTS[cost][1] * (1 if side == "BUY" else -1))
    return (math.ceil(adjusted / TICK - 1e-10) if side == "BUY" else math.floor(adjusted / TICK + 1e-10)) * TICK


def ticket(cash, equity, reference_price, target_price, stop_price):
    """在开盘前按压力成本计算份额和最高买价；开盘后只检验能否成交。"""
    if not (0 < stop_price < reference_price < target_price):
        return None, "INVALID_TARGET_STOP_GEOMETRY"
    entry = quote(reference_price, "BUY", "STRESS")
    target = quote(target_price, "SELL", "STRESS")
    stop = quote(stop_price, "SELL", "STRESS")
    if target <= entry or stop <= 0:
        return None, "NO_NET_UPSIDE"
    # 先用比例费用求份额上界，再逐整手检查最低佣金及最高买价。
    approximate_limit = (target * (1 - .0004) + RR * stop * (1 - .0004)) / ((1 + RR) * (1 + .0004))
    unit_risk_at_limit = approximate_limit * (1 + .0004) - stop * (1 - .0004)
    if approximate_limit < entry or unit_risk_at_limit <= 0:
        return None, "PLANNED_RR_BELOW_3"
    quantity = int(min(cash / max(approximate_limit, TICK), equity * RISK_FRACTION / unit_risk_at_limit) // 100) * 100
    while quantity >= 100:
        target_net = quantity * target - fee(quantity * target, "STRESS")
        stop_net = quantity * stop - fee(quantity * stop, "STRESS")
        cap = (target_net + RR * stop_net) / (1 + RR)
        limit = min(cap / (quantity * (1 + .0004)), (cap - 5) / quantity)
        limit = math.floor(limit / TICK + 1e-10) * TICK
        buy_cost = quantity * limit + fee(quantity * limit, "STRESS")
        loss = buy_cost - stop_net
        gain = target_net - buy_cost
        reference_cost = quantity * entry + fee(quantity * entry, "STRESS")
        if limit >= entry and loss > 0 and gain / loss >= RR - 1e-10 and loss <= equity * RISK_FRACTION + 1e-7 and buy_cost <= cash + 1e-7:
            return {"quantity": quantity, "buy_limit": limit, "target_raw": target_price,
                    "stop_raw": stop_price, "planned_loss_cny": loss, "planned_gain_cny": gain,
                    "planned_net_rr_at_limit": gain / loss,
                    "planned_net_rr_at_reference": (target_net - reference_cost) / (reference_cost - stop_net),
                    "risk_fraction_of_equity": loss / equity}, "READY"
        quantity -= 100
    return None, "RISK_OR_MINIMUM_LOT_LIMIT"


def prepare(d, information):
    """生成已发生的条件事实和候选；没有读取任何候选未来损益。"""
    d = d.copy().reset_index(drop=True)
    x = information.iloc[:len(d)].copy().reset_index(drop=True)
    assert d.date.tolist() == x.date.tolist()
    mean20, mean120 = d.ac.rolling(20).mean(), d.ac.rolling(120).mean()
    pullback = (d.ac > mean120) & (d.ac > mean20) & (d.ac.shift() <= mean20.shift())
    location = (d.ac - d.al) / (d.ah - d.al).replace(0, np.nan)
    shock = (np.log(d.ac.shift(1) / d.ac.shift(6)) < np.log(.97)) & (d.ac > d.ac.shift(1)) & (location >= .6)
    x["market_state"] = d.state.shift(1)
    x["flow_change3"] = (x.flow5 - x.flow5.shift(3)).where(x.flow_known & x.flow_known.shift(1, fill_value=False) & x.flow_known.shift(2, fill_value=False) & x.flow_known.shift(3, fill_value=False))
    x["breadth_change3"] = (x.flow_breadth5 - x.flow_breadth5.shift(3)).where(x.flow_change3.notna())
    x["state_information_known"] = x.funding_known & x.flow_known & x.pmi_known & x.flow_change3.notna() & x.breadth_change3.notna() & x.market_state.isin(["UP", "RANGE", "DOWN"])
    x["macro_state"] = np.select(
        [(x.pmi_level >= 0) & (x.pmi_change3 >= 0), (x.pmi_level >= 0) & (x.pmi_change3 < 0),
         (x.pmi_level < 0) & (x.pmi_change3 >= 0), (x.pmi_level < 0) & (x.pmi_change3 < 0)],
        ["EXPANDING_IMPROVING", "EXPANDING_SLOWING", "CONTRACTING_IMPROVING", "CONTRACTING_SLOWING"], default="UNKNOWN")
    x["funding_state"] = np.select([(x.funding_gap_pp > 0) & (x.funding_change5_pp > 0), x.funding_change5_pp <= 0], ["TIGHTENING", "EASING_OR_STABLE"], default="MIXED")
    x["stock_participation_support"] = (x.flow5 >= 0) & (x.flow_breadth5 >= .5)
    x["stock_participation_recovering"] = (x.flow_change3 >= 0) & (x.breadth_change3 >= 0)
    x["trend_mechanism_support"] = x.state_information_known & (x.pmi_level >= 0) & (x.pmi_change3 >= 0) & ~x.funding_tightening & x.stock_participation_support
    x["repair_mechanism_support"] = x.state_information_known & (x.pmi_change3 >= 0) & (x.funding_change5_pp <= 0) & x.stock_participation_recovering
    rows = []
    for t in range(120, len(d) - 1):
        for kind, condition in ((KINDS[0], pullback), (KINDS[1], shock)):
            if not bool(condition.iloc[t]):
                continue
            target = d.ac.iloc[t - 60:t].max() if kind == KINDS[0] else d.ac.iloc[t - 6]
            left = t - 4 if kind == KINDS[0] else t - 5
            stop = d.al.iloc[left:t + 1].min() - .1 * d.atr20.iloc[t]
            scale = d.close.iloc[t] / d.ac.iloc[t]
            # 除息日开盘新买没有前一登记日权益，已知除息机械调整所有价格锚点。
            cash_adjustment = float(d.dividend.iloc[t + 1])
            row = {"signal_id": f"{kind}_{d.date.iloc[t]}", "kind": kind, "signal_idx": t,
                   "signal_date": d.date.iloc[t], "entry_idx": t + 1, "entry_date": d.date.iloc[t + 1],
                   "reference_raw": float(d.close.iloc[t] - cash_adjustment),
                   "target_raw": float(target * scale - cash_adjustment), "stop_raw": float(stop * scale - cash_adjustment),
                   "target_index": float(target), "stop_index": float(stop), "entry_ex_adjustment": cash_adjustment,
                   "max_holding_sessions": 20 if kind == KINDS[0] else 5,
                   "target_economic_claim": "恢复到已出现过的六十日高位" if kind == KINDS[0] else "修复到急跌前收盘",
                   "fact_origin": "原第23轮T4或R3触发；价格目标及结构失效在本轮事前固定，尚非公平价值证明。"}
            plan, reason = ticket(200000, 200000, row["reference_raw"], row["target_raw"], row["stop_raw"])
            row["reference_ticket_status"] = reason
            row["reference_planned_net_rr"] = plan["planned_net_rr_at_reference"] if plan else np.nan
            for field in ("market_state", "macro_state", "funding_state", "state_information_known", "trend_mechanism_support", "repair_mechanism_support", "preholiday3", "known_scheduled_event", "global_shock"):
                row[field] = x[field].iloc[t + 1]
            rows.append(row)
    return x, pd.DataFrame(rows)


def allowed(policy, kind, row):
    if policy in ("TREND_ONLY", "REPAIR_ONLY", "UNION"):
        return (policy != "TREND_ONLY" or kind == KINDS[0]) and (policy != "REPAIR_ONLY" or kind == KINDS[1]), "STRATEGY_TYPE"
    price_ok = row.market_state == "UP" if kind == KINDS[0] else row.market_state in ("RANGE", "DOWN")
    if not price_ok:
        return False, "PRICE_STATE_NOT_FOR_THIS_STRATEGY"
    if policy in ("PRICE_ROUTE_MATCHED", "JOINT_ROUTE") and not bool(row.state_information_known):
        return False, "NO_VIEW_STATE_INPUTS"
    if policy == "JOINT_ROUTE":
        support = row.trend_mechanism_support if kind == KINDS[0] else row.repair_mechanism_support
        return bool(support), "MECHANISM_SUPPORTED" if support else "MACRO_FUNDING_PARTICIPATION_NOT_SUPPORTIVE"
    return True, "PRICE_STATE_SUPPORTED"


def can_fill(d, i, side):
    reference = float(d.close.iloc[i - 1] - d.dividend.iloc[i]) if i else float(d.open.iloc[i])
    price = float(d.open.iloc[i])
    if price <= 0 or float(d.volume.iloc[i]) <= 0:
        return False
    # 日线不能识别开盘排队；触及不利方向涨跌停一律不假设可成交。
    if side == "BUY" and price >= round(reference * 1.1, 3) - 1e-9:
        return False
    if side == "SELL" and price <= round(reference * .9, 3) + 1e-9:
        return False
    return True


def simulate(d, dividends, x, candidates, policy, cost, start, end):
    cash, quantity, receivable = 200000.0, 0, 0.0
    previous_equity, peak = 200000.0, 200000.0
    active, pending_exit = None, None
    book, ledger, trades, decisions = [], [], [], []
    groups = {int(k): g.to_dict("records") for k, g in candidates.groupby("entry_idx")} if len(candidates) else {}
    ex = {date: g for date, g in dividends.groupby("ex_date")}
    for i in range(start, end + 1):
        day, fees = d.iloc[i], 0.0
        accrued, paid = 0.0, 0.0
        old_quantity = quantity
        if day.date in ex and quantity:
            for event in ex[day.date].itertuples():
                amount = quantity * float(event.cash_dividend_per_share)
                book.append({"payment_date": event.payment_date, "amount": amount})
                receivable += amount
                accrued += amount
                active["dividend_cny"] += amount
                active["dividend_per_share"] += float(event.cash_dividend_per_share)
        remaining = []
        for item in book:
            if item["payment_date"] <= day.date:
                cash += item["amount"]
                receivable -= item["amount"]
                paid += item["amount"]
            else:
                remaining.append(item)
        book = remaining
        if quantity and policy in ("PRICE_ROUTE", "PRICE_ROUTE_MATCHED", "JOINT_ROUTE") and pending_exit is None:
            permitted, reason = allowed(policy, active["kind"], x.iloc[i])
            if not permitted and reason != "NO_VIEW_STATE_INPUTS":
                pending_exit = "KNOWN_STATE_INVALIDATION"
        if quantity and pending_exit is not None and i > active["entry_idx"]:
            if can_fill(d, i, "SELL"):
                px = quote(float(day.open), "SELL", cost)
                commission = fee(quantity * px, cost)
                cash += quantity * px - commission
                fees += commission
                active.update({"exit_idx": i, "exit_date": day.date, "exit_price": px, "exit_fee": commission,
                               "exit_reason": pending_exit, "holding_sessions": i - active["entry_idx"],
                               "net_pnl": quantity * (px - active["entry_price"]) + active["dividend_cny"] - active["entry_fee"] - commission})
                active["realized_R"] = active["net_pnl"] / active["planned_loss_cny"] if active.get("planned_loss_cny", 0) > 0 else np.nan
                trades.append(active.copy())
                decisions.append({"date": day.date, "idx": i, "signal_id": active["signal_id"], "action": "SELL_FILLED", "reason": pending_exit})
                quantity, active, pending_exit = 0, None, None
            else:
                decisions.append({"date": day.date, "idx": i, "signal_id": active["signal_id"], "action": "SELL_DEFERRED", "reason": pending_exit})
        if not old_quantity and policy not in ("CASH", "BUY_HOLD"):
            for candidate in groups.get(i, []):
                permitted, reason = allowed(policy, candidate["kind"], x.iloc[i])
                if not permitted:
                    decisions.append({"date": day.date, "idx": i, "signal_id": candidate["signal_id"], "action": "NO_ENTRY", "reason": reason})
                    continue
                plan, reason = ticket(cash, previous_equity, candidate["reference_raw"], candidate["target_raw"], candidate["stop_raw"])
                if plan is None:
                    decisions.append({"date": day.date, "idx": i, "signal_id": candidate["signal_id"], "action": "NO_ENTRY", "reason": reason})
                    continue
                px = quote(float(day.open), "BUY", cost)
                if not can_fill(d, i, "BUY") or px > plan["buy_limit"] + 1e-10 or float(day.open) <= plan["stop_raw"]:
                    decisions.append({"date": day.date, "idx": i, "signal_id": candidate["signal_id"], "action": "BUY_UNFILLED", "reason": "OPEN_GAP_LIMIT_OR_STOP_INVALIDATED", **plan})
                    break
                quantity = plan["quantity"]
                commission = fee(quantity * px, cost)
                cash -= quantity * px + commission
                fees += commission
                active = {**candidate, **plan, "entry_idx": i, "entry_date": day.date, "entry_price": px,
                          "entry_fee": commission, "entry_equity": previous_equity, "dividend_cny": 0.0,
                          "dividend_per_share": 0.0, "decision_time": str(x.decision_time.iloc[i]),
                          "entry_market_state": x.market_state.iloc[i], "entry_macro_state": x.macro_state.iloc[i],
                          "entry_funding_state": x.funding_state.iloc[i]}
                stress_buy = quantity * quote(float(day.open), "BUY", "STRESS")
                stop_net = quantity * quote(plan["stop_raw"], "SELL", "STRESS")
                target_net = quantity * quote(plan["target_raw"], "SELL", "STRESS")
                stress_buy += fee(stress_buy, "STRESS")
                stop_net -= fee(stop_net, "STRESS")
                target_net -= fee(target_net, "STRESS")
                active["actual_open_stress_net_rr"] = (target_net - stress_buy) / (stress_buy - stop_net) if stress_buy > stop_net else np.nan
                # 基础费用较小不意味着可绕过统一的压力成本门槛。
                if not np.isfinite(active["actual_open_stress_net_rr"]) or active["actual_open_stress_net_rr"] < RR - 1e-10:
                    cash += quantity * px + commission
                    fees -= commission
                    quantity, active = 0, None
                    decisions.append({"date": day.date, "idx": i, "signal_id": candidate["signal_id"], "action": "BUY_UNFILLED", "reason": "OPEN_STRESS_RR_BELOW_3"})
                    break
                decisions.append({"date": day.date, "idx": i, "signal_id": active["signal_id"], "action": "BUY_FILLED", "reason": "PREOPEN_RR3_LIMIT_TICKET", **plan})
                break
        elif policy == "BUY_HOLD" and i == start:
            px = quote(float(day.open), "BUY", cost)
            quantity = int(cash / (px * (1 + COSTS[cost][0])) // 100) * 100
            while quantity and quantity * px + fee(quantity * px, cost) > cash:
                quantity -= 100
            commission = fee(quantity * px, cost) if quantity else 0.0
            cash -= quantity * px + commission
            fees += commission
            active = {"signal_id": "BUY_HOLD", "kind": "BUY_HOLD", "entry_idx": i, "entry_date": day.date,
                      "entry_price": px, "entry_fee": commission, "dividend_cny": 0.0, "dividend_per_share": 0.0} if quantity else None
        if quantity and policy != "BUY_HOLD" and pending_exit is None:
            value = float(day.close) + active["dividend_per_share"]
            if value <= active["stop_raw"]:
                pending_exit = "STRUCTURAL_STOP_CLOSE_NEXT_OPEN"
            elif value >= active["target_raw"]:
                pending_exit = "FROZEN_TARGET_CLOSE_NEXT_OPEN"
            elif i - active["entry_idx"] + 1 >= active["max_holding_sessions"]:
                pending_exit = "STRATEGY_TIME_LIMIT"
        equity = cash + quantity * float(day.close) + receivable
        peak = max(peak, equity)
        assert cash >= -1e-7 and quantity % 100 == 0 and receivable >= -1e-7
        ledger.append({"idx": i, "date": day.date, "cash_cny": cash, "shares": quantity, "close": float(day.close),
                       "receivable_cny": receivable, "dividend_accrual_cny": accrued, "dividend_paid_cny": paid,
                       "fees_cny": fees, "equity_cny": equity, "daily_return": equity / previous_equity - 1,
                       "drawdown": equity / peak - 1, "exposure": quantity * float(day.close) / equity,
                       "active_signal_id": active["signal_id"] if active else "", "pending_exit": pending_exit or ""})
        previous_equity = equity
    reserve = 0.0
    if quantity:
        mark = float(d.close.iloc[end])
        px = quote(mark, "SELL", cost)
        reserve = quantity * (mark - px) + fee(quantity * px, cost)
    terminal = {"open_position": bool(quantity), "shares": quantity, "terminal_haircut_cny": reserve,
                "pending_exit": pending_exit, "active": active, "outstanding_dividends": book}
    return pd.DataFrame(ledger), pd.DataFrame(trades), pd.DataFrame(decisions), terminal


def summarize(ledger, trades, terminal):
    returns = ledger.daily_return.to_numpy(float).copy()
    ending = float(ledger.equity_cny.iloc[-1]) - terminal["terminal_haircut_cny"]
    prev = float(ledger.equity_cny.iloc[-2]) if len(ledger) > 1 else 200000.0
    returns[-1] = ending / prev - 1
    nav = np.r_[200000., 200000. * np.cumprod(1 + returns)]
    std = returns.std(ddof=1)
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 1e-14 else None
    cagr = float((ending / 200000.) ** (252 / len(ledger)) - 1)
    drawdown = float(-(nav / np.maximum.accumulate(nav) - 1).min())
    pnl = trades.net_pnl.to_numpy(float) if len(trades) else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {"start": str(ledger.date.iloc[0]), "end": str(ledger.date.iloc[-1]), "days": len(ledger),
            "ending_equity_cny": ending, "net_profit_cny": ending - 200000., "net_cagr": cagr,
            "net_sharpe": sharpe, "max_drawdown": drawdown, "completed_cycles": len(pnl),
            "win_rate": float(np.mean(pnl > 0)) if len(pnl) else None,
            "realized_cash_payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None,
            "mean_realized_R": float(trades.realized_R.mean()) if len(trades) and "realized_R" in trades else None,
            "completed_cycles_mean_net_pnl": float(pnl.mean()) if len(pnl) else None,
            "largest_cycle_net_pnl": float(pnl.max()) if len(pnl) else None,
            "remaining_saved_cycle_profit_after_largest": float(pnl.sum() - pnl.max()) if len(pnl) else None,
            "average_exposure": float(ledger.exposure.mean()), "commissions_cny": float(ledger.fees_cny.sum()),
            "numerical_target_pass": bool(cagr >= .1 and sharpe is not None and sharpe >= 1.2 and drawdown <= .1)}


def tests():
    plan, reason = ticket(200000, 200000, 4.0, 4.8, 3.85)
    assert reason == "READY" and plan["planned_net_rr_at_limit"] >= 3 - 1e-10
    assert plan["planned_loss_cny"] <= 2000 and plan["quantity"] % 100 == 0
    assert ticket(200000, 200000, 4.0, 4.1, 3.85)[0] is None
    dates = pd.date_range("2020-01-01", periods=7).strftime("%Y-%m-%d")
    d = pd.DataFrame({"date": dates, "open": [4., 4., 3.6, 4., 4., 4., 4.],
                      "close": [4., 3.7, 3.6, 4., 4., 4., 4.], "dividend": 0., "volume": 10000.})
    x = pd.DataFrame({"decision_time": pd.to_datetime(dates) + pd.Timedelta(hours=9), "market_state": "RANGE",
                      "macro_state": "EXPANDING_IMPROVING", "funding_state": "EASING_OR_STABLE",
                      "state_information_known": True, "trend_mechanism_support": True, "repair_mechanism_support": True})
    c = pd.DataFrame([{"signal_id": "人工", "kind": KINDS[1], "entry_idx": 1, "reference_raw": 4.,
                       "target_raw": 4.8, "stop_raw": 3.85, "max_holding_sessions": 5}])
    dv = pd.DataFrame(columns=["ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, trades, _, _ = simulate(d, dv, x, c, "REPAIR_ONLY", "STRESS", 0, 6)
    assert len(trades) == 1 and int(trades.entry_idx.iloc[0]) == 1 and int(trades.exit_idx.iloc[0]) >= 2
    assert float(trades.realized_R.iloc[0]) < -1
    gap = d.copy()
    gap.loc[1, "open"] = plan["buy_limit"] + .01
    nofill = simulate(gap, dv, x, c, "REPAIR_ONLY", "STRESS", 0, 6)
    assert nofill[0].shares.eq(0).all()
    dividend_case = d.copy()
    dividend_case["open"] = [4., 4., 3.9, 3.9, 3.9, 3.9, 3.9]
    dividend_case["close"] = dividend_case.open
    dividend_case.loc[2, "dividend"] = .1
    dv = pd.DataFrame([{"ex_date": dates[2], "payment_date": dates[4], "cash_dividend_per_share": .1}])
    dd = simulate(dividend_case, dv, x, c, "REPAIR_ONLY", "STRESS", 0, 6)
    assert dd[0].shares.iloc[2] > 0 and dd[0].receivable_cny.iloc[2] > 0
    assert dd[0].receivable_cny.iloc[4] == 0 and "STRUCTURAL_STOP" not in str(dd[1].get("exit_reason", ""))
    return ["压力3:1及1%预算、最低佣金整手", "不合格空间拒绝", "开盘跳高不成交", "T+1和跳空可能超过计划损失", "除息权益及到账不制造止损"]


def load(root):
    d = pd.read_parquet(root / "inputs/features.parquet")
    d = d.loc[d.date.le(END)].copy().reset_index(drop=True)
    v = pd.read_parquet(root / "inputs/known_information.parquet")
    dividends = pd.read_csv(root / "inputs/dividends.csv")
    return d, dividends, v


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("本轮已经冻结，禁止覆盖。")
    checks = tests()
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "code").mkdir(exist_ok=True)
    for name, source in SOURCES.items():
        shutil.copy2(WORKSPACE / source, root / "inputs" / name)
    shutil.copy2(Path(__file__), root / "code/state_mechanism_rr3_v1.py")
    d, dividends, information = load(root)
    x, candidates = prepare(d, information)
    # 原账户处理方式要求登记日是除息前的最近交易日，逐条确认本地事件满足。
    dates = d.date.to_list()
    for event in dividends.itertuples():
        if event.ex_date in dates:
            i = dates.index(event.ex_date)
            assert i > 0 and dates[i - 1] == event.record_date
    protocol = {
        "study_id": STUDY, "frozen_at": now(), "period": [START, END],
        "user_direction": ["不同市场状态、流动性、宏观对应不同策略", "只做能够用事实解释的高盈亏比机会", "确认入场前压力成本后至少3:1"],
        "primary": "JOINT_ROUTE", "primary_comparison": "JOINT_ROUTE_MINUS_PRICE_ROUTE_MATCHED", "policies": POLICIES,
        "method": "先固定两套不同触发、目标、失效点及持有期限，再检验宏观资金是否改变其适用性；不拟合收益，不以历史赢家配置状态。",
        "market_state": "沿用此前trend_z>1为UP、<-1为DOWN、其余RANGE；使用前一日收盘。",
        "trend_entry": "复用第23轮T4：含分红价高于120日均线，前一日不高于20日均线、当日重新站上。目标为触发前60日最高收盘；失效为包含触发日的5日最低价减0.1ATR20；最多20个交易日。",
        "repair_entry": "复用第23轮R3：前一日结束的5日累计跌超3%，今日收盘回升且收盘位置>=0.6。目标为该5日急跌前收盘；失效为含确认日6日最低价减0.1ATR20；最多5个交易日。",
        "price_route": "UP只允许趋势回调；RANGE或DOWN只允许急跌修复。",
        "joint_trend": "价格路由成立，已发布新订单>=50且连续3个月变化>=0，DR007没有同时高于政策率且5日上升，成分股成交分类5日强度>=0且正强度权重占比>=50%。",
        "joint_repair": "价格路由成立，已发布新订单3个月变化>=0，DR007最近5日不再上升，成分股成交分类强度及正强度占比均较3个可用观察日前不减。新订单水平可以仍低于50。",
        "mechanism_claim_limit": "这些是经济机制假设及可观测支持，目标价格是历史位置锚点，不是估值证明；成交分类不是具名机构净买入。必须报告反证和实际目标实现。",
        "clock": "执行日09:00使用既有已核定available_at；价格候选来自前日收盘。开盘前确定最高买价和份额，09:30仅检验可成交性；不用当天高低价判断开盘买入。",
        "rr_gate": {"minimum_planned_net_reward_risk": RR, "cost": "STRESS", "target_and_stop": "随信号固定，不为提高比例放大目标或缩窄止损。", "gap_up": "开盘压力报价使比例不足3则不成交", "sizing": "最高允许买价下计划失效损失不超过全账户权益1%，份额和现金均在开盘前确定。"},
        "execution": "100份、tick0.001、佣金最低5；T+1；停止/目标在收盘确认后下一个可卖开盘退出，不假定按线成交。已知适用状态失效在09:00可请求当日可卖开盘；缺资料不强行卖。退出请求持续，卖出日不从另一策略即时重入。",
        "dividends": "每笔只计实际登记持有权益，除息应收及支付分账；持仓目标比较含应得股息的每份清算价值。已知除息日新买无登记权益，价格锚点机械除息调整。",
        "missing": "PRICE_ROUTE_MATCHED与JOINT_ROUTE新入场使用同一信息覆盖；无资料仅NO_VIEW，无观点不等于看空。其他单策略与价格路由为不同覆盖对照，分别报告。",
        "holiday_and_events": "沿用已知节前三日、统计局事前日程、已公开海外冲击标签分层记录；前轮普遍避险未获支持，本轮不反转或优化那些日历规则。跨休市与T+1跳空损失均留在账户。",
        "account": {"capital_cny": 200000, "assets": ["510300.SH", "CASH_CNY"], "annual_days": 252, "cash_return": 0, "costs": COSTS, "planned_risk_fraction": RISK_FRACTION},
        "target": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1, "annual_cycle_quota": None},
        "inference": "一个主要配对增量；20日共同循环区块2000次、固定种子202609243；报告年化算术增量95%及单侧5%下界。各策略状态成交分组仅描述，不能用同批历史选择映射。",
        "old_research_difference": "第9轮是基于价格趋势/波动的机制收益追踪，第179轮是在两个旧合成账户内按状态历史夏普选胜者；本轮不用这两个合成账户、不恢复85/15。第23轮仅复用两个可解释触发，结构目标和3:1是用户本次新增合同。旧CF/DR条件图仍为有限描述，不把其部分通过升级。",
        "stop_after_frozen_failure": True, "parameter_grid": 0, "model_fits": 0,
        "source_first_vintage_authenticated": False, "independent_forward_observations": 0,
        "new_collection": False, "orders_authorized": False, "review_package": False,
        "pre_return_counts": {"candidate_signals_all_history": len(candidates), "reference_rr3_ready_all_history": int(candidates.reference_ticket_status.eq("READY").sum()), "common_state_days": int(x.state_information_known.sum())},
        "mechanism_checks": checks,
    }
    save_json(root / "protocol.json", protocol)
    paths = [root / "protocol.json", *[p for p in (root / "inputs").iterdir() if p.is_file()], *[p for p in (root / "code").iterdir() if p.is_file()]]
    save_json(root / "freeze.json", {"frozen_at": now(), "before_new_account_results": True,
                                  "files": [{"path": p.relative_to(root).as_posix(), "sha256": digest(p)} for p in paths]})
    print("状态、两套机制、事前3:1及统一风险预算已冻结；未查看新账户结果。", flush=True)


def run(root):
    frozen = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for item in frozen["files"]:
        assert digest(root / item["path"]) == item["sha256"], item["path"]
    if (root / "RUN_STARTED.json").exists():
        raise RuntimeError("本轮已经开始；不得改参数重跑。")
    save_json(root / "RUN_STARTED.json", {"started_at": now()})
    out = root / "results"
    out.mkdir(exist_ok=True)
    d, dividends, information = load(root)
    x, candidates = prepare(d, information)
    x.to_parquet(out / "known_states.parquet", index=False)
    candidates.to_parquet(out / "all_candidate_facts.parquet", index=False)
    start, end = int(d.index[d.date.ge(START)][0]), len(d) - 1
    accounts, metrics, slices = {}, [], []
    verification_rows = 0
    for cost in COSTS:
        for policy in POLICIES:
            result = simulate(d, dividends, x, candidates, policy, cost, start, end)
            ledger, trades, decisions, terminal = result
            accounts[(policy, cost)] = result
            folder = out / "accounts" / cost / policy
            folder.mkdir(parents=True, exist_ok=True)
            for name, frame in (("ledger", ledger), ("trades", trades), ("decisions", decisions)):
                frame.to_parquet(folder / f"{name}.parquet", index=False)
            save_json(folder / "terminal.json", terminal)
            m = {"policy": policy, "cost": cost, **summarize(ledger, trades, terminal)}
            save_json(folder / "metrics.json", m)
            metrics.append(m)
            np.testing.assert_allclose(ledger.cash_cny + ledger.shares * ledger.close + ledger.receivable_cny, ledger.equity_cny, atol=1e-7, rtol=0)
            if len(trades):
                assert (trades.exit_idx > trades.entry_idx).all()
                if policy != "BUY_HOLD":
                    assert trades.planned_net_rr_at_limit.ge(3 - 1e-10).all()
                    assert trades.actual_open_stress_net_rr.ge(3 - 1e-10).all()
                    for keys, g in trades.groupby(["kind", "entry_market_state", "entry_macro_state", "entry_funding_state"], dropna=False):
                        slices.append({"policy": policy, "cost": cost, "kind": keys[0], "market_state": keys[1], "macro_state": keys[2], "funding_state": keys[3], "cycles": len(g), "net_pnl": float(g.net_pnl.sum()), "mean_R": float(g.realized_R.mean()), "target_exit_count": int(g.exit_reason.eq("FROZEN_TARGET_CLOSE_NEXT_OPEN").sum()), "scope": "描述，不用于重新选择状态策略"})
            if not terminal["open_position"]:
                assert abs((trades.net_pnl.sum() if len(trades) else 0) - m["net_profit_cny"]) < 1e-6
            verification_rows += len(ledger)
        print(f"已完成{cost}八条完整账户，包含现金和买入持有对照。", flush=True)
    save_json(out / "account_metrics.json", metrics)
    save_json(out / "strategy_state_outcomes.json", slices)
    current, baseline = accounts[("JOINT_ROUTE", "STRESS")], accounts[("PRICE_ROUTE_MATCHED", "STRESS")]
    def returns(result):
        r = result[0].daily_return.to_numpy(float).copy()
        r[-1] = (result[0].equity_cny.iloc[-1] - result[3]["terminal_haircut_cny"]) / result[0].equity_cny.iloc[-2] - 1
        return r
    delta = returns(current) - returns(baseline)
    rng = np.random.default_rng(202609243)
    starts = rng.integers(0, len(delta), size=(2000, math.ceil(len(delta) / 20)))
    indices = ((starts[:, :, None] + np.arange(20)) % len(delta)).reshape(2000, -1)[:, :len(delta)]
    np.savez_compressed(out / "paired_draws.npz", indices=indices)
    samples = delta[indices].mean(axis=1) * 252
    paired = {"comparison": "JOINT_ROUTE_MINUS_PRICE_ROUTE_MATCHED", "annualized_arithmetic_increment": float(delta.mean() * 252),
              "ci95": np.quantile(samples, [.025, .975]), "one_sided_95_lower": float(np.quantile(samples, .05)),
              "reliable_positive_increment": bool(np.quantile(samples, .05) > 0)}
    save_json(out / "paired_increment.json", paired)
    primary = [r for r in metrics if r["policy"] == "JOINT_ROUTE"]
    eligible = candidates[candidates.entry_date.between(START, END)]
    counts = {"signals": len(eligible), "reference_rr3_ready": int(eligible.reference_ticket_status.eq("READY").sum()),
              "reference_rejections": eligible.reference_ticket_status.value_counts().to_dict(),
              "signals_by_strategy": eligible.kind.value_counts().to_dict(), "common_state_days": int(x.loc[x.date.between(START, END), "state_information_known"].sum())}
    save_json(out / "candidate_coverage.json", counts)
    verification = {"status": "PASS_FREEZE_ACCOUNT_IDENTITIES_T_PLUS_ONE_AND_ALL_ENTRY_RR3", "checked_at": now(),
                    "accounts": len(metrics), "account_rows": verification_rows, "model_fits": 0,
                    "independent_validation": False, "current_view": "NO_VIEW"}
    save_json(root / "verification.json", verification)
    reached = all(item["numerical_target_pass"] for item in primary) and paired["reliable_positive_increment"]
    save_json(root / "result.json", {"study_id": STUDY, "completed_at": now(),
              "status": "HISTORICAL_NUMERICAL_PASS_UNVALIDATED" if reached else "FROZEN_STATE_MECHANISM_RR3_TARGET_NOT_MET",
              "primary_accounts": primary, "candidate_coverage": counts, "paired_increment": paired,
              "numeric_primary_pass_both_costs": all(item["numerical_target_pass"] for item in primary),
              "new_accounts": len(metrics), "new_model_fits": 0, "goal_achieved": False,
              "independent_forward_observations": 0, "current_view": "NO_VIEW", "new_collection": False,
              "orders_authorized": False, "terminated_strategy_revived": False})
    print("本轮固定状态与事前3:1比较完成，全部结果已保存。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="510300状态、经济机制及3:1事前收益风险比较")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze(ROOT) if args.command == "freeze" else run(ROOT)
