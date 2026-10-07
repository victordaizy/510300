"""固定点位账户与 NR7 补充机会的日线执行器。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research import daily_supply_test_v1 as risk_base


CAPITAL = 200000.0
PARENT_A = "CORE_AUXILIARY_DRAWDOWN_GATE"
PARENT_B = "LAG_CONFIRMED_RUNS_AUXILIARY"
POLICIES = ("POINT_A", "POINT_B", "NR7_ONLY", "POINT_A_PLUS_NR7")
NAMES = {
    "POINT_A": "回撤限制点位账户",
    "POINT_B": "相关确认点位账户",
    "NR7_ONLY": "窄幅突破单独账户",
    "POINT_A_PLUS_NR7": "回撤限制点位加窄幅突破",
}


def nr7_signals(data: pd.DataFrame) -> pd.DataFrame:
    """准备日振幅严格小于此前六日；只承认紧接次日的收盘向上突破。"""
    d = data.reset_index(drop=True)
    if not d.date.is_monotonic_increasing or not d.date.is_unique:
        raise ValueError("日线日期必须唯一、递增。")
    raw = d[["high", "low"]].to_numpy(float)
    if not np.isfinite(raw).all() or (raw[:, 0] < raw[:, 1]).any():
        raise ValueError("日线高低价无效。")
    # 用报价刻度比较振幅，避免同样宽度因浮点误差成为严格新低。
    ticks = np.rint(raw / .001).astype(np.int64)
    width = pd.Series(ticks[:, 0] - ticks[:, 1])
    setup = width.lt(width.shift().rolling(6, min_periods=6).min()) & width.gt(0)
    previous_high = d.ah.shift()
    signal = setup.shift(fill_value=False) & d.ac.gt(previous_high + 1e-10)
    result = pd.DataFrame({
        "date": d.date, "range_ticks": width, "nr7_setup": setup,
        "entry_event": signal, "setup_date": d.date.shift().where(signal),
        "stop_index": d.al.shift().where(signal),
        "setup_high_index": previous_high.where(signal),
    })
    for name in ("daily_hist_rising", "weekly_hist_rising", "relative_volume", "rv_ratio"):
        if name in d:
            result[name] = d[name].to_numpy()
    return result


def fill(raw: float, side: int, cost: str) -> float:
    return risk_base.fill(raw, side, cost)


def fee(notional: float, cost: str) -> float:
    return risk_base.fee(notional, cost)


def open_blocked(d: pd.DataFrame, i: int, side: int) -> bool:
    """开盘执行仅用已知昨收、当天除息和开盘价，不读取当日收盘或成交量。"""
    raw = float(d.open.iloc[i])
    if not np.isfinite(raw) or raw <= 0:
        return True
    reference = float(d.close.iloc[i - 1] - d.dividend.iloc[i])
    boundary = round(reference * (1 + .1 * side), 3)
    return bool(raw >= boundary - .0005 if side > 0 else raw <= boundary + .0005)


def account(data, dividends, signals, parents, risks, policy, cost, start):
    """20万元现金账户；次日开盘成交、登记/应收/到账分开、期末自然持仓。"""
    if policy not in POLICIES:
        raise ValueError("未知固定账户方案。")
    d = data.reset_index(drop=True)
    sig = signals.reset_index(drop=True)
    if not pd.DatetimeIndex(d.date).equals(pd.DatetimeIndex(sig.date)):
        raise ValueError("窄幅信号与价格日期不一致。")
    first = int(np.flatnonzero(d.date.ge(pd.Timestamp(start)))[0])
    if first < 1:
        raise ValueError("缺少第一次开盘之前的决策日。")
    risk = dict(zip(risks.idx.astype(int), risks.es95))
    parent = parents.set_index("origin") if len(parents) else pd.DataFrame()
    parent_name = PARENT_B if policy == "POINT_B" else PARENT_A
    if policy != "NR7_ONLY":
        required = pd.DatetimeIndex(d.date.iloc[first - 1:-1])
        if parent_name not in parent or not required.isin(parent.index).all():
            raise ValueError("账户基线缺少事前父信号。")

    cash, q, receivable, previous_nav, peak = CAPITAL, 0, 0., CAPITAL, CAPITAL
    active, pending, stopped = None, None, False
    entitlements, outstanding, all_cycles = {}, {}, []
    events = dividends.to_dict("records")
    daily, orders, rejected, decisions = [], [], [], []

    def parent_at(i):
        if policy == "NR7_ONLY":
            return np.nan
        return float(parent.loc[d.date.iloc[i], parent_name])

    for i in range(first, len(d)):
        r = d.iloc[i]
        old_q, commission, slippage, accrued, paid = q, 0., 0., 0., 0.
        sold_today = False
        for k, event in enumerate(events):
            if event["ex_date"] == r.date and k in entitlements:
                eligible_q, owner = entitlements[k]
                amount = eligible_q * event["cash_dividend_per_share"]
                if amount:
                    outstanding[k] = amount
                    receivable += amount
                    accrued += amount
                    owner["dividend_cny"] += amount

        # 父信号归零与持仓风险要求都在之前收盘已知；受阻退出不撤销。
        prior_parent = parent_at(i - 1)
        if active is not None and active["source"] == "CORE" and pending is None and prior_parent == 0:
            pending = (0, "PARENT_TARGET_ZERO")
        if q and pending is not None:
            keep, reason = pending
            if i > active["entry_idx"] and not open_blocked(d, i, -1):
                sold = max(0, q - keep)
                if sold:
                    px = fill(float(r.open), -1, cost)
                    charge = fee(sold * px, cost)
                    cash += sold * px - charge
                    q -= sold
                    commission += charge
                    slippage += sold * (float(r.open) - px)
                    active["sell_net_cny"] += sold * px - charge
                    active["sell_fees"] += charge
                    sold_today = True
                    orders.append({"date": r.date, "origin": d.date.iloc[i - 1], "cycle_id": active["cycle_id"],
                                   "source": active["source"], "side": "SELL", "quantity": sold,
                                   "raw_open": r.open, "fill_price": px, "commission": charge,
                                   "slippage": sold * (float(r.open) - px), "reason": reason})
                    if q == 0:
                        active.update(exit_date=r.date, exit_idx=i, exit_reason=reason,
                                      holding_sessions=i - active["entry_idx"], status="COMPLETE")
                        active = None
                pending = None
            else:
                rejected.append({"date": r.date, "origin": d.date.iloc[i - 1], "source": active["source"],
                                 "reason": "SELL_DEFERRED_T1_OR_LIMIT"})

        signal = sig.iloc[i - 1]
        source = None
        if policy != "NR7_ONLY" and np.isfinite(prior_parent) and prior_parent > 0:
            source = "CORE"
        elif policy in {"NR7_ONLY", "POINT_A_PLUS_NR7"} and bool(signal.entry_event):
            # 组合只有在父信号明确为零时使用新机会；未知不解释为空仓。
            if policy == "NR7_ONLY" or prior_parent == 0:
                source = "NR7"
        decisions.append({"origin": d.date.iloc[i - 1], "execution_date": r.date,
                          "parent_target": prior_parent, "nr7_event": bool(signal.entry_event),
                          "source_request": source or "NONE", "position_before_entry": q})
        if source is not None:
            es = risk.get(i - 1, np.nan)
            reason = None
            if stopped:
                reason = "ACCOUNT_DRAWDOWN_STOPPED"
            elif q or sold_today:
                reason = "EXISTING_POSITION_OR_SAME_DAY_EXIT"
            elif not np.isfinite(es):
                reason = "MISSING_PRIOR_RISK_ESTIMATE"
            elif open_blocked(d, i, 1):
                reason = "OPEN_LIMIT_OR_MISSING_PRICE"
            elif source == "NR7" and float(r.ao) <= float(signal.stop_index):
                reason = "OPEN_ALREADY_BELOW_STRUCTURAL_STOP"
            else:
                nav_open = cash + receivable
                budget = risk_base.limits(nav_open, peak, es)
                reference = float(d.close.iloc[i - 1] - r.dividend)
                planned = risk_base.cap_quantity(reference, budget, 100000000, cash)
                filled = risk_base.cap_quantity(float(r.open), budget, planned, cash)
                if filled:
                    px = fill(float(r.open), 1, cost)
                    charge = fee(filled * px, cost)
                    debit = filled * px + charge
                    cash -= debit
                    q = filled
                    commission += charge
                    slip = q * (px - float(r.open))
                    slippage += slip
                    stop = float(signal.stop_index) if source == "NR7" else np.nan
                    target = float(r.ao) + 2 * (float(r.ao) - stop) if source == "NR7" else np.nan
                    active = {"cycle_id": len(all_cycles) + 1, "source": source, "entry_origin": d.date.iloc[i - 1],
                              "entry_date": r.date, "entry_idx": i, "entry_raw": float(r.open),
                              "entry_price": px, "entry_quantity": q, "entry_equity": nav_open,
                              "buy_debit": debit, "entry_fee": charge, "sell_net_cny": 0., "sell_fees": 0.,
                              "dividend_cny": 0., "stop_index": stop, "target_index": target,
                              "setup_date": signal.setup_date if source == "NR7" else pd.NaT,
                              "exit_date": pd.NaT, "status": "RIGHT_CENSORED"}
                    all_cycles.append(active)
                    orders.append({"date": r.date, "origin": d.date.iloc[i - 1], "cycle_id": active["cycle_id"],
                                   "source": source, "side": "BUY", "quantity": q, "raw_open": r.open,
                                   "fill_price": px, "commission": charge, "slippage": slip,
                                   "reason": "NEXT_OPEN", "known_es95": es, "planned_quantity": planned})
                else:
                    reason = "CASH_OR_RISK_BUDGET_BELOW_ONE_LOT"
            if reason:
                rejected.append({"date": r.date, "origin": d.date.iloc[i - 1], "source": source, "reason": reason})

        for k, event in enumerate(events):
            if event["record_date"] == r.date and q:
                entitlements[k] = (q, active)
            if event["payment_date"] <= r.date and k in outstanding:
                amount = outstanding.pop(k)
                receivable -= amount
                cash += amount
                paid += amount
        nav = cash + q * float(r.close) + receivable
        peak = max(peak, nav)
        drawdown = 1 - nav / peak
        if drawdown >= .1:
            stopped = True
        if q:
            reason = None
            if stopped:
                reason = "ACCOUNT_DRAWDOWN_STOP"
            elif active["source"] == "CORE" and parent_at(i) == 0:
                reason = "PARENT_TARGET_ZERO"
            elif active["source"] == "NR7":
                if r.ac <= active["stop_index"]:
                    reason = "CLOSE_STRUCTURAL_STOP"
                elif r.ac >= active["target_index"]:
                    reason = "CLOSE_2R_TARGET"
                elif i - active["entry_idx"] + 1 >= 10:
                    reason = "TIME_10_CLOSES"
            if reason:
                pending = (0, reason)
            elif pending is None:
                es = risk.get(i, np.nan)
                keep = risk_base.cap_quantity(float(r.close), risk_base.limits(nav, peak, es), q, existing=q)
                if keep < q:
                    pending = (keep, "PRIOR_CLOSE_RISK_REDUCTION")
        price_pnl = old_q * (float(r.open) - float(d.close.iloc[i - 1])) + q * (float(r.close) - float(r.open))
        error = nav - previous_nav - price_pnl - accrued + commission + slippage
        if abs(error) > 1e-6 or cash < -1e-6 or receivable < -1e-6 or q % 100:
            raise AssertionError("账户逐日财富、现金或整手恒等式失败。")
        daily.append({"date": r.date, "equity": nav, "cash": cash, "shares": q, "close": r.close,
                      "receivable": receivable, "net_return": nav / previous_nav - 1,
                      "price_pnl": price_pnl, "dividend_accrual": accrued, "dividend_paid": paid,
                      "commission": commission, "slippage": slippage, "exposure": q * r.close / nav,
                      "drawdown": drawdown, "accounting_error": error, "risk_stopped": stopped,
                      "source": active["source"] if active else "FLAT"})
        previous_nav = nav

    for trade in all_cycles:
        if trade["status"] == "COMPLETE":
            pnl = trade["sell_net_cny"] + trade["dividend_cny"] - trade["buy_debit"]
            trade["net_pnl"] = pnl
            trade["net_return"] = pnl / trade["buy_debit"]
        else:
            trade["net_pnl"], trade["net_return"] = np.nan, np.nan
    terminal = {"open_shares": q, "open_source": active["source"] if active else None,
                "pending_exit": pending, "stopped": stopped, "unpaid_dividend_cny": receivable,
                "estimated_future_exit_friction": q * (float(d.close.iloc[-1]) - fill(float(d.close.iloc[-1]), -1, cost))
                + fee(q * fill(float(d.close.iloc[-1]), -1, cost), cost) if q else 0.}
    columns = ["cycle_id", "source", "entry_origin", "entry_date", "exit_date", "status", "net_pnl", "net_return"]
    return {"daily": pd.DataFrame(daily), "trades": pd.DataFrame(all_cycles) if all_cycles else pd.DataFrame(columns=columns),
            "orders": pd.DataFrame(orders), "rejections": pd.DataFrame(rejected),
            "decisions": pd.DataFrame(decisions), "terminal": terminal}


def return_statistics(values):
    r = np.asarray(values, float)
    if not len(r) or not np.isfinite(r).all():
        raise ValueError("账户收益序列为空或存在缺失。")
    curve = np.cumprod(1 + r)
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.
    return {"days": len(r), "cumulative_return": float(curve[-1] - 1),
            "net_cagr": float(curve[-1] ** (252 / len(r)) - 1),
            "net_sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 1e-14 else np.nan,
            "annualized_volatility": sd * np.sqrt(252),
            "max_drawdown": float((1 - curve / np.maximum.accumulate(np.r_[1., curve])[1:]).max())}


def trade_statistics(trades):
    t = trades.loc[trades.status.eq("COMPLETE")]
    values = t.net_return.to_numpy(float)
    positive, negative = values[values > 0], values[values < 0]
    n = len(t)
    p, q = len(positive) / n if n else np.nan, len(negative) / n if n else np.nan
    b = positive.mean() / -negative.mean() if len(positive) and len(negative) else np.nan
    return {"completed_cycles": n, "wins": len(positive), "losses": len(negative),
            "win_rate": p, "payoff": b, "p_times_b": p * b,
            "mean_cycle_net_return": values.mean() if n else np.nan,
            "standard_expectancy_loss_units": p * b - q,
            "profit_factor": positive.sum() / -negative.sum() if len(negative) else np.nan,
            "completed_cycle_net_pnl": float(t.net_pnl.sum()) if n else 0.,
            "unfinished_cycles": int(trades.status.eq("RIGHT_CENSORED").sum())}


def metrics(result):
    d, t = result["daily"], result["trades"]
    complete = t.loc[t.status.eq("COMPLETE")]
    years = list(range(int(d.date.iloc[0].year), int(d.date.iloc[-1].year)))
    counts = [int(complete.exit_date.dt.year.eq(year).sum()) if len(complete) else 0 for year in years]
    return {**return_statistics(d.net_return), **trade_statistics(t),
            "ending_equity": float(d.equity.iloc[-1]), "total_commission": float(d.commission.sum()),
            "total_slippage": float(d.slippage.sum()), "gross_at_actual_quantities_pnl": float((d.price_pnl + d.dividend_accrual).sum()),
            "mean_exposure": float(d.exposure.mean()), "holding_days": int(d.shares.gt(0).sum()),
            "average_full_year_cycles": float(np.mean(counts)) if counts else np.nan,
            "zero_trade_full_years": sum(n == 0 for n in counts), "orders": len(result["orders"]),
            "account_risk_stopped": result["terminal"]["stopped"]}
