"""510300 连续形态与状态启停 V1：固定日线协议及完整账户研究。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

STUDY = "510300_SEQUENTIAL_PATTERNS_REGIME_V1"
FAMILIES = ("BREAKOUT", "RECLAIM", "REPAIR")
POLICIES = ("PATTERN_ONLY", "STATE_ONLY", "RECENT_ONLY", "FULL")
COSTS = {"BASE": (0.0002, 0.0005), "STRESS": (0.0004, 0.0010)}
STATE_MAP = {"BREAKOUT": ("UP",), "RECLAIM": ("RANGE",), "REPAIR": ("DOWN", "RANGE")}


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    (rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.14g")


def features(prices: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """只向前累计已发生分红，不用未来复权因子定义价格结构。"""
    d = prices.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    assert not d.date.duplicated().any() and d.date.is_monotonic_increasing
    assert not d[["open", "high", "low", "close", "volume"]].isna().any().any()
    assert (d[["open", "high", "low", "close"]] > 0).all().all()
    assert (d.high >= d[["open", "low", "close"]].max(axis=1)).all()
    assert (d.low <= d[["open", "high", "close"]].min(axis=1)).all()
    ex = dividends.groupby("ex_date").cash_dividend_per_share.sum().to_dict()
    d["dividend"] = d.date.map(ex).fillna(0.0)
    prev = float(d.close.iloc[0])
    indexed = []
    for i, row in d.iterrows():
        scale = prev / float(d.close.iloc[i - 1]) if i else 1.0
        values = [(float(row[k]) + float(row.dividend)) * scale for k in ("open", "high", "low", "close")]
        indexed.append(values)
        prev = values[-1]
    for k, col in enumerate(("ao", "ah", "al", "ac")):
        d[col] = np.asarray(indexed)[:, k]
    d["r"] = np.log(d.ac).diff()
    d["rv5"] = d.r.rolling(5).std(ddof=1)
    d["rv20"] = d.r.rolling(20).std(ddof=1)
    tr = pd.concat([d.ah - d.al, (d.ah - d.ac.shift()).abs(), (d.al - d.ac.shift()).abs()], axis=1).max(axis=1)
    d["atr20"] = tr.rolling(20).mean()
    d["high10"] = d.ah.rolling(10).max()
    d["low10"] = d.al.rolling(10).min()
    d["high20"] = d.ah.rolling(20).max()
    d["low20"] = d.al.rolling(20).min()
    d["prior_low20"] = d.low20.shift()
    d["range10"] = d.high10 - d.low10
    d["range20"] = d.high20 - d.low20
    d["trend_z"] = np.log(d.ac / d.ac.shift(20)) / (d.rv20 * np.sqrt(20))
    d["state"] = np.select([d.trend_z > 1, d.trend_z < -1], ["UP", "DOWN"], default="RANGE")
    d.loc[d.trend_z.isna(), "state"] = "UNKNOWN"
    d["high_vol"] = d.rv20 > d.rv20.shift().rolling(252, min_periods=126).quantile(0.8)
    d["volume_ratio"] = d.volume / d.volume.shift().rolling(20).median()
    d["downside_per_volume"] = (-d.r).clip(lower=0) / d.volume_ratio.replace(0, np.nan)
    d["shock_z"] = np.log(d.ac / d.ac.shift(3)) / (d.rv20.shift(3) * np.sqrt(3))
    d["compression"] = (d.rv5 <= 0.65 * d.rv20) & (d.range10 <= 0.70 * d.range20)
    return d


def detect(d: pd.DataFrame):
    """逐日状态机；未完成和失败过程均保留，每类最多一个在途过程。"""
    active = {k: None for k in FAMILIES}
    cooldown = {k: -1 for k in FAMILIES}
    episodes, steps, signals = [], [], []
    for i in range(252, len(d)):
        r = d.iloc[i]
        for fam in FAMILIES:
            ep = active[fam]
            if ep is not None:
                age = i - ep["setup_idx"]
                prior_low_probed = ep["low_probed"]
                action, trigger, stop = "OBSERVE", False, np.nan
                if fam == "BREAKOUT":
                    if r.al < ep["lower"]:
                        ep["low_probed"] = True
                    if r.ac > ep["upper"] + 0.1 * ep["atr"]:
                        action, trigger, stop = "CONFIRMED_BREAKOUT", True, ep["upper"]
                    elif r.ac < ep["lower"] - 0.1 * ep["atr"]:
                        action = "FAILED_DOWN_BREAK"
                    elif r.ah >= ep["upper"]:
                        action = "PROBE_REJECTED"
                elif fam == "RECLAIM":
                    if r.ac > ep["upper"] + 0.1 * ep["atr"]:
                        action, trigger, stop = "CONFIRMED_RECLAIM", True, ep["lower"] - 0.1 * ep["atr"]
                    elif r.ac < ep["lower"] - ep["atr"]:
                        action = "FAILED_CONTINUED_DOWN"
                else:
                    ep["trough"] = min(ep["trough"], float(r.al))
                    if r.ac >= ep["trough"] + 0.5 * (ep["upper"] - ep["trough"]) and r.ac > d.ac.iloc[i - 1]:
                        action, trigger, stop = "CONFIRMED_REPAIR", True, ep["trough"] - 0.1 * ep["atr"]
                    elif r.ac < ep["lower"] - ep["atr"]:
                        action = "FAILED_CONTINUED_DOWN"
                if not trigger and not action.startswith("FAILED") and age >= ep["max_wait"]:
                    action = "EXPIRED_WITHOUT_CONFIRMATION"
                steps.append({"episode_id": ep["episode_id"], "family": fam, "date": r.date, "idx": i, "age": age,
                              "action": action, "state": r.state, "close_index": r.ac, "upper": ep["upper"],
                              "lower": ep["lower"], "volume_ratio": r.volume_ratio,
                              "downside_per_volume": r.downside_per_volume})
                if trigger:
                    route = "DIRECT"
                    if fam == "BREAKOUT":
                        route = "PRIOR_DAY_LOW_PROBED_BEFORE_BREAKOUT" if prior_low_probed else (
                            "HIGHER_LOW" if d.al.iloc[max(ep["setup_idx"], i - 2):i + 1].min() > ep["lower"] + 0.25 * ep["atr"] else "DIRECT")
                    sig = {"signal_id": ep["episode_id"], "episode_id": ep["episode_id"], "family": fam,
                           "signal_idx": i, "signal_date": r.date, "setup_idx": ep["setup_idx"],
                           "setup_date": ep["setup_date"], "state": r.state, "high_vol": bool(r.high_vol),
                           "trend_z": r.trend_z, "stop_index": stop, "entry_basis": action,
                           "path_class": route, "volume_ratio": r.volume_ratio,
                           "downside_per_volume": r.downside_per_volume,
                           "close_index": r.ac, "atr_index": ep["atr"]}
                    signals.append(sig)
                if trigger or action.startswith("FAILED") or action.startswith("EXPIRED"):
                    ep.update(status=action, terminal_idx=i, terminal_date=r.date)
                    episodes.append(ep.copy())
                    active[fam] = None
                    cooldown[fam] = i + 5
                continue
            if i <= cooldown[fam]:
                continue
            setup = (bool(r.compression) if fam == "BREAKOUT" else
                     (r.ac < r.prior_low20 - 0.1 * d.atr20.iloc[i - 1]) if fam == "RECLAIM" else
                     (r.shock_z <= -2.0))
            if not setup:
                continue
            upper = r.high10 if fam == "BREAKOUT" else r.prior_low20 if fam == "RECLAIM" else d.ac.iloc[i - 3]
            lower = r.low10 if fam == "BREAKOUT" else r.al
            ep = {"episode_id": f"{fam}_{r.date}", "family": fam, "setup_idx": i, "setup_date": r.date,
                  "setup_state": r.state, "upper": float(upper), "lower": float(lower), "atr": float(r.atr20),
                  "trough": float(lower), "low_probed": False, "max_wait": 5 if fam == "BREAKOUT" else 3,
                  "status": "PENDING", "terminal_idx": None, "terminal_date": None}
            active[fam] = ep
            steps.append({"episode_id": ep["episode_id"], "family": fam, "date": r.date, "idx": i, "age": 0,
                          "action": "SETUP", "state": r.state, "close_index": r.ac, "upper": upper, "lower": lower,
                          "volume_ratio": r.volume_ratio, "downside_per_volume": r.downside_per_volume})
    episodes.extend(ep.copy() for ep in active.values() if ep is not None)
    return pd.DataFrame(episodes), pd.DataFrame(steps), pd.DataFrame(signals)


def fill_price(raw: float, side: str, cost: str) -> float:
    slip = COSTS[cost][1]
    price = raw * (1 + slip if side == "BUY" else 1 - slip)
    return (math.ceil(price * 1000 - 1e-9) if side == "BUY" else math.floor(price * 1000 + 1e-9)) / 1000


def commission(notional: float, cost: str) -> float:
    return max(5.0, notional * COSTS[cost][0])


def buy_quantity(cash: float, price: float, cost: str) -> int:
    q = int(cash / price / 100) * 100
    while q > 0 and q * price + commission(q * price, cost) > cash + 1e-8:
        q -= 100
    return q


def tradable(d: pd.DataFrame, i: int, side: str) -> bool:
    r = d.iloc[i]
    if r.volume <= 0:
        return False
    if i == 0:
        return True
    reference = float(d.close.iloc[i - 1] - r.dividend)
    upper = math.floor(reference * 1.1 * 1000 + 0.5) / 1000
    lower = math.floor(reference * 0.9 * 1000 + 0.5) / 1000
    return bool(r.open < upper - 0.00049) if side == "BUY" else bool(r.open > lower + 0.00049)


def trade_outcome(d: pd.DataFrame, sig: dict, cost: str, fixed_horizon=False) -> dict:
    """独立事件标签；退出只能在信号后下一开盘，入场当天不能卖出。"""
    s, e = int(sig["signal_idx"]), int(sig["signal_idx"]) + 1
    base = {"signal_id": sig["signal_id"], "family": sig["family"], "state": sig["state"],
            "signal_idx": s, "signal_date": sig["signal_date"], "cost": cost,
            "entry_idx": e, "exit_idx": None, "label_status": "PENDING", "exit_reason": None,
            "net_return": None, "raw_gross_return": None, "holding_sessions": None}
    if e >= len(d):
        return base
    if not tradable(d, e, "BUY") or (not fixed_horizon and d.ao.iloc[e] <= sig["stop_index"]):
        base["label_status"] = "UNFILLED_ENTRY"
        return base
    entry_px = fill_price(float(d.open.iloc[e]), "BUY", cost)
    q = buy_quantity(200000.0, entry_px, cost)
    due, reason = None, None
    for j in range(e, len(d)):
        if due is not None and j >= due and j > e and tradable(d, j, "SELL"):
            exit_px = fill_price(float(d.open.iloc[j]), "SELL", cost)
            div = float(d.dividend.iloc[e + 1:j + 1].sum())
            pnl = q * (exit_px - entry_px + div) - commission(q * entry_px, cost) - commission(q * exit_px, cost)
            base.update(exit_idx=j, entry_date=d.date.iloc[e], exit_date=d.date.iloc[j], label_status="MATURE",
                        exit_reason=reason, entry_price=entry_px, exit_price=exit_px, quantity=q,
                        net_return=pnl / (q * entry_px + commission(q * entry_px, cost)),
                        raw_gross_return=(float(d.open.iloc[j]) + div) / float(d.open.iloc[e]) - 1,
                        holding_sessions=j - e, dividend_per_share=div)
            return base
        if due is None:
            invalid = not fixed_horizon and float(d.ac.iloc[j]) < float(sig["stop_index"])
            if invalid or j - e + 1 >= 5:
                due, reason = j + 1, "INVALIDATED" if invalid else "TIME_EXIT"
    return base


def build_labels(d, signals):
    return pd.DataFrame([trade_outcome(d, s, cost) for s in signals.to_dict("records") for cost in COSTS])


def control_returns(d):
    """全体普通日期的固定五日持有对照；形成标签后才供更晚时点调用。"""
    rows = []
    for i in range(252, len(d)):
        sig = {"signal_idx": i, "signal_id": f"CONTROL_{d.date.iloc[i]}", "family": "CONTROL",
               "state": d.state.iloc[i], "signal_date": d.date.iloc[i]}
        item = trade_outcome(d, sig, "STRESS", fixed_horizon=True)
        if item["label_status"] == "MATURE":
            rows.append(item)
    return pd.DataFrame(rows)


def nonoverlap(records: pd.DataFrame) -> pd.DataFrame:
    chosen, last_exit = [], -1
    for row in records.sort_values(["signal_idx", "signal_id"]).to_dict("records"):
        if int(row["entry_idx"]) > last_exit:
            chosen.append(row)
            last_exit = int(row["exit_idx"])
    return pd.DataFrame(chosen, columns=records.columns)


def decisions(d, signals, labels, controls):
    """近期资格用已完成事件，停用期间继续观察假设事件，避免永久无法再启用。"""
    mature = labels[(labels.cost == "STRESS") & (labels.label_status == "MATURE")].copy()
    annotated = []
    for row in mature.to_dict("records"):
        t = int(row["signal_idx"])
        pool = controls[(controls.exit_idx <= t) & (controls.signal_idx >= t - 504) & (controls.state == row["state"])]
        pool = nonoverlap(pool).tail(20)
        baseline = float(pool.net_return.mean()) if len(pool) >= 10 else np.nan
        row.update(control_count=len(pool), control_mean_at_signal=baseline,
                   excess_return=row["net_return"] - baseline)
        annotated.append(row)
    train = pd.DataFrame(annotated, columns=list(mature.columns) + ["control_count", "control_mean_at_signal", "excess_return"])
    rows = []
    for t in range(252, len(d)):
        state = d.state.iloc[t]
        for fam in FAMILIES:
            pool = train[(train.family == fam) & (train.exit_idx <= t) & (train.signal_idx >= t - 504)]
            pool = nonoverlap(pool).tail(12)
            n = len(pool)
            mean = float(pool.net_return.mean()) if n else np.nan
            se = float(pool.net_return.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            lower = mean - 1.2815515655446004 * se if n > 1 else np.nan
            excess = float(pool.excess_return.mean()) if n else np.nan
            state_pool = pool[pool.state == state] if n else pool
            sn = len(state_pool)
            sm = float(state_pool.net_return.mean()) if sn else np.nan
            sx = float(state_pool.excess_return.mean()) if sn else np.nan
            enough = n >= 6 and pool.excess_return.notna().all()
            recent = enough and lower > 0 and excess > 0
            fixed_state = state in STATE_MAP[fam]
            full = recent and fixed_state and sn >= 4 and sm > 0 and sx > 0
            reason = "ENABLED" if full else "STATE_INELIGIBLE" if not fixed_state else "INSUFFICIENT_RECENT_EVENTS" if not enough else "RECENT_EDGE_UNCONFIRMED" if not recent else "INSUFFICIENT_STATE_EVENTS" if sn < 4 else "STATE_EDGE_NONPOSITIVE"
            rows.append({"idx": t, "date": d.date.iloc[t], "family": fam, "state": state,
                         "n_train": n, "train_signal_ids": "|".join(pool.signal_id.tolist()),
                         "latest_mature_exit_idx": int(pool.exit_idx.max()) if n else None,
                         "mean_net": mean, "mean_excess": excess, "normal90_lower": lower,
                         "state_n": sn, "state_mean_net": sm, "state_mean_excess": sx,
                         "PATTERN_ONLY": True, "STATE_ONLY": fixed_state,
                         "RECENT_ONLY": bool(recent), "FULL": bool(full), "reason": reason})
    return pd.DataFrame(rows), train


def account(d, dividends, signals, decision, start, end, policy, cost):
    """现金、股票、应收股息分账，完整逐日净值含空仓及研究终点估值。"""
    cash, q, receivable = 200000.0, 0, 0.0
    active, exit_due = None, None
    peak, previous_equity = 200000.0, 200000.0
    ledger, trades, rejected, receivable_book = [], [], [], []
    ev = signals.groupby("signal_idx") if len(signals) else None
    dec = decision.set_index(["idx", "family"])
    by_ex = {date: group for date, group in dividends.groupby("ex_date")}
    cycle, pending_buy = 0, None
    for i in range(start, end + 1):
        r = d.iloc[i]
        dividend_accrual, dividend_paid, fee_today, traded_today = 0.0, 0.0, 0.0, 0.0
        if r.date in by_ex and q:
            for item in by_ex[r.date].itertuples():
                amount = q * float(item.cash_dividend_per_share)
                receivable_book.append({"payment_date": item.payment_date, "amount": amount})
                receivable += amount
                dividend_accrual += amount
                if active:
                    active["dividend_cny"] += amount
        unpaid = []
        for item in receivable_book:
            if item["payment_date"] <= r.date:
                cash += item["amount"]
                receivable -= item["amount"]
                dividend_paid += item["amount"]
            else:
                unpaid.append(item)
        receivable_book = unpaid
        if q and exit_due is not None and i >= exit_due and i > active["entry_idx"]:
            if tradable(d, i, "SELL"):
                px = fill_price(float(r.open), "SELL", cost)
                fee = commission(q * px, cost)
                cash += q * px - fee
                fee_today += fee
                traded_today += q * px
                active.update(exit_idx=i, exit_date=r.date, exit_price=px, exit_fee=fee, quantity=q,
                              net_pnl=q * (px - active["entry_price"]) + active["dividend_cny"] - active["entry_fee"] - fee,
                              holding_sessions=i - active["entry_idx"])
                trades.append(active.copy())
                q, active, exit_due = 0, None, None
            else:
                rejected.append({"date": r.date, "signal_id": active["signal_id"], "reason": "SELL_DEFERRED_LIMIT_OR_NO_VOLUME"})
        if q == 0 and pending_buy is not None:
            sig = pending_buy
            invalid_open = policy != "BUY_HOLD" and r.ao <= sig["stop_index"]
            if tradable(d, i, "BUY") and not invalid_open:
                px = fill_price(float(r.open), "BUY", cost)
                q = buy_quantity(cash, px, cost)
                if q:
                    fee = commission(q * px, cost)
                    cash -= q * px + fee
                    fee_today += fee
                    traded_today += q * px
                    cycle += 1
                    active = dict(sig, cycle=cycle, entry_idx=i, entry_date=r.date, entry_price=px,
                                  entry_fee=fee, dividend_cny=0.0, entry_equity=previous_equity)
                else:
                    rejected.append({"date": r.date, "signal_id": sig["signal_id"], "reason": "INSUFFICIENT_CASH"})
            else:
                rejected.append({"date": r.date, "signal_id": sig["signal_id"], "reason": "ENTRY_GAP_INVALIDATED" if invalid_open else "BUY_UNFILLED_LIMIT_OR_NO_VOLUME"})
            pending_buy = None
        if policy == "BUY_HOLD" and i == start and q == 0:
            if tradable(d, i, "BUY"):
                px = fill_price(float(r.open), "BUY", cost)
                q = buy_quantity(cash, px, cost)
                fee = commission(q * px, cost)
                cash -= q * px + fee
                fee_today += fee
                traded_today += q * px
                active = {"signal_id": "BUY_HOLD", "family": "BUY_HOLD", "entry_idx": i, "entry_date": r.date,
                          "entry_price": px, "entry_fee": fee, "dividend_cny": 0.0, "entry_equity": previous_equity}
        if q and policy not in ("BUY_HOLD", "CASH") and exit_due is None:
            invalid = r.ac < active["stop_index"]
            timed = i - active["entry_idx"] + 1 >= 5
            disabled = policy in ("RECENT_ONLY", "FULL") and not bool(dec.loc[(i, active["family"]), policy])
            if invalid or timed or disabled:
                exit_due = i + 1
                active["exit_reason"] = "INVALIDATED" if invalid else "GATE_DISABLED" if disabled else "TIME_EXIT"
        equity = cash + q * float(r.close) + receivable
        peak = max(peak, equity)
        ledger.append({"idx": i, "date": r.date, "cash_cny": cash, "shares": q, "close": r.close,
                       "receivable_cny": receivable, "dividend_accrual_cny": dividend_accrual,
                       "dividend_paid_cny": dividend_paid, "fees_cny": fee_today, "notional_cny": traded_today,
                       "equity_cny": equity, "daily_return": equity / previous_equity - 1,
                       "drawdown": equity / peak - 1, "exposure": q * float(r.close) / equity,
                       "active_signal_id": active["signal_id"] if active else "", "exit_pending": exit_due is not None})
        assert cash >= -1e-6 and q % 100 == 0 and receivable >= -1e-6
        previous_equity = equity
        if i < end and policy in POLICIES and ev is not None and i in ev.groups:
            candidates = signals.loc[ev.groups[i]].copy()
            candidates["priority"] = candidates.family.map({fam: j for j, fam in enumerate(FAMILIES)})
            for sig in candidates.sort_values("priority").to_dict("records"):
                qualifies = bool(dec.loc[(i, sig["family"]), policy])
                if not qualifies:
                    rejected.append({"date": r.date, "signal_id": sig["signal_id"], "reason": "GATE_DISABLED"})
                elif q or pending_buy is not None:
                    rejected.append({"date": r.date, "signal_id": sig["signal_id"], "reason": "CAPITAL_OCCUPIED_OR_PRIORITY"})
                else:
                    pending_buy = sig
    terminal = {"open_position": bool(q), "shares": q, "last_signal_id": active["signal_id"] if active else None,
                "pending_sell": exit_due is not None, "terminal_haircut_cny": 0.0,
                "valuation_policy": "终点按收盘标记，不伪造已知终点清仓；另计双边买卖中的卖出成本储备。"}
    if q:
        px = fill_price(float(d.close.iloc[end]), "SELL", cost)
        terminal["terminal_haircut_cny"] = q * (float(d.close.iloc[end]) - px) + commission(q * px, cost)
    return pd.DataFrame(ledger), pd.DataFrame(trades), pd.DataFrame(rejected), terminal


def metrics(ledger, trades, terminal):
    r = ledger.daily_return.to_numpy(float)
    end = float(ledger.equity_cny.iloc[-1]) - terminal["terminal_haircut_cny"]
    adjusted = r.copy()
    previous = float(ledger.equity_cny.iloc[-2]) if len(ledger) > 1 else 200000.0
    adjusted[-1] = end / previous - 1
    nav = np.r_[200000.0, 200000.0 * np.cumprod(1 + adjusted)]
    std = adjusted.std(ddof=1)
    pnl = trades.net_pnl.to_numpy() if len(trades) else np.array([])
    win, loss = pnl[pnl > 0], pnl[pnl < 0]
    sharpe = adjusted.mean() / std * np.sqrt(252) if std > 1e-14 else None
    cagr = (end / 200000.0) ** (252 / len(ledger)) - 1
    mdd = float(-(nav / np.maximum.accumulate(nav) - 1).min())
    return {"days": len(ledger), "start": ledger.date.iloc[0], "end": ledger.date.iloc[-1],
            "net_cagr": cagr, "net_sharpe": sharpe, "max_drawdown": mdd,
            "ending_equity_with_exit_reserve": end, "net_profit_cny": end - 200000,
            "completed_cycles": len(pnl), "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
            "payoff_ratio": float(win.mean() / -loss.mean()) if len(win) and len(loss) else None,
            "average_exposure": float(ledger.exposure.mean()), "commissions_cny": float(ledger.fees_cny.sum()),
            "numerical_target_pass": bool(cagr >= 0.10 and sharpe is not None and sharpe >= 1.2 and mdd <= 0.10),
            **terminal}


def reserved_returns(ledger, terminal):
    r = ledger.daily_return.to_numpy(float).copy()
    prev = float(ledger.equity_cny.iloc[-2]) if len(ledger) > 1 else 200000.0
    r[-1] = (float(ledger.equity_cny.iloc[-1]) - terminal["terminal_haircut_cny"]) / prev - 1
    return r


def paired_bootstrap(ledgers, terminals, sample_path: Path):
    """固定20日联合区块，保存一次抽样索引；只评价主要近期压力账户。"""
    matrix = np.stack([reserved_returns(ledgers[p], terminals[p]) for p in POLICIES], axis=1)
    n = len(matrix)
    rng = np.random.default_rng(20260924)
    starts = rng.integers(0, n, size=(2000, math.ceil(n / 20)))
    indices = ((starts[:, :, None] + np.arange(20)) % n).reshape(2000, -1)[:, :n]
    np.savez_compressed(sample_path, indices=indices)
    samples = matrix[indices]
    cagr = np.prod(1 + samples, axis=1) ** (252 / n) - 1
    sharpe = np.divide(samples.mean(axis=1) * np.sqrt(252), samples.std(axis=1, ddof=1), out=np.zeros((2000, 4)), where=samples.std(axis=1, ddof=1) > 1e-14)
    rows = []
    for j, p in enumerate(POLICIES[:-1]):
        for name, values in (("CAGR_DELTA", cagr[:, -1] - cagr[:, j]), ("SHARPE_DELTA", sharpe[:, -1] - sharpe[:, j])):
            rows.append({"comparison": f"FULL_MINUS_{p}", "measure": name, "lower95": np.quantile(values, .025),
                         "upper95": np.quantile(values, .975), "median": np.median(values),
                         "draws": 2000, "block_sessions": 20,
                         "limitation": "单标的重叠研究与历史选择未被此区间消除；区间针对已保存账户日收益，不重新执行组合。"})
    return pd.DataFrame(rows)


def load_inputs(root):
    p = pd.read_parquet(root / "inputs/prices.parquet")
    div = pd.read_csv(root / "inputs/dividends.csv", dtype={"record_date": str, "ex_date": str, "payment_date": str})
    return p, div


def execute(root: Path):
    if (root / "results/summary.json").exists():
        raise RuntimeError("本次固定版本已有结果，禁止覆盖或重复研究运行。")
    freeze = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for name, h in freeze["hashes"].items():
        if digest(root / name) != h:
            raise RuntimeError(f"冻结文件改变：{name}")
    prices, div = load_inputs(root)
    d = features(prices, div)
    ep, steps, signals = detect(d)
    labels = build_labels(d, signals)
    controls = control_returns(d)
    decision, train = decisions(d, signals, labels, controls)
    out = root / "results"
    out.mkdir(exist_ok=True)
    for name, frame in (("features", d), ("episodes", ep), ("steps", steps), ("signals", signals),
                        ("event_labels", labels), ("controls", controls), ("training_events", train), ("daily_decisions", decision)):
        frame.to_parquet(out / f"{name}.parquet", index=False)
        if name != "features":
            save_csv(out / f"{name}.csv", frame)
    ranges = {
        "EARLY_FAILURE_DIAGNOSTIC": (int(d.index[d.date >= "2015-01-05"][0]), int(d.index[d.date <= "2019-12-31"][-1])),
        "LONG_REPLAY_DIAGNOSTIC": (int(d.index[d.date >= "2020-01-02"][0]), len(d) - 1),
        "RECENT_504_PRIMARY": (len(d) - 504, len(d) - 1)}
    rows, annual, saved_primary, primary_terminals = [], [], {}, {}
    for period, (a, b) in ranges.items():
        for cost in COSTS:
            for policy in (*POLICIES, "BUY_HOLD", "CASH"):
                ledger, trades, rejected, terminal = account(d, div, signals, decision, a, b, policy, cost)
                folder = out / "accounts" / period / cost / policy
                folder.mkdir(parents=True, exist_ok=True)
                ledger.to_parquet(folder / "daily.parquet", index=False)
                save_csv(folder / "daily.csv", ledger)
                save_csv(folder / "trades.csv", trades if len(trades) else pd.DataFrame(columns=["signal_id", "entry_idx", "exit_idx", "entry_date", "exit_date", "net_pnl", "holding_sessions"]))
                save_csv(folder / "rejected.csv", rejected if len(rejected) else pd.DataFrame(columns=["date", "signal_id", "reason"]))
                m = metrics(ledger, trades, terminal)
                rows.append(dict(period=period, cost=cost, policy=policy, **m))
                save_json(folder / "metrics.json", m)
                for year in sorted(ledger.date.str[:4].unique()):
                    sub = ledger[ledger.date.str.startswith(year)]
                    t = trades[trades.entry_date.str.startswith(year)] if len(trades) else trades
                    annual.append({"period": period, "cost": cost, "policy": policy, "year": year,
                                   "days": len(sub), "return": float(np.prod(1 + sub.daily_return) - 1),
                                   "complete_cycles_by_entry_year": len(t), "partial_year": year in (ledger.date.iloc[0][:4], ledger.date.iloc[-1][:4])})
                if period == "RECENT_504_PRIMARY" and cost == "STRESS":
                    saved_primary[policy] = ledger
                    primary_terminals[policy] = terminal
    result = pd.DataFrame(rows)
    save_csv(out / "account_comparison.csv", result)
    save_csv(out / "annual.csv", annual)
    ci = paired_bootstrap(saved_primary, primary_terminals, out / "bootstrap_indices.npz")
    save_csv(out / "paired_bootstrap.csv", ci)
    attribution = []
    full_r = reserved_returns(saved_primary["FULL"], primary_terminals["FULL"])
    for reference in ("PATTERN_ONLY", "BUY_HOLD"):
        ref_r = reserved_returns(saved_primary[reference], primary_terminals[reference])
        ratios = {"EXPOSURE": saved_primary["FULL"].exposure.mean() / max(saved_primary[reference].exposure.mean(), 1e-14),
                  "VOLATILITY": full_r.std(ddof=1) / max(ref_r.std(ddof=1), 1e-14)}
        for kind, ratio in ratios.items():
            scale = min(1.0, float(ratio))
            attribution.append({"reference": reference, "matching": kind, "scale": scale,
                                "full_cagr": np.prod(1 + full_r) ** (252 / len(full_r)) - 1,
                                "scaled_reference_cagr": np.prod(1 + ref_r * scale) ** (252 / len(ref_r)) - 1,
                                "evidence_class": "事后同暴露或同波动理想归因，非可执行账户，未重算整手及最低佣金。"})
    save_csv(out / "exposure_attribution.csv", attribution)
    maturity = labels[labels.label_status == "MATURE"]
    event_stats = maturity.groupby(["family", "cost", "state"]).agg(events=("net_return", "size"),
                 mean_net=("net_return", "mean"), median_net=("net_return", "median"),
                 win_rate=("net_return", lambda x: (x > 0).mean()), worst_net=("net_return", "min"),
                 mean_holding=("holding_sessions", "mean")).reset_index()
    save_csv(out / "event_by_state.csv", event_stats)
    transitions = decision.sort_values(["family", "idx"]).copy()
    transitions["previous_enabled"] = transitions.groupby("family").FULL.shift().fillna(False)
    transitions = transitions[transitions.FULL != transitions.previous_enabled]
    save_csv(out / "gate_transitions.csv", transitions)
    primary = result[(result.period == "RECENT_504_PRIMARY") & (result.policy == "FULL")]
    summary = {"study_id": STUDY, "completed_at": now(), "data_start": d.date.iloc[0], "data_end": d.date.iloc[-1],
               "price_rows": len(d), "episode_count": len(ep), "signal_count": len(signals),
               "episodes_by_family_status": ep.groupby(["family", "status"]).size().rename("count").reset_index().to_dict("records"),
               "primary_start": d.date.iloc[-504], "primary_end": d.date.iloc[-1],
               "primary_full_accounts": primary.to_dict("records"), "account_count": len(rows),
               "new_fits": 0, "parameter_grids": 0, "new_market_downloads": 0,
               "strict_forward_observations": 0, "current_view": "NO_VIEW_STALE_LOCAL_MARKET_DATA",
               "current_market_date_not_established": now()[:10],
               "current_action": "ABSTAIN", "position_target": "UNSET", "cash_target_implied": False,
               "evidence_class": "PREQUENTIAL_RETROSPECTIVE_REPLAY_ON_PREVIOUSLY_REUSED_HISTORY",
               "numerical_target_pass": bool(primary.numerical_target_pass.all()),
               "validated_goal_achieved": False,
               "status": "FROZEN_NUMERICAL_FAILURE" if not primary.numerical_target_pass.all() else "NUMERICAL_PASS_UNVALIDATED_FORWARD_REQUIRED"}
    save_json(out / "summary.json", summary)
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


def verify(root: Path):
    """只读复核保存账户与标签时钟，不新建账户、不拟合、不抽样或下载。"""
    out = root / "results"
    checks, errors = 0, []
    freeze = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for name, h in freeze["hashes"].items():
        checks += 1
        if digest(root / name) != h:
            errors.append("冻结身份不一致：" + name)
    for path in sorted((out / "accounts").rglob("daily.parquet")):
        d = pd.read_parquet(path)
        checks += len(d) * 3
        if not np.allclose(d.cash_cny + d.shares * d.close + d.receivable_cny, d.equity_cny, rtol=0, atol=1e-6):
            errors.append("账户恒等式：" + str(path))
        expected = d.equity_cny.to_numpy() / np.r_[200000.0, d.equity_cny.to_numpy()[:-1]] - 1
        if not np.allclose(expected, d.daily_return, rtol=0, atol=1e-12):
            errors.append("逐日收益：" + str(path))
        if (d.cash_cny < -1e-6).any() or (d.shares % 100 != 0).any():
            errors.append("资金或整手约束：" + str(path))
        tr = pd.read_csv(path.parent / "trades.csv")
        if len(tr) and (tr.exit_idx <= tr.entry_idx).any():
            errors.append("T+1：" + str(path))
        saved = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
        actual = metrics(d, tr, {k: saved[k] for k in ("open_position", "shares", "last_signal_id", "pending_sell", "terminal_haircut_cny", "valuation_policy")})
        for k in ("net_cagr", "net_sharpe", "max_drawdown", "net_profit_cny", "completed_cycles", "average_exposure"):
            checks += 1
            a, b = actual[k], saved[k]
            if a is None and b is None:
                continue
            if a is None or b is None or abs(a - b) > 1e-8:
                errors.append("指标复算：" + str(path) + ":" + k)
    decision = pd.read_parquet(out / "daily_decisions.parquet")
    train = pd.read_parquet(out / "training_events.parquet").set_index("signal_id")
    for r in decision.itertuples():
        for key in r.train_signal_ids.split("|") if r.train_signal_ids else []:
            checks += 1
            if train.loc[key, "exit_idx"] > r.idx or train.loc[key, "signal_idx"] < r.idx - 504:
                errors.append("启停引用未来标签：" + key)
    ci = pd.read_csv(out / "paired_bootstrap.csv")
    indices = np.load(out / "bootstrap_indices.npz")["indices"]
    matrix = np.stack([reserved_returns(pd.read_parquet(out / "accounts/RECENT_504_PRIMARY/STRESS" / p / "daily.parquet"),
                                       json.loads((out / "accounts/RECENT_504_PRIMARY/STRESS" / p / "metrics.json").read_text(encoding="utf-8"))) for p in POLICIES], axis=1)
    samples = matrix[indices]
    cagr = np.prod(1 + samples, axis=1) ** (252 / matrix.shape[0]) - 1
    std = samples.std(axis=1, ddof=1)
    sharpes = np.divide(samples.mean(axis=1) * np.sqrt(252), std, out=np.zeros_like(std), where=std > 1e-14)
    for j, p in enumerate(POLICIES[:-1]):
        for name, values in (("CAGR_DELTA", cagr[:, -1] - cagr[:, j]), ("SHARPE_DELTA", sharpes[:, -1] - sharpes[:, j])):
            r = ci[(ci.comparison == f"FULL_MINUS_{p}") & (ci.measure == name)].iloc[0]
            checks += 2
            if not np.allclose([r.lower95, r.upper95], np.quantile(values, [.025, .975]), atol=1e-11):
                errors.append("保存区块抽样复算失败")
    receipt = {"checked_at": now(), "checks": checks, "errors": errors,
               "status": "PASS_SAVED_OUTPUT_RECOMPUTATION" if not errors else "FAIL",
               "new_accounts": 0, "new_fits": 0, "new_random_draws": 0, "downloads": 0}
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if errors:
        raise AssertionError(receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description="连续形态与启停研究，固定版本只运行一次。")
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        execute(args.root)
    else:
        verify(args.root)


if __name__ == "__main__":
    main()
