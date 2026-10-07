"""510300周线定位、日线确认：只用日周线的固定点位研究。"""
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_weekly_daily_entry_locations_v1"
SOURCES = {
    "prices.parquet": ROOT / "reports/research/510300_sequential_patterns_regime_v1/inputs/prices.parquet",
    "dividends.csv": ROOT / "data/reference/510300_dividends.csv",
    "dividend_coverage.json": ROOT / "reports/research/510300_sequential_patterns_regime_v1/inputs/dividend_coverage.json",
    "authority_snapshot.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
}
POLICIES = ["P0_CONFIRM", "P1_SPACE", "P2_WEEKLY", "P3_CONTEXT"]
NAMES = {"P0_CONFIRM": "周线支撑日线确认", "P1_SPACE": "再要求计划净空间风险比至少2", "P2_WEEKLY": "再要求周MACD回升", "P3_CONTEXT": "再加日MACD低波动相对放量"}
COSTS = {"BASE": (.0002, .0005), "STRESS": (.0004, .001)}
BUDGET, TICK = 100000., .001


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if x is pd.NaT or x is pd.NA:
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    return x


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_table(name, frame):
    directory = OUT / "results"
    directory.mkdir(exist_ok=True)
    frame.to_csv(directory / (name + ".csv"), index=False, encoding="utf-8-sig", float_format="%.13g")
    frame.to_parquet(directory / (name + ".parquet"), index=False)


def load():
    p = pd.read_parquet(OUT / "inputs/prices.parquet").sort_values("date").reset_index(drop=True)
    p["date"] = pd.to_datetime(p.date).dt.normalize()
    div = pd.read_csv(OUT / "inputs/dividends.csv")
    for k in ["record_date", "ex_date", "payment_date"]:
        div[k] = pd.to_datetime(div[k]).dt.normalize()
    return p, div


def features(prices, div):
    """分红只向前平移价格；所有周线背景仅在下一自然交易周开始使用。"""
    d = prices.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date)
    assert not d.date.duplicated().any() and d.date.is_monotonic_increasing
    assert (d[["open", "high", "low", "close"]] > 0).all().all()
    assert not d[["open", "high", "low", "close", "volume"]].isna().any().any()
    assert (d.high >= d[["open", "close", "low"]].max(axis=1)).all()
    assert (d.low <= d[["open", "close", "high"]].min(axis=1)).all()
    ex = div.groupby("ex_date").cash_dividend_per_share.sum()
    d["dividend"] = d.date.map(ex).fillna(0.)
    d["cash_shift"] = d.dividend.cumsum()
    for raw, adjusted in [("open", "ao"), ("high", "ah"), ("low", "al"), ("close", "ac")]:
        d[adjusted] = d[raw] + d.cash_shift
    d["return_log"] = np.log((d.close + d.dividend) / d.close.shift())
    tr = pd.concat([d.ah - d.al, (d.ah - d.ac.shift()).abs(), (d.al - d.ac.shift()).abs()], axis=1).max(axis=1)
    d["atr20"] = tr.rolling(20).mean()
    d["rv20"] = d.return_log.rolling(20).std(ddof=1)
    d["prior_rv_ratio"] = d.rv20.shift() / d.rv20.shift().rolling(252, min_periods=126).median()
    d["low_volatility"] = d.prior_rv_ratio.le(1.)
    dif = d.ac.ewm(span=12, adjust=False).mean() - d.ac.ewm(span=26, adjust=False).mean()
    d["daily_macd_hist"] = 2 * (dif - dif.ewm(span=9, adjust=False).mean())
    d["daily_macd_rising"] = d.daily_macd_hist.gt(d.daily_macd_hist.shift())
    d["relative_volume"] = d.volume / d.volume.shift().rolling(20).median()
    d["relative_volume_high"] = d.relative_volume.ge(1.)
    d["period"] = d.date.dt.to_period("W-FRI")
    w = d.groupby("period", sort=True).agg(first_date=("date", "first"), last_date=("date", "last"), open=("ao", "first"), high=("ah", "max"), low=("al", "min"), close=("ac", "last"), volume=("volume", "sum")).reset_index()
    w["weekly_ema20"] = w.close.ewm(span=20, adjust=False).mean()
    wd = w.close.ewm(span=12, adjust=False).mean() - w.close.ewm(span=26, adjust=False).mean()
    w["weekly_macd_hist"] = 2 * (wd - wd.ewm(span=9, adjust=False).mean())
    w["weekly_macd_rising"] = w.weekly_macd_hist.gt(w.weekly_macd_hist.shift())
    w["weekly_trend_up"] = w.close.gt(w.weekly_ema20) & w.weekly_ema20.gt(w.weekly_ema20.shift(4))
    # 枢轴的右侧两周用于确认；只从确认周结束后开始可用，绝不回填枢轴周。
    w["new_pivot_low"] = np.nan
    w["new_pivot_high"] = np.nan
    for j in range(4, len(w)):
        center = j - 2
        neighbors = [center - 2, center - 1, center + 1, center + 2]
        if w.low.iloc[center] < w.low.iloc[neighbors].min() - 1e-9:
            w.loc[j, "new_pivot_low"] = w.low.iloc[center]
        if w.high.iloc[center] > w.high.iloc[neighbors].max() + 1e-9:
            w.loc[j, "new_pivot_high"] = w.high.iloc[center]
    period_to_idx = dict(zip(w.period, w.index))
    support = None
    weekly_rows = []
    for j, row in w.iterrows():
        if pd.notna(row.new_pivot_low):
            support = {"support_index": float(row.new_pivot_low), "support_pivot_date": w.last_date.iloc[j - 2], "support_confirm_date": row.last_date, "support_pivot_week": j - 2}
        r = {"week_idx": j, "weekly_last_date": row.last_date, "weekly_macd_hist": row.weekly_macd_hist, "weekly_macd_rising": bool(row.weekly_macd_rising), "weekly_trend_up": bool(row.weekly_trend_up), "weekly_close": row.close, "weekly_ema20": row.weekly_ema20, "weekly_available": j >= 129}
        r.update(support or {"support_index": np.nan, "support_pivot_date": pd.NaT, "support_confirm_date": pd.NaT, "support_pivot_week": np.nan})
        weekly_rows.append(r)
    wt = pd.DataFrame(weekly_rows).set_index("week_idx")
    # 当前周末那根完整周线也延至下一周才启用，避免节假日周末识别依赖未来。
    d["completed_week_idx"] = d.period.map(period_to_idx) - 1
    d = d.join(wt, on="completed_week_idx")
    d["context_available"] = d.weekly_available.fillna(False).astype(bool) & d.prior_rv_ratio.notna() & d.relative_volume.notna()
    d["weekly_macd_rising"] = d.weekly_macd_rising.fillna(False).astype(bool)
    d["weekly_trend_up"] = d.weekly_trend_up.fillna(False).astype(bool)
    d["period"] = d.period.astype(str)
    w["period"] = w.period.astype(str)
    return d, w


def detect(d, w):
    """每个已确认周低点只启动一次支撑测试；失败和无空间均保存。"""
    episodes, signals, steps = [], [], []
    used = set()
    active = None
    for i in range(1, len(d)):
        r = d.iloc[i]
        if r.date < pd.Timestamp("2015-01-01"):
            continue
        if active is None:
            if pd.isna(r.support_index) or not r.context_available:
                continue
            anchor = pd.Timestamp(r.support_pivot_date).strftime("%Y%m%d")
            if anchor in used or r.completed_week_idx - r.support_pivot_week > 52:
                continue
            atr = float(d.atr20.iloc[i - 1])
            if not (d.ac.iloc[i - 1] > r.support_index and r.al <= r.support_index + .5 * atr):
                continue
            used.add(anchor)
            known = w.iloc[:int(r.completed_week_idx) + 1]
            highs = known[(known.index >= r.completed_week_idx - 52) & known.new_pivot_high.gt(r.ac)]
            target = float(highs.new_pivot_high.min()) if len(highs) else np.nan
            target_row = highs.loc[highs.new_pivot_high.idxmin()] if len(highs) else None
            active = {"event_id": "WLOW_" + anchor, "setup_idx": i, "setup_date": r.date, "support_pivot_date": r.support_pivot_date, "support_confirm_date": r.support_confirm_date, "support_index": float(r.support_index), "target_index": target, "target_known_date": target_row.last_date if target_row is not None else pd.NaT, "atr_setup": atr, "test_low": float(r.al), "state": "PENDING"}
        age = i - active["setup_idx"]
        active["test_low"] = min(active["test_low"], float(r.al))
        state = "OBSERVE"
        if r.ac < active["support_index"] - 2 * active["atr_setup"]:
            state = "FAILED_SUPPORT_BREAK"
        elif r.ac >= active["support_index"] and r.ac > d.ah.iloc[i - 1] and r.ac > r.ao:
            state = "CONFIRMED" if np.isfinite(active["target_index"]) else "CONFIRMED_NO_KNOWN_RESISTANCE"
        elif age >= 5:
            state = "EXPIRED_NO_CONFIRMATION"
        steps.append({"event_id": active["event_id"], "date": r.date, "step": state, "close_index": r.ac, "support_index": active["support_index"], "test_low_so_far": active["test_low"]})
        if state.startswith("CONFIRMED"):
            sig = {**active, "signal_idx": i, "signal_date": r.date, "signal_close": r.close, "signal_cash_shift": r.cash_shift, "signal_close_index": r.ac, "stop_index": active["test_low"] - .5 * active["atr_setup"], "weekly_last_date": r.weekly_last_date, "weekly_macd_hist": r.weekly_macd_hist, "weekly_macd_rising": bool(r.weekly_macd_rising), "weekly_trend_up": bool(r.weekly_trend_up), "daily_macd_hist": r.daily_macd_hist, "daily_macd_rising": bool(r.daily_macd_rising), "prior_rv_ratio": r.prior_rv_ratio, "low_volatility": bool(r.low_volatility), "relative_volume": r.relative_volume, "relative_volume_high": bool(r.relative_volume_high), "state": state}
            sig["stop_raw_at_signal"] = sig["stop_index"] - r.cash_shift
            sig["target_raw_at_signal"] = sig["target_index"] - r.cash_shift
            sig["max_entry_raw_at_signal"] = entry_ceiling(sig["stop_raw_at_signal"], sig["target_raw_at_signal"])
            signals.append(sig)
        if state != "OBSERVE":
            active.update(state=state, terminal_idx=i, terminal_date=r.date)
            episodes.append(active)
            active = None
    if active is not None:
        episodes.append(active)
    return pd.DataFrame(episodes), pd.DataFrame(signals), pd.DataFrame(steps)


def fill(price, side, cost):
    adjusted = price * (1 + side * COSTS[cost][1]) / TICK
    return (math.ceil(adjusted - 1e-9) if side > 0 else math.floor(adjusted + 1e-9)) * TICK


def fee(notional, cost):
    return max(5., notional * COSTS[cost][0]) if notional > 0 else 0.


def quantity(price, cost):
    n = int(BUDGET // (price * 100)) * 100
    while n and n * price + fee(n * price, cost) > BUDGET:
        n -= 100
    return n


def trade_math(entry, exit_price, dividend, cost):
    bp, sp = fill(entry, 1, cost), fill(exit_price, -1, cost)
    qty = quantity(bp, cost)
    fees = fee(qty * bp, cost) + fee(qty * sp, cost)
    pnl = qty * (sp - bp + dividend) - fees
    return {"quantity": qty, "buy_price": bp, "sell_price": sp, "commission": fees, "net_pnl": pnl, "net_return": pnl / BUDGET, "gross_return": (exit_price + dividend) / entry - 1, "friction_return": (qty * (bp - entry + exit_price - sp) + fees) / BUDGET}


def planned_rr(entry, stop, target):
    if not np.isfinite([entry, stop, target]).all() or not (0 < stop < entry < target):
        return np.nan
    loss = -trade_math(entry, stop, 0., "STRESS")["net_return"]
    reward = trade_math(entry, target, 0., "STRESS")["net_return"]
    return reward / loss if loss > 0 else np.nan


def entry_ceiling(stop, target):
    """在信号收盘时枚举价格刻度的单调边界，不读取次日价格。"""
    if not np.isfinite([stop, target]).all() or stop <= 0 or target <= stop:
        return np.nan
    lo, hi = math.floor(stop / TICK) + 1, math.ceil(target / TICK) - 1
    answer = None
    while lo <= hi:
        mid = (lo + hi) // 2
        rr = planned_rr(mid * TICK, stop, target)
        if np.isfinite(rr) and rr >= 2.:
            answer, lo = mid, mid + 1
        else:
            hi = mid - 1
    return answer * TICK if answer is not None else np.nan


def at_limit(d, i, direction):
    previous = float(d.close.iloc[i - 1]) - float(d.dividend.iloc[i])
    price = round(previous * (1 + .1 * direction), 3)
    return d.open.iloc[i] >= price - TICK / 2 if direction > 0 else d.open.iloc[i] <= price + TICK / 2


def label_one(sig, d, div):
    entry_idx = int(sig["signal_idx"]) + 1
    if entry_idx >= len(d):
        return {**sig, "entry_status": "NO_NEXT_DAILY_OPEN", "label_status": "NOT_MATURE"}, []
    r = d.iloc[entry_idx]
    entry = float(r.open)
    stop, target = sig["stop_index"] - r.cash_shift, sig["target_index"] - r.cash_shift
    common = {**sig, "entry_idx": entry_idx, "entry_date": r.date, "entry_reference": entry, "entry_stop_raw": stop, "entry_target_raw": target, "planned_net_rr_at_open": planned_rr(entry, stop, target)}
    if not np.isfinite(target) or target <= 0:
        return {**common, "entry_status": "NO_KNOWN_TARGET", "label_status": "NOT_ENTERED"}, []
    if not (stop < entry < target):
        return {**common, "entry_status": "OUTSIDE_STOP_TARGET_INTERVAL", "label_status": "NOT_ENTERED"}, []
    if at_limit(d, entry_idx, 1):
        return {**common, "entry_status": "UP_LIMIT_NO_MODEL_FILL", "label_status": "NOT_ENTERED"}, []
    maximum = sig["max_entry_raw_at_signal"] + sig["signal_cash_shift"] - r.cash_shift
    space_ok = bool(np.isfinite(maximum) and entry <= maximum + 1e-9 and common["planned_net_rr_at_open"] >= 2. - 1e-9)
    common.update(entry_status="OPEN_PROXY_AVAILABLE", space_ok=space_ok, max_entry_raw_at_open=maximum, weekly_ok=bool(sig["weekly_macd_rising"]), all_context_ok=bool(sig["weekly_macd_rising"] and sig["daily_macd_rising"] and sig["low_volatility"] and sig["relative_volume_high"]))
    pending, decision_idx, exit_idx, reason = False, None, None, None
    for j in range(entry_idx, len(d)):
        if pending and j > entry_idx:
            if not at_limit(d, j, -1):
                exit_idx = j
                break
        if not pending:
            close = float(d.ac.iloc[j])
            reason = "CLOSE_BELOW_STRUCTURE" if close <= sig["stop_index"] else "CLOSE_REACHES_TARGET" if close >= sig["target_index"] else "TIME_20_SESSIONS" if j - entry_idx + 1 >= 20 else None
            if reason is not None:
                pending, decision_idx = True, j
    if exit_idx is None:
        return {**common, "label_status": "NOT_MATURE", "exit_reason": reason, "decision_idx": decision_idx}, []
    exit_day, exit_price = d.date.iloc[exit_idx], float(d.open.iloc[exit_idx])
    dv = float(div.loc[div.record_date.ge(r.date) & div.record_date.lt(exit_day) & div.ex_date.le(exit_day), "cash_dividend_per_share"].sum())
    path = d.iloc[entry_idx:exit_idx]
    common.update(label_status="MATURE", decision_idx=decision_idx, decision_date=d.date.iloc[decision_idx], exit_idx=exit_idx, exit_date=exit_day, exit_reference=exit_price, exit_reason=reason, holding_sessions=exit_idx - entry_idx, dividend_per_share=dv, worst_wealth_excursion=(min(float(path.al.min()), float(d.ao.iloc[exit_idx])) - entry - r.cash_shift) / entry)
    rows = []
    for cost in COSTS:
        math_result = trade_math(entry, exit_price, dv, cost)
        risk_fraction = -trade_math(entry, stop, 0., cost)["net_return"]
        rows.append({**common, "cost": cost, "realized_multiple_of_planned_risk": math_result["net_return"] / risk_fraction if risk_fraction > 0 else np.nan, **math_result})
    return common, rows


def policies_from_labels(labels):
    chunks = []
    for p in POLICIES:
        mask = pd.Series(True, index=labels.index)
        if p != "P0_CONFIRM":
            mask &= labels.space_ok
        if p == "P2_WEEKLY":
            mask &= labels.weekly_ok
        if p == "P3_CONTEXT":
            mask &= labels.all_context_ok
        x = labels[mask].copy()
        x["policy"] = p
        chunks.append(x)
    return pd.concat(chunks, ignore_index=True)


def nonoverlap(frame):
    rows = []
    for (policy, cost), g in frame.groupby(["policy", "cost"]):
        busy_until = -1
        for r in g.sort_values(["entry_idx", "event_id"]).to_dict("records"):
            take = r["entry_idx"] > busy_until
            rows.append({**r, "sequential_eligible": take, "sequential_reason": "ONE_POSITION_AVAILABLE" if take else "OVERLAPPING_EXISTING_CYCLE"})
            if take:
                busy_until = r["exit_idx"]
    return pd.DataFrame(rows)


def wilson(wins, n):
    if n == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    center = (wins / n + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(wins / n * (1 - wins / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - radius, center + radius


def metrics(r):
    r = np.asarray(r, dtype=float)
    wins, losses = r[r > 0], r[r < 0]
    lo, hi = wilson(len(wins), len(r))
    return {"events": len(r), "wins": len(wins), "losses": len(losses), "win_rate": len(wins) / len(r) if len(r) else np.nan, "win_rate_lower95": lo, "win_rate_upper95": hi, "avg_win": wins.mean() if len(wins) else np.nan, "avg_loss": -losses.mean() if len(losses) else np.nan, "payoff_ratio": wins.mean() / -losses.mean() if len(wins) and len(losses) else np.nan, "profit_factor": wins.sum() / -losses.sum() if len(losses) else np.nan, "mean_net": r.mean() if len(r) else np.nan, "median_net": np.median(r) if len(r) else np.nan, "mean_without_best": (r.sum() - r.max()) / (len(r) - 1) if len(r) > 1 else np.nan}


def block_interval(g, calendar):
    if len(g) < 10:
        return {"net_mean_lower95": np.nan, "net_mean_upper95": np.nan, "payoff_lower95": np.nan, "payoff_upper95": np.nan}
    z = g.groupby("entry_date").agg(total=("net_return", "sum"), count=("net_return", "size")).reindex(calendar, fill_value=0.)
    rng = np.random.default_rng(20261001)
    starts = rng.integers(0, len(z), (2000, math.ceil(len(z) / 20)))
    idx = ((starts[:, :, None] + np.arange(20)) % len(z)).reshape(2000, -1)[:, :len(z)]
    n = z["count"].to_numpy()[idx].sum(axis=1)
    means = z.total.to_numpy()[idx].sum(axis=1)[n > 0] / n[n > 0]
    by_date = g.assign(win_sum=g.net_return.clip(lower=0), loss_sum=(-g.net_return).clip(lower=0), win_n=g.net_return.gt(0).astype(int), loss_n=g.net_return.lt(0).astype(int)).groupby("entry_date")[["win_sum", "loss_sum", "win_n", "loss_n"]].sum().reindex(calendar, fill_value=0.)
    a = by_date.to_numpy()[idx].sum(axis=1)
    ok = (a[:, 2] > 0) & (a[:, 3] > 0) & (a[:, 1] > 0)
    payoff = (a[ok, 0] / a[ok, 2]) / (a[ok, 1] / a[ok, 3])
    return {"net_mean_lower95": np.quantile(means, .025), "net_mean_upper95": np.quantile(means, .975), "payoff_lower95": np.quantile(payoff, .025) if len(payoff) else np.nan, "payoff_upper95": np.quantile(payoff, .975) if len(payoff) else np.nan}


def summarize(trades, calendar):
    rows, yearly, backgrounds = [], [], []
    for p in POLICIES:
        for cost in COSTS:
            for scope in ["ALL_EVENTS", "NONOVERLAPPING_CYCLES"]:
                g = trades[trades.policy.eq(p) & trades.cost.eq(cost)]
                if scope == "NONOVERLAPPING_CYCLES":
                    g = g[g.sequential_eligible]
                values = metrics(g.net_return)
                values.update(block_interval(g, calendar))
                values.update(policy=p, cost=cost, scope=scope, mean_gross=g.gross_return.mean(), mean_planned_net_rr=g.planned_net_rr_at_open.mean(), median_holding_sessions=g.holding_sessions.median())
                values["point_estimate_joint_screen"] = bool(values["win_rate"] >= .55 and values["payoff_ratio"] >= 2 and values["mean_net"] > 0)
                values["evidence_lower_bounds_positive"] = bool(values["net_mean_lower95"] > 0 and values["win_rate_lower95"] > .5 and values["payoff_lower95"] > 1)
                rows.append(values)
                for label, start, end in [("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2023", "2020-01-01", "2023-12-31"), ("2024-2026", "2024-01-01", "2026-12-31")]:
                    sub = g[g.entry_date.between(start, end)]
                    yearly.append({"policy": p, "cost": cost, "scope": scope, "period": label, **metrics(sub.net_return)})
    parent = trades[trades.policy.eq("P1_SPACE") & trades.cost.eq("STRESS") & trades.sequential_eligible]
    for feature in ["weekly_macd_rising", "weekly_trend_up", "daily_macd_rising", "low_volatility", "relative_volume_high"]:
        for flag in [False, True]:
            g = parent[parent[feature].eq(flag)]
            backgrounds.append({"feature": feature, "flag": flag, **metrics(g.net_return)})
    return pd.DataFrame(rows), pd.DataFrame(yearly), pd.DataFrame(backgrounds)


def protocol():
    return {"study": "510300_WEEKLY_DAILY_ENTRY_LOCATIONS_V1", "frozen_at": now(), "user_request": "不用分钟线，我们只用日线，周线观察，找到进场胜率盈亏比大的点位", "data_resolution": ["DAILY", "COMPLETED_WEEKLY"], "minute_data_reads": 0, "assets": ["510300.SH", "CASH_CNY"], "evaluation_start": "2015-01-01", "evidence": "既有已使用历史上的发现；不是独立样本", "price_adjustment": "原始OHLC加截至当日已除息的累计现金分红；只向前累计，非可成交复权价格；成交使用原始价。", "weekly_clock": "以W-FRI聚合；每个日线决策仅用上一自然交易周及更早周线，包括本周五也不提前使用当前周。周MACD至少130周预热。", "pivot": "周线严格低于或高于左右各2周；右侧2周结束后确认，下一周才用于日线。不得把后确认的低点回填到形成周。", "support": "最新已确认周低点，最多52周；每个锚点只取确认后首次由上方接近支撑0.5个前日ATR20以内的日线测试。", "resistance": "准备时已知最近52周确认高点中，高于准备日收盘的最低一个；无已知上方阻力则保存但不构造目标价。", "confirmation": "准备日或之后最多5日，收盘在支撑之上、超过前日最高且为阳线；先收盘低于支撑2个准备ATR则失败；未确认超时。", "invalidation": "准备至确认的已发生最低价减0.5个准备时ATR20，确认后固定。", "entry": "确认次日开盘代理，只在结构失效线与阻力目标之间，涨停开盘不假设买到；P1及其子集另限价于信号日已算出的计划压力净空间风险比至少2的最高价，除息时平移报价；不追当日后续回落。", "policies": NAMES, "primary_policy": "P1_SPACE", "context": {"weekly": "上一完整周MACD12/26/9柱值比再前一周回升", "daily": "确认日MACD12/26/9柱值较前日回升", "volatility": "前日RV20不高于过去252日RV20中位数，至少126日", "volume": "确认日成交量不低于之前20日成交量中位数", "P3": "P1加周MACD回升、日MACD回升、相对低波动、相对放量；不调整参数"}, "exit": "买入后仅用日线收盘判断：到目标、跌破结构线或持有满20交易日，下一可卖开盘退出；入场当天收盘失效最早也次日开盘卖。跌停开盘顺延。无日内止盈止损假设。", "reward_risk": "事前目标净盈利/结构位净亏损不是胜率估计，非保证止损；事后盈亏比为平均正净收益/平均负净收益绝对值。", "costs": {k: {"commission": v[0], "slippage": v[1]} for k, v in COSTS.items()}, "minimum_commission": 5, "lot": 100, "tick": TICK, "event_budget": BUDGET, "account_reference_capital": 200000, "joint_screen": "为解释用户所指较高，事先采用压力净胜率>=55%、实际盈亏比>=2、净均值>0的研究筛选线；是本轮操作定义，不是用户指定阈值或保证。全部失败与样本数均披露，不设年度次数。", "statistical_scope": "20交易日循环块2000次，种子20261001；不足10条不报均值/盈亏比区间。胜率Wilson区间为描述性，未校正全项目多次检验。", "overlap": "全部事件与按时间顺序单仓不重叠周期同时保存，后者同日退出不假设再次入场。未成熟不作零收益。", "periods": ["2015-2019", "2020-2023", "2024-2026"], "stop": "四项固定表达均完整报告；主方案未满足联合条件则不以最好子组代替，不改参数或止盈止损补救。通过也仅为待进一步账户检验的候选。", "existing_failed_studies": ["weekly_daily_technical_v1", "sequential_patterns_regime_v1", "510300_smc_sweep_fvg_historical_v1"], "independent_new_alpha_family": False, "new_collection": False, "orders_authorized": False}


def freeze():
    assert not (OUT / "freeze.json").exists(), "已有冻结记录，禁止覆盖。"
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        shutil.copy2(source, OUT / "inputs" / name)
    p, div = load()
    d, w = features(p, div)
    quality = {"status": "PASS_DAILY_OHLC_AND_WEEKLY_CLOCK", "daily_rows": len(d), "first": d.date.min(), "last": d.date.max(), "weekly_rows_including_last_partial": len(w), "weekly_latest_used_at_last_day": d.weekly_last_date.iloc[-1], "minute_rows_read": 0, "prior_source_corrections_preserved": int(p.correction_applied.sum()) if "correction_applied" in p else None}
    save_json(OUT / "input_quality.json", quality)
    save_json(OUT / "protocol.json", protocol())
    save_json(OUT / "freeze.json", {"frozen_at": now(), "returns_not_computed": True, "code_sha256": digest(__file__), "protocol_sha256": digest(OUT / "protocol.json"), "inputs": {name: {"source": str(source), "snapshot": str(OUT / "inputs" / name), "sha256": digest(OUT / "inputs" / name)} for name, source in SOURCES.items()}})
    print("日周线点位规则与输入快照已固定；未计算事件收益。", flush=True)


def frozen_check():
    f = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert f["code_sha256"] == digest(__file__), "冻结代码改变"
    assert f["protocol_sha256"] == digest(OUT / "protocol.json"), "冻结协议改变"
    for v in f["inputs"].values():
        assert v["sha256"] == digest(v["snapshot"]), "输入快照改变"


def run():
    frozen_check()
    assert not (OUT / "result.json").exists(), "已有结果，不覆盖或重复搜索。"
    p, div = load()
    d, w = features(p, div)
    episodes, signals, steps = detect(d, w)
    save_json(OUT / "run_started.json", {"started_at": now()})
    print(f"周线锚点与日线过程重建：{len(episodes)}个过程，{len(signals)}次日线确认。", flush=True)
    statuses, labels = [], []
    for sig in signals.to_dict("records"):
        status, rows = label_one(sig, d, div)
        statuses.append(status)
        labels.extend(rows)
    if not labels:
        save_table("全部准备过程", episodes)
        save_table("全部日线确认", signals)
        save_table("全部入场与成熟状态", pd.DataFrame(statuses))
        save_json(OUT / "result.json", {"status": "NO_MATURE_EVENT", "research_complete": True, "strategy_goal_achieved": False})
        return
    all_labels = pd.DataFrame(labels)
    trades = nonoverlap(policies_from_labels(all_labels))
    stats, periods, backgrounds = summarize(trades, pd.DatetimeIndex(d.loc[d.date.ge("2015-01-01"), "date"]))
    main = stats[stats.cost.eq("STRESS") & stats.scope.eq("NONOVERLAPPING_CYCLES")]
    primary = main[main.policy.eq("P1_SPACE")].iloc[0]
    for name, frame in {"日线及当时可用周背景": d, "周线与枢轴确认": w, "全部准备过程": episodes, "逐日观察与失败": steps, "全部日线确认": signals, "全部入场与成熟状态": pd.DataFrame(statuses), "原始事件两档成本": all_labels, "四组全部交易与重叠标记": trades, "胜率实际盈亏比分组统计": stats, "固定时段统计": periods, "背景逐项观察": backgrounds}.items():
        save_table(name, frame)
    result = {"study": "510300_WEEKLY_DAILY_ENTRY_LOCATIONS_V1", "completed_at": now(), "status": "CANDIDATE_POINT_ESTIMATES_ONLY" if primary.point_estimate_joint_screen else "REJECTED_FIXED_PRIMARY_JOINT_POINT_SCREEN", "primary": primary.to_dict(), "all_four_stress_nonoverlapping": main.to_dict("records"), "daily_data_rows": len(d), "last_data_date": d.date.max(), "episodes": len(episodes), "confirmed": len(signals), "states": episodes.state.value_counts().to_dict(), "entry_states": pd.DataFrame(statuses).entry_status.value_counts().to_dict(), "label_states": pd.DataFrame(statuses).label_status.value_counts().to_dict(), "minute_data_reads": 0, "strategy_goal_achieved": False, "full_account_metrics": "NOT_COMPUTED_EVENT_POINT_RESEARCH", "current_view": "NO_VIEW_STALE_LOCAL_DATA", "orders_authorized": False}
    save_json(OUT / "result.json", result)
    print(json.dumps(clean({"状态": result["status"], "四组压力不重叠事件": main.to_dict("records")}), ensure_ascii=False), flush=True)


def verify():
    frozen_check()
    p, div = load()
    d = pd.read_parquet(OUT / "results/日线及当时可用周背景.parquet")
    sig = pd.read_parquet(OUT / "results/全部日线确认.parquet")
    labels = pd.read_parquet(OUT / "results/原始事件两档成本.parquet")
    assert (sig.weekly_last_date < sig.signal_date.dt.to_period("W-FRI").dt.start_time).all()
    assert (sig.support_confirm_date < sig.signal_date).all()
    assert (labels.entry_date > labels.signal_date).all()
    assert (labels.exit_date > labels.entry_date).all()
    assert (labels.decision_date < labels.exit_date).all()
    error = 0.
    for r in labels.itertuples():
        expected = trade_math(r.entry_reference, r.exit_reference, r.dividend_per_share, r.cost)
        error = max(error, abs(expected["net_return"] - r.net_return))
    assert error < 1e-12
    checks = 0
    for pos in np.unique(np.linspace(0, len(sig) - 1, min(12, len(sig))).astype(int)):
        row = sig.iloc[pos]
        dd, ww = features(p.iloc[:int(row.signal_idx) + 1], div[div.ex_date.le(row.signal_date)])
        _, ss, _ = detect(dd, ww)
        one = ss[ss.event_id.eq(row.event_id)].iloc[0]
        for k in ["support_index", "target_index", "stop_index", "max_entry_raw_at_signal", "weekly_macd_hist", "daily_macd_hist", "prior_rv_ratio", "relative_volume"]:
            assert np.isclose(one[k], row[k], equal_nan=True), k
        assert one.signal_date == row.signal_date
        checks += 1
    trades = pd.read_parquet(OUT / "results/四组全部交易与重叠标记.parquet")
    for _, g in trades[trades.sequential_eligible].groupby(["policy", "cost"]):
        g = g.sort_values("entry_idx")
        assert (g.entry_idx.iloc[1:].to_numpy() > g.exit_idx.iloc[:-1].to_numpy()).all()
    receipt = {"status": "PASS_SAVED_DAILY_WEEKLY_POINT_RECOMPUTATION", "verified_at": now(), "return_rows": len(labels), "max_return_error": error, "real_history_prefix_checks": checks, "all_weekly_inputs_from_prior_completed_week": True, "all_exits_T_plus_1_or_later": True, "nonoverlap_checked": True, "minute_reads": 0, "scope": "计算和时序复核，不是独立盈利验证。"}
    save_json(OUT / "saved_verification.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="510300日周线支撑点位研究")
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
