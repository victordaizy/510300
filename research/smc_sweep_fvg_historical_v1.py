"""510300 前日低点试探、收复和 FVG 的固定历史事件研究。"""
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


WORKSPACE = Path(__file__).resolve().parents[1]
STUDY = "510300_SMC_SWEEP_FVG_HISTORICAL_V1"
OUT = WORKSPACE / "reports/research/510300_smc_sweep_fvg_historical_v1"
FILES = {
    "minute": WORKSPACE / "data/raw/market/510300_1m_tushare_raw.parquet",
    "minute_metadata": WORKSPACE / "data/raw/market/510300_1m_tushare_raw.metadata.json",
    "daily": WORKSPACE / "data/raw/market/510300_daily_raw.parquet",
    "dividends": WORKSPACE / "data/reference/510300_dividends.csv",
    "mandate": WORKSPACE / "config/510300_existing_data_training_mandate_v1.json",
}
BASE_POLICIES = ["P0_RECLAIM", "P1_RANGE_BREAK", "P2_FVG", "P3_RETEST"]
POLICIES = BASE_POLICIES + ["P4_CONTEXT"]
NAMES = {"P0_RECLAIM": "破低收回", "P1_RANGE_BREAK": "再突破试探区间", "P2_FVG": "再要求已形成FVG", "P3_RETEST": "再等待FVG回踩收回", "P4_CONTEXT": "FVG叠加MACD回升低波动相对放量"}
COSTS = {"BASE": (0.0002, 0.0005), "STRESS": (0.0004, 0.0010)}
TICK = .001
ALLOCATION = 100000.
HORIZONS = ["M5", "M30", "T0_CLOSE", "T1_OPEN", "T1_CLOSE", "T2_CLOSE", "T5_CLOSE"]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if x is pd.NaT or x is pd.NA:
        return None
    return x


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.13g")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    m = pd.read_parquet(FILES["minute"]).copy()
    m["time"] = pd.to_datetime(m.trade_time)
    m = m.sort_values("time").reset_index(drop=True)
    m["date"] = m.time.dt.normalize()
    m["clock"] = m.time.dt.hour * 60 + m.time.dt.minute
    d = pd.read_parquet(FILES["daily"]).copy()
    d["date"] = pd.to_datetime(d.date).dt.normalize()
    d = d.sort_values("date").set_index("date")
    div = pd.read_csv(FILES["dividends"])
    for col in ["record_date", "ex_date", "payment_date"]:
        div[col] = pd.to_datetime(div[col])
    return m, d, div


def check_inputs(m, d):
    expected = {570, *range(571, 691), *range(781, 901)}
    assert m.time.is_monotonic_increasing and not m.time.duplicated().any()
    assert not d.index.duplicated().any()
    assert (m[["open", "high", "low", "close"]] > 0).all().all()
    assert not m[["open", "high", "low", "close", "vol", "amount"]].isna().any().any()
    assert (m.high >= m[["open", "close", "low"]].max(axis=1)).all()
    assert (m.low <= m[["open", "close", "high"]].min(axis=1)).all()
    assert all(set(g.clock) == expected for _, g in m.groupby("date"))
    a = m.groupby("date").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("vol", "sum"), amount=("amount", "sum"))
    joined = a.join(d, rsuffix="_daily")
    assert not joined.open_daily.isna().any()
    errors = {c: float((joined[c] - joined[c + "_daily"]).abs().max()) for c in ["open", "high", "low", "close"]}
    assert max(errors.values()) <= 1e-9
    ratios = {c: float(((joined[c] - joined[c + "_daily"]).abs() / joined[c + "_daily"]).max()) for c in ["volume", "amount"]}
    assert max(ratios.values()) < .001
    return {"status": "PASS_PRICE_AND_AGGREGATION", "minute_rows": len(m), "days": m.date.nunique(), "first": m.time.min(), "last": m.time.max(), "ohlc_max_error": errors, "volume_amount_max_relative_error": ratios, "timestamp_semantics": "提供方未承诺标签为分钟起点或终点，本研究保守延迟；这不是历史实时可得性认证。", "level2_available_in_this_input": False}


def aggregate(m):
    """按交易所午休分段，开盘集合竞价独立留在原始层。"""
    x = m[m.clock != 570].copy()
    x["session"] = np.where(x.clock <= 690, "AM", "PM")
    anchor = np.where(x.session.eq("AM"), 570, 780)
    x["slot"] = ((x.clock - anchor - 1) // 5).astype(int)
    b = x.groupby(["date", "session", "slot"], sort=True).agg(time=("time", "max"), open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("vol", "sum"), count=("time", "size")).reset_index()
    assert b["count"].eq(5).all()
    return b.sort_values("time").reset_index(drop=True)


def detect_day(b, reference, day_open, day):
    """每个交易日最多一个首破母事件；任何信号只依赖当时已完成柱。"""
    b = b.sort_values("time").reset_index(drop=True)
    event = {"event_id": day.strftime("%Y%m%d"), "date": day, "reference": reference, "day_open": day_open, "state": "NO_SWEEP"}
    if day_open < reference - 1e-8:
        event["state"] = "OPEN_BELOW_REFERENCE"
        return event, []
    hits = np.flatnonzero(b.low.to_numpy() <= reference - TICK + 1e-8)
    if not len(hits):
        return event, []
    s = int(hits[0])
    row = b.iloc[s]
    event.update(sweep_time=row.time, sweep_low=float(row.low), sweep_high=float(row.high), session=row.session, state="SWEEP_UNRECLAIMED")
    signal_rows = []
    end = int(np.flatnonzero(b.session.eq(row.session).to_numpy())[-1])
    reclaim_end = min(s + 6, end)
    rec = [j for j in range(s, reclaim_end + 1) if b.close.iloc[j] >= reference - 1e-8]
    if not rec:
        event["terminal_time"] = b.time.iloc[reclaim_end]
        return event, []
    r = rec[0]
    stop = float(b.low.iloc[s:r + 1].min())
    event.update(reclaim_time=b.time.iloc[r], structure_low=stop, state="RECLAIM_NO_RANGE_BREAK")

    def emit(policy, j, **extra):
        signal_rows.append({"event_id": event["event_id"], "date": day, "policy": policy, "signal_time": b.time.iloc[j], "available_at_conservative": b.time.iloc[j] + pd.Timedelta(minutes=1), "reference": reference, "structure_low": stop, **extra})

    emit("P0_RECLAIM", r)
    gaps = []
    for j in range(s + 2, r + 1):
        if b.session.iloc[j - 2] == row.session and b.low.iloc[j] >= b.high.iloc[j - 2] + TICK - 1e-8 and b.close.iloc[j - 1] > b.open.iloc[j - 1]:
            gaps.append((j, float(b.high.iloc[j - 2]), float(b.low.iloc[j])))
    confirm_end = min(r + 6, end)
    confirmed = None
    for j in range(r + 1, confirm_end + 1):
        if b.close.iloc[j] <= stop - TICK + 1e-8:
            event.update(state="STRUCTURE_FAILED_BEFORE_CONFIRM", terminal_time=b.time.iloc[j])
            break
        if j - 2 >= s and b.low.iloc[j] >= b.high.iloc[j - 2] + TICK - 1e-8 and b.close.iloc[j - 1] > b.open.iloc[j - 1]:
            gaps.append((j, float(b.high.iloc[j - 2]), float(b.low.iloc[j])))
        if b.close.iloc[j] >= event["sweep_high"] + TICK - 1e-8:
            confirmed = j
            break
    if confirmed is None:
        event.setdefault("terminal_time", b.time.iloc[confirm_end])
        return event, signal_rows
    c = confirmed
    # 确认前已经被收盘向下穿越的缺口不算活跃缺口。
    gaps = [(j, lo, hi) for j, lo, hi in gaps if b.close.iloc[j:c + 1].min() >= lo - 1e-8]
    event.update(confirmation_time=b.time.iloc[c], fvg_at_confirmation=bool(gaps), state="CONFIRMED_WITHOUT_FVG")
    emit("P1_RANGE_BREAK", c, fvg_at_confirmation=bool(gaps))
    if not gaps:
        return event, signal_rows
    g, lower, upper = gaps[-1]
    event.update(fvg_time=b.time.iloc[g], gap_lower=lower, gap_upper=upper, state="FVG_WITHOUT_RETEST")
    emit("P2_FVG", c, fvg_at_confirmation=True, gap_lower=lower, gap_upper=upper)
    retest_end = min(c + 6, end)
    for j in range(c + 1, retest_end + 1):
        if b.close.iloc[j] <= stop - TICK + 1e-8 or b.close.iloc[j] < lower - 1e-8:
            event.update(state="FVG_INVALIDATED_BEFORE_RETEST", terminal_time=b.time.iloc[j])
            break
        if b.low.iloc[j] <= upper + 1e-8 and b.close.iloc[j] >= upper - 1e-8:
            event.update(state="RETEST_CONFIRMED", retest_time=b.time.iloc[j])
            emit("P3_RETEST", j, gap_lower=lower, gap_upper=upper)
            break
    event.setdefault("terminal_time", b.time.iloc[retest_end])
    return event, signal_rows


def detect_all(bars, d, div):
    ex = div.groupby("ex_date").cash_dividend_per_share.sum().to_dict()
    events, signals = [], []
    for day, b in bars.groupby("date", sort=True):
        i = d.index.get_loc(day)
        if i == 0:
            continue
        # 只用前一交易日低点，并将已宣布的当日现金除息从参考价中扣除。
        ref = round(float(d.low.iloc[i - 1]) - float(ex.get(day, 0)), 6)
        ev, sig = detect_day(b, ref, float(d.open.iloc[i]), day)
        ev["reference_date"] = d.index[i - 1]
        ev["ex_adjustment"] = float(ex.get(day, 0))
        events.append(ev)
        signals.extend(sig)
    return pd.DataFrame(events), pd.DataFrame(signals)


def context_features(bars, d, div):
    """MACD只读已完成五分钟柱；日波动和同槽位成交量基准均来自过去。"""
    x = bars.copy()
    ex = div.groupby("ex_date").cash_dividend_per_share.sum().reindex(d.index, fill_value=0.)
    cumulative = ex.cumsum()
    adjusted = x.close + x.date.map(cumulative)
    fast = adjusted.ewm(span=12, adjust=False).mean()
    slow = adjusted.ewm(span=26, adjust=False).mean()
    dif = fast - slow
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)
    x["macd_hist_bps"] = hist / x.close * 10000
    x["macd_hist_change_bps"] = hist.diff() / x.close * 10000
    x["macd_rising"] = hist.gt(hist.shift())
    returns = np.log((d.close + ex) / d.close.shift())
    rv = returns.rolling(20).std(ddof=1) * np.sqrt(242)
    prior_rv = rv.shift()
    prior_reference = prior_rv.rolling(252, min_periods=126).median()
    x["rv20_annualized_prior"] = x.date.map(prior_rv)
    x["rv20_relative_prior"] = x.date.map(prior_rv / prior_reference)
    x["low_volatility"] = x.rv20_relative_prior.le(1.)
    volume_reference = x.groupby(["session", "slot"], sort=False).volume.transform(lambda s: s.shift().rolling(20, min_periods=20).median())
    x["relative_volume"] = x.volume / volume_reference.replace(0, np.nan)
    x["relative_volume_high"] = x.relative_volume.ge(1.)
    x["context_available"] = x.index.to_series().ge(260) & x[["macd_hist_change_bps", "rv20_relative_prior", "relative_volume"]].notna().all(axis=1)
    return x


def attach_context(signals, contextual_bars):
    fields = ["macd_hist_bps", "macd_hist_change_bps", "macd_rising", "rv20_annualized_prior", "rv20_relative_prior", "low_volatility", "relative_volume", "relative_volume_high", "context_available"]
    x = signals.merge(contextual_bars[["time"] + fields], left_on="signal_time", right_on="time", how="left", validate="many_to_one").drop(columns="time")
    chosen = x[x.policy.eq("P2_FVG") & x.context_available & x.macd_rising & x.low_volatility & x.relative_volume_high].copy()
    chosen["policy"] = "P4_CONTEXT"
    return pd.concat([x, chosen], ignore_index=True)


def fill_price(price, side, cost):
    slip = COSTS[cost][1]
    x = price * (1 + side * slip) / TICK
    return (math.ceil(x - 1e-9) if side > 0 else math.floor(x + 1e-9)) * TICK


def commission(notional, cost):
    return max(5., float(notional) * COSTS[cost][0]) if notional else 0.


def dividend_between(div, entry_day, exit_day):
    # 日内持有至登记日收盘才取得权益；退出日早盘出售不影响前一登记日已取得的权益。
    rows = div[(div.record_date >= entry_day) & (div.record_date < exit_day) & (div.ex_date <= exit_day)]
    return float(rows.cash_dividend_per_share.sum())


def trade_outcome(entry, exit_price, dividend, cost):
    bp, sp = fill_price(entry, 1, cost), fill_price(exit_price, -1, cost)
    qty = int(ALLOCATION // (bp * 100)) * 100
    while qty and qty * bp + commission(qty * bp, cost) > ALLOCATION:
        qty -= 100
    fee = commission(qty * bp, cost) + commission(qty * sp, cost)
    gross = (exit_price + dividend) / entry - 1
    pnl = qty * (sp - bp + dividend) - fee
    return {"gross_return": gross, "net_return": pnl / ALLOCATION, "net_pnl_cny": pnl, "quantity": qty, "buy_price": bp, "sell_price": sp, "commission_cny": fee, "slippage_rounding_cny": qty * (bp - entry + exit_price - sp)}


def entry_for_signal(sig, minute_index):
    t = pd.Timestamp(sig["signal_time"]) + pd.Timedelta(minutes=3)
    # 14:57之后包含收盘集合竞价；任何跨午休或跨日的价格均不能代替本次入场。
    clock = t.hour * 60 + t.minute
    st = pd.Timestamp(sig["signal_time"])
    if t.normalize() != st.normalize() or clock >= 897 or (st.hour < 12 and clock > 690):
        return None, "NO_CONTINUOUS_ENTRY_WINDOW"
    if t not in minute_index.index:
        return None, "NO_MINUTE_AT_DELAYED_ENTRY"
    row = minute_index.loc[t]
    if float(row.open) <= float(sig["structure_low"]) + 1e-8:
        return None, "STRUCTURE_ALREADY_BROKEN_AT_ENTRY"
    if float(row.vol) <= 0:
        return None, "NO_RECORDED_TRADES_IN_ENTRY_BAR"
    return row, "MODEL_FILL_PROXY_NOT_ACTUAL_EXECUTION"


def labels(signals, m, d, div):
    mi = m.set_index("time")
    rows, execution = [], []
    for sig in signals.to_dict("records"):
        entry_row, reason = entry_for_signal(sig, mi)
        base = {**sig, "entry_status": reason}
        if entry_row is None:
            execution.append(base)
            continue
        et = pd.Timestamp(sig["signal_time"]) + pd.Timedelta(minutes=3)
        ep = float(entry_row.open)
        day = et.normalize()
        i = d.index.get_loc(day)
        prior_close = float(d.close.iloc[i - 1]) - float(div.loc[div.ex_date.eq(day), "cash_dividend_per_share"].sum())
        if ep >= round(prior_close * 1.1, 3) - TICK / 2:
            execution.append({**base, "entry_status": "AT_UPPER_LIMIT_NO_FILL_ASSUMPTION"})
            continue
        execution.append({**base, "entry_time": et, "entry_reference_price": ep})
        for h in HORIZONS:
            xp, xt, legal, status = None, None, h not in ["M5", "M30", "T0_CLOSE"], "AVAILABLE"
            if h in ["M5", "M30"]:
                xt = et + pd.Timedelta(minutes=int(h[1:]))
                if xt in mi.index and xt.normalize() == day and (et.hour < 12) == (xt.hour < 12) and xt.hour * 60 + xt.minute < 897:
                    xp = float(mi.loc[xt, "open"])
            elif h == "T0_CLOSE":
                xt, xp = day + pd.Timedelta(hours=15), float(d.loc[day, "close"])
            else:
                shift = int(h[1:h.index("_")])
                if i + shift < len(d):
                    ed = d.index[i + shift]
                    field = "open" if h.endswith("OPEN") else "close"
                    xt = ed + (pd.Timedelta(hours=9, minutes=30) if field == "open" else pd.Timedelta(hours=15))
                    xp = float(d.iloc[i + shift][field])
                    if field == "open":
                        prev = float(d.close.iloc[i + shift - 1]) - float(div.loc[div.ex_date.eq(ed), "cash_dividend_per_share"].sum())
                        if xp <= round(prev * .9, 3) + TICK / 2:
                            status = "NEXT_OPEN_AT_DOWN_LIMIT_NOT_ASSUMED_FILLED"
                            xp = None
            if xp is None:
                rows.append({**base, "entry_time": et, "entry_reference_price": ep, "horizon": h, "legal_for_new_shares": legal, "label_status": status if status != "AVAILABLE" else "NO_VALID_ENDPOINT", "cost": "STRESS"})
                continue
            dv = dividend_between(div, day, xt.normalize()) if legal else 0.
            path = mi.loc[et:xt]
            # 开盘退出只观察该开盘价，不能把退出分钟随后高低价写入持有路径。
            if h in ["M5", "M30", "T1_OPEN"]:
                path = path[path.index < xt]
            path_complete = bool(xt <= mi.index.max())
            path_low = min(float(path.low.min()), xp) if len(path) else xp
            path_high = max(float(path.high.max()), xp) if len(path) else xp
            cross_ex = bool(div.ex_date.gt(day).mul(div.ex_date.le(xt.normalize())).any())
            # MAE为原始价格路径描述，跨除息时另标记，不能用它替代账户损失。
            for cost in COSTS:
                rows.append({**base, "entry_time": et, "entry_reference_price": ep, "exit_time": xt, "exit_reference_price": xp, "dividend_per_share": dv, "horizon": h, "legal_for_new_shares": legal, "label_status": status, "cost": cost, "raw_path_complete": path_complete, "raw_path_crosses_ex_date": cross_ex, "raw_price_mae": path_low / ep - 1 if path_complete else None, "raw_price_mfe": path_high / ep - 1 if path_complete else None, "structure_failed_before_next_open": bool((path.close < float(sig["structure_low"])).any() or xp < float(sig["structure_low"])) if h == "T1_OPEN" and path_complete else None, **trade_outcome(ep, xp, dv, cost)})
    return pd.DataFrame(rows), pd.DataFrame(execution)


def block_interval(frame, value, calendar, seed=20261001, iterations=2000):
    """在完整日历上按五个连续交易日抽块，不把稀疏事件当独立分钟样本。"""
    z = frame.groupby("date")[value].agg(["sum", "count"]).reindex(calendar, fill_value=0)
    if int(z["count"].sum()) < 10:
        return {"lower95": None, "upper95": None, "bootstrap_iterations": 0}
    a = z["sum"].to_numpy(float)
    n = z["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(z), size=(iterations, math.ceil(len(z) / 5)))
    ix = ((starts[:, :, None] + np.arange(5)) % len(z)).reshape(iterations, -1)[:, :len(z)]
    denominator = n[ix].sum(axis=1)
    good = denominator > 0
    values = a[ix].sum(axis=1)[good] / denominator[good]
    return {"lower95": float(np.quantile(values, .025)), "upper95": float(np.quantile(values, .975)), "bootstrap_iterations": int(good.sum())}


def summarize(lab, calendar):
    out, yearly = [], []
    valid = lab[lab.label_status.eq("AVAILABLE")].copy()
    for (policy, horizon, cost), g in valid.groupby(["policy", "horizon", "cost"], sort=True):
        r = g.net_return.to_numpy(float)
        win, lose = r[r > 0], r[r < 0]
        early, late = g[g.date.lt("2024-01-01")], g[g.date.ge("2024-01-01")]
        out.append({"policy": policy, "horizon": horizon, "cost": cost, "events": len(g), "mean_gross": g.gross_return.mean(), "mean_net": r.mean(), "median_net": np.median(r), "positive_fraction": (r > 0).mean(), "payoff_ratio": win.mean() / -lose.mean() if len(win) and len(lose) else None, "mean_without_best": (r.sum() - r.max()) / (len(r) - 1) if len(r) > 1 else None, "early_count": len(early), "late_count": len(late), "early_mean_net": early.net_return.mean(), "late_mean_net": late.net_return.mean(), "legal_for_new_shares": bool(g.legal_for_new_shares.iloc[0]), **block_interval(g, "net_return", calendar)})
        for year, part in g.groupby(g.date.dt.year):
            yearly.append({"policy": policy, "horizon": horizon, "cost": cost, "year": int(year), "events": len(part), "mean_gross": part.gross_return.mean(), "mean_net": part.net_return.mean(), "positive_fraction": part.net_return.gt(0).mean()})
    return pd.DataFrame(out), pd.DataFrame(yearly)


def paired_filters(lab, signals, execution, calendar):
    records, summaries = [], []
    available = lab[lab.label_status.eq("AVAILABLE") & lab.horizon.eq("T1_OPEN") & lab.cost.eq("STRESS")]
    for parent, child in list(zip(BASE_POLICIES, BASE_POLICIES[1:])) + [("P2_FVG", "P4_CONTEXT")]:
        base = available[available.policy.eq(parent)].set_index("event_id")
        offspring = available[available.policy.eq(child)].set_index("event_id")
        emitted = set(signals.loc[signals.policy.eq(child), "event_id"])
        unfilled = set(execution.loc[execution.policy.eq(child) & execution.entry_status.ne("MODEL_FILL_PROXY_NOT_ACTUAL_EXECUTION"), "event_id"])
        pairs = []
        for ident, r in base.iterrows():
            if child == "P4_CONTEXT" and not bool(r.context_available):
                continue
            if ident in offspring.index:
                nxt = offspring.loc[ident]
                cr, why = float(nxt.net_return), "CHILD_MODEL_FILL"
            elif ident not in emitted or ident in unfilled:
                cr, why = 0., "FROZEN_RULE_NO_POSITION"
            else:
                # 未来端点缺失不是未交易；不能填成零。
                continue
            row = {"comparison": child + "_MINUS_" + parent, "event_id": ident, "date": r.date, "parent_return": float(r.net_return), "child_contribution": cr, "difference": cr - float(r.net_return), "child_state": why}
            pairs.append(row)
            records.append(row)
        f = pd.DataFrame(pairs)
        if f.empty:
            continue
        ci = block_interval(f, "difference", calendar)
        summaries.append({"comparison": child + "_MINUS_" + parent, "parent_opportunities": len(f), "child_fills": int(f.child_state.eq("CHILD_MODEL_FILL").sum()), "parent_mean": f.parent_return.mean(), "child_mean_contribution": f.child_contribution.mean(), "mean_difference": f.difference.mean(), "interpretation": "每个父级可入场机会的等额收益贡献；不交易为零，不代表等风险因果效应或完整账户。", **ci})
    return pd.DataFrame(records), pd.DataFrame(summaries)


def primary_gate(summary, pairs):
    row = summary[summary.policy.eq("P4_CONTEXT") & summary.horizon.eq("T1_OPEN") & summary.cost.eq("STRESS")]
    inc = pairs[pairs.comparison.eq("P4_CONTEXT_MINUS_P2_FVG")] if len(pairs) else pd.DataFrame()
    if row.empty or inc.empty:
        return {"pass": False, "reason": "NO_PRIMARY_ESTIMATE", "strategy_goal_achieved": False}
    r, q = row.iloc[0], inc.iloc[0]
    tests = {
        "primary_net_mean_positive": bool(r.mean_net > 0),
        "primary_net_mean_lower95_positive": bool(pd.notna(r.lower95) and r.lower95 > 0),
        "filter_increment_lower95_positive": bool(pd.notna(q.lower95) and q.lower95 > 0),
        "both_fixed_halves_net_positive": bool(r.early_mean_net > 0 and r.late_mean_net > 0),
    }
    return {"pass": all(tests.values()), "checks": tests, "primary_policy": "P4_CONTEXT", "primary_horizon": "T1_OPEN", "primary_cost": "STRESS", "strategy_goal_achieved": False, "scientific_scope": "已使用历史上的条件机制筛查，不是独立样本验证，也未校正全项目历次试验。"}


def context_analysis(lab, calendar):
    """完整展示八个组合，不以最好单元格另造主策略。"""
    sample = lab[lab.label_status.eq("AVAILABLE") & lab.horizon.eq("T1_OPEN") & lab.cost.eq("STRESS") & lab.context_available]
    cells, marginal = [], []
    for policy in ["P0_RECLAIM", "P1_RANGE_BREAK", "P2_FVG"]:
        parent = sample[sample.policy.eq(policy)]
        for macd in [False, True]:
            for lowvol in [False, True]:
                for volume in [False, True]:
                    g = parent[parent.macd_rising.eq(macd) & parent.low_volatility.eq(lowvol) & parent.relative_volume_high.eq(volume)]
                    cells.append({"policy": policy, "macd_rising": macd, "low_volatility": lowvol, "relative_volume_high": volume, "events": len(g), "mean_gross": g.gross_return.mean(), "mean_net": g.net_return.mean(), "positive_fraction": g.net_return.gt(0).mean() if len(g) else None, "early_events": int(g.date.lt("2024-01-01").sum()), "late_events": int(g.date.ge("2024-01-01").sum()), "early_mean_net": g.loc[g.date.lt("2024-01-01"), "net_return"].mean(), "late_mean_net": g.loc[g.date.ge("2024-01-01"), "net_return"].mean(), **block_interval(g, "net_return", calendar)})
        for feature in ["macd_rising", "low_volatility", "relative_volume_high"]:
            yes, no = parent[parent[feature]], parent[~parent[feature]]
            contribution = parent[["date", "net_return", feature]].copy()
            contribution["difference"] = np.where(contribution[feature], 0., -contribution.net_return)
            marginal.append({"policy": policy, "feature": feature, "parent_events": len(parent), "kept_events": len(yes), "excluded_events": len(no), "kept_mean_net": yes.net_return.mean(), "excluded_mean_net": no.net_return.mean(), "difference_in_group_means": yes.net_return.mean() - no.net_return.mean(), "filter_contribution_per_parent": contribution.difference.mean(), **block_interval(contribution, "difference", calendar)})
    return pd.DataFrame(cells), pd.DataFrame(marginal)


def protocol():
    return {
        "study_id": STUDY,
        "created_at": now(),
        "user_request": "请继续，直到出研究成果；可以叠加MACD波动率量一起分析",
        "research_question": "事前前日低点被日内跌破后收回、再突破首次试探柱高点时，已形成FVG与随后回踩是否增加T+1可卖时点的净收益？",
        "mechanism_status": "待检验的价格过程假设；没有逐笔订单流，不能声称识别吸收、机构、止损单或操纵。",
        "relation_to_legacy": "属于既有价格形态和FVG相关家族的新条件检验。旧15分钟无条件FVG、日线RECLAIM及锚定网格结论保留；不称独立新家族，不用其结果挑新参数。",
        "bar_definition": "原一分钟09:31至11:30、13:01至15:00分别聚合连续五条为5分钟柱；不跨午休。09:30集合竞价不参与三柱形态。",
        "bar_choice_reason": "五分钟用于描述日内位置测试、收复和再次确认的顺序；本轮只检验这一分辨率，没有分钟窗口网格。",
        "reference": "前一交易日完整日线最低价减当日现金除息；开盘已低于该参考价另记OPEN_BELOW_REFERENCE，本轮不作为从上方跌破事件。",
        "mother_event": "全样本每交易日首个5分钟最低价低于参考价至少0.001元；每日最多一次，不因失败再启动。所有交易日及无事件状态保留。",
        "reclaim": "从首次跌破柱起至其后第6根柱，首次收盘不低于参考价；同一上午或下午内，最多等待30分钟。",
        "structure_low": "首次试探至收回确认时已经发生的最低价，之后固定。",
        "range_break": "收回后的后续第1至第6根柱，首次收盘超过首次试探柱最高价至少0.001；先发生收盘跌破固定结构低点至少0.001则失败。",
        "fvg": "三根柱同一时段且第一根不早于首次试探，第三根low>=第一根high+0.001，中间柱close>open；须在区间突破确认时已经形成，形成后没有收盘低于缺口下沿。",
        "retest": "只使用区间突破时最近一个已形成且活跃的FVG；之后第1至第6根柱low触及上沿、close回到上沿以上；先收盘跌破下沿或结构低点则作废。",
        "policies": NAMES,
        "context": {"macd": "五分钟12/26/9 EMA，柱值2*(DIF-DEA)，以已发生现金分红向前累计的价格计算；柱值大于上一根定义动能回升，不要求柱值已为正；至少260根预热。", "volatility": "此前20个完整交易日含息对数收益标准差乘sqrt(242)；不高于此前252个可得RV20中位数为相对低波动，基准至少126日。", "volume": "当前已完成五分钟柱成交量/此前20个交易日同上午或下午同槽位成交量中位数；>=1为相对放量。不用当日最终量作为输入。", "fixed_joint": "P4=P2且MACD回升且相对低波动且相对放量；三个背景全可用才可评价。", "joint_hypothesis": "在较稳定的日波动背景中，日内卖压试探后的动能回升和较充分成交可能支持反转延续；这只是事前假设，不能解释为观察到了订单流。", "all_cells": "对P0/P1/P2完整列出2*2*2八格和三项单独条件；仅描述，无事后择优晋升。", "missing_context": "保留原形态事件，背景记NO_VIEW，不把背景未知当成已确认不满足条件或零收益。"},
        "hypothesis_count": {"primary": 1, "nested_controls": 4, "context_cells_per_parent": 8, "context_parent_sets": 3, "single_context_comparisons": 9, "reported_horizons": len(HORIZONS), "cost_scenarios": len(COSTS)},
        "entry_clock": "信号标签t保守记为t+1分钟可得；采用原始标签t+3分钟open作为延迟成交代理，使两种标签语义下均晚于信息形成。拒绝跨午休、标签>=14:57、该分钟无成交、入场价已破固定结构低点、涨停价买入。",
        "fill_limit": "分钟open加不利滑点与最小价位取整仅为历史成交模型，不证明个人可成交、盘口容量或真实排队。",
        "outcomes": {"primary": "T1_OPEN", "legal_secondary": ["T1_CLOSE", "T2_CLOSE", "T5_CLOSE"], "nontradable_path_diagnostics": ["M5", "M30", "T0_CLOSE"]},
        "label_endpoint": "下一交易日及后续第2/5日由独立日线交易日历确定，日内端点不得跨午休；次日开盘跌停不假设完成卖出，保留无可成交标签。",
        "costs": {"BASE": {"commission_per_side": .0002, "slippage_per_side": .0005}, "STRESS": {"commission_per_side": .0004, "slippage_per_side": .001}, "minimum_commission": 5, "lot": 100, "tick": TICK, "event_allocation_cny": ALLOCATION, "account_capital_reference_cny": 200000, "stamp_duty": 0},
        "dividends": "按登记日持有条件和除息日在收益端确认应收权益，到账前不充当可投资现金。孤立事件收益不是连续账户。",
        "primary_comparison": "主比较为共同背景覆盖的P4减P2，在同一入场时点检验叠加MACD/波动/量的整体贡献；另保存P2减P1、P1减P0、P3减P2。不交易为零；背景或未来价格缺失不能记零。",
        "uncertainty": "完整交易日日历五日循环块抽样2000次，随机种子20261001；少于10条成熟事件不报告区间。",
        "fixed_halves": ["2021-08-12至2023-12-29", "2024-01-02至2026-08-12"],
        "evidence_scope": "全部属于已使用的回溯历史；日期分段仅用于描述，不称留出、未见OOS或前瞻。",
        "portfolio_gate": "唯一主方案P4压力T1_OPEN净均值及95%下界>0，P4减P2贡献差95%下界>0，两个固定半段净均值均>0，才可进入新冻结完整账户阶段。其他期限、单一条件或八格中最好者不能替代P4。",
        "stop": "主门失败则停止本次定义，完整账户NOT_RUN；不改方向、时间窗、参考位置、成本或等待长度补救。数据失败独立记NO_VIEW。",
        "real_order_flow_status": "NOT_RUN_NO_ADMITTED_TRADE_AND_ORDER_EVENTS",
        "execution_assets": ["510300.SH", "CASH_CNY"],
        "new_collection": False,
        "orders_authorized": False,
        "sources": [
            {"title": "ICT 2022 Mentorship Episode 6", "url": "https://www.youtube.com/watch?v=Bkt8B3kLATQ", "read_scope": "原视频元数据和简介，未取得完整字幕"},
            {"title": "Axia 假突破案例原文", "url": "https://axiafutures.com/blog/elite-trader-trades-false-break-setup-with-200-lots/", "read_scope": "作者配套全文，非收益认证"},
            {"title": "FVG工具定义", "url": "https://docs.luxalgo.com/platform/algos/price-action-concepts/imbalances", "read_scope": "三柱价格定义，不将供需叙事视为已证因果"},
            {"title": "上交所股票ETF T+1", "url": "https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml", "read_scope": "交易约束"},
        ],
    }


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("已有冻结记录，禁止覆盖。")
    m, d, div = inputs()
    quality = check_inputs(m, d)
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "input_quality.json", quality)
    save_json(OUT / "protocol.json", protocol())
    save_json(OUT / "freeze.json", {"frozen_at": now(), "results_not_computed": True, "code_sha256": sha(Path(__file__)), "protocol_sha256": sha(OUT / "protocol.json"), "inputs": {k: {"path": str(v), "bytes": v.stat().st_size, "sha256": sha(v)} for k, v in FILES.items()}, "existing_prices_have_been_used_in_prior_research": True})
    print("定义、输入与实现已固定，尚未计算本轮事件收益。", flush=True)


def frozen_check():
    f = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert f["code_sha256"] == sha(Path(__file__)), "冻结代码已改变"
    assert f["protocol_sha256"] == sha(OUT / "protocol.json"), "冻结定义已改变"
    for item in f["inputs"].values():
        assert item["sha256"] == sha(Path(item["path"])), "输入已改变"


def run():
    frozen_check()
    if (OUT / "result.json").exists():
        raise RuntimeError("本轮结果已存在，禁止重跑覆盖。")
    save_json(OUT / "run_started.json", {"started_at": now()})
    m, d, div = inputs()
    b = context_features(aggregate(m), d, div)
    events, raw_signals = detect_all(b, d, div)
    signals = attach_context(raw_signals, b)
    print(f"状态重建完成：{len(events)}个交易日、{len(signals)}条分层信号。", flush=True)
    lab, execution = labels(signals, m, d, div)
    calendar = pd.DatetimeIndex(m.date.unique()).sort_values()
    summary, yearly = summarize(lab, calendar)
    pairing, pair_summary = paired_filters(lab, signals, execution, calendar)
    cells, marginal = context_analysis(lab, calendar)
    gate = primary_gate(summary, pair_summary)
    (OUT / "results").mkdir(exist_ok=True)
    for name, frame in {"全部交易日与事件": events, "全部信号": signals, "入场及未入场": execution, "全部事件收益": lab, "分组统计": summary, "逐年统计": yearly, "逐机会增量": pairing, "条件增量统计": pair_summary, "MACD波动率量八格": cells, "单项背景增量": marginal}.items():
        csv(OUT / "results" / (name + ".csv"), frame)
        frame.to_parquet(OUT / "results" / (name + ".parquet"), index=False)
    b.to_parquet(OUT / "results/五分钟价格柱.parquet", index=False)
    result = {"study_id": STUDY, "completed_at": now(), "status": "PASS_HISTORICAL_DISCOVERY_ONLY" if gate["pass"] else "REJECTED_FROZEN_NO_PARAMETER_RESCUE", "data_days": len(events), "event_state_counts": events.state.value_counts().to_dict(), "signals_by_policy": signals.policy.value_counts().to_dict(), "execution_status_counts": execution.entry_status.value_counts().to_dict(), "primary_gate": gate, "main_rows": summary[summary.horizon.eq("T1_OPEN")].to_dict("records"), "filter_comparisons": pair_summary.to_dict("records"), "portfolio_status": "READY_FOR_SEPARATELY_FROZEN_ACCOUNT" if gate["pass"] else "NOT_RUN_PRIMARY_EVENT_GATE_FAILED", "new_market_downloads": 0, "new_fitted_models": 0, "new_parameter_searches": 0, "real_order_flow_computed": False, "strategy_goal_achieved": False, "orders_authorized": False}
    save_json(OUT / "result.json", result)
    print(json.dumps(clean({"status": result["status"], "primary_gate": gate, "primary_rows": result["main_rows"]}), ensure_ascii=False), flush=True)


def verify():
    """只读核对保存信号的时间顺序、前缀一致性及费用后收益。"""
    frozen_check()
    m, d, div = inputs()
    b = pd.read_parquet(OUT / "results/五分钟价格柱.parquet")
    signals = pd.read_parquet(OUT / "results/全部信号.parquet")
    lab = pd.read_parquet(OUT / "results/全部事件收益.parquet")
    valid = lab[lab.label_status.eq("AVAILABLE")]
    assert (valid.entry_time > valid.available_at_conservative).all()
    legal = valid[valid.legal_for_new_shares]
    assert (legal.exit_time.dt.normalize() > legal.entry_time.dt.normalize()).all()
    assert valid.quantity.mod(100).eq(0).all()
    maximum_error = 0.
    for r in valid.itertuples():
        expected = trade_outcome(r.entry_reference_price, r.exit_reference_price, r.dividend_per_share, r.cost)
        maximum_error = max(maximum_error, abs(expected["net_return"] - r.net_return))
    assert maximum_error < 1e-12
    tested = 0
    # 每类最多检查前后三个信号及三个中段信号；直接截断真实历史，不重算未来标签。
    for _, group in signals[signals.policy.ne("P4_CONTEXT")].groupby("policy"):
        positions = np.unique(np.linspace(0, len(group) - 1, min(9, len(group))).astype(int))
        for r in group.iloc[positions].itertuples():
            sub = b[b.date.eq(r.date) & b.time.le(r.signal_time)]
            _, pre = detect_day(sub, float(r.reference), float(d.loc[r.date, "open"]), r.date)
            found = [z for z in pre if z["policy"] == r.policy]
            assert len(found) == 1 and found[0]["signal_time"] == r.signal_time
            assert abs(found[0]["structure_low"] - r.structure_low) < 1e-12
            tested += 1
    feature_checks = 0
    for cut in np.unique(np.linspace(1500, len(b) - 1, 8).astype(int)):
        prefix_day = b.date.iloc[cut]
        prefix = context_features(aggregate(m[m.time.le(b.time.iloc[cut])]), d[d.index <= prefix_day], div[div.ex_date.le(prefix_day)])
        cols = ["macd_hist_bps", "macd_hist_change_bps", "rv20_relative_prior", "relative_volume"]
        assert np.allclose(prefix[cols].iloc[-1].to_numpy(float), b[cols].iloc[cut].to_numpy(float), equal_nan=True)
        feature_checks += 1
    joint = signals[signals.policy.eq("P4_CONTEXT")]
    expected_joint = signals[signals.policy.eq("P2_FVG") & signals.context_available & signals.macd_rising & signals.low_volatility & signals.relative_volume_high]
    assert set(joint.event_id) == set(expected_joint.event_id)
    summary = pd.read_parquet(OUT / "results/分组统计.parquet")
    for r in summary.itertuples():
        sub = valid[valid.policy.eq(r.policy) & valid.horizon.eq(r.horizon) & valid.cost.eq(r.cost)]
        assert len(sub) == r.events and abs(sub.net_return.mean() - r.mean_net) < 1e-12
    receipt = {"status": "PASS_SAVED_EVENT_RECOMPUTATION", "verified_at": now(), "saved_return_rows_recomputed": len(valid), "max_net_return_error": maximum_error, "real_data_signal_prefix_checks": tested, "real_data_context_prefix_checks": feature_checks, "joint_subset_verified": True, "summary_rows_verified": len(summary), "all_legal_exits_after_entry_day": True, "entry_later_than_conservative_availability": True, "no_new_backtest": True, "scope_limit": "计算与时序核对，不等于独立盈利验证或真实成交证明。"}
    save_json(OUT / "saved_verification.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="510300前日低点与FVG历史研究")
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
