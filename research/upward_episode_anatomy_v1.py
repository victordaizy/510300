"""从全部上涨段出发，分清技术指标的事后解释、确认时差和事前区分能力。"""
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
OUT = ROOT / "reports/research/510300_upward_episode_anatomy_v1"
SOURCE = ROOT / "reports/research/510300_weekly_daily_entry_locations_v1/inputs"
START = pd.Timestamp("2015-01-01")
STATES = {
    "above_ema20": "收盘在20日均线上",
    "daily_hist_positive": "日MACD在金叉侧",
    "daily_hist_rising": "日MACD柱回升",
    "daily_dif_positive": "日DIF在零轴上",
    "weekly_hist_rising": "已完成周MACD柱回升",
    "weekly_hist_positive": "已完成周MACD在金叉侧",
    "volume_expansion": "当日量至少为此前中位数1.5倍",
    "up_volume_dominant": "近5日上涨日成交量占优",
    "low_volatility": "20日波动不高于过去一年中位数",
    "breakout20": "收盘突破此前20日最高收盘",
}
STAGES = {"PRE5": "低点前5日", "BOTTOM": "事后低点日", "POST5": "低点后5日",
          "CONFIRM5": "首次涨到5%确认日", "PEAK": "事后高点日"}
ERAS = {"ALL": ("2015-01-01", "2026-12-31"), "2015_2019": ("2015-01-01", "2019-12-31"),
        "2020_2023": ("2020-01-01", "2023-12-31"), "2024_2026": ("2024-01-01", "2026-12-31")}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if OUT.exists():
        raise RuntimeError("研究目录已存在，保留原记录，不能覆盖后再改定义。")
    OUT.mkdir(parents=True)
    files = {}
    for src, dest in [(SOURCE / "prices.parquet", "inputs/prices.parquet"),
                      (SOURCE / "dividends.csv", "inputs/dividends.csv"),
                      (SOURCE / "dividend_coverage.json", "inputs/dividend_coverage.json"),
                      (Path(__file__).resolve(), "code/upward_episode_anatomy_v1.py"),
                      (ROOT / "tests/test_upward_episode_anatomy_v1.py", "code/test_upward_episode_anatomy_v1.py")]:
        target = OUT / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        files[dest] = {"source": src.relative_to(ROOT).as_posix(), "sha256": digest(target)}
    protocol = {
        "study": "510300_UPWARD_EPISODE_ANATOMY_V1", "at": now(),
        "user_instruction": "我们可以换个方向，看到他上涨，看能不能用技术指标解释，量价关系，然后我们再反推策略。",
        "scope": "510300日线、上一完整周；先解释全部上涨段并检验失败对照，不使用分钟线。",
        "evidence": "回顾性机制探索；历史已被多轮研究使用，不称独立验证或因果识别。",
        "episodes": "以逐日含现金分红再投资的收盘总回报指数，固定5%涨幅确认从低点进入上涨；从滚动高点回落5%后才确认上涨高点。低点、高点均为事后标签，确认日另存。无最短长度、收益筛选或参数网格。",
        "episode_admission": "仅以低点日期>=2015-01-01计入正式图谱；此前跨界段单列排除。期末没有回落确认的上涨段单列RIGHT_CENSORED，不假定末日为顶部。",
        "features": "价格只加截至当日已除息现金形成前向平移序列；EMA20，MACD12/26/9柱=2*(DIF-DEA)，ATR20，相对量=当日量/前20日中位量；RV20/此前252日RV20中位数。周MACD仅下一自然周可用，130周预热。",
        "volume_proxy": "近5日成交量按当日含息收益的正负加减后除以总量，只是上涨日/下跌日量的比较，不是主动买卖差、订单流或资金身份。",
        "states": STATES,
        "snapshots": STAGES,
        "timing": "每段自事后低点到高点第一次状态为真；低点已为真单列。时差和已涨幅是事后对齐描述，不把低点已知或高点可卖作为交易假设。",
        "daily_counterexample": "全部合格收盘日做原点；从次日开盘到第20个持有日收盘，按不再投资的应得股息计算回报。末期不成熟原点保留但不评价。>=5%为上涨、<=-5%为下跌，其余中间；另记途中涨到5%却期末未守住的样本。",
        "event_net_semantics": "每个原点独立10万元参考买入，100份整手，0.001刻度，单边万四佣金和千一滑点、最低5元。固定终点收盘是结果标签，不是实际收盘成交策略；事件会重叠，无账户净值，不计算策略夏普/交易次数。",
        "comparisons": "10项单项状态逐一与相反状态比较，另在首次上穿EMA20的共同价格事件上逐一增加条件；不枚举多项组合、不选最优阈值。",
        "dependence": "主要全交易日统计含20日重叠；固定20日循环区块2000次重抽样，95%区间为逐项描述区间，未作多重比较校正。另给固定日历每20日一个原点的非重叠结果，不按结果变更相位。",
        "eras": ERAS,
        "induction_rule": "仅把三段中各至少20个价格确认事件、净均值改善同向且总体区块区间下界>0的增量特征列为待检验假说；该门只判断下一步是否有证据，不自动产生策略。任何事后新想法另登记，不在本轮补做组合搜索。",
        "goal_preserved": "每个完整年至少5个完整交易周期，高胜率或大实际盈亏比且净期望正；20万元及现有完整账户风险/成本目标继续。图谱事件不冒充交易周期。",
        "old_studies": "旧趋势生命周期研究使用指数/成分结构；本轮研究单ETF技术量价。旧冻结失败与旧供给测试均保留，不借此复活。",
        "sources": ["https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd",
                    "https://www.schwab.com/learn/story/trading-volume-as-market-indicator"],
        "orders_authorized": False,
    }
    save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": digest(OUT / "protocol.json")}
    save_json(OUT / "freeze.json", {"at": now(), "before_new_label_read": True, "files": files})
    print("上涨图谱定义已固定：完整波段、事前时钟、十项指标和全部失败对照。", flush=True)


def features(prices, dividends):
    d = prices.sort_values("date").copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date)
    assert not d.date.duplicated().any()
    assert (d[["open", "high", "low", "close"]] > 0).all().all()
    assert d.symbol.eq("510300.SH").all()
    ex = dividends.groupby("ex_date").cash_dividend_per_share.sum()
    d["dividend"] = d.date.map(ex).fillna(0.)
    d["cash_shift"] = d.dividend.cumsum()
    for column, target in [("open", "ao"), ("high", "ah"), ("low", "al"), ("close", "ac")]:
        d[target] = d[column] + d.cash_shift
    d["return1"] = (d.close + d.dividend) / d.close.shift() - 1
    d["total_return_index"] = (1 + d.return1.fillna(0.)).cumprod()
    d["ema20"] = d.ac.ewm(span=20, adjust=False).mean()
    d["daily_dif"] = d.ac.ewm(span=12, adjust=False).mean() - d.ac.ewm(span=26, adjust=False).mean()
    d["daily_dea"] = d.daily_dif.ewm(span=9, adjust=False).mean()
    d["daily_hist"] = 2 * (d.daily_dif - d.daily_dea)
    tr = pd.concat([d.ah - d.al, (d.ah - d.ac.shift()).abs(), (d.al - d.ac.shift()).abs()], axis=1).max(axis=1)
    d["atr20"] = tr.rolling(20).mean()
    d["hist_atr"] = d.daily_hist / d.atr20
    d["relative_volume"] = d.volume / d.volume.shift().rolling(20).median()
    d["up_volume_balance5"] = (d.volume * np.sign(d.return1)).rolling(5).sum() / d.volume.rolling(5).sum()
    d["rv20"] = np.log1p(d.return1).rolling(20).std(ddof=1)
    d["rv_ratio"] = d.rv20 / d.rv20.shift().rolling(252, min_periods=252).median()
    d["past20_return"] = d.total_return_index / d.total_return_index.shift(20) - 1
    d["period"] = d.date.dt.to_period("W-FRI")
    w = d.groupby("period", sort=True).agg(last_date=("date", "last"), close=("ac", "last")).reset_index()
    dif = w.close.ewm(span=12, adjust=False).mean() - w.close.ewm(span=26, adjust=False).mean()
    w["weekly_hist"] = 2 * (dif - dif.ewm(span=9, adjust=False).mean())
    w["weekly_hist_rising"] = w.weekly_hist.gt(w.weekly_hist.shift())
    w["weekly_hist_positive"] = w.weekly_hist.gt(0.)
    w["weekly_available"] = w.index >= 129
    lookup = dict(zip(w.period, w.index))
    d["last_week_idx"] = d.period.map(lookup) - 1
    known = w.rename(columns={"last_date": "weekly_last_date"}).drop(columns=["period", "close"])
    d = d.join(known, on="last_week_idx")
    d["above_ema20"] = d.ac.gt(d.ema20)
    d["daily_hist_positive"] = d.daily_hist.gt(0.)
    d["daily_hist_rising"] = d.daily_hist.gt(d.daily_hist.shift())
    d["daily_dif_positive"] = d.daily_dif.gt(0.)
    d["volume_expansion"] = d.relative_volume.ge(1.5)
    d["up_volume_dominant"] = d.up_volume_balance5.gt(0.)
    d["low_volatility"] = d.rv_ratio.le(1.)
    d["breakout20"] = d.ac.gt(d.ac.shift().rolling(20).max())
    d["price_cross_ema20"] = d.above_ema20 & ~d.above_ema20.shift(fill_value=False)
    d["available"] = d.weekly_available.eq(True) & d.rv_ratio.notna() & d.relative_volume.notna()
    for state in STATES:
        d[state] = d[state].eq(True)
    d["period"] = d.period.astype(str)
    w["period"] = w.period.astype(str)
    return d, w


def upward_episodes(d, threshold=.05):
    """确认与极值分开保存；没有确认回落时不制造已完成顶部。"""
    q = d.total_return_index.to_numpy(float)
    mode, lo, hi = 0, 0, 0
    bottom, confirm_up = None, None
    rows = []

    def append(peak, confirm_down, state):
        rows.append({"episode_id": len(rows) + 1, "bottom_idx": bottom, "peak_idx": peak,
                     "confirm_up_idx": confirm_up, "confirm_down_idx": confirm_down,
                     "bottom_date": d.date.iloc[bottom], "peak_date": d.date.iloc[peak],
                     "confirm_up_date": d.date.iloc[confirm_up],
                     "confirm_down_date": d.date.iloc[confirm_down] if confirm_down is not None else pd.NaT,
                     "gross_rise": q[peak] / q[bottom] - 1, "rise_sessions": peak - bottom,
                     "confirmation_sessions": confirm_up - bottom,
                     "rise_at_confirmation": q[confirm_up] / q[bottom] - 1,
                     "status": state, "admitted": bool(d.date.iloc[bottom] >= START and d.available.iloc[bottom])})

    for i in range(1, len(d)):
        if mode == 0:
            if q[i] < q[lo]:
                lo = i
            if q[i] > q[hi]:
                hi = i
            if q[i] >= q[lo] * (1 + threshold):
                mode, bottom, confirm_up, hi = 1, lo, i, i
            elif q[i] <= q[hi] * (1 - threshold):
                mode, lo = -1, i
        elif mode == 1:
            if q[i] > q[hi]:
                hi = i
            if q[i] <= q[hi] * (1 - threshold):
                append(hi, i, "COMPLETE_RETROSPECTIVE")
                mode, lo = -1, i
        else:
            if q[i] < q[lo]:
                lo = i
            if q[i] >= q[lo] * (1 + threshold):
                mode, bottom, confirm_up, hi = 1, lo, i, i
    if mode == 1:
        append(hi, None, "RIGHT_CENSORED")
    return pd.DataFrame(rows)


def label_origins(d, dividends):
    rows = []
    for i in np.flatnonzero(d.date.ge(START) & d.available):
        e, x = i + 1, i + 20
        row = {"idx": i, "date": d.date.iloc[i], "entry_idx": e, "end_idx": x,
               "status": "MATURE" if x < len(d) else "RIGHT_CENSORED"}
        if x >= len(d):
            rows.append(row)
            continue
        entry_day, end_day = d.date.iloc[e], d.date.iloc[x]
        eligible = dividends.loc[dividends.record_date.ge(entry_day) & dividends.record_date.le(end_day)]
        cash = float(eligible.cash_dividend_per_share.sum())
        path = d.iloc[e:x + 1]
        # 标签按登记收盘形成权益；除息前仍留在原价中，尚未除息不额外计入现金。
        accrued = np.array([eligible.loc[eligible.ex_date.le(day), "cash_dividend_per_share"].sum() for day in path.date])
        cash = float(accrued[-1])
        wealth = (path.close.to_numpy() + accrued) / d.open.iloc[e] - 1
        gross = float(wealth[-1])
        bp = math.ceil(d.open.iloc[e] * 1.001 / .001 - 1e-9) * .001
        sp = math.floor(d.close.iloc[x] * .999 / .001 + 1e-9) * .001
        qty = int(100000 / (bp * 100)) * 100
        while qty * bp + max(5., qty * bp * .0004) > 100000:
            qty -= 100
        buy_cost = qty * bp + max(5., qty * bp * .0004)
        pnl = qty * (sp - bp + cash) - max(5., qty * bp * .0004) - max(5., qty * sp * .0004)
        row.update(entry_date=entry_day, end_date=end_day, gross_return=gross, net_reference_return=pnl / buy_cost,
                   max_close_excursion=float(wealth.max()), min_close_excursion=float(wealth.min()),
                   upside5=bool(gross >= .05), downside5=bool(gross <= -.05),
                   up_then_faded=bool(wealth.max() >= .05 and gross < .05), dividend_per_share=cash,
                   entry_raw=float(d.open.iloc[e]), end_raw=float(d.close.iloc[x]), quantity=qty,
                   buy_fill=bp, sell_fill=sp, buy_cost=buy_cost, net_pnl=pnl)
        rows.append(row)
    return pd.DataFrame(rows)


def snapshots_and_timing(d, episodes):
    snapshots, timing = [], []
    q = d.total_return_index.to_numpy(float)
    for ep in episodes.loc[episodes.admitted & episodes.status.eq("COMPLETE_RETROSPECTIVE")].to_dict("records"):
        b, p = int(ep["bottom_idx"]), int(ep["peak_idx"])
        for stage, i in {"PRE5": b - 5, "BOTTOM": b, "POST5": b + 5,
                         "CONFIRM5": int(ep["confirm_up_idx"]), "PEAK": p}.items():
            if not (0 <= i < len(d)) or not d.available.iloc[i]:
                continue
            r = d.iloc[i]
            row = {"episode_id": ep["episode_id"], "stage": stage, "date": r.date, "idx": i,
                   "after_peak": i > p, "gain_from_bottom": q[i] / q[b] - 1}
            row.update({k: bool(r[k]) for k in STATES})
            row.update({k: float(r[k]) for k in ["relative_volume", "up_volume_balance5", "rv_ratio", "hist_atr", "past20_return"]})
            snapshots.append(row)
        for state in STATES:
            local = d[state].iloc[b:p + 1].to_numpy(bool)
            indices = np.flatnonzero(local)
            first = b + int(indices[0]) if len(indices) else None
            timing.append({"episode_id": ep["episode_id"], "feature": state, "present_in_wave": first is not None,
                           "already_at_bottom": bool(local[0]), "first_idx": first,
                           "first_date": d.date.iloc[first] if first is not None else pd.NaT,
                           "lag_sessions": first - b if first is not None else np.nan,
                           "gain_already": q[first] / q[b] - 1 if first is not None else np.nan,
                           "fraction_log_move_used": np.log(q[first] / q[b]) / np.log(q[p] / q[b]) if first is not None else np.nan})
    return pd.DataFrame(snapshots), pd.DataFrame(timing)


def distribution(rows):
    r = rows.net_reference_return.to_numpy(float)
    pos, neg = r[r > 0], r[r < 0]
    return {"n": len(r), "upside5_rate": float(rows.upside5.mean()) if len(r) else np.nan,
            "downside5_rate": float(rows.downside5.mean()) if len(r) else np.nan,
            "up_then_faded_rate": float(rows.up_then_faded.mean()) if len(r) else np.nan,
            "gross_mean": float(rows.gross_return.mean()) if len(r) else np.nan,
            "net_mean": float(r.mean()) if len(r) else np.nan,
            "positive_net_rate": float((r > 0).mean()) if len(r) else np.nan,
            "payoff_reference": float(pos.mean() / -neg.mean()) if len(pos) and len(neg) else np.nan}


def conditional_tables(d, labels):
    full = labels.loc[labels.status.eq("MATURE")].merge(d[["date", *STATES, "price_cross_ema20"]], on="date", validate="one_to_one")
    rows, increments, nonoverlap = [], [], []
    for era, (start, end) in ERAS.items():
        g = full.loc[full.date.between(start, end)].reset_index(drop=True)
        rows.append({"era": era, "scope": "ALL_DAYS", "feature": "BASELINE", "condition": "ALL", **distribution(g)})
        for state in STATES:
            for flag in [True, False]:
                part = g.loc[g[state].eq(flag)]
                rows.append({"era": era, "scope": "ALL_DAYS", "feature": state, "condition": str(flag), **distribution(part)})
        cross = g.loc[g.price_cross_ema20]
        rows.append({"era": era, "scope": "PRICE_CROSS", "feature": "BASELINE", "condition": "ALL", **distribution(cross)})
        for state in STATES:
            if state == "above_ema20":
                continue
            selected = cross.loc[cross[state]]
            stats = distribution(selected)
            rows.append({"era": era, "scope": "PRICE_CROSS", "feature": state, "condition": "True", **stats})
            increments.append({"era": era, "feature": state, "selected_n": len(selected), "base_n": len(cross),
                               "selected_net_mean": stats["net_mean"], "base_net_mean": cross.net_reference_return.mean(),
                               "net_mean_increment": stats["net_mean"] - cross.net_reference_return.mean()})
        # 相位固定在全样本首个合格日，不为每个年代另换相位。
        sample = g.loc[(g.idx - int(full.idx.iloc[0])) % 20 == 0]
        nonoverlap.append({"era": era, "feature": "BASELINE", "condition": "ALL", **distribution(sample)})
        for state in STATES:
            nonoverlap.append({"era": era, "feature": state, "condition": "True", **distribution(sample.loc[sample[state]])})
    # 成对重抽整个日历块，保留信号稀疏性及相互重叠；所有指标共用抽样索引。
    rng = np.random.default_rng(202610013)
    n = len(full)
    offsets = np.arange(20)
    starts = rng.integers(0, n, size=(2000, math.ceil(n / 20)))
    indices = ((starts[:, :, None] + offsets) % n).reshape(2000, -1)[:, :n]
    returns = full.net_reference_return.to_numpy(float)
    boot_returns = returns[indices]
    intervals = []
    for scope in ["ALL_DAYS", "PRICE_CROSS"]:
        base_mask = np.ones(n, bool) if scope == "ALL_DAYS" else full.price_cross_ema20.to_numpy(bool)
        base_boot = base_mask[indices]
        base_average = np.divide((boot_returns * base_boot).sum(axis=1), base_boot.sum(axis=1),
                                 out=np.full(2000, np.nan), where=base_boot.sum(axis=1) > 0)
        for state in STATES:
            if scope == "PRICE_CROSS" and state == "above_ema20":
                continue
            mask = base_mask & full[state].to_numpy(bool)
            chosen = mask[indices]
            average = np.divide((boot_returns * chosen).sum(axis=1), chosen.sum(axis=1),
                                out=np.full(2000, np.nan), where=chosen.sum(axis=1) > 0)
            delta = average - base_average
            intervals.append({"scope": scope, "feature": state,
                              "net_mean_increment": float(returns[mask].mean() - returns[base_mask].mean()) if mask.any() else np.nan,
                              "lower95": float(np.nanquantile(delta, .025)) if np.isfinite(delta).any() else np.nan,
                              "upper95": float(np.nanquantile(delta, .975)) if np.isfinite(delta).any() else np.nan,
                              "valid_resamples": int(np.isfinite(delta).sum())})
    return pd.DataFrame(rows), pd.DataFrame(increments), pd.DataFrame(nonoverlap), pd.DataFrame(intervals)


def verify(d, w, div, episodes, labels):
    prices = pd.read_parquet(OUT / "inputs/prices.parquet")
    prefixes = [700, 1100, 1700, 2300, 2900, len(d) - 7]
    max_error = 0.
    for n in prefixes:
        partial_div = div.loc[div.ex_date.le(pd.Timestamp(prices.date.iloc[n - 1]))]
        dd, _ = features(prices.iloc[:n], partial_div)
        for name in ["ac", "daily_hist", "weekly_hist", "rv_ratio", "relative_volume", "total_return_index"]:
            a, b = dd[name].to_numpy(float), d[name].iloc[:n].to_numpy(float)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)
            good = np.isfinite(a) & np.isfinite(b)
            max_error = max(max_error, float(np.max(np.abs(a[good] - b[good]))) if good.any() else 0.)
        for state in STATES:
            assert dd[state].equals(d[state].iloc[:n])
        old = upward_episodes(dd)
        confirmed = old.loc[old.status.eq("COMPLETE_RETROSPECTIVE")]
        expected = episodes.loc[episodes.confirm_down_idx.lt(n)]
        assert confirmed[["bottom_idx", "peak_idx", "confirm_up_idx", "confirm_down_idx"]].to_dict("records") == expected[["bottom_idx", "peak_idx", "confirm_up_idx", "confirm_down_idx"]].to_dict("records")
    known = d.loc[d.available]
    assert (known.weekly_last_date < known.date).all()
    assert (known.weekly_last_date.dt.to_period("W-FRI") < known.date.dt.to_period("W-FRI")).all()
    for ep in episodes.loc[episodes.status.eq("COMPLETE_RETROSPECTIVE")].itertuples():
        b, p, u, x = int(ep.bottom_idx), int(ep.peak_idx), int(ep.confirm_up_idx), int(ep.confirm_down_idx)
        assert b < u <= p < x
        assert d.total_return_index.iloc[u] / d.total_return_index.iloc[b] >= 1.05 - 1e-12
        assert d.total_return_index.iloc[x] / d.total_return_index.iloc[p] <= .95 + 1e-12
    error = 0.
    for r in labels.loc[labels.status.eq("MATURE")].itertuples():
        eligible = div.loc[div.record_date.ge(r.entry_date) & div.record_date.le(r.end_date) & div.ex_date.le(r.end_date)]
        expected = (r.end_raw + eligible.cash_dividend_per_share.sum()) / r.entry_raw - 1
        error = max(error, abs(expected - r.gross_return))
    assert error < 1e-12
    return {"status": "PASS_FEATURE_CLOCK_AND_LABEL_RECOMPUTATION", "prefix_checks": len(prefixes),
            "maximum_feature_error": max_error, "maximum_label_error": error,
            "mature_origin_count": int(labels.status.eq("MATURE").sum()),
            "weekly_current_bar_used": False, "retrospective_labels_enter_features": False,
            "scope": "必要时序和标签复核；不是外部独立验证。"}


def run():
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert all(digest(OUT / name) == record["sha256"] for name, record in frozen["files"].items())
    if (OUT / "result.json").exists():
        raise RuntimeError("已完成结果不覆盖。")
    p = pd.read_parquet(OUT / "inputs/prices.parquet")
    div = pd.read_csv(OUT / "inputs/dividends.csv")
    for column in ["record_date", "ex_date", "payment_date"]:
        div[column] = pd.to_datetime(div[column])
    d, w = features(p, div)
    episodes = upward_episodes(d)
    snapshots, timing = snapshots_and_timing(d, episodes)
    print("已建立上涨段，正在以全部交易日计算成功与失败对照。", flush=True)
    labels = label_origins(d, div)
    stats, increments, nonoverlap, intervals = conditional_tables(d, labels)
    for name, frame in [("features", d), ("weekly", w), ("上涨段全集", episodes), ("阶段快照", snapshots),
                        ("首次确认时差", timing), ("全部原点结果标签", labels), ("单项条件与失败对照", stats),
                        ("价格相同条件下的增量", increments), ("非重叠日历对照", nonoverlap), ("区块区间", intervals)]:
        table(name, frame)
    stage_rates = snapshots.groupby("stage")[list(STATES)].mean().reindex(STAGES)
    table("各阶段特征出现率", stage_rates.reset_index())
    ts = timing.groupby("feature").agg(episodes=("episode_id", "size"),
             in_wave=("present_in_wave", "sum"), at_bottom=("already_at_bottom", "sum"),
             median_lag=("lag_sessions", "median"), median_gain_already=("gain_already", "median"),
             median_fraction_used=("fraction_log_move_used", "median")).reset_index()
    table("指标时差汇总", ts)
    derivation = []
    for state in STATES:
        if state == "above_ema20":
            continue
        inc = increments.loc[increments.feature.eq(state) & ~increments.era.eq("ALL")]
        ci = intervals.loc[intervals.feature.eq(state) & intervals.scope.eq("PRICE_CROSS")].iloc[0]
        enough = bool(inc.selected_n.ge(20).all())
        same = bool(inc.net_mean_increment.gt(0).all())
        lower_positive = bool(ci.lower95 > 0)
        derivation.append({"feature": state, "minimum_selected_era_events": int(inc.selected_n.min()),
                           "enough_events": enough, "all_eras_positive_increment": same,
                           "overall_lower95_positive": lower_positive,
                           "eligible_hypothesis": enough and same and lower_positive})
    derivation = pd.DataFrame(derivation)
    table("反推规则证据表", derivation)
    verification = verify(d, w, div, episodes, labels)
    save_json(OUT / "verification.json", verification)
    eligible = episodes.loc[episodes.admitted]
    result = {"at": now(), "status": "EXPLANATORY_ATLAS_COMPLETE_STRATEGY_NOT_ESTABLISHED",
              "data_start": str(d.date.iloc[0].date()), "data_end": str(d.date.iloc[-1].date()),
              "analysis_start": str(START.date()), "total_price_days": len(d),
              "complete_up_episodes": int(eligible.status.eq("COMPLETE_RETROSPECTIVE").sum()),
              "right_censored_up_episodes": int(eligible.status.eq("RIGHT_CENSORED").sum()),
              "mature_daily_origins": int(labels.status.eq("MATURE").sum()),
              "censored_daily_origins": int(labels.status.eq("RIGHT_CENSORED").sum()),
              "hypotheses_passing_fixed_evidence_screen": derivation.loc[derivation.eligible_hypothesis, "feature"].tolist(),
              "full_account_metrics": "NOT_COMPUTED_EVENT_LABELS_ONLY", "goal_achieved": False,
              "independent_validation": "NOT_ESTABLISHED_HISTORY_PREVIOUSLY_USED", "orders_authorized": False}
    save_json(OUT / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="先解释上涨，再看同类特征是否能区分失败。")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        run()
