"""固定三个确认时点，研究标的多空点位的胜率、实际盈亏比和频率。"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as a

OUT = ROOT / "reports/research/510300_directional_entry_timing_v1"
FAMILIES = {"MOMENTUM_TURN": "MACD柱早期转折", "EMA20_CROSS": "收盘穿过EMA20", "RANGE20_BREAK": "收盘突破20日区间"}
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"


def save(name, frame):
    target = OUT / "results" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target.with_suffix(".parquet"), index=False)
    frame.to_csv(target.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if OUT.exists():
        raise RuntimeError("多空点位实验已登记，不覆盖旧定义。")
    OUT.mkdir(parents=True)
    sources = [(a.OUT / "results/features.parquet", "inputs/features.parquet"),
               (a.OUT / "results/weekly.parquet", "inputs/weekly.parquet"),
               (a.OUT / "results/全部原点结果标签.parquet", "inputs/origin_labels.parquet"),
               (a.OUT / "inputs/dividends.csv", "inputs/dividends.csv"),
               (a.OUT / "result.json", "inputs/anatomy_result.json"),
               (CONTEXT / "point_only_scope_clarification_20261001.json", "inputs/user_scope.json"),
               (Path(__file__), "code/directional_entry_timing_v1.py"),
               (Path(a.__file__), "code/upward_episode_anatomy_v1.py"),
               (ROOT / "tests/test_directional_entry_timing_v1.py", "code/test_directional_entry_timing_v1.py")]
    files = {}
    for source, dest in sources:
        target = OUT / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[dest] = {"source": source.relative_to(ROOT).as_posix(), "sha256": a.digest(target)}
    protocol = {
        "study": "510300_DIRECTIONAL_ENTRY_TIMING_V1", "at": a.now(),
        "user_scope": "只研究点位、多空盈亏比和胜率；不选期权，不计算期权或完整账户收益。",
        "motivation": "上涨图谱显示指标确认时差不同，但单项增量区间均跨零。本轮只比较早/中/晚三个既定确认时点在同一失效退出口径下的表现，不因事后子组改方向或叠加过滤。",
        "design": "3个确认族；每族包含多头与空头；同一族只容许一个活跃点位。保留所有信号、跳空拒绝、重叠忽略及未完成点位。无参数搜索或优胜者拼接。",
        "signals": {
            "MOMENTUM_TURN": "多头：日MACD柱<0，本日柱变化>0而前日柱变化<=0；空头：柱>0，本日柱变化<0而前日柱变化>=0。",
            "EMA20_CROSS": "多头：收盘从不高于EMA20转为高于；空头：从不低于EMA20转为低于。",
            "RANGE20_BREAK": "多头首次收盘高于此前20日最高收盘；空头首次收盘低于此前20日最低收盘。连续创新高/低的同一状态只取第一次，状态退出后可再形成。"},
        "stop": "信号日及前4日的最低/最高前向平移价格，再向外加0.5个前日ATR20。失效线当时固定。",
        "target": "信号收盘沿方向加2倍收盘至失效线的距离，固定不追移。2是预设目标距离，不是已实现平均盈亏比。",
        "entry": "次日开盘参考价；开盘必须仍在已知失效与目标之间。以原始价撮合代理，进入方向涨跌停或无量则取消，不等待更有利开盘。",
        "exit": "只在收盘检查固定目标/失效，或已持有20个交易日；随后下一可平仓开盘退出，遇不利方向涨跌停/无量则顺延。无日内高低价先后成交假设。",
        "cost_reference": "方向研究使用每点位10万元固定名义规模、100份整手、0.001价格刻度，单边万四佣金及千一滑点、最低5元作统一摩擦敏感性；另外单列毛方向收益。",
        "cost_limit": "空头只是标的方向的参考损益，未假设可融券，不含借券费，也不是期权权利金收益；不能作为真实交易或期权胜率。",
        "dividend": "多头享有登记日权益，空头方向标签扣对应股息；技术价只随已除息现金向前平移，不提前复权。",
        "sequential": "每族同一时刻只占一个点位，多空共用占用状态；退出开盘之后当日收盘可产生下一个信号。重叠信号不算独立交易。",
        "annual": "每个完整自然年按入场年计至少5个已完成点位周期，2015—2025含零；2026截至数据截止日单列。",
        "working_edge": "毛口径与统一摩擦口径分别看胜率>=55%或平均盈利收益率/平均亏损收益率>=2，并要求均值>0；数字是工作比较线。",
        "evidence": "2015—2019、2020—2023、2024—2026固定分段。按完整入场日历20日循环区块2000次估计净均值区间；历史多次使用、三项同源比较，不能称独立验证。",
        "downward_atlas": "对同一含息总回报指数，以从高点回落5%确认下跌，从低点反弹5%确认结束；高低点为事后标签，确认日另存。上涨定义保持原样。",
        "scope_limit": "不将上涨或下跌事后极值注入信号；不输出当前行情点位；数据截至2026-09-16，当前日期的市场看法未建立。",
        "goal_achieved": False, "orders_authorized": False,
    }
    a.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": a.digest(OUT / "protocol.json")}
    a.save_json(OUT / "freeze.json", {"at": a.now(), "before_directional_point_returns": True, "files": files})
    print("多空点位比较已固定：三个确认时点，共用失效与退出，不选择期权合约。", flush=True)


def directional_episodes(d):
    q = d.total_return_index.to_numpy(float)
    mode, lo, hi, start, confirm = 0, 0, 0, None, None
    rows = []

    def append(end, end_confirm, status):
        rows.append({"episode_id": len(rows) + 1, "direction": mode, "start_idx": start, "end_idx": end,
                     "confirm_idx": confirm, "end_confirm_idx": end_confirm,
                     "start_date": d.date.iloc[start], "end_date": d.date.iloc[end],
                     "confirm_date": d.date.iloc[confirm],
                     "end_confirm_date": d.date.iloc[end_confirm] if end_confirm is not None else pd.NaT,
                     "directional_move": mode * (q[end] / q[start] - 1), "sessions": end - start,
                     "admitted": bool(d.date.iloc[start] >= a.START and d.available.iloc[start]), "status": status})

    for i in range(1, len(d)):
        if mode == 0:
            if q[i] < q[lo]:
                lo = i
            if q[i] > q[hi]:
                hi = i
            if q[i] >= q[lo] * 1.05:
                mode, start, confirm, hi = 1, lo, i, i
            elif q[i] <= q[hi] * .95:
                mode, start, confirm, lo = -1, hi, i, i
        elif mode == 1:
            if q[i] > q[hi]:
                hi = i
            if q[i] <= q[hi] * .95:
                append(hi, i, "COMPLETE_RETROSPECTIVE")
                mode, start, confirm, lo = -1, hi, i, i
        else:
            if q[i] < q[lo]:
                lo = i
            if q[i] >= q[lo] * 1.05:
                append(lo, i, "COMPLETE_RETROSPECTIVE")
                mode, start, confirm, hi = 1, lo, i, i
    if mode:
        append(hi if mode == 1 else lo, None, "RIGHT_CENSORED")
    return pd.DataFrame(rows)


def point_signals(d):
    slope = d.daily_hist.diff()
    states = {
        ("MOMENTUM_TURN", 1): d.daily_hist.lt(0) & slope.gt(0) & slope.shift().le(0),
        ("MOMENTUM_TURN", -1): d.daily_hist.gt(0) & slope.lt(0) & slope.shift().ge(0),
        ("EMA20_CROSS", 1): d.ac.gt(d.ema20) & d.ac.shift().le(d.ema20.shift()),
        ("EMA20_CROSS", -1): d.ac.lt(d.ema20) & d.ac.shift().ge(d.ema20.shift()),
    }
    for direction, state in [(1, d.ac.gt(d.ac.shift().rolling(20).max())),
                              (-1, d.ac.lt(d.ac.shift().rolling(20).min()))]:
        states[("RANGE20_BREAK", direction)] = state & ~state.shift(fill_value=False)
    low, high, atr = d.al.rolling(5).min(), d.ah.rolling(5).max(), d.atr20.shift()
    rows = []
    for (family, direction), mask in states.items():
        for i in np.flatnonzero(mask & d.available & d.date.ge(a.START)):
            stop = float(low.iloc[i] - .5 * atr.iloc[i] if direction == 1 else high.iloc[i] + .5 * atr.iloc[i])
            risk = direction * (float(d.ac.iloc[i]) - stop)
            if not (risk > 0):
                raise RuntimeError("失效距离不是正值。")
            rows.append({"family": family, "direction": direction, "signal_idx": int(i), "signal_date": d.date.iloc[i],
                         "signal_close": float(d.close.iloc[i]), "stop_index": stop,
                         "target_index": float(d.ac.iloc[i] + direction * 2 * risk),
                         "stop_raw_at_signal": stop - float(d.cash_shift.iloc[i]),
                         "target_raw_at_signal": float(d.ac.iloc[i] + direction * 2 * risk - d.cash_shift.iloc[i]),
                         "weekly_hist": float(d.weekly_hist.iloc[i]), "relative_volume": float(d.relative_volume.iloc[i]),
                         "rv_ratio": float(d.rv_ratio.iloc[i])})
    return pd.DataFrame(rows).sort_values(["family", "signal_idx", "direction"]).reset_index(drop=True)


def fill(price, side):
    adjusted = price * (1 + side * .001) / .001
    return (math.ceil(adjusted - 1e-9) if side > 0 else math.floor(adjusted + 1e-9)) * .001


def friction_result(entry, exit_price, dividend, direction):
    ef, xf = fill(entry, direction), fill(exit_price, -direction)
    qty = int(100000 / (entry * 100)) * 100
    fees = max(5., qty * ef * .0004) + max(5., qty * xf * .0004)
    gross = direction * (exit_price - entry + dividend) / entry
    net = (direction * qty * (xf - ef + dividend) - fees) / (qty * entry)
    return {"entry_fill": ef, "exit_fill": xf, "quantity": qty, "fees": fees,
            "gross_return": gross, "net_reference_return": net}


def at_limit(d, i, action):
    previous = float(d.close.iloc[i - 1] - d.dividend.iloc[i])
    limit = round(previous * (1 + action * .1), 3)
    return bool(d.open.iloc[i] >= limit - .0005 if action > 0 else d.open.iloc[i] <= limit + .0005)


def evaluate_point(sig, d, div):
    i = int(sig["signal_idx"]) + 1
    direction = int(sig["direction"])
    row = dict(sig)
    if i >= len(d):
        return {**row, "status": "NO_NEXT_OPEN"}
    entry = float(d.open.iloc[i])
    entry_index = float(d.ao.iloc[i])
    row.update(entry_idx=i, entry_date=d.date.iloc[i], entry_raw=entry)
    if direction * (entry_index - sig["stop_index"]) <= 0 or direction * (sig["target_index"] - entry_index) <= 0:
        return {**row, "status": "OPEN_OUTSIDE_KNOWN_INTERVAL"}
    if d.volume.iloc[i] <= 0 or at_limit(d, i, direction):
        return {**row, "status": "ENTRY_LIMIT_OR_NO_VOLUME"}
    stop_raw = sig["stop_index"] - d.cash_shift.iloc[i]
    target_raw = sig["target_index"] - d.cash_shift.iloc[i]
    loss = friction_result(entry, stop_raw, 0., direction)["net_reference_return"]
    reward = friction_result(entry, target_raw, 0., direction)["net_reference_return"]
    row.update(entry_stop_raw=stop_raw, entry_target_raw=target_raw,
               planned_reference_rr=reward / -loss if loss < 0 else np.nan)
    decision, exit_idx, reason = None, None, None
    for j in range(i, len(d)):
        if decision is not None:
            if d.volume.iloc[j] > 0 and not at_limit(d, j, -direction):
                exit_idx = j
                break
        else:
            if direction * (d.ac.iloc[j] - sig["stop_index"]) <= 0:
                decision, reason = j, "STRUCTURE_FAILED"
            elif direction * (d.ac.iloc[j] - sig["target_index"]) >= 0:
                decision, reason = j, "TARGET_CONFIRMED"
            elif j - i + 1 >= 20:
                decision, reason = j, "TIME_20_SESSIONS"
    if exit_idx is None:
        return {**row, "status": "OPEN_RIGHT_CENSORED", "exit_reason": reason}
    exit_day = d.date.iloc[exit_idx]
    dv = float(div.loc[div.record_date.ge(d.date.iloc[i]) & div.record_date.lt(exit_day) & div.ex_date.le(exit_day), "cash_dividend_per_share"].sum())
    results = friction_result(entry, float(d.open.iloc[exit_idx]), dv, direction)
    row.update(status="COMPLETE", exit_idx=exit_idx, exit_date=exit_day, decision_idx=decision,
               decision_date=d.date.iloc[decision], exit_reason=reason, holding_sessions=exit_idx - i,
               exit_raw=float(d.open.iloc[exit_idx]), dividend_per_share=dv,
               realized_r=results["net_reference_return"] / -loss if loss < 0 else np.nan, **results)
    return row


def sequences(signals, d, div):
    rows = []
    for family in FAMILIES:
        busy_until = -1
        for sig in signals.loc[signals.family.eq(family)].to_dict("records"):
            if sig["signal_idx"] < busy_until:
                rows.append({**sig, "status": "OVERLAPPING_ACTIVE_POINT"})
                continue
            record = evaluate_point(sig, d, div)
            rows.append(record)
            if record["status"] == "COMPLETE":
                busy_until = record["exit_idx"]
            elif record["status"] == "OPEN_RIGHT_CENSORED":
                busy_until = len(d)
    return pd.DataFrame(rows)


def stats(values):
    r = np.asarray(values, float)
    pos, neg = r[r > 0], r[r < 0]
    return {"cycles": len(r), "wins": len(pos), "losses": len(neg),
            "win_rate": float((r > 0).mean()) if len(r) else np.nan,
            "payoff": float(pos.mean() / -neg.mean()) if len(pos) and len(neg) else np.nan,
            "mean_return": float(r.mean()) if len(r) else np.nan,
            "median_return": float(np.median(r)) if len(r) else np.nan,
            "mean_win": float(pos.mean()) if len(pos) else np.nan,
            "mean_loss": float(neg.mean()) if len(neg) else np.nan}


def summarize(points, d):
    complete = points.loc[points.status.eq("COMPLETE")].copy()
    rows, counts, confidence = [], [], []
    dates = d.loc[d.date.ge(a.START), "date"].reset_index(drop=True)
    date_to_idx = dict(zip(dates, dates.index))
    rng = np.random.default_rng(202610014)
    starts = rng.integers(0, len(dates), size=(2000, math.ceil(len(dates) / 20)))
    draw = ((starts[:, :, None] + np.arange(20)) % len(dates)).reshape(2000, -1)[:, :len(dates)]
    for family in FAMILIES:
        group = complete.loc[complete.family.eq(family)]
        for year in range(2015, 2027):
            yy = group.loc[group.entry_date.dt.year.eq(year)]
            counts.append({"family": family, "year": year, "full_year": year < 2026, "cycles": len(yy),
                           "long": int(yy.direction.eq(1).sum()), "short": int(yy.direction.eq(-1).sum()),
                           "at_least_five": len(yy) >= 5})
        for era, (start, end) in a.ERAS.items():
            g = group.loc[group.entry_date.between(start, end)]
            for direction in [0, 1, -1]:
                part = g if direction == 0 else g.loc[g.direction.eq(direction)]
                for cost, column in [("GROSS", "gross_return"), ("REFERENCE_FRICTION", "net_reference_return")]:
                    rows.append({"family": family, "era": era, "direction": direction, "cost": cost, **stats(part[column])})
        for direction in [0, 1, -1]:
            part = group if direction == 0 else group.loc[group.direction.eq(direction)]
            numerator, denominator = np.zeros(len(dates)), np.zeros(len(dates))
            for point in part.itertuples():
                index = date_to_idx[point.entry_date]
                numerator[index] += point.net_reference_return
                denominator[index] += 1
            nn, dd = numerator[draw].sum(axis=1), denominator[draw].sum(axis=1)
            samples = np.divide(nn, dd, out=np.full(2000, np.nan), where=dd > 0)
            confidence.append({"family": family, "direction": direction,
                               "mean_return": float(part.net_reference_return.mean()),
                               "lower95": float(np.nanquantile(samples, .025)),
                               "upper95": float(np.nanquantile(samples, .975))})
    return pd.DataFrame(rows), pd.DataFrame(counts), pd.DataFrame(confidence)


def down_description(d, w, episodes):
    week_change = w.weekly_hist.diff()
    states = {
        "above_ema20": d.ac.lt(d.ema20), "daily_hist_positive": d.daily_hist.lt(0),
        "daily_hist_rising": d.daily_hist.lt(d.daily_hist.shift()), "daily_dif_positive": d.daily_dif.lt(0),
        "weekly_hist_rising": d.last_week_idx.map(week_change).lt(0),
        "weekly_hist_positive": d.weekly_hist.lt(0), "volume_expansion": d.volume_expansion,
        "up_volume_dominant": d.up_volume_balance5.lt(0), "low_volatility": d.low_volatility,
        "breakout20": d.ac.lt(d.ac.shift().rolling(20).min()),
    }
    rows, timing = [], []
    for ep in episodes.loc[episodes.admitted & episodes.direction.eq(-1) & episodes.status.eq("COMPLETE_RETROSPECTIVE")].to_dict("records"):
        b, end = int(ep["start_idx"]), int(ep["end_idx"])
        for stage, idx in {"PRE5": b - 5, "BOTTOM": b, "POST5": b + 5, "CONFIRM5": int(ep["confirm_idx"]), "PEAK": end}.items():
            if idx >= len(d) or not d.available.iloc[idx]:
                continue
            rows.append({"episode_id": ep["episode_id"], "stage": stage, "date": d.date.iloc[idx],
                         "after_end": idx > end, **{key: bool(value.iloc[idx]) for key, value in states.items()}})
        for key, state in states.items():
            indices = np.flatnonzero(state.iloc[b:end + 1].to_numpy(bool))
            first = b + int(indices[0]) if len(indices) else None
            timing.append({"episode_id": ep["episode_id"], "feature": key, "present_in_wave": first is not None,
                           "already_at_start": bool(state.iloc[b]), "first_idx": first,
                           "lag_sessions": first - b if first is not None else np.nan,
                           "move_already": 1 - d.total_return_index.iloc[first] / d.total_return_index.iloc[b] if first is not None else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(timing)


def verify(d, episodes, signals, points, div):
    for n in [900, 1700, 2500, len(d) - 11]:
        partial = point_signals(d.iloc[:n])
        expected = signals.loc[signals.signal_idx.lt(n)].reset_index(drop=True)
        pd.testing.assert_frame_equal(partial, expected)
        short = directional_episodes(d.iloc[:n])
        observed = short.loc[short.status.eq("COMPLETE_RETROSPECTIVE")]
        known = episodes.loc[episodes.end_confirm_idx.lt(n)]
        assert observed[["direction", "start_idx", "end_idx", "confirm_idx", "end_confirm_idx"]].to_dict("records") == known[["direction", "start_idx", "end_idx", "confirm_idx", "end_confirm_idx"]].to_dict("records")
    old = a.upward_episodes(d)
    left = old.loc[old.status.eq("COMPLETE_RETROSPECTIVE"), ["bottom_idx", "peak_idx"]].to_numpy()
    right = episodes.loc[episodes.direction.eq(1) & episodes.status.eq("COMPLETE_RETROSPECTIVE"), ["start_idx", "end_idx"]].to_numpy()
    np.testing.assert_array_equal(left, right)
    error = 0.
    for family, group in points.loc[points.status.eq("COMPLETE")].groupby("family"):
        ordered = group.sort_values("entry_idx")
        assert (ordered.entry_idx.to_numpy()[1:] > ordered.exit_idx.to_numpy()[:-1]).all()
        for row in group.itertuples():
            dividend = float(div.loc[div.record_date.ge(row.entry_date) & div.record_date.lt(row.exit_date) & div.ex_date.le(row.exit_date), "cash_dividend_per_share"].sum())
            gross = row.direction * (row.exit_raw - row.entry_raw + dividend) / row.entry_raw
            fees = max(5., row.quantity * row.entry_fill * .0004) + max(5., row.quantity * row.exit_fill * .0004)
            net = (row.direction * row.quantity * (row.exit_fill - row.entry_fill + dividend) - fees) / (row.quantity * row.entry_raw)
            error = max(error, abs(gross - row.gross_return), abs(net - row.net_reference_return))
            assert row.entry_idx <= row.decision_idx < row.exit_idx
            assert row.signal_idx < row.entry_idx
    assert error < 1e-12
    return {"status": "PASS_POINT_CLOCK_AND_DIRECTIONAL_RETURN_RECOMPUTATION", "prefixes": 4,
            "complete_point_rows": int(points.status.eq("COMPLETE").sum()), "maximum_return_error": error,
            "old_upward_labels_identical": True, "full_account": "NOT_COMPUTED_USER_REQUESTED_POINTS_ONLY"}


def run():
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert all(a.digest(OUT / name) == record["sha256"] for name, record in frozen["files"].items())
    if (OUT / "result.json").exists():
        raise RuntimeError("已有结果不覆盖。")
    d = pd.read_parquet(OUT / "inputs/features.parquet")
    w = pd.read_parquet(OUT / "inputs/weekly.parquet")
    div = pd.read_csv(OUT / "inputs/dividends.csv")
    for name in ["record_date", "ex_date", "payment_date"]:
        div[name] = pd.to_datetime(div[name])
    episodes = directional_episodes(d)
    signals = point_signals(d)
    points = sequences(signals, d, div)
    summaries, annual, intervals = summarize(points, d)
    snapshots, timing = down_description(d, w, episodes)
    for name, frame in [("多空波段全集", episodes), ("所有候选点位", signals), ("点位逐笔与拒绝原因", points),
                        ("胜率实际盈亏比", summaries), ("逐年次数", annual), ("净均值区间", intervals),
                        ("下跌阶段快照", snapshots), ("下跌确认时差", timing)]:
        save(name, frame)
    passing = []
    for family in FAMILIES:
        main = summaries.loc[summaries.family.eq(family) & summaries.era.eq("ALL") & summaries.direction.eq(0) & summaries.cost.eq("REFERENCE_FRICTION")].iloc[0]
        frequency = bool(annual.loc[annual.family.eq(family) & annual.full_year, "at_least_five"].all())
        edge = bool((main.win_rate >= .55 or main.payoff >= 2.) and main.mean_return > 0)
        ci = intervals.loc[intervals.family.eq(family) & intervals.direction.eq(0)].iloc[0]
        passing.append({"family": family, "annual_frequency_pass": frequency, "descriptive_edge_pass": edge,
                        "positive_mean_lower95": bool(ci.lower95 > 0), "independent_validation": False})
    verification = verify(d, episodes, signals, points, div)
    a.save_json(OUT / "verification.json", verification)
    a.save_json(OUT / "result.json", {"at": a.now(), "status": "DIRECTIONAL_POINT_COMPARISON_COMPLETE_NOT_VALIDATED_STRATEGY",
                "user_scope": "标的多空点位，非期权收益", "data_end": str(d.date.iloc[-1].date()),
                "complete_up_episodes": int((episodes.admitted & episodes.direction.eq(1) & episodes.status.eq("COMPLETE_RETROSPECTIVE")).sum()),
                "complete_down_episodes": int((episodes.admitted & episodes.direction.eq(-1) & episodes.status.eq("COMPLETE_RETROSPECTIVE")).sum()),
                "censored_episodes": int((episodes.admitted & episodes.status.eq("RIGHT_CENSORED")).sum()),
                "families": passing, "goal_achieved": False, "orders_authorized": False})
    print(summaries.loc[summaries.era.eq("ALL") & summaries.direction.eq(0)].round(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="标的多空点位研究，不选择期权合约。")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        run()
