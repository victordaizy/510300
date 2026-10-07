"""用户改为双向点位、频率软目标后，补充周阻力镜像；原多头规则保持。"""
from __future__ import annotations

import argparse
import importlib.util
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
from research import directional_entry_timing_v1 as b

OUT = ROOT / "reports/research/510300_weekly_anchor_bidirectional_v1"
PARENT = ROOT / "reports/research/510300_weekly_daily_entry_locations_v1"
POLICIES = {"LONG_ONLY": "原周支撑多头", "SHORT_ONLY": "周阻力镜像空头", "BOTH": "周支撑与周阻力双向"}


def helper():
    spec = importlib.util.spec_from_file_location("weekly_anchor_frozen_helper", OUT / "code/weekly_daily_entry_locations_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(name, df):
    p = OUT / "results" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p.with_suffix(".parquet"), index=False)
    df.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if OUT.exists():
        raise RuntimeError("本次镜像研究已经登记。")
    OUT.mkdir(parents=True)
    files = {}
    for source, dest in [(PARENT / "inputs/prices.parquet", "inputs/prices.parquet"),
                         (PARENT / "inputs/dividends.csv", "inputs/dividends.csv"),
                         (PARENT / "results/四组全部交易与重叠标记.parquet", "inputs/old_points.parquet"),
                         (ROOT / "research/weekly_daily_entry_locations_v1.py", "code/weekly_daily_entry_locations_v1.py"),
                         (Path(b.__file__), "code/directional_entry_timing_v1.py"),
                         (Path(__file__), "code/weekly_anchor_bidirectional_v1.py"),
                         (b.CONTEXT / "active_goal_effective_requirements.json", "inputs/current_user_scope.json")]:
        target = OUT / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[dest] = {"source": source.relative_to(ROOT).as_posix(), "sha256": a.digest(target)}
    protocol = {
        "study": "510300_WEEKLY_ANCHOR_BIDIRECTIONAL_V1", "at": a.now(),
        "authority_change": "用户明确只研究点位、多空胜率及盈亏比，并取消每年五次硬门槛，保留提高次数的软目标。",
        "old_result_preserved": "旧P1主方案及其冻结失败保持。旧P0多头已知19笔高胜率，作为已有描述基线；本轮新增未经评价的对称空头，不宣称重新发现多头优势。",
        "long": "原P0的周支撑首次测试与日线确认全部定义保持，不新增空间、MACD、波动率或成交量过滤。",
        "short": "最新已确认周高点作阻力，枢轴左右各2周、右2周结束后的下一周才可用；最多52周，每个锚点仅首次从下方接近阻力0.5个前日ATR内时启动。",
        "short_confirmation": "准备日或随后最多5日，收盘<=阻力、低于前日最低且为阴线；先收盘>阻力+2个准备ATR则失败；超时结束。",
        "short_stop": "从准备到确认已出现的最高价加0.5个准备ATR，确认后固定。",
        "short_target": "准备时已知最近52周确认低点中，低于准备日收盘的最高一个；未知目标保留并不进入。",
        "execution": "沿用本任务固定方向点位引擎：确认次日开盘位于失效与目标之间；收盘触及目标或失效、或持有20日后，下一可平仓开盘退出。股息方向对称、涨跌停及无量限制相同。",
        "cost": "10万元固定名义规模，单边万四佣金、千一滑点、最低5元；归一于原始入场名义金额。多头与旧记录的数量预算略有不同，仅比较同样日期毛回报与同一新归一口径，不冒充期权或融券收益。",
        "policies": POLICIES,
        "overlap": "各政策独立顺序研究；BOTH同一时间只容许一个方向。若同日多空都确认，双方均记冲突不进入。持有期间信号不重复计数。",
        "frequency": "逐年次数、完整年份均值、零交易年份、最长空档，均为软目标描述，无每年五次通过门。",
        "uncertainty": "20交易日入场日历循环区块2000次，固定种子202610015；胜率Wilson区间为描述区间。三个政策同源，旧多头已被选择与查看，不称独立验证。",
        "prohibited": "不根据空头结果更改镜像、阈值、等待期、目标、止损或做多做空权重。",
        "goal_achieved": False, "orders_authorized": False,
    }
    a.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": a.digest(OUT / "protocol.json")}
    a.save_json(OUT / "freeze.json", {"at": a.now(), "before_new_short_results": True, "files": files})
    print("周阻力空头镜像已固定；周支撑原样作比较，频率是软目标。", flush=True)


def mirror(d, w):
    md, mw = d.copy(), w.copy()
    for target, source in [("ao", "ao"), ("ac", "ac"), ("ah", "al"), ("al", "ah")]:
        md[target] = -d[source]
    for target, source in [("open", "open"), ("close", "close"), ("high", "low"), ("low", "high"),
                           ("new_pivot_low", "new_pivot_high"), ("new_pivot_high", "new_pivot_low")]:
        mw[target] = -w[source]
    support, rows = None, []
    for j, row in mw.iterrows():
        if pd.notna(row.new_pivot_low):
            support = {"support_index": float(row.new_pivot_low), "support_pivot_date": row.last_date if j < 2 else mw.last_date.iloc[j - 2],
                       "support_confirm_date": row.last_date, "support_pivot_week": j - 2}
        rows.append(support.copy() if support else {"support_index": np.nan, "support_pivot_date": pd.NaT,
                                                   "support_confirm_date": pd.NaT, "support_pivot_week": np.nan})
    known = pd.DataFrame(rows)
    for column in known.columns:
        md[column] = md.completed_week_idx.map(known[column])
    return md, mw


def detect(h, d, w):
    long_ep, long_signals, long_steps = h.detect(d, w)
    md, mw = mirror(d, w)
    short_ep, short_signals, short_steps = h.detect(md, mw)
    signals = []
    for direction, frame in [(1, long_signals), (-1, short_signals)]:
        for s in frame.to_dict("records"):
            signals.append({"direction": direction, "signal_idx": int(s["signal_idx"]), "signal_date": s["signal_date"],
                            "event_id": ("L_" if direction == 1 else "S_") + s["event_id"], "setup_date": s["setup_date"],
                            "anchor_date": s["support_pivot_date"], "anchor_confirm_date": s["support_confirm_date"],
                            "stop_index": direction * s["stop_index"], "target_index": direction * s["target_index"],
                            "target_known_date": s["target_known_date"], "state": s["state"],
                            "signal_close": s["signal_close"], "relative_volume": s["relative_volume"],
                            "daily_macd_rising": s["daily_macd_rising"], "weekly_macd_rising": s["weekly_macd_rising"]})
    def annotate(frame, direction):
        result = frame.copy()
        result["direction"] = direction
        result["price_semantics"] = "原价格" if direction == 1 else "符号镜像，仅用于检测；信号已转回原价格"
        return result
    return (pd.DataFrame(signals).sort_values(["signal_idx", "direction"]).reset_index(drop=True),
            pd.concat([annotate(long_ep, 1), annotate(short_ep, -1)], ignore_index=True),
            pd.concat([annotate(long_steps, 1), annotate(short_steps, -1)], ignore_index=True))


def sequences(signals, d, div):
    rows = []
    for policy in POLICIES:
        sub = signals if policy == "BOTH" else signals.loc[signals.direction.eq(1 if policy == "LONG_ONLY" else -1)]
        conflict = set(sub.loc[sub.signal_idx.duplicated(False), "signal_idx"])
        busy = -1
        for sig in sub.to_dict("records"):
            sig["policy"] = policy
            if sig["signal_idx"] in conflict:
                rows.append({**sig, "status": "SAME_DAY_DIRECTION_CONFLICT"})
            elif sig["signal_idx"] < busy:
                rows.append({**sig, "status": "OVERLAPPING_ACTIVE_POINT"})
            elif not np.isfinite(sig["target_index"]):
                rows.append({**sig, "status": "NO_KNOWN_TARGET"})
            else:
                point = b.evaluate_point(sig, d, div)
                rows.append(point)
                if point["status"] == "COMPLETE":
                    busy = point["exit_idx"]
                elif point["status"] == "OPEN_RIGHT_CENSORED":
                    busy = len(d)
    return pd.DataFrame(rows)


def summarize(points, d):
    rows, annual, confidence, frequency = [], [], [], []
    valid = points.loc[points.status.eq("COMPLETE")]
    calendar = d.loc[d.date.ge(a.START), "date"].reset_index(drop=True)
    mapping = dict(zip(calendar, calendar.index))
    rng = np.random.default_rng(202610015)
    starts = rng.integers(0, len(calendar), size=(2000, math.ceil(len(calendar) / 20)))
    draw = ((starts[:, :, None] + np.arange(20)) % len(calendar)).reshape(2000, -1)[:, :len(calendar)]
    for policy in POLICIES:
        g = valid.loc[valid.policy.eq(policy)]
        for era, (start, end) in a.ERAS.items():
            part = g.loc[g.entry_date.between(start, end)]
            for direction in [0, 1, -1]:
                side = part if direction == 0 else part.loc[part.direction.eq(direction)]
                for cost, col in [("GROSS", "gross_return"), ("REFERENCE_FRICTION", "net_reference_return")]:
                    rows.append({"policy": policy, "era": era, "direction": direction, "cost": cost, **b.stats(side[col])})
        for year in range(2015, 2027):
            yy = g.loc[g.entry_date.dt.year.eq(year)]
            annual.append({"policy": policy, "year": year, "full_year": year < 2026, "cycles": len(yy),
                           "long": int(yy.direction.eq(1).sum()), "short": int(yy.direction.eq(-1).sum())})
        numerator, denominator = np.zeros(len(calendar)), np.zeros(len(calendar))
        for point in g.itertuples():
            idx = mapping[point.entry_date]
            numerator[idx] += point.net_reference_return
            denominator[idx] += 1
        nn, dd = numerator[draw].sum(axis=1), denominator[draw].sum(axis=1)
        mean = np.divide(nn, dd, out=np.full(2000, np.nan), where=dd > 0)
        conf = b.stats(g.net_reference_return)
        low, high = helper().wilson(conf["wins"], conf["cycles"])
        confidence.append({"policy": policy, "net_mean": g.net_reference_return.mean(), "mean_lower95": np.nanquantile(mean, .025),
                           "mean_upper95": np.nanquantile(mean, .975), "win_lower95": low, "win_upper95": high})
        earliest = int(d.index[d.date.ge(a.START)][0])
        previous = earliest - 1
        gaps = []
        for point in g.sort_values("entry_idx").itertuples():
            gaps.append(max(0, int(point.entry_idx) - previous - 1))
            previous = int(point.exit_idx)
        gaps.append(max(0, len(d) - 1 - previous))
        full_years = [x["cycles"] for x in annual if x["policy"] == policy and x["full_year"]]
        frequency.append({"policy": policy, "full_year_average": float(np.mean(full_years)), "zero_years": full_years.count(0),
                          "minimum_full_year": min(full_years), "maximum_full_year": max(full_years),
                          "maximum_flat_sessions_including_edges": max(gaps), "hard_frequency_gate": False})
    return pd.DataFrame(rows), pd.DataFrame(annual), pd.DataFrame(confidence), pd.DataFrame(frequency)


def run():
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert all(a.digest(OUT / p) == item["sha256"] for p, item in frozen["files"].items())
    if (OUT / "result.json").exists():
        raise RuntimeError("已有研究结果，不能覆盖。")
    h = helper()
    p = pd.read_parquet(OUT / "inputs/prices.parquet")
    div = pd.read_csv(OUT / "inputs/dividends.csv")
    for name in ["record_date", "ex_date", "payment_date"]:
        div[name] = pd.to_datetime(div[name])
    d, w = h.features(p, div)
    signals, episodes, steps = detect(h, d, w)
    points = sequences(signals, d, div)
    summary, annual, confidence, frequency = summarize(points, d)
    for name, df in [("周线锚点过程", episodes), ("逐日确认过程", steps), ("候选多空点位", signals),
                     ("点位逐笔与拒绝原因", points), ("胜率实际盈亏比", summary), ("逐年次数", annual),
                     ("不确定区间", confidence), ("频率与等待", frequency)]:
        save(name, df)
    old = pd.read_parquet(OUT / "inputs/old_points.parquet")
    old = old.loc[old.policy.eq("P0_CONFIRM") & old.cost.eq("STRESS") & old.sequential_eligible].sort_values("entry_date")
    long = points.loc[points.policy.eq("LONG_ONLY") & points.status.eq("COMPLETE")].sort_values("entry_date")
    assert old.entry_date.tolist() == long.entry_date.tolist()
    assert old.exit_date.tolist() == long.exit_date.tolist()
    error = float(np.max(np.abs(old.gross_return.to_numpy() - long.gross_return.to_numpy())))
    assert error < 1e-12
    for n in [1100, 1900, 2700, len(d) - 9]:
        dd, ww = h.features(p.iloc[:n], div.loc[div.ex_date.le(pd.Timestamp(p.date.iloc[n - 1]))])
        partial, _, _ = detect(h, dd, ww)
        expected = signals.loc[signals.signal_idx.lt(n)].reset_index(drop=True)
        pd.testing.assert_frame_equal(partial, expected)
    known = signals.loc[signals.target_index.notna()]
    assert (known.anchor_confirm_date < known.signal_date).all()
    assert (known.target_known_date < known.signal_date).all()
    for _, group in points.loc[points.status.eq("COMPLETE")].groupby("policy"):
        group = group.sort_values("entry_idx")
        assert (group.entry_idx.to_numpy()[1:] > group.exit_idx.to_numpy()[:-1]).all()
    verification = {"status": "PASS_ORIGINAL_LONG_IDENTITY_AND_NEW_SHORT_CLOCK", "prefixes": 4,
                    "original_long_complete_cycles": len(long), "original_long_gross_return_max_error": error,
                    "new_short_return_math": "已通过统一点位引擎的空头股息与费用单元测试；保存逐笔可重算。"}
    a.save_json(OUT / "verification.json", verification)
    a.save_json(OUT / "result.json", {"at": a.now(), "status": "BIDIRECTIONAL_WEEKLY_POINT_STUDY_COMPLETE",
                "scope": "用户只研究标的点位；频率软目标，无期权或完整账户收益",
                "independent_validation": False, "goal_achieved": False, "orders_authorized": False})
    print(summary.loc[summary.era.eq("ALL") & summary.direction.eq(0)].round(5).to_string(index=False), flush=True)
    print(confidence.round(5).to_string(index=False), flush=True)
    print(frequency.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="周支撑与周阻力的双向点位研究。")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        run()
