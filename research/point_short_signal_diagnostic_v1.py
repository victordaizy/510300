"""固定七笔既有空头点位，区分相对时段强弱、绝对价格方向和退出兑现。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common
from research import point_entry_exit_contribution_v1 as source
from research.adaptive_allocation_v1 import normalize_dividends
from research.historic_cycles_point_translation_v1 import statistics

OUT = ROOT / "reports/research/510300_point_short_signal_diagnostic_v1"
SOURCE = ROOT / "reports/research/510300_point_entry_exit_contribution_v1"
PERIODS = {"earlier_diagnostic": ("2014-12-31", "2019-12-30"), "evaluation": ("2019-12-31", "2026-08-13")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not OUT.exists(), "既有空头点位诊断已登记，不覆盖。")
    files = {}
    for src, relative in ((SOURCE / "inputs/source_daily.parquet", "inputs/daily.parquet"),
                          (SOURCE / "inputs/dividends.csv", "inputs/dividends.csv"),
                          (SOURCE / "results/连续规则全部点位.parquet", "inputs/points.parquet"),
                          (SOURCE / "results/连续规则指标.parquet", "inputs/metrics.parquet"),
                          (Path(__file__), "code/" + Path(__file__).name),
                          (Path(source.__file__), "code/" + Path(source.__file__).name)):
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        files[relative] = {"source": str(src.relative_to(ROOT)), "sha256": common.digest(target)}
    common.save_json(OUT / "protocol.json", {
        "study": "510300_POINT_SHORT_SIGNAL_DIAGNOSTIC_V1", "at": common.now(),
        "scope": "保持原SHORT_GUARDS七笔完整点位的进入退出、份额、保护和成本，只解释已有失败。",
        "question": "日内弱于隔夜的相对差是否对应绝对价格下跌，收益不足来自方向、费用还是已观察价格路径。",
        "known_results": "2015—2019三笔全亏；2020—2026年8月14日四笔中三胜，但pB约0.831。",
        "entry_components": "只读取进入前收盘已知的60日日内、隔夜与二者之和；原连续两日阈值不改。",
        "frequency": "统计原空头条件为真的日数及连续段，不把同一段的重复信号当多个独立交易。",
        "path": "从原进入收盘至原实际退出前收盘逐日计算固定份额净参考标记；最大浮盈是事后描述，不是可兑现退出。",
        "fixed_attribution": ["原完成点位净回报", "同一时点份额移除摩擦的毛回报", "原退出决定日收盘净参考标记"],
        "no_rescue": "不改阈值、不运行筛选后策略、不调整持有期限或止损、不镜像多头模型、不把最佳路径标记当策略。",
        "new_point_replays": 0, "new_model_fits": 0, "independent_validation": "NOT_ESTABLISHED",
        "orders_authorized": False, "goal_achieved": False,
    })
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("七笔既有空头点位的方向、路径和费用诊断已固定，原交易规则不变。", flush=True)


def phase(intraday, overnight):
    if intraday >= 0 and overnight >= 0:
        return "日内与隔夜均上涨，日内相对弱"
    if intraday < 0 and overnight < 0:
        return "日内与隔夜均下跌，日内更弱"
    return "日内下跌、隔夜上涨" if intraday < 0 else "日内上涨、隔夜下跌"


def run():
    require(not (OUT / "summary.json").exists(), "诊断已完成，不覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for relative, record in frozen["files"].items():
        require(common.digest(OUT / relative) == record["sha256"], "固定诊断输入改变。")
    dividends = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    data = source.make_features(pd.read_parquet(OUT / "inputs/daily.parquet"), dividends)
    data["intraday60"] = data.intraday_log.rolling(60).sum()
    data["overnight60"] = data.overnight_log.rolling(60).sum()
    data["total60"] = data.total_log.rolling(60).sum()
    require(np.allclose(data.intraday60+data.overnight60, data.total60, rtol=0, atol=1e-12, equal_nan=True), "日内隔夜和不对应绝对涨跌。")
    saved = pd.read_parquet(OUT / "inputs/points.parquet")
    points = saved.loc[saved.policy.eq("SHORT_GUARDS")].copy()
    require(len(points) == 7 and points.status.eq("COMPLETE").all() and points.direction.eq(-1).all(), "原七笔空头范围不同。")
    details, paths, summaries, frequency = [], [], [], []
    for p in points.itertuples():
        e, x, decision = int(p.entry_idx), int(p.exit_idx), int(p.first_exit_decision_idx)
        before = data.iloc[e-1]
        require(before.date == p.entry_origin and data.date.iloc[x] == p.exit_date and before.entry_short, "点位不对应原收盘条件。")
        actual = source.return_values(p.entry_raw, p.exit_raw, source.dividend_amount(dividends, p.entry_date, p.exit_date, True), -1, p.quantity)
        require(abs(actual["point_net_return"]-p.point_net_return) < 1e-12 and abs(actual["point_gross_return"]-p.point_gross_return) < 1e-12,
                "空头方向损益、费用或股息复算不同。")
        local = []
        for t in range(e, x):
            row = data.iloc[t]
            mark = source.return_values(p.entry_raw, float(row.close), source.dividend_amount(dividends, p.entry_date, row.date, False), -1, p.quantity)
            item = {"period": p.period, "entry_date": p.entry_date, "date": row.date, "origin_index": t,
                    "net_reference_mark": mark["point_net_return"], "gross_reference_mark": mark["point_gross_return"],
                    "future_path_description_only": True}
            local.append(item); paths.append(item)
        marks = pd.DataFrame(local)
        decision_mark = marks.loc[marks.origin_index.eq(decision), "net_reference_mark"].iloc[0]
        details.append({"period": p.period, "entry_date": p.entry_date, "exit_date": p.exit_date, "entry_origin": p.entry_origin,
                        "entry_score60": before.session_score60, "prior_intraday60_log": before.intraday60,
                        "prior_overnight60_log": before.overnight60, "prior_total60_return": np.expm1(before.total60),
                        "entry_phase": phase(before.intraday60, before.overnight60), "prior_price_direction": "DOWN" if before.total60 < 0 else "UP_OR_FLAT",
                        "net_return": p.point_net_return, "gross_return": p.point_gross_return,
                        "friction_drag": p.point_gross_return-p.point_net_return, "decision_close_mark": decision_mark,
                        "open_delay_impact": p.point_net_return-decision_mark, "maximum_close_net_mark": marks.net_reference_mark.max(),
                        "minimum_close_net_mark": marks.net_reference_mark.min(), "first_close_net_mark": marks.net_reference_mark.iloc[0],
                        "loss_after_positive_close": bool(p.point_net_return < 0 and marks.net_reference_mark.max() > 0),
                        "holding_sessions": p.holding_sessions, "exit_reasons": p.exit_reasons})
    details, paths = pd.DataFrame(details), pd.DataFrame(paths)
    old_metrics = pd.read_parquet(OUT / "inputs/metrics.parquet")
    for period, (start, end) in PERIODS.items():
        subset = details.loc[details.period.eq(period)]
        old = old_metrics.loc[old_metrics.policy.eq("SHORT_GUARDS") & old_metrics.period.eq(period)].iloc[0]
        base = statistics(subset.net_return)
        require(base["n"] == old["n"] and np.isclose(base["mean"], old["mean"]), "原指标没有复现。")
        for column, label in (("net_return", "ACTUAL_NET"), ("gross_return", "GROSS_SAME_POINTS"), ("decision_close_mark", "DECISION_CLOSE_MARK")):
            summaries.append({"period": period, "comparison": label, **statistics(subset[column])})
        local = data.loc[data.date.between(start, end)]
        active = local.entry_short
        clusters = active & ~active.shift(fill_value=False)
        frequency.append({"period": period, "observed_decision_days": len(local), "raw_short_signal_days": int(active.sum()),
                          "raw_short_signal_clusters": int(clusters.sum()), "raw_signal_days_with_positive_total60": int((active & local.total60.ge(0)).sum()),
                          "actual_entries": len(subset), "actual_entries_with_positive_total60": int(subset.prior_price_direction.eq("UP_OR_FLAT").sum()),
                          "losses": int(subset.net_return.lt(0).sum()), "losses_after_positive_close": int(subset.loss_after_positive_close.sum()),
                          "mean_friction_drag": float(subset.friction_drag.mean()), "mean_open_delay_impact": float(subset.open_delay_impact.mean())})
    summaries, frequency = pd.DataFrame(summaries), pd.DataFrame(frequency)
    for name, frame in (("七笔空头方向与路径", details), ("逐日收盘路径", paths), ("固定归因比较", summaries), ("原信号密度与方向", frequency)):
        save_table(name, frame)
    summary = {"study": "510300_POINT_SHORT_SIGNAL_DIAGNOSTIC_V1", "at": common.now(), "status": "FROZEN_SHORT_POINTS_ATTRIBUTED",
               "completed_points": len(details), "entries_after_nonnegative_total60": int(details.prior_price_direction.eq("UP_OR_FLAT").sum()),
               "losses_after_positive_close": int(details.loss_after_positive_close.sum()), "new_point_replays": 0, "new_models": 0,
               "periods": frequency.to_dict("records"), "short_candidate_ready": False, "goal_achieved": False, "orders_authorized": False}
    common.save_json(OUT / "summary.json", summary)
    lines = ["# 七笔既有空头点位的失败归因", "",
             "本轮只拆解已经失败的日内／隔夜相对强弱镜像空头，保持七笔原进出时点、份额与费用。没有重新运行筛选策略，也没有改阈值或退出。", "",
             "相对因子取60日日内收益减隔夜收益，绝对价格方向取两者之和。差值为负不要求总和为负；以下均在进入前一个收盘观察，未来持仓路径单列。", "",
             "| 进入 | 退出 | 入场前60日总回报 | 原净收益 | 最大收盘净浮盈 | 决定收盘到退出开盘影响 | 退出原因 |", "|---|---|---:|---:|---:|---:|---|"]
    for r in details.itertuples():
        lines.append(f"| {r.entry_date.date()} | {r.exit_date.date()} | {r.prior_total60_return:.2%} | {r.net_return:.2%} | {r.maximum_close_net_mark:.2%} | {r.open_delay_impact:+.2%} | {r.exit_reasons} |")
    lines += ["", "最大收盘净浮盈只是已经发生的价格路径，不能假定当时知道最佳退出点。决定收盘标记也不能在观察到收盘以后按该价成交。", "",
              "| 时期 | 比较 | 笔数 | 胜率 | 实际B | pB | 平均净／毛回报 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summaries.itertuples():
        b, pb = (f"{r.b:.3f}", f"{r.product:.3f}") if np.isfinite(r.b) else ("未定义", "未定义")
        lines.append(f"| {r.period} | {r.comparison} | {r.n} | {r.p:.1%} | {b} | {pb} | {r.mean:.2%} |")
    lines += ["", "以上移除摩擦或收盘标记仅作归因，不能选较好的一项升格为策略。早期若全亏，则没有平均盈利，实际B和pB保留未定义。", "",
              "| 时期 | 条件为真的日数 | 连续条件段数 | 其中60日总回报非负日数 | 实际进入 | 非负总回报时进入 |", "|---|---:|---:|---:|---:|---:|"]
    for r in frequency.itertuples():
        lines.append(f"| {r.period} | {r.raw_short_signal_days} | {r.raw_short_signal_clusters} | {r.raw_signal_days_with_positive_total60} | {r.actual_entries} | {r.actual_entries_with_positive_total60} |")
    lines += ["", "同一条件段内重复成立的日线不是多个独立入场机会。当前研究没有据这些分类添加新的趋势过滤，也没有把亏损点位反向交易。后续空头假说应首先解释绝对价格下降及失败对照，而不是直接假设多头因子的符号翻转就能产生空头优势。", ""]
    (OUT / "研究结论.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定既有空头点位的方向和收益归因。")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
