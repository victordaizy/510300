"""固定已有点位，分解目标空间、方向路径、交易摩擦及退出延迟。"""
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
from research import directional_entry_timing_v1 as b
from research import weekly_anchor_bidirectional_v1 as c

OUT = ROOT / "reports/research/510300_point_payoff_bottleneck_v1"
NAMES = {**b.FAMILIES, **c.POLICIES}


def save_table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if OUT.exists():
        raise RuntimeError("已有诊断登记，不覆盖定义。")
    OUT.mkdir(parents=True)
    inputs = [(a.OUT / "results/features.parquet", "inputs/features.parquet"),
              (a.OUT / "inputs/dividends.csv", "inputs/dividends.csv"),
              (b.OUT / "results/点位逐笔与拒绝原因.parquet", "inputs/dense_points.parquet"),
              (c.OUT / "results/点位逐笔与拒绝原因.parquet", "inputs/weekly_points.parquet"),
              (b.CONTEXT / "active_goal_effective_requirements.json", "inputs/current_requirements.json"),
              (Path(__file__), "code/point_payoff_bottleneck_v1.py")]
    files = {}
    for src, name in inputs:
        dst = OUT / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files[name] = {"source": src.relative_to(ROOT).as_posix(), "sha256": a.digest(dst)}
    protocol = {
        "study": "510300_POINT_PAYOFF_BOTTLENECK_V1", "at": a.now(),
        "question": "为什么已有六组点位的胜率乘实际盈亏比均不大于1：信号方向、可用目标空间、兑现过程还是交易摩擦？",
        "authority": "只研究日线及已完成周线的标的多空点位，频率软目标；p×实际B>1为额外硬门槛。",
        "source_rules": "全部保存的完成点位保持原入场、方向、失效、目标、退出、顺序和份额。不新增回测、不改变交易规则。",
        "actual_recomputation": "从原入场/退出原价、成交价、份额、登记和除息日期独立重算毛方向及参考摩擦回报。",
        "three_fixed_comparisons": ["ACTUAL_SAVED：原结果", "GROSS_SAME_POINTS：原时点份额移除摩擦的归因", "DECISION_CLOSE_MARK：原退出决定日收盘作参考标记的归因"],
        "counterfactual_limit": "决定收盘标记不代表能在观察到收盘后按该收盘成交；三个比较不能选优升格为策略。",
        "known_geometry": "事前目标盈利、事前失效亏损、二者的净计划比；固定分箱<=1、(1,2]、>2仅用于归因，不据事后好坏新增过滤。",
        "risk_units": "实际净回报除以各点位入场时计划失效净亏损，得到R；与等名义收益口径并列，不因哪个较好而换验收分母。",
        "paths": "保留入场日起至实际退出前一交易日的所有收盘浮盈亏，并单列真实退出开盘。最大浮盈是事后路径描述，不假定能卖在最高点。",
        "loss_categories": "亏损且曾有扣摩擦正浮盈；亏损且从未有扣摩擦正浮盈；盈利。另看亏损点位是否曾达到事前1R。",
        "delay": "实际退出参考回报减原决定日收盘参考标记，正数代表后续开盘有利，负数代表不利；不是新的退出策略。",
        "sampling": "六组同源比较含重复日期/点位，分组各自统计，不能相加当独立样本。旧样本已反复研究，不称独立验证。",
        "prohibited": "不扫描持有天数、目标倍数、止损距离、分箱阈值或技术指标；不从失败子组反向交易。",
        "data_cutoff": "2026-09-16", "new_rule_accounts": 0, "new_trades": 0, "orders_authorized": False,
    }
    a.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": a.digest(OUT / "protocol.json")}
    a.save_json(OUT / "freeze.json", {"at": a.now(), "before_new_path_diagnostics": True, "files": files})
    print("点位收益来源诊断已固定：不改变原点位，只重算与解释。", flush=True)


def fill(price, action):
    scaled = price * (1 + action * .001) / .001
    return (math.ceil(scaled - 1e-9) if action > 0 else math.floor(scaled + 1e-9)) * .001


def dividend_at(div, entry_date, date, at_open=False):
    record = div.record_date.lt(date) if at_open else div.record_date.le(date)
    eligible = div.record_date.ge(entry_date) & record & div.ex_date.le(date)
    return float(div.loc[eligible, "cash_dividend_per_share"].sum())


def mark(point, raw, cash_dividend):
    exit_fill = fill(raw, -int(point["direction"]))
    qty = float(point["quantity"])
    fees = max(5., qty * point["entry_fill"] * .0004) + max(5., qty * exit_fill * .0004)
    net = (point["direction"] * qty * (exit_fill - point["entry_fill"] + cash_dividend) - fees) / (qty * point["entry_raw"])
    gross = point["direction"] * (raw - point["entry_raw"] + cash_dividend) / point["entry_raw"]
    return net, gross


def stats(values):
    r = np.asarray(values, float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {"n": 0, "wins": 0, "losses": 0, "p": np.nan, "b": np.nan, "product": np.nan, "mean": np.nan, "mean_win": np.nan, "mean_loss": np.nan, "profit_factor": np.nan}
    pos, neg = r[r > 0], r[r < 0]
    p = len(pos) / len(r)
    payoff = pos.mean() / -neg.mean() if len(pos) and len(neg) else np.nan
    return {"n": len(r), "wins": len(pos), "losses": len(neg), "p": p, "b": payoff, "product": p * payoff,
            "mean": float(r.mean()), "mean_win": float(pos.mean()) if len(pos) else np.nan,
            "mean_loss": float(neg.mean()) if len(neg) else np.nan,
            "profit_factor": float(pos.sum() / -neg.sum()) if len(neg) else np.nan}


def diagnose(d, div, points):
    rows, paths = [], []
    maximum = 0.
    for point in points.to_dict("records"):
        e, x, k = int(point["entry_idx"]), int(point["exit_idx"]), int(point["decision_idx"])
        assert d.date.iloc[e] == point["entry_date"] and d.date.iloc[x] == point["exit_date"]
        cash = dividend_at(div, point["entry_date"], point["exit_date"], True)
        actual_net, actual_gross = mark(point, float(d.open.iloc[x]), cash)
        maximum = max(maximum, abs(actual_net - point["net_reference_return"]), abs(actual_gross - point["gross_return"]))
        plan_stop_net, _ = mark(point, point["entry_stop_raw"], 0.)
        plan_target_net, _ = mark(point, point["entry_target_raw"], 0.)
        loss_unit = -plan_stop_net
        assert loss_unit > 0
        plan_rr = plan_target_net / loss_unit
        assert abs(plan_rr - point["planned_reference_rr"]) < 1e-10
        assert abs(actual_net / loss_unit - point["realized_r"]) < 1e-10
        local = []
        for j in range(e, x):
            dv = dividend_at(div, point["entry_date"], d.date.iloc[j])
            net, gross = mark(point, float(d.close.iloc[j]), dv)
            local.append(net)
            paths.append({"rule": point["rule"], "point_id": point["point_id"], "date": d.date.iloc[j],
                          "holding_session": j - e + 1, "reference_close_net": net, "reference_close_gross": gross,
                          "reference_close_r": net / loss_unit, "future_path_label_only": True})
        decision_dividend = dividend_at(div, point["entry_date"], d.date.iloc[k])
        decision_net, _ = mark(point, float(d.close.iloc[k]), decision_dividend)
        peak = max(local)
        trough = min(local)
        loss_category = "WIN" if actual_net > 0 else "LOSS_GAVE_BACK_POSITIVE_MARK" if peak > 0 else "LOSS_NO_POSITIVE_MARK"
        bin_label = "RR_LE_1" if plan_rr <= 1 else "RR_1_TO_2" if plan_rr <= 2 else "RR_GT_2"
        rows.append({**point, "actual_recomputed_net": actual_net, "actual_recomputed_gross": actual_gross,
                     "plan_loss": loss_unit, "plan_reward": plan_target_net, "plan_rr": plan_rr, "geometry_bin": bin_label,
                     "actual_r_recomputed": actual_net / loss_unit, "decision_close_net": decision_net,
                     "exit_delay_effect": actual_net - decision_net, "friction_effect": actual_gross - actual_net,
                     "future_best_close_net": peak, "future_worst_close_net": trough,
                     "future_best_close_r": peak / loss_unit, "future_worst_close_r": trough / loss_unit,
                     "future_positive_mark": peak > 0, "future_reached_1r": peak >= loss_unit,
                     "loss_category": loss_category, "exit_wait_sessions": x - k})
    assert maximum < 1e-12
    return pd.DataFrame(rows), pd.DataFrame(paths), maximum


def summarize(rows):
    comparisons, attribution, geometry = [], [], []
    for rule, g in rows.groupby("rule", sort=False):
        for scenario, column in [("ACTUAL_SAVED", "actual_recomputed_net"), ("GROSS_SAME_POINTS", "actual_recomputed_gross"),
                                 ("DECISION_CLOSE_MARK", "decision_close_net"), ("RISK_UNIT_DESCRIPTION", "actual_r_recomputed")]:
            comparisons.append({"rule": rule, "scenario": scenario, **stats(g[column])})
        losses = g.loc[g.actual_recomputed_net.lt(0)]
        wins = g.loc[g.actual_recomputed_net.gt(0)]
        attribution.append({"rule": rule, "n": len(g), "losses": len(losses),
                            "losses_ever_positive": int(losses.future_positive_mark.sum()),
                            "losses_never_positive": int((~losses.future_positive_mark).sum()),
                            "losses_ever_1r": int(losses.future_reached_1r.sum()),
                            "plan_rr_median": g.plan_rr.median(), "plan_rr_le1": int(g.plan_rr.le(1).sum()),
                            "plan_rr_gt2": int(g.plan_rr.gt(2).sum()),
                            "wins_reaching_planned_target": int(wins.exit_reason.eq("TARGET_CONFIRMED").sum()),
                            "mean_target_winner_return": wins.loc[wins.exit_reason.eq("TARGET_CONFIRMED"), "actual_recomputed_net"].mean(),
                            "mean_time_winner_return": wins.loc[wins.exit_reason.eq("TIME_20_SESSIONS"), "actual_recomputed_net"].mean(),
                            "mean_loss_return": losses.actual_recomputed_net.mean(),
                            "mean_friction": g.friction_effect.mean(), "mean_exit_delay_effect": g.exit_delay_effect.mean(),
                            "exit_delay_negative_count": int(g.exit_delay_effect.lt(0).sum()),
                            "maximum_extra_wait_sessions": int(g.exit_wait_sessions.max() - 1),
                            "median_best_close_r": g.future_best_close_r.median(),
                            "loss_median_best_close_r": losses.future_best_close_r.median()})
        for label in ["RR_LE_1", "RR_1_TO_2", "RR_GT_2"]:
            sub = g.loc[g.geometry_bin.eq(label)]
            geometry.append({"rule": rule, "geometry_bin": label, **stats(sub.actual_recomputed_net),
                             "post_result_diagnostic_only": True})
    return pd.DataFrame(comparisons), pd.DataFrame(attribution), pd.DataFrame(geometry)


def report(comparisons, attribution, geometry):
    comp = comparisons.pivot(index="rule", columns="scenario", values="product")
    lines = ["# 510300点位收益来源诊断", "",
             "本轮固定上一批全部入场、方向、目标、失效、退出与份额，只做归因。没有新策略回测，没有调整参数。当前门槛仍为扣参考摩擦后的p×实际盈亏比>1，交易次数为软目标。", "",
             "## 原结果、摩擦与退出标记", "",
             "| 规则 | 原实际乘积 | 原时点移除摩擦 | 原决定日收盘标记 | 按事前1R归一的实际乘积 |",
             "|---|---:|---:|---:|---:|"]
    for rule in NAMES:
        r = comp.loc[rule]
        lines.append(f"| {NAMES[rule]} | {r.ACTUAL_SAVED:.3f} | {r.GROSS_SAME_POINTS:.3f} | {r.DECISION_CLOSE_MARK:.3f} | {r.RISK_UNIT_DESCRIPTION:.3f} |")
    lines.extend(["", "移除摩擦仅是相同时点份额的收益归因；决定日收盘标记不代表观察到收盘后还能按该价成交；1R归一也不是另一条自动合格策略。三者都不能择优当作可执行结果。", "",
                  "## 亏损之前发生了什么", "",
                  "| 规则 | 完成点位 | 亏损 | 亏损前曾有正浮盈 | 亏损前曾到1R | 计划净比≤1的点位 | 入场计划净比中位数 | 平均退出延迟影响 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in attribution.itertuples():
        lines.append(f"| {NAMES[r.rule]} | {r.n} | {r.losses} | {r.losses_ever_positive} | {r.losses_ever_1r} | {r.plan_rr_le1} | {r.plan_rr_median:.3f} | {r.mean_exit_delay_effect:+.3%} |")
    lines.extend(["", "正浮盈只按入场后至真实退出前的日收盘计算，并已扣参考摩擦；不把盘中最高/最低先后顺序当成可成交依据。曾有浮盈不表示当时知道应当退出。1R是入场时已知的计划失效亏损。", "",
                  "## 事前空间分箱（仅归因）", "",
                  "固定分箱为计划净目标/计划净亏损<=1、(1,2]、>2。分箱不是结果后新增交易条件，尤其不能从几笔高乘积子组直接宣布成功。所有组，包括负结果与小样本，均在结果表保存。", "",
                  "| 规则 | 计划净比 | 笔数 | 胜率 | 实际盈亏比 | 乘积 | 平均参考净回报 |", "|---|---|---:|---:|---:|---:|---:|"])
    for r in geometry.itertuples():
        lines.append(f"| {NAMES[r.rule]} | {r.geometry_bin} | {r.n} | {r.p:.1%} | {r.b:.3f} | {r.product:.3f} | {r.mean:.2%} |")
    lines.extend(["", "## 文件与含义", "",
                  "逐笔诊断保存了当时已知的计划目标/失效空间，以及明确标记为未来路径的逐日浮盈亏。后者只能作为结果标签，不能回填为入场信号。实际收益从原价格、份额、双边费用和股息权益独立复算。", "",
                  "每个规则单独统计，六组含重复点位和同源行情，不能加总为独立样本。本轮没有计算期权收益、融券收益或完整账户收益。数据截至2026-09-16。"])
    (OUT / "研究结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert all(a.digest(OUT / name) == item["sha256"] for name, item in frozen["files"].items())
    if (OUT / "result.json").exists():
        raise RuntimeError("已有诊断结果不能覆盖。")
    dense = pd.read_parquet(OUT / "inputs/dense_points.parquet").rename(columns={"family": "rule"})
    weekly = pd.read_parquet(OUT / "inputs/weekly_points.parquet").rename(columns={"policy": "rule"})
    points = pd.concat([dense, weekly], ignore_index=True)
    points = points.loc[points.status.eq("COMPLETE")].copy()
    points["point_id"] = points.rule + "_" + points.entry_date.dt.strftime("%Y%m%d") + "_" + points.direction.astype(str)
    assert not points.point_id.duplicated().any()
    d = pd.read_parquet(OUT / "inputs/features.parquet")
    div = pd.read_csv(OUT / "inputs/dividends.csv")
    for column in ["record_date", "ex_date", "payment_date"]:
        div[column] = pd.to_datetime(div[column])
    rows, paths, error = diagnose(d, div, points)
    comparisons, attribution, geometry = summarize(rows)
    for name, frame in [("逐笔收益来源", rows), ("逐日浮盈亏标签", paths), ("固定归因比较", comparisons),
                        ("亏损路径与空间", attribution), ("事前空间分箱归因", geometry)]:
        save_table(name, frame)
    report(comparisons, attribution, geometry)
    a.save_json(OUT / "verification.json", {"status": "PASS_SAVED_POINT_RETURN_AND_PLANNED_RISK_RECOMPUTATION",
                "saved_complete_records_with_comparison_duplicates": len(points), "daily_path_rows": len(paths),
                "maximum_actual_return_error": error, "new_strategy_backtests": 0, "new_orders": 0})
    a.save_json(OUT / "result.json", {"at": a.now(), "status": "FIXED_POINTS_DIAGNOSTIC_COMPLETE_NO_STRATEGY_PROMOTION",
                "goal_achieved": False, "comparisons": comparisons.to_dict("records"),
                "diagnostic_only_geometry_product_above_one_rows": geometry.loc[geometry["product"].gt(1)].to_dict("records"),
                "warning": "事后子组和退出标记不能作为新策略通过证据。"})
    print(attribution.round(5).to_string(index=False), flush=True)
    print(comparisons.round(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定点位的盈亏结构诊断。")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        run()
