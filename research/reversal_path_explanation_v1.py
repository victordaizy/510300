"""沿用固定月末原点，区分途中回撤与期末反转；仅生成描述证据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_reversal_path_explanation_v1"
REV = ROOT / "reports/research/510300_reversal_monthly_diagnostic_v1"
POLICY = ROOT / "reports/research/510300_policy_information_clock_v1"
EPS = 1e-12
GREEN, RED, BLUE = "#17866e", "#cc5b4b", "#2875a6"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def freeze():
    if OUT.exists():
        raise FileExistsError("本轮目录已存在，不能覆盖协议或重选路径")
    for name in ["inputs", "results", "figures", "code", "evidence", "sources"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    pairs = [
        (REV / "inputs/market.parquet", "market.parquet"),
        (REV / "全部月末原点与二十日收益.csv", "monthly_origins.csv"),
        (REV / "summary.json", "prior_reversal_summary.json"),
        (REV / "protocol.json", "prior_reversal_protocol.json"),
        (POLICY / "results/跨通道政策链_完整事实与时钟.csv", "policy_nodes.csv"),
    ]
    for source, name in pairs:
        shutil.copy2(source, OUT / "inputs" / name)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    plan = {
        "study_id": "510300_REVERSAL_PATH_EXPLANATION_V1",
        "created_at": stamp(),
        "classification": "POST_HOC_DESCRIPTIVE_EXTENSION_ALREADY_OBSERVED_HISTORY",
        "user_request": "A股均值回归、反转多于动量；结合此前涨幅较大后按回撤退出与政策动态更新要求，用真实路径澄清。",
        "monthly_sample": "继承前轮全部170个月末原点及固定20交易日标签，不换期限、样本或涨跌幅档位。",
        "policy_sample": "继承手工验证的24个政策来源节点；同一复核日只计算一条价格路径。不是独立政策事件全集，不统计政策胜率。",
        "horizon": 20,
        "wealth": "第0点为次日实际开盘价归一为1；随后20个收盘点加持有期现金分红权益，入场除息日不享有分红，不再投资；无费用、无账户。",
        "pullback": "只要此前出现高于入场价的收盘现金权益高点、之后任一收盘低于该已出现高点，即记有途中回撤。浮点容差1e-12，不设经济阈值；一次很小波动也会计入，绝不当成可获利卖点。",
        "reversal": "此前20日回报与未来20日最终回报符号相反；原轮分类保持原样。",
        "statistics": ["全部路径与最终回报", "各此前方向的途中回撤/期末方向交叉计数", "收盘最大回撤分布"],
        "illustration": "此前上涨且途中回撤的期末仍涨、期末转跌两组中，各取日期最早的一条；机械选择用于说明形态，不作代表性推断。",
        "policy_illustration": "R01宣布降息与R02实施确认两节点，并展示所有其余节点的映射和路径；只解释可成交起点差异，不比较哪天交易最优。",
        "new_models": 0, "account_runs": 0, "exit_strategy_runs": 0, "bootstrap_draws": 0,
        "acceptance": "只检查路径、分红、时间与保存期末回报一致；不能以此放行任何入场或退出策略。",
        "inputs": [{"path": "inputs/" + name, "origin": str(source), "sha256": sha(source)} for source, name in pairs],
    }
    save(OUT / "protocol.json", plan)
    save(OUT / "freeze.json", {"created_at": stamp(), "before_new_path_statistics": True,
         "historical_prices_and_prior_results_already_seen": True,
         "protocol_sha256": sha(OUT / "protocol.json"), "code_sha256": sha(Path(__file__))})
    print("已冻结路径说明口径；这仍是事后描述，不是新预测检验。", flush=True)


def path_rows(market, entry_index, key, kind):
    selected = market.iloc[entry_index:entry_index + 20]
    if len(selected) != 20:
        raise ValueError("标签未成熟，不能改变持有期限")
    entry_open = float(selected.open.iloc[0])
    dividends = selected.dividend.to_numpy(float).copy()
    dividends[0] = 0.0
    entitlement = np.cumsum(dividends)
    wealth = np.r_[1.0, (selected.close.to_numpy(float) + entitlement) / entry_open]
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1
    rows = [{"path_id": key, "kind": kind, "step": 0,
             "date": str(selected.date.iloc[0].date()), "observation": "开盘基点",
             "price": entry_open, "cumulative_dividend": 0.0, "wealth": 1.0,
             "running_peak": 1.0, "drawdown": 0.0}]
    for j, row in enumerate(selected.itertuples(), start=1):
        rows.append({"path_id": key, "kind": kind, "step": j,
                     "date": str(row.date.date()), "observation": "收盘",
                     "price": float(row.close), "cumulative_dividend": float(entitlement[j - 1]),
                     "wealth": float(wealth[j]), "running_peak": float(peak[j]), "drawdown": float(drawdown[j])})
    pullback = bool(np.any((peak > 1 + EPS) & (wealth < peak - EPS)))
    fields = {"path_id": key, "entry_date": str(selected.date.iloc[0].date()),
              "exit_date": str(selected.date.iloc[-1].date()), "entry_open": entry_open,
              "final_return": float(wealth[-1] - 1), "peak_return": float(wealth.max() - 1),
              "trough_return": float(wealth.min() - 1), "max_close_drawdown": float(drawdown.min()),
              "peak_step": int(wealth.argmax()), "trough_step": int(wealth.argmin()),
              "max_drawdown_step": int(drawdown.argmin()), "positive_peak_then_pullback": pullback}
    return rows, fields


def run():
    if (OUT / "run_started.json").exists() or (OUT / "summary.json").exists():
        raise FileExistsError("已启动的路径计算不能再次运行")
    plan = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    if sha(OUT / "protocol.json") != frozen["protocol_sha256"] or sha(Path(__file__)) != frozen["code_sha256"]:
        raise ValueError("协议或计算代码不再等于冻结版本")
    for item in plan["inputs"]:
        if sha(OUT / item["path"]) != item["sha256"]:
            raise ValueError("继承输入身份变化")
    save(OUT / "run_started.json", {"started_at": stamp(), "new_path_statistics_only": True})
    m = pd.read_parquet(OUT / "inputs/market.parquet").sort_values("date").reset_index(drop=True)
    m["date"] = pd.to_datetime(m.date)
    monthly = pd.read_csv(OUT / "inputs/monthly_origins.csv")
    policy = pd.read_csv(OUT / "inputs/policy_nodes.csv")
    paths, records = [], []
    for row in monthly.itertuples():
        key = "M_" + row.date
        pp, fields = path_rows(m, int(row.origin_index) + 1, key, "固定月末")
        if abs(fields["final_return"] - row.future20) > 1e-12:
            raise ValueError("新路径期末值与前轮固定标签不一致")
        fields.update(origin_date=row.date, past20=float(row.past20), pattern=row.pattern)
        paths.extend(pp)
        records.append(fields)
    p_records, map_records = [], []
    for review, group in policy.groupby("review_close", sort=True):
        found = m.index[m.date == pd.Timestamp(review)].to_list()
        if len(found) != 1:
            raise ValueError("政策复核日没有唯一行情")
        key = "P_" + review
        pp, fields = path_rows(m, found[0] + 1, key, "政策来源节点")
        if set(group.research_execution_open) != {fields["entry_date"]}:
            raise ValueError("政策次开盘时钟与前轮不同")
        fields.update(review_close=review, node_ids="|".join(group.node_id),
                      source_node_count=len(group), titles="；".join(group.title))
        p_records.append(fields)
        paths.extend(pp)
        for row in group.itertuples():
            map_records.append({"node_id": row.node_id, "path_id": key, "title": row.title,
                                "channel": row.channel, "source_available_upper": row.source_available_upper,
                                "review_close": review, "entry_date": fields["entry_date"],
                                "source_url": row.source_url, "source_sha256": row.source_sha256,
                                "expectation_status": row.expectation_status})
    d = pd.DataFrame(records)
    p = pd.DataFrame(p_records)
    full = pd.DataFrame(paths)
    counts = []
    for label, subset in [("全部月末", d), ("此前上涨", d[d.past20 > 0]), ("此前下跌", d[d.past20 < 0])]:
        pb = subset.positive_peak_then_pullback
        counts.append({"group": label, "origins": len(subset), "positive_peak_then_pullback": int(pb.sum()),
                       "pullback_final_positive": int((pb & (subset.final_return > EPS)).sum()),
                       "pullback_final_negative": int((pb & (subset.final_return < -EPS)).sum()),
                       "pullback_final_zero": int((pb & (subset.final_return.abs() <= EPS)).sum()),
                       "no_positive_peak_pullback": int((~pb).sum()),
                       "median_max_close_drawdown": float(subset.max_close_drawdown.median())})
    cases = {}
    for name, cond in [("回撤后期末仍涨", d.final_return > EPS), ("回撤后期末转跌", d.final_return < -EPS)]:
        subset = d[(d.past20 > 0) & d.positive_peak_then_pullback & cond].sort_values("origin_date")
        cases[name] = None if subset.empty else subset.iloc[0].path_id
    summary = {"study_id": plan["study_id"], "status": "COMPLETED_DESCRIPTIVE_PATH_DISTINCTION_NO_PREDICTIVE_GATE",
               "monthly_origins": len(d), "monthly_path_points": int((full.kind == "固定月末").sum()),
               "policy_source_nodes": len(policy), "policy_unique_review_paths": len(p),
               "policy_path_points": int((full.kind == "政策来源节点").sum()),
               "counts": counts, "illustration_cases": cases,
               "new_models": 0, "account_runs": 0, "exit_strategy_runs": 0,
               "new_independent_forward_events": 0, "whole_macro_objective_complete": False,
               "causal_pressure_mechanism_established": False,
               "economic_pullback_threshold_established": False}
    for frame, name in [(d, "月末路径完整统计"), (p, "政策去重路径完整统计"), (full, "所有路径全部日频点"),
                        (pd.DataFrame(map_records), "全部政策节点到路径映射"), (pd.DataFrame(counts), "途中回撤与期末方向交叉表")]:
        frame.to_csv(OUT / "results" / (name + ".csv"), index=False, encoding="utf-8-sig", float_format="%.17g")
        frame.to_parquet(OUT / "results" / (name + ".parquet"), index=False)
    save(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def style():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False,
                         "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#83919a", "axes.labelcolor": "#334854"})


def write_figure(fig, name):
    for ext in ["png", "svg"]:
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def render():
    style()
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    d = pd.read_parquet(OUT / "results/月末路径完整统计.parquet")
    paths = pd.read_parquet(OUT / "results/所有路径全部日频点.parquet")
    policies = pd.read_parquet(OUT / "results/政策去重路径完整统计.parquet")
    maps = pd.read_parquet(OUT / "results/全部政策节点到路径映射.parquet")
    prior = json.loads((OUT / "inputs/prior_reversal_summary.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(14, 10.2), facecolor="#f6f8fa")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=.50, wspace=.28)
    ax = fig.add_subplot(grid[0, 0])
    counts = summary["counts"][1]
    vals = [counts["pullback_final_positive"], counts["pullback_final_negative"], counts["pullback_final_zero"], counts["no_positive_peak_pullback"]]
    labels = ["途中回撤，期末仍涨", "途中回撤，期末转跌", "途中回撤，期末持平", "未出现该类回撤"]
    bars = ax.barh(labels, vals, color=[GREEN, RED, "#83919a", BLUE], height=.55)
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(val + .6, bar.get_y() + bar.get_height() / 2, str(val), va="center")
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_xlabel("次数；分母为此前上涨的全部93个月末原点")
    ax.set_title("中途回撤，可以与期末上涨同时发生", loc="left", pad=13)
    ax = fig.add_subplot(grid[0, 1])
    up = d[d.past20 > 0]
    for label, cond, color in [("期末仍涨", up.final_return > EPS, GREEN), ("期末转跌", up.final_return < -EPS, RED), ("期末持平", up.final_return.abs() <= EPS, "#83919a")]:
        sub = up[cond]
        ax.scatter(sub.peak_return * 100, sub.final_return * 100, s=34, alpha=.78, color=color, label=f"{label} {len(sub)}")
    upper = max(up.peak_return.max(), .01) * 100 * 1.05
    ax.plot([0, upper], [0, upper], color="#95a2ad", lw=.9, ls="--")
    ax.axhline(0, color="#83919a", lw=.8)
    ax.set_xlabel("路径最高现金权益收益（含开盘基点，%）")
    ax.set_ylabel("第20日最终现金权益收益（%）")
    ax.set_title("低于途中高点，不等于最终亏损", loc="left", pad=13)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=.15)
    for col, (name, key) in enumerate(summary["illustration_cases"].items()):
        ax = fig.add_subplot(grid[1, col])
        if key is None:
            ax.text(.5, .5, "没有符合该形态的路径", ha="center", transform=ax.transAxes)
            continue
        path = paths[paths.path_id == key]
        row = d[d.path_id == key].iloc[0]
        color = GREEN if row.final_return > 0 else RED
        ax.plot(path.step, (path.wealth - 1) * 100, lw=2.2, marker="o", ms=3, color=color)
        ax.axhline(0, color="#83919a", lw=.8)
        for step, kind in [(row.peak_step, "高点"), (row.trough_step, "低点")]:
            point = path[path.step == step].iloc[0]
            ax.annotate(f"{kind} {(point.wealth - 1) * 100:+.2f}%", (step, (point.wealth - 1) * 100),
                        xytext=(8, 13 if kind == "高点" else -20), textcoords="offset points", fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": "#83919a"})
        ax.set_title(f"{name}｜{row.entry_date}—{row.exit_date}", loc="left", fontsize=10.5, pad=25)
        ax.text(0, 1.025, f"此前20日 {row.past20:+.2%} · 期末 {row.final_return:+.2%} · 途中最大收盘回撤 {row.max_close_drawdown:.2%}", transform=ax.transAxes, fontsize=9, color="#526676")
        ax.set_xlabel("入场开盘后的交易日；0=开盘基点")
        ax.set_ylabel("现金权益收益（%）")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xlim(-.5, 22)
        ax.margins(y=.3)
        ax.grid(alpha=.15)
    fig.suptitle("510300：看到上涨中的回撤，不能直接推出反转优势", x=.07, y=.98, ha="left", fontsize=19, weight="bold", color="#17394c")
    fig.text(.07, .927, f"固定170个月末，20交易日尺度｜此前上涨93次：随后上涨51、下跌41、持平1｜反转总占比 {prior['reversal_fraction']:.1%}", color="#526676")
    fig.text(.07, .028, "回撤口径：高于入场的收盘高点之后出现任意回落，微小波动也计入；没有设定有效卖点或手续费收益。\n两条实线路径按各类最早日期机械选取；高低点是事后标注。全部170条路径和3570个点随包保留。", fontsize=9.5, color="#526676", linespacing=1.6)
    fig.subplots_adjust(left=.16, right=.96, top=.84, bottom=.16)
    write_figure(fig, "510300_途中回撤与期末反转")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), sharey=True, facecolor="#f6f8fa")
    for ax, node in zip(axes, ["R01", "R02"]):
        key = maps[maps.node_id == node].path_id.iloc[0]
        row = policies[policies.path_id == key].iloc[0]
        path = paths[paths.path_id == key]
        ax.plot(path.step, (path.wealth - 1) * 100, color=BLUE, lw=2.2, marker="o", ms=3)
        ax.axhline(0, color="#83919a", lw=.8)
        ax.set_title(("9月24日宣布后，9月25日开盘起" if node == "R01" else "9月27日实施来源确认，10月8日开盘起"), loc="left", fontsize=11, pad=29)
        ax.text(0, 1.045, f"入场参考 {row.entry_open:.3f} 元 · 最终 {row.final_return:+.2%} · 路径高点 {row.peak_return:+.2%}", transform=ax.transAxes, fontsize=9.5, color="#526676")
        for step, kind in [(row.peak_step, "高点"), (row.trough_step, "低点")]:
            point = path[path.step == step].iloc[0]
            if step == 0:
                kind = "开盘基点"
            ax.annotate(f"{point.date[5:]} {kind}\n{(point.wealth - 1) * 100:+.2f}%", (step, (point.wealth - 1) * 100), xytext=(7, 10 if kind in ["高点", "开盘基点"] else -30),
                        textcoords="offset points", fontsize=9, arrowprops={"arrowstyle": "-", "color": "#83919a"})
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xlim(-.8, 22)
        ax.set_xlabel("分别从各自开盘起计算的20个交易日")
        ax.grid(alpha=.15)
        ax.margins(y=.25)
    axes[0].set_ylabel("每份现金权益收益（%）")
    fig.suptitle("同一轮政策，进入时点不同，持有路径也不同", x=.075, y=.965, ha="left", fontsize=19, weight="bold", color="#17394c")
    fig.text(.075, .887, "2024年9月政策链的两个来源节点；用于说明时钟，不是比较最优买点，也不能把实施再次称为同等意外。", color="#526676", fontsize=10)
    fig.text(.075, .035, "固定规则：来源可复核后首次收盘观察，下一开盘开始；实施资料只有日期上界，叠加节假日后落在10月8日。\n两窗口重叠并包含其他消息，不能当成两次独立政策试验；所有24个来源节点到路径的映射均保留。", color="#526676", fontsize=9.5, linespacing=1.6)
    fig.subplots_adjust(left=.075, right=.96, top=.70, bottom=.22, wspace=.12)
    write_figure(fig, "510300_政策宣布与实施后的不同路径")
    print("两幅路径说明图已由保存结果绘制，未新增模型或退出规则。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定二十日路径说明")
    parser.add_argument("action", choices=["freeze", "run", "render"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "render": render}[args.action]()
