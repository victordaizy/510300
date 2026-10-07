"""从冻结结果生成LPR股债图表、研究结论及审阅包；不再拟合。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_lpr_joint_response_20d_v1"
PRIOR = ROOT / "reports/research/510300_reversal_path_explanation_v1"
PARENT = ROOT / "deliverables/510300_回撤反转与政策路径说明_V1_GPT审阅_20260922.zip"
PARENT_SHA = "9851f915c32449dafcc864aceaeb5ffe4617e1c25951892fc3d5458b59760c20"
ARCHIVE = ROOT / "deliverables/510300_LPR变更与股债联合反应_V1_GPT审阅_20260922.zip"
ATTACHMENT = Path(r"E:\CodexData\.codex\attachments\9985585b-6e77-41bf-b927-0e02209a9c57\pasted-text-1.txt")
COLORS = {"P0": "#627184", "P_LPR": "#168986", "P_JOINT": "#D78337",
          "P_REV": "#517CAD", "P_MOM": "#AB8294", "MEAN": "#9AA19C", "ZERO": "#465455"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write(name, value):
    (STUDY / name).write_text(value, encoding="utf-8")


def table(name):
    return pd.read_parquet(STUDY / "results" / (name + ".parquet"))


def save_chart_data(name, data):
    data.to_csv(STUDY / "figures" / (name + ".csv"), index=False, encoding="utf-8-sig", float_format="%.17g")
    data.to_parquet(STUDY / "figures" / (name + ".parquet"), index=False)


def finish(fig, name, footer):
    fig.text(.06, .025, footer, fontsize=10, color="#586574", va="bottom")
    for ext in ("png", "svg"):
        fig.savefig(STUDY / "figures" / (name + "." + ext), dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_saved():
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#CFD7DC", "axes.labelcolor": "#263C48",
                         "xtick.color": "#50606C", "ytick.color": "#50606C", "svg.fonttype": "none"})
    summary = read_json(STUDY / "results/summary.json")
    lpr = pd.read_parquet(STUDY / "inputs/lpr.parquet")
    daily = table("全日频股债LPR状态")
    levels = lpr[["month", "lpr_1y_percent", "lpr_5y_percent"]].rename(columns={"month": "lpr_month"})
    daily = daily.merge(levels, on="lpr_month", how="left", validate="many_to_one")
    daily["date"] = pd.to_datetime(daily.date)
    save_chart_data("全部日频图形数据_含报价水平", daily)
    shown = daily[(daily.date >= "2019-08-20") & (daily.date <= "2026-07-31")]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [1.45, 1, 1]})
    fig.subplots_adjust(left=.085, right=.95, top=.865, bottom=.13, hspace=.17)
    fig.suptitle("510300、LPR与国债收益率：看同一时间轴上的真实走势", x=.06, ha="left", y=.97, fontsize=21, weight="bold")
    fig.text(.06, .925, "2019年8月—2026年7月｜84个月LPR原文｜灰色区间为45个完整评价月；走势对照不等于因果归因", fontsize=11, color="#536777")
    axes[0].plot(shown.date, shown.close, color="#1D536F", lw=1.3, label="510300收盘价")
    axes[0].set_ylabel("510300价格（元）")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].step(shown.date, shown.lpr_1y_percent, where="post", color="#168986", lw=1.8, label="1年期 LPR")
    axes[1].step(shown.date, shown.lpr_5y_percent, where="post", color="#D78337", lw=1.8, label="5年期以上 LPR")
    axes[1].set_ylabel("公布报价（%）")
    axes[1].legend(loc="upper right", ncol=2, frameon=False)
    axes[2].plot(shown.date, shown.yield10, color="#826392", lw=1.2, label="中债10年国债收益率曲线")
    axes[2].set_ylabel("日终曲线（%）")
    axes[2].legend(loc="upper right", frameon=False)
    for ax in axes:
        ax.axvspan(pd.Timestamp("2022-11-01"), pd.Timestamp("2026-07-31"), color="#657C89", alpha=.065, zorder=0)
        ax.grid(axis="y", alpha=.16)
        ax.margins(x=0)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    finish(fig, "510300_LPR与国债收益率_完整走势",
           "LPR按当次公布后的首个完整股票交易日进入；9:30同时开盘的历史月份顺延，保留9:15/9:00原文时钟。\n国债曲线是日终估计收益率，次日决策才使用；不是可交易债券回报。上图价格未含分红，研究标签包含应得现金分红。")

    metrics = pd.DataFrame(summary["metrics"])
    comparisons = pd.DataFrame(summary["comparisons"])
    annual = table("逐年损失贡献")
    save_chart_data("图形_全部基准与损失", metrics)
    save_chart_data("图形_全部增量及固定区间", comparisons)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.09, right=.955, top=.85, bottom=.13, hspace=.47, wspace=.42)
    fig.suptitle("预测改变了，但可靠增量尚未建立", x=.05, ha="left", y=.97, fontsize=22, weight="bold")
    fig.text(.05, .925, "2022年11月—2026年7月｜45个月等权、231次复核｜所有固定候选与基准完整报告", fontsize=11, color="#536777")
    keys = ["P0", "P_LPR", "P_JOINT", "P_REV", "P_MOM", "MEAN", "ZERO"]
    labels = ["价格＋信息年龄", "加入 LPR 变化", "再加入股债联合", "单一反转约束", "单一动量约束", "当时成熟训练均值", "零收益预测"]
    values = [float(metrics.set_index("model").loc[k, "rmse_pp"]) for k in keys]
    ax = axes[0, 0]
    ax.barh(np.arange(7), values, color=[COLORS[k] for k in keys], height=.62)
    ax.set_yticks(np.arange(7), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 6)
    for i, value in enumerate(values):
        ax.text(value + .055, i, f"{value:.4f}", va="center", fontsize=10)
    ax.set_xlabel("未来20日收益 RMSE（百分点，越低越好）")
    ax.set_title("收益误差接近简单基准", loc="left", fontsize=14, pad=13)
    direct = [("P_LPR", "P0"), ("P_JOINT", "P_LPR")]
    for ax, pairs, scale, xlabel, title in [
        (axes[0, 1], direct, 10000, "MSE减少量（百分点²，右侧为改善）", "收益：直接增量区间均跨过零"),
        (axes[1, 0], [("R_LPR", "R0"), ("R_JOINT", "R_LPR")], 1,
         "QLIKE减少量（右侧为改善）", "风险：LPR小幅改善，联合反应未继续改善")]:
        ax.axvline(0, color="#74848D", lw=1, ls="--")
        for i, (candidate, reference) in enumerate(pairs):
            row = comparisons[(comparisons.candidate == candidate) & (comparisons.reference == reference)].iloc[0]
            low, mid, high = [float(row[v]) * scale for v in ("one_sided_98_75pct_lower", "mean_improvement", "central_97_5pct_upper")]
            color = "#168986" if i == 0 else "#D78337"
            ax.plot([low, high], [i, i], color=color, lw=3, solid_capstyle="round")
            ax.scatter([mid], [i], color=color, s=65, zorder=3)
            ax.text(mid, i - .18, f"{mid:+.4f}", ha="center", fontsize=10)
        ax.set_yticks([0, 1], ["加入 LPR", "再加入股债联合"])
        ax.set_ylim(1.55, -.65)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=13, pad=13)
        ax.grid(axis="x", alpha=.13)
    ax = axes[1, 1]
    annual_values = annual.P_LPR_improvement.to_numpy() * 10000
    ax.bar(annual.year.astype(str), annual_values, color=["#168986" if x > 0 else "#C47867" for x in annual_values], width=.55)
    ax.axhline(0, color="#74848D", lw=.8)
    for i, (value, months) in enumerate(zip(annual_values, annual.months)):
        ax.text(i, value + (.035 if value >= 0 else -.035), f"{value:+.3f}\n{months}个月", ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    ax.set_ylim(-.5, 1.25)
    ax.set_ylabel("LPR相对共同基准的年内平均 MSE 减少（百分点²）")
    ax.set_title("年度贡献诊断：改善集中在2024年", loc="left", fontsize=13, pad=13)
    finish(fig, "510300_LPR股债_全部对照与增量",
           "区间为冻结的六个月区块、10,000次保存抽样所得97.5%双侧区间；门槛使用98.75%单侧下界。\n四个候选分别验收，当前全部未通过。年度贡献只解释已保存结果；未删年、改样本、重新拟合或运行账户。")

    updates = table("额外LPR复核内容更新诊断")
    counts = []
    for row in summary["extra_lpr_review_diagnostic"]:
        gain = updates["improvement_" + row["model"]]
        counts.append({"model": row["model"], "better": int((gain > 1e-12).sum()), "same": int((gain.abs() <= 1e-12).sum()), "worse": int((gain < -1e-12).sum())})
    counts = pd.DataFrame(counts)
    save_chart_data("图形_公告更新全部次数", counts)
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.subplots_adjust(left=.19, right=.95, top=.75, bottom=.20)
    fig.suptitle("39次额外公告复核：11次内容变化，28次预测不变", x=.06, ha="left", y=.96, fontsize=20, weight="bold")
    fig.text(.06, .88, "固定同一模型、当前价格、国债变化和信息年龄，仅替换上一周仍沿用的两项LPR变化值。", fontsize=11, color="#536777")
    pos = np.arange(4)
    left = np.zeros(4)
    for key, label, color in [("better", "误差减小", "#168986"), ("same", "误差相同", "#CAD4D8"), ("worse", "误差扩大", "#C47867")]:
        values = counts[key].to_numpy()
        ax.barh(pos, values, left=left, height=.58, color=color, label=label)
        for i, (start, value) in enumerate(zip(left, values)):
            if value:
                ax.text(start + value / 2, i, str(value), ha="center", va="center", color="white" if key != "same" else "#324856", weight="bold")
        left += values
    ax.set_yticks(pos, ["收益：加入 LPR", "收益：再加股债", "风险：加入 LPR", "风险：再加股债"])
    ax.invert_yaxis()
    ax.set_xticks([0, 10, 20, 30, 39])
    ax.set_xlim(0, 39)
    ax.set_xlabel("非周度LPR公告复核次数")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), frameon=False, ncol=3)
    finish(fig, "510300_LPR_公告内容更新对照",
           "这是一项预先固定的内容替换诊断，不是完整周度决策与公告触发账户的比较，也不是交易胜率。\n11次内容变化包括新降息及随后未变报价令上一笔变化归零；39次有重叠的20日标签，不能当独立事件推断有效。")


def prepare():
    if ARCHIVE.exists():
        raise FileExistsError("本轮最终归档已存在，不能覆盖")
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("上一轮归档身份不符")
    (STUDY / "history").mkdir(exist_ok=True)
    shutil.copy2(PARENT, STUDY / "history" / PARENT.name)
    save_json(STUDY / "evidence/前阶段归档身份.json", {"checked_at": now(), "name": PARENT.name, "sha256": PARENT_SHA})
    for name in ["verify_lpr_joint_response_20d_v1.py", "deliver_lpr_joint_response_20d_v1.py"]:
        shutil.copy2(ROOT / "research" / name, STUDY / "code" / name)
    for name in ["前轮_510300_M1M2与宏观政策走势.png", "前轮_510300二十日反转与延续.png", "510300_途中回撤与期末反转.png"]:
        relative = "figures/" + name
        with zipfile.ZipFile(PARENT) as archive:
            content = archive.read(relative)
        (STUDY / relative).write_bytes(content)
    shutil.copy2(ATTACHMENT, STUDY / "用户初始目标原文.txt")
    shutil.copy2(ROOT / "reports/research/510300_macro_research_program_status_v1.json", STUDY / "evidence/本轮登记前宏观主线状态.json")
    summary = read_json(STUDY / "results/summary.json")
    metrics = {row["model"]: row for row in summary["metrics"]}
    monthly = table("主要月度损失")
    gain = monthly.loss_P0 - monthly.loss_P_LPR
    yr2024 = monthly.month.str.startswith("2024")
    diagnostic = {
        "scope": "SAVED_RESULT_EXPLANATION_ONLY_NO_NEW_GATE",
        "mse_reduction_percent_LPR_vs_P0": (1 - metrics["P_LPR"]["mean_loss"] / metrics["P0"]["mean_loss"]) * 100,
        "mse_increase_percent_JOINT_vs_LPR": (metrics["P_JOINT"]["mean_loss"] / metrics["P_LPR"]["mean_loss"] - 1) * 100,
        "lpr_2024_share_of_total_improvement_percent": float(gain[yr2024].sum() / gain.sum() * 100),
        "other_years_mean_improvement": float(gain[~yr2024].mean()),
        "new_model_fits": 0, "new_random_samples": 0, "candidate_decisions_changed": False,
    }
    save_json(STUDY / "evidence/保存结果解释性汇总.json", diagnostic)
    plot_saved()
    write("00_README_FIRST.md", """# 从这里阅读

本轮完成84个月LPR原文与日终国债曲线时钟整理，并完成固定20日、周度加公告复核的四项信息门检验。四项均未建立可靠增量，账户未运行，整个宏观计划仍未完成。

1. 先读《研究结论.md》，看三张新图及完整简单基准。
2. 《用户要求与当前范围.md》保留用户后续纠偏；《用户初始目标原文.txt》是初始附件原文。
3. `source_plan.json` → 原文与请求回执 → `protocol.json` → `freeze.json` → `run_started.json` → `results/summary.json` 是执行顺序。
4. `results`保留421个全部复核原点、231个逐期预测、45个月损失、368个训练记录、39次公告内容对照以及所有失败比较。CSV和Parquet均保留。
5. `figures`保留全部日频绘图数据、PNG和SVG；前轮M1/M2及反转图原样继承，不能当本轮新样本。
6. `code/verify_lpr_joint_response_20d_v1.py --study-dir 解压目录`从保存输入核对原文、信息时钟、成熟训练月、保存参数和误差区间；不重新拟合。
7. `history`含先校验哈希的前轮ZIP；内部旧状态是历史快照。新包以根目录FILE_INDEX.csv为成员身份索引。
8. 可复制《GPT审阅提问.md》。本包未上传，也未获得外部审阅。
""")
    write("用户要求与当前范围.md", """# 用户范围与本轮对应

用户初始完整目标见同目录txt；以下是之后对话的要求摘录与整理，不冒充全部对话逐字记录。

- 需要510300与M1、M2同比增速剪刀差的真实走势对比图；不仅看差值，还要看实际相对市场事前预期。
- 同时研究其他宏观政策影响，补齐信息再得出结论；允许互联网和微信公众号查询，线索须区分原始出处及历史时点。
- 主线是宏观传导、可观察市场状态和持续更新；既有货币公告五日实验失败，只拒绝该具体实现。资金资料缺口不应阻断独立通道。
- 上涨较多后，再结合回撤决定退出；用户提出“A股是一个均值回归，反转多于动量的市场”。需与ETF单一期限、回撤、动量和可交易利润分别对应。
- 20万元为主，2万元为费用对照；完整账户采用242日、零现金及无风险收益率等既有口径。

本轮只检验新的一条贷款定价与日频股债联合信息通道，加入固定反转/动量及均值/零收益基准；不预设反转成立。LPR实际变化不是市场意外，84个月共识和意外字段均缺失。并非替代整个政策、预期和资金研究。

前轮M1/M2图及固定反转诊断随包保留。本轮四候选均未通过自身预测门，因此20万元/2万元账户、连续账户更新、固定入场后的涨后回撤退出保持NOT_RUN，不填写虚构年化或夏普。
""")
    rows = []
    labels = {"P0": "共同基准：价格＋公告信息年龄", "P_LPR": "共同基准＋两期限LPR变化", "P_JOINT": "再加10年收益率变化与股债交互", "P_REV": "固定反转约束", "P_MOM": "固定动量约束", "MEAN": "当时成熟训练均值", "ZERO": "恒定零收益"}
    for key in ["P0", "P_LPR", "P_JOINT", "P_REV", "P_MOM", "MEAN", "ZERO"]:
        rows.append(f"|{labels[key]}|{metrics[key]['rmse_pp']:.6f}|{metrics[key]['positive_forecasts']}|")
    result_table = "\n".join(rows)
    write("研究结论.md", f"""# LPR变更、股债联合反应与动态持有判断

**LPR实际变化给共同基准带来约{diagnostic['mse_reduction_percent_LPR_vs_P0']:.3f}%的MSE下降，但没有稳定胜过简单基准；再加入日频股债联合项未带来进一步改善。收益与下行风险四项候选均未通过固定门槛。**这轮补齐了一条完整月度政策报价系列并检验其持续更新价值，尚不能形成当前买卖判断。反转约束在本段历史中略好于动量约束，却与训练均值、零收益几乎相同；不能据此宣布510300有可交易的普遍均值回归规律。

## 实际走势：利率下行不能直接换成股票上涨

![三条时间轴](figures/510300_LPR与国债收益率_完整走势.png)

图中保留该范围全部日频点，时间轴一致，不倒置利率坐标或移动曲线找吻合。完整3476日及两个报价水平的绘图数据另存。LPR两期限分开表示；没有用差值吞掉方向相同或来源不同的变化。图本身是描述，没有建立政策因果关系。

## 这轮具体检验了什么

固定问题是：贷款定价变化及已经发生的股债联合反应，能否改善从下一个可执行开盘起20个交易日的收益均值或下行风险预测。周度最后股票交易日以及LPR公布后的首个完整股票交易日触发复核，重合日合并。观察到的当日反应只作输入，不计入随后收益。

价格及信息年龄共同基准使用过去20日回报、当日回报、20日波动（风险模型用下行方差）和公告年龄。P_LPR/R_LPR只增加最近公布的1年、5年期以上LPR各自月度变化，下一次未变报价会把变化更新为零。P_JOINT/R_JOINT再增加同一股票交易区间的10年国债曲线变化以及股价日回报乘该变化的一个交互项。未预设四种股债符号必然对应某个经济状态。

扩展训练只使用全部标签已经成熟的完整自然月；同一月内各复核点等权，各月总权重相等。最少36个训练月，固定ridge惩罚1，全部标准化仅用当前成熟训练集，模型集合不变时复用系数。421个计划原点全部具备共同输入；45个完整评价月为2022-11至2026-07，共231次预测、46个训练集合、368个保存模型。20日标签重叠，231次不被当作231次独立宏观观察。

收益标签从下一股票日开盘开始，包含持有期应得现金分红，不享有入场除息日分红且不再投资；风险标签是这段权益路径20个日收益负部平方的均值。本轮不是账户净值，不含费用后绩效。

## 收益：小幅点估计变化尚不可靠

|固定方法|20日收益RMSE（百分点，越低越好）|231次中正预测次数|
|---|---:|---:|
{result_table}

P_LPR对共同基准的平均MSE改善为0.197490百分点²，冻结区块区间为[-0.184790, 0.683228]；再加股债的直接改善为-0.013241百分点²，区间[-0.166352, 0.172253]。这些是97.5%双侧区间，验收使用相同左端作为98.75%单侧下界。零被区间覆盖，且P_LPR、P_JOINT的点误差仍高于训练均值、零收益、反转基准，均未通过。

![所有候选和基准](figures/510300_LPR股债_全部对照与增量.png)

加入LPR与再加股债后，分别有37、46次预测正负号相对共同基准改变。**方向改变次数不是正确次数，也不是达到实际往返成本门槛的成交次数。**本次没有用方向改变代替误差或账户验收。

保存结果的逐年贡献诊断显示，2024年贡献了LPR相对共同基准总改善的{diagnostic['lpr_2024_share_of_total_improvement_percent']:.2f}%；其余年份合计抵消了一部分。这里仅解释保存结果，没有删掉任何年份或据此重新选择样本。两个半段对共同基准均有微小点改善，也不能替代完整不确定性和简单基准对照。

## 风险：不能用收益模型结论代替风险验收

QLIKE越低越好；共同风险基准R0为{metrics['R0']['mean_loss']:.6f}，加入LPR为{metrics['R_LPR']['mean_loss']:.6f}，再加股债为{metrics['R_JOINT']['mean_loss']:.6f}。LPR相对R0改善0.005639，区间[-0.001163, 0.015714]，未建立稳定直接增量。它相对历史平均下行方差和当前过去20日下行方差两项简单参考通过了相应比较，但仍未通过与共同风险基准的关键比较，不能把模型其余部分的贡献归给LPR。

股债项相对LPR风险模型的改善为-0.009330，区间[-0.037521, 0.001261]，前后半段方向也不一致。两类风险候选均未进入账户。区间来自保存的六个月连续区块、10,000次抽样，四项候选作局部多重检验控制，不能校正仓库全部上游研究选择，也不是独立前向验证。

## 持有期间的新消息是否改变判断

![公告内容更新](figures/510300_LPR_公告内容更新对照.png)

39次非周度LPR复核中，11次新旧报价变化内容不同并改变预测，28次预测相同；11次包括降息后下一月未变报价使变化归零，并非11次新降息。两个收益候选都为7次误差减小、4次扩大；两风险候选分别8/3、7/4。

这里严格保持同一当前模型、价格、债券、年龄和未来目标，只替换沿用的旧LPR变化内容。该对照能够隔离这一小项内容变化，却**不是完整旧周度决策与新公告决策的连续账户比较**。样本少且标签重叠，不计算独立事件成功率；不能用这项次级诊断挽救主要门失败。

## 公布时间、真实预期和来源边界

84个月从2019-08至2026-07连续完整，包括未变报价。直接历史接口一次TLS EOF失败已保存，随后使用央行正式中文目录及84份静态原文完成来源，不依赖只有降息日的手工样本。[央行LPR原文目录](https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/3876551/index.html)

逐月原文时间覆盖历史9:30、9:15和9:00；研究使用开盘严格晚于公告上界的第一个完整交易日，9:30恰与开盘相同则顺延。不能把现行9:00说明回填全部历史。[2019年8月原文](https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/3876551/3877436/index.html)、[交易中心现行说明](https://www.chinamoney.com.cn/chinese/bklpr/)

中债官网说明曲线工作日日终17:30发布；本轮再以观测日23:59:59作保守可用上界，下一股票交易日09:00决策、09:30开始收益。曲线来自报价、成交等信息的估计，不能当成可交易债券收益或单一消息冲击。保存的2509日曲线与早期衍生账簿在2501个公共日期全部一致，旧09:30衍生时钟原样保留，只用于数值核对。[中债曲线说明](https://yield.chinabond.com.cn/cbweb-cbrc-web/cbrc/showCbrc)

**所有LPR事前共识和政策意外字段都为空。**利率下调是实际变化，不能说成超预期宽松；今天取得的历史网页也不等于认证了当年第一版本。当前日曲线值的一致性不能替代历史首版认证。独立前向事件为0。

文献以高频股价与利率共动区分政策冲击和信息冲击，为本轮研究提供动机；本轮使用日收益与10年期日终曲线，覆盖多种同时消息，并未复现高频识别或建立因果解释。[Jarociński与Karadi，2020，论文及摘要](https://www.aeaweb.org/articles?id=10.1257/mac.20180090)

作者公开的446条政策日期及Target/Path因素只用于前期方法去重线索，没有进入本轮输入。该作者日历未列2024-09-24首次综合宣布，不能自动作为所有首次政策发布全集；这反映公告范围差异，不能简单宣布其论文数据有误。新LPR系列只是自己的完整月度母集，仍不能代表全部货币、财政、地产或外部政策。

## 对“反转多于动量”和“涨多后回撤卖”的回答

此前固定月末诊断覆盖170个原点：过去20日和随后次开盘20日，81次反转、88次延续、1次零；非零原点中反转47.93%，六个月区块95%区间41.18%—55.03%。这些是前轮保存统计继承，本次没有再次抽样。不能由这一结果概括全A股或所有持有期限，更没有检验价格是否围绕某个固定合理价值平稳回归。

![前轮固定反转诊断](figures/前轮_510300二十日反转与延续.png)

93次此前上涨原点中，84次后续有正浮盈高点后的回撤，但其中51次最终仍涨、32次最终跌、1次持平。任何微小回落都计入该路径定义，所以常见回撤不能自动成为有利润的退出信号。反转、途中回撤与费用后卖出优势是不同检验。

用户希望上涨较大后根据回撤退出，本轮保留既有单一执行设定：入场前20日日波动σ，浮盈达2σ启动，自持有收盘高点回撤σ后次开盘退出，最迟20日结束。但尚未运行或确认阈值有效，不能凭事后高点声称可成交。

## 当前结论和继续研究的边界

本轮四项全部为`REJECTED_FROZEN_NO_RELIABLE_INCREMENT`，账户均`NOT_RUN_OWN_PREDICTION_GATE`。20万元主账户、2万元费用对照、持续账户复核和回撤退出比较保持未运行。不能填零收益，也没有完成年化10%、净夏普1.2目标；当前状态`NO_VIEW_NO_VALIDATED_POLICY_RULE`不代表预测下跌或现金仓位指令。

这次检验覆盖贷款定价变化及日频联合反应，不代替真实预期、盈利增长、政策传导进度、资金临时压力的独立研究。M1/M2预期差、新订单、财政执行及旧85/15终止结果均保持。下一步应优先选择具有完整事前来源的独立政策意外或传导变量，先说明为何在当前价格之后仍有信息，再做固定收益或风险对照；继续微调这四个失败模型的窗口、符号、样本、权重和卖点，不能据此建立新证据。

本阶段已完成新来源与一次固定检验；整个宏观信息与动态持有研究尚未完成。行情冻结至2026-09-11，研究原点止于2026-07-31，图不代表2026-09-22实时判断。
""")
    write("GPT审阅提问.md", """请先阅读00_README_FIRST.md、研究结论.md、protocol.json、freeze.json与全部失败结果，并把用户原始范围与本轮有限通道区分。

请批评并核对：
1. 完整84个月LPR原文、当次9:30/9:15/9:00时钟、同刻开盘的保守顺延和曲线日终17:30之后才使用，是否足够支持这里有限的历史研究；指出第一版本未认证的剩余问题。
2. 将利率实际变化持续到下一次报价、分别保留1年和5年变化而无真实预期，经济表达有何不足。不要把变化命名为意外。
3. 421个原点、231预测、45等权月、368个保存模型是否一致；逐期训练月全部标签成熟、月内及月间权重、ridge参数、方向约束、风险smearing、简单基准和冻结区块均可用独立只读脚本复核。
4. LPR收益MSE只改善约0.744%、区间跨零且未胜过简单基准；风险相对R0的区间也跨零。是否正确保持四项拒绝，而未将2024年贡献或39次更新子集当作救援。
5. 新旧报价内容替换不是完整每周与公告触发动态账户；反转略好于动量不代表交易优势。170月末反转47.93%、93次此前上涨中84回撤但51最终仍涨，是否被恰当解释。
6. 日频10年曲线交互不是高频政策/信息冲击识别，作者446日期也不是本轮母集。请区分相关性、预测与因果。

请同时给出下一项独立策略/研究方向、优先级、最少必要数据、真实历史公布与预期时钟、简单对照、验证与停止条件。保持宏观传导、市场状态和持续更新的总目标，不再压成单一五日剪刀差模型；不得通过改变已失败的窗口、方向、成本、样本或卖点来包装成功。资金数据缺口只约束其相应模块。若建议进入账户，必须说明依据哪个已通过的自身信息门；完整20万元/2万元账户、242日、现金无风险零及涨后回撤退出均还没有本轮结果。

本包只经过本地结构与保存数据核对，不是外部审阅结论；没有实盘、Paper、Shadow或订单权限扩展。
""")
    write("交付范围与排除说明.md", """本包包含当前研究全部84份LPR直接原文、官方目录及时间说明、全部下载回执（包括一次公开接口失败）、初版保守日期表及冻结前精确时钟细化、完整冻结行情/日历/曲线/旧曲线交叉账簿、源代码/冻结协议、全部预测/训练权重/区块索引/失败比较/图形和只读复核器。前轮ZIP经SHA核对后原样嵌入，其内部更早状态只作快照。

根索引覆盖每一分发成员及SHA-256；自身与外部交付回执排除以避免自引用。80MiB是本轮打包内部上限，不宣称是外部平台限制。排除虚拟环境、无关工作区、缓存、凭证、全部旧供应商曲线分段下载缓存及完整论文附件；本轮完整实际输入保留。继承的原始行情供应商全量下载包未重复纳入。作者日历仅作去重线索，没有用于拟合，未当作已认证的首次公告全集。

可在安装numpy、pandas、pyarrow、beautifulsoup4的Python环境中，运行code/verify_lpr_joint_response_20d_v1.py并通过--study-dir指定解压目录。它核对保存的系数最优条件及预测等式，不重新训练或重新抽样。绘图还需要matplotlib及中文字体。原一次性研究脚本保留项目绝对来源路径与防重跑标记，不承诺在任意目录重新获取全部历史来源或重新拟合。包版本记录见evidence/运行依赖版本.json。

结构验证与保存等式检查不等于独立样本、因果证明、外部审阅或对历史第一版本的认证。当前0新增账户、0退出模拟、0严格前向事件；全年化、夏普、回撤等完整账户指标为NOT_RUN。
""")
    import importlib.metadata
    save_json(STUDY / "evidence/运行依赖版本.json", {"python": sys.version, **{key: importlib.metadata.version(key) for key in ["numpy", "pandas", "pyarrow", "beautifulsoup4", "matplotlib"]}})
    scope = [
        ("M1/M2真实走势与预期差", "前阶段完成整理及有限五日拒绝", "history中的原包及继承图", "不外推为全部宏观无效"),
        ("反转与途中回撤区分", "已完成固定描述", "170月末诊断与3990路径点在前包", "未建立可交易均值回归或退出优势"),
        ("LPR完整来源与实际变化", "84月原文已完成", "lpr_observations_precise.csv与raw", "真实事前共识缺失"),
        ("股债联合收益与风险信息", "四项固定检验均拒绝", "results/全部预定增量比较.csv", "不重调参数抢救"),
        ("新消息的内容更新", "39次固定内容对照已完成", "results/额外LPR复核内容更新诊断.csv", "未建立完整连续账户价值"),
        ("其他宏观政策传导与真实预期", "未完成", "本轮登记前状态及前轮政策目录", "独立通道继续；不当作全国政策母集"),
        ("真实资金暂时压力", "资料与公开时钟仍有缺口", "前轮来源缺口记录", "只限制对应模块"),
        ("20万元/2万元账户与回撤卖出", "NOT_RUN_OWN_PREDICTION_GATE", "results/summary.json", "年化10%/夏普1.2未建立"),
    ]
    pd.DataFrame(scope, columns=["需求", "状态", "证据", "边界"]).to_csv(STUDY / "需求完成与未完成项.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2), flush=True)


def package():
    if ARCHIVE.exists():
        raise FileExistsError("最终归档已存在，不能覆盖")
    visual = read_json(STUDY / "evidence/图表目视核对.json")
    if visual["status"] != "PASS":
        raise ValueError("图形尚未完成目视核对")
    result = subprocess.run([sys.executable, str(STUDY / "code/verify_lpr_joint_response_20d_v1.py"), "--study-dir", str(STUDY)], check=True, capture_output=True, text=True, encoding="utf-8")
    checked = json.loads(result.stdout)
    save_json(STUDY / "evidence/保存数据独立复核.json", checked)
    files = sorted(p for p in STUDY.rglob("*") if p.is_file() and p.name not in {"FILE_INDEX.csv", "delivery_receipt.json"} and "__pycache__" not in p.parts and p.suffix != ".pyc")
    entries = [{"path": p.relative_to(STUDY).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(entries)
    index = buf.getvalue().encode("utf-8-sig")
    building = ARCHIVE.with_suffix(".building.zip")
    with zipfile.ZipFile(building, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for p in files:
            archive.write(p, p.relative_to(STUDY).as_posix())
        archive.writestr("FILE_INDEX.csv", index)
    if building.stat().st_size > 80 * 1024 * 1024:
        raise ValueError("超过本轮80MiB交付上限")
    with tempfile.TemporaryDirectory(prefix="lpr_joint_saved_verify_") as temp:
        with zipfile.ZipFile(building) as archive:
            names = archive.namelist()
            if archive.testzip() is not None or len(names) != len(set(names)):
                raise ValueError("ZIP结构检查失败")
            if set(names) != {row["path"] for row in entries} | {"FILE_INDEX.csv"}:
                raise ValueError("索引覆盖不同")
            for row in entries:
                data = archive.read(row["path"])
                if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError("成员大小或哈希不同")
            archive.extractall(temp)
        result = subprocess.run([sys.executable, str(Path(temp) / "code/verify_lpr_joint_response_20d_v1.py"), "--study-dir", temp], check=True, capture_output=True, text=True, encoding="utf-8")
        if json.loads(result.stdout) != checked:
            raise ValueError("解压后的保存结果复核不同")
    building.replace(ARCHIVE)
    (STUDY / "FILE_INDEX.csv").write_bytes(index)
    receipt = {"created_at": now(), "archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size, "sha256": sha(ARCHIVE),
               "members": len(entries) + 1, "indexed_members": len(entries), "status": "PASS_STRUCTURAL_AND_SAVED_LPR_JOINT_RECOMPUTATION",
               "crc": "PASS", "duplicates": 0, "index_size_hash": "PASS", "fresh_extraction_verification": checked,
               "external_review_completed": False, "new_package_model_fits": 0, "new_accounts": 0,
               "whole_macro_objective_complete": False}
    save_json(ARCHIVE.with_suffix(".receipt.json"), receipt)
    save_json(STUDY / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从冻结结果整理LPR股债研究交付")
    parser.add_argument("action", choices=["prepare", "package"])
    {"prepare": prepare, "package": package}[parser.parse_args().action]()
