"""生成已有策略失败归因的中文说明、普通结果表和图，不制作审阅包。"""
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now

SOURCE = ROOT / "reports/research/510300_forward_eps_corrected_account_attribution_v1_1"
OUT = ROOT / "deliverables/510300前瞻EPS失败归因与进出场影响_20260907"


def percent(value):
    return f"{100 * value:.2f}%"


def main(plot_only=False):
    checked = read(SOURCE / "saved_numerical_verification.json")
    assert checked["status"].startswith("PASS_")
    result = read(SOURCE / "result.json")
    summaries = pd.read_csv(SOURCE / "账户规模与时变损益分解.csv")
    base = summaries.loc[summaries.cost_scenario.eq("BASE") & summaries.period.eq("FULL_EVALUATION")].set_index("model")
    active = summaries.loc[summaries.cost_scenario.eq("BASE") & summaries.period.eq("AFTER_FIRST_EPS_MODEL")].set_index("model")
    errors = pd.read_csv(SOURCE / "不同市场状态下预测误差.csv")
    full_error = errors.loc[errors.group.eq("全部已兑现预测")].set_index("model")
    OUT.mkdir(parents=True, exist_ok=plot_only)
    if not plot_only:
        for name in ["账户规模与时变损益分解.csv", "同期间策略增量分解.csv", "逐年持仓和损益.csv",
                     "已有账户全部进出场.csv", "盈利集中程度.csv", "已有预测与兑现结果.csv", "不同市场状态下预测误差.csv"]:
            shutil.copy2(SOURCE / name, OUT / name)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    assert font_path.exists()
    font = FontProperties(fname=str(font_path))
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.12, 1]})
    fig.patch.set_facecolor("#f5f7f8")
    keys = ["BUY_HOLD", "PRICE", "DAILY_TREND_EXIT", "REVISION_DISTRIBUTION", "PRICE_EPS", "EPS", "OPTIONAL_VALUATION", "MATCHED_EPS", "VALUATION"]
    labels = ["买入持有", "共同月份价格", "EPS＋每日趋势退出", "EPS＋修正分布", "价格＋EPS", "原EPS", "EPS＋可选估值", "估值共同月份EPS", "EPS＋研报估值"]
    values = base.loc[keys, "net_sharpe"].to_numpy()
    axes[0].barh(labels, values, color=["#8b9ba3" if k in ["BUY_HOLD", "PRICE", "MATCHED_EPS"] else "#247e83" for k in keys], height=.63)
    for i, value in enumerate(values):
        axes[0].text(value + .018, i, f"{value:.3f}", va="center", fontsize=10)
    axes[0].axvline(1.2, color="#b5533c", linewidth=1.5, linestyle="--")
    axes[0].text(1.19, .975, "目标1.2", ha="right", va="top", color="#b5533c", transform=axes[0].get_xaxis_transform())
    axes[0].set_xlim(0, 1.29)
    axes[0].set_title("完整账户成本后夏普", loc="left", fontweight="bold", pad=15)
    axes[0].set_xlabel("2020-01-02至2026-08-14，基础费用，1604个交易日")
    groups = ["判断时位于120日均线之上", "判断时不高于120日均线"]
    locations = np.arange(2)
    records = {}
    for name in ["E3_FORWARD_EPS", "E1_PRICE_FORWARD_EPS"]:
        records[name] = errors.loc[errors.model.eq(name)].set_index("group").loc[groups]
    actual = records["E3_FORWARD_EPS"].mean_actual_return.to_numpy() * 100
    pure = records["E3_FORWARD_EPS"].mean_predicted_return.to_numpy() * 100
    mixed = records["E1_PRICE_FORWARD_EPS"].mean_predicted_return.to_numpy() * 100
    for delta, series, label, color in [(-.25, actual, "实际兑现", "#247e83"), (0, pure, "原EPS预测", "#ba704f"), (.25, mixed, "价格＋EPS预测", "#8b9ba3")]:
        bars = axes[1].bar(locations + delta, series, width=.23, label=label, color=color)
        for bar, value in zip(bars, series):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + .1, f"{value:.2f}", ha="center", fontsize=9)
    axes[1].set_xticks(locations, ["判断时在120日均线上方\n20次已兑现预测", "判断时不高于120日均线\n13次已兑现预测"])
    axes[1].set_ylim(0, 6.3)
    axes[1].set_ylabel("未来60交易日平均收益（%）")
    axes[1].set_title("不同状态下的预测偏差", loc="left", fontweight="bold", pad=15)
    axes[1].legend(loc="upper left", frameon=False, ncol=1)
    for axis in axes:
        axis.set_facecolor("#f5f7f8")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x" if axis is axes[0] else "y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("现有前瞻EPS策略：账户结果与预测偏差", x=.02, ha="left", fontsize=17, fontweight="bold", y=.98)
    fig.text(.02, .02, "全部为已观察历史；右图60日标签有重叠，分组结果不等于独立验证。没有删除年份或重新运行策略。", fontsize=10, color="#53656d")
    fig.subplots_adjust(left=.17, right=.98, top=.85, bottom=.18, wspace=.36)
    chart = OUT / "完整账户表现与预测偏差.png"
    fig.savefig(chart, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    if plot_only:
        print("图表目标标签位置已调整：", str(chart))
        return
    rows = []
    for key in ["EPS", "PRICE_EPS", "PRICE", "DAILY_TREND_EXIT", "REVISION_DISTRIBUTION", "VALUATION", "MATCHED_EPS", "OPTIONAL_VALUATION", "BUY_HOLD"]:
        row = base.loc[key]
        rows.append(f"| {row['策略']} | {row.net_sharpe:.3f} | {percent(row.annualized_return)} | {percent(row.max_drawdown)} | {row.net_cumulative_cny:,.2f} | {int(row.trade_count)} |")
    table = "\n".join(rows)
    eps = base.loc["EPS"]
    trend = base.loc["DAILY_TREND_EXIT"]
    loss = float(eps.net_cumulative_cny - trend.net_cumulative_cny)
    extra_cost = float(eps.cost_cumulative_cny - trend.cost_cumulative_cny)
    active_eps = active.loc["EPS"]
    mse = full_error.loc["E3_FORWARD_EPS", "mean_squared_error"]
    naive_mse = full_error.loc["E3_FORWARD_EPS", "training_mean_baseline_squared_error"]
    source_document = ROOT / "deliverables/510300前瞻EPS多机构覆盖与完整规则_20260907/前瞻EPS多机构覆盖进度与三种策略完整规则.md"
    markdown = f"""# 前瞻EPS策略为什么仍未达到夏普1.2

更新于：{now()}。本次直接分析已保存的十八条基础及压力账户、三十六个账户期间和一百零八条预测记录，没有新增策略、重新训练或改写历史账户。当前第十九轮多机构原件归档及后续程序继续运行，新结果尚未产生。不准备GPT数值审阅包。

## 结论与实际影响

**目前的问题同时出在数据覆盖、预测可靠性，以及预测转为实际仓位和退出的方式上。** 加因子、增加退出条件或切换策略，都需要检查实际账户效果。已经出现预测误差更小、账户收益反而更低的情况，也出现增加退出规则后错过持仓收益的情况。

原EPS模型首次给出预测是在2023年8月31日，下一交易日是2023年9月1日；实际账户在2020至2023年都没有持仓。原EPS在2024年的单年夏普约1.308，2025年约1.197，但原完整期间只有0.619。即使另看首次预测可用后的714日，诊断夏普也只有0.928。后一个区间只用于解释数据可用之后的行为，目标验收仍采用完整1604个交易日。

## 完整期间的真实账户结果

初始20万元，2020年1月2日至2026年8月14日开盘终点，基础费用；净利润已经包含佣金、滑点和实际分红记账。最大回撤以负号表示。

| 策略 | 成本后夏普 | 年化收益 | 最大回撤 | 累计净利润（元） | 成交笔数 |
| --- | --- | --- | --- | --- | --- |
{table}

表格全部保留原有评价时间。估值主方案与它的共同月份EPS对照必须一起看：二者只差约1201元，不能把估值主方案相对原完整EPS的全部收益差额归于估值因子。

![完整账户表现与预测偏差](<{chart.as_posix()}>)

## 失败原因一：历史覆盖不足，尚未检验完整周期能力

原EPS来源从2022年6月才达到原有覆盖要求，再等待至少十二个已兑现月末样本，首次预测到2023年8月底才出现。2020至2022年的空仓属于缺少合格输入；2023年后四个月已有预测但仍没有实际持仓。两种情况应分开解释，不能把长期现金都说成模型提前判断风险。

第十九轮正在补充东吴6734份历史成员报告，2017至2021年有3072份。完整数据处理后，将检验更早能够作出盈利判断时的真实账户结果；不能提前认定新增历史一定提高夏普。

## 失败原因二：方向和收益幅度预测仍不可靠

原EPS共有36次月末预测，其中33次的未来60交易日结果已经完整兑现。33次里方向正确18次；实际上涨19次，因此“始终判断上涨”在这个样本中也会对19次。该比较只说明当前方向优势尚不明显，不代表应该采用永远买入的规则。

原EPS平均绝对预测误差约8.60个百分点；均方误差为{mse:.6f}，比每次仅采用当时成熟训练样本平均收益的基准高约{100*(mse/naive_mse-1):.1f}%。这个简单基准只用于预测误差对照，没有据此生成新的交易账户。

在判断日位于120日均线上方的20次已兑现预测里，原EPS有18次预计上涨，实际只有9次上涨；预计未来60日平均收益约3.98%，实际约0.52%。在不高于均线的13次里，原EPS预计平均约0.23%，实际约5.35%。这显示本样本中存在高估部分上涨状态、低估部分反弹阶段的情况，但样本少且60日标签重叠，不能据此直接宣布一种新的行情切换规则有效。

一月恰逢预测目标年度切换，只有三次已兑现观测，暂不足以把误差归因为年度切换。上一轮已经确认样本公司增减会改变因子中位数；这些来源结构问题继续与模型误差分开检查。

## 失败原因三：预测更准确，不一定带来更好的仓位

价格加EPS在33次已兑现预测中对20次，平均绝对误差约7.21个百分点，均方误差约0.007598，均优于原EPS对应数值。但是，它完整账户只赚约7.64万元，原EPS赚约9.46万元，前者少约1.82万元。

原因不能仅靠预测误差判断。仓位取决于预测大小、波动估计、上一期持仓、整手和成本；两个模型即使方向相同，也可能在关键阶段持有不同数量。保存路径的损益分解显示，这项差额主要来自持仓随时点变化所形成的市场损益变化。后续需要同时比较预测和实际持仓，不能只把均方误差最低的模型作为交易赢家。

## 失败原因四：退出规则改变了收益获取过程

原EPS使用月末统一判断进入、增加、减少和退出；增加期限与每日趋势退出后，成交从10笔升至23笔，累计净利润从约9.46万元降至约0.70万元，少赚{loss:,.2f}元。其中额外费用为{extra_cost:,.2f}元，约占总差额的{100*extra_cost/loss:.2f}%。

差距主要来自持仓变化造成的市场损益，包括提前退出后少持仓、再入场与退出衔接等影响。费用金额是实际账本中的费用差额；没有假装把费用删除后重新得到了可实施账户。原退出规则的全部成交都保留在附表里。

这也说明退出不能只写一个看似合理的条件：必须同步明确清仓后什么时候重入、新的盈利预测怎样续期、未成交怎样处理，以及期间持仓会怎样变化。当前第十九轮已把这些规则完整登记，保持原月末收益与风险评分，不临时加入新的每日卖点。

## 原EPS收益究竟来自平均持仓还是持仓时点

从原EPS首次可用预测后的固定714日观察，其累计净利润约{active_eps.net_cumulative_cny:,.2f}元，可以按保存的实际份额作如下会计拆分：

- 按该期间平均持有份额计算的市场损益项：{active_eps.mean_size_cumulative_cny:,.2f}元。
- 份额高于或低于期间平均值所对应的时点变化项：{active_eps.time_variation_cumulative_cny:,.2f}元。
- 实际佣金及滑点贡献：{active_eps.cost_cumulative_cny:,.2f}元。

三项合计与实际净利润一致。这个窗口中两项市场损益都为正，因此也不能把原EPS收益全部解释为固定持仓。平均份额是看完整段历史后计算的诊断量，不能用于当时下单；时点变化项也不自动等于独立、稳定的择时能力。

## 接下来怎样推进

现有东吴采集与顺序接续程序继续：先完成全量原件和修正EPS，再计算两机构因子，运行三组已登记策略及十条评价账户。新策略全部有进入、加仓、减仓、退出和重新进入规则，见[完整中文因子与买卖规则](<{source_document.as_posix()}>)。

新结果出来后，首先检查扩大早期数据覆盖是否改变完整周期表现，再比较两机构与单机构、信息质量与纯盈利信息。模型误差、仓位贡献、退出损益和样本构成共同解释结果。当前诊断没有修改第十九轮规则，没有新增账户或把诊断区间改成验收区间。

成本后夏普1.2及稳定超额的目标保持未完成。现有最高完整期间历史观察值仍约0.687，原有增量不确定性区间跨过零，反复观察的历史也不能当作独立验证。

## 普通结果文件及核对

本目录附七份CSV：完整账户与期间拆分、策略增量拆分、逐年损益、全部进出场、盈利集中程度、每次预测与实际兑现，以及不同市场状态的预测误差。费用支出的正额与对收益的负贡献分别保存，原分解数值逐项保持。

来源结果：`reports/research/510300_forward_eps_corrected_account_attribution_v1_1/result.json`。核对通过十八个原账户、三十六个期间、二十八组比较和一百零八条预测的时钟；没有重新训练、生成账户、下载数据或安全审计。
"""
    document = OUT / "前瞻EPS失败归因_预测仓位与退出.md"
    document.write_text(markdown, encoding="utf-8")
    save(OUT / "delivery_receipt.json", {"created_at": now(), "source_study": result["study_id"],
        "source_verification_status": checked["status"], "main_document": document.relative_to(ROOT).as_posix(),
        "csv_files": 7, "chart": chart.relative_to(ROOT).as_posix(), "visual_check_pending": True,
        "new_account_evaluations": 0, "gpt_review_package_created": False}, exclusive=True)
    print(str(document))
    print(str(chart))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    main(parser.parse_args().plot_only)
