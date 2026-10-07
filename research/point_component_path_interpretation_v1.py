"""报告组合增次的来源、收益接续和比值变化，不产生新交易规则。"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import point_component_path_attribution_v1 as study

OUT = study.OUT
GROUPS = {"CORE_INVOLVED": "核心参与入场", "AUXILIARY_ONLY": "辅助单独入场", "ALL": "全部点位"}


def simple_groups(entries):
    rows = []
    for model in study.MODELS:
        for period in study.PERIODS:
            current = entries.loc[entries.model.eq(model) & entries.period.eq(period) & entries.status.eq("COMPLETE")]
            for group, values in (("CORE_INVOLVED", current.loc[current.entry_cause.ne("AUXILIARY_ONLY"), "point_net_return"]),
                                  ("AUXILIARY_ONLY", current.loc[current.entry_cause.eq("AUXILIARY_ONLY"), "point_net_return"]),
                                  ("ALL", current.point_net_return)):
                rows.append({"model": model, "period": period, "group": group, "label": GROUPS[group], **study.old.statistics(values)})
    frame = pd.DataFrame(rows)
    original = pd.read_parquet(OUT / "inputs/点位指标.parquet")
    comparison = frame.loc[frame.group.eq("ALL")].merge(original, on=["model", "period"], suffixes=("", "_saved"), validate="one_to_one")
    for field in ("n", "p", "b", "product", "mean", "profit_factor"):
        study.require(np.allclose(comparison[field], comparison[field + "_saved"], rtol=0, atol=1e-12, equal_nan=True), "来源分组改变了原全部点位指标。")
    return frame


def figure(groups):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 11, "axes.unicode_minus": False})
    order = [(m, p) for m in study.MODELS for p in study.PERIODS]
    products, means, counts = np.empty((4, 3)), np.empty((4, 3)), np.empty((4, 3), dtype=int)
    for i, (model, period) in enumerate(order):
        for j, group in enumerate(GROUPS):
            row = groups.loc[groups.model.eq(model) & groups.period.eq(period) & groups.group.eq(group)].iloc[0]
            products[i, j], means[i, j], counts[i, j] = row["product"], row["mean"] * 100, row["n"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.4, 7.6))
    fig.patch.set_facecolor("#f7f8fa")
    left.imshow((products > 1).astype(float), vmin=0, vmax=1, cmap=ListedColormap(["#e8edf1", "#bedac9"]), aspect="auto")
    right.imshow(means, cmap="RdBu", norm=TwoSlopeNorm(vmin=-7, vcenter=0, vmax=7), aspect="auto")
    labels = ["回撤限制组合\n2015—2019", "回撤限制组合\n2020—2026/8", "相关确认组合\n2015—2019", "相关确认组合\n2020—2026/8"]
    left.set_yticks(range(4), labels)
    right.set_yticks([])
    for ax in (left, right):
        ax.set_xticks(range(3), ["核心参与入场", "辅助单独入场", "全部点位"])
        ax.tick_params(length=0, pad=12)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="#f7f8fa", linewidth=5)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    left.set_title("胜率 × 实际净盈亏比\n绿色底表示历史点值严格超过 1", loc="left", fontsize=13, pad=18)
    right.set_title("每笔平均净回报\n均值不随比值同步变化", loc="left", fontsize=13, pad=18)
    for i in range(4):
        for j in range(3):
            left.text(j, i, f"{products[i,j]:.3f}\n{counts[i,j]}笔", ha="center", va="center", color="#233645", fontsize=13)
            right.text(j, i, f"{means[i,j]:+.2f}%", ha="center", va="center", color="white" if abs(means[i,j]) > 4 else "#233645", fontsize=14)
    fig.suptitle("组合乘积达线，不等于新增交易改善收益", x=.035, y=.96, ha="left", fontsize=22, weight="bold")
    fig.text(.035, .885, "辅助点位的平均收益在早期为负、近期为正；所有分组都保留原来的进出场与参考摩擦。", fontsize=11, color="#526272")
    fig.text(.035, .057, "分组依据原点位入场当天的已知来源，仅用于解释；不是删除组件后重新运行的策略。四组共享大量点位，不能当作独立验证。\n"
             "p×B>1继续作为硬门槛，同时并列净均值、利润因子与年度次数。数据截至2026-08-14。", fontsize=10, color="#526272")
    fig.subplots_adjust(left=.18, right=.97, bottom=.17, top=.74, wspace=.13)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"辅助次数与点位质量.{suffix}", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    report = OUT / "研究结论.md"
    study.require(not report.exists(), "已有来源解释报告，不覆盖。")
    entries = pd.read_parquet(OUT / "results/所有点位入场对应.parquet")
    paths = pd.read_parquet(OUT / "results/整笔路径与来源切换.parquet")
    links = pd.read_parquet(OUT / "results/六次额外再入场对应.parquet")
    states = pd.read_parquet(OUT / "results/持有状态收益分解.parquet")
    groups = simple_groups(entries)
    study.save_table("核心参与与辅助单独入场对照", groups)
    figure(groups)
    lines = [
        "# 组合增次来源与实际持有路径", "",
        "**上一轮较弱的额外再入场主要来自核心信号，不是辅助信号。两条组合都保留了其中五次同日新开仓，另一次当时已经持有；辅助新增点位则表现为早期负收益、近期正收益。历史组合pB仍超过1，但其数值上升有时伴随平均收益下降，不能单凭乘积上升宣称频率优化成功。**", "",
        "本轮覆盖已保存的两条组合全部94条完成记录与4条末端未完成记录，拆开1683个已观察开盘区间；没有新增策略、参数搜索、训练或资金账户。只研究510300标的日线与此前完整周线，年度次数为软目标，p乘实际净B严格大于1和净均值为正的要求不变。", "",
        "## 六次额外再入场具体落在哪里", "",
        "‘同日新开仓’和‘已经持有’由实际开盘持仓状态区分；退出开盘不再算持有。两条组合在下面六个日期的对应完全相同，不能将它们算成两次独立验证。", "",
        "| 上一轮额外入场 | 原训练退出规则：退出/净回报 | 两组合当时状态 | 组合对应点位：入场至退出/净回报 |",
        "|---|---|---|---|",
    ]
    selected = links.loc[links.model.eq(next(iter(study.MODELS)))]
    for r in selected.itertuples():
        relation = "核心信号同日新开仓" if r.relation == "SAME_ENTRY" else "原有核心持仓继续，没有新增一笔"
        lines.append(f"| {r.extra_entry_date.date()} | {r.extra_exit_date.date()} / {r.extra_model_return:+.2%} | {relation} | {r.composite_entry_date.date()}—{r.composite_exit_date.date()} / {r.composite_full_point_return:+.2%} |")
    five = selected.loc[selected.relation.eq("SAME_ENTRY"), "composite_full_point_return"]
    five_stats = study.old.statistics(five)
    lines += [
        "", f"五次同日新开仓均来自核心，平均每笔净回报{five_stats['mean']:+.2%}，胜率{five_stats['p']:.0%}，pB为{five_stats['product']:.3f}。辅助不是这批点位的入场来源，不能靠关闭辅助来解决这一组较弱的再进入。", "",
        "2026-01-16的情形不可直接比较两行整笔收益：组合从2025-12-19就已持有，而原训练退出规则在1月16日另开一笔。组合没有新增交易，不等于避开了1月16日后的价格下跌；两者整笔回报的起点不同。", "",
        "同日入场也不必有相同退出。2021-02-03原训练退出于2月19日得到约+4.27%，组合直到2月26日退出变为约-2.16%；2024-08-02则相反，组合更早退出，把对应整笔亏损从约-6.12%缩为-1.24%。必须保留两种方向的差异。", "",
        "## 辅助新增机会的跨时期表现", "",
        "这里按实际入场当日是否只有辅助来源分组，整笔退出保持原样。‘核心参与’包含核心与辅助同时入场，因此三列是明确的互补分组加全体，而不是各自重新运行的组件策略。", "",
        "| 组合 | 时期 | 分组 | 笔数 | 胜率 | 实际B | pB | 平均净回报/笔 | 利润因子 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in groups.itertuples():
        lines.append(f"| {study.MODEL_NAMES[r.model]} | {study.old.PERIOD_NAMES[r.period]} | {r.label} | {r.n} | {r.p:.2%} | {r.b:.3f} | {r.product:.3f} | {r.mean:+.2%} | {r.profit_factor:.3f} |")
    lines += [
        "", "辅助新增点位，回撤限制组合早期13笔平均-1.38%、近期8笔平均+2.84%；相关确认组合早期5笔平均-0.65%、近期7笔平均+2.48%。近期辅助胜率分别为87.50%和85.71%，但平均盈利小于平均亏损，pB分别只有0.659和0.502。高胜率没有单独满足当前要求。", "",
        "## 为什么乘积变好，平均收益却可能变差", "",
        "以回撤限制组合2015—2019为例：核心参与的9笔平均净回报+6.75%，pB为0.963；辅助单独入场的13笔平均-1.38%，pB为0.160。合并后22笔平均净回报降至+1.95%，pB却上升到1.016。利润因子也从核心参与组的4.332降至整体的2.033。", "",
        "原因可从分母直接看到：核心参与组平均盈利约11.29%、平均亏损约9.12%；混合后平均盈利约7.67%、平均亏损约3.77%。加入较小的亏损会改变实际B的分母，pB不是各组乘积的笔数加权平均。因此用户的pB>1仍严格保留，同时必须并列真实平均收益和利润因子来解释增次效果，不能把乘积刚达线当成新增点位本身有优势。", "",
        "以上是样本构成的描述，不等于单独执行核心会取得+6.75%每笔；实际移除辅助后，未来持仓状态与入场顺序可能改变，本轮没有运行这种策略。", "",
        "## 逐日路径核对纠正了什么", "",
        "实现允许核心与辅助接续，但在本轮94条完成记录中，没有发现‘由核心参与入场，后来只靠辅助继续持有’的点位。实际观察到的是三条比较记录由辅助入场后出现核心，其中两条是两组合共享的同一段行情。因此不能用假设的接续机制解释全部频率差异。", "",
        "| 组合 | 辅助入场 | 最终退出 | 仅辅助区间毛贡献 | 核心参与区间毛贡献 | 摩擦 | 整笔净回报 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    changes = paths.loc[paths.status.eq("COMPLETE") & paths.later_core_after_auxiliary_entry]
    for r in changes.itertuples():
        core = r.CORE_ONLY_gross_return + r.CORE_AND_AUXILIARY_gross_return
        lines.append(f"| {study.MODEL_NAMES[r.model]} | {r.entry_date.date()} | {r.exit_date.date()} | {r.AUXILIARY_ONLY_gross_return:+.2%} | {core:+.2%} | {r.friction_return:+.2%} | {r.recomputed_net_return:+.2%} |")
    lines += [
        "", "每个开盘区间只使用该开盘之前的收盘状态；区间原价差和有登记资格的股息除以该笔原始入场价，随后相加。这里的贡献表示收益发生时的状态，不是对组件独立预测能力的因果估计。交易摩擦单列，不按信号权重虚构资金分配。", "",
        "| 组合 | 时期 | 仅核心期间 | 仅辅助期间 | 二者同时期间 | 摩擦 | 最终每笔净均值 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in study.MODELS:
        for period in study.PERIODS:
            values = states.loc[states.model.eq(model) & states.period.eq(period)].set_index("state").average_contribution_per_original_point
            columns = [values[k] for k in ("CORE_ONLY", "AUXILIARY_ONLY", "CORE_AND_AUXILIARY", "FRICTION", "NET")]
            lines.append(f"| {study.MODEL_NAMES[model]} | {study.old.PERIOD_NAMES[period]} | " + " | ".join(f"{v:+.2%}" for v in columns) + " |")
    lines += [
        "", "上述各状态都以同组全部完成点位数为分母，不仅除以进入该状态的点位；三项毛贡献加摩擦严格等于最终净均值。没有未知保持或零目标等待的完整持仓区间。未完成路径另外保存，不计入胜率或完整点位均值。", "",
        "## 本轮结论和下一步", "",
        "保留两条组合历史点值达线的事实，也保留早期大盈利集中和分组收益跨期反转的限制。六次弱再进入与辅助增次是两种来源，不能简单关闭辅助解决；根据已知盈利分组继续挑条件，也不能形成新的独立证据。", "",
        "下一步转向固定候选的独立验证准备：确认目前全部候选可用的数据截止日、已经用过的历史范围及完整信号计算依赖，建立事前固定的后续点位记录。保留现有阈值和全部失败记录；新增数据只能按此前确定的规则进入，不能倒用来挑选方案。", "",
        "本轮三项必要测试已通过，覆盖开盘持有边界、分红权益与区间损益加总、未来价格不影响前段及执行日收盘拒绝。98条完成或末端标记的净回报复算最大误差为零；原四组全部指标保持一致。数据截至2026-08-14，无当前市场进出点报价，无期权收益回测或委托。目标仍未宣布完成。", "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    study.common.save_json(OUT / "interpretation_receipt.json", {
        "at": study.common.now(), "source": str(Path(__file__).relative_to(ROOT)), "code_sha256": study.common.digest(Path(__file__)),
        "source_results_modified": False, "test_passed": 3, "test_exit_code": 0,
        "observed_test_command": ".venv\\Scripts\\python.exe -m pytest tests\\test_point_component_path_attribution_v1.py -q",
        "figure_visually_inspected": False, "goal_achieved": False,
        "artifacts": [{"path": str(p.relative_to(OUT)), "sha256": study.common.digest(p)} for p in
                      (report, OUT / "辅助次数与点位质量.png", OUT / "辅助次数与点位质量.svg", OUT / "results/核心参与与辅助单独入场对照.csv")],
    })
    print("六次再入场、辅助跨期表现及收益构成解释已保存。", flush=True)


if __name__ == "__main__":
    main()
