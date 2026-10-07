"""解释固定入场退出对照，区分退出改善与额外再入场，不修改原策略。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import point_entry_exit_contribution_v1 as study

OUT = study.OUT
NAMES = {
    "LONG_GUARDS": "多头：价格保护",
    "LONG_MODEL": "多头：价格保护＋训练退出",
    "SHORT_GUARDS": "空头：价格保护镜像",
    "BOTH_GUARDS": "多空：价格保护",
}


def number(value, digits=3):
    if not np.isfinite(value):
        return "未定义"
    if abs(value - 1) < 1e-6:
        return f"{value:.9f}"
    return f"{value:.{digits}f}"


def decompose(points, pairs):
    """仅按已经完成的两条连续规则对齐日期，不创建过滤后的策略。"""
    rows, extra_rows = [], []
    for period in study.PERIODS:
        complete = points.loc[points.period.eq(period) & points.status.eq("COMPLETE")]
        guards = complete.loc[complete.policy.eq("LONG_GUARDS")]
        learned = complete.loc[complete.policy.eq("LONG_MODEL")]
        paired = pairs.loc[pairs.period.eq(period) & pairs.both_complete]
        groups = [
            ("SAME_ENTRIES_GUARDS", "原入场、价格保护", paired.guard_return),
            ("SAME_ENTRIES_MODEL", "同入场、加入训练退出", paired.model_return),
            ("ADDITIONAL_MODEL_ENTRIES", "连续模型规则另有的入场", learned.loc[~learned.entry_date.isin(guards.entry_date), "point_net_return"]),
            ("CONTINUOUS_MODEL", "连续模型规则全部", learned.point_net_return),
        ]
        for group, label, values in groups:
            rows.append({"period": period, "group": group, "label": label, **study.old.statistics(values)})
        extra = learned.loc[~learned.entry_date.isin(guards.entry_date)].copy()
        extra["diagnostic_group"] = "ADDITIONAL_MODEL_ENTRIES"
        extra_rows.append(extra)
        same = learned.loc[learned.entry_date.isin(guards.entry_date)]
        aligned = same.merge(paired[["entry_date", "model_exit_date", "model_return"]], on="entry_date", validate="one_to_one")
        study.require(len(same) == len(aligned) == len(paired), "连续模型与配对的同入场覆盖发生变化。")
        study.require(aligned.exit_date.eq(aligned.model_exit_date).all(), "同入场模型退出日期不同，不能作当前分解。")
        study.require(np.allclose(aligned.point_net_return, aligned.model_return, rtol=0, atol=1e-12), "同入场回报不能对应。")
        study.require(abs(same.point_net_return.sum() + extra.point_net_return.sum() - learned.point_net_return.sum()) < 1e-12, "连续模型分组回报不能相加。")
        study.require(guards.entry_date.isin(learned.entry_date).all(), "存在被替换的旧入场，必须重新解释分组。")
    return pd.DataFrame(rows), pd.concat(extra_rows, ignore_index=True)


def make_figure(metrics, pairs):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10.5, "axes.unicode_minus": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(14.2, 8.1), gridspec_kw={"width_ratios": [1.08, 1.]})
    fig.patch.set_facecolor("#f7f8fa")
    colors = ["#597994", "#cb6d30"]
    order = list(NAMES)
    for j, period in enumerate(study.PERIODS):
        part = metrics.loc[metrics.period.eq(period)].set_index("policy").loc[order]
        positions = np.arange(len(order)) + (j - .5) * .30
        left.barh(positions, part["product"].fillna(0), height=.26, color=colors[j], label="2015—2019" if j == 0 else "2020—2026/8/14")
        for y, row in zip(positions, part.itertuples()):
            if not np.isfinite(row.product):
                left.text(.025, y, f"未定义：{row.wins}胜/{row.n}笔", va="center", color=colors[j], fontsize=10)
            else:
                label = "0.999999826 < 1" if abs(row.product-1) < 1e-6 else f"{row.product:.3f}"
                left.text(row.product + .025, y, f"{label}  ·  {row.n}笔", va="center", fontsize=9.5)
    left.axvline(1, color="#8d4140", linestyle="--", linewidth=1.3)
    left.set_yticks(np.arange(4), ["多头：价格保护", "多头：加训练退出", "空头：镜像保护", "多空：价格保护"])
    left.invert_yaxis()
    left.set_xlim(0, 1.60)
    left.set_xlabel("胜率 × 实际净盈亏比")
    left.set_title("四种连续规则：八组均未严格超过 1", loc="left", fontsize=13, pad=18)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.63, .922), ncol=2, frameon=False, fontsize=10)
    left.grid(axis="x", alpha=.12)
    recent = pairs.loc[pairs.period.eq("evaluation") & pairs.both_complete].sort_values("entry_date")
    values = recent.net_return_increment.to_numpy() * 100
    positions = np.arange(len(recent))
    right.barh(positions, values, height=.64, color=["#367a69" if x > 1e-10 else "#bb603e" if x < -1e-10 else "#8c9298" for x in values])
    for y, value in zip(positions, values):
        right.text(value + (.16 if value >= 0 else -.16), y, f"{value:+.2f}", ha="left" if value >= 0 else "right", va="center", fontsize=9)
    right.set_yticks(positions, recent.entry_date.dt.strftime("%Y-%m-%d"))
    right.invert_yaxis()
    right.axvline(0, color="#6d7378", linewidth=.8)
    right.set_xlim(-8, 12)
    right.set_xlabel("同一入场：训练退出 − 原退出（百分点）")
    right.set_title("2020 年后相同的 13 个入场：8 改善、3 变差、2 不变", loc="left", fontsize=12, pad=18)
    right.grid(axis="x", alpha=.12)
    for ax in (left, right):
        ax.set_facecolor("#f7f8fa")
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("退出改善了一部分点位，增加的再入场稀释了优势", x=.03, y=.965, ha="left", fontsize=21, weight="bold")
    fig.text(.03, .902, "日线收盘判断、下一开盘执行，保留原入场与保护阈值；没有根据结果调整参数。", color="#526272", fontsize=11)
    fig.text(.03, .045, "同入场对照是解释退出贡献的配对样本，并非另一套已经验证的连续策略。连续模型规则还多出 6 次再入场，平均每笔 −0.77%。\n"
             "均已扣统一参考摩擦并计入有资格的股息；八组共享点位。空头为标的方向参考收益，数据截至 2026-08-14。", color="#526272", fontsize=10)
    fig.subplots_adjust(left=.14, right=.98, top=.80, bottom=.17, wspace=.44)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"退出贡献与连续点位差异.{suffix}", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    report = OUT / "解读与下一步.md"
    study.require(not report.exists(), "已有解释报告，不覆盖。")
    points = pd.read_parquet(OUT / "results/连续规则全部点位.parquet")
    metrics = pd.read_parquet(OUT / "results/连续规则指标.parquet")
    pairs = pd.read_parquet(OUT / "results/同入场配对.parquet")
    paired_summary = pd.read_parquet(OUT / "results/同入场增量汇总.parquet")
    yearly = pd.read_parquet(OUT / "results/逐年次数.parquet")
    groups, extras = decompose(points, pairs)
    study.save_table("退出增量与新增入场分解", groups)
    study.save_table("连续模型新增的六个入场", extras)
    errors = []
    for row in metrics.to_dict("records"):
        values = points.loc[points.policy.eq(row["policy"]) & points.period.eq(row["period"]) & points.status.eq("COMPLETE"), "point_net_return"]
        rebuilt = study.old.statistics(values)
        for field in ("p", "b", "product", "mean", "profit_factor"):
            if np.isfinite(rebuilt[field]):
                errors.append(abs(rebuilt[field]-row[field]))
        study.require(rebuilt["point_estimate_pass"] == row["point_estimate_pass"], "显示精度改变了原通过判断。")
    study.require(max(errors, default=0) < 1e-12, "报告点位统计不能重现。")
    make_figure(metrics, pairs)
    lines = [
        "# 新增研究结果：提前退出的改善，被额外再入场部分抵消", "",
        "**本轮四种连续规则、两个时期共八组均未达到严格的 p×实际净盈亏比>1。训练退出在近期同入场对照中有改善，但新增六次再入场的平均收益为负，不能把提前释放持仓自动等同于更多有效机会。**", "",
        "当前只研究510300标的日线及此前完整周线的多空点位；次数为软目标，不要求每年五笔。本轮不进行期权收益、完整账户夏普或分钟线研究。", "",
        "## 正确落实用户的数学门槛", "",
        "p为盈利笔数占比，q为亏损笔数占比；B为平均正净收益率除以平均负净收益率绝对值。标准期望以平均亏损为单位是pB−q，无平手时为pB−(1−p)；利润因子为pB/q。因此pB>1是用户增加的较严格门槛，仍要求每笔净收益均值为正。数学期望的基本定义可核对[CME原文](https://www.cmegroup.com/education/courses/trading-psychology/the-mathematics-of-trading-success)。", "",
        "B一律来自已经完成点位的实际净收益，不用预设止盈止损比替代。没有盈利或没有亏损时，样本B未定义，不视作达标。", "",
        "## 八组完整结果", "",
        "| 连续规则 | 时期 | 完成数 | 胜率 | 实际B | pB | 平均净回报/笔 | 完整年均次数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics.itertuples():
        lines.append(f"| {NAMES[r.policy]} | {study.old.PERIOD_NAMES[r.period]} | {r.n} | {r.p:.2%} | {number(r.b)} | {number(r.product)} | {r.mean:+.2%} | {r.full_year_average:.2f} |")
    lines += [
        "", "多头训练退出的2015—2019乘积精确值为0.9999998258376445，三位小数会显示1.000，但它没有严格大于1。这个极小距离不是稳健优势的证据，也不据此修改阈值。近期同规则乘积只有0.895388。", "",
        "每笔以约十万元初始原价名义份额归一化；每边佣金万四且至少5元、滑点千一，成交价按0.001元不利取整，股息按登记资格计入多头或扣减空头。表中收益为标的点位的统一参考净回报，不是资金账户年化收益。", "",
        "## 分清退出改善与额外入场", "",
        "原规则的入场固定为最近60个交易日开收盘相对隔夜强弱连续两天大于1；空头为小于-1的镜像。它只使用每日开收盘数据。收盘判定6%成本亏损、8%持有路径回撤、最多60个持仓收盘或方向状态连续两天反转，下一可成交开盘退出。这里的6%是触发阈值，不是成交后亏损上限。", "",
        "训练退出在以上保护上增加一个条件：根据入场日首次收盘可取得的模型，连续两次预测继续持有收益为负，则下一可成交开盘退出。模型版本固定到本次退出，初始没有成熟模型就不补入未来版本。本轮没有重新训练。", "",
        "同入场配对固定原价格保护规则的全部入场，只比较两种退出；连续规则则各自独立运行入场、退出、等待和再次进入。它们回答不同问题。", "",
        "| 时期 | 两侧完成配对 | 有成熟模型 | 平均回报增量 | 改善/变差/不变 | 平均减少持有交易日 | 年份区块描述区间 |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for r in paired_summary.itertuples():
        lines.append(f"| {study.old.PERIOD_NAMES[r.period]} | {r.both_complete} | {r.model_available_pairs} | {r.mean_increment*100:+.2f}个百分点 | {r.improved_pairs}/{r.worsened_pairs}/{r.unchanged_pairs} | {r.mean_sessions_saved:.2f} | [{r.year_block_mean_p025*100:+.2f}, {r.year_block_mean_p975*100:+.2f}]个百分点 |")
    lines += [
        "", "近期13个相同入场，平均每笔改善2.03个百分点，但年份分块区间仍跨零。例子同时包括改善和损失：2020-02-05入场由约-0.95%变为+8.64%；2024-02-19入场则由约+8.81%变为+2.70%。早期九个入场只有四个开始时具备成熟模型，不能把全部早期表现归因于训练。", "",
        "| 2020年后分解 | 笔数 | 胜率 | 实际B | pB | 平均净回报/笔 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in groups.loc[groups.period.eq("evaluation")].itertuples():
        lines.append(f"| {r.label} | {r.n} | {r.p:.2%} | {number(r.b)} | {number(r.product)} | {r.mean:+.2%} |")
    lines += [
        "", "同入场加训练退出的13笔pB为1.204；加入六次额外入场后，连续规则19笔pB降为0.895。六笔额外点位为三赢三亏，平均盈利约2.89%、平均亏损约4.44%，实际B约0.652，每笔平均净回报约-0.77%。这是已完成两条路径的归因，不是可以事后删除亏损后使用的新策略。", "",
        "| 连续模型额外入场 | 退出 | 参考净回报 |",
        "|---|---|---:|",
    ]
    for r in extras.itertuples():
        lines.append(f"| {r.entry_date.date()} | {r.exit_date.date()} | {r.point_net_return:+.2%} |")
    lines += [
        "", "13笔配对来自原规则的一整套入场时钟；提前退出后是否继续等待原周期结束，必须作为入场前明确的规则另行评价，不能在结果出现后将上述分组自动升级。近期配对的pB达线也不能弥补早期未达线或提供独立验证。", "",
        "## 频率和空头证据", "",
        "2020—2025完整年份，多头训练退出从年均1.83笔增加到2.50笔；2026截至8月14日从2笔增至4笔，不年化。2023年两条多头规则均为零笔，最长空仓间隔均为346个交易日。", "",
        "空头镜像早期3笔全部亏损，近期4笔3赢1亏且pB仍只有0.831。近期胜率75%来自四笔样本，不能据此认定高胜率做空成立。纯保护的多空合并也未达线。", "",
        "| 年份 | 多头保护 | 多头加训练退出 | 空头保护 | 多空保护 |",
        "|---|---:|---:|---:|---:|",
    ]
    annual = yearly.pivot(index=["period", "year", "full_year"], columns="policy", values="completed_points").sort_values("year")
    for (_, year, full), r in annual.iterrows():
        label = str(year) if full else f"{year}截至8月14日"
        lines.append(f"| {label} | " + " | ".join(str(int(r[p])) for p in NAMES) + " |")
    unique = points.loc[points.status.eq("COMPLETE"), ["direction", "entry_date", "exit_date"]].drop_duplicates()
    lines += [
        "", "## 现有线索与后续研究边界", "",
        "上一轮两条组合点位线索的四组历史pB仍约为1.016/1.239和1.015/1.154；本轮研究的是更深层原始入场与退出组件，不能用组件未达线覆盖组合测量结果。组合仍存在早期大盈利集中、共同历史反复使用和缺少独立验证的问题。", "",
        "本轮将‘多做几次’的障碍具体化为提前退出后的再进入质量。下一步应对原有两条组合线索的入场来源和完整持有路径进行对应，检查辅助点位是否与这批额外再入场重合，解释频率和盈亏比的权衡；保持全部已固定阈值，不直接删除本轮观察到的六笔，不重新搜索使失败规则达线。", "",
        f"八组连续重放合计86条完成比较记录，只有{len(unique)}组不同方向与进出日期，且仍可能相互重叠；另3条末端未完成比较记录不进入胜率。原训练退出28个完成点位日期全部复现；核对141条保存模型的当时成熟训练成员，重算391条已保存预测，最大误差为零。", "",
        "七项必要测试已经通过，覆盖模型版本固定、无成熟模型不后补、退出锁定、空头股息和费用、再进入等待、模型时钟及日线前缀不变。本报告按逐笔结果再计算八组核心指标，最大误差为零。没有新增训练、参数搜索或完整资金账户回测。", "",
        "数据截至2026-08-14。目标保持进行中；本轮得到可复算的机制解释，尚未得到经过独立验证且符合要求的多空规则。", "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    receipt = {
        "at": study.common.now(), "source": str(Path(__file__).relative_to(ROOT)),
        "code_sha256": study.common.digest(Path(__file__)), "source_results_modified": False,
        "maximum_metrics_recomputation_error": max(errors, default=0),
        "additional_completed_model_entries": len(extras),
        "completed_unique_direction_entry_exit_groups": len(unique),
        "test_command": ".venv\\Scripts\\python.exe -m pytest tests\\test_point_entry_exit_contribution_v1.py -q",
        "observed_tests_passed": 7, "observed_test_exit_code": 0,
        "figure_visually_inspected": False, "goal_achieved": False,
        "artifacts": [{"path": str(path.relative_to(OUT)), "sha256": study.common.digest(path)} for path in
                      [report, OUT / "退出贡献与连续点位差异.png", OUT / "退出贡献与连续点位差异.svg",
                       OUT / "results/退出增量与新增入场分解.csv", OUT / "results/连续模型新增的六个入场.csv"]],
    }
    study.common.save_json(OUT / "interpretation_receipt.json", receipt)
    print("八组指标、同入场与新增六笔的分解、年度频率和解释图已保存。", flush=True)


if __name__ == "__main__":
    main()
