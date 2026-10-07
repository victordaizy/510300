"""交付本轮已完成的固定比较，最新进度不回退到旧研究。"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, digest, now
import research.persistent_state_research_delivery_v1 as earlier
import research.index_state_strategy_value_v1 as original
import research.index_state_strategy_value_saved_completion_v1 as value
import research.index_state_active_allocation_v1 as active

SUMMARY = earlier.SUMMARY
NAMES = {"EQUAL_EXPERTS": "固定等权", "HISTORY": "同窗历史价值分配", "PRICE_HMM": "价格状态价值分配",
         "MACRO_HMM": "宏观状态价值分配", "PARENT_MACRO_HMM": "允许参考空仓等待的父策略"}


def paragraph(metric):
    return f"压力夏普{metric['net_sharpe']:.3f}、复合年化{metric['annualized_return']:.2%}、最大回撤{abs(metric['max_drawdown']):.2%}，20万元变为{metric['ending_equity']:,.2f}元"


def comparison_lines(result):
    return [f"- 相对{NAMES[row['right']]}：算术年化{row['annual_arithmetic_difference']*100:+.3f}个百分点，95%区间[{row['lower_95']*100:+.3f}, {row['upper_95']*100:+.3f}]个百分点。" for row in result["comparisons"]]


def reports(value_result, active_result):
    for folder, result, title in [(value.OUT, value_result, "状态用于具体策略的收益分配"),
                                   (active.OUT, active_result, "分配只允许当日有信号的参考策略")]:
        metric = result["primary"]
        rows = [f"{title}实验已完成，{paragraph(metric)}。三项历史点值门槛{'同时通过，但独立验证仍未完成' if metric['historical_point_targets_met'] else '未同时通过'}。", "",
                "评价为2021年1月4日至2026年8月14日的完整账户，共1,361个交易日，252日年化，现金和无风险收益均按0计。空仓、亏损、T+1、佣金、滑点、分红和期末未清仓风险都包含在结果中。", "",
                "四个固定压力账户：", ""]
        for row in result["all_accounts"]:
            if row["cost"] == "STRESS":
                rows.append(f"- {NAMES[row['policy']]}：夏普{row['net_sharpe']:.3f}，年化{row['annualized_return']:.2%}，回撤{abs(row['max_drawdown']):.2%}，费用{row['cost_cny']:,.2f}元。")
        rows += ["", "参考策略固定为D60_INTRA、R2_Z_CONFIRM和PANIC_ONLY；各自按原有价格入场、退出和压力成本连续运行。它们是候选收益路径，不被视为已经独立成立的Alpha。模型预测各参考当日开盘交易前至第五个后续开盘交易前的权益收益，排除了入场前跳空。", "",
                 "训练权重复用刚完成的两年每日状态模型。所有收益标签的退出开盘严格早于本次09:00判断时点。三策略均值与协方差每日估计，协方差50%收缩至对角，固定二次风险惩罚。余量为现金。没有重训隐藏状态或选择新的窗口。", "",
                 "三条参考的计划持仓经分配形成一个真实的指数目标；实际账户单独结算，不能用参考净值线性相加替代账户。所有政策使用同一MACRO_HMM指数五日尾部估计，因此PRICE_HMM只是分配信息的价格对照，整个账户仍含相同的宏观风险预算信息。", "",
                 "目标仓位至多50%，五日ES95预算2.5%，10%标的跳空风险预算至多权益5%且至多消耗距90%峰值余量的一半。调仓与退出成本计入预算，风险投影之后应用10个百分点调仓带；明确退出或风险超限优先。约束无法保证实际回撤不超过10%。", ""]
        if folder == value.OUT:
            rows += ["原始实验在一条浮点数1.0000000000000002触发严格边界后中断。新目录只将绝对超界不超过1e-12的合成仓位截断到[0,1]，超出容差仍拒绝。原代码、原预测与中断文件保留。全部8账户完成，已完成的等权基础账户逐列重现；没有新拟合或经济参数变更。", "",
                     "该分配允许给目前空仓、未来可能出现信号的参考策略配置资金；这些资金当前留在现金。它同时反映未来五日参考路径的价值与等待机会，不能简单解释成今天必须买入的信号强弱。", ""]
        else:
            rows += ["本次唯一经济变化：参考当日无持仓信号时，其优化权重必须为0；当日有信号才允许分配。保持原预测均值、协方差和效用不变。全无信号或无正效用仍为现金。", "",
                     "这项约束是一项研究假设，并非修复已证实的错误。父策略在744个有参考信号日中有263日分配为0，其中55日存在有信号且预测均值为正的参考；强制只选当日有信号路径，可能提高利用率，也可能失去有价值的等待。", "",
                     f"实际对照显示，主压力夏普由{value_result['primary']['net_sharpe']:.3f}变为{metric['net_sharpe']:.3f}，年化由{value_result['primary']['annualized_return']:.2%}变为{metric['annualized_return']:.2%}。不能用更多参与天数冒充改进。旧父策略及所有对照均保留。", "",
                     "本次仍沿用原无条件参考五日标签，没有解决当前持仓阶段与未来参考收益是否匹配的问题。这个限制不能作为临时挑选参数或挽救本轮结果的理由。", ""]
        rows += ["固定配对比较：", "", *comparison_lines(result), "",
                 "这些区间来自完整同日收益差的20日区块2,000次抽样，未作整个研究家族的选择校正。所有历史已经参加过研究，新增独立观察为0。点估计变化不等于稳定增量或独立验证。", "",
                 "账户现金、分红、T+1、实际买入预算和保存优化结果已核对。没有新行情采集、期权、订单或当前市场判断；资料不覆盖当前决策，保持NO_VIEW。"]
        (folder / "研究结论.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_comparison(value_result, active_result):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    series = [(value.OUT, "MACRO_HMM", value_result["primary"], "宏观状态分配：允许参考等待", "#267D8D", "-"),
              (active.OUT, "MACRO_HMM", active_result["primary"], "宏观状态分配：仅当日有信号", "#B15F42", "-")]
    equal = next(row for row in active_result["all_accounts"] if row["cost"] == "STRESS" and row["policy"] == "EQUAL_EXPERTS")
    series.append((active.OUT, "EQUAL_EXPERTS", equal, "固定等权参考", "#7A8793", "--"))
    sources = []
    for folder, policy, metric, label, color, style in series:
        path = folder / "accounts/STRESS" / f"{policy}_ledger.parquet"
        ledger = pd.read_parquet(path)
        nav = np.r_[200000., ledger.equity.to_numpy(float)]
        dd = (nav / np.maximum.accumulate(nav) - 1)[1:]
        axes[0].plot(ledger.date, ledger.equity / 10000, label=f"{label}｜夏普{metric['net_sharpe']:.3f}，年化{metric['annualized_return']:.2%}", color=color, ls=style, lw=1.6)
        axes[1].plot(ledger.date, dd, color=color, ls=style, lw=1.2)
        sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("账户权益（万元）")
    axes[1].set_ylabel("历史峰值回撤")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].margins(y=.38)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("状态信息用于策略分配：等待是否有价值", x=.09, y=.97, ha="left", fontsize=17, weight="bold")
    fig.text(.09, .927, "2021年1月4日至2026年8月14日；相同预测、资本和风险约束，只改变可分配的当日动作。", fontsize=10.5, color="#56616C")
    fig.text(.09, .045, "初始20万元；压力成本；252日年化；包含空仓、T+1、费用及分红。", fontsize=9)
    fig.text(.09, .022, "重复使用的历史开发比较；没有新增独立验证。", fontsize=9, color="#56616C")
    fig.subplots_adjust(left=.09, right=.97, top=.88, bottom=.11, hspace=.1)
    fig.savefig(active.OUT / "状态分配与等待价值.png", dpi=170)
    plt.close(fig)
    save(active.OUT / "figure_sources.json", {"figure": "状态分配与等待价值.png", "sources": sources})


def main():
    earlier.main()
    state_result = read(earlier.STATE / "result.json")
    factorial = read(earlier.FACTORIAL / "result.json")
    value_result, active_result = read(value.OUT / "result.json"), read(active.OUT / "result.json")
    reports(value_result, active_result)
    plot_comparison(value_result, active_result)
    study_ids = [factorial["study_id"], state_result["study_id"], original.STUDY, value.STUDY, active.STUDY]
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    if mandate.get("current_round") not in study_ids:
        print("本轮历史报告与图已生成；已有后续研究，未覆盖最新项目状态。", flush=True)
        return
    p, a = value_result["primary"], active_result["primary"]
    message = (f"退出/分配四格、持续状态、状态策略价值及当日动作约束四项比较已完成。状态价值压力夏普{p['net_sharpe']:.3f}、年化{p['annualized_return']:.2%}、回撤{abs(p['max_drawdown']):.2%}；"
               f"增加当日可用约束后为{a['net_sharpe']:.3f}/{a['annualized_return']:.2%}/{abs(a['max_drawdown']):.2%}。目标尚未完成，保持进行中。")
    progress = {"at": now(), "study_ids": study_ids, "economic_comparisons": 4,
        "new_unique_full_accounts": 28, "account_computations": 31, "repeat_account_checks": 3,
        "reused_old_terminal_accounts": 2, "new_internal_dependency_accounts": 44, "new_reference_accounts": 3,
        "new_hidden_model_fits": state_result["new_hidden_model_fits"], "strategy_moment_estimations": 4083,
        "additional_constrained_optimizations": 4083, "last_primary": a, "strategy_value_primary": p,
        "goal_achieved": False, "goal_workflow": "ACTIVE_RESEARCH", "new_independent_observations": 0,
        "current_market_view": "NO_VIEW", "orders_authorized": False}
    progress_path = SUMMARY / "state_allocation_progress_20260925.json"
    save(progress_path, progress)
    report = [message, "",
        "本轮的关键发现是：描述市场状态、预测具体策略收益、限制当前可执行动作，是三个不同环节。持续状态直接预测指数收益没有形成正的成本后收益；同一状态用于参考策略价值时有所改善，但未达到门槛；只允许给当天有信号的策略分配资金，实际表现反而更差。不能把更多交易、更复杂模型或更少空仓当成优势。", "",
        f"- 持续状态直接预测指数：{paragraph(state_result['primary'])}。",
        f"- 状态条件化三类策略价值：{paragraph(p)}。",
        f"- 当日有信号才允许分配：{paragraph(a)}。", "",
        "这三项使用相同2021年1月4日至2026年8月14日评价区间和252日年化。退出/分配四格使用原迁移的2020年起区间和242日年化，只能在其内部比较。", "",
        "迁移分解的压力点估计：改变退出学习块的算术年化影响约-0.968个百分点，仅改变参考分配约-0.071个百分点，交互约+0.315个百分点；区间均跨零，不能据此做确定归因。四格只在全部日更格满足完整新训练合同。", "",
        "原版历史压力夏普1.219、年化10.29%、回撤6.51%的重现结论继续保留；它可能包含真实的历史优势，但全家族选择、局部稳健性和独立观察不足仍未解决。此前两年日更0.636、共享训练0.916、风险先行交易带0.954及五日标签0.712也继续保留，均未满足全部目标。", "",
        "本轮新增28个末端策略与成本组合、44个内部依赖账户、3个参考账户；另有3次已完成账户重复核对。实际拟合2,722个隐藏状态模型，计算4,083组策略条件矩和4,083组追加受限优化；这些不是独立试验样本。新增独立行情为0。", "",
        "数值修正只处理一条1.0000000000000002的浮点超界，保留原中断证据和原冻结代码，复用全部已拟合结果。策略变更的当日动作约束单独立项，未混入修复。", "",
        "完整来源：", "",
        "- reports/research/510300_selected_mix_migration_factorial_v1/研究结论.md",
        "- reports/research/510300_index_persistent_state_daily_v1/研究结论.md",
        "- reports/research/510300_index_state_strategy_value_saved_completion_v1/研究结论.md",
        "- reports/research/510300_index_state_active_allocation_v1/研究结论.md", "",
        "复用运行和查询入口：scripts/run_510300_index_state_research.ps1。已完成步骤复用保存结果；不自动覆盖有中断记录的研究。", "",
        "仍按20万元、净夏普1.2、年化10%、最大回撤10%及当前尾部预算评估。只有510300与现金，采集暂停，当前判断NO_VIEW。研究整体目标保持进行中，尚未找到满足新合同且经过独立验证的策略。"]
    (SUMMARY / "最新研究结论.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    current_path = SUMMARY / "current_status.json"
    current = read(current_path)
    completed = current.get("completed_followup_studies", [])
    for study_id in study_ids:
        if study_id not in completed:
            completed.append(study_id)
    current.update(updated_at=now(), goal_status="active", goal_achieved=False, completed_followup_studies=completed,
        latest_user_requested_study=active.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=active_result["status"],
        latest_goal_turn_classification="PROGRESS_FOUR_FIXED_STATE_AND_EXECUTION_COMPARISONS_COMPLETED",
        latest_state_allocation_progress=progress, same_condition_consecutive_no_progress_goal_turns=0,
        active_blocker_id=None, active_blocker_description=None, report="最新研究结论.md",
        goal_metadata_note="目标工具状态为active；本轮有实际研究进展，未宣布目标达成。",
        remaining_research_question="现有状态对可执行收益的增量尚未建立；当日动作可用性约束未改善。仍需可重复的收益来源和未用于选择的验证。")
    save(current_path, current)
    mandate.update(last_research_result=message, goal_achieved=False, latest_progress_receipt=progress_path.relative_to(ROOT).as_posix())
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    old = status_path.read_text(encoding="utf-8")
    marker, end_marker = "<!-- INDEX_STATE_COMPLETION_20260925 -->", "<!-- END_INDEX_STATE_COMPLETION_20260925 -->"
    if marker in old and end_marker in old:
        begin, end = old.index(marker), old.index(end_marker) + len(end_marker)
        old = old[:begin] + old[end:].lstrip("\n")
    banner = marker + "\n\n> 2026-09-25 " + message + " 详见[最新研究结论](reports/research/510300_integrated_research_continuation_20260924/最新研究结论.md)。\n\n" + end_marker + "\n\n"
    status_path.write_text(banner + old, encoding="utf-8")
    print("本轮四项经济比较及数值完成记录已交付，目标保持进行中且未达成。", flush=True)


if __name__ == "__main__":
    main()
