"""整理迁移分解和持续状态研究的实际结果，不改写冻结策略或旧失败。"""
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

FACTORIAL = ROOT / "reports/research/510300_selected_mix_migration_factorial_v1"
STATE = ROOT / "reports/research/510300_index_persistent_state_daily_v1"
SUMMARY = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
NAMES = {"HISTORY": "同窗历史分布", "PRICE_HMM": "价格持续状态",
         "MACRO_EMISSION": "宏观条件、不继承状态", "MACRO_HMM": "宏观持续状态"}


def main():
    diagnostic, result = read(FACTORIAL / "result.json"), read(STATE / "result.json")
    p = result["primary"]
    models = read(STATE / "results/saved_models.json")
    predictions = pd.read_parquet(STATE / "results/predictions.parquet")
    x = pd.read_parquet(STATE / "inputs/decision_information.parquet")
    x = x.loc[x.date.between("2021-01-04", "2026-08-14")]
    passed = p["historical_point_targets_met"]
    rows = [f"持续状态实验完成。主候选{'达到历史点值门槛，仍待独立验证' if passed else '未达到高夏普目标'}：压力夏普{p['net_sharpe']:.3f}、复合年化{p['annualized_return']:.2%}、最大回撤{abs(p['max_drawdown']):.2%}，20万元变为{p['ending_equity']:,.2f}元。", "",
            "评价为2021年1月4日至2026年8月14日的完整账户，包括空仓、亏损、费用、分红及T+1限制。此研究为与已有宏观共同资料匹配而采用该区间，按252日年化；不能与2020年起、按242日年化的原图迁移直接排序。", "",
            "本轮四个固定压力账户：", ""]
    for row in result["all_accounts"]:
        if row["cost"] == "STRESS":
            rows.append(f"- {NAMES[row['policy']]}：夏普{row['net_sharpe']:.3f}，年化{row['annualized_return']:.2%}，回撤{abs(row['max_drawdown']):.2%}，费用{row['cost_cny']:,.2f}元。")
    rows += ["", "本轮检验的是状态持续性的增量：过去状态及转移能否帮助解释同样当前形态的不同后续结果。双状态模型没有被命名为已知牛熊，也没有用未来收益给状态改标签。两状态的高斯均值、方差及转移每天在最近两日历年内重新估计，固定40次参数更新，没有更换初值挑最好结果，也没有沿用包含过期资料的旧参数。", "",
             "价格变量只有五日抛压、二十日趋势和短长波动率比；宏观只增加社融同比三月变化及DR007相对七日逆回购政策利率。二者分别是信用变化与资金利差代理，不等于NFCI或名义利率减中性利率。发布时钟沿用已有可用性记录，历史供应商首次交付时点并未因此获得独立认证。", "",
             "MACRO_EMISSION与主候选使用同一个拟合模型，但只用平稳先验乘当日特征似然，不沿用昨日状态概率；主候选则沿时间前向更新。条件收益分布使用当时五日标签已成熟的行，按前向状态概率加权，未用事后平滑状态给历史收益分组。加权有效行数不足126时回到同窗历史分布。", "",
             f"实际完成{result['new_hidden_model_fits']:,}个模型拟合、{result['daily_predictions']:,}条分布预测和8条完整账户。每天拟合的两个模型对应价格与宏观，40次更新是单个模型内部计算，不能算成40个独立模型。评价期内信用可用时间共{x.credit_available_at.nunique()}个不同值；每日沿用同一宏观公告不会增加独立信息。", "",
             "主候选相对各对照的压力账户算术年化收益差：", ""]
    for row in result["comparisons"]:
        rows.append(f"- 相对{NAMES[row['right']]}：{row['annual_arithmetic_difference']*100:+.3f}个百分点，95%区间[{row['lower_95']*100:+.3f}, {row['upper_95']*100:+.3f}]个百分点。")
    rows += ["", "区间为同日收益差的20日区块2,000次计算，未经完整研究家族选择校正。预测均方误差、尾部分位覆盖和完整账户分别保存，不能以状态图好看或预测误差下降替代成本后账户优势。", "",
             "四政策使用同一事前效用和风险规则：目标仓位至多50%、五日ES95损失预算为权益2.5%、标的10%跳空损失不超过权益5%且不超过距90%历史峰值余量的一半。调仓与未来退出费用一并预算。09:00确定计划，开盘只允许因超预算缩小买入，不能利用开盘结果扩大方向判断。回撤触发10%后申请退出，本轮不恢复。", "",
             "这些限制不能保证实际最大回撤：价格跳空、T+1和涨跌停会影响成交。末日正常收盘计价，仍有持仓时扣除压力清算费用储备；这笔储备不冒称实际卖出成交。", "",
             f"主候选全部滚动两年三项目标通过窗口数为{result['rolling_two_year_joint_passes']['MACRO_HMM']}；固定五日相位的尾部监控记录{result['tail_monitor_alerts']['MACRO_HMM']}次观察报警。重叠窗口及报警均不自动带来新的交易规则。", "",
             "必要检查完成：小样本枚举全部隐藏状态路径核对概率计算、填充行不产生转移、前向状态不读取随后观察、保存权重和分布复算、未成熟标签扰动、现金/分红/T+1及实际买入的尾部预算。", "",
             "本轮所有历史此前均已参加研究，新增独立行情为0；保持NO_VIEW，不输出当前仓位观点。未采集新行情，没有期权、自动交易、审核ZIP或用户汇总表格。"]
    (STATE / "研究结论.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {"HISTORY": "#798998", "PRICE_HMM": "#387694", "MACRO_EMISSION": "#BD8555", "MACRO_HMM": "#67509B"}
    sources = []
    for row in result["all_accounts"]:
        if row["cost"] != "STRESS":
            continue
        policy = row["policy"]
        path = STATE / "accounts/STRESS" / f"{policy}_ledger.parquet"
        ledger = pd.read_parquet(path)
        nav = np.r_[200000., ledger.equity.to_numpy(float)]
        dd = (nav / np.maximum.accumulate(nav) - 1)[1:]
        axes[0].plot(ledger.date, ledger.equity / 10000, label=f"{NAMES[policy]}｜夏普{row['net_sharpe']:.3f}，年化{row['annualized_return']:.2%}", color=colors[policy], lw=1.5)
        axes[1].plot(ledger.date, dd, color=colors[policy], lw=1.1)
        sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("账户权益（万元）")
    axes[1].set_ylabel("历史峰值回撤")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].margins(y=.4)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("两年日更：宏观信息与状态记忆的压力账户比较", x=.09, ha="left", y=.97, fontsize=16, weight="bold")
    fig.text(.09, .927, "2021年1月4日至2026年8月14日；各政策使用相同资本、费用和尾部预算。", fontsize=11, color="#56616C")
    fig.text(.09, .045, "初始20万元；年化252日；现金与无风险收益0；包含空仓、T+1、佣金、滑点和分红。", fontsize=9)
    fig.text(.09, .022, "全部历史已被研究；这些结果不构成独立前向验证。", fontsize=9, color="#56616C")
    fig.subplots_adjust(left=.09, right=.97, top=.88, bottom=.11, hspace=.1)
    fig.savefig(STATE / "持续状态压力账户比较.png", dpi=170)
    plt.close(fig)
    save(STATE / "figure_sources.json", {"figure": "持续状态压力账户比较.png", "sources": sources})

    permitted = {"510300_SELECTED_MIX_MIGRATION_FACTORIAL_V1", "510300_INDEX_PERSISTENT_STATE_DAILY_V1",
                 "510300_INDEX_STATE_STRATEGY_VALUE_V1", "510300_INDEX_STATE_STRATEGY_VALUE_SAVED_COMPLETION_V1",
                 "510300_INDEX_STATE_ACTIVE_ALLOCATION_V1"}
    current_mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    if current_mandate.get("current_round") not in permitted:
        print("历史持续状态报告与图已生成；当前存在后续研究，未回退最新进度。", flush=True)
        return

    progress = {"at": now(), "studies": [diagnostic["study_id"], result["study_id"]],
                "new_full_accounts": 14, "reused_full_accounts": 2, "new_internal_dependency_accounts": 44,
                "new_model_fits": result["new_hidden_model_fits"], "latest_primary": p,
                "goal_achieved": False, "goal_workflow": "ACTIVE_RESEARCH", "new_independent_observations": 0}
    save(SUMMARY / "persistent_state_progress_20260925.json", progress)
    message = (f"继续完成退出／分配四格诊断及宏观持续状态实验。主持续状态压力夏普{p['net_sharpe']:.3f}、年化{p['annualized_return']:.2%}、回撤{abs(p['max_drawdown']):.2%}。"
               + ("历史点值通过，独立验证未完成。" if passed else "当前三项目标未同时完成。"))
    report = [message, "", "迁移分解的压力点估计显示：改变退出学习块的算术年化影响约-0.968个百分点，仅改变参考分配约-0.071个百分点，交互项约+0.315个百分点；三项区间均跨零，不能据此宣称确定归因。所有格统一当前末端风险预算，只有全部日更格符合完整合同。", "",
              "继续检验的持续状态实验及全部对照见：reports/research/510300_index_persistent_state_daily_v1/研究结论.md。",
              "迁移诊断见：reports/research/510300_selected_mix_migration_factorial_v1/研究结论.md。", "",
              f"本次新增14条末端账户、44条内部依赖账户和{result['new_hidden_model_fits']:,}个模型拟合；新增独立行情0。两个研究的评价日历不同，不能把绩效直接排序。", "",
              "原版历史夏普1.219及此前两年日更0.636、共享退出0.916、执行顺序0.954、五日标签0.712的结果继续保留。训练、实施和诊断完成并不等于高夏普目标完成。", "",
              "当前仍按20万元、净夏普1.2、年化10%、最大回撤10%、50%目标仓位及既有尾部预算推进；仅510300与现金，不采集，NO_VIEW。整体目标保持进行中，未宣称找到已独立验证的策略。"]
    (SUMMARY / "最新研究结论.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    current = read(SUMMARY / "current_status.json")
    completed = current.get("completed_followup_studies", [])
    for study in progress["studies"]:
        if study not in completed:
            completed.append(study)
    current.update(updated_at=now(), goal_status="active", goal_achieved=False,
                   completed_followup_studies=completed, latest_user_requested_study=STATE.relative_to(ROOT).as_posix(),
                   latest_user_requested_study_status=result["status"], latest_persistent_state_progress=progress,
                   latest_goal_turn_classification="PROGRESS_FACTORIAL_AND_PERSISTENT_STATE_EXPERIMENTS_COMPLETED",
                   same_condition_consecutive_no_progress_goal_turns=0, active_blocker_id=None, active_blocker_description=None,
                   report="最新研究结论.md", remaining_research_question="当前条件下仍需证明可重复的成本后收益来源。")
    save(SUMMARY / "current_status.json", current)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(last_research_result=message, goal_achieved=False,
                   latest_progress_receipt=(SUMMARY / "persistent_state_progress_20260925.json").relative_to(ROOT).as_posix())
    save(mandate_path, mandate)
    status = ROOT / "RESEARCH_STATUS.md"
    existing = status.read_text(encoding="utf-8")
    marker = "<!-- PERSISTENT_STATE_PROGRESS_20260925 -->"
    if marker not in existing:
        status.write_text(marker + "\n\n> 2026-09-25 " + message + " 详见[持续状态研究](reports/research/510300_index_persistent_state_daily_v1/研究结论.md)。\n\n" + existing, encoding="utf-8")
    print("迁移诊断与持续状态结果已整理，项目状态保持目标未完成。", flush=True)


if __name__ == "__main__":
    main()
