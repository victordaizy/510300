"""保存资金定盘分化研究的中文结果、账户图及本次连续研究进展。"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now

OUT = ROOT / "reports/research/510300_repo_segmentation_daily_v1"
SOURCE = ROOT / "reports/research/510300_repo_fixing_segmentation_source_v1"
HUIJIN = ROOT / "reports/research/510300_huijin_etf_event_saved_completion_v1"
MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
PRIMARY = "PRICE_AND_REPO_SEGMENTATION"


def run():
    result, source = read(OUT / "result.json"), read(SOURCE / "result.json")
    huijin = read(HUIJIN / "completed_round.json")
    primary = result["primary"]
    baseline = next(row for row in result["all_accounts"] if row["policy"] == "PRICE" and row["cost"] == "STRESS")
    pair = next(row for row in result["comparisons"] if row["cost"] == "STRESS")
    pred_all, pred_down = result["forecast_evaluation"]
    gross = []
    accounts = {}
    for name in ["PRICE", PRIMARY]:
        ledger = pd.read_parquet(OUT / "accounts/STRESS" / name / "ledger.parquet")
        accounts[name] = ledger
        row = {"policy": name, "same_share_path_gross_pnl": float((ledger.price_pnl+ledger.dividend_recognized).sum()),
               "actual_execution_cost": float((ledger.commission+ledger.slippage_cost).sum()),
               "terminal_exit_reserve": float(ledger.terminal_exit_reserve.iloc[-1]),
               "net_profit": float(ledger.equity.iloc[-1]-200000),
               "description": "固定已经成交的份额路径，仅做会计归因；不是另一个零成本策略账户。"}
        assert abs(row["same_share_path_gross_pnl"]-row["actual_execution_cost"]-row["terminal_exit_reserve"]-row["net_profit"]) < 1e-6
        gross.append(row)
    save(OUT / "results/saved_account_cost_attribution.json", gross)
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {"PRICE": "#367bb5", PRIMARY: "#b45a3c"}
    labels = {"PRICE": "价格信息", PRIMARY: "价格信息＋回购定盘差"}
    for name, ledger in accounts.items():
        axes[0].plot(ledger.date, ledger.equity/10000, color=colors[name], linewidth=1.6, label=labels[name])
        equity = np.r_[200000., ledger.equity.to_numpy()]
        dd = (equity/np.maximum.accumulate(equity)-1)[1:]*100
        axes[1].plot(ledger.date, dd, color=colors[name], linewidth=1.2)
    axes[0].axhline(20, color="#6f7782", linewidth=.8, linestyle="--")
    axes[0].set_ylabel("账户权益（万元）", fontproperties=font)
    axes[1].set_ylabel("回撤（%）", fontproperties=font)
    axes[0].set_title("新增资金分化信息：误差略降，完整账户收益未改善", fontproperties=font, fontsize=16, loc="left", pad=18)
    axes[0].legend(prop=font, frameon=False, loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dfe3e8", linewidth=.7)
    fig.text(.08, .015, "20万元连续账户｜2020-01-02—2026-09-24｜压力费用、尾部预算、空仓日全部计入｜历史开发证据", fontproperties=font, fontsize=10, color="#586271")
    fig.tight_layout(rect=(.025, .045, .99, .99))
    figure = OUT / "完整账户比较.png"
    fig.savefig(figure, dpi=140, facecolor="white")
    plt.close(fig)
    report = f"""本次连续研究已完成两项新证据工作和8个完整账户，仍未找到符合要求的高夏普策略。汇金披露规则未通过；随后新增的FR007-FDR007定盘差，在价格弱势样本中只带来{pred_down['mse_improvement_fraction']:.2%}的预测均方误差点值下降，完整压力账户夏普{primary['sharpe']:.6f}、年化{primary['annual_return']:.4%}、最大回撤{abs(primary['max_drawdown']):.4%}，没有超过价格基准。总目标保持进行中。

当前要求保持为20万元、仅510300与现金、压力成本净夏普至少1.2、复合年化至少10%、最大回撤不超过10%；训练只用最近两个日历年的成熟标签，每天更新。最大目标仓位50%、5日ES95预算2.5%、10%假设跳空预算5%，并受账户距90%峰值剩余空间的一半约束。已估计调仓和退出成本计入风险预算；T+1与跳空意味着预算不是最终损失保证。

第一项：汇金ETF公开披露。

当前官网2012—2026年15个目录的114篇文章已经补齐，ETF相关8篇，包含6篇已买入相关、1篇出售及1篇仅承诺。合并重复披露后，2015年以来账户期只有4个独立买入事件。公告参与与公告后价格确认各做两档成本，4个账户全部完成；压力夏普分别{huijin['control_stress']['sharpe']:.6f}和{huijin['primary']['sharpe']:.6f}，期末分别{huijin['control_stress']['end_equity']:.2f}和{huijin['primary']['end_equity']:.2f}元。2015年增持公告后的继续下跌被完整保留，两规则该次均亏6248.20元。

其完整结论见[汇金新证据与账户结论](<{(HUIJIN / '汇金新证据与账户结论.md').as_posix()}> )。这批资料只覆盖当前官网目录，没有认证历史首版，也不包含精确510300每日买卖数量。

第二项：同上午回购定盘差。

官方FR007与FDR007采用不同参与者和抵押品范围的上午成交定盘。二者之差可作为资金市场分化的一个代理，不能等同于纯非银融资成本，也不是全天加权R007-DR007，更不能解释成名义利率减中性利率。[中国货币网编制方法](https://www.chinamoney.com.cn/chinese/bkfrr/)

这次从同一官方JSON的两个字段直接提取，共{source['rows']}日，2017-05-31至2026-09-24；原2305个FDR数值精确一致，额外下载补齐23日。按官方11:30起发布的方法，历史重建用当日12:00计划可用时点，再合并到下一个A股执行日09:00；历史实际首次送达仍未认证。没有把数据采集日写成历史发布日期，没有在FDR推出前补值。

价格基准固定使用五日抛压、二十日趋势、短长波动比；新增方案只加入一个FR007-FDR007差值。每天重建最近两日历年、退出开盘已严格早于当日的训练池，固定至少252条共同成熟样本和126近邻经验分布，参数没有扫描。主比较两方案使用同样资料覆盖、同样整手/T+1/分红账本、同样压力成本事前预算，仅允许五日弱势时新增库存。

实际保存{result['daily_distribution_updates']}份分布更新，即{result['unique_prediction_days']}个日期、每日期3个模型口径；其中HISTORY是同池经验均值对照。共有{result['mature_prediction_days']}个成熟评价日期，五日弱势样本793个；这些标签重叠，不能按同等数量独立交易计数。全部成熟样本MSE点值改善{pred_all['mse_improvement_fraction']:.2%}，弱势样本改善{pred_down['mse_improvement_fraction']:.2%}；在弱势样本中，新增方案的MSE仍高于简单历史均值，未建立稳定预测优势。

完整账户覆盖2020-01-02至2026-09-24的{primary['days']}个交易日，包括9个无完整预测日期和全部空仓日，不重置年度本金。两个候选、两档成本的4个账户全部完成，压力结果为：

- 价格基准：夏普{baseline['sharpe']:.6f}、复合年化{baseline['annual_return']:.4%}、最大回撤{abs(baseline['max_drawdown']):.4%}，期末{baseline['end_equity']:.2f}元；42个已闭合周期，另有1个未闭合周期。
- 加入定盘差：夏普{primary['sharpe']:.6f}、复合年化{primary['annual_return']:.4%}、最大回撤{abs(primary['max_drawdown']):.4%}，期末{primary['end_equity']:.2f}元；45个已闭合周期，另有1个未闭合周期。

压力账户算术年化增量为{pair['annual_arithmetic_difference']*100:.4f}个百分点，95%区块区间[{pair['lower_95']*100:.4f}, {pair['upper_95']*100:.4f}]个百分点；2020—2023和2024—末端两个固定时期的增量都为负。主方案所有滚动两年联合通过次数为0。区块区间未做项目内全部尝试的选择校正。

账户差距主要来自实际持仓路径的毛损益下降：价格方案毛损益{gross[0]['same_share_path_gross_pnl']:.2f}元，新增信息方案{gross[1]['same_share_path_gross_pnl']:.2f}元，相差{gross[0]['same_share_path_gross_pnl']-gross[1]['same_share_path_gross_pnl']:.2f}元；两者实际交易费用相差仅{gross[1]['actual_execution_cost']-gross[0]['actual_execution_cost']:.2f}元。这个拆分固定已经成交的份额路径，只用于会计解释，不是零成本可交易账户，也不把两方案不同的份额路径视为严格因果识别。预测误差略降没有转化为更好的持仓选择。

![压力成本完整账户](<{figure.as_posix()}>)

所有保存分布均从原入选样本复算一致；两处截断未来标签后判断相同，四账户现金、分红、T+1、调仓费用与入场尾部预算通过一致性检查。只用已成熟的固定相位五日标签建立了风险退化监控，最近60条分位突破率超过15%才报警；本样本没有触发。没有触发只表示该特定监控条件未出现，不能当作策略有效性证明，也没有据结果反调监控门槛。

本次连续研究合计8个新的完整账户；汇金2853份风险估计和本项4872份分布更新各自记录，不能当成独立市场验证。新增独立前向观察仍为0。本次两个固定用途均不晋升，原失败保留。此前按原规则追加行情后的旧版压力夏普1.2017、年化10.15%、回撤6.51%的历史结果继续保留；它与当前两年日更、账户尾部预算合同不同，原始选择偏差尚未解除。

当前高夏普目标仍未完成。已完成的研究代码、协议、原始来源、逐日预测及账户都保存在各目录；没有制作审核包或用户表格，没有恢复旧自动采集或执行订单。
"""
    (OUT / "本轮新证据与账户结论.md").write_text(report, encoding="utf-8")
    completed = {"at": now(), "status": "TWO_NEW_INFORMATION_STUDIES_AND_EIGHT_ACCOUNTS_COMPLETED_TARGET_UNMET",
                 "studies": [*huijin["studies"], "510300_REPO_FIXING_SEGMENTATION_SOURCE_V1", "510300_REPO_SEGMENTATION_DAILY_V1"],
                 "new_unique_full_accounts": huijin["new_full_accounts"]+result["new_full_accounts"],
                 "official_huijin_documents": 114, "independent_huijin_buy_episodes_in_account_period": 4,
                 "paired_repo_fixing_days": source["rows"], "new_repo_fixing_dates": source["additional_rows"],
                 "huijin_daily_tail_estimates": 2853, "repo_daily_distribution_updates": result["daily_distribution_updates"],
                 "unique_repo_prediction_days": result["unique_prediction_days"], "new_parameter_searches": 0,
                 "primary": primary, "price_baseline_stress": baseline, "huijin_primary": huijin["primary"],
                 "goal_status": "active", "goal_achieved": False, "independent_forward_observations": 0,
                 "review_package_created": False, "orders_authorized": False}
    save(OUT / "completed_round.json", completed)
    relative = OUT.relative_to(ROOT).as_posix()
    status_path = MAIN / "current_status.json"
    status = read(status_path)
    for name in completed["studies"]:
        if name not in status["completed_followup_studies"]:
            status["completed_followup_studies"].append(name)
    status.update(updated_at=now(), goal_status="active", goal_achieved=False,
                  latest_goal_turn_classification="PROGRESS_NEW_OFFICIAL_INFORMATION_AND_EIGHT_FULL_ACCOUNTS",
                  same_condition_consecutive_no_progress_goal_turns=0,
                  latest_new_information_round=completed,
                  latest_user_requested_study=relative, latest_user_requested_study_status=completed["status"],
                  research_execution_state=completed["status"], latest_continuation_report=relative+"/本轮新证据与账户结论.md",
                  active_blocker_id=None, active_blocker_description=None,
                  goal_metadata_note="目标active；本轮来源和8个完整账户为实质进展，目标未完成。",
                  remaining_research_question="新披露和资金定盘差的固定用途均未通过；现行两年日更与尾部预算下仍须找到可执行收益优势及独立验证。")
    save(status_path, status)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(latest_progress_receipt=relative+"/completed_round.json", latest_integrated_experiment=result["study_id"],
                   last_research_result="两项新证据与8个完整账户已完成；披露信号和资金定盘差均未达标，高夏普目标仍active。",
                   latest_continuation_report=relative+"/本轮新证据与账户结论.md",
                   latest_continuation_classification=status["latest_goal_turn_classification"],
                   research_execution_state=completed["status"], goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    banner = ("> 2026-09-25 新披露与资金定盘差连续研究完成：汇金114篇官方文章、4独立买入事件，新增FR007-FDR007共同2328日、补23日，"
              "共8个新完整账户。定盘差主压力夏普-0.097、年化-0.304%、回撤6.84%，弱势样本MSE略降0.52%但账户收益未改善。"
              f"目标ACTIVE、未完成，独立前向0；见[本轮结论]({relative}/本轮新证据与账户结论.md)。\n\n")
    global_file = ROOT / "RESEARCH_STATUS.md"
    previous = global_file.read_text(encoding="utf-8-sig")
    if "新披露与资金定盘差连续研究完成" not in previous:
        global_file.write_text(banner+previous, encoding="utf-8")
    print("两项新证据、8个完整账户及中文结果图已保存，主线状态为active且未达标。", flush=True)


if __name__ == "__main__":
    run()
