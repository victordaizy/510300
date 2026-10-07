"""保存香港人民币下午融资变化的研究结论与主线进度，不改冻结实验。"""
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

OUT = ROOT / "reports/research/510300_hkma_afternoon_liquidity_daily_v1"
SOURCE = ROOT / "reports/research/510300_hkma_rmb_liquidity_source_v1"
MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
PRIMARY = "PRICE_AND_AFTERNOON_FUNDING"


def run():
    result = read(OUT / "result.json")
    source = read(SOURCE / "result.json")
    assert result["status"] == "FROZEN_NO_QUALIFIED_HKMA_LIQUIDITY_STRATEGY"
    assert result["new_full_accounts"] == 4
    primary = result["primary"]
    control = next(row for row in result["all_accounts"] if row["policy"] == "PRICE" and row["cost"] == "STRESS")
    pair = next(row for row in result["comparisons"] if row["cost"] == "STRESS")
    pred_all, pred_down = result["forecast_evaluation"]
    accounts, attribution = {}, []
    for name in ["PRICE", PRIMARY]:
        ledger = pd.read_parquet(OUT / "accounts/STRESS" / name / "ledger.parquet")
        accounts[name] = ledger
        value = {
            "policy": name,
            "same_share_path_gross_pnl": float((ledger.price_pnl + ledger.dividend_recognized).sum()),
            "actual_execution_cost": float((ledger.commission + ledger.slippage_cost).sum()),
            "terminal_exit_reserve": float(ledger.terminal_exit_reserve.iloc[-1]),
            "net_profit": float(ledger.equity.iloc[-1] - 200000),
            "scope": "固定已经成交的份额路径做会计归因，不是零成本策略，也不是不同份额路径的因果识别。",
        }
        assert abs(value["same_share_path_gross_pnl"] - value["actual_execution_cost"] - value["terminal_exit_reserve"] - value["net_profit"]) < 1e-6
        attribution.append(value)
    save(OUT / "results/saved_account_cost_attribution.json", attribution)

    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, color, label in [("PRICE", "#367bb5", "同样资料覆盖的价格对照"),
                                (PRIMARY, "#ae5748", "价格＋香港人民币下午融资变化")]:
        ledger = accounts[name]
        axes[0].plot(ledger.date, ledger.equity / 10000, color=color, linewidth=1.6, label=label)
        equity = np.r_[200000., ledger.equity.to_numpy()]
        dd = (equity / np.maximum.accumulate(equity) - 1)[1:] * 100
        axes[1].plot(ledger.date, dd, color=color, linewidth=1.2)
    axes[0].axhline(20, color="#737b83", linewidth=.8, linestyle="--")
    axes[0].set_title("下午融资变化未改善510300完整账户", fontproperties=font, fontsize=16, loc="left", pad=18)
    axes[0].set_ylabel("账户权益（万元）", fontproperties=font)
    axes[1].set_ylabel("回撤（%）", fontproperties=font)
    axes[0].legend(prop=font, frameon=False, loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dfe3e8", linewidth=.7)
    fig.text(.08, .015, "20万元｜2020-01-02—2026-09-24｜两年训练、每日更新｜压力费用与全部空仓日｜历史开发证据", fontproperties=font, fontsize=10, color="#586271")
    fig.tight_layout(rect=(.025, .045, .99, .99))
    figure = OUT / "完整账户比较.png"
    fig.savefig(figure, dpi=140, facecolor="white")
    plt.close(fig)

    period_before, period_after = pair["fixed_periods"]
    report = f"""本轮已完成香港金管局新来源与4个完整账户。新增的人民币日间设施下午使用变化没有改善510300策略：主压力账户夏普{primary['sharpe']:.6f}、复合年化{primary['annual_return']:.4%}、最大回撤{abs(primary['max_drawdown']):.4%}，20万元变为{primary['end_equity']:.2f}元。固定实验不予晋升，高夏普总目标仍未完成。

当前要求保持为仅510300与现金、20万元、压力成本净夏普至少1.2、复合年化至少10%、最大回撤不超过10%；仅用最近两个日历年的成熟标签，每个交易日更新。单笔3:1不再是筛选门槛；沿用50%目标仓位上限、5日ES95为权益2.5%、标的10%跳空损失预算为权益5%且至多消耗距90%权益峰值余量的一半。调仓与退出费用计入预算。T+1和实际开盘跳空意味着预算不保证最终损失。

本轮的新证据是什么。

取得官方API共{source['api_pages']}个全量页面、{source['rows']}个日期，覆盖2016-11-01至2026-09-24；另保留1个先前探测页面及8份官方说明或日档案。每个日期的14时和16时日间回购使用金额均完整。最初探测包含2026-09-25数据，但该日期没有进入历史特征或账户评价。金额正变化883日、零变化553日、负变化1007日。

官方档案显示设施额度曾变化。2025-10-09起，日间额度从200亿元改为300亿元，隔夜额度从200亿元改为100亿元，并新增期限安排。因而不能用固定额度除全部历史金额，也不能把API旧字段相加当成全部设施使用量。[香港金管局数据字段](https://apidocs.hkma.gov.hk/documentation/market-data-and-statistics/daily-monetary-statistics/usage-rmb-liquidity-fac/)，[2025年规则通知](https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20250926-4-EN/20250926-4-EN.pdf)。

新增信息在接触候选策略收益前固定为“(16时金额−14时金额)/(1＋16时金额＋14时金额)”，金额单位百万元，均零时结果为零。不加入隔夜、PLP、额外期限或第二个新指标。它记录下午借入余额的变化，可能同时受结算需求、融资条件及政策供给影响，不能直接解释成A股资金流或纯粹市场恐慌。使用同日下午变化减少跨日额度差异的机械影响，不能彻底消除制度变化。

来源按次日00:00计划可用时点合并到A股执行日09:00，来源年龄限1至7个自然日，严格排除当天16时的未来信息。历史首版和实际首次送达没有认证，属于历史开发重建；新增下载并不产生新的独立前向验证。

实验如何保持可比。

价格对照固定使用五日抛压、二十日趋势和短长波动比；主方案仅增加这一个下午变化信息。两者使用相同可用日期和成熟训练池，训练内标准化并截断至[-5,5]、固定126近邻、至少252共同成熟标签，未扫描特征方向、窗口、近邻数或阈值。HISTORY仅作为同池经验分布的预测参考。每日09:00确定计划，开盘只在现金和预算需要时缩小买入份额，五日价格弱势时才增加库存；整手、T+1、分红、费用与全部空仓日均纳入完整账户。

本次共同资料覆盖与上轮FR007-FDR007不同，所以本轮价格对照不等于上轮价格对照。有效比较是本轮两方案的同日差异，不能跨轮挑选较弱基准。共保存{result['daily_distribution_updates']}份分布，对应{result['unique_prediction_days']}个日期、每日期3种模型；成熟评价日期{result['mature_prediction_days']}个，五日弱势样本797个，重叠标签不视为独立交易。指标延续既有研究的242日年化、现金收益及无风险基准按零计算的口径；没有把当期另行估计的存款利息加入账户，也没有把低波动现金收益包装成策略优势。

结果与实际含义。

- 价格对照的压力账户：夏普{control['sharpe']:.6f}，复合年化{control['annual_return']:.4%}，最大回撤{abs(control['max_drawdown']):.4%}，期末{control['end_equity']:.2f}元；44个已闭合周期、1个未闭合周期。
- 加入下午融资变化：夏普{primary['sharpe']:.6f}，复合年化{primary['annual_return']:.4%}，最大回撤{abs(primary['max_drawdown']):.4%}，期末{primary['end_equity']:.2f}元；41个已闭合周期，末端无仓位。

新增方案全部成熟样本的均方误差上升{-pred_all['mse_improvement_fraction']:.2%}，弱势样本上升{-pred_down['mse_improvement_fraction']:.2%}。两口径也没有超过简单历史均值的均方误差。分位损失同时变差，未发现收益或尾部预测的增量证据。

完整压力账户的算术年化收益增量为{pair['annual_arithmetic_difference']*100:.4f}个百分点，预先固定的20日区块区间为[{pair['lower_95']*100:.4f}, {pair['upper_95']*100:.4f}]个百分点，包含零。2020—2023及2024—末端两个时期的点值增量分别为{period_before['annual_arithmetic_difference']*100:.4f}和{period_after['annual_arithmetic_difference']*100:.4f}个百分点，均为负；区间未对项目内全部尝试作选择校正，不能据此断言该信息在所有用途下都无效。所有滚动两年联合通过次数为{result['primary_rolling_two_year_joint_passes']}。

固定已成交份额做会计分解，价格对照的毛损益为{attribution[0]['same_share_path_gross_pnl']:.2f}元、实际交易成本{attribution[0]['actual_execution_cost']:.2f}元、末端退出储备{attribution[0]['terminal_exit_reserve']:.2f}元；新增方案的毛损益为{attribution[1]['same_share_path_gross_pnl']:.2f}元、实际交易成本{attribution[1]['actual_execution_cost']:.2f}元。主方案成本更少仍然亏损，费用并不是这次失败的全部原因。该分解不是零成本交易策略，不用来调低费用救回结果。

![压力成本连续账户](<{figure.as_posix()}>)

2025-10-09制度变化前后指标按事前安排分别保存，仅描述，不据结果选择起点或拼接策略。保存的全部{result['saved_distributions_recomputed']}份分布已从入选训练样本复算；两处未来标签截断检查通过；四账户现金、分红、T+1及调仓后风险预算一致。固定的已成熟五日相位风险监控未报警，它只监测分位突破，不能证明均值优势存在。

关于旧版夏普约1.2的判断保持不变：旧规则追加行情后的压力夏普1.2017、年化10.15%、回撤6.51%是真实保存的历史模拟点值，不应直接断言全部是过拟合；它也没有获得独立验证，且与当前两年日更及尾部预算合同不同。旧退出与当前尾部预算的既有对照年化约4.37%，说明只改变更新频率不能恢复原年化。本轮没有重新搜索旧版参数。

本轮只计4个新完整账户，上轮8个不重复计入。新独立前向观察仍为0，尚无满足现行要求并获得独立验证的策略。研究结果、代码、来源与账户已保存；总目标保持active且未完成，当前市场观点保持NO_VIEW。本轮不制作审核包或用户表格。
"""
    report_path = OUT / "本轮新证据与账户结论.md"
    report_path.write_text(report, encoding="utf-8")
    completed = {
        "at": now(),
        "status": "NEW_HKMA_INFORMATION_AND_FOUR_ACCOUNTS_COMPLETED_TARGET_UNMET",
        "studies": [source["study_id"], result["study_id"]],
        "new_source_rows": source["rows"],
        "full_api_pages": source["api_pages"],
        "additional_probe_pages": 1,
        "explanatory_document_requests": source["new_document_requests"],
        "new_unique_full_accounts": result["new_full_accounts"],
        "daily_distribution_updates": result["daily_distribution_updates"],
        "unique_prediction_days": result["unique_prediction_days"],
        "mature_prediction_days": result["mature_prediction_days"],
        "new_parameter_searches": 0,
        "primary": primary,
        "price_baseline_stress": control,
        "source_first_vintage_verified": False,
        "goal_status": "active",
        "goal_achieved": False,
        "independent_forward_observations": 0,
        "review_package_created": False,
        "orders_authorized": False,
    }
    save(OUT / "completed_round.json", completed)
    relative = OUT.relative_to(ROOT).as_posix()
    status_path = MAIN / "current_status.json"
    status = read(status_path)
    for study in completed["studies"]:
        if study not in status["completed_followup_studies"]:
            status["completed_followup_studies"].append(study)
    if "previous_new_information_round_before_hkma" not in status:
        status["previous_new_information_round_before_hkma"] = status.get("latest_new_information_round")
    classification = "PROGRESS_NEW_HKMA_INFORMATION_AND_FOUR_FULL_ACCOUNTS"
    status.update(
        updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification=classification,
        same_condition_consecutive_no_progress_goal_turns=0,
        latest_new_information_round=completed,
        latest_hkma_information_round=completed,
        latest_user_requested_study=relative,
        latest_user_requested_study_status=completed["status"],
        research_execution_state=completed["status"],
        latest_continuation_report=relative + "/本轮新证据与账户结论.md",
        active_blocker_id=None, active_blocker_description=None,
        goal_metadata_note="目标active；香港官方资料与4个完整账户为实质进展，高夏普目标未完成。",
        remaining_research_question="汇金披露、大陆回购定盘差及香港设施下午变化的固定用途均未通过；仍须找到当前两年日更与尾部预算下的可执行收益优势并独立验证。",
    )
    save(status_path, status)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(
        current_round=result["study_id"], current_protocol=relative + "/protocol.json",
        latest_progress_receipt=relative + "/completed_round.json",
        latest_integrated_experiment=result["study_id"],
        last_research_result="香港官方资料与4个账户已完成；下午融资变化未改善预测或账户，高夏普目标仍active。",
        latest_continuation_report=relative + "/本轮新证据与账户结论.md",
        latest_continuation_classification=classification,
        research_execution_state=completed["status"], goal_status="active", goal_achieved=False,
    )
    save(mandate_path, mandate)
    banner = (
        "> 2026-09-25 香港人民币下午融资变化研究完成：2,443个官方日期、4个新完整账户。"
        f"主压力夏普{primary['sharpe']:.3f}、年化{primary['annual_return']:.3%}、回撤{abs(primary['max_drawdown']):.2%}；"
        "新增信息未改善预测与账户。目标ACTIVE且未完成，独立前向0；"
        f"见[本轮结论]({relative}/本轮新证据与账户结论.md)。\n\n"
    )
    global_path = ROOT / "RESEARCH_STATUS.md"
    old = global_path.read_text(encoding="utf-8-sig")
    if "香港人民币下午融资变化研究完成" not in old:
        global_path.write_text(banner + old, encoding="utf-8")
    print("香港人民币新证据、四账户结论与主线状态已保存；高夏普目标尚未完成。", flush=True)


if __name__ == "__main__":
    run()
