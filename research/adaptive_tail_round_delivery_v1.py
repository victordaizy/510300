"""汇总本次日更风险检验、决策敏感度和已成熟宏观标签；不制作审阅包。"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.nfci_tail_forecast_increment_v1 import fz0, interval

RISK = ROOT / "reports/research/510300_adaptive_tail_daily_v1"
SENSITIVITY = ROOT / "reports/research/510300_tail_decision_sensitivity_v1"
M1 = ROOT / "reports/research/510300_m1_pending_label_settlement_v1"
INTEGRATED = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def verify_saved(risk):
    checked = 0
    for directory in [RISK, SENSITIVITY, M1]:
        frozen = read(directory / "freeze.json")
        assert digest(directory / "protocol.json") == frozen["protocol_sha256"]
        for name, expected in frozen["sources"].items():
            assert digest(ROOT / name) == expected, "冻结来源变化：" + name
            checked += 1
    scores = pd.read_parquet(RISK / "mature_forecast_scores.parquet")
    np.testing.assert_allclose(fz0(scores.return5, scores.q05, -scores.es95), scores.fz0, atol=1e-14, rtol=0)
    phase = scores[scores.fixed_nonoverlap_phase]
    rng = np.random.default_rng(20260925)
    for record in risk["comparisons"]:
        left = phase[phase.period.eq(record["period"]) & phase.policy.eq(record["policy"])].sort_values("origin_index")
        right = phase[phase.period.eq(record["period"]) & phase.policy.eq(record["comparator"])].sort_values("origin_index")
        recomputed = interval(left.fz0.to_numpy() - right.fz0.to_numpy(), rng)
        for field in ["mean_difference", "lower_95", "upper_95"]:
            assert recomputed[field] == record[field]
    inner = pd.read_parquet(RISK / "nested_validation.parquet")
    used = inner[inner.status.eq("SCORED")]
    assert (used.validation_maturity_index < used.outer_origin_index).all()
    assert (used.latest_training_exit_index < used.validation_origin_index).all()
    assert 2 * len(used) == risk["nested_validation_predictions"]
    return {"frozen_source_checks": checked, "saved_scores_recomputed": len(scores),
            "paired_intervals_recomputed": 4, "nested_scored_rows": len(used),
            "new_fits_or_accounts_in_delivery": 0, "result": "PASS_SAVED_RESULTS_AND_TIME_ORDER"}


def chart(risk):
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    weights = pd.DataFrame(read(RISK / "daily_weight_receipts.json"))
    weights["origin"] = pd.to_datetime(weights.origin)
    fig = plt.figure(figsize=(12, 8), facecolor="#fafbf8")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.1, 1], hspace=.47, top=.88, bottom=.10, left=.20, right=.95)
    ax = fig.add_subplot(grid[0])
    ax.set_facecolor("#fafbf8")
    ax.plot(weights.origin, weights.filtered_weight, color="#196b71", linewidth=1.35)
    ax.axhline(.5, color="#888b8a", linestyle="--", linewidth=1, label="固定各半对照")
    ax.axvline(pd.Timestamp("2020-01-01"), color="#c5c7c4", linewidth=.9)
    ax.set_ylim(.25, 1.)
    ax.yaxis.set_major_formatter(PercentFormatter(1.))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("波动尺度模型权重")
    ax.set_title("每日权重只依据当时两年窗口内已成熟的预测误差", loc="left", fontsize=12, pad=10)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=.16)
    ax2 = fig.add_subplot(grid[1])
    ax2.set_facecolor("#fafbf8")
    labels = []
    for index, row in enumerate(risk["comparisons"]):
        period = "2015—2019" if row["period"] == "earlier" else "2020—2026/9"
        comparator = "原风险模型" if row["comparator"] == "BASELINE" else "固定各半"
        labels.append(period + "  对比" + comparator)
        color = "#196b71" if row["upper_95"] < 0 else "#b07442"
        ax2.errorbar(row["mean_difference"], index,
                     xerr=[[row["mean_difference"] - row["lower_95"]], [row["upper_95"] - row["mean_difference"]]],
                     fmt="o", color=color, capsize=4, markersize=6, linewidth=1.6)
    ax2.axvline(0, color="#4c5352", linewidth=1)
    ax2.set_yticks(range(4), labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("自适应减对照的联合风险评分差；负值表示改善")
    ax2.set_title("2020年以来，两项差值区间都跨过零", loc="left", fontsize=12, pad=12)
    ax2.grid(axis="x", alpha=.16)
    fig.suptitle("日更风险组合：前期有增量，近期未确认", x=.05, ha="left", fontsize=18, fontweight="bold", y=.97)
    fig.text(.05, .918, "2,846个决策日  ·  最近两年内嵌套验证  ·  567个不重叠五日评价区间", color="#535b5a", fontsize=11)
    fig.text(.05, .026, "区间为固定20交易日区块自助法95%区间；未校正项目反复选择。图中是风险预测评分，不是账户夏普。", fontsize=10, color="#5c6461")
    fig.savefig(RISK / "日更风险组合检验.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def write(path, paragraphs):
    path.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")


def main():
    if (RISK / "completed_round.json").exists():
        raise RuntimeError("本轮已经交付，不覆盖完成记录")
    risk, sensitivity, m1 = [read(p / "result.json") for p in [RISK, SENSITIVITY, M1]]
    verification = verify_saved(risk)
    chart(risk)
    source = "[Patton、Ziegel、Chen的联合风险评分论文](https://public.econ.duke.edu/~ap172/Patton_Ziegel_Chen_JoE_2019.pdf)"
    risk_text = [
        "本轮已实现并运行最近两年内的嵌套顺序验证、每日风险模型权重更新。2015—2019年改善明确，2020年以来没有确认优于原模型或固定各半；未通过事前门槛，没有生成新账户。高夏普目标仍未达到。",
        "两名专家固定为原五日经验损失分布和同一分布的20日波动尺度调整。每个决策日T，当前预测只用T前两年内已经成熟的五日结果。模型权重也在这同一个窗口内重新顺序验证，避免使用历史专家在更早年代训练的预测，从而间接引入四年资料。",
        "内层只取最近一年、每隔五个交易日的固定相位验证点。每个验证点只使用当时成熟且仍在当前T两年窗口内的样本；不少于40个成熟观察才更新权重。对平均联合评分取固定温度1的软权重，不按随后账户利润选择。三项对照为原模型、固定尺度模型、固定各半；主方案必须在两个时期同时优于原模型及固定各半，才能进入另立协议的账户实验。",
        f"VaR与ES采用联合风险评分，数值越低越好，方法来源见{source}。论文没有证明本轮软权重规则在510300有效；权重、窗口和门槛是本轮研究设计。",
    ]
    for item in risk["comparisons"]:
        period = "2015—2019" if item["period"] == "earlier" else "2020-01-02至2026-09-16"
        control = "原模型" if item["comparator"] == "BASELINE" else "固定各半"
        risk_text.append(f"{period}，自适应组合相对{control}的平均评分差{item['mean_difference']:+.4f}，"
                         f"95%区间[{item['lower_95']:+.4f}, {item['upper_95']:+.4f}]；"
                         f"分位数损失差{item['pinball_mean_difference']:+.7f}。")
    risk_text.extend([
        "实际跌破预测5%损失分位线的比例：2015—2019由原模型的10.29%降至自适应的7.00%；2020年以来由4.32%变为4.01%。仅降低跌破率并不足以证明预测更好，需要同时考察损失评分。",
        f"本次完成{risk['daily_adaptive_weight_updates']:,}次每日权重更新、{risk['new_combined_risk_estimates']:,}份新增组合预测，逐值复现{risk['reproduced_saved_expert_predictions']:,}份旧专家预测。内层累计{risk['nested_validation_predictions']:,}份风险预测反复共享历史，不是这么多独立市场事件。评价只有243与324个不重叠五日区间；不重叠也不保证统计独立。",
        f"波动尺度专家权重范围{risk['filtered_weight_minimum']:.2%}至{risk['filtered_weight_maximum']:.2%}，中位数{risk['filtered_weight_median']:.2%}。全部决策日均满足预定内层样本数，未触发固定各半回退。",
        "未来标签、过期标签和未来波动扰动均不改变当日预测或权重；常波动下两专家和组合一致；所有日期重现原专家，成熟时序检查及保存评分复算通过。代码位于research/adaptive_tail_daily_v1.py及research/adaptive_tail_daily_core_v1.py。",
        "失败状态FROZEN_NO_RELIABLE_ADAPTIVE_TAIL_INCREMENT保留。所有历史已被反复研究，区间未校正全项目选择，本轮仍是开发证据，没有新增独立前向观察。不能以2015—2019自适应、2020以后固定各半的事后拼接宣称通过。"
    ])
    write(RISK / "风险模型研究结论.md", risk_text)
    changed = next(r for r in sensitivity["results"] if r["cost"] == "STRESS" and r["case"] == "ADAPTIVE_TWO_YEAR")
    zero = next(r for r in sensitivity["results"] if r["cost"] == "STRESS" and r["case"] == "ZERO_ES_DIAGNOSTIC_ONLY")
    sensitivity_text = [
        "风险模型改善只有在改变实际决策时才可能影响账户。本项固定每个历史时点的原账户权益、持仓、峰值、信号与费用，仅替换ES输入，不将反事实份额传播到后一天，没有构建新账户或计算夏普。",
        f"压力费用下，自适应预测改变{changed['changed_decisions']}/{changed['all_decisions']}个决策日，约{changed['changed_fraction_of_all']:.2%}；"
        f"在原信号有仓位需求的{changed['source_positive_decisions']}日中占{changed['changed_fraction_of_source_positive']:.2%}。各日仓位差有增有减，全日平均净差约{changed['mean_exposure_difference_all']*100:.3f}个百分点。",
        f"将五日ES市场损失项暂设为0、仍保留交易费用、50%上限、10%跳空与回撤余量，只在{zero['changed_decisions']}/{zero['all_decisions']}日改变决策，约{zero['changed_fraction_of_all']:.2%}；"
        f"原信号有仓位需求的日子中占{zero['changed_fraction_of_source_positive']:.2%}，全日平均允许仓位增加{zero['mean_exposure_difference_all']*100:.3f}个百分点。",
        "ES=0是不可交易的诊断输入。上述比例只描述原账户状态附近的局部敏感度，不是收益上界，也不能据此声称任何风险模型都不可能改善收益。它表明，多数原状态的仓位还受原信号、跳空预算、回撤余量和调仓带影响；单靠细调五日尾部预测，尚无证据填补当前较大的年化收益缺口。",
        f"两档费用共{sensitivity['saved_baseline_decisions_reproduced']}个原决策逐值复现。所有案例与原参数完整保存，未解除上一项失败门槛。代码research/tail_decision_sensitivity_v1.py；结果result.json。"
    ]
    write(SENSITIVITY / "研究结论.md", sensitivity_text)
    m1_text = [
        f"此前缺行情的2026年8月M1事件现已结算。原定9月16日开盘至9月22日收盘的五日毛收益为{m1['actual_gross_return']:+.2%}，三份保存预测均为负，方向全部错误。没有重新训练或重启原失败规则。",
        "保存的价格模型为−0.6395%，加入M1/M2预期差的模型为−0.6955%，成熟样本均值为−0.5914%。此前四个缺失日期已由新取得的双源行情补齐；原价格前缀、16事件特征及三条输出逐值复现，观察日之后价格不影响原输出。",
        f"合并原10个顺序评价事件后，共11事件。加入预期差的均方误差仍比价格基准高{-m1['surprise_mse_improvement_vs_price']:.2%}，没有确认消息增量。各模型都不会因这条负预测新增做多，未新增账户。",
        "该事件模型实际保存于9月22日20:37，已经晚于本事件收盘结算。因此，这是旧模型的历史重建输出结算，不能计作及时独立前向验证；本轮新增独立前向事件仍为0。标签补齐不改原STOP_CURRENT_REPRESENTATION_NO_PARAMETER_RESCUE。",
        "来源沿用reports/research/510300_original_frozen_sse_completion_20260925/protocol.json的双行情源与上交所公告覆盖规则，不冒充旧双官网准入成功。没有新增下载、拟合或账户。代码research/m1_pending_label_settlement_v1.py。"
    ]
    write(M1 / "研究结论.md", m1_text)
    headline = "已完成日更风险组合、仓位敏感度和M1待评价事件结算；未找到新的合格高夏普策略，目标保持active。"
    summary = [headline,
        "当前目标仍为20万元完整账户，压力费用后夏普至少1.2、复合年化至少10%、最大回撤不超过10%；只研究510300与现金。训练限最近两个日历年并每天更新，单笔3:1已改为账户尾部预算。",
        "日更风险组合已实际实现并跑完2,846个决策日。它在2015—2019优于原模型及固定各半；2020年以来，相对原模型的联合评分差−0.0584，95%区间[−0.2330,+0.0599]；相对固定各半差+0.00065，95%区间[−0.01007,+0.01504]，均未确认改善。没有用有利年份拼接、改变窗口或改权重温度补救。",
        "固定原账户状态的诊断显示，自适应预测只改变158/1,628个压力决策日；即使局部删除五日ES市场损失项，也只改变125日。这不是收益上界，但说明当前问题不能只寄望于更细的风险模型。需要证明在已有风险预算内可成交、费用后仍足够大的条件收益。",
        f"新行情还结算了旧M1事件：原固定五日实际毛收益{m1['actual_gross_return']:+.2%}，三份保存输出全部预测下跌。加入预期差后，11事件均方误差仍比价格基准高{-m1['surprise_mse_improvement_vs_price']:.2%}，原失败结论保留。",
        "当前合同下保存的原两年日更对照仍为压力夏普0.954、年化3.62%、回撤5.06%。旧夏普1.2版本已固定回放至9月24日，历史压力夏普1.2017、年化10.15%、回撤6.51%，但训练与风险合同不同，历史选择与收益集中问题尚未消除。这两组结果不能拼成同一个达标策略。",
        "本轮新增2,846次权重更新、5,692份组合风险预测，复现5,692份旧专家预测；内层270,056份重复风险预测不当作独立样本。另复现3,256个原账户决策并补齐1个宏观标签。新增回归拟合0、新账户0、下载0、独立前向事件0。",
        "风险估计、执行和收益优势需要分别检验。当前证据支持把后续研究重点放回可执行的收益来源与完整账户，而不是继续变换同一风险估计的权重。下一项研究必须先确认与已失败用途的差异，并保留原失败和所有评价区间；有限历史不能保证找到永不失效的策略。",
        "本轮结果均为开发研究，未校正整个项目的反复选择。目标工具已确认active且未标记完成；这批计算已结束，没有后台研究进程、订单或恢复计划采集任务，没有创建审核ZIP或面向用户的表格。",
        "详细文件：风险模型研究结论.md；../510300_tail_decision_sensitivity_v1/研究结论.md；../510300_m1_pending_label_settlement_v1/研究结论.md。旧综合与A/B/C/D等结果保留在综合目录current_status.json及上一轮../510300_new_evidence_resume_20260925/本轮新增证据与策略结论.md。"
    ]
    write(RISK / "本轮继续研究结论.md", summary)
    receipt = {"at": now(), "status": "THREE_FOLLOWUP_STUDIES_COMPLETED_HIGH_SHARPE_TARGET_UNMET",
               "studies": [x["study_id"] for x in [risk, sensitivity, m1]],
               "new_daily_weight_updates": risk["daily_adaptive_weight_updates"],
               "new_combined_risk_predictions": risk["new_combined_risk_estimates"],
               "nested_risk_predictions_not_independent": risk["nested_validation_predictions"],
               "saved_expert_predictions_reproduced": risk["reproduced_saved_expert_predictions"],
               "baseline_decisions_reproduced": sensitivity["saved_baseline_decisions_reproduced"],
               "newly_completed_macro_labels": 1, "new_regression_fits": 0, "new_accounts": 0,
               "new_downloads": 0, "new_independent_forward_events": 0,
               "goal_status": "active", "goal_achieved": False, "orders_authorized": False,
               "verification": verification, "sources": {p.relative_to(ROOT).as_posix(): digest(p / "result.json") for p in [RISK, SENSITIVITY, M1]}}
    save(RISK / "completed_round.json", receipt, True)
    current = read(INTEGRATED / "current_status.json")
    current.update(updated_at=now(), goal_status="active", goal_achieved=False,
                   research_execution_state=receipt["status"], latest_goal_turn_classification="PROGRESS_FIXED_ADAPTIVE_RISK_AND_NEWLY_MATURE_MACRO_LABEL",
                   same_condition_consecutive_no_progress_goal_turns=0, active_blocker_id=None, active_blocker_description=None,
                   latest_user_requested_study=RISK.relative_to(ROOT).as_posix(), latest_user_requested_study_status=receipt["status"],
                   latest_adaptive_risk_round=receipt, latest_adaptive_risk_result=risk["status"],
                   latest_pending_m1_label_status=m1["status"], remaining_research_question="现行两年日更与尾部预算下仍缺足够的可执行收益优势及独立验证；日更风险权重与旧M1消息表示未确认增量。",
                   goal_metadata_note="本轮get_goal再次确认active；三项实质续研完成，目标未完成。")
    for study in receipt["studies"]:
        if study not in current["completed_followup_studies"]:
            current["completed_followup_studies"].append(study)
    save(INTEGRATED / "current_status.json", current)
    write(INTEGRATED / "最新研究结论.md", summary)
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(path)
    mandate.update(current_round="510300_ADAPTIVE_TAIL_DAILY_V1", latest_integrated_experiment="510300_ADAPTIVE_TAIL_DAILY_V1",
                   current_protocol=(RISK / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(RISK / "completed_round.json").relative_to(ROOT).as_posix(),
                   latest_continuation_report=(RISK / "本轮继续研究结论.md").relative_to(ROOT).as_posix(),
                   latest_continuation_classification="PROGRESS_FIXED_ADAPTIVE_RISK_AND_NEWLY_MATURE_MACRO_LABEL",
                   research_execution_state=receipt["status"], last_research_result=headline, goal_status="active", goal_achieved=False)
    save(path, mandate)
    status = ROOT / "RESEARCH_STATUS.md"
    banner = "<!-- ADAPTIVE_RISK_FOLLOWUP_20260925 -->\n\n> " + headline + " 详见[本轮继续研究结论](reports/research/510300_adaptive_tail_daily_v1/本轮继续研究结论.md)。以下旧受阻记录为历史状态。\n\n<!-- END_ADAPTIVE_RISK_FOLLOWUP_20260925 -->\n\n"
    status.write_text(banner + status.read_text(encoding="utf-8"), encoding="utf-8")
    print(headline, flush=True)


if __name__ == "__main__":
    main()
