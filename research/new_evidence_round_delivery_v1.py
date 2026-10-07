"""汇总本轮已保存的新证据与固定检验，不重跑模型、不制作审核包或用户表格。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_new_evidence_resume_20260925"
INTEGRATED = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
EXIT = ROOT / "reports/research/510300_selected_mix_nfci_increment_daily_v1"
RISK = ROOT / "reports/research/510300_nfci_tail_forecast_increment_v1"
SCALED = ROOT / "reports/research/510300_index_volatility_scaled_tail_v1"
ORIGINAL = ROOT / "reports/research/510300_original_frozen_sse_completion_20260925"


def write_report(folder, paragraphs):
    (folder / "研究结论.md").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")


def main():
    macro, risk, scaled, original = [read(folder / "result.json") for folder in [EXIT, RISK, SCALED, ORIGINAL]]
    source = read(ROOT / "reports/research/510300_nfci_graph_vintages_source_v1_3/result.json")
    risk_text = ["NFCI新历史版本的风险预测用途未通过，未据此运行新账户。总指标趋紧与信用相对风险变化，分别只增加一个状态条件；均不能稳定改善原五日损失预测。",
        "每天只使用最近两个日历年、当时已经完整成熟的五日含分红开盘至开盘收益。原动量分组不变；宏观匹配少于60行或宏观过期时，按预先规则退回原分布。不是根据账户收益挑选宏观状态。",
        "主评价从预定首原点每隔五个交易日取一次。2015—2019有243个不重叠持有段，2020年以来有324个；不重叠仍不代表统计独立，因此使用20交易日区块区间。各区间只评价本区间内已经成熟的结果。",
        "联合评分同时检查损失分位数和尾部平均损失，采用Patton、Ziegel、Chen的FZ0评分；数值越低越好。"
        "这是评分方法来源，不构成510300有效性证据。[作者论文](https://public.econ.duke.edu/~ap172/Patton_Ziegel_Chen_JoE_2019.pdf)。"]
    for row in risk["comparisons"]:
        if row["policy"] == "RELATIVE_CREDIT":
            period = "2015—2019" if row["period"] == "earlier" else "2020年以来"
            risk_text.append(f"{period}主方案相对原对照的平均联合评分差为{row['mean_difference']:+.4f}，"
                             f"95%区间[{row['lower_95']:+.4f}, {row['upper_95']:+.4f}]；实际跌破5%损失分位线的比例{row['breach_rate']:.2%}。正评分差表示变差。")
    risk_text.extend([f"共形成{risk['new_conditional_risk_estimates']}次新增条件风险估计，逐日复现{risk['reproduced_baseline_forecasts']}个已保存原风险预测。没有新增回归拟合，没有新增账户。",
                      "预指定主方案在两个区间均未确认改善，状态FROZEN_NO_RELIABLE_NFCI_TAIL_INCREMENT，账户NOT_RUN_FAILED_RISK_GATE。窗口、方向、门槛和账户验收不因结果而改动。",
                      "代码：research/nfci_tail_forecast_increment_v1.py；协议protocol.json；全部结果result.json；全部成熟评分mature_forecast_scores.parquet；逐年评分yearly_risk_scores.json。项目多次选择尚未由区间校正，本项仍是开发研究。"])
    write_report(RISK, risk_text)
    scaled_text = ["20日事前波动尺度调整能改善2015—2019的风险预测，但2020年以来没有确认增量，未通过跨时期门槛，未运行新账户。",
        "使用与原风险模型完全相同的两年成熟样本；将每个历史五日对数收益按今天20日波动／历史决策原点20日波动缩放，再转回简单收益。只使用当时已知波动，不用未来五日实际波动，不搜索别的波动窗口。",
        "这一方法受过滤历史模拟启发，属于固定五日尺度近似，不保证未来波动或跳空已被完整描述。"
        "相关方法和顺周期问题见[英国央行研究](https://www.bankofengland.co.uk/working-paper/2015/filtered-historical-simulation-value-at-risk-models-and-their-competitors)。",
        "旧研究中的直接按波动缩放仓位、十日幅度均值预测，与本次五日损失分布检验分开保留，未改写旧裁决。"]
    for row in scaled["comparisons"]:
        period = "2015—2019" if row["period"] == "earlier" else "2020年以来"
        b = next(x for x in scaled["period_scores"] if x["period"] == row["period"] and x["policy"] == "BASELINE")
        scaled_text.append(f"{period}：5%损失分位线实际跌破率由{b['breach_rate']:.2%}变为{row['breach_rate']:.2%}；"
                           f"联合评分差{row['mean_difference']:+.4f}，95%区间[{row['lower_95']:+.4f}, {row['upper_95']:+.4f}]。评分差越低越好；只让跌破率接近5%不能单独证明预测更好。")
    scaled_text.extend([f"本次形成{scaled['daily_risk_estimates']}次固定尺度风险估计，新增账户0个。状态FROZEN_NO_RELIABLE_SCALED_TAIL_INCREMENT；因门槛失败，账户NOT_RUN_FAILED_RISK_GATE。",
                        "这项假设是在发现旧风险估计的校准偏差后提出，属于开发证据。代码research/index_volatility_scaled_tail_v1.py；协议protocol.json；完整结果result.json；全部成熟预测评分mature_forecast_scores.parquet。"])
    write_report(SCALED, scaled_text)
    primary = next(x for x in original["all_metrics"] if x["cost"] == "STRESS")
    new_period = next(x for x in original["new_periods"] if x["cost"] == "STRESS")
    ledger = pd.read_parquet(ORIGINAL / "accounts_run/accounts/STRESS/SELECTED_MIX_BAND10_SIMPLE2/ledger.parquet")
    np.testing.assert_allclose(ledger.equity.iloc[-1], new_period["end_equity"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(ledger.equity.iloc[-1] / 200000 - 1, primary["cumulative_return"], atol=1e-12, rtol=0)
    original_text = [f"原固定策略已经无调参续算至2026-09-24。压力成本全历史夏普{primary['net_sharpe']:.4f}、年化{primary['annualized_return']:.2%}、最大回撤{abs(primary['max_drawdown']):.2%}；新增六个交易日净收益{new_period['new_period_net_return']:.2%}。仍未证明独立稳定优势。",
        "新增日期为9月17日、18日、21日至24日。两家行情源逐字段一致，旧价格和因素前缀完全保留；原22个内部账户合计新增132条账本记录、复用39,006条旧记录，没有新增拟合或候选搜索。",
        "基金官网产品页三次出现代理请求失败。原双官网接纳失败保留；本次另立来源合同，使用新取得的双源行情、完整上交所2026年16条公告和原已核实分红账本，确认没有新分红或拆合事件。没有修改全局分红账本或旧正式覆盖回执。",
        "新六日均未达到全部22个输入决定在开盘前实际保存的条件。第一日对应检查点实际保存于9月17日11时23分附近，已晚于当日开盘；后续日期的决定由本次回放生成。没有借用更早研究起点的时钟冒充及时决定。",
        f"新增期间压力账户有{new_period['filled_days']}个成交日，{new_period['exposed_days']}个持仓日，费用与滑点合计{new_period['fees']:.2f}元；六日结果不年化、不计算短期夏普。期末仍按收盘价计量剩余持仓，没有强行按终点清仓。",
        "旧版并非已被证明完全没有真实优势，也没有被证明不受过拟合影响。它曾参与大量历史选择，旧主区间最大一笔约占利润42.3%，前五笔约85.5%；新六日不足以化解这些问题。旧版还不符合当前最近两年每日训练和账户尾部预算，保持固定研究对照身份。",
        "9月25日至27日中秋休市，下一交易日为9月28日。日历来自[上交所2026年休市安排](https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20251222_10802510.shtml)。本研究没有恢复订单或原计划采集任务。",
        "代码research/original_new_evidence_sse_completion_v1.py；固定协议protocol.json；完整结果result.json；新增日收益new_observed_daily_returns.parquet；来源时钟actual_source_recording_times.json。"]
    write_report(ORIGINAL, original_text)
    headline = "新证据与四项固定检验已完成，高夏普目标未完成。NFCI未确认决策或风险增量；波动尺度仅局部改善；旧版追加六个交易日后历史压力夏普1.202，仍未证明稳定。"
    summary = [headline,
        "本轮使用用户新授权寻找公开证据，实际取得NFCI总体、信用、风险三个官方历史版本序列，共739个版本日、2,217条版本状态，并取得510300新增六个交易日行情。"
        "宏观回放保留当时可见版本，避免今天修订值回填过去。官方说明：[NFCI发布与修订](https://www.chicagofed.org/research/data/nfci/current-data)、[ALFRED版本定义](https://alfred.stlouisfed.org/help/downloaddata)。",
        "本轮评价目标保持20万元全账户、压力费用后夏普至少1.2、复合年化至少10%、最大回撤不超过10%；只研究510300与现金。主策略训练仍为最近两个日历年逐日更新，单笔3:1已按用户此前答复改为账户尾部预算。",
        "NFCI用于持有与退出的固定比较，账户区间为2020-01-02至2026-09-16：\n\n"
        "- 原两年日更对照：压力夏普0.954、年化3.62%、最大回撤5.06%。\n"
        "- 加入总体金融条件：压力夏普0.919、年化3.49%、最大回撤5.06%。\n"
        "- 再加入一项信用与风险差异：压力夏普0.861、年化3.33%、最大回撤5.06%。",
        "主方案相对原对照的算术年化收益差约−0.275个百分点，20交易日区块95%区间约[−1.047，+0.339]个百分点。"
        "没有确认增量，所有滚动两年窗口也未同时达到三项数值要求。不是把宏观解释写得更完整，就获得了可交易收益。",
        "NFCI另一个预先固定的用途是五日风险预测。主方案在2015—2019、2020年以来两段的联合评分都变差，预测门槛失败；没有据此运行新账户。",
        "风险检查发现原模型前期低估损失：预计5%的分位线实际被25/243个不重叠五日结果跌破，约10.3%。"
        "固定20日波动尺度调整把它降至15/243，约6.2%，且前期联合评分显著改善；2020年以来未确认评分改善，因此没有按年份拼接两个风险模型，也没有晋升为新账户。",
        f"原夏普1.2固定版本另行无调参回放至9月24日：新增六日压力净收益{new_period['new_period_net_return']:.2%}，全历史压力夏普{primary['net_sharpe']:.4f}、年化{primary['annualized_return']:.2%}、最大回撤{abs(primary['max_drawdown']):.2%}。"
        "这保留了原策略可能包含部分真实优势的可能性，但新增样本少、决定文件不及时、历史选择与收益集中问题仍在，不能认定已排除过拟合。原版与当前两年日更及尾部预算合同也不同。",
        "本次实质新增为10,524次宏观退出回归拟合、4个新增完整策略账户、2个对照完整复现、66个内部依赖账户、8,538次新增风险估计；"
        "旧固定策略另有22个内部账户的132条新增日记录。6个新市场日期不等于6个及时独立前向验证日，后者本轮为0。",
        "当前仍缺：同时符合现行训练与风险合同的足够收益优势，以及规则冻结之后的独立验证。NFCI历史版本缺口已经解决，不再列为未取到数据；国内中性利率或预期通胀的合格历史版本问题没有因此被解决。"
        "不能把美国NFCI当成中国信用增速，也不能把DR007减政策利率改名为中性利率缺口。",
        "已失败的宏观用途、风险用途和尺度用途各自冻结；没有放宽验收或保留有利年份。目标未完成，也没有证明所有可能策略在数学上都不可能。稳定性必须继续接受新信息与新观察检验，有限历史不能保证未来永不失效。",
        "本轮没有期权、杠杆、订单、实盘、Paper/Shadow执行，没有审核ZIP或面向用户的表格；原计划采集任务保持暂停。研究回放持仓不是用户实际持仓或当前交易建议。",
        "文件导航：\n\n"
        "- NFCI完整账户比较：../510300_selected_mix_nfci_increment_daily_v1/研究结论.md。\n"
        "- NFCI风险预测：../510300_nfci_tail_forecast_increment_v1/研究结论.md。\n"
        "- 波动尺度风险预测：../510300_index_volatility_scaled_tail_v1/研究结论.md。\n"
        "- 原固定版本新增日期：../510300_original_frozen_sse_completion_20260925/研究结论.md。\n"
        "- 既有综合研究与A/B/C/D、节假日、宏观事件、状态策略、退出等旧结果：../510300_integrated_research_continuation_20260924/current_status.json。"]
    (OUT / "本轮新增证据与策略结论.md").write_text("\n\n".join(summary)+"\n", encoding="utf-8")
    progress = {"at": now(), "status": "NEW_PUBLIC_EVIDENCE_AND_FOUR_FIXED_STUDIES_COMPLETED_TARGET_UNMET",
        "new_source": source, "studies": [r["study_id"] for r in [macro, risk, scaled, original]],
        "new_regression_fits": macro["training_checks"]["new_fits"], "new_final_accounts": 4,
        "baseline_accounts_reproduced": 2, "internal_dependency_accounts": 66,
        "new_risk_estimates": risk["new_conditional_risk_estimates"]+scaled["daily_risk_estimates"],
        "old_frozen_incremental_account_rows": original["incremental_account_rows"],
        "new_market_trading_days": 6, "new_timely_forward_validation_days": 0,
        "original_full_stress": primary, "nfci_primary_stress": macro["primary"],
        "price_coverage": {"nfci_experiment": "2026-09-16", "fixed_original_replay": "2026-09-24"},
        "goal_achieved": False, "orders_authorized": False,
        "artifacts": {folder.relative_to(ROOT).as_posix(): digest(folder / "result.json") for folder in [EXIT, RISK, SCALED, ORIGINAL]}}
    save(OUT / "completed_round.json", progress, True)
    current = read(INTEGRATED / "current_status.json")
    current.update(updated_at=now(), scope="已有本地资料与本次新授权的公开证据研究",
        research_execution_state="NEW_PUBLIC_EVIDENCE_ROUND_COMPLETED_TARGET_UNMET",
        latest_goal_turn_classification="PROGRESS_NEW_VINTAGES_NEW_MARKET_DATES_AND_FIXED_EXPERIMENTS",
        same_condition_consecutive_no_progress_goal_turns=0, active_blocker_id=None, active_blocker_description=None,
        latest_original_strategy_price_date="2026-09-24", latest_available_price_source_end="2026-09-24",
        latest_strategy_price_source_end="2026-09-16", latest_new_evidence_round=progress,
        latest_user_requested_study=OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=progress["status"],
        remaining_research_question="新NFCI版本未确认决策或风险增量；尺度校准仅前期改善。仍需满足现行两年日更与尾部预算的可执行收益优势及后续独立证据。",
        goal_metadata_note="用户本轮已恢复公开证据研究，并完成实质进展；目标工具最后返回blocked且无恢复接口，未伪造工具状态，未把目标标为完成。",
        goal_achieved=False, current_market_view="NO_VIEW")
    for study in progress["studies"]:
        if study not in current["completed_followup_studies"]:
            current["completed_followup_studies"].append(study)
    save(INTEGRATED / "current_status.json", current)
    (INTEGRATED / "最新研究结论.md").write_text("\n\n".join(summary)+"\n", encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(current_round=progress["status"], research_execution_state=current["research_execution_state"],
                   latest_progress_receipt=(OUT / "completed_round.json").relative_to(ROOT).as_posix(),
                   last_research_result=headline, goal_achieved=False,
                   latest_new_public_price_source=(ORIGINAL / "candidate_features.parquet").relative_to(ROOT).as_posix())
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    banner = "<!-- NEW_EVIDENCE_ROUND_COMPLETE_20260925 -->\n\n> " + headline + " 详见[本轮新增证据与策略结论](reports/research/510300_new_evidence_resume_20260925/本轮新增证据与策略结论.md)。后面的旧受阻记录是历史状态。\n\n<!-- END_NEW_EVIDENCE_ROUND_COMPLETE_20260925 -->\n\n"
    status_path.write_text(banner+status_path.read_text(encoding="utf-8"), encoding="utf-8")
    save(OUT / "delivery_checks.json", {"at": now(), "source_results_read_from_saved_files": True,
        "original_new_equity_recomputed": True, "timely_validation_not_confused_with_new_price_dates": True,
        "nfci_plot_visual_status": read(EXIT / "delivery_checks.json")["image_visual_check"],
        "new_training_or_accounts_in_delivery": 0, "goal_achieved": False}, True)
    print(headline, flush=True)


if __name__ == "__main__":
    main()
