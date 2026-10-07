"""将已完成的固定日更研究同步到当前项目摘要，保留全部历史结果。"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now

FOLDERS = [ROOT / "reports/research" / ("510300_" + name) for name in [
    "selected_mix_daily_two_year_v1", "selected_mix_pooled_daily_v1",
    "selected_mix_risk_before_band_v1", "selected_mix_bounded_value_daily_v1"]]
SUMMARY = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
LABELS = ["原图两年日更、5周期门槛", "三类过程共享退出", "风险目标先行、固定调仓带", "固定五日继续价值"]
MARKER = "<!-- SELECTED_MIX_DAILY_PROGRESS_20260925 -->"


def main():
    results = [read(folder / "result.json") for folder in FOLDERS]
    fits = sum(result["checks"].get("actual_model_fits", result["checks"].get("fitted_models", 0)) for result in results)
    summaries = [{"study_id": result["study_id"], "primary": result["primary"], "status": result["status"],
                  "result": (folder / "result.json").relative_to(ROOT).as_posix()}
                 for folder, result in zip(FOLDERS, results)]
    receipt = {"at": now(), "latest_user_instruction": "不找到不中止   找到再停止",
               "status": "FOUR_FIXED_RESEARCH_EXPERIMENTS_COMPLETED_GOAL_UNMET", "goal_achieved": False,
               "goal_workflow": "ACTIVE_RESEARCH", "studies": summaries, "new_model_fits": fits,
               "new_internal_dependency_accounts": 88, "new_full_terminal_policy_accounts": 12,
               "new_reference_account_reconstructions": 3, "new_independent_market_observations": 0,
               "market_collection_resumed": False, "orders_authorized": False,
               "current_market_view": "NO_VIEW", "price_source_end": "2026-09-16",
               "constraint_question": "年化10%与当前尾部预算取舍的可选问题尚未收到变更；继续保持原三项目标和全部风险限额。"}
    save(SUMMARY / "selected_mix_daily_progress_20260925.json", receipt)
    explanation = (
        "原版85/15在原规则下可复现压力夏普1.219、年化10.29%、最大回撤6.51%；历史优势可能存在，选择膨胀与独立验证仍未解决。"
        "本轮将其迁移到最近两年、逐日更新和当前尾部预算，并完成共享训练、执行顺序、五日标签三项固定后续比较，均未同时达到1.2/10%/10%。"
        "当前目标保持未完成。"
    )
    rows = [explanation, "", "本轮实际完成的压力账户：", ""]
    for label, result, folder in zip(LABELS, results, FOLDERS):
        p = result["primary"]
        rows.append(f"- {label}：夏普{p['sharpe']:.3f}，年化{p['annual_return']:.2%}，最大回撤{abs(p['max_drawdown']):.2%}，期末{p['end_equity']:,.2f}元。来源：{folder.relative_to(ROOT).as_posix()}/研究结论.md。")
    rows += ["", "旧模型加同一尾部预算的对照压力夏普1.061、年化4.41%、回撤3.51%；它使用原先较长周期的模型，不能冒充符合两年训练的新候选。严格10周期对照、所有基础成本结果、所有年份及滚动两年也完整保留。",
             "", f"本轮新增{fits:,}次逐日模型拟合、88条内部依赖账户和12条末端完整账户，重建3个价格参考。大量训练记录共享同一价格历史，新增独立市场观察仍为0。内部账户没有接末端尾部预算，不能从内部选出好结果当作当前合格策略。",
             "", "发现并检验的具体执行改进：先形成风险可行目标、再应用固定10个百分点调仓带。压力成交次数由343减至171，总费用从9,230.76元降至7,998.31元；夏普由0.916升至0.954，但配对增量区间仍跨零，年化也没有达到10%。强制减险和原信号归零不受带宽阻拦。",
             "", "五日标签让近期状态更快进入训练，但压力夏普降为0.712。没有据此改五日窗口或继续放宽样本门槛。这个结果与原自然退出标签结果并列保留。",
             "", "两年日更与账户尾部风险均已实现：每交易日只纳入最近两个日历年内的可用训练信息，模型不合格时不沿用过期系数；三类共享训练保留类别差异。末端50%目标上限、五日ES95为权益2.5%、10%跳空预算与回撤余量继续有效。预算不是最大回撤保证。",
             "", "完整账本覆盖2020年1月2日至2026年9月16日全部1,627个交易日。成本、100份、T+1、现金、分红和次开盘执行均计入，未利用终点日强行平仓。模型样本时钟、保存系数、未来标签扰动、标签截断、风险优先级与账户等式检查完成。",
             "", "现在的瓶颈是符合当前条件的可重复收益来源。减少费用、增加训练行数、增加更新频率均未自动建立该优势。历史已被反复研究，这些结果不能作为独立样本外证明；原版可能含有真实优势，也仍可能受到选择膨胀影响。",
             "", "运行入口：scripts/run_510300_selected_mix_daily_research.ps1。默认顺序执行尚未完成的固定步骤、复用已有结果；-Mode Status只读取状态，不重跑研究。源码、协议、保存模型、每日决策、账户与结果均在各研究目录。",
             "", "可选的收益／尾部预算取舍问题尚未收到变更，因此本轮没有降低目标或放松风险限额。行情采集仍暂停，价格只到9月16日，当前观点NO_VIEW。没有期权、审核ZIP、用户汇总表格或交易订单。"]
    report = "\n".join(rows) + "\n"
    (SUMMARY / "两年日更实施与研究进展.md").write_text(report, encoding="utf-8")
    (SUMMARY / "最新研究结论.md").write_text(report, encoding="utf-8")
    current = read(SUMMARY / "current_status.json")
    completed = current.get("completed_followup_studies", [])
    for result in results:
        if result["study_id"] not in completed:
            completed.append(result["study_id"])
    current.update(updated_at=now(), goal_status="active", goal_achieved=False,
                   completed_followup_studies=completed, latest_goal_turn_classification="PROGRESS_FOUR_FIXED_DAILY_RESEARCH_EXPERIMENTS_COMPLETED",
                   same_condition_consecutive_no_progress_goal_turns=0,
                   latest_user_requested_study=FOLDERS[-1].relative_to(ROOT).as_posix(),
                   latest_user_requested_study_status=results[-1]["status"],
                   latest_research_workflow="DAILY_TWO_YEAR_IMPLEMENTED_CURRENT_HIGH_SHARPE_TARGET_UNMET",
                   latest_strategy_price_source_end="2026-09-16", active_blocker_id=None, active_blocker_description=None,
                   remaining_research_question="当前两年日更和尾部预算下尚未找到达标的收益来源；旧版历史达标不等于新合同达标或独立验证。",
                   report="两年日更实施与研究进展.md", latest_daily_research_progress=receipt)
    save(SUMMARY / "current_status.json", current)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(latest_user_instruction="不找到不中止   找到再停止", goal_achieved=False,
                   last_research_result=explanation,
                   latest_progress_receipt=(SUMMARY / "selected_mix_daily_progress_20260925.json").relative_to(ROOT).as_posix())
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    original = status_path.read_text(encoding="utf-8")
    if not original.startswith(MARKER):
        addition = (MARKER + "\n\n> 2026-09-25 已完成原85/15的两年日更迁移、三类共享训练、风险目标先行与固定五日价值四项研究。"
                    "压力夏普分别0.636、0.916、0.954、0.712；当前1.2/10%/10%目标未完成，研究目标保持进行中。"
                    "旧原版1.219的历史复现保留。详见[本轮实施与进展](reports/research/510300_integrated_research_continuation_20260924/两年日更实施与研究进展.md)。\n\n")
        status_path.write_text(addition + original, encoding="utf-8")
    print(f"四项固定研究进展已同步：{fits:,}次拟合、12条末端账户，整体目标保持未完成。", flush=True)


if __name__ == "__main__":
    main()
