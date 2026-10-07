"""将R163解释和R164/R165实际金融结果一次归档到项目长期事实文件。"""
from __future__ import annotations
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.point_first_passage_study_v1 import read, write_json, require, now, digest
from research.trend_expansion_phase_study_v1 import OUT, EXPLANATION

STATE_PATH = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FORWARD_KEYS = (
    "forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
    "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
    "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
    "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation",
)
REPORT = "reports/research/510300_trend_expansion_phase_study_v1/研究结果与下一步.md"
SUMMARY = (
    "2026-10-04按用户顺序完成TECH.R163全体展开解释及R164登记/R165实际账户。原3488日、85联合出生、"
    "182突破、275放量事件保留；三解释测试、三前缀、四图。唯一日DIF>0/上一完整周柱>0/价在EMA20上出生进、"
    "已知失效出的完整政策，六必要行为测试、八控制精确复现、四新账户一次完成。压力较早37完成/1开放、"
    "胜率27.03%、B2.955/pB0.799、净年化0.0364%/夏普0.0273/DD8.23%；近期46完成、19.57%、"
    "B3.167/pB0.620、−0.7823%/−0.2663/DD7.37%。四经济门及稳定门全败，固定政策关闭不营救。"
    "84实际周期、83完成/1开放及全部85资格已还原；2024次开4.656触及限价无成交，2019春季+16.14%及"
    "2020后段+13.47%不能代表全体。近期实际毛损益−2151.90元，非只因费用；两期未停机。"
    "0拟合/新训练标签/采集；52金融及28解释来源保持。次数增加但收益夏普未改善、独立和去过拟合未达。"
    "最新金融R165、预测R158、原退出R145；目标active/阻塞0，原前瞻12值保持。下一全体触发时钟归因仅解释，"
    "不按分组过滤重跑；新金融未登记，已准入待跑0。"
)
ENTRY = """
## TECH.R163：日周展开阶段与量价到达时钟的全体解释（2026-10-04）

**假设**：价格已经在EMA20上后，日DIF与上一完整周MACD同时正的阶段出生，可能解释从修复到展开；须先覆盖具体上涨及全部失败，不能直接当进场优势。

**验证方法**：先冻结现有EMA20、日DIF、上一完整周柱的四象限及均线下/未知，出生需前原点已知且尚未同时成立；既有20日突破和相对量1.5倍首次成立只作观察通道。3488日先标，2015起2855原点全保留，原2846标签2826成熟/20未知、尾部9无旧标签不填。原61分段/49正式波段原样，事后低高点仅图示；三测试、三个真实前缀、28冻结来源及四原案例图完成。无新拟合/训练标签/账户/采集。

**结果**：联合出生85次，旧20日标签45.88%为正、平均−0.0083%、pB0.539；原突破182次pB0.570，放量275次/274成熟pB0.532。全体时期保留，不切换较有利通道。2019-01-18日DIF到正；2020-06-08上一完整周柱到正；2024-09-30才周柱确认，次开执行未知；2015-06-17仍日DIF/周柱正却迅速失败。

**为什么接受/拒绝**：接受日DIF与日柱、周线时钟及全体原值事实；拒绝旧标签证明高质量或可成交。旧20日终点不是后续完整阶段失效退出的新入组门。在读取旧标签前已经写明唯一完整政策意图，未因弱诊断换条件。

**是否需要重新验证**：解释已完成，不改定义重算更好数字。完整政策另按TECH.R164登记；全部历史为开发，首版未认证。原R161及所有旧失败保持，最新预测模型R158和原退出R145不改变。

依据：[完整解释](../reports/research/510300_trend_expansion_phase_explanation_v1/研究结果与下一步.md)、[解释合同](../reports/research/510300_trend_expansion_phase_explanation_v1/protocol.json)、[实际解释](../reports/research/510300_trend_expansion_phase_explanation_v1/summary.json)。

## TECH.R164：唯一联合阶段出生与失效完整用途登记（2026-10-04）

**假设**：从价在EMA20上、日DIF正和上一完整周柱正的联合出生，到任一已知条件失效，可能改善后段参与及失败退出，从而提高共同账户净收益和净夏普。

**验证方法**：前述用途在R163旧标签连接前已经固定，本次六项必要阶段/账户测试通过，3488原阶段精确复现；四原A和四原纯价格的日账、订单和周期逐项一致。仅一个候选，次真实开尝试、失败不追；已知失效次合法开退出、未知本身不卖、风险只减，无固定2R/20日/加仓/终点清仓。冻结52来源，使用原20万元、252日、两时期/两费用、现金/股息/风险和T+1口径。

**结果**：接受登记及共同执行可比性，金融账户当时尚未运行。计划唯一主策略四账户，复用八精确控制；完整门为全四场景净CAGR/净Sharpe均高于零和两控制、DD<=10%、实际pB>1及pB-q>0；20/252日配对区块各2000、固定种子、全比较报告。次数软目标。

**为什么接受/拒绝**：这是不同于旧量→日柱→价格严格顺序、MA60连续确认及周牛突破/ATR用途的信息时钟；不适用原退出MSE/115成员门，不营救任何原失败。旧20日解释弱的事实保留，不临时创设账户入组门或选另一个观察通道。

**是否需要重新验证**：登记不是结果、去过拟合或独立验证；账户仅允许一次执行，实际裁决见R165。所有历史DEVELOPMENT_CALIBRATION，首版未认证。

依据：[唯一金融协议](../reports/research/510300_trend_expansion_phase_study_v1/protocol.json)、[六必要测试](../reports/research/510300_trend_expansion_phase_study_v1/tests_receipt.json)、[八控制预核对](../reports/research/510300_trend_expansion_phase_study_v1/control_preflight.json)。

## TECH.R165：联合日周展开完整账户失败及全部点位归档（2026-10-04）

**假设**：固定联合出生/失效政策能同时改善原A和纯价格完整账户的收益、夏普及实际交易质量。

**验证方法**：按R164冻结一次运行四个新主账户，不重跑或调参；原八控制复用，六必要行为测试通过。期末开放保留。保存账本资金、库存、T+1、周期净回报和共同指标复算通过，最大恒等误差4.46e-11元；全部52金融及28前置解释来源不变。所有85合格出生连接决定、真实订单、拒单和周期，当前及前原点特征与未来结果分开；0模型拟合/新训练标签/行情采集。

**结果**：较早BASE/STRESS净年化0.2923%/0.0364%、夏普0.1061/0.0273、DD7.90%/8.23%、37完成+1开放、胜率27.03%、pB0.8854/0.7986。近期BASE/STRESS年化−0.5739%/−0.7823%、夏普−0.1833/−0.2663、DD6.76%/7.37%、46完成、胜率21.74%/19.57%、pB0.6772/0.6197。四经济与稳定门均失败。压力原A较早1.8371%/0.4380、近期3.9908%/1.2169；本政策较早弱于两控制，近期仅弱负值优于纯价格，远弱于A。完整年平均7.4/6.83次虽高于A4.4/4.17，却没有足够质量。

忽略费用复刻共84实际周期（83完成/1开放）；2019-01-21至03-27净+16.14%、2020-06-17至07-27+13.47%，2015-06-18至19−4.91%、2020-06-09至16−1.04%，2024-09-30信号的10-08开4.656达到原涨停边界、没有成交，后续11月交易−2.88%。近期压力按实际份额毛损益−2151.90元、摩擦7789.52元、期末净亏9941.42元，四账户均未触发停机。不是只因费用或停机；赢家不能归入原A组合增益，也不能删掉反复失败。

**为什么接受/拒绝**：拒绝该固定完整政策并关闭，不修改指标窗口、门槛、周线延迟、退出、费用、时期或加过滤救回。接受全体点位、未成交、成本与毛路径解释；不能把增加交易次数等同于提高净收益夏普，不能以少量具体上涨解释全样本优势。

**是否需要重新验证/下一步具体实验**：旧固定政策不在同历史重试。下一仅全体85出生触发时钟归因：当前/前一已知状态确定最后到达的价格、日DIF、周柱或同日多项，全部实际83完成/1开放/1未成交保留，不把分组选择当失败政策新版本。不同机制未明确前，新金融候选未登记、已准入待跑0。目标active、本轮真实progress、阻塞0；最新金融R165，实际预测R158，原退出R145；独立验证与去过拟合未达，global DSR/PBO未计算，原真实前瞻12值、10-08 15:05时钟和其他宏观/盘口分支保持。

依据：[完整结果](../reports/research/510300_trend_expansion_phase_study_v1/研究结果与下一步.md)、[实际金融裁决](../reports/research/510300_trend_expansion_phase_study_v1/summary.json)、[保存核对](../reports/research/510300_trend_expansion_phase_study_v1/saved_result_verification.json)、[全部真实点位](../reports/research/510300_trend_expansion_phase_study_v1/results/全部实际进出点位_当时指标与A覆盖.csv)、[全部资格与拒单](../reports/research/510300_trend_expansion_phase_study_v1/results/全部合格出生信号_真实执行与未成交.csv)、[下一解释提案](../reports/research/510300_trend_expansion_phase_study_v1/next_all_event_clock_diagnostic_proposal.json)。
"""


def main():
    require(not (OUT / "closeout_receipt.json").exists(), "此固定结果已经更新长期事实，不重复。")
    result = read(OUT / "summary.json")
    require(result["technical_decision"] == "TECH.R165" and not result["all_four_economic_gates_passed"],
            "实际裁决与当前归档不符。")
    verification = read(OUT / "saved_result_verification.json")
    require(verification["status"] == "PASS_FOUR_SAVED_ACCOUNTS_METRICS_AND_LEDGER_IDENTITIES", "保存核对未通过。")
    detail = read(OUT / "actual_point_detail_summary.json")
    require(detail["distinct_actual_cycles_ignoring_cost_replications"] == 84 and detail["distinct_qualified_origins"] == 85,
            "全部资格与执行数量改变。")
    goal = read(OUT / "goal_tool_status.json")
    require(goal["goal"]["status"] == "active", "目标工具状态未确认active。")
    before = read(STATE_PATH)
    require(before["latest_technical_decision"] == "TECH.R162", "项目技术状态已由其他工作推进，不能覆盖。")
    require(all(k in before for k in FORWARD_KEYS), "原前瞻十二字段不完整。")
    forward = {k: copy.deepcopy(before[k]) for k in FORWARD_KEYS}
    require(before["latest_actual_prediction_model_decision"] == "TECH.R158"
            and before["original_exit_model_latest_decision"] == "TECH.R145", "两个原预测用途指针改变。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "归档来源不对应固定账户。")
    write_json(OUT / "project_state_before_R163_R165.json", before, exclusive=True)
    timestamp = now()
    next_action = "仅按全部85已知出生做价格/日DIF/上一完整周柱/同日多项的触发时钟归因；83完成/1开放/1未成交全部报告，不把分组反选为R165过滤或改退出。不同机制成立后才登记新金融策略。"
    state = copy.deepcopy(before)
    state.update({
        "updated_at": timestamp, "at": timestamp, "status": "research_active", "goal_achieved": False,
        "latest_completed_study": OUT.relative_to(ROOT).as_posix(),
        "latest_result": (OUT / "summary.json").relative_to(ROOT).as_posix(), "latest_report": REPORT,
        "latest_research_status": result["status"], "latest_progress": SUMMARY, "latest_overall_summary": SUMMARY,
        "latest_technical_decision": "TECH.R165", "latest_registration_decision": "TECH.R164",
        "latest_model_decision": "TECH.R165", "latest_actual_model_decision": "TECH.R165",
        "latest_actual_model_kind": "COMPLETE_RULE_POLICY_WITHOUT_ESTIMATED_PREDICTION_MODEL",
        "latest_financial_strategy_decision": "TECH.R165",
        "latest_financial_strategy_result": (OUT / "summary.json").relative_to(ROOT).as_posix(),
        "new_account_return_sharpe_result": (OUT / "summary.json").relative_to(ROOT).as_posix(),
        "current_study": result["study"], "current_phase": "FIXED_JOINT_PHASE_FINANCIAL_POLICY_REJECTED_AND_CLOSED",
        "current_direction": "具体展开阶段解释及唯一完整政策已完成并失败；下一全体触发时钟归因，新的金融策略未登记。",
        "current_priority": next_action, "next_research_action": next_action, "next_available_action": next_action,
        "next_research_plan": (OUT / "next_all_event_clock_diagnostic_proposal.json").relative_to(ROOT).as_posix(),
        "next_financial_experiment": "NOT_REGISTERED_NO_READY_CANDIDATE",
        "current_admitted_unrun_numeric_candidates": 0, "next_candidate_field_status": "NO_NEW_FINANCIAL_CANDIDATE_ADMITTED",
        "latest_trend_expansion_explanation": (EXPLANATION / "summary.json").relative_to(ROOT).as_posix(),
        "latest_trend_expansion_saved_verification": (OUT / "saved_result_verification.json").relative_to(ROOT).as_posix(),
        "latest_trend_expansion_actual_point_detail": (OUT / "actual_point_detail_summary.json").relative_to(ROOT).as_posix(),
        "next_trend_expansion_clock_explanation_status": "PROPOSED_NOT_RUN_NO_FINANCIAL_CANDIDATE",
        "new_accounts_this_continuation": 4, "new_accounts_in_current_phase": 4,
        "new_financial_candidate_accounts_this_continuation": 4,
        "new_investment_account_evaluations_this_continuation": 4,
        "new_point_replays_this_continuation": 0, "internal_reference_replays_this_continuation": 8,
        "saved_account_controls_replayed_this_continuation": 8,
        "new_model_fits_this_continuation": 0, "new_model_training_labels_this_continuation": 0,
        "new_training_labels_this_continuation": 0, "new_market_requests_this_continuation": 0,
        "necessary_tests_passed_this_continuation": 6, "necessary_test_executions_this_continuation": 9,
        "current_structural_signal_events": 85, "current_executed_unique_new_entry_events": 84,
        "current_new_policy_complete_cycles": 83, "current_new_policy_open_cycles": 1,
        "current_new_policy_unexecuted_origins": 1,
        "current_full_account_economic_gate_passed": False,
        "return_and_sharpe_changed_this_continuation": True,
        "return_and_sharpe_improved_this_continuation": False, "returns_and_sharpe_improved": False,
        "current_unmet_evidence": "联合阶段增加次数但全四场景收益夏普/实际pB未过；近期毛损益为负。不同有效金融机制和独立证据仍缺，去过拟合未建立。",
        "independent_validation_status": "NOT_ESTABLISHED", "overfitting_removed": False,
        "overfit_removed": False, "whole_model_overfitting_removed": False, "global_DSR_PBO": "NOT_COMPUTED",
        "previous_goal_turn_classification": "progress", "current_goal_turn_classification": "progress",
        "current_goal_turn_classification_reason": "完成R163全体解释、R164固定及R165四真实成本账户，有限负结果与全部实际执行改变下一研究问题。",
        "previous_goal_turn_classification_reason": "R162完整路径及覆盖解释有实际新证据，未陷入连续无进展。",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0, "blocked_audit_key": None, "blocked_reason": None,
        "goal_tool_status_confirmed": "active", "latest_goal_tool_status_receipt": (OUT / "goal_tool_status.json").relative_to(ROOT).as_posix(),
        "live_own_process_handle": None, "verified_wait": False, "orders_authorized": False,
        "current_goal_turn_actual_work": {
            "all_known_phase_rows": 3488, "joint_onsets": 85, "old_breakout_onsets": 182, "old_volume_onsets": 275,
            "old_daily_labels_preserved": 2846, "old_mature_labels": 2826, "old_unknown_labels": 20, "tail_without_old_labels": 9,
            "original_episode_rows": 61, "original_admitted_waves": 49, "charts": 4, "actual_prefix_checks": 3,
            "unique_necessary_tests_passed": 6, "test_executions_passed": 9,
            "new_primary_policy_accounts": 4, "original_A_replays": 4, "original_price_replays": 4,
            "distinct_actual_cycles": 84, "complete_cycles": 83, "open_cycles": 1, "unexecuted_known_origins": 1,
            "frozen_explanation_sources": 28, "frozen_financial_sources": 52,
            "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        },
    })
    require(all(state[k] == value for k, value in forward.items()), "当前更新改写原真实前瞻。")
    require(state["latest_actual_prediction_model_decision"] == before["latest_actual_prediction_model_decision"]
            and state["original_exit_model_latest_decision"] == before["original_exit_model_latest_decision"],
            "完整规则裁决不能覆盖原预测用途。")
    head = "\n\n> 技术线最新金融裁决 TECH.R165（优先于下方历史快照，原预测R158/退出R145分别保留）：" + SUMMARY + " [完整报告](../" + REPORT + ")。\n"
    files = [
        ROOT / "docs/PROJECT_STATE.md", ROOT / "docs/PROJECT_STATE_TECHNICAL_LINE.md",
        ROOT / "docs/RESEARCH_DECISIONS.md", ROOT / "docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md",
    ]
    old_docs = {p: p.read_text(encoding="utf-8-sig") for p in files}
    require(all("\n## TECH.R165：" not in s for s in old_docs.values()), "实际裁决已写入部分事实文件，不能重复追加。")
    documents = []
    for p, old in old_docs.items():
        first, rest = old.split("\n", 1)
        content = first + head + "\n" + rest
        content = content.rstrip() + "\n\n" + ENTRY.strip() + "\n"
        p.write_text(content, encoding="utf-8")
        require(p.read_text(encoding="utf-8") == content, "事实文档保存不一致。")
        documents.append({"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)})
    write_json(STATE_PATH, state)
    restored = read(STATE_PATH)
    require(all(restored[k] == value for k, value in forward.items()), "保存后真实前瞻十二值发生变化。")
    write_json(OUT / "closeout_receipt.json", {
        "at": timestamp, "status": "PASS_R163_R164_R165_SINGLE_FACT_UPDATE_AND_FORWARD_PRESERVATION",
        "documents": documents, "project_state_sha256": digest(STATE_PATH),
        "prior_project_state": (OUT / "project_state_before_R163_R165.json").relative_to(ROOT).as_posix(),
        "latest_financial_decision": "TECH.R165", "latest_actual_prediction_decision": "TECH.R158",
        "original_exit_decision": "TECH.R145", "forward_keys_preserved": forward,
        "goal_status": "active", "goal_achieved": False, "current_turn_classification": "progress", "blocked_count": 0,
        "new_policy_accounts_in_closeout": 0, "old_frozen_failures_preserved": True,
        "closeout_source": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": digest(Path(__file__))},
    }, exclusive=True)
    print("R163解释/R164登记/R165四实际账户已一次写入四长期事实文件及状态；原十二前瞻值、R158/R145保留，目标active。", flush=True)


if __name__ == "__main__":
    main()
