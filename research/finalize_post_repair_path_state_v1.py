"""一次写回已完成的全事件解释，保留原金融裁决及真实前瞻。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "reports/research/510300_post_repair_path_explanation_v1"
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
DOCS = [ROOT / "docs" / n for n in ["PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"]]
FORWARD_KEYS = [
    "forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
    "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
    "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
    "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation",
]
FINANCIAL_KEYS = [
    "latest_model_decision", "latest_actual_model_decision", "latest_actual_prediction_model_decision",
    "latest_actual_prediction_result", "latest_prediction_technical_decision", "latest_financial_strategy_decision",
    "latest_financial_strategy_result", "original_exit_model_latest_decision", "original_exit_model_latest_result",
    "actual_candidate_trials", "new_account_return_sharpe_result",
]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    receipt_path = OUT / "project_state_update_receipt.json"
    snapshot_path = OUT / "state_before_TECH_R162.json"
    if receipt_path.exists() or snapshot_path.exists():
        raise RuntimeError("本次解释状态写回已经开始或完成。")
    summary = read(OUT / "summary.json")
    verification = read(OUT / "saved_explanation_verification.json")
    protocol = read(OUT / "protocol.json")
    goal = read(OUT / "goal_tool_status.json")
    if verification["status"] != "PASS_ALL_KNOWN_EVENTS_OLD_LABELS_CYCLES_AND_ORIGINAL_EPISODE_MEMBERSHIP":
        raise RuntimeError("全体路径保存结果未完成核对。")
    if summary["new_accounts"] or summary["new_fits"] or summary["goal_achieved"]:
        raise RuntimeError("本轮解释阶段不得冒充新策略或目标完成。")
    if goal["goal_tool_result"]["goal"]["status"] != "active":
        raise RuntimeError("目标工具不是active。")
    for source in protocol["sources"]:
        if sha((ROOT / source["path"]).read_bytes()) != source["sha256"]:
            raise RuntimeError("冻结来源变化：" + source["path"])
    originals = {p: p.read_bytes() for p in [STATE, *DOCS]}
    state = json.loads(originals[STATE].decode("utf-8-sig"))
    forward = {k: state[k] for k in FORWARD_KEYS}
    financial = {k: state[k] for k in FINANCIAL_KEYS}
    if state["latest_technical_decision"] != "TECH.R161":
        raise RuntimeError("项目已出现另一项技术裁决，需先核对并发状态。")
    out_rel = "reports/research/510300_post_repair_path_explanation_v1"
    progress = (
        "2026-10-04 TECH.R162按用户顺序完成全体修复、失效、再次确认及A覆盖解释。"
        "3488已知状态、187价格确认、原186成熟标签及1尾部未知、187原实际周期（186完成/1开放）、原61分段/49正式上涨段全部保留。"
        "动量保留29、重建64、未正93、首次1；保留组旧20日正比例62.07%，原实际交易31.03%/平均−0.18%/pB0.577；全体三常规组实际pB均<1。"
        "2020后段原价格周期06-17入/07-27出净+13.65%，周柱正但量方向负，不能用单例反选过滤。"
        "正式49上涨段有15无A库存、18无正目标、7无区间内均线上穿；187确认有60已有库存或正目标，127均无，不是127盈利点。"
        "2019/2020波段A分别持4/72和4/75收盘，不是完全缺席；2024确认日库存0但目标正。"
        "3测试最终通过，首次None/NaN前缀表示失败及原实现保留，登记前统一Int64未知，不改分类或放宽断言；4实际前缀、28冻结来源、4图和案例逐日原值完成。"
        "本轮0新账户/拟合/训练标签/采集。下一先解释均线之上趋势展开及量价扩张/失败，旧突破等定义先核对，不营救R161；新金融候选0。"
        "最新金融仍R161、实际预测R158、原退出R145；收益夏普目标/独立/去过拟合未达，目标active，阻塞计数0，原前瞻保持。"
    )
    for k, v in list(state.items()):
        if k.endswith("_this_continuation") and not k.startswith("prior_"):
            if isinstance(v, bool):
                state[k] = False
            elif isinstance(v, (int, float)):
                state[k] = 0
    work = {
        "all_origin_known_path_rows": 3488, "all_known_price_confirmation_events": 187,
        "reused_original_20d_labels": 186, "tail_events_without_new_labels": 1,
        "reused_original_actual_price_cycles": 187, "complete_cycles": 186, "open_cycles": 1,
        "original_episode_rows": 61, "original_admitted_upward_episodes": 49,
        "charts": 4, "necessary_tests_final_passed": 3, "pre_registration_test_failures_preserved": 1,
        "actual_prefix_checks": 4, "frozen_sources": 28, "new_policy_accounts": 0,
        "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
    }
    state.update({
        "at": summary["at"], "updated_at": summary["at"], "status": "research_active", "goal_achieved": False,
        "latest_technical_decision": "TECH.R162", "latest_completed_study": out_rel,
        "latest_result": out_rel + "/summary.json", "latest_report": out_rel + "/研究结果与下一步.md",
        "latest_research_report": out_rel + "/研究结果与下一步.md", "latest_research_status": summary["status"],
        "latest_progress": progress, "latest_overall_summary": progress, "latest_continuation_outcome": progress,
        "latest_continuation_receipt": out_rel + "/summary.json", "latest_goal_tool_status_receipt": out_rel + "/goal_tool_status.json",
        "goal_tool_status_confirmed": "active", "current_phase": "ALL_EVENT_REPAIR_AND_RECONFIRMATION_EXPLANATION_COMPLETED",
        "current_study": protocol["study"], "current_direction": "修复与再次确认解释完成；趋势已在均线上方的展开阶段待解释，下一金融实验未登记。",
        "current_priority": "保留全体失败，先解释均线上方趋势展开、量价承接与扩张及A覆盖；核对旧突破/方向/压缩等完整用途，不复活旧失败。",
        "current_unmet_evidence": "动量保留标签正比例高，但原实际pB<1；无新独立政策或收益夏普改善，独立验证及去过拟合未建立。",
        "next_available_action": "解释原49段确认阶段之后的全部已有价格突破/量价扩张时点及失败对照；当时特征不能含事后低高点，先审旧定义再判断不同机制。",
        "next_experiment_status": "TREND_EXPANSION_PATH_EXPLANATION_PROPOSED_BEFORE_NEW_FINANCIAL_DEFINITION",
        "next_financial_experiment": "NOT_REGISTERED_NO_READY_CANDIDATE",
        "next_strategy_increment_status": "NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY", "current_admitted_unrun_numeric_candidates": 0,
        "latest_post_repair_path_explanation": out_rel + "/summary.json",
        "latest_post_repair_path_verification": out_rel + "/saved_explanation_verification.json",
        "latest_post_repair_coverage": out_rel + "/coverage_findings.json",
        "current_goal_turn_actual_work": work, "current_phase_trial_accounting": work,
        "actual_candidate_trials_role": "LATEST_FINANCIAL_R161_TRIAL_ACCOUNTING_NOT_CURRENT_EXPLANATION",
        "previous_goal_turn_classification": "progress",
        "previous_goal_turn_classification_reason": "R160—R161一次实际八账户及九点位，有限负结果改变下一研究方向。",
        "current_goal_turn_classification": "PROGRESS_NEW_ALL_EVENT_PATH_AND_COVERAGE_EVIDENCE",
        "current_goal_turn_classification_reason": "全体已知路径与两种旧结果分开，还原A覆盖及标签/退出矛盾，否定机械再确认晋级，明确均线外展开的下一解释范围。",
        "goal_turn_progress_classification": "NEW_EXPLANATORY_EVIDENCE_NO_NEW_FINANCIAL_POLICY",
        "necessary_tests_passed_this_continuation": 3, "necessary_tests_passed_in_current_phase": 3,
        "saved_field_rows_recomputed_this_continuation": 3488, "saved_daily_information_rows_this_continuation": 3488,
        "saved_old_account_rows_extracted_this_continuation": 187,
        "saved_account_rows_extracted_without_recomputation_this_continuation": 187,
        "saved_old_account_rows_reused_this_continuation": 187, "saved_account_overlap_tables_read_this_continuation": 1,
        "source_freeze_count_this_continuation": 28, "latest_source_freeze_count": 28,
        "new_accounts_in_current_phase": 0, "new_network_requests_this_continuation": 0,
        "literature_lookup_this_continuation": {"page_open_attempts": 0, "paper_find_requests": 0, "market_data_requests": 0},
        "validation_method_this_continuation": "3测试最终通过/初次未知编码失败保留；4真实前缀，187原点/186旧标签/186完成与1开放周期、原61分段完全保留核对，四图已查看；0金融复跑。",
        "current_member_support_this_continuation": "187当前价格确认全部入解释，原186标签与尾部无旧标签分别保留；原实际187周期含1开放，不由未来结果筛已知状态。",
        "current_study_prediction_gate_status": "NOT_APPLICABLE_EXPLANATION_ONLY",
        "economic_stage_status": "NOT_RUN_NO_NEW_FINANCIAL_POLICY_REGISTERED",
        "account_return_sharpe_this_continuation": "NOT_COMPUTED_NEW_POLICY_NONE_OLD_CYCLE_SUBSETS_DESCRIPTIVE_ONLY",
        "account_return_sharpe_reason": "本轮只复用原实际周期作路径解释，未生成新的政策账户或收益夏普。",
        "return_and_sharpe_improved_this_continuation": False, "returns_and_sharpe_improved": False,
        "independent_validation_status": "NOT_ESTABLISHED", "whole_model_overfitting_removed": False,
        "overfit_removed": False, "overfitting_removed": False, "global_DSR_PBO": "NOT_COMPUTED",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0, "blocked_reason": None,
        "blocked_audit_key": None, "blocking_decision": None, "verified_wait": False, "live_own_process_handle": None,
        "code_files_added_this_continuation": ["research/post_repair_path_inputs_v1.py", "research/post_repair_path_explanation_v1.py",
                                             "research/post_repair_path_closeout_v1.py", "research/finalize_post_repair_path_state_v1.py",
                                             "tests/test_post_repair_path_inputs_v1.py"],
        "report_programs_added_this_continuation": ["research/post_repair_path_closeout_v1.py", "research/finalize_post_repair_path_state_v1.py"],
        "original_strategy_source_files_changed_this_continuation": 0,
    })
    if {k: state[k] for k in FORWARD_KEYS} != forward or {k: state[k] for k in FINANCIAL_KEYS} != financial:
        raise RuntimeError("原前瞻或金融/预测/退出裁决发生变化。")
    section = """
## TECH.R162：全部修复、失效及再次确认路径解释（2026-10-04）

**假设**：再次站回EMA20时，上一确认的MACD正段若穿过价格失效仍保留，可能解释较好的延续；先解释全体结果及A覆盖，不能直接从2020后段赢家反推过滤。

**验证方法**：逐日先生成当时路径：上次合格价格确认、此后首次价格失效、MACD当前正段是否持续穿越该过程，只有保留/重建/未正及首个/未知一种固定分层。3488原点、当前187确认全保留，再连接原186成熟20日标签和原纯价格账户187实际周期（186完成/1开放）；1尾部无旧标签不填。原61分段/49正式上涨段不重分，事后低高点只作图示与覆盖对齐。三测试后固定28来源，旧三时期和ALL全部40单元报告，不搜索量/周线/波动组合。四原案例四面板与逐日原值已查看，四真实前缀通过。本轮0新拟合/训练标签/账户/采集。

**结果**：187确认为动量保留29、重建64、未正93、首个1。保留组旧20日18/29为正（62.07%、pB0.619），原实际只有9/29盈利（31.03%、平均净−0.18%、pB0.577）；重建/未正全体原实际pB0.893/0.469，均未满足用户pB>1。2024—2026重建组pB1.709而早两期0.843/0.516，不选有利时期。原实际子集不是独立分组策略账户，原现金/份额路径保持，不推算新收益夏普或证明尚未运行策略必亏。

2020具体路径：04-07早段重建；04-14动量保留但量方向负、周柱负，原纯价格交易04-15至05-25净−0.14%；06-01突破20日高点但日柱略负，原周期净+0.40%；06-16周柱正、日柱重建、量方向仍负，原周期06-17至07-27净+13.65%。这个后段赢家不是R161严格顺序政策收益。原A四确认日均库存0/目标0，但6月30日给正目标、7月1/2/3/6实际持仓，波段内共4/75收盘。

49正式上涨段中15段无A库存、18无正目标、7无区间内均线价格上穿；187确认中60已有A库存或正目标，127两者均无。2019波段A只持4/72收盘；2024确认日库存0却目标正。覆盖短/覆盖缺口是回顾事实，不等于可成交新增利润或127盈利点位。

**为什么接受/拒绝**：接受跨阶段状态与全体反例、标签和实际退出差以及A覆盖事实；拒绝机械动量保留再确认已具备高质量优势、用20日62.07%当实际胜率、用后段赢家或最近分组作为新策略有效证据。独立金融实验没有登记，不把解释层拒绝扩大为未运行的新完整策略必然失败；全部旧固定失败保持。

**是否需要重新验证/下一步具体实验**：本固定解释完成，不改分类或选时期争取更好的统计。下一解释价格已经在均线上方的趋势展开及量价承接/扩张，覆盖原49段全部已确认阶段、既有突破与失败对照；先核对旧RANGE20_BREAK、5%方向确认、压缩/回踩/供给/价量相关完整用途，避免复活失败。不用事后低高点或未来涨幅定义候选，不调整R161顺序/退出营救。实质不同机制成立后才登记新金融合同；当前待跑0，原十二前瞻字段保持。

首次前缀测试因未知持续天数None与NaN编码不同失败，原实现/失败保留；登记前统一为明确Int64空值，没有改分类或放宽断言，三测试最终通过。最新技术解释R162，最新实际金融政策仍R161、预测模型R158、原退出增量R145。目标工具active、本轮progress、阻塞计数0；独立验证、去过拟合与提高收益夏普未达，global DSR/PBO未算；其他宏观/盘口分支状态与授权保持。

依据：[完整解释](../reports/research/510300_post_repair_path_explanation_v1/研究结果与下一步.md)、[全部当时已知确认](../reports/research/510300_post_repair_path_explanation_v1/results/全部价格确认_当时路径与A已知覆盖.csv)、[全部旧结果与原实际分层](../reports/research/510300_post_repair_path_explanation_v1/results/动量路径分层_旧标签及原实际周期全部报告.csv)、[原61段覆盖](../reports/research/510300_post_repair_path_explanation_v1/results/原上涨段全集_完整路径及A覆盖回顾.csv)、[唯一解释口径](../reports/research/510300_post_repair_path_explanation_v1/protocol.json)、[实际解释结果](../reports/research/510300_post_repair_path_explanation_v1/summary.json)。
"""
    banner = "> 技术线最新解释 TECH.R162（优先于下方历史摘要，金融裁决仍R161）：" + progress + " [完整报告](../" + out_rel + "/研究结果与下一步.md)。\n\n"
    replacements = {STATE: (json.dumps(state, ensure_ascii=False, indent=2)+"\n").encode("utf-8")}
    for path in DOCS:
        text = originals[path].decode("utf-8-sig")
        if "## TECH.R162：" in text:
            raise RuntimeError("事实文件已记录本次解释："+str(path))
        first, rest = text.split("\n", 1)
        replacements[path] = (first+"\n\n"+banner+rest+"\n"+section).encode("utf-8")
    for path, data in originals.items():
        if path.read_bytes() != data:
            raise RuntimeError("同一事实文件已被另一工作修改："+str(path))
    with snapshot_path.open("xb") as f:
        f.write(originals[STATE])
    for path, data in replacements.items():
        with path.open("wb") as f:
            f.write(data)
    updated = read(STATE)
    if {k: updated[k] for k in FORWARD_KEYS} != forward or {k: updated[k] for k in FINANCIAL_KEYS} != financial:
        raise RuntimeError("写回后原前瞻或实际裁决改变。")
    for p in DOCS:
        if originals[p].decode("utf-8-sig").split("\n", 1)[1] not in p.read_bytes().decode("utf-8-sig"):
            raise RuntimeError("原事实正文没有完整保留："+str(p))
    payload = {"status": "PASS_R162_SINGLE_EXPLANATION_FACT_UPDATE_FINANCIAL_AND_FORWARD_PRESERVED",
               "scope": "DAILY_WEEKLY_TECHNICAL_LINE_ONLY", "financial_pointers_preserved": financial,
               "forward_twelve_fields_preserved": forward, "goal_status": "active", "goal_achieved": False,
               "new_fits_or_account_runs": 0, "files": [{"path": p.relative_to(ROOT).as_posix(),
                   "before_sha256": sha(originals[p]), "after_sha256": sha(p.read_bytes())} for p in originals]}
    with receipt_path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("已一次写回R162解释及四事实文件；原金融R161/预测R158/退出R145与十二前瞻字段保持，目标active未达。")


if __name__ == "__main__":
    main()
