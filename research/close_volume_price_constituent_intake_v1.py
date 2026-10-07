"""关闭未登记成分参与提案：核对旧用途和未变化原源，不重算旧收益。"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.finalize_downtrend_break_state_v1 import STATE, FORWARD, DOCS

OUT = ROOT / "reports/research/510300_volume_price_constituent_intake_v1"
FINANCE = ROOT / "reports/research/510300_volume_lead_price_confirm_study_v1"
PRIOR = ROOT / "reports/research/510300_point_breadth_prior_source_routes_v1"


def run():
    require(not (OUT / "summary.json").exists(), "未登记成分提案已经裁决，不重复。")
    sources = [Path(__file__), FINANCE / "下一不同信息准入提案_成分参与广度_未登记.md",
               FINANCE / "summary.json", FINANCE / "project_state_update_receipt.json",
               PRIOR / "summary.json", PRIOR / "cases.json", PRIOR / "source_facts.json",
               PRIOR / "evidence_manifest.json", PRIOR / "saved_output_verification_receipt.json",
               ROOT / "research/point_breadth_prior_source_routes_v1.py",
               ROOT / "research/breadth_majority_inputs_v1.py", ROOT / "research/breadth_learned_gate_inputs_v1.py",
               ROOT / "reports/research/510300_breadth_majority_trend_v1/acceptance_outcome.json",
               ROOT / "reports/research/510300_breadth_learned_entry_gate_v1_output_fix/acceptance_outcome.json",
               ROOT / "data/raw/constituents/000300_constituent_daily.parquet",
               ROOT / "data/raw/constituents/000300_historical_weights.parquet"]
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_VOLUME_PRICE_CONSTITUENT_INTAKE_V1", "decision": "TECH.R190",
        "scope": "仅关闭R189未登记泛成分参与提案：当前两原文件字节与旧R110清单/元数据一致，六原路由及两真实旧用途裁决读取，不重算源上界/收益，不形成新完整策略。",
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_requests": 0,
        "no_global_invalidity_claim": True,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources],
    }, exclusive=True)
    manifest = read(PRIOR / "evidence_manifest.json")
    facts = read(PRIOR / "source_facts.json")
    observed = []
    for name, rows, first, last in (
        ("data/raw/constituents/000300_constituent_daily.parquet", 484200, "2019-12-23", "2026-08-19"),
        ("data/raw/constituents/000300_historical_weights.parquet", 36000, "2016-08-31", "2026-07-31")):
        matches = [entry for entry in manifest if entry["path"].replace("\\", "/") == name]
        require(len(matches) == 1 and digest(ROOT / name) == matches[0]["sha256"], "当前原数据已变化，不能复用旧日期事实。")
        meta = [entry for entry in facts["raw_metadata"] if entry["path"] == name]
        require(len(meta) == 1 and (meta[0]["rows"], meta[0]["date_start"], meta[0]["date_end"]) == (rows, first, last), "真实日期列核对与原元数据不一致。")
        observed.append({**meta[0], "current_bytes_same_as_R110": True,
                         "current_terminal_date_column_read_confirmed": True,
                         "metadata_evidence": "本轮终端已实际读取date/trade_date两列，确认行数、起止及日期数；原清单哈希完全相同，不重读或补样本。"})
    prior = read(PRIOR / "summary.json")
    receipt = read(PRIOR / "saved_output_verification_receipt.json")
    require(prior["technical_decision"] == "TECH.R110" and prior["completed_prior_routes"] == 6
            and receipt["status"] == "PASS_SAVED_SIX_BREADTH_ROUTES_AND_FOUR_SOURCE_UPPER_BOUND_VERIFICATION_V1_0_1",
            "六旧用途实际核对未完成。")
    cases = read(PRIOR / "cases.json")
    statuses = {"majority": read(ROOT / "reports/research/510300_breadth_majority_trend_v1/acceptance_outcome.json")["status"],
                "learned_gate": read(ROOT / "reports/research/510300_breadth_learned_entry_gate_v1_output_fix/acceptance_outcome.json")["status"]}
    route_rows = [{"id": c["id"], "name": c["name"], "old_actual_routing_status": c["routing_status"],
                   "old_result_description": c["result"], "old_complete_use_only_not_new_financial_result": True} for c in cases]
    conclusion = (
        "R189未登记泛成分参与提案不进入金融：六种既有广度/传播用途已经在R110有限核对，旧多数广度仅局部优于弱价格对照、未达目标，"
        "旧学习进入广度过滤在两时期两费用均更差；不能重新包装为新候选。当前原成分OHLC/量/额仍484200行，2019-12-23至2026-08-19，"
        "未覆盖原2015—2019研究主体；权重36000行/120月、2016-08-31至2026-07-31，首次公布证据和行业历史合同未建立。两文件与R110字节完全相同，未新增合格资料。"
        "旧2015起breadth20缓存不等于未存在的同期成员原始量额，不能由比例还原成交额参与、用今天成员倒填、拼原A补未知或选近期缩样本。"
        "本提案未提供经旧用途核对的不同完整规则，故NOT_ADMITTED，不重算旧回测。"
        "原R110的115模型完整支持上界仅适用于原预测用途；本次不把该门套给无拟合规则，不断言所有成分机制或外部资料无效。")
    next_action = (
        "当前无已准入待跑金融实验。后续金融须先有实质不同的可知信息和完整用途，或真正新样本/真实来源实现错误；旧失败和泛成分提案不重开。"
        "原A/POINT前瞻按既定来源回执和真实日期推进，最早尚未发生合格收盘2026-10-08T15:05:00+08:00；现在不得回填或制造观察。"
        "既有样本无独立验证，不能宣称已提高完整收益夏普或去除过拟合，目标保持active。")
    report = OUT / "成分参与提案实际核对与关闭.md"
    lines = ["# 未登记成分参与提案的实际核对", "", conclusion, "",
             "这里关闭的是这张尚无不同完整用途、原源未变化的提案，不关闭整个项目，也不否定所有成分信息。0新账户/拟合/标签/采集，旧第110轮四源上界不重算，旧115模型门不扩大适用。", "",
             "| 原路由 | 旧完整用途 | 原实际状态 |", "|---|---|---|"]
    for row in route_rows:
        lines.append(f"| {row['id']} | {row['name']} | {row['old_actual_routing_status']} |")
    lines.extend(["", f"旧多数广度裁决：{statuses['majority']}；旧学习进入过滤裁决：{statuses['learned_gate']}。旧242日/旧终点账户不能与当前252日/截至9月30日直接排名。", "",
                  next_action, "", "最近真实金融仍R189：早压力净年化0.8591%/夏普0.2726，近压力0.7435%/0.2606，均弱于A，原固定完整用途已拒绝。", "",
                  "[R189具体上涨和完整账户](../510300_volume_lead_price_confirm_study_v1/研究结果与下一步.md)、[六旧路由实际核对](../510300_point_breadth_prior_source_routes_v1/summary.json)。", ""])
    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    summary = {"at": now(), "technical_decision": "TECH.R190",
               "status": "NOT_ADMITTED_UNREGISTERED_CONSTITUENT_PARTICIPATION_PROPOSAL_PRIOR_USES_AND_UNCHANGED_INCOMPLETE_RAW_SOURCE",
               "current_raw_source_facts": observed, "old_six_routes": route_rows, "actual_old_statuses": statuses,
               "current_new_complete_policy_defined": False, "new_admitted_unrun_candidates": 0,
               "legacy_115_model_gate_applied_to_new_rule": False, "new_accounts": 0, "new_fits": 0,
               "new_training_labels": 0, "new_market_requests": 0, "goal_achieved": False,
               "goal_turn_classification": "progress", "independent_validation": "NOT_ESTABLISHED",
               "global_DSR_PBO": "NOT_COMPUTED", "next_action": next_action, "conclusion": conclusion}
    write_json(OUT / "summary.json", summary, exclusive=True)
    state = read(STATE)
    require(state["latest_technical_decision"] == "TECH.R189", "技术状态已由其他阶段推进，不覆盖。")
    state_hash = digest(STATE)
    forward_before, next_before = {k: state[k] for k in FORWARD}, state["next_experiment"]
    state["archived_R189_phase_before_R190_local_intake"] = {
        "at": now(), "phase": state["current_phase"], "trial_accounting": state["current_phase_trial_accounting"],
        "next_research_plan": state["next_research_plan"], "priority": state["current_priority"]}
    rel = OUT.relative_to(ROOT).as_posix()
    marker = "> 技术线当前事实 TECH.R190（优先于下方技术线历史快照，2026-10-05）："
    for name in DOCS:
        p = ROOT / "docs" / name
        original = p.read_text(encoding="utf-8-sig")
        require(marker not in original, "R190事实已经更新，不重复。")
        first, rest = original.split("\n", 1)
        top = marker+conclusion+"最近实际金融R189/登记R188、原预测R158/退出R145保持，独立0，目标active，待跑金融0。\n\n"+f"依据：[当前实际核对](../{rel}/{report.name})。下一步：{next_action}\n\n"
        appendix = "\n\n## TECH.R190：关闭未登记成分参与提案，保留旧用途与不完整原源（2026-10-05）\n\n"
        if name.startswith("RESEARCH_DECISIONS"):
            appendix += "**假设**：标的量价之外的成分参与信息是否已有不同用途及可用资料，足以进入下一金融实验？\n\n**验证方法**：读取原6路完整定义与实际裁决；当前两原文件哈希对R110逐字一致、date/trade_date真实列核对行数及日期；不重算旧115源上界、旧收益或新规则。\n\n**结果**："+conclusion+"\n\n**为什么接受/拒绝**：拒绝准入这张未登记提案，不宣称所有成分研究无效；多数广度和进入过滤原裁决保持，原源没有新增。\n\n**是否需要重新验证**：只在实质不同完整用途、合格当时成员/权重/量额/发布资料或真正新样本/真实错误出现时重新准入，不通过选近期、补A/等权或旧缓存代用营救。\n"
        else:
            appendix += conclusion+"\n\n"+next_action+"\n"
        p.write_text(first+"\n\n"+top+rest.lstrip("\n")+appendix, encoding="utf-8", newline="\n")
    state.update({"at": now(), "updated_at": now(), "latest_technical_decision": "TECH.R190",
        "latest_completed_study": rel, "latest_result": rel+"/summary.json", "latest_report": rel+"/"+report.name,
        "latest_research_report": rel+"/"+report.name, "latest_research_status": summary["status"],
        "latest_progress": conclusion, "latest_overall_summary": conclusion, "latest_continuation_outcome": conclusion,
        "current_study": "510300_VOLUME_PRICE_CONSTITUENT_INTAKE_V1",
        "current_phase": "UNREGISTERED_COMPONENT_PARTICIPATION_PROPOSAL_CLOSED_PRIOR_USES_AND_UNCHANGED_RAW_SOURCE",
        "current_priority": next_action, "next_research_action": next_action, "next_available_action": next_action,
        "next_research_question": next_action, "next_research_plan": rel+"/summary.json",
        "next_experiment_status": "NO_NEW_ADMITTED_FINANCIAL_POLICY_AFTER_CONSTITUENT_LOCAL_INTAKE",
        "next_candidate_field_status": "UNREGISTERED_COMPONENT_PARTICIPATION_PROPOSAL_NOT_ADMITTED",
        "next_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE", "current_admitted_unrun_numeric_candidates": 0,
        "latest_continuation_receipt": rel+"/project_state_update_receipt.json",
        "latest_new_information_admission_boundary": rel+"/summary.json",
        "current_phase_trial_accounting": {"scope": "TECH_R190_UNREGISTERED_COMPONENT_PROPOSAL_LOCAL_INTAKE_ONLY",
            "current_raw_sources_confirmed_unchanged": 2, "old_routes_read": 6, "new_accounts": 0, "new_fits": 0,
            "new_labels": 0, "new_market_requests": 0, "new_numeric_candidates": 0},
        "current_goal_turn_actual_work": {"financial_phase": state["actual_candidate_trials"], "post_result_intake": "TECH.R190", "new_accounts_total": 4},
        "current_goal_turn_classification": "progress", "goal_turn_progress_classification": "FOUR_R189_ACTUAL_ACCOUNTS_AND_CLOSED_UNREGISTERED_COMPONENT_PROPOSAL",
        "current_goal_turn_classification_reason": "量先价后完整用途真实拒绝并归档；下一泛成分提案已由旧用途及未变化不完整原源关闭，未循环重算旧指标。",
        "latest_constituent_intake": rel+"/summary.json", "new_accounts_in_current_phase": 0,
        "goal_status": "active", "goal_tool_status_confirmed": "active", "goal_achieved": False,
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0,
        "current_unmet_evidence": "R189四场景仍弱于A、交易更少；未登记成分提案不准入，独立0，去过拟合未建立，待跑金融0。"})
    require({k: state[k] for k in FORWARD} == forward_before and state["next_experiment"] == next_before, "原前瞻改变。")
    require(digest(STATE) == state_hash, "状态同时被修改，不覆盖。")
    write_json(STATE, state)
    write_json(OUT / "project_state_update_receipt.json", {
        "at": now(), "status": "PASS_R190_FOUR_FACT_FILES_AND_STATE_UPDATED_ONCE",
        "latest_actual_financial": "TECH.R189", "latest_registration": "TECH.R188",
        "latest_prediction": "TECH.R158", "original_exit": "TECH.R145", "goal_status": "active",
        "goal_achieved": False, "new_financial_candidate_accounts": 0, "new_admitted_unrun_candidates": 0,
        "forward_values_unchanged": forward_before, "next_forward_experiment_unchanged": next_before,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                    for p in [Path(__file__), OUT / "summary.json", report, STATE, *(ROOT / "docs" / n for n in DOCS)]]}, exclusive=True)
    print("R190两原文件未变化、六旧用途和真实源范围已记录；未登记提案不准入，原R189金融/原策略/真实前瞻保持。", flush=True)


if __name__ == "__main__":
    run()
