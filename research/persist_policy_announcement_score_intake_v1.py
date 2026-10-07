"""将实际政策准入结果写入长期事实来源，保留最新现有正文与前瞻协议。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pandas as pd

from research import policy_announcement_score_intake_export_adapter_v1 as study

core = study.parent
ROOT, OUT = core.ROOT, study.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
DOCS = [ROOT / "docs" / name for name in ["PROJECT_STATE.md", "RESEARCH_DECISIONS.md",
                                         "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"]]
PROTECTED = ["forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
             "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
             "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
             "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation", "next_experiment"]


def main():
    if (OUT / "project_state_update_receipt.json").exists():
        raise FileExistsError("本研究的事实更新已经完成，不重复写入。")
    summary = core.load(OUT / "summary.json")
    assert summary["decision"] == "TECH.R194" and summary["new_accounts"] == summary["new_model_fits"] == 0
    stamp = core.now()
    before = core.load(STATE)
    protected = {key: copy.deepcopy(before.get(key)) for key in PROTECTED}
    core.save(OUT / "state_before_TECH_R194.json", before)
    backup_dir = OUT / "docs_before_TECH_R194"
    backup_dir.mkdir()
    report = "../reports/research/510300_policy_announcement_score_intake_v1_export_adapter/政策信息加入评分_具体案例反例与准入结论.md"
    text = (
        "TECH.R193—R194政策评分输入核对（2026-10-05，当前技术线来源裁决；最近实际金融仍TECH.R192）："
        "已核对全部24节点/20不同原件、六链、490目录、原3488日/四案例240行/61分段49上涨，以及九份既有用途结果。"
        "预告目标与实际状态确为不同信息：2024-09-24前日DR0071.885%减统计日利率1.7%=0.185pp，而减预告目标1.5%=0.385pp；"
        "原因子经济含义不改，前日统计合同不是实现错误。24政策节点预期均UNKNOWN，目录不是完整政策母集；"
        "84个月LPR槽中56有调查/28缺失，也不能转交7天逆回购预期。旧7次降息五次一般下调4负、20日费用后均值−2.2710%；"
        "旧政策20日账户压力夏普0.6896/年化4.6075%、首笔约80.4%净盈利，失败保持。"
        "接受来源语义分离，拒绝本提案进入数值评分，未宣称所有政策机制无效。"
        "0新账户/拟合/股票标签/网络请求，新收益夏普NOT_COMPUTED；独立验证/去过拟合/完整目标未达，goal active。"
        "原元数据保存错误及隔离格式适配保留，0金融重跑。原A/POINT、12前瞻值及原下一前瞻计划不变。"
    )
    next_text = (
        "当前无已准入待跑金融候选、无登记中的下一政策源申请。后续只能在可覆盖的公告母集、当时原件/版本钟与"
        "实质不同完整用途成立后另登记；声称政策意外须同工具事前共识。不得用24选择节点/经济日期补0或按2024行情加分，"
        "不得调原联合树或旧政策规则救援。真实新资料与原前瞻按既定来源合同推进，最早合格新收盘仍2026-10-08T15:05+08。"
    )
    marker = "> 当前技术线来源事实 TECH.R193—R194"
    top = f"{marker}（优先于下方技术线历史快照）：{text}\n\n依据：[具体量价、政策时钟、全部反例与准入结论]({report})。\n\n{next_text}\n\n"
    ledger = f"""
### TECH.R193｜政策公告目标与原评分的来源、用途核对登记

- **假设**：预告目标、已生效资金状态与事前共识提供不同信息，可核对是否支持新完整评分用途。
- **验证方法**：登记固定24节点/7旧事实/7降息/490目录，原3488日/四案例/49上涨，同16:00保守上界连接；九份原用途结果不重跑；两固定说明日历不检验新增收益。
- **结果**：本地原件与连接已完成；元数据Parquet混合类型失败保留，隔离适配只变保存表示，来源钟及数值不改。
- **为什么接受/拒绝**：接受来源核对登记，不接受其为新的金融候选、首版证据或目标达成。
- **是否需要重新验证**：仅真实来源/实现错误补充核对；不能重新筛选节点与日期，实际裁决见R194。

### TECH.R194｜政策新信息确不同，但当前选择链与预期不足以进入新评分

- **假设**：当前原政策信息足以形成不同完整评分输入，改善原联合用途。
- **验证方法**：24节点原件定位实际通过、20不同原件/六链；全490目录/3488日/240案例/61分段49上涨连接；旧全部7降息、LPR84槽与九用途实际结果核对，保持工具/时期/单位及公布钟。
- **结果**：{text}
- **为什么接受/拒绝**：接受宣布中的未来条件与实现值不同；拒绝本次数值准入。政策母集不完整、24共识未知，LPR共识不可跨工具转交；旧政策账户与反例保留，未证明追加政策能修复原树。单一2024收益不可拟合永久利好权重。
- **是否需要重新验证**：只有明确可覆盖的公告系列及当时原件/版本钟、同工具预期或真正新样本、实质不同完整用途才另立实验；当前不自动另登记源/金融变体。不认定所有政策或宏观方向无效。
- **依据**：[实际连接、全部反例与准入结论]({report})；[事前用途卡](510300_POLICY_ANNOUNCEMENT_SCORE_INTAKE_V1.md)。
"""
    changes = []
    for path in DOCS:
        payload = path.read_bytes()
        shutil.copy2(path, backup_dir / path.name)
        body = payload.decode("utf-8-sig")
        if marker in body:
            raise ValueError("重复技术来源事实：" + path.name)
        first_line, separator, rest = body.partition("\n")
        result = first_line + "\n\n" + top + rest.lstrip("\r\n")
        if "RESEARCH_DECISIONS" in path.name:
            result = result.rstrip() + "\n" + ledger + "\n"
        if path.read_bytes() != payload:
            raise RuntimeError("正文在本次更新时变化，保持他线程内容后再合并：" + path.name)
        path.write_text(result, encoding="utf-8")
        changes.append({"path": core.rel(path), "before_sha256": core.digest(backup_dir / path.name),
                        "after_sha256": core.digest(path), "preserved_entire_previous_body": True})
    current = core.load(STATE)
    if current != before:
        raise RuntimeError("状态在本次文档更新时变化，未覆盖状态文件。")
    archived = {key: value for key, value in before.items() if key.startswith("current_phase") or key.endswith("this_continuation")
                or key in ["current_phase", "new_accounts_in_current_phase", "current_goal_turn_actual_work", "goal_turn_classification"]}
    current["archived_before_TECH_R194_source_intake"] = {"at": stamp, "fields": archived}
    for key in list(current):
        if key.endswith("this_continuation") and isinstance(current[key], (int, float)) and not isinstance(current[key], bool):
            current[key] = 0
    current.update({
        "updated_at": stamp, "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_completed_study": core.rel(OUT), "latest_result": core.rel(OUT / "summary.json"),
        "latest_report": core.rel(OUT / "政策信息加入评分_具体案例反例与准入结论.md"),
        "latest_research_status": summary["status"], "latest_progress": text,
        "current_study": summary["study"], "current_phase": "POLICY_ANNOUNCEMENT_SOURCE_INTAKE_CLOSED_NOT_ADMITTED",
        "current_direction": "先解释具体上涨和全部反例，区分政策预告、实际资金状态、事前共识，再判定不同完整用途。",
        "latest_technical_decision": "TECH.R194", "latest_information_intake_technical_decision": "TECH.R194",
        "latest_prior_review_technical_decision": "TECH.R194", "latest_policy_information_intake": core.rel(OUT / "summary.json"),
        "latest_actual_model_kind": "PAIRED_TECH_AND_MACRO_THREE_CLASS_FIRST_PASSAGE_DECISION_TREES_TECH_R192",
        "latest_actual_financial_decision": "TECH.R192", "new_accounts_in_current_phase": 0,
        "necessary_tests_passed_in_current_phase": 0, "current_phase_required_tests": 0,
        "current_phase_financial_candidate_configurations": 0, "current_admitted_unrun_numeric_candidates": 0,
        "current_phase_known_daily_rows": 3488, "current_phase_original_state_rows": 3488,
        "current_phase_information_scope": "LOCAL_POLICY_ANNOUNCEMENT_VS_REALIZED_STATE_AND_MATCHED_EXPECTATIONS_SOURCE_INTAKE",
        "current_phase_definition_browsing": "NINE_FIXED_PRIOR_RECORDS_AND_ALL_24_ARCHIVED_POLICY_NODES_LOCAL_ONLY",
        "current_phase_required_directions": [],
        "current_phase_trial_accounting": {"scope": "TECH_R193_R194_LOCAL_POLICY_SOURCE_INTAKE_ONLY", "new_accounts": 0,
                                            "new_fits": 0, "new_labels": 0, "new_requests": 0, "source_nodes": 24,
                                            "unique_raw_files": 20, "catalog_leads": 490, "daily_origins": 3488,
                                            "case_rows": 240, "episodes": 61, "admitted_episodes": 49,
                                            "prior_records": 9, "new_numeric_candidates": 0},
        "current_goal_turn_actual_work": {"scope": "TECH_R193_R194_LOCAL_POLICY_SOURCE_INTAKE_ONLY", "source_nodes_checked": 24,
                                          "unique_raw_files": 20, "new_source_context_links": True, "closed_policy_score_proposal": True,
                                          "new_accounts": 0, "new_fits": 0, "new_labels": 0, "network_requests": 0},
        "previous_goal_turn_classification": before.get("current_goal_turn_classification"),
        "current_goal_turn_classification": "progress", "goal_turn_classification": "PROGRESS_ACTUAL_SOURCE_CONTEXT_LINK_AND_POLICY_INPUT_DISPOSITION",
        "goal_turn_progress_classification": "PROGRESS_NEW_SOURCE_CONTEXT_RESULT_NOT_FINANCIAL_IMPROVEMENT",
        "current_goal_turn_classification_reason": "实际跨原评分与完整来源连接确认不同含义、反例与未完整母集，关闭此前未登记政策评分提案；文档整理本身不作为进展。",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0, "blocked_reason": None, "blocked_key": None,
        "next_candidate_field_status": "NO_ADMITTED_UNRUN_NUMERIC_CANDIDATE_AFTER_POLICY_INTAKE",
        "next_macro_information_intake": {"status": "COMPLETED_CLOSED_NOT_ADMITTED", "decision": "TECH.R194",
                                           "result": core.rel(OUT / "summary.json"), "new_numeric_candidates": 0,
                                           "next_source_registered": False, "existing_macro_policy_failures_preserved": True},
        "next_research_question": next_text, "new_financial_result_computed_this_continuation": False,
        "return_and_sharpe_improved_this_continuation": False, "return_and_sharpe_changed_this_continuation": False,
        "code_files_changed_this_continuation": 0,
        "code_files_added_this_continuation": ["research/policy_announcement_score_intake_v1.py",
                                              "research/policy_announcement_score_intake_export_adapter_v1.py",
                                              "research/report_policy_announcement_score_intake_v1.py",
                                              "research/persist_policy_announcement_score_intake_v1.py"],
    })
    checks = pd.read_parquet(OUT / "results/全部24节点原件定位核对.parquet")
    identities = core.load(OUT / "protocol.json")["identities"]
    current["current_phase_source_freeze_count"] = len(set(identities) | set(checks.source_path))
    assert all(current.get(key) == value for key, value in protected.items())
    assert current["latest_actual_model_decision"] == current["latest_prediction_technical_decision"] == "TECH.R192"
    core.save(STATE, current, exclusive=False)
    description = core.load(OUT / "description_receipt.json")
    description.update(figure_visual_review="PASS_ACTUALLY_VIEWED_CHINESE_FONT_LAYOUT_AND_PLOTTED_VALUES", reviewed_at=stamp)
    core.save(OUT / "description_receipt.json", description, exclusive=False)
    core.save(OUT / "project_state_update_receipt.json", {
        "at": stamp, "status": "PASS_CURRENT_DOCUMENT_BODY_AND_FORWARD_CONTRACT_PRESERVED",
        "decision": "TECH.R194", "documents": changes, "protected_forward_fields": PROTECTED,
        "all_protected_fields_unchanged": True, "latest_actual_financial_decision": "TECH.R192",
        "new_accounts": 0, "new_fits": 0, "goal_status": "active", "goal_achieved": False,
    })
    print("TECH.R194实际来源裁决已写入四份长期事实来源与状态，原金融和13项前瞻值不变。")


if __name__ == "__main__":
    main()
