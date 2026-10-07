"""把实际源诊断与唯一金融结果写入长期事实文件，保留其他分支原文。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

from research import funding_availability_semantics_v1 as diagnosis
from research import macro_funding_source_contract_v2 as version
from research import macro_technical_first_passage_study_v1 as study

ROOT, OUT = version.ROOT, version.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FORWARD = ["forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
           "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
           "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
           "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation", "next_experiment"]


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def run():
    receipt = OUT / "project_state_update_receipt.json"
    study.require(not receipt.exists(), "本次长期事实已归档，不覆盖。")
    summary = study.read(OUT / "summary.json")
    description = study.read(OUT / "description_receipt.json")
    goal = study.read(OUT / "goal_service_status.json")
    study.require(goal["goal"]["status"] == "active", "实际目标服务未恢复，不能自行写恢复。")
    before = STATE.read_bytes()
    state = json.loads(before.decode("utf-8-sig"))
    (OUT / "state_before_TECH_R198_resumed.json").write_bytes(before)
    frozen_forward = {k: copy.deepcopy(state[k]) for k in FORWARD}
    backup = OUT / "documents_before_TECH_R198"
    backup.mkdir()
    report = OUT / "多信息源评分_具体上涨与完整结论.md"
    progress = ("TECH.R195—R196区分资金源可用与旧分位统计支持，128时点源已知却旧门不足；"
                "联合完整原点2534→2608，74新增为2015年70/2020年4，142训练合同77变化。"
                "R197登记唯一源合同修正，R198一次完整金融：125配对月/250实际拟合、2345可评分原点、"
                "2007不同预测且用宏观路径、4原A精确复现、4联合及4配对新账户与8保存复算。"
                "较早压力净年化0.4132%/夏普0.7904、2全赢但B不可估；近期−0.79645%/−0.62898，"
                "17完成4胜13负、B1.09085/pB0.25667/标准期望−0.50803、DD7.5665%。"
                "较旧R192增加5笔全部亏损，完整年次数2→2.833，期末低4187.99932元；"
                "四经济门0通过、稳定门失败，拒绝唯一固定用途。五源合同测试通过，四原图已查看，"
                "原61/49和四案例240行保持，0新股票标签/采集/原策略改动。"
                "本轮有实际来源与金融进展，服务实际active，旧三轮blocked计数归档，本恢复周期连续受阻0。"
                "独立/去过拟合和完整目标未达，当前无待跑金融；原13前瞻值不变。")
    next_source = ("只核对2026本地原始DR007、实际政策利率和融资源的覆盖/公布钟，"
                   "区分未纳入旧派生缓存与真实缺口；原DR源已至2026-08-14。"
                   "0金融拟合，待可验证的不同完整用途或真正新样本成立才另登记。"
                   "不按R198亏损调门槛、权重、树、窗口或退出；原E03前瞻计划不变。")
    metrics = pd.read_parquet(OUT / "results/十二账户与统一完整年频率.parquet")
    forecasts = pd.read_parquet(OUT / "results/全部事前配对模型预测与实际宏观路径.parquet")
    macro_entries = forecasts.loc[forecasts.policy.eq("MACRO_TECH_TREE") & forecasts.entry_event]
    trials = {"scope": "TECH_R197_R198_SOURCE_CONTRACT_ONLY_FIXED_MACRO_TECH_COMPARISON",
              "candidate_configurations": 1, "new_candidate_accounts": 4, "new_matched_control_accounts": 4,
              "new_accounts": 8, "original_A_successful_exact_replays": 4, "original_A_format_failed_attempts": 0,
              "monthly_fit_records": 142, "paired_months_fitted": summary["paired_months_fitted"],
              "actual_fit_calls": summary["actual_fit_calls"], "all_known_daily_rows": 3488,
              "macro_complete_origins": 2608, "common_available_forecasts_per_policy": 2345,
              "actual_macro_path_origins": 2007, "new_necessary_tests_passed": 5,
              "parent_eight_tests_previously_passed_not_rerun": True, "saved_account_checks": 8,
              "new_labels": 0, "new_market_requests": 0, "single_scenario_economic_passes": 0,
              "all_four_economic_gates_passed": False, "historical_stability_gate_passed": False,
              "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False}
    state["previous_actual_candidate_trials_before_TECH_R198"] = copy.deepcopy(state.get("actual_candidate_trials"))
    state["historical_blocked_cycle_before_TECH_R195"] = {k: state.get(k) for k in
        ["goal_status", "status", "consecutive_blocked_goal_turns", "blocked_reason", "blocked_key",
         "latest_progress_check", "goal_tool_blocked_receipt", "goal_blocked_at"]}
    state.update(updated_at=study.now(), status="research_active", goal_status="active", goal_achieved=False,
        latest_completed_study=relative(OUT), latest_result=relative(OUT / "summary.json"), latest_report=relative(report),
        latest_research_status=summary["status"], latest_progress=progress, latest_continuation_outcome=progress,
        latest_overall_summary=progress, next_research_question=next_source, current_priority=next_source,
        current_unmet_evidence="R198四经济门/稳定门失败；独立验证NOT_ESTABLISHED，去过拟合未建立，收益夏普目标未达。",
        current_study=summary["study"], current_phase="SOURCE_CONTRACT_FIXED_MACRO_FINANCIAL_COMPLETE_REJECTED",
        current_direction="先解释原具体上涨和全部反例，分别判断源可用、特征可算、训练支持和入场门，再核完整账户。",
        current_goal_turn_classification="progress", goal_turn_progress_classification="PROGRESS_SOURCE_SEMANTICS_AND_FIXED_FINANCIAL_RESULT",
        current_goal_turn_classification_reason="实际128源可用分歧、77训练合同变化及8新账户，不以文件更新充作进展。",
        consecutive_blocked_goal_turns=0, blocked_audit_count=0, blocked_reason=None, blocked_key=None,
        blocked_audit_key=None, blocked_scope=None, goal_blocked_at=None,
        blocking_decision="PREVIOUS_BLOCKED_CYCLE_ARCHIVED_ACTUAL_SERVICE_RESUMED_AND_MEANINGFUL_PROGRESS",
        blocked_transition_status="HISTORICAL_BLOCKED_SUPERSEDED_BY_ACTUAL_RESUME_AND_SOURCE_DIAGNOSIS",
        goal_tool_status_confirmed="active", latest_goal_service_status="active", latest_goal_service_status_observed_at=study.now(),
        latest_goal_tool_status_receipt=relative(OUT / "goal_service_status.json"),
        latest_continuation_receipt=relative(receipt), latest_progress_check=relative(OUT / "description_receipt.json"),
        latest_technical_decision="TECH.R198", latest_registration_decision="TECH.R197",
        latest_financial_registration_decision="TECH.R197", latest_actual_financial_decision="TECH.R198",
        latest_financial_strategy_decision="TECH.R198", latest_financial_strategy_result=relative(OUT / "summary.json"),
        latest_prediction_technical_decision="TECH.R198", latest_actual_prediction_model_decision="TECH.R198",
        latest_actual_model_kind="PAIRED_TECH_AND_MACRO_THREE_CLASS_FIRST_PASSAGE_TREES_SOURCE_CONTRACT_V2_TECH_R198",
        latest_macro_joint_technical_decision="TECH.R198", latest_macro_joint_registration="TECH.R197",
        latest_information_intake_technical_decision="TECH.R196", latest_prior_review_technical_decision="TECH.R196",
        latest_funding_source_semantics=relative(diagnosis.OUT / "summary.json"),
        latest_funding_source_semantics_erratum=relative(OUT / "source_diagnostic_status_erratum.json"),
        latest_concrete_case_explanation=relative(OUT / "explanation_summary.json"),
        latest_joint_macro_case_description=relative(OUT / "description_receipt.json"),
        new_account_return_sharpe_result=relative(OUT / "summary.json"), actual_candidate_trials=trials,
        new_accounts_in_current_phase=8, return_and_sharpe_improved_this_continuation=False,
        return_and_sharpe_changed_this_continuation=True, new_financial_result_computed_this_continuation=True,
        current_information_mechanisms_admitted_this_continuation=0, current_fields_admitted_this_continuation=0,
        current_member_support_this_continuation="3488完整日历，2608联合完整，125配对拟合月/250次拟合、2345可评分日；非独立证据。",
        current_phase_source_freeze_count=len(study.read(OUT / "protocol.json")["sources"]),
        current_phase_known_daily_rows=3488, current_phase_original_state_rows=3488,
        current_phase_original_case_rows=240, current_phase_original_episode_rows=61, current_phase_original_admitted_waves=49,
        current_phase_information_scope="SAME_FUNDING_SOURCE_VALUES_CLOCK_AVAILABILITY_DISTINCT_FROM_QUANTILE_SUPPORT",
        current_phase_definition_browsing="ALL_3307_CACHE_ROWS_3488_ORIGINS_142_MONTHS_LOCAL_ONLY",
        necessary_tests_passed_this_continuation=5, necessary_test_executions_this_continuation=5,
        actual_prefix_checks_this_continuation=0, new_accounts_this_continuation=8, new_primary_accounts_this_continuation=4,
        new_financial_candidate_accounts_this_continuation=4, new_strategy_accounts_this_continuation=4,
        new_matched_control_accounts_this_continuation=4, saved_account_controls_replayed_this_continuation=4,
        internal_reference_replays_this_continuation=4, saved_accounts_checked_this_continuation=8,
        new_strategy_configurations_this_continuation=1, new_model_fits_this_continuation=250,
        new_training_labels_this_continuation=0, new_return_labels_this_continuation=0,
        new_market_requests_this_continuation=0, original_strategy_source_files_changed_this_continuation=0,
        code_files_changed_this_continuation=0, code_files_changed_role="原已有源码修改0，新增隔离源码4与测试1另列。",
        current_phase_required_tests=5, current_phase_financial_candidate_configurations=1,
        validation_method_this_continuation="五新源钟边界测试实际通过，四原A精确复现，八新账户一次及八保存复算；原八模型测试未重复。",
        current_study_prediction_gate_status="FULL_FIXED_ECONOMIC_AND_HISTORICAL_STABILITY_FAILED_NOT_INDEPENDENT",
        latest_prediction_economic_stage_status="EIGHT_ACTUAL_ACCOUNTS_COMPLETED_FIXED_CANDIDATE_REJECTED",
        current_full_account_economic_gate_passed=False, current_historical_stability_gate_passed=False,
        current_all_four_actual_point_quality_passed=False, current_individual_economic_gate_passes=0,
        current_point_quality_evidence_role="EARLY_TWO_ALL_WIN_B_NOT_ESTIMABLE_RECENT_SEVENTEEN_PB_0_25667_NEGATIVE_EXPECTANCY",
        current_trade_map_scope="ONE_COST_ALL_MACRO_TECH_MODEL_CYCLES_R198", current_trade_map_cycles=19,
        current_trade_map_complete_cycles=19, current_trade_map_open_cycles=0,
        current_new_policy_complete_cycles=19, current_new_policy_open_cycles=0,
        current_executed_unique_new_entry_events=19, current_structural_signal_events=None,
        current_model_daily_qualified_origins=len(macro_entries), current_new_policy_unexecuted_origins=len(macro_entries)-19,
        current_admitted_unrun_numeric_candidates=0, next_candidate_field_status="NO_ADMITTED_UNRUN_CANDIDATE_AFTER_FIXED_V2_REJECTION",
        next_financial_experiment="NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE", next_information_source_proposal=next_source,
        whole_model_overfitting_removed=False, background_monitor_running=False,
        current_numeric_leads_role="EXISTING_FORWARD_REGISTRY_ONLY_NOT_R198_CANDIDATES",
        latest_long_point_metrics={"scope": "R198_ALL_MACRO_TECH_COMPLETED_CYCLES_DEVELOPMENT_NOT_CURRENT_MARKET",
            "STRESS": json.loads(metrics.loc[metrics.cost.eq("STRESS") & metrics.policy.eq("MACRO_TECH_TREE")].to_json(orient="records"))},
        code_files_added_this_continuation=["research/funding_availability_semantics_v1.py", "research/macro_funding_source_contract_v2.py",
            "research/describe_macro_funding_source_contract_v2.py", "research/finalize_macro_funding_source_contract_v2.py",
            "tests/test_funding_source_contract_v2.py"])
    study.require(all(state[k] == v for k, v in frozen_forward.items()), "原前瞻计划或十三状态发生改变。")
    STATE.write_bytes((json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    facts = ("> 最新技术线事实 TECH.R195—R198（2026-10-05，优先于下方本线旧快照）：" + progress + "\n\n" +
             "目标服务已实际恢复active；下方三轮blocked属于此前已结束恢复周期，不能带入本轮计数。" +
             "本轮有源语义发现及唯一完整账户结果，连续受阻0，完整目标仍未完成。其他宏观/分钟分支原文保持。\n\n" +
             "下一步：" + next_source + "\n\n" +
             "依据：[具体上涨、全部周期与完整结论](../" + relative(report) + ")；" +
             "[实际金融结果](../" + relative(OUT / "summary.json") + ")；" +
             "[原源诊断及77合同变化](../" + relative(diagnosis.OUT / "summary.json") + ")；" +
             "[源诊断状态文字更正](../" + relative(OUT / "source_diagnostic_status_erratum.json") + ")。\n\n")
    decisions = """
## TECH.R195—R196：资金源可用与旧分位统计支持分开

- 假设：旧fund_known混合源字段可用与K05分位支持，可能错误缩小联合信息和训练支持。
- 验证方法：先登记R195；同3307缓存保留所有源值与钟，逐行核源龄/公布钟/统计日/有限值；精确重现旧3488输入和142原训练成员，对比完整成员、成熟标签、唯一性权重和两模型全部特征，0拟合/账户。
- 结果：128源已知而旧分位不足；2534→2608完整原点，新增74为2015年70/2020年4；77训练合同变化，最后2023-03。近期训练和输入均非精确相同，不能直接沿用旧近期成绩。
- 为什么接受/拒绝：接受输入语义修正的事实；原R192正确执行冻结门，未判实现违反。原R196status文字硬编码无变化与数字冲突，另存更正并保留原件/哈希。源可用不等于特征可算、训练足够或可入场。
- 是否重验：本源诊断关闭；仅新字段/源钟实现改变或实际错误才重验，不以修正名义搜索参数。金融另登记R197。

## TECH.R197—R198：只修正源可用合同的唯一宏观联合用途

- 假设：分开旧分位支持后，固定八技术/六宏观评分能提高同口径完整账户收益与夏普及实际pB。
- 验证方法：R197先冻结；五新源钟测试通过，只替换fund_known派生副本，其他源值、统计日政策配对、缓存截止、142月训练与树参数、标签、估计入场门、退出、20万元风险/费用/T+1及两时期两费用全部继承；4原A精确控制、4联合和4配对账户一次，全部20/252日各2000配对区间，8保存复算。
- 结果：125配对月/250拟合，每模型2345可评分日、2007预测不同且用宏观路径。较早压力净年化0.4132%/夏普0.7904、2全赢而B不可估；近期−0.79645%/−0.62898、17完成4胜13负、B1.09085/pB0.25667/标准期望−0.50803、DD7.5665%。较旧版增加5笔全部亏损，年均2→2.833，期末低4187.99932元；4经济门0、稳定门失败。原A/配对技术账户数值不变，原61/49及四案例240行保留。
- 为什么接受/拒绝：宏观数据可进入并影响路径的事实接受；次数增加却实际质量、收益和夏普更差，拒绝并关闭本完整配置，旧R192失败同时保留。不能用较早夏普或全赢两笔替代完整目标，不断言所有宏观或多源机制无效。
- 是否重验：禁止改权重、叶大小、深度、窗口、阈值、符号、样本或退出营救。本次历史仍DEVELOPMENT_CALIBRATION/首版NOT_CERTIFIED/独立NOT_ESTABLISHED/global DSR-PBO NOT_COMPUTED。只在实质不同可知信息和完整用途、真正新样本或真实源实现错误成立后另登记；下一项仅2026原统计源覆盖/公布钟核对，未准入新金融。
"""
    records = []
    for name in ["PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"]:
        p = ROOT / "docs" / name
        original = p.read_bytes()
        (backup / name).write_bytes(original)
        first_newline = original.index(b"\n") + 1
        header, body = original[:first_newline], original[first_newline:]
        newline = "\r\n" if header.endswith(b"\r\n") else "\n"
        insertion = ("\n" + facts).replace("\n", newline).encode("utf-8")
        tail = decisions.replace("\n", newline).encode("utf-8") if name.startswith("RESEARCH_DECISIONS") else b""
        updated = header + insertion + body + tail
        p.write_bytes(updated)
        study.require(p.read_bytes() == updated and updated[first_newline+len(insertion):first_newline+len(insertion)+len(body)] == body,
                      "长期文件原分支正文未逐字保留。")
        records.append({"path": relative(p), "before_sha256": hashlib.sha256(original).hexdigest(),
                        "after_sha256": study.digest(p), "prior_entire_body_bytes_preserved": True})
    final_state = study.read(STATE)
    study.require(all(final_state[k] == v for k, v in frozen_forward.items()), "保存后前瞻十三项不同。")
    study.write_json(receipt, {"at": study.now(), "status": "PASS_ACTIVE_STATE_AND_FOUR_DURABLE_FACT_FILES",
        "state_sha256": study.digest(STATE), "documents": records, "goal_service_status": "active",
        "consecutive_blocked_goal_turns": 0, "previous_blocked_three_turn_cycle_archived": True,
        "meaningful_progress": "ACTUAL_SOURCE_SEMANTICS_AND_FIXED_FINANCIAL_EXPERIMENT_NOT_DOCUMENT_WRITES",
        "latest_technical_decision": "TECH.R198", "latest_actual_financial_decision": "TECH.R198",
        "original_forward_thirteen_values_exactly_unchanged": True, "original_forward_values": frozen_forward,
        "current_admitted_unrun_financial_candidates": 0, "new_fits_or_accounts_in_persistence": 0,
        "goal_achieved": False}, exclusive=True)
    print("R195—R198写入四长期文件；本轮实际进展，目标active/受阻0，原十三前瞻值保持。", flush=True)


if __name__ == "__main__":
    run()
