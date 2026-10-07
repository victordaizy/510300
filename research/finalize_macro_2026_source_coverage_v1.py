"""按实际新增原点结束来源核对，单点评估原冻结参数，写入长期事实。"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import macro_2026_source_coverage_clock_adapter_v1 as adapter
from research import macro_2026_source_coverage_intake_v1 as source
from research.finalize_macro_funding_source_contract_v2 import FORWARD

OUT, ROOT = adapter.OUT, source.ROOT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def run():
    source.study.require(not (OUT / "summary.json").exists(), "实际源结果已归档，不覆盖。")
    protocol = source.study.read(OUT / "single_recovered_origin_effect_protocol.json")
    code = source.study.read(OUT / "single_effect_code_registration.json")
    source.study.require(code["sha256"] == source.h(Path(__file__)), "登记的单点效应代码改变。")
    for item in protocol["sources"]:
        source.study.require(source.h(ROOT / item["path"]) == item["sha256"], "单点登记来源改变。")
    source.save_json(OUT / "single_effect_STARTED.json", {"at": source.study.now(), "new_fits": 0, "new_accounts": 0})
    coverage = pd.read_parquet(OUT / "results/全部3488源覆盖比较_联合未知保留.parquet")
    year = coverage.loc[coverage.date.dt.year.eq(2026)]
    extra = coverage.loc[coverage.additional_joint_origin]
    source.study.require(len(extra) == 1 and extra.date.iloc[0] == pd.Timestamp("2026-01-05"), "实际新增原点不再唯一。")
    source.study.require(not (coverage.old_joint_features_known & ~coverage.joint_features_known).any(), "旧完整原点丢失。")
    data = pd.read_parquet(source.BASE)
    funding = pd.read_parquet(OUT / "results/完整3488资金源派生_仅扩已有日历不拟合.parquet")
    derived = source.model.views(data, pd.read_parquet(source.availability.ORDERS), funding, pd.read_parquet(source.availability.MARGIN))
    models = source.study.read(source.PARENT / "saved_models.json")
    forecasts = pd.read_parquet(source.PARENT / "results/全部事前配对模型预测与实际宏观路径.parquet")
    origin = int(np.flatnonzero(derived.date.eq(extra.date.iloc[0]))[0])
    rows = []
    for policy in source.model.POLICIES:
        previous = forecasts.loc[forecasts.origin_index.eq(origin) & forecasts.policy.eq(policy)]
        source.study.require(len(previous) == 1, "原点旧模型预测身份不唯一。")
        fit_index = int(previous.fit_index.iloc[0])
        record = next(x for x in models if x["fit_index"] == fit_index)
        source.study.require(record["status"] == "FIT_COMPLETE", "原冻结模型无训练支持，不得补拟合。")
        model = record["models"][policy]
        vector = derived.iloc[origin][model["features"]].to_numpy(float)
        probability, leaf, used = source.model.predict(model, vector)
        quality = source.model.original.quality(model, probability)
        entry = bool(quality["predicted_p_times_b"] > 1 and quality["predicted_net_expectation"] > 0)
        rows.append({"date": extra.date.iloc[0], "policy": policy, "original_fit_index": fit_index,
                     "original_status": previous.status.iloc[0], "old_entry_event": bool(previous.entry_event.iloc[0]),
                     "source_available_without_new_fit": True, "leaf": leaf, "entry_event_under_original_gate": entry,
                     "macro_features_on_path": "|".join(x for x in dict.fromkeys(used) if x in source.model.MACRO), **quality})
    scores = pd.DataFrame(rows)
    source.OUT = OUT
    source.table("唯一新增完整原点_原冻结两模型评分不重拟合", scores)
    source.table("唯一新增完整原点_完整技术宏观向量", derived.loc[derived.date.eq(extra.date.iloc[0])])
    monthly = pd.read_parquet(OUT / "results/2026逐月源覆盖_不计算收益.parquet")
    raw_coverage = pd.read_parquet(OUT / "results/已定位全部融资原件覆盖_不以ETF替代两市.parquet")
    notices = pd.read_parquet(OUT / "results/2026全部政策操作原件及钟核对.parquet")
    available = year.loc[year.funding_known]
    same_before_2026 = coverage.loc[coverage.date.lt("2026-01-01")]
    source.study.require(same_before_2026.old_joint_features_known.equals(same_before_2026.joint_features_known), "旧历史支持变化。")
    source.study.require(year.margin_known.sum() == 1 and year.joint_features_known.sum() == 1, "2026源支持不是已核的一条。")
    no_new_entry = not scores.entry_event_under_original_gate.any()
    summary = {"at": source.study.now(), "study": "510300_2026_MACRO_SOURCE_COVERAGE_INTAKE_V1",
        "registration_decision": "TECH.R199", "decision": "TECH.R200",
        "status": "COMPLETED_FUNDING_COVERAGE_RECOVERED_ONE_ROLLOVER_ORIGIN_REMAINING_MARKET_MARGIN_MISSING",
        "previous_goal_turn": "PROGRESS_SOURCE_SEMANTICS_AND_EIGHT_ACCOUNT_EXPERIMENT_R198",
        "current_goal_turn": "PROGRESS_2026_RAW_COVERAGE_AND_SINGLE_RECOVERED_ORIGIN_EFFECT",
        "original_funding_prefix_rows_exact_after_lossless_clock_normalization": 3307,
        "all_calendar_slots": len(coverage), "slots_2026": len(year), "dr_raw_2026_rows_exactly_reproduced": 153,
        "dr_source_last": "2026-08-14", "funding_known_2026_under_inherited_contract": int(year.funding_known.sum()),
        "funding_change5_known_2026": int(year.funding_gap_change5.notna().sum()),
        "funding_known_2026_first": available.date.min().isoformat(), "funding_known_2026_last": available.date.max().isoformat(),
        "policy_notices_2026_original_fields_exactly_reparsed": len(notices),
        "policy_rate_levels_2026": sorted(notices.rate.unique().tolist()), "last_policy_notice": notices.notice_date.max().isoformat(),
        "all_located_market_or_exchange_tables": int(raw_coverage.scope.ne("ETF_ONLY_510300").sum()),
        "market_summary_source_rows_2026": int(raw_coverage.loc[raw_coverage.scope.ne("ETF_ONLY_510300"), "rows_2026"].sum()),
        "etf_only_source_tables": 2, "etf_only_source_rows_2026": int(raw_coverage.loc[raw_coverage.scope.eq("ETF_ONLY_510300"), "rows_2026"].sum()),
        "etf_margin_substitution": False, "market_margin_known_decision_slots_2026": 1,
        "rollover_margin_stat_date": extra.margin_stat_date.iloc[0].isoformat(), "joint_known_2026": 1,
        "joint_origins_before": int(coverage.old_joint_features_known.sum()), "joint_origins_after": int(coverage.joint_features_known.sum()),
        "additional_joint_origins": 1, "additional_origin": "2026-01-05", "old_joint_mask_exactly_unchanged": False,
        "single_source_exception_observed_and_preserved": True, "original_source_comparison_failure_preserved": True,
        "source_cardinality_guard_failure_preserved": True, "new_origin_fixed_saved_model_evaluations": 2,
        "new_origin_admitted_entry_events": int(scores.entry_event_under_original_gate.sum()),
        "single_origin_no_entry_both_models": bool(no_new_entry),
        "financial_admission": "NOT_ADMITTED_NO_NEW_ENTRY_AND_180_OF_181_MARKET_MARGIN_SLOTS_UNKNOWN" if no_new_entry else
                               "NOT_REGISTERED_REQUIRES_SEPARATE_COMPLETE_PURPOSE_NOT_ONE_ORIGIN_PROMOTION",
        "new_fits": 0, "new_accounts": 0, "new_training_labels": 0, "new_return_label_reads": 0, "new_network_requests": 0,
        "new_net_sharpe": "NOT_COMPUTED", "new_net_cagr": "NOT_COMPUTED", "latest_actual_financial_decision": "TECH.R198",
        "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False,
        "goal_achieved": False, "next_source_need": "2026-01-05至2026-09-30沪深两市每日汇总融资余额和融资买入原件及公布钟，先有限官方源可用性申请。"}
    source.save_json(OUT / "summary.json", summary)
    text = "# 2026宏观源覆盖：资金恢复155槽，联合仅恢复一个跨年原点\n\n"
    text += "TECH.R199—TECH.R200完成本地源用途和实际例外裁决。前轮R198实际金融失败保持；本轮0新拟合、账户、股票标签、网络请求，不宣称提高收益或夏普。\n\n"
    text += "原3307资金缓存的统计日、公布钟、政策钟、实施日、DR007、利率、价差和源龄精确保持。初次比较因微秒/纳秒表示差异失败，原记录保留；隔离适配逐列精确往返到原表示后再按纳秒精确比较，未改变任一经济时点或数值。\n\n"
    text += f"原2026 DR响应153行逐值复现；原181个ETF观察槽中，按原资金合同恢复{len(available)}个源及五日变化槽，最后支持日2026-08-24。2026央行7天操作公告{len(notices)}份，逐份重解析原HTML的日期、公布钟、公告号和7天行利率，水平均1.4%，最后公告8月10日。公告缺日不补，本次计算可用性不是当年首版认证，不能把重复水平当作147次新冲击。\n\n"
    text += monthly.to_markdown(index=False) + "\n\n"
    text += "资料不是全部缺2026：资金源可恢复；但12份已定位交易所/两市融资汇总资料均无2026原行。两份510300自身融资合计150条2026记录，经济对象不同，不能替代两市汇总。\n\n"
    text += "例外必须保留：2026年1月5日使用2025年12月31日的上一ETF交易日融资、1月4日DR007和已公布12月订单，恢复一个联合完整原点；随后180槽两市融资未知。原2608联合原点增加到2609，不是完全不变。原预设不变断言已终止，该失败保持，新增点没有删除。\n\n"
    text += scores.to_markdown(index=False, floatfmt=".6f") + "\n\n"
    text += ("唯一恢复原点在原冻结两模型参数和质量门下均不入场。没有重新拟合，也没有新增账户收益；一个源完整原点不足以说明全年覆盖或独立优势。\n\n" if no_new_entry else
             "唯一恢复原点存在通过估计门的模型，尚未登记完整金融用途，不以一个预测晋升；本次未执行账户。\n\n")
    text += "下一具体资料需求是2026-01-05至2026-09-30的沪深两交易所每日汇总融资余额、融资买入额和公布时钟。先登记有限官方源可用性申请，能取得并对齐才考虑完整信息接入；不得以自身ETF融资替代、改变量意义、缩样本或调树/阈值/退出救援旧失败。\n\n"
    text += "[全部181观察槽](results/2026全部观察槽_订单资金两市融资支持分开.csv)、[原参数单点评分](results/唯一新增完整原点_原冻结两模型评分不重拟合.csv)、[原点完整向量](results/唯一新增完整原点_完整技术宏观向量.csv)、[12汇总与2自身融资资料](results/已定位全部融资原件覆盖_不以ETF替代两市.csv)、[全部操作原件钟](results/2026全部政策操作原件及钟核对.csv)、[实际结果](summary.json)。\n"
    report = OUT / "2026资料覆盖与唯一原点的实际结论.md"
    report.write_bytes(text.encode("utf-8"))
    before = STATE.read_bytes()
    state = json.loads(before.decode("utf-8-sig"))
    (OUT / "state_before_TECH_R200.json").write_bytes(before)
    frozen = {k: copy.deepcopy(state[k]) for k in FORWARD}
    state["previous_goal_turn_classification"] = state.get("current_goal_turn_classification")
    state["previous_actual_phase_trials_before_TECH_R200"] = copy.deepcopy(state.get("actual_candidate_trials"))
    progress = ("TECH.R200实际核2026源：DR原153行逐值复现、147政策原件的7天行/日期/钟精确重解析，"
                "3307原缓存源值和钟精确保持。181槽资金及五日变化恢复155、末2026-08-24；"
                "12汇总融资表无2026原行，两自身ETF表150行不替代。2026-01-05仍可用2025-12-31融资，"
                "联合原点2608→2609，唯一例外完整保存；原两模型参数评分2次，无新拟合或账户。"
                "初次精度比较失败和后续不变断言失败均保留，不删除新增点。"
                "收益夏普新值NOT_COMPUTED，最近实际金融仍TECH.R198拒绝；独立与去过拟合未建立。")
    progress += "唯一原点两模型均未过原门。" if no_new_entry else "单点估计门通过不晋升，金融用途尚未登记。"
    next_source = summary["next_source_need"]
    state.update(updated_at=source.study.now(), latest_technical_decision="TECH.R200", latest_information_intake_technical_decision="TECH.R200",
        latest_prior_review_technical_decision="TECH.R200", latest_registration_decision="TECH.R199",
        latest_completed_study=source.rel(OUT), latest_result=source.rel(OUT / "summary.json"), latest_report=source.rel(report),
        latest_research_status=summary["status"], latest_progress=progress, latest_continuation_outcome=progress,
        current_study=summary["study"], current_phase="2026_MACRO_SOURCE_COVERAGE_COMPLETE_NO_NEW_FINANCIAL_PURPOSE",
        current_phase_information_scope="DR007_POLICY_ORIGINALS_AND_MARKET_VS_ETF_MARGIN_COVERAGE",
        current_phase_definition_browsing="2026_ALL_181_CALENDAR_SLOTS_147_POLICY_ARTICLES_12_SUMMARY_AND_TWO_ETF_TABLES_LOCAL_ONLY",
        current_goal_turn_classification="progress", goal_turn_progress_classification="PROGRESS_RAW_SOURCE_AND_SINGLE_ORIGIN_EFFECT",
        current_goal_turn_classification_reason="真实2026原件和覆盖、跨年例外与原冻结参数两点评估改变下一资料动作。",
        consecutive_blocked_goal_turns=0, blocked_audit_count=0, current_admitted_unrun_numeric_candidates=0,
        next_information_source_proposal=next_source, next_research_question=next_source, current_priority=next_source,
        current_unmet_evidence="R198完整金融失败保持；180/181个2026观察槽缺两市融资，独立和去过拟合未建立。",
        next_candidate_field_status="NO_ADMITTED_FINANCIAL_CANDIDATE_AFTER_2026_SOURCE_INTAKE",
        next_financial_experiment="NOT_DEFINED_OR_REGISTERED_REQUIRES_FULL_SOURCE_AND_PURPOSE",
        latest_2026_macro_source_coverage=source.rel(OUT / "summary.json"),
        latest_2026_macro_single_origin_effect=source.rel(OUT / "results/唯一新增完整原点_原冻结两模型评分不重拟合.csv"),
        latest_continuation_receipt=source.rel(OUT / "project_state_update_receipt.json"),
        latest_progress_check=source.rel(OUT / "summary.json"), current_phase_source_freeze_count=len(source.study.read(OUT / "protocol.json")["sources"]),
        new_accounts_in_current_phase=0, new_financial_result_computed_this_continuation=False,
        return_and_sharpe_improved_this_continuation=False, return_and_sharpe_changed_this_continuation=False,
        current_information_mechanisms_admitted_this_continuation=0, current_fields_admitted_this_continuation=0,
        current_phase_financial_candidate_configurations=0, current_phase_required_tests=0,
        current_member_support_this_continuation="原3488源日历；2026年181槽/155资金/1跨年融资及联合原点，180两市融资未知。",
        new_accounts_this_continuation=0, new_primary_accounts_this_continuation=0,
        new_financial_candidate_accounts_this_continuation=0, new_strategy_accounts_this_continuation=0,
        new_matched_control_accounts_this_continuation=0, saved_account_controls_replayed_this_continuation=0,
        internal_reference_replays_this_continuation=0, saved_accounts_checked_this_continuation=0,
        new_strategy_configurations_this_continuation=0, new_model_fits_this_continuation=0,
        new_training_labels_this_continuation=0, new_return_labels_this_continuation=0, new_market_requests_this_continuation=0,
        necessary_tests_passed_this_continuation=0, necessary_test_executions_this_continuation=0,
        actual_prefix_checks_this_continuation=1, actual_prefix_check_scope="3307行源值与全部时钟，不是模型完整未来信息检验。",
        fixed_saved_model_evaluations_this_continuation=2,
        validation_method_this_continuation="原DR153逐值、政策147原7天行和钟、3307源前缀精确往返比较、181完整槽；原冻结两模型单点2评估，0金融重跑。",
        current_study_prediction_gate_status="SOURCE_COVERAGE_AND_FIXED_SAVED_SINGLE_ORIGIN_ONLY_NOT_NEW_MODEL",
        latest_prediction_economic_stage_status="NO_NEW_FINANCIAL_RUN_OR_METRIC_AFTER_R198",
        current_numeric_leads_role="ORIGINAL_FORWARD_REGISTRY_ONLY_NOT_ONE_RECOVERED_ORIGIN_CANDIDATE",
        original_strategy_source_files_changed_this_continuation=0, code_files_changed_this_continuation=0,
        code_files_added_this_continuation=["research/macro_2026_source_coverage_intake_v1.py",
             "research/macro_2026_source_coverage_clock_adapter_v1.py", "research/finalize_macro_2026_source_coverage_v1.py"],
        goal_achieved=False, whole_model_overfitting_removed=False)
    source.study.require(all(state[k] == v for k, v in frozen.items()), "原十三项前瞻改变。")
    STATE.write_bytes((json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))
    facts = ("> 最新来源事实 TECH.R200（2026-10-05，优先于本线下方旧快照；最近实际金融与模型仍TECH.R198）：" + progress +
             "本轮实际来源/例外进展，连续受阻0；目标保持active且未达，不把文件更新算进展。\n\n" +
             "下一步：" + next_source + "未登记下一来源申请或金融用途，原13项前瞻状态及原下一前瞻实验不变。\n\n" +
             "依据：[2026全部资料、单点与实际结论](../"+source.rel(report)+")；[完整源结果](../"+source.rel(OUT/"summary.json")+")。\n\n")
    decision = """
## TECH.R199—TECH.R200：2026资金、政策和两市融资实际源覆盖

- 假设：2026资金未知可能只是旧派生缓存截止，两市融资必须单独核，不以510300自身融资替代。
- 验证方法：先冻结全部本地原件和观察日历。2026 DR响应153行逐值复现、147央行原件重解析7天行与钟；原3307资金源和钟精确保持，逐日查看181槽以及12汇总表和2自身融资表。0拟合、账户、股票标签、网络。
- 结果：资金及五日变化155槽可算，最后2026-08-24；操作原件147份水平1.4%、末8月10日。汇总表无2026原行，自身融资150行不替代。跨年2026-01-05仍用前交易日2025-12-31融资，恢复1完整原点，2608→2609；180/181缺两市融资。初次时钟精度比较失败和后续不变断言失败均保存，数据例外不删除。补充用途只评原冻结两模型的唯一完整向量，不读结果标签或新拟合，评分在实际表中。
- 为什么接受/拒绝：接受源存在、155资金槽、一个跨年完整点的事实；拒绝把一个点或自身ETF融资当全年联合支持。两市融资母体和公布钟缺口未解除，当前无新金融候选，收益夏普NOT_COMPUTED；R198失败保持。
- 是否重验：本源核对结束，原件/钟或真实错误改变才重验。下一只限定2026-01-05至09-30沪深每日汇总融资余额/买入原件和公布钟的有限官方可获得性申请，再按完整用途准入；不营救旧树/阈值/样本/退出。不认证历史首版、独立性或去过拟合。
"""
    backups = OUT / "documents_before_TECH_R200"
    backups.mkdir()
    documents = []
    for name in ["PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"]:
        p = ROOT / "docs" / name
        b = p.read_bytes()
        (backups / name).write_bytes(b)
        cut = b.index(b"\n") + 1
        newline = "\r\n" if b[:cut].endswith(b"\r\n") else "\n"
        insertion = ("\n"+facts).replace("\n", newline).encode("utf-8")
        tail = decision.replace("\n", newline).encode("utf-8") if name.startswith("RESEARCH_DECISIONS") else b""
        updated = b[:cut] + insertion + b[cut:] + tail
        p.write_bytes(updated)
        source.study.require(updated[cut+len(insertion):cut+len(insertion)+len(b[cut:])] == b[cut:], "原分支正文未完整保持。")
        documents.append({"path": source.rel(p), "original_body_bytes_preserved": True, "sha256": source.h(p)})
    source.save_json(OUT / "project_state_update_receipt.json", {"at": source.study.now(),
        "status": "PASS_SOURCE_EXCEPTION_AND_FOUR_DURABLE_FILES_FINANCIAL_R198_PRESERVED",
        "documents": documents, "state_sha256": source.h(STATE), "previous_turn": "progress", "current_turn": "progress",
        "consecutive_blocked_goal_turns": 0, "goal_status": "active", "latest_source_decision": "TECH.R200",
        "latest_actual_financial_decision": state["latest_actual_financial_decision"], "original_forward_thirteen_unchanged": True,
        "new_fits_or_accounts_in_finalization": 0, "goal_achieved": False})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
