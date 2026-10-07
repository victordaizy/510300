"""只解释已保存结果并更新事实文档，不重新训练、运行账户或采集。"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research.cnh_macro_first_passage_study_v1 import OUT, model, read, write_json, digest, now, require

STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
DOCS = [ROOT / "docs" / name for name in ["PROJECT_STATE.md", "RESEARCH_DECISIONS.md",
        "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"]]
FORWARD = ["forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
    "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at", "current_validated_candidates",
    "forward_account_comparison_protocol", "latest_forward_account_check", "new_prospective_sessions_this_continuation",
    "new_prospective_cycles_this_continuation", "next_experiment"]


def main():
    require(not (OUT / "project_state_update_receipt.json").exists(), "本结果事实文件已更新，不重复。")
    summary = read(OUT / "summary.json")
    protocol = read(OUT / "protocol.json")
    delivery = read(OUT / "delivery_receipt.json")
    require(delivery["saved_account_checks"] == 12, "保存账户交付尚未完成。")
    stats = pd.read_parquet(OUT / "results/十六完整账户共同口径比较.parquet")
    forecasts = pd.read_parquet(OUT / "results/全部三模型事前预测与实际技术宏观外汇路径.parquet")
    candidate = forecasts.loc[forecasts.policy.eq(model.POLICIES[2])].reset_index(drop=True)
    control = forecasts.loc[forecasts.policy.eq(model.POLICIES[1])].reset_index(drop=True)
    known = candidate.status.eq("AVAILABLE")
    yearly = pd.DataFrame({"year": candidate.date.dt.year, "known": known,
        "fx_path": known & candidate.fx_features_on_path.ne(""),
        "entry_different": candidate.entry_event.ne(control.entry_event),
        "candidate_entries": candidate.entry_event, "macro_entries": control.entry_event}).groupby("year").sum().reset_index()
    yearly.to_parquet(OUT / "results/逐年真实外汇路径及同池进场差异.parquet", index=False)
    yearly.to_csv(OUT / "results/逐年真实外汇路径及同池进场差异.csv", index=False, encoding="utf-8-sig")
    economic_checks = []
    for period in ["2015_2019", "2020_2026"]:
        for cost in ["BASE", "STRESS"]:
            path = OUT / f"results/accounts/{period}/{cost}"
            a = pd.read_parquet(path / model.POLICIES[1] / "daily.parquet")
            b = pd.read_parquet(path / model.POLICIES[2] / "daily.parquet")
            fields = ["date", "equity", "cash", "shares", "net_return", "price_pnl", "commission", "slippage"]
            economic_checks.append({"period": period, "cost": cost, "candidate_and_macro_daily_economic_values_exactly_same": bool(a[fields].equals(b[fields]))})
    pressure = stats.loc[stats.cost.eq("STRESS")].set_index(["period", "policy"])
    recent = pressure.loc[("2020_2026", model.POLICIES[2])]
    early = pressure.loc[("2015_2019", model.POLICIES[2])]
    path = OUT / "results/accounts/2015_2019/STRESS"
    a = pd.read_parquet(path / model.POLICIES[1] / "trades.parquet")
    b = pd.read_parquet(path / model.POLICIES[2] / "trades.parquet")
    extra = b.loc[~b.entry_origin.isin(a.entry_origin)]
    require(len(extra) == 1 and extra.status.eq("COMPLETE").all(), "新增点位解释与保存交易不符。")
    detail = {"at": now(), "status": "SAVED_ONLY_INCREMENT_AND_COST_ATTRIBUTION_COMPLETE",
        "comparison_scope": "同一个16字段成熟共同池，14字段控制与16字段候选；不替代原R198不同池结果。",
        "daily_economic_checks": economic_checks,
        "entry_different_origins_total": int(yearly.entry_different.sum()),
        "entry_different_years": yearly.loc[yearly.entry_different.gt(0), "year"].astype(int).tolist(),
        "early_pressure_extra_completed_points": extra.to_dict("records"),
        "early_pressure_ending_equity_increment": float(early.ending_equity - pressure.loc[("2015_2019", model.POLICIES[1]), "ending_equity"]),
        "recent_pressure_gross_pnl_at_actual_quantities": float(recent.gross_at_actual_quantities_pnl),
        "recent_pressure_actual_friction": float(recent.total_commission + recent.total_slippage),
        "recent_pressure_net_account_pnl": float(recent.ending_equity - 200000.),
        "cost_counterfactual_strategy_not_run": True, "independent_validation": "NOT_ESTABLISHED",
        "charts_visually_inspected": delivery["charts"], "charts_visually_inspected_count": 4,
        "new_accounts_rerun": 0, "new_fits": 0, "new_labels": 0, "new_requests": 0}
    write_json(OUT / "description_receipt.json", detail, exclusive=True)
    report_path = OUT / "不同信息多因子评分_全部结论与点位.md"
    report = report_path.read_text(encoding="utf-8").replace("勝率", "胜率")
    report += "\n补充实际增量解释：外汇字段进入284个可评分原点的树路径，相对14字段控制也有284个预测改变，但只有2017年12个原点改变进场门。实际较早压力账户多出2017-01-03决定、01-04开盘进、02-08开盘出的1笔20收盘到期交易，净赚192.82580元；期末多221.32476元还包含随后共同交易数量变化。候选早期胜率升至4/6，但实际B降至0.72743、pB0.48496；增加小盈利可以提高胜率而降低平均盈利与亏损比。\n\n"
    report += "2020—2026两种费用下候选和14字段控制的逐日净值、现金、持股、净收益、毛价格损益及费用完全相同，不能把多出的字段数当成交易改善。近期压力17笔为5胜12负，实际B1.37395、pB0.40410、标准期望−0.30178；按实际成交数量毛损益−3902.50元、佣金滑点3269.28120元、净账户损益−7171.78120元。毛损益已经负，因此本次失败含点位筛选问题；没有运行零费或不同数量的反事实策略。\n\n"
    report += "四案例解读：2015反弹处于日柱和周柱走弱、融资下降，反弹后的继续下跌为失败对照；该年本模型无足够共同成熟训练，评分NO_VIEW，不能把后来建成的模型回填。2019-01-08日柱转正/上一周仍负，01-14周柱也转正，量仍低于近期中位数；固定宏观及外汇树都未提前识别此启动。2020日柱多次修复和回落，必须连同失败段保留，人民币路径并未改变当期实际进场。2024-09-24相对量3.33842倍、日柱/ATR0.30279、上一完整周柱/ATR−0.27422，体现快修复与周线滞后；三模型估计pB分别0.82774、0.56206、0.56206，均未满足原>1进场门，2024全年度0个候选进场信号。不能因后来看见上涨就降门槛补买。完整51行关键日和720行四案例评分已保留。\n\n"
    report += "本次接受来源接入、保守可知钟、同池增量可测和保存账户复算；拒绝唯一固定16字段配置可提升完整账户收益夏普的假设。保留原A仅作为共同口径控制，它的早期pB未达1、两期压力夏普均未达1.5，因此也不等于完整目标已完成。原独立前瞻13项状态及既定E03计划不改；无已准入待跑配置，不重试本配置，不宣称所有宏观或外汇机制无效。四图已实际查看，缺失断线保持。\n"
    report_path.write_text(report, encoding="utf-8")
    before_bytes = STATE.read_bytes()
    (OUT / "state_before_TECH_R208.json").write_bytes(before_bytes)
    state = read(STATE)
    forward_before = {key: state[key] for key in FORWARD}
    goal_receipt = read(OUT / "goal_service_status_after_result.json")
    require(goal_receipt["goal"]["status"] == "active", "目标服务真实状态不是active。")
    previous = {key: state.get(key) for key in ["latest_technical_decision", "latest_result", "latest_report", "latest_progress",
        "current_phase", "current_phase_trial_accounting", "current_goal_turn_actual_work", "goal_turn_classification"]}
    state["previous_current_phase_before_TECH_R208"] = previous
    prefix = OUT.absolute().relative_to(ROOT).as_posix()
    brief = ("TECH.R207—R208唯一技术8/宏观6/人民币2同池配置一次完成：3488日/2433同知，142月中124支持/372拟合，2164共同评分；"
        "4原A精确复现、4候选和8配对控制共12新账户、12保存复算、5新必要测试。实际外汇路径284，仅2017年12原点改变进场、多1实际盈利192.8258元。"
        "较早压力年化0.15991%/夏普0.16480/6笔4胜2负/B0.72743/pB0.48496；近期−0.56092%/−0.42111/17笔5胜12负/B1.37395/pB0.40410/期望−0.30178/DD6.44613%。"
        "近期与14字段控制经济路径完全相同，毛损益−3902.50元/摩擦3269.2812元；四经济门0通过/稳定失败，拒绝固定用途。0新标签/采集/原策略改动；独立、去过拟合及完整目标未达。")
    trials = {"scope": "TECH_R207_R208_FIXED_CNH_MACRO_FIRST_PASSAGE_SAME_POOL",
        "candidate_configurations": 1, "new_candidate_accounts": 4, "new_matched_control_accounts": 8, "new_accounts": 12,
        "original_A_exact_replays": 4, "monthly_fit_records": 142, "paired_months_fitted": 124, "actual_fit_calls": 372,
        "common_available_forecasts_per_policy": 2164, "joint_known_daily_rows": 2433, "actual_fx_path_origins": 284,
        "new_labels": 0, "new_market_requests": 0, "necessary_new_tests_passed": 5, "saved_account_checks": 12,
        "all_four_economic_gates_passed": False, "historical_stability_gate_passed": False,
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False}
    updates = {"status": "research_active", "goal_status": "active", "goal_achieved": False,
        "current_phase": "FIXED_CNH_MACRO_FIRST_PASSAGE_FULL_ACCOUNT_COMPLETED_REJECTED",
        "latest_technical_decision": "TECH.R208", "latest_actual_financial_decision": "TECH.R208",
        "latest_registration_decision": "TECH.R207", "latest_financial_registration_decision": "TECH.R207",
        "latest_financial_strategy_decision": "TECH.R208", "latest_actual_model_decision": "TECH.R208",
        "latest_financial_strategy_result": prefix + "/summary.json", "latest_result": prefix + "/summary.json",
        "latest_completed_study": prefix, "latest_report": prefix + "/不同信息多因子评分_全部结论与点位.md",
        "latest_research_report": prefix + "/不同信息多因子评分_全部结论与点位.md", "latest_research_status": summary["status"],
        "latest_progress": brief, "latest_progress_check": prefix + "/summary.json",
        "latest_cnh_macro_first_passage": prefix + "/summary.json",
        "latest_joint_macro_case_description": prefix + "/description_receipt.json",
        "latest_concrete_case_explanation": prefix + "/具体上涨与共同支持_拟合前.md",
        "current_phase_trial_accounting": trials, "current_candidate_trials": trials,
        "current_goal_turn_actual_work": {"new_complete_financial_uses": 1, "new_candidate_configurations": 1,
            "new_fits": 372, "new_accounts": 12, "original_A_exact_replays": 4, "saved_account_checks": 12,
            "necessary_new_tests_passed": 5, "new_labels": 0, "native_market_requests": 0, "new_quote_rows": 0},
        "goal_turn_classification": "PROGRESS_DIFFERENT_SOURCE_FULL_FINANCIAL_PURPOSE_COMPLETED_AND_REJECTED",
        "consecutive_blocked_goal_turns": 0, "blocked_key": None, "blocked_reason": None, "blocked_scope": None,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": now(),
        "latest_goal_tool_status_receipt": prefix + "/goal_service_status_after_result.json",
        "current_admitted_unrun_complete_uses": 0, "current_admitted_unrun_numeric_candidates": 0,
        "current_financial_candidate_admission": "FIXED_USE_REJECTED_NOT_PROMOTED",
        "new_financial_result_computed_this_continuation": True,
        "new_account_return_sharpe": "COMPUTED_FULL_ACCOUNT_GATE_FAILED",
        "new_account_return_sharpe_result": prefix + "/results/十六完整账户共同口径比较.parquet",
        "account_return_sharpe_reason": "完整四场景已计算，固定配置失败；不是NOT_COMPUTED。",
        "current_full_account_economic_gate_passed": False,
        "current_phase_known_daily_rows": 3488, "current_phase_original_state_rows": 3488,
        "current_model_daily_qualified_origins": 2433, "current_phase_information_scope": "TECH8_MACRO6_CNH_FIXING2",
        "current_fields_admitted_this_continuation": 16, "current_phase_source_freeze_count": len(protocol["sources"]),
        "current_member_support_this_continuation": "3488日/2433同知，142月中124共同成熟支持；372拟合/每政策2164评分，成员权重逐月保存，不是独立样本认证。",
        "current_phase_original_case_rows": 240, "current_phase_original_episode_rows": 61, "current_phase_original_admitted_waves": 49,
        "necessary_tests_passed_this_continuation": 5, "necessary_test_executions_this_continuation": 5,
        "current_phase_required_tests": 5, "current_phase_financial_candidate_configurations": 1,
        "new_accounts_this_continuation": 12, "new_primary_accounts_this_continuation": 4,
        "new_financial_candidate_accounts_this_continuation": 4, "new_matched_control_accounts_this_continuation": 8,
        "new_strategy_accounts_this_continuation": 12, "saved_account_controls_replayed_this_continuation": 4,
        "internal_reference_replays_this_continuation": 4, "saved_accounts_checked_this_continuation": 12,
        "new_strategy_configurations_this_continuation": 1, "new_model_fits_this_continuation": 372,
        "new_training_labels_this_continuation": 0, "new_return_labels_this_continuation": 0, "new_market_requests_this_continuation": 0,
        "original_strategy_source_files_changed_this_continuation": 0, "code_files_changed_this_continuation": 0,
        "code_files_added_this_continuation": ["research/cnh_macro_first_passage_inputs_v1.py", "research/cnh_macro_first_passage_study_v1.py",
            "research/summarize_cnh_macro_first_passage_v1.py", "tests/test_cnh_macro_first_passage_v1.py"],
        "source_fields_prepared_this_continuation": 0,
        "validation_method_this_continuation": "新接入五测试通过；同成熟成员权重三模型；4原A精确复现、12新账户一次及12保存复算；四图已查看。",
        "next_financial_experiment": "NONE_ADMITTED_UNRUN_AFTER_TECH_R208_REJECTION",
        "next_information_source_proposal": "无已登记待跑配置或下一资料请求；仅在可核验的实质不同信息机制/完整用途、真实来源错误或真正新样本成立后另登记。原E03及13项前瞻不改，旧失败不参数营救。",
        "overfitting_removed": False, "whole_model_overfitting_removed": False,
        "independent_validation": "NOT_ESTABLISHED", "first_vintage": "NOT_CERTIFIED", "global_DSR_PBO": "NOT_COMPUTED"}
    state.update(updates)
    require({key: state[key] for key in FORWARD} == forward_before, "原13项前瞻被意外改变。")
    require(STATE.read_bytes() == before_bytes, "根状态同时被更新，需重新读取后合并。")
    write_json(STATE, state)
    sources = (f"依据：[固定完整用途](510300_CNH_MACRO_FIRST_PASSAGE_V1.md)；[全部结论与具体点位](../{prefix}/不同信息多因子评分_全部结论与点位.md)；"
               f"[四场景和全部配对区间](../{prefix}/summary.json)；[实际路径及费用分解](../{prefix}/description_receipt.json)。")
    state_note = ("\n\n> 本日周线技术线最新完整金融事实 TECH.R207—R208（2026-10-05，优先于下方本线旧快照）：" + brief +
        " 本轮实际新金融进展，目标服务active，连续受阻0；固定用途拒绝不等于目标实现，也不表示所有宏观/外汇机制无效。原13项前瞻、旧拒绝、原策略和其他研究线状态保持。\n\n" + sources +
        "\n\n下一步：当前无已准入待跑金融配置。仅在实质不同信息机制/完整用途、真实来源错误或真正新样本成立后另登记；原独立E03计划不改，不按本失败改树、窗口、门槛、符号、费用或退出，不拼接历史分支。\n\n")
    decision_note = ("\n\n> 最新研究裁决 TECH.R207登记 → TECH.R208实际结果（2026-10-05，本日周线技术线）：假设是离岸相对官方定盘偏离及五槽变化，在原技术/宏观上提供点位和完整账户增量。\n\n"
        "验证方法：先解释61/49分段、四案例240行和17关键日，全部3488日未知保留；唯一固定深3/每叶60/756日/原142月，技术8、技术宏观14和技术宏观人民币16同成熟成员同权重，原标签/退出/风险/资金/费用。5新必要测试、4原A精确复现、12新账户一次、12保存复算及24组20/252配对区间，0新股票标签或采集。\n\n"
        "结果：2433同知、124共同月/372拟合/2164共同评分，外汇路径284；仅2017年12原点改变进场，多1压力净赚192.8258元。早压力年化0.15991%/夏普0.16480/4胜2负/B0.72743/pB0.48496；近期−0.56092%/−0.42111/5胜12负/B1.37395/pB0.40410/标准期望−0.30178，与14字段控制经济路径完全相同，毛损益已负。四经济门0/4、稳定门失败。\n\n"
        "接受/拒绝及理由：接受来源接入、保守可知钟和同池可测，拒绝唯一16字段完整金融用途；字段更多/局部胜率更高并未改善完整收益夏普，且实际pB未达原>1门。原A保留为控制，早期pB及整体夏普1.5目标仍不足，不晋升。历史DEVELOPMENT_CALIBRATION、首版NOT_CERTIFIED、独立NOT_ESTABLISHED、global DSR/PBO NOT_COMPUTED，过拟合未移除。\n\n"
        "是否重新验证：本固定失败不重试或调参营救。只有实质不同信息机制/完整用途、确认真实来源错误或真正独立新样本才能另登记；原独立前瞻13项和E03保持。当前无待跑金融配置。实际新金融进展使本恢复周期连续受阻0，目标服务active、完整目标未达；不宣称全部宏观或外汇机制无效。\n\n" + sources + "\n\n")
    backups = OUT / "documents_before_TECH_R208"
    backups.mkdir(exist_ok=True)
    doc_receipts = []
    for path in DOCS:
        prior_bytes = path.read_bytes()
        (backups / path.name).write_bytes(prior_bytes)
        first = prior_bytes.find(b"\n") + 1
        require(first > 0, "事实文档没有标题行。")
        note = decision_note if "DECISIONS" in path.name else state_note
        encoded = note.replace("\n", "\r\n").encode("utf-8")
        replacement = prior_bytes[:first] + encoded + prior_bytes[first:]
        require(path.read_bytes() == prior_bytes, "事实文档同时被更新，需合并。")
        path.write_bytes(replacement)
        require(path.read_bytes()[first + len(encoded):] == prior_bytes[first:], "既有正文没有逐字节保留。")
        doc_receipts.append({"path": path.relative_to(ROOT).as_posix(), "sha256_after": digest(path), "prior_body_bytes_preserved": True})
    write_json(OUT / "project_state_update_receipt.json", {"at": now(), "status": "PASS_FOUR_FACT_DOCUMENTS_AND_ROOT_STATE_SYNCED",
        "latest_technical_decision": "TECH.R208", "latest_actual_financial_decision": "TECH.R208", "goal_status": "active",
        "consecutive_blocked_goal_turns": 0, "goal_achieved": False, "forward_fields_preserved_exactly": FORWARD,
        "current_admitted_unrun_complete_uses": 0, "state_sha256_after": digest(STATE), "documents": doc_receipts,
        "new_financial_runs_in_summary_stage": 0, "new_fits_in_summary_stage": 0, "new_market_requests": 0}, exclusive=True)
    print("TECH.R208完整点位解释和四份事实文件已同步，原13项前瞻逐值保留；目标未完成。", flush=True)


if __name__ == "__main__":
    main()
