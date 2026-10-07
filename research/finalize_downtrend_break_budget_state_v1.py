"""保存日均价旧用途核对，并一次更新R174长期事实，金融R173与前瞻保持。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.downtrend_break_budget_attribution_v1 import OUT, PRIOR, FORWARD_KEYS, DOCS
from research.point_first_passage_study_v1 import read, write_json, digest, now, require

STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def old_daily_vwap_review():
    """同表达只定位旧实际结果，不重跑、反号或替换旧实验口径。"""
    path = OUT / "old_vwap_duplicate_recheck.json"
    require(not path.exists(), "本轮日均价旧用途已经核对。")
    intake_path = ROOT / "reports/research/510300_point_method_intake_20261002/prior_direction_review.json"
    registered_path = ROOT / "reports/research/registered_factor_research.json"
    account_path = ROOT / "reports/backtest/registered_factor_backtests.json"
    old = read(intake_path)["daily_close_vwap"]
    registered = read(registered_path)
    selected = [row for row in registered["results"] if row["hypothesis_id"] in ("H008", "H018")]
    require(len(selected) == 2 and all(row["factor_id"] == "ETF_CLOSE_VWAP" for row in selected),
            "未找到旧日成交均价的两实际用途。")
    require(old["formula"] == "Close/(Amount/Volume)-1", "旧字段表达发生变化。")
    require({row["hypothesis_id"]: row["evidence"]["rating"] for row in selected} ==
            {"H008": "REJECTED", "H018": "INCONCLUSIVE"}, "旧实际终态与当前记录不同。")
    accounts = [row for row in read(account_path)["factors"] if row["factor_id"] == "ETF_CLOSE_VWAP"]
    require(len(accounts) == 1 and accounts[0]["candidate_status"] == "NOT_ELIGIBLE", "旧单因子准入状态不同。")
    write_json(path, {
        "at": now(), "role": "FINITE_SAVED_OLD_EXPRESSION_RECHECK_NOT_A_NEW_FINANCIAL_STUDY",
        "formula": old["formula"], "factor_id": "ETF_CLOSE_VWAP", "exact_prior_match": old["exact_prior_match"],
        "actual_old_prediction_results": [{"hypothesis_id": row["hypothesis_id"], "target": row["target"],
                                           "horizon": row["horizon"], "evidence": row["evidence"]} for row in selected],
        "old_single_factor_candidate_status": accounts[0]["candidate_status"],
        "old_study_account_scope": "旧10万元/242日；不和当前20万元/252日金融结果混排。",
        "decision": "STOP_DUPLICATE_EXPRESSION_NO_NEW_FIELD_OR_POLICY_ADMISSION",
        "scope_limit": "本相同字段及其旧实际用途关闭；不是所有VWAP用途的普遍否定。96因子A05事件锚定多日代理原NOT_RUN身份保持，不由其名义状态推准入。",
        "new_accounts": 0, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in
                    [Path(__file__), intake_path, registered_path, account_path,
                     ROOT / "config/factor_registry.yaml", ROOT / "research/registered_factors.py"]],
    }, exclusive=True)


def run():
    require(not (OUT / "project_state_update_receipt.json").exists(), "R174长期事实已经更新，不重复追加。")
    summary, delivery = read(OUT / "summary.json"), read(OUT / "delivery_receipt.json")
    require(summary["technical_decision"] == "TECH.R174" and summary["actual_orders_verified"] == 288,
            "全部原订单归因尚未完成。")
    require(delivery["status"] == "PASS_ALL_SAVED_BUDGETS_ORDERS_AND_FOUR_CASE_COVERAGE_DELIVERED",
            "具体案例与全体记录交付尚未完成。")
    state = read(STATE)
    require(state["latest_financial_strategy_decision"] == "TECH.R173", "接手的最新金融策略不是R173。")
    before = {key: state[key] for key in FORWARD_KEYS}
    original_next_forward = state["next_experiment"]
    original_financial_trials = state["actual_candidate_trials"]
    marker = "> 技术线当前事实 TECH.R174（优先于下方技术线历史快照，2026-10-05）："
    texts = {name: (ROOT / "docs" / name).read_text(encoding="utf-8-sig") for name in DOCS}
    require(all(marker not in text for text in texts.values()), "事实文件已存在R174当前条目。")
    write_json(OUT / "project_state_before.json", state, exclusive=True)
    old_daily_vwap_review()
    next_boundary = {
        "at": now(), "status": "NO_NEW_ADMITTED_FINANCIAL_CANDIDATE_AFTER_ALL_BUDGET_DIAGNOSTICS",
        "latest_actual_financial_decision": "TECH.R173", "latest_completed_diagnostic": "TECH.R174",
        "accepted_new_fact": "全132资格、128空仓计划、1062持有决定和288原买卖订单已精确归因；主要入场约束为跳空与ES，费用/小额减仓不足以解释近期对A差距。",
        "next_required_input": "实质不同的日线点位信息与完整用途、真实独立新样本，或真实来源/实现错误。先明确对象、可知时钟、旧用途和全体对照，再决定准入。",
        "closed_current_proposals": ["R173参数/过滤/退出/费用/风险预算营救", "仅省小额减仓手续费追平A", "同日Close/(Amount/Volume)-1当作新字段", "机械等额/放大或混A"],
        "no_unregistered_policy_run": True, "new_admitted_unrun_candidates": 0,
        "new_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "historical_sample_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "existing_forward_protocol": state["forward_account_comparison_protocol"],
        "registered_earliest_new_close": state["next_new_close_eligible_at"],
        "forward_next_experiment_unchanged": original_next_forward,
        "scope_limit": "本预算归因及同表达用途关闭不是全项目否定；原四场景收益/夏普与真实前瞻登记保持，不能从日历未到或旧文件构造独立收益。",
        "goal_achieved": False,
    }
    write_json(OUT / "next_point_information_admission_boundary.json", next_boundary, exclusive=True)
    overview = (
        "完成R174全体原资金诊断：132资格、128空仓预算、1062原持有决定及288原订单（106买/182卖）精确复现，"
        "106周期104完成/2开放保持，28入场/28持有减仓/12费用单元及原四固定案例资金图交付。"
        "压力64空仓中55跳空预算绑定、9ES；62跳空预算来自剩余回撤余量、2与基础预算同值，无现金不足买入。"
        "2019-01-18/2024-09-24当时回撤6.8862%/6.8733%，实际投入16.0547%/16.1377%；2020-05-29由ES约束投入30.7331%。"
        "近期压力18风险减仓仅118.50元/总摩擦5227.61元约2.27%，实际份额毛+6693.60元仍低于A净+57847.49元，"
        "差额−51153.89元，省费/小额减仓不能单独追平A；持有320日对A441、全日历暴露5.1342%对8.4586%，不能据此放大或换预算。"
        "112预算来源、0新账户/拟合/训练标签/采集，固定失效/预算保持。有限日均价核原结果H008/20日REJECTED0/7、"
        "H018/5日INCONCLUSIVE3/7、单因子NOT_ELIGIBLE，同表达不新准入，旧10万/242日不混当前20万/252日。"
        "最新技术诊断R174、金融R173/登记R172/实际预测R158/原退出R145保持，四场景pB正及完整账户拒绝不改。"
        "独立/去过拟合与完整收益夏普目标未达，目标active/本轮progress/阻塞0；当前无已准入新策略，下一只不同信息/完整用途或真实新样本的准入，十二前瞻和原1008实际交易日计划不变。"
    )
    top = marker + (
        "全128空仓预算、1062持有决定和288订单归因完成；压力64空仓55受跳空预算约束/9受ES。"
        "近期小额风险减仓摩擦118.50元仅占总摩擦2.27%，实际份额毛盈利6693.60元仍远低A净57847.49元，费用不能单独解释差距。"
        "0新账户，金融R173及其四场景pB正/整体拒绝保持；日均价同表达已核旧实际终态，不重新准入。"
        "目标未达且active，下一不同信息用途/真实新样本准入，无待跑新金融候选，原真实前瞻保持。"
        " [全体预算与资金覆盖诊断](../reports/research/510300_downtrend_break_budget_attribution_v1/研究结果与下一步.md)。\n\n"
    )
    detailed = """
## TECH.R174：正交易期望与弱完整账户之间的全体资金诊断（2026-10-05）

**假设**：R173四场景已完成交易期望为正而近期完整账户显著低于原A，可能涉及已知风险预算压缩入场、持有减仓和摩擦，需要完整归因。

**验证方法**：固定一次保存结果诊断、112来源，不跑账户。全132资格中128空仓原点按当时NAV/现金/应收/峰值/已知ES及50%意向复算原cap_quantity，4已有持仓资格不新生周期；次真实开复算旧H2取消、限价、开盘风险/现金与买入摩擦。全部1062持有原点因果复算只上移确认低点、已知失效或cap_quantity(existing=q)，全部288订单逐项匹配份额/次开/T+1/成交价/佣金/滑点。两时期两费用七风险集合各28入场/28减仓单元、三订单类型12费用单元全报告，空组未知均值保留，原106周期104完成/2开放和全部取消保持。新测试0，新账户/拟合/训练标签/采集0；这些是为解释而做的全部保存账本验证，不声称重跑新策略。原四案例资金图已目视核对。

**结果**：压力64次空仓请求55跳空绑定、9ES绑定；62原跳空预算来自半剩余10%回撤余量、2基础与余量同值，没有现金不足买入。早期27空仓6ES/21跳空、近期37空仓3ES/34跳空，两费用同绑定数量。2019-01-18当时回撤6.8862%、已知ES5.2765%、跳空预算3593.98元，计划11100份/实际10900份，投入16.0547%；2019-03-18失败点投入27.1317%，亏损完整保留。2020-05-29回撤1.8409%、ES7.8047%绑定，计划15900/实际15500份、投入30.7331%。2024-09-24回撤6.8733%、ES4.8991%、跳空预算3302.78元，计划9400/实际9100份，投入16.1377%，该日A库存0但原已知目标31.9188%。不得据后来输赢反改预算。

压力持有减仓早24次（17ES/7跳空）、近期18次（6ES/12跳空）；全部减仓摩擦早174.497元、近118.50元，占近全部5227.6082元约2.27%，小额最低佣金不能解释主要缺口。近期STRESS突破账户实际份额毛增6693.60元，对A净增57847.49296元，差−51153.89296元；BASE毛7514元对A净61823.03124元，差−54309.03124元。早期BASE毛22754.20/A净21532.34702、STRESS毛22352.10/A净18410.68428元也完整列出，不能只选近期或称四期毛差都负。近期真实持有320日对A441，全日历暴露STRESS5.1342%对8.4586%；早211对435、4.4561%对11.8125%。毛增是原实际份额账本事实，不是零费账户或零费夏普。R173完成周期协方差恒等式保持，不能把均值项当等额策略。

有限重核日成交均价信息：直接定位原注册源码Close/(Amount/Volume)-1及原结果H008/20日REJECTED0/7、H018/5日INCONCLUSIVE3/7、单因子NOT_ELIGIBLE。旧研究10万元/242日不得和当前20万元/252日混排；同表达停止、不反号/换窗。它不是所有VWAP用途普遍无效，也不由96因子A05原NOT_RUN推金融准入。20/60量加权和新低锚的原用途保持。

**为什么接受/拒绝**：接受原预算/执行/持有约束与费用事实，排除“省小额减仓手续费即可追平A”及“费用是近期全部不足”的解释。风险约束压缩部分赢家是事实，不等于放宽风险或筛预算类别可提升夏普；入场与全体持有路径同样需要新的信息证据。原A、R173固定政策拒绝、R166/R167和全部旧配仓/波动/半仓/自身盈利终态不变。全体发展样本、first-vintage未认证、独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED、去过拟合与完整收益夏普目标未达到；本轮没有新的金融提升。

**是否需要重新验证/下一步**：本固定归因一次完成，不变原金融参数或再分赢家组。只有真实来源/实现错误才复核并保留原证据。下一须先明确实质不同的日线点位信息及完整用途，或真实独立新样本；当前新金融准入/待跑0，无新策略可直接回测。已有A/POINT真实前瞻十二值及1008实际交易日口径保持，原登记最早资格新收盘仍2026-10-08 15:05，未发生不虚构。目标active、本轮progress/阻塞0；最新技术诊断R174、实际金融R173、登记R172、预测R158/原退出R145保持。单项关闭不是全项目停止。

依据：[全体预算及四案例资金覆盖](../reports/research/510300_downtrend_break_budget_attribution_v1/研究结果与下一步.md)、[固定分解定义](../reports/research/510300_downtrend_break_budget_attribution_v1/protocol.json)、[全体实际核对](../reports/research/510300_downtrend_break_budget_attribution_v1/summary.json)、[旧日均价实际终态重核](../reports/research/510300_downtrend_break_budget_attribution_v1/old_vwap_duplicate_recheck.json)、[下一信息准入边界](../reports/research/510300_downtrend_break_budget_attribution_v1/next_point_information_admission_boundary.json)、[原R173实际金融](../reports/research/510300_downtrend_break_study_v1/summary.json)。
"""
    for name, text in texts.items():
        first, remainder = text.split("\n", 1)
        append = detailed if name != "PROJECT_STATE.md" else "\n## 技术线TECH.R174当前更新（2026-10-05）\n\n" + overview + "\n\n完整假设→验证→结果→接受/拒绝→重验入口见[技术线决策](RESEARCH_DECISIONS_TECHNICAL_LINE.md)；共享其他分支范围和终态保持。\n"
        (ROOT / "docs" / name).write_text(first + "\n\n" + top + remainder.lstrip("\n") + "\n" + append,
                                         encoding="utf-8", newline="\n")
    relative = OUT.relative_to(ROOT).as_posix()
    next_action = "只有实质不同日线点位信息/完整用途、真实独立新样本或真实来源/实现错误才进一步准入；当前无新已准入金融候选。日均价同表达停止，不能改R173预算/过滤/退出/费用/窗口或混A救回；原真实前瞻登记保持。"
    work = {
        "scope": "TECH_R174_FIXED_ALL_SAVED_BUDGET_HOLDING_AND_ORDER_DIAGNOSTIC",
        "qualification_rows": 132, "unique_origins": 66, "native_origin_budgets_verified": 128,
        "holding_decisions_verified": 1062, "original_orders_verified": 288,
        "original_buys_verified": 106, "original_sells_verified": 182,
        "original_cycle_rows": 106, "original_completed_cycles": 104, "original_open_cycles": 2,
        "entry_budget_cells": 28, "holding_budget_cells": 28, "order_cost_cells": 12,
        "frozen_budget_sources": 112, "charts": 1, "necessary_new_tests": 0,
        "new_policy_accounts": 0, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "risk_budget_changes": 0, "old_vwap_actual_use_rechecks": 3,
    }
    state.update({
        "at": now(), "updated_at": now(), "latest_completed_study": relative,
        "latest_result": relative + "/summary.json", "latest_report": relative + "/研究结果与下一步.md",
        "latest_research_report": relative + "/研究结果与下一步.md", "latest_research_status": summary["status"],
        "latest_progress": overview, "latest_overall_summary": overview, "latest_continuation_outcome": overview,
        "current_study": "510300_DOWNTREND_BREAK_SAVED_BUDGET_ATTRIBUTION_V1",
        "current_phase": "ALL_SAVED_BUDGETS_HOLDING_AND_ORDERS_EXPLAINED_FINANCIAL_R173_UNCHANGED",
        "current_direction": "全体原预算/资金覆盖/费用解释完成；费用和小额减仓不足以解释近期对A差距，旧日均价重复用途关闭。",
        "current_priority": next_action, "next_research_action": next_action, "next_available_action": next_action,
        "next_research_plan": relative + "/next_point_information_admission_boundary.json",
        "next_experiment_status": "DISTINCT_INFORMATION_OR_TRUE_NEW_SAMPLE_REQUIRED_NO_ADMITTED_FINANCIAL_CANDIDATE",
        "next_candidate_field_status": "OLD_DAILY_CLOSE_VWAP_DUPLICATE_STOPPED_NO_NEW_FIELD_ADMITTED",
        "next_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "next_strategy_increment_status": "NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY", "current_admitted_unrun_numeric_candidates": 0,
        "latest_technical_decision": "TECH.R174", "latest_actual_model_decision": "TECH.R173",
        "latest_financial_strategy_decision": "TECH.R173", "latest_registration_decision": "TECH.R172",
        "latest_financial_registration_decision": "TECH.R172",
        "latest_actual_prediction_model_decision": "TECH.R158", "latest_original_exit_decision": "TECH.R145",
        "latest_continuation_receipt": relative + "/delivery_receipt.json",
        "latest_downtrend_break_budget_attribution": relative + "/summary.json",
        "latest_downtrend_break_all_budget_points": relative + "/results/全部132资格_原预算开盘执行与真实资金.csv",
        "latest_old_daily_vwap_recheck": relative + "/old_vwap_duplicate_recheck.json",
        "latest_new_information_admission_boundary": relative + "/next_point_information_admission_boundary.json",
        "current_phase_trial_accounting": work, "current_goal_turn_actual_work": work,
        "actual_candidate_trials_role": "LATEST_FINANCIAL_R173_TRIAL_ACCOUNTING_PRESERVED_NOT_R174_DIAGNOSTIC",
        "new_accounts_this_continuation": 0, "new_accounts_in_current_phase": 0,
        "new_primary_accounts_this_continuation": 0, "new_financial_candidate_accounts_this_continuation": 0,
        "new_investment_account_evaluations_this_continuation": 0, "internal_reference_replays_this_continuation": 0,
        "saved_account_controls_replayed_this_continuation": 0, "saved_accounts_checked_this_continuation": 4,
        "saved_original_entry_quantities_verified_this_continuation": 106,
        "saved_original_sell_quantities_verified_this_continuation": 182,
        "necessary_tests_passed_this_continuation": 0, "necessary_tests_passed_in_current_phase": 0,
        "actual_prefix_checks_this_continuation": 0, "new_model_fits_this_continuation": 0,
        "historical_model_refits_this_continuation": 0, "new_return_labels_this_continuation": 0,
        "new_training_labels_this_continuation": 0, "new_model_training_labels_this_continuation": 0,
        "new_market_requests_this_continuation": 0, "new_financial_result_computed_this_continuation": False,
        "new_allocation_method_admitted": False, "new_method_admitted": "SAVED_BUDGET_DIAGNOSTIC_ONLY_NO_NEW_POLICY",
        "new_market_information_admitted": False, "returns_and_sharpe_improved": False,
        "return_and_sharpe_improved_this_continuation": False,
        "original_strategy_source_files_changed_this_continuation": 0,
        "code_files_added_this_continuation": ["research/downtrend_break_budget_attribution_v1.py", "research/finalize_downtrend_break_budget_state_v1.py"],
        "previous_goal_turn_classification": "progress", "current_goal_turn_classification": "progress",
        "goal_turn_progress_classification": "NEW_ALL_ORIGINAL_BUDGET_AND_ORDER_EVIDENCE_NO_FINANCIAL_POLICY_CHANGE",
        "current_goal_turn_classification_reason": "全128空仓/1062持有/288原订单核对明确跳空与ES来源，量化小额减仓及毛资金对A不足，排除省费解释，并核旧日均价同表达，改变下一准入入口。",
        "goal_status": "active", "goal_tool_status_confirmed": "active", "goal_achieved": False,
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0,
        "blocked_reason": None, "blocked_audit_key": None, "blocking_decision": None,
        "whole_model_overfitting_removed": False, "overfitting_removed": False,
        "overfit_removed": False, "independent_validation_status": "NOT_ESTABLISHED",
        "current_unmet_evidence": "完整金融R173收益/夏普门及稳定失败保持；费用/小额减仓不足以追平A。无新已准入不同信息完整策略、真实独立新样本/去过拟合未建立。",
        "validation_method_this_continuation": "固定112来源，一次全132资格/128原预算/1062只上移失效和持有风险决定/288原买卖核对，28+28预算/12费用单元、原四例资金图，0新账户或测试重跑。",
    })
    require({key: state[key] for key in FORWARD_KEYS} == before, "十二真实前瞻值意外变化。")
    require(state["next_experiment"] == original_next_forward and state["actual_candidate_trials"] == original_financial_trials,
            "原前瞻计划或金融R173试验账意外变化。")
    write_json(STATE, state)
    write_json(OUT / "project_state_update_receipt.json", {
        "at": now(), "status": "PASS_FOUR_DURABLE_FACT_FILES_AND_CURRENT_STATE_UPDATED_ONCE",
        "latest_technical_diagnostic": "TECH.R174", "latest_financial_preserved": "TECH.R173",
        "registration_preserved": "TECH.R172", "actual_prediction_preserved": "TECH.R158", "original_exit_preserved": "TECH.R145",
        "forward_values_unchanged": before, "forward_next_experiment_unchanged": original_next_forward,
        "actual_financial_trial_accounting_unchanged": original_financial_trials,
        "goal_status": "active", "goal_achieved": False, "new_accounts": 0, "new_admitted_unrun_candidates": 0,
        "sources": [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in
                    [Path(__file__), STATE, *(ROOT / "docs" / name for name in DOCS)]],
    }, exclusive=True)
    print("四长期事实及当前状态已一次更新至R174；实际金融R173、十二前瞻和原1008日计划保持，目标active。", flush=True)


if __name__ == "__main__":
    run()
