"""一次归档R187—R189实际结果，更新四长期事实及技术线当前状态。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.volume_lead_price_confirm_study_v1 import OUT, EXPLANATION, PRIMARY
from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.finalize_downtrend_break_state_v1 import STATE, FORWARD, DOCS


def run():
    require(not (OUT / "project_state_update_receipt.json").exists(), "本轮事实已更新，不重复。")
    financial, explanation, delivery = [read(p) for p in
        (OUT / "summary.json", EXPLANATION / "summary.json", OUT / "delivery_receipt.json")]
    require(financial["technical_decision"] == "TECH.R189" and financial["new_primary_accounts"] == 4, "R189实际金融未完成。")
    require(delivery["status"] == "PASS_ALL_SAVED_VOLUME_PRICE_POINTS_AND_CONCRETE_PHASES_DELIVERED"
            and delivery["all_qualification_rows"] == 88 and delivery["all_actual_cycle_rows"] == 24
            and delivery["complete_exit_rows"] == 20 and delivery["individual_economic_gate_passes"] == 0,
            "全体交付和当前实际结论不一致。")
    state = read(STATE)
    require(state["latest_technical_decision"] == "TECH.R186", "状态已经由另一阶段推进，不覆盖。")
    state_before_hash = digest(STATE)
    forward_before = {key: state[key] for key in FORWARD}
    next_forward_before = state["next_experiment"]
    texts = {name: (ROOT / "docs" / name).read_text(encoding="utf-8-sig") for name in DOCS}
    hashes = {name: digest(ROOT / "docs" / name) for name in DOCS}
    marker = "> 技术线当前事实 TECH.R189（优先于下方技术线历史快照，2026-10-05）："
    require(all(marker not in text for text in texts.values()), "本轮已进入长期事实，不能重复。")
    rel, expl_rel = OUT.relative_to(ROOT).as_posix(), EXPLANATION.relative_to(ROOT).as_posix()
    next_path = OUT / "下一不同信息准入提案_成分参与广度_未登记.md"
    next_text = """# 下一不同信息准入提案：成分参与广度，尚未登记

当前R189累计量先行/价格后确认完整用途已拒绝并关闭，不能改其确认窗口、量符号、参考锚、固定低点、时间退出、费用、预算或与A混合营救。原技术/宏观/成分研究终态分别保持。

下一次仅进行有限可行性核对：原上涨及失败段中，是否存在标的日线价量之外、当时可知的指数成分参与信息。拟核对已保存成分股上涨/下跌成交及参与比例的原完整用途、成员与权重时钟、缺失/停牌/退市处理及已有实际裁决。它只是一个待核对的信息方向，不是新公式、配置或准入的交易策略。

执行顺序固定：先查本地原代码/协议/实际结果并列出旧等价用途；再检查原本地当时成员资料及每日报价是否足以重建资料可得性。不能使用今天的成分名单倒填历史，不能把事后成员或未知资料变成信号。发现同等完整用途已拒绝，则记录旧裁决而不重开；未能建立当时可得性则保留NO_VIEW/NOT_ADMITTED，不以ETF量或跨ETF代理替代。

只有确有实质不同信息与完整用途、且上述条件成立，才另外保存完整用途卡，之后先展开具体上涨/失败，再按原两时期两费用账户做唯一测量。该提案不登记新金融配置、不产生预测成员、账户、采集、持仓或订单。执行仍限510300.SH/CASH_CNY，成分只作观察。

原A/POINT真实前瞻协议、十二状态值及1008实际交易日终点评价不变；第一个尚未发生合格收盘不会用历史回填。全部既有历史DEVELOPMENT_CALIBRATION，独立0。项目目标保持active，本失败分支关闭不等于项目停止。
"""
    with next_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(next_text)
    overview = (
        "R187先解释具体上涨与全部失败：2019日柱1月8日、上一完整周柱1月14日、DIF1月18日陆续转正，量价同步改善而未形成本用途启动确认；"
        "2020年5月28日量准备/29日价格确认，日周柱仍负；2024年9月23日量准备/24日同锚价格确认，DIF/上周柱仍负，26日/30日慢指标才转正；2015准备失败保留。"
        "全3488日434高点锚/151准备/53全历史确认（2015起123/44），原49上涨31底峰间无确认/35确认至峰间无确认，全部59量失效/19低点失效/20准备后替换保持。"
        "R188登记唯一完整用途，6输入/6账户测试首轮通过、4真实前缀、8原A/价格控制精确复现；R189原4场景一次实际金融及保存核对完成。"
        "早BASE净年化0.8694%/夏普0.2756、STRESS0.8591%/0.2726，7完成/1开放、1胜6负、压力B3.8519/pB0.5503/标准期望−0.3069；"
        "近BASE0.7589%/0.2650、STRESS0.7435%/0.2606，3完成/1开放、1胜2负、压力B5.4688/pB1.8229/标准期望1.1563。"
        "四场景收益夏普均低于原A，完整年均次数1.4/0.5低于A4.4/4.1667；四经济门0通过、整体及历史稳定门失败，固定完整用途拒绝关闭。"
        "2020实际2020-06-01至2022-10-31+11.9506%；2024实际9月25日买入截至2026-09-30未完成，不进入完成胜率/B。"
        "全部88资格/24周期（20完成/4开放）及4资金恒等式交付；0拟合/新训练标签/市场采集，原冻结策略不改。"
        "全部历史DEVELOPMENT_CALIBRATION/first-vintage NOT_CERTIFIED/独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED，收益夏普完整目标及去过拟合未达，目标active，新准入待跑金融0。"
    )
    next_action = (
        "下一次只先核对本地成分参与广度信息的旧完整用途/实际裁决及历史成员和资料可得性；同等旧失败不重开、PIT未知NO_VIEW/NOT_ADMITTED。"
        "只有实质不同信息及完整用途成立才另写用途卡，随后具体上涨/失败解释和原4场景唯一测量；当前提案未准入、未登记、未跑金融。"
        "不改R189锚/固定线/时间退出或添加指标营救。原A/POINT十二前瞻值与1008实际交易日计划保持。")
    top = marker+overview+"\n\n"+f"依据：[具体上涨与完整账户](../{rel}/研究结果与下一步.md)、[原图与所有准备](../{expl_rel}/具体上涨与点位反推.md)、[实际点位](../{rel}/results/全部实际进出点位_准备确认及真实资金.csv)。下一步：{next_action}\n\n"
    decisions = (
        "\n\n## TECH.R187：先解释量价顺序和全部准备失败（2026-10-05）\n\n"
        "**假设**：累计涨跌成交份额能否先超过一个已确认高点中心日的累计量，价格随后突破同锚，形成慢动量恢复前的准备与确认？量不是主动买单或净申购，不证明上涨因果。\n\n"
        "**验证方法**：完整用途在新状态计算/结果连接前固定；已发生现金分红整数价方向×实际整数份额连续累计，首日0、缺失吸收未知；原严格2/2枢轴和交替尾部组件，今天新确认高点先替换旧锚。量先价未过且价格高于已确认低点准备，固定准备低点；之后不同日价过/量仍过确认，准备低点或量失效消耗不重试。有限完整用途与真实结果核对R116量模型、R37 VWMA、NVI、价量相关、排序、R170结构、R173高点突破及R185缺口收复；旧失败/成本敏感归档保持，不宣称穷尽等价策略。全3488/434锚、原61/49段、4案例及全部失败；6必要输入测试和4真实前缀。\n\n"
        "**结果**：151准备中53全历史价格确认，59量失效/19低点失效/20准备后替换；2015起123准备/44确认。49上涨31底峰间无确认/35原5%确认至峰无确认。2019没有1月启动确认，3月22日准备25日失效；2020年5月28日准备29日确认，量0.929/0.833、日周柱负；2024年9月23日准备24日确认，前日量0.736/RV0.917、确认量3.338/RV1.574，DIF/周柱仍负。2015准备失败，四图已查看。原2846标签/2826成熟/20删失/9尾部保持。\n\n"
        "**为什么接受/拒绝**：接受部分修复阶段及当时可知时钟的解释，拒绝称全部上涨共同启动规律或机构吸筹；0解释金融账户。\n\n"
        "**是否需要重新验证**：只有真实来源/实现错误复核，不能依据好案例改锚、累计窗口、确认顺序或添加MACD/量比。基础公式见TradingView官方OBV说明，分红/顺序/锚/失效为项目定义，官方不证明预测优势。\n\n"
        "## TECH.R188：唯一量先价后完整账户登记（2026-10-05）\n\n"
        "**假设**：可知准备与确认位置能否转为实际正期望，并提高原完整账户收益夏普及机会。\n\n"
        "**验证方法**：空仓价格确认次真实开一次，开盘到/低于准备低点或现金坐标未知取消。持有固定准备低点/参考高点/参考累计量，已知收盘到/低于低点，或量到/低于参考且价到/低于原高点共同失效次合法开退出，低点失败优先；未知自身不出、原风险只减，无加仓/抬线/目标/时间退出/期末清仓/混A或指标过滤。原20万元/252日/现金0/50%/ES跳空DD/股息/T+1/100份/.001/最低佣金5、两时期两费用及BASE买入STRESS风险检查保持；4场景净CAGR/Sharpe均>0/A/PRICE、DD<=10%、实际pB>1且pB−q>0，原20/252配对块各2000/固定种子，全部95%下界正才历史稳定门。\n\n"
        "**结果**：6输入/6账户必要测试首轮通过、8原控制精确复现，准入1完整配置/4新主账户的一次开发测量，0预测拟合或新训练标签。\n\n"
        "**为什么接受/拒绝**：接受可比较测量，不代表已完成高收益夏普、独立验证或模型修复；原预测MSE/成员门仅适用原预测用途。\n\n"
        "**是否需要重新验证**：本配置不得按结果改严格关系、2日确认、累计量、唯一锚/消耗、固定低点或共同退出、费用/预算/时期，不能改时间退出或混A营救。\n\n"
        "## TECH.R189：能解释2024准备，但完整用途低频且弱于A，拒绝关闭（2026-10-05）\n\n"
        "**假设**：具体好点位必须接受全体失败、持仓占用、实际退出及完整原控制。\n\n"
        "**验证方法**：4新账户一次测量/保存核对、全部88资格/24实际周期（20完成/4开放）、全部20退出与4资金恒等式交付；所有占用/取消/开放保留，未按事后最高点卖出。四段日周指标全部跨零时钟只是保存背景解释，不作为新增金融条件。\n\n"
        "**结果**：早BASE净CAGR0.8694%/Sharpe0.2756/DD9.66%，早STRESS0.8591%/0.2726/DD9.68%；7完成/1开放、p14.29%，压力B3.8519/pB0.5503/标准期望−0.3069。近BASE0.7589%/0.2650/DD9.15%，近STRESS0.7435%/0.2606/DD9.16%；3完成/1开放、p33.33%，压力B5.4688/pB1.8229/标准期望1.1563。近期正pB只基于1胜2负，不称可靠高胜率。四场景收益夏普均低于原A，原A近STRESS3.9908%/1.2169/DD2.75%。完整年均次数1.4/0.5低于A4.4/4.1667；四经济门0通过，整体/历史稳定门败。2020年6月1日进入至2022年10月31日退出+11.9506%，2024年9月25日进入截至2026年9月30日仍开放，不入完成胜率/B。2019启动没有实际周期；2015相交持仓源自3月而非6月反弹。\n\n"
        "**为什么接受/拒绝**：接受具体量先价后及慢动量滞后的开发事实，拒绝此固定完整用途作为提高模型收益夏普的证据并关闭。早期等周期期望负但实际年化正，来自原实际资金/持仓计价，不能互相替代。不能通过选近期、改时间/目标退出、抬线或添加MACD/量比补救。历史DEVELOPMENT_CALIBRATION、独立未建立/global DSR/PBO未算，目标active但未实现去过拟合。\n\n"
        "**是否需要重新验证/下一具体实验**：固定用途不重跑。下一只先核对本地成分参与广度的旧完整用途/裁决和当时成员/资料可得性；同等旧失败不重开，缺PIT保留NO_VIEW/NOT_ADMITTED。这是未登记提案，待跑金融0，不自动加入原A/POINT前瞻。之后只有实质不同信息及完整用途、真实新样本或真实来源/实现错误才新准入。\n\n"
        f"依据：[全部实际结果](../{rel}/研究结果与下一步.md)、[原解释](../{expl_rel}/具体上涨与点位反推.md)、[冻结](../{rel}/protocol.json)、[下一未登记提案](../{rel}/{next_path.name})。\n")
    for name in DOCS:
        path = ROOT / "docs" / name
        require(digest(path) == hashes[name], "事实文件同时被其他工作修改，不覆盖。")
        first, rest = texts[name].split("\n", 1)
        appendix = decisions if name.startswith("RESEARCH_DECISIONS") else "\n\n## 技术线R187—R189当前状态（2026-10-05）\n\n"+overview+"\n\n"+next_action+"\n"
        path.write_text(first+"\n\n"+top+rest.lstrip("\n")+appendix, encoding="utf-8", newline="\n")
    counts = delivery["scenario_counts"]
    pressure = [row for row in counts if row["cost"] == "STRESS"]
    accounting = {"scope": "TECH_R187_R188_R189_ONE_FIXED_VOLUME_FIRST_PRICE_LATER_POLICY", "candidate_configurations": 1,
        "new_policy_accounts": 4, "primary_accounts": 4, "saved_original_A_replays": 4, "saved_price_control_replays": 4,
        "all_known_daily_rows": 3488, "all_reference_anchors": 434, "all_volume_leads": 151, "all_price_confirmations": 53,
        "volume_leads_since_2015": 123, "price_confirmations_since_2015": 44, "all_qualification_rows_two_costs": 88,
        "all_actual_cycle_rows_two_costs": 24, "completed_rows_two_costs": 20, "open_rows_two_costs": 4,
        "actual_cycles_per_cost": sum(r["actual_cycles"] for r in pressure), "completed_per_cost": sum(r["completed"] for r in pressure),
        "open_per_cost": sum(r["open"] for r in pressure), "scenario_counts": counts,
        "original_episode_rows": 61, "original_admitted_waves": 49, "case_cycle_rows_two_costs": delivery["case_cycle_rows_two_costs"],
        "necessary_tests_final_passed": 12, "necessary_test_executions": 12, "actual_prefix_checks": 4, "charts": 4,
        "explanation_frozen_sources": explanation["frozen_sources"], "financial_frozen_sources": len(read(OUT / "protocol.json")["sources"]),
        "capital_identity_cells": 4, "all_four_point_quality_passed": False, "single_scenario_economic_passes": 0,
        "overall_economic_gate_passed": False, "historical_stability_gate_passed": False,
        "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0, "risk_budget_changes": 0}
    previous = {key: value for key, value in state.items() if key not in FORWARD and
                (key.startswith("current_phase_") or key.endswith("_this_continuation"))}
    state["archived_previous_current_phase_fields_before_R189"] = {"at": now(), "fields": previous}
    state["archived_financial_trial_accounting_R185_before_R189"] = state["actual_candidate_trials"]
    for key in previous:
        del state[key]
    state.update({
        "at": now(), "updated_at": now(), "latest_completed_study": rel, "latest_result": rel+"/summary.json",
        "latest_report": rel+"/研究结果与下一步.md", "latest_research_report": rel+"/研究结果与下一步.md",
        "latest_research_status": financial["status"], "latest_progress": overview, "latest_overall_summary": overview,
        "latest_continuation_outcome": overview, "current_study": "510300_VOLUME_LEAD_PRICE_CONFIRM_STUDY_V1",
        "current_phase": "VOLUME_FIRST_PRICE_LATER_FULL_POLICY_REJECTED_BELOW_A_AND_LOWER_FREQUENCY",
        "current_direction": "先解释四具体上涨量价/日周动量，再量先价后点位与原4场景金融；固定完整用途低频弱于A。",
        "current_priority": next_action, "next_research_action": next_action, "next_available_action": next_action,
        "next_research_question": next_action, "next_research_plan": next_path.relative_to(ROOT).as_posix(),
        "next_experiment_status": "CONSTITUENT_PARTICIPATION_INFORMATION_LOCAL_FEASIBILITY_PROPOSAL_NOT_ADMITTED",
        "next_candidate_field_status": "NO_NEW_ADMITTED_FINANCIAL_POLICY_AFTER_VOLUME_PRICE_REJECTION",
        "next_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "next_strategy_increment_status": "NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY", "current_admitted_unrun_numeric_candidates": 0,
        "latest_technical_decision": "TECH.R189", "latest_actual_model_decision": "TECH.R189",
        "latest_financial_strategy_decision": "TECH.R189", "latest_registration_decision": "TECH.R188",
        "latest_financial_registration_decision": "TECH.R188", "latest_financial_strategy_result": rel+"/summary.json",
        "latest_actual_prediction_model_decision": "TECH.R158", "latest_original_exit_decision": "TECH.R145",
        "new_account_return_sharpe_result": rel+"/summary.json", "new_account_return_sharpe": financial["status"],
        "latest_actual_model_kind": "COMPLETE_RULE_POLICY_WITHOUT_ESTIMATED_PREDICTION_MODEL",
        "latest_continuation_receipt": rel+"/delivery_receipt.json", "latest_volume_price_explanation": expl_rel+"/summary.json",
        "latest_volume_price_saved_verification": rel+"/saved_result_verification.json",
        "latest_volume_price_actual_points": rel+"/results/全部实际进出点位_准备确认及真实资金.csv",
        "latest_all_population_capital_identity": rel+"/results/四场景完成周期真实资金恒等式.csv",
        "latest_new_information_admission_boundary": rel+"/next_information_admission_boundary.json",
        "economic_stage_status": "FIXED_VOLUME_PRICE_SEQUENCE_REJECTED_ZERO_OF_FOUR_GATES",
        "current_economic_stage_status": "COMPLETED_FIXED_FULL_ACCOUNT_GATE_FAILED",
        "current_study_prediction_gate_status": "NOT_APPLICABLE_COMPLETE_RULE_POLICY_NO_ESTIMATED_PREDICTION_MODEL",
        "current_prediction_stage_status": "NOT_APPLICABLE_RULE_POLICY_ORIGINAL_PREDICTION_R158_PRESERVED",
        "current_full_account_economic_gate_passed": False, "current_historical_stability_gate_passed": False,
        "current_all_four_actual_point_quality_passed": False, "current_individual_economic_gate_passes": 0,
        "current_point_quality_evidence_role": "RECENT_THREE_COMPLETE_CYCLES_PB_POSITIVE_EARLY_NEGATIVE_NOT_PROMOTION",
        "returns_and_sharpe_improved": False, "return_and_sharpe_improved_this_continuation": False,
        "return_and_sharpe_changed_this_continuation": True, "new_financial_result_computed_this_continuation": True,
        "actual_candidate_trials": accounting, "current_candidate_trials": accounting, "current_phase_trial_accounting": accounting,
        "current_goal_turn_actual_work": accounting, "actual_candidate_trials_role": "LATEST_FINANCIAL_R189_TRIAL_ACCOUNTING",
        "current_executed_unique_new_entry_events": accounting["actual_cycles_per_cost"], "current_structural_signal_events": 44,
        "current_trade_map_cycles": accounting["actual_cycles_per_cost"], "current_trade_map_complete_cycles": accounting["completed_per_cost"],
        "current_trade_map_open_cycles": accounting["open_per_cost"], "current_trade_map_scope": "ONE_COST_FULL_POPULATION_SAVED_R189_PRIMARY_ACCOUNTS",
        "current_new_policy_complete_cycles": accounting["completed_per_cost"], "current_new_policy_open_cycles": accounting["open_per_cost"],
        "current_new_policy_unexecuted_origins": 44-accounting["actual_cycles_per_cost"],
        "current_information_mechanisms_admitted_this_continuation": 1, "current_fields_admitted_this_continuation": 0,
        "current_member_support_this_continuation": "全3488原点、434高点锚、151准备/53全历史确认、原61/49段、全部失败，0新预测成员。",
        "current_phase_required_directions": ["LONG"], "current_phase_source_freeze_count": accounting["financial_frozen_sources"],
        "current_phase_known_daily_rows": 3488, "current_phase_original_state_rows": 3488,
        "current_phase_information_scope": "CUMULATIVE_ACTUAL_SIGNED_VOLUME_LEADS_THEN_SAME_CONFIRMED_PRICE_HIGH_WITH_FIXED_KNOWN_FAILURE",
        "current_phase_definition_browsing": "TRADINGVIEW_OFFICIAL_OBV_BASIC_FORMULA_PROJECT_CASH_DIRECTION_AND_COMPLETE_PURPOSE",
        "necessary_tests_passed_this_continuation": 12, "necessary_tests_passed_in_current_phase": 12,
        "necessary_test_executions_this_continuation": 12, "actual_prefix_checks_this_continuation": 4,
        "new_accounts_this_continuation": 4, "new_accounts_in_current_phase": 4, "new_primary_accounts_this_continuation": 4,
        "new_financial_candidate_accounts_this_continuation": 4, "new_strategy_accounts_this_continuation": 4,
        "saved_account_controls_replayed_this_continuation": 8, "internal_reference_replays_this_continuation": 8,
        "saved_accounts_checked_this_continuation": 12, "new_strategy_configurations_this_continuation": 1,
        "new_model_fits_this_continuation": 0, "new_training_labels_this_continuation": 0, "new_return_labels_this_continuation": 0,
        "new_market_requests_this_continuation": 0, "original_strategy_source_files_changed_this_continuation": 0,
        "new_allocation_method_admitted": False, "new_method_admitted": "ONE_FIXED_VOLUME_PRICE_SEQUENCE_EXECUTED_AND_REJECTED",
        "new_market_information_admitted": False,
        "code_files_added_this_continuation": ["research/volume_lead_price_confirm_inputs_v1.py", "research/volume_lead_price_confirm_explanation_v1.py",
            "research/volume_lead_price_confirm_account_v1.py", "research/volume_lead_price_confirm_study_v1.py",
            "research/deliver_volume_lead_price_confirm_results_v1.py", "research/finalize_volume_lead_price_confirm_state_v1.py",
            "tests/test_volume_lead_price_confirm_inputs_v1.py", "tests/test_volume_lead_price_confirm_account_v1.py"],
        "pre_result_input_test_fixture_repairs": 0, "pre_registration_new_source_representation_fixes": 0,
        "current_numeric_leads_role": "EXISTING_FORWARD_REGISTRY_ONLY_NOT_R189_CANDIDATES",
        "current_goal_turn_classification": "progress", "goal_turn_progress_classification": "NEW_VOLUME_PRICE_PHASE_EXPLANATION_AND_FOUR_ACTUAL_FINANCIAL_RESULTS",
        "current_goal_turn_classification_reason": "部分慢指标前量价准备成立，四账户低频、弱于A、早期点位期望负；固定完整用途实际拒绝关闭。",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0, "blocked_reason": None, "blocked_audit_key": None,
        "blocking_decision": None, "goal_status": "active", "goal_tool_status_confirmed": "active", "goal_achieved": False,
        "whole_model_overfitting_removed": False, "overfitting_removed": False, "overfit_removed": False,
        "independent_validation_status": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "current_unmet_evidence": "R189四场景收益夏普低于A，1.4/0.5完整年均次数减少，早期pB<1、近期pB>1只3完成；独立和去过拟合未建立，待跑金融0。",
        "validation_method_this_continuation": "12必要测试首轮、4真实前缀、8原控制精确复现、4主账户一次与保存核对，全部88资格/24周期/20退出/4资金恒等式。",
        "latest_long_point_metrics": {"scope": "R189_COMPLETE_POLICY_DEVELOPMENT_NOT_CURRENT_MARKET",
            "STRESS": pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet").query("policy == @PRIMARY and cost == 'STRESS'").to_dict("records")},
    })
    require({key: state[key] for key in FORWARD} == forward_before and state["next_experiment"] == next_forward_before,
            "原真实前瞻或1008实际交易日计划改变。")
    require(digest(STATE) == state_before_hash, "当前状态被同时修改，不覆盖。")
    write_json(STATE, state)
    write_json(OUT / "project_state_update_receipt.json", {
        "at": now(), "status": "PASS_FOUR_DURABLE_FACT_FILES_AND_CURRENT_STATE_UPDATED_ONCE",
        "latest_technical_and_financial": "TECH.R189", "registration": "TECH.R188",
        "actual_prediction_preserved": "TECH.R158", "original_exit_preserved": "TECH.R145",
        "goal_status": "active", "goal_achieved": False, "new_admitted_unrun_candidates": 0,
        "forward_values_unchanged": forward_before, "existing_forward_next_experiment_unchanged": next_forward_before,
        "new_primary_accounts": 4, "original_strategy_source_changes": 0, "new_fits": 0,
        "whole_model_overfitting_removed": False, "next_local_feasibility_only": next_path.relative_to(ROOT).as_posix(),
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                    for p in [Path(__file__), STATE, next_path, *(ROOT / "docs" / name for name in DOCS)]],
    }, exclusive=True)
    print("四长期事实及当前状态一次更新至R189，原预测/退出/前瞻保持；目标active，固定用途关闭，待跑金融0。", flush=True)


if __name__ == "__main__":
    run()
