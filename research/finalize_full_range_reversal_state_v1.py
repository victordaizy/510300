"""一次写入R179—R181的四份长期事实，原策略及真实前瞻保持。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.full_range_reversal_study_v1 import OUT, EXPLANATION, PRIMARY
from research.finalize_downtrend_break_state_v1 import DOCS, FORWARD
from research.point_first_passage_study_v1 import read, write_json, digest, now, require

STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def run():
    require(not (OUT / "project_state_update_receipt.json").exists(), "R181长期事实已经更新。")
    financial, explanation, delivery = read(OUT / "summary.json"), read(EXPLANATION / "summary.json"), read(OUT / "delivery_receipt.json")
    require(financial["technical_decision"] == "TECH.R181" and financial["new_primary_accounts"] == 4,
            "四账户尚未完成。")
    require(financial["status"] == "REJECTED_FIXED_FULL_RANGE_REVERSAL_FULL_ACCOUNT_GATE_FAILED",
            "本次状态文本只对应实际拒绝裁决。")
    require(delivery["status"] == "PASS_ALL_RANGE_REVERSAL_POINTS_CASES_EXITS_AND_SAVED_IDENTITIES_DELIVERED",
            "全部实际资格和周期尚未交付。")
    for path in (OUT / "protocol.json", EXPLANATION / "protocol.json"):
        for source in read(path)["sources"]:
            require(digest(ROOT / source["path"]) == source["sha256"], "冻结来源改变。")
    state = read(STATE)
    require(state["latest_technical_decision"] == "TECH.R178" and state["latest_financial_strategy_decision"] == "TECH.R177",
            "当前状态不是接手时的R178/R177，不能覆盖另一阶段。")
    forward, next_experiment = {key: state[key] for key in FORWARD}, state["next_experiment"]
    texts = {name: (ROOT / "docs" / name).read_text(encoding="utf-8-sig") for name in DOCS}
    marker = "> 技术线当前事实 TECH.R181（优先于下方技术线历史快照，2026-10-05）："
    require(all(marker not in text for text in texts.values()), "四事实文件已经有本期记录。")
    write_json(OUT / "project_state_before.json", state, exclusive=True)
    metric_rows = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    primary = metric_rows.loc[metric_rows.policy.eq(PRIMARY)]
    early = primary.loc[primary.period.eq("2015_2019") & primary.cost.eq("STRESS")].iloc[0]
    recent = primary.loc[primary.period.eq("2020_2026") & primary.cost.eq("STRESS")].iloc[0]
    financial_sources = len(read(OUT / "protocol.json")["sources"])
    out_rel, expl_rel = OUT.relative_to(ROOT).as_posix(), EXPLANATION.relative_to(ROOT).as_posix()
    next_action = read(OUT / "next_all_reversal_lifecycle_diagnostic_proposal.json")["next_required_input"]
    overview = (
        "先具体量价解释再反推点位完成TECH.R179解释、R180登记、R181四实际账户。"
        "LOW<昨LOW且CLOSE>昨HIGH收盘后确认、次开一次，初始线为事件LOW；开盘到/低于线取消，"
        "新同向事件只抬事件低点，已知收盘跌线或HIGH>昨HIGH且CLOSE<昨LOW的相反形态次合法开退出；未知不强退、风险只减。"
        "全3488日/2855当期原点、98同向/71相反事件、98路径失效（42线上相反事件）、原61/49段及四例保留。"
        "2019-01-04量1.623倍、日周MACD仍负，实际压力01-07至01-18净+1.94%；02-25至04-23净+11.52%。"
        "2020-03-10放量局部强反转仍处下跌途中，实际03-11至03-16净−5.04%；05-07至05-25−2.98%。"
        "2024九月急涨没有该形态，10-18才到达，10-21至11-18净+0.78%；2015-07-10至08-19净−3.16%。"
        f"压力早29完成/1开放、胜率{early.win_rate:.2%}/B{early.payoff:.6f}/pB{early.p_times_b:.6f}，"
        f"净年化{early.net_cagr:.6%}/夏普{early.net_sharpe:.6f}/DD{early.max_drawdown:.4%}；"
        f"近期41完成、{recent.win_rate:.2%}/{recent.payoff:.6f}/{recent.p_times_b:.6f}、"
        f"{recent.net_cagr:.6%}/{recent.net_sharpe:.6f}/DD{recent.max_drawdown:.4%}。"
        "早期两费用夏普高于A但收益低A，早STRESS pB不足1；近期收益/夏普/标准期望均负，实际份额毛损益也负。"
        "完整年均次数5.8/6.0高于A4.4/4.1667，但四经济门0通过、整体及历史稳定门全败，固定完整政策拒绝关闭不营救。"
        "196资格/142周期（140完成/2开放）、27每费用已有持仓不新建，14案例周期行及140退出、四资金恒等式全交付。"
        f"九必要测试/四真实前缀、八A/纯价格控制精确复现、四主账户一次测量及保存核对通过；33解释/{financial_sources}金融来源。"
        "首次测试例除息相等算术差.001只修正测试、逻辑不改，首次失败和旧测试保存，0拟合/新训练标签/采集。"
        "历史DEVELOPMENT_CALIBRATION、first-vintage NOT_CERTIFIED、独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED，"
        "去过拟合与完整收益夏普目标未达，目标active/本轮progress/阻塞0。最新技术/金融R181、登记R180，预测R158/退出R145及所有旧终态保持。"
        "下一全体保存生命周期、已知位置、资金/次开时钟与原49上涨无形态覆盖诊断，0新账户，不挑分组加过滤；新金融未登记/待跑0。十二真实前瞻与原1008实际交易日计划、其他分支保持。")
    top = marker+(
        "R179具体量价解释/R180登记/R181四账户完成：98同向/71相反整日收盘反转，196资格/142周期全保存。"
        f"压力早净年化{early.net_cagr:.4%}/夏普{early.net_sharpe:.4f}/pB{early.p_times_b:.4f}，"
        f"近期{recent.net_cagr:.4%}/{recent.net_sharpe:.4f}/{recent.p_times_b:.4f}。"
        "早夏普单项高于A但收益低、压力pB不足1；近期实际毛损益负。次数5.8/6.0增加，四经济门与稳定门全败，固定政策拒绝关闭。"
        "2019价格先转强而MACD滞后、2020局部反转后续失败、2024急涨无该形态均解释；九测试/八原控制/四保存账户核对通过。"
        "下一只全体保存位置和资金时钟归因，待跑金融0；目标未达且active，原策略/旧失败/真实前瞻保持。"
        f" [具体解释与全部结果](../{out_rel}/研究结果与下一步.md)。\n\n")
    decisions = """
## TECH.R179：具体上涨与失败的整日收盘跨区间解释（2026-10-05）

**假设**：同日覆盖昨低后收盘越过昨高能描述局部转强，日周MACD、量和RV解释当时位置；并非所有上涨先给该形态，也不保证更大尺度下降已结束。

**验证方法**：整数.001现金平移LOW<昨LOW且CLOSE>昨HIGH，同向出生；HIGH>昨HIGH且CLOSE<昨LOW为相反形态。全3488原点、2015起2855、全部98同向/71相反及全部失效/开放、原61/49段及原四案例展开；四输入测试、四真实前缀。完整未来用途在新结果连接前固定，原2846标签/2826成熟/20删失/9尾部只事后对齐。有限核对旧V5二十日低点收复、供给测试、分钟FVG、顺序形态和R177缺口完整用途，不因猜测不存在的结果路径推断旧顺序形态NOT_RUN，不声称穷尽所有旧形态。

**结果**：全部98路径失效，其中42在事件低点线上出现相反事件。2019-01-04量1.623倍、日周MACD仍负，局部价格先到；2020-03-10量1.983倍和近零正日柱仍在后续下跌前出现，05-06修复中段同向而06-11相反。2024九月急涨没有该形态，10-18日周动量同正/RV比4.907时才出现；2015固定反弹窗后段07-09日周动量负/RV3.620的强反转仍不能证明持有盈利。四图已查看、33来源、0账户/拟合/新标签/采集。

**为什么接受/拒绝**：接受具体可知时钟与阶段解释，拒绝把局部转强等同于所有上涨启动或真正主动订单流/因果。日线不能得知HIGH/LOW先后及撤单/主动买卖；旧冻结失败保持。完整账户优劣另见R180—R181。

**是否需要重新验证**：解释一次完成，只有真实来源/实现错误才复核；不挑2019赢家加量/MACD过滤。首次输入测试例9.978+.033=10.011而非预期10.010，只改测试CLOSE为9.977，原逻辑未改，首次失败和测试原件保留，主账户结果读取0。

## TECH.R180：唯一整日反转完整用途登记（2026-10-05）

**假设**：上一整日区间两端跨越和对称相反失效提供独立完整入出用途，可能增加实际交易机会并改善扣费收益夏普；不是只把旧收复窗口20改1，也不营救R177。

**验证方法**：空仓同向出生次开一次；初始线为出生当日LOW，开盘到/低于线或坐标未知取消。持有新同向事件只抬LOW，已知收盘跌线或相反整日反转次合法开退出，同时优先收盘线失败；未知不强退、原风险只减，无加仓/混A/固定获利或时间退出/量或MACD过滤。20万元/252日、完整现金日、股息、T+1、100份/.001、最低佣金5元、原ES/跳空/DD预算，两时期两费用；原risk_ok在BASE也按STRESS预算检查保持。全部四场景净收益夏普须同时高于0、A及纯价格，DD<=10%、实际pB>1/pB−q>0；20/252固定配对区块各2000/种子510300154，所有差值95%下界正才过历史稳定。

**结果**：四输入与五必要账户行为测试通过，八原A/纯价格日账/周期/订单精确复现，R179全部状态一致，仅准入1完整配置/4实际账户的开发测量。

**为什么接受/拒绝**：接受一次可比较测量，不接受已证实高胜率、高夏普或独立样本。全历史已使用，形态选择偏差没有被固定规则消除；原退出MSE门不扩展到该完整规则用途。

**是否需要重新验证**：不据结果改跨越条件、参考日/窗口、初始线/反向退出、预算/费用/时期，不加指标过滤或混A；实际裁决见R181。

## TECH.R181：早期夏普单项提高，完整整日反转用途拒绝关闭（2026-10-05）

**假设**：完整政策必须在两时期两费用同时改善收益夏普及实际点位质量，增加次数和单个好上涨不能代替完整目标。

**验证方法**：4主账户一次运行，8原控制复用，4保存账户/资金/库存/T+1/股息/成本/指标核对；196资格、142周期（140完成/2开放）、140退出、14案例周期行及四实际资金恒等式全部交付。每费用98资格=71周期（70完成/1开放）+27已有持仓；预首日和期末请求保留，开放案例只观察到自身账户末日。

**结果**：早BASE29完成/1开放，净年化2.1142%/夏普0.5694、胜率41.38%/B2.6111/pB1.0805/DD6.02%；早STRESS1.7124%/0.4971、41.38%/2.3504/0.9726/DD6.58%。近BASE41完成、−1.4898%/−0.9162、29.27%/0.7316/0.2141/DD9.28%；近STRESS−1.5164%/−0.9765、29.27%/0.6473/0.1894/DD9.44%。早夏普两档高于A，但收益两档低A、压力pB不足1；近期标准期望亦负。四单场景经济门0通过、整体与历史稳定门全败，两期均未风险停机。完整年均次数5.8/6.0高于A4.4/4.1667，不构成更好模型。

压力2019-01-07至01-18净+1.94%，02-25至04-23+11.52%；2020-03-11至03-16−5.04%、05-07至05-25−2.98%；2024-10-21至11-18+0.78%，无法捕捉9月启动；2015-07-10至08-19−3.16%。全部相交周期包含窗口前已有交易，不能只列出生在窗内的赢家。近期实际份额毛损益−15764.60元，佣金966.71元/滑点2156.70元，最终净−18888.01元；不单是费用，不能据此放宽预算。

**为什么接受/拒绝**：接受早期局部转强及某些实际盈利点、早夏普单项改善和次数增加的开发事实；拒绝完整政策作为全账户模型提升，固定用途关闭。不得只选早期、2019赢家或低费用，也不从已见结果加MACD/量/RV、改失效/窗口或混A营救。独立NOT_ESTABLISHED、first-vintage NOT_CERTIFIED、global DSR/PBO NOT_COMPUTED，去过拟合及收益夏普目标未达，目标active；原A/旧失败和原真实前瞻保持。

**是否需要重新验证/下一具体实验**：本金融用途一次完成，不改参重跑。下一对全98出生及全部保存周期作当时可知位置、只抬事件低点、相反退出/收盘失效、实际资金和次开差额的归因，同时保留原49上涨无形态覆盖，0新账户。不把最高账面利润当可实现退出或把分组转过滤。之后只有实质不同信息及完整用途、真实新样本或真实来源/实现错误进一步准入；新金融未定义/登记、待跑0。原A/POINT十二真实前瞻值及1008实际交易日计划不变，不自动将该历史形态加入前瞻或实盘。
"""
    links = (f"\n\n依据：[具体上涨解释](../{expl_rel}/具体上涨与点位反推.md)、[全部结果](../{out_rel}/研究结果与下一步.md)、"
             f"[冻结](../{out_rel}/protocol.json)、[全部区间](../{out_rel}/summary.json)、"
             f"[全部真实点位](../{out_rel}/results/全部实际进出点位_已知整日反转量价及真实资金.csv)、"
             f"[保存核对](../{out_rel}/saved_result_verification.json)、[下一全体归因](../{out_rel}/next_all_reversal_lifecycle_diagnostic_proposal.json)。\n")
    for name, text in texts.items():
        first, remainder = text.split("\n", 1)
        appended = decisions if name != "PROJECT_STATE.md" else "\n## 技术线TECH.R179—R181当前更新（2026-10-05）\n\n"+overview
        (ROOT / "docs" / name).write_text(first+"\n\n"+top+remainder.lstrip("\n")+"\n"+appended+links,
                                           encoding="utf-8", newline="\n")
    accounting = {
        "scope": "TECH_R179_R180_R181_ONE_FIXED_FULL_RANGE_REVERSAL_COMPLETE_POLICY",
        "candidate_configurations": 1, "new_policy_accounts": 4, "primary_accounts": 4,
        "saved_original_A_replays": 4, "saved_price_control_replays": 4, "reused_control_scenarios": 8,
        "all_known_daily_rows": 3488, "origins_since_2015": 2855,
        "bullish_reversal_events": 98, "bearish_reversal_events_observation_and_long_exit_only": 71,
        "all_qualification_rows_two_costs": 196, "all_actual_cycle_rows_two_costs": 142,
        "actual_cycles_per_cost": 71, "completed_per_cost": 70, "open_per_cost": 1,
        "already_holding_no_new_cycle_per_cost": 27, "actual_complete_exit_rows_two_costs": 140,
        "original_episode_rows": 61, "original_admitted_waves": 49, "case_cycle_rows_two_costs": delivery["case_cycle_rows"],
        "necessary_tests_final_passed": 9, "actual_necessary_test_executions_including_repaired_fixture": 13,
        "pre_result_input_fixture_failures_preserved": 1, "actual_prefix_checks": 4, "charts": 4,
        "explanation_frozen_sources": explanation["frozen_sources"], "financial_frozen_sources": financial_sources,
        "capital_identity_cells": 4, "all_four_point_quality_passed": False,
        "single_scenario_economic_passes": delivery["individual_economic_gate_passes"],
        "overall_economic_gate_passed": False, "historical_stability_gate_passed": False,
        "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0, "risk_budget_changes": 0,
    }
    legacy = {key: value for key, value in state.items() if key.startswith("current_phase_")}
    state["archived_previous_current_phase_fields_before_R181"] = {"at": now(), "fields": legacy}
    state["archived_financial_trial_accounting_R177_before_R181"] = state["actual_candidate_trials"]
    for key in legacy:
        del state[key]
    state.update({
        "at": now(), "updated_at": now(), "latest_completed_study": out_rel, "latest_result": out_rel+"/summary.json",
        "latest_report": out_rel+"/研究结果与下一步.md", "latest_research_report": out_rel+"/研究结果与下一步.md",
        "latest_research_status": financial["status"], "latest_progress": overview, "latest_overall_summary": overview,
        "latest_continuation_outcome": overview, "current_study": "510300_FULL_RANGE_REVERSAL_STUDY_V1",
        "current_phase": "FULL_RANGE_REVERSAL_FREQUENCY_AND_EARLY_SHARPE_INCREASED_FULL_POLICY_REJECTED",
        "current_direction": "先解释具体上涨/失败量价，再完整整日收盘反转用途，早夏普单项正而跨期经济拒绝。",
        "current_priority": next_action, "next_research_action": next_action, "next_available_action": next_action,
        "next_research_question": next_action, "next_research_plan": out_rel+"/next_all_reversal_lifecycle_diagnostic_proposal.json",
        "next_experiment_status": "ALL_SAVED_REVERSAL_LIFECYCLE_DIAGNOSTIC_PROPOSED_NO_NEW_POLICY",
        "next_candidate_field_status": "NO_NEW_ADMITTED_FINANCIAL_POLICY_AFTER_RANGE_REVERSAL_REJECTION",
        "next_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "next_strategy_increment_status": "NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY", "current_admitted_unrun_numeric_candidates": 0,
        "latest_technical_decision": "TECH.R181", "latest_actual_model_decision": "TECH.R181",
        "latest_financial_strategy_decision": "TECH.R181", "latest_registration_decision": "TECH.R180",
        "latest_financial_registration_decision": "TECH.R180", "latest_financial_strategy_result": out_rel+"/summary.json",
        "new_account_return_sharpe_result": out_rel+"/summary.json", "new_account_return_sharpe": financial["status"],
        "latest_actual_model_kind": "COMPLETE_RULE_POLICY_WITHOUT_ESTIMATED_PREDICTION_MODEL",
        "latest_actual_prediction_model_decision": "TECH.R158", "latest_original_exit_decision": "TECH.R145",
        "latest_continuation_receipt": out_rel+"/delivery_receipt.json",
        "latest_full_range_reversal_explanation": expl_rel+"/summary.json",
        "latest_full_range_reversal_protocol": out_rel+"/protocol.json",
        "latest_full_range_reversal_saved_verification": out_rel+"/saved_result_verification.json",
        "latest_full_range_reversal_actual_points": out_rel+"/results/全部实际进出点位_已知整日反转量价及真实资金.csv",
        "latest_all_population_capital_identity": out_rel+"/results/全部四场景完成周期_实际资金恒等式.csv",
        "latest_new_information_admission_boundary": out_rel+"/next_all_reversal_lifecycle_diagnostic_proposal.json",
        "economic_stage_status": "FIXED_RANGE_REVERSAL_REJECTED_ZERO_OF_FOUR_SCENARIO_GATES_PASSED",
        "current_economic_stage_status": "COMPLETED_FIXED_FULL_ACCOUNT_GATE_FAILED",
        "current_study_prediction_gate_status": "NOT_APPLICABLE_COMPLETE_RULE_POLICY_NO_ESTIMATED_PREDICTION_MODEL",
        "current_prediction_stage_status": "NOT_APPLICABLE_RULE_POLICY_ORIGINAL_PREDICTION_R158_PRESERVED",
        "current_full_account_economic_gate_passed": False, "current_historical_stability_gate_passed": False,
        "current_all_four_actual_point_quality_passed": False,
        "current_point_quality_evidence_role": "EARLY_BASE_PB_POSITIVE_PRESSURE_BELOW_ONE_AND_RECENT_NEGATIVE_NOT_PROMOTION",
        "current_individual_economic_gate_passes": 0, "returns_and_sharpe_improved": False,
        "return_and_sharpe_improved_this_continuation": False, "return_and_sharpe_changed_this_continuation": True,
        "new_financial_result_computed_this_continuation": True, "actual_candidate_trials": accounting,
        "current_candidate_trials": accounting, "current_phase_trial_accounting": accounting,
        "current_goal_turn_actual_work": accounting, "actual_candidate_trials_role": "LATEST_FINANCIAL_R181_TRIAL_ACCOUNTING",
        "current_executed_unique_new_entry_events": 71, "current_structural_signal_events": 98,
        "current_trade_map_cycles": 71, "current_trade_map_complete_cycles": 70, "current_trade_map_open_cycles": 1,
        "current_trade_map_scope": "ONE_COST_FULL_POPULATION_SAVED_R181_PRIMARY_ACCOUNTS",
        "current_new_policy_complete_cycles": 70, "current_new_policy_open_cycles": 1, "current_new_policy_unexecuted_origins": 27,
        "current_fields_admitted_this_continuation": 0, "current_information_mechanisms_admitted_this_continuation": 1,
        "current_member_support_this_continuation": "全3488原报价/2855当期/98同向整日反转；全部实际周期和失败，旧标签仅回顾，0新预测成员。",
        "current_phase_required_directions": ["LONG"], "current_phase_source_freeze_count": financial_sources,
        "current_phase_known_daily_rows": 3488, "current_phase_original_state_rows": 3488,
        "current_phase_information_scope": "PREVIOUS_COMPLETE_DAILY_RANGE_TWO_ENDPOINT_CLOSE_REVERSAL_AND_MIRRORED_FAILURE",
        "current_phase_definition_browsing": "NONE_LOCAL_FROZEN_QUOTES_AND_FINITE_COMPLETE_OLD_USES",
        "necessary_tests_passed_this_continuation": 9, "necessary_tests_passed_in_current_phase": 9,
        "necessary_test_executions_this_continuation": 13, "actual_prefix_checks_this_continuation": 4,
        "new_accounts_this_continuation": 4, "new_accounts_in_current_phase": 4,
        "new_primary_accounts_this_continuation": 4, "new_financial_candidate_accounts_this_continuation": 4,
        "new_investment_account_evaluations_this_continuation": 4, "new_strategy_accounts_this_continuation": 4,
        "internal_reference_replays_this_continuation": 8, "saved_account_controls_replayed_this_continuation": 8,
        "saved_accounts_checked_this_continuation": 12, "new_strategy_configurations_this_continuation": 1,
        "new_model_fits_this_continuation": 0, "historical_model_refits_this_continuation": 0,
        "new_return_labels_this_continuation": 0, "new_training_labels_this_continuation": 0,
        "new_model_training_labels_this_continuation": 0, "new_allocation_method_admitted": False,
        "new_method_admitted": "ONE_FIXED_FULL_RANGE_REVERSAL_COMPLETE_POLICY_EXECUTED_AND_REJECTED",
        "new_market_information_admitted": False, "new_market_requests_this_continuation": 0,
        "original_strategy_source_files_changed_this_continuation": 0,
        "code_files_added_this_continuation": ["research/full_range_reversal_inputs_v1.py", "research/full_range_reversal_explanation_v1.py",
            "research/full_range_reversal_account_v1.py", "research/full_range_reversal_study_v1.py",
            "research/deliver_full_range_reversal_results_v1.py", "research/finalize_full_range_reversal_state_v1.py",
            "tests/test_full_range_reversal_inputs_v1.py", "tests/test_full_range_reversal_account_v1.py"],
        "pre_result_input_test_fixture_repairs": 1, "pre_registration_new_source_representation_fixes": 0,
        "initial_source_backup_timing_limit_role": "R177_HISTORICAL_EVIDENCE_PRESERVED_NOT_CURRENT_R181_FAILURE",
        "current_numeric_leads_role": "EXISTING_FORWARD_REGISTRY_ONLY_NOT_R181_CANDIDATES",
        "current_goal_turn_classification": "progress", "previous_goal_turn_classification": "progress",
        "goal_turn_progress_classification": "NEW_CONCRETE_RANGE_REVERSAL_CASES_AND_FOUR_ACTUAL_ACCOUNT_RESULTS",
        "current_goal_turn_classification_reason": "新完整用途检验改变证据：2019较早转强、2020失败/2024急涨无形态，早夏普单项增加但收益及跨期质量拒绝，未以好案例推广。",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0, "blocked_reason": None,
        "blocked_audit_key": None, "blocking_decision": None, "goal_status": "active", "goal_tool_status_confirmed": "active",
        "goal_achieved": False, "whole_model_overfitting_removed": False, "overfitting_removed": False, "overfit_removed": False,
        "independent_validation_status": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "current_unmet_evidence": "R181次数和早夏普单项增加但早收益低A/压力pB不足1、近期毛损益及经济期望负，四经济/稳定门败；独立和去过拟合未建立，新待跑金融0。",
        "validation_method_this_continuation": "九必要测试/四前缀/八原控制精确复现/四主账户一次与保存核对，196资格/142周期/140退出/14案例行/四资金恒等式全交付；首次测试例失败保留。",
        "latest_long_point_metrics": {"scope": "R181_COMPLETE_POLICY_DEVELOPMENT_NOT_CURRENT_MARKET", "STRESS": primary.loc[primary.cost.eq("STRESS")].to_dict("records")},
    })
    require({key: state[key] for key in FORWARD} == forward and state["next_experiment"] == next_experiment,
            "原十二真实前瞻或原1008实际交易日计划改变。")
    write_json(STATE, state)
    write_json(OUT / "project_state_update_receipt.json", {
        "at": now(), "status": "PASS_FOUR_DURABLE_FACT_FILES_AND_CURRENT_STATE_UPDATED_ONCE",
        "latest_technical_and_financial": "TECH.R181", "financial_registration": "TECH.R180",
        "actual_prediction_preserved": "TECH.R158", "original_exit_preserved": "TECH.R145",
        "goal_status": "active", "goal_achieved": False, "admitted_unrun_candidates": 0,
        "forward_values_unchanged": forward, "existing_forward_next_experiment_unchanged": next_experiment,
        "overall_full_account_gate_passed": False, "historical_stability_gate_passed": False,
        "original_strategy_source_changes": 0, "new_primary_accounts": 4, "control_replays": 8,
        "legacy_current_phase_fields_archived": len(legacy),
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                    for p in [Path(__file__), STATE, *(ROOT / "docs" / name for name in DOCS)]],
    }, exclusive=True)
    print("四长期事实与状态一次更新至R181；原策略/前瞻保持，完整目标未达且active。", flush=True)


if __name__ == "__main__":
    run()
