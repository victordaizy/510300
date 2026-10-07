"""一次更新R183—R185长期事实，保留旧冻结策略、其他分支和原真实前瞻。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.down_gap_reclaim_study_v1 import OUT, EXPLANATION, PRIMARY
from research.finalize_downtrend_break_state_v1 import DOCS, FORWARD
from research.point_first_passage_study_v1 import read, write_json, digest, now, require

STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def run():
    require(not (OUT / "project_state_update_receipt.json").exists(), "R185长期事实已更新，不重复。")
    financial, explanation, delivery = read(OUT / "summary.json"), read(EXPLANATION / "summary.json"), read(OUT / "delivery_receipt.json")
    require(financial["status"] == "REJECTED_FIXED_DOWN_GAP_RECLAIM_FULL_ACCOUNT_GATE_FAILED" and financial["new_primary_accounts"] == 4, "本期状态文本只对应实际四账户拒绝。")
    require(delivery["status"] == "PASS_ALL_RECLAIM_POINTS_CASES_EXITS_AND_SAVED_IDENTITIES_DELIVERED", "全部实际点位尚未交付。")
    for protocol in (OUT / "protocol.json", EXPLANATION / "protocol.json"):
        for item in read(protocol)["sources"]:
            require(digest(ROOT / item["path"]) == item["sha256"], "本期冻结来源改变。")
    state_hash = digest(STATE)
    state = read(STATE)
    require(state["latest_technical_decision"] == "TECH.R182" and state["latest_financial_strategy_decision"] == "TECH.R181", "当前阶段已被其他工作更新，不覆盖。")
    original_forward, original_next = {key: state[key] for key in FORWARD}, state["next_experiment"]
    texts = {name: (ROOT / "docs" / name).read_text(encoding="utf-8-sig") for name in DOCS}
    original_doc_hashes = {name: digest(ROOT / "docs" / name) for name in DOCS}
    marker = "> 技术线当前事实 TECH.R185（优先于下方技术线历史快照，2026-10-05）："
    require(all(marker not in text for text in texts.values()), "长期事实已有本期记录。")
    write_json(OUT / "project_state_before.json", state, exclusive=True)
    all_metrics = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    primary = all_metrics.loc[all_metrics.policy.eq(PRIMARY)]
    early = primary.loc[primary.period.eq("2015_2019") & primary.cost.eq("STRESS")].iloc[0]
    recent = primary.loc[primary.period.eq("2020_2026") & primary.cost.eq("STRESS")].iloc[0]
    pressure_counts = [row for row in delivery["scenario_counts"] if row["cost"] == "STRESS"]
    cycle_count = sum(row["actual_cycles"] for row in pressure_counts)
    completed = sum(row["completed"] for row in pressure_counts)
    opened = sum(row["open"] for row in pressure_counts)
    unused = sum(row["reclaim_origins"]-row["actual_cycles"] for row in pressure_counts)
    frozen_count = len(read(OUT / "protocol.json")["sources"])
    out_rel, expl_rel = OUT.relative_to(ROOT).as_posix(), EXPLANATION.relative_to(ROOT).as_posix()
    next_action = read(OUT / "next_all_reclaim_lifecycle_diagnostic_proposal.json")["next_required_input"]
    overview = (
        "按用户要求先解释具体上涨量价再反推点位，完成R183全体解释/R184登记/R185四真实账户。"
        "此前HIGH<昨LOW完整向下缺口取最新锚，新缺口替换未收复锚，出生以后首次CLOSE>上沿消耗锚；"
        "空仓次开一次，开盘<=原锚下沿或未知取消，无重试；持有固定原下沿不抬线，收盘<=线或新向下缺口次合法开出，"
        "同时优先收盘失败，未知自身不出/原风险只减，无加仓/混A/获利或时间退出/量或MACD过滤。"
        "全3488/2855原点，112原锚（2015起91），74收复/37被替换/1开放，2015起60首次收复。"
        "2019直到03-18才是回调收复（量0.975，日DIF/上一周柱正、日柱负），不能捕捉1月启动；"
        "2020-04-02日柱正而DIF/上一周柱负、量0.741，06-01另一修复；2024-09-24量3.338、"
        "日柱正但DIF/上一周柱负，09-25真实入、2025-04-08出净+5.65%。"
        "原49上涨19底峰间无收复/28确认峰间无收复；原61/49段及2846旧标签/2826成熟/20删失/9尾部保持。"
        f"压力早{int(early.completed_cycles)}完成/{int(early.unfinished_cycles)}开放、胜率{early.win_rate:.2%}/B{early.payoff:.6f}/pB{early.p_times_b:.6f}、"
        f"净年化{early.net_cagr:.6%}/夏普{early.net_sharpe:.6f}/DD{early.max_drawdown:.4%}；"
        f"近{int(recent.completed_cycles)}完成/{int(recent.unfinished_cycles)}开放、{recent.win_rate:.2%}/{recent.payoff:.6f}/{recent.p_times_b:.6f}、"
        f"{recent.net_cagr:.6%}/{recent.net_sharpe:.6f}/DD{recent.max_drawdown:.4%}。"
        "近两费用实际pB>1且标准期望正、收益夏普高于纯价格，但两时期两费用均低于原A；早期收益夏普与标准期望负。"
        f"四单场景经济门通过{delivery['individual_economic_gate_passes']}，整体和历史稳定门败，固定完整用途拒绝关闭，不加指标/改锚或退出营救。"
        f"全部{delivery['all_qualification_rows']}资格/{delivery['all_actual_cycle_rows']}周期（{delivery['all_completed_cycles']}完成/{delivery['all_open_cycles']}开放）、"
        f"每费用{cycle_count}实际/{completed}完成/{opened}开放/{unused}开盘取消，全部退出及14案例行、四资金恒等式交付。"
        f"九必要测试首轮通过/四真实前缀/八原A纯价格控制精确复现/四主账户一次与保存核对；"
        f"{explanation['frozen_sources']}解释/{frozen_count}金融来源，四图已查看，解释表格只重排保存行并留初稿，0拟合/新训练标签/采集。"
        "旧gap_recovery完整定义及实际失败已实核：同日低开收复昨C/C<=当日O退出与本用途不同，旧终态保持；有限旧核对不证明全球新颖。"
        "历史DEVELOPMENT_CALIBRATION、first-vintage NOT_CERTIFIED、独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED，"
        "去过拟合和完整收益夏普目标未达，目标active/本轮progress/阻塞0。最新技术金融R185/登记R184，预测R158/原退出R145及R181/R182旧终态保持。"
        "下一一次全体保存收复位置/资金/失效时钟及无收复上涨覆盖归因，0新账户；新准入待跑0，十二真实前瞻与原1008实际交易日计划保持。")
    top = marker+(
        "R183具体量价解释/R184登记/R185四账户完成：此前完整向下缺口最新锚首次收复，120资格/102周期全体保存。"
        "2024-09-24量3.338、日柱先正而DIF/上一周柱负，真实09-25至2025-04-08+5.65%；2019仅3月回调点、不是1月启动。"
        f"压力早净年化{early.net_cagr:.4%}/夏普{early.net_sharpe:.4f}/pB{early.p_times_b:.4f}，"
        f"近期{recent.net_cagr:.4%}/{recent.net_sharpe:.4f}/{recent.p_times_b:.4f}。"
        "近期两费用点位pB>1/标准期望正且高于纯价格，但原A收益夏普更高；早期负，四经济和稳定门失败，固定完整用途拒绝关闭。"
        "下一全体保存资金及失效位置归因，待跑金融0；完整目标未达且active，原策略、旧失败和真实前瞻保持。"
        f" [具体量价与实际结果](../{out_rel}/研究结果与下一步.md)。\n\n")
    decisions = (
        "\n\n## TECH.R183：具体上涨与此前完整向下缺口首次收复解释（2026-10-05）\n\n"
        "**假设**：此前日线完整向下区间被后续已知收盘否定，可能给出不同于同日低开收复/完整反转的修复信息；日周动量、量和RV解释当时阶段，不等同于主动订单流或上涨因果。\n\n"
        "**验证方法**：原.001整数现金平移HIGH<昨LOW出生，最新锚替换、首次CLOSE>上沿消耗；完整未来用途在连接任何新事件结果之前固定。所有3488日连续状态、所有原锚和替换/收复/开放、原61/49段、四案例及失败展开，四输入测试及四真实前缀；原2846标签/2826成熟/20删失/9尾部只事后解释。更宽搜索找到旧gap_recovery，核实同日OPEN+分红<昨C、CLOSE+分红>昨C进入及CLOSE<=当日OPEN退出，实际历史目标未达，旧失败保持；有限核对旧V5/供给测试/顺序形态/R177/R181，不声称穷尽全球。\n\n"
        "**结果**：全历史112原锚74收复/37被替换/1开放，2015起91出生、60首次收复（锚跨时期连续，出生数不代替交易数）。2019-03-18是趋势中回调修复而非1月启动；2020-04-02量0.741、日柱正而DIF/上一周柱负，6月另一个收复；2024-09-24收盘3.427收复9月9日原上沿3.300、量3.338、日柱正但DIF/上一周柱负，收盘后才知。2015-07-09在原短反弹之后、日周负/RV3.620。原49上涨19底峰间无收复、28确认峰间无收复，四图已查看。\n\n"
        "**为什么接受/拒绝**：接受具体修复阶段及可知时钟解释，拒绝称全部上涨必要启动、机构吸筹或因果。日周指标滞后不自动支持符号过滤；0新金融账户。\n\n"
        "**是否需要重新验证**：只有真实来源/实现错误才复核定义，不能按赢家改锚/严格端点/消耗或加量MACD。解释表格根据原保存行重排并补阶段含义，初稿保存，原冻结代码和结果未改。\n\n"
        "## TECH.R184：唯一向下整日缺口首次收复完整用途登记（2026-10-05）\n\n"
        "**假设**：完整已知入场及失效规则能否同时提高实际pB、交易机会、扣费收益和全日历夏普，须以原共同账户检验。\n\n"
        "**验证方法**：空仓首次收复次开一次，开盘现金价<=原锚下沿或坐标未知取消、无延迟追入。持有原下沿固定不抬线，CLOSE<=线或新完整向下缺口次合法开退出，两者同现优先收盘失败；未知自身不出、风险只减，无加仓/混A/目标或时间退出/期末清仓/指标过滤。原20万元/252日/现金0/50%/股息/ES和跳空预算/DD10%/T+1/100份/.001/最低佣金5，两时期两费用；BASE买入原STRESS风险检查保持。四场景净CAGR和净Sharpe均须>0/A/纯价格、DD<=10%、实际pB>1且pB−q>0；原20/252配对区块各2000、种子510300154，所有差值95%下界正才过历史稳定门。\n\n"
        "**结果**：四输入和五账户行为测试首轮通过，八A/纯价格日账/订单/周期精确复现；只准入1完整配置/4新主账户的一次开发测量。\n\n"
        "**为什么接受/拒绝**：接受可比较一次测量，不接受已证实高收益夏普或独立验证；旧MSE/115成员门限于原预测用途，本次没有拟合或新训练标签。\n\n"
        "**是否需要重新验证**：不按结果改严格端点、最新锚/消耗、固定线或新缺口退出、预算/费用/时期，不加量MACD/RV、不混A。实际裁决见R185。\n\n"
        "## TECH.R185：近期实际pB正、原A更强，完整收复用途拒绝关闭（2026-10-05）\n\n"
        "**假设**：局部或某一时期的有效修复点必须接受全体失败和原控制比较，不能因2024启动识别就宣称完整策略已提高。\n\n"
        "**验证方法**：四主账户一次测量、八保存控制复用、四保存账本/资金/库存/次开/T+1/股息/成本/指标核对；全部120资格/102周期（100完成/2开放）、100退出、14案例行及四资金恒等式交付，9开盘到/低于固定线的取消每费用保留、不补追入。\n\n"
        "**结果**：早BASE20完成/1开放、胜率25%/B1.9275/pB0.4819、净年化−0.0862%/夏普−0.0145/DD8.58%；早STRESS25%/1.7731/0.4433、−0.2675%/−0.0778/DD8.68%。近BASE30完成、33.33%/5.2984/1.7661、2.1648%/0.5366/DD5.47%；近STRESS33.33%/4.7540/1.5847、1.9145%/0.4821/DD5.32%，标准期望0.9180损失单位。近期两费用胜率低但实际B大、pB>1/标准期望正，收益夏普高于纯价格；原A近压力3.9908%/1.2169更高，早期两费用收益夏普/期望负。四单场景经济门0通过、整体与历史稳定门败；完整年均次数4.0/4.3333，A4.4/4.1667，不能称全面增加。\n\n"
        "实际压力2015-07-10至08-24−9.88%；2019-03-19至03-26−2.94%；2020-04-03至05-25+2.46%、06-02至08-21+16.45%；2024-09-25至2025-04-08+5.65%。窗口前已有2018-12-14至17−1.91%、2020-03-03至10−3.20%亦保留。2024可以在9月启动阶段确认，但实际固定退出不是10月事后最高点。\n\n"
        "**为什么接受/拒绝**：接受近期两费用pB>1和部分修复实际盈利的开发事实，拒绝此固定完整用途作为提高原模型收益夏普的证据并关闭；低胜率可以伴高B，但近期仍弱于A，早期负。不能选时期、照2024量3.338设置放量过滤、改固定退出或混A营救。历史DEVELOPMENT_CALIBRATION、first-vintage NOT_CERTIFIED、独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED，完整目标/去过拟合未达，目标active；旧终态/原前瞻保持。\n\n"
        "**是否需要重新验证/下一具体实验**：本金融用途不改参重跑。下一一次全体保存生命周期、原锚至首次收复、固定线/新缺口失效、实际库存/资金/股息、最后次开差额及无收复上涨覆盖归因，0新账户；最高收盘财富不是新增可实现止盈。之后新准入需实质不同信息及完整用途、真实新样本或来源/实现错误，待跑0。原A/POINT十二前瞻值及1008实际交易日计划保持，本历史形态不自动加入前瞻。\n\n"
        f"依据：[具体解释](../{expl_rel}/具体上涨与点位反推.md)、[全体实际结果](../{out_rel}/研究结果与下一步.md)、"
        f"[冻结](../{out_rel}/protocol.json)、[保存核对](../{out_rel}/saved_result_verification.json)。\n")
    for name in DOCS:
        path = ROOT / "docs" / name
        require(digest(path) == original_doc_hashes[name], "事实文件同时被其他工作修改，不覆盖。")
        first, rest = texts[name].split("\n", 1)
        appendix = decisions if name.startswith("RESEARCH_DECISIONS") else "\n\n## 技术线R183—R185当前状态与下一步（2026-10-05）\n\n"+overview+"\n\n"+f"依据：[全体结果](../{out_rel}/研究结果与下一步.md)、[全部点位](../{out_rel}/results/全部实际进出点位_已知收复量价及真实资金.csv)。\n"
        path.write_text(first+"\n\n"+top+rest.lstrip("\n")+appendix, encoding="utf-8", newline="\n")
    accounting = {
        "scope": "TECH_R183_R184_R185_ONE_FIXED_DOWN_GAP_RECLAIM_COMPLETE_POLICY", "candidate_configurations": 1,
        "new_policy_accounts": 4, "primary_accounts": 4, "saved_original_A_replays": 4, "saved_price_control_replays": 4,
        "reused_control_scenarios": 8, "all_known_daily_rows": explanation["all_daily_rows"], "origins_since_2015": explanation["origins_since_2015"],
        "all_down_gap_anchors": explanation["all_down_gap_anchors"], "down_gap_births_since_2015": explanation["down_gap_anchors_since_2015"],
        "first_reclaims_since_2015": explanation["reclaims_since_2015"], "all_anchor_terminal_counts": explanation["all_anchor_terminal_counts"],
        "all_qualification_rows_two_costs": delivery["all_qualification_rows"], "all_actual_cycle_rows_two_costs": delivery["all_actual_cycle_rows"],
        "actual_cycles_per_cost": cycle_count, "completed_per_cost": completed, "open_per_cost": opened, "cancelled_per_cost": unused,
        "actual_complete_exit_rows_two_costs": delivery["actual_complete_exit_rows"], "original_episode_rows": 61, "original_admitted_waves": 49,
        "case_cycle_rows_two_costs": delivery["case_cycle_rows"], "necessary_tests_final_passed": 9, "actual_necessary_test_executions": 9,
        "pre_result_input_fixture_failures": 0, "actual_prefix_checks": 4, "charts": 4, "explanation_frozen_sources": explanation["frozen_sources"],
        "financial_frozen_sources": frozen_count, "capital_identity_cells": 4, "all_four_point_quality_passed": delivery["all_four_actual_point_quality_passed"],
        "single_scenario_economic_passes": delivery["individual_economic_gate_passes"], "overall_economic_gate_passed": False,
        "historical_stability_gate_passed": False, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0, "risk_budget_changes": 0,
    }
    legacy = {key: value for key, value in state.items() if key.startswith("current_phase_")}
    state["archived_previous_current_phase_fields_before_R185"] = {"at": now(), "fields": legacy}
    state["archived_financial_trial_accounting_R181_before_R185"] = state["actual_candidate_trials"]
    for key in legacy:
        del state[key]
    state.update({
        "at": now(), "updated_at": now(), "latest_completed_study": out_rel, "latest_result": out_rel+"/summary.json",
        "latest_report": out_rel+"/研究结果与下一步.md", "latest_research_report": out_rel+"/研究结果与下一步.md",
        "latest_research_status": financial["status"], "latest_progress": overview, "latest_overall_summary": overview,
        "latest_continuation_outcome": overview, "current_study": "510300_DOWN_GAP_RECLAIM_STUDY_V1",
        "current_phase": "RECENT_RECLAIM_POINT_EXPECTANCY_POSITIVE_FULL_POLICY_BELOW_A_AND_REJECTED",
        "current_direction": "具体上涨量价/此前完整向下缺口修复，近期实际pB正，原A更强/早期负。",
        "current_priority": next_action, "next_research_action": next_action, "next_available_action": next_action, "next_research_question": next_action,
        "next_research_plan": out_rel+"/next_all_reclaim_lifecycle_diagnostic_proposal.json", "next_experiment_status": "ALL_SAVED_RECLAIM_LIFECYCLE_DIAGNOSTIC_PROPOSED_NO_NEW_POLICY",
        "next_candidate_field_status": "NO_NEW_ADMITTED_FINANCIAL_POLICY_AFTER_RECLAIM_REJECTION", "next_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "next_strategy_increment_status": "NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY", "current_admitted_unrun_numeric_candidates": 0,
        "latest_technical_decision": "TECH.R185", "latest_actual_model_decision": "TECH.R185", "latest_financial_strategy_decision": "TECH.R185",
        "latest_registration_decision": "TECH.R184", "latest_financial_registration_decision": "TECH.R184", "latest_financial_strategy_result": out_rel+"/summary.json",
        "new_account_return_sharpe_result": out_rel+"/summary.json", "new_account_return_sharpe": financial["status"], "latest_actual_model_kind": "COMPLETE_RULE_POLICY_WITHOUT_ESTIMATED_PREDICTION_MODEL",
        "latest_actual_prediction_model_decision": "TECH.R158", "latest_original_exit_decision": "TECH.R145", "latest_continuation_receipt": out_rel+"/delivery_receipt.json",
        "latest_down_gap_reclaim_explanation": expl_rel+"/summary.json", "latest_down_gap_reclaim_protocol": out_rel+"/protocol.json",
        "latest_down_gap_reclaim_saved_verification": out_rel+"/saved_result_verification.json", "latest_down_gap_reclaim_actual_points": out_rel+"/results/全部实际进出点位_已知收复量价及真实资金.csv",
        "latest_all_population_capital_identity": out_rel+"/results/全部四场景完成周期_实际资金恒等式.csv",
        "latest_new_information_admission_boundary": out_rel+"/next_all_reclaim_lifecycle_diagnostic_proposal.json",
        "economic_stage_status": "FIXED_DOWN_GAP_RECLAIM_REJECTED_ZERO_OF_FOUR_SCENARIO_GATES", "current_economic_stage_status": "COMPLETED_FIXED_FULL_ACCOUNT_GATE_FAILED",
        "current_study_prediction_gate_status": "NOT_APPLICABLE_COMPLETE_RULE_POLICY_NO_ESTIMATED_PREDICTION_MODEL",
        "current_prediction_stage_status": "NOT_APPLICABLE_RULE_POLICY_ORIGINAL_PREDICTION_R158_PRESERVED",
        "current_full_account_economic_gate_passed": False, "current_historical_stability_gate_passed": False, "current_all_four_actual_point_quality_passed": False,
        "current_point_quality_evidence_role": "RECENT_BOTH_COSTS_PB_ABOVE_ONE_AND_STANDARD_POSITIVE_EARLY_NEGATIVE_NOT_PROMOTION",
        "current_individual_economic_gate_passes": delivery["individual_economic_gate_passes"], "returns_and_sharpe_improved": False,
        "return_and_sharpe_improved_this_continuation": False, "return_and_sharpe_changed_this_continuation": True, "new_financial_result_computed_this_continuation": True,
        "actual_candidate_trials": accounting, "current_candidate_trials": accounting, "current_phase_trial_accounting": accounting,
        "current_goal_turn_actual_work": accounting, "actual_candidate_trials_role": "LATEST_FINANCIAL_R185_TRIAL_ACCOUNTING",
        "current_executed_unique_new_entry_events": cycle_count, "current_structural_signal_events": explanation["reclaims_since_2015"],
        "current_trade_map_cycles": cycle_count, "current_trade_map_complete_cycles": completed, "current_trade_map_open_cycles": opened,
        "current_trade_map_scope": "ONE_COST_FULL_POPULATION_SAVED_R185_PRIMARY_ACCOUNTS", "current_new_policy_complete_cycles": completed,
        "current_new_policy_open_cycles": opened, "current_new_policy_unexecuted_origins": unused,
        "current_fields_admitted_this_continuation": 0, "current_information_mechanisms_admitted_this_continuation": 1,
        "current_member_support_this_continuation": "全3488原点、112原缺口锚、60当期首次收复、全体失败/替换/开放，旧标签仅回顾，0新预测成员。",
        "current_phase_required_directions": ["LONG"], "current_phase_source_freeze_count": frozen_count, "current_phase_known_daily_rows": 3488,
        "current_phase_original_state_rows": 3488, "current_phase_information_scope": "PAST_FULL_DOWN_GAP_LATEST_ANCHOR_FIRST_STRICT_CLOSE_RECLAIM_AND_FIXED_FLOOR_FAILURE",
        "current_phase_definition_browsing": "NONE_LOCAL_FROZEN_QUOTES_AND_FINITE_COMPLETE_OLD_USES",
        "necessary_tests_passed_this_continuation": 9, "necessary_tests_passed_in_current_phase": 9, "necessary_test_executions_this_continuation": 9,
        "actual_prefix_checks_this_continuation": 4, "new_accounts_this_continuation": 4, "new_accounts_in_current_phase": 4, "new_primary_accounts_this_continuation": 4,
        "new_financial_candidate_accounts_this_continuation": 4, "new_investment_account_evaluations_this_continuation": 4, "new_strategy_accounts_this_continuation": 4,
        "internal_reference_replays_this_continuation": 8, "saved_account_controls_replayed_this_continuation": 8, "saved_accounts_checked_this_continuation": 12,
        "new_strategy_configurations_this_continuation": 1, "new_model_fits_this_continuation": 0, "historical_model_refits_this_continuation": 0,
        "new_return_labels_this_continuation": 0, "new_training_labels_this_continuation": 0, "new_model_training_labels_this_continuation": 0,
        "new_allocation_method_admitted": False, "new_method_admitted": "ONE_FIXED_DOWN_GAP_RECLAIM_COMPLETE_POLICY_EXECUTED_AND_REJECTED",
        "new_market_information_admitted": False, "new_market_requests_this_continuation": 0, "original_strategy_source_files_changed_this_continuation": 0,
        "code_files_added_this_continuation": ["research/down_gap_reclaim_inputs_v1.py", "research/down_gap_reclaim_explanation_v1.py", "research/down_gap_reclaim_account_v1.py",
            "research/down_gap_reclaim_study_v1.py", "research/deliver_down_gap_reclaim_explanation_v1.py", "research/deliver_down_gap_reclaim_results_v1.py",
            "research/finalize_down_gap_reclaim_state_v1.py", "tests/test_down_gap_reclaim_inputs_v1.py", "tests/test_down_gap_reclaim_account_v1.py"],
        "pre_result_input_test_fixture_repairs": 0, "pre_registration_new_source_representation_fixes": 0, "current_numeric_leads_role": "EXISTING_FORWARD_REGISTRY_ONLY_NOT_R185_CANDIDATES",
        "current_goal_turn_classification": "progress", "previous_goal_turn_classification": "progress",
        "goal_turn_progress_classification": "NEW_CONCRETE_RECLAIM_STAGE_EXPLANATION_AND_FOUR_ACTUAL_ACCOUNT_RESULTS",
        "current_goal_turn_classification_reason": "新证据：2024启动时可知收复/2020部分修复实际盈利、近期pB>1，但原A更高/早期负，固定完整用途拒绝。",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0, "blocked_reason": None, "blocked_audit_key": None, "blocking_decision": None,
        "goal_status": "active", "goal_tool_status_confirmed": "active", "goal_achieved": False, "whole_model_overfitting_removed": False,
        "overfitting_removed": False, "overfit_removed": False, "independent_validation_status": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "current_unmet_evidence": "R185近期两费用pB正而原A收益夏普更高，早期负；四经济/历史稳定门失败，独立和去过拟合未建立，新待跑金融0。",
        "validation_method_this_continuation": "九必要测试首轮/四前缀/八原控制精确复现/四主账户一次与保存核对，120资格/102周期/100退出/14案例行/四资金恒等式全交付。",
        "latest_long_point_metrics": {"scope": "R185_COMPLETE_POLICY_DEVELOPMENT_NOT_CURRENT_MARKET", "STRESS": primary.loc[primary.cost.eq("STRESS")].to_dict("records")},
    })
    require({key: state[key] for key in FORWARD} == original_forward and state["next_experiment"] == original_next, "原前瞻或1008实际交易日计划改变。")
    require(digest(STATE) == state_hash, "当前状态同时被其他工作修改，不覆盖。")
    write_json(STATE, state)
    write_json(OUT / "project_state_update_receipt.json", {
        "at": now(), "status": "PASS_FOUR_DURABLE_FACT_FILES_AND_CURRENT_STATE_UPDATED_ONCE", "latest_technical_and_financial": "TECH.R185",
        "financial_registration": "TECH.R184", "actual_prediction_preserved": "TECH.R158", "original_exit_preserved": "TECH.R145",
        "goal_status": "active", "goal_achieved": False, "admitted_unrun_candidates": 0, "forward_values_unchanged": original_forward,
        "existing_forward_next_experiment_unchanged": original_next, "overall_full_account_gate_passed": False,
        "historical_stability_gate_passed": False, "original_strategy_source_changes": 0, "new_primary_accounts": 4,
        "control_replays": 8, "legacy_current_phase_fields_archived": len(legacy),
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in [Path(__file__), STATE, *(ROOT / "docs" / name for name in DOCS)]],
    }, exclusive=True)
    print("四长期事实与状态一次更新至R185，原策略及前瞻保持，完整目标未达且active。", flush=True)


if __name__ == "__main__":
    run()
