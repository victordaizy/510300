"""一次整理R175—R177长期事实；保留旧冻结策略、其他分支和真实前瞻。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.full_daily_gap_study_v1 import OUT, EXPLANATION, PRIMARY
from research.finalize_downtrend_break_state_v1 import FORWARD, DOCS
from research.point_first_passage_study_v1 import read, write_json, digest, now, require

STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def run():
    require(not (OUT / "project_state_update_receipt.json").exists(), "R177长期事实已更新，不重复。")
    financial, explanation, delivery = read(OUT / "summary.json"), read(EXPLANATION / "summary.json"), read(OUT / "delivery_receipt.json")
    require(financial["technical_decision"] == "TECH.R177" and financial["new_primary_accounts"] == 4,
            "四新账户尚未完成。")
    require(delivery["status"] == "PASS_ALL_FULL_GAP_POINTS_CASES_EXITS_AND_SAVED_IDENTITIES_DELIVERED",
            "全体资格及真实周期交付未完成。")
    require(delivery["open_case_windows_clipped_at_own_account_end"], "开放案例超出自身账户末日。")
    for path in (OUT / "protocol.json", EXPLANATION / "protocol.json"):
        for source in read(path)["sources"]:
            require(digest(ROOT / source["path"]) == source["sha256"], "本轮冻结来源改变。")
    state = read(STATE)
    require(state["latest_technical_decision"] == "TECH.R174" and state["latest_financial_strategy_decision"] == "TECH.R173",
            "接手状态不再是R174诊断/R173金融，不能覆盖另一阶段。")
    before_forward = {key: state[key] for key in FORWARD}
    before_next = state["next_experiment"]
    marker = "> 技术线当前事实 TECH.R177（优先于下方技术线历史快照，2026-10-05）："
    texts = {name: (ROOT / "docs" / name).read_text(encoding="utf-8-sig") for name in DOCS}
    require(all(marker not in text for text in texts.values()), "四事实文件已有本轮标记。")
    write_json(OUT / "project_state_before.json", state, exclusive=True)
    metric_rows = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    primary = metric_rows.loc[metric_rows.policy.eq(PRIMARY)]
    early = primary.loc[primary.period.eq("2015_2019") & primary.cost.eq("STRESS")].iloc[0]
    recent = primary.loc[primary.period.eq("2020_2026") & primary.cost.eq("STRESS")].iloc[0]
    economic = bool(financial["all_four_economic_gates_passed"])
    stable = bool(financial["historical_stability_gate_passed"])
    point_quality = bool(delivery["all_four_actual_point_quality_passed"])
    out_relative, explanation_relative = OUT.relative_to(ROOT).as_posix(), EXPLANATION.relative_to(ROOT).as_posix()
    overview = (
        "按先解释具体上涨/失败及量价指标、再反推当时可知点位完成TECH.R175解释、R176唯一完整日线缺口登记和R177四真实账户。"
        "当前完整日low严格高于上一完整日high为出生，次开一次，开盘到/低于下沿取消；新完整缺口仅提高下沿，已知日low回补即次合法开退出，收盘在线上也不撤销，未知不强退、风险只减。"
        "使用整数.001与截至当时已发生分红，除息相等边界不生成伪缺口；原3488日、2855当期原点、97向上/91向下只观察、61/49段、旧2846/2826/20标签及9尾部保持。"
        "每档费用97资格=65真实周期（64完成/1早期开放）+1次开取消+31已有持仓，194资格/130周期（128完成/2开放）和22案例周期行全交付。"
        f"压力较早29完成/1开放、胜率{early.win_rate:.2%}/B{early.payoff:.6f}/pB{early.p_times_b:.6f}，净年化{early.net_cagr:.6%}/夏普{early.net_sharpe:.6f}/DD{early.max_drawdown:.4%}；"
        f"近期35完成、{recent.win_rate:.2%}/B{recent.payoff:.6f}/pB{recent.p_times_b:.6f}、{recent.net_cagr:.6%}/{recent.net_sharpe:.6f}/DD{recent.max_drawdown:.4%}。"
        "早期两费用pB<1，近期虽pB>1且相对R173提高，仍远低原A压力3.9908%/1.2169；四单场景经济门0通过、整体与历史稳定均败，固定完整政策拒绝关闭。"
        "完整年均次数5.8/5.1667高于A4.4/4.1667，但不等于收益夏普改进。2019-01-09缺口压力次01-10入/03-11出+18.51%；2020-03-25次03-26入/04-22出+2.25%，后续多次失败；"
        "2024-09-25次09-26入/2025-03-11出+14.83%，A当日有21800份/目标31.92%；2015-07-10次07-13入/07-28出−10.28%。"
        "97路径44低点已回补而收盘仍在线上，实际128完成退出中54为此类（两费用），全记录而非新退出比较。"
        "八必要测试/四真实前缀、八原控制精确复现、四主账户一次运行和保存核对PASS；34解释/74金融来源。"
        "首次日期ms/ns接口失败修正时主金融结果0，旧输入/源码不变；备份捕获修正后版本的时序局限及重建标识如实保存。"
        "四完成资金恒等式保持，压力早完整周期净−2006.35元而账户净增869.86元，开放差额不填胜率；近期净11674.60元。"
        "历史DEVELOPMENT_CALIBRATION、first-vintage NOT_CERTIFIED、独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED，去过拟合及完整收益夏普目标未达；目标active/本轮progress/阻塞0。"
        "最新技术/金融R177、登记R176，预测R158/原退出R145和所有旧终态保持。下一只全体保存缺口生命周期与次开时钟归因，不加过滤救本规则；未登记另一策略/待跑0，十二真实前瞻和原1008实际交易日计划、其他分支保持。"
    )
    top = marker + (
        "已完成R175具体量价解释/R176登记/R177四实际账户。97完整向上缺口，每费用65周期（64完成/1开放）、1取消/31已有持仓；194资格/130周期全保留。"
        f"压力早净年化{early.net_cagr:.4%}/夏普{early.net_sharpe:.4f}/pB{early.p_times_b:.4f}，近{recent.net_cagr:.4%}/{recent.net_sharpe:.4f}/pB{recent.p_times_b:.4f}；"
        "年均完成5.8/5.17增加，早pB不足1、近期仍低A，四经济门与稳定门全败，固定政策关闭。"
        "八测试/四前缀/八对照/四保存账户核对通过；下一全体保存时钟归因只诊断，待跑金融0。"
        "收益夏普/独立/去过拟合目标未达且active，原A/旧终态/前瞻保持。"
        " [具体上涨、全部点位及完整结果](../reports/research/510300_full_daily_gap_study_v1/研究结果与下一步.md)。\n\n"
    )
    decisions = """
## TECH.R175：先解释相邻完整日线区间分离及回补（2026-10-05）

**假设**：日线完整向上缺口可在部分上涨修复较早阶段到达，其后低点回补与后续缺口的顺序能解释走势；量、日MACD和上一完整周只作当时状态观察。

**验证方法**：完整日low严格大于上一日high，整数.001报价及当时已发生分红，相等非缺口。全3488日、97向上缺口和91向下仅观察、全部出生至回补路径、原61分段/49正式波段及四固定案例保留，四输入测试和四真实行情前缀。旧20日2846标签/2826成熟/20删失/9尾部仅回顾描述。完整未来金融用途在连接旧结果前固定。有限旧用途区分十五分钟三柱、分钟扫低FVG、RR3开盘取消和R173收盘突破，不声称穷尽全项目。

**结果**：全部97出生路径完成，44条日低点回补时收盘仍在线上。2019-01-09价区间上移、相对量1.400、日柱正而DIF/上一周负；2020-03-25相对量1.666、高波动且日周动量仍负，修复较早到达；2024首次完整缺口09-25，晚于09-24上涨一日，放量而长周期指标仍滞后；2015固定06-29/30反弹无此点位，07-10后来的缺口在高波动下失败。34来源冻结、四图已查看，0账户/拟合/新训练标签/采集。

**为什么接受/拒绝**：接受可知时钟和具体阶段/失效解释；没有接受真实订单流、造成上涨的因果或预测优越。缺少主动逐笔和Level2，量价指标不补造订单流。旧FVG终态和R173固定失败保持。旧20日描述pB约0.758不代替完整动态账户。

**是否需要重新验证**：固定解释已完成，只因真实来源/实现错误复核，不据赢家调缺口幅度或指标；完整金融另见R176—R177，独立样本未建立。

## TECH.R176：唯一完整日线缺口金融用途登记（2026-10-05）

**假设**：相邻整日区间向上分离、用日低点回补及只上移缺口下沿管理失效，可能提高更多实际点位及全账户收益夏普。

**验证方法**：空仓出生后次真实开一次，整数现金平移开盘<=上一日high取消不追；初始下沿为上一high，持有新缺口仅提高到max(旧线,新上一high)，已知日low<=线次合法开退出，未知自身不卖、风险只减，无加仓/A/2R/20日/期末清仓。原20万元/252日/现金0、两时期两费用、50%/ES/跳空/DD/T+1/100份/.001/股息口径保持。四输入/四账户测试、四真实前缀及八共同控制精确复现，74金融来源冻结。全部四场景净CAGR/夏普>0且同时优于原A及纯价格、DD<=10%、实际pB>1/pB−q>0；固定20/252日配对区块各2000、seed510300154全比较，次数软目标。原退出MSE/115成员门不用于该规则，R145/R158不营救。

**结果**：一完整新政策、四主账户由R177一次执行完，登记不再待跑。共同对照首次ms/ns日期表示不匹配保存，经济日期完全相同，仅新金融接口满足冻结输入ns再恢复原账户分辨率，旧输入不改。修正时新主金融结果读取0，八原控制随后精确复现。备份命令异步复制实际捕获修正后源码的局限单独纠正，重建副本明确标注而不冒充首次快照。

**为什么接受/拒绝**：接受一次不同完整用途的开发测量，不接受独立样本、策略有效或实盘；不是R173加缺口过滤，也不把分钟FVG结果移植为日线收益证据。

**是否需要重新验证**：不重登记、不据金融结果改缺口大小/时钟/量/MACD/RV/费用/预算/时期或混A；实际裁决见R177。只有真实实现/来源错误才复核，并保留原证据。

## TECH.R177：交易次数增加，固定完整缺口账户拒绝关闭（2026-10-05）

**假设**：R176固定入出用途必须在所有时期费用中同时提高扣费后的收益、全日历夏普和实际交易质量，不能由两个好案例证明。

**验证方法**：四新账户一次完成、八原A/纯价格控制复用，保存指标/资金/库存/T+1/成本/股息核对PASS。194资格及130真实周期全部交付，128完成/2开放；每费用97出生=65实际周期（64完成/1早期开放）+1次开取消+31已有持仓。预首日及期末无次开请求核对，未成交/开放不填完成收益，22案例周期行按自身账户末日截断。全体完成退出和四资金恒等式仅保存账本派生，0新账户重跑。

**结果**：早BASE29完成/1开放，净年化0.4264%/夏普0.1446、胜率27.59%/B2.9500/pB0.8138/DD6.46%；早STRESS0.0898%/0.0441、27.59%/2.6803/0.7394/DD6.64%。近BASE35完成、1.1744%/0.3590、31.43%/3.7401/1.1755/DD4.95%；近STRESS0.8777%/0.2856、28.57%/3.7866/1.0819/DD5.24%。四标准期望pB−q均正，但早pB不足1，四单场景经济门0通过，整体/历史稳定均败。近期相对R173收益夏普提高，仍低原A压力3.9908%/1.2169，不构成整体模型改进。完整年均5.8/5.1667高于原A4.4/4.1667，次数不是利润保证。

压力实际2019-01-10至03-11净+18.51%、04-02至04-29净−2.35%；2020-03-26至04-22+2.25%，后续05-27、06-02、06-29和07-07入点均净负，不能把后续波段总涨幅给早入点；2024-09-26至2025-03-11+14.83%，09-25原A已有21800份/目标31.92%；2015-07-13至07-28−10.28%同样保留。初始当时原报价下沿2019-01-09为3.110元、2020-03-25为3.623元、2024-09-25为3.429元，只是当时已知失效，非事后低价成交。实际128完成退出中54低点已回补但收盘仍在线上（两费用），不是另一退出策略回测。

四完成利润恒等式ΣQr=ΣQ×平均r+Σ[(Q−平均Q)(r−平均r)]：压力早547.05−2553.40=−2006.35元，NAV净增869.86元含开放周期，不能提前填入完成胜率；近期11853.09−178.49=11674.60元。原实际份额毛/费用、投入及所有现金日保持，均值项不是等额配仓账户。

**为什么接受/拒绝**：接受个别可知点位、具体量价顺序及次数增加的开发事实；拒绝该固定完整政策作为全账户收益夏普改进，关闭并保留。不能只挑2019/2024赢家、近期或低费用，不能调回补线、补MACD/量/RV过滤、拼旧突破退出或混A营救。全部历史DEVELOPMENT_CALIBRATION、first-vintage NOT_CERTIFIED、独立NOT_ESTABLISHED、global DSR/PBO NOT_COMPUTED，去过拟合及完整收益夏普目标未达；目标active，原A/所有旧终态和十二真实前瞻保持。

**是否需要重新验证/下一步具体实验**：本固定用途一次完成，不重跑改参。下一只对全部97出生和四保存账户做生命周期/执行时钟归因，解释出生、后续缺口提高下沿、日低点回补及次开卖出的先后；2019早期修复、2020反复中断、2024持续持有和所有失败/开放一起记录，不以结果形成新过滤。该诊断无新账户，金融下一政策未定义/登记、已准入待跑0；进一步策略须实质不同信息和完整用途、真正新样本或真实实现错误。真实前瞻A/POINT原1008实际交易日计划不变，不能将历史点位自动登记为新前瞻候选。其他宏观/盘口分支保持。

依据：[具体上涨及完整结果](../reports/research/510300_full_daily_gap_study_v1/研究结果与下一步.md)、[全体解释定义](../reports/research/510300_full_daily_gap_explanation_v1/protocol.json)、[R176冻结](../reports/research/510300_full_daily_gap_study_v1/protocol.json)、[全四场景与所有区间](../reports/research/510300_full_daily_gap_study_v1/summary.json)、[保存账本核对](../reports/research/510300_full_daily_gap_study_v1/saved_result_verification.json)、[全部真实进出](../reports/research/510300_full_daily_gap_study_v1/results/全部实际进出点位_已知缺口量价及真实资金.csv)、[下一全体时钟归因](../reports/research/510300_full_daily_gap_study_v1/next_all_gap_lifecycle_diagnostic_proposal.json)。
"""
    for name, text in texts.items():
        first, remainder = text.split("\n", 1)
        append = decisions if name != "PROJECT_STATE.md" else (
            "\n## 技术线TECH.R175—R177当前更新（2026-10-05）\n\n"+overview+
            "\n\n完整假设→验证→结果→接受/拒绝→重验入口见[技术线研究决策](RESEARCH_DECISIONS_TECHNICAL_LINE.md)；共享其他分支终态保持。\n")
        (ROOT / "docs" / name).write_text(first+"\n\n"+top+remainder.lstrip("\n")+"\n"+append, encoding="utf-8", newline="\n")
    accounting = {
        "scope":"TECH_R175_R176_R177_ONE_FIXED_FULL_DAILY_GAP_COMPLETE_POLICY",
        "candidate_configurations":1,"new_policy_accounts":4,"primary_accounts":4,
        "saved_original_A_replays":4,"saved_price_control_replays":4,"reused_control_scenarios":8,
        "all_known_daily_rows":3488,"origins_since_2015":2855,"full_up_gap_events":97,"down_gaps_observation_only":91,
        "all_qualification_rows_two_costs":194,"all_actual_cycle_rows_two_costs":130,
        "actual_cycles_per_cost":65,"completed_per_cost":64,"open_per_cost":1,
        "open_cancelled_per_cost":1,"already_holding_no_new_cycle_per_cost":31,
        "original_episode_rows":61,"original_admitted_waves":49,"actual_complete_exit_rows_two_costs":128,
        "low_fill_exit_close_above_floor_two_costs":54,"case_cycle_rows_two_costs":22,
        "necessary_tests_final_passed":8,"pre_registration_date_interface_failures_preserved":1,
        "actual_prefix_checks":4,"charts":4,"explanation_frozen_sources":34,"financial_frozen_sources":74,
        "capital_identity_cells":4,"all_four_point_quality_passed":point_quality,
        "single_scenario_economic_passes":int(delivery["individual_economic_gate_passes"]),
        "overall_economic_gate_passed":economic,"historical_stability_gate_passed":stable,
        "new_model_fits":0,"new_training_labels":0,"new_market_requests":0,"risk_budget_changes":0,
    }
    next_action = "全97出生、四保存账户及所有失败/开放作缺口出生、后续抬线、日低点回补和次开执行时钟归因；不跑新账户或据结果加指标/改阈值/退出/预算/费用/混A营救R177。之后只实质不同信息及完整用途、真实新样本或真实来源/实现错误准入，原真实前瞻保持。"
    legacy_phase = {key:value for key,value in state.items() if key.startswith("current_phase_")}
    state["archived_previous_current_phase_fields_before_R177"] = {
        "at":now(),"role":"OLD_BRANCH_FIELDS_PRESERVED_NOT_CURRENT_FULL_DAILY_GAP_PHASE", "fields":legacy_phase}
    for key in legacy_phase:
        del state[key]
    state.update({
        "at":now(),"updated_at":now(),"latest_completed_study":out_relative,"latest_result":out_relative+"/summary.json",
        "latest_report":out_relative+"/研究结果与下一步.md","latest_research_report":out_relative+"/研究结果与下一步.md",
        "latest_research_status":financial["status"],"latest_progress":overview,"latest_overall_summary":overview,
        "latest_continuation_outcome":overview,"current_study":"510300_FULL_DAILY_UP_GAP_STUDY_V1",
        "current_phase":"FULL_GAP_POINT_COUNT_INCREASED_FULL_ACCOUNT_FIXED_POLICY_REJECTED",
        "current_direction":"先全体完整日线缺口量价解释，再唯一完整入出账户；次数增加但完整收益夏普拒绝，下一保存生命周期时钟归因。",
        "current_priority":next_action,"next_research_action":next_action,"next_available_action":next_action,
        "next_research_plan":out_relative+"/next_all_gap_lifecycle_diagnostic_proposal.json",
        "next_experiment_status":"ALL_SAVED_GAP_LIFECYCLE_DIAGNOSTIC_PROPOSED_NO_NEW_POLICY",
        "next_candidate_field_status":"NO_NEW_ADMITTED_FINANCIAL_POLICY_AFTER_FULL_GAP_REJECTION",
        "next_financial_experiment":"NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "next_strategy_increment_status":"NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY","current_admitted_unrun_numeric_candidates":0,
        "latest_technical_decision":"TECH.R177","latest_actual_model_decision":"TECH.R177",
        "latest_financial_strategy_decision":"TECH.R177","latest_registration_decision":"TECH.R176",
        "latest_financial_registration_decision":"TECH.R176","latest_financial_strategy_result":out_relative+"/summary.json",
        "new_account_return_sharpe_result":out_relative+"/summary.json",
        "latest_actual_model_kind":"COMPLETE_RULE_POLICY_WITHOUT_ESTIMATED_PREDICTION_MODEL",
        "latest_actual_prediction_model_decision":"TECH.R158","latest_original_exit_decision":"TECH.R145",
        "latest_continuation_receipt":out_relative+"/delivery_receipt.json",
        "latest_full_daily_gap_explanation":explanation_relative+"/summary.json",
        "latest_full_daily_gap_protocol":out_relative+"/protocol.json",
        "latest_full_daily_gap_saved_verification":out_relative+"/saved_result_verification.json",
        "latest_full_daily_gap_actual_points":out_relative+"/results/全部实际进出点位_已知缺口量价及真实资金.csv",
        "latest_all_population_capital_identity":out_relative+"/results/全部四场景完成周期_实际资金恒等式.csv",
        "latest_new_information_admission_boundary":out_relative+"/next_all_gap_lifecycle_diagnostic_proposal.json",
        "economic_stage_status":"FIXED_FULL_GAP_POLICY_REJECTED_ZERO_OF_FOUR_SCENARIO_GATES_PASSED",
        "current_economic_stage_status":"COMPLETED_FIXED_FULL_ACCOUNT_GATE_FAILED",
        "current_study_prediction_gate_status":"NOT_APPLICABLE_COMPLETE_RULE_POLICY_NO_ESTIMATED_PREDICTION_MODEL",
        "current_prediction_stage_status":"NOT_APPLICABLE_RULE_POLICY_ORIGINAL_PREDICTION_R158_PRESERVED",
        "current_full_account_economic_gate_passed":economic,"current_historical_stability_gate_passed":stable,
        "current_all_four_actual_point_quality_passed":point_quality,"current_point_quality_evidence_role":"MIXED_DEVELOPMENT_ESTIMATES_EARLY_PB_BELOW_ONE_NOT_STRATEGY_PROMOTION",
        "current_individual_economic_gate_passes":int(delivery["individual_economic_gate_passes"]),
        "returns_and_sharpe_improved":False,"return_and_sharpe_improved_this_continuation":False,
        "new_financial_result_computed_this_continuation":True,"actual_candidate_trials":accounting,
        "current_phase_trial_accounting":accounting,"current_goal_turn_actual_work":accounting,
        "actual_candidate_trials_role":"LATEST_FINANCIAL_R177_TRIAL_ACCOUNTING",
        "current_executed_unique_new_entry_events":65,"current_structural_signal_events":97,
        "current_trade_map_cycles":65,"current_trade_map_complete_cycles":64,"current_trade_map_open_cycles":1,
        "current_trade_map_scope":"ONE_COST_FULL_POPULATION_SAVED_R177_PRIMARY_ACCOUNTS",
        "current_new_policy_complete_cycles":64,"current_new_policy_open_cycles":1,"current_new_policy_unexecuted_origins":32,
        "current_fields_admitted_this_continuation":0,"current_information_mechanisms_admitted_this_continuation":1,
        "current_member_support_this_continuation":"全部3488原报价状态、2855当期原点、97完整向上缺口；全部真实周期与失败，旧标签仅事后对齐，0新预测成员。",
        "current_phase_required_directions":["LONG"],"current_phase_source_freeze_count":74,
        "current_phase_known_daily_rows":3488,"current_phase_original_state_rows":3488,
        "current_phase_information_scope":"ADJACENT_FULL_DAILY_HIGH_LOW_SEPARATION_AND_LOW_FILL_CLOCK_NOT_LEVEL2_ORDER_FLOW",
        "current_phase_definition_browsing":"NONE_LOCAL_FROZEN_QUOTES_AND_SAVED_OLD_PURPOSES",
        "necessary_tests_passed_this_continuation":8,"necessary_tests_passed_in_current_phase":8,
        "actual_prefix_checks_this_continuation":4,"new_accounts_this_continuation":4,"new_accounts_in_current_phase":4,
        "new_primary_accounts_this_continuation":4,"new_financial_candidate_accounts_this_continuation":4,
        "new_investment_account_evaluations_this_continuation":4,"internal_reference_replays_this_continuation":8,
        "saved_account_controls_replayed_this_continuation":8,"saved_accounts_checked_this_continuation":12,
        "new_model_fits_this_continuation":0,"historical_model_refits_this_continuation":0,
        "new_return_labels_this_continuation":0,"new_training_labels_this_continuation":0,
        "new_model_training_labels_this_continuation":0,"new_allocation_method_admitted":False,
        "new_method_admitted":"ONE_FIXED_FULL_DAILY_GAP_COMPLETE_POLICY_EXECUTED_AND_REJECTED",
        "new_market_information_admitted":False,"new_market_requests_this_continuation":0,
        "original_strategy_source_files_changed_this_continuation":0,
        "code_files_added_this_continuation":["research/full_daily_gap_inputs_v1.py","research/full_daily_gap_explanation_v1.py",
            "research/full_daily_gap_account_v1.py","research/full_daily_gap_study_v1.py","research/deliver_full_daily_gap_results_v1.py",
            "research/finalize_full_daily_gap_state_v1.py","tests/test_full_daily_gap_inputs_v1.py","tests/test_full_daily_gap_account_v1.py"],
        "pre_registration_new_source_representation_fixes":1,"initial_source_backup_timing_limit_preserved":True,
        "current_numeric_leads_role":"EXISTING_FORWARD_REGISTRY_ONLY_NOT_R177_VALIDATED_CANDIDATES",
        "current_signal_role":"HISTORICAL_20260930_ORIGINAL_FORWARD_SNAPSHOT_NOT_CURRENT_20261005_MARKET_VIEW",
        "current_goal_turn_classification":"progress","previous_goal_turn_classification":"progress",
        "goal_turn_progress_classification":"NEW_CASE_EXPLANATION_AND_FULL_GAP_ACCOUNT_EVIDENCE_FIXED_POLICY_REJECTED",
        "current_goal_turn_classification_reason":"独立日线完整区间分离用途的四实际账户及全97时钟改变已知证据：次数增加、早pB不足1、近期低A，具体修复/中断/追入差异已保存。",
        "blocked_audit_count":0,"consecutive_blocked_goal_turns":0,"blocked_reason":None,"blocked_audit_key":None,"blocking_decision":None,
        "goal_status":"active","goal_tool_status_confirmed":"active","goal_achieved":False,
        "whole_model_overfitting_removed":False,"overfitting_removed":False,"overfit_removed":False,
        "independent_validation_status":"NOT_ESTABLISHED","global_DSR_PBO":"NOT_COMPUTED",
        "current_unmet_evidence":"R177次数增加但早期pB不足1、近期收益夏普低A，整体经济/稳定门败，独立/去过拟合未建立；无新待跑金融策略。",
        "validation_method_this_continuation":"四输入/四账户测试、四实际前缀、八控制精确复现、四主账户一次运行与保存核对；194资格/130周期/128退出/22案例周期及四资金恒等式全部交付，首次日期接口及备份时序局限保留。",
    })
    require({key:state[key] for key in FORWARD} == before_forward, "十二真实前瞻值改变。")
    require(state["next_experiment"] == before_next, "原1008实际交易日计划改变。")
    write_json(STATE, state)
    write_json(OUT / "project_state_update_receipt.json", {
        "at":now(),"status":"PASS_FOUR_DURABLE_FACT_FILES_AND_CURRENT_STATE_UPDATED_ONCE",
        "latest_technical_and_financial":"TECH.R177","financial_registration":"TECH.R176",
        "actual_prediction_preserved":"TECH.R158","original_exit_preserved":"TECH.R145",
        "goal_status":"active","goal_achieved":False,"admitted_unrun_candidates":0,
        "forward_values_unchanged":before_forward,"existing_forward_next_experiment_unchanged":before_next,
        "all_four_actual_point_quality_passed":point_quality,"individual_economic_gate_passes":int(delivery["individual_economic_gate_passes"]),
        "overall_full_account_gate_passed":economic,"historical_stability_gate_passed":stable,
        "original_strategy_source_changes":0,"new_primary_accounts":4,"control_replays":8,
        "legacy_current_phase_fields_archived":len(legacy_phase),
        "sources":[{"path":p.relative_to(ROOT).as_posix(),"sha256":digest(p)} for p in
                   [Path(__file__),STATE,*(ROOT/"docs"/name for name in DOCS)]],
    }, exclusive=True)
    print("四长期事实及当前状态一次更新至R177；固定缺口规则拒绝，原前瞻和其他分支保持，目标active。", flush=True)


if __name__ == "__main__":
    run()
