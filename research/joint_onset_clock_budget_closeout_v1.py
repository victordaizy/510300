"""R166时钟与R167预算解释一次写入四长期事实来源，金融R165保持。"""
from __future__ import annotations
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.point_first_passage_study_v1 import read, write_json, require, digest, now
from research.joint_onset_clock_diagnostic_v1 import OUT as CLOCK
from research.joint_onset_budget_attribution_v1 import OUT

STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FORWARD_KEYS = (
    "forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
    "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
    "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
    "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation",
)
SUMMARY = (
    "2026-10-04 TECH.R166—R167完成全85联合出生的时钟、投入权重和原预算归因。价格最后51、日DIF最后10、"
    "周柱最后22、同日多项2；原84周期（83完成/1开放）及1限价未成交保留。压力日DIF组等权+0.7192%/"
    "pB1.288，却实际合计−1413.96元、投入加权−0.3349%，两个赢家都在较早期、近期四笔全负。"
    "36全体资金恒等式核对：该组乘积项+3036.06元、协方差项−4450.02元，非等额账户。"
    "原170决定/168实际买入精确复算；85压力原点75跳空预算/10ES绑定，83跳空预算由剩余DD余量更紧决定。"
    "2019赢家入场资金9.77%，解释权重差，不能据此放宽预算。三测试/三前缀、101及112冻结来源，"
    "32时钟/56预算全单元、全体图已查看；0新账户/拟合/训练标签/采集。旧波动率/半仓、"
    "波动限价、自身盈利过滤的实际关闭已核对，不营救R165、不选好分组。下一严格2/2已确认日收盘双高双低"
    "抬升结构只提案，先核旧完整定义并解释所有上涨/失败，再决定金融登记；已准入待跑0。"
    "最新技术R167，最新金融R165、预测R158、原退出R145；目标active/阻塞0，独立和去过拟合未达，"
    "收益夏普未提高，原十二前瞻值和其他分支保持。"
)
ENTRY = """
## TECH.R166：全部触发时钟及投入权重解释（2026-10-04）

**假设**：R165的失败可能包含价格重新确认、日DIF转正、上一完整周确认等不同到达时钟；先全体解释，不据分组结果营救失败策略。

**验证方法**：当前/前一已知原点唯一分类；同日多项单列、未知不伪造出生、周柱新增必须有真实新完整周来源。3488出生标记及三真实前缀一致，三必要测试通过，101来源冻结后一次解释。原85资格、84周期（83完成/1开放）和1限价未成交、两费用/ALL与三个时期共32单元全保留，事后结果不进入分类；0新账户/模型/训练标签/采集。

**结果**：价格最后51、日DIF最后10、周柱最后22、同日多项2。压力价格组pB0.746/原合计−5872.85元；日DIF组pB1.288/等权+0.7192%却实际−1413.96元；周柱组22资格/20完成，pB0.514、−1460.23元，1开放/1未成交未知保留；同日2全负，B/pB未定义。日DIF两个赢家在较早期6笔，近期3+1笔全负，不据ALL晋升。

结果后追加的是全体资金代数解释，非事前登记模型：36整体/子集恒等式全部报告，Σ(Qr)=ΣQ×平均r+Σ[(Q−平均Q)(r−平均r)]。日DIF两项+3036.06/−4450.02元，投入加权周期均值−0.3349%；不把第一项称等额账户收益。全体较早完成191.38元与期末352.31元差来自原开放周期财富；近期完成−9941.42元且等权平均也负，不能只归因资金配置。

**为什么接受/拒绝**：接受时钟与实际资金差异，拒绝以某组pB>1或等权均值选择过滤并重跑R165；原账户子集不是新策略或独立样本，pB、资金加权回报与全日历Sharpe须分开。原四案例与全部反例图已经查看，不选择某费用/时期营救。

**是否需要重新验证/下一步**：固定解释完成不改分类。接续R167仅还原原预算，不放宽风险；下一独立日线结构提案需旧完整定义核对、因果时钟及全体案例解释后才决定金融登记。独立及去过拟合未建立，最新金融R165、预测R158、原退出R145保持。

依据：[完整时钟/资金报告](../reports/research/510300_joint_onset_clock_diagnostic_v1/研究结果与下一步.md)、[固定合同](../reports/research/510300_joint_onset_clock_diagnostic_v1/protocol.json)、[实际结果](../reports/research/510300_joint_onset_clock_diagnostic_v1/summary.json)、[全32单元](../reports/research/510300_joint_onset_clock_diagnostic_v1/results/四时钟原实际分组_全部32单元.csv)、[资金恒等式与事后来源](../reports/research/510300_joint_onset_clock_diagnostic_v1/allocation_identity_summary.json)。

## TECH.R167：全体原风险预算与实际份额归因（2026-10-04）

**假设**：负的投入—回报协方差可能涉及前期回撤余量或ES压缩后续赢家；只能解释原预算来源，不能用未来利润放宽预算。

**验证方法**：使用原信号收盘自身现金/应收/NAV/峰值/已知ES复算50%意向及cap_quantity；加100份会违反的仓位/ES/跳空项完整集合，不择一主因。原min(5%NAV,50%max[0,NAV−90%峰值])跳空预算不改。次开报价、限价、股息应收、现金及STRESS压力费用评估单列为执行信息。冻结112来源、一次归因，170原决定与168已发生买入份额/价格/费用全部一致，原点严格早于执行；七非空预算集合/四时期/两费用共56单元全保留。无新账户/拟合/训练标签/采集，没有新增镜像测试代替保存账本核对。

**结果**：85压力资格75次份额加一手违反跳空预算、10次违反ES；跳空预算来源83次剩余DD余量更紧、2次与原5%同值，两统计含义分别保留。2019-01-18NAV197574.83/峰值215035.40、回撤8.12%、ES5.2765%，跳空预算2021.49元；原50%意向31200份降至预定6200、真实6100，9.77%资金捕获+16.14%。2015-02-13亏损点资金41.52%；2020后段赢家32.11%；2024-09-30的8900份请求仍限价未成交，不填收益。

**为什么接受/拒绝**：接受预算压缩与原投入的可重建事实；不把负资金协方差当程序错误、不从事后赢家加仓、不放宽50%/ES/跳空/DD、不构造等额或零费反事实并晋升。近期全体等权回报已负，信号问题仍在。旧第41轮波动率/半仓、第50轮波动限价及原自身盈利过滤的实际失败/禁止细调已查，原年化口径不与当前直接混排，不恢复旧研究。

**是否需要重新验证/下一步具体实验**：本固定归因完成。下一提案是严格左2/右2已确认日收盘交替高低点中双高、双低同时抬升；b点只能b+2可知，同类更极端只更新当前尾点，不重写过去。原A04第九项低高回调比、旧交替枢轴/ATR/DIF背离、旧周枢轴支撑已有限核对，组件有旧试验，不能称全项目未试；仍须核完整旧金融触发/退出及终态。先解释原3488日、61分段/49正式上涨及全体失败，再决定是否另立唯一金融协议。提案未来用途已写明出生次开、已知结构失效或跌到最新确认低点次合法开退出、未知不迫使卖出、风险只减、无混A/加仓/固定2R或20日；不是已准入策略，当前新金融未登记、待跑0。

最新技术R167、最新金融R165、预测模型R158、原退出R145保持。目标active，本轮progress/阻塞0，收益夏普未提高、独立和去过拟合未建立、global DSR/PBO未算；原十二前瞻值和其他宏观/盘口分支状态保持。

依据：[原预算完整报告](../reports/research/510300_joint_onset_budget_attribution_v1/研究结果与下一步.md)、[固定合同](../reports/research/510300_joint_onset_budget_attribution_v1/protocol.json)、[全部实际核对](../reports/research/510300_joint_onset_budget_attribution_v1/summary.json)、[全部原点预算与真实数量](../reports/research/510300_joint_onset_budget_attribution_v1/results/全部原资格_事前预算来源与实际开盘份额.csv)、[下一不同结构提案](../reports/research/510300_joint_onset_budget_attribution_v1/next_confirmed_structure_explanation_proposal.json)。
"""


def main():
    require(not (OUT / "closeout_receipt.json").exists(), "两项解释已经更新长期事实，不重复。")
    clocks, budget = read(CLOCK / "summary.json"), read(OUT / "summary.json")
    require(clocks["technical_decision"]=="TECH.R166" and budget["technical_decision"]=="TECH.R167", "实际解释未完成。")
    require(budget["origin_decisions_verified"]==170 and budget["actual_buy_quantities_verified"]==168, "原决定或买入复算缺失。")
    goal = read(OUT / "goal_tool_status.json")
    require(goal["goal"]["status"]=="active", "目标工具未确认active。")
    for folder in [CLOCK, OUT]:
        for source in read(folder/"protocol.json")["sources"]:
            require(digest(ROOT/source["path"])==source["sha256"], "固定解释来源改变。")
    before = read(STATE)
    require(before["latest_technical_decision"]=="TECH.R165", "技术线已由其他工作推进，不覆盖。")
    preserve_keys = (*FORWARD_KEYS, "latest_financial_strategy_decision", "latest_financial_strategy_result",
                     "latest_actual_prediction_model_decision", "original_exit_model_latest_decision",
                     "latest_model_decision", "latest_actual_model_decision", "latest_registration_decision")
    preserved = {k:copy.deepcopy(before[k]) for k in preserve_keys}
    write_json(OUT/"project_state_before_R166_R167.json",before,exclusive=True)
    stamp=now()
    proposal=(OUT/"next_confirmed_structure_explanation_proposal.json").relative_to(ROOT).as_posix()
    action="先进一步核对旧A04/日线交替枢轴背离/周支撑的完整金融用途，固定严格2/2已确认日收盘双高双低抬升结构的全体解释及未来唯一用途；先解释原四例/49正式波段和全部失败，再决定金融登记。不选R166好时钟、不放宽预算、不救R165。"
    state=copy.deepcopy(before)
    state.update({
        "updated_at":stamp,"at":stamp,"status":"research_active","goal_achieved":False,
        "latest_completed_study":OUT.relative_to(ROOT).as_posix(),
        "latest_result":(OUT/"summary.json").relative_to(ROOT).as_posix(),
        "latest_report":(OUT/"研究结果与下一步.md").relative_to(ROOT).as_posix(),
        "latest_research_status":budget["status"],"latest_progress":SUMMARY,"latest_overall_summary":SUMMARY,
        "latest_technical_decision":"TECH.R167","current_study":budget["study"],
        "current_phase":"ALL_EVENT_CLOCK_CAPITAL_WEIGHT_AND_ORIGINAL_BUDGET_EXPLANATION_COMPLETED",
        "current_direction":"联合时钟与原预算解释完成；下一日线因果高低结构仅提案，金融策略未登记。",
        "current_priority":action,"next_available_action":action,"next_research_action":action,
        "next_research_plan":proposal,"next_financial_experiment":"NOT_REGISTERED_NO_READY_CANDIDATE",
        "next_candidate_field_status":"DIFFERENT_DAILY_CONFIRMED_STRUCTURE_EXPLANATION_PROPOSED_NOT_ADMITTED",
        "current_admitted_unrun_numeric_candidates":0,
        "latest_joint_birth_clock_explanation":(CLOCK/"summary.json").relative_to(ROOT).as_posix(),
        "latest_joint_birth_allocation_identity":(CLOCK/"allocation_identity_summary.json").relative_to(ROOT).as_posix(),
        "latest_joint_budget_attribution":(OUT/"summary.json").relative_to(ROOT).as_posix(),
        "next_trend_expansion_clock_explanation_status":"COMPLETED_FIXED_R166_NO_FILTER_SELECTION",
        "new_accounts_this_continuation":0,"new_accounts_in_current_phase":0,
        "new_financial_candidate_accounts_this_continuation":0,"new_investment_account_evaluations_this_continuation":0,
        "new_point_replays_this_continuation":0,"internal_reference_replays_this_continuation":0,
        "saved_account_controls_replayed_this_continuation":0,"saved_original_entry_quantities_verified_this_continuation":168,
        "new_model_fits_this_continuation":0,"new_model_training_labels_this_continuation":0,
        "new_training_labels_this_continuation":0,"new_market_requests_this_continuation":0,
        "necessary_tests_passed_this_continuation":3,"necessary_test_executions_this_continuation":3,
        "return_and_sharpe_changed_this_continuation":False,"return_and_sharpe_improved_this_continuation":False,
        "returns_and_sharpe_improved":False,"current_full_account_economic_gate_passed":False,
        "current_unmet_evidence":"日DIF组等权和pB较好却实际资金亏，近期四笔全负；新结构尚未解释或准入，原预算内净收益夏普改善未成立，独立与去过拟合未建立。",
        "goal_tool_status_confirmed":"active","latest_goal_tool_status_receipt":(OUT/"goal_tool_status.json").relative_to(ROOT).as_posix(),
        "independent_validation_status":"NOT_ESTABLISHED","overfitting_removed":False,"overfit_removed":False,
        "whole_model_overfitting_removed":False,"global_DSR_PBO":"NOT_COMPUTED",
        "previous_goal_turn_classification":"progress","current_goal_turn_classification":"progress",
        "previous_goal_turn_classification_reason":"R163—R165真实四账户失败和全体执行归档，改变下一问题。",
        "current_goal_turn_classification_reason":"R166全体时钟、36资金恒等式及R167原170决定/168买入复现揭示等权与现金分歧及预算来源，排除好分组过滤/机械配仓，明确独立日线结构待解释。",
        "blocked_audit_count":0,"consecutive_blocked_goal_turns":0,"blocked_audit_key":None,"blocked_reason":None,
        "live_own_process_handle":None,"verified_wait":False,"orders_authorized":False,
        "current_goal_turn_actual_work":{
            "all_known_daily_rows":3488,"reused_qualified_origins":85,"reused_original_cycles":84,
            "complete_cycles_per_cost":83,"open_cycles_per_cost":1,"unexecuted_origins_per_cost":1,
            "clock_partition_counts":clocks["clock_counts"],"clock_group_cells":32,"allocation_identity_cells":36,
            "original_origin_decisions_verified":170,"original_buy_quantities_verified":168,"budget_group_cells":56,
            "necessary_tests_passed":3,"actual_prefix_checks":3,"charts":1,"clock_frozen_sources":101,"budget_frozen_sources":112,
            "new_policy_accounts":0,"new_model_fits":0,"new_training_labels":0,"new_market_requests":0,"risk_budget_changes":0,
        },
    })
    require(all(state[k]==v for k,v in preserved.items()),"解释覆盖了金融/预测/前瞻指针。")
    report=(OUT/"研究结果与下一步.md").relative_to(ROOT).as_posix()
    head="\n\n> 技术线最新解释 TECH.R167（优先于下方旧解释，金融裁决仍R165）："+SUMMARY+" [完整报告](../"+report+")。\n"
    paths=[ROOT/"docs/PROJECT_STATE.md",ROOT/"docs/PROJECT_STATE_TECHNICAL_LINE.md",
           ROOT/"docs/RESEARCH_DECISIONS.md",ROOT/"docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md"]
    originals={p:p.read_text(encoding="utf-8-sig") for p in paths}
    require(all("\n## TECH.R167：" not in text for text in originals.values()),"部分事实文件已更新，不能重复。")
    docs=[]
    for p,old in originals.items():
        first,rest=old.split("\n",1)
        content=(first+head+"\n"+rest).rstrip()+"\n\n"+ENTRY.strip()+"\n"
        p.write_text(content,encoding="utf-8")
        require(p.read_text(encoding="utf-8")==content,"事实文件保存不一致。")
        docs.append({"path":p.relative_to(ROOT).as_posix(),"sha256":digest(p)})
    write_json(STATE,state)
    restored=read(STATE)
    require(all(restored[k]==v for k,v in preserved.items()),"保存后金融/预测/前瞻指针变化。")
    write_json(OUT/"closeout_receipt.json",{
        "at":stamp,"status":"PASS_R166_R167_SINGLE_FACT_UPDATE_WITH_FINANCIAL_AND_FORWARD_PRESERVED",
        "documents":docs,"state_sha256":digest(STATE),"preserved_values":preserved,"goal_status":"active",
        "latest_technical_decision":"TECH.R167","latest_financial_decision":"TECH.R165",
        "latest_actual_prediction_decision":"TECH.R158","original_exit_decision":"TECH.R145",
        "current_turn_classification":"progress","blocked_count":0,"goal_achieved":False,
        "new_accounts_in_closeout":0,"next_financial_policy_admitted":False,
        "source":{"path":Path(__file__).relative_to(ROOT).as_posix(),"sha256":digest(Path(__file__))},
    },exclusive=True)
    print("R166—R167一次归档到四长期事实与状态；金融R165、预测R158、退出R145和十二前瞻值保持，目标active。",flush=True)


if __name__=="__main__":
    main()
