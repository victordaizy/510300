"""一次更新本轮结构解释、真实账户和四个长期事实文件；前瞻状态保持。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.confirmed_structure_study_v1 import OUT, EXPLANATION, PRIMARY
from research.point_first_passage_study_v1 import read, write_json, digest, now, require

STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FORWARD = ("forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
           "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
           "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
           "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation")
DOCS = ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md")


def run():
    require(not (OUT / "project_state_update_receipt.json").exists(), "本轮长期事实已经更新，不追加同一轮。")
    summary, explanation, delivery = read(OUT / "summary.json"), read(EXPLANATION / "summary.json"), read(OUT / "delivery_receipt.json")
    require(summary["technical_decision"] == "TECH.R170" and summary["new_primary_accounts"] == 4, "本轮实际账户未完成。")
    require(delivery["open_case_windows_clipped_at_own_account_end"], "案例观察末日修正未完成。")
    protocol = read(OUT / "protocol.json")
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "金融冻结来源发生变化。")
    state = read(STATE)
    forward_before = {k: state[k] for k in FORWARD}
    write_json(OUT / "project_state_before.json", state, exclusive=True)
    metric = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    primary = metric.loc[metric.policy.eq(PRIMARY)]
    earlier = primary.loc[primary.period.eq("2015_2019") & primary.cost.eq("STRESS")].iloc[0]
    recent = primary.loc[primary.period.eq("2020_2026") & primary.cost.eq("STRESS")].iloc[0]
    overview = (
        "2026-10-05按用户先解释具体上涨、再反推点位的顺序完成TECH.R168解释/R169登记/R170实际账户。"
        "严格日收盘2/2因果交替枢轴、两高两低抬升及已知失效；3488原点、941确认到达、111出生/失效、原61分段/49正式波段和四案例完整保留。"
        "2019-01-23可确认慢涨结构，2020修复多次重建，2024急涨直到10-21才形成完整结构，2015放量反弹未形成；不是事后高低点入出。"
        "四输入/三账户测试最终通过，四真实前缀、八A/纯价格对照精确复现，四新账户一次完成。"
        f"压力较早52完成/1开放，胜率{earlier.win_rate:.2%}、B{earlier.payoff:.3f}/pB{earlier.p_times_b:.3f}、净年化{earlier.net_cagr:.4%}/夏普{earlier.net_sharpe:.4f}/DD{earlier.max_drawdown:.2%}；"
        f"近期58完成、{recent.win_rate:.2%}、B{recent.payoff:.3f}/pB{recent.p_times_b:.3f}、{recent.net_cagr:.4%}/{recent.net_sharpe:.4f}/DD{recent.max_drawdown:.2%}。"
        "完整年均次数10.4/8.83，四经济门及历史稳定门全败，固定政策关闭不营救；近期毛损益已负，非仅费用。"
        "222资格/222实际周期原值及四案例全体记录完成，2019开放周期只观察到自身末日；首次测试失败与案例跨期派生错误保留并修正，未改金融逻辑或重跑。"
        "32解释/71金融来源保持，0拟合/新训练标签/采集。最新技术与金融R170、预测R158、原退出R145；收益夏普目标未达、独立和去过拟合未建立，目标active/阻塞0。"
        "下一下降结构首次破坏的全体解释仅提案，先核旧完整定义；金融用途未定义/登记，已准入待跑0，原十二前瞻值和其他分支保持。"
    )
    marker = "> 技术线当前事实 TECH.R170（优先于下方所有技术线历史快照，2026-10-05）："
    top = marker + overview + " [具体上涨解释、全部点位与完整账户](../reports/research/510300_confirmed_structure_study_v1/研究结果与下一步.md)。\n\n"
    technical_state = """
## TECH.R168—R170：具体上涨解释、确认高低结构与完整账户（2026-10-05）

本轮先按原四案例2019/2020/2024上涨及2015反弹失败解释日线量价、MACD和RV，再进行完整账户测量。日线严格2/2中心只在b+2确认，交替尾点同类仅更极端时向前替换，历史快照不重写；最近两高、两低同时抬升且当前收盘高于最新低点才成立，已知未成立至成立出生。次真实开尝试，已知结构失效或跌到确认低点次合法开退出，未知不迫使卖出，原预算只减，无加仓/A混合/固定2R或20日。

原3488状态/941确认到达、2015起2855原点/111出生、63低点跌破及48抬升结构失效路径、61分段/49正式波段均保留。49正式波段中27段确认后至高点无结构成立、29段无新出生；这不是27/29个盈利机会。原2846结果标签（2826成熟/20删失）及9尾部无标签不变。旧20日描述pB0.502不是本结构失效账户的训练标签或收益结果。

具体2019-01-23成立且相对量0.863/量方向−0.031，后续两次重建；2020-03-09先失败，04-01/04-27/06-17重建，06-17量仅0.987而后段仍上行；2024-09-24量3.338/日柱正但上一完整周柱负，至10-08急涨峰值仍缺完整结构，10-21才成立；2015-06-30量2.271但日周柱负、波动约2.7倍、下降结构仍在。这些是路径解释，不是因果上涨证明或反选过滤。

原A04回调比例预测、3/3 OHLC/ATR/DIF配对背离及海龟、完整周支撑收复2R/20日的完整用途已有限核对。原A04和周支撑实际拒绝保持；旧图谱本次只核定义，不凭源码或旧config冒称当前终态。交替枢轴组件已使用，不能声称全项目未试；新完整双抬升成立/失效用途另立一次开发测量，非原预测MSE补救。

四输入及三实际账户行为测试最终通过，四真实前缀和八控制精确复现；四新主账户一次运行，保存资金/库存/T+1/指标核对通过。初次日期精度表示失败保留并统一ns；未知报价测试原错误继承完整报价高点的第26日退出预期，实际该高点确认窗口有缺失，改测试后续明确已知跌破、第28日退出，输入/引擎不改。案例相交派生表曾把2019开放周期延伸入2020/2024，原表保留并截断到自身观察末日，原金融账本/指标不变。

压力较早52完成/1开放：胜率25%、B2.280189、pB0.570047、净年化−0.232759%、夏普−0.106075、DD9.0988%；近期58完成：胜率15.5172%、B2.294740、pB0.356080、净年化−0.552442%、夏普−0.268155、DD9.6111%。完整年均10.4/8.833次，原A为4.4/4.167次；次数增加但收益夏普均不及A。四场景经济门与两尺度/两比较稳定门全部失败，固定政策拒绝并关闭。较早毛+2376.70/摩擦4618.51元；近期毛−2261.90/摩擦4803.16元，不能只归因费用。均未停机，2019期末开放不人工清仓。两费用222资格/222实际周期行含220完成/2开放；其中压力111周期含110完成/1开放。

0模型拟合/新训练标签/采集，32解释及71金融来源保持；真实前瞻十二值原样、最新预测R158及原退出R145保持。全部历史DEVELOPMENT_CALIBRATION，first-vintage NOT_CERTIFIED、独立NOT_ESTABLISHED、global DSR/PBO NOT_COMPUTED；完整收益夏普/去过拟合未达，目标active、本轮progress/阻塞0。其他宏观/盘口分支不改变。

下一具体实验仅提案：继续原2/2确认时钟，先解释下降结构最后一个已确认高点首次被收盘突破，核旧快速结构/供给测试/简单量反转/海龟完整用途后再决定准入；相同完整用途已终态则停止。四例、原61/49段及全体失败保留，量价只观察，完整金融退出/参考标签未定义或登记；已准入待跑0。不缩确认、加门槛、混A或放宽原预算救R170。

依据：[具体路径与完整结果](../reports/research/510300_confirmed_structure_study_v1/研究结果与下一步.md)、[R168解释](../reports/research/510300_confirmed_structure_explanation_v1/summary.json)、[固定金融定义](../reports/research/510300_confirmed_structure_study_v1/protocol.json)、[实际金融结果及区间](../reports/research/510300_confirmed_structure_study_v1/summary.json)、[全部点位](../reports/research/510300_confirmed_structure_study_v1/results/全部实际进出点位_已知高低结构与A覆盖.csv)、[下一仅解释提案](../reports/research/510300_confirmed_structure_study_v1/next_structure_break_explanation_proposal.json)。
"""
    technical_decisions = """
## TECH.R168：先具体上涨及失败，再解释当时已确认高低结构（2026-10-04至05）

**假设**：持续上涨可能表现为事前已确认的两高、两低逐次抬升；须先解释真实路径和失败，不从事后低高点入场。

**验证方法**：原严格日收盘2/2及已发生现金分红平移；确认到达序列交替，同类只向前替换当前更极端尾点，过去不重写。两高与两低同时抬升、收盘高于最新低点为成立，已知未成立至成立为出生；不足四点已知未成立，报价/分红未知NO_VIEW，恢复不伪造出生。唯一未来出生次开、失效次合法开政策在连接旧标签前固定。有限核原A04预测、旧交替枢轴/ATR/DIF背离和周支撑完整触发退出。四输入测试最终通过，四真实前缀一致，32来源冻结后一次解释全部3488状态/941确认、2855当期原点、111出生/失效、61/49分段、四图及全部失败，0账户/拟合/新训练标签/采集。

**结果**：111收盘描述路径有63低点跌破、48非抬升失效。2019-01-23可识别慢涨；2020有03-09失败、04-01/04-27/06-17重建；2024主升内无完整结构、10-21才成立；2015一日放量反弹仍下降。49正式段有27无确认后结构成立、29无新出生，不能称盈利点。原20日标签111出生pB0.502、均值+0.0686%，近期分段均值负；原2846/2826/20及9尾部无标签全部保留。初次日期dtype前缀不一致只统一ns并保留失败源码。

**为什么接受/拒绝**：接受因果时钟、具体量价和确认滞后解释；没有接受策略优势、因果上涨解释或指标过滤。枢轴组件有旧用途，新完整HH/HL成立与失效不是旧回调比例预测、背离或周支撑2R；有限核对不声称全项目穷尽。旧图谱只核定义，不从config声明实际当前终态；原A04及周支撑拒绝保持。

**是否需要重新验证**：本固定解释完成。原定义只遇来源或实现真实错误才复核，不换窗口或据旧20日标签选政策。完整金融用途由R169登记/R170实际履行，独立验证仍未成立。

## TECH.R169：唯一确认高低结构的完整账户用途登记（2026-10-05）

**假设**：当时已确认双高、双低同时抬升的出生与已知失效，可提供独立于MACD零轴/完整周同步的完整入出时钟，需直接检验收益夏普。

**验证方法**：R168唯一用途不变；空仓出生次开一次尝试、已知双抬升失效或收盘<=最新确认低点次合法开退出，未知不退出、风险只减、无加仓/混A/固定2R/20日。原20万元/252日/两时期两费用/50%/ES/跳空/剩余DD/T+1/股息/期末口径；经济门净CAGR和净Sharpe同时>0且优于A与纯价格、DD<=10%、实际pB>1及pB−q>0，次数软目标；20/252配对区块各2000种子510300154全比较。四输入测试不无故重跑，三新账户测试最终通过，八保存控制日账/订单/周期精确复现；71來源在新账户结果前冻结。未知报价测试原错误预期及原源码保留，只补后续明确跌破路径，不改输入/引擎逻辑。原退出MSE/成员门只约束原用途。

**结果**：一配置、四新主账户计划，金融来源固定；登记已由R170全部履行，不是待跑任务。0模型拟合/新训练标签/采集；原失败和真实前瞻保持。

**为什么接受/拒绝**：接受一次独立完整开发测量，不接受收益优势或独立样本资格；原111旧20日弱结果保留但不是状态失效政策训练门。不把交替枢轴说成未用，不由结果添加门槛或调窗口。

**是否需要重新验证**：无需重新登记或重跑；实际结果见R170。本登记不授权实盘、其他资产、分钟或期权收益。

## TECH.R170：实际四账户拒绝，次数提高而收益夏普未提高（2026-10-05）

**假设**：R169唯一完整双抬升政策须在全部四场景中改善净收益、全日历夏普及实际交易质量，而非仅解释几个赢家。

**验证方法**：四新账户一次运行，八控制直接复用原精确核验账本；资金/库存、次开/T+1、成本/股息和保存指标核对全部通过。111出生两费用222资格/222周期全体保留，220完成/2开放；压力111实际周期110完成/1较早期末开放，不人为末日清仓。日线量价和H/L确认原值、原A库存/已知目标、四案例全部相交周期公开，不以未来盈亏筛选。案例派生表初次开放周期跨期错误保留，改截断到自身账户末日，金融账本不变。

**结果**：压力较早52完成/1开放，胜率25%、B2.280189/pB0.570047、均值−0.2842%、净年化−0.232759%/夏普−0.106075/DD9.0988%；近期58完成，15.5172%、B2.294740/pB0.356080、均值−0.7867%、−0.552442%/−0.268155/DD9.6111%。BASE较早−0.0372%/−0.0071/pB0.6067、近期−0.4911%/−0.2320/pB0.3946，同样失败。完整年均次数10.4/8.833（原A4.4/4.167），四经济门和历史稳定门全败；最近毛损益已负，两期未停机。

具体压力2019-01-24至03-11+16.25%、2020-04-02至04-27+3.27%、06-18至07-27+10.46%真实存在，但2020-03-10至03-13−4.26%及其他失败完整计入。2024-10-22至10-28+0.48%属于急涨后的点，不能继承09-13至10-08主升。原2015-06-11至06-16−2.38%也保留。较早毛+2376.70/摩擦4618.51元；近期毛−2261.90/摩擦4803.16元，费用不是全部解释。

**为什么接受/拒绝**：拒绝固定完整双抬升政策作为收益夏普改进，保留具体上涨解释及真实赢家事实；不从单例加量/MACD过滤、缩短确认、混合A、放宽预算或再调失效营救。原A及所有旧失败/其他分支保持。全部历史开发复用、first-vintage未认证，独立和去过拟合未建立，global DSR/PBO未计算，完整目标仍未达到。

**是否需要重新验证/下一步**：本政策关闭，无同数据重跑或参数救援。只有实质新信息、真实新样本或来源/实现真实错误才可独立重验并保留本终态。下一仅解释下降结构首次被破坏：原2/2时钟下上一原点最后确认下降高点首次被收盘突破，先核旧完整用途、再四例/原61/49段/全体失败；完整金融入出未定义或登记，已准入待跑0。目标active，本轮progress/阻塞0；最新技术与金融R170，实际预测R158、原退出R145，原前瞻十二值不变。

依据：[完整路径、实际点位及结果](../reports/research/510300_confirmed_structure_study_v1/研究结果与下一步.md)、[R168固定解释](../reports/research/510300_confirmed_structure_explanation_v1/protocol.json)、[R169固定金融口径](../reports/research/510300_confirmed_structure_study_v1/protocol.json)、[R170实际结果](../reports/research/510300_confirmed_structure_study_v1/summary.json)、[保存核对](../reports/research/510300_confirmed_structure_study_v1/saved_result_verification.json)、[下一仅解释提案](../reports/research/510300_confirmed_structure_study_v1/next_structure_break_explanation_proposal.json)。
"""
    routing = "\n## 技术线TECH.R168—R170更新（2026-10-05）\n\n" + overview + "\n\n完整假设→验证→结果→处置→重验条件见[技术线账本](RESEARCH_DECISIONS_TECHNICAL_LINE.md)，共享宏观/盘口分支范围与终态保持。本轮技术金融登记R169已经由R170履行，不能继续列为待运行。\n"
    for name in DOCS:
        path = ROOT / "docs" / name
        text = path.read_text(encoding="utf-8-sig")
        require(marker not in text, "事实文件已包含本轮，拒绝重复更新。")
        first, remainder = text.split("\n", 1)
        append = technical_decisions if name == "RESEARCH_DECISIONS_TECHNICAL_LINE.md" else technical_state if name == "PROJECT_STATE_TECHNICAL_LINE.md" else routing
        path.write_text(first+"\n\n"+top+remainder.lstrip("\n")+"\n"+append, encoding="utf-8", newline="\n")
    out_relative = OUT.relative_to(ROOT).as_posix()
    explanation_relative = EXPLANATION.relative_to(ROOT).as_posix()
    next_action = "先有限核旧快速结构/日线供给测试/简单量价反转/海龟的完整用途和实际终态，再固定下降结构最后确认高点首次被收盘突破的全体解释；四例、原61/49段及全部失败保留，完整金融用途尚未定义/登记。R170固定双抬升关闭，不改窗口/指标阈值/退出/风险预算营救。"
    state.update({
        "updated_at": now(), "latest_completed_study": out_relative, "latest_result": out_relative+"/summary.json",
        "latest_report": out_relative+"/研究结果与下一步.md", "latest_research_status": summary["status"],
        "latest_progress": overview, "latest_overall_summary": overview, "latest_continuation_outcome": overview,
        "current_study": "510300_CONFIRMED_STRUCTURE_STUDY_V1", "current_phase": "CONFIRMED_STRUCTURE_EXPLANATION_AND_FIXED_FULL_ACCOUNT_REJECTED",
        "current_direction": "具体四例及全体结构解释完成，双抬升完整账户失败关闭；下一下降结构首次破坏仅解释提案。",
        "current_priority": next_action, "next_research_action": next_action, "next_available_action": next_action,
        "next_research_plan": out_relative+"/next_structure_break_explanation_proposal.json",
        "next_experiment_status": "DIFFERENT_STRUCTURE_BREAK_EXPLANATION_PROPOSED_BEFORE_FULL_POLICY_DEFINITION",
        "next_candidate_field_status": "DIFFERENT_STRUCTURE_BREAK_EXPLANATION_PROPOSED_NOT_ADMITTED",
        "next_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "next_strategy_increment_status": "NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY", "current_admitted_unrun_numeric_candidates": 0,
        "latest_technical_decision": "TECH.R170", "latest_actual_model_decision": "TECH.R170",
        "latest_financial_registration_decision": "TECH.R169", "latest_financial_strategy_decision": "TECH.R170",
        "latest_financial_strategy_result": out_relative+"/summary.json",
        "latest_actual_model_kind": "COMPLETE_RULE_POLICY_WITHOUT_ESTIMATED_PREDICTION_MODEL",
        "latest_actual_prediction_model_decision": "TECH.R158", "latest_original_exit_decision": "TECH.R145",
        "latest_continuation_receipt": out_relative+"/delivery_receipt.json",
        "latest_confirmed_structure_explanation": explanation_relative+"/summary.json",
        "latest_confirmed_structure_protocol": out_relative+"/protocol.json",
        "latest_confirmed_structure_saved_verification": out_relative+"/saved_result_verification.json",
        "economic_stage_status": "COMPLETED_REJECTED_FIXED_CONFIRMED_STRUCTURE_ALL_FOUR_GATES_FAILED",
        "necessary_tests_passed_this_continuation": 7, "necessary_tests_passed_in_current_phase": 7,
        "actual_prefix_checks_this_continuation": 4, "new_accounts_this_continuation": 4, "new_accounts_in_current_phase": 4,
        "new_investment_account_evaluations_this_continuation": 4, "internal_reference_replays_this_continuation": 8,
        "saved_accounts_checked_this_continuation": 4, "new_model_fits_this_continuation": 0,
        "historical_model_refits_this_continuation": 0, "new_return_labels_this_continuation": 0,
        "new_allocation_method_admitted": False, "new_method_admitted": "ONE_FIXED_COMPLETE_CONFIRMED_STRUCTURE_RULE_MEASUREMENT_EXECUTED_AND_REJECTED",
        "new_market_information_admitted": False, "new_market_requests_this_continuation": 0,
        "original_strategy_source_files_changed_this_continuation": 0, "code_files_changed_this_continuation": 0,
        "code_files_added_this_continuation": ["research/confirmed_structure_inputs_v1.py", "research/confirmed_structure_explanation_v1.py",
            "research/confirmed_structure_account_v1.py", "research/confirmed_structure_study_v1.py", "research/deliver_confirmed_structure_results_v1.py",
            "research/finalize_confirmed_structure_state_v1.py", "tests/test_confirmed_structure_inputs_v1.py", "tests/test_confirmed_structure_account_v1.py"],
        "current_goal_turn_classification": "progress", "previous_goal_turn_classification": "progress", "blocked_audit_count": 0,
        "blocked_reason": None, "blocked_audit_key": None, "blocking_decision": None,
        "goal_tool_status_confirmed": "active", "goal_achieved": False, "whole_model_overfitting_removed": False,
        "current_unmet_evidence": "交易次数增加但R170四场景完整净收益/夏普及实际pB门均失败；下一方向仅解释提案，独立样本及去过拟合仍未建立。",
        "validation_method_this_continuation": "四输入/三账户测试最终通过，四真实前缀，八原A/价格对照精确复现，四新账户一次运行及保存核对；首次测试与案例派生错误保留、修正不改金融逻辑。",
    })
    require({k: state[k] for k in FORWARD} == forward_before, "真实前瞻状态意外变化。")
    write_json(STATE, state)
    write_json(OUT / "project_state_update_receipt.json", {
        "at": now(), "status": "PASS_FOUR_DURABLE_FACT_FILES_AND_CURRENT_STATE_UPDATED_ONCE",
        "goal_status": "active", "goal_achieved": False, "latest_technical_and_financial": "TECH.R170",
        "actual_prediction_preserved": "TECH.R158", "original_exit_preserved": "TECH.R145",
        "forward_values_unchanged": forward_before, "admitted_unrun_candidates": 0,
        "original_strategy_source_changes": 0, "new_primary_accounts": 4, "control_replays": 8,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in
                    [Path(__file__), STATE, *(ROOT / "docs" / n for n in DOCS)]],
    }, exclusive=True)
    print("四个长期事实文件及当前状态已一次更新至R170；原十二前瞻值不变，目标继续active。", flush=True)


if __name__ == "__main__":
    run()
