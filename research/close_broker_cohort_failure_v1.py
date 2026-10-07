"""把一次固定账户实验与必要归因写入长期事实，保留所有旧正文和前瞻。"""
from __future__ import annotations

import json

from research import broker_cohort_failure_study_v1 as study
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import prepared_prepend, digest

ROOT, OUT = study.ROOT, study.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def main():
    if (OUT / "project_state_update_receipt.json").exists() or (OUT / "state_before_TECH_R216.json").exists():
        raise RuntimeError("本金融结果已经归档或开始，不重复写入。")
    summary = study.read(OUT / "summary.json")
    preflight = study.read(OUT / "control_preflight.json")
    delivery = study.read(OUT / "delivery_receipt.json")
    diagnosis = study.read(OUT / "post_run_diagnosis.json")
    viewed = study.read(OUT / "figure_delivery_view_receipt.json")
    service = study.read(OUT / "goal_service_status_after_result.json")
    protocol = study.read(OUT / "protocol.json")
    if (summary["decision"] != "TECH.R216" or summary["new_accounts"] != 12
            or summary["saved_account_checks"] != 12 or summary["necessary_tests_passed"] != 8
            or delivery["saved_accounts_recomputed"] != 12 or delivery["all_saved_metric_rows_recomputed"] != 20
            or viewed["viewed"] != 4 or preflight["original_R212_replays"] != 4
            or len(diagnosis["four_scenario_full_financial_equivalence"]) != 4
            or service["goal"]["status"] != "active" or summary["goal_achieved"] is not False):
        raise ValueError("实际金融、保存复算、图形、归因或目标服务状态不一致。")
    if summary["all_economic_gates_passed"] or summary["historical_stability_passed"]:
        raise ValueError("实际结果与拒绝归档不一致。")
    frozen = []
    for item in protocol["sources"]:
        actual = study.digest(ROOT / item["path"])
        if actual != item["sha256"]:
            raise ValueError("本用途冻结代码、源或原账户改变：" + item["path"])
        frozen.append({"path": item["path"], "sha256": actual})
    old_state = STATE.read_bytes()
    state = json.loads(old_state.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R214" or state["latest_actual_financial_decision"] != "TECH.R212":
        raise ValueError("项目事实已被其他研究推进，先核实。")
    forward = {key: state[key] for key in FORWARD}
    report = study.relative(OUT / "真实进入后固定组失效_完整结果与反例.md")
    result = study.relative(OUT / "summary.json")
    proof = study.relative(OUT / "goal_service_status_after_result.json")
    explanation = study.relative(OUT / "post_run_diagnosis.json")
    note = f"""> 最新完整金融事实（2026-10-05，TECH.R215—R216，优先于下方R214假设未跑记录）：真实成交后固定价格组，仅第5个观察收盘一次检查，两组均未过半上涨且同期ETF总回报非正时额外退出；原进入/宏观/周线/风险/费用/股息/T+1全部保持。12新账户一次完成，原R212四账户五账表及终态精确复现，八必要测试、12保存新账户及20指标行复算通过，四完整账户图已逐幅查看。当前固定用途拒绝：四经济门0/4、稳定失败、所有主政策实际pB<1；独立及去过拟合未建立。目标服务实际active，本轮PROGRESS，连续受阻0；完整收益夏普目标未达，旧策略/失败和13前瞻逐值保持。

压力同口径：2015—2019主政策净CAGR1.6483%/夏普0.41208/DD6.3228%/28完成、11胜17负/B2.35037/pB0.92336/标准期望0.31622/完整年均5.6，低于原阶段2.1871%/0.52919及原A1.8371%/0.43804。2020—2026主政策0.0125%/0.01892/DD7.2653%/47完成、13胜34负/B2.76590/pB0.76504/标准期望0.04163/完整年均7.167；原阶段−0.2400%/−0.06608/44完成/年均6.667，原A3.9908%/1.21691/32完成/年均4.167。基础/压力全部20行及四对照20/252块各2000次区间保留，不择期/择尺度。

最重要发现：主政策与同成分覆盖、只看ETF价格的对照，四场景六账表及终态规范政策名称后精确相同；已知源且原退出未触发的价格失效，较早1/近期11次全部两组弱势，成分方向没有额外区分。较早第五日源已知11/未知12；全日期价格对照额外6次，主政策只1次，说明覆盖会影响政策；近期已知32/未知2，三政策金融值相同。未知不填零，不降覆盖；该结果不否定所有成分或主线信息。

退出反例：2019-06-12修复，原07-10退出盈利1989.10元，新06-20提前退出亏50.28392元，还释放资金07-02新买亏2539.48748元。较早完整账户终值少5606.82388元，包含后续份额与未平仓变化，不能将首笔差当全部。近期11额外退出中10原亏损：8截短、2反更差；另1原盈利的2025-06-26恢复，原净+2.6999%被截为+0.0524%。2024-09-24放量重新定价仍09-25进入、10-17退出、净周期12.1365%，局部成功不证明全期。

费用与风险：近期实际数量毛7713.70元/摩擦7550.94788元/净162.75212元；加回摩擦固定数量年化0.5846%，仍远低于原A扣费3.9908%，不能只归因于费用。主政策/原阶段无实际拒单或10%DD停买，原风险预算份额与所有现金日保持。较早主政策1未完成、open_pnl2750.9388元入完整NAV；近期未平仓0。原Aopen_pnl与拒单表未保存，本轮显式UNKNOWN；完整NAV包含自然持仓。最坏日/周期、集中度、暴露、最长空仓、成交、股息应收和逐年统计全部保留。

下一研究问题：在ETF和多数成分同时走弱后，如何用不同的当时信息区分恢复与继续下跌；当前仅加同义投票没有增量。优先解释主线对沪深300的传导、集中与轮动，先核历史分类/成员、实际披露/源钟，产业或参与者结构不能事后选赢家。当前没有已准入待跑新用途；不改五日/阈值/源钟/路线或费用救援本配置。券商框架可允许行为按状态变化，但状态及行动须当时可知，小资金容量优势不代替预测收益。

实现记录：登记前首个原R212核验因0行0列空表保存后的列索引类型不同终止；只在核对双方空表时规范索引，0新候选账户，无策略变化，另四成功原核验；失败回执保持。报告层原A拒单0更正UNKNOWN，不改任何冻结金融表。首版NOT_CERTIFIED、独立NOT_ESTABLISHED、全项目搜索选择校正NOT_COMPUTED。

依据：[完整结果及全部反例](../{report})；[固定用途卡](510300_BROKER_COHORT_FAILURE_V1.md)；[全部指标与区间](../{result})；[重复信息与退出归因](../{explanation})；[目标服务实际active](../{proof})。
"""
    decisions = f"""### TECH.R215—R216：真实进入后固定组失效的完整账户与重复信息（2026-10-05）

R215唯一配置登记，R21612账户一次完成。主政策与同覆盖价格、全日期价格两个归因对照；原A仅读取保存账户、原R212四精确复现。8必要测试、12保存新账户/20行指标复算、四图已查看。旧R214观察结论和旧策略/失败保留，不能把已知信号锚点13亏损当成本策略收益。

| 方向 | 假设 | 验证方法 | 实际结果 | 为什么接受/拒绝 | 是否需要重新验证 |
|---|---|---|---|---|---|
| 五日固定组与价格共同失效 | 成交后多数成员未响应可及时止损，提高完整收益夏普 | 实际进入前一源日定组，单次第5观察收盘、前一源日五个回报；原规则/两价格对照/原A，两时期两费用 | 四经济门0/4、稳定失败；较早压力1.6483%/0.41208/pB0.92336，近期0.0125%/0.01892/pB0.76504 | 拒绝本固定用途晋升；次数增加而质量不达标，较早反受损，所有主政策pB<1 | 不调整五日/覆盖/符号/费用营救；不同完整机制另登记 |
| 成分方向提供价格以外的信息 | 两组是否共同弱可排除价格假失效 | 同源覆盖、同进入/观察时刻/原退出，仅移除组方向 | 四场景六账表与终态规范标签后精确相同；可执行已知价格失效1/11全部两组弱 | 接受信息重复事实，拒绝当前成分方向增量；不推成所有主线信息无效 | 不继续同义投票；不同传导/结构信息须新用途 |
| 覆盖门无影响 | 去掉成分资格只影响描述，不影响收益 | 全日期仅价格退出对照，未知不补 | 较早23检查、11已知/12未知；主额外1，全价格6；全价格压力2.5610%/0.69234/pB1.13303，近期三政策相同 | 拒绝覆盖无影响；保留全价格局部点值，非策略晋升，不能改主政策为赢家 | 若不同完整用途，覆盖效应与方向效应须分开、完整时期及独立验证 |
| 早退只节省亏损 | 价格/组同弱即可丢弃无继续价值的交易 | 全部额外实际退出与原同进入上下文、后续资金路径 | 较早2019-06-12原+1989.10转新−50.28，新增07-02亏2539.49；近期10原亏8改善2变差，1原盈利被截 | 拒绝只省亏假设；全期回报不能由单笔差相加，继续价值不充分 | 是，用实质不同的当时信息区分弱势后恢复与继续下跌 |
| 数量增加即可提高目标 | 放出资金产生更多机会有利 | 完整周期次数、pB、全账户风险费用同时验收 | 早/近压力完成28/47，完整年均5.6/7.167；pB均<1 | 接受次数事实，拒绝以次数代替质量 | 次数软目标，须净收益/夏普及pB联合通过 |
| 费用或执行是唯一问题 | 小资金降低摩擦就能满足目标 | 实际数量毛/费/净拆解、拒单与DD停买记录 | 近期毛7713.70、摩擦7550.95、净162.75；固定数量无费年化0.5846%，仍低于A扣费3.9908%；主/原阶段无拒单及停买 | 拒绝仅费用或成交障碍解释，不抬仓/降费救援 | 未来信息改进仍以原冻结成本及风险评估 |
| 2024快速重新定价 | 强放量突破在宏观/周线慢确认前可进入且持有 | 原2024-09-24已知量/价格/日柱、次开买入，原确认和结构退出 | 主与对照仍09-25至10-17、净完成12.1365% | 接受具体实际周期；不能证明全期、因果或独立 | 全部反例和真实独立证据仍要求 |
| 主线传导及适应行为 | 产业/资金/博弈/周期的不同信息可以区分继续价值 | 先以原上涨和失败逐日说明、当时分类/成员/披露钟合同，再立完整对照 | 本轮未构造行业主线因子，金融NOT_RUN；现有分类钟尚不足直接准入 | 接受下一研究问题，不把券商叙事或小资金直接当收益证据 | 是，先取得不同可知信息与完整用途，未知保留 |

全部历史开发/校准，五日受已知R214启发，非盲测。原Aopen_pnl和拒单表本轮未知，报告未知更正不改变冻结金融。登记前无列空表核验失败保存，4成功核验之外另1失败尝试。独立/去过拟合未建立、首版未认证、搜索选择校正未计算。服务实际active、PROGRESS、受阻0，原13前瞻逐值保持。

依据：[完整结果](../{report})、[用途卡](510300_BROKER_COHORT_FAILURE_V1.md)、[所有区间](../{result})、[必要归因](../{explanation})。
"""
    prepared = []
    for name, content in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
            ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)):
        path = ROOT / f"docs/{name}.md"
        prepared.append((path, *prepared_prepend(path, content)))
        if (OUT / f"{name}_before_TECH_R216.md").exists():
            raise RuntimeError("本次事实备份已存在。")
    state["previous_phase_before_TECH_R215_R216"] = {key: state.get(key) for key in (
        "latest_technical_decision", "latest_registration_decision", "latest_result", "latest_report", "current_phase",
        "latest_actual_financial_decision", "latest_actual_financial_result", "current_phase_trial_accounting",
        "next_fixed_cohort_financial_hypothesis", "next_isolated_strategy_design")}
    pressure = [row for row in summary["metrics"] if row["cost"] == "STRESS" and row["policy"] == study.candidate.POLICIES[0]]
    progress = "TECH.R215—R216一次12新账户、原R212四精确核对、八测试、12保存/20指标复算、四图查看；成分与同覆盖价格完全重复，四门0/4、稳定失败；早退有省亏和误伤，完整目标未达。"
    state.update({"updated_at": study.parent.original.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_goal_service_status": "active", "goal_tool_status_confirmed": "active", "latest_goal_service_status_observed_at": service["at"],
        "latest_goal_tool_status_receipt": proof, "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "blocked_key": None, "blocked_reason": None, "blocked_audit_key": None, "blocked_scope": None,
        "goal_turn_classification": "PROGRESS_R215_R216_TWELVE_ACCOUNTS_AND_INFORMATION_REDUNDANCY_DIAGNOSIS",
        "current_goal_turn_classification": "progress", "latest_technical_decision": "TECH.R216", "latest_registration_decision": "TECH.R215",
        "latest_actual_financial_decision": "TECH.R216", "latest_actual_financial_result": result,
        "latest_result": result, "latest_report": report, "latest_completed_study": study.relative(OUT),
        "latest_research_status": summary["status"], "current_study": "510300_BROKER_COHORT_FAILURE_V1",
        "current_phase": "FIXED_COHORT_HOLDING_FAILURE_COMPLETED_REJECTED_INFORMATION_REDUNDANCY",
        "latest_progress": progress, "latest_overall_summary": progress, "latest_continuation_outcome": progress,
        "new_accounts_in_current_phase": 12, "necessary_tests_passed_in_current_phase": 8,
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "latest_long_point_metrics": {"scope": "R216_PRIMARY_STRESS_COMPLETE_CYCLES_DEVELOPMENT_NOT_CURRENT_MARKET", "STRESS": pressure},
        "current_financial_candidate_admission": "FIXED_COHORT_FAILURE_REJECTED_NOT_PROMOTED",
        "current_phase_trial_accounting": {"scope": "TECH_R215_R216_SINGLE_FAILURE_POLICY_AND_TWO_PRICE_CONTROLS",
            "new_accounts": 12, "original_R212_exact_replays": 4, "pre_registration_original_attempts": 5,
            "pre_registration_failed_attempts": 1, "original_A_new_replays": 0, "new_fits": 0, "new_labels": 0,
            "new_market_requests": 0, "necessary_tests_passed": 8, "saved_accounts_recomputed": 12,
            "all_metric_rows_recomputed": 20, "figures_viewed": 4},
        "current_goal_turn_actual_work": {"new_accounts": 12, "original_R212_replays": 4, "new_fits": 0,
            "new_labels": 0, "new_market_requests": 0, "saved_accounts_recomputed": 12,
            "actual_full_financial_equivalence_checks": 4, "post_run_diagnosis": explanation},
        "next_fixed_cohort_financial_hypothesis": {"status": summary["status"], "registration": "TECH.R215", "decision": "TECH.R216",
            "new_financial_run": "COMPLETED_TERMINAL_REJECTED", "new_accounts": 12},
        "next_isolated_strategy_design": {"hypothesis": "成交后单次五日固定组与价格共同失效",
            "status": summary["status"], "registration_decision": "TECH.R215", "result_decision": "TECH.R216",
            "current_financial_run": "TERMINAL_REJECTED", "old_results_and_strategies_preserved": True},
        "next_research_question": "ETF与多数成员共同走弱后，如何以不同当时信息区分恢复/继续下跌：主线向沪深300传导、集中与轮动，先核历史分类/成员/真实披露源钟，全部反例与完整用途后再金融。",
        "next_information_source_proposal": "行业主线金融仍NOT_ADMITTED；不能用生成分类生效钟当真实首次披露，也不把缺失补为0。先核不同信息来源与用途，不继续同义投票或调五日营救。",
        "current_unmet_evidence": "当前配置所有主pB<1、成分方向无增量、较早受损；完整收益夏普、历史稳定和独立未达到。不同主线传导用途尚未准入。",
        "latest_holding_failure_diagnosis": explanation, "independent_validation": "NOT_ESTABLISHED",
        "overfitting_removed": False, "whole_model_overfitting_removed": False, "historical_search_selection_correction": "NOT_COMPUTED"})
    if {key: state[key] for key in FORWARD} != forward:
        raise ValueError("拟写入事实改变了原13前瞻字段。")
    with (OUT / "state_before_TECH_R216.json").open("xb") as stream:
        stream.write(old_state)
    docs = []
    for path, old, new, body in prepared:
        with (OUT / f"{path.stem}_before_TECH_R216.md").open("xb") as stream:
            stream.write(old)
        path.write_bytes(new)
        if path.read_bytes() != new or not new.endswith(body):
            raise ValueError("旧事实正文未精确保留。")
        docs.append({"path": study.relative(path), "old_sha256": digest(old), "new_sha256": digest(new), "old_body_preserved_exact": True})
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if {key: study.read(STATE)[key] for key in FORWARD} != forward:
        raise ValueError("保存后的13前瞻字段有变化。")
    study.write(OUT / "project_state_update_receipt.json", {"at": study.parent.original.now(),
        "decision": "TECH.R216", "status": "FOUR_LONG_TERM_FACT_DOCUMENTS_AND_GOAL_STATE_UPDATED",
        "documents": docs, "old_forward_field_values_preserved": list(FORWARD), "frozen_sources_unchanged": frozen,
        "actual_goal_service_status": "active", "research_progress": True, "consecutive_blocked_goal_turns": 0,
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False})
    print("R216完整结果和必要失败归因已写入四份长期事实；原正文与13前瞻保持，目标实际active且未达。", flush=True)


if __name__ == "__main__":
    main()
