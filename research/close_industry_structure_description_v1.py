"""一次归档R222实际行业描述，保留金融拒绝与独立前瞻。"""
from __future__ import annotations

import json

from research import industry_structure_description_study_v1 as study
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import digest, prepared_prepend

ROOT, OUT, parent = study.ROOT, study.OUT, study.parent
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def main():
    if (OUT / "project_state_update_receipt.json").exists() or (OUT / "state_before_TECH_R222.json").exists():
        raise RuntimeError("行业结构描述已经归档或开始，不重复更新。")
    summary = parent.read(OUT / "summary.json")
    diagnosis = parent.read(OUT / "post_run_diagnosis.json")
    service = parent.read(OUT / "goal_service_status_after_result.json")
    report = parent.read(OUT / "report_receipt.json")
    views = parent.read(OUT / "figure_view_receipt.json")
    if (summary["decision"] != "TECH.R222" or summary["daily_rows"] != 3488
            or summary["evaluation_rows"] != 2855 or summary["known_evaluation_rows"] != 1956
            or summary["industry_daily_rows"] != 38004 or summary["stage_events"] != 143
            or summary["event_anchor_states"]["KNOWN_FIXED_INDUSTRY_ANCHOR"] != 83
            or summary["event_profile_rows"] != 3003 or summary["identity_rows"] != 43800
            or summary["case_rows"] != 240 or summary["key_rows"] != 17
            or summary["source_files_unchanged"] != 18 or summary["necessary_tests_passed"] != 4
            or len(summary["prefix_checks"]) != 17 or summary["new_accounts"] != 0
            or diagnosis["old_stock_join_rows"] != 429 or views["viewed"] != 4
            or service["goal"]["status"] != "active"
            or parent.digest(ROOT / report["report"]) != report["sha256"]):
        raise ValueError("实际保存结果、报告、查看或目标状态不一致。")
    raw = STATE.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R220" or state["latest_actual_financial_decision"] != "TECH.R216":
        raise ValueError("当前事实由其他研究推进，先核实。")
    original_financial = (state["latest_actual_financial_decision"], state["latest_actual_financial_result"])
    forward = {key: state[key] for key in FORWARD}
    result = study.relative(OUT / "summary.json")
    report_path = report["report"]
    proof = study.relative(OUT / "goal_service_status_after_result.json")
    date = parent.original.now()[:10]
    note = f"""> 最新行业数值描述（{date}，TECH.R221—R222；实际金融仍R216）：原3488观察槽/2855评价槽、143事件/3003固定观察、四原案例240行/17关键日完成，38004逐点行业记录、32949固定行业记录、43800成员锚身份。1956评价日结构可知/899未知，83事件锚可形成/60未知；固定观察934可知（含83零槽）/2069未知。4必要测试、17输入截断精确通过、18冻结文件保持。0新账户/拟合/训练目标/行情请求；收益夏普NOT_COMPUTED，完整目标未达。服务实际active，本轮PROGRESS，连续受阻0。

当时数据方法：前一源日实际成员与已公布分类、20日相对ETF强弱固定前三行业、两个不重叠5日块排名变化和保留行业上涨比例；分类及20报价明确至少294/300、行业至少5成员且98%报价历史完整，至少4行业。固定锚后不追逐新赢家。实际代表145—250成员、中位220，不是指数权重/贡献或资金流；209已知评价槽分类源龄超过365日。历史开发/校准、首版NOT_CERTIFIED、独立NOT_ESTABLISHED。

具体上涨：2019-01-08观察行业上涨78.95%，日线已正、上周负；2020-04-01上涨77.78%、日线刚修复，上周及融资仍弱。2024-09-24以前一09-23源看保险/汽车/资本市场服务，16行业212成员，上涨75%，领先5日2.92%/ETF1.23%、20日相对9.23个百分点；当日相对量3.3384而周线和PMI/融资未全面确认。09-26扩散16/16、轮动0.2583；09-30扩散17/17、轮动回升0.4191，集合变化不能当纯轮动加速度。没有支持统一轮动低才买、高就卖。

反例：2019-06-19上涨行业1/19，次日观察13/19；原R216在该原周期提前退出而损失盈利，下一07-02上涨比例82.35%仍不能保证原进入赚钱。2015全部22案例行业未知；2020原95行34可知/61未知，最新2020Q1原件失败保持，不能延用旧表；2025-06-26/07-03/04分别293/291/291同时明确，行业未知。四案例逐点已知0/92/34/31，固定锚已知0/1/8/12；两类可观察性不得混淆。原图60日标题分母错误保持，交付图仅修正22/92/95/31且4幅实际查看。

新增假设依据：全部429固定1/5/20槽与原股票观察完整对照。近期原完成周期第5槽ETF锚后累计正、原领先行业非正8次，7次信息早于原退出，原8次均亏；2025-03-21原股票多数仍正但固定领先行业−0.437%/ETF+1.773%，体现不同信息。不是8个新策略样本或100%胜率。近期同槽两者均正12原周期仍3赢9输，支持不足以修复旧进入。

下一具体用途尚未登记/运行：实际买入前一源日固定行业，实际进入后5个已形成源日，检验ETF仍正而原领先行业非正是否应下一开盘退出，原风险优先、未知按原规则。配原阶段、同覆盖ETF价格条件退出和全日历价格退出对照；完整原账户/成本/股息/两个时期/风险门，不能只加总原亏损节约额。源钟、覆盖门和实际计时须在新收益前固定，历史通过仍需独立证据。原R212/R216失败、R218整表失败、R220源未知及13独立前瞻保持；当前无已准入待跑金融。

依据：[完整行业研究](../{report_path})、[实际结果](../{result})、[固定用途卡](510300_INDUSTRY_STRUCTURE_DESCRIPTION_V1.md)、[实际目标状态](../{proof})。
"""
    decision = f"""### TECH.R221—R222：行业结构、扩散与固定主线描述（{date}）

原3488槽、143事件/3003观察、四案例240行/17关键日完成。0新账户/拟合/训练目标/行情，实际金融仍R216，收益夏普NOT_COMPUTED。服务active、本轮PROGRESS、受阻0；完整目标未达。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 券商周期启发转化 | 策略可按已知阶段和主线变化改变行为 | 复用招商/兴业原文，独立固定前一源日行业强弱/扩散/排名，原策略保留 | 38004逐点行业与32949固定观察完成 | 接受不同状态、信息角色的研究，未接受券商叙事直接为交易策略 | 后续完整用途需实际账户与独立证据 |
| 行业数据可用 | 全分类表明确即可每天得到行业结构 | 原成员、分类与连续20日报价同时明确294/300，保留全2855分母 | 1956已知/899未知，事件83已知锚/60未知 | 接受有限描述，拒绝忽略成员/报价缺口；2015年仅28/244可知 | 新金融明确未知分支，不能按收益调门 |
| 行业强弱能解释早修复 | 价格与行业参与可以先于周线和慢数据 | 四原案例和17原关键日、量价/已知宏观同时显示 | 2019-01-08上涨78.95%、2020-04-01为77.78%，日柱正而上周负 | 接受具体顺序解释，不接受提前必赚或必须全部指标正 | 全部假启动及完整账户检验 |
| 2024主线向宽基扩散 | 重新定价同时出现行业参与扩散 | 09-24看09-23源，09-26/30保持保守钟 | 保险/汽车/资本市场服务；75%升100%，源龄174—180日 | 接受已发生扩散证据，不接受行业收益是指数权重贡献/资金净流入 | 完整账户与独立样本检验 |
| 统一低轮动买高轮动卖 | 上涨只能伴随轮动下降 | 全四案例逐点排名及集合数量保留 | 2024扩散保持100%时轮动0.2583升0.4191，行业16变17；2019亦非单调 | 拒绝统一交易命名与相邻集合变动冒充纯加速度 | 不由当前结果选轮动阈值救规则 |
| 固定主线跟踪可知性 | 逐点已知意味着最初行业锚后传播也全知 | 固定成员/分类/领先身份，逐组报价98%完整及组比例门 | 四案例逐点0/92/34/31已知，固定0/1/8/12已知 | 拒绝将每天重选领先当最初主线延续；局部均值不是总体可知 | 后续用途按实际测量对象固定资格，原结果不改 |
| 行业普遍转弱就退出 | 行业转弱足以判定修复失败 | 保留2019被截断盈利及次日恢复、2025未知反例 | 06-19行业1/19正，06-20为13/19；原R216损失盈利；2025点位未知 | 拒绝凭一次弱势改退出，06-20收盘亦晚于原开盘退出 | 新不同机制需完整同风险验证 |
| ETF仍正而原主线弱 | 固定行业均值可提供价格与股票多数之外的信息 | 全429固定时点对照，原1/5/20和所有未知/晚信息同列 | 近期第5槽8原亏周期/7及时；2025-03-21股票多数正、行业−0.437%/ETF+1.773% | 接受下一问题，不接受新策略100%胜率或独立证据 | 是，以实际进入起点重新固定、完整价格及同覆盖对照 |
| 行业主线正足以修复旧进入 | 行业与ETF均正即可提升旧策略 | 原周期所有状态完整展示 | 近期第5槽两者正12次，原3赢9输 | 拒绝充分条件；原进入和摩擦问题仍待解决 | 完整新用途与风险/费用/稳定门 |
| 图标题实现 | 四原案例均为60行 | 原保存表真实分母核对，独立交付图重画 | 实际22/92/95/31；4图实际查看 | 接受标题修正，原图/代码/结果保持，无重复描述 | 不需要重复数值实验 |
| 收益/夏普改善 | 描述清楚即可认为策略优化成功 | 本用途0新账户，保留原金融结果与13前瞻 | NOT_COMPUTED、首版未认证、独立未建立 | 拒绝收益已提高或过拟合已去除 | 是，完整同口径账户后仍需独立前瞻 |

来源：[完整研究](../{report_path})、[实际摘要](../{result})、[全状态及未知诊断](../{study.relative(OUT / 'post_run_diagnosis.json')})、[用途卡](510300_INDUSTRY_STRUCTURE_DESCRIPTION_V1.md)。
"""
    prepared = []
    for name, text in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
            ("RESEARCH_DECISIONS", decision), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decision)):
        path = ROOT / f"docs/{name}.md"
        old, new, body = prepared_prepend(path, text)
        prepared.append((path, old, new, body))
    with (OUT / "state_before_TECH_R222.json").open("xb") as stream:
        stream.write(raw)
    docs = []
    for path, old, new, body in prepared:
        path.write_bytes(new)
        if path.read_bytes() != new or not path.read_bytes().endswith(body):
            raise ValueError("原长期事实正文未逐字节保持。")
        docs.append({"path": study.relative(path), "old_sha256": digest(old), "new_sha256": digest(new), "old_body_preserved_exact": True})
    state["previous_phase_before_TECH_R221_R222"] = {key: state.get(key) for key in
        ("latest_technical_decision", "latest_registration_decision", "latest_report", "latest_result", "current_phase", "next_mainline_structure_description")}
    state.update({"updated_at": parent.original.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": parent.original.now(),
        "latest_goal_tool_status_receipt": proof, "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "latest_technical_decision": "TECH.R222", "latest_registration_decision": "TECH.R221", "latest_report": report_path, "latest_result": result,
        "latest_progress": "全行业结构数值与原四案例/143事件完成；价格与固定行业不一致提供下一失效假设，0新金融。",
        "current_phase": "INDUSTRY_STRUCTURE_DESCRIPTION_COMPLETED_NEXT_ENTRY_ANCHORED_MAINLINE_FAILURE_COMPLETE_ACCOUNT",
        "goal_turn_classification": "PROGRESS_R221_R222_ALL_INDUSTRY_STRUCTURE_AND_429_STOCK_CONTEXT_COMPARISONS",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "new_accounts_in_current_phase": 0, "necessary_tests_passed_in_current_phase": 4,
        "current_financial_candidate_admission": "NEW_ENTRY_ANCHORED_MAINLINE_FAILURE_PROPOSAL_NOT_REGISTERED_OR_RUN",
        "current_phase_trial_accounting": {"scope": "TECH_R221_R222_INDUSTRY_STRUCTURE_DESCRIPTION",
            "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0,
            "all_calendar_slots": 3488, "evaluation_slots": 2855, "known_evaluation_slots": 1956, "unknown_evaluation_slots": 899,
            "industry_daily_records": 38004, "stage_events": 143, "known_event_anchors": 83, "unknown_event_anchors": 60,
            "event_profile_records": 3003, "known_fixed_observations_including_83_zero_slots": 934, "unknown_fixed_observations": 2069,
            "fixed_industry_records": 32949, "anchor_member_records": 43800, "case_records": 240,
            "key_records": 17, "necessary_tests": 4, "prefix_checks": 17, "unchanged_frozen_files": 18,
            "complete_old_stock_context_joins": 429, "delivery_figures_viewed": 4, "figure_title_denominator_only_corrected": True},
        "current_goal_turn_actual_work": {"numeric_description_purposes_completed": 1, "all_daily_structure_rows": 3488,
            "all_original_event_profile_rows": 3003, "complete_stock_context_joins": 429,
            "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0, "new_admitted_financial_purposes": 0},
        "latest_industry_structure_description": result,
        "next_mainline_structure_description": {"status": "COMPLETED_FIXED_DESCRIPTION_ALL_CASES_AND_UNKNOWNS_RETAINED",
            "registration": "TECH.R221", "decision": "TECH.R222", "result": result, "financial_run": "NOT_RUN"},
        "next_entry_anchored_industry_failure_proposal": {"status": "PROPOSED_NOT_REGISTERED_OR_RUN",
            "question": "ETF仍有支撑而实际进入时固定的领先行业失效，是否应该缩短持有。",
            "clock": "实际买入前一源日固定身份，实际进入后首次5个已形成源日，不复用信号后时钟。",
            "candidate_action": "ETF同期累计正且原领先行业非正，下一开盘退出；原风险优先、未知保持原流程。",
            "comparisons": ["原阶段账户", "同观察覆盖ETF价格条件退出", "全日历ETF价格条件退出"],
            "source_and_coverage": "按该用途真正测量对象事先固定源钟和组完整门，不根据新收益调门。",
            "mandatory_account": "原20万元/股息权益/原风险/T+1/整手/下一开盘/两时期两费用，pB>1且标准净期望正；完整对比原A和阶段基线。",
            "independent_status": "NOT_ESTABLISHED_ALL_EXISTING_HISTORY_DEVELOPMENT", "financial_metrics": "NOT_COMPUTED"},
    })
    with STATE.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    saved = parent.read(STATE)
    if {key: saved[key] for key in FORWARD} != forward:
        raise ValueError("原13独立前瞻字段被改变。")
    if (saved["latest_actual_financial_decision"], saved["latest_actual_financial_result"]) != original_financial:
        raise ValueError("描述结果改写原金融结论。")
    parent.write(OUT / "project_state_update_receipt.json", {"at": parent.original.now(), "status": "ALL_NUMERIC_INDUSTRY_DESCRIPTION_ARCHIVED",
        "docs": docs, "old_state_sha256": digest(raw), "new_state_sha256": parent.digest(STATE),
        "forward_fields_preserved_exact": list(FORWARD), "latest_technical_decision": "TECH.R222",
        "latest_actual_financial_decision": saved["latest_actual_financial_decision"], "goal_status": saved["goal_status"],
        "goal_turn_classification": saved["goal_turn_classification"], "consecutive_blocked_goal_turns": 0,
        "new_accounts": 0, "financial_metrics": "NOT_COMPUTED"})
    print("R222行业数值描述及四长期事实文件一次归档；原正文、金融R216和13独立前瞻保持，目标active。", flush=True)


if __name__ == "__main__":
    main()
