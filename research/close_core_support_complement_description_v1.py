"""一次归档完整来源描述，保持实际金融和独立前瞻。"""
from __future__ import annotations

import json

from research import core_support_complement_completion_1_0_1 as study
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import prepared_prepend, digest

ROOT, OUT = study.ROOT, study.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FINANCE = ("latest_actual_financial_decision","latest_actual_financial_result","latest_actual_financial_status")


def main():
    receipt = OUT / "project_state_update_receipt.json"
    backup = OUT / "state_before_TECH_R238.json"
    if receipt.exists() or backup.exists():
        raise RuntimeError("本轮归档已完成或开始，不重复。")
    result = study.read(OUT / "summary.json")
    diagnosis = study.read(OUT / "post_run_diagnosis.json")
    delivery = study.read(OUT / "delivery_receipt.json")
    view = study.read(OUT / "figure_view_receipt.json")
    service = study.read(OUT / "goal_service_status_after_result.json")
    proposal = study.read(OUT / "next_shared_cash_source_ownership_proposal.json")
    if result["decision"] != "TECH.R238" or result["all_context_rows"] != 72 or result["all_original_A_entry_context_rows"] != 330:
        raise ValueError("完整范围改变。")
    if len(result["original_key_prefixes"]) != 19 or len(result["all_signal_cutoffs"]) != 12 or result["new_accounts"] != 0:
        raise ValueError("时序范围或金融身份不完整。")
    if diagnosis["original_A_potential_unique_points"] != 5 or diagnosis["old_news_conflicts_unique"] != 3 or diagnosis["old_news_conflicts_with_original_birth_eligible"] != 2:
        raise ValueError("来源/资金冲突事实改变。")
    if service["goal"]["status"] != "active" or not view["actually_viewed"] or view["panels_actually_viewed"] != 10:
        raise ValueError("实际目标状态或查看图未完成。")
    if view["figures"] != delivery["figures"] or study.digest(ROOT / delivery["report"]) != delivery["report_sha256"]:
        raise ValueError("完整报告或实际查看版本不同。")
    for item in [*delivery["figures"],*delivery["tables"]]:
        if study.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("保存输出改变。")
    frozen = study.frozen_exact()
    raw = STATE.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R236" or state["latest_actual_financial_decision"] != "TECH.R236":
        raise ValueError("当前项目状态已经推进。")
    financial = {k:state[k] for k in FINANCE}
    forward = {k:state[k] for k in FORWARD}
    old_phase = {k:state.get(k) for k in ("latest_technical_decision","latest_registration_decision","latest_report","latest_result",
        "current_phase","current_study","goal_turn_classification","current_phase_trial_accounting")}
    relative = study.original.relative
    report,result_path = delivery["report"],relative(OUT / "summary.json")
    diag_path,proposal_path = relative(OUT / "post_run_diagnosis.json"),relative(OUT / "next_shared_cash_source_ownership_proposal.json")
    note = f"""> 最新实际研究（2026-10-06，TECH.R237—R238，原A与支持来源完整互补描述）：12接受点、72费用/背景行、55原A进入的330背景行，三保存来源×两期两费12账户；238保存周期费用复本234完成/4开放。5必要测试、19原关键日和12接受日共31截断全部精确；{frozen}原冻结来源保持，9组全表及10面板图实际查看。原范围144计数断言失败留档，1.0.1只按身份矩阵修正72，经济函数未改，0新账户/金融/拟合/标签/请求。最新实际金融三字段仍R236拒绝，新策略收益夏普NOT_COMPUTED、目标未达/active/PROGRESS、受阻0；原正文及13独立前瞻保持。

压力原A有5个当时零目标且空仓的潜在补充（早2近3），下一真实开盘一次容量均正，包含2019较早修复与2023/2025旧支持亏损。原A背景2019容量11600份，独立支持实际29700份；原现金/峰值回撤/风险已改变预算，不能将独立利润相加或报5次新共同交易。三背景两费用的30潜在行仅5个事件，不是30独立机会。

旧支持在原A进入时三次占用：2015Sep9支持对应ASep29进入、2019Jan9对应AFeb28，这两支持出生时A确实零/空；2024Sep25对应AOct9，但该支持出生时A已持仓，不能当新共同冲突必然发生。2021Dec9的R236虽保护后现金，原CORE仍正；原源优先而不是误当零，已有持仓计划次开卖也不能同开盘复买接受。

下一不同整体机制为统一共同现金的CORE优先和固定持仓归属：空仓CORE正用原权重，明确零才尝试当天支持接受，未知不补充；首次真实买入固定CORE/SUPPORT，各沿原退出，不按另一源换身份或独立加仓，接受当日消费。原价接受无支持、同共同进入原固定退出两个归因及原A/原支持/R236完整对照，统一20万元风险股息/整手T+1次开，两时期两费用，全部净CAGR/Sharpe增量、DD<=10%、实际pB>1及标准期望正、原两尺度区间；次数软。此提案PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN、0金融准入，完成唯一动作和原A适配必要核对后另登记，不按2019/PMI赢家救旧配置。

依据：[完整点位/量价宏观与冲突](../{report})、[完整结果](../{result_path})、[全归因](../{diag_path})、[下一共同账户提案](../{proposal_path})、[原计数失败](../{relative(study.original.OUT / 'RUN_FAILURE.json')})。历史全部开发，首版/独立/全国政策覆盖未建立、DSR/PBO未算。
"""
    decisions = f"""### TECH.R237—R238：全部支持来源与原A的时钟、现金和持仓冲突（2026-10-06）

12点位/72背景/330原A入口背景，12保存账户238周期复本234完成4开放；5测试和31截断精确，10面板图实际查看。原计数失败保留，1.0.1不改经济动作。0新金融，新收益NOT_COMPUTED，最新金融R236三字段/13前瞻保持；目标active/PROGRESS、未实现。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 来源互补资格 | 支持接受能补原A空仓 | 全12点在原源0/正/未知和实际持仓核对，下一开盘另列 | 5潜在：Sep2015/Jan2019/Apr2020/Jun2023/May2025，各费用均容量正 | 接受不同來源问题，拒绝背景行就是新策略交易 | 新共同账户现金/占用须另登记一次完整实验 |
| 小资金独立利润可相加 | 独立支持赚钱等于给原A直接添利 | 真实原现金/峰值/应收/风险及整手容量 | Jan2019原背景11600份，独立29700份；两近期亏损亦保留 | 拒绝收益相加和五笔新胜率，接受资金路径会改变收益 | 共同账户逐日执行后再评价 |
| 两来源自然不冲突 | 更早进入只添机会不占用后续 | 全55原A进入的330背景与支持真实身份匹配 | 三次旧支持占用，2015/2019两次出生资格真；2024出生时A已持仓 | 拒绝无占用推论，支持固定归属/CORE优先的不同机制 | 所有迟到/吸收/放弃进入必须纳入完整账户 |
| 空仓就能补源 | 保护后现金或待卖等于原零资格 | 原A/原R236/原支持同接受日逐行 | Dec2021主现金但CORE正，Sep2024A当天已买；计划次开退出仍持仓 | 拒绝把现金误当源零与同开复买；已消费不复活 | 只按当前实际归属和原目标0/正/未知执行 |
| 宏观多数票 | 同公告工具与确认可加独立分，低PMI自动排除 | 公布/首次观察/锚/接受时钟和共同源身份，原值完整保留 | 65节点归共同源；订单PMI48.3/49.2亏损也有低值赢家 | 拒绝独立投票及按亏损挑新门槛；接受背景用途 | 新宏观作用须事前唯一固定，不能救R232/R236 |
| 范围断言正确 | 两期两费三背景应是144点位行 | 每点时期唯一的完整身份矩阵、原失败接续 | 正确72，原第一次范围失败保存，31前缀保持精确 | 接受纯计数修复，未改经济/时序/金融 | 不重跑旧入口或放宽前缀 |
| 共同现金固定归属 | CORE与零目标补充能提升总净收益夏普 | 下一一主两归因/完整原对照，同风险两期两费 | PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN | 接受待验证的不同用途，未接受交易策略 | 唯一动作/原A精确适配/必要时序现金后另登记 |

依据：[完整报告](../{report})、[实际结果](../{result_path})、[全归因](../{diag_path})、[下一不同用途](../{proposal_path})。所有旧终态拒绝、未知和开放保留；本描述不是收益验证。
"""
    prepared = []
    for name,text in (("PROJECT_STATE",note),("PROJECT_STATE_TECHNICAL_LINE",note),("RESEARCH_DECISIONS",decisions),("RESEARCH_DECISIONS_TECHNICAL_LINE",decisions)):
        p = ROOT / f"docs/{name}.md"
        prepared.append((p,*prepared_prepend(p,text)))
    state.update({"updated_at":study.original.previous.parent.original.now(),"status":"research_active","goal_status":"active","goal_achieved":False,
        "latest_goal_service_status":"active","latest_goal_service_status_observed_at":service["observed_at_utc"],
        "latest_goal_tool_status_receipt":relative(OUT / "goal_service_status_after_result.json"),
        "consecutive_blocked_goal_turns":0,"blocked_audit_count":0,"latest_technical_decision":"TECH.R238",
        "latest_registration_decision":"TECH.R237","latest_result":result_path,"latest_report":report,
        "latest_completed_study":relative(OUT),"latest_research_status":result["status"],
        "previous_phase_before_TECH_R237_R238":old_phase,
        "current_study":"510300_CORE_SUPPORT_COMPLEMENT_DESCRIPTION_V1",
        "current_phase":"CORE_SUPPORT_FULL_SOURCE_DESCRIPTION_COMPLETE_NEXT_SHARED_CASH_SOURCE_OWNERSHIP",
        "goal_turn_classification":"PROGRESS_R237_R238_COMPLETE_SOURCE_CLOCK_AND_CASH_CONFLICT_DESCRIPTION",
        "latest_progress":"全部点位资格/一次容量和原A持仓冲突已验证时序；下一不同共同现金归属机制待唯一固定，不相加旧利润。",
        "current_admitted_unrun_numeric_candidates":0,"current_admitted_unrun_complete_uses":0,
        "current_financial_candidate_admission":"NEXT_SHARED_CASH_SOURCE_OWNERSHIP_NOT_ADMITTED",
        "new_accounts_in_current_phase":0,"necessary_tests_passed_in_current_phase":5,
        "current_new_strategy_return_sharpe":"NOT_COMPUTED",
        "current_phase_trial_accounting":{"scope":"TECH_R237_R238_CORE_SUPPORT_FULL_SAVED_SOURCE_DESCRIPTION",
            "saved_accounts":12,"support_points":12,"context_rows":72,"A_entry_context_rows":330,"saved_cycle_replicates":238,
            "completed_replicates":234,"open_replicates":4,"necessary_tests":5,"prefixes":31,
            "original_frozen_sources_exact":frozen,"full_saved_output_generations":1,"count_assertion_failure_preserved":True,
            "diagnostic_reconstruction_only":1,"figures_actually_viewed":1,"figure_panels":10,
            "new_accounts":0,"new_financial_runs":0,"new_fits":0,"new_labels":0,"new_market_requests":0},
        "current_goal_turn_actual_work":{"complete_source_and_cash_description":True,"unique_potential_original_A_points":5,
            "possible_eligible_support_original_A_conflicts":2,"new_financial_runs":0,"new_accounts":0},
        "next_core_support_source_complement_description_proposal":{**state["next_core_support_source_complement_description_proposal"],
            "status":"COMPLETED_FULL_DESCRIPTION_COUNT_CORRECTION_ONLY","registration":"TECH.R237","decision":"TECH.R238","result":result_path},
        "next_shared_cash_source_ownership_proposal":{**proposal,"proposal_path":proposal_path}})
    if {k:state[k] for k in FINANCE} != financial or {k:state[k] for k in FORWARD} != forward:
        raise AssertionError("实际金融或独立前瞻改变。")
    with backup.open("xb") as stream:
        stream.write(raw)
    document_receipts = []
    for p,old,new,body in prepared:
        if p.read_bytes() != old:
            raise RuntimeError("长期事实在准备后变化。")
        p.write_bytes(new)
        if p.read_bytes() != new or not new.endswith(body):
            raise AssertionError("原正文未保持。")
        document_receipts.append({"path":relative(p),"old_sha256":digest(old),"new_sha256":digest(new),"old_body_preserved_exact":True})
    STATE.write_text(json.dumps(study.original.previous.parent.original.clean(state),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    saved = study.read(STATE)
    if {k:saved[k] for k in FINANCE} != financial or {k:saved[k] for k in FORWARD} != forward:
        raise AssertionError("保存状态身份不一致。")
    study.write(receipt,{"at":study.original.previous.parent.original.now(),"decision":"TECH.R238","docs":document_receipts,
        "old_state_backup":relative(backup),"state_sha256":digest(STATE.read_bytes()),"financial_fields_preserved_exact":financial,
        "forward_fields_preserved_exact":list(FORWARD),"original_frozen_sources_exact":frozen,"new_accounts":0,
        "new_financial_runs":0,"actual_document_state_updates":1,"goal_status":"active","goal_achieved":False,
        "consecutive_blocked_goal_turns":0,"next_financial_admission":"NOT_ADMITTED"})
    print("R237—R238四份长期事实和状态已一次更新，R236金融/13独立前瞻精确保持。",flush=True)


if __name__ == "__main__":
    main()
