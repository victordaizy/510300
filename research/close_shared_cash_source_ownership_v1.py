"""一次保存共同现金的实际金融终态，保留所有旧事实和前瞻。"""
from __future__ import annotations

import json

from research import shared_cash_source_ownership_study_v1 as study
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import prepared_prepend,digest
from research.write_shared_cash_source_ownership_report_v1 import frozen_exact,inventory

ROOT,OUT = study.ROOT,study.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FINANCE = ("latest_actual_financial_decision","latest_actual_financial_result","latest_actual_financial_status")


def main():
    receipt,backup = OUT / "project_state_update_receipt.json",OUT / "state_before_TECH_R240.json"
    if receipt.exists() or backup.exists():
        raise RuntimeError("共同现金终态已归档或开始，不重复。")
    result = study.read(OUT / "summary.json")
    pre = study.read(OUT / "control_and_prefix_preflight.json")
    diagnosis = study.read(OUT / "post_run_diagnosis.json")
    delivery = study.read(OUT / "delivery_receipt.json")
    view = study.read(OUT / "figure_view_receipt.json")
    service = study.read(OUT / "goal_service_status_after_result.json")
    proposal = study.read(OUT / "next_source_expectation_transmission_review_proposal.json")
    if result["decision"] != "TECH.R240" or len(result["metrics"]) != 24 or len(result["comparisons"]) != 20 or len(result["gates"]) != 4:
        raise ValueError("完整金融范围改变。")
    if result["new_accounts"] != 12 or result["saved_account_checks"] != 12 or len(result["timing_checks"]) != 12 or result["necessary_tests_passed"] != 11:
        raise ValueError("必要现金/时序核对不足。")
    if len(pre["adapter_checks"]) != 8 or len(pre["all_original_parent_prefixes"]) != 19 or len(diagnosis["all_new_account_diagnoses"]) != 12:
        raise ValueError("原适配及归因不足。")
    if diagnosis["accepted_unique_primary_complements"] != 5 or delivery["all_interval_rows"] != 40 or delivery["new_account_files_exact"] != 84:
        raise ValueError("全部补充和保存范围不完整。")
    if not view["all_two_actually_viewed"] or view["panels_actually_viewed"] != 9 or service["goal"]["status"] != "active":
        raise ValueError("实际查看/目标状态尚未完成。")
    if view["figures"] != delivery["figures"] or study.digest(ROOT / delivery["report"]) != delivery["report_sha256"]:
        raise ValueError("完整报告与查看版本不同。")
    for item in delivery["figures"]:
        if study.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("查看之后图改变。")
    frozen = frozen_exact()
    if inventory() != delivery["new_account_files"]:
        raise ValueError("完成金融的保存账户改变。")
    raw = STATE.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R238" or state["latest_actual_financial_decision"] != "TECH.R236":
        raise ValueError("当前项目已推进，不覆盖。")
    forward = {k:state[k] for k in FORWARD}
    financial = {k:state[k] for k in FINANCE}
    old_phase = {k:state.get(k) for k in ("latest_technical_decision","latest_registration_decision","latest_report","latest_result",
        "current_phase","current_study","goal_turn_classification","current_phase_trial_accounting","current_new_strategy_return_sharpe")}
    report = delivery["report"]
    summary_path,diag_path = study.relative(OUT / "summary.json"),study.relative(OUT / "post_run_diagnosis.json")
    next_path = study.relative(OUT / "next_source_expectation_transmission_review_proposal.json")
    service_path = study.relative(OUT / "goal_service_status_after_result.json")
    stress = [x for x in result["metrics"] if x["cost"] == "STRESS"]
    lines = [f"| {x['period']} | {study.NAMES[x['policy']]} | {x['net_cagr']:.4%} | {x['net_sharpe']:.6f} | {x['max_drawdown']:.4%} | {x['wins']}/{x['losses']} | {'UNKNOWN' if x['p_times_b'] is None else format(x['p_times_b'],'.6f')} | {x['average_full_year_cycles']:.4f} |" for x in stress]
    metrics_table = "| 时期 | 账户 | 净年化 | 净夏普 | 最大回撤 | 赢/亏 | 实际净pB | 完整年均次数 |\n|---|---|---|---|---|---|---|---|\n"+"\n".join(lines)
    passed = sum(x["economic_passed"] for x in result["gates"])
    note = f"""> 最新实际金融（2026-10-06，TECH.R239—R240，共同现金CORE优先与来源持仓归属）：一主两预定归因×两期两费12新账户、12保存对照、24指标及五对照20比较/40两尺度区间，11必要测试、四原A及四原支持8完整适配、19原CORE前缀通过；R238的31来源背景前缀只复用不重跑。12现金/12归属时序、{frozen}冻结来源/84新账户文件保持，二图9面板实际查看。登记前订单字段和字符串类型两失败保持，修正未改经济动作，金融一次。经济{passed}/4、稳定={result['historical_stability_passed']}，终态{result['status']}。目标收益/夏普同时提高未实现，独立/去过拟合未建立；目标active/PROGRESS、受阻0。最新实际金融由R236更新R240，所有旧拒绝/正文及13独立前瞻保持。

{metrics_table}

实际增加机会：早期主23完成/1开放、近期35完成/0开放，相对原A22/1和32/0；完整年均4.4→4.6、4.1667→4.6667。五真实支持补充均保留：2015Sep10—Nov3初买8000份净+973.25；2019Jan10—Mar27初买11500份净+7225.05；2020Apr3—Apr14初买17200份净+672.76；2023Jun16—Jun27初买27400份净−3203.09；2025May13—May28初买29500份净−2201.58。2019原A背景单次11600份/旧独立29700份+16105.95与真正共同账户区别明确；2015补充已改变后续现金/峰值/风险，亏损数量也变化，不能独立收益相加。

早期主CAGR2.7025%/Sharpe.606417高A1.8371%/.438043，但实际pB.824837未>1，夏普低R236 .635416和旧支持1.109941。近期主3.6362%/1.096314低A3.9908%/1.216910，实际pB.983214亦未>1；基础近期pB1.083567不能替代压力。标准净期望早.346576/近.583214为正不等于满足用户更强pB>1或提高完整收益。仅价格补充虽年均10.0/10.1667，压力年化−.1687%/.0765%、夏普−.021661/.040139；更多频率不建立优势。近期固定退出归因3.9325%/1.174684高主但仍低A，不能看结果升格对照或拼时期。

完整原A入口外连接：压力早21真实日期保留、2原日期仅旧、3新日期，近期32原日期全部保留另3新；吸收/迟到与现金数量反馈纳入全部账户。所有24账户周期/开放、12新事件消费/固定归属、实际毛优势/费/风险和拒绝归因保存；固定实际数量去费只解释，不是新可执行账户。旧R232周规则及旧原A动作没有救改。

下一优先审查实质不同的事前信息：政策增量相对此前市场预期、主线传播到ETF的证据。先核对7原券商来源/65原节点及所有5真实补充的来源/公开上界/版本可用性；只有事后故事、重复信息、未知版本或靠已亏PMI/窗口调参则NO_VIEW/NOT_ADMITTED。本新提案PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN，0待跑金融；不同整体用途须事前固定，不救R240/R232/R236、不拼早期主/近期A。原E03仍独立，历史全部开发、首版/全国政策覆盖/独立未建立，DSR/PBO未算。

依据：[完整金融及点位失败归因](../{report})、[全部实际结果](../{summary_path})、[全归因](../{diag_path})、[唯一规则](510300_SHARED_CASH_SOURCE_OWNERSHIP_V1.md)、[下一不同来源审查](../{next_path})、[目标active实际回执](../{service_path})。
"""
    decisions = f"""### TECH.R239—R240：统一现金的CORE优先与固定来源归属（2026-10-06）

12新完整账户/12保存对照/24指标/40区间、11必要测试、8原适配精确、19原目标前缀、12现金/12归属时序，二图9面板查看。订单字段及StringDtype登记前失败保持，经济未改，金融一次；经济{passed}/4、稳定={result['historical_stability_passed']}，本配置固定拒绝。最新金融R240、目标active/PROGRESS未实现、受阻0，原13前瞻保持。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 共同现金来源补充 | CORE零时新支持补充同时提高收益夏普 | 一主两预定归因×两期两费，同风险/现金/自然开放和五对照 | 压力早2.7025%/.606417、近3.6362%/1.096314，{passed}/4门 | 本配置拒绝，近期低A、早近pB<1、所有区间未稳定 | 不改本阈值/退出/归属救回；不同信息或新样本另用途 |
| 更多交易 | 原12来源事件能够补真实机会 | 全部成交/拒绝、原A进入外连接、逐年完整统计 | 原年均4.4/4.1667→4.6/4.6667，真实支持5笔；早23/1、近35/0 | 接受频率实际增量，拒绝据此收益和胜率必高 | 后续只评新增来源的净增量，不硬凑次数 |
| 2019独立大赢家可搬入 | 旧16105.95元可加到原A | 真实现金峰值预算、五补充及所有CORE反馈 | 实际11500份+7225.05，原背景11600、旧独立29700；两近亏损也放大 | 拒绝独立盈亏相加或赢家大仓/亏损小仓拼接 | 必须同账户真实时钟与现金执行 |
| 不需支持信息 | 去掉宏观要求会更快增加有效点位 | 同归属/现金/风险仅原价格接受对照 | 年均10/10.1667，压力CAGR−.1687%/.0765%、Sharpe−.021661/.040139 | 拒绝本价格用途有足够交易优势，保留全部失败 | 不按赢家增加量/MACD窗口救此控制 |
| 延长补充持有 | 原周递升退出优于固定退出 | 同来源进入/归属，原固定退出预定归因 | 近固定3.9325%/1.174684高主仍低原A | 拒绝本延续普遍优越；对照不事后晋升 | 不换标控制规避失败，须实质不同信息 |
| 净期望正即达目标 | 标准期望正可代替pB>1及全账户增量 | p/B/亏损概率和完整日收益一起核对 | 主压力标准期望.346576/.583214正，但pB.824837/.983214且近收益降 | 拒绝自动达标推论，接受各自实际数学事实 | 保持两独立验收条件，不调口径 |
| 原实现未改变 | 来源组合代码关闭另一功能能复现原策略 | 四A/四支持逐日日净值、订单和周期精确 | 8适配全通过；登记前字段/StringDtype失败保存并修类型 | 接受实现对照与一次金融，未证明独立预测 | 无需重跑已经完成金融，旧原账户保持 |
| 政策预期和ETF传播 | 新增事前信息能解释持续或短反弹 | 下一来源/时钟/版本准入审查，保留五真实成功失败 | PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN，0金融准入 | 接受待查不同信息问题，未接受新策略 | 缺公开上界/首版或只有后验故事则NO_VIEW/不准入 |

依据：[完整金融](../{report})、[实际结果](../{summary_path})、[所有组件归因](../{diag_path})、[唯一规则](510300_SHARED_CASH_SOURCE_OWNERSHIP_V1.md)、[下一信息审查](../{next_path})。历史点值/区块不是独立验证，旧R232/R236及所有终态保留。
"""
    prepared = []
    for name,text in (("PROJECT_STATE",note),("PROJECT_STATE_TECHNICAL_LINE",note),("RESEARCH_DECISIONS",decisions),("RESEARCH_DECISIONS_TECHNICAL_LINE",decisions)):
        p = ROOT / f"docs/{name}.md"
        prepared.append((p,*prepared_prepend(p,text)))
    state.update({"updated_at":study.previous.parent.original.now(),"status":"research_active","goal_status":"active","goal_achieved":False,
        "latest_goal_service_status":"active","latest_goal_service_status_observed_at":service["observed_at_utc"],
        "latest_goal_tool_status_receipt":service_path,"consecutive_blocked_goal_turns":0,"blocked_audit_count":0,
        "latest_technical_decision":"TECH.R240","latest_registration_decision":"TECH.R239","latest_result":summary_path,
        "latest_report":report,"latest_completed_study":study.relative(OUT),"latest_research_status":result["status"],
        "latest_actual_financial_decision":"TECH.R240","latest_actual_financial_result":summary_path,"latest_actual_financial_status":result["status"],
        "latest_actual_financial_failure_diagnosis":diag_path,"previous_actual_financial_before_TECH_R239_R240":financial,
        "previous_phase_before_TECH_R239_R240":old_phase,"current_study":"510300_SHARED_CASH_SOURCE_OWNERSHIP_V1",
        "current_phase":"SHARED_CASH_SOURCE_OWNERSHIP_COMPLETE_FIXED_REJECTION_NEXT_DIFFERENT_INFORMATION_REVIEW",
        "goal_turn_classification":"PROGRESS_R239_R240_FULL_SHARED_CASH_SOURCE_OWNERSHIP_AND_FREQUENCY_ATTRIBUTION_COMPLETED",
        "latest_progress":"12新完整账户验证来源互补/频率/资金占用，本配置不通过；下一实质不同信息来源时钟审查。",
        "current_admitted_unrun_numeric_candidates":0,"current_admitted_unrun_complete_uses":0,
        "current_financial_candidate_admission":"R240_FIXED_REJECTION_NEXT_SOURCE_REVIEW_NOT_ADMITTED",
        "current_new_strategy_return_sharpe":"COMPUTED_R240_FULL_ACCOUNT_FIXED_CONFIGURATION_REJECTED",
        "latest_actual_financial_primary_four_scene_metrics":[x for x in result["metrics"] if x["policy"] == study.rules.POLICIES[0]],
        "new_accounts_in_current_phase":12,"necessary_tests_passed_in_current_phase":11,
        "current_phase_trial_accounting":{"scope":"TECH_R239_R240_FIXED_SHARED_CASH_SOURCE_OWNERSHIP_FINANCIAL",
            "primary_configurations":1,"predeclared_component_controls":2,"new_full_accounts":12,"saved_controls_reused":12,
            "successful_original_A_adapter_replays":4,"successful_original_support_adapter_replays":4,
            "all_metrics":24,"all_interval_rows":40,"necessary_tests":11,"raw_CORE_target_prefixes":19,
            "saved_source_context_prefixes_reused_not_recomputed":31,"accounting_checks":12,"timing_checks":12,
            "frozen_source_files_exact":frozen,"new_account_files_exact":84,"financial_run_starts":1,"financial_replays_after_result":0,
            "pre_freeze_synthetic_schema_failed_attempts":1,"pre_freeze_adapter_string_schema_failed_attempts":1,
            "actual_primary_unique_complements":5,"actual_primary_complement_cost_replicates":10,
            "figures_actually_viewed":2,"figure_panels":9,"new_fits":0,"new_labels":0,"new_market_requests":0},
        "current_goal_turn_actual_work":{"new_financial_purposes_completed":1,"new_full_accounts":12,"all_metrics":24,"all_interval_rows":40,
            "all_source_owner_components_and_A_entry_feedback_completed":True,"new_source_review_proposal":proposal["name"],"new_fits":0,"new_market_requests":0},
        "next_shared_cash_source_ownership_proposal":{**state["next_shared_cash_source_ownership_proposal"],
            "status":"COMPLETED_FINANCIAL_FIXED_CONFIGURATION_TERMINAL","registration":"TECH.R239","decision":"TECH.R240",
            "result":summary_path,"financial_admission":"COMPLETED_NO_REPLAY_NO_PARAMETER_RESCUE","new_financial_runs":1},
        "next_source_expectation_transmission_review_proposal":{**proposal,"proposal_path":next_path}})
    if {k:state[k] for k in FORWARD} != forward:
        raise AssertionError("独立前瞻字段改变。")
    with backup.open("xb") as stream:
        stream.write(raw)
    documents = []
    for p,old,new,body in prepared:
        if p.read_bytes() != old:
            raise RuntimeError("准备后长期事实改变。")
        p.write_bytes(new)
        if p.read_bytes() != new or not new.endswith(body):
            raise AssertionError("原正文未保持。")
        documents.append({"path":study.relative(p),"old_sha256":digest(old),"new_sha256":digest(new),"old_body_preserved_exact":True})
    STATE.write_text(json.dumps(study.previous.parent.original.clean(state),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    saved = study.read(STATE)
    if {k:saved[k] for k in FORWARD} != forward or saved["latest_actual_financial_decision"] != "TECH.R240":
        raise AssertionError("保存金融或独立前瞻身份不同。")
    study.write(receipt,{"at":study.previous.parent.original.now(),"decision":"TECH.R240","docs":documents,
        "old_state_backup":study.relative(backup),"state_sha256":digest(STATE.read_bytes()),"old_financial_fields_preserved_in_backup":financial,
        "new_actual_financial_decision":"TECH.R240","forward_fields_preserved_exact":list(FORWARD),
        "frozen_source_files_exact":frozen,"new_account_files_exact":84,"new_full_financial_accounts":12,
        "actual_financial_replays_after_result":0,"actual_document_state_updates":1,"goal_status":"active","goal_achieved":False,
        "goal_turn_classification":saved["goal_turn_classification"],"consecutive_blocked_goal_turns":0,"next_financial_admission":"NOT_ADMITTED"})
    print("R239—R240四份长期事实/状态已一次更新，实际金融固定拒绝，13独立前瞻精确保持。",flush=True)


if __name__ == "__main__":
    main()
