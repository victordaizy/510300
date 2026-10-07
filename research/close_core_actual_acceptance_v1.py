"""一次保存R235—R236长期事实；金融结果更新，独立前瞻不改变。"""
from __future__ import annotations

import json

from research import core_actual_acceptance_study_v1 as study
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import prepared_prepend, digest
from research.write_core_actual_acceptance_report_v1 import frozen_exact, finance_inventory

ROOT, OUT = study.ROOT, study.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FINANCIAL = ("latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status")


def main():
    receipt_path, backup = OUT / "project_state_update_receipt.json", OUT / "state_before_TECH_R236.json"
    if receipt_path.exists():
        raise RuntimeError("本轮已归档，不重复。")
    resuming_archive_only_failure = backup.exists()
    if resuming_archive_only_failure:
        failure = study.read(OUT / "archive_only_failure_and_recovery_basis.json")
        if failure["exec_session"] != 5438 or failure["exit_code"] != 1 or STATE.read_bytes() != backup.read_bytes():
            raise RuntimeError("已有备份不是本次正文写入前的归档接口失败。")
        if len(failure["unchanged_docs"]) != 4:
            raise ValueError("原四份正文证明不足。")
        for item in failure["unchanged_docs"]:
            if study.digest(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError("归档失败之后正文改变，禁止重复接续。")
    result = study.read(OUT / "summary.json")
    diagnosis = study.read(OUT / "post_run_diagnosis.json")
    delivery = study.read(OUT / "delivery_receipt.json")
    view = study.read(OUT / "figure_view_receipt.json")
    service = study.read(OUT / "goal_service_status_after_result.json")
    proposal = study.read(OUT / "next_core_support_source_complement_description_proposal.json")
    tests = study.read(OUT / "tests_receipt.json")
    pre = study.read(OUT / "control_and_prefix_preflight.json")
    if (result["decision"] != "TECH.R236" or result["new_accounts"] != 12 or len(result["metrics"]) != 20
            or len(result["comparisons"]) != 16 or len(result["gates"]) != 4 or len(result["timing_checks"]) != 12
            or result["saved_account_checks"] != 12 or result["necessary_tests_passed"] != 11 or result["all_prefix_checks"] != 19
            or tests["passed"] != 11 or tests["exit_code"] != 0 or len(pre["adapter_checks"]) != 4
            or len(pre["all_original_prefixes"]) != 19 or len(pre["saved_metrics_checked"]) != 8
            or len(diagnosis["all_new_account_diagnoses"]) != 12 or len(diagnosis["information_vs_protection_pairs"]) != 4
            or len(diagnosis["all_original_A_entry_matches"]) != 12 or delivery["all_metrics"] != 20
            or delivery["all_interval_rows"] != 32 or delivery["new_account_files_exact"] != 84
            or not view["all_two_actually_viewed"] or view["panels_actually_viewed"] != 10
            or service["goal"]["status"] != "active" or proposal["financial_admission"] != "NOT_ADMITTED"
            or result["independent_validation"] != "NOT_ESTABLISHED" or result["goal_achieved"]):
        raise ValueError("金融范围、必要核对、实际查看或目标状态不完整。")
    if view["figures"] != delivery["figures"] or study.digest(ROOT / delivery["report"]) != delivery["report_sha256"]:
        raise ValueError("实际查看图或完整报告版本不同。")
    for item in delivery["figures"]:
        if study.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("查看后的图改变。")
    frozen_count = frozen_exact()
    if finance_inventory() != delivery["new_account_files"]:
        raise ValueError("已完成的金融账户改变。")
    raw = STATE.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R234" or state["latest_actual_financial_decision"] != "TECH.R232":
        raise ValueError("项目已推进，不能覆盖。")
    forward = {key: state[key] for key in FORWARD}
    old_finance = {key: state[key] for key in FINANCIAL}
    old_phase = {key:state.get(key) for key in ("latest_technical_decision","latest_registration_decision","latest_report","latest_result",
        "current_phase","current_study","goal_turn_classification","current_phase_trial_accounting","next_core_actual_acceptance_carry_proposal")}
    report, summary_path = delivery["report"], study.relative(OUT / "summary.json")
    diag_path, service_path = study.relative(OUT / "post_run_diagnosis.json"), study.relative(OUT / "goal_service_status_after_result.json")
    next_path = study.relative(OUT / "next_core_support_source_complement_description_proposal.json")
    pressure = [x for x in result["metrics"] if x["cost"] == "STRESS"]
    metric_lines = []
    for x in pressure:
        metric_lines.append(f"| {x['period']} | {study.NAMES[x['policy']]} | {x['net_cagr']:.4%} | {x['net_sharpe']:.6f} | {x['max_drawdown']:.4%} | {x['wins']}/{x['losses']} | {x['p_times_b']:.6f} | {x['average_full_year_cycles']:.4f} |")
    metric_table = "| 时期 | 账户 | 净年化 | 净夏普 | 最大回撤 | 赢/亏 | 实际净pB | 完整年均次数 |\n|---|---|---|---|---|---|---|---|\n"+"\n".join(metric_lines)
    passed = sum(x["economic_passed"] for x in result["gates"])
    note = f"""> 最新实际金融（{study.parent.original.now()[:10]}，TECH.R235—R236，原A实际进入后的接受保护与新增信息延续）：唯一配置12新完整账户、8保存对照、20指标及全部32两尺度区间；11必要测试、19整段来源/原目标身份前缀、4原A适配器逐日订单周期精确、8旧指标、12现金/12时序通过。登记前全NaT公布钟秒/微秒格式失败保留；纳秒明确和新增时间测试在登记前完成，0金融重跑。{frozen_count}冻结来源/84新账户文件精确，两图10面板实际查看；{passed}/4经济门、稳定={result['historical_stability_passed']}，终态{result['status']}。目标收益/夏普同时提高尚未实现，独立验证/去过拟合未建立；目标active/PROGRESS、受阻0。最新金融由R232更新R236，原旧失败/正文和13独立前瞻保持。

{metric_table}

核心归因：早期主与仅保护在两费用的全部金融序列精确相同，主虽2晋级/10撤销，但原目标零/未知时延续收盘为0，参与/宏观无额外金融增量。近期主8晋级/8撤销、42原零/未知延续收盘，延续确实发生，却低于仅保护的年化和夏普。压力近期主CAGR3.8170%/Sharpe1.217442，相对原A3.9908%/1.216910是年化下降和极小夏普点升；保护3.8930%/1.261572同样年化下降。早期主2.6802%/0.635416虽高于A1.8371%/0.438043，但实际pB0.872547仍未>1，且与归因对照相同。不能事后选保护作已通过策略或拼接早期主/近期A。

所有新策略保留原A实际进入日期，次数未增加；早期22完成/1开放、近期32完成/0开放，完整年均4.4/4.1667。实际赢/亏主早10/12、近17/15，相对A11/11和20/12；近期pB虽提高到1.367079、标准期望0.898329，完整年化仍降低。更多盈亏比并不保证全账户收益更高。匹配入口的现金/数量也会反馈变化，不把单笔人民币差当纯持有增量。六原案例含亏损与确认反例，开盘成交价和图收盘坐标区别明确。

下一优先问题从继续叠加持有指标转到来源互补的时钟和资金冲突：在统一日历和现金约束下，原保存支持—价格接受全部12点位是否确实补原A55进入未持仓，还是冲突/重复/未知或新增亏损。先完整描述原A/本轮/原支持账户的所有机会、拒绝、持有、原0/未知和下一开盘可执行性；不挑2019成功、重用已消费公告、不早期原型/近期A拼接，不救R232/R236。当前下一描述提案已保存但未登记、0新金融准入/待跑；不同完整共同现金机制需有该证据再唯一固定。当前配置不调参重跑。

依据：[完整金融报告](../{report})、[实际结果](../{summary_path})、[全部归因](../{diag_path})、[固定规则](510300_CORE_ACTUAL_ACCEPTANCE_V1.md)、[下一来源互补问题](../{next_path})、[目标实际active](../{service_path})。历史截至Sep30全部开发，首版/全国政策覆盖/独立验证未建立，DSR/PBO未算；次数软目标、20万元日周线范围保持。
"""
    decisions = f"""### TECH.R235—R236：原A实际进入后的接受保护及新增信息延续（{study.parent.original.now()[:10]}）

11测试、19来源/目标段前缀、4A适配器/8旧指标、12完整新账户现金及时序、20指标/32区间和两图10面板完整。登记前时间类型失败留档、经济动作不变、金融一次；{passed}/4经济门，稳定={result['historical_stability_passed']}，独立验证未建立。目标active/PROGRESS、受阻0，最新实际金融R236，原13前瞻保持。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 保留进入改持有 | 原A入场后不同接受/失败动作同时提高收益夏普 | 一主两归因×两期两费、同风险完整现金、四对照两尺度 | 早期主2.6802%/.635416，近期3.8170%/1.217442；{passed}/4门 | 本固定配置拒绝，近期年化低A，早期pB<1，独立未知 | 不修改本位/规则/时窗救回，实质不同信息或新样本另用途 |
| 接受后保护的点值 | 先保留修复，接受后保护能减少部分损失 | 同进入仅保护完整对照，全部赢亏/现金反馈 | 早期与主金融精确相同；近期3.8930%/1.261572但CAGR低A | 接受原点值归因，不能事后晋升此对照为已通过策略 | 无独立验证，不能换标控制规避本失败 |
| 参与宏观延续增量 | 更多源可确认更好的延长持有 | 主对价格延续及仅保护、0/未知真实延续逐日 | 早期2晋级/10撤销却0额外延续、主保护序列同；近期42收盘却主低保护 | 拒绝本用途附加增量充分性，保留信息角色事实 | 不加指标/门槛救本配置；新来源/用途需事前固定 |
| 提高pB即提高年化 | 近期pB更大能自动满足总目标 | 实际完成净回报与所有现金日/自然开放同时核对 | pB1.367079、EV.898329；年化3.8170%低原3.9908% | 拒绝自动推论，接受实际pB及正期望事实 | 资金占用、数量反馈、放弃赢家与成本必须共同验 |
| 增加频率 | 持有阶段变化自然增加次数 | 全55进入外连接、逐年完整统计 | 各新策略进入日期均保留，年均4.4/4.1667未增 | 拒绝本用途能增加机会的事实主张；次数仍软 | 下一检查完整来源互补，不能将描述空仓作新好买点 |
| 更早支持源补原A | 同信息持续与价格接受能补较迟CORE进入 | 下一全12支持点位/55A入口、两期两费共同时钟及现金冲突描述 | 新描述提案PROPOSED_NOT_REGISTERED_NOT_RUN，0金融准入 | 接受不同的来源配置问题，不接受组合已盈利 | 先保留所有冲突/亏损/未知，若有机制证据再固定完整共同账户 |
| 登记前时间格式 | 全NaT与真实公布钟可稳定精确截断 | 第一秒/微秒失败、原表保留、明确纳秒新增必要测试 | 11最终测试/19前缀通过，经济动作未改、金融只一次 | 接受格式修复与时序，不改变过去冻结源 | 无需再次跑已完成金融，真实值不等仍失败 |

依据：[完整金融](../{report})、[实际结果](../{summary_path})、[全部现金/费用/持有归因](../{diag_path})、[唯一规则](510300_CORE_ACTUAL_ACCEPTANCE_V1.md)、[下一不同问题](../{next_path})。历史点值不等于独立成功，旧R232/R212等留档，不把一次终局变成项目停止。
"""
    prepared = []
    for name, text in (("PROJECT_STATE",note),("PROJECT_STATE_TECHNICAL_LINE",note),("RESEARCH_DECISIONS",decisions),("RESEARCH_DECISIONS_TECHNICAL_LINE",decisions)):
        p = ROOT / f"docs/{name}.md"
        prepared.append((p,*prepared_prepend(p,text)))
    state.update({"updated_at":study.parent.original.now(),"status":"research_active","goal_status":"active","goal_achieved":False,
        "latest_goal_service_status":"active","latest_goal_service_status_observed_at":service["observed_at_utc"],
        "latest_goal_tool_status_receipt":service_path,"consecutive_blocked_goal_turns":0,"blocked_audit_count":0,
        "latest_technical_decision":"TECH.R236","latest_registration_decision":"TECH.R235","latest_result":summary_path,
        "latest_report":report,"latest_completed_study":study.relative(OUT),"latest_research_status":result["status"],
        "latest_actual_financial_decision":"TECH.R236","latest_actual_financial_result":summary_path,
        "latest_actual_financial_status":result["status"],"latest_actual_financial_failure_diagnosis":diag_path,
        "previous_actual_financial_before_TECH_R235_R236":old_finance,"previous_phase_before_TECH_R235_R236":old_phase,
        "current_study":"510300_CORE_ACTUAL_ACCEPTANCE_V1","current_phase":"CORE_ACTUAL_ACCEPTANCE_COMPLETE_FIXED_CONFIGURATION_TERMINAL_NEXT_SOURCE_COMPLEMENT_DESCRIPTION",
        "goal_turn_classification":"PROGRESS_R235_R236_FULL_ACCOUNT_HOLDING_MECHANISM_AND_COMPONENT_ATTRIBUTION_COMPLETED",
        "latest_progress":"12完整新账户、20指标/32区间及原A全部进入的持有增量已检验；本配置拒绝，下一转来源互补时钟与现金冲突。",
        "current_admitted_unrun_numeric_candidates":0,"current_admitted_unrun_complete_uses":0,
        "current_financial_candidate_admission":"R236_FIXED_CONFIGURATION_REJECTED_NEXT_SOURCE_DESCRIPTION_NOT_ADMITTED",
        "new_accounts_in_current_phase":12,"necessary_tests_passed_in_current_phase":11,
        "current_phase_trial_accounting":{"scope":"TECH_R235_R236_FIXED_CORE_ACCEPTANCE_HOLDING_FINANCIAL",
            "primary_configurations":1,"predeclared_component_controls":2,"new_full_accounts":12,"saved_controls_reused":8,
            "original_A_adapter_replays":4,"all_metrics":20,"all_interval_rows":32,"necessary_tests":11,
            "source_and_episode_prefix_checks":19,"accounting_checks":12,"timing_checks":12,
            "frozen_source_files_exact":frozen_count,"new_account_files_exact":84,"financial_run_starts":1,"financial_replays_after_result":0,
            "pre_freeze_format_failed_attempts":1,"figures_actually_viewed":2,"figure_panels":10,"new_fits":0,"new_labels":0,"new_market_requests":0},
        "current_goal_turn_actual_work":{"new_financial_purposes_completed":1,"new_full_accounts":12,"all_metrics":20,"all_interval_rows":32,
            "full_A_entry_identity_matching_completed":True,"new_materially_different_description_proposal":proposal["name"],"new_fits":0,"new_market_requests":0},
        "next_core_actual_acceptance_carry_proposal":{**state["next_core_actual_acceptance_carry_proposal"],
            "status":"COMPLETED_FINANCIAL_FIXED_CONFIGURATION_TERMINAL","registration":"TECH.R235","decision":"TECH.R236",
            "result":summary_path,"financial_admission":"COMPLETED_NO_REPLAY_NO_PARAMETER_RESCUE","new_financial_runs":1},
        "next_core_support_source_complement_description_proposal":{**proposal,"proposal_path":next_path}})
    if {key:state[key] for key in FORWARD} != forward:
        raise AssertionError("独立前瞻字段改变。")
    if not resuming_archive_only_failure:
        with backup.open("xb") as f:
            f.write(raw)
    document_receipts = []
    for p,old,new,body in prepared:
        if p.read_bytes() != old:
            raise RuntimeError("长期事实在准备后改变。")
        p.write_bytes(new)
        if p.read_bytes() != new or not new.endswith(body):
            raise AssertionError("长期事实写入或原正文保留失败。")
        document_receipts.append({"path":study.relative(p),"old_sha256":digest(old),"new_sha256":digest(new),"old_body_preserved_exact":True})
    STATE.write_text(json.dumps(study.parent.original.clean(state),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    actual = study.read(STATE)
    if {key:actual[key] for key in FORWARD} != forward or actual["latest_actual_financial_decision"] != "TECH.R236":
        raise AssertionError("保存状态与金融终态或独立前瞻不同。")
    study.write(receipt_path,{"at":study.parent.original.now(),"decision":"TECH.R236","docs":document_receipts,
        "old_state_backup":study.relative(backup),"state_sha256":digest(STATE.read_bytes()),
        "old_financial_fields_preserved_in_backup":old_finance,"new_actual_financial_decision":"TECH.R236",
        "forward_fields_preserved_exact":list(FORWARD),"frozen_source_files_exact":frozen_count,"new_account_files_exact":84,
        "new_full_financial_accounts":12,"actual_financial_replays_after_result":0,"goal_service_status":"active",
        "goal_turn_classification":actual["goal_turn_classification"],"consecutive_blocked_goal_turns":0,"goal_achieved":False,
        "next_financial_admission":"NOT_ADMITTED","archive_only_first_failure_recovered":resuming_archive_only_failure,
        "same_original_state_backup_reused":resuming_archive_only_failure,"actual_document_state_updates":1})
    print("R235—R236一次更新四份长期事实和状态；R236实际金融拒绝留档，13独立前瞻精确保持。",flush=True)


if __name__ == "__main__":
    main()
