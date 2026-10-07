"""一次归档完整账户和实际持仓失败，保留旧正文与原独立前瞻合同。"""
from __future__ import annotations

import json
from pathlib import Path

from research import support_price_acceptance_acceptance_v1_0_1 as acceptance
from research import support_price_acceptance_study_v1 as study
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import digest, prepared_prepend
from research.write_support_price_acceptance_report_v1 import check_frozen

ROOT, OUT, SOURCE = study.ROOT, acceptance.OUT, study.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def relative(path: Path) -> str:
    return path.absolute().relative_to(ROOT).as_posix()


def main() -> None:
    receipt_path = OUT / "project_state_update_receipt.json"
    backup = OUT / "state_before_TECH_R232.json"
    if receipt_path.exists() or backup.exists():
        raise RuntimeError("本用途已经归档或开始，不重复更新项目事实。")
    summary = study.read(OUT / "summary.json")
    diagnosis = study.read(OUT / "post_run_diagnosis.json")
    delivery = study.read(OUT / "delivery_receipt.json")
    viewed = study.read(OUT / "figure_view_receipt.json")
    service = study.read(OUT / "goal_service_status_after_result.json")
    proposal = study.read(OUT / "next_actual_holding_information_proposal.json")
    original_tests = study.read(SOURCE / "tests_receipt.json")
    date_tests = study.read(OUT / "tests_receipt.json")
    if (summary["decision"] != "TECH.R232" or summary["original_new_accounts_completed"] != 12
            or summary["original_saved_controls_reused"] != 8 or len(summary["metrics"]) != 20
            or summary["new_account_replays_this_completion"] != 0 or summary["date_value_comparisons"] != 16
            or summary["original_prefix_checks"] != 19 or summary["original_cash_account_checks"] != 12
            or summary["all_economic_gates_passed"] or summary["historical_stability_passed"]
            or diagnosis["all_economic_scenarios_passed"] != 0 or diagnosis["all_saved_interval_rows"] != 32
            or diagnosis["all_primary_stress_completed_cycles"] != 9 or diagnosis["recent_promoted_losses"] != 4
            or diagnosis["all_A_completed_cycles"] != 54 or diagnosis["all_A_cycles_including_open"] != 55
            or len(diagnosis["confirmation_after_one_actual_holding_close"]) != 2
            or delivery["all_holding_context_rows"] != 45 or delivery["all_original_key_days"] != 19
            or original_tests["passed"] != 8 or original_tests["exit_code"] != 0
            or date_tests["passed"] != 2 or date_tests["exit_code"] != 0
            or not viewed["all_two_actually_viewed"] or viewed["panels_actually_viewed"] != 6
            or delivery["figures_actually_viewed"] != 2 or service["goal"]["status"] != "active"
            or summary["goal_achieved"] or proposal["financial_admission"] != "NOT_ADMITTED"):
        raise ValueError("原账户、完整比较、时序、未知、必要检查或目标终态不匹配。")
    if study.digest(ROOT / delivery["report"]) != delivery["report_sha256"]:
        raise ValueError("最终报告与交付回执不同。")
    for expected, actual in zip(delivery["figures"], viewed["figures"], strict=True):
        if expected != actual or study.digest(ROOT / expected["path"]) != expected["sha256"]:
            raise ValueError("图表不是实际查看的同一文件。")
    frozen_before = check_frozen()
    raw = STATE.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R230" or state["latest_actual_financial_decision"] != "TECH.R224":
        raise ValueError("项目已有其他实际推进，不覆盖另一用途的状态。")
    forward_before = {key: state[key] for key in FORWARD}
    previous = {key: state.get(key) for key in (
        "latest_technical_decision", "latest_registration_decision", "latest_report", "latest_result",
        "latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status",
        "current_phase", "current_study", "goal_turn_classification", "current_phase_trial_accounting",
        "next_support_price_acceptance_state_proposal")}
    report, result = delivery["report"], relative(OUT / "summary.json")
    proof = relative(OUT / "goal_service_status_after_result.json")
    diagnostic = relative(OUT / "post_run_diagnosis.json")
    proposal_path = relative(OUT / "next_actual_holding_information_proposal.json")
    date = study.parent.original.now()[:10]
    note = f"""> 最新实际研究（{date}，TECH.R231—R232，支持信息→价格接受→延续→失效的完整账户）：12新账户一次运行，8原A/原阶段保存账户复用，共20指标、全部32两尺度区间；8原测试/2必要日期测试、19整段前缀、8原对照、12现金复算/12持仓时序、16比较日期值检查通过。原账户已运行完成，首个比较因ms/ns类型断言退出，失败留档；1.0.1只规范临时比较日期，0账户重跑、0动作修改，63原冻结输入/80接续文件（含12账户72文件）保持。两图六面板已实际查看。四经济门0/4、历史稳定失败，本配置拒绝；最新实际金融由R224更新为R232，完整收益夏普目标未实现。目标服务实际active、PROGRESS、连续受阻0；原13独立前瞻/E03保持。

压力费用完整结果：2015—2019主净年化2.9743%/夏普1.109941/DD2.4139%、3完成且3盈利0亏损、完整年均0.6；没有亏损样本，实际B/pB/标准亏损单位期望UNKNOWN，不能当可靠100%胜率。原A同期1.8371%/0.438043。2020—2026主0.01966%/0.020075/DD4.5507%、6完成2赢4亏、年均1.0，实际净B2.3729、pB0.79097、标准净期望+0.12430；标准净期望正，但仍未达更严格pB>1。同期原A3.9908%/1.216910、同支持固定退出0.1115%/0.095929；新原型低于两者。无支持价格对照年均9.0/9.5但两个时期净亏，不能用增加次数代替质量。

具体上涨：2019Jan7支持观察、Jan9接受、Jan10实际买、Jan14晋延续、Mar27退出，净利16105.95；2024Sep24宣布观察、Sep25接受、Sep26买、Sep30延续、Oct17退出，净利7747.24。同原周期数量/时点/持有不同，不是单因素归因。2020Apr3买来自Mar30操作/Apr2价格决定，不能用Apr3 16:57:32公告事前解释（其首次ETF可知Apr7）。原两假启动避开，但9主实际周期全晋延续，近期仍4亏；7个确认周含买前交易日。Dec10’21与Jun16’23仅经历1持仓收盘、确认周4买前/1买日起，后来亏2686.01/2058.31。这是冻结规则允许的信息构成，不当日期bug修复，不事后改周纯度/等待参数救配置。

完整失败归因：26支持激活/12价格接受，早期3接受在已有持仓时消费，主共9完成周期；无拒单或DD停买，风险减仓15/7。近期毛1457.80元、佣金滑点1202.36、净255.44，固定原数量零费用解释年化0.1119%/夏普0.05959，仍低于A；不是可执行零費策略。近期92.31%的盈利金额来自2024一笔，资金平均曝光早2.90%/近0.98%。早期压力对A点增量虽正，两尺度区间跨0；近期对A全部区间全负。历史多次选择未校正、独立验证未建立，未宣称去过拟合。

下一优先方向：实际进入后的新增延续信息与原A全部机会顺序。本轮已保存原A54完成/1开放、主9周期、45激活/接受/实际进入/晋级/退出决定节点及原19关键日；A的52完成进入决定是NO_ACTIVE_OBSERVATION、51买入日新原型收盘为空仓，这不代表无政策或51新增好交易。Jun2’20是旧阶段修复决定（盈利10756.16），原A该日实际进入查询空集保留，不能错记原A。研究旧锚消费后的新价格/参与结构、持有过程中形成的新增证据；不是继续加静态入场过滤、按赢家重用旧公告或早期新原型/近期A拼接。新提案PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN，0已准入待跑金融，新的整体动作须先完整解释与唯一登记。

来源与权限：固定65节点/25保存操作、同源与宣布/确认分开；空来源不是无政策。全国公告覆盖、首版、最早公布均未建立。历史截至2026Sep30全为开发，日线/上一完整周、20万元510300.SH/CASH_CNY、50%及原风险/费用/整手/T+1/股息保持；未新增行情/拟合/标签。旧R212/R216/R224与原策略留档，本终局只约束R232这套配置，不是全部自适应策略无效或项目停止。

依据：[完整金融与具体上涨失败报告](../{report})、[实际结果](../{result})、[全部失败归因](../{diagnostic})、[固定用途](510300_SUPPORT_PRICE_ACCEPTANCE_V1.md)、[下一不同机制提案](../{proposal_path})、[实际目标active](../{proof})。
"""
    decisions = f"""### TECH.R231—R232：支持信息、价格接受与真实持仓延续（{date}）

12新账户一次完成、8保存对照，共20指标/32区间。8原+2日期测试、19前缀、12现金/持仓时序、16日期值比较；两图六面板已实际查看。首个比较日期类型失败保留，1.0.1仅接续比较，0金融重跑/拟合/新行情；63原输入/80保存接续文件精确。四经济门0/4/稳定失败，固定配置终局拒绝，最新金融R232；目标active/PROGRESS、受阻0和13前瞻不变。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 按可知阶段改变动作 | 支持先观察、接受再买、确认后持有能改善完整账户 | 唯一EARLY/CARRY机制、四对照/两个时期/两费用 | 早期压力2.9743%/1.10994；近期0.01966%/0.020075，0/4门 | 接受状态动作的实现与解释，拒绝本完整配置绩效 | 不改旧配置；不同新增证据机制需另登记 |
| 支持信息单次激活 | 每次正信息一次价格锚可筛优质机会 | 65全部资格/25操作、26激活/12接受和真实现金 | 主9完成；近期6交易4亏、年均1.0；无支持406点位亦失败 | 拒绝已能兼顾质量频率；空源不是无政策 | 先解释全部A机会与后续腿，不直接反复使用旧新闻 |
| 价格比慢指标先确认 | 日正、价格接受时周柱或订单仍弱可进入 | 原2019/2024案例与全部假启动，同原16:00钟 | Jan10’19盈利16105.95；Sep26’24盈利7747.24；其他亏损保留 | 接受具体顺序解释，不接受两个赢家代替完整胜率 | 不选案例门槛；实质不同整体动作再检验 |
| 周线延续确认充分性 | 完整周低和收盘抬高能证明买入后的新延续 | 全9真实周期/45节点，周成分按实际买入钟拆分 | 全9晋级，近期4晋级后亏；7确认周含买前日，两例4前/1后 | 拒绝充分性；没有未来数据不等于新增持仓证据充分 | 周纯度/等几日不是当前bug修复，需新的机制用途 |
| 早期3笔可靠高胜率 | 3/3盈利能证明实际B和稳健成功 | 实际净B/pB/标准亏损单位期望，亏损样本保持空 | B/pB/标准期望UNKNOWN、年均0.6、区间跨0 | 拒绝可靠100%/无穷B/计划2R代实际B | 独立样本未建立，不另挑早期扩大结论 |
| pB与标准期望是同一门 | pB<1必然数学期望负或标准正即可采用 | 实际净回报p=2/6、B2.3729，标准pB−q | 近期pB0.79097、标准+0.12430 | 接受标准正与严格pB失败并存；拒绝偷换门 | 后续全部门共同保留 |
| 费用是全部失败原因 | 小资金减成本足以超过原A | 真实毛/费/净和逐日固定原数量零费解释 | 近期1457.80/1202.36/255.44；零费0.1119%/0.05959仍低A | 拒绝费用单因；接受摩擦吞掉82.48%弱毛优势 | 不调费或放风险救配置，解释边界非策略 |
| 账户约束造成失败 | 拒单或DD停买漏掉优势 | 全实际拒单/停买/风险減仓保存 | 主无拒单/停买，减仓15/7；早期3接受已有持仓 | 拒绝全部归因执行；现金与风险路径继续保持 | 新机制同资金约束，不把事件当成交 |
| 更多价格交易机会 | 无政策价格状态提高次数也提高收益 | 同持有动作无支持对照完整账户 | 压力早期−0.5653%/−0.12408、近期−1.1032%/−0.42946，年均9/9.5 | 拒绝密交易即收益；次数软目标不能营救毛优势 | 不对406点位逐参数挖掘；不同解释再登记 |
| 历史收益稳定性 | 早期显著点增量可推广近期 | 全四场景/四对照20和252日各2000，全部32区间 | 早期对A跨0；近期对A两尺度全部负，稳定失败 | 拒绝通用增量/去过拟合/独立验证 | 真正新样本或不同机制，不择时期拼接 |
| 日期格式接续 | ms/ns存储不同应阻止相同日期比较 | 保存原失败，临时日期ns化并逐值/索引一致，2测试 | 原12完整账户0重跑、16日期精确、全部区间完成 | 接受技术接续，不更改源日期/金融动作或经济门 | 新日期实际不同仍硬失败，不以格式修复经济失败 |
| 原A全部机会及持仓新增信息 | 新形成的结构、参与/资金与失效能指导持有并增加有质量机会 | 已完成54原A完成+1开放、9主周期/45节点/19日；下一完整日序解释 | 52A决定无活动支持，51A买日主空；Jun2’20只属旧阶段，A空集保留 | 接受最优先不同问题，未准入新账户或假设已能提升收益 | 下一完整描述、唯一动作/未知/对照登记，然后同资金风险金融 |

本轮原配置终局拒绝；下一PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN，0已准入待跑金融。保留原E03/13独立前瞻、旧失败和源码，历史都是开发，收益夏普目标未完成。

依据：[完整报告](../{report})、[金融结果](../{result})、[全周期及归因](../{diagnostic})、[下一问题](../{proposal_path})、[固定口径](510300_SUPPORT_PRICE_ACCEPTANCE_V1.md)。
"""
    prepared = []
    for name, text in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
                       ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)):
        path = ROOT / f"docs/{name}.md"
        prepared.append((path, *prepared_prepend(path, text)))
    state.update({
        "updated_at": study.parent.original.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": service["observed_at_utc"],
        "latest_goal_tool_status_receipt": proof, "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "latest_technical_decision": "TECH.R232", "latest_registration_decision": "TECH.R231",
        "latest_report": report, "latest_result": result, "latest_completed_study": relative(OUT),
        "latest_completed_description_study": relative(OUT),
        "latest_actual_financial_decision": "TECH.R232", "latest_actual_financial_result": result,
        "latest_actual_financial_status": summary["status"], "latest_research_status": summary["status"],
        "current_study": "510300_SUPPORT_PRICE_ACCEPTANCE_V1",
        "current_phase": "FIXED_SUPPORT_PRICE_ACCEPTANCE_COMPLETE_ACCOUNT_REJECTED_NEXT_ACTUAL_HOLDING_NEW_INFORMATION",
        "goal_turn_classification": "PROGRESS_R231_R232_TWELVE_NEW_FULL_ACCOUNTS_AND_HOLDING_DIAGNOSIS",
        "previous_phase_before_TECH_R231_R232": previous,
        "latest_progress": "完成12新账户及全部对照区间，配置拒绝；全9主周期及原A54完成/1开放解释留档，下一实际进入后新增延续信息和机会顺序。",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "current_financial_candidate_admission": "R232_FIXED_CONFIGURATION_REJECTED_NEXT_ACTUAL_HOLDING_INFORMATION_NOT_ADMITTED",
        "new_accounts_in_current_phase": 12, "necessary_tests_passed_in_current_phase": 10,
        "latest_support_price_acceptance_financial": result, "latest_holding_failure_diagnosis": diagnostic,
        "current_phase_trial_accounting": {
            "scope": "TECH_R231_R232_ONE_CONFIGURATION_TWELVE_NEW_FULL_ACCOUNTS_AND_FULL_DIAGNOSIS",
            "new_accounts": 12, "saved_controls_reused": 8, "all_account_metrics": 20,
            "new_financial_configuration_runs": 1, "date_completion_account_replays": 0,
            "original_tests": 8, "necessary_date_tests": 2, "prefix_checks": 19,
            "original_saved_control_checks": 8, "cash_account_checks": 12, "holding_timing_checks": 12,
            "date_value_comparisons": 16, "increment_interval_rows": 32,
            "all_source_observation_rows": 3488, "all_source_qualifications": 65,
            "support_activation_days": 26, "support_accepted_points": 12, "price_only_accepted_points": 406,
            "all_primary_stress_completed_cycles": 9, "all_A_completed_cycles": 54, "all_A_cycles_including_open": 55,
            "actual_holding_context_rows": 45, "all_original_key_points": 19,
            "original_frozen_inputs_exact": 63, "saved_completion_files_exact": 80,
            "figures_actually_viewed": 2, "figure_panels": 6,
            "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0},
        "current_goal_turn_actual_work": {
            "new_financial_purposes_completed": 1, "new_accounts": 12, "all_account_comparisons": 20,
            "all_increment_intervals": 32, "all_actual_primary_cycles_described": 9,
            "all_original_A_completed_and_open_cycles_described": 55,
            "new_different_mechanism_proposal": proposal["name"],
            "new_source_requests": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0},
        "next_support_price_acceptance_state_proposal": {
            **state["next_support_price_acceptance_state_proposal"],
            "status": "FIXED_CONFIG_REJECTED_AFTER_COMPLETE_FINANCE", "financial_admission": "COMPLETED_REJECTED_NOT_UNRUN",
            "registration": "TECH.R231", "decision": "TECH.R232", "result": result,
            "new_financial_runs": 1, "new_accounts": 12, "controls_reused": 8,
            "all_economic_gates_passed": False, "historical_stability_passed": False,
            "actual_rule_card": "docs/510300_SUPPORT_PRICE_ACCEPTANCE_V1.md"},
        "next_actual_holding_information_proposal": {**proposal, "proposal_path": proposal_path},
        "next_information_source_proposal": "先用全部原A与真实持仓本地可知信息完成新增证据时序；不无限收集全国新闻或反选本已拒绝参数。",
        "source_coverage_boundary_current_purpose": "RECORDED_POSITIVE_EVIDENCE_ONLY_EMPTY_NOT_POLICY_ABSENCE",
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "current_target_achieved": False, "overfitting_removed": False,
        "current_long_term_fact_sources": ["docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md",
                                           "docs/PROJECT_STATE_TECHNICAL_LINE.md", "docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md"]})
    if {key: state[key] for key in FORWARD} != forward_before:
        raise ValueError("原13前瞻字段改变，停止归档。")
    with backup.open("xb") as handle:
        handle.write(raw)
    docs = []
    for path, old, new, body in prepared:
        path.write_bytes(new)
        if path.read_bytes() != new or not path.read_bytes().endswith(body):
            raise ValueError("长期事实旧正文未精确保留。")
        docs.append({"path": relative(path), "old_sha256": digest(old), "new_sha256": digest(new), "old_body_preserved_exact": True})
    with STATE.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    saved = study.read(STATE)
    if {key: saved[key] for key in FORWARD} != forward_before or saved["latest_actual_financial_decision"] != "TECH.R232":
        raise ValueError("保存后的实际金融或原前瞻约定不一致。")
    frozen_after = check_frozen()
    if frozen_after != frozen_before:
        raise ValueError("归档影响原金融文件。")
    study.write(receipt_path, {"at": study.parent.original.now(), "decision": "TECH.R232",
        "docs": docs, "old_state_backup": relative(backup), "state_sha256": study.digest(STATE),
        "forward_fields_preserved_exact": list(FORWARD), "frozen_financial_before": frozen_before,
        "frozen_financial_after": frozen_after, "actual_financial_status": summary["status"],
        "all_economic_gates_passed": False, "historical_stability_passed": False,
        "latest_actual_financial_decision_updated": "TECH.R232", "previous_actual_financial_decision_archived": "TECH.R224",
        "goal_service_status": "active", "goal_turn_classification": saved["goal_turn_classification"],
        "consecutive_blocked_goal_turns": 0, "goal_achieved": False, "next_financial_admission": "NOT_ADMITTED"})
    print("R231—R232完整结果一次更新四份长期事实及项目状态：配置拒绝、目标active，旧策略与13独立前瞻保持。", flush=True)


if __name__ == "__main__":
    main()
