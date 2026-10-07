"""一次归档全账户新增信息描述，保留原金融拒绝、旧正文及独立前瞻。"""
from __future__ import annotations

import json

from research import actual_holding_information_format_v1_0_1 as repair
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import digest, prepared_prepend
from research.write_actual_holding_information_report_v1 import frozen_exact

parent, ROOT, OUT, SOURCE = repair.parent, repair.ROOT, repair.OUT, repair.SOURCE
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FINANCIAL = ("latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status")


def main():
    receipt_path, backup = OUT / "project_state_update_receipt.json", OUT / "state_before_TECH_R234.json"
    if receipt_path.exists() or backup.exists():
        raise RuntimeError("真实持仓信息用途已归档或开始，不重复更新。")
    result = parent.read(OUT / "summary.json")
    diagnosis = parent.read(OUT / "post_run_diagnosis.json")
    delivery = parent.read(OUT / "delivery_receipt.json")
    viewed = parent.read(OUT / "figure_view_receipt.json")
    service = parent.read(OUT / "goal_service_status_after_result.json")
    proposal = parent.read(OUT / "next_core_actual_acceptance_carry_proposal.json")
    tests = parent.read(SOURCE / "tests_receipt.json")
    time_tests = parent.read(OUT / "tests_receipt.json")
    if (result["decision"] != "TECH.R234" or result["saved_accounts_described"] != 8
            or result["all_daily_sequence_rows"] != 11268 or result["all_actual_saved_cycles_including_open"] != 128
            or result["all_actual_saved_completed_cycles"] != 126 or result["all_actual_saved_open_cycles"] != 2
            or len(result["prefix_checks"]) != 19 or not all(x["all_states_and_clocks_exact"] for x in result["prefix_checks"])
            or result["new_accounts"] != 0 or result["new_strategy_return_sharpe"] != "NOT_COMPUTED"
            or diagnosis["all_pressure_A_completed_cycles"] != 54 or diagnosis["all_pressure_A_open_cycles"] != 1
            or diagnosis["all_pressure_primary_completed_cycles"] != 9 or diagnosis["A_low_before_acceptance_existing_positive"] != 6
            or diagnosis["A_low_after_acceptance_existing_positive"] != 2 or diagnosis["A_low_without_acceptance_existing_negative"] != 11
            or diagnosis["A_positive_without_new_whole_week"] != 13 or diagnosis["A_recent_positive_without_new_whole_week"] != 10
            or delivery["all_conditions_rows"] != 64 or delivery["all_sequence_order_rows"] != 30
            or tests["passed"] != 7 or tests["exit_code"] != 0 or time_tests["passed"] != 2 or time_tests["exit_code"] != 0
            or not viewed["all_one_actually_viewed"] or viewed["panels_actually_viewed"] != 6
            or delivery["figure_actually_viewed"] is not True or service["goal"]["status"] != "active"
            or proposal["financial_admission"] != "NOT_ADMITTED" or result["goal_achieved"]):
        raise ValueError("全范围事实、时序、原结果、必要测试或目标状态不匹配。")
    if (parent.digest(ROOT / delivery["report"]) != delivery["report_sha256"]
            or delivery["figure"] != viewed["figure"]
            or parent.digest(ROOT / delivery["figure"]["path"]) != delivery["figure"]["sha256"]):
        raise ValueError("完整报告或实际查看图与回执不同。")
    before_frozen = frozen_exact()
    raw = STATE.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R232" or state["latest_actual_financial_decision"] != "TECH.R232":
        raise ValueError("项目已由其他用途推进，不能覆盖。")
    forward_before = {key: state[key] for key in FORWARD}
    financial_before = {key: state[key] for key in FINANCIAL}
    previous = {key: state.get(key) for key in ("latest_technical_decision", "latest_registration_decision", "latest_report", "latest_result",
        "current_phase", "current_study", "goal_turn_classification", "current_phase_trial_accounting", "next_actual_holding_information_proposal")}
    report, summary = delivery["report"], repair.relative(OUT / "summary.json")
    diagnostic, proof = repair.relative(OUT / "post_run_diagnosis.json"), repair.relative(OUT / "goal_service_status_after_result.json")
    proposed = repair.relative(OUT / "next_core_actual_acceptance_carry_proposal.json")
    date = parent.previous.parent.original.now()[:10]
    note = f"""> 最新实际研究（{date}，TECH.R233—R234，真实进入后的新增量价/宏观信息与延续失败顺序）：8原保存账户（原A与R232支持原型、两时期/两费用），11268持仓及退出后日行、128费用复本周期（126完成/2开放，非128独立机会）。压力原A54完成/1开放、支持原型9完成全部解释，原3488观察日/65来源节点/19关键日保持。7必要测试/2时间类型测试、19整段前缀120账户对/93508重叠已有日行精确；1图六案例实际查看。原一次生成八序列在首个前缀全NaT秒/ns类型断言退出，失败和输入保存；1.0.1只临时规范时间类型逐值核对，原八全序列0重建、0金融。62原冻结文件/7接续文件精确。新收益/夏普/策略胜率NOT_COMPUTED，最新实际金融仍R232拒绝、目标未实现；服务active/PROGRESS、受阻0，13独立前瞻/E03及旧正文保持。

核心事实：压力原A54完成中36曾在持有中价格接受，20有全部买入日起的新完整周接受，后者原结果18正2负；存在持有存续和价格路径选择，不可称新策略90%胜率。13原A盈利未形成新周，其中近期10笔。Jul1—Jul7’20原净利13302.43、Sep25—Sep30’24净利11030.84只有4/3持仓收盘；快速重定价不能普遍要求进入前等新周。R232的四近期亏损持有中均未出现新完整周，但‘后来未出现’须等原持有结束才知道，不能算四笔已能事前避免。

顺序事实：原A7笔先低点失败后价格接受，原结果6正1负；7笔先接受后低点失败，2正5负；11笔低点失败且在原持有中没有价格接受，全为负。最后完整类别在第一次低点失败时不可知。全部8笔曾低点失败但原最终盈利保留，其中6先跌后修复、2先接受后失败；不能直接统一进入日低止损或按后验类别报新pB。价格/周结构使用实际进入日高低在其收盘后固定，只用于后来持有，未倒用于当天开盘。

信息角色：原A54完成仅7笔在持有中遇到本有限集合真正新支持公布，不能等同全国无政策。09:30真实进入与16:00原ETF观察、公布上界分别保留；进入前公布但买后首次观察、买后真新公开、目标操作确认与反向操作不独立投票。每日融资/资金更新不自动当冲击，PMI为新订单调查背景；行业逐点分类/源龄/领先身份保留，未认证固定进入成员传播，不把变化比例直接作新扩散或ETF成交量作真实净流入。未知不填0；截至Sep30历史全开发、首版/独立未建立。

具体案例：2015Jun3原A曾价格和新周接受，仍Jun23净亏5715.56，确认不保证。2019支持原型Jan10盈利16105.95与原AFeb28盈利1295.87来自不同进入/仓位，不相减当持有增量。2020Apr3进入依Mar30操作/Apr2决定，Apr3晚公告最早Apr7；Jun2是旧阶段修复（10756.16），原A该日没有进入，真正原AJul1快上涨保留。2024原ASep25买/Sep30退盈利11030.84，Oct9重入/Oct10亏2808.61；支持原型Sep26—Oct17盈利7747.24，延长一笔必须纳入后续现金路径。2021/2023买前四日的确认周与买后新完整周区别保留。图标真实成交日期，纵坐标对齐当日现金平移收盘，实际成交价在原订单。

下一最值得继续的唯一完整用途：保留原A进入与权重来源，实际进入后根据接受、失效、新完整周及已知参与证据改变持有，宏观真新公开信息撤销延续资格，未知保持CORE；强制退出消费旧原正目标段，避免立即重复旧信号。原进入保持，先解释和固定持有增量；同持有无新增信息、同接受保护不延长、原A/原阶段完整对照，同资金风险费用两时期，CAGR/净Sharpe增量、DD<=10%、实际pB>1与标准期望正、全部两尺度区间。新提案PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN，0已准入待跑金融；保护位/转换/锁定/未知细节须唯一事前固定。不是救R232周纯度、反选窗口/来源或拼接时期。

依据：[全部真实交易与具体上涨完整解释](../{report})、[实际描述结果](../{summary})、[全部归因](../{diagnostic})、[固定用途](510300_ACTUAL_HOLDING_INFORMATION_V1.md)、[下一不同整体机制](../{proposed})、[服务active](../{proof})。最新金融R232三字段逐值保持；本轮不声称提高收益已成功或去过拟合。
"""
    decisions = f"""### TECH.R233—R234：真实进入后的新证据及完整价格顺序（{date}）

八保存账户/11268日序列/128费用复本126完成2开放，压力A54完成1开放+原型9全部保留；3488原槽/65节点/19关键日，7+2测试、19截断120对93508重叠行精确、六案例图查看。原全NaT时间断言失败留档、类型接续0全序列重建/金融；62原文件/7接续文件精确。最新实际金融R232拒绝、新收益NOT_COMPUTED，目标active/PROGRESS、受阻0和13前瞻保持。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 实际进入后的新结构 | 完整周可知意味着均是买后形成 | 固定真实买入高低，逐完整周首末及当前钟，全部8账户 | R232两旧确认含4买前/1买后；新周条件另列、不改旧规则 | 接受新证据身份描述，拒绝旧确认即买后新结构 | 新整体持有用途须唯一固定，未金融检验周纯度 |
| 新周接受组等于新高胜率 | 原20已确认交易18盈利可报新90% | 全54完成、开放和费用复本，条件只在买后出现 | 原结果18正2负，存续/价格路径选择，进入未知是否确认 | 拒绝直接策略胜率/无穷B/独立成功，接受原结果标注 | 完整事前状态与现金账户才能验证 |
| 统一等周进场 | 所有盈利需完整买后周确认 | 全缺确认赢家保留、原快上涨时钟 | 13盈利无新周，近期10；Jul’20/Sep’24仅4/3收盘盈利13302.43/11030.84 | 拒绝普适慢确认门；快速重定价与长修复不同 | 保留进入先测不同持有动作，不按赢家设新进入门 |
| 统一进入日低止损 | 触低即证明修复无效 | 54原压力周期，首次接受和低失败顺序 | 8触低后仍原盈利，6先跌后接受、2先接受后失败 | 拒绝统一充分性，接受不同顺序的机制问题 | 需完整提前卖出/现金重入/费用影响，不能直接算节约 |
| 后验顺序可实时预知 | 11触低不接受全亏说明首次触低可正确退出 | 全原顺序7前6正/7后2正/11不接受全负 | ‘后来不接受’只在原持有终点可知 | 拒绝未来类别当标签/即时规则，保留全部路径解释 | 新状态只能用截至当前信息，不选择未来命运 |
| 新公开与新观察相同 | 原记录首次见到是买后的新支持冲击 | 原公布上界/首次ETF观察、09:30入场/16:00观察 | A54只7遇本集合买后新支持；旧观察和目标确认单列 | 接受角色分离，拒绝同公告多票和新消息硬门 | 后续宏观可撤销状态，旧背景及未知不能自动触发 |
| 行业比例差是主线扩散 | 逐点比例升高代表固定成员新传播 | 分类/源龄/领先身份/known完整保留 | 逐点集合变动，固定进入群体同一性未认证 | 接受逐点参与，拒绝把比例差或排名直接作同组新扩散 | 新用途需事前测量身份，当前不新增投票 |
| 原A单笔与原型单笔同资金 | 相减2019/2024盈利可证明持有改善 | 全A加减仓/目标与原型只减仓，实际日期和全部资金路径 | A2024三收盘赢11030.84后Oct9亏2808.61，原型更晚长持赢7747.24 | 拒绝单笔差作纯持有增量；接受统一进入后对照需要 | 下一完整策略保持A源/权重，纳入后续现金反馈 |
| 时间存储差等于特征不因果 | 全NaT秒/ns类型断言失败应改经济规则 | 原失败保留，临时已声明时间ns化、逐值/索引与其他特征严格，2测试 | 19前缀120对93508既有重叠行精确，原序列0重建 | 接受格式接续，有限无时区公布钟或真实差异仍拒绝 | 不以类型问题营救金融失败或更换日期 |
| 下一实际接受/失效持有 | A进入保留，价格/新周/参与和真新宏观改变持有能提高收益夏普 | 本轮128费用复本及全序列解释完成；下一唯一动作及四对照 | PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN，0新金融 | 接受最优先不同完整问题；尚未接受绩效或过拟合去除 | 先完整规则/必要对照与测试冻结，再一次完整账户 |

依据：[完整解释](../{report})、[实际完成](../{summary})、[全路径归因](../{diagnostic})、[下一提案](../{proposed})、[固定用途](510300_ACTUAL_HOLDING_INFORMATION_V1.md)。原金融和旧正文保持，本用途结束不等于项目停止。
"""
    prepared = []
    for name, text in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
                       ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)):
        path = ROOT / f"docs/{name}.md"
        prepared.append((path, *prepared_prepend(path, text)))
    state.update({
        "updated_at": parent.previous.parent.original.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": service["observed_at_utc"],
        "latest_goal_tool_status_receipt": proof, "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "latest_technical_decision": "TECH.R234", "latest_registration_decision": "TECH.R233",
        "latest_result": summary, "latest_report": report, "latest_completed_study": repair.relative(OUT),
        "latest_completed_description_study": repair.relative(OUT), "latest_research_status": result["status"],
        "current_study": "510300_ACTUAL_HOLDING_INFORMATION_V1",
        "current_phase": "ACTUAL_ENTRY_NEW_INFORMATION_FULL_DESCRIPTION_COMPLETE_NEXT_CORE_ACCEPTANCE_FAILURE_CARRY",
        "previous_phase_before_TECH_R233_R234": previous,
        "goal_turn_classification": "PROGRESS_R233_R234_ALL_SAVED_ACCOUNT_NEW_INFORMATION_AND_FAILURE_ORDER_COMPLETED",
        "latest_progress": "全八保存账户实际进入后的11268日序列和128费用复本周期完成，明确低点失败/接受顺序及快速上涨；下一同A进入的新增信息持有完整机制。",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "current_financial_candidate_admission": "R232_REJECTED_UNCHANGED_NEXT_CORE_ACTUAL_ACCEPTANCE_CARRY_NOT_ADMITTED",
        "new_accounts_in_current_phase": 0, "necessary_tests_passed_in_current_phase": 9,
        "latest_actual_holding_information_description": summary,
        "latest_actual_financial_failure_diagnosis": "reports/research/510300_support_price_acceptance_v1/implementation_v1_0_1/post_run_diagnosis.json",
        "latest_holding_information_sequence_diagnosis": diagnostic,
        "current_phase_trial_accounting": {
            "scope": "TECH_R233_R234_ALL_SAVED_ACCOUNT_ACTUAL_ENTRY_NEW_INFORMATION_DESCRIPTION",
            "original_observation_rows": 3488, "policy_receipt_nodes": 65, "saved_accounts_described": 8,
            "full_sequence_rows": 11268, "saved_cycles_with_cost_replicates": 128, "completed_replicates": 126, "open_replicates": 2,
            "pressure_A_completed": 54, "pressure_A_open": 1, "pressure_primary_completed": 9,
            "all_original_key_days": 19, "prefix_checks": 19, "account_prefix_pairs": 120,
            "overlapping_existing_rows_exactly_compared_not_new_observations": 93508,
            "original_necessary_tests": 7, "necessary_time_tests": 2, "original_frozen_files_exact": 62, "completion_files_exact": 7,
            "original_saved_full_sequence_regenerations": 0, "figures_actually_viewed": 1, "figure_panels": 6,
            "new_accounts": 0, "new_financial_runs": 0, "new_fits": 0, "new_labels": 0, "new_market_requests": 0},
        "current_goal_turn_actual_work": {
            "new_complete_description_purposes_completed": 1, "all_saved_accounts": 8, "full_actual_holding_and_after_exit_rows": 11268,
            "all_pressure_original_A_completed_and_open": 55, "all_pressure_primary_completed": 9,
            "new_materially_different_complete_use_proposal": proposal["name"],
            "new_financial_purposes_completed": 0, "new_accounts": 0, "new_source_requests": 0, "new_fits": 0, "new_labels": 0},
        "next_actual_holding_information_proposal": {
            **state["next_actual_holding_information_proposal"], "status": "COMPLETED_FULL_DESCRIPTION_NEXT_CORE_ACCEPTANCE_CARRY_NOT_ADMITTED",
            "registration": "TECH.R233", "decision": "TECH.R234", "result": summary, "saved_accounts": 8,
            "financial_admission": "DESCRIPTION_ONLY_NO_NEW_FINANCIAL_RUN", "new_financial_runs": 0},
        "next_core_actual_acceptance_carry_proposal": {**proposal, "proposal_path": proposed},
        "next_information_source_proposal": "已有公开钟与全持有时序用途完成；下一优先同A进入的接受/失效/参与与宏观角色持有机制，不为全国新闻全集无限延后。",
        "current_new_strategy_return_sharpe": "NOT_COMPUTED", "history_role": "DEVELOPMENT_CALIBRATION",
        "independent_validation": "NOT_ESTABLISHED", "current_target_achieved": False, "overfitting_removed": False})
    if {key: state[key] for key in FORWARD} != forward_before or {key: state[key] for key in FINANCIAL} != financial_before:
        raise ValueError("原独立前瞻或金融三字段改变，停止归档。")
    with backup.open("xb") as handle:
        handle.write(raw)
    docs = []
    for path, old, new, body in prepared:
        path.write_bytes(new)
        if path.read_bytes() != new or not path.read_bytes().endswith(body):
            raise ValueError("原长期事实正文未精确保留。")
        docs.append({"path": repair.relative(path), "old_sha256": digest(old), "new_sha256": digest(new), "old_body_preserved_exact": True})
    with STATE.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    saved = parent.read(STATE)
    if {key: saved[key] for key in FORWARD} != forward_before or {key: saved[key] for key in FINANCIAL} != financial_before:
        raise ValueError("保存状态未保持前瞻与金融事实。")
    after_frozen = frozen_exact()
    if after_frozen != before_frozen:
        raise ValueError("项目归档改变原固定来源或序列。")
    parent.write(receipt_path, {"at": parent.previous.parent.original.now(), "decision": "TECH.R234", "docs": docs,
        "old_state_backup": repair.relative(backup), "state_sha256": parent.digest(STATE),
        "old_financial_fields_preserved_exact": financial_before, "forward_fields_preserved_exact": list(FORWARD),
        "frozen_before": before_frozen, "frozen_after": after_frozen, "new_financial_runs": 0,
        "goal_service_status": "active", "goal_turn_classification": saved["goal_turn_classification"],
        "consecutive_blocked_goal_turns": 0, "goal_achieved": False, "next_financial_admission": "NOT_ADMITTED"})
    print("R233—R234一次更新四份长期事实和项目状态，R232金融三字段与13独立前瞻精确保持。", flush=True)


if __name__ == "__main__":
    main()
