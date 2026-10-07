"""一次归档官方回顾目录与宣布覆盖缺口，保留金融失败和前瞻字段。"""
from __future__ import annotations

import json
from pathlib import Path

from research import official_policy_catalog_format_repair_v1_0_1 as repair
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import digest, prepared_prepend

ROOT, OUT, SOURCE, parent = repair.ROOT, repair.OUT, repair.SOURCE, repair.parent
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FINANCIAL = ("latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status")


def relative(path: Path) -> str:
    return path.absolute().relative_to(ROOT).as_posix()


def main() -> None:
    receipt_path = OUT / "project_state_update_receipt.json"
    backup = OUT / "state_before_TECH_R228.json"
    if receipt_path.exists() or backup.exists():
        raise RuntimeError("官方政策目录已归档或开始，禁止重复插入事实。")
    initial = parent.read(SOURCE / "summary.json")
    result = parent.read(OUT / "summary.json")
    diagnosis = parent.read(OUT / "post_run_diagnosis.json")
    delivery = parent.read(OUT / "delivery_receipt.json")
    viewed = parent.read(OUT / "figure_view_receipt.json")
    service = parent.read(OUT / "goal_service_status_after_result.json")
    if (initial["registration"] != "TECH.R227" or initial["decision"] != "TECH.R228"
            or initial["actual_catalog_rows"] != 68 or initial["all_source_instance_records"] != 1790
            or initial["native_requests_this_registered_run"] != 49 or initial["earlier_native_probes"] != 3
            or initial["saved_probe_reuses"] != 2 or initial["native_fetch_failures"] != 1
            or initial["necessary_tests_passed"] != 6 or len(initial["prefix_checks"]) != 19
            or not all(item["all_recorded_chain_prefix_rows_exact"] for item in initial["prefix_checks"])
            or result["all_catalog_rows"] != 88 or result["fixed_selected_documents"] != 46
            or result["source_instance_records"] != 1814 or result["logical_records"] != 1143
            or result["exact_day_unknown_source_rows"] != 6 or result["exact_day_unknown_logical_rows"] != 3
            or result["necessary_repair_tests_passed"] != 3 or result["unchanged_source_rows_exact"] != 1464
            or result["new_requests"] != 0 or result["new_daily_chain_runs"] != 0 or result["new_accounts"] != 0
            or diagnosis["source_documents_received"] != 45 or diagnosis["recorded_chain_new_event_count"] != 1
            or diagnosis["direct_same_day_policy_entry_gate_admitted"] or not viewed["all_one_actually_viewed"]
            or service["goal"]["status"] != "active"):
        raise ValueError("实际来源、原失败、未知日期、必要检查或目标状态不匹配。")
    if parent.digest(ROOT / delivery["report"]) != delivery["report_sha256"]:
        raise ValueError("实际交付报告与保存结果不一致。")
    if parent.digest(ROOT / delivery["figure"]) != delivery["figure_sha256"] or viewed["sha256"] != delivery["figure_sha256"]:
        raise ValueError("实际查看的图与交付图不一致。")

    raw = STATE.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R226" or state["latest_actual_financial_decision"] != "TECH.R224":
        raise ValueError("项目已由其他研究推进，不能覆盖其当前状态。")
    forward_before = {key: state[key] for key in FORWARD}
    financial_before = {key: state[key] for key in FINANCIAL}
    previous = {key: state.get(key) for key in (
        "latest_technical_decision", "latest_registration_decision", "latest_report", "latest_result",
        "current_phase", "goal_turn_classification", "current_phase_trial_accounting",
        "next_official_policy_announcement_information_proposal")}
    report = delivery["report"]
    result_path = relative(OUT / "summary.json")
    proof = relative(OUT / "goal_service_status_after_result.json")
    date = parent.original.now()[:10]
    note = f"""> 最新实际研究（{date}，TECH.R227—R228，官方政策来源及格式实现1.0.1）：固定央行大事记五页88入口和2015—2025全部、2026上半年46源文，45收到/2019Q2一TLS失败。整理1814源实例、1143规范全文逻辑记录；6实例/3逻辑记录具体日未知，523重复逻辑记录及671额外源实例保留。1143不是独立政策事件。登记后49原生请求、2原探测复用，先前3探测（含1未保存正文）总52；格式处理0新请求。原68目录/1790记录/失败不覆盖，7文段落及旧题目格式另实现，1464原行精确保持、57原文件不变。6原必要测试、3格式测试、19原整段截断通过，1时钟图已实际查看。0新金融/拟合/行情，最新实际金融仍R224拒绝，收益夏普目标未达；目标active、本轮PROGRESS、受阻0，13独立前瞻逐值保持。

实质结论：大事记是追溯目录，不是当时宣布全集。2019-01-04降准在本固定目录最早回顾公开为05-24 17:00，应复用原六降准研究的01-04政府报道及日期级上界，不能把5月资料倒填。2024-09-24原直播链R01/H02上界09:19:36、C01上界11:42:50，已记录降准降息、住房支持及资本市场工具额度；这些同源节点不是三条独立新闻，额度也不是实际股票购买。原利率背景仍旧07-22不表示当天没有政策；09-27实施来源日终上界、09-30原利率首次观察/PMI/完整周线确认分别保存。

覆盖缺口：原24链集中2024—2025；全部143原进入事件仅2024-09-24与链中新公布区间相交，不能用无记录等于无政策，也不能把一次交集作为高胜率门。大事记09-24四逻辑条目为住房条件通知及全文版本差异，不能代替直播三节点；2017-03-16目录有委员会人员调整、非完整操作加息公告，2020-03-30目录0记录也不证明没有公告。2015反弹、2017/2026假启动、所有‘其他’条目与来源失败保持；当前首版未认证、全公告覆盖与独立验证均未建立。

下一具体用途：原25个利率记录身份、完整准备金率导航45逻辑记录、资本市场工具3目录记录及原24链，固定逐条追溯操作公告/发布会/宣布—实施—更新与共同来源，不按原盈亏选源。25不是25次政策变化、45不是45次降准、3不是3个独立冲击。形成逐条已核对/未核对母集后，再解释‘宣布改变预期—量价启动—行业传播—持有失效’及完整反例，之后才另登记完整金融。当前源追溯PROPOSED，0已准入待跑金融；不新增同日政策硬门，不改旧失败或原E03下一实验。源登记/格式各一次完成，本次并未完成全国政策宣布历史全集。

依据：[完整报告与全部19点位](../{report})、[实际源结果](../{result_path})、[固定来源用途](510300_OFFICIAL_POLICY_ANNOUNCEMENT_CATALOG_V1.md)、[实际目标active](../{proof})。旧政策490目录/24链和六降准直接买入拒绝均保留；旧研究自身绝对收益门不自动移到新阶段用途。
"""
    decisions = f"""### TECH.R227—R228：官方政策目录、宣布钟与真正缺口（{date}）

固定88入口/46源文，45收到、1原TLS失败；1814源实例/1143规范全文逻辑记录、6实例具体日未知。原生49新请求及3前探测总52，格式实现0请求。6原测试/19前缀/3格式测试、1实际查看图；0新账户/拟合/行情，最新金融R224拒绝、目标active/PROGRESS、受阻0。下面接受来源与机制问题，不接受新交易策略。

| 方向 | 假设 | 验证方法 | 实际结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 固定栏目母目录 | 官方大事记可建立全部可见回顾索引 | 固定5页88入口，保留累计/季度版本和所有其他条目 | 原旧题目使68入口，新格式补齐88；46目标源集合及顺序不改 | 接受固定栏目完整，不能解释为全国全部政策完整 | 网站新发布另立新样本，原失败不覆盖 |
| 回顾日期当宣布可知钟 | 条目事件日可以作为策略已知日 | 保留事件日、实施日、来源实际公布钟并核对原链 | 2019-01-04条目本目录最早05-24公开；2024实施条目回顾晚于宣布 | 拒绝倒填来源钟；接受追溯导航，下一步复用或取得原公告 | 是，需原宣布来源，当前首版仍未认证 |
| 工具导航即政策强弱 | 分类和数量可直接多因子正向加分 | 七工具加其他全文保留，逐条考察持平/方向/版本 | 45准备金记录非45次降准；109LPR包括持平，住房/基准利率可能在其他 | 拒绝直接方向打分和只选标签；接受导航 | 实际经济含义及宣布变化需完整不同用途 |
| 多来源或多节点独立确认 | 相同文本/一发布会的多个措施可相加提高置信度 | 保存523重复逻辑、671额外源实例及原链共同源 | R01/H02/C01同一直播；不同全文称谓仍可能多逻辑ID | 拒绝来源数当独立冲击/胜率，接受共同来源关系 | 下一追溯先定义同公告经济身份和工具，不能靠收益合并 |
| 生效利率覆盖政策启动 | 原利率表无新钟代表当天无政策 | 原19具体点位/143事件与24核对链同钟连接 | 09-24旧利率与上午新宣布并存；09-30原慢指标确认 | 拒绝完整新闻覆盖，接受生效背景和不同信息顺序 | 宣布、实施、首次观察必须各保留自身钟 |
| 链空值代表没有政策 | 无24链新节点可以过滤所有其他启动 | 全3488日、143原事件与19关键点保留覆盖字段 | 仅1原事件与新链相交，链集中2024—2025；2017目录为人员调整，2020-03-30无条目 | 拒绝无记录=无政策；同日政策硬门未准入 | 是，先全范围源追溯，不按成功日期选取 |
| 月份和下旬补具体日 | 月份段落可用月底或首日补齐 | 2017两版本1月、2022四版本3月下旬原文保留 | 6实例/3逻辑记录日期NaT，仅期间边界，0未归类正文丢弃 | 接受未知和边界；拒绝人为给策略日期 | 有实质独立原公告才可另立核对，不改原未知 |
| 源文结构格式处理 | 旧题目/字体拆分应使可读正文缺失 | 保存原失败，单独1.0.1重连7源段落和旧题目，3必要测试 | 补20旧入口、2016Q3恢复18日期实例；1464未影响行精确，57原文件保持 | 接受技术处理，不是经济调参，不重算原链或金融 | 旧输入保持；未来不同结构另用途，不盲目重试TLS |
| 真正宣布与传播新机制 | 不同工具宣布可解释修复/重新定价而慢指标延后 | 下一固定25利率身份+45准备金逻辑+3资本目录+24原链，全母集原文/生效/同源/未知 | 目前PROPOSED_NOT_REGISTERED_OR_RUN；完整政策覆盖尚未建立 | 接受最值得继续的来源用途和时序假设；未接受买入门或收益提升 | 必须先完整源准入与反例，再冻结同资金成本风险账户和独立验证 |

原生来源失败不被年度重叠文本改写，原68/1790结果及解析失败完整保留。旧六降准直接买入全账户拒绝不重跑；其S1.2/CAGR10%合同只约束其原用途，不能机械代替当前收益/夏普增量与p×B>1、标准净期望正的合同。新历史来源不是独立未来样本。原金融拒绝和13前瞻字段逐值保持。

依据：[完整报告](../{report})、[实际结果](../{result_path})、[覆盖归因](../{relative(OUT / 'post_run_diagnosis.json')})、[固定用途](510300_OFFICIAL_POLICY_ANNOUNCEMENT_CATALOG_V1.md)。
"""
    prepared = []
    for name, text in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
                       ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)):
        path = ROOT / f"docs/{name}.md"
        old, new, body = prepared_prepend(path, text)
        prepared.append((path, old, new, body))
    with backup.open("xb") as stream:
        stream.write(raw)
    doc_receipts = []
    for path, old, new, body in prepared:
        path.write_bytes(new)
        if path.read_bytes() != new or not path.read_bytes().endswith(body):
            raise ValueError("原长期事实正文未逐字节保持。")
        doc_receipts.append({"path": relative(path), "old_sha256": digest(old), "new_sha256": digest(new), "old_body_preserved_exact": True})
    state.update({
        "updated_at": parent.original.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": parent.original.now(),
        "latest_goal_tool_status_receipt": proof, "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "latest_technical_decision": "TECH.R228", "latest_registration_decision": "TECH.R227",
        "latest_report": report, "latest_result": result_path,
        "previous_phase_before_TECH_R227_R228": previous,
        "latest_progress": "官方回顾五页88入口、46源45成功和完整未知归档；回顾不是宣布全集，下一全母集真正宣布与工具共同源追溯。",
        "current_phase": "OFFICIAL_POLICY_RETROSPECTIVE_CATALOG_COMPLETED_NEXT_ORIGINAL_ANNOUNCEMENT_SOURCE_TRACE",
        "goal_turn_classification": "PROGRESS_R227_R228_OFFICIAL_POLICY_SOURCE_CATALOG_AND_ANNOUNCEMENT_GAPS_COMPLETED",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "new_accounts_in_current_phase": 0, "necessary_tests_passed_in_current_phase": 9,
        "current_financial_candidate_admission": "R224_REJECTED_UNCHANGED_POLICY_SOURCE_NOT_NEW_FINANCIAL_ADMISSION",
        "current_source_admission": "FIXED_RETROSPECTIVE_CATALOG_COMPLETE_88_PARTIAL_ANNOUNCEMENT_CHAIN_NOT_ALL_POLICY_NEWS_NOT_FIRST_VINTAGE",
        "latest_official_policy_announcement_catalog": result_path,
        "current_phase_trial_accounting": {
            "scope": "TECH_R227_R228_OFFICIAL_POLICY_RETROSPECTIVE_SOURCE_AND_CHAIN_COVERAGE",
            "new_native_requests_after_registration": 49, "earlier_native_probes": 3, "native_total_including_unsaved_probe": 52,
            "saved_probe_reuses": 2, "fixed_index_entries": 88, "fixed_documents": 46, "received_documents": 45,
            "source_failures_kept": 1, "source_instance_records": 1814, "logical_records_not_independent_events": 1143,
            "exact_date_unknown_instances": 6, "necessary_original_tests": 6, "necessary_format_tests": 3,
            "original_prefix_checks": 19, "format_repair_new_requests": 0, "format_repair_chain_recomputations": 0,
            "all_original_daily_rows": 3488, "all_original_event_rows": 143, "all_original_case_rows": 240,
            "all_original_key_rows": 19, "figures_actually_viewed": 1,
            "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0, "new_financial_runs": 0},
        "current_goal_turn_actual_work": {
            "new_source_purposes_completed": 1, "new_native_requests_after_registration": 49,
            "all_official_index_entries": 88, "source_instance_records": 1814,
            "all_original_entry_events_coverage_checked": 143, "all_19_key_points_source_clocks_retained": True,
            "new_financial_purposes_completed": 0, "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0},
        "next_official_policy_announcement_information_proposal": {
            "status": "PARTIAL_PURPOSE_COMPLETED_CATALOG_NOT_ANNOUNCEMENT_COVERAGE",
            "registration": "TECH.R227", "decision": "TECH.R228", "result": result_path,
            "retrospective_index_entries": 88, "original_chain_nodes_reused": 24,
            "original_events_with_same_interval_new_chain": 1,
            "complete_policy_announcement_coverage": "NOT_ESTABLISHED",
            "direct_same_day_policy_gate_admitted": False, "financial_admission": "NOT_ADMITTED_NOT_RUN"},
        "next_information_source_proposal": "固定原25利率记录身份、45准备金率导航逻辑、3资本工具目录和24原链，全部追溯官方操作公告/发布会宣布、实施、共同来源与未知；不按原盈亏找源。",
        "next_original_policy_announcement_trace_proposal": {
            "status": "PROPOSED_NOT_REGISTERED_OR_RUN",
            "question": "当时真正可知的宣布、实施与工具类型，是否构成上涨启动和后续传播的可核对信息。",
            "fixed_scope": {"original_rate_record_identities_not_changes": 25,
                "all_rrr_navigation_logical_records_not_cuts": 45,
                "capital_tool_catalog_records_not_independent_shocks": 3, "original_verified_chain_nodes": 24},
            "required_sources": ["官方操作公告", "官方发布会与直播", "原已保存公告和政府报道", "原大事记仅追溯导航"],
            "required_fields": ["宣布与来源公开上界", "实施或执行日", "原文工具与方向", "同公告及共同来源关系", "未核对或未知"],
            "all_success_and_negative_cases_retained": True,
            "new_financial_run": "NOT_ADMITTED_NOT_RUN", "independent_validation": "NOT_ESTABLISHED"},
    })
    with STATE.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    saved = parent.read(STATE)
    if {key: saved[key] for key in FORWARD} != forward_before or {key: saved[key] for key in FINANCIAL} != financial_before:
        raise ValueError("旧独立前瞻或最新实际金融拒绝被改变。")
    parent.write(receipt_path, {
        "at": parent.original.now(), "status": "R228_POLICY_CATALOG_AND_SOURCE_LIMITS_ARCHIVED",
        "docs": doc_receipts, "old_state_sha256": digest(raw), "new_state_sha256": parent.digest(STATE),
        "forward_fields_preserved_exact": list(FORWARD), "latest_financial_fields_preserved_exact": financial_before,
        "latest_technical_decision": "TECH.R228", "goal_status": "active", "goal_achieved": False,
        "goal_turn_classification": saved["goal_turn_classification"], "consecutive_blocked_goal_turns": 0,
        "all_catalog_entries": 88, "fixed_source_documents": 46, "source_documents_received": 45,
        "source_instances": 1814, "logical_records_not_independent_policy_events": 1143,
        "new_accounts": 0, "next_original_announcement_trace": "PROPOSED_NOT_REGISTERED_OR_RUN"})
    print("R228官方目录、宣布覆盖缺口及下一用途已归档四长期事实；金融仍R224拒绝、原13前瞻保持，目标active。", flush=True)


if __name__ == "__main__":
    main()
