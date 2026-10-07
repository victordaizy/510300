"""一次保存全目录来源与覆盖事实，保留R216金融拒绝和全部独立前瞻。"""
from __future__ import annotations

import json

from research import official_industry_full_sequence_intake_v1 as intake
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import digest, prepared_prepend

ROOT, OUT, parent = intake.ROOT, intake.OUT, intake.parent
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def main():
    if (OUT / "project_state_update_receipt.json").exists() or (OUT / "state_before_TECH_R220.json").exists():
        raise RuntimeError("全序列来源已经归档或开始，不重复更新。")
    summary = parent.read(OUT / "implementation_v1_0_1/summary.json")
    source = parent.read(OUT / "summary.json")
    delivery = parent.read(OUT / "delivery_receipt.json")
    service = parent.read(OUT / "goal_service_status_after_result.json")
    views = parent.read(OUT / "source_figure_view_receipt.json")
    if (summary["decision"] != "TECH.R220" or summary["new_pdf_parses"] != 25 or summary["reused_pdf_parses"] != 8
            or summary["total_pages_read_or_reused"] != 2879 or summary["total_classification_rows"] != 124926
            or summary["complete_snapshots_passed"] != 19 or summary["row_source_structurally_eligible_snapshots"] != 32
            or summary["slots"] != 3488 or summary["evaluation_slots"] != 2855
            or summary["actual_member_rows"] != 846900 or summary["known_member_rows"] != 764101
            or summary["unknown_member_rows"] != 82799 or summary["new_accounts"] != 0
            or source["logical_gets"] != 54 or source["http_successes"] != 51 or source["request_failures"] != 3
            or delivery["clock_violations"] != 0 or views["viewed"] != 5
            or service["goal"]["status"] != "active" or summary["frozen_files_unchanged"] != 179):
        raise ValueError("来源、全日历、原失败或目标状态与实际保存结果不一致。")
    state_raw = STATE.read_bytes()
    state = json.loads(state_raw.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R218" or state["latest_actual_financial_decision"] != "TECH.R216":
        raise ValueError("项目事实已由其他研究推进，先核实。")
    previous_keys = ("latest_technical_decision", "latest_registration_decision", "latest_actual_financial_decision",
        "latest_actual_financial_result", "latest_report", "latest_result", "current_phase", "current_phase_trial_accounting",
        "current_source_admission", "next_information_source_proposal", "next_official_mainline_complete_source_proposal")
    previous = {key: state.get(key) for key in previous_keys}
    forward = {key: state[key] for key in FORWARD}
    report = relative(OUT / "行业主线_全发布来源与日历覆盖.md")
    result = relative(OUT / "implementation_v1_0_1/summary.json")
    proof = relative(OUT / "goal_service_status_after_result.json")
    date = parent.original.now()[:10]
    note = f"""> 最新全序列来源事实（{date}归档，TECH.R219—R220；实际金融仍R216拒绝）：固定保存目录36节点、复用原8，新增54GET/51成功/3失败、25PDF2177页；共33原件2879页/124926跨快照分类行。新用途在取得前区分完整快照门与明确证券行来源资格：19整表通过/32逐行结构合格，原R218三个整表失败保持；个别未知行不补、最新结构不合格不回退。全3488槽/2855评价槽、846900实际成员行已汇总，764101已知/82799未知，121039行业数量行。评价1561日300全知/995部分/299零已知（成员源缺32、最新源结构/取得失败267），全日期分母保留。0新账户/拟合/未来标签/行情日线，收益夏普NOT_COMPUTED；目标服务实际active、本轮PROGRESS、连续受阻0，完整目标未达。

实质新信息：原2019-01-08/06-19和2020-04-01在新逐行用途均300成员分类明确，但没有补写原5缺码或修改旧整表失败。2017Q1原PDF第14页同603026记两个公司，逐行结构不合格；2019Q2、2020Q1、2021Q1取得失败，各段全部保留。306评价槽源龄超过365日、最长820日；2022—2023旧表存在明确归属不等于已验证实时业务未变，旧源龄不隐去。CSRC与CAPCO版本不混申万，数量不是指数权重/贡献；当前取得未认证历史首版。

公布钟：实际公布日23:59，全3488槽晚于观察使用0；2024-09-30仍用2023H2，2026-09-30仍用2025H2，不提前使用当日2026H1。原2015前一成员缺口及2026-08-14之后成员缺口保持，不填未来或0。三个逐行测试和一个存储回归通过，原次轮失败回执保留；25PDF全解析后Arrow混合时间类型存储失败，隔离v1_0_1只统一时间类型ns，从已保存解析完成一次全日历；0重GET/解析，179文件保持、5原件图已查看。两项事前测试断言/类型失败和报告日期格式修正不改变来源门/策略。

下一步：固定行业集合、相对ETF强弱、扩散与轮动的数值描述，使用已有成分含息收益和保守前一源日；原四上涨、全部原阶段事件及恢复被截断反例同时解释。源龄与未知分别显示，不把数量当贡献，不按结果选赢家或窗口。当前为下一描述设计，尚非待跑金融；之后另立完整账户用途，配原A、价格和同源覆盖对照、原成本风险及pB门。原R212/R216拒绝、旧事实及13独立前瞻保持。

依据：[完整来源与覆盖报告](../{report})；[全部保存结果](../{result})；[全序列用途卡](510300_OFFICIAL_INDUSTRY_FULL_SEQUENCE_V1.md)；[目标服务实际active](../{proof})。
"""
    decisions = f"""### TECH.R219—R220：全目录发布序列与逐行来源用途（{date}归档）

新用途在新增28节点取得前固定，原8来源/解析复用，原R218整体失败不改。54新增GET/51成功/3失败、25新PDF与全部33原件2879页；全3488日历来源覆盖一次完成。0新账户/拟合/未来标签/行情日线，实际金融仍R216拒绝，服务active、PROGRESS、受阻0，完整目标未达。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 整表完整性与逐行来源 | 非成员未知分类无需否定其他明确行的不同观察用途 | 新用途预定原标题/证券集合/唯一身份与行出处，原整表门另列；未知股票不倒补 | 19完整快照通过、32逐行结构合格；原R218三整表失败保持；2019/2020三案例300已知 | 接受逐行来源用途，不接受改写原整表通过或交易门；字段含义不同 | 金融另登记，保持未知/源龄/版本与同覆盖对照 |
| 全保存目录来源 | 原8案例不足全日历，需要全部可见36发布节点 | 固定36、复用8，28节点每URL一次有限取得，全部错误保存 | 54GET51成功3失败，33PDF2879页/124926跨快照行 | 接受全已保存目录取得结果，不宣称全部历史没有其他源；缺2021Q4/2022未知 | 新不同来源用途可补，不重试当前冻结请求 |
| 非唯一证券身份 | 当行业表证券代码重复时可以按名称猜正确码 | 原表行/文本集合及PDF第14页画面核查 | 2017Q1的603026连续对应石大胜华与科森科技，重复2行 | 拒绝猜码或选一行；完整门与逐行结构门均失败，72评价槽保持未知 | 有明确更正或不同原件才新用途核实 |
| 三取得失败可回退旧表 | 缺最新原件仍可从较老原件提供当前已更新分类 | 最新公布节点固定选择，不自动退旧源 | 2019Q2/2020Q1/2021Q1共195评价槽取得失败未知，含72重复节点共267 | 拒绝当前最新失败回退，保留全部分母和源错误 | 新明确来源/钟合同可补；不是收益调参邀请 |
| 分类明确等于实时行业真相 | 表内有明确代码即可忽略行业表年龄 | 全日历原件年龄、年份和未知保留 | 306槽超过365日、最大820；2022/2023均无300全知日，但平均成员覆盖较高 | 拒绝明确分类即实时业务未变；接受旧已公布分类描述，源龄仅诊断无交易筛选 | 后续固定行业组须说明旧源角色，真正实时归属或首版另验证 |
| 全日历覆盖 | 原件取得成功即可宣称所有日可做主线 | 严格原前一成员、最新发布和逐行左连，全3488/2855分母 | 846900实际成员，764101已知/82799未知；1561全知、995部分、299零知 | 接受覆盖事实，拒绝把未知移出分母或填0/现金建议 | 描述及金融要定义未知分支与同覆盖对照，不选择有利阈值 |
| 同日发布提前可用 | 09-30新分类可以进入同日收盘 | 全槽available_at与15:00检查，两个真实终点核对 | 3488槽违规0，2024仍2023H2、2026仍2025H2 | 拒绝提前2026H1，接受实际23:59保守钟 | 有明确日内证据需不同版本，不由结果缩短钟 |
| 存储类型修复 | 混合旧JSON字符串与新Timestamp不应阻断来源汇总 | 保存ArrowTypeError原失败，单一类型修复回归，复用25新8旧解析一次完成 | 原3逐行测试/1存储回归通过，179文件保持；0重GET/解析 | 接受仅表示类型修复，拒绝把它叫改策略/补收益；事前测试失败保持 | 不重复全PDF，后续使用明确类型规范 |
| 行业结构改善收益夏普 | 行业相对强弱/扩散/轮动可区分弱势后的继续价值 | 下一个用途先描述全部原阶段事件和上涨/失效，再固定完整金融及归因对照 | 尚未数值构造行业收益结构/新账户，NOT_RUN/NOT_COMPUTED | 接受下一问题，不接受为已有收益策略；小资金不替代方向证据 | 是，完整同费用风险账户、pB>1及真正独立验证 |

所有历史仍开发/校准，首版NOT_CERTIFIED、独立NOT_ESTABLISHED。原13前瞻及R212/R216冻结拒绝保持；当前没有已准入待跑新金融。

依据：[完整来源报告](../{report})、[保存结果](../{result})、[新用途卡](510300_OFFICIAL_INDUSTRY_FULL_SEQUENCE_V1.md)。
"""
    prepared = []
    for name, text in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
            ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)):
        path = ROOT / f"docs/{name}.md"
        old, new, body = prepared_prepend(path, text)
        prepared.append((path, old, new, body))
    with (OUT / "state_before_TECH_R220.json").open("xb") as stream:
        stream.write(state_raw)
    docs = []
    for path, old, new, body in prepared:
        with (OUT / f"{path.stem}_before_TECH_R220.md").open("xb") as stream:
            stream.write(old)
        path.write_bytes(new)
        if path.read_bytes() != new or not path.read_bytes().endswith(body):
            raise ValueError("原事实正文未逐字节保持。")
        docs.append({"path": relative(path), "old_sha256": digest(old), "new_sha256": digest(new), "old_body_preserved_exact": True})
    state["previous_phase_before_TECH_R219_R220"] = previous
    state.update({"updated_at": parent.original.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": parent.original.now(),
        "latest_goal_tool_status_receipt": proof, "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "latest_technical_decision": "TECH.R220", "latest_registration_decision": "TECH.R219",
        "latest_report": report, "latest_result": result,
        "latest_progress": "全保存目录36发布节点、33原件/2879页、3488槽/846900成员来源与源龄完成；新逐行用途32结构合格，0新金融。",
        "current_phase": "OFFICIAL_FULL_VISIBLE_RELEASE_SEQUENCE_ROW_SOURCE_COVERAGE_COMPLETED_NEXT_INDUSTRY_STRUCTURE_DESCRIPTION",
        "goal_turn_classification": "PROGRESS_R219_R220_FULL_RELEASE_SOURCE_AND_846900_MEMBER_CLASSIFICATION_PANEL",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "new_accounts_in_current_phase": 0, "necessary_tests_passed_in_current_phase": 4,
        "current_financial_candidate_admission": "INDUSTRY_ROW_SOURCE_PANEL_ONLY_NO_ADMITTED_FINANCIAL_CANDIDATE",
        "current_source_admission": "ALL_VISIBLE_RELEASES_ROW_SOURCE_PANEL_32_STRUCTURALLY_ELIGIBLE_SNAPSHOTS_NOT_FIRST_VINTAGE_OR_FINANCIAL",
        "current_phase_trial_accounting": {"scope": "TECH_R219_R220_FULL_VISIBLE_OFFICIAL_SOURCE_ROW_COVERAGE",
            "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0,
            "native_source_gets": 54, "http_successes": 51, "request_failures": 3,
            "new_pdf_parses": 25, "reused_pdf_parses": 8, "total_pdf_pages_read_or_reused": 2879,
            "source_classification_rows": 124926, "complete_snapshots_passed": 19, "row_source_structurally_eligible_snapshots": 32,
            "all_calendar_slots": 3488, "evaluation_slots": 2855, "member_rows": 846900,
            "known_member_rows": 764101, "unknown_member_rows": 82799, "industry_count_rows": 121039,
            "all300_known_days": 1561, "partial_known_days": 995, "zero_known_days": 299,
            "source_age_over_one_year_days": 306, "max_source_age_days": 820,
            "necessary_row_tests": 3, "timestamp_regressions": 1, "pre_registration_test_failures_retained": 2,
            "original_parser_storage_failure_retained": 1, "completed_all_calendar_summary_runs": 1,
            "report_format_failure_corrected": 1, "source_figures_viewed": 5, "clock_violations": 0},
        "current_goal_turn_actual_work": {"native_source_gets": 54, "new_official_pdfs": 25, "new_pdf_pages": 2177,
            "full_calendar_source_rows": 3488, "source_member_panel_rows": 846900,
            "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0,
            "new_admitted_financial_purposes": 0, "new_source_description_purposes_completed": 1},
        "latest_official_industry_full_sequence_coverage": result,
        "next_official_mainline_complete_source_proposal": {"status": "COMPLETED_FIXED_FULL_VISIBLE_SOURCE_PURPOSE_FAILURES_RETAINED",
            "registration": "TECH.R219", "decision": "TECH.R220", "received_pdf_nodes": 33,
            "missing_or_structurally_failed_nodes": ["2017Q1", "2019Q2", "2020Q1", "2021Q1"], "financial_run": "NOT_RUN"},
        "next_information_source_proposal": "当前33原件/32逐行结构来源已形成全日历分类，下一步复用已有成分含息收益做行业相对强弱/扩散/轮动描述；源龄、缺码、3请求失败与2017重复全部保留。",
        "next_mainline_structure_description": {"status": "PROPOSED_NOT_REGISTERED_OR_RUN",
            "purpose": "固定当时行业集合并观察相对ETF强弱、行业参与与轮动，解释全部原事件和上涨/过早退出反例。",
            "inputs": [result, "已保存成分含息日收益及四状态可知表", "原阶段技术宏观已知表"],
            "new_financial_run": "NOT_RUN", "full_source_age_and_unknowns_preserved": True},
    })
    with STATE.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    saved = parent.read(STATE)
    if {key: saved[key] for key in FORWARD} != forward:
        raise ValueError("原13项独立前瞻被改变。")
    if (saved["latest_actual_financial_decision"] != previous["latest_actual_financial_decision"]
            or saved["latest_actual_financial_result"] != previous["latest_actual_financial_result"]):
        raise ValueError("来源归档改写最新真实金融。")
    parent.write(OUT / "project_state_update_receipt.json", {"at": parent.original.now(),
        "status": "FULL_VISIBLE_SOURCE_COVERAGE_ARCHIVED_OLD_FINANCIAL_FAILURE_PRESERVED",
        "docs": docs, "old_state_sha256": digest(state_raw), "new_state_sha256": parent.digest(STATE),
        "forward_fields_preserved_exact": list(FORWARD), "latest_technical_decision": saved["latest_technical_decision"],
        "latest_actual_financial_decision": saved["latest_actual_financial_decision"], "goal_status": saved["goal_status"],
        "goal_turn_classification": saved["goal_turn_classification"], "consecutive_blocked_goal_turns": 0,
        "new_accounts": 0, "new_financial_metrics": "NOT_COMPUTED"})
    print("全来源R220及四事实文档一次归档，原正文/13前瞻保持；金融仍R216，目标active。", flush=True)


if __name__ == "__main__":
    main()
