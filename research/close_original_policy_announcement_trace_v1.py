"""一次归档原公告时序与不同机制的下一用途，保留旧金融及独立前瞻。"""
from __future__ import annotations

import json
from pathlib import Path

from research import original_policy_trace_format_v1_0_1 as repair
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import digest, prepared_prepend

ROOT, OUT, SOURCE, parent = repair.ROOT, repair.OUT, repair.SOURCE, repair.parent
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FINANCIAL = ("latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status")


def relative(path: Path) -> str:
    return path.absolute().relative_to(ROOT).as_posix()


def main() -> None:
    receipt_path = OUT / "project_state_update_receipt.json"
    backup = OUT / "state_before_TECH_R230.json"
    if receipt_path.exists() or backup.exists():
        raise RuntimeError("原公告用途已归档或开始，不重复写入。")
    initial = parent.read(SOURCE / "summary.json")
    result = parent.read(OUT / "summary.json")
    diagnostic = parent.read(OUT / "post_run_diagnosis.json")
    delivery = parent.read(OUT / "delivery_receipt.json")
    viewed = parent.read(OUT / "figure_view_receipt.json")
    service = parent.read(OUT / "goal_service_status_after_result.json")
    proposal = parent.read(OUT / "next_support_acceptance_state_proposal.json")
    if (initial["original_reference_identities"] != 97 or initial["original_rate_sources_reused_and_verified"] != 25
            or initial["original_chain_nodes_reused"] != 24 or initial["fixed_lookup_groups"] != 42
            or initial["new_native_requests"] != 22 or initial["new_native_received"] != 18
            or initial["new_sources_content_and_clock_matched"] != 6 or initial["all_key_rows"] != 17
            or result["new_source_content_and_clock_matched"] != 15 or result["clock_conflicts_kept_unknown"] != 2
            or result["all_actual_nodes"] != 65 or result["unique_common_sources"] != 60
            or result["all_daily_rows"] != 3488 or result["all_events"] != 143 or result["all_key_rows"] != 19
            or len(result["prefix_checks"]) != 19 or not all(row["all_prior_rows_exact"] for row in result["prefix_checks"])
            or result["new_requests"] != 0 or result["new_accounts"] != 0
            or initial["necessary_tests_passed"] != 6 or result["necessary_format_tests_passed"] != 3
            or diagnostic["still_unknown_reference_identities"] != 26 or diagnostic["events_with_new_recorded_node"] != 2
            or diagnostic["old_gov_source_visible_date"] != "2019-01-05"
            or diagnostic["same_day_new_policy_hard_gate_admitted"]
            or not viewed["all_one_actually_viewed"] or viewed["panels_actually_viewed"] != 6
            or service["goal"]["status"] != "active" or proposal["financial_admission"] != "NOT_ADMITTED_NOT_RUN"):
        raise ValueError("实际来源、原案例、必要时序检查、未知或目标状态不匹配。")
    if (parent.digest(ROOT / delivery["report"]) != delivery["report_sha256"]
            or parent.digest(ROOT / delivery["figure"]) != delivery["figure_sha256"]
            or viewed["sha256"] != delivery["figure_sha256"]):
        raise ValueError("交付或实际查看文件与保存回执不同。")
    raw = STATE.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R228" or state["latest_actual_financial_decision"] != "TECH.R224":
        raise ValueError("项目由其他用途推进，不能覆盖其状态。")
    before_forward = {key: state[key] for key in FORWARD}
    before_financial = {key: state[key] for key in FINANCIAL}
    previous = {key: state.get(key) for key in (
        "latest_technical_decision", "latest_registration_decision", "latest_report", "latest_result",
        "current_phase", "current_study", "goal_turn_classification", "current_phase_trial_accounting",
        "next_original_policy_announcement_trace_proposal")}
    report, result_path = delivery["report"], relative(OUT / "summary.json")
    proof = relative(OUT / "goal_service_status_after_result.json")
    date = parent.original.now()[:10]
    note = f"""> 最新实际研究（{date}，TECH.R229—R230，原政策公告追溯及已保存格式实现1.0.1）：97原参考身份/42固定检索组，25原操作逐文利率与秒钟一致、24原链复用；22新原生请求18收到/4失败不重试。原实现6新源核对/17关键日留档，格式实现0请求，最终15新源内容/公开钟合格、2可见日与元数据跨年冲突/1字面匹配未知。97身份49原复用/22新或既有源关联/26仍未知；不是97独立政策。原55节点逐值保持，最终65节点/60共同源；全3488日、143原事件、原17+两假启动19关键日及19整段前缀完成，6原测试/3格式测试、1图六案例已实际查看。0新账户/拟合/行情，最新实际金融仍R224拒绝，收益夏普目标未达；目标active/PROGRESS、连续受阻0，13独立前瞻逐值保持。

事实纠正：R228所谓gov_20190104.html‘01-04政府来源’不准确。保存页面标2019-01-05，正文描述01-04决定；日期级公开上界01-05 23:59:59，本原ETF日历最早01-07 16:00可知。不能用该保存页事前解释01-04价格，也不能把5月回顾倒填。旧文档、旧六降准拒绝及旧账户不覆盖；本新事实为当前来源口径。2020-04-03央行降准公告16:57:32晚于原16:00，首次ETF可知04-07；03-30操作2.20%、04-01PMI/价格修复与后续宣布各自时点保留。

具体机制：2015操作利率下降仍伴随下跌，2.70%相对前一保存3.35%的差不是认证当日一次65bp降息；2017-03-16操作2.35→2.45%上调后04-05技术/PMI向好原假突破仍亏1181.26元。2019已知支持后01-08行业15/19上涨、日正/周负/PMI49.7，原01-09修复盈利7978.96元；2024-09-24上午同源宣布，相对量3.3384、行业75%、日正/周负，慢指标随后确认，原周期盈利3776.25元。2026-05-11日周及行业向好、旧1.40%背景、PMI环比负/融资未知，原周期亏1753.77元；无新节点不等于当天无政策。这些是原开发案例解释，不是新策略胜率或因果证明。

全体结果：仅2020-04-07和2024-09-24两个原进入事件与本固定来源新公布区间相交；原成功修复可以发生在更早支持之后。同日新公告硬门未准入，空记录不能当无政策，工具额度不能当真实股票流入；同文件多工具及宣布/实施不能独立投票。SOURCE_008可见2024-10-18与元数据2026-08-01、SOURCE_014可见2021-07-13与元数据2022-07-15冲突保持未知；旧24节点按原协议留档不偷偷改写。全国公告覆盖/历史首版/独立验证均未建立。

下一优先完整实验：已核对支持信息激活观察→固定已知价格区域并等待接受→确认后延续持有→结构失效或已知反向操作退出。研究信息持续作用及价格失效，不每个进入日重新要求全部宏观变好。唯一动作/失效规则与同价格无政策、同进入固定退出、原A/原阶段对照须在收益前固定；同20万元/费用/风险/两个时期、p×实际净B>1及标准净期望正、收益夏普增量共同核对。来源固定用途本轮结束，下步优先机制与完整账户，不能以追求全国公告全集无限延期。目前PROPOSED_NOT_REGISTERED_OR_RUN，0已准入待跑金融；旧失败、原E03下一实验保持。

依据：[具体上涨及全部反例完整报告](../{report})、[来源实际结果](../{result_path})、[固定源用途](510300_ORIGINAL_POLICY_ANNOUNCEMENT_TRACE_V1.md)、[下一机制提案](../{relative(OUT / 'next_support_acceptance_state_proposal.json')})、[实际目标active](../{proof})。原源码与原17实现都保留，只有新格式/病例连接另立实现；绘图时间存储ns/us不匹配失败留档，仅图表精度统一后接续，不重新采集或金融。
"""
    decisions = f"""### TECH.R229—R230：原公告、具体上涨与持续机会状态（{date}）

固定97原参考/42检索组；22新原生18收到/4失败，最终15新源钟内容合格/2跨日冲突/1匹配未知。25原操作/24链复用、49原身份/22关联/26仍未知；65节点60共同源，不是独立冲击数。原17描述保存，新实现原17+2反例19日/19整段前缀、6原及3格式测试、1图六案例实际查看。0新金融；原R224拒绝、目标active/PROGRESS、受阻0和13前瞻保持。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 策略随可知阶段变化 | 启动、传播、延续和失效可以需要不同证据与动作 | 招商等券商框架结合全日历及固定成功/假启动原案例 | 2019支持后修复、2024上午宣布后快速量价响应，慢统计/周线后到 | 接受新机制研究；未接受已能提升收益，更不复活旧配置 | 是，唯一新机制需完整账户及独立样本 |
| 当前生效利率是政策新闻全集 | 无新利率身份等于无新政策 | 25实际操作逐文核对、24链及新来源连接 | 09-24宣布早于09-29保存操作/09-30原利率首次观察 | 拒绝全集含义，保留实际操作与生效背景角色 | 公告和操作分开核对；无需为技术源重跑旧失败 |
| 保存操作差等于即时政策降幅 | 相邻保存记录差就是当日决策变化 | 25原利率和秒钟与冻结表一致，保存前值缺口 | 1前值未知、20相对保存值下降/4上升；2015Jun25距前保存跨度长 | 拒绝即时幅度和最早宣布认证，接受观察值变化事实 | 完整中间操作原件可支持不同用途，不补写本未知 |
| 政策宣布当天硬门 | 只在新宣布与进入同一区间才能赚钱 | 全143原事件与已核对节点按原16:00连接 | 仅2020Apr7和2024Sep24两事件相交；2019修复支持提前已知 | 单独硬门未准入，不能叫新策略拒绝/两笔100%胜率 | 下一完整状态用途允许信息在价格失效前持续作用 |
| 政策宽松直接买入 | 操作下降本身保证抛压结束 | 2015下跌和25全部操作、原政策直买失败保持 | 6月操作下降与价格下跌共存，旧六降准完整账户已拒绝 | 拒绝直接推买点；保留支持信息作为观察条件的假设 | 不救旧直买；不同价格接受机制另固定 |
| 常见指标多数正必然优质 | 日周柱、放量、PMI/行业多数正足够进入 | 19原关键日/2017和2026假启动及原周期 | Apr5’17及May11’26指标向好仍分别亏1181.26/1753.77元 | 拒绝直接多数投票，单案例不反选阈值 | 新机制要解释信息状态、价格接受及失效 |
| 多节点/多转载独立加分 | 同公告多工具或不同转载能提高独立置信度 | 保存65节点/60共同源、原事件和角色 | Sep24同源三节点，Sep27实施与Sep29操作重复确认 | 拒绝节点数当独立冲击/资金流，接受角色关系 | 需要经济事件归一和独立信息核对，当前未认证 |
| 2019政府来源日期 | 文件名gov_20190104意味着Jan4可知 | 原HTML可见页标、正文及原日历连接 | 正文Jan4宣布、页面Jan5、上界Jan5日終、ETF最早Jan7 | 接受事实纠正；R228旧表述不作为当前来源钟 | 新用途用纠正钟，旧冻结拒绝和旧文件留档 |
| 2020当天收盘已知公告 | Apr3收盘价格可用Apr3降准解释为事前信号 | 原公告时刻16:57:32与原16:00比较 | 首次ETF决定Apr7才可知 | 拒绝提前进入Apr3信号；接受后来价格传播用途 | 后续统一宣布/观察/次开实际成交时序 |
| 原生网页格式缺失等于无政策 | 不同DOM或数字排版不能匹配就表示没有内容 | 原6匹配/17描述留档，12保存格式修正、原17+2反例 | 最终15可核对，2时间冲突/1字面不符/4取得失败保持 | 接受技术读取和病例完整，拒绝消灭未知或日期倒填 | 无必要盲目重试；有新原件可新用途核对 |
| 现代网页首版与公布钟 | 可见历史日与元数据冲突可以自动选择有利的早日 | Source008/014保留可见日和PubDate，原24链保持原协议 | 两跨年冲突不给新源已知节点；全球首发/首版未认证 | 接受未知而非自动补历史；旧链只是原目的留档 | 需独立旧版本/发布证据，不能靠盈亏选择钟 |
| 小资金提高收益的来源 | 选择灵活和容量低即可得到高Sharpe | 保留20万元、整手/最低佣金/T+1/滑点和原R224费用事实 | 最新金融四经济门0/4、稳定失败；本轮NOT_COMPUTED | 接受执行灵活性作为设计依据，不接受自动方向优势 | 新完整账户同口径；次数软目标不能代替净收益 |
| 支持—价格接受—延续—失效 | 信息先激活，价格接受后进入，确认后改变持有，失效退出 | 下一唯一规则及同价格无政策/固定退出/原A/阶段对照事前固定 | 当前提案保存PROPOSED_NOT_REGISTERED_OR_RUN，0金融准入/运行 | 接受最优先不同机制研究，不声称策略已成功 | 下一优先完整账户，不把补全国新闻变成无限前置任务 |

完整固定母集追溯到此完成；全国公告覆盖与全网最早宣布仍未建立，缺失不作为无政策证据。所有历史是开发资料、金融目标未达。R212/R216/R224、旧六降准及13前瞻保留；旧合同绝对收益门不自动转移到新用途。当前是来源与机制进展，不是交易授权。

依据：[完整报告](../{report})、[实际结果](../{result_path})、[全部范围及归因](../{relative(OUT / 'post_run_diagnosis.json')})、[下一状态用途](../{relative(OUT / 'next_support_acceptance_state_proposal.json')})。
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
            raise ValueError("长期事实原正文未精确保留。")
        doc_receipts.append({"path": relative(path), "old_sha256": digest(old), "new_sha256": digest(new), "old_body_preserved_exact": True})
    state.update({
        "updated_at": parent.original.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": service["observed_at_utc"],
        "latest_goal_tool_status_receipt": proof, "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "latest_technical_decision": "TECH.R230", "latest_registration_decision": "TECH.R229",
        "latest_report": report, "latest_result": result_path, "latest_completed_study": relative(OUT),
        "current_study": "510300_ORIGINAL_POLICY_ANNOUNCEMENT_TRACE_V1",
        "latest_research_status": result["status"], "previous_phase_before_TECH_R229_R230": previous,
        "latest_progress": "固定原公告追溯、实际操作/宣布/实施与19原关键日完成；下一支持信息持续作用、价格接受和持有失效的完整新机制。",
        "current_phase": "ORIGINAL_POLICY_SOURCE_TRACE_COMPLETED_NEXT_SUPPORT_PRICE_ACCEPTANCE_STATE",
        "goal_turn_classification": "PROGRESS_R229_R230_ORIGINAL_ANNOUNCEMENT_SEQUENCE_AND_ALL_NEGATIVE_CASES_COMPLETED",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "new_accounts_in_current_phase": 0, "necessary_tests_passed_in_current_phase": 9,
        "current_financial_candidate_admission": "R224_REJECTED_UNCHANGED_NEXT_SUPPORT_STATE_NOT_REGISTERED_OR_RUN",
        "current_source_admission": "FIXED_97_REFERENCE_TRACE_COMPLETE_PARTIAL_ANNOUNCEMENTS_UNKNOWN_KEPT_NOT_FIRST_VINTAGE",
        "latest_original_policy_announcement_trace": result_path,
        "current_phase_trial_accounting": {
            "scope": "TECH_R229_R230_ORIGINAL_POLICY_SOURCE_TRACE_AND_CASE_SEQUENCE",
            "fixed_reference_identities_not_independent_policies": 97, "fixed_lookup_groups": 42,
            "fixed_queries_completed": 42, "title_or_index_followup_queries": 8, "official_index_opens": 6,
            "new_native_requests": 22, "new_native_received": 18, "source_fetch_failures_kept": 4,
            "original_rate_sources_verified": 25, "original_chain_nodes_reused": 24,
            "original_new_sources_matched": 6, "final_new_sources_matched": 15,
            "public_clock_conflicts_unknown": 2, "literal_matching_unknown": 1,
            "original_source_reference_identities": 49, "matched_new_or_local_references": 22, "unknown_references": 26,
            "actual_source_nodes": 65, "unique_common_sources": 60, "economic_events_independently_counted": False,
            "original_initial_key_rows": 17, "final_key_rows": 19, "all_original_daily_rows": 3488,
            "all_original_events": 143, "prefix_checks": 19, "original_necessary_tests": 6, "format_necessary_tests": 3,
            "format_new_requests": 0, "figures_actually_viewed": 1, "figure_case_panels": 6,
            "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0, "new_financial_runs": 0},
        "current_goal_turn_actual_work": {
            "new_source_purposes_completed": 1, "new_native_requests": 22, "all_fixed_reference_identities": 97,
            "all_original_events": 143, "all_original_and_negative_key_points": 19,
            "source_date_correction_saved": "gov_20190104_VISIBLE_20190105_NOT_EVENT_DAY_PUBLICATION",
            "next_materially_different_mechanism": proposal["name"],
            "new_financial_purposes_completed": 0, "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0},
        "next_original_policy_announcement_trace_proposal": {
            "status": "FIXED_REFERENCE_PURPOSE_COMPLETED_NOT_ALL_POLICY_ANNOUNCEMENT_COVERAGE",
            "registration": "TECH.R229", "decision": "TECH.R230", "result": result_path,
            "original_references": 97, "original_rate_sources": 25, "original_chain_nodes": 24,
            "unknown_reference_identities": 26, "complete_policy_announcement_coverage": "NOT_ESTABLISHED",
            "same_day_policy_hard_gate_admitted": False, "financial_admission": "NOT_ADMITTED_NOT_RUN"},
        "next_support_price_acceptance_state_proposal": {**proposal, "proposal_path": relative(OUT / "next_support_acceptance_state_proposal.json")},
        "next_information_source_proposal": "固定原公告追溯结束；下一优先支持信息持续作用、价格接受和持有失效的完整唯一机制，不以全国公告全集无限延期。",
        "next_research_question": "已核对支持出现后价格被接受并继续传播时，如何以可失效机会状态改变进入与持有，并较无政策及固定退出对照提高完整净CAGR和Sharpe？",
        "current_direction": "支持信息→固定当时价格区域→价格接受进入→确认延续→结构或反向操作失效；完整反例和原风险费用后再判断金融。",
    })
    with STATE.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    saved = parent.read(STATE)
    if {key: saved[key] for key in FORWARD} != before_forward or {key: saved[key] for key in FINANCIAL} != before_financial:
        raise ValueError("旧前瞻或最新实际金融拒绝字段改变。")
    parent.write(receipt_path, {
        "at": parent.original.now(), "status": "R230_ORIGINAL_POLICY_SEQUENCE_AND_CASES_ARCHIVED",
        "docs": doc_receipts, "old_state_sha256": digest(raw), "new_state_sha256": parent.digest(STATE),
        "forward_fields_preserved_exact": list(FORWARD), "latest_financial_fields_preserved_exact": before_financial,
        "latest_technical_decision": "TECH.R230", "goal_status": "active", "goal_achieved": False,
        "goal_turn_classification": saved["goal_turn_classification"], "consecutive_blocked_goal_turns": 0,
        "fixed_reference_identities": 97, "new_content_clock_sources": 15, "unknown_references": 26,
        "all_key_points": 19, "next_financial_admission": "NOT_ADMITTED_NOT_RUN", "new_accounts": 0,
    })
    print("R230原公告时序、日期纠正和下一不同机制归档四事实；旧金融R224/13前瞻保持，目标active。", flush=True)


if __name__ == "__main__":
    main()
