"""归档进入信息完整解释，保留金融拒绝、历史正文和独立前瞻。"""
from __future__ import annotations

import json

from research import entry_information_sequence_description_study_v1 as study
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import digest, prepared_prepend

ROOT, OUT, parent = study.ROOT, study.OUT, study.parent
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def main():
    if (OUT / "project_state_update_receipt.json").exists() or (OUT / "state_before_TECH_R226.json").exists():
        raise RuntimeError("进入信息描述已归档或开始，不重复更新。")
    summary = parent.read(OUT / "summary.json")
    diag = parent.read(OUT / "post_run_diagnosis.json")
    delivery = parent.read(OUT / "delivery_receipt.json")
    views = parent.read(OUT / "figure_view_receipt.json")
    service = parent.read(OUT / "goal_service_status_after_result.json")
    if (summary["decision"] != "TECH.R226" or summary["daily_rows"] != 3488 or summary["evaluation_rows"] != 2855
            or summary["original_events"] != 143 or summary["case_rows"] != 240 or summary["negative_window_rows"] != 38
            or summary["new_accounts"] != 0 or summary["necessary_tests_passed"] != 6 or len(summary["prefix_checks"]) != 19
            or not all(item["all_prior_rows_exact"] for item in summary["prefix_checks"])
            or delivery["all_figures_actually_viewed"] != 6 or not views["all_six_actually_viewed"]
            or service["goal"]["status"] != "active" or diag["orders_new_publication_events"] != 11
            or diag["policy_rate_new_publication_events"] != 0 or diag["single_freshness_gate_admitted"]):
        raise ValueError("进入信息完整结果、交付或实际目标状态不匹配。")
    raw = STATE.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R224" or state["latest_actual_financial_decision"] != "TECH.R224":
        raise ValueError("项目状态已被其他实际研究推进。")
    forward = {key: state[key] for key in FORWARD}
    old_financial = {key: state[key] for key in ("latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status")}
    report = delivery["report"]
    result = study.relative(OUT / "summary.json")
    proof = study.relative(OUT / "goal_service_status_after_result.json")
    date = parent.original.now()[:10]
    note = f"""> 最新实际研究（{date}，TECH.R225—R226）：进入信息更新与量价行业顺序已一次完整解释。3488日/2855研究日、143原事件、四原案例240行/17关键日、两个新增亏损38窗口行；6测试、19整段因果截断、6图实际查看。0新账户/拟合/未来训练目标/行情/金融重跑。最新实际金融仍R224固定行业失效拒绝，收益夏普目标未达；目标服务active、本轮PROGRESS、受阻0，原13前瞻保持。

核心事实：PMI/对应利率/日常资金/融资身份首次观察，与公布钟处于上一ETF决定之后到当前决定之间，分别描述；未知与恢复旧身份不补0。宏观仍原16:00，行业仍前一完整源日。原2855日利率25首次记录、只有1区间新钟、24首次见到旧钟；143事件142利率可知但0新利率公布，不能据此说没有政策宣布。PMI142可知/1未知、11区间新公布，资金142、融资135更新是日常记录。首版未认证。

具体上涨：2019-01-08日柱正/完整周负、行业15/19上涨、PMI49.7且环比−0.7；01-09修复R224原盈利7978.96元，是同利率同路线此前2次后的信号。2020-04-01日柱转正/周负、14/18行业上涨、PMI29.3→52.0，新生效利率记录已知钟旧于当前决定区间；06-02盈利修复同利率同路线此前3次，不能硬要求首次。2024-09-24量3.3384、日正/周负、12/16行业上涨，PMI48.9/融资负/利率仍07-22记录；原盈利3776.25元保持，09-30慢指标才确认。2015反弹仍日柱负，22日行业未知，不用后来分类证明当时无主线。

反例：2017-04-05首次同PMI/同利率突破，相对量1.5408/日周柱正、PMI53.3且+0.3，行业未知，原04-06至04-19亏1181.26元。2026-05-11同PMI首次突破，相对量2.3905/日周正、行业75%上涨、领先5日+5.9283%、资金缺口−0.1013，PMI50.6但−1.0、融资未知，原05-12至05-19亏1753.77元；对应利率同路线此前9次/源龄368天。首次PMI或常见技术多数向好不够。

全体检验：PMI区间新公布11事件，原阶段4已有周期3正1负、R224同4周期4正，另7无实际周期，不能当新100%胜率/11交易。PMI同突破路线重复21事件原阶段/R224都0实际完成周期，单纯删除重复信号不能假设净值改变；同利率重复较差上下文也不能反选阈值。当前首次/新消息作为单独进入门未准入，而非重新计算或拒绝一套尚未运行金融策略。

下一最值得做：不同来源的官方政策宣布信息与工具类型，分别保存宣布时刻、生效日、降准/利率/资本市场工具；先说明整个研究日历/全部事件的覆盖和未知，不仅收集成功日期，不能把生效利率当完整新闻钟。2019/2024成功段与2017/2026反例共同解释后才登记启动—传播—持有—失效完整机制；当前只是PROPOSED，未登记/采集/运行，0已准入待跑金融。R212/R216/R224失败、R220源失败、R222行业事实及E03均保留，历史开发/独立未建立。

依据：[具体上涨与假启动完整解释](../{report})、[一次结果](../{result})、[固定用途](510300_ENTRY_INFORMATION_SEQUENCE_DESCRIPTION_V1.md)、[实际目标状态](../{proof})。
"""
    decisions = f"""### TECH.R225—R226：进入信息更新与量价行业顺序（{date}）

完整3488日/143原事件/240原案例/两个新增亏损38窗口行，6测试、19整段前缀精确、6图已实际查看。0新金融/拟合/训练目标/行情/重跑；最新实际金融R224拒绝保持，目标active、PROGRESS、受阻0。资料首版未认证、历史开发、独立未建立。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 首次观察与新公布 | 原表首次见到记录可作新消息 | 四源身份/时钟/原known因果处理，左开右闭区间，整段19截断 | 原2855日利率25首次身份只有1区间新钟、24是旧公布 | 接受分别描述；拒绝首次数据=新宣布 | 新不同公告来源需要自身公布/生效钟，当前不改旧钟 |
| 对应政策利率新闻覆盖 | funding_policy_known_at覆盖全部政策宣布 | 3488日与143事件、具体2019/2024时序对照 | 142事件利率可知但0新钟，2024-09-24仍指07-22 | 拒绝完整新闻覆盖假设；保留已生效利率背景，不能推没有新政策 | 是，真正官方宣布与工具类型属于不同来源用途，当前未登记 |
| 日常更新等于冲击 | 资金/融资每日新记录说明启动 | 全143更新与量价行业案例/失败对照 | 资金142、融资135更新，但两新增亏损也有更新 | 拒绝日常更新直接当独立政策冲击 | 不据此新建买入门；不同具体信息需事前定义 |
| 同PMI首次突破 | 同一背景只买首次能避免释放现金的新亏损 | 全143同来源同路线只计过去，两新增亏损全保留 | 2017-04-05/2026-05-11均同PMI首次突破且实际亏损 | 拒绝单独首次标识充分性，不等于跑了新过滤策略 | 新机制另登记完整账户，不调首次或天数救规则 |
| 日周柱与多数行业 | 常见技术背景同时正足以证明进入 | 原四成功/反弹与两假启动量价/行业/宏观逐点对照 | 2026日周正/行业75%/领先5日+5.93%仍亏；2024周负即上涨 | 拒绝单独同正或多数阈值充分性，保留描述输入 | 不从这两个已知盈亏反推更优排名阈值 |
| 删除重复信号提高收益 | 原重复突破都是可删的多余交易 | 全信号对原阶段/R224真实现金周期，未知/未成交保持 | 同PMI重复突破21事件均0实际完成周期 | 拒绝事件数=交易数/收益改善；接受持仓与现金路径事实 | 新进入机制要整账检验，不从无交易事件报节约 |
| PMI新公布高胜率 | 11公布区间事件可当11笔高胜率样本 | 全11与原两账户已有周期保留，0新标签/策略 | 7无实际周期；阶段4周期3正1负、R224同4全正 | 不准入单独新消息门，也不报新100%胜率；样本少且现金选择已知 | 只有事前完整用途与新独立样本，不能挑4笔 |
| 新机制必须全来源首次 | 同利率路线重复都是失败 | 全事件及2019/2020成功段说明 | 2019盈利修复同利率此前2次、2020-06盈利修复此前3次 | 拒绝全来源必须首次的普适条件，保留不同阶段/机制 | 官方宣布、传播与持有完整机制需另立，不按原赢亏反选 |
| 官方宣布/工具类型新来源 | 不同政策工具能帮助辨别修复和重新定价 | 下一用途先全日历/事件覆盖/宣布生效钟/工具类型，成功反例同框 | 仅PROPOSED，未登记、采集或新金融 | 接受最值得继续的问题，不宣称已经验证方向或因果 | 是，先完成不同来源准入再固定完整账户；缺失保留 |

来源：[完整解释](../{report})、[一次描述](../{result})、[归因](../{study.relative(OUT / 'post_run_diagnosis.json')})、[用途](510300_ENTRY_INFORMATION_SEQUENCE_DESCRIPTION_V1.md)。原13独立前瞻字段和最新金融R224拒绝逐值保持。
"""
    prepared = []
    for name, text in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
            ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)):
        path = ROOT / f"docs/{name}.md"
        old, new, body = prepared_prepend(path, text)
        prepared.append((path, old, new, body))
    with (OUT / "state_before_TECH_R226.json").open("xb") as stream:
        stream.write(raw)
    doc_receipts = []
    for path, old, new, body in prepared:
        path.write_bytes(new)
        if path.read_bytes() != new or not path.read_bytes().endswith(body):
            raise ValueError("原长期事实正文未完整保持。")
        doc_receipts.append({"path": study.relative(path), "old_sha256": digest(old), "new_sha256": digest(new), "old_body_preserved_exact": True})
    state.update({"updated_at": parent.original.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": parent.original.now(),
        "latest_goal_tool_status_receipt": proof, "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "latest_technical_decision": "TECH.R226", "latest_registration_decision": "TECH.R225",
        "latest_report": report, "latest_result": result,
        "latest_progress": "全部进入信息更新与量价行业顺序完成；两首次假启动和旧生效利率新闻覆盖不足，下一官方宣布与工具信息用途。",
        "current_phase": "ENTRY_INFORMATION_SEQUENCE_DESCRIPTION_COMPLETE_NEXT_OFFICIAL_POLICY_ANNOUNCEMENT_INFORMATION",
        "goal_turn_classification": "PROGRESS_R225_R226_ALL_EVENTS_ENTRY_INFORMATION_SEQUENCE_AND_KNOWN_NEGATIVE_CASES_COMPLETED",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "new_accounts_in_current_phase": 0, "necessary_tests_passed_in_current_phase": 6,
        "current_financial_candidate_admission": "R224_REJECTED_UNCHANGED_NO_NEW_FINANCIAL_ADMISSION_FROM_DESCRIPTION",
        "current_phase_trial_accounting": {"scope": "TECH_R225_R226_ALL_ENTRY_INFORMATION_SEQUENCE_DESCRIPTION",
            "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0, "parameter_grid": False,
            "necessary_tests": 6, "prefix_checks": 19, "all_daily_slots": 3488, "all_evaluation_slots": 2855,
            "all_original_events": 143, "all_original_case_rows": 240, "negative_signal_rows": 2,
            "negative_window_rows": 38, "figures_viewed": 6, "financial_reruns": 0},
        "current_goal_turn_actual_work": {"new_financial_purposes_completed": 0, "new_accounts": 0,
            "new_description_purposes_completed": 1, "all_original_entry_events_explained": 143,
            "all_known_new_negative_entry_cases_explained": 2, "new_fits": 0, "new_labels": 0,
            "new_market_bars": 0, "new_admitted_unrun_financial_purposes": 0},
        "next_mainline_entry_information_description": {"status": "COMPLETED_ALL_EVENTS_NOT_FINANCIAL",
            "registration": "TECH.R225", "decision": "TECH.R226", "result": result,
            "single_freshness_entry_gate_admitted": False, "current_rate_clock_full_policy_coverage": False,
            "independent_validation": "NOT_ESTABLISHED"},
        "next_information_source_proposal": "官方政策宣布/生效时钟与工具类型，先原完整日历/全部进入事件覆盖和未知，再解释修复/重新定价及两假启动；未登记或采集。",
        "next_official_policy_announcement_information_proposal": {"status": "PROPOSED_NOT_REGISTERED_OR_RUN",
            "question": "现有生效利率背景未覆盖的真正政策宣布及不同工具，能否解释启动机制而非事后挑选事件。",
            "scope": "原完整研究日历及143阶段事件来源覆盖；2019/2024成功、2015反弹与2017/2026假启动共同解释，不只收集赢家。",
            "fields": ["官方原文", "保存公布钟", "宣布时刻", "预计及实际生效日", "直接可核对工具类型", "检索与时钟未知"],
            "financial_admission": "NOT_ADMITTED_NOT_RUN", "independent_validation": "NOT_ESTABLISHED"},
    })
    with STATE.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    saved = parent.read(STATE)
    if {key: saved[key] for key in FORWARD} != forward or {key: saved[key] for key in old_financial} != old_financial:
        raise ValueError("独立前瞻或最新实际金融事实被改变。")
    parent.write(OUT / "project_state_update_receipt.json", {"at": parent.original.now(), "status": "R226_FULL_ENTRY_INFORMATION_DESCRIPTION_ARCHIVED",
        "docs": doc_receipts, "old_state_sha256": digest(raw), "new_state_sha256": parent.digest(STATE),
        "forward_fields_preserved_exact": list(FORWARD), "latest_financial_fields_preserved_exact": old_financial,
        "latest_technical_decision": "TECH.R226", "goal_status": "active", "goal_achieved": False,
        "goal_turn_classification": saved["goal_turn_classification"], "consecutive_blocked_goal_turns": 0,
        "all_entry_events": 143, "new_accounts": 0, "next_source_proposal": "PROPOSED_NOT_REGISTERED_OR_RUN"})
    print("R226进入信息解释已归档四长期事实；原正文、金融拒绝与13前瞻保持，目标active。", flush=True)


if __name__ == "__main__":
    main()
