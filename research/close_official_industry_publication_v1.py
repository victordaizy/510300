"""一次归档新来源事实，保留最新金融拒绝、旧事实正文和独立前瞻字段。"""
from __future__ import annotations

import json

from research import official_industry_publication_intake_v1 as intake
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import prepared_prepend, digest

ROOT, OUT = intake.ROOT, intake.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def main():
    if (OUT / "project_state_update_receipt.json").exists() or (OUT / "state_before_TECH_R218.json").exists():
        raise RuntimeError("本来源阶段已经归档或开始，不重复更新事实。")
    summary = intake.parent.read(OUT / "implementation_v1_0_1/summary.json")
    received = intake.parent.read(OUT / "summary.json")
    delivery = intake.parent.read(OUT / "delivery_receipt.json")
    tests = intake.parent.read(OUT / "implementation_v1_0_1/tests_receipt.json")
    catalog = intake.parent.read(OUT / "complete_catalog_publication_index.json")
    service = intake.parent.read(OUT / "goal_service_status_after_result.json")
    viewed = intake.parent.read(OUT / "source_figure_view_receipt.json")
    if (summary["decision"] != "TECH.R218" or summary["pages_read"] != 702 or summary["classification_rows"] != 32223
            or summary["case_member_rows"] != 2400 or summary["parse_passed_snapshots"] != 5
            or summary["parse_failed_snapshots"] != 3 or summary["unknown_classification_rows"] != 5
            or received["logical_gets"] != 20 or received["http_successes"] != 20
            or tests["passed"] != 2 or tests["exit_code"] != 0 or delivery["nine_saved_case_rows_read"] != 9
            or catalog["publication_nodes"] != 36 or catalog["remaining_unreceived"] != 28
            or viewed["viewed"] != 11 or service["goal"]["status"] != "active" or summary["new_accounts"] != 0):
        raise ValueError("保存的真实来源结果、失败、汇总或目标状态与归档事实不一致。")
    files = []
    for name in ("protocol.json", "parser_protocol.json", "implementation_v1_0_1/protocol.json"):
        protocol = intake.parent.read(OUT / name)
        for item in protocol["files"]:
            actual = intake.parent.digest(ROOT / item["path"])
            if actual != item["sha256"]:
                raise ValueError("本来源冻结/复用文件改变：" + item["path"])
            files.append(item)
    old_state = STATE.read_bytes()
    state = json.loads(old_state.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R216" or state["latest_actual_financial_decision"] != "TECH.R216":
        raise ValueError("项目事实已由其他研究更新，先核实。")
    forward = {key: state[key] for key in FORWARD}
    previous_keys = ("latest_technical_decision", "latest_registration_decision", "latest_actual_financial_decision",
        "latest_actual_financial_result", "latest_result", "latest_report", "current_phase", "goal_turn_classification",
        "current_phase_trial_accounting", "current_financial_candidate_admission", "next_information_source_proposal")
    previous = {key: state.get(key) for key in previous_keys}
    report = relative(OUT / "官方分类_案例覆盖与下一实验.md")
    result = relative(OUT / "implementation_v1_0_1/summary.json")
    proof = relative(OUT / "goal_service_status_after_result.json")
    catalog_path = relative(OUT / "complete_catalog_publication_index.json")
    note = f"""> 最新新来源事实（2026-10-05，TECH.R217—R218；最新实际金融仍R216拒绝）：官方分类固定8快照/3目录/1方法页，本机20逻辑GET全部成功；8PDF全部702页、32,223快照证券行和公布钟已取得，原完整快照解析5通过/3失败、5个缺码行保留。九固定案例完成、2400实际成员行/221行业数量行：2024-09-24/09-30/10-08均300已知，2015-06-29及2025-07-03各299已知/1未知，2019/2020三个案例最新快照失败NO_VIEW，2015-01-05缺前一源日成员、分母未知。3份原件失败不能改写成通过，分类数量不等于指数权重或收益贡献。0新拟合/未来标签/账户/行情日线，收益夏普NOT_COMPUTED；完整目标未达。目标服务实际active，本轮新来源PROGRESS、连续受阻0。

源限制：CSRC季度与CAPCO2023半年度版本不混同申万代理；实际公布日23:59保守钟，2024H1虽09-30公布，09-30收盘仍用2023H2，10-08才可用。当前官方PDF取得不认证历史首版（现代链接含2026迁移目录），全部原历史仅开发/来源解释。保存目录36节点日期已建立索引，8已取得/28未取得；目录未列2021Q4/2022仍UNKNOWN，不据此宣称不存在。八案例快照不是完整2015—2026金融来源，行业策略未准入、独立验证未建立。

实现：原冻结解析完成八PDF后，首个0成员案例错误按缺industry_key空表归组，KeyError终止；失败及原8解析保留。隔离v1_0_1仅修空成员案例汇总，从已保存解析完成一次9案例；0重请求/重解析，原5缺码不补、质量门/钟/版本/9日期不改，原6必要测试和另2必要回归通过，54个复用文件保持。8首页/3相关原件页已查看。2025未知001391.SZ、2015未知600958.SH不倒补，未知原因不推测。

下一步：另立全发布序列来源用途，固定当前36目录节点、复用8原件，对28未取得页面与明确附件有限获取；先报告全日历逐行分类/覆盖/源龄及缺失。来源合格后再登记主线相对强弱/扩散/轮动能否区分ETF走弱后恢复与继续下跌的完整用途，保留原A、价格及同覆盖对照、原成本风险。此处是明确下一来源设计，未登记待跑新金融；原R212/R216拒绝不营救，原13独立前瞻及旧事实正文保持。

依据：[全部来源与案例报告](../{report})；[八解析与九案例结果](../{result})；[36发布节点索引](../{catalog_path})；[来源用途卡](510300_OFFICIAL_INDUSTRY_PUBLICATION_INTAKE_V1.md)；[目标服务实际active](../{proof})。
"""
    decisions = f"""### TECH.R217—R218：官方分类公布序列与主线案例来源（2026-10-05）

本用途为来源与案例覆盖，不是新交易策略检验。20本机GET全部成功，8PDF/702页/32,223快照证券行，九案例/2400成员/221行业数量；原6必要测试及另2空成员回归。原5通过/3失败和5缺码保持，0新账户/拟合/未来标签/行情日线。最新金融仍R216拒绝；服务实际active、PROGRESS、受阻0，完整目标未达。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否需要重新验证 |
|---|---|---|---|---|---|
| 官方历史行业分类来源 | 带实际公布日原件能提供不同于ETF价格的行业归属观察 | 固定8节点/3目录/1方法页，有限一次GET，保存原响应/日期/PDF/全部行 | 20GET成功，8PDF702页，5完整快照通过/3失败 | 接受公开来源取得与有限案例用途，不准入完整金融；当前取回版本首版未知 | 是，全序列与逐行版本/成员覆盖先固定，不能将当前分类提前 |
| 原快照缺码 | 单行门类缺码可能只是表格提取错误 | 查看8首页和3缺码相关原页，原表单元格/出处保留 | 5原证券行门类没有可明确括号代码，旧表全部证券集合仍匹配 | 拒绝推测补码；原整份快照5通过/3失败不改 | 有明确源证据或另立逐行用途才复核，不能依据后来分类/收益补 |
| 2024启动主线来源可观察 | 入场时可得到覆盖沪深300的行业集合 | 严格前一成员源日、实际公布日23:59选择所选最新快照 | 09-24、09-30用2023H2，10-08用2024H1，三个案例均300已知 | 接受有限案例归属和时钟，不接受收益预测/产业因果 | 是，主线强弱/传播须全日历反例、同价格对照及新完整金融 |
| 季末即分类可知 | 覆盖期末或公布日收盘可以直接使用新行业表 | 核实际页面日期与15:00/23:59先后 | 2024H1实际09-30公布；当日收盘不可用 | 拒绝按覆盖期末提前，接受保守钟；未知日内时刻保留 | 新时刻证据需新版本，不能按结果缩短钟 |
| 成员覆盖自动完整 | 有一份解析通过PDF即能归全300 | 原成员左连接，缺成员与缺分类分别保留 | 两案例299/300，首个缺成员源，三个案例原快照失败全未知 | 拒绝有PDF即全覆盖或缺失填零；未知证券600958.SH/001391.SZ保留 | 是，逐次来源及新公司可知分类；不从未来/旧表倒补 |
| 数量结构等于指数贡献 | 行业成员比例可直接代表ETF权重或资金传导 | 明确数量字段来源，与真实权重/资金证据分开 | 221数量行，2024电子42成员/14%；权重及贡献NOT_COMPUTED | 拒绝数量冒充权重/贡献/资金；接受观察对象分布 | 有合格权重及逐时价格源才另用途验证 |
| 原8快照够完整策略 | 覆盖四上涨与退出反例即可代表全部日历 | 三保存目录全部可见2014Q3以后的节点建立索引 | 36发布节点，8已取得、28未取得；2021Q4/2022目录缺口未知 | 拒绝把案例覆盖当完整2015—2026来源；接受下一有限全序列取得 | 是，新来源用途先固定、复用旧8，不重跑已冻源请求 |
| 空成员实现修复 | 缺源案例应记录未知，而不是汇总异常 | 两必要回归；复用保存8解析一次完成9案例，54文件哈希保持 | 原KeyError失败保留，v1_0_1仅空表归组保护，0重GET/解析/账户 | 接受实现修复，不改质量门、钟或5缺码 | 原实现失败无需收益重跑，后续新用途使用明确空源分支 |
| 行业主线提高收益夏普 | 相对强弱/扩散/轮动可分辨同样ETF弱势的继续价值 | 尚未登记完整金融；拟原A/价格/同覆盖对照及全账户成本风险 | 金融NOT_RUN，收益夏普NOT_COMPUTED，独立未建立 | 接受研究问题，不接受为有收益策略；小资金容量不替代方向优势 | 是，来源及用途合格后全日历同口径一次检验，不只看2024 |

原R212分阶段策略与R216失效策略拒绝保留；不用窗口、费用或权重事后选优救援。当前PDF版不认证首版，全项目搜索选择校正NOT_COMPUTED，原13独立前瞻保持。

依据：[全部来源与案例报告](../{report})、[实际汇总](../{result})、[36发布序列](../{catalog_path})、[来源用途卡](510300_OFFICIAL_INDUSTRY_PUBLICATION_INTAKE_V1.md)。
"""
    prepared = []
    for name, text in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
            ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)):
        path = ROOT / f"docs/{name}.md"
        old, new, body = prepared_prepend(path, text)
        prepared.append((path, old, new, body))
    with (OUT / "state_before_TECH_R218.json").open("xb") as stream:
        stream.write(old_state)
    docs = []
    for path, old, new, body in prepared:
        with (OUT / f"{path.stem}_before_TECH_R218.md").open("xb") as stream:
            stream.write(old)
        path.write_bytes(new)
        if path.read_bytes() != new or not path.read_bytes().endswith(body):
            raise ValueError("旧事实正文未逐字节保留。")
        docs.append({"path": relative(path), "old_sha256": digest(old), "new_sha256": digest(new), "old_body_preserved_exact": True})
    state["previous_phase_before_TECH_R217_R218"] = previous
    state.update({"updated_at": intake.parent.original.now(), "status": "research_active", "goal_status": "active",
        "goal_achieved": False, "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": intake.parent.original.now(),
        "latest_goal_tool_status_receipt": proof, "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "latest_technical_decision": "TECH.R218", "latest_registration_decision": "TECH.R217",
        "latest_result": result, "latest_report": report,
        "latest_progress": "官方20GET成功，8PDF702页、32223分类行、9案例2400成员；原5通过/3失败保持，36发布节点已列出，0新金融。",
        "current_phase": "OFFICIAL_INDUSTRY_SELECTED_SOURCE_CASE_COVERAGE_COMPLETED_FULL_SEQUENCE_PROPOSED",
        "goal_turn_classification": "PROGRESS_R217_R218_EIGHT_OFFICIAL_PDFS_PUBLICATION_CLOCKS_AND_NINE_CASE_SOURCE_COVERAGE",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "new_accounts_in_current_phase": 0, "necessary_tests_passed_in_current_phase": 8,
        "current_financial_candidate_admission": "SOURCE_CASE_COVERAGE_ONLY_NO_ADMITTED_FINANCIAL_CANDIDATE",
        "current_phase_trial_accounting": {"scope": "TECH_R217_R218_DIFFERENT_OFFICIAL_CLASSIFICATION_SOURCE_NOT_FINANCIAL",
            "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0, "native_gets": 20,
            "http_successes": 20, "official_pdfs_received": 8, "pages_read": 702, "classification_rows": 32223,
            "parse_passed_snapshots": 5, "parse_failed_snapshots": 3, "source_unknown_classification_rows": 5,
            "case_rows": 9, "case_member_rows": 2400, "industry_count_rows": 221,
            "original_necessary_tests": 6, "implementation_regressions": 2,
            "original_partial_parser_failures": 1, "completed_case_summary_runs": 1, "original_figures_viewed": 11,
            "complete_saved_catalog_nodes": 36, "remaining_unreceived_catalog_nodes": 28},
        "current_goal_turn_actual_work": {"new_official_publication_sources": 8, "new_official_pdf_pages": 702,
            "native_source_gets": 20, "source_classification_rows": 32223, "source_failures_retained": 3,
            "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0,
            "new_admitted_financial_purposes": 0},
        "latest_official_industry_source_case_coverage": result, "latest_official_industry_publication_catalog": catalog_path,
        "next_information_source_proposal": "新用途固定36已列发布节点，复用8原件，对28尚未取得页面/附件有限获取；保留缺码、成员未知、版本/实际公布钟/源龄，再评估逐行分类。行业收益金融仍未准入。",
        "next_official_mainline_complete_source_proposal": {"status": "PROPOSED_SOURCE_PURPOSE_NOT_REGISTERED_OR_RUN",
            "visible_nodes": 36, "reuse_received_nodes": 8, "unreceived_nodes": 28,
            "incomplete_catalog_periods": ["2021Q4", "2022"], "financial_run": "NOT_RUN",
            "purpose": "全发布序列/逐行分类可知与覆盖，为行业传播和弱势后恢复研究准备不同观察信息。"},
        "current_source_admission": "EIGHT_SELECTED_CURRENT_OFFICIAL_PDFS_CASE_COVERAGE_ONLY_NOT_FULL_HISTORY_NOT_FIRST_VINTAGE",
    })
    with STATE.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    saved = intake.parent.read(STATE)
    if {key: saved[key] for key in FORWARD} != forward:
        raise ValueError("原13项独立前瞻改变。")
    if saved["latest_actual_financial_decision"] != previous["latest_actual_financial_decision"] or saved["latest_actual_financial_result"] != previous["latest_actual_financial_result"]:
        raise ValueError("最新金融拒绝被来源用途改变。")
    intake.parent.write(OUT / "project_state_update_receipt.json", {"at": intake.parent.original.now(),
        "status": "SOURCE_PROGRESS_ARCHIVED_NO_FINANCIAL_PROMOTION", "docs": docs,
        "old_state_sha256": digest(old_state), "new_state_sha256": intake.parent.digest(STATE),
        "forward_fields_preserved_exact": list(FORWARD), "original_source_and_implementation_files_verified": len(files),
        "latest_technical_decision": saved["latest_technical_decision"], "latest_actual_financial_decision": saved["latest_actual_financial_decision"],
        "goal_status": saved["goal_status"], "goal_turn_classification": saved["goal_turn_classification"],
        "consecutive_blocked_goal_turns": 0, "new_accounts": 0, "new_financial_metrics": "NOT_COMPUTED"})
    print("新来源R218和四事实文档一次归档；旧正文及13前瞻精确保持，金融仍R216拒绝，目标active。", flush=True)


if __name__ == "__main__":
    main()
