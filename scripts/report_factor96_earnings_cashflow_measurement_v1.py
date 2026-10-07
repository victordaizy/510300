"""保存财报字段反证、影响范围及本轮交付说明，不改变已冻结测量。"""
from pathlib import Path
import json
import re
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.factor96_earnings_cashflow_measurement_v1 import OUT, STUDY, digest, now, save


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def contradiction_cards():
    # 以下金额来自本轮已读的公告原页；只写反证卡，不覆盖或重算原测量值。
    specs = [
        ("1214959878", "TOTAL_ASSETS_END", "北方华创2022三季报", "金额截断", 40895706832.37,
         [(1, "总资产（元）40,895,706,832.37"), (7, "资产总计40,895,706,832.37")],
         "总资产完整金额被档案保存为40，原页和合并资产负债表一致。"),
        ("1216653495", "TOTAL_ASSETS_END", "中海油服2023一季报", "跨页单位遗漏", 76677224967.,
         [(1, "单位：百万元"), (2, "总资产76,677.2"), (7, "单位：元"), (7, "资产总计76,677,224,967")],
         "摘要总资产为76,677.2百万元，合并表精确值76,677,224,967元；档案漏掉百万单位。"),
        ("1219823220", "TOTAL_ASSETS_END", "中海油服2024一季报", "跨页单位遗漏", 82463076897.,
         [(1, "单位：百万元"), (2, "总资产82,463.1"), (8, "单位：元"), (8, "资产总计82,463,076,897")],
         "摘要与明细存在正常显示精度差异；按元保留82,463.1属于数量级错误。"),
        ("1219906975", "PARENT_NET_PROFIT_YTD", "复星医药2024一季报", "脚注编号当金额", 609733627.26,
         [(3, "单位：元"), (3, "归属于上市公司股东的净利润注2609,733,627.26")],
         "注2为脚注，真实归母利润为609,733,627.26元。"),
        ("1223412328", "PARENT_NET_PROFIT_YTD", "复星医药2025一季报", "脚注编号当金额", 764757750.06,
         [(2, "单位：元"), (2, "归属于上市公司股东的净利润注2764,757,750.06")],
         "注2为脚注，不能把2元作为当期利润。"),
        ("1223412328", "OPERATING_CASH_FLOW_YTD", "复星医药2025一季报", "脚注编号当金额", 1055679878.45,
         [(2, "单位：元"), (2, "经营活动产生的现金流量净额注41,055,679,878.45")],
         "注4为脚注，真实经营现金流为1,055,679,878.45元。"),
        ("1211423261", "OPERATING_CASH_FLOW_YTD", "北方华创2021三季报", "跨行取错指标", -602204358.37,
         [(1, "经营活动产生的现金流量净额（元）————-602,204,358.37"),
          (1, "基本每股收益（元/股）0.7068"), (11, "经营活动产生的现金流量净额-602,204,358.37")],
         "0.7068为单季基本每股收益；年初至期末经营现金流是负602,204,358.37元。"),
    ]
    extra = OUT / "anomaly_evidence"
    flagged = pd.read_parquet(extra / "flagged_fields_before_source_review.parquet")
    cards = []
    for aid, metric, title, kind, correct, anchors, note in specs:
        row = flagged[(flagged.announcement_id == aid) & (flagged.metric_id == metric)].iloc[0]
        doc, receipt = read(extra / "text" / (aid + ".json")), read(extra / "receipts" / (aid + ".json"))
        assert receipt["matches_archived_extraction_pdf"] and receipt["sha256"] == row.official_pdf_sha256
        assert correct != row.verified_value
        proof = []
        for page, fragment in anchors:
            text = re.sub(r"\s+", "", doc["pages"][page - 1]["text"])
            assert fragment in text, (aid, page, fragment)
            proof.append({"page": page, "normalized_fragment": fragment})
        cards.append({"announcement_id": aid, "ts_code": row.ts_code, "report_period": row.report_period,
            "metric_id": metric, "report_title": title, "error_kind": kind, "archived_value_cny": row.verified_value,
            "original_page_value_cny": correct, "raw_original_snapshot_value": row.source_raw_value,
            "archived_unit": row.source_unit, "source_pdf_url": row.official_pdf_url,
            "pdf_sha256": receipt["sha256"], "proof": proof, "explanation": note,
            "status": "CONFIRMED_ORIGINAL_PDF_CONTRADICTS_ARCHIVED_FIELD", "correction_applied_to_frozen_data": False})
    return cards


def impact(cards):
    dep = pd.read_parquet(OUT / "formula_dependencies.parquet")
    keys = pd.DataFrame([{"source_announcement_id": r["announcement_id"], "metric_id": r["metric_id"], "error_kind": r["error_kind"]} for r in cards])
    affected = dep.merge(keys, on=["source_announcement_id", "metric_id"], how="inner", validate="many_to_one")
    assert affected.status.eq("VERIFIED_SAVED_ORIGINAL_FACT").all()
    measured = pd.read_parquet(OUT / "member_report_measurements.parquet")
    flagged = measured.loc[measured.announcement_id.isin(set(affected.target_announcement_id))].copy()
    panel = pd.read_parquet(OUT / "daily_member_measurements.parquet")
    dates = panel.loc[panel.announcement_id.isin(set(affected.target_announcement_id)) & panel.joint_known,
                      ["date", "ts_code", "announcement_id", "report_period", "L02", "L04_change", "joint_known"]].copy()
    dates["status"] = "CONFIRMED_FIELD_DEPENDENCY_CONTAMINATED_NO_TRADING_ADMISSION"
    summary = {"confirmed_wrong_fields": len(cards), "confirmed_documents": len(set(r["announcement_id"] for r in cards)),
        "error_kinds": sorted(set(r["error_kind"] for r in cards)), "affected_formula_dependency_occurrences": len(affected),
        "affected_historical_union_reports": len(flagged), "affected_member_report_events": int(flagged.member_at_available.sum()),
        "affected_mechanically_joint_reports_in_union": int(flagged.joint_known.sum()),
        "affected_mechanically_joint_member_report_events": int((flagged.member_at_available & flagged.joint_known).sum()),
        "affected_daily_joint_member_rows": len(dates), "affected_calendar_sessions": int(dates.date.nunique()),
        "affected_issuers": sorted(flagged.ts_code.unique().tolist()),
        "exhaustiveness": "只计算已确认七个字段的直接依赖，是已知影响下界；未被这些字段命中的行不等于正确。",
        "current_dataset_admission": "BLOCKED_CONFIRMED_SEMANTIC_EXTRACTION_ERRORS", "T11": "NOT_RUN",
        "new_accounts": 0, "new_returns": 0, "goal_status": "active", "goal_achieved": False,
        "old_frozen_files_modified": False, "corrected_feature_values_computed": False,
        "next_step": "冻结新版本摘要表取数规则，修复跨页单位、脚注、金额截断和列行身份；以七条原页作回归反例，再按同类来源规则评估全档案影响，不按收益筛修补对象。"}
    return affected, flagged, dates, summary


def main():
    assert not (OUT / "round_status.json").exists()
    extra = OUT / "anomaly_evidence"
    result = read(OUT / "result.json")
    cards = contradiction_cards()
    affected, flagged, dates, summary = impact(cards)
    save(extra / "confirmed_field_contradictions.json", cards)
    affected.to_parquet(extra / "affected_formula_dependencies.parquet", index=False)
    flagged.to_parquet(extra / "affected_report_measurements.parquet", index=False)
    dates.to_parquet(extra / "affected_daily_member_measurements.parquet", index=False)
    save(OUT / "source_invalidation_addendum.json", summary)
    visual_names = ["1214959878_p1.png", "1211423261_p1.png", "1223412328_p2.png", "1216653495_p1.png", "1216653495_p2.png",
                    "1219823220_p1.png", "1219823220_p2.png", "1219906975_p3.png", "1216653495_p7.png", "1219823220_p8.png"]
    save(extra / "visual_review_receipt.json", {"at": now(), "reviewer": "本任务助手，非外部独立审阅",
        "rendered_pages": 12, "visually_inspected_pages": len(visual_names),
        "pages": [{"path": "visual/" + n, "sha256": digest(extra / "visual" / n)} for n in visual_names],
        "finding": "七个字段均由可读全页确认；跨页单位与表格边界、脚注上标及现金流/每股收益行列已分辨。",
        "uninspected_extra_renderings": ["1211423261_p11.png", "1214959878_p7.png"],
        "global_source_semantic_pass": False})
    save(extra / "semantic_verification_receipt.json", {"status": "PASS_SEVEN_ORIGINAL_PAGE_CONTRADICTIONS_AND_DEPENDENCY_IMPACT",
        "confirmed_wrong_fields": 7, "pdf_documents": 6, "all_pdf_hashes_match_old_extraction_archive": True,
        "proof_fragments": sum(len(r["proof"]) for r in cards), "impact": summary, "external_review": "NOT_PERFORMED"})
    shutil.copy2(Path(__file__), extra / Path(__file__).name)
    save(extra / "review_freeze.json", {"at": now(), "measurement_result_sha256": digest(OUT / "result.json"),
        "invalidation_addendum_sha256": digest(OUT / "source_invalidation_addendum.json"),
        "files": [{"path": p.relative_to(extra).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                  for p in sorted(extra.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]})
    status = {"at": now(), "study_id": STUDY, "this_turn_classification": "PROGRESS_MEASUREMENT_AND_SEVEN_SOURCE_CONTRADICTIONS",
        "progress": "完成单季/TTM公式及依赖测量，并由六份同哈希原公告证实七个字段错误；不将机械完整率当有效数据。",
        "new_accounts": 0, "new_returns": 0, "new_official_pdf_documents": 6, "collector_direct_http_requests": 6,
        "cumulative_admitted_account_scenarios": 432, "cumulative_invalid_implementation_accounts": 152,
        "cumulative_executed_account_scenarios": 584, "qualified_candidates": [], "new_admitted_trading_features": 0,
        "T11": "NOT_RUN_BLOCKED_CONFIRMED_SOURCE_ERRORS", "goal_status": "active", "goal_achieved": False,
        "current_market_view": "NO_VIEW", "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "round_status.json", status)
    rows = []
    for r in cards:
        rows.append(f"| {r['report_title']} | {r['metric_id']} | {r['archived_value_cny']:,.4f} | {r['original_page_value_cny']:,.2f} | {r['error_kind']} |")
    annual = []
    for r in result["yearly_daily_coverage"]:
        if r["year"] >= 2018:
            annual.append(f"| {r['year']} | {r['trading_days']} | {r['joint_median']:g} | {r['coverage_median']:.1%} |")
    report = f"""目标尚未实现。T11“现金质量支持的盈利更新”完成了财务测量，但尚未运行账户：35,673条目标字段可按保存内容复算，6,977个当时成员公告能机械计算L02与L04；随后发现并由六份原公告确认七个字段的语义提取错误。这些错误已影响{summary['affected_mechanically_joint_member_report_events']}个上述事件和{summary['affected_daily_joint_member_rows']}个逐日成员观测。当前数据不准入交易因子，不能把可计算数量当作有效证据。

本轮只操作研究文件，交易范围仍是510300.SH与人民币现金。新增账户0、收益计算0，累计正式432加否定实现152等于584个账户情景，独立前向观察0，合格候选仍为空。计划采集继续暂停；本轮六个指定官方PDF请求与该计划无关。

已完成的测量包括：单季利润由本年累计相减，分母为上一季资产；盈利意外相对前两年同季计算；经营现金流和归母利润按“本期累计加上年全年减上年同期”分别得到真正TTM；按名义披露日后的首个已覆盖交易日保守可用。441,735条依赖中27条因晚于目标时刻不能使用；缺项、三条身份异常及零历史波动均保持未知。2018年4月18日是首个机械可计算成员事件，涉及444家公司。

原公告反证如下，金额均按人民币元列示；两份中海油服摘要采用百万元，精确元值另由合并资产负债表核对。

| 公告 | 字段 | 档案错误值 | 原页数值 | 错误类型 |
|---|---|---:|---:|---|
{chr(10).join(rows)}

六份本次取得的PDF与旧提取档案的SHA-256全部一致，因此这些矛盾不能解释为本次下载了不同版本。完整PDF、文本、字段卡、页面定位及10页目视核对记录见anomaly_evidence/。反证金额只写入对照卡，未覆盖冻结数据，也未计算所谓修正后收益。

七个字段共命中{summary['affected_formula_dependency_occurrences']}条公式依赖，涉及{summary['affected_historical_union_reports']}个历史成员并集报告、{summary['affected_member_report_events']}个披露时成员报告；其中{summary['affected_mechanically_joint_member_report_events']}个原本被机械标为两因子完整。它们继续传播到{summary['affected_calendar_sessions']}个交易日、{summary['affected_daily_joint_member_rows']}条当前成员观测。这只是已知影响下界，不能据此宣称其余数据正确，也不把这七个字段简单剔除后继续试收益。

下表仅记录发现错误前的机械覆盖，供定位来源缺口；不构成数据有效率或交易准入。

| 年份 | 交易日 | 每日两因子可算公司数中位 | 相对非金融成员的机械覆盖中位 |
|---|---:|---:|---:|
{chr(10).join(annual)}

原始事实档案的旧整体覆盖失败状态保留。旧盈利广度研究采用累计值同比和年化近似，本轮采用单季与真正TTM，两者没有拼接；旧研究的240家门槛也没有降低。本轮保留25个冻结文件、全部原测量与错误依赖；8项季度会计/时序边界测试通过，保存输入能只读重算，这些检查不能代替原页语义正确性。

接下来先建立一个新的来源修正版：以这七条真实反例修复摘要表脚注、跨页单位、金额截断及跨行选数，按相同解析路径评估全档案影响，不按策略收益选择修补对象。来源修正及可得时钟满足后，才另行冻结行业归一化、ETF聚合、首完整日反应和两年日更状态规则，运行T11固定账户。O02当前NOT_COMPUTED，T11当前NOT_RUN；不得靠裁剪极端值、调阈值或挑年份绕过已确认错误。

仍需保留的限制是：两年同季只有两个基准观测；合并经营现金流与归母利润并非完全相同权益口径；相邻报告的重述和合并范围未统一；行业可用时刻是供应商生效日期代理；名义公告日及本次相同PDF哈希不证明历史首次发布。2026覆盖截至8月14日，不能据此形成当前市场观点。

主要证据：protocol.json和freeze.json为测量前冻结；result.json为不可覆盖的机械测量；source_invalidation_addendum.json为发现错误后的准入否定；anomaly_evidence/protocol_addendum.json和request_freeze.json为六份原文请求前冻结；confirmed_field_contradictions.json、affected_formula_dependencies.parquet、affected_daily_member_measurements.parquet为具体反证和传播路径。完整复核包另有FILE_INDEX.csv、原附件、旧包原字节快照及可复制审阅提示。仅本地交付，外部审阅NOT_PERFORMED。
"""
    (OUT / "研究结论.md").write_text(report, encoding="utf-8")
    prompt = """请先读02_研究结论.md，再读source_invalidation_addendum.json，最后检查原始measurement的result.json。目标是只交易510300与现金，20万元完整账户成本后夏普至少1.2、年化至少10%、目标回撤不超过10%；不得把源数据研究当目标达成。

请批评本轮单季利润、真正TTM、两年前同季标准化、名义披露日与原始版本的处理；直接核对六份原PDF是否支持七个提取错误，以及441735条依赖中的已知传播范围。指出机械复算通过与财务语义正确的差别。不要把6977个可算事件或未命中七条错误的行当有效样本。

请给出下一步具体来源修正方案、真实反例回归验证、同类解析路径的扩展核查边界，以及修正完成后如何事前固定行业处理、ETF聚合、O02首日反应和两年日更训练。明确哪些条件仍应NO_VIEW/NOT_RUN、何时应放弃T11；不得通过缩尾、调阈值、选盈利年份或覆盖旧失败救援结果。本轮未运行新账户、未做任何收益判断。旧档案/代码/测量和历史包均需保留，外部审阅尚未执行；本包不授权任何订单。
"""
    (OUT / "01_GPT_REVIEW_PROMPT.txt").write_text(prompt, encoding="utf-8")
    program = ROOT / "reports/research/510300_factor96_program_v1"
    old_status = read(program / "status.json")
    old_status.update({"at": now(), "latest_round": STUDY, "latest_result": (OUT / "round_status.json").relative_to(ROOT).as_posix(),
        "admitted_account_scenarios_this_round": 0, "invalid_implementation_accounts_this_round": 0,
        "new_source_documents_this_round": 6, "new_searchable_text_documents_this_round": 6,
        "new_archived_source_http_responses_this_round": 6, "new_source_occurrences_this_round": 0,
        "source_field_candidates_this_round": 7, "source_field_candidate_kind": "七个已确认的财务语义提取错误，非新交易字段",
        "new_explicit_version_links_this_round": 0, "new_version_roles_this_round": 0,
        "reused_repurchase_pdf_documents_this_round": 0, "completed_calendar_examples_this_round": 0,
        "latest_earnings_measurement": (OUT / "result.json").relative_to(ROOT).as_posix(),
        "latest_earnings_source_invalidation": (OUT / "source_invalidation_addendum.json").relative_to(ROOT).as_posix(),
        "next_candidates": ["T11_SEMANTIC_FINANCIAL_PARSER_REMEDIATION", "T12_FREE_FLOAT_AND_REMAINING_EXECUTION_PURPOSE_CHAIN",
                            "T13_EVENT_IDENTITIES_AND_FREE_FLOAT", "T04_HISTORICAL_WEIGHTS"]})
    save(program / "status.json", old_status, exclusive=False)
    for filename, ids in [("factor_progress.json", {"L02", "L04"}), ("strategy_progress.json", {"T11"})]:
        items = read(program / filename)
        for item in items:
            if item["id"] in ids:
                item.update({"current_status": "NOT_RUN_BLOCKED_CONFIRMED_FINANCIAL_SOURCE_ERRORS",
                    "current_evidence": "机械单季/TTM测量完成；六份同哈希PDF确认七个单位、脚注或跨行提取错误，需来源新版本，未运行T11收益。",
                    "current_evidence_path": (OUT / "source_invalidation_addendum.json").relative_to(ROOT).as_posix(), "individual_performance": None})
        save(program / filename, items, exclusive=False)
        csv_name = "96因子当前进度.csv" if filename.startswith("factor") else "18策略当前进度.csv"
        pd.DataFrame(items).to_csv(program / csv_name, index=False, encoding="utf-8-sig")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update({"current_round": STUDY, "latest_progress_receipt": (OUT / "round_status.json").relative_to(ROOT).as_posix(),
        "current_protocol": (OUT / "protocol.json").relative_to(ROOT).as_posix(),
        "latest_continuation_report": (OUT / "研究结论.md").relative_to(ROOT).as_posix(),
        "latest_continuation_classification": status["this_turn_classification"],
        "research_execution_state": "T11_MEASUREMENT_SOURCE_SEMANTICS_REJECTED_NO_NEW_ACCOUNT",
        "last_source_result": "完成财报单季/TTM依赖测量；七个字段由六份原PDF确认错误，T11不得直接采用，等待来源修正版。"})
    save(mandate_path, mandate, exclusive=False)
    (OUT / "program_after").mkdir()
    for path in program.iterdir():
        if path.is_file():
            shutil.copy2(path, OUT / "program_after" / path.name)
    shutil.copy2(mandate_path, OUT / "program_after/mandate.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
