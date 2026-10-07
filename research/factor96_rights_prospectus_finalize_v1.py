"""保存74份当前发行章节的字段对照及数量口径；未核实字段保持未知。"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import csv
import json
import re

from research.factor96_rights_prospectus_v1 import OUT, PREVIOUS, SOURCE, read, save, digest, normalized, now

AH = {
    "002202.SZ": (675678627, 552167067, 123511560),
    "600999.SH": (2009822798, 1715702444, 294120354),
    "600030.SH": (1939016404, 1597267249, 341749155),
    "600958.SH": (1958223624, 1670641224, 287582400),
    "601916.SH": (6380609033, 5014409033, 1366200000),
}
BASIS = {
    "601012.SH": ("2018年9月30日", 2791680001, 837504000),
    "600919.SH": ("2020年6月30日", 11544508967, 3463352690),
    "600909.SH": ("2021年3月31日", 3621046771, 1086314031),
    "601108.SH": ("2021年12月20日", 3589014449, 1076704335),
}
BUYBACK = {
    "601555.SH_A_RIGHTS_2019_2984": (3000000000, 2901700, 899129490, 900000000),
    "601555.SH_A_RIGHTS_2021_3337": (3880518908, 41701514, 1151645218, 1164155672),
}


class Section:
    def __init__(self, source, section):
        self.source = source
        self.section = section
        self.pages = {p["page"]: normalized(p["text"]) for p in read(source["text_path"])["pages"]}

    def literal(self, literal):
        hits = []
        for part in self.section["section_pages"]:
            text = self.pages[part["page"]]
            for m in re.compile(re.escape(literal)).finditer(text, part["start"], part["end"]):
                hits.append({"document_id": self.source["document_id"], "page": part["page"],
                             "start": m.start(), "end": m.end(), "literal": m.group(),
                             "context": text[max(0,m.start()-100):m.end()+180]})
        assert hits, (self.source["document_id"], literal)
        return hits

    def number(self, value):
        return self.literal(format(value, ","))


def finalize():
    targets = {r["document_id"]: r for r in read(OUT / "targets.json")}
    sections = {r["document_id"]: r for r in read(OUT / "current_issue_sections_v1_0_1.json")}
    rows = deepcopy(read(OUT / "current_section_comparison_v1_0_1.json"))
    for row in rows:
        rid, symbol = row["document_id"], row["symbol"]
        doc = Section(targets[rid], sections[rid])
        source = targets[rid]["source"]
        row.update(source_url=source["source_url"], raw_path=str(SOURCE/source["raw_snapshot"]),
                   raw_sha256=source["raw_sha256"], text_path=str(SOURCE/source["text_snapshot"]),
                   text_sha256=source["text_sha256"], field_review_notes=[], source_scope_evidence={})
        if rid == "1201906629":
            hits = doc.literal("2016年1月15日至") + doc.literal("2016年1月21日(T+1日)至(T+5日)")
            row["field_candidates"]["payment_window"] = [{**h, "value": ["2016-01-15", "2016-01-21"],
                 "derivation": "同一日历表格行跨第22和23页，已目检两页"} for h in hits]
            row["unique_values"]["payment_window"] = [["2016-01-15", "2016-01-21"]]
            row["comparisons"]["payment_window"] = "MATCHES_SAME_DAY_NOTICE"
            row["field_review_notes"].append("跨页表格不是两个区间；完整缴款期为1月15—21日，已按两页图像核对。")
        if rid == "1205988163":
            hits = doc.literal("本次配股价") + doc.literal("格为4.65元/股")
            row["field_candidates"]["issue_price_cents"] = [{**h, "value": 465,
                "derivation": "同一句配股价格说明跨第21和22页，已目检两页"} for h in hits]
            row["unique_values"]["issue_price_cents"] = [465]
            row["comparisons"]["issue_price_cents"] = "MATCHES_SAME_DAY_NOTICE"
        if symbol == "000661.SZ":
            row["calendar_status"] = "RELATIVE_T_DAY_SCHEDULE_ONLY_IN_CURRENT_SECTION"
            row["source_scope_evidence"]["relative_calendar"] = doc.literal("股权登记日T日正常交易")
            row["field_review_notes"].append("当前发行日程只列T日等相对安排，不把同日发行公告的完整日期填进本文件摘录。")
        elif row["unique_values"]["month_day_record"]:
            row["calendar_status"] = "MONTH_DAY_ONLY_MATCHES_NOTICE_WITHOUT_ASSIGNING_YEAR"
            row["field_review_notes"].append("原表只写月日，保留无年份字段；只与同日主公告的月日比较，不增加绝对日期来源。")
        else:
            row["calendar_status"] = "ABSOLUTE_DATES_MATCH_SAME_DAY_NOTICE"
        if not row["unique_values"]["issue_price_cents"]:
            row["field_review_notes"].append("当前发行章节只提取到定价原则，未得到明确数字价格；其他文件价格不填入本文件。")
        quantities = row["unique_values"]["quantity_scenarios"]
        if symbol in AH:
            total, a_shares, h_shares = AH[symbol]
            assert total == a_shares + h_shares and a_shares == row["same_day_notice_plan_max_shares"]
            row["quantity_review"] = {"status": "A_H_TOTAL_SEPARATED", "total_A_H_shares": total,
                                      "A_share_scenario": a_shares, "H_share_scenario": h_shares,
                                      "actual_subscription_known_from_this_clause": False}
            for field, value in [("A_H_total",total), ("A_shares",a_shares), ("H_shares",h_shares)]:
                row["source_scope_evidence"][field] = doc.number(value)
        elif symbol in BASIS:
            basis_day, capital, quantity = BASIS[symbol]
            assert quantities == [quantity] and abs(capital*3 - quantity*10) < 10, (rid, quantities, quantity)
            row["quantity_review"] = {"status": "CONDITIONAL_SHARE_CAPITAL_BASIS_ESTIMATE", "basis_date_literal": basis_day,
                                      "basis_total_shares": capital, "prospectus_estimate": quantity,
                                      "proportional_product_tenths_of_share": capital*3,
                                      "rounding_note": "保留公告原数；比例乘积可能有不足1股的差额，不施加统一向下取整。",
                                      "same_day_notice_plan_max_shares": row["same_day_notice_plan_max_shares"],
                                      "notice_minus_prospectus_shares": row["same_day_notice_plan_max_shares"] - quantity,
                                      "not_an_actual_subscription_correction": True}
            row["source_scope_evidence"].update(basis_date=doc.literal(basis_day), basis_capital=doc.number(capital), estimate=doc.number(quantity))
        elif row["comparison_event_id"] in BUYBACK:
            capital, treasury, lower, upper = BUYBACK[row["comparison_event_id"]]
            assert set(quantities) == {lower, upper} and (capital-treasury)*3//10 == lower and capital*3//10 == upper
            assert lower == row["same_day_notice_plan_max_shares"]
            row["quantity_review"] = {"status": "TREASURY_DISPOSAL_ALTERNATIVE_SCENARIOS", "total_capital": capital,
                                      "treasury_shares": treasury, "excluding_treasury_shares_scenario": lower,
                                      "all_treasury_disposed_scenario": upper, "not_two_issuances": True}
            row["source_scope_evidence"].update(capital=doc.number(capital), treasury=doc.number(treasury),
                                              lower_scenario=doc.number(lower), upper_scenario=doc.number(upper))
        elif symbol == "601016.SH":
            assert not quantities
            row["quantity_review"] = {"status": "RATIO_STATED_NO_SCALAR_QUANTITY_EXTRACTED_IN_CURRENT_SECTION", "ratio_per_10": "3"}
            row["source_scope_evidence"]["ratio"] = doc.literal("按每10股配售3股的比例")
        else:
            assert quantities == [row["same_day_notice_plan_max_shares"]], (rid, quantities)
            row["quantity_review"] = {"status": "CURRENT_SECTION_SCENARIO_EQUALS_NOTICE_PLANNED_QUANTITY",
                                      "scenario_quantity": quantities[0], "not_actual_subscription": True}
        row["full_body_reviewed"] = False
        row["current_section_field_comparison_finished"] = True
        row["quantity_admitted_as_final_issuance"] = False
        row["listing_restriction_full_review_complete"] = False
    save(OUT / "reviewed_current_section_comparisons.json", rows)
    pending = [r for r in targets.values() if r["body_role_candidate"] != "IMPLEMENTATION_PROSPECTUS"]
    save(OUT / "remaining_121_plan_admin_documents.json", {"at": now(), "documents": pending,
         "status": "BODY_REVIEW_PENDING_TITLE_CLASSIFICATION_ONLY", "count": len(pending)})
    # 限售及上市条款另存候选，避免当前三类字段核对被误读为完整上市准入。
    restrictions = []
    for row in rows:
        for part in sections[row["document_id"]]["section_pages"]:
            text = part["text"]
            positions = []
            for m in re.finditer("无限售|有限售|限售|锁定|持有期限|禁售", text):
                a,b = max(0,m.start()-80),min(len(text),m.end()+220)
                if positions and a <= positions[-1][1]:
                    positions[-1][1] = max(b,positions[-1][1])
                else:
                    positions.append([a,b])
            for a,b in positions:
                restrictions.append({"document_id":row["document_id"], "page":part["page"],
                    "start":part["start"]+a, "end":part["start"]+b, "literal":text[a:b], "reviewed":False})
    save(OUT / "listing_restriction_candidates.json", restrictions)
    result = {"at": now(), "study_id": "510300_FACTOR96_RIGHTS_PROSPECTUS_V1",
        "status": "74_CURRENT_ISSUE_SECTION_FIELDS_COMPARED_WITH_QUANTITY_SCOPE_SEPARATED",
        "fixed_nonreminder_documents": 195, "candidate_clauses": 2784,
        "implementation_prospectuses_and_abstracts_compared": 74, "current_chapter_page_spans": 584,
        "actual_all_584_pages_manually_read": False,
        "comparison_issuer_events": len({r["comparison_event_id"] for r in rows}),
        "same_day_primary_issuance_notices": len({r["same_day_issuance_notice_id"] for r in rows}),
        "new_earlier_publication_dates_established": 0,
        "calendar_statuses": dict(Counter(r["calendar_status"] for r in rows)),
        "price_numeric_comparisons": sum(r["comparisons"]["issue_price_cents"]=="MATCHES_SAME_DAY_NOTICE" for r in rows),
        "price_not_numerically_extracted_from_current_chapter": sum(not r["unique_values"]["issue_price_cents"] for r in rows),
        "quantity_statuses": dict(Counter(r["quantity_review"]["status"] for r in rows)),
        "current_core_approval_support_documents": sum(r["comparison_identity_basis"]=="CURRENT_SECTION_APPROVAL_AND_SAME_DAY_ISSUER_NOTICE" for r in rows),
        "same_issuer_same_day_comparison_only_documents": sum(r["comparison_identity_basis"]=="SAME_ISSUER_SAME_DAY_IMPLEMENTATION_NOTICE_COMPARISON_ONLY" for r in rows),
        "retained_and_excluded_cross_row_registration_candidates": 2,
        "cross_page_field_reconstructions": 2, "visual_pages_reviewed": 7,
        "remaining_plan_admin_documents": len(pending), "restriction_clause_candidates_not_reviewed": len(restrictions),
        "comparison_records_are_not_new_canonical_event_updates": True,
        "full_intermediate_version_coverage": False, "full_M06_calendar_established": False,
        "historical_first_publication_verified": False, "free_float_denominator_established": False,
        "strict_M06_pressure": None, "trading_feature_admitted": False,
        "new_network_requests": 0, "new_accounts": 0, "new_returns": 0, "new_models": 0,
        "orders_authorized": False, "external_review": "NOT_PERFORMED", "delivery_package_created": False}
    assert len(pending) == 121
    assert result["calendar_statuses"] == {"ABSOLUTE_DATES_MATCH_SAME_DAY_NOTICE":66, "RELATIVE_T_DAY_SCHEDULE_ONLY_IN_CURRENT_SECTION":2, "MONTH_DAY_ONLY_MATCHES_NOTICE_WITHOUT_ASSIGNING_YEAR":6}
    assert result["price_numeric_comparisons"] == 67
    save(OUT / "result.json", result)
    flat = [{"文档ID":r["document_id"], "证券":r["symbol"], "目录代理日期":r["known_at"][:10],
             "同日发行公告ID":r["same_day_issuance_notice_id"], "日历核对":r["calendar_status"],
             "价格核对":r["comparisons"]["issue_price_cents"], "数量口径":r["quantity_review"]["status"],
             "原文地址":r["source_url"], "交易特征准入":False} for r in rows]
    with (OUT / "74份说明书字段对照.csv").open("x",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    finalize()
