"""保存固定范围提示公告的字段对照与例外证据；不生成交易特征或账户。"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import csv
import json

from research.factor96_rights_intermediate_v1 import OUT, BASE, SOURCE, read, save
from research.factor96_rights_event_chains_v1 import digest, normalized, now
from research.factor96_rights_event_review_v1 import Document, base_fact, state_at
from research.factor96_rights_reminder_review_v1 import FIELDS

CONNECT_IDS = {"1204534489", "1204577532", "1204587123"}
TABLE_ROW_IDS = {"1204493232", "1204494961", "1204501796", "1204506208", "1204512788"}
PREPLAN_ID = "1203316299"
CONNECT_DATES = ["2018-03-29", "2018-04-03", "2018-04-04"]


def source_documents():
    rows = read(OUT / "remaining_local_documents.json") + read(OUT / "additional_documents.json")
    rows += read(OUT / "ambiguous_title_supplement/documents.json")
    return {row["document_id"]: row for row in rows}


def anchor(hit):
    return {"document_id": hit["document_id"], "page": hit["page"],
            "normalized_start": hit["start"], "normalized_end": hit["end"],
            "literal": hit["literal"], "context": hit["context"]}


def make_fact(source, event, role):
    doc = Document({**source, "review_role": role})
    fact = base_fact(doc, event.split("_A_RIGHTS_")[1] if event else None)
    fact.update(full_document_semantics_reviewed=False, calendar_fields_reconciled=False,
                review_method="LABELLED_CLAUSE_EXTRACTION_AND_PRIOR_KNOWN_NOTICE_COMPARISON_WITH_TARGETED_EXCEPTIONS")
    return doc, fact


def regular_reminder(row, source):
    doc, fact = make_fact(source, row["review_event_id"], "PAYMENT_REMINDER")
    assert fact["event_id"]
    evidence = deepcopy(row["parsed_evidence"])
    rejected = []
    if row["document_id"] in TABLE_ROW_IDS:
        for hit in evidence["record_date"]:
            if hit["value"] == "2018-03-20":
                assert hit["page"] == 3 and hit["literal"] == "股权登记日2018年3月20日"
                rejected.append({**hit, "reason": "PDF表格登记日行标签被错误连接至下一行缴款起日；原PDF对应登记日为3月19日。"})
        assert len(rejected) == 1
        evidence["record_date"] = [h for h in evidence["record_date"] if h["value"] != "2018-03-20"]
        fact["notes"].append("保留一条跨表行错误候选；正文明确登记日为2018-03-19，与日程表T日一致。")
    values = {}
    for field in FIELDS:
        unique = list(dict.fromkeys(tuple(h["value"]) if isinstance(h["value"], list) else h["value"]
                                    for h in evidence[field]))
        assert len(unique) == 1, (row["document_id"], field, unique)
        values[field] = unique[0]
        prior = row["prior_known_source_state"]
        expected = (prior.get("payment_start"), prior.get("payment_end")) if field == "payment_window" else prior.get(field)
        assert unique[0] == expected, (row["document_id"], field, unique[0], expected)
        if field == "payment_window":
            fact["updates"].update(payment_start=unique[0][0], payment_end=unique[0][1])
            fact["evidence"]["payment_start"] = [anchor(h) for h in evidence[field]]
            fact["evidence"]["payment_end"] = [anchor(h) for h in evidence[field]]
        else:
            fact["updates"][field] = unique[0]
            fact["evidence"][field] = [anchor(h) for h in evidence[field]]
    if row["document_id"] == "1211684963":
        fact["evidence"]["approval_key"] = doc.literal("并获中国证监会证监") + doc.literal("许可〔2021〕2718号文核准")
        fact["notes"].append("核准文号跨PDF第2、3页；已实际查看两页并连接同一句，不按日期邻近推定身份。")
    else:
        fact["evidence"]["approval_key"] = [
            {"document_id": row["document_id"], "page": h["page"],
             "normalized_start": h["start"], "normalized_end": h["end"], "literal": h["literal"], "context": h["context"]}
            for h in row["identity_anchors"] if h["matched_event_id"] == fact["event_id"]]
        assert fact["evidence"]["approval_key"]
    codes = row["parsed_values"]["rights_code"]
    fact["source_rights_codes"] = codes
    fact["evidence"]["source_rights_codes"] = [anchor(h) for h in evidence["rights_code"]]
    if row["document_id"] == "1203224562":
        assert set(codes) == {"082673", "086273"}
        fact["updates"]["reminder_code_conflict_open"] = True
        fact["updates"]["reminder_code_conflicting_values"] = codes
        fact["evidence"]["reminder_code_conflict_open"] = fact["evidence"]["source_rights_codes"]
        fact["evidence"]["reminder_code_conflicting_values"] = fact["evidence"]["source_rights_codes"]
        fact["notes"].append("首次提示第1页082673、第3页086273，内部冲突保留；次日更正不能倒填至本公告时点。")
    else:
        assert len(codes) <= 1, (row["document_id"], codes)
    fact.update(calendar_fields_reconciled=True, comparison_status="FOUR_FIELDS_MATCH_PRIOR_KNOWN_SOURCE",
                rejected_table_candidates=rejected,
                payment_segments_not_used_as_full_window=evidence["payment_segment"])
    fact["notes"].append("核对范围为登记日、主要缴款区间、配股价格和计划最大股数；不等同于逐页人工审阅或所有条款完整性。")
    return fact


def connect_reminder(row, source):
    doc, fact = make_fact(source, "600256.SH_A_RIGHTS_2018_157", "CONNECT_PAYMENT_REMINDER")
    fact.update(full_document_semantics_reviewed=True, calendar_fields_reconciled=True,
                review_method="READ_BOTH_PAGES_AND_EXPLICIT_PROSPECTUS_REFERENCE")
    fact["updates"] = {"record_date": "2018-03-19", "connect_payment_dates": CONNECT_DATES,
                       "actual_subscribed_shares": 1515101106, "actual_subscription_cents": 386350782030,
                       "connect_hk_authorization_reported": True,
                       "connect_hk_authorization_economic_date": "2018-03-28",
                       "expected_resumption_date": "2018-04-10"}
    fact["evidence"] = {
        "record_date": doc.literal("2018年3月19日"),
        "connect_payment_dates": doc.anchors(r"2018年3月29日、(?:2018年)?4月3日(?:及|、)(?:2018年)?4月4日"),
        "actual_subscribed_shares": doc.literal("1,515,101,106股"),
        "actual_subscription_cents": doc.literal("3,863,507,820.30元"),
        "connect_hk_authorization_reported": doc.literal("取得香港证监会发出的“原则同意授权函”"),
        "connect_hk_authorization_economic_date": doc.literal("2018年3月28日"),
        "expected_resumption_date": doc.literal("2018年4月10日复牌交易"),
        "identity_bridge": doc.literal("《广汇能源股份有限公司配股说明书》全文及其他相关资料刊载于2018年3月15日"),
        "listing_date_not_yet_announced": doc.literal("上市时间将另行公告"),
    }
    fact["notes"].append("同发行人、明确引用3月15日配股说明书、登记日及境内认购结果共同连接当次发行。三个离散缴款日不能填成连续区间。")
    fact["notes"].append("3月28日获得授权是事件发生日；现有补充提示的最早目录代理时钟为3月29日日末，不倒填可得时间。")
    return fact


ADMIN = {
    "1203316299": ("PREPLAN_TRADING_RESUMPTION", "董事会通过筹划方案后的复牌，股东大会及监管批准仍待取得；不是2018年实际配股缴款结束复牌。",
                   ["公司股票将自2017年4月18日开市起复牌", "还需经公司股东大会审议批准,并获得相关部门的批准"]),
    "1201094804": ("PROPOSED_PROCEEDS_USE_DETAIL", "筹划期120亿元预算及用途细化，仍在审核；未给出实施登记或缴款日期。",
                   ["目前正处于中国证监会审核过程中", "募集资金总额不超过人民币120亿元"]),
    "1201238063": ("REGULATORY_MEETING_CANCELLED", "取消2015年第147次发审委工作会议，不能据此认定发行终止。",
                   ["2015年第147次发行审核委员会工作会议因故取消"]),
    "1201417538": ("PROPOSED_BUDGET_REVISED", "筹划预算由120亿元降至45亿元；不是实际认购额或2016年发行条款的回填。",
                   ["目前正处于中国证监会审核过程中", "按每10股配售3股", "不超过人民币120亿元", "调整为“不超过人民币45亿元"]),
    "1204215173": ("REGULATORY_MEETING_REVIEW_CANCELLED_PENDING_CHECK", "审查会议取消并要求补充核查，不等于发行终止。",
                   ["决定取消第十七届发审委第68次发审委会议对公司配股审报文件的审核", "积极准备补充材料"]),
    "1204222957": ("REGULATORY_REVIEW_SCHEDULED", "已提交补充材料，安排12月18日审核，尚未获核准。",
                   ["于2017年12月13日提交发审委", "2017年12月18日对公司配股项目申请进行审核", "尚需获得中国证监会的核准"]),
    "1201641243": ("PROPOSED_RATIO_AND_ILLUSTRATIVE_QUANTITY", "每10股配2股，391433206股以2015年6月30日股本测算，允许随实施前股本变动调整；不是已实施股数。",
                   ["目前正处于中国证监会审核过程中", "按每10股配售2股", "截至2015年6月30日总股本1,957,166,032股为基数测算", "配股数量按照总股本变动的比例相应调整"]),
}


def administrative_fact(rid, source):
    status, note, literals = ADMIN[rid]
    doc, fact = make_fact(source, None, "PRE_IMPLEMENTATION_ADMINISTRATIVE_DOCUMENT")
    fact.update(administrative_status=status, full_document_semantics_reviewed=True,
                review_method="ALL_EXTRACTED_BODY_TEXT_READ", implemented_calendar_admitted=False)
    fact["notes"].append(note)
    fact["evidence"]["administrative_scope"] = [hit for literal in literals for hit in doc.literal(literal)]
    return fact


def save_pending_scope():
    index = read(OUT / "intermediate_document_index.json")
    pending = [row for row in index if row["title_class"] != "PAYMENT_OR_RESUMPTION_REMINDER_TITLE"]
    assert len(pending) == 195
    save(OUT / "remaining_candidate_only_documents.json", {
        "at": now(), "scope": "尚未完成非提示版本语义核对的195份已存文档；机器候选和局部检索不算全文完成。",
        "title_counts": dict(Counter(r["title_class"] for r in pending)),
        "multiple_event_candidate_documents": [r["document_id"] for r in pending if len(r["candidate_event_ids"]) > 1],
        "partial_use_only": {"1210115278": "仅引用第27页提示公告5次的计划次数，全文仍待核对。"},
        "documents": pending, "new_accounts": 0, "trading_feature_admitted": False})


def build():
    assert not (OUT / "result.json").exists(), "保留已有结果，禁止覆盖。"
    sources = source_documents()
    rows = read(OUT / "reminder_comparison_refined_v1_0_1.json")
    facts = []
    for row in rows:
        rid = row["document_id"]
        if rid == PREPLAN_ID:
            facts.append(administrative_fact(rid, sources[rid]))
        elif rid in CONNECT_IDS:
            facts.append(connect_reminder(row, sources[rid]))
        else:
            facts.append(regular_reminder(row, sources[rid]))
    facts += [administrative_fact(rid, sources[rid]) for rid in ADMIN if rid != PREPLAN_ID]
    assert len(facts) == 189
    save(OUT / "reminder_and_administrative_facts.json", facts)
    combined = deepcopy(read(BASE / "reviewed_document_facts.json"))
    correction = next(r for r in combined if r["document_id"] == "1203236200")
    correction["revises_external_notice"].update(document_id="1203224562", standalone_original_obtained_in_this_scope=True)
    correction["updates"].update(reminder_code_conflict_open=False, reminder_code_conflicting_values=[])
    correction["evidence"]["reminder_code_conflict_open"] = correction["evidence"]["corrected_reminder_rights_code"]
    correction["evidence"]["reminder_code_conflicting_values"] = correction["evidence"]["corrected_reminder_rights_code"]
    save(OUT / "western_correction_link_addendum.json", correction)
    combined += facts
    save(OUT / "combined_source_facts.json", combined)
    groups = defaultdict(list)
    for fact in combined:
        if fact["event_id"]:
            groups[fact["event_id"]].append(fact)
    boundaries, events = [], []
    for event_id, members in sorted(groups.items()):
        clocks = sorted({f["known_at"] for f in members})
        for clock in clocks:
            before = (datetime.fromisoformat(clock) - timedelta(seconds=1)).isoformat()
            a, b = state_at(members, before), state_at(members, clock)
            newly_known = {f["document_id"] for f in members if f["known_at"] == clock}
            assert not newly_known & set(a["source_documents"])
            assert all(s["known_at"] <= clock for s in b["field_sources"].values())
            boundaries.append({"event_id": event_id, "boundary": clock, "before": a, "at": b})
        regular = [r for r in members if r["role"] == "PAYMENT_REMINDER"]
        events.append({"event_id": event_id, "symbol": members[0]["symbol"],
                       "regular_reminder_count": len(regular), "regular_reminder_ids": [r["document_id"] for r in regular],
                       "connect_reminder_ids": [r["document_id"] for r in members if r["role"] == "CONNECT_PAYMENT_REMINDER"],
                       "latest_reviewed_state": state_at(members, clocks[-1]),
                       "full_intermediate_version_coverage": False, "trading_feature_admitted": False})
    assert len(events) == 36 and sum(e["regular_reminder_count"] for e in events) == 179
    save(OUT / "event_chains_with_reminders.json", events)
    save(OUT / "publication_boundary_queries.json", boundaries)
    western = groups["002673.SZ_A_RIGHTS_2017_316"]
    western_checks = [state_at(western, t) for t in ["2017-03-29T23:59:59+08:00", "2017-03-30T23:59:59+08:00", "2017-03-31T23:59:59+08:00"]]
    assert "reminder_code_conflict_open" not in western_checks[0]
    assert western_checks[1]["reminder_code_conflict_open"] is True
    assert western_checks[2]["reminder_code_conflict_open"] is False
    guanghui = groups["600256.SH_A_RIGHTS_2018_157"]
    connect_checks = [state_at(guanghui, t) for t in ["2018-03-28T23:59:59+08:00", "2018-03-29T23:59:59+08:00", "2018-04-03T23:59:59+08:00", "2018-04-04T23:59:59+08:00"]]
    assert "connect_hk_authorization_reported" not in connect_checks[0]
    assert connect_checks[1]["connect_hk_authorization_reported"] is True
    assert connect_checks[2]["known_connect_payment_dates_on_or_after_day"] == CONNECT_DATES[1:]
    assert connect_checks[3]["known_connect_payment_dates_on_or_after_day"] == CONNECT_DATES[2:]
    save(OUT / "specific_time_queries.json", {"western": western_checks, "guanghui": connect_checks})
    huaan = next(e for e in events if e["symbol"] == "600909.SH")
    prospectus = Document(sources["1210115278"])
    gap = {"event_id": huaan["event_id"], "status": "ONE_PLANNED_REMINDER_ORIGINAL_NOT_LOCATED",
           "planned_reminder_count": 5, "located_reminder_count": 4,
           "located_ids": huaan["regular_reminder_ids"], "uncovered_scheduled_trading_date": "2021-06-03",
           "not_proof_of_nonpublication": True, "planning_evidence": prospectus.literal("配股提示性公告(5次)"),
           "planning_source_document_id": "1210115278", "planning_source_url": sources["1210115278"]["source_url"],
           "research_note": "本地目录无该日提示原件；6条定向网页查询及1次搜狐页面打开未定位新的原件。未猜测公告ID或把未找到当作未发布。"}
    save(OUT / "huaan_reminder_coverage_gap.json", gap)
    visual_names = ["1204493232_p3.png", "1211684963_p2.png", "1211684963_p3.png", "1203224562_p3.png"]
    save(OUT / "visual_review_receipt.json", {"at": now(), "scope": "已实际查看4张页面图片，非全部PDF逐页目检。",
         "findings": ["广汇登记日为3月19日，下一行3月20日是缴款起日", "宁波核准文号跨2、3页", "西部第3页确实印为086273"],
         "pages": [{"path": str(OUT / "page_previews" / name), "sha256": digest(OUT / "page_previews" / name)} for name in visual_names]})
    counts = Counter(f["role"] for f in facts)
    result = {"at": now(), "study_id": "510300_FACTOR96_RIGHTS_INTERMEDIATE_V1",
              "status": "FIXED_SCOPE_REMINDER_FIELDS_RECONCILED_WITH_EXPLICIT_GAPS",
              "local_remaining_documents_candidate_scanned": 365, "additional_targeted_originals": 19,
              "new_archived_source_http_responses": 19, "title_rows_reclassified": 717,
              "indexed_documents": 378, "indexed_pages": 21765, "field_candidate_clauses": 5831,
              "reminder_title_documents_classified": 183, "new_source_facts": len(facts),
              "source_fact_roles": dict(counts), "regular_reminders_four_fields_reconciled": 179,
              "connect_reminders": 3, "preimplementation_or_administrative_documents": 7,
              "remaining_indexed_documents_candidate_only": 195,
              "targeted_prospectus_planned_count_evidence_documents": 1,
              "events": 36, "combined_source_facts": len(combined),
              "publication_boundary_pairs": len(boundaries), "publication_boundary_queries": len(boundaries)*2,
              "wrong_registration_candidates_retained_and_excluded": 5,
              "western_original_code_conflict_and_next_day_correction_linked": True,
              "huaan_missing_planned_reminder_originals": 1,
              "review_scope": "179常规提示的四字段自动定位并与当时已知主公告比对；异常及10份行政/沪股通短文人工核对，4页面目检。未逐页人工读完整378文档。",
              "full_intermediate_version_coverage": False, "full_M06_calendar_established": False,
              "free_float_denominator_established": False, "historical_first_publication_verified": False,
              "strict_M06_pressure": None, "trading_feature_admitted": False,
              "new_accounts": 0, "new_returns": 0, "new_models": 0,
              "orders_authorized": False, "external_review": "NOT_PERFORMED", "delivery_package_created": False}
    save(OUT / "result.json", result)
    flat = [{"发行身份": e["event_id"], "常规提示份数": e["regular_reminder_count"],
             "沪股通提示份数": len(e["connect_reminder_ids"]),
             "登记日": e["latest_reviewed_state"]["record_date"],
             "主要缴款开始": e["latest_reviewed_state"]["payment_start"],
             "主要缴款结束": e["latest_reviewed_state"]["payment_end"],
             "中间版本覆盖完整": False, "严格M06压力": "NOT_COMPUTED",
             "提示原件ID": "|".join(e["regular_reminder_ids"]+e["connect_reminder_ids"])} for e in events]
    with (OUT / "36组配股提示覆盖.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    save_pending_scope()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    build()
