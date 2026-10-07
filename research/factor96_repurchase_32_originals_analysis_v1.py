"""登记32个原方案的已读条款，保留源间冲突、待批状态及数量未知，不生成交易信号。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from research.factor96_repurchase_32_originals_v1 import OUT, PRIOR, ROOT, digest, now, read, save
from research.factor96_repurchase_remaining_changes_v1 import locate, normalize


# 每行来自逐篇条款阅读。数字保留原文单位；金额类别单列，不把估算额变成承诺下限。
# 顺序：固定窗口序号、原文、用途、用途原句、金额类别、下限/定额、上限、期限月数、期限原句。
CARDS = [
    (0, "1212540927", "EMPLOYEE_OR_INCENTIVE", "将用于实施公司股权激励计划及/或员工持股计划", "CASH_RANGE", "25亿元", "50亿元", 12, "自董事会审议通过回购股份方案之日起12个月内"),
    (1, "1214984635", "EMPLOYEE_OR_INCENTIVE", "用于实施公司员工持股计划或股权激励", "CASH_RANGE", "1.5亿元", "3亿元", 12, "自股东大会审议通过本次回购方案之日起12个月内"),
    (2, "1211684889", "INCENTIVE", "用于后期实施公司及子公司核心团队股权激励", "FIXED_CASH_AMOUNT", "6亿元", None, 12, "自公司董事会审议通过本次回购股份方案之日起不超过12个月"),
    (3, "1212811720", "RESTRICTED_STOCK_INCENTIVE", "实施A股限制性股票股权激励计划", "ESTIMATED_CASH_FROM_SHARE_BOUNDS", "2.03亿元", "4.06亿元", 12, "自公司董事会审议通过回购方案之日起12个月内"),
    (4, "1214710579", "EMPLOYEE_OR_INCENTIVE", "用于股权激励计划或员工持股计划", "CASH_RANGE", "75,000万元", "150,000万元", 12, "自董事会审议通过之日起12个月内"),
    (5, "1210795177", "EMPLOYEE_OR_INCENTIVE", "回购股份用于实施股权激励计划或员工持股计划", "CASH_RANGE", "30,000万元", "60,000万元", 12, "自董事会审议通过本回购方案之日起12个月"),
    (6, "1210543790", "EMPLOYEE_OR_INCENTIVE", "用于实施员工持股计划或股权激励", "CASH_RANGE", "4,000万元", "6,000万元", 12, "自董事会审议通过回购股份方案之日起12个月内"),
    (7, "1210601313", "EMPLOYEE_OR_INCENTIVE", "用于实施员工持股计划或股权激励", "CASH_RANGE", "4,000万元", "6,000万元", 12, "自董事会审议通过回购股份方案之日起12个月内"),
    (8, "1213133675", "EMPLOYEE_OR_INCENTIVE", "用于实施员工持股计划或股权激励", "CASH_RANGE", "10,000万元", "20,000万元", 12, "自董事会审议通过回购股份方案之日起12个月内"),
    (9, "1214473189", "EMPLOYEE_PLAN", "将用于员工持股计划", "CASH_RANGE", "13,600万元", "20,000万元", 12, "自公司董事会审议通过本次回购方案之日起12个月内"),
    (10, "1215080695", "EMPLOYEE_OR_INCENTIVE", "拟用于员工持股计划或股权激励计划", "CASH_RANGE", "3亿元", "6亿元", 12, "自公司董事会审议通过本次回购股份方案之日起12个月内"),
    (11, "1212352493", "INCENTIVE_EMPLOYEE_OR_CANCELLATION", "用于后期实施股权激励计划、员工持股计划或减少注册资本", "CASH_RANGE", "5亿元", "10亿元", 12, "股东大会审议通过回购股份方案之日起12个月"),
    (12, "1219186496", "EMPLOYEE_OR_INCENTIVE", "用于实施员工持股计划或股权激励", "CASH_RANGE", "5,000万元", "10,000万元", 6, "自董事会审议通过本次回购股份方案之日起不超过6个月"),
    (13, "1212828984", "EMPLOYEE_OR_INCENTIVE", "用于实施股权激励计划或员工持股计划", "CASH_RANGE", "10,000万元", "20,000万元", 12, "自董事会审议通过回购股份议案之日起不超过12个月"),
    (14, "1214607954", "EMPLOYEE_OR_INCENTIVE", "用于实施股权激励计划或员工持股计划", "CASH_RANGE", "15,000万元", "30,000万元", 12, "自董事会审议通过回购股份议案之日起不超过12个月"),
    (15, "1214738642", "EMPLOYEE_OR_INCENTIVE", "用于实施股权激励计划或员工持股计划", "CASH_RANGE", "30,000万元", "60,000万元", 12, "自董事会审议通过回购股份议案之日起不超过12个月"),
    (16, "1211374711", "EMPLOYEE_OR_INCENTIVE", "用于实施员工持股计划或股权激励计划", "CASH_RANGE", "3亿元", "5亿元", 12, "自公司董事会审议通过本次回购股份方案之日起12个月内"),
    (17, "1215627569", "PARTNER_PLAN_OR_RESTRICTED_INCENTIVE", "用于实施公司后续的事业合伙人持股计划或限制性股票激励计划", "CASH_RANGE", "5,000万元", "10,000万元", 12, "自公司董事会审议通过回购股份方案之日起不超过12个月"),
    (18, "1215043805", "EMPLOYEE_OR_INCENTIVE", "用于将来实施员工持股计划或股权激励", "CASH_RANGE", "5亿元", "10亿元", 12, "自公司董事会审议通过回购方案之日起12个月内"),
    (19, "1217147079", "EMPLOYEE_OR_INCENTIVE", "拟回购股份用途：员工持股计划或股权激励计划", "CASH_RANGE", "30,000.00万元", "50,000.00万元", 6, "自公司董事局审议通过本回购股份方案之日起6个月内"),
    (20, "1218247673", "MAINTAIN_VALUE", "用于维护安徽海螺水泥股份有限公司（以下简称“公司”）价值及股东权益", "CASH_RANGE", "4亿元", "6亿元", 3, "自董事会审议通过本次回购A股股份方案之日起不超过3个月"),
    (21, "1212752470", "EMPLOYEE_OR_INCENTIVE", "本次回购股份计划将用于股权激励/员工持股计划", "CASH_RANGE", "15亿元", "30亿元", 12, "自董事会审议通过回购股份方案之日起12个月内"),
    (22, "1216663730", "EMPLOYEE_PLAN", "本次回购股份计划将用于员工持股计划", "CASH_RANGE", "15亿元", "30亿元", 12, "自董事会审议通过回购股份方案之日起12个月内"),
    (23, "1211446638", "INCENTIVE", "拟用于实施股权激励计划", "SHARE_RANGE_WITH_CASH_CAP", None, "5.18亿元", 12, "自公司董事会审议通过本次回购股份方案之日起不超过12个月"),
    (24, "1214232168", "EMPLOYEE_OR_INCENTIVE", "用于员工持股计划或股权激励", "CASH_RANGE", "20,000万元", "40,000万元", 12, "自公司董事会审议通过回购股份方案之日起12个月内"),
    (25, "1217554934", "EMPLOYEE_OR_INCENTIVE", "本次回购的股份拟全部用于后续实施员工持股计划或者股权激励", "CASH_RANGE", "5亿元", "10亿元", None, "自董事会审议通过本次回购股份方案之日至2023年9月28日"),
    (26, "1218978576", "MAINTAIN_VALUE", "本次回购股份拟用于维护公司价值及股东权益", "CASH_RANGE", "6亿元", "12亿元", None, "自董事会审议通过本次回购股份方案之日起至2024年2月29日"),
    (27, "1217118049", "EMPLOYEE_OR_INCENTIVE", "公司本次回购股份拟用于员工持股计划及/或股权激励", "CASH_RANGE", "20,000万元", "30,000万元", 12, "自公司股东大会审议通过回购股份方案之日起不超过12个月"),
    (28, "1214962074", "EMPLOYEE_OR_INCENTIVE", "本次回购的股份拟用于股权激励或员工持股计划", "CASH_RANGE", "1.5亿元", "3亿元", 6, "自董事会审议通过之日起6个月内"),
    (29, "1217543818", "EMPLOYEE_OR_INCENTIVE", "回购的股份将在未来适宜时机全部用于员工持股计划或者股权激励", "CASH_RANGE", "30,000万元", "60,000万元", 12, "自董事会审议通过本次回购方案之日起12个月内"),
    (30, "1217775185", "MAINTAIN_VALUE", "拟回购股份的用途：维护公司价值及股东权益", "CASH_RANGE", "3亿元", "6亿元", 3, "自董事会审议通过本次回购A股股份方案之日起不超过3个月"),
    (31, "1219044004", "MAINTAIN_VALUE", "拟回购股份的用途：维护公司价值及股东权益", "CASH_RANGE", "3亿元", "6亿元", 3, "自董事会审议通过本次回购A股股份方案之日起不超过3个月"),
]


def money(text):
    if text is None:
        return None
    unit = 100000000 if text.endswith("亿元") else 10000
    return int(Decimal(text[:-2].replace(",", "")) * unit)


def approval_proof(pages, pending):
    alternatives = (["尚需提交公司股东大会审议", "仍需公司股东大会审议通过后方可实施"] if pending else
                    ["无需提交股东大会审议", "无需提交公司股东大会审议", "无需提交至公司股东大会审议",
                     "无需经股东大会审议", "无须提交股东大会审议"])
    for phrase in alternatives:
        if any(normalize(phrase) in normalize(p) for p in pages):
            return locate(pages, phrase)
    raise AssertionError("未找到已读原方案的审批状态原句")


def reviewed_history_at(root, edges, when):
    when = pd.Timestamp(when)
    if when < pd.Timestamp(root["known_at"]):
        return {"status": "NO_VIEW_BEFORE_ORIGINAL_SOURCE"}
    purpose = root["initial_purpose"]
    source_ids = [root["document_id"]]
    latest = None
    conflict = False
    for edge in sorted(edges, key=lambda r: r["known_at"]):
        if edge["root_id"] != root["root_id"] or pd.Timestamp(edge["known_at"]) > when:
            continue
        latest = edge
        source_ids.append(edge["change_document_id"])
        conflict = conflict or edge["purpose_conflict_with_original"]
        if edge["effective_new_purpose_from_notice"] is not None:
            purpose = edge["effective_new_purpose_from_notice"]
    return {"status": latest["approval_state"] if latest else root["initial_approval_state"],
            "last_reviewed_effective_purpose": None if conflict else purpose,
            "purpose_conflict": conflict, "source_document_ids": source_ids,
            "proposed_purpose": latest["proposed_new_purpose"] if latest else None,
            "cancellation_completion_established": False, "new_cashflow_established": False,
            "scope": "REVIEWED_DOCUMENT_HISTORY_ONLY_NOT_COMPLETE_CURRENT_PLAN_STATE"}


def build():
    documents = {d["document_id"]: d for d in read(OUT / "documents_with_supplement.json")}
    windows = read(OUT / "protocol.json")["windows"]
    notice_cards = {r["document_id"]: r for r in read(PRIOR / "review_cards.json")}
    change_sources = {r["document_id"]: r for r in read(PRIOR / "protocol.json")["targets"]}
    roots, edges, conflicts = [], [], []
    for index, doc_id, purpose, phrase, kind, lower, upper, months, term_phrase in CARDS:
        window, doc = windows[index], documents[doc_id]
        assert window["key"] in doc["window_keys"] and doc["status"] == "PDF_TEXT_SAVED"
        assert doc["symbol"] == window["symbol"] and doc["code_in_first_page"]
        pages = read(OUT / doc["text_path"])
        for field in ("raw", "text"):
            assert digest(OUT / doc[field + "_path"]) == doc[field + "_sha256"]
        board_date = "2021-07-28" if index == 7 else window["original_approval_date"]
        y, m, d = map(int, board_date.split("-"))
        pending = index in (1, 11, 27)
        proof = {"board_date": locate(pages, f"{y}年{m}月{d}日"), "purpose": locate(pages, phrase),
                 "duration": locate(pages, term_phrase), "approval": approval_proof(pages, pending)}
        if lower is not None:
            proof["amount_a"] = locate(pages, lower)
        if upper is not None:
            proof["amount_b"] = locate(pages, upper)
        root = {
            "window_key": window["key"], "document_id": doc_id, "root_id": doc["symbol"] + "_ORIGINAL_" + doc_id,
            "symbol": doc["symbol"], "current_holder_symbol": window["current_holder_symbol"],
            "original_board_date": board_date, "later_notice_reported_board_date": window["original_approval_date"],
            "known_at": doc["known_at"], "source_url": doc["source_url"], "raw_sha256": doc["raw_sha256"],
            "initial_purpose": purpose, "initial_approval_state": "BOARD_PENDING_SHAREHOLDERS" if pending else "BOARD_NO_SHAREHOLDER_REQUIRED",
            "amount_kind": kind, "cash_budget_floor_cny": money(lower) if kind == "CASH_RANGE" else None,
            "cash_budget_ceiling_cny": money(upper) if kind in ("CASH_RANGE", "SHARE_RANGE_WITH_CASH_CAP") else None,
            "fixed_stated_cash_amount_cny": money(lower) if kind == "FIXED_CASH_AMOUNT" else None,
            "estimated_cash_low_cny": money(lower) if kind == "ESTIMATED_CASH_FROM_SHARE_BOUNDS" else None,
            "estimated_cash_high_cny": money(upper) if kind == "ESTIMATED_CASH_FROM_SHARE_BOUNDS" else None,
            "duration_months": months, "duration_anchor": "SHAREHOLDERS_APPROVAL_DATE_NOT_ESTABLISHED" if pending else "BOARD_APPROVAL_DATE",
            "explicit_calendar_end_date": {25: "2023-09-28", 26: "2024-02-29"}.get(index),
            "past_deadline_implies_completed": False, "historical_first_publication_verified": False,
            "full_lifecycle_established": False, "trading_feature_admitted": False, "evidence": proof,
        }
        if index == 3:
            root.update(share_floor=7621088, share_ceiling=15242175, earlier_terms_document_id="1212751542",
                        note="方案原公告未明示董事会日期，使用明确日期的后续报告书确认，最早可见日期按该报告书保存；现金区间为股数估算。")
            proof["share_floor"] = locate(pages, "762.1088万股")
            proof["share_ceiling"] = locate(pages, "1,524.2175万股")
            proof["estimate_qualifier"] = locate(pages, "拟用于回购的资金总额约为2.03亿元至4.06亿元")
        if index == 23:
            root.update(share_floor=6600000, share_ceiling=7700000,
                        note="股数660万至770万、现金仅设5.18亿元上限，不用股数下限乘价格上限制造现金下限。")
            proof["share_range"] = locate(pages, "回购股份数量为660万股至770万股")
        if index == 30:
            root.update(earlier_unavailable_document_id="1217718984", earlier_catalogue_date="2023-08-31",
                        note="更早方案PDF两次传输失败，现用已取得的2023-09-05报告书确认，不借目录将其正文追溯到8月31日。")
        if index in (30, 31):
            proof["predecessor_name"] = locate(pages, "海通证券股份有限公司")
            root["issuer_mapping"] = "ORIGINAL_600837_TO_SUCCESSOR_601211_INVENTORY_ONLY"
        roots.append(root)
        for change_id in window["change_document_ids"]:
            card = notice_cards[change_id]
            assert pd.Timestamp(root["known_at"]) < pd.Timestamp(card["known_at"])
            pooled = card["original_reference_scope"] in ("POOLED_REFERENCES_ALLOCATION_UNKNOWN", "PREDECESSOR_POOLED_INVENTORY_AFTER_MERGER")
            allocation = card.get("explicit_original_allocations", {}).get(window["original_approval_date"])
            if allocation is None and not pooled:
                allocation = card["affected_inventory_shares"]
            relation = "POOLED_ORIGIN_REFERENCE_ALLOCATION_UNKNOWN" if pooled else "CONFIRMED_TARGET_ORIGINAL"
            change_pages = read(ROOT / change_sources[change_id]["text_path"])
            edge = {"root_id": root["root_id"], "original_document_id": doc_id, "change_document_id": change_id,
                    "known_at": card["known_at"], "relation": relation, "allocated_change_shares": allocation,
                    "approval_state": card["approval_state"], "proposed_new_purpose": card["proposed_new_purpose"],
                    "effective_new_purpose_from_notice": card["effective_new_purpose_from_this_notice"],
                    "purpose_conflict_with_original": index == 25, "board_date_conflict": index == 7,
                    "change_source_sha256": card["raw_sha256"],
                    "historical_first_publication_verified": False, "full_lifecycle_established": False}
            if index == 7:
                proof["identity_number"] = locate(pages, "2021-136")
                edge["change_identity_evidence"] = locate(change_pages, "2021-136")
                corroborating = documents["1210612639"]
                corroborating_pages = read(OUT / corroborating["text_path"])
                conflicts.append({"kind": "BOARD_DATE_CONFLICT_IDENTITY_RESOLVED_BY_EXPLICIT_DOCUMENT_NUMBER",
                    "root_id": root["root_id"], "change_document_id": change_id,
                    "original_board_date": board_date, "later_reported_date": "2021-07-29",
                    "identity_number": "2021-136", "later_evidence": card["evidence"]["2021-07-29"],
                    "corroborating_document_id": "1210612639", "corroborating_raw_sha256": corroborating["raw_sha256"],
                    "corroborating_date_evidence": locate(corroborating_pages, "2021年7月28日"),
                    "resolution": "方案身份由明确编号和会议届次确认；两个源的日期分别保留，不把后述日期覆写原文。"})
            if index == 25:
                conflicts.append({"kind": "ORIGINAL_PURPOSE_DIFFERS_FROM_LATER_RETROSPECTIVE_DESCRIPTION",
                    "root_id": root["root_id"], "change_document_id": change_id,
                    "initial_purpose": purpose, "later_retrospective_purpose": "MAINTAIN_VALUE",
                    "later_evidence": locate(change_pages, "回购股份用于维护公司价值及股东权益"),
                    "original_evidence": proof["purpose"],
                    "resolution": "同公司、日期和预算对应；初始用途以当时原文记录，后述用途差异保留为冲突，未证明中间变更链，当前现行用途未知。"})
            edges.append(edge)
    assert len(roots) == len({r["root_id"] for r in roots}) == 32
    assert len(edges) == 34
    return documents, roots, edges, conflicts


def run():
    assert not (OUT / "result.json").exists(), "不覆盖已完成条款研究"
    documents, roots, edges, conflicts = build()
    save(OUT / "analysis_freeze.json", {"at": now(), "scope": "32个固定原方案的条款与冲突，未读新收益。",
        "code_sha256": digest(Path(__file__)), "source_protocol_sha256": digest(OUT / "protocol.json"),
        "source_manifest_sha256": digest(OUT / "documents_with_supplement.json"),
        "original_roots": roots, "edges": edges, "conflicts": conflicts})
    save(OUT / "original_roots.json", roots)
    save(OUT / "origin_to_change_edges.json", edges)
    save(OUT / "source_conflicts.json", conflicts)
    queries = []
    for root in roots:
        times = {pd.Timestamp(root["known_at"]) - timedelta(seconds=1), pd.Timestamp(root["known_at"])}
        for edge in edges:
            if edge["root_id"] == root["root_id"]:
                t = pd.Timestamp(edge["known_at"])
                times.update((t - timedelta(seconds=1), t, t + timedelta(days=365)))
        for timestamp in sorted(times):
            queries.append({"root_id": root["root_id"], "query_at": timestamp.isoformat(),
                            "result": reviewed_history_at(root, edges, timestamp)})
        assert reviewed_history_at(root, edges, pd.Timestamp(root["known_at"]) - timedelta(seconds=1)) == {"status": "NO_VIEW_BEFORE_ORIGINAL_SOURCE"}
    by_index = {index: roots[n] for n, (index, *_) in enumerate(CARDS)}
    tianqi = by_index[9]
    tianqi_edges = sorted([e for e in edges if e["root_id"] == tianqi["root_id"]], key=lambda e: e["known_at"])
    assert reviewed_history_at(tianqi, edges, tianqi_edges[0]["known_at"])["last_reviewed_effective_purpose"] == "RESTRICTED_STOCK_INCENTIVE"
    assert reviewed_history_at(tianqi, edges, tianqi_edges[-1]["known_at"])["last_reviewed_effective_purpose"] == "RESTRICTED_STOCK_INCENTIVE"
    howell = by_index[25]
    latest = next(e for e in edges if e["root_id"] == howell["root_id"])
    assert reviewed_history_at(howell, edges, howell["known_at"])["last_reviewed_effective_purpose"] == "EMPLOYEE_OR_INCENTIVE"
    assert reviewed_history_at(howell, edges, latest["known_at"])["last_reviewed_effective_purpose"] is None
    assert by_index[3]["cash_budget_floor_cny"] is None and by_index[23]["cash_budget_floor_cny"] is None
    assert by_index[2]["cash_budget_floor_cny"] is None and by_index[2]["fixed_stated_cash_amount_cny"] == 600000000
    assert sum(r["initial_approval_state"] == "BOARD_PENDING_SHAREHOLDERS" for r in roots) == 3
    save(OUT / "asof_reviewed_history.json", queries)
    ledger = deepcopy(read(PRIOR / "combined_change_ledger.json"))
    for row in ledger:
        additions = [e for e in edges if e["change_document_id"] == row["document_id"]]
        if not additions:
            continue
        row["additional_original_relations"] = additions
        row["confirmed_target_roots"] = sorted(set(row["confirmed_target_roots"]) | {e["root_id"] for e in additions if e["relation"] == "CONFIRMED_TARGET_ORIGINAL"})
        row["pooled_origin_reference_roots"] = sorted(set(row.get("pooled_origin_reference_roots", [])) | {e["root_id"] for e in additions if e["relation"] == "POOLED_ORIGIN_REFERENCE_ALLOCATION_UNKNOWN"})
        row["original_reference_documents_obtained"] = True
        row["source_conflicts"] = [c for c in conflicts if c["change_document_id"] == row["document_id"]]
        row["review_status"] = "ORIGINAL_TERMS_LINKED_WITH_CONFLICTS_PRESERVED_FULL_LIFECYCLE_PENDING"
    save(OUT / "combined_change_ledger.json", ledger)
    records = [{k: v for k, v in r.items() if k != "evidence"} for r in roots]
    pd.DataFrame(records).to_csv(OUT / "32个原方案条款.csv", index=False, encoding="utf-8-sig")
    receipts = [read(p) for p in (OUT / "receipts").glob("*.json")]
    result = {"at": now(), "study_id": "510300_FACTOR96_REPURCHASE_32_ORIGINALS_V1",
        "status": "ALL_32_ORIGINAL_REFERENCES_DOCUMENTED_TWO_SOURCE_CONFLICTS_PRESERVED",
        "complete_query_windows": 32, "saved_pdf_text_documents": sum(d["status"] == "PDF_TEXT_SAVED" for d in documents.values()),
        "new_http_requests": sum("requested_at" in r for r in receipts), "reviewed_original_plans": len(roots),
        "new_original_to_notice_relations": len(edges), "relation_counts": dict(Counter(e["relation"] for e in edges)),
        "amount_kinds": dict(Counter(r["amount_kind"] for r in roots)),
        "initial_approval_counts": dict(Counter(r["initial_approval_state"] for r in roots)),
        "source_conflicts": len(conflicts), "remaining_unmatched_original_reference_targets": 0,
        "earliest_catalogue_pdf_unavailable_but_later_report_obtained": 1, "earlier_terms_need_later_explicit_board_date": 1,
        "date_conflict_resolved_identity_but_retained_values": 1, "current_purpose_conflict_unresolved": 1,
        "asof_queries": len(queries), "free_float_denominator_established": False, "full_M02_lifecycle_established": False,
        "strict_T12": "NOT_RUN_FREE_FLOAT_AND_COMPLETE_PURPOSE_CLOCK_CHAIN", "new_accounts": 0, "new_returns": 0,
        "goal_status": "active", "goal_achieved": False, "delivery_package_created": False, "orders_authorized": False}
    save(OUT / "result.json", result)
    print(f"已登记32个原方案、34条变更关联和{len(conflicts)}处源间冲突；T12未回测，未制作交付包。")


if __name__ == "__main__":
    run()
