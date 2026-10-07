"""补记三份本地原方案的对应关系，并核对已保存公告的证据位置和时钟。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.factor96_repurchase_remaining_changes_v1 import (
    OUT, digest, locate, normalize, now, read, save,
)


def main():
    assert not (OUT / "local_original_link_addendum.json").exists(), "不覆盖已保存的对应关系"
    cards = read(OUT / "review_cards.json")
    card_by_id = {r["document_id"]: r for r in cards}
    protocol = read(OUT / "protocol.json")
    targets = {r["document_id"]: r for r in protocol["targets"]}
    base = ROOT / "reports/research/510300_factor96_repurchase_purpose_v1"
    documents = {r["document_id"]: r for r in read(base / "inputs/documents.json")}
    nodes = read(base / "inputs/plan_nodes.json")
    definitions = [
        ("1225215686", "1222926231", "2025-03-27", "EMPLOYEE_PLAN", "POOL_REFERENCE_ALLOCATION_UNKNOWN",
         "公司于2025年3月27日召开了第十一届董事会第十三次会议", "无需经股东大会审议",
         "本次回购股份计划将用于员工持股计划"),
        ("1225215686", "1225038318", "2026-03-26", "EMPLOYEE_PLAN", "POOL_REFERENCE_ALLOCATION_UNKNOWN",
         "公司于2026年3月26日召开了第十二届董事会第四次会议", "无需经股东会审议",
         "本次回购股份计划将用于员工持股计划"),
        ("1225458188", "1225079477", "2026-04-03", "EMPLOYEE_OR_INCENTIVE", "CONFIRMED_TARGET_ORIGINAL",
         "2026年4月3日，公司召开第七届董事会第十四次会议", "无需提交至公司股东会审议",
         "本次回购的股份拟全部用于后续实施员工持股计划或者股权激励"),
    ]
    links = []
    for change_id, original_id, date, purpose, role, board, approval, purpose_phrase in definitions:
        card, source = card_by_id[change_id], documents[original_id]
        matches = [n for n in nodes if n["document_id"] == original_id]
        assert len(matches) == 1
        node = matches[0]
        assert source["symbol"] == card["symbol"] == node["symbol"]
        assert date in card["reported_original_approval_dates"]
        assert pd.Timestamp(node["known_at"]) < pd.Timestamp(card["known_at"])
        raw, txt = base / source["raw_snapshot"], base / source["text_snapshot"]
        assert digest(raw) == source["raw_sha256"]
        assert digest(txt) == source["text_sha256"]
        pages = read(txt)
        link = {
            "change_document_id": change_id, "original_document_id": original_id,
            "symbol": card["symbol"], "root_id": node["root_id"], "original_board_date": date,
            "original_approving_body": "BOARD", "original_purpose": purpose,
            "original_approval_state": "BOARD_NO_SHAREHOLDER_REQUIRED",
            "original_known_at": node["known_at"], "change_known_at": card["known_at"],
            "relation_known_at": card["known_at"], "clock_grade": "CATALOGUE_DATE_END_PROXY",
            "historical_first_publication_verified": False, "relation": role,
            "allocated_change_shares": card["affected_inventory_shares"] if role == "CONFIRMED_TARGET_ORIGINAL" else None,
            "later_change_approval_state": card["approval_state"],
            "later_change_effective_purpose": card["effective_new_purpose_from_this_notice"],
            "source_url": source["source_url"],
            "raw_path": raw.relative_to(ROOT).as_posix(), "raw_sha256": digest(raw),
            "text_path": txt.relative_to(ROOT).as_posix(), "text_sha256": digest(txt),
            "original_evidence": {"board": locate(pages, board), "approval": locate(pages, approval),
                                  "purpose": locate(pages, purpose_phrase)},
            "change_origin_evidence": card["evidence"][date],
            "full_plan_lifecycle_established": False, "trading_feature_admitted": False,
        }
        if role == "POOL_REFERENCE_ALLOCATION_UNKNOWN":
            link["note"] = "明确对应公告列举的原方案，未披露逐方案拟注销数量；不能把合计数量重复分配给每个原方案。"
        else:
            link["note"] = "原方案确认了批准机构和用途；本次变更仍须股东会批准，原方案免于股东会审议不能沿用到后续变更。"
        links.append(link)
    save("local_original_link_addendum.json", {
        "at": now(), "basis": "复用本地原始公告，保留先前逐篇记录，不回填旧冻结研究。",
        "links": links, "referenced_original_documents": 3, "definite_target_links": 1,
        "pooled_reference_links": 2, "new_network_requests": 0, "new_accounts": 0,
    })

    # 逐条核对保存位置，防止页码或归一化位置错误将其他数字当成依据。
    anchor_count = 0
    for row in cards:
        target = targets[row["document_id"]]
        for kind in ("raw", "text"):
            assert digest(ROOT / target[kind + "_path"]) == target[kind + "_sha256"]
        pages = read(ROOT / target["text_path"])
        for evidence in row["evidence"].values():
            for item in evidence:
                text = normalize(pages[item["page"] - 1])
                assert text[item["start"]:item["end"]] == item["phrase"]
                anchor_count += 1
        assert not row["event_represents_new_market_purchase"]
        assert row["new_purchase_cashflow_from_this_notice"] is None
        assert not row["cancellation_completed_by_this_notice"]
    queries = read(OUT / "asof_notice_queries.json")
    assert len(queries) == 3 * len(cards) == 93
    for query in queries:
        row = card_by_id[query["document_id"]]
        state = query["result"]
        if pd.Timestamp(query["query_at"]) < pd.Timestamp(row["known_at"]):
            assert state == {"status": "NO_VIEW_BEFORE_DOCUMENT", "document_id": None}
        else:
            assert state["status"] == row["approval_state"]
            assert state["effective_new_purpose"] == row["effective_new_purpose_from_this_notice"]
            assert not state["cancellation_completed"] and not state["new_market_purchase"]
    assert card_by_id["1223023057"]["approval_state"] == "CHAIR_PROPOSAL_PENDING_BOARD_AND_SHAREHOLDERS"
    assert card_by_id["1223100990"]["prior_proposal_document_id"] == "1223023057"
    assert card_by_id["1221385315"]["effective_new_purpose_from_this_notice"] == "RESTRICTED_STOCK_INCENTIVE"
    assert card_by_id["1224628842"]["prior_purpose_change_document_id"] == "1221385315"
    assert card_by_id["1225575738"]["future_use_completed_by_this_notice"] is False
    assert card_by_id["1225142099"]["affected_inventory_shares"] == 191
    assert card_by_id["1225186448"]["predecessor_issuer_symbol"] is None

    resolved = {(r["change_document_id"], r["original_board_date"]) for r in links}
    remaining = defaultdict(list)
    for card in cards:
        issuer = card.get("predecessor_issuer_name") or card["symbol"]
        for date in card["reported_original_approval_dates"]:
            if (card["document_id"], date) not in resolved:
                remaining[(issuer, date)].append(card["document_id"])
    missing = [{"issuer": issuer, "reported_approval_date": date, "change_document_ids": ids,
                "status": "ORIGINAL_DOCUMENT_REQUIRED", "search_completed_this_round": False}
               for (issuer, date), ids in sorted(remaining.items())]
    save("remaining_original_targets.json", missing)
    save("saved_verification_receipt.json", {
        "at": now(), "status": "PASS_SAVED_NOTICE_ANCHORS_CLOCKS_AND_THREE_LOCAL_ORIGIN_RELATIONS",
        "documents": len(cards), "anchors_checked": anchor_count, "asof_queries_checked": len(queries),
        "original_links": 3, "definite_target_links": 1, "pooled_reference_links": 2,
        "remaining_original_references": sum(len(r["change_document_ids"]) for r in missing),
        "remaining_unique_original_targets": len(missing),
        "input_identities": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                             for p in (OUT / "protocol.json", OUT / "review_cards.json",
                                       OUT / "local_original_link_addendum.json", Path(__file__))],
        "full_M02_lifecycle_established": False, "free_float_denominator_established": False,
        "new_accounts": 0, "goal_achieved": False, "delivery_package_created": False,
    })
    print(f"已核对31份公告的{anchor_count}处定位、93个查询及3份本地原方案；还有{len(missing)}个不同原方案待取得。")


if __name__ == "__main__":
    main()
