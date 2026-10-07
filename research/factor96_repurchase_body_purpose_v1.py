"""复核固定194个正向披露批次的同文档用途正文，不更改现金增量或交易准入。"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

import pandas as pd

from research.factor96_repurchase_purpose_v1 import ROOT, digest, now, normalize, read, save

OUT = ROOT / "reports/research/510300_factor96_repurchase_body_purpose_v1"
BASE = ROOT / "reports/research/510300_factor96_repurchase_purpose_v1"
JOIN = ROOT / "reports/research/510300_factor96_repurchase_execution_join_v1"
STUDY = "510300_FACTOR96_REPURCHASE_BODY_PURPOSE_V1"


def freeze():
    assert not (OUT / "protocol.json").exists()
    batches = read(JOIN / "publication_batches.json")
    selected = [b for b in batches if b["positive_disclosed_increment"] and b["purpose_consensus"] is None]
    ids = {key for b in selected for key in b["frontier_document_ids"]}
    documents = [d for d in read(BASE / "inputs/documents.json") if d["document_id"] in ids]
    fields = {r["document_id"]: r for r in read(BASE / "execution_purpose_fields.json")}
    assert len(selected) == 194 and len(ids) == len(documents) == 195
    for doc in documents:
        assert digest(BASE / doc["text_snapshot"]) == doc["text_sha256"]
    save(OUT / "target_batches.json", selected, True)
    save(OUT / "target_documents.json", documents, True)
    sources = [JOIN / "publication_batches.json", BASE / "inputs/documents.json", BASE / "execution_purpose_fields.json"]
    save(OUT / "protocol.json", {
        "at": now(), "study_id": STUDY, "batches": 194, "documents": 195,
        "selection": "前轮588个正向披露增量批次中用途未知的全部194批次及其195个最新统计日文档；不按收益或用途选择。",
        "prior_unknown_statuses": dict(Counter(fields[key]["purpose"]["status"] for key in ids)),
        "source": "仅使用现有同一执行公告的已存全文与原PDF，不跨公告、跨方案或未来文件填充。",
        "review": "先提取同文档用途候选句和表格，逐项保存精确原文与页码；正文明确、原方案回顾、附条件后备用途、待批准变更、存在冲突、原文未说明分别记载。",
        "current_rule": "只有同文档明确归属于本次方案且未见相反/待决变更的用途才可标为同文档报告用途；历史回顾另标，不宣称完成独立版本链。",
        "fallback": "三年未用于激励则注销、届时无法使用则注销等条件性后备措施不认定为当前注销用途。",
        "multiple_purposes": "保留全部明确用途，不推断资金分配比例；维护价值的动机与注销这一法定去向分字段保存。",
        "clock": "继承原known_at和日期末代理等级，不升级为历史首次公开时刻；仅在原时钟后可查。",
        "unknown_rule": "无法从同文档确认则保留未知；未找到用途不等于没有用途。",
        "cash": "不修改已披露金额、累计差、批次划分和修订冲突隔离。",
        "limits": "自由流通股本仍缺失，严格M01保持空，T12不回测，旧失败策略不恢复。",
        "new_accounts": 0, "new_returns": 0, "new_network_requests": 0,
        "delivery_package_required": False, "goal_achieved": False,
        "inputs": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources],
    }, True)
    print("已固定194个批次、195份同文档用途复核范围。")


def extract():
    assert not (OUT / "review_cards.json").exists()
    fields = {r["document_id"]: r for r in read(BASE / "execution_purpose_fields.json")}
    cards = []
    for doc in read(OUT / "target_documents.json"):
        pages = read(BASE / doc["text_snapshot"])
        snippets = []
        for page_no, raw in enumerate(pages, 1):
            page = normalize(raw)
            intervals = []
            for m in re.finditer("回购用途|用作|用于|注销|减少注册资本|维护公司价值|股权激励|员工持股|变更", page):
                start = max(0, m.start() - 75)
                end = min(len(page), m.end() + 130)
                if intervals and start <= intervals[-1][1]:
                    intervals[-1][1] = end
                else:
                    intervals.append([start, end])
            snippets.extend({"page": page_no, "start": a, "end": b, "text": page[a:b]} for a, b in intervals)
        old = fields[doc["document_id"]]
        cards.append({"document_id": doc["document_id"], "symbol": doc["symbol"], "title": doc["title"],
                      "root_id": old["root_id"], "known_at": old["known_at"],
                      "source_url": doc["source_url"], "raw_sha256": doc["raw_sha256"],
                      "text_sha256": doc["text_sha256"], "previous_purpose_status": old["purpose"]["status"],
                      "old_menu_blocks": old["purpose"]["blocks"], "snippets": snippets})
    save(OUT / "review_cards.json", cards, True)
    print(f"已提取{len(cards)}份公告的用途相关片段，共{sum(len(c['snippets']) for c in cards)}段；尚未写入用途结论。")


def reported_purpose_at(batches, root_id, timestamp):
    eligible = [b for b in batches if b["root_id"] == root_id and pd.Timestamp(b["known_at"]) <= pd.Timestamp(timestamp)]
    if not eligible:
        return {"status": "NO_VIEW", "purpose_consensus": None, "batch_id": None}
    latest = max(eligible, key=lambda b: pd.Timestamp(b["known_at"]))
    return {"status": "SAME_DOCUMENT_REPORTED_PURPOSE" if latest["purpose_consensus"] is not None else "LATEST_BATCH_PURPOSE_UNKNOWN",
            "purpose_consensus": latest["purpose_consensus"], "batch_id": latest["batch_id"],
            "known_at": latest["known_at"], "effective_legal_purposes": None}


def assemble():
    assert not (OUT / "result.json").exists()
    protocol = read(OUT / "protocol.json")
    for source in protocol["inputs"]:
        assert digest(ROOT / source["path"]) == source["sha256"]
    assert digest(OUT / "reviewed_source_rows.json") == read(OUT / "source_review_receipt.json")["reviewed_source_rows_sha256"]
    save(OUT / "assembly_start_receipt.json", {"at": now(), "module_sha256": digest(Path(__file__)),
        "reviewed_rows_sha256": digest(OUT / "reviewed_source_rows.json"), "before_aggregation": True}, True)
    reviews = {r["document_id"]: r for r in read(OUT / "reviewed_source_rows.json")}
    old_contexts = read(JOIN / "execution_context_rows.json")
    old_batches = read(JOIN / "publication_batches.json")
    documents = {d["document_id"]: d for d in read(OUT / "target_documents.json")}
    for row in reviews.values():
        doc = documents[row["document_id"]]
        pages = [normalize(p) for p in read(BASE / doc["text_snapshot"])]
        assert digest(BASE / doc["text_snapshot"]) == row["text_sha256"]
        for group in row["evidence"]:
            for proof in group:
                assert pages[proof["page"] - 1][proof["start"]:proof["end"]] == proof["phrase"]
    contexts = []
    for old in old_contexts:
        new = dict(old)
        review = reviews.get(old["document_id"])
        if review:
            assert review["root_id"] == old["root_id"] and review["known_at"] == old["known_at"]
            assert old["same_document_purposes"] is None
            new["legacy_menu_status"] = old["same_document_purpose_status"]
            new["same_document_purpose_status"] = review["review_status"]
            new["same_document_purposes"] = review["resolved_reported_purposes"]
            new["same_document_purpose_review"] = review
        contexts.append(new)
    purpose_by_id = {r["document_id"]: r["same_document_purposes"] for r in contexts}
    batches = []
    updated_ids = []
    for old in old_batches:
        new = dict(old)
        targets = [key for key in old["frontier_document_ids"] if key in reviews]
        if targets:
            assert old["positive_disclosed_increment"] and old["purpose_consensus"] is None
            values = [purpose_by_id[key] for key in old["frontier_document_ids"]]
            chosen = values[0] if all(v is not None and v == values[0] for v in values) else None
            new["legacy_purpose_consensus"] = None
            new["purpose_consensus"] = chosen
            new["purpose_source_document_ids"] = targets
            new["purpose_source_scopes"] = [reviews[key]["evidence_scope"] for key in targets]
            new["purpose_review_statuses"] = [reviews[key]["review_status"] for key in targets]
            new["full_effective_version_chain_established"] = False
            if chosen is not None:
                updated_ids.append(old["batch_id"])
        batches.append(new)
    cash_identity_fields = ("batch_id", "root_id", "known_at", "document_ids", "frontier_document_ids", "frontier_cumulative_cents",
                            "reported_increment_cents", "increment_status", "positive_disclosed_increment", "previous_publication_document_id",
                            "strict_M01_value", "trading_feature_admitted", "source_clock_grade")
    for old, new in zip(old_batches, batches):
        assert all(old[k] == new[k] for k in cash_identity_fields)
        if old["purpose_consensus"] is not None:
            assert old["purpose_consensus"] == new["purpose_consensus"]
    positive = [b for b in batches if b["positive_disclosed_increment"]]
    unresolved = [b for b in positive if b["purpose_consensus"] is None]
    # 从逐份已读的条款登记附条件后备注销，后备条款不改变当前用途。
    cards = sorted(read(OUT / "review_cards.json"), key=lambda r: (r["symbol"], r["known_at"]))
    fallback_indices = [40, 50, 52, 64, 66, 68, 90, 91, 92, 93, 94, 98, 105, 111, 113, 118, 126, 135, 138, 168, 184, 186]
    fallback = []
    for i in fallback_indices:
        card = cards[i]
        row = reviews[card["document_id"]]
        assert "CANCEL_CAPITAL" not in row["resolved_reported_purposes"]
        snippets = [s for s in card["snippets"] if "注销" in s["text"]]
        assert snippets
        fallback.append({"document_id": row["document_id"], "resolved_reported_purposes": row["resolved_reported_purposes"],
                         "conditional_cancellation_only": True, "same_document_contexts": snippets})
    # 对时间边界和同日双文档做针对性检查，不采用未来完成公告补早期未知用途。
    cases = ["1221816103", "1222206565", "1224919220", "1221640298", "1221917357", "1222707114",
             "1225536562", "1225536563", "1225480467", "1221203760", "1224886159", "1224401365"]
    queries = []
    for key in cases:
        row = reviews[key]
        stamp = pd.Timestamp(row["known_at"])
        before = reported_purpose_at(batches, row["root_id"], stamp - pd.Timedelta(seconds=1))
        current = reported_purpose_at(batches, row["root_id"], stamp)
        assert current["purpose_consensus"] == row["resolved_reported_purposes"]
        if before.get("known_at"):
            assert pd.Timestamp(before["known_at"]) < stamp
        queries.append({"document_id": key, "before_query_at": (stamp-pd.Timedelta(seconds=1)).isoformat(),
                        "before": before, "at_query_at": stamp.isoformat(), "at": current})
    assert reviews["1221816103"]["resolved_reported_purposes"] is None
    assert reviews["1222206565"]["resolved_reported_purposes"] == ["EMPLOYEE_INCENTIVE"]
    assert reviews["1224919220"]["resolved_reported_purposes"] == ["CANCEL_CAPITAL"]
    assert all(reviews[k]["resolved_reported_purposes"] is None for k in ("1221640298", "1221917357"))
    assert reviews["1222707114"]["resolved_reported_purposes"] == ["CANCEL_CAPITAL"]
    assert all(reviews[k]["resolved_reported_purposes"] == ["CANCEL_CAPITAL"] for k in ("1221203760", "1224886159"))
    assert len(batches) == 946 and len(positive) == 588 and len(contexts) == 948
    save(OUT / "execution_context_rows.json", contexts, True)
    save(OUT / "publication_batches.json", batches, True)
    save(OUT / "remaining_unknown_positive_batches.json", unresolved, True)
    save(OUT / "conditional_fallback_review.json", fallback, True)
    save(OUT / "asof_boundary_queries.json", queries, True)
    pd.DataFrame([{"document_id": r["document_id"], "symbol": r["symbol"], "root_id": r["root_id"], "known_at": r["known_at"],
                   "reported_purposes": "|".join(r["same_document_reported_purposes"] or []),
                   "resolved_reported_purposes": "|".join(r["resolved_reported_purposes"] or []),
                   "status": r["review_status"], "scope": r["evidence_scope"], "note": r["review_note"], "source_url": r["source_url"]}
                  for r in reviews.values()]).to_csv(OUT / "195份执行公告用途复核.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(batches).to_csv(OUT / "946个公开批次.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(unresolved).to_csv(OUT / "21个用途仍未知正向批次.csv", index=False, encoding="utf-8-sig")
    result = {"at": now(), "study_id": STUDY, "status": "SAME_DOCUMENT_PURPOSE_RECOVERY_COMPLETE_STRICT_M01_BLOCKED",
              "reviewed_documents": len(reviews), "reviewed_positive_batches": 194,
              "review_statuses": dict(Counter(r["review_status"] for r in reviews.values())),
              "new_resolved_documents": sum(r["resolved_reported_purposes"] is not None for r in reviews.values()),
              "new_resolved_positive_batches": len(updated_ids),
              "positive_batches": len(positive), "positive_batches_with_reported_purpose": len(positive)-len(unresolved),
              "positive_batches_remaining_unknown": len(unresolved),
              "all_batches_with_reported_purpose": sum(b["purpose_consensus"] is not None for b in batches),
              "all_batches_remaining_unknown": sum(b["purpose_consensus"] is None for b in batches),
              "all_execution_documents_with_reported_purpose": sum(r["same_document_purposes"] is not None for r in contexts),
              "new_resolved_document_categories": dict(Counter("|".join(r["resolved_reported_purposes"]) for r in reviews.values() if r["resolved_reported_purposes"])),
              "conditional_fallback_documents_not_misclassified_as_current_cancellation": len(fallback),
              "mixed_purpose_documents": sum(len(r["resolved_reported_purposes"] or []) > 1 for r in reviews.values()),
              "share_allocation_bounds_not_cash_allocation_documents": 2,
              "purpose_and_share_disposition_separately_recorded_documents": 3,
              "other_plan_purpose_exclusions": 2, "visual_pages_reviewed": 5,
              "cash_amounts_identity_clocks_unchanged": True,
              "full_effective_purpose_lifecycle_established": False, "free_float_denominator_established": False,
              "strict_M01_non_null_rows": 0, "T12_status": "NOT_RUN",
              "new_accounts": 0, "new_network_requests": 0, "new_returns": 0,
              "goal_achieved": False, "delivery_package_created": False,
              "evidence_boundary": "同文档披露用途补充，不代表全库完整用途生效链，也不代表源字段已可用于交易。"}
    assert result["new_resolved_documents"] == 174 and result["new_resolved_positive_batches"] == 173
    assert result["positive_batches_remaining_unknown"] == 21
    save(OUT / "result.json", result, True)
    print(f"用途明确的正向批次由394增至{result['positive_batches_with_reported_purpose']}，仍未知{len(unresolved)}；现金增量和公开时钟均保持原值。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="同文档回购用途正文复核")
    parser.add_argument("action", choices=["freeze", "extract", "assemble"])
    {"freeze": freeze, "extract": extract, "assemble": assemble}[parser.parse_args().action]()
