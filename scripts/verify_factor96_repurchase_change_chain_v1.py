"""只读核查已保存的回购变更字段、页内证据与时钟查询，不重新采集或回测。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identity(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return path.stat().st_size, checksum.hexdigest()


def norm(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def verify(root):
    freeze = read(root / "freeze.json")
    for item in freeze["files"]:
        assert identity(root / item["path"]) == (item["bytes"], item["sha256"]), item["path"]
    assert read(root / "run_started.json")["freeze_sha256"] == identity(root / "freeze.json")[1]
    assert stamp(freeze["at"]) <= stamp(read(root / "run_started.json")["at"])
    protocol, result = read(root / "protocol.json"), read(root / "result.json")
    changes, nodes = read(root / "inputs/changes.json"), read(root / "inputs/plan_nodes.json")
    metas = {r["document_id"]: r for r in read(root / "inputs/metadata.json")}
    node_by_id = {r["document_id"]: r for r in nodes}
    documents = {r["document_id"]: r for r in read(root / "inputs/documents.json")}
    cards, ledger = read(root / "review_cards.json"), read(root / "change_ledger.json")
    mentions, queries = read(root / "candidate_mentions.json"), read(root / "asof_change_queries.json")
    assert len(ledger) == len(changes) == protocol["change_documents"] == 54
    assert {r["document_id"] for r in ledger} == {r["document_id"] for r in changes}
    assert len(documents) == protocol["selected_source_documents"] == 80
    assert len(cards) == 23 and all(k in cards for k, rows in mentions.items() if rows)
    for source in documents.values():
        for kind in ["raw", "text", "receipts"]:
            assert identity(root / source[kind + "_path"])[1] == source[kind + "_sha256"]
        assert (root / source["raw_path"]).read_bytes().startswith(b"%PDF")
    for item in changes:
        meta = metas[item["document_id"]]
        expected = []
        for node in nodes:
            if node["symbol"] != item["symbol"] or stamp(node["known_at"]) >= stamp(item["known_at"]):
                continue
            a = sorted(set(meta["referenced_announcements"]) & set(node["document_announcements"]))
            b = sorted(set(meta["original_board_dates"]) & set(node["original_board_dates"]))
            if a or b:
                expected.append({"root_id": node["root_id"], "document_id": node["document_id"],
                                 "announcement_hits": a, "board_date_hits": b,
                                 "role": "MENTION_ONLY_REQUIRES_TARGET_CLAUSE"})
        assert mentions[item["document_id"]] == expected
    before = {r["document_id"]: r for r in changes}
    phrase_count = anchor_count = 0
    for row in ledger:
        key = row["document_id"]
        assert row["known_at"] == before[key]["known_at"]
        assert row["prior_root_id"] == before[key]["root_id"]
        assert row["raw_sha256"] == documents[key]["raw_sha256"]
        assert row["candidate_mentions"] == mentions[key]
        assert not row["historical_first_publication_verified"] and not row["trading_feature_admitted"]
        assert not row["full_plan_lifecycle_established"] and row["new_executed_cashflow"] == "NOT_COMPUTED"
        if key not in cards:
            assert row["approval_state"] == row["action"] == "UNRESOLVED" and not row["confirmed_target_roots"]
            continue
        card = cards[key]
        for field, value in card.items():
            if field not in ["proofs", "targets"]:
                assert row[field] == value, (key, field)
        pages = read(root / documents[key]["text_path"])
        assert set(card["proofs"]) == set(row["evidence"])
        for label, phrase in card["proofs"].items():
            phrase_count += 1
            expected = []
            needle = norm(phrase)
            for page_num, page in enumerate(pages, 1):
                text = norm(page)
                for match in re.finditer(re.escape(needle), text):
                    start, end = match.span()
                    if needle[0].isdigit() and start > 0 and text[start - 1].isdigit():
                        continue
                    if needle[-1].isdigit() and end < len(text) and text[end].isdigit():
                        continue
                    expected.append({"page": page_num, "start": start, "end": end,
                                     "normalized_quote": needle, "context": text[max(0, start - 70):end + 70]})
            assert expected and expected == row["evidence"][label], (key, label)
            anchor_count += len(expected)
        roots = []
        assert len(row["targets"]) == len(card["targets"])
        for saved, target in zip(row["targets"], card["targets"]):
            assert all(saved[k] == v for k, v in target.items())
            if target["original_id"]:
                node = node_by_id[target["original_id"]]
                assert node["symbol"] == row["symbol"] and stamp(node["known_at"]) < stamp(row["known_at"])
                assert node["root_id"] in {m["root_id"] for m in mentions[key]}
                assert saved["root_id"] == node["root_id"]
                assert saved["identity_status"] == "EXPLICIT_TARGET_WITH_EXISTING_PRIOR_ORIGINAL"
                roots.append(node["root_id"])
            else:
                assert saved["root_id"] is None and saved["identity_status"] == "TARGET_IDENTIFIED_ORIGINAL_DOCUMENT_MISSING"
        assert row["confirmed_target_roots"] == sorted(set(roots))
        assert row["context_only_roots"] == sorted({m["root_id"] for m in mentions[key]} - set(roots))
    expected_query_keys = []
    for row in ledger:
        for root_id in row["confirmed_target_roots"]:
            expected_query_keys.extend([(root_id, stamp(row["known_at"]) - timedelta(seconds=1), "BEFORE"),
                                        (root_id, stamp(row["known_at"]), "AT")])
    assert [(q["root_id"], stamp(q["query_at"]), q["side"]) for q in queries] == expected_query_keys
    for query in queries:
        eligible = [r for r in ledger if query["root_id"] in r["confirmed_target_roots"]
                    and stamp(r["known_at"]) <= stamp(query["query_at"])]
        eligible.sort(key=lambda r: stamp(r["known_at"]))
        actual = query["result"]
        if not eligible:
            assert actual == {"status": "NO_REVIEWED_CHANGE_BEFORE_QUERY", "document_ids": [], "effective_new_purpose": None}
            continue
        latest = [r for r in eligible if r["known_at"] == eligible[-1]["known_at"]]
        if len(latest) > 1:
            assert actual == {"status": "SAME_CLOCK_MULTIPLE_DOCUMENTS_UNRESOLVED",
                              "document_ids": sorted(r["document_id"] for r in latest), "effective_new_purpose": None}
            continue
        row = latest[0]
        approved = row["approval_state"] == "BOARD_APPROVED_NO_SHAREHOLDER_REQUIRED"
        assert actual == {"status": row["approval_state"], "document_ids": [row["document_id"]],
                          "action": row["action"], "known_at": row["known_at"],
                          "proposed_purpose": row.get("proposed_purpose"),
                          "effective_new_purpose": row.get("proposed_purpose") if approved else None,
                          "approved_terms": row.get("terms", {}) if approved else {},
                          "scope": "LATEST_REVIEWED_CHANGE_ONLY_NOT_COMPLETE_PLAN_STATE"}
    by_id = {r["document_id"]: r for r in ledger}
    for key in ["1224735615", "1225508551", "1225519613"]:
        assert not by_id[key]["confirmed_target_roots"] and by_id[key]["context_only_roots"]
    for key in ["1224772736", "1225266149", "1225371015", "1225451851"]:
        row = by_id[key]
        assert sum(t["proposed_affected_shares"] for t in row["targets"]) == row["proposed_total_shares"]
        assert sum(t["original_id"] is not None for t in row["targets"]) == 1
    pair = [by_id[k] for k in ["1225525350", "1225530137"]]
    assert pair[0]["raw_sha256"] != pair[1]["raw_sha256"]
    assert pair[0]["known_at"] == pair[1]["known_at"]
    assert by_id["1224960309"]["budget_increase_is_new_executed_cashflow"] is False
    derived = {"change_documents": len(ledger), "reviewed_cards": len(cards),
               "documents_with_prior_candidates": sum(bool(x) for x in mentions.values()),
               "documents_with_confirmed_target_root": sum(bool(r["confirmed_target_roots"]) for r in ledger),
               "distinct_confirmed_target_roots": len({v for r in ledger for v in r["confirmed_target_roots"]}),
               "new_document_root_links_vs_prior": sum(bool(r["confirmed_target_roots"]) and r["prior_root_id"] is None for r in ledger),
               "context_only_root_mentions": sum(len(r.get("context_only_roots", [])) for r in ledger),
               "partial_target_documents": sum(any(t.get("original_id") is None for t in r.get("targets", [])) and bool(r["confirmed_target_roots"]) for r in ledger),
               "review_status_counts": dict(Counter(r["review_status"] for r in ledger)),
               "approval_state_counts": dict(Counter(r["approval_state"] for r in ledger)),
               "asof_queries": len(queries),
               "same_clock_ambiguous_queries": sum(q["result"]["status"] == "SAME_CLOCK_MULTIPLE_DOCUMENTS_UNRESOLVED" for q in queries)}
    assert all(result[k] == v for k, v in derived.items())
    assert result["new_accounts"] == result["new_returns"] == result["new_network_requests"] == 0
    assert result["T12"] == "NOT_RUN" and not result["goal_achieved"]
    assert not result["full_purpose_termination_chain_established"] and not result["free_float_denominator_established"]
    with (root / "逐公告变更对象与审批状态.csv").open(encoding="utf-8-sig", newline="") as stream:
        flat = list(csv.DictReader(stream))
    assert len(flat) == len(ledger)
    for row, item in zip(ledger, flat):
        assert all(item[k] == row[k] for k in ["document_id", "symbol", "known_at", "title", "action", "approval_state", "review_status"])
        assert item["confirmed_target_roots"] == "|".join(row["confirmed_target_roots"])
        assert item["context_only_roots"] == "|".join(row.get("context_only_roots", []))
        assert int(item["missing_target_originals"]) == sum(t["original_id"] is None for t in row.get("targets", []))
        assert item["trading_feature_admitted"] == "False"
    visual = read(root / "visual_review_receipt.json")
    assert len(visual["pages"]) == 3
    for page in visual["pages"]:
        assert identity(root / page["image"])[1] == page["sha256"]
    return {"status": "PASS_SAVED_CHANGE_FIELDS_EVIDENCE_AND_ASOF_QUERIES",
            "frozen_files": len(freeze["files"]), "source_documents": len(documents),
            "change_documents": len(ledger), "reviewed_cards": len(cards), "proof_phrases": phrase_count,
            "page_local_anchors": anchor_count, "asof_queries": len(queries),
            "confirmed_document_root_links": derived["documents_with_confirmed_target_root"],
            "context_only_mentions_rejected": derived["context_only_root_mentions"],
            "same_clock_ambiguous_queries": derived["same_clock_ambiguous_queries"],
            "semantic_review_scope": "23张人工原文复核卡的保存证据与输出一致；不自动证明完整历史语义或首次发布时间。",
            "new_accounts": 0, "network_requests": 0, "prior_accounts_replayed": 0,
            "external_review": "NOT_PERFORMED", "goal_achieved": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读核查回购变更记录")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False))
