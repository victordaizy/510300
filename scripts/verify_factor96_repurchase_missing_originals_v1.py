"""只读重算七个来源关联，并独立检查日期不触发自动完成。"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    freeze = read(root / "analysis_freeze.json")
    for row in freeze["files"]:
        p = root / row["path"]
        assert p.stat().st_size == row["bytes"] and digest(p) == row["sha256"], row["path"]
    assert read(root / "analysis_started.json")["freeze_sha256"] == digest(root / "analysis_freeze.json")
    assert datetime.fromisoformat(freeze["at"]) <= datetime.fromisoformat(read(root / "analysis_started.json")["at"])
    request_freeze = read(root / "source_request_freeze.json")
    for row in request_freeze["files"]:
        assert digest(root / row["path"]) == row["sha256"]
    receipts = [read(p) for p in sorted((root / "receipts").glob("*.json"))]
    assert len(receipts) == 20
    for receipt in receipts:
        assert receipt["status"] == "HTTP_OK" and receipt["http_status"] == 200
        assert digest(root / receipt["raw_path"]) == receipt["sha256"]
        assert (root / receipt["raw_path"]).stat().st_size == receipt["bytes"]
    catalogues = read(root / "catalogues.json")
    count = 0
    for query in catalogues:
        assert query["status"] == "COMPLETE_QUERY" and len(query["rows"]) == query["total"]
        assert len(query["receipts"]) == 1
        body = read(root / read(root / query["receipts"][0])["raw_path"])
        assert int(body["totalAnnouncement"]) == query["total"]
        assert {str(r["announcementId"]) for r in body["announcements"]} == {r["document_id"] for r in query["rows"]}
        assert all(r["secCode"] == query["window"]["symbol"][:6] and r["orgId"] == query["window"]["org_id"] for r in body["announcements"])
        count += len(query["rows"])
    assert len(catalogues) == 7 and count == 22
    for source in read(root / "documents.json"):
        assert source["status"] == "PDF_TEXT_SAVED"
        assert digest(root / source["raw_path"]) == source["raw_sha256"]
        assert digest(root / source["text_path"]) == source["text_sha256"]
        assert (root / source["raw_path"]).read_bytes().startswith(b"%PDF")
    for source in read(root / "local_source_documents.json"):
        for kind in ["raw", "text", "receipts"]:
            assert digest(root / source[kind + "_path"]) == source[kind + "_sha256"]
    code = root / "code/factor96_repurchase_missing_originals_analysis_v1.py"
    spec = importlib.util.spec_from_file_location("frozen_original_analysis", code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = root
    recomputed = module.build()
    names = ["original_root_ledger.json", "updated_change_ledger.json", "followup_ledger.json", "reviewed_events.json"]
    for name, value in zip(names, recomputed):
        assert read(root / name) == value, name
    roots, updated, followups, events = recomputed
    assert len(roots) == 7 and len(updated) == 54 and len(followups) == 6
    queries = read(root / "asof_event_queries.json")
    for query in queries:
        available = [r for r in events if query["root_id"] in r["target_roots"] and r["known_at"] <= query["query_at"]]
        actual = query["result"]
        if not available:
            assert actual == {"status": "NO_REVIEWED_EVENT_BEFORE_QUERY", "document_ids": []}
            continue
        newest = max(r["known_at"] for r in available)
        selected = [r for r in available if r["known_at"] == newest]
        assert len(selected) == 1
        row = selected[0]
        assert actual["document_ids"] == [row["document_id"]] and actual["known_at"] == newest
        assert actual["status"] == row["status"]
        assert actual["reported_event_date"] == row.get("event_date")
        assert actual["completion_confirmed_by_document"] == row.get("completion_confirmed_by_document", False)
        assert actual["root_affected_shares"] == row.get("root_shares", {}).get(query["root_id"])
        assert actual["remaining_inventory_shares"] == row.get("remaining_inventory_shares")
        assert not actual["trading_feature_admitted"]
    assert len(queries) == 68
    result = read(root / "result.json")
    assert result["change_documents_with_any_confirmed_root"] == sum(bool(r["confirmed_target_roots"]) for r in updated) == 20
    assert result["distinct_confirmed_target_roots"] == len({v for r in updated for v in r["confirmed_target_roots"]}) == 22
    assert result["confirmed_change_to_root_edges"] == sum(len(r["confirmed_target_roots"]) for r in updated) == 24
    assert result["unreviewed_change_documents"] == sum(r["action"] == "UNRESOLVED" for r in updated) == 31
    assert result["known_missing_original_targets_in_reviewed_cards"] == sum(t["original_id"] is None for r in updated for t in r.get("targets", [])) == 0
    assert sum(r["completion_confirmed_by_document"] for r in followups) == result["followup_explicit_completed_documents"] == 2
    assert sum(bool(r["target_roots"]) and not r["completion_confirmed_by_document"] for r in followups) == 3
    assert result["original_announcement_number_unresolved"] == sum(r["announcement_number"] is None for r in roots) == 1
    assert result["new_accounts"] == result["new_returns"] == 0 and result["T12"] == "NOT_RUN"
    assert not result["goal_achieved"] and not result["free_float_denominator_established"] and not result["full_M02_lifecycle_established"]
    return {"status": "PASS_SAVED_ORIGINAL_ROOTS_AND_LIFECYCLE_CLOCKS", "frozen_files": len(freeze["files"]),
            "collector_http_receipts": len(receipts), "query_windows": len(catalogues), "catalogue_rows": count,
            "new_pdf_documents": 13, "reused_pdf_documents": 13, "original_roots": len(roots),
            "proof_phrases": sum(len(r["evidence"]) for r in roots + followups),
            "page_anchors": sum(len(v) for r in roots + followups for v in r["evidence"].values()),
            "asof_queries": len(queries), "new_accounts": 0, "network_requests": 0, "prior_accounts_replayed": 0,
            "recomputation_method": "冻结解释程序重算保存台账，另按时间排序检查计划不自动完成；不是独立外部语义审阅。",
            "external_review": "NOT_PERFORMED", "goal_achieved": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读核查七个原方案与后续时钟")
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False))
