"""按原文复核卡补七个方案身份，并分开公告时钟和实际或计划事件日。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_repurchase_missing_originals_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(value):
    return datetime.fromisoformat(value)


def locate(pages, phrase):
    normal = lambda text: re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    needle, hits = normal(phrase), []
    for number, page in enumerate(pages, 1):
        text = normal(page)
        for match in re.finditer(re.escape(needle), text):
            start, end = match.span()
            if needle[0].isdigit() and start and text[start - 1].isdigit():
                continue
            if needle[-1].isdigit() and end < len(text) and text[end].isdigit():
                continue
            hits.append({"page": number, "start": start, "end": end, "quote": needle,
                         "context": text[max(0, start - 50):end + 50]})
    assert hits, phrase
    return hits


def latest_event(events, root_id, timestamp):
    available = [r for r in events if root_id in r["target_roots"] and stamp(r["known_at"]) <= stamp(timestamp)]
    if not available:
        return {"status": "NO_REVIEWED_EVENT_BEFORE_QUERY", "document_ids": []}
    latest = max(stamp(r["known_at"]) for r in available)
    rows = [r for r in available if stamp(r["known_at"]) == latest]
    if len(rows) != 1:
        return {"status": "SAME_CLOCK_UNRESOLVED", "document_ids": sorted(r["document_id"] for r in rows)}
    row = rows[0]
    # 仅依据文件明确声明；计划日已过、事件日较早都不会使记录提前生效或自动完成。
    return {"status": row["status"], "document_ids": [row["document_id"]], "known_at": row["known_at"],
            "reported_event_date": row.get("event_date"),
            "completion_confirmed_by_document": row.get("completion_confirmed_by_document", False),
            "root_affected_shares": row.get("root_shares", {}).get(root_id),
            "remaining_inventory_shares": row.get("remaining_inventory_shares"),
            "scope": "LATEST_REVIEWED_EVENT_ONLY_NOT_COMPLETE_CURRENT_PLAN_STATE",
            "trading_feature_admitted": False}


def build():
    documents = {r["document_id"]: r for name in ["documents.json", "local_source_documents.json"] for r in read(OUT / name)}
    original_cards, followup_cards = read(OUT / "root_review_cards.json"), read(OUT / "followup_review_cards.json")
    old = read(OUT / "inputs/prior_change_ledger.json")
    updated = deepcopy(old)
    changes = {r["document_id"]: r for r in updated}
    missing = read(OUT / "inputs/prior_identified_missing_originals.json")
    assert {r["change_document_id"] for r in missing} == {r["change_document_id"] for r in original_cards}
    roots, events = [], []
    root_ids = {n["document_id"]: n["root_id"] for n in read(OUT / "inputs/plan_nodes.json")}
    for card in original_cards:
        source = documents[card["original_id"]]
        assert source["symbol"] == card["symbol"] and source["status"] == "PDF_TEXT_SAVED"
        assert source["code_in_first_page"]
        pages = read(OUT / source["text_path"])
        root_id = card["symbol"] + "_ORIGINAL_" + card["original_id"]
        known_at = source["catalogue_date"] + "T23:59:59+08:00"
        change = changes[card["change_document_id"]]
        assert stamp(known_at) < stamp(change["known_at"])
        assert card["board_date"] <= source["catalogue_date"]
        proof = {label: locate(pages, phrase) for label, phrase in card["proofs"].items()}
        root = {k: v for k, v in card.items() if k != "proofs"}
        root.update(root_id=root_id, source_url=source["source_url"], raw_sha256=source["raw_sha256"],
                    known_at=known_at, evidence=proof, historical_first_publication_verified=False,
                    trading_feature_admitted=False, full_lifecycle_established=False)
        roots.append(root)
        root_ids[card["original_id"]] = root_id
        matches = [t for t in change["targets"] if t["original_id"] is None and t["original_approval_date"] == card["board_date"]]
        assert len(matches) == 1
        matches[0].update(original_id=card["original_id"], root_id=root_id,
                          identity_status="ORIGINAL_DOCUMENT_MATCHED_BY_ISSUER_BOARD_BUDGET_AND_PURPOSE")
        change["prior_review_status"] = change["review_status"]
        change["review_status"] = "TARGET_ORIGINALS_SUPPLEMENTED_FULL_LIFECYCLE_PENDING"
        change["confirmed_target_roots"] = sorted({t["root_id"] for t in change["targets"]})
        change["supplemented_original_id"] = card["original_id"]
        change["supplemental_source_known_at"] = known_at
        change["prior_snapshot_rewritten"] = False
        events.append({"document_id": card["original_id"], "target_roots": [root_id], "known_at": known_at,
                       "status": "INITIAL_PLAN_TERMS_REPORTED", "event_date": card["board_date"],
                       "completion_confirmed_by_document": False, "initial_purpose": card["purpose"],
                       "full_lifecycle_established": False, "trading_feature_admitted": False})
    for card in original_cards:
        change = changes[card["change_document_id"]]
        events.append({"document_id": change["document_id"], "target_roots": change["confirmed_target_roots"],
                       "known_at": change["known_at"], "status": change["approval_state"],
                       "completion_confirmed_by_document": False,
                       "root_shares": {t["root_id"]: t["proposed_affected_shares"] for t in change["targets"]},
                       "full_lifecycle_established": False, "trading_feature_admitted": False})
    metas = {m["document_id"]: m for m in read(OUT / "inputs/metadata.json")}
    followups = []
    for card in followup_cards:
        source = documents[card["document_id"]]
        row = {k: v for k, v in card.items() if k != "proofs"}
        row.update(known_at=metas[card["document_id"]]["known_at"],
                   evidence={label: locate(read(OUT / source["text_path"]), phrase) for label, phrase in card["proofs"].items()},
                   source_url=source["source_url"], raw_sha256=source["raw_sha256"],
                   target_roots=[root_ids[key] for key in card["target_original_ids"]],
                   root_shares={root_ids[key]: value for key, value in card.get("target_shares", {}).items()},
                   historical_first_publication_verified=False, trading_feature_admitted=False,
                   full_lifecycle_established=False)
        if row.get("target_shares"):
            assert sum(row["target_shares"].values()) == row["total_affected_shares"]
        if row["target_roots"]:
            assert all(root_id.startswith(row["symbol"] + "_") for root_id in row["target_roots"])
            events.append(row)
        followups.append(row)
    return roots, updated, followups, events


def freeze():
    assert not (OUT / "analysis_freeze.json").exists()
    assert read(OUT / "prefreeze_test_receipt.json")["exit_code"] == 0
    roots, updated, followups, events = build()
    for path in [Path(__file__), ROOT / "tests/test_factor96_repurchase_missing_originals_v1.py"]:
        shutil.copyfile(path, OUT / "code" / path.name)
    save(OUT / "analysis_protocol.json", {"at": datetime.now().astimezone().isoformat(),
         "scope": "7个固定缺失目标及同批准日的本地后续文件；先读原文再冻结语义卡与时钟查询，不涉及价格收益选择。",
         "root_selection": "每个目标选择窗口内明确说明对应原批准、初始预算与用途的方案文件；公告号有占位文字时保留未知，不猜测数字。",
         "completion_rule": "明确已完成或已使用完才记录完成；日期已到、标题含实施、申请已递交均不自动等同登记完成。",
         "retrospective_clock": "公告回述先前批准或实际日期时，仅在当前保存公告的known_at之后可见；不倒填到事件日。",
         "remaining_limits": "全体执行用途及M01增量链、自由流通分母、首次历史发布版本仍未完成；T12不运行。",
         "new_accounts": 0, "new_returns": 0, "goal_achieved": False})
    files = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(OUT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    save(OUT / "analysis_freeze.json", {"at": datetime.now().astimezone().isoformat(), "files": files})
    print(f"已冻结7张原方案卡、6张后续复核卡，{len(files)}个输入与代码文件。", flush=True)


def run():
    assert not (OUT / "result.json").exists()
    freeze = read(OUT / "analysis_freeze.json")
    for item in freeze["files"]:
        assert digest(OUT / item["path"]) == item["sha256"]
    save(OUT / "analysis_started.json", {"at": datetime.now().astimezone().isoformat(), "freeze_sha256": digest(OUT / "analysis_freeze.json")})
    roots, updated, followups, events = build()
    for name, value in [("original_root_ledger.json", roots), ("updated_change_ledger.json", updated),
                        ("followup_ledger.json", followups), ("reviewed_events.json", events)]:
        save(OUT / name, value)
    queries = []
    for row in events:
        for root_id in row["target_roots"]:
            for kind, at in [("BEFORE", stamp(row["known_at"]) - timedelta(seconds=1)), ("AT", stamp(row["known_at"]))]:
                query_at = at.isoformat()
                queries.append({"root_id": root_id, "query_at": query_at, "kind": kind,
                                "result": latest_event(events, root_id, query_at)})
    for row in followups:
        for root_id in row["target_roots"]:
            query_at = row["event_date"] + "T12:00:00+08:00"
            queries.append({"root_id": root_id, "query_at": query_at, "kind": "EVENT_DATE_NOON",
                            "result": latest_event(events, root_id, query_at)})
            query_at = "2026-09-22T23:59:59+08:00"
            queries.append({"root_id": root_id, "query_at": query_at, "kind": "FIXED_LATER_QUERY_NO_AUTO_COMPLETION",
                            "result": latest_event(events, root_id, query_at)})
    save(OUT / "asof_event_queries.json", queries)
    sources = read(OUT / "source_result.json")
    result = {"at": datetime.now().astimezone().isoformat(), "study_id": "510300_FACTOR96_REPURCHASE_MISSING_ORIGINALS_V1",
              "status": "SEVEN_ORIGINAL_TARGETS_MATCHED_LIFECYCLE_AND_DENOMINATOR_PENDING",
              "new_confirmed_original_roots": len(roots), "seven_previously_identified_missing_targets_resolved": len(roots),
              "known_missing_original_targets_in_reviewed_cards": sum(t.get("original_id") is None for row in updated for t in row.get("targets", [])),
              "change_documents_with_any_confirmed_root": sum(bool(r["confirmed_target_roots"]) for r in updated),
              "distinct_confirmed_target_roots": len({v for r in updated for v in r["confirmed_target_roots"]}),
              "confirmed_change_to_root_edges": sum(len(r["confirmed_target_roots"]) for r in updated),
              "unreviewed_change_documents": sum(r["action"] == "UNRESOLVED" for r in updated),
              "followup_documents_reviewed": len(followups), "followup_target_documents": sum(bool(r["target_roots"]) for r in followups),
              "followup_background_documents_rejected": sum(not r["target_roots"] for r in followups),
              "followup_explicit_completed_documents": sum(r["completion_confirmed_by_document"] for r in followups),
              "followup_announced_dates_without_saved_completion_confirmation": sum(bool(r["target_roots"]) and not r["completion_confirmed_by_document"] for r in followups),
              "original_announcement_number_unresolved": sum(r.get("announcement_number") is None for r in roots),
              "asof_queries": len(queries), "new_source_pdf_documents": sources["complete_pdf_texts"],
              "collector_direct_http_requests": sources["new_http_requests"],
              "free_float_denominator_established": False, "full_M02_lifecycle_established": False,
              "historical_first_publication_verified": False, "T12": "NOT_RUN", "new_accounts": 0,
              "new_returns": 0, "goal_status": "active", "goal_achieved": False, "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复核七个原方案与后续公告")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
