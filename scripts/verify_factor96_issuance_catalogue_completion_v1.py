"""从保存响应只读核对标题修正、发行人代码分离和失败窗口补全。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re

import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify(root):
    base = root.parent / "510300_factor96_issuance_catalogue_v1"
    frozen, began = read(root / "freeze.json"), read(root / "run_started.json")
    assert pd.Timestamp(frozen["at"]) < pd.Timestamp(began["at"])
    assert digest(root / "freeze.json") == began["freeze_sha256"]
    for item in frozen["files"]:
        folder = root if item["scope"] == "local" else base
        assert digest(folder / item["path"]) == item["sha256"], item
    targets = read(root / "targets.json")
    base_jobs = {p.name: read(p) for p in (base / "jobs").glob("*.json")}
    assert {t["base_job"] for t in targets} == {name for name, job in base_jobs.items() if not job["complete"]}
    groups = {g["group_id"]: g["companies"] for g in read(base / "inputs/groups.json")}
    actual = pd.read_parquet(root / "catalogue_occurrences.parquet")
    raw_cache, receipt_cache = {}, {}

    def source(scope, receipt_name):
        key = (scope, receipt_name)
        if key not in receipt_cache:
            folder = base if scope == "base" else root
            receipt = read(folder / receipt_name)
            if receipt.get("raw_path"):
                raw = folder / receipt["raw_path"]
                assert raw.stat().st_size == receipt["bytes"] and digest(raw) == receipt["sha256"]
                if receipt["status"] == "HTTP_OK":
                    raw_cache[key] = read(raw)
            receipt_cache[key] = receipt
        return receipt_cache[key], raw_cache.get(key)

    def identity(row):
        return (row["symbol"], row["document_id"], row["query"], row["reported_security_code"],
                row["source_scope"], row["raw_response_path"])

    actual_counter = Counter()
    for row in actual.to_dict("records"):
        receipt, body = source(row["source_scope"], row["request_receipt"])
        assert receipt["status"] == "HTTP_OK"
        assert row["raw_response_path"] == receipt["raw_path"] and row["raw_response_sha256"] == receipt["sha256"]
        payload = receipt["payload"]
        found = [r for r in body["announcements"] if str(r["announcementId"]) == row["document_id"]
                 and r["secCode"] == row["reported_security_code"] and r["orgId"] == row["org_id"]]
        assert len(found) == 1
        raw = found[0]
        assert raw["announcementTitle"] == row["raw_title"]
        assert row["title"] == html.unescape(re.sub(r"</?em(?:\s+[^>]*)?>", "", raw["announcementTitle"], flags=re.I))
        keyword = {"categories": "", "issuance_title": "发行", "rights_title": "配股"}[row["query"]]
        assert payload["searchkey"] == keyword and (not keyword or keyword in row["title"])
        group_id = row["window_id"].split("_")[0]
        company = [c for c in groups[group_id] if c["org_id"] == row["org_id"]]
        assert len(company) == 1
        company = company[0]
        assert row["symbol"] == company["symbol"] and row["query_security_code"] == company["code"]
        expected_relationship = "SAME_STOCK_CODE" if raw["secCode"] == company["code"] else "OTHER_CODE_SAME_ORGANIZATION_UNRESOLVED"
        assert row["code_relationship"] == expected_relationship
        assert payload["stock"] == ";".join(c["code"] + "," + c["org_id"] for c in groups[group_id])
        clock = pd.to_datetime(raw["announcementTime"], unit="ms", utc=True).tz_convert("Asia/Shanghai")
        assert row["catalogue_timestamp"] == clock.isoformat() and row["catalogue_date"] == clock.date().isoformat()
        start, end = payload["seDate"].split("~")
        assert start <= row["catalogue_date"] <= end
        assert row["source_url"] == "https://static.cninfo.com.cn/" + raw["adjunctUrl"]
        assert not row["historical_first_publication_verified"] and not row["trading_feature_admitted"]
        actual_counter[identity(row)] += 1
    expected_counter, effective, title_changes = Counter(), [], []
    completed_gaps, complete_leaves = 0, 0
    for name, old in sorted(base_jobs.items()):
        group_id, query = old["group_id"], old["query"]
        if old["complete"]:
            for row in old["rows"]:
                expected_counter[(row["symbol"], row["document_id"], row["query"], row["sec_code"], "base", row["raw_response_path"])] += 1
                receipt, body = source("base", row["request_receipt"])
                raw = next(r for r in body["announcements"] if str(r["announcementId"]) == row["document_id"] and r["secCode"] == row["sec_code"])
                corrected = html.unescape(re.sub(r"</?em(?:\s+[^>]*)?>", "", raw["announcementTitle"], flags=re.I))
                if row["title"] != corrected:
                    title_changes.append({"symbol": row["symbol"], "document_id": row["document_id"], "query": row["query"],
                        "old_title": row["title"], "corrected_title": corrected, "raw_response_path": row["raw_response_path"]})
            effective.append({"group_id": group_id, "query": query, "complete": True, "source": "base", "rows": len(old["rows"]), "base_rows": len(old["rows"])})
            continue
        job = read(root / "jobs" / name)
        assert job["companies"] == old["companies"] and job["query"] == query
        windows = {w["window_id"]: w for w in job["windows"]}
        top = job["windows"][0]
        assert top["start"] == "2015-01-01" and top["end"] == "2025-12-31"
        assert job["complete"] == top["complete"]
        completed_gaps += int(job["complete"])
        expected_counter.update(identity(r) for r in job["rows"])
        for window in windows.values():
            if window.get("request_receipt"):
                receipt, body = source("completion", window["request_receipt"])
                assert receipt["payload"]["pageNum"] == "1" and receipt["payload"]["pageSize"] == "1500"
                assert receipt["payload"]["seDate"] == window["start"] + "~" + window["end"]
            if window.get("child_windows"):
                left, right = [windows[key] for key in window["child_windows"]]
                assert left["start"] == window["start"] and right["end"] == window["end"]
                assert pd.Timestamp(left["end"]) + pd.Timedelta(days=1) == pd.Timestamp(right["start"])
                if window["complete"]:
                    assert left["complete"] and right["complete"]
                    assert window["received"] == left["received"] + right["received"] == window["declared_total"]
                continue
            if window["complete"]:
                assert window["status"] == "COMPLETE_SINGLE_RESPONSE" and receipt["status"] == "HTTP_OK"
                batch = body.get("announcements") or []
                selected = [r for r in job["rows"] if r["window_id"] == window["window_id"]]
                assert len(batch) == len(selected) == window["received"] == window["declared_total"] == body["totalAnnouncement"]
                keys = [(str(r["orgId"]), str(r["secCode"]), str(r["announcementId"])) for r in batch]
                assert len(keys) == len(set(keys))
                assert keys == [(r["org_id"], r["reported_security_code"], r["document_id"]) for r in selected]
                complete_leaves += 1
        effective.append({"group_id": group_id, "query": query, "complete": job["complete"], "source": "completion", "rows": len(job["rows"]), "base_rows": len(old["rows"])})
    assert actual_counter == expected_counter
    assert read(root / "effective_jobs.json") == effective
    assert read(root / "corrected_titles.json") == title_changes
    rebuilt = []
    for _, group in actual.groupby(["org_id", "document_id"], sort=True):
        row = group.iloc[0].to_dict()
        fields = ["title", "catalogue_timestamp", "catalogue_date", "source_url", "title_role"]
        row.update(query_sources="|".join(sorted(group["query"].unique())), occurrence_count=len(group),
            reported_security_codes="|".join(sorted(group.reported_security_code.unique())),
            code_relationships="|".join(sorted(group.code_relationship.unique())),
            catalogue_metadata_conflicts="|".join(c for c in fields if group[c].nunique(dropna=False) != 1))
        rebuilt.append(row)
    unique = pd.read_parquet(root / "unique_documents.parquet")
    pd.testing.assert_frame_equal(unique, pd.DataFrame(rebuilt), check_dtype=False)
    receipts = list((root / "receipts").glob("*.json"))
    for file in receipts:
        source("completion", file.relative_to(root).as_posix())
    result = read(root / "result.json")
    assert result["query_jobs"] == len(effective) == 987
    assert result["targeted_gap_jobs"] == len(targets) and result["completed_gap_jobs"] == completed_gaps
    assert result["effective_complete_jobs"] == sum(j["complete"] for j in effective)
    assert result["source_occurrences"] == len(actual) and result["unique_issuer_documents"] == len(unique)
    assert result["corrected_prior_occurrence_titles"] == len(title_changes)
    assert result["other_security_code_documents"] == int(unique.code_relationships.str.contains("OTHER_CODE").sum())
    assert result["metadata_conflict_documents"] == int(unique.catalogue_metadata_conflicts.ne("").sum())
    assert result["title_roles"] == dict(Counter(unique.title_role))
    assert result["not_in_categories_but_title_query"] == int((~unique.query_sources.str.contains("categories")).sum())
    assert result["http_requests"] == len(receipts) <= 512
    assert not result["full_M06_calendar_established"] and not result["historical_code_effective_dates_verified"]
    assert result["new_accounts"] == result["new_returns"] == 0 and result["T13"] == "NOT_RUN"
    return {"status": "PASS_SAVED_ISSUANCE_TITLE_IDENTITY_AND_TARGETED_COMPLETION", "query_jobs": 987,
        "effective_complete_jobs": result["effective_complete_jobs"], "completed_gap_jobs": completed_gaps,
        "complete_single_response_leaves": complete_leaves, "source_occurrences": len(actual),
        "unique_issuer_documents": len(unique), "new_accounts": 0, "network_requests": 0,
        "external_review": "NOT_PERFORMED", "scope": "保存响应与修正目录一致；完整只针对固定检索器，不是经济事件全集或交易证据。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    arguments = parser.parse_args()
    output = verify(arguments.root)
    if arguments.receipt:
        arguments.receipt.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)
