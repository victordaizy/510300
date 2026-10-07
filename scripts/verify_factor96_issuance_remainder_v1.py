"""核对新增响应、原表完整继承、跨轮日期闭合和文档并集；不联网。"""
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
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verify(root):
    frozen, began = read(root / "freeze.json"), read(root / "run_started.json")
    assert pd.Timestamp(frozen["at"]) < pd.Timestamp(began["at"])
    assert digest(root / "freeze.json") == began["freeze_sha256"]
    for item in frozen["files"]:
        assert digest(root / item["path"]) == item["sha256"]
    previous = pd.read_parquet(root / "inputs/prior_occurrences.parquet")
    current = pd.read_parquet(root / "catalogue_occurrences.parquet")
    inherited = current[current.source_scope != "remainder"].reset_index(drop=True)
    pd.testing.assert_frame_equal(inherited, previous.reset_index(drop=True), check_dtype=False)
    targets, fresh, target_map = read(root / "targets.json"), [], {}
    for target in targets:
        job = read(root / "jobs" / (target["window_id"] + ".json"))
        assert job["target"] == target
        windows = {w["window_id"]: w for w in job["windows"]}
        top = job["windows"][0]
        assert top["window_id"] == target["window_id"] and top["start"] == target["start"] and top["end"] == target["end"]
        assert job["complete"] == top["complete"]
        target_map[target["window_id"]] = job
        for window in windows.values():
            if window.get("child_windows"):
                left, right = [windows[x] for x in window["child_windows"]]
                assert left["start"] == window["start"] and right["end"] == window["end"]
                assert pd.Timestamp(left["end"]) + pd.Timedelta(days=1) == pd.Timestamp(right["start"])
                if window["complete"]:
                    assert left["complete"] and right["complete"]
                    assert left["received"] + right["received"] == window["received"] == window["declared_total"]
                continue
            rows = [r for r in job["rows"] if r["window_id"] == window["window_id"]]
            if not window["complete"]:
                assert not rows
                continue
            assert window["status"] == "COMPLETE_SINGLE_RESPONSE"
            receipt = read(root / window["request_receipt"])
            assert receipt["status"] == "HTTP_OK"
            assert digest(root / receipt["raw_path"]) == receipt["sha256"]
            body = read(root / receipt["raw_path"])
            batch = body.get("announcements") or []
            assert len(rows) == len(batch) == window["received"] == window["declared_total"] == body["totalAnnouncement"]
            payload = receipt["payload"]
            assert payload["pageNum"] == "1" and payload["pageSize"] == "30"
            assert payload["seDate"] == window["start"] + "~" + window["end"]
            assert payload["stock"] == ";".join(c["code"] + "," + c["org_id"] for c in target["companies"])
            by_org = {c["org_id"]: c for c in target["companies"]}
            for row, raw in zip(rows, batch):
                assert row["org_id"] == raw["orgId"] and row["document_id"] == str(raw["announcementId"])
                assert row["reported_security_code"] == raw["secCode"] and row["query"] == target["query"]
                company = by_org[row["org_id"]]
                assert row["symbol"] == company["symbol"] and row["query_security_code"] == company["code"]
                relationship = "SAME_STOCK_CODE" if raw["secCode"] == company["code"] else "OTHER_CODE_SAME_ORGANIZATION_UNRESOLVED"
                assert row["code_relationship"] == relationship
                title = html.unescape(re.sub(r"</?em(?:\s+[^>]*)?>", "", raw["announcementTitle"], flags=re.I))
                assert row["title"] == title and row["raw_title"] == raw["announcementTitle"]
                assert not payload["searchkey"] or payload["searchkey"] in title
                clock = pd.to_datetime(raw["announcementTime"], unit="ms", utc=True).tz_convert("Asia/Shanghai")
                assert row["catalogue_timestamp"] == clock.isoformat() and row["catalogue_date"] == clock.date().isoformat()
                assert window["start"] <= row["catalogue_date"] <= window["end"]
                assert row["source_url"] == "https://static.cninfo.com.cn/" + raw["adjunctUrl"]
                assert row["raw_response_sha256"] == receipt["sha256"] and row["raw_response_path"] == receipt["raw_path"]
                assert row["source_scope"] == "remainder" and not row["trading_feature_admitted"] and not row["historical_first_publication_verified"]
        fresh.extend(job["rows"])
    expected = pd.concat([previous, pd.DataFrame(fresh)], ignore_index=True).sort_values(
        ["catalogue_date", "symbol", "document_id", "query", "reported_security_code"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(current.reset_index(drop=True), expected, check_dtype=False)
    assert not current.duplicated(["org_id", "document_id", "query", "reported_security_code"]).any()
    jobs = read(root / "effective_jobs.json")
    before = read(root / "inputs/prior_effective_jobs.json")
    for original, state in zip(before, jobs):
        assert (original["group_id"], original["query"]) == (state["group_id"], state["query"])
        if original["complete"]:
            assert state == original
            continue
        saved = read(root / "reconciled_queries" / f"{state['group_id']}_{state['query']}.json")
        old = read(root / "inputs/prior_jobs" / f"{state['group_id']}_{state['query']}.json")
        old_map = {w["window_id"]: w for w in old["windows"]}
        windows = {w["window_id"]: w for w in saved["windows"]}
        assert saved["summary"] == state
        top = windows[old["windows"][0]["window_id"]]
        assert state["complete"] == top["complete"]
        for window in windows.values():
            if window.get("child_windows"):
                left, right = [windows[key] for key in window["child_windows"]]
                assert left["start"] == window["start"] and right["end"] == window["end"]
                assert pd.Timestamp(left["end"]) + pd.Timedelta(days=1) == pd.Timestamp(right["start"])
                if window["complete"]:
                    assert left["complete"] and right["complete"]
                    assert window["received"] == left["received"] + right["received"] == window["declared_total"]
            elif window["provenance"] == "previous_completion":
                assert {k: v for k, v in window.items() if k != "provenance"} == old_map[window["window_id"]]
        selected = current[current.window_id.str.startswith(state["group_id"] + "_") & current["query"].eq(state["query"])]
        assert state["rows"] == len(selected)
        if state["complete"]:
            assert len(selected) == top["declared_total"] == top["received"]
    rebuilt = []
    for _, group in current.groupby(["org_id", "document_id"], sort=True):
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
    for path in receipts:
        receipt = read(path)
        if receipt.get("raw_path"):
            assert digest(root / receipt["raw_path"]) == receipt["sha256"]
            assert (root / receipt["raw_path"]).stat().st_size == receipt["bytes"]
    result = read(root / "result.json")
    assert len(jobs) == result["query_jobs"] == 987
    assert result["effective_complete_jobs"] == sum(j["complete"] for j in jobs)
    assert result["source_occurrences"] == len(current) and result["new_source_occurrences"] == len(fresh)
    assert result["unique_issuer_documents"] == len(unique) and result["http_requests"] == len(receipts) <= 768
    assert result["title_roles"] == dict(Counter(unique.title_role))
    original = pd.read_parquet(root / "inputs/base_unique_documents.parquet")
    absent = set(zip(original.org_id, original.document_id)) - set(zip(unique.org_id, unique.document_id))
    assert result["base_document_ids_not_covered"] == len(absent)
    assert result["new_accounts"] == result["new_returns"] == result["new_models"] == 0
    return {"status": "PASS_SAVED_REMAINDER_RESPONSES_INHERITANCE_AND_DATE_CLOSURE", "effective_complete_jobs": result["effective_complete_jobs"],
        "source_occurrences": len(current), "new_source_occurrences": len(fresh), "unique_issuer_documents": len(unique),
        "base_document_ids_not_covered": len(absent), "new_accounts": 0, "network_requests": 0,
        "scope": "固定前版表逐行继承和新增原响应复算；前版原始来源的完整证据在前版ZIP及其核验回执，当前不递归重跑旧包。",
        "external_review": "NOT_PERFORMED"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    output = verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)
