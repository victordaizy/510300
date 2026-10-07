"""只读复核发行目录的固定范围、原响应、分页缺口与三路并集。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def verify(root):
    frozen, began = read(root / "freeze.json"), read(root / "run_started.json")
    assert pd.Timestamp(frozen["at"]) < pd.Timestamp(began["at"])
    assert digest(root / "freeze.json") == began["freeze_sha256"]
    for item in frozen["files"]:
        assert digest(root / item["path"]) == item["sha256"], item["path"]
    protocol = read(root / "protocol.json")
    groups = read(root / "inputs/groups.json")
    companies = [c for g in groups for c in g["companies"]]
    membership = pd.read_parquet(root / "inputs/membership.parquet")
    desired = sorted(membership.loc[pd.to_datetime(membership.membership_date).between(*protocol["period"]), "symbol"].unique())
    assert [c["symbol"] for c in companies] == desired
    assert len(groups) == 329 and len(companies) == 658
    assert len({c["code"] for c in companies}) == 658
    queries = {"categories": "", "issuance_title": "发行", "rights_title": "配股"}
    all_rows, all_windows, complete_jobs, complete_leaves = [], [], 0, 0
    raw_cache, receipt_cache, checked_occurrences = {}, {}, 0

    def source_receipt(name):
        if name not in receipt_cache:
            receipt = read(root / name)
            if receipt.get("raw_path"):
                assert digest(root / receipt["raw_path"]) == receipt["sha256"]
                assert (root / receipt["raw_path"]).stat().st_size == receipt["bytes"]
            receipt_cache[name] = receipt
        return receipt_cache[name]

    def source_body(name, checksum):
        if name not in raw_cache:
            assert digest(root / name) == checksum
            raw_cache[name] = read(root / name)
        return raw_cache[name]

    for group in groups:
        allowed = {c["code"]: c["org_id"] for c in group["companies"]}
        for query, keyword in queries.items():
            job = read(root / "jobs" / f"{group['group_id']}_{query}.json")
            assert job["companies"] == group["companies"] and job["query"] == query
            windows = job["windows"]
            window_map = {w["window_id"]: w for w in windows}
            assert len(window_map) == len(windows)
            assert job["complete"] == windows[0]["complete"]
            complete_jobs += int(job["complete"])
            for row in job["rows"]:
                assert row["symbol"][:6] in allowed and row["org_id"] == allowed[row["symbol"][:6]]
                receipt = source_receipt(row["request_receipt"])
                assert receipt["status"] == "HTTP_OK" and receipt["raw_path"] == row["raw_response_path"]
                assert receipt["sha256"] == row["raw_response_sha256"]
                assert receipt["payload"]["pageNum"] == str(row["page"])
                assert receipt["payload"]["searchkey"] == keyword
                assert receipt["payload"]["stock"] == ";".join(code + "," + org for code, org in allowed.items())
                body = source_body(row["raw_response_path"], row["raw_response_sha256"])
                matching = [r for r in body["announcements"] if str(r["announcementId"]) == row["document_id"] and r["secCode"] == row["sec_code"]]
                assert matching
                original = matching[0]
                title = re.sub("<[^>]+>", "", original["announcementTitle"])
                stamp = pd.to_datetime(original["announcementTime"], unit="ms", utc=True).tz_convert("Asia/Shanghai")
                assert row["title"] == title and (not keyword or keyword in title)
                assert row["catalogue_timestamp"] == stamp.isoformat() and row["catalogue_date"] == stamp.date().isoformat()
                window = window_map[row["window_id"]]
                assert window["start"] <= row["catalogue_date"] <= window["end"]
                assert row["source_url"] == "https://static.cninfo.com.cn/" + original["adjunctUrl"]
                assert row["announcement_type"] == original.get("announcementType")
                assert row["column_id"] == original.get("columnId")
                assert row["historical_first_publication_verified"] is False and row["trading_feature_admitted"] is False
                checked_occurrences += 1
            for window in windows:
                if window.get("child_windows"):
                    first, second = [window_map[k] for k in window["child_windows"]]
                    assert first["start"] == window["start"] and second["end"] == window["end"]
                    assert pd.Timestamp(first["end"]) + pd.Timedelta(days=1) == pd.Timestamp(second["start"])
                    if window["complete"]:
                        assert first["complete"] and second["complete"]
                        assert window["received"] == first["received"] + second["received"] == window["declared_total"]
                    continue
                selected = [r for r in job["rows"] if r["window_id"] == window["window_id"]]
                if not window["complete"]:
                    assert window["status"] != "COMPLETE_FILTERED_QUERY"
                    continue
                assert window["status"] == "COMPLETE_FILTERED_QUERY"
                count = window["declared_total"]
                assert len(selected) == window["received"] == count
                assert len({(r["symbol"], r["document_id"]) for r in selected}) == count
                rebuilt = []
                for page in range(1, max(1, math.ceil(count / 30)) + 1):
                    base = f"receipts/{window['window_id']}_p{page:03d}"
                    receipt = source_receipt(base + "_a1.json")
                    if receipt["status"] != "HTTP_OK":
                        receipt = source_receipt(base + "_a2.json")
                    assert receipt["status"] == "HTTP_OK"
                    payload = receipt["payload"]
                    assert payload["pageSize"] == "30" and payload["pageNum"] == str(page)
                    assert payload["seDate"] == window["start"] + "~" + window["end"]
                    body = source_body(receipt["raw_path"], receipt["sha256"])
                    assert body["totalAnnouncement"] == count
                    batch = body.get("announcements") or []
                    assert len(batch) <= 30
                    rebuilt.extend((r["secCode"], str(r["announcementId"])) for r in batch)
                assert rebuilt == [(r["sec_code"], r["document_id"]) for r in selected]
                complete_leaves += 1
            all_rows.extend(job["rows"])
            all_windows.extend(windows)
    # 失败尝试、分割前首屏和未使用的响应也有不可变来源回执。
    receipt_files = list((root / "receipts").glob("*.json"))
    for file in receipt_files:
        source_receipt(file.relative_to(root).as_posix())
    expected = pd.DataFrame(all_rows).sort_values(["catalogue_date", "symbol", "document_id", "query"]).reset_index(drop=True)
    actual = pd.read_parquet(root / "catalogue_occurrences.parquet").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    saved_unique = pd.read_parquet(root / "unique_documents.parquet")
    merged = []
    for _, group in expected.groupby(["symbol", "document_id"], sort=True):
        fields = ["org_id", "title", "catalogue_timestamp", "catalogue_date", "source_url", "title_role"]
        row = group.iloc[0].to_dict()
        row.update(query_sources="|".join(sorted(group["query"].unique())), occurrence_count=len(group),
                   catalogue_metadata_conflicts="|".join(c for c in fields if group[c].nunique(dropna=False) != 1),
                   raw_reference_count=group.raw_response_path.nunique())
        merged.append(row)
    pd.testing.assert_frame_equal(saved_unique, pd.DataFrame(merged), check_dtype=False)
    actual_windows = read(root / "query_windows.json")
    assert {w["window_id"]: w for w in actual_windows} == {w["window_id"]: w for w in all_windows}
    result = read(root / "result.json")
    assert result["query_jobs"] == 987 and result["complete_jobs"] == complete_jobs
    assert result["source_occurrences"] == len(expected) == checked_occurrences
    assert result["unique_issuer_documents"] == len(saved_unique)
    assert result["unique_pdf_ids"] == saved_unique.document_id.nunique()
    assert result["title_roles"] == dict(Counter(saved_unique.title_role))
    assert result["not_in_categories_but_title_query"] == int((~saved_unique.query_sources.str.contains("categories")).sum())
    assert result["metadata_conflict_documents"] == saved_unique.catalogue_metadata_conflicts.ne("").sum()
    assert result["full_M06_calendar_established"] is False and result["free_float_denominator_established"] is False
    assert result["T13"] == "NOT_RUN" and result["new_accounts"] == result["new_returns"] == 0
    return {"status": "PASS_SAVED_ISSUANCE_CATALOGUE_RESPONSES_PAGINATION_AND_UNION", "companies": 658,
        "query_jobs": 987, "complete_jobs": complete_jobs, "complete_leaf_windows": complete_leaves,
        "source_occurrences": checked_occurrences, "unique_issuer_documents": len(saved_unique),
        "request_receipts": len(receipt_files), "new_accounts": 0, "network_requests": 0, "external_review": "NOT_PERFORMED",
        "scope": "核对固定检索范围、收到的原响应、完整分页与保留缺口和去重并集；不证明经济事件全集、原文发行日期或策略收益。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    output = verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)
