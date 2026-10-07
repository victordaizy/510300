"""仅补上一轮预算结束后留下的日期叶窗口，并合并已存完整来源。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import json
from pathlib import Path
import shutil
import threading
import time

import pandas as pd
import requests

from research.factor96_issuance_catalogue_v1 import QUERIES, digest, now, read, save
from research.factor96_issuance_catalogue_completion_v1 import merge_documents, normalize

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "reports/research/510300_factor96_issuance_catalogue_completion_v1"
BASE = ROOT / "reports/research/510300_factor96_issuance_catalogue_v1"
OUT = ROOT / "reports/research/510300_factor96_issuance_remainder_v1"
STUDY = "510300_FACTOR96_ISSUANCE_REMAINDER_V1"
HTTP_LIMIT = 768
COUNTER = 0
LOCK = threading.Lock()
STOP = threading.Event()


def reconcile(window_id, old_windows, replacements):
    original = old_windows[window_id]
    if window_id in replacements:
        replacement = replacements[window_id]
        top = replacement["windows"][0]
        assert top["window_id"] == window_id
        assert (top["start"], top["end"]) == (original["start"], original["end"])
        return dict(top), [dict(w, provenance="new_remainder") for w in replacement["windows"]]
    current = dict(original, provenance="previous_completion")
    if not original.get("child_windows"):
        return current, [current]
    first, first_rows = reconcile(original["child_windows"][0], old_windows, replacements)
    second, second_rows = reconcile(original["child_windows"][1], old_windows, replacements)
    assert first["start"] == original["start"] and second["end"] == original["end"]
    assert date.fromisoformat(first["end"]) + timedelta(days=1) == date.fromisoformat(second["start"])
    current.update(received=first["received"] + second["received"],
        complete=first["complete"] and second["complete"], status="COMPOSITE_DISJOINT_WINDOWS")
    if current["complete"] and current["declared_total"] is not None and current["received"] != current["declared_total"]:
        current.update(complete=False, status="OLD_PARENT_NEW_CHILD_TOTAL_MISMATCH")
    return current, [current, *first_rows, *second_rows]


def freeze():
    assert not (OUT / "freeze.json").exists()
    assert read(OUT / "prefreeze_test_receipt.json")["exit_code"] == 0
    assert read(PRIOR / "saved_verification_receipt.json")["status"] == "PASS_SAVED_ISSUANCE_TITLE_IDENTITY_AND_TARGETED_COMPLETION"
    copies = {
        "inputs/prior_occurrences.parquet": PRIOR / "catalogue_occurrences.parquet",
        "inputs/prior_unique_documents.parquet": PRIOR / "unique_documents.parquet",
        "inputs/prior_effective_jobs.json": PRIOR / "effective_jobs.json",
        "inputs/prior_result.json": PRIOR / "result.json",
        "inputs/prior_saved_verification.json": PRIOR / "saved_verification_receipt.json",
        "inputs/base_unique_documents.parquet": BASE / "unique_documents.parquet",
        "inputs/groups.json": BASE / "inputs/groups.json",
        "inputs/prior_delivery_receipt.json": ROOT / "reports/research/510300_factor96_issuance_repurchase_sources_v1/delivery_receipt.json",
        "inputs/current_mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
    }
    for name, source in copies.items():
        destination = OUT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    targets = []
    for state in read(PRIOR / "effective_jobs.json"):
        if state["complete"]:
            continue
        filename = f"{state['group_id']}_{state['query']}.json"
        original = read(PRIOR / "jobs" / filename)
        destination = OUT / "inputs/prior_jobs" / filename
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(PRIOR / "jobs" / filename, destination)
        for window in original["windows"]:
            if not window["complete"] and not window.get("child_windows"):
                assert window["status"] == "NOT_REQUESTED_SOURCE_OR_FIXED_BUDGET_STOP", window
                targets.append({"window_id": window["window_id"], "start": window["start"], "end": window["end"],
                    "group_id": state["group_id"], "query": state["query"], "companies": original["companies"],
                    "prior_job": filename, "prior_window_status": window["status"]})
    assert len({t["prior_job"] for t in targets}) == 16
    save(OUT / "targets.json", targets, True)
    for relative in ["research/factor96_issuance_remainder_v1.py", "research/factor96_issuance_catalogue_v1.py",
        "research/factor96_issuance_catalogue_completion_v1.py", "tests/test_factor96_issuance_remainder_v1.py"]:
        destination = OUT / "code" / Path(relative).name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "phase": "REMAINING_SOURCE_WINDOWS_ONLY",
        "previous_turn_classification": "PROGRESS_SOURCE_FIELDS_AND_ARCHIVE_COMPLETED",
        "selection": "上一轮全部16未完成查询中的全部未请求叶窗口；已完整窗口和971个完整查询均复用，原512次请求上限和结果不改写。",
        "target_queries": 16, "target_leaf_windows": len(targets), "maximum_new_http_requests": HTTP_LIMIT,
        "paging": "请求容量30，仅取第一页；声明总量大于实收就按不交叠日期二分，单日仍不全保留缺口，拒绝跨页重复排序。",
        "clock": "目录时钟、下载时钟与经济日期分开；首次历史HTTP可得仍未证明。",
        "transport": "2并发，每请求后0.35秒，连接10秒读取25秒；仅传输或5xx最多2次，403/429停止新请求。",
        "union": "前一版全部91043条收到记录原样继承，补入本轮互斥日期窗口；按组织及文档ID并集，保留证券代码差异与元数据冲突。",
        "boundaries": "完整只指原三个固定检索器；不是经济事件全集、收益验证或自由流通分母。",
        "new_accounts": 0, "new_returns": 0, "new_models": 0, "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "external_review": "NOT_PERFORMED"}, True)
    save(OUT / "freeze.json", {"at": now(), "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)}
        for p in sorted(OUT.rglob("*")) if p.is_file()]}, True)
    print(f"已冻结16组查询的{len(targets)}个未完成日期窗口。", flush=True)


def request(session, key, payload):
    global COUNTER
    for attempt in [1, 2]:
        with LOCK:
            if STOP.is_set() or COUNTER >= HTTP_LIMIT:
                return None, {"status": "NOT_REQUESTED_FIXED_BUDGET_OR_SOURCE_STOP"}
            COUNTER += 1
        receipt = {"requested_at": now(), "url": "https://www.cninfo.com.cn/new/hisAnnouncement/query", "payload": dict(payload), "attempt": attempt}
        content = None
        try:
            response = session.post(receipt["url"], data=payload,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}, timeout=(10, 25))
            content = response.content
            path = OUT / "raw" / f"{key}_a{attempt}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(content)
            receipt.update(status="HTTP_OK" if response.status_code == 200 else "HTTP_ERROR", http_status=response.status_code,
                raw_path=path.relative_to(OUT).as_posix(), sha256=digest(path), bytes=len(content))
            if response.status_code in [403, 429]:
                STOP.set()
        except requests.RequestException as error:
            receipt.update(status="REQUEST_FAILED", error_type=type(error).__name__)
        receipt["completed_at"] = now()
        name = f"receipts/{key}_a{attempt}.json"
        save(OUT / name, receipt, True)
        receipt["receipt_path"] = name
        time.sleep(.35)
        if receipt["status"] != "REQUEST_FAILED" and not 500 <= receipt.get("http_status", 0) <= 599:
            return content, receipt
    return content, receipt


def collect(session, target, start, end):
    key = f"{target['group_id']}_{target['query']}_{start.replace('-', '')}_{end.replace('-', '')}"
    state = {"window_id": key, "start": start, "end": end, "complete": False, "declared_total": None, "received": 0}
    query = QUERIES[target["query"]]
    payload = {"pageNum": "1", "pageSize": "30", "column": "szse", "tabName": "fulltext", "plate": "",
        "stock": ";".join(c["code"] + "," + c["org_id"] for c in target["companies"]), "searchkey": query["keyword"],
        "secid": "", "category": query["category"], "trade": "", "seDate": start + "~" + end,
        "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
    content, receipt = request(session, key, payload)
    state["request_receipt"] = receipt.get("receipt_path")
    if receipt["status"] != "HTTP_OK":
        state["status"] = receipt["status"]
        return [], [state]
    try:
        total, rows = normalize(json.loads(content), target["companies"], target["query"], start, end, 30)
    except (ValueError, TypeError, KeyError) as error:
        state.update(status="SOURCE_RESPONSE_NOT_ADMITTED", reason=str(error))
        return [], [state]
    state.update(declared_total=total, received=len(rows))
    if total == len(rows):
        state.update(complete=True, status="COMPLETE_SINGLE_RESPONSE")
        for row in rows:
            row.update(window_id=key, source_scope="remainder", raw_response_path=receipt["raw_path"],
                raw_response_sha256=receipt["sha256"], request_receipt=receipt["receipt_path"])
        return rows, [state]
    if start == end:
        state["status"] = "SINGLE_DAY_NOT_COMPLETE"
        return [], [state]
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    middle = first + timedelta(days=(last - first).days // 2)
    left, lw = collect(session, target, start, middle.isoformat())
    right, rw = collect(session, target, (middle + timedelta(days=1)).isoformat(), end)
    state.update(complete=lw[0]["complete"] and rw[0]["complete"], received=len(left) + len(right),
        status="SPLIT_DISJOINT_DATE_WINDOWS", child_windows=[lw[0]["window_id"], rw[0]["window_id"]])
    if state["complete"] and state["received"] != total:
        state.update(complete=False, status="PARENT_CHILD_TOTAL_MISMATCH")
    return left + right, [state, *lw, *rw]


def collect_target(target):
    with requests.Session() as session:
        rows, windows = collect(session, target, target["start"], target["end"])
    result = {"at": now(), "target": target, "complete": windows[0]["complete"], "windows": windows, "rows": rows}
    save(OUT / "jobs" / (target["window_id"] + ".json"), result, True)
    return result


def run():
    assert not (OUT / "run_started.json").exists()
    for item in read(OUT / "freeze.json")["files"]:
        assert digest(OUT / item["path"]) == item["sha256"]
    for relative in ["research/factor96_issuance_remainder_v1.py", "research/factor96_issuance_catalogue_v1.py", "research/factor96_issuance_catalogue_completion_v1.py"]:
        assert digest(ROOT / relative) == digest(OUT / "code" / Path(relative).name)
    save(OUT / "run_started.json", {"at": now(), "freeze_sha256": digest(OUT / "freeze.json")}, True)
    targets = read(OUT / "targets.json")
    replacements, new_rows = {}, []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(collect_target, target) for target in targets]
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            replacements[result["target"]["window_id"]] = result
            new_rows.extend(result["rows"])
            print(f"剩余日期窗口完成{number}/{len(targets)}，本窗口完整={result['complete']}。", flush=True)
    previous = pd.read_parquet(OUT / "inputs/prior_occurrences.parquet")
    frame = pd.concat([previous, pd.DataFrame(new_rows)], ignore_index=True).sort_values(
        ["catalogue_date", "symbol", "document_id", "query", "reported_security_code"])
    assert not frame.duplicated(["org_id", "document_id", "query", "reported_security_code"]).any(), "日期窗口出现重复，不能合并为完整"
    frame.to_parquet(OUT / "catalogue_occurrences.parquet", index=False)
    unique = merge_documents(frame)
    unique.to_parquet(OUT / "unique_documents.parquet", index=False)
    jobs = read(OUT / "inputs/prior_effective_jobs.json")
    completed_before = sum(job["complete"] for job in jobs)
    for job in jobs:
        if job["complete"]:
            continue
        filename = f"{job['group_id']}_{job['query']}.json"
        old = read(OUT / "inputs/prior_jobs" / filename)
        windows = {w["window_id"]: w for w in old["windows"]}
        top, reconciled = reconcile(old["windows"][0]["window_id"], windows, replacements)
        rows = frame[(frame.window_id.str.startswith(job["group_id"] + "_")) & (frame["query"] == job["query"])]
        if top["complete"]:
            assert len(rows) == top["received"] == top["declared_total"]
        job.update(complete=top["complete"], source="prior_and_remainder", rows=len(rows))
        save(OUT / "reconciled_queries" / filename, {"summary": job, "windows": reconciled}, True)
    save(OUT / "effective_jobs.json", jobs, True)
    original = pd.read_parquet(OUT / "inputs/base_unique_documents.parquet")
    before_keys = set(zip(original.org_id, original.document_id))
    after_keys = set(zip(unique.org_id, unique.document_id))
    restored = original[[key in after_keys for key in zip(original.org_id, original.document_id)]]
    restored[["org_id", "document_id"]].to_csv(OUT / "首版文档覆盖核对.csv", index=False, encoding="utf-8-sig")
    result = {"at": now(), "study_id": STUDY,
        "status": "FIXED_FILTER_CATALOGUE_COMPLETE" if all(j["complete"] for j in jobs) else "FIXED_FILTER_CATALOGUE_WITH_GAPS",
        "companies_in_saved_membership_union": 658, "query_jobs": len(jobs), "complete_queries_before": completed_before,
        "effective_complete_jobs": sum(j["complete"] for j in jobs), "new_source_occurrences": len(new_rows),
        "source_occurrences": len(frame), "unique_issuer_documents": len(unique), "http_requests": COUNTER,
        "target_windows": len(targets), "completed_target_windows": sum(r["complete"] for r in replacements.values()),
        "base_document_ids_not_covered": len(before_keys - after_keys), "metadata_conflict_documents": int(unique.catalogue_metadata_conflicts.ne("").sum()),
        "title_roles": unique.title_role.value_counts().to_dict(), "full_M06_calendar_established": False,
        "free_float_denominator_established": False, "T13": "NOT_RUN", "new_accounts": 0, "new_returns": 0, "new_models": 0,
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False, "external_review": "NOT_PERFORMED"}
    save(OUT / "result.json", result, True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "run"])
    arguments = parser.parse_args()
    freeze() if arguments.stage == "freeze" else run()
