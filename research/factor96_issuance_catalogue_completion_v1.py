"""保存发行目录清洗修正，并仅补齐首版失败查询；不产生事件事实或交易。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import html
import json
from pathlib import Path
import re
import shutil
import threading
import time

import pandas as pd
import requests

from research.factor96_issuance_catalogue_v1 import QUERIES, classify_title, digest, now, read, save

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/research/510300_factor96_issuance_catalogue_v1"
OUT = ROOT / "reports/research/510300_factor96_issuance_catalogue_completion_v1"
STUDY = "510300_FACTOR96_ISSUANCE_CATALOGUE_COMPLETION_V1"
CAPACITY = 1500
STOP = threading.Event()
LOCK = threading.Lock()
COUNTER = 0


def clean_title(value):
    # 公告标题里的中文尖括号可能是正文，不能用通用标签正则删除。
    return html.unescape(re.sub(r"</?em(?:\s+[^>]*)?>", "", value, flags=re.I))


def normalize(body, companies, query, start, end, capacity=CAPACITY):
    total = body.get("totalAnnouncement")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("声明总量无效")
    batch = body.get("announcements") or []
    if not isinstance(batch, list) or len(batch) > capacity or len(batch) > total or (total > 0 and not batch):
        raise ValueError("单次响应长度无效")
    by_org = {x["org_id"]: x for x in companies}
    assert len(by_org) == len(companies)
    rows, seen = [], set()
    for raw in batch:
        organization = str(raw["orgId"])
        if organization not in by_org:
            raise ValueError("返回组织不在固定发行人范围")
        company = by_org[organization]
        title = clean_title(raw["announcementTitle"])
        keyword = QUERIES[query]["keyword"]
        if keyword and keyword not in title:
            raise ValueError("保留原文尖括号后仍不符合关键词")
        stamp = pd.to_datetime(raw["announcementTime"], unit="ms", utc=True).tz_convert("Asia/Shanghai")
        day = stamp.date().isoformat()
        if not start <= day <= end:
            raise ValueError("目录日期超出窗口")
        adjunct = str(raw["adjunctUrl"])
        if not adjunct.startswith("finalpage/") or ".." in adjunct or not adjunct.lower().endswith(".pdf"):
            raise ValueError("PDF地址无效")
        security = str(raw["secCode"])
        key = (organization, security, str(raw["announcementId"]))
        if key in seen:
            raise ValueError("同次响应重复返回同一组织证券文档")
        seen.add(key)
        rows.append({"symbol": company["symbol"], "query_security_code": company["code"],
            "reported_security_code": security, "org_id": organization,
            "code_relationship": "SAME_STOCK_CODE" if security == company["code"] else "OTHER_CODE_SAME_ORGANIZATION_UNRESOLVED",
            "document_id": str(raw["announcementId"]), "title": title, "raw_title": raw["announcementTitle"],
            "catalogue_timestamp": stamp.isoformat(), "catalogue_date": day,
            "source_url": "https://static.cninfo.com.cn/" + adjunct, "adjunct_url": adjunct,
            "title_role": classify_title(title), "query": query,
            "historical_first_publication_verified": False, "trading_feature_admitted": False})
    return total, rows


def freeze():
    assert (BASE / "result.json").exists(), "必须等首版生产者结束"
    assert not (OUT / "freeze.json").exists()
    assert read(OUT / "prefreeze_test_receipt.json")["exit_code"] == 0
    groups = {g["group_id"]: g for g in read(BASE / "inputs/groups.json")}
    targets = []
    for file in sorted((BASE / "jobs").glob("*.json")):
        job = read(file)
        if not job["complete"]:
            targets.append({"group_id": job["group_id"], "query": job["query"],
                "companies": groups[job["group_id"]]["companies"], "base_job": file.name,
                "base_sha256": digest(file), "base_retained_rows": len(job["rows"]),
                "base_last_full_page_received": job["windows"][0]["received"],
                "base_reasons": [w.get("reason", w["status"]) for w in job["windows"] if not w["complete"]]})
    save(OUT / "targets.json", targets, True)
    for file in [Path(__file__), ROOT / "tests/test_factor96_issuance_catalogue_completion_v1.py"]:
        destination = OUT / "code" / file.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(file, destination)
    protocol = {"at": now(), "study_id": STUDY, "phase": "SOURCE_CATALOGUE_CORRECTION_AND_GAP_COMPLETION_ONLY",
        "base_result_sha256": digest(BASE / "result.json"), "base_freeze_sha256": digest(BASE / "freeze.json"),
        "targets": len(targets), "period": ["2015-01-01", "2025-12-31"],
        "selection": "首版987个固定查询中全部未完成查询；已完成查询只从已存响应重建清洗字段，不重采。",
        "title_correction": "只移除明确em高亮标签并还原HTML实体；保留中文引用文件名使用的尖括号。",
        "issuer_identity": "查询组织ID须完全匹配；原证券代码另存。同组织债券代码和历史代码不自动成为当时A股代码，保留OTHER_CODE_SAME_ORGANIZATION_UNRESOLVED。",
        "single_response_rule": "只请求第一页，容量1500。实际返回数量须等于声明总量且组织证券文档键无重复才算完整；服务端截断则按不交叠日期中点分割，单日仍不齐保留缺口。",
        "transport": "2并发、每请求后0.35秒、连接10秒读取25秒，单请求最多2次且仅传输或5xx重试；最多512次HTTP请求，403/429停止。",
        "partial_window_count_note": "首版received是最后完整解析页累计量，遇重复的当页可能已保存部分新行；同时保留真实保存行数，不覆盖原状态。",
        "historical_membership_boundary": "658个代码是已存成员表取值并集；组织ID仅用于收集，历史证券代码有效期和更名不因当前目录自动获得证明。",
        "new_accounts": 0, "new_returns": 0, "full_M06_calendar_established": False,
        "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "protocol.json", protocol, True)
    files = [{"scope": "local", "path": p.relative_to(OUT).as_posix(), "sha256": digest(p)}
             for p in sorted(OUT.rglob("*")) if p.is_file()]
    for name in ["result.json", "freeze.json", "inputs/groups.json", "catalogue_occurrences.parquet", "unique_documents.parquet"]:
        files.append({"scope": "base", "path": name, "sha256": digest(BASE / name)})
    for p in sorted((BASE / "jobs").glob("*.json")):
        files.append({"scope": "base", "path": p.relative_to(BASE).as_posix(), "sha256": digest(p)})
    save(OUT / "freeze.json", {"at": now(), "files": files}, True)
    print(f"修正合同已冻结：只补{len(targets)}个失败查询，其他查询复用原响应。", flush=True)


def request(session, key, payload):
    global COUNTER
    for attempt in [1, 2]:
        with LOCK:
            if STOP.is_set() or COUNTER >= 512:
                return None, {"status": "NOT_REQUESTED_SOURCE_OR_FIXED_BUDGET_STOP"}
            COUNTER += 1
        receipt = {"requested_at": now(), "url": "https://www.cninfo.com.cn/new/hisAnnouncement/query",
                   "payload": dict(payload), "attempt": attempt}
        raw_path = OUT / "raw" / f"{key}_a{attempt}.json"
        raw_path.parent.mkdir(exist_ok=True)
        content = None
        try:
            response = session.post(receipt["url"], data=payload,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}, timeout=(10, 25))
            content = response.content
            raw_path.write_bytes(content)
            receipt.update(status="HTTP_OK" if response.status_code == 200 else "HTTP_ERROR", http_status=response.status_code,
                raw_path=raw_path.relative_to(OUT).as_posix(), bytes=len(content), sha256=digest(raw_path))
            if response.status_code in [403, 429]:
                STOP.set()
        except requests.RequestException as exc:
            receipt.update(status="REQUEST_FAILED", error_type=type(exc).__name__)
        receipt["completed_at"] = now()
        name = f"receipts/{key}_a{attempt}.json"
        save(OUT / name, receipt, True)
        receipt["receipt_path"] = name
        time.sleep(.35)
        if receipt["status"] != "REQUEST_FAILED" and not 500 <= receipt.get("http_status", 0) <= 599:
            return content, receipt
    return content, receipt


def collect_window(session, target, start, end):
    key = f"{target['group_id']}_{target['query']}_{start.replace('-', '')}_{end.replace('-', '')}"
    state = {"window_id": key, "start": start, "end": end, "complete": False, "status": "NOT_REQUESTED",
             "declared_total": None, "received": 0}
    query = QUERIES[target["query"]]
    payload = {"pageNum": "1", "pageSize": str(CAPACITY), "column": "szse", "tabName": "fulltext", "plate": "",
        "stock": ";".join(c["code"] + "," + c["org_id"] for c in target["companies"]), "searchkey": query["keyword"],
        "secid": "", "category": query["category"], "trade": "", "seDate": start + "~" + end,
        "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
    content, receipt = request(session, key, payload)
    state["request_receipt"] = receipt.get("receipt_path")
    if receipt["status"] != "HTTP_OK":
        state["status"] = receipt["status"]
        return [], [state]
    try:
        total, rows = normalize(json.loads(content), target["companies"], target["query"], start, end)
    except (ValueError, KeyError, TypeError) as exc:
        state.update(status="SOURCE_RESPONSE_NOT_ADMITTED", reason=str(exc))
        return [], [state]
    state.update(declared_total=total, received=len(rows))
    if len(rows) == total:
        state.update(complete=True, status="COMPLETE_SINGLE_RESPONSE")
        for row in rows:
            row.update(window_id=key, source_scope="completion", raw_response_path=receipt["raw_path"],
                       raw_response_sha256=receipt["sha256"], request_receipt=receipt["receipt_path"])
        return rows, [state]
    if start == end:
        state["status"] = "SINGLE_DAY_RESPONSE_NOT_COMPLETE"
        return [], [state]
    lo, hi = date.fromisoformat(start), date.fromisoformat(end)
    middle = lo + timedelta(days=(hi - lo).days // 2)
    a, wa = collect_window(session, target, start, middle.isoformat())
    b, wb = collect_window(session, target, (middle + timedelta(days=1)).isoformat(), end)
    state.update(status="SPLIT_DISJOINT_DATE_WINDOWS", complete=wa[0]["complete"] and wb[0]["complete"],
                 received=len(a) + len(b), child_windows=[wa[0]["window_id"], wb[0]["window_id"]])
    if state["complete"] and state["received"] != total:
        state.update(status="PARENT_CHILD_TOTAL_MISMATCH", complete=False)
    return a + b, [state, *wa, *wb]


def collect_job(target):
    with requests.Session() as session:
        rows, windows = collect_window(session, target, "2015-01-01", "2025-12-31")
    result = {"at": now(), "group_id": target["group_id"], "query": target["query"], "companies": target["companies"],
              "complete": windows[0]["complete"], "windows": windows, "rows": rows}
    save(OUT / "jobs" / target["base_job"], result, True)
    return result


def merge_documents(frame):
    merged = []
    for _, group in frame.groupby(["org_id", "document_id"], sort=True):
        row = group.iloc[0].to_dict()
        fields = ["title", "catalogue_timestamp", "catalogue_date", "source_url", "title_role"]
        row.update(query_sources="|".join(sorted(group["query"].unique())), occurrence_count=len(group),
            reported_security_codes="|".join(sorted(group.reported_security_code.unique())),
            code_relationships="|".join(sorted(group.code_relationship.unique())),
            catalogue_metadata_conflicts="|".join(c for c in fields if group[c].nunique(dropna=False) != 1))
        merged.append(row)
    return pd.DataFrame(merged)


def run():
    assert not (OUT / "run_started.json").exists()
    for item in read(OUT / "freeze.json")["files"]:
        folder = OUT if item["scope"] == "local" else BASE
        assert digest(folder / item["path"]) == item["sha256"], item
    assert digest(Path(__file__)) == digest(OUT / "code" / Path(__file__).name)
    save(OUT / "run_started.json", {"at": now(), "freeze_sha256": digest(OUT / "freeze.json")}, True)
    targets = read(OUT / "targets.json")
    replacements = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(collect_job, target) for target in targets]
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            replacements[(result["group_id"], result["query"])] = result
            print(f"失败窗口补查完成{number}/{len(targets)}，本组完整={result['complete']}。", flush=True)
    rows, jobs, corrected_titles = [], [], []
    raw_cache = {}
    for file in sorted((BASE / "jobs").glob("*.json")):
        old = read(file)
        key = (old["group_id"], old["query"])
        if key in replacements:
            current = replacements[key]
            rows.extend(current["rows"])
            jobs.append({"group_id": key[0], "query": key[1], "complete": current["complete"],
                "source": "completion", "rows": len(current["rows"]), "base_rows": len(old["rows"])})
            continue
        assert old["complete"]
        for previous in old["rows"]:
            raw_name = previous["raw_response_path"]
            if raw_name not in raw_cache:
                body = read(BASE / raw_name)
                receipt = read(BASE / previous["request_receipt"])
                start, end = receipt["payload"]["seDate"].split("~")
                _, decoded = normalize(body, old["companies"], old["query"], start, end, 30)
                raw_cache[raw_name] = {(r["reported_security_code"], r["document_id"]): r for r in decoded}
            row = dict(raw_cache[raw_name][(previous["sec_code"], previous["document_id"])])
            row.update(window_id=previous["window_id"], source_scope="base", raw_response_path=raw_name,
                raw_response_sha256=previous["raw_response_sha256"], request_receipt=previous["request_receipt"])
            rows.append(row)
            if row["title"] != previous["title"]:
                corrected_titles.append({"symbol": row["symbol"], "document_id": row["document_id"], "query": row["query"],
                    "old_title": previous["title"], "corrected_title": row["title"], "raw_response_path": raw_name})
        jobs.append({"group_id": key[0], "query": key[1], "complete": True, "source": "base",
            "rows": len(old["rows"]), "base_rows": len(old["rows"])})
    frame = pd.DataFrame(rows).sort_values(["catalogue_date", "symbol", "document_id", "query", "reported_security_code"])
    frame.to_parquet(OUT / "catalogue_occurrences.parquet", index=False)
    unique = merge_documents(frame)
    unique.to_parquet(OUT / "unique_documents.parquet", index=False)
    save(OUT / "effective_jobs.json", jobs, True)
    save(OUT / "corrected_titles.json", corrected_titles, True)
    pd.crosstab(pd.to_datetime(unique.catalogue_date).dt.year, unique.title_role).to_csv(OUT / "逐年发行目录角色.csv", encoding="utf-8-sig")
    result = {"at": now(), "study_id": STUDY,
        "status": "FIXED_FILTER_CATALOGUE_COMPLETE" if all(j["complete"] for j in jobs) else "FIXED_FILTER_CATALOGUE_WITH_SOURCE_GAPS",
        "companies_in_saved_membership_union": 658, "query_jobs": len(jobs), "base_complete_jobs": sum(j["source"] == "base" for j in jobs),
        "targeted_gap_jobs": len(targets), "completed_gap_jobs": sum(r["complete"] for r in replacements.values()),
        "effective_complete_jobs": sum(j["complete"] for j in jobs), "source_occurrences": len(frame),
        "unique_issuer_documents": len(unique), "unique_pdf_ids": int(unique.document_id.nunique()),
        "corrected_prior_occurrence_titles": len(corrected_titles),
        "other_security_code_documents": int(unique.code_relationships.str.contains("OTHER_CODE").sum()),
        "metadata_conflict_documents": int(unique.catalogue_metadata_conflicts.ne("").sum()),
        "not_in_categories_but_title_query": int((~unique.query_sources.str.contains("categories")).sum()),
        "title_roles": unique.title_role.value_counts().to_dict(), "http_requests": COUNTER,
        "full_M06_calendar_established": False, "free_float_denominator_established": False,
        "historical_code_effective_dates_verified": False, "T13": "NOT_RUN", "new_accounts": 0, "new_returns": 0,
        "goal_status": "active", "goal_achieved": False, "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "run"])
    arguments = parser.parse_args()
    freeze() if arguments.stage == "freeze" else run()
