"""按执行公告给出的历史日期补原始方案，连接逻辑只用公告当时可知资料。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import re
import sys
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest
import research.corporate_repurchase_plan_identity_v1 as identity
import research.corporate_repurchase_index_documents_v1 as downloads
import research.corporate_repurchase_index_catalogue_v1 as catalogue
import research.corporate_repurchase_public_completion_v1 as public

OUT = ROOT / "reports/research/510300_corporate_repurchase_original_plan_completion_v1"
STUDY = "510300_CORPORATE_REPURCHASE_ORIGINAL_PLAN_COMPLETION_V1"


def metadata(row):
    result = identity.metadata(row)
    title = identity.prior.normalized(row["title"])
    text = identity.prior.page_text(read(ROOT / row["text_path"]))
    excluded = any(word in title for word in ["提议", "提案", "董事长", "子公司", "联营", "合营", "限制性", "补偿", "法律意见", "独立", "注销完成", "借款", "贷款", "转让", "尚未", "未实施", "期限届满", "期限过半"])
    excluded = excluded or ("H股" in title and "A股" not in title)
    explicit_plan = "方案" in title or "报告书" in title
    plain_original = (re.search(r"回购(?:公司)?(?:A股)?股份的公告$", title) is not None
        and bool(result["approvals"]) and "回购方案" in text)
    eligible = (row["title_category"] == "PLAN_OR_OTHER_REPURCHASE_DOCUMENT" and not excluded
        and (explicit_plan or plain_original) and ("集中竞价" in text or "二级市场" in text))
    amended = eligible and any(word in title for word in ["增加", "调整", "变更", "延长", "延期", "终止", "更正"])
    result["plan_kind"] = "AMENDMENT" if amended else "ORIGINAL_PLAN_DOCUMENT" if eligible else "NOT_A_PLAN_NODE"
    return result


def freeze():
    for folder in ["code", "raw", "receipts", "results", "documents", "text"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    links = read(identity.OUT / "results/execution_identity_links.json")
    metas = {r["document_id"]: r for r in read(identity.OUT / "results/document_metadata.json")}
    directory = {r["code"]: r for r in read(public.OUT / "cninfo_stock_directory.json")["stockList"] if r["category"] == "A股"}
    requests_by_key = {}
    for row in links:
        if row["identity"]["status"] == "LINKED_ASOF_ORIGINAL_PLAN":
            continue
        first = row["explicit_first_publication"]
        if first is not None:
            anchors = [(pd.Timestamp(first), "EXPLICIT_FIRST_PUBLICATION", 1)]
        else:
            approvals = metas[row["document_id"]]["approvals"]
            original = [a for a in approvals if a["action"] == "ORIGINAL"]
            anchors = [(pd.Timestamp(a["date"]), "ORIGINAL_APPROVAL", 7) for a in original]
        for day, method, days_after in anchors:
            # 本轮只补采集窗口以前的原方案；窗口内身份问题继续用已有全文解决。
            if day >= pd.Timestamp(catalogue.START):
                continue
            key = row["symbol"] + "_" + day.strftime("%Y%m%d")
            if key not in requests_by_key:
                company = directory[row["symbol"][:6]]
                requests_by_key[key] = {"key": key, "symbol": row["symbol"], "code": row["symbol"][:6],
                    "org_id": company["orgId"], "column": "sse" if row["symbol"].endswith(".SH") else "szse",
                    "anchor": day, "method": method, "start": (day - pd.Timedelta(days=1)).date().isoformat(),
                    "end": (day + pd.Timedelta(days=days_after)).date().isoformat(), "referencing_execution_ids": []}
            requests_by_key[key]["referencing_execution_ids"].append(row["document_id"])
    targets = sorted(requests_by_key.values(), key=lambda r: r["key"])
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "action": "只按未解执行公告明确给出的窗口前日期，补查原始回购方案；不根据后续价格或账户结果选择公司。",
        "targets": targets, "query_windows": len(targets),
        "rules": "首次披露日前后1日；仅有明确批准日则前1日至后7日。每窗口最多3页，每页30条；查询无命中不解释为没有回购。原文下载最多两次传输尝试。",
        "metadata_correction": "用于注销的二级市场回购仍属于原方案；未实施进展、纯注销、补偿、H股等分别处理。标准标题未写方案但正文明确原方案批准的公告可以作为方案文件。",
        "source_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in [identity.OUT / "results/execution_identity_links.json", identity.OUT / "results/document_metadata.json", downloads.OUT / "documents.json"]},
        "code_sha256": digest(Path(__file__)), "new_accounts": 0, "new_fits": 0}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(f"已固定{len(targets)}个历史日期窗口，仅补原方案身份资料。", flush=True)


def query(window):
    destination = OUT / "results" / (window["key"] + "_catalogue.json")
    if destination.exists():
        return read(destination)
    rows, total = [], None
    status = "PARTIAL_QUERY"
    with requests.Session() as session:
        for page in range(1, 4):
            key = window["key"] + f"_page{page}"
            payload = {"pageNum": str(page), "pageSize": "30", "column": window["column"], "tabName": "fulltext",
                "stock": window["code"] + "," + window["org_id"], "searchkey": "回购", "secid": "", "category": "", "trade": "",
                "seDate": window["start"] + "~" + window["end"], "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
            body = None
            for attempt in [1, 2]:
                receipt_path = OUT / "receipts" / f"{key}_{attempt}.json"
                if receipt_path.exists():
                    receipt = read(receipt_path)
                elif downloads.STOP.is_set():
                    status = "NOT_REQUESTED_AFTER_SOURCE_LIMIT"
                    break
                else:
                    receipt = {"requested_at": now(), "source_url": public.QUERY_URL, "payload": payload}
                    try:
                        response = session.post(public.QUERY_URL, data=payload, headers=public.HEADERS, timeout=(10, 25))
                        path = OUT / "raw" / f"{key}_{attempt}.json"
                        path.write_bytes(response.content)
                        receipt.update(status="HTTP_OK" if response.status_code == 200 else "HTTP_ERROR", http_status=response.status_code,
                            raw_path=path.relative_to(ROOT).as_posix(), sha256=digest(path))
                        if response.status_code in [403, 429]:
                            downloads.STOP.set()
                    except requests.RequestException as exc:
                        receipt.update(status="REQUEST_FAILED", error_type=type(exc).__name__)
                    receipt["completed_at"] = now()
                    save(receipt_path, receipt, True)
                    time.sleep(0.3)
                if receipt["status"] == "HTTP_OK":
                    body = json.loads((ROOT / receipt["raw_path"]).read_bytes())
                    break
                status = receipt["status"]
                if receipt["status"] != "REQUEST_FAILED":
                    break
            if body is None:
                break
            count = int(body["totalAnnouncement"])
            if total is not None and count != total:
                status = "QUERY_TOTAL_CHANGED"
                break
            total = count
            batch = body.get("announcements") or []
            for entry in batch:
                assert entry["secCode"] == window["code"] and entry["orgId"] == window["org_id"]
                title = re.sub("<[^>]+>", "", entry["announcementTitle"])
                stamp = pd.to_datetime(entry["announcementTime"], unit="ms", utc=True).tz_convert("Asia/Shanghai")
                assert pd.Timestamp(window["start"]).date() <= stamp.date() <= pd.Timestamp(window["end"]).date()
                assert "回购" in title and entry["adjunctUrl"].startswith("finalpage/")
                rows.append({"symbol": window["symbol"], "document_id": str(entry["announcementId"]), "title": title,
                    "catalogue_date": stamp.date().isoformat(), "catalogue_timestamp": stamp,
                    "source_url": "https://static.cninfo.com.cn/" + entry["adjunctUrl"],
                    "title_category": catalogue.catalogue_kind(title), "identity_source_only": True})
            if len({r["document_id"] for r in rows}) != len(rows):
                status = "DUPLICATE_PAGINATION_ROWS"
                break
            if len(rows) == total:
                status = "COMPLETE_QUERY"
                break
            if not batch:
                status = "MISSING_QUERY_PAGE"
                break
    result = {"window": window, "status": status, "total": total, "rows": rows}
    save(destination, result, True)
    return result


def run():
    protocol = read(OUT / "protocol.json")
    assert digest(Path(__file__)) == protocol["code_sha256"]
    for name, sha in protocol["source_hashes"].items():
        assert digest(ROOT / name) == sha
    catalogues = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(query, window) for window in protocol["targets"]]
        for future in as_completed(futures):
            catalogues.append(future.result())
            if len(catalogues) % 15 == 0:
                print(f"历史方案日期窗口已查{len(catalogues)}/{len(protocol['targets'])}。", flush=True)
    save(OUT / "results/catalogues.json", catalogues, True)
    all_rows = {r["document_id"]: r for c in catalogues for r in c["rows"]}
    old = {r["document_id"]: r for r in read(downloads.OUT / "documents.json")}
    # 只下载可能提供原方案身份的文档；进展与持股名单没有必要再次下载。
    targets = [r for r in all_rows.values() if r["title_category"] == "PLAN_OR_OTHER_REPURCHASE_DOCUMENT"]
    results, pending = [], []
    for row in targets:
        destination = OUT / "documents" / (row["document_id"] + ".json")
        if destination.exists():
            results.append(read(destination))
        elif row["document_id"] in old:
            results.append(old[row["document_id"]])
        else:
            pending.append(row)
    previous_out = downloads.OUT
    try:
        downloads.OUT = OUT
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(downloads.request_pdf, row) for row in pending]
            for future in as_completed(futures):
                record = downloads.extract_text(future.result())
                save(OUT / "documents" / (record["document_id"] + ".json"), record, True)
                results.append(record)
                if len(results) % 15 == 0:
                    print(f"历史方案原文已取得{len(results)}/{len(targets)}份。", flush=True)
    finally:
        downloads.OUT = previous_out
    save(OUT / "results/supplement_documents.json", results, True)
    combined = {**old, **{r["document_id"]: r for r in results if r["status"] == "PDF_TEXT_SAVED"}}
    metas = [metadata(row) for row in combined.values()]
    nodes, excluded = identity.build_nodes(metas)
    indexed = {m["document_id"]: m for m in metas}
    old_links = read(identity.OUT / "results/execution_identity_links.json")
    links = []
    for row in old_links:
        meta = indexed[row["document_id"]]
        resolved = identity.resolve(meta, nodes)
        links.append({**row, "identity": resolved, "prior_identity_status": row["identity"]["status"]})
    save(OUT / "results/document_metadata.json", metas, True)
    save(OUT / "results/plan_nodes.json", nodes, True)
    save(OUT / "results/unresolved_plan_nodes.json", excluded, True)
    save(OUT / "results/execution_identity_links.json", links, True)
    status_counts = pd.Series([r["identity"]["status"] for r in links]).value_counts().to_dict()
    method_counts = pd.Series([r["identity"].get("method", "UNRESOLVED") for r in links]).value_counts().to_dict()
    result = {"at": now(), "study_id": STUDY, "status": "HISTORICAL_PLAN_SOURCES_AND_IDENTITY_COMPLETED_GAPS_RETAINED",
        "query_windows": len(catalogues), "complete_query_windows": sum(c["status"] == "COMPLETE_QUERY" for c in catalogues),
        "queried_documents": len(all_rows), "selected_supplement_documents": len(results),
        "complete_supplement_texts": sum(r["status"] == "PDF_TEXT_SAVED" for r in results),
        "plan_nodes": len(nodes), "original_plan_roots": len({n['root_id'] for n in nodes}),
        "unresolved_plan_nodes": len(excluded), "execution_records": len(links),
        "identity_status_counts": status_counts, "identity_method_counts": method_counts,
        "new_accounts": 0, "new_fits": 0, "independent_forward_observations": 0, "current_market_view": "NO_VIEW",
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)
    print({"原方案补充完成": True, "原文数": len(results), "执行身份状态": status_counts}, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "freeze":
        freeze()
    else:
        run()
