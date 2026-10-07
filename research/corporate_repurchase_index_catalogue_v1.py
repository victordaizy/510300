"""按历史成分并集收集回购公告目录，为指数实际需求研究确定覆盖边界。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import re
import sys
import threading
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest
import research.corporate_repurchase_public_completion_v1 as pilot

OUT = ROOT / "reports/research/510300_corporate_repurchase_index_catalogue_v1"
STUDY = "510300_CORPORATE_REPURCHASE_INDEX_CATALOGUE_V1"
WEIGHTS = ROOT / "data/raw/constituents/000300_historical_weights.parquet"
START, END = "2024-09-01", "2026-09-24"
MAX_PAGES, PAGE_SIZE = 10, 30
STOP = threading.Event()


def catalogue_kind(title):
    """标题只用于安排原文核实，不直接生成回购金额。"""
    if "限制性股票" in title:
        return "RESTRICTED_SHARE_ADMINISTRATION"
    if "股东持股" in title or "持股情况" in title:
        return "HOLDER_LIST"
    if "债权人" in title:
        return "CREDITOR_NOTICE"
    if "价格上限" in title or "回购价格" in title:
        return "PRICE_AMENDMENT"
    if any(v in title for v in ["法律意见", "独立意见", "核查意见", "独立董事意见"]):
        return "LEGAL_OR_REVIEW_DOCUMENT"
    if "回购" in title and any(v in title for v in ["首次", "进展", "结果", "完成"]):
        return "EXECUTION_DISCLOSURE_CANDIDATE"
    return "PLAN_OR_OTHER_REPURCHASE_DOCUMENT"


def prior_snapshot(weights, timestamp):
    # 回购发生后纳入指数的公司，可以被收集，但不能提前视为指数成分。
    days = weights.trade_date.unique()
    days = days[days < pd.Timestamp(timestamp).tz_localize(None).normalize().to_datetime64()]
    return pd.Timestamp(max(days)) if len(days) else None


def freeze():
    for name in ["code", "raw", "receipts", "results"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    weights = pd.read_parquet(WEIGHTS)
    anchor = weights.loc[weights.trade_date.lt(START), "trade_date"].max()
    selected = weights[weights.trade_date.between(anchor, END)].copy()
    assert selected.groupby("trade_date").size().eq(300).all()
    assert not selected.duplicated(["trade_date", "con_code"]).any()
    directory_path = pilot.OUT / "cninfo_stock_directory.json"
    directory = read(directory_path)["stockList"]
    mapping = {}
    for row in directory:
        if row["category"] == "A股":
            if row["code"] in mapping:
                assert mapping[row["code"]]["orgId"] == row["orgId"]
            mapping[row["code"]] = row
    companies = []
    for symbol in sorted(selected.con_code.unique()):
        code = symbol[:6]
        group = selected[selected.con_code.eq(symbol)]
        record = mapping.get(code)
        companies.append({"symbol": symbol, "code": code,
            "org_id": record["orgId"] if record else None,
            "name": record["zwjc"] if record else None,
            "column": "sse" if symbol.endswith(".SH") else "szse",
            "first_reference_snapshot": group.trade_date.min(),
            "last_reference_snapshot": group.trade_date.max(),
            "reference_snapshot_count": group.trade_date.nunique()})
    protocol = {"at": now(), "study_id": STUDY, "period": [START, END],
        "question": "从两家公司扩展到历史成分并集，确认实际回购披露的覆盖和后续原文工作量；目录本身不作为买盘信号。",
        "universe_rule": "2024-09-01前最近权重快照及此后已保存月度快照的全部公司并集，只作采集范围；每份公告另标记其日期之前最近快照中的身份，不把后纳入公司提前加入指数。",
        "anchor_snapshot": anchor, "last_snapshot": selected.trade_date.max(),
        "reference_snapshots": selected.trade_date.nunique(), "universe_companies": len(companies),
        "companies": companies, "source": pilot.QUERY_URL,
        "query": "按公司及机构ID检索标题含回购的公告，30条一页，最多10页；必须数量、唯一ID和范围同时核对。",
        "empty_rule": "正确公司查询且明确totalAnnouncement=0仅表示标题查询无命中，不证明实际回购为零。",
        "reuse": "原两家公司同区间的完整目录直接复用，不重新请求。",
        "concurrency": 3, "minimum_seconds_between_each_worker_requests": 0.5,
        "rate_limit_rule": "遇到HTTP403或429停止发出新的请求，保留未取得和部分取得状态。",
        "membership_limit": "月度历史供应商快照的首发时间未认证；公告日之前最近保存快照只作历史参考，不冒充每日正式生效名单。末快照以后身份覆盖单独标记。",
        "title_classification_limit": "标题候选必须继续读发行人原文；进展与完成不直接计入实际金额，未覆盖资料不得填零。",
        "new_accounts": 0, "new_fits": 0, "new_returns_loaded": False,
        "goal_achieved": False, "orders_authorized": False,
        "source_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in
            [WEIGHTS, directory_path, pilot.OUT / "results/catalogue.parquet", pilot.OUT / "results/catalogue_status.json"]},
        "code_sha256": digest(Path(__file__))}
    save(OUT / "protocol.json", protocol, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    selected.to_parquet(OUT / "results/reference_weights.parquet", index=False)
    print(f"已固定历史范围：{len(companies)}家公司、{selected.trade_date.nunique()}个月度快照；只收目录，不读策略收益。", flush=True)


def request_page(session, company, page):
    key = f"{company['symbol']}_{page:02d}"
    destination = OUT / "receipts" / (key + ".json")
    if destination.exists():
        receipt = read(destination)
        content = (OUT / receipt["raw_path"]).read_bytes() if receipt.get("raw_path") else None
        return content, receipt
    if STOP.is_set():
        return None, {"status": "NOT_REQUESTED_AFTER_SOURCE_LIMIT"}
    payload = {"pageNum": str(page), "pageSize": str(PAGE_SIZE), "column": company["column"],
        "tabName": "fulltext", "stock": company["code"] + "," + company["org_id"],
        "searchkey": "回购", "secid": "", "category": "", "trade": "",
        "seDate": START + "~" + END, "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
    receipt = {"requested_at": now(), "source_url": pilot.QUERY_URL, "payload": payload}
    content = None
    try:
        response = session.post(pilot.QUERY_URL, data=payload, headers=pilot.HEADERS, timeout=(10, 25))
        content = response.content
        path = OUT / "raw" / (key + ".json")
        path.write_bytes(content)
        receipt.update(http_status=response.status_code, raw_path=path.relative_to(OUT).as_posix(),
            sha256=digest(path), bytes=len(content), status="HTTP_OK" if response.status_code == 200 else "HTTP_ERROR")
        if response.status_code in [403, 429]:
            STOP.set()
    except requests.RequestException as exc:
        receipt.update(status="REQUEST_FAILED", error_type=type(exc).__name__)
    receipt["completed_at"] = now()
    save(destination, receipt, True)
    time.sleep(0.5)
    return content, receipt


def collect_company(company):
    target = OUT / "results" / (company["symbol"] + ".json")
    if target.exists():
        return read(target)
    result = {"symbol": company["symbol"], "name": company["name"], "complete": False,
        "status": "NOT_REQUESTED", "total": None, "rows": [], "reused": False}
    if not company["org_id"]:
        result["status"] = "NO_ORGANIZATION_MAPPING"
    elif company["symbol"] in ["600519.SH", "300750.SZ"]:
        rows = pd.read_parquet(pilot.OUT / "results/catalogue.parquet")
        rows = rows[rows.symbol.eq(company["symbol"])].to_dict("records")
        original = next(r for r in read(pilot.OUT / "results/catalogue_status.json") if r["code"] == company["code"])
        assert original["complete"] and len(rows) == original["total"]
        result.update(status="COMPLETE_TITLE_QUERY", complete=True, total=len(rows), rows=rows, reused=True)
    else:
        with requests.Session() as session:
            for page in range(1, MAX_PAGES + 1):
                content, receipt = request_page(session, company, page)
                if receipt["status"] != "HTTP_OK":
                    result["status"] = receipt["status"]
                    break
                try:
                    body = json.loads(content)
                    count = int(body["totalAnnouncement"])
                    if count < 0 or (result["total"] is not None and result["total"] != count):
                        raise ValueError("分页总量异常或快照变化")
                    result["total"] = count
                    batch = body.get("announcements") or []
                    if not batch and count:
                        raise ValueError("总量非零但分页为空")
                    parsed = []
                    for row in batch:
                        if row["secCode"] != company["code"] or row["orgId"] != company["org_id"]:
                            raise ValueError("返回公司身份与查询不一致")
                        title = re.sub("<[^>]+>", "", row["announcementTitle"])
                        stamp = pd.to_datetime(row["announcementTime"], unit="ms", utc=True).tz_convert("Asia/Shanghai")
                        if "回购" not in title or not (pd.Timestamp(START).date() <= stamp.date() <= pd.Timestamp(END).date()):
                            raise ValueError("返回标题或日期不符合查询")
                        path = row["adjunctUrl"]
                        if not path.startswith("finalpage/") or ".." in path:
                            raise ValueError("原文路径格式异常")
                        parsed.append({"symbol": company["symbol"], "document_id": str(row["announcementId"]),
                            "title": title, "catalogue_timestamp": stamp, "catalogue_date": stamp.date().isoformat(),
                            "source_url": "https://static.cninfo.com.cn/" + path,
                            "official_pdf_path": path, "query_page": page})
                    result["rows"].extend(parsed)
                    unique = len({r["document_id"] for r in result["rows"]})
                    if unique != len(result["rows"]) or unique > count:
                        raise ValueError("公告ID重复或超出查询总量")
                    if unique == count:
                        result.update(status="COMPLETE_TITLE_QUERY", complete=True)
                        break
                    result["status"] = "PARTIAL_PAGE_LIMIT"
                except (ValueError, KeyError, TypeError) as exc:
                    result.update(status="RESPONSE_VALIDATION_FAILED", error=str(exc))
                    break
    result["completed_at"] = now()
    save(target, result, True)
    return result


def run():
    protocol = read(OUT / "protocol.json")
    assert digest(Path(__file__)) == protocol["code_sha256"]
    for path, sha in protocol["source_hashes"].items():
        assert digest(ROOT / path) == sha, path
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(collect_company, c) for c in protocol["companies"]]
        for done in as_completed(futures):
            result = done.result()
            results.append(result)
            if len(results) % 15 == 0:
                print(f"目录核实{len(results)}/{len(futures)}家公司，完整{sum(r['complete'] for r in results)}家，已取得{sum(len(r['rows']) for r in results)}条。", flush=True)
    weights = pd.read_parquet(OUT / "results/reference_weights.parquet")
    snapshots = {pd.Timestamp(day): dict(zip(group.con_code, group.weight)) for day, group in weights.groupby("trade_date")}
    rows, statuses = [], []
    for result in sorted(results, key=lambda r: r["symbol"]):
        statuses.append({k: v for k, v in result.items() if k != "rows"})
        for entry in result["rows"]:
            record = dict(entry)
            day = pd.Timestamp(record["catalogue_date"])
            snapshot = prior_snapshot(weights, day)
            present = snapshot is not None and record["symbol"] in snapshots[snapshot]
            record.update(title_category=catalogue_kind(record["title"]), reference_snapshot=snapshot,
                in_latest_prior_saved_snapshot=present,
                reference_weight_percent=snapshots[snapshot].get(record["symbol"]) if snapshot is not None else None,
                snapshot_age_days=(day - snapshot).days if snapshot is not None else None,
                after_last_saved_snapshot=day > weights.trade_date.max(),
                company_catalogue_complete=result["complete"], amount_parsed=False)
            rows.append(record)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "results/catalogue.parquet", index=False)
    save(OUT / "results/company_status.json", statuses, True)
    result = {"at": now(), "study_id": STUDY,
        "status": "COMPLETE_INDEX_UNION_TITLE_CATALOGUE_FACTS_PENDING" if all(r["complete"] for r in results) else "PARTIAL_INDEX_UNION_TITLE_CATALOGUE_FACTS_PENDING",
        "companies": len(results), "complete_companies": sum(r["complete"] for r in results),
        "companies_without_title_matches": sum(r["complete"] and r["total"] == 0 for r in results),
        "reused_companies": sum(r["reused"] for r in results), "documents": len(frame),
        "unique_document_ids": frame.document_id.nunique(),
        "candidate_execution_documents": int(frame.title_category.eq("EXECUTION_DISCLOSURE_CANDIDATE").sum()),
        "candidate_execution_companies": int(frame.loc[frame.title_category.eq("EXECUTION_DISCLOSURE_CANDIDATE"), "symbol"].nunique()),
        "title_category_counts": frame.title_category.value_counts().to_dict(),
        "in_prior_snapshot_documents": int(frame.in_latest_prior_saved_snapshot.sum()),
        "outside_prior_snapshot_documents": int((~frame.in_latest_prior_saved_snapshot).sum()),
        "after_last_saved_snapshot_documents": int(frame.after_last_saved_snapshot.sum()),
        "new_actual_amounts_parsed": 0, "new_accounts": 0, "new_fits": 0,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False,
        "independent_forward_observations": 0, "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)
    print(f"指数历史并集目录完成：{result['complete_companies']}/{len(results)}家公司取得完整标题查询，{len(frame)}条；金额事实及策略检验仍待完成。", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "freeze":
        freeze()
    else:
        run()
