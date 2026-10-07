"""按全部历史成员建立发行/配股检索目录，保留类别和标题检索的差异。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_issuance_catalogue_v1"
PROBE = ROOT / "reports/research/510300_factor96_issuance_source_probe_v1"
STUDY = "510300_FACTOR96_ISSUANCE_CATALOGUE_V1"
QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
START, END = "2015-01-01", "2025-12-31"
PAGE_SIZE, MAX_LEAF_ROWS = 30, 1500
STOP = threading.Event()
QUERIES = {"categories": {"keyword": "", "category": "category_pg_szsh;category_zf_szsh"},
           "issuance_title": {"keyword": "发行", "category": ""},
           "rights_title": {"keyword": "配股", "category": ""}}


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def classify_title(title):
    """只分派原文工作，不把标题当事件、金额、供给日期或事实。"""
    if re.search("首次公开发行|首次公开.*上市|初次公开发行|首次发行", title):
        return "IPO_OUTSIDE_M06_CORE"
    if re.search("债券|可转债|可转换|优先股|境外|H股|Ｈ股|全球存托|GDR|债务融资|短期融资|中期票据|资产支持", title):
        return "OTHER_FINANCING_OR_MIXED_TITLE"
    if re.search("法律意见|律师事务所|核查意见|保荐书|审计报告|验资报告|鉴证报告|独立财务顾问", title):
        return "SUPPORTING_DOCUMENT"
    if re.search("终止|撤回|不予|未获|未通过|失效", title):
        return "TERMINATION_OR_REJECTION_CANDIDATE"
    if re.search("更正|补充|修订|调整|更新|延期|延长", title):
        return "AMENDMENT_CANDIDATE"
    if re.search("分配股息|转配股息|汽配股份|配股权质押", title):
        return "LEXICAL_FALSE_POSITIVE_CANDIDATE"
    if re.search("发行情况|发行结果|发行报告|发行公告|上市公告|股份变动|新增股份|缴款|配股提示|配股说明书|认购结果", title):
        return "ISSUER_CALENDAR_DOCUMENT_CANDIDATE"
    if re.search("募集说明书|配股|增发|发行.*股票|发行.*股份|发行.*A股|发行.*Ａ股", title):
        return "EQUITY_PLAN_OR_STAGE_CANDIDATE"
    return "UNRESOLVED_TITLE_RETAINED"


def normalize_page(body, allowed, query, start, end):
    total = body.get("totalAnnouncement")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("目录总数类型或数值无效")
    batch = body.get("announcements") or []
    if not isinstance(batch, list) or len(batch) > PAGE_SIZE or (total > 0 and not batch) or (total == 0 and batch):
        raise ValueError("分页长度与目录总数不一致")
    output = []
    for raw in batch:
        code, org = str(raw["secCode"]), str(raw["orgId"])
        if code not in allowed or org != allowed[code]:
            raise ValueError("返回发行人不在事前固定双公司范围")
        title = re.sub("<[^>]+>", "", raw["announcementTitle"])
        keyword = QUERIES[query]["keyword"]
        if keyword and keyword not in title:
            raise ValueError("标题关键词过滤没有生效")
        timestamp = pd.to_datetime(raw["announcementTime"], unit="ms", utc=True).tz_convert("Asia/Shanghai")
        day = timestamp.date().isoformat()
        if not start <= day <= end:
            raise ValueError("返回档案日期超出固定窗口")
        adjunct = str(raw["adjunctUrl"])
        if not adjunct.startswith("finalpage/") or ".." in adjunct or not adjunct.lower().endswith(".pdf"):
            raise ValueError("原文地址不是预期公开PDF目录")
        output.append({"symbol": code + (".SH" if code.startswith("6") else ".SZ"), "sec_code": code,
            "org_id": org, "document_id": str(raw["announcementId"]), "title": title,
            "catalogue_timestamp": timestamp.isoformat(), "catalogue_date": day,
            "source_url": "https://static.cninfo.com.cn/" + adjunct, "adjunct_url": adjunct,
            "announcement_type": raw.get("announcementType"), "column_id": raw.get("columnId"),
            "title_role": classify_title(title), "query": query,
            "historical_first_publication_verified": False, "trading_feature_admitted": False})
    return total, output


def freeze():
    assert not (OUT / "freeze.json").exists(), "目录范围已经冻结，不能覆盖"
    sources = {
        "inputs/membership.parquet": "reports/research/510300_factor96_known_supply_sources_v1/inputs/membership.parquet",
        "inputs/company_directory.json": "reports/research/510300_corporate_repurchase_public_completion_v1/cninfo_stock_directory.json",
        "source_evidence/current_mandate.json": "config/510300_existing_data_training_mandate_v1.json",
        "source_evidence/prior_supply_result.json": "reports/research/510300_factor96_known_supply_sources_v1/event_fields_v1/result.json",
        "source_evidence/prior_version_result.json": "reports/research/510300_factor96_supply_version_ledger_v1/result.json",
    }
    for name, original in sources.items():
        destination = OUT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / original, destination)
    for path in PROBE.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            destination = OUT / "source_probe" / path.relative_to(PROBE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for path in (ROOT / "reports/research/510300_factor96_program_v1").iterdir():
        if path.is_file():
            destination = OUT / "program_before" / path.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(path, destination)
    members = pd.read_parquet(OUT / "inputs/membership.parquet")
    symbols = sorted(members.loc[pd.to_datetime(members.membership_date).between(START, END), "symbol"].unique())
    directory = {r["code"]: r for r in read(OUT / "inputs/company_directory.json")["stockList"] if r["category"] == "A股"}
    companies = []
    for symbol in symbols:
        match = directory.get(symbol[:6])
        companies.append({"symbol": symbol, "code": symbol[:6], "org_id": match["orgId"] if match else None,
                          "directory_name": match["zwjc"] if match else None})
    groups = [{"group_id": f"g{i//2:04d}", "companies": companies[i:i+2]} for i in range(0, len(companies), 2)]
    save(OUT / "inputs/groups.json", groups, True)
    test = read(OUT / "prefreeze_test_receipt.json")
    assert test["exit_code"] == 0
    for name in ["research/factor96_issuance_catalogue_v1.py", "tests/test_factor96_issuance_catalogue_v1.py"]:
        target = OUT / "code" / Path(name).name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / name, target)
    protocol = {"at": now(), "study_id": STUDY, "phase": "CATALOGUE_ONLY_BEFORE_NEW_DOCUMENT_FIELDS_OR_RETURNS",
        "purpose": "补充M06发行和配股的原公告目录，检查类别检索遗漏；不能由该目录直接构成完整事件日历。",
        "period": [START, END], "companies": len(companies), "groups": len(groups), "queries": QUERIES,
        "universe": "2015至2025每日点时沪深300成员的完整历史并集，仅用于收集范围；后纳入公司不能提前作为当时成员。",
        "prior_count_explanation": "此前2192个解禁目录目标涉及526个有匹配公告的发行人，不能把它误当全部历史成员；本轮从完整成员表重新取并集。",
        "grouping": "股票代码排序后每两家公司一组；事前27=21+6的混合交易所实例及完整文档集合一致支持过滤。",
        "paging": "每页30，叶窗口最多1500条；首屏总量更高时按日历中点递归分割，不按公告内容或收益选择范围。每个叶窗口须总量稳定、无重复且全部收齐。",
        "transport": "3并发，单请求连接10秒读取25秒，每次请求后0.35秒；仅传输异常和5xx最多两次，403/429停止新请求，来源错误保留。",
        "definition_boundary": "类别与标题三路并集只保证已请求过滤器的接收完整，不保证经济发行事件全集；债券、IPO、辅助意见、更正、终止及未知全部分开保留。",
        "free_float_boundary": "普通流通股、总股本和分级靠档后的调整股本都不自动等于M01/M06自由流通量；当前2024年规则不能后填2015年方法与时点。",
        "clock": "目录时间、PDF署期、经济发行/缴款/上市日和本次下载时钟需要区分；此阶段只保存目录时间，不宣称首次历史HTTP可得。",
        "new_accounts": 0, "new_models": 0, "new_returns_loaded": False, "external_review": "NOT_PERFORMED",
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False}
    save(OUT / "protocol.json", protocol, True)
    save(OUT / "freeze.json", {"at": now(), "before_catalogue_requests_and_outcomes": True,
        "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)} for p in sorted(OUT.rglob("*")) if p.is_file()]}, True)
    print(f"发行来源范围已冻结：{len(companies)}家公司、{len(groups)}个双公司组、三路目录。", flush=True)


def acquire(session, key, payload):
    for attempt in [1, 2]:
        receipt_path = OUT / "receipts" / f"{key}_a{attempt}.json"
        if receipt_path.exists():
            receipt = read(receipt_path)
            content = (OUT / receipt["raw_path"]).read_bytes() if receipt.get("raw_path") else None
        elif STOP.is_set():
            return None, {"status": "NOT_REQUESTED_AFTER_SOURCE_LIMIT"}
        else:
            receipt = {"requested_at": now(), "url": QUERY_URL, "payload": payload, "attempt": attempt}
            content = None
            try:
                response = session.post(QUERY_URL, data=payload, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}, timeout=(10, 25))
                content = response.content
                raw = OUT / "raw" / f"{key}_a{attempt}.json"
                raw.parent.mkdir(exist_ok=True)
                raw.write_bytes(content)
                receipt.update(http_status=response.status_code, raw_path=raw.relative_to(OUT).as_posix(),
                    bytes=len(content), sha256=digest(raw), status="HTTP_OK" if response.status_code == 200 else "HTTP_ERROR")
                if response.status_code in [403, 429]:
                    STOP.set()
            except requests.RequestException as exc:
                receipt.update(status="REQUEST_FAILED", error_type=type(exc).__name__)
            receipt["completed_at"] = now()
            save(receipt_path, receipt, True)
            time.sleep(.35)
        receipt = {**receipt, "receipt_path": receipt_path.relative_to(OUT).as_posix()}
        if receipt["status"] != "REQUEST_FAILED" and not 500 <= receipt.get("http_status", 0) <= 599:
            return content, receipt
    return content, receipt


def collect_window(session, group, query, start, end):
    label = f"{group['group_id']}_{query}_{start.replace('-', '')}_{end.replace('-', '')}"
    allowed = {r["code"]: r["org_id"] for r in group["companies"]}
    state = {"window_id": label, "group_id": group["group_id"], "query": query, "start": start, "end": end,
             "status": "NOT_REQUESTED", "complete": False, "declared_total": None, "received": 0}
    if not all(allowed.values()):
        state["status"] = "NO_ORGANIZATION_MAPPING"
        return [], [state]
    payload = {"pageNum": "1", "pageSize": str(PAGE_SIZE), "column": "szse", "tabName": "fulltext", "plate": "",
        "stock": ";".join(code + "," + org for code, org in allowed.items()), "searchkey": QUERIES[query]["keyword"],
        "secid": "", "category": QUERIES[query]["category"], "trade": "", "seDate": start + "~" + end,
        "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
    rows, seen, total = [], set(), None
    for page in range(1, MAX_LEAF_ROWS // PAGE_SIZE + 1):
        payload["pageNum"] = str(page)
        content, receipt = acquire(session, label + f"_p{page:03d}", dict(payload))
        if receipt["status"] != "HTTP_OK":
            state["status"] = receipt["status"]
            break
        try:
            count, batch = normalize_page(json.loads(content), allowed, query, start, end)
            if total is not None and count != total:
                raise ValueError("分页期间总量变化")
            total = count
            state["declared_total"] = count
            if page == 1 and total > MAX_LEAF_ROWS:
                lo, hi = date.fromisoformat(start), date.fromisoformat(end)
                if lo == hi:
                    state["status"] = "SINGLE_DAY_EXCEEDS_FIXED_PAGE_LIMIT"
                    break
                middle = lo + timedelta(days=(hi - lo).days // 2)
                a, wa = collect_window(session, group, query, start, middle.isoformat())
                b, wb = collect_window(session, group, query, (middle + timedelta(days=1)).isoformat(), end)
                state.update(status="SPLIT_INTO_DISJOINT_DATE_WINDOWS", complete=bool(wa[0]["complete"] and wb[0]["complete"]),
                             received=len(a) + len(b), child_windows=[wa[0]["window_id"], wb[0]["window_id"]])
                if state["complete"] and len(a) + len(b) != total:
                    state.update(status="PARENT_CHILD_TOTAL_MISMATCH", complete=False)
                return a + b, [state, *wa, *wb]
            for row in batch:
                identity = (row["document_id"], row["symbol"])
                if identity in seen:
                    raise ValueError("同一叶窗口出现重复文档，不能算已收齐")
                seen.add(identity)
                rows.append({**row, "window_id": label, "page": page, "raw_response_path": receipt["raw_path"],
                             "raw_response_sha256": receipt["sha256"], "request_receipt": receipt["receipt_path"]})
            state["received"] = len(rows)
            if len(rows) >= total:
                if len(rows) != total:
                    raise ValueError("收到行数超过声明总量")
                state.update(status="COMPLETE_FILTERED_QUERY", complete=True)
                break
        except (ValueError, TypeError, KeyError) as exc:
            state.update(status="SOURCE_RESPONSE_NOT_ADMITTED", reason=str(exc))
            break
    if not state["complete"] and state["status"] == "NOT_REQUESTED":
        state["status"] = "FIXED_PAGE_LIMIT_NOT_COMPLETE"
    return rows, [state]


def collect_job(group, query):
    path = OUT / "jobs" / f"{group['group_id']}_{query}.json"
    assert not path.exists(), "当前运行不能覆盖已有任务结果"
    with requests.Session() as session:
        rows, windows = collect_window(session, group, query, START, END)
    result = {"at": now(), "group_id": group["group_id"], "query": query, "companies": group["companies"],
              "complete": windows[0]["complete"], "windows": windows, "rows": rows}
    save(path, result, True)
    return result


def run():
    assert (OUT / "freeze.json").exists() and not (OUT / "run_started.json").exists(), "只允许固定运行一次"
    for item in read(OUT / "freeze.json")["files"]:
        assert digest(OUT / item["path"]) == item["sha256"], item["path"]
    assert digest(Path(__file__)) == digest(OUT / "code" / Path(__file__).name)
    save(OUT / "run_started.json", {"at": now(), "freeze_sha256": digest(OUT / "freeze.json"), "new_accounts": 0}, True)
    groups, all_rows, all_windows, complete = read(OUT / "inputs/groups.json"), [], [], 0
    jobs = [(group, query) for group in groups for query in QUERIES]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(collect_job, group, query): (group["group_id"], query) for group, query in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            complete += int(result["complete"])
            all_rows.extend(result["rows"])
            all_windows.extend(result["windows"])
            if number % 20 == 0 or number == len(jobs):
                save(OUT / "live_progress.json", {"at": now(), "completed_jobs": number, "expected_jobs": len(jobs),
                    "complete_queries": complete, "received_occurrences": len(all_rows), "source_limit_stop": STOP.is_set()})
                print(f"发行目录完成{number}/{len(jobs)}组查询，完整{complete}组，已收到{len(all_rows)}条来源记录。", flush=True)
    frame = pd.DataFrame(all_rows).sort_values(["catalogue_date", "symbol", "document_id", "query"])
    frame.to_parquet(OUT / "catalogue_occurrences.parquet", index=False)
    merged = []
    for (symbol, document_id), group in frame.groupby(["symbol", "document_id"], sort=True):
        fields = ["org_id", "title", "catalogue_timestamp", "catalogue_date", "source_url", "title_role"]
        conflicts = [field for field in fields if group[field].nunique(dropna=False) != 1]
        row = group.iloc[0].to_dict()
        row.update(query_sources="|".join(sorted(group["query"].unique())), occurrence_count=len(group),
                   catalogue_metadata_conflicts="|".join(conflicts), raw_reference_count=group.raw_response_path.nunique())
        merged.append(row)
    unique = pd.DataFrame(merged)
    unique.to_parquet(OUT / "unique_documents.parquet", index=False)
    pd.DataFrame(all_windows).to_parquet(OUT / "query_windows.parquet", index=False)
    save(OUT / "query_windows.json", all_windows, True)
    years = pd.to_datetime(unique.catalogue_date).dt.year
    coverage = pd.crosstab(years, unique.title_role).reindex(range(2015, 2026), fill_value=0)
    coverage.index.name = "year"
    coverage.to_csv(OUT / "逐年发行目录角色.csv", encoding="utf-8-sig")
    result = {"at": now(), "study_id": STUDY, "status": "CATALOGUE_FIXED_FILTERS_COMPLETE" if complete == len(jobs) else "CATALOGUE_PARTIAL_SOURCE_GAPS_RETAINED",
        "historical_member_companies": sum(len(g["companies"]) for g in groups), "query_jobs": len(jobs), "complete_jobs": complete,
        "source_occurrences": len(frame), "unique_issuer_documents": len(unique), "unique_pdf_ids": unique.document_id.nunique(),
        "document_issuers": unique.symbol.nunique(), "title_roles": unique.title_role.value_counts().to_dict(),
        "metadata_conflict_documents": int(unique.catalogue_metadata_conflicts.ne("").sum()),
        "not_in_categories_but_title_query": int((~unique.query_sources.str.contains("categories")).sum()),
        "new_accounts": 0, "new_returns": 0, "new_models": 0, "full_M06_calendar_established": False,
        "free_float_denominator_established": False, "T13": "NOT_RUN", "goal_status": "active", "goal_achieved": False,
        "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
