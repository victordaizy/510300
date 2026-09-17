"""按完整月份获取巨潮原始公告索引，保留分页、失败与覆盖证据。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_unlock_announcement_source_v1.json"


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path: Path, value: dict | list) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def clean_title(value: str) -> str:
    return re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]+>", "", value)))


def title_kind(value: str, config: dict) -> str:
    title = clean_title(value)
    if config["title_required"] not in title or not re.search(config["title_qualifier_pattern"], title):
        return "NOT_TARGET_TITLE"
    if re.search(r"更正|补充|取消|撤销|延期|延迟|暂缓|终止", title):
        return "REVISION_TITLE_RETAINED_SEPARATELY"
    if re.search(config["title_exclusion_pattern"], title):
        return "SUPPORTING_OPINION_NOT_COUNTED"
    return "ELIGIBLE_ORIGINAL_DISCLOSURE"


def month_windows(config: dict) -> list[tuple[str, str, str]]:
    start, end = pd.Timestamp(config["start_date"]), pd.Timestamp(config["end_date"])
    return [(str(period), max(start, period.start_time).strftime("%Y-%m-%d"), min(end, period.end_time.normalize()).strftime("%Y-%m-%d"))
            for period in pd.period_range(start, end, freq="M")]


def expected_pages(total: int, page_size: int) -> int:
    return max(1, math.ceil(total / page_size))


def collect_month(window: tuple[str, str, str], out: Path, config: dict) -> tuple[dict, list]:
    month, start, end = window
    directory = out / month
    directory.mkdir()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
                            "Origin": "https://www.cninfo.com.cn", "Accept": "application/json, text/plain, */*"})
    declared, planned_pages, records, page_receipts = None, 1, [], []
    report = {"month": month, "start_date": start, "end_date": end, "started_at": now(), "status": "RUNNING"}
    try:
        page = 1
        while page <= planned_pages:
            body = {"pageNum": str(page), "pageSize": str(config["page_size"]), "column": "szse", "tabName": "fulltext", "plate": "",
                    "stock": "", "searchkey": config["search_keyword"], "secid": "", "category": "", "trade": "",
                    "seDate": start + "~" + end, "sortName": "time", "sortType": "asc", "isHLtitle": "true"}
            payload = None
            for attempt in range(1, config["max_attempts_per_page"] + 1):
                stem = f"page_{page:03d}_attempt_{attempt}"
                receipt = {"url": config["url"], "method": "POST", "body": body, "requested_at": now(), "tls_verify": True}
                try:
                    response = session.post(config["url"], data=body, timeout=(config["request_connect_timeout_seconds"], config["request_read_timeout_seconds"]))
                    raw = directory / (stem + ".json")
                    with raw.open("xb") as stream:
                        stream.write(response.content)
                    receipt.update(http_status=response.status_code, response_url=response.url, received_at=now(),
                                   raw_path=raw.relative_to(ROOT).as_posix(), bytes=len(response.content),
                                   sha256=hashlib.sha256(response.content).hexdigest(), http_date=response.headers.get("Date"))
                    response.raise_for_status()
                    candidate = response.json()
                    if not isinstance(candidate, dict) or "totalAnnouncement" not in candidate or not isinstance(candidate.get("announcements"), (list, type(None))):
                        raise ValueError("响应不是公告分页结构")
                    payload = candidate
                    receipt["status"] = "HTTP_AND_SCHEMA_PASS"
                except Exception as error:
                    receipt.update(status="PAGE_ATTEMPT_FAILED", error_type=type(error).__name__, error=str(error))
                write_json(directory / (stem + "_receipt.json"), receipt)
                page_receipts.append(receipt)
                if payload is not None:
                    break
            if payload is None:
                raise RuntimeError(f"第{page}页在规定次数内无法取得")
            total = int(payload["totalAnnouncement"])
            if total < 0:
                raise ValueError("总记录数为负")
            if declared is None:
                declared, planned_pages = total, expected_pages(total, config["page_size"])
                if planned_pages > config["max_pages_per_month"]:
                    raise ValueError("当月页数超过预定预算")
            elif total != declared:
                raise ValueError("分页过程中声明总量变化")
            received = payload.get("announcements") or []
            expected = min(config["page_size"], max(0, declared - (page - 1) * config["page_size"]))
            if len(received) != expected:
                raise ValueError(f"第{page}页实得{len(received)}条，预期{expected}条")
            for item in received:
                record = dict(item)
                clock = pd.to_datetime(int(record["announcementTime"]), unit="ms", utc=True).tz_convert("Asia/Shanghai")
                date = clock.strftime("%Y-%m-%d")
                if not start <= date <= end:
                    raise ValueError("公告日期越过请求月份")
                record.update(announcement_date=date, announcement_timestamp_shanghai=clock.isoformat(),
                              clean_title=clean_title(record["announcementTitle"]),
                              title_kind=title_kind(record["announcementTitle"], config), source_month=month,
                              source_page=page, raw_response_path=page_receipts[-1]["raw_path"],
                              raw_response_sha256=page_receipts[-1]["sha256"])
                records.append(record)
            page += 1
        ids = [str(row["announcementId"]) for row in records]
        if len(records) != declared or len(ids) != len(set(ids)):
            raise ValueError("全部分页的实得量或公告ID唯一性不符合声明")
        report.update(status="PASS_COMPLETE_DECLARED_MONTH", declared_records=declared, received_records=len(records),
                      actual_pages=planned_pages, successful_page_count=planned_pages,
                      reported_totalpages_first_response=json.loads((directory / "page_001_attempt_1.json").read_text(encoding="utf-8")).get("totalpages") if (directory / "page_001_attempt_1.json").exists() and page_receipts[0].get("status") == "HTTP_AND_SCHEMA_PASS" else None,
                      eligible_original_disclosures=sum(row["title_kind"] == "ELIGIBLE_ORIGINAL_DISCLOSURE" for row in records),
                      revision_titles=sum(row["title_kind"] == "REVISION_TITLE_RETAINED_SEPARATELY" for row in records))
    except Exception as error:
        report.update(status="INCOMPLETE_MONTH_NO_ZERO_FILL", declared_records=declared, received_records=len(records),
                      error_type=type(error).__name__, error=str(error))
    finally:
        session.close()
    report.update(completed_at=now(), total_http_attempts=len(page_receipts), failed_http_attempts=sum(r["status"] != "HTTP_AND_SCHEMA_PASS" for r in page_receipts))
    write_json(directory / "month_receipt.json", report)
    write_json(directory / "records.json", records)
    return report, records


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report_directory = ROOT / config["report_directory"]
    report_directory.mkdir(parents=True, exist_ok=True)
    claim = report_directory / "collection_claim.json"
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_0800")
    out = ROOT / config["output_parent"] / stamp
    write_json(claim, {"started_at": now(), "pid": __import__("os").getpid(), "output": out.relative_to(ROOT).as_posix(),
                       "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    out.mkdir(parents=True, exist_ok=False)
    windows = month_windows(config)
    summaries, all_records = [], []
    print(f"开始取得{len(windows)}个自然月公告；原始响应目录：{out}", flush=True)
    with ThreadPoolExecutor(max_workers=config["month_workers"]) as pool:
        futures = {pool.submit(collect_month, window, out, config): window[0] for window in windows}
        for future in as_completed(futures):
            summary, records = future.result()
            summaries.append(summary)
            all_records.extend(records)
            print(f"[{len(summaries)}/{len(windows)}] {summary['month']} {summary['status']} 实得{summary.get('received_records', 0)}条", flush=True)
    summaries.sort(key=lambda r: r["month"])
    all_records.sort(key=lambda r: (r["announcement_date"], str(r["announcementId"])))
    write_json(out / "all_announcement_records.json", all_records)
    pd.DataFrame(summaries).to_csv(out / "month_coverage.csv", index=False, encoding="utf-8-sig")
    selected = [{key: row.get(key) for key in ["secCode", "secName", "orgId", "announcementId", "announcementTitle", "announcementTime",
                 "announcement_date", "announcement_timestamp_shanghai", "adjunctUrl", "clean_title", "title_kind", "source_month", "source_page",
                 "raw_response_path", "raw_response_sha256"]} for row in all_records]
    frame = pd.DataFrame(selected)
    frame.to_parquet(out / "all_announcement_records.parquet", index=False)
    frame.loc[frame.title_kind.eq("ELIGIBLE_ORIGINAL_DISCLOSURE")].to_parquet(out / "eligible_original_disclosures.parquet", index=False)
    frame.loc[frame.title_kind.eq("REVISION_TITLE_RETAINED_SEPARATELY")].to_parquet(out / "revision_titles.parquet", index=False)
    complete = all(row["status"] == "PASS_COMPLETE_DECLARED_MONTH" for row in summaries)
    duplicates = int(frame.announcementId.duplicated().sum())
    status = "PASS_COMPLETE_PRIMARY_ANNOUNCEMENT_METADATA" if complete and duplicates == 0 else "NO_VIEW_INCOMPLETE_PRIMARY_ANNOUNCEMENT_COVERAGE"
    result = {"source_study_id": config["source_study_id"], "status": status, "completed_at": now(),
              "raw_output": out.relative_to(ROOT).as_posix(), "requested_months": len(windows),
              "completed_months": sum(row["status"] == "PASS_COMPLETE_DECLARED_MONTH" for row in summaries),
              "all_records": len(frame), "duplicate_global_announcement_ids": duplicates,
              "eligible_title_records": int(frame.title_kind.eq("ELIGIBLE_ORIGINAL_DISCLOSURE").sum()),
              "revision_title_records": int(frame.title_kind.eq("REVISION_TITLE_RETAINED_SEPARATELY").sum()),
              "incomplete_months": [row["month"] for row in summaries if row["status"] != "PASS_COMPLETE_DECLARED_MONTH"],
              "total_http_attempts": sum(row["total_http_attempts"] for row in summaries),
              "failed_http_attempts": sum(row["failed_http_attempts"] for row in summaries),
              "new_return_labels": 0, "new_models": 0, "new_accounts": 0, "goal_achieved": False,
              "historical_intraday_first_delivery_proven": False,
              "output_files": {name: {"sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(), "bytes": (out / name).stat().st_size}
                               for name in ["all_announcement_records.json", "all_announcement_records.parquet", "eligible_original_disclosures.parquet", "revision_titles.parquet", "month_coverage.csv"]}}
    write_json(out / "completion.json", result)
    write_json(report_directory / "source_collection_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if status != "PASS_COMPLETE_PRIMARY_ANNOUNCEMENT_METADATA":
        sys.exit(2)


if __name__ == "__main__":
    main()
