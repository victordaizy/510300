"""用有界日期分区修复历史公告分页；旧失败与原始响应均保留。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_510300_unlock_announcements_v1 import clean_title, title_kind

CONFIG = ROOT / "config/510300_unlock_announcement_source_v1_1.json"
PROTOCOL = ROOT / "docs/510300_UNLOCK_ANNOUNCEMENT_SOURCE_V1_1_REPAIR.md"
CORE_KEYS = ("secCode", "orgId", "announcementId", "announcementTitle", "announcementTime", "adjunctUrl")


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def merge_records(groups: list[list[dict]], expected: int) -> list[dict]:
    by_id = {}
    for rows in groups:
        for row in rows:
            key = str(row["announcementId"])
            if key in by_id:
                check(all(by_id[key].get(k) == row.get(k) for k in CORE_KEYS), "相同公告编号的身份字段变化")
            else:
                by_id[key] = row
    check(len(by_id) <= expected, "编号并集超过固定声明量")
    return sorted(by_id.values(), key=lambda x: (x["announcement_date"], str(x["announcementId"])))


def split_dates(start: str, end: str) -> tuple[tuple[str, str], tuple[str, str]]:
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    check(a < b, "单日不能再按日期二分")
    mid = a + timedelta(days=(b-a).days // 2)
    return ((a.isoformat(), mid.isoformat()), ((mid+timedelta(days=1)).isoformat(), b.isoformat()))


class MonthCollector:
    def __init__(self, month: str, directory: Path, config: dict, original: dict):
        self.month, self.directory, self.config, self.original = month, directory, config, original
        self.attempts = 0
        self.failures = 0
        self.nodes = []
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search", "Origin": "https://www.cninfo.com.cn", "Accept": "application/json"})

    def request(self, start: str, end: str, plate: str, page: int, traversal: int) -> tuple[int, list[dict]]:
        body = {"pageNum": str(page), "pageSize": str(self.config["page_size"]), "column": "szse", "tabName": "fulltext", "plate": plate, "stock": "", "searchkey": self.original["search_keyword"], "secid": "", "category": "", "trade": "", "seDate": f"{start}~{end}", "sortName": "time", "sortType": "asc", "isHLtitle": "true"}
        last = None
        for attempt in range(1, self.config["max_http_attempts_per_page"]+1):
            check(self.attempts < self.config["max_total_http_attempts_per_month"], "当月HTTP预算耗尽")
            self.attempts += 1
            stem = f"request_{self.attempts:03d}"
            receipt = {"url": self.original["url"], "method": "POST", "body": body, "requested_at": now(), "tls_verify": True, "traversal": traversal, "page_attempt": attempt}
            try:
                response = self.session.post(receipt["url"], data=body, timeout=(self.config["connect_timeout_seconds"], self.config["read_timeout_seconds"]))
                raw = self.directory / (stem+".json")
                raw.parent.mkdir(parents=True, exist_ok=True)
                with raw.open("xb") as stream:
                    stream.write(response.content)
                receipt.update(received_at=now(), http_status=response.status_code, http_date=response.headers.get("Date"), response_url=response.url, raw_path=raw.relative_to(ROOT).as_posix(), sha256=sha(raw), bytes=len(response.content))
                response.raise_for_status()
                payload = response.json()
                total = int(payload["totalAnnouncement"])
                check(total >= 0, "声明总量为负")
                check(isinstance(payload.get("announcements"), (list, type(None))), "公告字段不是数组")
                rows = []
                for item in payload.get("announcements") or []:
                    clock = pd.to_datetime(int(item["announcementTime"]), unit="ms", utc=True).tz_convert("Asia/Shanghai")
                    day = clock.strftime("%Y-%m-%d")
                    check(start <= day <= end, "日期越过请求区间")
                    code = str(item["secCode"])
                    if plate == "sh":
                        check(code.startswith(("6", "9")), "沪市分区混入其他代码")
                    if plate == "sz":
                        check(code.startswith(("0", "2", "3")), "深市分区混入其他代码")
                    rows.append({**item, "announcement_date": day, "announcement_timestamp_shanghai": clock.isoformat(), "clean_title": clean_title(item["announcementTitle"]), "title_kind": title_kind(item["announcementTitle"], self.original), "source_month": self.month, "source_page": page, "raw_response_path": receipt["raw_path"], "raw_response_sha256": receipt["sha256"]})
                check(len(rows) == min(30, max(0, total-(page-1)*30)), "响应条数不符合声明分页")
                receipt.update(status="HTTP_AND_SCHEMA_PASS", declared_records=total, received_records=len(rows))
                save(self.directory/(stem+"_receipt.json"), receipt)
                return total, rows
            except Exception as error:
                last = error
                self.failures += 1
                receipt.update(status="PAGE_ATTEMPT_FAILED", error_type=type(error).__name__, error=str(error))
                save(self.directory/(stem+"_receipt.json"), receipt)
        raise RuntimeError(f"规定次数内未取得分页：{last}")

    def collect(self, start: str, end: str, plate: str = "", expected: int | None = None) -> tuple[int, list[dict]]:
        total, first = self.request(start, end, plate, 1, 1)
        if expected is not None:
            check(total == expected, "区间声明量与父来源不符")
        node = {"start": start, "end": end, "plate": plate, "declared_records": total}
        self.nodes.append(node)
        if total <= 30:
            rows = merge_records([first], total)
            check(len(rows) == total, "单响应内有重复且不足声明量")
            node.update(status="PASS_SINGLE_RESPONSE", unique_records=len(rows))
            return total, rows
        if start < end:
            left, right = split_dates(start, end)
            a, rows_a = self.collect(*left, plate)
            b, rows_b = self.collect(*right, plate)
            check(a+b == total, "日期二分总量不等于父区间")
            rows = merge_records([rows_a, rows_b], total)
            check(len(rows) == total, "日期二分编号集合不完整或交叠")
            node.update(status="PASS_DATE_PARTITION", child_totals=[a, b], unique_records=len(rows))
            return total, rows
        if not plate:
            children = [self.collect(start, end, market) for market in self.config["single_day_overflow_partition"]]
            check(sum(x[0] for x in children) == total, "板块总量不等于单日总量")
            rows = merge_records([x[1] for x in children], total)
            check(len(rows) == total, "板块分区编号集合不完整或交叠")
            node.update(status="PASS_MARKET_PARTITION", child_totals=[x[0] for x in children], unique_records=len(rows))
            return total, rows
        pages = math.ceil(total/30)
        groups = [first]
        for traversal in range(1, self.config["max_passes_for_single_day_single_plate"]+1):
            for page in range(2 if traversal == 1 else 1, pages+1):
                repeated_total, group = self.request(start, end, plate, page, traversal)
                check(repeated_total == total, "多遍单日查询声明总量变化")
                groups.append(group)
            rows = merge_records(groups, total)
            if len(rows) == total:
                node.update(status="PASS_BOUNDED_UNION_EQUAL_DECLARED", traversals=traversal, unique_records=len(rows))
                return total, rows
        raise ValueError("单日单板块有界分页后仍未覆盖声明编号数量")


def repair_month(old: dict, output: Path, config: dict, original: dict) -> tuple[dict, list[dict]]:
    collector = MonthCollector(old["month"], output/old["month"], config, original)
    report = {"month": old["month"], "start_date": old["start_date"], "end_date": old["end_date"], "original_status": old["status"], "started_at": now(), "original_declared_records": old["declared_records"]}
    rows = []
    try:
        total, rows = collector.collect(old["start_date"], old["end_date"], expected=int(old["declared_records"]))
        report.update(status="PASS_REPAIRED_DECLARED_MONTH", declared_records=total, received_records=len(rows))
    except Exception as error:
        report.update(status="NO_VIEW_REPAIR_INCOMPLETE", error_type=type(error).__name__, error=str(error), received_records=len(rows))
    finally:
        collector.session.close()
    report.update(completed_at=now(), http_attempts=collector.attempts, failed_attempts=collector.failures)
    save(collector.directory/"partition_receipts.json", collector.nodes)
    save(collector.directory/"month_receipt.json", report)
    save(collector.directory/"records.json", rows)
    return report, rows


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    original = json.loads((ROOT/config["original_config"]).read_text(encoding="utf-8"))
    old_result_path = ROOT/config["original_result"]
    old_result = json.loads(old_result_path.read_text(encoding="utf-8"))
    check(old_result["status"] == "NO_VIEW_INCOMPLETE_PRIMARY_ANNOUNCEMENT_COVERAGE", "原失败状态不符")
    old_output = ROOT/config["original_output"]
    old_frame = pd.read_parquet(old_output/"all_announcement_records.parquet")
    check(sha(old_output/"all_announcement_records.parquet") == old_result["output_files"]["all_announcement_records.parquet"]["sha256"], "旧来源哈希变化")
    report_dir = ROOT/config["report_directory"]
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_0800")
    output = ROOT/config["output_parent"]/stamp
    protected = [CONFIG, PROTOCOL, Path(__file__), ROOT/config["original_config"], old_result_path, ROOT/"scripts/collect_510300_unlock_announcements_v1.py"]
    save(report_dir/"repair_freeze.json", {"status": "FROZEN_SOURCE_REPAIR_BEFORE_LABELS", "frozen_at": now(), "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in protected], "original_result_status_preserved": old_result["status"], "new_return_labels": 0, "new_models": 0, "new_accounts": 0})
    save(report_dir/"repair_claim.json", {"started_at": now(), "pid": os.getpid(), "output": output.relative_to(ROOT).as_posix(), "config_sha256": sha(CONFIG)})
    output.mkdir(parents=True, exist_ok=False)
    old_coverage = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(old_output.glob("????-??/month_receipt.json"))]
    check(len(old_coverage) == 140, "旧来源月份不符")
    complete, repair = [], []
    all_rows = []
    for record in old_coverage:
        if record["status"] == "PASS_COMPLETE_DECLARED_MONTH":
            rows = old_frame.loc[old_frame.source_month.eq(record["month"])].to_dict("records")
            check(len(rows) == int(record["declared_records"]) and len({x["announcementId"] for x in rows}) == len(rows), "复用月份编号量不符")
            complete.append({**record, "status": "PASS_REUSED_ORIGINAL_COMPLETE_MONTH", "http_attempts": 0, "failed_attempts": 0})
            all_rows.extend(rows)
        else:
            repair.append(record)
    print(f"保留旧60个月完整来源；修复{len(repair)}个月。输出：{output}", flush=True)
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(repair_month, row, output, config, original) for row in repair]
        for index, future in enumerate(as_completed(futures), 1):
            report, rows = future.result()
            complete.append(report)
            all_rows.extend(rows)
            print(f"[{index}/{len(repair)}] {report['month']} {report['status']} {report.get('received_records', 0)}条 {report['http_attempts']}次请求", flush=True)
    complete.sort(key=lambda row: row["month"])
    all_rows = merge_records([all_rows], sum(int(row["declared_records"]) for row in old_coverage))
    frame = pd.DataFrame(all_rows)
    selected_columns = list(old_frame.columns)
    frame = frame[selected_columns]
    frame.to_parquet(output/"all_announcement_records.parquet", index=False)
    frame.loc[frame.title_kind.eq("ELIGIBLE_ORIGINAL_DISCLOSURE")].to_parquet(output/"eligible_original_disclosures.parquet", index=False)
    frame.loc[frame.title_kind.eq("REVISION_TITLE_RETAINED_SEPARATELY")].to_parquet(output/"revision_titles.parquet", index=False)
    pd.DataFrame(complete).to_csv(output/"month_coverage.csv", index=False, encoding="utf-8-sig")
    old_ids, new_ids = set(old_frame.announcementId), set(frame.announcementId)
    missing = sorted(old_ids-new_ids)
    added = sorted(new_ids-old_ids)
    save(output/"difference_from_original.json", {"original_rows": len(old_frame), "original_unique_ids": len(old_ids), "new_unique_ids": len(new_ids), "added_ids": added, "missing_old_ids": missing})
    status = "PASS_COMPLETE_PRIMARY_ANNOUNCEMENT_METADATA_AFTER_SOURCE_REPAIR" if all(row["status"].startswith("PASS_") for row in complete) and not missing and len(frame) == int(old_frame.shape[0]) else "NO_VIEW_REPAIR_INCOMPLETE"
    result = {"source_study_id": config["source_study_id"], "status": status, "completed_at": now(), "raw_output": output.relative_to(ROOT).as_posix(), "old_result": config["original_result"], "old_result_sha256": sha(old_result_path), "original_status_preserved": old_result["status"], "requested_months": 140, "reused_months": 60, "repair_months": len(repair), "complete_months": sum(row["status"].startswith("PASS_") for row in complete), "all_unique_records": len(frame), "added_record_ids": len(added), "missing_old_record_ids": len(missing), "eligible_original_records": int(frame.title_kind.eq("ELIGIBLE_ORIGINAL_DISCLOSURE").sum()), "revision_records": int(frame.title_kind.eq("REVISION_TITLE_RETAINED_SEPARATELY").sum()), "new_http_attempts": sum(row["http_attempts"] for row in complete), "new_failed_http_attempts": sum(row["failed_attempts"] for row in complete), "new_return_labels": 0, "new_models": 0, "new_accounts": 0, "goal_achieved": False, "historical_intraday_first_delivery_proven": False, "output_files": {name: {"sha256": sha(output/name), "bytes": (output/name).stat().st_size} for name in ["all_announcement_records.parquet", "eligible_original_disclosures.parquet", "revision_titles.parquet", "month_coverage.csv", "difference_from_original.json"]}}
    save(output/"completion.json", result)
    save(report_dir/"source_repair_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not status.startswith("PASS_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
