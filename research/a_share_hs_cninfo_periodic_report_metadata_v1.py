from __future__ import annotations

import calendar
import gzip
import hashlib
import html
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class CalendarInterval:
    label: str
    start_date: date
    end_date: date


@dataclass(frozen=True)
class IntervalReceipt:
    label: str
    start_date: str
    end_date: str
    status: str
    source: str
    page_count: int
    declared_total: int | None
    retrieved_rows: int
    unique_announcement_ids: int
    checkpoint_file: str | None
    checkpoint_sha256: str | None
    error: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def monthly_intervals(start_date: date, end_date: date) -> list[CalendarInterval]:
    if start_date > end_date:
        raise ValueError("查询开始日不得晚于结束日")
    intervals: list[CalendarInterval] = []
    cursor = date(start_date.year, start_date.month, 1)
    while cursor <= end_date:
        month_last = date(
            cursor.year,
            cursor.month,
            calendar.monthrange(cursor.year, cursor.month)[1],
        )
        interval_start = max(start_date, cursor)
        interval_end = min(end_date, month_last)
        intervals.append(
            CalendarInterval(
                label=f"{cursor.year:04d}-{cursor.month:02d}",
                start_date=interval_start,
                end_date=interval_end,
            )
        )
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return intervals


def build_session(search_page_url: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/140 Safari/537.36"
            ),
            "Referer": search_page_url,
            "Origin": "https://www.cninfo.com.cn",
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
        }
    )
    response = session.get(search_page_url, timeout=30)
    response.raise_for_status()
    return session


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout_seconds: int,
    maximum_attempts: int,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_error: Exception | None = None
    for attempt in range(1, maximum_attempts + 1):
        try:
            response = session.request(method, url, data=data, timeout=timeout_seconds)
            response.raise_for_status()
            return json.loads(response.content.decode("utf-8"))
        except Exception as error:  # noqa: BLE001 - 网络失败必须进入逐分片收据
            final_error = error
            if attempt < maximum_attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"请求失败：{url}；{final_error}")


def checkpoint_path(root: Path, interval: CalendarInterval) -> Path:
    return root / f"{interval.label}.json.gz"


def _atomic_gzip_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, path)


def load_checkpoint(path: Path, interval: CalendarInterval) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "label": interval.label,
        "start_date": interval.start_date.isoformat(),
        "end_date": interval.end_date.isoformat(),
    }
    actual = {key: payload.get("interval", {}).get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"检查点分片身份不符：{path}")
    if payload.get("status") != "COMPLETE":
        raise ValueError(f"检查点不是完整状态：{path}")
    records = payload.get("records") or []
    declared_total = int(payload["declared_total"])
    if len(records) != declared_total:
        raise ValueError(f"检查点行数不等于声明总数：{path}")
    identifiers = [str(record.get("announcementId") or "") for record in records]
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"检查点公告ID缺失或重复：{path}")
    return payload


def query_interval(
    interval: CalendarInterval,
    *,
    config: dict[str, Any],
    checkpoint_root: Path,
) -> tuple[list[dict[str, Any]], IntervalReceipt]:
    path = checkpoint_path(checkpoint_root, interval)
    if path.exists():
        try:
            payload = load_checkpoint(path, interval)
            records = payload["records"]
            return records, IntervalReceipt(
                label=interval.label,
                start_date=interval.start_date.isoformat(),
                end_date=interval.end_date.isoformat(),
                status="COMPLETE",
                source="REUSED_VERIFIED_CHECKPOINT",
                page_count=int(payload["page_count"]),
                declared_total=int(payload["declared_total"]),
                retrieved_rows=len(records),
                unique_announcement_ids=len(records),
                checkpoint_file=path.relative_to(ROOT).as_posix(),
                checkpoint_sha256=sha256_file(path),
                error=None,
            )
        except Exception as error:  # noqa: BLE001 - 损坏检查点不得被静默复用
            return [], IntervalReceipt(
                label=interval.label,
                start_date=interval.start_date.isoformat(),
                end_date=interval.end_date.isoformat(),
                status="FAILED_INVALID_EXISTING_CHECKPOINT",
                source="EXISTING_CHECKPOINT",
                page_count=0,
                declared_total=None,
                retrieved_rows=0,
                unique_announcement_ids=0,
                checkpoint_file=path.relative_to(ROOT).as_posix(),
                checkpoint_sha256=sha256_file(path),
                error=str(error),
            )

    query = config["query"]
    source = config["source"]
    categories = ";".join(query["categories"]) + ";"
    session: requests.Session | None = None
    records: list[dict[str, Any]] = []
    declared_total: int | None = None
    page_count = 0
    try:
        session = build_session(source["search_page_url"])
        previous_first_id: str | None = None
        for page_number in range(1, int(query["maximum_pages_per_interval"]) + 1):
            body = {
                "pageNum": page_number,
                "pageSize": int(query["page_size"]),
                "column": query["column"],
                "tabName": query["tab_name"],
                "plate": "",
                "stock": "",
                "searchkey": "",
                "secid": "",
                "category": categories,
                "trade": "",
                "seDate": (
                    f"{interval.start_date.isoformat()}~{interval.end_date.isoformat()}"
                ),
                "sortName": "",
                "sortType": "",
                "isHLtitle": "true",
            }
            payload = request_json(
                session,
                "POST",
                source["query_url"],
                timeout_seconds=int(query["timeout_seconds"]),
                maximum_attempts=int(query["maximum_attempts"]),
                data=body,
            )
            if declared_total is None:
                declared_total = int(payload.get("totalAnnouncement") or 0)
            elif declared_total != int(payload.get("totalAnnouncement") or 0):
                raise RuntimeError("同一分片翻页期间接口声明总数发生变化")
            page_records = payload.get("announcements") or []
            if bool(payload.get("hasMore")) and not page_records:
                raise RuntimeError("接口声明仍有下一页但当前页为空")
            if page_records:
                first_id = str(page_records[0].get("announcementId") or "")
                if page_number > 1 and first_id == previous_first_id:
                    raise RuntimeError("翻页返回重复首页，拒绝形成伪完整档案")
                previous_first_id = first_id
            for record in page_records:
                copied = dict(record)
                copied["query_interval"] = interval.label
                records.append(copied)
            page_count = page_number
            if not bool(payload.get("hasMore")):
                break
            time.sleep(float(query["inter_page_delay_seconds"]))
        else:
            raise RuntimeError("翻页达到预注册上限仍未结束")

        if declared_total is None or len(records) != declared_total:
            raise RuntimeError(
                f"实际取得{len(records)}行，不等于接口声明{declared_total}行"
            )
        identifiers = [str(record.get("announcementId") or "") for record in records]
        if not all(identifiers):
            raise RuntimeError("公告ID存在空值")
        if len(set(identifiers)) != len(identifiers):
            raise RuntimeError("分片内公告ID重复")
        checkpoint_payload = {
            "status": "COMPLETE",
            "retrieved_at": datetime.now(TIMEZONE).isoformat(),
            "source_query_url": source["query_url"],
            "interval": {
                "label": interval.label,
                "start_date": interval.start_date.isoformat(),
                "end_date": interval.end_date.isoformat(),
            },
            "page_count": page_count,
            "declared_total": declared_total,
            "records": records,
        }
        _atomic_gzip_json(path, checkpoint_payload)
        return records, IntervalReceipt(
            label=interval.label,
            start_date=interval.start_date.isoformat(),
            end_date=interval.end_date.isoformat(),
            status="COMPLETE",
            source="LIVE_CNINFO_QUERY",
            page_count=page_count,
            declared_total=declared_total,
            retrieved_rows=len(records),
            unique_announcement_ids=len(set(identifiers)),
            checkpoint_file=path.relative_to(ROOT).as_posix(),
            checkpoint_sha256=sha256_file(path),
            error=None,
        )
    except Exception as error:  # noqa: BLE001 - 失败必须返回结构化NO_VIEW证据
        return records, IntervalReceipt(
            label=interval.label,
            start_date=interval.start_date.isoformat(),
            end_date=interval.end_date.isoformat(),
            status="FAILED",
            source="LIVE_CNINFO_QUERY",
            page_count=page_count,
            declared_total=declared_total,
            retrieved_rows=len(records),
            unique_announcement_ids=len(
                {
                    str(record.get("announcementId") or "")
                    for record in records
                    if record.get("announcementId")
                }
            ),
            checkpoint_file=None,
            checkpoint_sha256=None,
            error=str(error),
        )
    finally:
        if session is not None:
            session.close()


def normalize_title(value: Any) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "", str(value or "")))
    return re.sub(r"[\s\u3000]+", "", text).replace("－", "-")


def classify_title(
    title: Any,
    *,
    excluded_tokens: Iterable[str],
) -> tuple[str | None, pd.Timestamp | None, str]:
    normalized = normalize_title(title)
    for token in excluded_tokens:
        if token in normalized:
            return None, None, f"EXCLUDED_TITLE_TOKEN_{token}"
    match = re.search(r"(?<!\d)(20\d{2})年", normalized)
    if match is None:
        return None, None, "NO_REPORT_YEAR_IN_TITLE"
    report_year = int(match.group(1))
    if "第一季度报告" in normalized or "一季度报告" in normalized:
        return "Q1", pd.Timestamp(report_year, 3, 31), "ACCEPTED_ORIGINAL_FULL"
    if any(token in normalized for token in ("半年度报告", "中期报告", "半年报")):
        return "H1", pd.Timestamp(report_year, 6, 30), "ACCEPTED_ORIGINAL_FULL"
    if "第三季度报告" in normalized or "三季度报告" in normalized:
        return "Q3", pd.Timestamp(report_year, 9, 30), "ACCEPTED_ORIGINAL_FULL"
    if (
        ("年度报告" in normalized or "年报" in normalized)
        and "半年度报告" not in normalized
        and "半年报" not in normalized
    ):
        return "FY", pd.Timestamp(report_year, 12, 31), "ACCEPTED_ORIGINAL_FULL"
    return None, None, "NOT_PERIODIC_REPORT_FULL_TITLE"


def ts_code_from_cninfo_code(value: Any) -> str | None:
    code = str(value or "").zfill(6)
    if re.fullmatch(r"(600|601|603|605|688|689)\d{3}", code):
        return f"{code}.SH"
    if re.fullmatch(r"(000|001|002|003|300|301)\d{3}", code):
        return f"{code}.SZ"
    return None


def url_publication_date(value: Any) -> pd.Timestamp | None:
    match = re.search(r"finalpage/(\d{4}-\d{2}-\d{2})/", str(value or ""))
    return pd.Timestamp(match.group(1)) if match else None


def timestamp_at(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(int(value), unit="ms", utc=True).tz_convert(TIMEZONE)


def normalize_records(
    records: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any],
    retrieved_at: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exclusions = config["event_contract"]["excluded_title_tokens"]
    rows: list[dict[str, Any]] = []
    for record in records:
        title = normalize_title(record.get("announcementTitle"))
        period_type, report_period, title_status = classify_title(
            title, excluded_tokens=exclusions
        )
        official_timestamp = timestamp_at(record.get("announcementTime"))
        official_timestamp_date = (
            pd.Timestamp(official_timestamp.date())
            if official_timestamp is not None
            else None
        )
        relative_url = str(record.get("adjunctUrl") or "")
        official_url_date = url_publication_date(relative_url)
        rows.append(
            {
                "announcement_id": str(record.get("announcementId") or ""),
                "sec_code": str(record.get("secCode") or "").zfill(6),
                "ts_code": ts_code_from_cninfo_code(record.get("secCode")),
                "sec_name": str(record.get("secName") or ""),
                "org_id": str(record.get("orgId") or ""),
                "page_column": str(record.get("pageColumn") or ""),
                "announcement_title": title,
                "title_status": title_status,
                "period_type": period_type,
                "report_period": report_period,
                "announcement_time_ms": record.get("announcementTime"),
                "official_timestamp_at": official_timestamp,
                "official_timestamp_date": official_timestamp_date,
                "official_url_date": official_url_date,
                "official_internal_date_equal": (
                    official_timestamp_date == official_url_date
                    if official_timestamp_date is not None and official_url_date is not None
                    else False
                ),
                "relative_pdf_url": relative_url,
                "official_pdf_url": urljoin(config["source"]["pdf_base_url"], relative_url),
                "adjunct_type": str(record.get("adjunctType") or ""),
                "adjunct_size_kb": record.get("adjunctSize"),
                "query_interval": str(record.get("query_interval") or ""),
                "retrieved_at": retrieved_at,
            }
        )
    all_records = pd.DataFrame(rows)
    if all_records.empty:
        return all_records, all_records.copy()
    accepted = all_records.loc[
        all_records["title_status"].eq("ACCEPTED_ORIGINAL_FULL")
        & all_records["ts_code"].notna()
        & all_records["official_internal_date_equal"]
        & all_records["adjunct_type"].str.upper().eq("PDF")
    ].copy()
    accepted = accepted.sort_values(
        [
            "ts_code",
            "report_period",
            "official_url_date",
            "announcement_time_ms",
            "announcement_id",
        ],
        kind="stable",
    )
    events = accepted.drop_duplicates(["ts_code", "report_period"], keep="first").copy()
    events = events.rename(columns={"official_url_date": "event_publication_date"})
    columns = [
        "announcement_id",
        "ts_code",
        "sec_code",
        "sec_name",
        "org_id",
        "page_column",
        "announcement_title",
        "period_type",
        "report_period",
        "event_publication_date",
        "official_timestamp_at",
        "official_timestamp_date",
        "official_internal_date_equal",
        "relative_pdf_url",
        "official_pdf_url",
        "adjunct_type",
        "adjunct_size_kb",
        "query_interval",
        "retrieved_at",
    ]
    return all_records, events[columns].reset_index(drop=True)


def verify_pdf_prefix(
    session: requests.Session,
    url: str,
    *,
    accepted_host: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    host = (urlparse(url).hostname or "").lower()
    result: dict[str, Any] = {
        "url": url,
        "official_host": host == accepted_host.lower(),
        "status_code": None,
        "content_type": None,
        "pdf_header": False,
        "error": None,
    }
    try:
        with session.get(url, timeout=timeout_seconds, stream=True) as response:
            result["status_code"] = int(response.status_code)
            result["content_type"] = response.headers.get("Content-Type")
            response.raise_for_status()
            result["pdf_header"] = response.raw.read(5) == b"%PDF-"
    except Exception as error:  # noqa: BLE001 - 抽检失败必须逐条保留
        result["error"] = str(error)
    return result


def deterministic_pdf_sample(events: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if sample_size < 1:
        raise ValueError("PDF抽样数必须大于等于1")
    ranked = events.assign(
        _sample_hash=events["announcement_id"].map(
            lambda value: hashlib.sha256(
                f"CNINFO_PERIODIC_REPORT_METADATA_V1|{value}".encode("utf-8")
            ).hexdigest()
        )
    ).sort_values("_sample_hash", kind="stable")
    return ranked.head(min(sample_size, len(ranked))).drop(columns="_sample_hash")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def render_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# A股沪深定期报告官方元数据采集 V1",
        "",
        f"- 状态：`{report['status']}`",
        f"- 证据截止日：`{report['evidence_cutoff']}`",
        f"- 采集时间：`{report['started_at']}` 至 `{report['completed_at']}`",
        "- 阶段：只采集官方事件元数据；未读取市场价格或未来收益。",
        "",
        "## 完整性",
        "",
        f"- 自然月分片：{metrics['completed_intervals']}/{metrics['total_intervals']} 完成",
        f"- 官方原始记录：{metrics['raw_record_count']}",
        f"- 全局唯一公告ID：{metrics['unique_announcement_id_count']}",
        f"- 原始完整定期报告事件：{metrics['event_count']}",
        f"- 覆盖证券：{metrics['event_symbol_count']}",
        f"- 官方内部日期一致率：{metrics['official_internal_date_agreement_ratio']:.6%}",
        f"- PDF抽检：{metrics['pdf_verified_count']}/{metrics['pdf_sample_count']} 通过",
        "",
        "## 验收门",
        "",
    ]
    for gate in report["gates"]:
        lines.append(
            f"- {'PASS' if gate['passed'] else 'FAIL'} `{gate['gate_id']}`：{gate['description']}"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本产物只能证明官方定期报告事件日，不证明供应商财务数值的历史vintage。",
            "- 本产物没有计算ORJ、IC、未来收益、组合、成本、持仓或订单。",
            "- 只有全部验收门通过后，才允许另行冻结唯一事件信号研究。",
            "",
            "## 产物",
            "",
            f"- 官方事件档案：`{report['artifacts']['normalized_event_archive']}`",
            f"- 分片清单：`{report['artifacts']['checkpoint_inventory']}`",
            f"- 采集收据：`{report['artifacts']['acquisition_receipt']}`",
            "",
        ]
    )
    return "\n".join(lines)

