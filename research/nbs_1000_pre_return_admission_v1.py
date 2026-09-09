"""510300国家统计局10时事件的官方日程采集与预收益准入。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

from research.stk_mins_source_admission_v1 import (
    atomic_json,
    atomic_parquet,
    atomic_text,
    canonical_json_bytes,
    load_protocol as load_minute_protocol,
    now_local,
    run_source_adjudication,
    sha256_bytes,
    sha256_file,
    verify_frozen_manifests as verify_minute_manifests,
)


MODEL_ID = "510300_NBS_1000_NEGATIVE_INFORMATION_DRIFT_V1"
TIMEZONE = ZoneInfo("Asia/Shanghai")
PROTOCOL_PATH = Path("config/510300_nbs_1000_negative_information_drift_v1.yaml")
PROTOCOL_MANIFEST_PATH = Path(
    "config/510300_nbs_1000_negative_information_drift_v1_manifest.json"
)
IMPLEMENTATION_MANIFEST_PATH = Path(
    "config/510300_nbs_1000_pre_return_implementation_manifest.json"
)
LEDGER_MANIFEST_PATH = Path(
    "reports/data_quality/510300_nbs_1000_negative_information_drift_v1/"
    "pre_return_event_ledger_manifest.json"
)
SUMMARY_PATH = Path(
    "reports/research/510300_nbs_1000_negative_information_drift_v1/"
    "PRE_RETURN_G0_G1_SUMMARY.md"
)


class NbsProtocolError(RuntimeError):
    """NBS冻结协议或实现完整性错误。"""


class NbsAcquisitionError(RuntimeError):
    """NBS官网采集错误。"""


class NbsParseError(RuntimeError):
    """NBS年度发布日程解析错误。"""


def load_protocol(project_root: Path) -> dict[str, Any]:
    """读取NBS策略冻结协议。"""

    payload = yaml.safe_load((project_root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NbsProtocolError("NBS协议YAML顶层必须是对象")
    identity = payload.get("protocol", {})
    if identity.get("model_id") != MODEL_ID:
        raise NbsProtocolError("NBS协议MODEL_ID不匹配")
    if identity.get("version") != "1.0.1":
        raise NbsProtocolError("NBS协议版本必须为1.0.1")
    if identity.get("state") != "FROZEN_BEFORE_EVENT_RETURN_READ":
        raise NbsProtocolError("NBS协议没有处于预收益冻结状态")
    return payload


def _verify_hash_mapping(project_root: Path, mapping: Mapping[str, str]) -> None:
    for relative, expected in mapping.items():
        path = project_root / Path(relative)
        if not path.is_file():
            raise NbsProtocolError(f"冻结文件缺失：{relative}")
        actual = sha256_file(path)
        if actual != str(expected).lower():
            raise NbsProtocolError(
                f"冻结文件哈希漂移：{relative}；expected={expected}；actual={actual}"
            )


def verify_protocol_manifest(project_root: Path) -> dict[str, Any]:
    """验证策略协议、来源依赖和固定外部输入。"""

    path = project_root / PROTOCOL_MANIFEST_PATH
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID:
        raise NbsProtocolError("NBS协议清单MODEL_ID不匹配")
    if manifest.get("version") != "1.0.1":
        raise NbsProtocolError("NBS协议清单版本必须为1.0.1")
    if manifest.get("freeze_state") != "PROTOCOL_FROZEN_BEFORE_EVENT_RETURN_READ":
        raise NbsProtocolError("NBS协议清单冻结状态不正确")
    _verify_hash_mapping(project_root, manifest.get("frozen_files", {}))
    dependency = manifest["source_dependency"]
    source_manifest_path = project_root / Path(dependency["manifest_path"])
    if sha256_file(source_manifest_path) != dependency["manifest_sha256"]:
        raise NbsProtocolError("分钟来源协议清单哈希漂移")
    _verify_hash_mapping(project_root, manifest.get("fixed_external_inputs", {}))
    return manifest


def verify_frozen_manifests(
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """联网或构造账本前验证策略、分钟来源和实现清单。"""

    protocol_manifest = verify_protocol_manifest(project_root)
    verify_minute_manifests(project_root)
    path = project_root / IMPLEMENTATION_MANIFEST_PATH
    if not path.is_file():
        raise NbsProtocolError("缺少NBS预收益实现冻结清单")
    implementation = json.loads(path.read_text(encoding="utf-8"))
    if implementation.get("model_id") != MODEL_ID:
        raise NbsProtocolError("NBS预收益实现清单MODEL_ID不匹配")
    if implementation.get("freeze_state") != "PRE_RETURN_IMPLEMENTATION_FROZEN":
        raise NbsProtocolError("NBS预收益实现未冻结")
    if implementation.get("protocol_manifest_sha256") != sha256_file(
        project_root / PROTOCOL_MANIFEST_PATH
    ):
        raise NbsProtocolError("NBS实现清单引用的协议清单哈希不匹配")
    _verify_hash_mapping(project_root, implementation.get("implementation_files", {}))
    return protocol_manifest, implementation


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).replace("\u3000", " ").split())


def _allowed_url(url: str, allowed_hosts: Sequence[str]) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in set(allowed_hosts)


def _request_official(
    session: requests.Session,
    url: str,
    allowed_hosts: Sequence[str],
    maximum_attempts: int = 4,
) -> tuple[bytes, str, str]:
    """获取官方HTML；仅重试传输错误、429和5xx。"""

    if not _allowed_url(url, allowed_hosts):
        raise NbsAcquisitionError(f"URL不在冻结官方域名内：{url}")
    last_error = ""
    for attempt in range(1, maximum_attempts + 1):
        try:
            response = session.get(
                url,
                timeout=(15, 60),
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Encoding": "gzip",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                    "User-Agent": f"{MODEL_ID}/1.0 pre-return-source-admission",
                },
                allow_redirects=True,
            )
            final_url = str(response.url)
            if not _allowed_url(final_url, allowed_hosts):
                raise NbsAcquisitionError(f"官网请求被重定向到非准入域名：{final_url}")
            if response.status_code == 200:
                content = bytes(response.content)
                if not content:
                    raise NbsAcquisitionError(f"官网成功响应为空：{url}")
                try:
                    content.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise NbsAcquisitionError(f"官网页面不是UTF-8：{url}") from exc
                return content, final_url, now_local().isoformat()
            last_error = f"HTTP_{response.status_code}"
            if response.status_code != 429 and response.status_code < 500:
                break
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = type(exc).__name__
        if attempt < maximum_attempts:
            time.sleep(float(2 ** (attempt - 1)))
    raise NbsAcquisitionError(f"官网采集失败：{url}；{last_error}")


def discover_schedule_links(
    index_content: bytes,
    index_url: str,
    allowed_hosts: Sequence[str],
    start_year: int,
    end_year: int,
) -> dict[int, str]:
    """从官方目录发现每个年度日程页。"""

    soup = BeautifulSoup(index_content, "html.parser", from_encoding="utf-8")
    pattern = re.compile(r"(20\d{2})年国家统计局主要统计信息发布日程表")
    links: dict[int, str] = {}
    for anchor in soup.select("a[href]"):
        text = _normalize_text(anchor.get_text(" ", strip=True))
        match = pattern.search(text)
        if not match:
            continue
        year = int(match.group(1))
        if not start_year <= year <= end_year:
            continue
        target = urljoin(index_url, str(anchor["href"]))
        if not _allowed_url(target, allowed_hosts):
            raise NbsAcquisitionError(f"年度日程链接越出官方域名：{target}")
        if year in links and links[year] != target:
            raise NbsAcquisitionError(f"年度日程目录出现冲突链接：{year}")
        links[year] = target
    expected = set(range(start_year, end_year + 1))
    if set(links) != expected:
        missing = sorted(expected - set(links))
        extra = sorted(set(links) - expected)
        raise NbsAcquisitionError(f"年度日程目录不完整：missing={missing} extra={extra}")
    return dict(sorted(links.items()))


def _inventory_without_fingerprint(inventory: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(inventory)
    payload.pop("inventory_fingerprint", None)
    return payload


def validate_source_inventory(
    project_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """验证官方原始页清单和全部文件哈希。"""

    path = project_root / Path(protocol["nbs_event_contract"]["source_inventory"])
    if not path.is_file():
        raise NbsAcquisitionError("缺少NBS年度日程来源清单")
    inventory = json.loads(path.read_text(encoding="utf-8"))
    expected_fingerprint = inventory.get("inventory_fingerprint")
    actual_fingerprint = sha256_bytes(
        canonical_json_bytes(_inventory_without_fingerprint(inventory))
    )
    if expected_fingerprint != actual_fingerprint:
        raise NbsAcquisitionError("NBS来源清单指纹不匹配")
    records = inventory.get("records", [])
    if len(records) != 11:
        raise NbsAcquisitionError("NBS来源清单必须包含1个目录页和10个年度页")
    years = sorted(int(record["year"]) for record in records if record["kind"] == "ANNUAL")
    if years != list(range(2017, 2027)):
        raise NbsAcquisitionError(f"NBS年度页不完整：{years}")
    for record in records:
        raw_path = project_root / Path(record["raw_relative_path"])
        if not raw_path.is_file():
            raise NbsAcquisitionError(f"NBS原始页缺失：{raw_path}")
        if sha256_file(raw_path) != record["sha256"]:
            raise NbsAcquisitionError(f"NBS原始页哈希漂移：{raw_path}")
        if raw_path.stat().st_size != int(record["bytes"]):
            raise NbsAcquisitionError(f"NBS原始页大小漂移：{raw_path}")
    return inventory


def acquire_official_schedules(
    project_root: Path,
    protocol: Mapping[str, Any],
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """采集目录和2017至2026年度日程，保留不可变原始HTML。"""

    inventory_path = project_root / Path(
        protocol["nbs_event_contract"]["source_inventory"]
    )
    if inventory_path.is_file():
        existing = validate_source_inventory(project_root, protocol)
        reused = dict(existing)
        reused["state"] = "PASS_NBS_SCHEDULE_ACQUISITION_REUSED"
        return reused

    contract = protocol["nbs_event_contract"]
    allowed_hosts = [str(value) for value in contract["allowed_hosts"]]
    index_url = str(contract["schedule_index"])
    client = session or requests.Session()
    index_content, final_index_url, index_retrieved = _request_official(
        client, index_url, allowed_hosts
    )
    links = discover_schedule_links(
        index_content,
        final_index_url,
        allowed_hosts,
        2017,
        2026,
    )
    raw_root = project_root / Path(contract["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    retrieval_id = now_local().strftime("retrieved_at=%Y%m%dT%H%M%S%f%z")
    staging = raw_root / f".staging-{uuid.uuid4().hex}"
    final_directory = raw_root / retrieval_id
    staging.mkdir(parents=False, exist_ok=False)
    records: list[dict[str, Any]] = []

    index_file = staging / "schedule_index.html"
    index_file.write_bytes(index_content)
    records.append(
        {
            "kind": "INDEX",
            "year": None,
            "requested_url": index_url,
            "final_url": final_index_url,
            "retrieved_at": index_retrieved,
            "raw_relative_path": (
                final_directory / index_file.name
            ).relative_to(project_root).as_posix(),
            "bytes": len(index_content),
            "sha256": sha256_bytes(index_content),
        }
    )
    for number, (year, url) in enumerate(links.items(), start=1):
        if number > 1:
            time.sleep(0.5)
        content, final_url, retrieved_at = _request_official(
            client, url, allowed_hosts
        )
        filename = f"year={year}.html"
        (staging / filename).write_bytes(content)
        records.append(
            {
                "kind": "ANNUAL",
                "year": year,
                "requested_url": url,
                "final_url": final_url,
                "retrieved_at": retrieved_at,
                "raw_relative_path": (
                    final_directory / filename
                ).relative_to(project_root).as_posix(),
                "bytes": len(content),
                "sha256": sha256_bytes(content),
            }
        )
        print(f"国家统计局年度日程 {year}：已保存，bytes={len(content)}", flush=True)
    os.replace(staging, final_directory)
    inventory = {
        "model_id": MODEL_ID,
        "phase": "NBS_OFFICIAL_SCHEDULE_ACQUISITION",
        "state": "PASS_NBS_SCHEDULE_ACQUISITION",
        "generated_at": now_local().isoformat(),
        "schedule_index": index_url,
        "allowed_hosts": allowed_hosts,
        "years": list(range(2017, 2027)),
        "raw_retrieval_directory": final_directory.relative_to(project_root).as_posix(),
        "records": records,
        "all_raw_hashes_present": True,
        "event_return_reads": 0,
        "model_training_run": False,
        "portfolio_evaluation_run": False,
        "position_impact": 0,
    }
    inventory["inventory_fingerprint"] = sha256_bytes(canonical_json_bytes(inventory))
    atomic_json(inventory, inventory_path)
    return inventory


def _parse_chinese_date(value: str) -> pd.Timestamp:
    match = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", value)
    if not match:
        raise NbsParseError(f"无法解析官方成文日期：{value}")
    return pd.Timestamp(date(*(int(item) for item in match.groups())))


def _publication_metadata(content: bytes, year: int) -> tuple[pd.Timestamp, str]:
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    text = _normalize_text(soup.get_text(" ", strip=True))
    title = f"{year}年国家统计局主要统计信息发布日程表"
    if title not in text and not (
        "信息类别 发布日程表" in text and "发布机构 国家统计局" in text
    ):
        raise NbsParseError(f"年度页标题和官方元数据均不匹配：{year}")
    date_match = re.search(
        r"成文日期\s*(\d{4}年\d{1,2}月\d{1,2}日)",
        text,
    )
    if not date_match:
        date_match = re.search(
            r"(\d{4})/(\d{1,2})/(\d{1,2})\s+\d{1,2}:\d{2}",
            text,
        )
        if not date_match:
            raise NbsParseError(f"年度页缺少成文日期：{year}")
        publication = pd.Timestamp(
            date(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))
        )
    else:
        publication = _parse_chinese_date(date_match.group(1))
    index_match = re.search(r"索\s*引\s*号\s*([0-9A-Za-z-]+)", text)
    return publication, index_match.group(1) if index_match else ""


def _canonical_schedule_rows(
    content: bytes,
    canonical_titles: Sequence[str],
) -> tuple[str, list[str], list[str]]:
    try:
        tables = pd.read_html(BytesIO(content), encoding="utf-8")
    except ValueError as exc:
        raise NbsParseError("年度页没有可解析表格") from exc
    canonical_set = set(canonical_titles)
    candidates: list[list[str]] = []
    for table in tables:
        if table.shape[1] < 14:
            continue
        normalized = table.iloc[:, :14].map(_normalize_text)
        for _, row in normalized.iterrows():
            values = row.tolist()
            if values[0] == "1" and values[1] in canonical_set:
                candidates.append(values)
    date_rows = [row for row in candidates if any("/" in cell for cell in row[2:14])]
    time_rows = [
        row
        for row in candidates
        if any(re.search(r"\d{1,2}\s*[:：]\s*\d{2}", cell) for cell in row[2:14])
    ]
    if len(date_rows) != 1 or len(time_rows) != 1:
        raise NbsParseError(
            f"国民经济运行情况日期行或时刻行不唯一：dates={len(date_rows)} times={len(time_rows)}"
        )
    if date_rows[0][1] != time_rows[0][1]:
        raise NbsParseError("国民经济运行情况日期行与时刻行标题不一致")
    return date_rows[0][1], date_rows[0][2:14], time_rows[0][2:14]


def _parse_day_cell(cell: str, year: int, month: int) -> tuple[pd.Timestamp | None, str]:
    normalized = _normalize_text(cell)
    if not normalized or normalized.lower() == "nan" or "…" in normalized:
        return None, ""
    match = re.search(r"(\d{1,2})\s*/\s*([一二三四五六日天])", normalized)
    if not match:
        raise NbsParseError(f"无法解析{year}-{month:02d}日期单元格：{normalized}")
    try:
        scheduled_date = pd.Timestamp(date(year, month, int(match.group(1))))
    except ValueError as exc:
        raise NbsParseError(f"年度日程包含非法日期：{year}-{month}-{match.group(1)}") from exc
    marker = match.group(2).replace("天", "日")
    weekday_markers = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}
    if weekday_markers[scheduled_date.weekday()] != marker:
        raise NbsParseError(
            f"年度日程星期标记不一致：{scheduled_date.date()} cell={normalized}"
        )
    return scheduled_date, marker


def _parse_time_cell(cell: str, year: int, month: int) -> tuple[str | None, str]:
    normalized = _normalize_text(cell)
    if not normalized or normalized.lower() == "nan" or "…" in normalized:
        return None, ""
    match = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})", normalized)
    if not match:
        raise NbsParseError(f"无法解析{year}-{month:02d}时刻单元格：{normalized}")
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise NbsParseError(f"年度日程包含非法时刻：{normalized}")
    return f"{hour:02d}:{minute:02d}:00", normalized


def parse_annual_schedule(
    content: bytes,
    year: int,
    source_url: str,
    document_sha256: str,
    canonical_titles: Sequence[str],
) -> list[dict[str, Any]]:
    """解析单个年度页中唯一的国民经济运行情况事件家族。"""

    publication_date, official_index_number = _publication_metadata(content, year)
    title, date_cells, time_cells = _canonical_schedule_rows(content, canonical_titles)
    rows: list[dict[str, Any]] = []
    for month in range(1, 13):
        scheduled_date, weekday_marker = _parse_day_cell(
            date_cells[month - 1], year, month
        )
        scheduled_time, raw_time_cell = _parse_time_cell(
            time_cells[month - 1], year, month
        )
        if scheduled_date is None and scheduled_time is None:
            continue
        if scheduled_date is None or scheduled_time is None:
            raise NbsParseError(f"{year}-{month:02d}日期与时刻一方缺失")
        scheduled_at = pd.Timestamp(
            f"{scheduled_date.date().isoformat()} {scheduled_time}"
        )
        initial_before_year = publication_date.date() < date(year, 1, 1)
        rows.append(
            {
                "schedule_year": year,
                "schedule_month": month,
                "event_title": title,
                "scheduled_date": scheduled_date,
                "scheduled_time": scheduled_time,
                "scheduled_at": scheduled_at,
                "schedule_day_cell": date_cells[month - 1],
                "schedule_time_cell": raw_time_cell,
                "weekday_marker": weekday_marker,
                "schedule_publication_date": publication_date,
                "schedule_publication_before_event": bool(
                    publication_date.date() < scheduled_date.date()
                ),
                "schedule_source_url": source_url,
                "schedule_source_url_hash": sha256_bytes(source_url.encode("utf-8")),
                "schedule_document_sha256": document_sha256,
                "schedule_official_index_number": official_index_number,
                "schedule_snapshot_state": (
                    "ORIGINAL_PRE_YEAR_OFFICIAL_SCHEDULE"
                    if initial_before_year
                    else "IN_YEAR_OFFICIAL_SNAPSHOT_OR_REVISION"
                ),
                "original_scheduled_at": scheduled_at if initial_before_year else pd.NaT,
                "revised_scheduled_at": scheduled_at if not initial_before_year else pd.NaT,
                "revision_known_at": publication_date if not initial_before_year else pd.NaT,
            }
        )
    if not 10 <= len(rows) <= 11:
        raise NbsParseError(f"{year}年度国民经济运行情况应有10至11个事件，实际{len(rows)}")
    return rows


def _load_minute_quality_ledger(project_root: Path) -> pd.DataFrame:
    path = (
        project_root
        / "data/curated/510300_stk_mins_source_admission_v1/daily_quality_ledger.parquet"
    )
    if not path.is_file():
        raise NbsParseError("缺少分钟来源日级质量账本")
    frame = pd.read_parquet(path)
    required = {
        "trade_date",
        "exact_standard_241_labels",
        "pre_window_complete",
        "reaction_window_complete",
        "entry_window_complete",
        "exit_window_complete",
    }
    if not required.issubset(frame.columns):
        raise NbsParseError(f"分钟质量账本缺字段：{sorted(required - set(frame.columns))}")
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    if frame["trade_date"].isna().any() or frame["trade_date"].duplicated().any():
        raise NbsParseError("分钟质量账本日期无效或重复")
    return frame


def _event_exclusion_reason(
    row: Mapping[str, Any],
    start_date: date,
    end_date: date,
) -> str:
    scheduled_date = pd.Timestamp(row["scheduled_date"]).date()
    if scheduled_date < start_date:
        return "EXCLUDED_OUT_OF_SCOPE_BEFORE_START_DATE"
    if scheduled_date > end_date:
        return "EXCLUDED_OUT_OF_SCOPE_AFTER_END_DATE"
    if row["scheduled_time"] != "10:00:00":
        return "EXCLUDED_NON_1000_RELEASE"
    if not bool(row["schedule_publication_before_event"]):
        return "EXCLUDED_UNVERIFIABLE_SCHEDULE_REVISION"
    if not bool(row["sse_open_day"]):
        return "EXCLUDED_NON_TRADING_DAY"
    if row["minute_data_state"] != "PASS_FOUR_WINDOWS_COMPLETE":
        return "NO_VIEW_INCOMPLETE_MINUTE_WINDOW"
    return "PASS_ELIGIBLE_PRE_RETURN"


def build_pre_return_event_ledger(
    project_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只构造事件资格；不访问分钟价格、收益或标签。"""

    inventory = validate_source_inventory(project_root, protocol)
    annual_records = {
        int(record["year"]): record
        for record in inventory["records"]
        if record["kind"] == "ANNUAL"
    }
    contract = protocol["nbs_event_contract"]
    parsed_rows: list[dict[str, Any]] = []
    parse_counts: dict[str, int] = {}
    for year in range(2017, 2027):
        record = annual_records[year]
        raw_path = project_root / Path(record["raw_relative_path"])
        rows = parse_annual_schedule(
            raw_path.read_bytes(),
            year,
            str(record["final_url"]),
            str(record["sha256"]),
            [str(value) for value in contract["canonical_titles"]],
        )
        parsed_rows.extend(rows)
        parse_counts[str(year)] = len(rows)
    ledger = pd.DataFrame(parsed_rows).sort_values("scheduled_at").reset_index(drop=True)
    if ledger.duplicated(["scheduled_date"]).any():
        raise NbsParseError("同一日期出现多个国民经济运行情况日程事件")

    quality = _load_minute_quality_ledger(project_root)
    quality["date_key"] = quality["trade_date"].dt.date
    quality_by_date = quality.set_index("date_key")
    window_columns = [
        "pre_window_complete",
        "reaction_window_complete",
        "entry_window_complete",
        "exit_window_complete",
    ]
    open_dates = set(quality_by_date.index)
    complete_open_dates = {
        index
        for index, row in quality_by_date.iterrows()
        if bool(row["exact_standard_241_labels"])
        and all(bool(row[column]) for column in window_columns)
    }
    ledger["sse_open_day"] = ledger["scheduled_date"].dt.date.isin(open_dates)
    ledger["minute_data_state"] = ledger["scheduled_date"].dt.date.map(
        lambda value: (
            "PASS_FOUR_WINDOWS_COMPLETE"
            if value in complete_open_dates
            else "NOT_SSE_OPEN_DAY"
            if value not in open_dates
            else "NO_VIEW_INCOMPLETE_MINUTE_WINDOW"
        )
    )
    start = pd.Timestamp(protocol["protocol"]["start_date"]).date()
    end = pd.Timestamp(protocol["protocol"]["end_date"]).date()
    ledger["exclusion_reason"] = ledger.apply(
        lambda row: _event_exclusion_reason(row, start, end), axis=1
    )
    ledger["final_event_eligibility"] = ledger["exclusion_reason"].eq(
        "PASS_ELIGIBLE_PRE_RETURN"
    )
    ledger["event_id"] = ledger.apply(
        lambda row: (
            f"NBS_NATIONAL_ECONOMY_{row['scheduled_date']:%Y%m%d}_"
            f"{str(row['scheduled_time']).replace(':', '')[:4]}"
        ),
        axis=1,
    )

    family_event_dates = {
        value.date()
        for value in ledger.loc[
            ledger["scheduled_date"].dt.date.between(start, end), "scheduled_date"
        ]
    }
    valid_non_event_dates = sorted(complete_open_dates - family_event_dates)
    baseline_counts: list[int] = []
    for row in ledger.itertuples(index=False):
        event_date = row.scheduled_date.date()
        baseline_counts.append(sum(value < event_date for value in valid_non_event_dates))
    ledger["strictly_prior_complete_non_event_days"] = baseline_counts
    required_baseline = int(protocol["feature"]["causal_baseline_days"])
    ledger["causal_baseline_precheck_pass"] = (
        ledger["strictly_prior_complete_non_event_days"] >= required_baseline
    )
    ledger["model_pre_return_eligibility"] = (
        ledger["final_event_eligibility"] & ledger["causal_baseline_precheck_pass"]
    )
    ledger["model_pre_return_exclusion_reason"] = ledger.apply(
        lambda row: (
            "PASS_MODEL_PRE_RETURN_ELIGIBILITY"
            if row["model_pre_return_eligibility"]
            else "NO_VIEW_INSUFFICIENT_CAUSAL_BASELINE"
            if row["final_event_eligibility"]
            and not row["causal_baseline_precheck_pass"]
            else row["exclusion_reason"]
        ),
        axis=1,
    )
    ledger["model_event_ordinal"] = pd.Series(pd.NA, index=ledger.index, dtype="Int64")
    eligible_index = ledger.index[ledger["model_pre_return_eligibility"]]
    ledger.loc[eligible_index, "model_event_ordinal"] = range(1, len(eligible_index) + 1)
    ledger["era"] = "NOT_MODEL_ELIGIBLE"
    ordinals = ledger["model_event_ordinal"]
    ledger.loc[ordinals.between(1, 36, inclusive="both").fillna(False), "era"] = (
        "TRAINING_ORIGIN"
    )
    ledger.loc[ordinals.between(37, 56, inclusive="both").fillna(False), "era"] = "ERA_1"
    ledger.loc[ordinals.between(57, 76, inclusive="both").fillna(False), "era"] = "ERA_2"
    ledger.loc[ordinals.ge(77).fillna(False), "era"] = "ERA_3"

    ordered_columns = [
        "event_id",
        "schedule_year",
        "schedule_month",
        "event_title",
        "scheduled_date",
        "scheduled_time",
        "scheduled_at",
        "schedule_publication_date",
        "schedule_publication_before_event",
        "schedule_source_url",
        "schedule_source_url_hash",
        "schedule_document_sha256",
        "schedule_official_index_number",
        "schedule_snapshot_state",
        "original_scheduled_at",
        "revised_scheduled_at",
        "revision_known_at",
        "schedule_day_cell",
        "schedule_time_cell",
        "weekday_marker",
        "sse_open_day",
        "minute_data_state",
        "strictly_prior_complete_non_event_days",
        "causal_baseline_precheck_pass",
        "final_event_eligibility",
        "exclusion_reason",
        "model_pre_return_eligibility",
        "model_pre_return_exclusion_reason",
        "model_event_ordinal",
        "era",
    ]
    ledger = ledger[ordered_columns]
    pre_return_path = project_root / Path(contract["pre_return_ledger"])
    event_ledger_path = project_root / Path(contract["curated_event_ledger"])
    atomic_parquet(ledger, pre_return_path)
    atomic_parquet(ledger, event_ledger_path)

    reason_counts = {
        str(key): int(value)
        for key, value in ledger["exclusion_reason"].value_counts().sort_index().items()
    }
    manifest = {
        "model_id": MODEL_ID,
        "phase": "PRE_RETURN_EVENT_LEDGER",
        "state": "PASS_PRE_RETURN_EVENT_LEDGER_BUILT",
        "generated_at": now_local().isoformat(),
        "source_inventory_fingerprint": inventory["inventory_fingerprint"],
        "source_inventory_sha256": sha256_file(
            project_root / Path(contract["source_inventory"])
        ),
        "annual_parse_counts": parse_counts,
        "event_rows": int(len(ledger)),
        "in_scope_rows": int(
            ledger["scheduled_date"].dt.date.between(start, end).sum()
        ),
        "schedule_eligible_rows": int(ledger["final_event_eligibility"].sum()),
        "model_pre_return_eligible_rows": int(
            ledger["model_pre_return_eligibility"].sum()
        ),
        "exclusion_reason_counts": reason_counts,
        "pre_return_ledger": pre_return_path.relative_to(project_root).as_posix(),
        "pre_return_ledger_sha256": sha256_file(pre_return_path),
        "event_ledger": event_ledger_path.relative_to(project_root).as_posix(),
        "event_ledger_sha256": sha256_file(event_ledger_path),
        "event_return_reads": 0,
        "event_labels_created": 0,
        "model_training_run": False,
        "portfolio_evaluation_run": False,
        "position_impact": 0,
    }
    fingerprint_payload = dict(manifest)
    fingerprint_payload.pop("generated_at")
    manifest["ledger_fingerprint"] = sha256_bytes(
        canonical_json_bytes(fingerprint_payload)
    )
    atomic_json(manifest, project_root / LEDGER_MANIFEST_PATH)
    return ledger, manifest


def _validate_ledger_manifest(
    project_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = project_root / LEDGER_MANIFEST_PATH
    if not path.is_file():
        raise NbsParseError("缺少预收益事件账本清单")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("ledger_fingerprint")
    payload = dict(manifest)
    payload.pop("generated_at", None)
    payload.pop("ledger_fingerprint", None)
    if expected != sha256_bytes(canonical_json_bytes(payload)):
        raise NbsParseError("预收益事件账本清单指纹不匹配")
    ledger_path = project_root / Path(
        protocol["nbs_event_contract"]["pre_return_ledger"]
    )
    if sha256_file(ledger_path) != manifest["pre_return_ledger_sha256"]:
        raise NbsParseError("预收益事件账本哈希漂移")
    ledger = pd.read_parquet(ledger_path)
    if len(ledger) != int(manifest["event_rows"]):
        raise NbsParseError("预收益事件账本行数漂移")
    return ledger, manifest


def _era_counts(ledger: pd.DataFrame) -> dict[str, int]:
    return {
        name: int(ledger["era"].eq(name).sum())
        for name in ("TRAINING_ORIGIN", "ERA_1", "ERA_2", "ERA_3")
    }


def adjudicate_g0_g1(
    project_root: Path,
    protocol: Mapping[str, Any],
    replay_expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """只统计G0/G1；不读取价格、收益、标签或模型指标。"""

    ledger, ledger_manifest = _validate_ledger_manifest(project_root, protocol)
    source_report_path = (
        project_root
        / "reports/data_quality/510300_stk_mins_source_admission_v1/source_adjudication.json"
    )
    if not source_report_path.is_file():
        raise NbsParseError("缺少分钟来源裁决")
    source = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_coverage = source["coverage"]
    g0_rules = protocol["gates"]["g0"]
    g0_checks = {
        "minute_source_general_admission_pass": source.get("state")
        == protocol["dependencies"]["minute_source"]["required_state"],
        "open_day_minute_coverage": float(source_coverage["open_day_any_data_coverage"])
        >= float(g0_rules["open_day_minute_coverage_minimum"]),
        "standard_241_bar_day_coverage": float(
            source_coverage["standard_241_bar_day_coverage"]
        )
        >= float(g0_rules["standard_241_bar_day_coverage_minimum"]),
        "eligible_event_four_window_coverage": source_coverage[
            "eligible_event_four_window_coverage"
        ]
        is not None
        and float(source_coverage["eligible_event_four_window_coverage"])
        >= float(g0_rules["eligible_event_four_window_coverage_minimum"]),
        "daily_ohlc_reconciliation_rate": float(
            source_coverage["daily_ohlc_reconciliation_rate"]
        )
        >= float(g0_rules["daily_ohlc_reconciliation_rate_minimum"]),
        "timestamp_semantics_evidence": source["timestamp_semantics_evidence"]["state"]
        == "PASS_BAR_END_DOCUMENT_AND_SESSION_EVIDENCE",
        "nbs_schedule_raw_hashes_complete": bool(
            validate_source_inventory(project_root, protocol)["all_raw_hashes_present"]
        ),
    }
    g0_pass = all(g0_checks.values())

    start = pd.Timestamp(protocol["protocol"]["start_date"]).date()
    end = pd.Timestamp(protocol["protocol"]["end_date"]).date()
    in_scope = ledger[ledger["scheduled_date"].dt.date.between(start, end)]
    eligible = ledger[ledger["final_event_eligibility"]]
    model_eligible = ledger[ledger["model_pre_return_eligibility"]]
    era_counts = _era_counts(ledger)
    g1_rules = protocol["gates"]["g1"]
    eligible_count = int(len(eligible))
    complete_count = int(
        eligible["minute_data_state"].eq("PASS_FOUR_WINDOWS_COMPLETE").sum()
    )
    model_count = int(len(model_eligible))
    evaluation_count = max(model_count - int(protocol["models"]["minimum_training_events"]), 0)
    g1_checks = {
        "eligible_1000_events": eligible_count
        >= int(g1_rules["eligible_1000_events_minimum"]),
        "complete_window_events": complete_count
        >= int(g1_rules["complete_window_events_minimum"]),
        "training_events": model_count >= int(g1_rules["training_events_minimum"]),
        "era_1_events": era_counts["ERA_1"] >= int(g1_rules["era_1_events_minimum"]),
        "era_2_events": era_counts["ERA_2"] >= int(g1_rules["era_2_events_minimum"]),
        "era_3_events": era_counts["ERA_3"] >= int(g1_rules["era_3_events_minimum"]),
        "evaluation_events": evaluation_count
        >= int(g1_rules["evaluation_events_minimum"]),
    }
    g1_pass = all(g1_checks.values())
    if not g0_pass:
        state = "BLOCKED_G0_SOURCE_ADMISSION"
    elif not g1_pass:
        state = str(g1_rules["failure_state"])
    else:
        state = "PASS_G0_G1_PRE_RETURN_ADMISSION"

    exclusion_counts = {
        str(key): int(value)
        for key, value in in_scope["exclusion_reason"].value_counts().sort_index().items()
    }
    counts = {
        "total_schedule_rows_all_annual_pages": int(len(ledger)),
        "total_schedule_rows_in_scope": int(len(in_scope)),
        "scheduled_1000_rows_in_scope": int(in_scope["scheduled_time"].eq("10:00:00").sum()),
        "non_1000_rows_in_scope": int(
            in_scope["exclusion_reason"].eq("EXCLUDED_NON_1000_RELEASE").sum()
        ),
        "non_trading_day_rows_in_scope": int(
            in_scope["exclusion_reason"].eq("EXCLUDED_NON_TRADING_DAY").sum()
        ),
        "unverifiable_schedule_revision_rows_in_scope": int(
            in_scope["exclusion_reason"]
            .eq("EXCLUDED_UNVERIFIABLE_SCHEDULE_REVISION")
            .sum()
        ),
        "eligible_1000_events": eligible_count,
        "eligible_complete_window_events": complete_count,
        "eligible_but_insufficient_60_day_baseline": int(
            (
                ledger["final_event_eligibility"]
                & ~ledger["causal_baseline_precheck_pass"]
            ).sum()
        ),
        "model_pre_return_eligible_events": model_count,
        "evaluation_events_after_first_36": evaluation_count,
        "era_counts": era_counts,
        "exclusion_reason_counts_in_scope": exclusion_counts,
    }
    report = {
        "model_id": MODEL_ID,
        "phase": "G0_G1_PRE_RETURN_ADJUDICATION",
        "state": state,
        "generated_at": now_local().isoformat(),
        "g0_pass": g0_pass,
        "g0_checks": g0_checks,
        "g1_pass": g1_pass,
        "g1_checks": g1_checks,
        "counts": counts,
        "minute_source": {
            "state": source["state"],
            "decision_fingerprint": source["decision_fingerprint"],
            "open_day_minute_coverage": source_coverage["open_day_any_data_coverage"],
            "standard_241_bar_day_coverage": source_coverage[
                "standard_241_bar_day_coverage"
            ],
            "daily_ohlc_reconciliation_rate": source_coverage[
                "daily_ohlc_reconciliation_rate"
            ],
            "eligible_event_four_window_coverage": source_coverage[
                "eligible_event_four_window_coverage"
            ],
        },
        "nbs_schedule": {
            "source_inventory_fingerprint": ledger_manifest[
                "source_inventory_fingerprint"
            ],
            "ledger_fingerprint": ledger_manifest["ledger_fingerprint"],
            "all_raw_hashes_present": True,
        },
        "return_evaluation": "NOT_ALLOWED",
        "event_return_reads": 0,
        "event_labels_created": 0,
        "model_training_run": False,
        "g2_run": False,
        "g3_run": False,
        "g4_run": False,
        "sharpe_calculated": False,
        "order_generated": False,
        "position_impact": 0,
    }
    fingerprint_payload = dict(report)
    fingerprint_payload.pop("generated_at")
    report["decision_fingerprint"] = sha256_bytes(
        canonical_json_bytes(fingerprint_payload)
    )
    if replay_expected_fingerprint is not None:
        if report["decision_fingerprint"] != replay_expected_fingerprint:
            raise NbsParseError(
                "G0/G1干净进程重放指纹不一致："
                f"expected={replay_expected_fingerprint} actual={report['decision_fingerprint']}"
            )
        report["replay_state"] = "PASS_CLEAN_NEW_PROCESS_REPLAY"
        return report
    output_path = project_root / Path(protocol["outputs"]["g0_g1"])
    atomic_json(report, output_path)
    atomic_text(_summary_markdown(report), project_root / SUMMARY_PATH)
    return report


def _summary_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    source = report["minute_source"]
    era = counts["era_counts"]
    return f"""# 510300国家统计局10时负向信息漂移V1：预收益G0/G1裁决

生成时间：{report['generated_at']}

```text
STATE={report['state']}
G0_PASS={str(report['g0_pass']).upper()}
G1_PASS={str(report['g1_pass']).upper()}
RETURN_EVALUATION=NOT_ALLOWED
EVENT_RETURN_READS=0
MODEL_TRAINING_RUN=FALSE
G2_RUN=FALSE
G3_RUN=FALSE
G4_RUN=FALSE
POSITION_IMPACT=0
```

## 分钟来源

- 来源状态：{source['state']}
- 开放日任意分钟覆盖率：{source['open_day_minute_coverage']:.4%}
- 标准241根覆盖率：{source['standard_241_bar_day_coverage']:.4%}
- 日级OHLC对账率：{source['daily_ohlc_reconciliation_rate']:.4%}
- 合格事件四窗口覆盖率：{source['eligible_event_four_window_coverage']}

## 官方事件账本

- 研究期内年度日程事件：{counts['total_schedule_rows_in_scope']}
- 日程标示10:00：{counts['scheduled_1000_rows_in_scope']}
- 非10:00：{counts['non_1000_rows_in_scope']}
- 非交易日：{counts['non_trading_day_rows_in_scope']}
- 无法证明事前日程版本：{counts['unverifiable_schedule_revision_rows_in_scope']}
- 逐事件准入且四窗口完整：{counts['eligible_complete_window_events']}
- 60日前序非事件日预检后可建模：{counts['model_pre_return_eligible_events']}
- 首36个之后评价事件：{counts['evaluation_events_after_first_36']}
- 时代：训练{era['TRAINING_ORIGIN']}，Era1={era['ERA_1']}，Era2={era['ERA_2']}，Era3={era['ERA_3']}

本阶段只使用官方日程、交易日和分钟完整性布尔字段。没有读取窗口价格、事件收益或标签，也没有训练模型、生成净值、Sharpe、仓位或订单。
"""


def run_all_pre_return(
    project_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """依次采集、建账、回填事件窗口来源门并裁决G0/G1。"""

    acquire_official_schedules(project_root, protocol)
    _, ledger_manifest = build_pre_return_event_ledger(project_root, protocol)
    minute_protocol = load_minute_protocol(project_root)
    run_source_adjudication(
        project_root,
        minute_protocol,
        event_ledger_path=project_root
        / Path(protocol["nbs_event_contract"]["pre_return_ledger"]),
    )
    report = adjudicate_g0_g1(project_root, protocol)
    report["pre_return_ledger_fingerprint"] = ledger_manifest["ledger_fingerprint"]
    return report
