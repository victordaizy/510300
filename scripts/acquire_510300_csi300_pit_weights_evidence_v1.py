"""采集沪深300历史权重的可复验来源凭证并做数值级交叉核验。

本脚本只访问指数成分、指数权重、来源说明和来源授权清单。它不会读取价格、
收益、未来标签、模型输出、仓位或订单。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.csi300_pit_membership_weights_source_remediation_v1 import (  # noqa: E402
    REMEDIATION_ID,
    normalize_symbol,
    sha256_file,
)


RAW_ROOT = ROOT / "data" / "raw" / "510300_csi300_pit_membership_weights_source_remediation_v1"
CURATED_ROOT = ROOT / "data" / "curated" / "510300_csi300_pit_membership_weights_source_remediation_v1"
EXISTING_WEIGHT_PATH = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
SECOND_RETRIEVAL_PATH = CURATED_ROOT / "tushare_index_weight_second_retrieval.parquet"
OFFICIAL_CROSSCHECK_PATH = CURATED_ROOT / "official_weight_snapshot_crosscheck.parquet"
OUTPUT_MANIFEST_PATH = CURATED_ROOT / "pit_weight_source_evidence_manifest.json"

OFFICIAL_CURRENT_WEIGHT_URL = (
    "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/"
    "autofile/closeweight/000300closeweight.xls"
)
CSI_AUTHORIZED_AGENT_URL = (
    "https://www.csindex.com.cn/csindex-home/dataServer/queryAgentslist"
)
CSI_AUTHORIZED_USER_URL = (
    "https://www.csindex.com.cn/csindex-home/dataServer/queryAuthorizeddatauser"
)
CSI_WEB_APP_URL = "https://www.csindex.com.cn/static/js/app.5663471f.js"
CSI_VENDOR_CHUNK_URL = (
    "https://www.csindex.com.cn/static/js/chunk-40c40a5d.3f3115c8.js"
)
TUSHARE_INDEX_WEIGHT_DOC_URL = "https://tushare.pro/document/2?doc_id=96"
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"
ALLOWED_PROXY_ENDPOINTS = {
    "https://fast.xiaodefa.cn",
    "https://tt.xiaodefa.cn",
}


@dataclass(frozen=True)
class ParsedOfficialSnapshot:
    """一个从官方 closeweight 文件解析出的完整权重快照。"""

    snapshot_date: date
    frame: pd.DataFrame
    weight_sum_pct: float
    weight_decimal_places: int


class TushareRetrievalFailure(RuntimeError):
    """携带已归档失败响应的 Tushare 采集失败。"""

    def __init__(self, message: str, sources: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.sources = sources


def now_shanghai() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def archive_content_addressed(
    directory: Path,
    prefix: str,
    payload: bytes,
    suffix: str,
) -> tuple[Path, str]:
    digest = sha256_bytes(payload)
    path = directory / f"{prefix}_{digest}{suffix.lower()}"
    if path.exists():
        if sha256_file(path) != digest:
            raise RuntimeError(f"已有归档对象内容哈希异常：{path}")
    else:
        atomic_write_bytes(path, payload)
    return path, digest


def build_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.csindex.com.cn/",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def checked_get(
    session: requests.Session,
    url: str,
    timeout_seconds: float,
    validator: Callable[[bytes], bool] | None = None,
    attempts: int = 4,
) -> tuple[bytes, dict[str, Any]]:
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=timeout_seconds)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.content
            if not payload:
                raise RuntimeError("返回空内容")
            if validator is not None and not validator(payload):
                raise RuntimeError(
                    f"内容类型或文件签名不合格：{response.headers.get('Content-Type', '')}"
                )
            return payload, {
                "status_code": response.status_code,
                "content_type": str(response.headers.get("Content-Type") or ""),
                "final_url": str(response.url),
            }
        except (requests.RequestException, RuntimeError) as exc:
            errors.append(f"第{attempt}次：{type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"来源连续失败：{url}；{'；'.join(errors)}")


def is_excel_payload(payload: bytes) -> bool:
    return payload.startswith(bytes.fromhex("D0CF11E0A1B11AE1")) or payload.startswith(
        b"PK\x03\x04"
    )


def is_json_payload(payload: bytes) -> bool:
    try:
        json.loads(payload.decode("utf-8"))
        return True
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False


def source_record(
    kind: str,
    url: str,
    path: Path,
    payload: bytes,
    response_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "url": url,
        "method": "GET",
        "tls_verification": True,
        "archive_path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "response_status_code": int(response_meta["status_code"]),
        "response_content_type": str(response_meta["content_type"]),
        "response_final_url": str(response_meta["final_url"]),
        "retrieval_mode": "NETWORK_TLS_VERIFIED",
    }


def parse_official_closeweight(payload: bytes) -> ParsedOfficialSnapshot:
    if not is_excel_payload(payload):
        raise ValueError("官方权重文件不是可识别的 XLS/XLSX")
    frame = pd.read_excel(io.BytesIO(payload), dtype=object)
    if frame.shape[1] < 7:
        raise ValueError(f"官方权重文件列数不足：{frame.shape[1]}")

    def find_column(english_label: str, fallback_index: int) -> object:
        matches = [
            column
            for column in frame.columns
            if english_label.lower() in str(column).lower()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            exact = [
                column
                for column in matches
                if str(column).strip().lower().endswith(english_label.lower())
            ]
            if len(exact) == 1:
                return exact[0]
        if fallback_index >= frame.shape[1]:
            raise ValueError(
                f"官方权重文件无法定位字段 {english_label}，列为 {list(frame.columns)}"
            )
        return frame.columns[fallback_index]

    date_column = find_column("Date", 0)
    index_column = find_column("Index Code", 1)
    security_column = find_column("Constituent Code", 4 if frame.shape[1] >= 9 else 3)
    weight_column = find_column("Weight(%)", frame.shape[1] - 1)
    frame = frame[[date_column, index_column, security_column, weight_column]].copy()
    frame.columns = ["snapshot_date", "index_code", "security_code", "weight_pct"]
    frame["index_code"] = (
        frame["index_code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    )
    frame = frame.loc[frame["index_code"].eq("000300")].copy()
    if frame.empty:
        raise ValueError("官方权重文件没有 000300 记录")
    raw_dates = frame["snapshot_date"].astype(str).str.strip()
    frame["snapshot_date"] = pd.to_datetime(raw_dates, errors="raise").dt.date
    unique_dates = sorted(frame["snapshot_date"].unique())
    if len(unique_dates) != 1:
        raise ValueError(f"官方权重文件应只有一个快照日，实际为 {unique_dates}")
    frame["symbol"] = frame["security_code"].map(normalize_symbol)

    def decimal_places(value: object) -> int:
        text = f"{float(value):.12f}".rstrip("0").rstrip(".")
        return len(text.rsplit(".", 1)[1]) if "." in text else 0

    observed_decimal_places = max(frame["weight_pct"].map(decimal_places))
    frame["weight_pct"] = pd.to_numeric(frame["weight_pct"], errors="raise")
    if len(frame) != 300 or frame["symbol"].nunique() != 300:
        raise ValueError(
            f"官方权重文件不是 300 个唯一证券：行数={len(frame)}，唯一数={frame['symbol'].nunique()}"
        )
    if frame["symbol"].duplicated().any():
        raise ValueError("官方权重文件存在重复证券")
    weight_sum = float(frame["weight_pct"].sum())
    if not 98.0 <= weight_sum <= 102.0:
        raise ValueError(f"官方权重和超出 98%-102%：{weight_sum}")
    output = frame[["snapshot_date", "symbol", "weight_pct"]].sort_values("symbol").reset_index(drop=True)
    return ParsedOfficialSnapshot(
        snapshot_date=unique_dates[0],
        frame=output,
        weight_sum_pct=weight_sum,
        weight_decimal_places=observed_decimal_places,
    )


def archive_official_web_evidence(
    session: requests.Session,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    users: list[dict[str, Any]] = []
    specifications = (
        (
            "CSI_AUTHORIZED_STATIC_DATA_AGENTS",
            CSI_AUTHORIZED_AGENT_URL,
            "authorized_static_data_agents",
            ".json",
            is_json_payload,
        ),
        (
            "CSI_AUTHORIZED_DATA_USERS",
            CSI_AUTHORIZED_USER_URL,
            "authorized_data_users",
            ".json",
            is_json_payload,
        ),
        (
            "CSI_WEB_APP_DATA_SERVICE_DEFINITION",
            CSI_WEB_APP_URL,
            "csi_web_app",
            ".js",
            None,
        ),
        (
            "CSI_WEB_APP_AUTHORIZED_VENDOR_TABLE",
            CSI_VENDOR_CHUNK_URL,
            "csi_vendor_chunk",
            ".js",
            None,
        ),
        (
            "TUSHARE_INDEX_WEIGHT_DOCUMENTATION",
            TUSHARE_INDEX_WEIGHT_DOC_URL,
            "tushare_index_weight_doc",
            ".html",
            None,
        ),
    )
    for kind, url, prefix, suffix, validator in specifications:
        payload, meta = checked_get(session, url, timeout_seconds, validator=validator)
        if kind == "CSI_WEB_APP_DATA_SERVICE_DEFINITION":
            required = (
                b"table.data.service.monthly.closing.weight",
                b"queryAuthorizeddatauser",
            )
            if not all(item in payload for item in required):
                raise RuntimeError("中证官网应用脚本缺少月度权重或授权用户定义")
        if kind == "CSI_WEB_APP_AUTHORIZED_VENDOR_TABLE":
            required = (b"monthlyClosingWeight", b"indexDailyStaticData")
            if not all(item in payload for item in required):
                raise RuntimeError("中证官网授权资讯商表脚本缺少静态数据字段")
        if kind == "TUSHARE_INDEX_WEIGHT_DOCUMENTATION":
            text = payload.decode("utf-8", errors="ignore")
            if "index_weight" not in text and "指数成分和权重" not in text:
                raise RuntimeError("Tushare 文档页面没有 index_weight 说明")
        path, _ = archive_content_addressed(RAW_ROOT / "weight_provenance", prefix, payload, suffix)
        sources.append(source_record(kind, url, path, payload, meta))
        if kind in {"CSI_AUTHORIZED_STATIC_DATA_AGENTS", "CSI_AUTHORIZED_DATA_USERS"}:
            parsed = json.loads(payload.decode("utf-8"))
            if str(parsed.get("code")) != "200" or not parsed.get("success"):
                raise RuntimeError(f"中证官网授权清单业务状态异常：{kind}")
            records = parsed.get("data")
            if not isinstance(records, list) or not records:
                raise RuntimeError(f"中证官网授权清单为空：{kind}")
            if kind == "CSI_AUTHORIZED_STATIC_DATA_AGENTS":
                agents = records
            else:
                users = records
    return sources, agents, users


def query_wayback_cdx(
    session: requests.Session,
    timeout_seconds: float,
) -> tuple[bytes, dict[str, Any], list[dict[str, str]]]:
    params: list[tuple[str, str]] = [
        ("url", "csindex.com.cn"),
        ("matchType", "domain"),
        ("output", "json"),
        ("fl", "timestamp,original,statuscode,mimetype,digest,length"),
        ("filter", "statuscode:200"),
        ("filter", r"original:.*000300closeweight\.xls.*"),
        ("collapse", "digest"),
    ]
    errors: list[str] = []
    for attempt in range(1, 7):
        try:
            response = session.get(WAYBACK_CDX_URL, params=params, timeout=timeout_seconds)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.content
            parsed = json.loads(payload.decode("utf-8"))
            if not isinstance(parsed, list) or len(parsed) < 2:
                raise RuntimeError("CDX 没有返回历史 closeweight 记录")
            header = parsed[0]
            expected_header = [
                "timestamp",
                "original",
                "statuscode",
                "mimetype",
                "digest",
                "length",
            ]
            if header != expected_header:
                raise RuntimeError(f"CDX 字段异常：{header}")
            records = [dict(zip(header, row, strict=True)) for row in parsed[1:]]
            records = [
                record
                for record in records
                if "000300closeweight.xls" in record["original"].lower()
            ]
            unique: dict[str, dict[str, str]] = {}
            for record in records:
                unique.setdefault(record["digest"], record)
            ordered = sorted(unique.values(), key=lambda item: item["timestamp"])
            if not ordered:
                raise RuntimeError("CDX 过滤后没有历史 closeweight 记录")
            meta = {
                "status_code": response.status_code,
                "content_type": str(response.headers.get("Content-Type") or ""),
                "final_url": str(response.url),
            }
            return payload, meta, ordered
        except (requests.RequestException, RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"第{attempt}次：{type(exc).__name__}: {exc}")
            if attempt < 6:
                time.sleep(min(2 ** (attempt - 1), 12))
    raise RuntimeError(f"Wayback CDX 连续失败：{'；'.join(errors)}")


def acquire_official_snapshots(
    session: requests.Session,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], ParsedOfficialSnapshot]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    snapshots: list[tuple[dict[str, Any], ParsedOfficialSnapshot]] = []
    failures: list[dict[str, Any]] = []

    cdx_payload, cdx_meta, captures = query_wayback_cdx(session, timeout_seconds)
    cdx_path, _ = archive_content_addressed(
        RAW_ROOT / "weight_provenance" / "cdx",
        "000300closeweight_cdx",
        cdx_payload,
        ".json",
    )
    cdx_source = source_record(
        "INTERNET_ARCHIVE_CDX_FOR_CSI_OFFICIAL_CLOSEWEIGHT",
        str(cdx_meta["final_url"]),
        cdx_path,
        cdx_payload,
        cdx_meta,
    )
    cdx_source["transport_role"] = "THIRD_PARTY_IMMUTABLE_CAPTURE_INDEX_ONLY"
    sources.append(cdx_source)

    for capture in captures:
        payload = b""
        replay_url = (
            f"https://web.archive.org/web/{capture['timestamp']}id_/"
            f"{capture['original']}"
        )
        try:
            payload, meta = checked_get(
                session,
                replay_url,
                timeout_seconds,
                validator=is_excel_payload,
                attempts=5,
            )
            path, _ = archive_content_addressed(
                RAW_ROOT / "weight_provenance" / "wayback_official_files",
                f"capture_{capture['timestamp']}",
                payload,
                ".xls",
            )
            record = source_record(
                "CSI_OFFICIAL_CLOSEWEIGHT_VIA_INTERNET_ARCHIVE_CAPTURE",
                replay_url,
                path,
                payload,
                meta,
            )
            parsed = parse_official_closeweight(payload)
            record.update(
                {
                    "official_original_url": capture["original"],
                    "wayback_capture_timestamp_utc": capture["timestamp"],
                    "wayback_cdx_digest": capture["digest"],
                    "wayback_cdx_reported_length": int(capture["length"]),
                    "snapshot_date": parsed.snapshot_date.isoformat(),
                    "constituent_count": len(parsed.frame),
                    "weight_sum_pct": parsed.weight_sum_pct,
                    "weight_decimal_places": parsed.weight_decimal_places,
                    "transport_role": "THIRD_PARTY_TRANSPORT_OF_OFFICIAL_FILE",
                }
            )
            sources.append(record)
            snapshots.append((record, parsed))
            print(
                f"Wayback 官方权重 {capture['timestamp']} -> {parsed.snapshot_date}："
                f"{len(parsed.frame)} 行",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - 单个历史留存失败需保留后继续
            archived_path = None
            archived_sha256 = None
            if "payload" in locals() and isinstance(payload, bytes) and is_excel_payload(payload):
                archived_path_object, archived_sha256 = archive_content_addressed(
                    RAW_ROOT / "weight_provenance" / "wayback_official_files_unparsed",
                    f"capture_{capture['timestamp']}",
                    payload,
                    ".xls",
                )
                archived_path = archived_path_object.relative_to(ROOT).as_posix()
            failures.append(
                {
                    "capture_timestamp_utc": capture["timestamp"],
                    "official_original_url": capture["original"],
                    "wayback_cdx_digest": capture["digest"],
                    "archive_path": archived_path,
                    "sha256": archived_sha256,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(
                f"Wayback 官方权重 {capture['timestamp']} 获取失败：{type(exc).__name__}: {exc}",
                flush=True,
            )

    current_payload, current_meta = checked_get(
        session,
        OFFICIAL_CURRENT_WEIGHT_URL,
        timeout_seconds,
        validator=is_excel_payload,
    )
    current_snapshot = parse_official_closeweight(current_payload)
    current_path, _ = archive_content_addressed(
        RAW_ROOT / "weight_provenance" / "official_current",
        "000300closeweight_current",
        current_payload,
        ".xls",
    )
    current_record = source_record(
        "CSI_OFFICIAL_CLOSEWEIGHT_DIRECT_CURRENT",
        OFFICIAL_CURRENT_WEIGHT_URL,
        current_path,
        current_payload,
        current_meta,
    )
    current_record.update(
        {
            "snapshot_date": current_snapshot.snapshot_date.isoformat(),
            "constituent_count": len(current_snapshot.frame),
            "weight_sum_pct": current_snapshot.weight_sum_pct,
            "weight_decimal_places": current_snapshot.weight_decimal_places,
            "transport_role": "DIRECT_OFFICIAL_SOURCE",
        }
    )
    sources.append(current_record)
    snapshots.append((current_record, current_snapshot))
    return sources, snapshots, failures


def month_periods(start_date: date, end_date: date) -> Iterable[pd.Period]:
    return pd.period_range(
        start=pd.Timestamp(start_date).to_period("M"),
        end=pd.Timestamp(end_date).to_period("M"),
        freq="M",
    )


def proxy_endpoints() -> tuple[str, list[str]]:
    load_dotenv(ROOT / ".env")
    token = (os.getenv("TUSHARE_PROXY_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("项目 .env 没有 TUSHARE_PROXY_TOKEN")
    configured = (
        os.getenv("TUSHARE_PROXY_URL", "https://fast.xiaodefa.cn").strip().rstrip("/")
    )
    endpoints = [configured]
    for alternate in ("https://fast.xiaodefa.cn", "https://tt.xiaodefa.cn"):
        if alternate not in endpoints:
            endpoints.append(alternate)
    invalid = [endpoint for endpoint in endpoints if endpoint not in ALLOWED_PROXY_ENDPOINTS]
    if invalid:
        raise RuntimeError(f"Tushare 代理地址不在允许清单：{invalid}")
    return token, endpoints


def parse_tushare_response(payload: bytes, expected_period: pd.Period) -> pd.DataFrame:
    body = json.loads(payload.decode("utf-8"))
    code = body.get("code")
    if code not in {0, "0", None}:
        raise RuntimeError(f"Tushare 代理业务错误 {code}：{body.get('msg')}")
    data = body.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Tushare 代理响应缺少 data")
    fields = data.get("fields")
    items = data.get("items")
    if not isinstance(fields, list) or not isinstance(items, list):
        raise RuntimeError("Tushare 代理响应缺少 fields 或 items")
    frame = pd.DataFrame.from_records(items, columns=fields)
    required = {"index_code", "con_code", "trade_date", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Tushare 权重响应缺少字段：{sorted(missing)}")
    if frame.empty:
        raise RuntimeError(f"Tushare 权重响应为空：{expected_period}")
    frame = frame[["index_code", "con_code", "trade_date", "weight"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="raise")
    periods = frame["trade_date"].dt.to_period("M").unique().tolist()
    if periods != [expected_period]:
        raise RuntimeError(f"Tushare 响应月份异常：期望 {expected_period}，实际 {periods}")
    frame["symbol"] = frame["con_code"].map(normalize_symbol)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    if frame.duplicated(["trade_date", "symbol"]).any():
        raise RuntimeError(f"Tushare 权重响应存在重复行：{expected_period}")
    dates = sorted(frame["trade_date"].dt.date.unique())
    if len(dates) != 1:
        raise RuntimeError(f"Tushare 每月应只有一个权重日：{expected_period} -> {dates}")
    if len(frame) != 300 or frame["symbol"].nunique() != 300:
        raise RuntimeError(
            f"Tushare 权重响应不是 300 个唯一证券：{expected_period} -> {len(frame)}"
        )
    weight_sum = float(frame["weight"].sum())
    if not 98.0 <= weight_sum <= 102.0:
        raise RuntimeError(f"Tushare 权重和异常：{expected_period} -> {weight_sum}")
    return frame


def retrieve_tushare_weights(
    session: requests.Session,
    start_date: date,
    end_date: date,
    timeout_seconds: float,
    request_interval_seconds: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    token, endpoints = proxy_endpoints()
    retrieval_started_at = now_shanghai()
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    selected_endpoint = ""
    for sequence, period in enumerate(month_periods(start_date, end_date), start=1):
        parameters = {
            "index_code": "399300.SZ",
            "start_date": period.start_time.strftime("%Y%m%d"),
            "end_date": period.end_time.strftime("%Y%m%d"),
        }
        request_payload = {"api_name": "index_weight", "params": parameters}
        errors: list[str] = []
        response_payload: bytes | None = None
        response_meta: dict[str, Any] | None = None
        used_endpoint = ""
        for endpoint in endpoints:
            for attempt in range(1, 4):
                try:
                    response = session.post(
                        endpoint,
                        json=request_payload,
                        headers={"x-api-key": token},
                        timeout=timeout_seconds,
                    )
                    if response.status_code != 200:
                        raise RuntimeError(f"HTTP {response.status_code}")
                    if token.encode("utf-8") in response.content:
                        raise RuntimeError("代理响应意外回显凭据，拒绝归档")
                    raw_body = response.json()
                    raw_code = raw_body.get("code") if isinstance(raw_body, dict) else None
                    if raw_code not in {0, "0", None}:
                        error_path, error_digest = archive_content_addressed(
                            RAW_ROOT / "weight_provenance" / "tushare_failed_retrieval",
                            f"index_weight_{period.strftime('%Y%m')}_{urlparse(endpoint).netloc}",
                            response.content,
                            ".json",
                        )
                        failure_source = {
                            "kind": "TUSHARE_COMPATIBLE_INDEX_WEIGHT_FAILED_RESPONSE",
                            "endpoint_host": urlparse(endpoint).netloc,
                            "method": "POST",
                            "api_name": "index_weight",
                            "params": parameters,
                            "credential_kind": "TUSHARE_PROXY_TOKEN_IN_REQUEST_HEADER_NOT_ARCHIVED",
                            "tls_verification": True,
                            "archive_path": error_path.relative_to(ROOT).as_posix(),
                            "sha256": error_digest,
                            "size_bytes": len(response.content),
                            "response_status_code": response.status_code,
                            "provider_code": str(raw_code),
                            "provider_message": str(raw_body.get("msg") or ""),
                            "retrieval_mode": "NETWORK_TLS_VERIFIED",
                        }
                        sources.append(failure_source)
                        raise RuntimeError(
                            f"Tushare 代理业务错误 {raw_code}：{raw_body.get('msg')}"
                        )
                    frame = parse_tushare_response(response.content, period)
                    response_payload = response.content
                    response_meta = {
                        "status_code": response.status_code,
                        "content_type": str(response.headers.get("Content-Type") or ""),
                        "final_url": str(response.url),
                    }
                    used_endpoint = endpoint
                    break
                except (requests.RequestException, RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    errors.append(
                        f"{urlparse(endpoint).netloc}第{attempt}次：{type(exc).__name__}: {exc}"
                    )
                    if "业务错误 40101" in str(exc):
                        break
                    if attempt < 3:
                        time.sleep(min(2 ** (attempt - 1), 4))
            if response_payload is not None:
                break
        if response_payload is None or response_meta is None:
            raise TushareRetrievalFailure(
                f"Tushare 权重 {period} 获取失败：{'；'.join(errors)}",
                sources,
            )
        path, digest = archive_content_addressed(
            RAW_ROOT / "weight_provenance" / "tushare_second_retrieval",
            f"index_weight_{period.strftime('%Y%m')}",
            response_payload,
            ".json",
        )
        source = {
            "kind": "TUSHARE_COMPATIBLE_INDEX_WEIGHT_SECOND_RETRIEVAL",
            "endpoint_host": urlparse(used_endpoint).netloc,
            "method": "POST",
            "api_name": "index_weight",
            "params": parameters,
            "credential_kind": "TUSHARE_PROXY_TOKEN_IN_REQUEST_HEADER_NOT_ARCHIVED",
            "tls_verification": True,
            "archive_path": path.relative_to(ROOT).as_posix(),
            "sha256": digest,
            "size_bytes": len(response_payload),
            "response_status_code": int(response_meta["status_code"]),
            "response_content_type": str(response_meta["content_type"]),
            "snapshot_date": str(frame["trade_date"].dt.date.iloc[0]),
            "row_count": len(frame),
            "retrieval_mode": "NETWORK_TLS_VERIFIED",
        }
        sources.append(source)
        frame["source"] = "tushare.index_weight"
        frame["retrieved_at"] = retrieval_started_at
        frames.append(frame)
        selected_endpoint = used_endpoint
        print(
            f"Tushare 历史权重 {period}：{len(frame)} 行，"
            f"{frame['trade_date'].dt.date.iloc[0]}",
            flush=True,
        )
        if sequence < len(list(month_periods(start_date, end_date))):
            time.sleep(request_interval_seconds)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    expected_month_count = len(list(month_periods(start_date, end_date)))
    if combined["trade_date"].nunique() != expected_month_count:
        raise RuntimeError(
            f"二次抓取快照数量异常：期望 {expected_month_count}，实际 {combined['trade_date'].nunique()}"
        )
    return combined, sources, selected_endpoint


def compare_retrievals(current: pd.DataFrame, existing_path: Path) -> dict[str, Any]:
    if not existing_path.is_file():
        raise FileNotFoundError(f"首次历史权重文件不存在：{existing_path}")
    existing = pd.read_parquet(existing_path)
    required = {"con_code", "trade_date", "weight", "source", "retrieved_at"}
    missing = required - set(existing.columns)
    if missing:
        raise ValueError(f"首次历史权重文件缺少字段：{sorted(missing)}")
    existing = existing.copy()
    existing["trade_date"] = pd.to_datetime(existing["trade_date"], errors="raise")
    existing["symbol"] = existing["con_code"].map(normalize_symbol)
    existing["weight"] = pd.to_numeric(existing["weight"], errors="raise")
    left = existing[["trade_date", "symbol", "weight"]].rename(
        columns={"weight": "first_weight_pct"}
    )
    right = current[["trade_date", "symbol", "weight"]].rename(
        columns={"weight": "second_weight_pct"}
    )
    merged = left.merge(
        right,
        on=["trade_date", "symbol"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    both = merged.loc[merged["_merge"].eq("both")].copy()
    differences = (both["first_weight_pct"] - both["second_weight_pct"]).abs()
    mismatch_count = int(differences.gt(1e-12).sum())
    retrieval_times = sorted(
        {str(value) for value in existing["retrieved_at"].dropna().astype(str).unique()}
    )
    return {
        "first_file": existing_path.relative_to(ROOT).as_posix(),
        "first_file_sha256": sha256_file(existing_path),
        "first_retrieval_time_min": retrieval_times[0] if retrieval_times else None,
        "first_retrieval_time_max": retrieval_times[-1] if retrieval_times else None,
        "first_row_count": len(existing),
        "second_row_count": len(current),
        "matched_row_count": len(both),
        "left_only_row_count": int(merged["_merge"].eq("left_only").sum()),
        "right_only_row_count": int(merged["_merge"].eq("right_only").sum()),
        "weight_mismatch_count_tolerance_1e_12": mismatch_count,
        "maximum_absolute_weight_difference_pct": (
            float(differences.max()) if not differences.empty else None
        ),
        "exact_value_match": bool(
            len(merged) == len(existing) == len(current)
            and merged["_merge"].eq("both").all()
            and mismatch_count == 0
        ),
    }


def load_existing_weight_frame(path: Path) -> pd.DataFrame:
    """加载首次权重文件，供官方历史版本的数值交叉核验使用。"""

    if not path.is_file():
        raise FileNotFoundError(f"首次历史权重文件不存在：{path}")
    frame = pd.read_parquet(path)
    required = {"index_code", "con_code", "trade_date", "weight", "source", "retrieved_at"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"首次历史权重文件缺少字段：{sorted(missing)}")
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise")
    frame["symbol"] = frame["con_code"].map(normalize_symbol)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    if frame.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("首次历史权重文件存在日期和证券重复行")
    return frame


def crosscheck_official_snapshots(
    official_snapshots: list[tuple[dict[str, Any], ParsedOfficialSnapshot]],
    tushare_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    available_dates = set(tushare_frame["trade_date"].dt.date.unique())
    for source, parsed in official_snapshots:
        snapshot_date = parsed.snapshot_date
        base = {
            "source_kind": source["kind"],
            "snapshot_date": snapshot_date.isoformat(),
            "source_archive_path": source["archive_path"],
            "source_sha256": source["sha256"],
            "wayback_capture_timestamp_utc": source.get("wayback_capture_timestamp_utc"),
            "official_original_url": source.get("official_original_url", source.get("url")),
            "official_weight_decimal_places": parsed.weight_decimal_places,
        }
        if snapshot_date not in available_dates:
            summaries.append(
                {
                    **base,
                    "comparison_status": "OUTSIDE_TUSHARE_RETRIEVAL_WINDOW",
                    "official_row_count": len(parsed.frame),
                    "tushare_row_count": 0,
                    "set_difference_count": None,
                    "precision_aware_tolerance_pct": None,
                    "weight_mismatch_count_precision_aware": None,
                    "maximum_absolute_weight_difference_pct": None,
                }
            )
            continue
        vendor = tushare_frame.loc[
            tushare_frame["trade_date"].dt.date.eq(snapshot_date),
            ["symbol", "weight"],
        ].rename(columns={"weight": "vendor_weight_pct"})
        official = parsed.frame[["symbol", "weight_pct"]].rename(
            columns={"weight_pct": "official_weight_pct"}
        )
        merged = official.merge(vendor, on="symbol", how="outer", indicator=True, validate="one_to_one")
        both = merged["_merge"].eq("both")
        merged["absolute_weight_difference_pct"] = (
            merged["official_weight_pct"] - merged["vendor_weight_pct"]
        ).abs()
        for item in merged.to_dict(orient="records"):
            rows.append(
                {
                    **base,
                    "symbol": item["symbol"],
                    "official_weight_pct": item["official_weight_pct"],
                    "vendor_weight_pct": item["vendor_weight_pct"],
                    "absolute_weight_difference_pct": item[
                        "absolute_weight_difference_pct"
                    ],
                    "merge_state": str(item["_merge"]),
                }
            )
        differences = merged.loc[both, "absolute_weight_difference_pct"]
        precision_aware_tolerance = 0.5 * (10 ** (-parsed.weight_decimal_places))
        summaries.append(
            {
                **base,
                "comparison_status": "COMPARED",
                "official_row_count": len(official),
                "tushare_row_count": len(vendor),
                "set_difference_count": int((~both).sum()),
                "precision_aware_tolerance_pct": precision_aware_tolerance,
                "weight_mismatch_count_precision_aware": int(
                    differences.gt(precision_aware_tolerance + 1e-12).sum()
                ),
                "maximum_absolute_weight_difference_pct": (
                    float(differences.max()) if not differences.empty else None
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["snapshot_date", "source_kind", "symbol"]).reset_index(drop=True)
    return frame, summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2016, 8, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2026, 7, 31))
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--request-interval-seconds", type=float, default=0.6)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.start_date > arguments.end_date:
        raise ValueError("开始日期不得晚于结束日期")
    if arguments.request_interval_seconds < 0.5:
        raise ValueError("请求间隔不得低于 0.5 秒")

    session = build_session()
    acquired_at = now_shanghai()
    web_sources, authorized_agents, authorized_users = archive_official_web_evidence(
        session,
        arguments.timeout_seconds,
    )
    official_sources, official_snapshots, wayback_failures = acquire_official_snapshots(
        session,
        arguments.timeout_seconds,
    )
    second_retrieval_error: dict[str, Any] | None = None
    selected_endpoint = ""
    try:
        tushare_frame, tushare_sources, selected_endpoint = retrieve_tushare_weights(
            session,
            arguments.start_date,
            arguments.end_date,
            arguments.timeout_seconds,
            arguments.request_interval_seconds,
        )
        retrieval_comparison = compare_retrievals(tushare_frame, EXISTING_WEIGHT_PATH)
        output_frame = tushare_frame[
            ["index_code", "con_code", "trade_date", "weight", "source", "retrieved_at"]
        ].copy()
        atomic_write_parquet(SECOND_RETRIEVAL_PATH, output_frame)
        second_retrieval_status = "PASS_SECOND_RETRIEVAL"
    except TushareRetrievalFailure as exc:
        tushare_sources = exc.sources
        second_retrieval_error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "proxy_token_expires_at": os.getenv("TUSHARE_PROXY_TOKEN_EXPIRES_AT"),
        }
        retrieval_comparison = {
            "status": "EXTERNAL_SOURCE_FAILED_TOKEN_EXPIRED",
            "exact_value_match": False,
            **second_retrieval_error,
        }
        output_frame = None
        tushare_frame = load_existing_weight_frame(EXISTING_WEIGHT_PATH)
        second_retrieval_status = "EXTERNAL_SOURCE_FAILED_TOKEN_EXPIRED"
        print(
            f"Tushare 二次抓取未完成，保留失败凭证并仅用首次文件做官方锚点交叉核验：{exc}",
            flush=True,
        )

    official_crosscheck_frame, official_crosschecks = crosscheck_official_snapshots(
        official_snapshots,
        tushare_frame,
    )
    atomic_write_parquet(OFFICIAL_CROSSCHECK_PATH, official_crosscheck_frame)

    compared_official = [
        item for item in official_crosschecks if item["comparison_status"] == "COMPARED"
    ]
    clean_official = [
        item
        for item in compared_official
        if item["set_difference_count"] == 0
        and item["weight_mismatch_count_precision_aware"] == 0
    ]
    primary_official_versions = [
        item
        for item in official_sources
        if item["kind"] == "CSI_OFFICIAL_CLOSEWEIGHT_VIA_INTERNET_ARCHIVE_CAPTURE"
        and item.get("snapshot_date", "") >= "2021-01-01"
    ]
    static_agents = [
        {
            "name": item.get("userName"),
            "name_en": item.get("userEnName"),
            "website": item.get("companyWebsite"),
            "mainland": bool(item.get("chineseMainland")),
            "outside_china": bool(item.get("chinaOutside")),
        }
        for item in authorized_agents
        if item.get("indexDailyStaticData")
    ]
    authorized_names = {
        str(item.get("userName") or "") for item in authorized_agents + authorized_users
    }
    tushare_named_as_authorized = any("Tushare" in name or "tushare" in name for name in authorized_names)

    evidence_pass = bool(
        output_frame is not None
        and retrieval_comparison["exact_value_match"]
        and len(output_frame) == 36_000
        and output_frame["trade_date"].nunique() == 120
        and len(compared_official) >= 3
        and len(clean_official) == len(compared_official)
    )
    official_partial_pass = bool(
        len(primary_official_versions) >= 2
        and len(compared_official) >= 3
        and len(clean_official) == len(compared_official)
    )
    if evidence_pass:
        acquisition_status = "PASS_WEIGHT_EVIDENCE_ACQUIRED_NOT_ADMITTED"
    elif official_partial_pass and second_retrieval_error is not None:
        acquisition_status = "PARTIAL_WEIGHT_EVIDENCE_ACQUIRED_EXTERNAL_TOKEN_EXPIRED"
    else:
        acquisition_status = "FAILED_WEIGHT_EVIDENCE_ACQUISITION_OR_CROSSCHECK"
    manifest = {
        "remediation_id": REMEDIATION_ID,
        "version": "1.0.0",
        "status": acquisition_status,
        "acquired_at": acquired_at,
        "window": {
            "start_date": arguments.start_date.isoformat(),
            "end_date": arguments.end_date.isoformat(),
        },
        "official_data_service_evidence": {
            "authorized_static_data_agent_count": len(static_agents),
            "authorized_static_data_agents": static_agents,
            "authorized_data_user_count": len(authorized_users),
            "tushare_named_in_official_authorized_agent_or_user_lists": tushare_named_as_authorized,
            "interpretation": (
                "中证官网将月度收盘权重列为授权数据；当前本地 Tushare 兼容来源不在官网列示的"
                "授权代理商或授权用户名称中。"
            ),
        },
        "tushare_second_retrieval": {
            "status": second_retrieval_status,
            "endpoint_host": urlparse(selected_endpoint).netloc if selected_endpoint else None,
            "snapshot_count": (
                int(output_frame["trade_date"].nunique()) if output_frame is not None else 0
            ),
            "row_count": len(output_frame) if output_frame is not None else 0,
            "first_snapshot_date": (
                str(output_frame["trade_date"].min().date())
                if output_frame is not None
                else None
            ),
            "last_snapshot_date": (
                str(output_frame["trade_date"].max().date())
                if output_frame is not None
                else None
            ),
            "output_path": (
                SECOND_RETRIEVAL_PATH.relative_to(ROOT).as_posix()
                if output_frame is not None
                else None
            ),
            "output_sha256": (
                sha256_file(SECOND_RETRIEVAL_PATH) if output_frame is not None else None
            ),
            "raw_response_count": len(tushare_sources),
            "comparison_to_2026_08_13_retrieval": retrieval_comparison,
            "failed_attempt": second_retrieval_error,
            "existing_file_used_only_for_official_numeric_crosscheck": (
                EXISTING_WEIGHT_PATH.relative_to(ROOT).as_posix()
                if output_frame is None
                else None
            ),
            "existing_file_sha256": (
                sha256_file(EXISTING_WEIGHT_PATH) if output_frame is None else None
            ),
        },
        "official_versioned_snapshot_evidence": {
            "wayback_cdx_capture_count": len(
                [
                    item
                    for item in official_sources
                    if item["kind"]
                    == "CSI_OFFICIAL_CLOSEWEIGHT_VIA_INTERNET_ARCHIVE_CAPTURE"
                ]
            ),
            "wayback_failed_capture_count": len(wayback_failures),
            "wayback_failures": wayback_failures,
            "primary_2021_plus_wayback_official_snapshot_count": len(primary_official_versions),
            "direct_current_official_snapshot_count": len(
                [
                    item
                    for item in official_sources
                    if item["kind"] == "CSI_OFFICIAL_CLOSEWEIGHT_DIRECT_CURRENT"
                ]
            ),
            "numeric_crosschecks": official_crosschecks,
            "clean_numeric_crosscheck_count": len(clean_official),
            "crosscheck_path": OFFICIAL_CROSSCHECK_PATH.relative_to(ROOT).as_posix(),
            "crosscheck_sha256": sha256_file(OFFICIAL_CROSSCHECK_PATH),
        },
        "frozen_weight_provenance_gate_assessment": {
            "historical_values_retrieved_twice_with_exact_match": bool(
                retrieval_comparison["exact_value_match"]
            ),
            "versioned_official_numeric_anchors_present": bool(primary_official_versions),
            "licensed_or_official_version_proof_for_every_required_snapshot": False,
            "admission_sufficient": False,
            "status": "BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS",
            "reason": (
                "公开留存仅形成少量官方历史版本锚点；现有 Tushare 兼容来源不在中证官网列示的"
                "授权代理商或授权用户名称中，且没有每一期历史版本或 as-of 凭证。二次抓取又因"
                "代理凭据过期而失败，因此不能越过冻结硬门槛。"
            ),
        },
        "source_objects": web_sources + official_sources + tushare_sources,
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "model_read": False,
        "portfolio_evaluation": "NOT_ALLOWED",
        "trading_authorization": False,
    }
    atomic_write_json(OUTPUT_MANIFEST_PATH, manifest)
    result = {
        "status": manifest["status"],
        "weight_admission_status": manifest["frozen_weight_provenance_gate_assessment"]["status"],
        "snapshot_count": manifest["tushare_second_retrieval"]["snapshot_count"],
        "row_count": manifest["tushare_second_retrieval"]["row_count"],
        "exact_second_retrieval_match": retrieval_comparison["exact_value_match"],
        "official_clean_crosscheck_count": len(clean_official),
        "primary_versioned_official_snapshot_count": len(primary_official_versions),
        "manifest_path": OUTPUT_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(OUTPUT_MANIFEST_PATH),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if evidence_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
