"""归档沪深300临时调样的中证指数与上交所官方证据。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_hs_csi300_official_addition_forced_demand_v1 import (  # noqa: E402
    extract_official_attachment_urls,
)
from research.csi300_pit_membership_weights_source_remediation_v1 import (  # noqa: E402
    REMEDIATION_ID,
    normalize_symbol,
    sha256_file,
)


RAW_ROOT = ROOT / "data" / "raw" / "510300_csi300_pit_membership_weights_source_remediation_v1"
OUTPUT_PATH = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "official_special_rebalance_manifest.json"
)
REUSED_LIST_PATH = (
    ROOT
    / "data"
    / "raw"
    / "a_share_hs_csi300_official_addition_forced_demand_v1"
    / "list"
    / "announcement_list_d97a309d21bfa57dc1fb940cb2327a466f8fc00009de34d67ce1f0fbaea7803f.json"
)

SPECIAL_ANNOUNCEMENTS = (
    {
        "cycle_id": "2017_SPECIAL_WISCO_DELISTING",
        "announcement_id": 4794,
        "subject_security_code": "600005",
        "discovery_search_input": "武钢股份",
        "sse_delisting_url": None,
    },
    {
        "cycle_id": "2025_SPECIAL_HAITONG_DELISTING",
        "announcement_id": 15546,
        "subject_security_code": "600837",
        "sse_delisting_url": (
            "https://www.sse.com.cn/disclosure/announcement/listing/stock/"
            "c/c_20250226_10773005.shtml"
        ),
    },
    {
        "cycle_id": "2025_SPECIAL_CSIC_DELISTING",
        "announcement_id": 1006022,
        "subject_security_code": "601989",
        "sse_delisting_url": (
            "https://www.sse.com.cn/disclosure/announcement/listing/stock/"
            "c/c_20250829_10790128.shtml"
        ),
    },
)


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


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
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


def checked_get(session: requests.Session, url: str, timeout_seconds: float) -> bytes:
    response = session.get(url, timeout=timeout_seconds)
    if response.status_code != 200:
        raise RuntimeError(f"官方来源 HTTP 状态异常：{url} -> {response.status_code}")
    if not response.content:
        raise RuntimeError(f"官方来源返回空内容：{url}")
    return response.content


def checked_post_json(
    session: requests.Session,
    url: str,
    body: dict[str, Any],
    timeout_seconds: float,
) -> bytes:
    response = session.post(url, json=body, timeout=timeout_seconds)
    if response.status_code != 200:
        raise RuntimeError(f"官方来源 HTTP 状态异常：{url} -> {response.status_code}")
    if not response.content:
        raise RuntimeError(f"官方来源返回空内容：{url}")
    return response.content


def archive_content_addressed(directory: Path, prefix: str, payload: bytes, suffix: str) -> tuple[Path, str]:
    digest = sha256_bytes(payload)
    path = directory / f"{prefix}_{digest}{suffix.lower()}"
    if path.exists():
        if sha256_file(path) != digest:
            raise RuntimeError(f"已有归档对象内容哈希异常：{path}")
    else:
        atomic_write_bytes(path, payload)
    return path, digest


def source_record(kind: str, url: str, path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "kind": kind,
        "url": url,
        "method": "GET",
        "tls_verification": true_value(),
        "archive_path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "retrieval_mode": "NETWORK_TLS_VERIFIED",
    }


def true_value() -> bool:
    """集中返回布尔真，避免来源凭证出现字符串化布尔值。"""

    return True


def parse_detail(payload: bytes, expected_id: int) -> dict[str, Any]:
    parsed = json.loads(payload.decode("utf-8"))
    if str(parsed.get("code")) != "200" or not parsed.get("success"):
        raise RuntimeError(f"中证公告详情业务状态异常：{parsed.get('code')} {parsed.get('msg')}")
    detail = parsed.get("data")
    if not isinstance(detail, dict) or int(detail.get("id")) != expected_id:
        raise RuntimeError(f"中证公告详情 ID 与请求不一致：{expected_id}")
    return detail


def infer_suffix(url: str, payload: bytes) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".xls", ".xlsx", ".pdf"}:
        return suffix
    if payload.startswith(b"PK\x03\x04"):
        return ".xlsx"
    if payload.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        return ".xls"
    if payload.startswith(b"%PDF"):
        return ".pdf"
    raise ValueError(f"无法识别官方附件格式：{url}")


def parse_csi300_special_change(payload: bytes) -> dict[str, str]:
    excel = pd.ExcelFile(io.BytesIO(payload))
    matches: list[dict[str, str]] = []
    split_sheet_matches: dict[str, dict[str, str]] = {}
    for sheet_name in excel.sheet_names:
        frame = pd.read_excel(io.BytesIO(payload), sheet_name=sheet_name, header=None, dtype=object)
        for row_index, values in frame.iterrows():
            cells = ["" if pd.isna(value) else str(value).strip() for value in values.tolist()]
            if not cells:
                continue
            first_digits = "".join(character for character in cells[0] if character.isdigit())
            if first_digits != "000300":
                continue
            if len(cells) >= 6 and cells[4]:
                matches.append(
                    {
                        "sheet_name": str(sheet_name),
                        "row_index_zero_based": str(row_index),
                        "index_code": "000300",
                        "index_name": cells[1],
                        "deletion_code": normalize_symbol(cells[2]),
                        "deletion_name": cells[3],
                        "addition_code": normalize_symbol(cells[4]),
                        "addition_name": cells[5],
                    }
                )
                continue
            if len(cells) < 4:
                raise ValueError(f"官方临时调样表的沪深300行列数不足：{cells}")
            role = None
            if "调入" in str(sheet_name):
                role = "addition"
            elif "调出" in str(sheet_name):
                role = "deletion"
            if role is None:
                raise ValueError(f"无法确定官方临时调样工作表角色：{sheet_name}")
            split_sheet_matches[role] = {
                "sheet_name": str(sheet_name),
                "row_index_zero_based": str(row_index),
                "index_code": "000300",
                "index_name": cells[1],
                "security_code": normalize_symbol(cells[2]),
                "security_name": cells[3],
            }
    if len(matches) == 1 and not split_sheet_matches:
        return matches[0]
    if not matches and set(split_sheet_matches) == {"addition", "deletion"}:
        addition = split_sheet_matches["addition"]
        deletion = split_sheet_matches["deletion"]
        return {
            "sheet_name": f"{deletion['sheet_name']}+{addition['sheet_name']}",
            "row_index_zero_based": (
                f"deletion={deletion['row_index_zero_based']};addition={addition['row_index_zero_based']}"
            ),
            "index_code": "000300",
            "index_name": addition["index_name"],
            "deletion_code": deletion["security_code"],
            "deletion_name": deletion["security_name"],
            "addition_code": addition["security_code"],
            "addition_name": addition["security_name"],
        }
    raise ValueError(
        "官方临时调样附件中的 000300 记录结构异常："
        f"合并行 {len(matches)} 条，拆分工作表角色 {sorted(split_sheet_matches)}"
    )


def decode_official_html(payload: bytes) -> str:
    for encoding in ("utf-8", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("上交所官方页面无法按 UTF-8 或 GB18030 解码")


def extract_delisting_effective_date(html: str, security_code: str) -> str:
    compact = re.sub(r"\s+", "", html)
    if security_code not in compact:
        raise ValueError(f"上交所终止上市公告未出现目标证券代码 {security_code}")
    patterns = (
        r"自(20\d{2})年(\d{1,2})月(\d{1,2})日起终止其股票在本所上市交易",
        r"自(20\d{2})年(\d{1,2})月(\d{1,2})日起终止.*?上市交易",
    )
    for pattern in patterns:
        match = re.search(pattern, compact)
        if match:
            year, month, day = (int(value) for value in match.groups())
            return f"{year:04d}-{month:02d}-{day:02d}"
    raise ValueError(f"上交所终止上市公告未提取到 {security_code} 的生效日")


def extract_detail_effective_date(content_html: str, publish_date: str, security_code: str) -> str:
    compact = re.sub(r"\s+", "", content_html)
    if security_code not in compact:
        raise ValueError(f"中证临时调样公告未出现目标证券代码 {security_code}")
    full_date = re.search(r"自.*?(20\d{2})年(\d{1,2})月(\d{1,2})日退市之日起", compact)
    if full_date:
        year, month, day = (int(value) for value in full_date.groups())
        return f"{year:04d}-{month:02d}-{day:02d}"
    partial_date = re.search(r"自.*?(\d{1,2})月(\d{1,2})日退市之日起", compact)
    if partial_date:
        year = int(str(publish_date)[:4])
        month, day = (int(value) for value in partial_date.groups())
        return f"{year:04d}-{month:02d}-{day:02d}"
    raise ValueError(f"中证临时调样公告未提取到 {security_code} 的生效日")


def acquire(timeout_seconds: float) -> dict[str, Any]:
    if not REUSED_LIST_PATH.is_file():
        raise FileNotFoundError(f"冻结的中证公告列表不存在：{REUSED_LIST_PATH}")
    session = build_session()
    bootstrap = session.get("https://www.csindex.com.cn/", timeout=timeout_seconds)
    if bootstrap.status_code != 200:
        raise RuntimeError(f"中证官网会话初始化失败：HTTP {bootstrap.status_code}")

    list_payload = REUSED_LIST_PATH.read_bytes()
    list_json = json.loads(list_payload.decode("utf-8"))
    listed_ids = {
        int(item["id"])
        for item in list_json.get("data") or []
        if isinstance(item, dict) and item.get("id") is not None
    }
    cycles: list[dict[str, Any]] = []
    all_sources: list[dict[str, Any]] = [
        {
            "kind": "OFFICIAL_ANNOUNCEMENT_LIST_REUSED_IMMUTABLE_RAW",
            "url": "https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementByVo",
            "method": "POST",
            "tls_verification": True,
            "archive_path": REUSED_LIST_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(list_payload),
            "size_bytes": len(list_payload),
            "retrieval_mode": "REUSED_IMMUTABLE_RAW",
        }
    ]

    for specification in SPECIAL_ANNOUNCEMENTS:
        announcement_id = int(specification["announcement_id"])
        if announcement_id not in listed_ids:
            list_url = "https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementByVo"
            discovery_body = {
                "lang": "cn",
                "searchInput": str(specification["discovery_search_input"]),
                "startDate": "2016-01-01",
                "endDate": "2018-01-01",
                "classList": ["index"],
                "indexList": ["csi_index"],
                "relatedTopics": ["index_rebalance"],
                "typeList": ["announcement"],
                "page": {"key": "", "order": "asc", "page": 1, "rows": 500, "sortBy": "publish_date"},
            }
            discovery_payload = checked_post_json(session, list_url, discovery_body, timeout_seconds)
            discovery_json = json.loads(discovery_payload.decode("utf-8"))
            discovered_ids = {
                int(item["id"])
                for item in discovery_json.get("data") or []
                if isinstance(item, dict) and item.get("id") is not None
            }
            if announcement_id not in discovered_ids:
                raise RuntimeError(f"官方定向公告查询未包含临时调样公告 {announcement_id}")
            discovery_path, _ = archive_content_addressed(
                RAW_ROOT / "list",
                f"announcement_search_{announcement_id}",
                discovery_payload,
                ".json",
            )
            discovery_source = source_record(
                "CSI_OFFICIAL_SPECIAL_REBALANCE_DISCOVERY_LIST",
                list_url,
                discovery_path,
                discovery_payload,
            )
            discovery_source["method"] = "POST"
            discovery_source["request_body_sha256"] = sha256_bytes(
                json.dumps(
                    discovery_body,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            all_sources.append(discovery_source)

        detail_url = (
            "https://www.csindex.com.cn/csindex-home/announcement/"
            f"queryAnnouncementById?id={announcement_id}&lang=cn"
        )
        detail_payload = checked_get(session, detail_url, timeout_seconds)
        detail = parse_detail(detail_payload, announcement_id)
        detail_path, _ = archive_content_addressed(
            RAW_ROOT / "details",
            f"announcement_{announcement_id}",
            detail_payload,
            ".json",
        )
        detail_source = source_record("CSI_OFFICIAL_SPECIAL_REBALANCE_DETAIL", detail_url, detail_path, detail_payload)
        all_sources.append(detail_source)

        attachment_urls = extract_official_attachment_urls(detail)
        if not attachment_urls:
            raise RuntimeError(f"中证临时调样公告 {announcement_id} 没有附件")
        parsed_changes: list[dict[str, str]] = []
        attachment_sources: list[dict[str, Any]] = []
        for sequence, attachment in enumerate(attachment_urls, start=1):
            url = str(attachment["url"])
            payload = checked_get(session, url, timeout_seconds)
            suffix = infer_suffix(url, payload)
            attachment_path, _ = archive_content_addressed(
                RAW_ROOT / "attachments",
                f"announcement_{announcement_id}_attachment_{sequence}",
                payload,
                suffix,
            )
            attachment_source = source_record(
                "CSI_OFFICIAL_SPECIAL_REBALANCE_ATTACHMENT",
                url,
                attachment_path,
                payload,
            )
            attachment_source["file_name"] = str(attachment.get("file_name") or "")
            attachment_sources.append(attachment_source)
            all_sources.append(attachment_source)
            if suffix in {".xls", ".xlsx"}:
                parsed_changes.append(parse_csi300_special_change(payload))
        if len(parsed_changes) != 1:
            raise RuntimeError(
                f"中证临时调样公告 {announcement_id} 应解析出一条沪深300变更，实际 {len(parsed_changes)}"
            )
        change = parsed_changes[0]
        expected_deletion = normalize_symbol(specification["subject_security_code"])
        if change["deletion_code"] != expected_deletion:
            raise RuntimeError(
                f"中证临时调样公告 {announcement_id} 的调出证券不是目标证券：{change['deletion_code']}"
            )

        if specification["sse_delisting_url"]:
            sse_url = str(specification["sse_delisting_url"])
            sse_payload = checked_get(session, sse_url, timeout_seconds)
            sse_path, _ = archive_content_addressed(
                RAW_ROOT / "sse_delisting",
                f"security_{specification['subject_security_code']}",
                sse_payload,
                ".html",
            )
            effective_source = source_record(
                "SSE_OFFICIAL_DELISTING_EFFECTIVE_DATE",
                sse_url,
                sse_path,
                sse_payload,
            )
            all_sources.append(effective_source)
            effective_date = extract_delisting_effective_date(
                decode_official_html(sse_payload),
                str(specification["subject_security_code"]),
            )
            effective_date_rule = "CSI_FROM_DELISTING_DATE_PLUS_SSE_EXACT_DELISTING_DATE"
        else:
            effective_source = detail_source
            effective_date = extract_detail_effective_date(
                str(detail.get("content") or ""),
                str(detail["publishDate"])[:10],
                str(specification["subject_security_code"]),
            )
            effective_date_rule = "CSI_AND_SSE_JOINT_DETAIL_EXACT_DELISTING_DATE"

        cycles.append(
            {
                "cycle_id": str(specification["cycle_id"]),
                "announcement_id": announcement_id,
                "announcement_date": str(detail["publishDate"])[:10],
                "announcement_title": str(detail["title"]),
                "effective_date": effective_date,
                "effective_date_rule": effective_date_rule,
                "index_code": "000300",
                "deletions": [
                    {
                        "symbol": change["deletion_code"],
                        "name": change["deletion_name"],
                    }
                ],
                "additions": [
                    {
                        "symbol": change["addition_code"],
                        "name": change["addition_name"],
                    }
                ],
                "extraction": change,
                "detail_source": detail_source,
                "attachment_sources": attachment_sources,
                "effective_date_source": effective_source,
            }
        )

    ordered_cycles = sorted(cycles, key=lambda item: item["effective_date"])
    if ordered_cycles != cycles:
        raise RuntimeError("临时调样周期未按生效日升序排列")
    result = {
        "remediation_id": REMEDIATION_ID,
        "version": "1.0.0",
        "status": "PASS_OFFICIAL_SPECIAL_REBALANCE_SOURCES_ACQUIRED",
        "retrieved_at": now_shanghai(),
        "tls_verification": True,
        "special_cycle_count": len(cycles),
        "cycles": cycles,
        "source_objects": all_sources,
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "return_evaluation": "NOT_ALLOWED",
        "trading_authorization": False,
    }
    atomic_write_json(OUTPUT_PATH, result)
    result["output_path"] = OUTPUT_PATH.relative_to(ROOT).as_posix()
    result["output_sha256"] = sha256_file(OUTPUT_PATH)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    result = acquire(arguments.timeout_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
