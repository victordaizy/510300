"""取得并归档沪深300在 2015 条件扩展窗口的官方成分链证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_hs_csi300_official_addition_forced_demand_v1 import (  # noqa: E402
    extract_effective_date,
    extract_official_attachment_urls,
)
from research.csi300_pit_membership_weights_source_remediation_v1 import (  # noqa: E402
    REMEDIATION_ID,
    normalize_symbol,
    parse_2015_extension_cycle_changes,
    sha256_file,
)
from scripts.acquire_510300_csi300_pit_weights_evidence_v1 import (  # noqa: E402
    parse_official_closeweight,
)


RAW_ROOT = ROOT / "data" / "raw" / "510300_csi300_pit_membership_weights_source_remediation_v1"
CURATED_ROOT = ROOT / "data" / "curated" / "510300_csi300_pit_membership_weights_source_remediation_v1"
OUTPUT_PATH = CURATED_ROOT / "official_2015_extension_manifest.json"
WEIGHT_EVIDENCE_PATH = CURATED_ROOT / "pit_weight_source_evidence_manifest.json"
CSI_LIST_URL = "https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementByVo"
CSI_DETAIL_URL = "https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementById"
SSE_QUERY_URL = "https://query.sse.com.cn/commonQuery.do"

SEARCH_BODY = {
    "lang": "cn",
    "searchInput": "沪深300",
    "startDate": "2015-01-01",
    "endDate": "2016-06-13",
    "classList": ["index"],
    "indexList": ["csi_index"],
    "relatedTopics": ["index_rebalance"],
    "typeList": ["announcement"],
    "page": {
        "key": "",
        "order": "asc",
        "page": 1,
        "rows": 500,
        "sortBy": "publish_date",
    },
}

TARGET_CYCLES = (
    {
        "cycle_id": "2015_SPECIAL_HYSEC_MERGER",
        "announcement_id": 6271,
        "announcement_date": "2015-01-22",
        "effective_date": "2015-01-26",
        "parse_mode": "INLINE_INDEXED_HTML_TABLES",
        "change_count": 1,
    },
    {
        "cycle_id": "2015_SPECIAL_CNR_OPG_DELISTING",
        "announcement_id": 6802,
        "announcement_date": "2015-05-14",
        "effective_date": "2015-05-20",
        "parse_mode": "INLINE_INDEXED_HTML_TABLES",
        "change_count": 2,
    },
    {
        "cycle_id": "2015_H1_REGULAR",
        "announcement_id": 4084,
        "announcement_date": "2015-06-01",
        "effective_date": "2015-06-15",
        "parse_mode": "STANDARD_OFFICIAL_PARSER",
        "change_count": 18,
    },
    {
        "cycle_id": "2015_H2_REGULAR",
        "announcement_id": 4272,
        "announcement_date": "2015-11-30",
        "effective_date": "2015-12-14",
        "parse_mode": "STANDARD_OFFICIAL_PARSER",
        "change_count": 20,
    },
    {
        "cycle_id": "2015_SPECIAL_CHINA_MERCHANTS_PROPERTY_MERGER",
        "announcement_id": 2882,
        "announcement_date": "2015-12-28",
        "effective_date": "2015-12-30",
        "parse_mode": "COMBINED_EXCEL_000300_ROW",
        "change_count": 1,
    },
)

EXPECTED_SEARCH_IDS = {
    6271,
    6802,
    4084,
    4261,
    4272,
    4290,
    2882,
    4311,
    4330,
    4354,
    4378,
    4417,
    3855,
    4460,
}
SMART_INDEX_IDS = {4261, 4290, 4311, 4330, 4354, 4378, 4417, 4460}
HANDOVER_2016_H1_ANNOUNCEMENT_ID = 3855
REQUIRED_ATTACHMENT_IDS = {4272, 2882}
LEGACY_REWRITTEN_ATTACHMENT_IDS = {6802, 4084}
ANCHOR_DATES = {"2015-03-02", "2015-07-01", "2015-09-01"}


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
    atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def build_session(referer: str) -> requests.Session:
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
            "Referer": referer,
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def checked_get(
    session: requests.Session,
    url: str,
    timeout_seconds: float,
    *,
    params: Mapping[str, Any] | None = None,
) -> tuple[bytes, str, str]:
    response = session.get(url, params=params, timeout=timeout_seconds)
    if response.status_code != 200 or not response.content:
        raise RuntimeError(
            f"官方来源 GET 失败：{url} -> HTTP {response.status_code}，字节={len(response.content)}"
        )
    return response.content, str(response.url), str(response.headers.get("Content-Type") or "")


def checked_post_json(
    session: requests.Session,
    url: str,
    body: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[bytes, str, str]:
    response = session.post(url, json=dict(body), timeout=timeout_seconds)
    if response.status_code != 200 or not response.content:
        raise RuntimeError(
            f"官方来源 POST 失败：{url} -> HTTP {response.status_code}，字节={len(response.content)}"
        )
    return response.content, str(response.url), str(response.headers.get("Content-Type") or "")


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
            raise RuntimeError(f"已有归档对象哈希异常：{path}")
    else:
        atomic_write_bytes(path, payload)
    return path, digest


def source_record(
    kind: str,
    url: str,
    final_url: str,
    content_type: str,
    method: str,
    path: Path,
    payload: bytes,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "url": url,
        "final_url": final_url,
        "method": method,
        "tls_verification": True,
        "content_type": content_type,
        "archive_path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "retrieval_mode": "NETWORK_TLS_VERIFIED",
    }


def parse_detail(payload: bytes, expected_id: int) -> dict[str, Any]:
    wrapped = json.loads(payload.decode("utf-8"))
    if str(wrapped.get("code")) != "200" or not wrapped.get("success"):
        raise RuntimeError(f"中证公告详情业务状态异常：{wrapped.get('code')} {wrapped.get('msg')}")
    detail = wrapped.get("data")
    if not isinstance(detail, dict) or int(detail.get("id")) != expected_id:
        raise RuntimeError(f"中证公告详情 ID 与请求不一致：{expected_id}")
    return detail


def excel_suffix(url: str, payload: bytes) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".xls", ".xlsx"}:
        raise ValueError(f"官方附件不是 XLS/XLSX：{url}")
    if not (
        payload.startswith(bytes.fromhex("D0CF11E0A1B11AE1"))
        or payload.startswith(b"PK\x03\x04")
    ):
        raise ValueError(f"官方附件文件签名不是 XLS/XLSX：{url}")
    return suffix


def parse_sse_jsonp(payload: bytes, expected_code: str) -> dict[str, Any]:
    text = payload.decode("utf-8").strip()
    match = re.fullmatch(r"[A-Za-z0-9_]+\((.*)\)", text, flags=re.DOTALL)
    if not match:
        raise ValueError("上交所终止上市接口没有返回合法 JSONP")
    wrapped = json.loads(match.group(1))
    records = wrapped.get("result")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError(f"上交所终止上市接口目标记录数不是 1：{expected_code}")
    record = records[0]
    if str(record.get("COMPANY_CODE")) != expected_code:
        raise ValueError(f"上交所终止上市接口证券代码不一致：{expected_code}")
    return record


def load_official_2015_anchors() -> list[dict[str, Any]]:
    evidence = json.loads(WEIGHT_EVIDENCE_PATH.read_text(encoding="utf-8"))
    records = (
        evidence.get("official_versioned_snapshot_evidence", {})
        .get("numeric_crosschecks", [])
    )
    anchors: list[dict[str, Any]] = []
    for record in records:
        snapshot_date = str(record.get("snapshot_date"))
        if snapshot_date not in ANCHOR_DATES:
            continue
        path = ROOT / str(record["source_archive_path"])
        expected_sha256 = str(record["source_sha256"])
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"2015 官方权重锚点不存在或哈希漂移：{snapshot_date}")
        parsed = parse_official_closeweight(path.read_bytes())
        if parsed.snapshot_date.isoformat() != snapshot_date:
            raise ValueError(f"2015 官方权重锚点文件内日期不一致：{snapshot_date}")
        set_payload = "\n".join(sorted(parsed.frame["symbol"].tolist())).encode("utf-8")
        anchors.append(
            {
                "snapshot_date": snapshot_date,
                "constituent_count": len(parsed.frame),
                "constituent_set_sha256": sha256_bytes(set_payload),
                "archive_path": path.relative_to(ROOT).as_posix(),
                "sha256": expected_sha256,
                "official_original_url": str(record["official_original_url"]),
                "wayback_capture_timestamp_utc": str(record["wayback_capture_timestamp_utc"]),
                "retrieval_mode": "REUSED_IMMUTABLE_CSI_OFFICIAL_FILE_VIA_INTERNET_ARCHIVE",
            }
        )
    if {item["snapshot_date"] for item in anchors} != ANCHOR_DATES:
        raise ValueError("2015 官方权重锚点日期不完整")
    return sorted(anchors, key=lambda item: item["snapshot_date"])


def acquire(timeout_seconds: float) -> dict[str, Any]:
    csi_session = build_session("https://www.csindex.com.cn/")
    bootstrap, _, _ = checked_get(csi_session, "https://www.csindex.com.cn/", timeout_seconds)
    if len(bootstrap) < 1000:
        raise RuntimeError("中证官网会话初始化内容异常")

    list_payload, list_final_url, list_content_type = checked_post_json(
        csi_session,
        CSI_LIST_URL,
        SEARCH_BODY,
        timeout_seconds,
    )
    list_wrapped = json.loads(list_payload.decode("utf-8"))
    if str(list_wrapped.get("code")) != "200" or not list_wrapped.get("success"):
        raise RuntimeError("中证公告定向查询业务状态失败")
    list_records = list_wrapped.get("data") or []
    observed_ids = {int(item["id"]) for item in list_records}
    if observed_ids != EXPECTED_SEARCH_IDS or int(list_wrapped.get("total", -1)) != 14:
        raise ValueError(
            "2015-2016H1 沪深300官方定向公告集合发生变化："
            f"缺少={sorted(EXPECTED_SEARCH_IDS - observed_ids)}，"
            f"新增={sorted(observed_ids - EXPECTED_SEARCH_IDS)}"
        )
    list_path, _ = archive_content_addressed(
        RAW_ROOT / "2015_extension" / "list",
        "csi300_2015_2016h1_targeted_search",
        list_payload,
        ".json",
    )
    list_source = source_record(
        "CSI_OFFICIAL_2015_EXTENSION_TARGETED_LIST",
        CSI_LIST_URL,
        list_final_url,
        list_content_type,
        "POST",
        list_path,
        list_payload,
    )
    list_source["request_body_sha256"] = sha256_bytes(
        json.dumps(SEARCH_BODY, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    list_source["complete_single_page"] = True
    list_source["observed_total"] = 14

    official_anchors = load_official_2015_anchors()
    anchor_sets: dict[str, frozenset[str]] = {}
    for anchor in official_anchors:
        parsed = parse_official_closeweight((ROOT / anchor["archive_path"]).read_bytes())
        anchor_sets[anchor["snapshot_date"]] = frozenset(parsed.frame["symbol"])

    sse_session = build_session("https://www.sse.com.cn/assortment/stock/list/delisting/")
    sse_sources: dict[str, dict[str, Any]] = {}
    delisting_dates: dict[str, str] = {}
    for code in ("601299", "600832"):
        params = {
            "jsonCallBack": "jsonpCallback",
            "isPagination": "true",
            "sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L",
            "STOCK_CODE": code,
            "CSRC_CODE": "",
            "REG_PROVINCE": "",
            "STOCK_TYPE": "1,2",
            "COMPANY_STATUS": "3",
            "type": "inParams",
            "pageHelp.cacheSize": "1",
            "pageHelp.beginPage": "1",
            "pageHelp.pageSize": "25",
            "pageHelp.pageNo": "1",
        }
        payload, final_url, content_type = checked_get(
            sse_session,
            SSE_QUERY_URL,
            timeout_seconds,
            params=params,
        )
        record = parse_sse_jsonp(payload, code)
        delisting_date = str(record.get("DELIST_DATE") or "")
        if delisting_date != "20150520":
            raise ValueError(f"上交所 {code} 终止上市日不是 2015-05-20：{delisting_date}")
        delisting_dates[code] = "2015-05-20"
        path, _ = archive_content_addressed(
            RAW_ROOT / "2015_extension" / "sse_delisting",
            f"security_{code}",
            payload,
            ".jsonp",
        )
        record_source = source_record(
            "SSE_OFFICIAL_DELISTING_QUERY",
            SSE_QUERY_URL,
            final_url,
            content_type,
            "GET",
            path,
            payload,
        )
        record_source["request_parameters_sha256"] = sha256_bytes(
            json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        record_source["security_code"] = code
        record_source["delisting_date"] = "2015-05-20"
        sse_sources[code] = record_source

    target_by_id = {int(item["announcement_id"]): item for item in TARGET_CYCLES}
    cycles: list[dict[str, Any]] = []
    source_objects: list[dict[str, Any]] = [list_source, *sse_sources.values()]

    for announcement_id, specification in target_by_id.items():
        detail_url = f"{CSI_DETAIL_URL}?id={announcement_id}&lang=cn"
        detail_payload, detail_final_url, detail_content_type = checked_get(
            csi_session,
            detail_url,
            timeout_seconds,
        )
        detail = parse_detail(detail_payload, announcement_id)
        if str(detail.get("publishDate"))[:10] != specification["announcement_date"]:
            raise ValueError(f"公告 {announcement_id} 发布日发生漂移")
        detail_path, _ = archive_content_addressed(
            RAW_ROOT / "2015_extension" / "details",
            f"announcement_{announcement_id}",
            detail_payload,
            ".json",
        )
        detail_source = source_record(
            "CSI_OFFICIAL_2015_EXTENSION_DETAIL",
            detail_url,
            detail_final_url,
            detail_content_type,
            "GET",
            detail_path,
            detail_payload,
        )
        source_objects.append(detail_source)

        attachment_sources: list[dict[str, Any]] = []
        attachment_payloads: list[dict[str, Any]] = []
        attachment_urls = extract_official_attachment_urls(detail)
        if announcement_id in REQUIRED_ATTACHMENT_IDS:
            valid_excel_urls = [
                item for item in attachment_urls if Path(urlparse(str(item["url"])).path).suffix.lower() in {".xls", ".xlsx"}
            ]
            if len(valid_excel_urls) != 1:
                raise ValueError(f"公告 {announcement_id} 有效官方 Excel 附件数量不是 1")
            attachment = valid_excel_urls[0]
            payload, final_url, content_type = checked_get(
                csi_session,
                str(attachment["url"]),
                timeout_seconds,
            )
            suffix = excel_suffix(str(attachment["url"]), payload)
            path, _ = archive_content_addressed(
                RAW_ROOT / "2015_extension" / "attachments",
                f"announcement_{announcement_id}_attachment_1",
                payload,
                suffix,
            )
            attachment_source = source_record(
                "CSI_OFFICIAL_2015_EXTENSION_ATTACHMENT",
                str(attachment["url"]),
                final_url,
                content_type,
                "GET",
                path,
                payload,
            )
            attachment_source["file_name"] = str(attachment.get("file_name") or "")
            attachment_sources.append(attachment_source)
            attachment_payloads.append(
                {**attachment_source, "payload": payload, "suffix": suffix}
            )
            source_objects.append(attachment_source)
        elif announcement_id in LEGACY_REWRITTEN_ATTACHMENT_IDS:
            if not attachment_urls or not all(
                str(item["url"]).endswith("/20250513/NHR.txt") for item in attachment_urls
            ):
                raise ValueError(f"公告 {announcement_id} 的旧附件重写状态发生变化")

        parsed = parse_2015_extension_cycle_changes(
            detail,
            attachment_payloads,
            str(specification["parse_mode"]),
            int(specification["change_count"]),
        )
        if announcement_id == 6802:
            if delisting_dates != {"601299": "2015-05-20", "600832": "2015-05-20"}:
                raise ValueError("2015 年 5 月两项退市日未共同确认")
            effective_sources = [sse_sources["601299"], sse_sources["600832"]]
            extracted_effective_date = "2015-05-20"
            effective_date_rule = "CSI_FROM_EACH_DELISTING_DATE_PLUS_SSE_EXACT_DELISTING_QUERY"
        else:
            extracted_effective_date = extract_effective_date(
                str(detail.get("content") or ""),
                str(detail["publishDate"])[:10],
            )
            effective_sources = [detail_source]
            effective_date_rule = "CSI_DETAIL_EXPLICIT_EFFECTIVE_DATE"
        if extracted_effective_date != specification["effective_date"]:
            raise ValueError(
                f"公告 {announcement_id} 生效日不一致：{extracted_effective_date}"
            )

        additions = [
            {
                "symbol": normalize_symbol(item["security_code"]),
                "name": str(item.get("security_name") or ""),
            }
            for item in parsed["additions"]
        ]
        deletions = [
            {
                "symbol": normalize_symbol(item["security_code"]),
                "name": str(item.get("security_name") or ""),
            }
            for item in parsed["deletions"]
        ]
        if announcement_id == 4272:
            pre_state = anchor_sets["2015-09-01"]
            parsed_additions = {item["symbol"] for item in additions}
            parsed_deletions = {item["symbol"] for item in deletions}
            if parsed_additions & pre_state or not parsed_deletions <= pre_state:
                raise ValueError("2015H2 官方工作簿的调入/调出工作表角色未通过 9 月官方锚点校验")

        cycles.append(
            {
                **dict(specification),
                "announcement_title": str(detail["title"]),
                "effective_date_rule": effective_date_rule,
                "source_format": str(parsed["source_format"]),
                "extraction_method": str(parsed["method"]),
                "additions": additions,
                "deletions": deletions,
                "detail_source": detail_source,
                "attachment_sources": attachment_sources,
                "effective_date_sources": effective_sources,
                "legacy_attachment_resolution": (
                    "CURRENT_CSI_API_REWRITES_LEGACY_LINK_TO_NHR_TXT_INLINE_TABLE_USED"
                    if announcement_id in LEGACY_REWRITTEN_ATTACHMENT_IDS
                    else "NOT_APPLICABLE"
                ),
            }
        )

    cycles.sort(key=lambda item: item["effective_date"])
    if [item["cycle_id"] for item in cycles] != [item["cycle_id"] for item in TARGET_CYCLES]:
        raise ValueError("2015 官方扩展周期顺序异常")

    for anchor in official_anchors:
        source_objects.append(
            {
                "kind": "CSI_OFFICIAL_2015_MEMBERSHIP_ANCHOR_VIA_INTERNET_ARCHIVE",
                "url": anchor["official_original_url"],
                "method": "GET_BY_PRIOR_ARCHIVE_CAPTURE",
                "tls_verification": True,
                "archive_path": anchor["archive_path"],
                "sha256": anchor["sha256"],
                "size_bytes": (ROOT / anchor["archive_path"]).stat().st_size,
                "retrieval_mode": anchor["retrieval_mode"],
                "snapshot_date": anchor["snapshot_date"],
                "wayback_capture_timestamp_utc": anchor["wayback_capture_timestamp_utc"],
            }
        )

    search_classification = [
        {
            "announcement_id": int(item["id"]),
            "publish_date": str(item["publishDate"])[:10],
            "title": str(item["title"]),
            "classification": (
                "INCLUDED_2015_STANDARD_CSI300_CHANGE"
                if int(item["id"]) in target_by_id
                else (
                    "EXCLUDED_DIFFERENT_INDEX_930667_CSI300_SMART"
                    if int(item["id"]) in SMART_INDEX_IDS
                    else "HANDOVER_2016H1_STANDARD_CSI300_CHANGE_ALREADY_IN_PRIMARY_CHAIN"
                )
            ),
        }
        for item in sorted(list_records, key=lambda value: (str(value["publishDate"]), int(value["id"])))
    ]
    if sum(item["classification"].startswith("INCLUDED") for item in search_classification) != 5:
        raise ValueError("2015 官方公告分类后的纳入数量不是 5")
    if sum(item["classification"].startswith("EXCLUDED") for item in search_classification) != 8:
        raise ValueError("2015 官方公告分类后的精明指数排除数量不是 8")
    if sum(item["classification"].startswith("HANDOVER") for item in search_classification) != 1:
        raise ValueError("2016H1 交接公告分类数量不是 1")

    result = {
        "remediation_id": REMEDIATION_ID,
        "version": "1.0.2-2015-extension-source",
        "status": "PASS_OFFICIAL_2015_MEMBERSHIP_EXTENSION_SOURCES_ACQUIRED",
        "retrieved_at": now_shanghai(),
        "coverage": {
            "start_date": "2015-01-01",
            "end_date": "2016-06-12",
            "handover_standard_announcement_id": HANDOVER_2016_H1_ANNOUNCEMENT_ID,
            "targeted_search_result_count": len(list_records),
            "included_cycle_count": len(cycles),
            "excluded_csi300_smart_announcement_count": len(SMART_INDEX_IDS),
        },
        "announcement_search_classification": search_classification,
        "official_membership_anchors": official_anchors,
        "cycle_count": len(cycles),
        "cycles": cycles,
        "source_objects": source_objects,
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "model_read": False,
        "portfolio_evaluation": "NOT_ALLOWED",
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
