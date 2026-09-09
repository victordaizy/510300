"""补齐 510300 非对称压力风险 V1 的三项冻结来源。

仅获取申万官方行业历史、央行公开市场公告，并为授权 DR007 导出提供
失败即关闭的导入入口。本脚本不读取 BAD10、未来收益、模型或组合结果。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.asymmetric_stress_hazard_source_remediation_v1_0_1 import (  # noqa: E402
    CHINAMONEY_DAILY_URL,
    OBSERVATION_CUTOFF,
    OBSERVATION_START,
    PBOC_LIST_ROOT,
    REMEDIATION_ID,
    SW_CODEBOOK_URL,
    SW_HISTORY_URL,
    build_csi300_member_day_sw_industry,
    parse_pboc_open_market_notice,
    parse_sw_official_workbooks,
    sha256_bytes,
    sha256_file,
    summarize_pboc_policy_rate_ledger,
    validate_dr007_daily,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)
RAW_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "remediation"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1"
)
CURATED_ROOT = (
    ROOT
    / "data"
    / "curated"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1"
)
MEMBERSHIP_PATH = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "000300_daily_pit_membership_20150101_20260814.parquet"
)
PBOC_ANCHOR_SEARCH_START = date(2014, 1, 1)
CHINAMONEY_PROBE_DATES = ("2026-08-14", "2026-08-19", "2026-09-01")
ALLOWED_SOURCES = frozenset({"sw", "pboc", "dr007-probe", "dr007-import"})


def now_shanghai() -> str:
    return datetime.now(TIMEZONE).isoformat()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


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


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _serializable_headers(headers: Any) -> dict[str, str]:
    retained = {"content-type", "content-length", "last-modified", "etag", "date"}
    return {
        key: str(value)
        for key, value in headers.items()
        if key.lower() in retained
    }


def request_bytes(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 45.0,
) -> tuple[bytes, dict[str, str]]:
    merged_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/json,application/xhtml+xml,*/*;q=0.8",
    }
    if headers:
        merged_headers.update(headers)
    context = ssl.create_default_context()
    error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = Request(
                url,
                data=body,
                headers=merged_headers,
                method=method,
            )
            with urlopen(request, timeout=timeout_seconds, context=context) as response:
                payload = response.read()
                if response.status != 200:
                    raise RuntimeError(f"HTTP 状态异常：{response.status}")
                if not payload:
                    raise RuntimeError("来源返回空内容")
                return payload, _serializable_headers(response.headers)
        except Exception as exc:  # noqa: BLE001 - 网络失败需重试并保留原异常
            error = exc
            if attempt < 3:
                time.sleep(float(attempt))
    raise RuntimeError(f"官方来源获取失败：{url}；{type(error).__name__}: {error}")


def load_or_acquire_raw(
    path: Path,
    *,
    url: str,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    sidecar = path.with_name(f"{path.name}.source.json")
    request_body_sha256 = sha256_bytes(body) if body is not None else None
    if path.is_file():
        if not sidecar.is_file():
            raise RuntimeError(f"冻结原始文件缺少来源侧车：{relative(path)}")
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        payload = path.read_bytes()
        if record.get("url") != url or record.get("method") != method:
            raise RuntimeError(f"冻结原始文件请求身份漂移：{relative(path)}")
        if record.get("request_body_sha256") != request_body_sha256:
            raise RuntimeError(f"冻结原始文件请求正文漂移：{relative(path)}")
        if record.get("sha256") != sha256_bytes(payload):
            raise RuntimeError(f"冻结原始文件哈希校验失败：{relative(path)}")
        return payload, {**record, "retrieval_mode": "FROZEN_RAW_RESUME"}

    payload, response_headers = request_bytes(
        url,
        method=method,
        body=body,
        headers=headers,
    )
    acquired_at = now_shanghai()
    record = {
        "url": url,
        "method": method,
        "request_body_sha256": request_body_sha256,
        "archive_path": relative(path),
        "source_sidecar_path": relative(sidecar),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "retrieved_at": acquired_at,
        "tls_verification": True,
        "response_headers": response_headers,
        "retrieval_mode": "NETWORK_TLS_VERIFIED",
    }
    atomic_write_bytes(path, payload)
    atomic_write_json(sidecar, record)
    return payload, record


def artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def acquire_sw() -> dict[str, Any]:
    started_at = now_shanghai()
    history_raw = RAW_ROOT / "sw" / "StockClassifyUse_stock.xls"
    codebook_raw = RAW_ROOT / "sw" / "SwClassCode_2021.xls"
    history_payload, history_receipt = load_or_acquire_raw(
        history_raw, url=SW_HISTORY_URL
    )
    codebook_payload, codebook_receipt = load_or_acquire_raw(
        codebook_raw, url=SW_CODEBOOK_URL
    )
    retrieved_at = max(
        history_receipt["retrieved_at"], codebook_receipt["retrieved_at"]
    )
    history, codebook, source_metrics = parse_sw_official_workbooks(
        history_payload,
        codebook_payload,
        retrieved_at=retrieved_at,
    )
    membership = pd.read_parquet(MEMBERSHIP_PATH)
    member_day, coverage_metrics = build_csi300_member_day_sw_industry(
        membership,
        history,
    )

    history_path = CURATED_ROOT / "sw_industry_history_events_through_20260814.parquet"
    codebook_path = CURATED_ROOT / "sw_2021_codebook.parquet"
    member_day_path = (
        CURATED_ROOT / "000300_member_day_sw_l1_pit_20150105_20260814.parquet"
    )
    atomic_write_parquet(history_path, history)
    atomic_write_parquet(codebook_path, codebook)
    atomic_write_parquet(member_day_path, member_day)
    manifest = {
        "remediation_id": REMEDIATION_ID,
        "source_id": "SW_OFFICIAL_INDUSTRY_HISTORY_WITH_RECORD_UPDATE_CLOCK",
        "status": "PASS_SOURCE_ACQUIRED_PIT_NO_VIEW_PRESERVED",
        "started_at": started_at,
        "completed_at": now_shanghai(),
        "observation_start": OBSERVATION_START.isoformat(),
        "observation_cutoff": OBSERVATION_CUTOFF.isoformat(),
        "raw_sources": [history_receipt, codebook_receipt],
        "source_metrics": source_metrics,
        "coverage_metrics": coverage_metrics,
        "artifacts": {
            "history_events": artifact_record(history_path, rows=len(history)),
            "codebook": artifact_record(codebook_path, rows=len(codebook)),
            "member_day_pit": artifact_record(member_day_path, rows=len(member_day)),
        },
        "availability_clock": (
            "RECORD_UPDATED_AT_NOT_LATER_THAN_MEMBER_DATE_15_00_ASIA_SHANGHAI"
        ),
        "missing_value_rule": "NO_VIEW_NO_INTERPOLATION",
        "future_revision_backfill_used": False,
        "feature_construction_performed": False,
        "future_returns_read": False,
    }
    manifest_path = CURATED_ROOT / "sw_source_acquisition_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest": artifact_record(manifest_path)}


def pboc_list_url(page_number: int) -> str:
    if page_number == 1:
        return f"{PBOC_LIST_ROOT}/index.html"
    return f"{PBOC_LIST_ROOT}/17081-{page_number}.html"


def parse_pboc_list_page(payload: bytes) -> tuple[int, list[dict[str, Any]]]:
    try:
        html = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("央行公告列表不是 UTF-8 HTML") from exc
    soup = BeautifulSoup(html, "html.parser")
    paging = soup.find("input", attrs={"name": "article_paging_list_hidden"})
    if paging is None or not str(paging.get("totalpage", "")).isdigit():
        raise ValueError("央行公告列表缺少 totalpage")
    total_pages = int(str(paging["totalpage"]))
    records: list[dict[str, Any]] = []
    for anchor in soup.find_all("a", href=True):
        title = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()
        if not re.search(r"公开市场业务交易公告\s*\[20\d{2}\]第\d+号", title):
            continue
        row = anchor.find_parent("tr")
        row_text = (
            re.sub(r"\s+", " ", row.get_text(" ", strip=True)).strip()
            if row is not None
            else title
        )
        date_match = re.search(r"20\d{2}-\d{2}-\d{2}", row_text)
        if date_match is None:
            raise ValueError(f"央行公告列表条目缺少日期：{title}")
        article_date = date.fromisoformat(date_match.group(0))
        url = urljoin("https://www.pbc.gov.cn", str(anchor["href"]))
        records.append(
            {"list_date": article_date, "list_title": title, "source_url": url}
        )
    unique = {record["source_url"]: record for record in records}
    if not unique:
        raise ValueError("央行公告列表页没有识别到公告")
    return total_pages, list(unique.values())


def _acquire_pboc_list_page(page_number: int) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    url = pboc_list_url(page_number)
    path = RAW_ROOT / "pboc" / "list_pages" / f"page_{page_number:03d}.html"
    payload, receipt = load_or_acquire_raw(path, url=url)
    total_pages, records = parse_pboc_list_page(payload)
    return total_pages, records, receipt


def acquire_pboc_list_index(workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_pages, first_records, first_receipt = _acquire_pboc_list_page(1)
    records = list(first_records)
    receipts = [first_receipt]
    page_number = 2
    reached_anchor_floor = min(item["list_date"] for item in first_records) < PBOC_ANCHOR_SEARCH_START
    while page_number <= total_pages and not reached_anchor_floor:
        batch = list(range(page_number, min(total_pages + 1, page_number + workers)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_acquire_pboc_list_page, value): value for value in batch
            }
            batch_results = []
            for future in as_completed(futures):
                batch_results.append((futures[future], future.result()))
        for _, (observed_total, page_records, receipt) in sorted(batch_results):
            if observed_total != total_pages:
                raise RuntimeError("央行公告分页总页数在同次冻结获取中发生漂移")
            records.extend(page_records)
            receipts.append(receipt)
        batch_dates = [
            item["list_date"] for _, (_, page_records, _) in batch_results for item in page_records
        ]
        reached_anchor_floor = min(batch_dates) < PBOC_ANCHOR_SEARCH_START
        page_number = batch[-1] + 1
        if page_number % 20 == 2 or reached_anchor_floor:
            print(
                json.dumps(
                    {
                        "stage": "PBOC_LIST",
                        "pages_archived": len(receipts),
                        "earliest_list_date": min(
                            item["list_date"] for item in records
                        ).isoformat(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    unique = {record["source_url"]: record for record in records}
    selected = [
        record
        for record in unique.values()
        if PBOC_ANCHOR_SEARCH_START <= record["list_date"] <= OBSERVATION_CUTOFF
    ]
    selected.sort(key=lambda value: (value["list_date"], value["source_url"]))
    if not selected:
        raise RuntimeError("央行公告列表没有覆盖锚定搜索期至观察截止日")
    return selected, receipts


def _pboc_article_path(record: dict[str, Any]) -> Path:
    match = re.search(r"/([^/]+)/index\.html$", record["source_url"])
    if match is None:
        raise ValueError(f"央行公告 URL 身份不可解析：{record['source_url']}")
    return (
        RAW_ROOT
        / "pboc"
        / "articles"
        / f"{record['list_date'].isoformat()}_{match.group(1)}.html"
    )


def _acquire_pboc_article(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _pboc_article_path(record)
    payload, receipt = load_or_acquire_raw(path, url=record["source_url"])
    parsed = parse_pboc_open_market_notice(payload, source_url=record["source_url"])
    if parsed.notice_date != record["list_date"]:
        raise ValueError(
            f"央行公告列表日期与正文日期不一致：{record['source_url']}"
        )
    output = asdict(parsed)
    output["notice_date"] = parsed.notice_date.isoformat()
    output["published_at"] = parsed.published_at.isoformat()
    output["list_title"] = record["list_title"]
    output["raw_path"] = relative(path)
    return output, receipt


def acquire_pboc(workers: int) -> dict[str, Any]:
    started_at = now_shanghai()
    article_index, list_receipts = acquire_pboc_list_index(workers)
    parsed_records: list[dict[str, Any]] = []
    article_receipts: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_acquire_pboc_article, record): record
            for record in article_index
        }
        completed = 0
        for future in as_completed(futures):
            record = futures[future]
            try:
                parsed, receipt = future.result()
                parsed_records.append(parsed)
                article_receipts.append(receipt)
            except Exception as exc:  # noqa: BLE001 - 逐公告失败必须完整落账
                failures.append(
                    {
                        "source_url": record["source_url"],
                        "list_date": record["list_date"].isoformat(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            completed += 1
            if completed % 100 == 0 or completed == len(article_index):
                print(
                    json.dumps(
                        {
                            "stage": "PBOC_ARTICLES",
                            "completed": completed,
                            "total": len(article_index),
                            "failure_count": len(failures),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    if failures:
        attempt_path = RAW_ROOT / "attempts" / "pboc_acquisition_failure_receipt.json"
        atomic_write_json(
            attempt_path,
            {
                "remediation_id": REMEDIATION_ID,
                "status": "BLOCKED_PBOC_ARTICLE_ACQUISITION_OR_PARSE_FAILURE",
                "started_at": started_at,
                "completed_at": now_shanghai(),
                "article_count": len(article_index),
                "failure_count": len(failures),
                "failures": failures,
                "future_returns_read": False,
            },
        )
        raise RuntimeError(
            f"央行公告有 {len(failures)} 条获取或解析失败；见 {relative(attempt_path)}"
        )

    ledger = pd.DataFrame(parsed_records)
    ledger["notice_date"] = pd.to_datetime(ledger["notice_date"])
    ledger["published_at"] = pd.to_datetime(ledger["published_at"])
    ledger = ledger.sort_values(["notice_date", "published_at", "source_url"])
    published_mask = pd.to_numeric(
        ledger["seven_day_rate_percent"], errors="coerce"
    ).notna()
    pre_window_rates = ledger.loc[
        published_mask & (ledger["notice_date"] < pd.Timestamp(OBSERVATION_START))
    ]
    if pre_window_rates.empty:
        raise RuntimeError("央行 7 天逆回购操作利率缺少观察期前官方锚点")
    anchor = pre_window_rates.sort_values(["notice_date", "published_at"]).iloc[-1]
    retained = ledger.loc[
        (ledger["notice_date"] >= pd.Timestamp(OBSERVATION_START))
        | ledger["source_url"].eq(anchor["source_url"])
    ].reset_index(drop=True)
    summary = summarize_pboc_policy_rate_ledger(retained)
    changes = summary.pop("rate_changes")
    published = retained.loc[
        pd.to_numeric(retained["seven_day_rate_percent"], errors="coerce").notna()
    ].copy()
    published = (
        published.sort_values(["notice_date", "published_at"])
        .drop_duplicates("notice_date", keep="last")
        .reset_index(drop=True)
    )

    ledger_path = (
        CURATED_ROOT / "pboc_open_market_notice_ledger_20150105_20260814.parquet"
    )
    published_path = (
        CURATED_ROOT
        / "pboc_7d_reverse_repo_published_rates_anchor_20150105_20260814.parquet"
    )
    changes_path = (
        CURATED_ROOT
        / "pboc_7d_reverse_repo_rate_changes_anchor_20150105_20260814.parquet"
    )
    atomic_write_parquet(ledger_path, retained)
    atomic_write_parquet(published_path, published)
    atomic_write_parquet(changes_path, changes)
    manifest = {
        "remediation_id": REMEDIATION_ID,
        "source_id": "PBOC_7D_REVERSE_REPO_ACTUAL_PUBLISHED_OPERATION_RATE",
        "status": "PASS_SOURCE_ACQUIRED_WITH_PRE_WINDOW_OFFICIAL_ANCHOR",
        "started_at": started_at,
        "completed_at": now_shanghai(),
        "observation_start": OBSERVATION_START.isoformat(),
        "observation_cutoff": OBSERVATION_CUTOFF.isoformat(),
        "anchor_search_start": PBOC_ANCHOR_SEARCH_START.isoformat(),
        "pre_window_anchor": {
            "notice_date": anchor["notice_date"].date().isoformat(),
            "published_at": anchor["published_at"].isoformat(),
            "rate_percent": float(anchor["seven_day_rate_percent"]),
            "source_url": anchor["source_url"],
            "raw_sha256": anchor["raw_sha256"],
        },
        "list_page_count_archived": len(list_receipts),
        "article_count_archived": len(article_receipts),
        "summary": summary,
        "artifacts": {
            "notice_ledger": artifact_record(ledger_path, rows=len(retained)),
            "published_rates": artifact_record(published_path, rows=len(published)),
            "rate_changes": artifact_record(changes_path, rows=len(changes)),
        },
        "raw_manifest_digest": sha256_bytes(
            json.dumps(
                sorted(
                    [
                        {
                            "archive_path": item["archive_path"],
                            "sha256": item["sha256"],
                            "url": item["url"],
                        }
                        for item in list_receipts + article_receipts
                    ],
                    key=lambda value: value["archive_path"],
                ),
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ),
        "interpolation_performed": False,
        "inferred_change_dates_used": False,
        "zero_operation_rate_guessed": False,
        "feature_construction_performed": False,
        "future_returns_read": False,
    }
    manifest_path = CURATED_ROOT / "pboc_source_acquisition_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest": artifact_record(manifest_path)}


def _chinamoney_body(probe_date: str) -> bytes:
    return urlencode(
        {
            "lang": "en",
            "indexType": "markInterBankVOList",
            "searchDate": probe_date,
            "publishedTime": "2200",
        }
    ).encode("ascii")


def acquire_dr007_public_probe() -> dict[str, Any]:
    started_at = now_shanghai()
    probes: list[dict[str, Any]] = []
    raw_receipts: list[dict[str, Any]] = []
    for probe_date in CHINAMONEY_PROBE_DATES:
        body = _chinamoney_body(probe_date)
        path = RAW_ROOT / "chinamoney" / f"dr007_public_probe_{probe_date}.json"
        payload, receipt = load_or_acquire_raw(
            path,
            url=CHINAMONEY_DAILY_URL,
            method="POST",
            body=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Referer": "https://www.chinamoney.com.cn/chinese/mtdexdaily/?tab=2",
                "Accept": "application/json, text/plain, */*",
            },
        )
        parsed = json.loads(payload.decode("utf-8"))
        if str(parsed.get("head", {}).get("rep_code")) != "200":
            raise RuntimeError(f"ChinaMoney DR007 探针返回业务失败：{probe_date}")
        dr007_rows = [
            row
            for row in parsed.get("records", [])
            if str(row.get("instrmntCd", "")).upper() == "DR007"
        ]
        if len(dr007_rows) > 1:
            raise RuntimeError(f"ChinaMoney 同日返回多个 DR007：{probe_date}")
        probes.append(
            {
                "probe_date": probe_date,
                "public_start_date": parsed.get("data", {}).get("startDate"),
                "public_end_date": parsed.get("data", {}).get("endDate"),
                "dr007_present": len(dr007_rows) == 1,
                "dr007_weighted_average_rate_percent": (
                    float(dr007_rows[0]["wghtdAvgRepoRate"])
                    if dr007_rows
                    else None
                ),
                "raw_path": relative(path),
                "raw_sha256": sha256_bytes(payload),
            }
        )
        raw_receipts.append(receipt)
    starts = sorted(
        {
            item["public_start_date"]
            for item in probes
            if item["public_start_date"]
        }
    )
    if not starts or min(starts) <= OBSERVATION_START.isoformat():
        raise RuntimeError("ChinaMoney 公共端点保留窗口探针结果不符合预期")
    manifest = {
        "remediation_id": REMEDIATION_ID,
        "source_id": "CHINAMONEY_PUBLIC_DR007_RETENTION_PROBE",
        "status": "BLOCKED_PUBLIC_ENDPOINT_RETENTION_DOES_NOT_COVER_FROZEN_WINDOW",
        "started_at": started_at,
        "completed_at": now_shanghai(),
        "observation_start": OBSERVATION_START.isoformat(),
        "observation_cutoff": OBSERVATION_CUTOFF.isoformat(),
        "public_retention_start_dates_observed": starts,
        "probes": probes,
        "raw_sources": raw_receipts,
        "selected_series": "DR007",
        "substitute_used": False,
        "credential_bypass_attempted": False,
        "required_next_action": (
            "IMPORT_LICENSED_DR007_WEIGHTED_AVERAGE_DAILY_EXPORT_WITH_PROVENANCE"
        ),
        "future_returns_read": False,
    }
    manifest_path = CURATED_ROOT / "dr007_public_retention_probe_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest": artifact_record(manifest_path)}


def load_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"不支持的 DR007 导出格式：{path.suffix}")


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"DR007 来源证明缺少字段：{key}")
    return value


def import_dr007(source_path: Path, provenance_path: Path) -> dict[str, Any]:
    started_at = now_shanghai()
    if not source_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("DR007 导出文件或来源证明文件不存在")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_identity = _required_text(provenance, "source_identity")
    series_code = _required_text(provenance, "series_code").upper()
    value_semantics = _required_text(provenance, "value_semantics").upper()
    date_column = _required_text(provenance, "date_column")
    value_column = _required_text(provenance, "value_column")
    availability_rule = _required_text(provenance, "availability_rule").upper()
    declared_hash = _required_text(provenance, "original_file_sha256").lower()
    _required_text(provenance, "source_url")
    _required_text(provenance, "retrieved_at")
    _required_text(provenance, "license_or_account_basis")
    if series_code != "DR007":
        raise ValueError(f"DR007 来源证明 series_code 不是 DR007：{series_code}")
    if value_semantics != "WEIGHTED_AVERAGE_RATE_PERCENT":
        raise ValueError(
            "DR007 来源证明 value_semantics 必须是 WEIGHTED_AVERAGE_RATE_PERCENT"
        )
    if availability_rule != "NEXT_TRADING_DAY_OPEN_AFTER_RATE_DATE":
        raise ValueError(
            "DR007 来源证明 availability_rule 必须是 "
            "NEXT_TRADING_DAY_OPEN_AFTER_RATE_DATE"
        )
    actual_hash = sha256_file(source_path)
    if declared_hash != actual_hash:
        raise ValueError("DR007 来源证明的 original_file_sha256 与导出文件不一致")
    frame = load_tabular(source_path)
    missing_columns = [
        column for column in (date_column, value_column) if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"DR007 导出文件缺少声明列：{missing_columns}")
    normalized = frame.loc[:, [date_column, value_column]].rename(
        columns={date_column: "date", value_column: "dr007"}
    )
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["dr007"] = pd.to_numeric(normalized["dr007"], errors="coerce")
    membership = pd.read_parquet(MEMBERSHIP_PATH, columns=["membership_date"])
    market_dates = membership["membership_date"].drop_duplicates().sort_values()
    metrics = validate_dr007_daily(
        normalized,
        market_dates=market_dates,
        source_identity=source_identity,
    )
    normalized = normalized.sort_values("date").reset_index(drop=True)
    normalized["series_code"] = "DR007"
    normalized["value_semantics"] = "WEIGHTED_AVERAGE_RATE_PERCENT"
    normalized["source_identity"] = source_identity
    normalized["source_url"] = provenance["source_url"]
    normalized["retrieved_at"] = provenance["retrieved_at"]
    normalized["availability_rule"] = availability_rule
    normalized["original_file_sha256"] = actual_hash
    normalized["missing_value_rule"] = "NO_VIEW_NO_INTERPOLATION"

    raw_copy = RAW_ROOT / "dr007_licensed_import" / f"{actual_hash}{source_path.suffix.lower()}"
    if raw_copy.is_file() and sha256_file(raw_copy) != actual_hash:
        raise RuntimeError("已有 DR007 原始导入归档哈希异常")
    if not raw_copy.is_file():
        atomic_write_bytes(raw_copy, source_path.read_bytes())
    raw_provenance = RAW_ROOT / "dr007_licensed_import" / f"{actual_hash}.provenance.json"
    provenance_archive = {
        **provenance,
        "archived_at": now_shanghai(),
        "archive_path": relative(raw_copy),
    }
    atomic_write_json(raw_provenance, provenance_archive)

    output_path = CURATED_ROOT / "dr007_daily_20150105_20260814.parquet"
    atomic_write_parquet(output_path, normalized)
    status = (
        "PASS_DR007_SOURCE_ADMITTED_FULL_MARKET_SESSION_COVERAGE"
        if metrics["missing_market_session_count"] == 0
        else "PASS_DR007_SOURCE_ADMITTED_WITH_DATE_LEVEL_NO_VIEW"
    )
    manifest = {
        "remediation_id": REMEDIATION_ID,
        "source_id": "LICENSED_DR007_WEIGHTED_AVERAGE_DAILY_EXPORT",
        "status": status,
        "started_at": started_at,
        "completed_at": now_shanghai(),
        "metrics": metrics,
        "provenance": provenance_archive,
        "raw_artifact": artifact_record(raw_copy),
        "curated_artifact": artifact_record(output_path, rows=len(normalized)),
        "substitute_used": False,
        "interpolation_performed": False,
        "future_returns_read": False,
    }
    manifest_path = CURATED_ROOT / "dr007_source_acquisition_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest": artifact_record(manifest_path)}


def parse_sources(value: str) -> list[str]:
    sources = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(sources).difference(ALLOWED_SOURCES))
    if unknown:
        raise argparse.ArgumentTypeError(f"未知来源：{unknown}")
    if not sources:
        raise argparse.ArgumentTypeError("至少选择一个来源")
    return list(dict.fromkeys(sources))


def existing_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    return str(json.loads(path.read_text(encoding="utf-8")).get("status"))


def build_overall_manifest(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sw_status = existing_status(CURATED_ROOT / "sw_source_acquisition_manifest.json")
    pboc_status = existing_status(CURATED_ROOT / "pboc_source_acquisition_manifest.json")
    dr007_status = existing_status(CURATED_ROOT / "dr007_source_acquisition_manifest.json")
    sw_pass = bool(sw_status and sw_status.startswith("PASS_"))
    pboc_pass = bool(pboc_status and pboc_status.startswith("PASS_"))
    dr007_pass = bool(dr007_status and dr007_status.startswith("PASS_"))
    all_pass = sw_pass and pboc_pass and dr007_pass
    return {
        "remediation_id": REMEDIATION_ID,
        "status": (
            "READY_FOR_VERSIONED_SOURCE_ADMISSION"
            if all_pass
            else "BLOCKED_REQUIRED_SOURCES_NOT_ALL_ACQUIRED"
        ),
        "generated_at": now_shanghai(),
        "source_status": {
            "sw_pit_industry": sw_status or "NOT_ACQUIRED",
            "pboc_7d_reverse_repo_rate": pboc_status or "NOT_ACQUIRED",
            "dr007_daily": dr007_status or "BLOCKED_LICENSED_HISTORY_REQUIRED",
        },
        "sources_run_this_invocation": sorted(results),
        "all_required_sources_acquired": all_pass,
        "feature_construction_allowed": False,
        "feature_construction_requires_separate_admission_receipt": True,
        "model_training_allowed": False,
        "portfolio_evaluation_allowed": False,
        "future_returns_read": False,
        "substitute_or_proxy_used": False,
        "required_next_action": (
            "RUN_VERSIONED_SOURCE_ADMISSION"
            if all_pass
            else "IMPORT_LICENSED_DR007_WEIGHTED_AVERAGE_DAILY_EXPORT_WITH_PROVENANCE"
        ),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=parse_sources,
        default=parse_sources("sw,pboc,dr007-probe"),
        help="逗号分隔：sw,pboc,dr007-probe,dr007-import",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dr007-import", type=Path)
    parser.add_argument("--dr007-provenance", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise ValueError("workers 必须在 1 到 8 之间")
    if "dr007-import" in args.sources and (
        args.dr007_import is None or args.dr007_provenance is None
    ):
        raise ValueError("dr007-import 必须同时提供导出文件和来源证明 JSON")
    results: dict[str, dict[str, Any]] = {}
    if "sw" in args.sources:
        results["sw"] = acquire_sw()
        print(json.dumps({"stage": "SW", "status": results["sw"]["status"]}, ensure_ascii=False), flush=True)
    if "pboc" in args.sources:
        results["pboc"] = acquire_pboc(args.workers)
        print(json.dumps({"stage": "PBOC", "status": results["pboc"]["status"]}, ensure_ascii=False), flush=True)
    if "dr007-probe" in args.sources:
        results["dr007-probe"] = acquire_dr007_public_probe()
        print(json.dumps({"stage": "DR007_PROBE", "status": results["dr007-probe"]["status"]}, ensure_ascii=False), flush=True)
    if "dr007-import" in args.sources:
        results["dr007-import"] = import_dr007(
            args.dr007_import.resolve(), args.dr007_provenance.resolve()
        )
        print(json.dumps({"stage": "DR007_IMPORT", "status": results["dr007-import"]["status"]}, ensure_ascii=False), flush=True)
    overall = build_overall_manifest(results)
    manifest_path = CURATED_ROOT / "source_remediation_acquisition_manifest.json"
    atomic_write_json(manifest_path, overall)
    print(json.dumps({**overall, "manifest_path": relative(manifest_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
