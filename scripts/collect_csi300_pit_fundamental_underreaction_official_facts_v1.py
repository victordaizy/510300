from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.csi300_pit_fundamental_underreaction_official_facts_v1 import (  # noqa: E402
    PARSER_VERSION,
    REQUIRED_METRICS,
    extract_official_pdf_facts,
)


CONFIG_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")
THREAD_LOCAL = threading.local()
FROZEN_STATUS = (
    "FROZEN_BEFORE_BULK_OFFICIAL_PDF_ACQUISITION_AND_ANY_FUTURE_RETURN_READ"
)
TERMINAL_CHECKPOINT_STATUSES = {"PARSED_COMPLETE", "PARSED_INCOMPLETE"}
QUARTER_DEFINITIONS = {
    1: (3, 31, "Q1"),
    2: (6, 30, "H1"),
    3: (9, 30, "Q3"),
    4: (12, 31, "FY"),
}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
    )


def atomic_write_gzip_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    os.replace(temporary, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def timestamp_iso(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"日期为空：{value}")
    return timestamp.date().isoformat()


def utc_or_local_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def verify_manifest(
    path: Path,
    *,
    expected_status: str,
    expected_content_sha256: str | None,
    verify_own_content_payload: bool = False,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"冻结清单不存在：{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != expected_status:
        raise RuntimeError(
            f"冻结状态不一致：{relative_path(path)}|{manifest.get('status')}"
        )
    if expected_content_sha256 and manifest.get("content_sha256") != expected_content_sha256:
        raise RuntimeError(f"冻结内容哈希不一致：{relative_path(path)}")
    mismatches = [
        item["path"]
        for item in manifest.get("files", [])
        if not project_path(item["path"]).exists()
        or sha256_file(project_path(item["path"])) != item["sha256"]
    ]
    if mismatches:
        raise RuntimeError(f"冻结文件漂移：{mismatches}")
    if verify_own_content_payload:
        payload = {
            "status": manifest["status"],
            "protocol_id": manifest["protocol_id"],
            "version": manifest["version"],
            "parent_freeze": manifest["parent_freeze"],
            "inputs": manifest["inputs"],
            "files": manifest["files"],
        }
        if canonical_sha256(payload) != manifest.get("content_sha256"):
            raise RuntimeError(f"冻结清单自身内容校验失败：{relative_path(path)}")
    return manifest


def load_and_verify_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["extraction"]["parser_version"] != PARSER_VERSION:
        raise RuntimeError("配置中的解析器版本与代码不一致")
    if tuple(config["extraction"]["required_metrics"]) != tuple(REQUIRED_METRICS):
        raise RuntimeError("配置中的必需指标顺序与解析器不一致")

    parent = config["parent_freeze"]
    parent_manifest = verify_manifest(
        project_path(parent["path"]),
        expected_status=parent["expected_status"],
        expected_content_sha256=parent["expected_content_sha256"],
    )
    self_manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    self_manifest = verify_manifest(
        self_manifest_path,
        expected_status=FROZEN_STATUS,
        expected_content_sha256=None,
        verify_own_content_payload=True,
    )

    inventory_path = project_path(config["inputs"]["target_inventory"]["path"])
    events_path = project_path(config["inputs"]["official_events"]["path"])
    for label, path, expected in (
        (
            "无收益目标事件清单",
            inventory_path,
            config["inputs"]["target_inventory"]["expected_sha256"],
        ),
        (
            "官方定期报告事件档案",
            events_path,
            config["inputs"]["official_events"]["expected_sha256"],
        ),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label}不存在：{path}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"{label}哈希漂移：{relative_path(path)}")
    provenance = {
        "parent_manifest_path": relative_path(project_path(parent["path"])),
        "parent_manifest_sha256": sha256_file(project_path(parent["path"])),
        "parent_manifest_content_sha256": parent_manifest["content_sha256"],
        "official_fact_manifest_path": relative_path(self_manifest_path),
        "official_fact_manifest_sha256": sha256_file(self_manifest_path),
        "official_fact_manifest_content_sha256": self_manifest["content_sha256"],
        "target_inventory_path": relative_path(inventory_path),
        "target_inventory_sha256": sha256_file(inventory_path),
        "official_events_path": relative_path(events_path),
        "official_events_sha256": sha256_file(events_path),
    }
    return config, provenance


def standard_quarter_dependencies(
    report_period: Any,
    period_type: str,
    *,
    quarter_count: int = 9,
) -> list[dict[str, Any]]:
    target = pd.Timestamp(report_period).date()
    target_quarter = (target.month - 1) // 3 + 1
    month, day_value, expected_type = QUARTER_DEFINITIONS[target_quarter]
    if (target.month, target.day) != (month, day_value):
        raise ValueError(f"报告期不是标准季度末：{target.isoformat()}")
    if str(period_type).upper() != expected_type:
        raise ValueError(
            f"报告期与类型不一致：{target.isoformat()}|{period_type}|{expected_type}"
        )
    absolute_quarter = target.year * 4 + target_quarter - 1
    rows: list[dict[str, Any]] = []
    for lag in range(quarter_count):
        dependency_index = absolute_quarter - lag
        dependency_year = dependency_index // 4
        dependency_quarter = dependency_index % 4 + 1
        dependency_month, dependency_day, dependency_type = QUARTER_DEFINITIONS[
            dependency_quarter
        ]
        rows.append(
            {
                "dependency_lag_quarters": lag,
                "dependency_role": "TARGET_REPORT" if lag == 0 else f"LAG_{lag}_QUARTERS",
                "required_report_period": date(
                    dependency_year, dependency_month, dependency_day
                ).isoformat(),
                "required_period_type": dependency_type,
            }
        )
    return rows


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label}缺少字段：{missing}")


def build_requirement_ledger(
    inventory: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
    *,
    enforce_expected_target_count: bool = True,
) -> pd.DataFrame:
    target_config = config["inputs"]["target_inventory"]
    target_flag = target_config["target_flag"]
    _require_columns(
        inventory,
        (
            "announcement_id",
            "event_publication_date",
            "official_pdf_url",
            "period_type",
            "report_period",
            "ts_code",
            "industry_l1",
            "industry_l1_code",
            "publication_year",
            target_flag,
        ),
        "无收益目标事件清单",
    )
    targets = inventory.loc[inventory[target_flag].fillna(False).astype(bool)].copy()
    targets["announcement_id"] = targets["announcement_id"].astype("string")
    if targets["announcement_id"].duplicated().any():
        raise ValueError("目标事件公告ID不唯一")
    expected_count = int(target_config["expected_target_event_count"])
    if enforce_expected_target_count and len(targets) != expected_count:
        raise RuntimeError(f"目标事件数漂移：{len(targets)}|期望{expected_count}")

    rows: list[dict[str, Any]] = []
    quarter_count = int(config["dependency_contract"]["quarters_including_target"])
    for target in targets.to_dict("records"):
        target_period = timestamp_iso(target["report_period"])
        target_publication_date = timestamp_iso(target["event_publication_date"])
        for dependency in standard_quarter_dependencies(
            target_period,
            str(target["period_type"]),
            quarter_count=quarter_count,
        ):
            rows.append(
                {
                    "target_announcement_id": str(target["announcement_id"]),
                    "target_ts_code": str(target["ts_code"]),
                    "target_report_period": target_period,
                    "target_period_type": str(target["period_type"]),
                    "target_publication_date": target_publication_date,
                    "target_official_pdf_url": str(target["official_pdf_url"]),
                    "target_publication_year": int(target["publication_year"]),
                    "target_industry_l1": target["industry_l1"],
                    "target_industry_l1_code": str(target["industry_l1_code"]),
                    **dependency,
                }
            )
    requirements = pd.DataFrame(rows)

    event_columns = (
        "announcement_id",
        "ts_code",
        "sec_code",
        "sec_name",
        "announcement_title",
        "period_type",
        "report_period",
        "event_publication_date",
        "official_timestamp_at",
        "official_pdf_url",
        "relative_pdf_url",
        "org_id",
        "adjunct_size_kb",
    )
    _require_columns(events, event_columns, "官方定期报告事件档案")
    lookup = events[list(event_columns)].copy()
    lookup["ts_code"] = lookup["ts_code"].astype("string")
    lookup["required_report_period"] = pd.to_datetime(
        lookup["report_period"], errors="raise"
    ).dt.date.astype(str)
    if lookup.duplicated(["ts_code", "required_report_period"]).any():
        duplicates = int(lookup.duplicated(["ts_code", "required_report_period"]).sum())
        raise ValueError(f"官方事件证券-报告期键不唯一：{duplicates}")
    lookup = lookup.drop(columns="report_period").rename(
        columns={
            "announcement_id": "dependency_announcement_id",
            "sec_code": "dependency_sec_code",
            "sec_name": "dependency_sec_name",
            "announcement_title": "dependency_announcement_title",
            "period_type": "dependency_period_type",
            "event_publication_date": "dependency_publication_date",
            "official_timestamp_at": "dependency_official_timestamp_at",
            "official_pdf_url": "dependency_official_pdf_url",
            "relative_pdf_url": "dependency_relative_pdf_url",
            "org_id": "dependency_org_id",
            "adjunct_size_kb": "dependency_adjunct_size_kb",
        }
    )
    requirements["target_ts_code"] = requirements["target_ts_code"].astype("string")
    merged = requirements.merge(
        lookup,
        left_on=["target_ts_code", "required_report_period"],
        right_on=["ts_code", "required_report_period"],
        how="left",
        validate="many_to_one",
    ).drop(columns="ts_code")

    dependency_dates = pd.to_datetime(merged["dependency_publication_date"], errors="coerce")
    target_dates = pd.to_datetime(merged["target_publication_date"], errors="raise")
    status = pd.Series(
        config["dependency_contract"]["missing_dependency_status"],
        index=merged.index,
        dtype="object",
    )
    found = merged["dependency_announcement_id"].notna()
    status.loc[found & dependency_dates.le(target_dates)] = (
        "PASS_OFFICIAL_DEPENDENCY_AVAILABLE_AT_TARGET"
    )
    status.loc[found & dependency_dates.gt(target_dates)] = (
        "NO_VIEW_REQUIRED_OFFICIAL_REPORT_NOT_PUBLIC_AT_TARGET"
    )
    period_type_mismatch = found & merged["dependency_period_type"].astype(str).ne(
        merged["required_period_type"].astype(str)
    )
    status.loc[period_type_mismatch] = "BLOCKED_OFFICIAL_REPORT_PERIOD_TYPE_MISMATCH"
    accepted_host = config["official_pdf"]["accepted_host"]
    invalid_host = found & merged["dependency_official_pdf_url"].map(
        lambda value: (
            False
            if value is None or pd.isna(value)
            else urlparse(str(value)).hostname != accepted_host
        )
    )
    status.loc[invalid_host] = "BLOCKED_NON_OFFICIAL_PDF_HOST"
    target_id_mismatch = (
        merged["dependency_lag_quarters"].eq(0)
        & found
        & merged["dependency_announcement_id"].astype(str).ne(
            merged["target_announcement_id"].astype(str)
        )
    )
    status.loc[target_id_mismatch] = "BLOCKED_TARGET_EVENT_ARCHIVE_ID_MISMATCH"
    merged["dependency_temporal_status"] = status
    merged["queued_for_official_pdf"] = status.eq(
        "PASS_OFFICIAL_DEPENDENCY_AVAILABLE_AT_TARGET"
    )
    merged["market_price_read"] = False
    merged["future_return_read"] = False
    merged["future_label_created"] = False
    return merged.sort_values(
        [
            "target_publication_date",
            "target_ts_code",
            "target_announcement_id",
            "dependency_lag_quarters",
        ],
        kind="stable",
    ).reset_index(drop=True)


def build_document_queue(ledger: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    available = ledger.loc[ledger["queued_for_official_pdf"]].copy()
    metadata_columns = [
        "dependency_announcement_id",
        "target_ts_code",
        "dependency_sec_code",
        "dependency_sec_name",
        "dependency_announcement_title",
        "required_report_period",
        "required_period_type",
        "dependency_publication_date",
        "dependency_official_timestamp_at",
        "dependency_official_pdf_url",
        "dependency_relative_pdf_url",
        "dependency_org_id",
        "dependency_adjunct_size_kb",
    ]
    if available.empty:
        return pd.DataFrame(
            columns=[
                "announcement_id",
                "ts_code",
                "sec_code",
                "sec_name",
                "announcement_title",
                "report_period",
                "period_type",
                "event_publication_date",
                "official_timestamp_at",
                "official_pdf_url",
                "relative_pdf_url",
                "org_id",
                "adjunct_size_kb",
                "dependent_target_event_count",
                "checkpoint_path",
            ]
        )
    for column in metadata_columns:
        inconsistency = available.groupby("dependency_announcement_id", dropna=False)[
            column
        ].nunique(dropna=False)
        if int((inconsistency > 1).sum()):
            raise ValueError(f"同一依赖公告元数据不一致：{column}")
    counts = (
        available.groupby("dependency_announcement_id")["target_announcement_id"]
        .nunique()
        .rename("dependent_target_event_count")
    )
    queue = (
        available[metadata_columns]
        .drop_duplicates("dependency_announcement_id")
        .set_index("dependency_announcement_id")
        .join(counts)
        .reset_index()
        .rename(
            columns={
                "dependency_announcement_id": "announcement_id",
                "target_ts_code": "ts_code",
                "dependency_sec_code": "sec_code",
                "dependency_sec_name": "sec_name",
                "dependency_announcement_title": "announcement_title",
                "required_report_period": "report_period",
                "required_period_type": "period_type",
                "dependency_publication_date": "event_publication_date",
                "dependency_official_timestamp_at": "official_timestamp_at",
                "dependency_official_pdf_url": "official_pdf_url",
                "dependency_relative_pdf_url": "relative_pdf_url",
                "dependency_org_id": "org_id",
                "dependency_adjunct_size_kb": "adjunct_size_kb",
            }
        )
    )
    queue["announcement_id"] = queue["announcement_id"].astype(str)
    queue["checkpoint_path"] = queue.apply(
        lambda row: relative_path(checkpoint_path(config, row.to_dict())), axis=1
    )
    if queue["announcement_id"].duplicated().any():
        raise ValueError("官方PDF队列公告ID不唯一")
    return queue.sort_values(
        ["report_period", "ts_code", "announcement_id"], kind="stable"
    ).reset_index(drop=True)


def checkpoint_path(config: dict[str, Any], row: dict[str, Any]) -> Path:
    report_year = timestamp_iso(row["report_period"])[:4]
    return (
        project_path(config["artifacts"]["checkpoint_root"])
        / report_year
        / f"{str(row['announcement_id'])}.json.gz"
    )


def read_checkpoint(
    path: Path,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:  # noqa: BLE001 - 损坏断点必须重新获取，不能冒充完成
        return None
    if str(payload.get("announcement_id")) != str(row["announcement_id"]):
        return None
    canonical_url = payload.get("canonical_official_pdf_url") or payload.get(
        "official_pdf_url"
    )
    if str(canonical_url) != str(row["official_pdf_url"]):
        return None
    if str(payload.get("report_period")) != timestamp_iso(row["report_period"]):
        return None
    if str(payload.get("period_type")) != str(row["period_type"]):
        return None
    if payload.get("parser_version") != PARSER_VERSION:
        return None
    status = payload.get("checkpoint_status")
    if status in TERMINAL_CHECKPOINT_STATUSES:
        if len(str(payload.get("official_pdf_sha256") or "")) != 64:
            return None
        if not isinstance(payload.get("metrics"), list):
            return None
    payload["checkpoint_file_sha256"] = sha256_file(path)
    return payload


def download_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "official_fact_session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/140 Safari/537.36"
                ),
                "Referer": "https://www.cninfo.com.cn/",
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            }
        )
        THREAD_LOCAL.official_fact_session = session
    return session


def official_query_session(config: dict[str, Any]) -> requests.Session:
    session = getattr(THREAD_LOCAL, "official_query_session", None)
    if session is None:
        search_page_url = config["official_pdf"]["search_page_url"]
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
        response = session.get(search_page_url, timeout=(30, 60))
        response.raise_for_status()
        THREAD_LOCAL.official_query_session = session
    return session


def title_matches_period(title: Any, report_period: Any, period_type: str) -> bool:
    normalized = str(title or "").replace(" ", "")
    report_year = timestamp_iso(report_period)[:4]
    if f"{report_year}年" not in normalized:
        return False
    normalized_type = str(period_type).upper()
    if normalized_type == "Q1":
        return "第一季度报告" in normalized or "一季度报告" in normalized
    if normalized_type == "H1":
        return any(marker in normalized for marker in ("半年度报告", "中期报告", "半年报"))
    if normalized_type == "Q3":
        return "第三季度报告" in normalized or "三季度报告" in normalized
    if normalized_type == "FY":
        return (
            any(marker in normalized for marker in ("年度报告", "年报"))
            and "半年度报告" not in normalized
            and "半年报" not in normalized
        )
    return False


def query_same_event_official_pdf_candidates(
    row: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], str | None]:
    pdf_config = config["official_pdf"]
    if not pdf_config.get("same_event_official_full_pdf_sibling_fallback"):
        return [], None
    if not row.get("sec_code") or not row.get("org_id"):
        return [], "缺少证券代码或巨潮机构ID，无法查询同事件PDF"
    category_by_period = {
        "FY": "category_ndbg_szsh",
        "H1": "category_bndbg_szsh",
        "Q1": "category_yjdbg_szsh",
        "Q3": "category_sjdbg_szsh",
    }
    category = category_by_period.get(str(row["period_type"]).upper())
    if category is None:
        return [], f"未知报告类型：{row['period_type']}"
    publication_date = timestamp_iso(row["event_publication_date"])
    body = {
        "pageNum": 1,
        "pageSize": int(pdf_config["sibling_query_page_size"]),
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{row['sec_code']},{row['org_id']}",
        "searchkey": "",
        "secid": "",
        "category": category + ";",
        "trade": "",
        "seDate": f"{publication_date}~{publication_date}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    final_error: Exception | None = None
    payload: dict[str, Any] | None = None
    for attempt in range(1, 4):
        try:
            response = official_query_session(config).post(
                pdf_config["query_url"],
                data=body,
                timeout=(30, int(pdf_config["timeout_seconds"])),
            )
            response.raise_for_status()
            payload = response.json()
            break
        except Exception as error:  # noqa: BLE001 - 官方查询失败保留为来源缺口
            final_error = error
            if attempt < 3:
                time.sleep(float(attempt))
    if payload is None:
        return [], f"{type(final_error).__name__}: {final_error}"[:2000]
    if bool(payload.get("hasMore")):
        return [], "同证券同日报告候选超过冻结单页上限"

    excluded_tokens = tuple(pdf_config["sibling_excluded_title_tokens"])
    accepted_host = pdf_config["accepted_host"]
    primary_id = str(row["announcement_id"])
    primary_url = str(row["official_pdf_url"])
    candidates: list[dict[str, Any]] = []
    for record in payload.get("announcements") or []:
        title = str(record.get("announcementTitle") or "")
        if any(token in title for token in excluded_tokens):
            continue
        if str(record.get("secCode") or "").zfill(6) != str(row["sec_code"]).zfill(6):
            continue
        if not title_matches_period(title, row["report_period"], row["period_type"]):
            continue
        relative_url = str(record.get("adjunctUrl") or "")
        official_url = urljoin(pdf_config["pdf_base_url"], relative_url)
        if urlparse(official_url).hostname != accepted_host:
            continue
        announcement_id = str(record.get("announcementId") or "")
        if not announcement_id or announcement_id == primary_id or official_url == primary_url:
            continue
        candidates.append(
            {
                **row,
                "official_pdf_announcement_id": announcement_id,
                "official_pdf_url": official_url,
                "official_pdf_title": title,
                "official_pdf_announcement_time_ms": record.get("announcementTime"),
                "official_pdf_adjunct_size_kb": record.get("adjunctSize"),
                "source_route": "SAME_EVENT_OFFICIAL_FULL_PDF_SIBLING",
            }
        )
    candidates.sort(
        key=lambda item: (
            int(item.get("official_pdf_announcement_time_ms") or 0),
            str(item["official_pdf_announcement_id"]),
        )
    )
    maximum = int(pdf_config["sibling_maximum_candidates"])
    return candidates[:maximum], None


def download_official_pdf(
    row: dict[str, Any], config: dict[str, Any]
) -> tuple[bytes | None, dict[str, Any]]:
    pdf_config = config["official_pdf"]
    accepted_host = pdf_config["accepted_host"]
    url = str(row["official_pdf_url"])
    if urlparse(url).hostname != accepted_host:
        return None, {
            "attempts": 0,
            "http_status": None,
            "content_type": None,
            "final_url": None,
            "download_error": "PDF地址不是冻结的官方主机",
            "download_error_kind": "INVALID_OFFICIAL_HOST",
        }

    maximum_attempts = int(pdf_config["maximum_attempts_per_run"])
    backoffs = list(pdf_config.get("retry_backoff_seconds", []))
    maximum_bytes = int(pdf_config["maximum_pdf_bytes"])
    chunk_bytes = int(pdf_config["download_chunk_bytes"])
    timeout = int(pdf_config["timeout_seconds"])
    last_meta: dict[str, Any] = {}
    for attempt in range(1, maximum_attempts + 1):
        response: requests.Response | None = None
        try:
            response = download_session().get(
                url,
                stream=True,
                timeout=(30, timeout),
                allow_redirects=True,
            )
            response.raise_for_status()
            if urlparse(response.url).hostname != accepted_host:
                raise RuntimeError("官方PDF请求被重定向到非冻结主机")
            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > maximum_bytes:
                raise ValueError(f"PDF声明大小超过上限：{declared_size}")
            content = bytearray()
            for chunk in response.iter_content(chunk_size=chunk_bytes):
                if not chunk:
                    continue
                content.extend(chunk)
                if len(content) > maximum_bytes:
                    raise ValueError(f"PDF实际大小超过上限：{len(content)}")
            payload = bytes(content)
            required_header = str(pdf_config["required_header"]).encode("ascii")
            if not payload.startswith(required_header):
                raise ValueError("响应没有冻结要求的PDF文件头")
            return payload, {
                "attempts": attempt,
                "http_status": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "final_url": response.url,
                "download_error": None,
                "download_error_kind": None,
            }
        except Exception as error:  # noqa: BLE001 - 网络失败必须进入可重试状态
            last_meta = {
                "attempts": attempt,
                "http_status": response.status_code if response is not None else None,
                "content_type": (
                    response.headers.get("Content-Type") if response is not None else None
                ),
                "final_url": response.url if response is not None else None,
                "download_error": f"{type(error).__name__}: {error}"[:2000],
                "download_error_kind": type(error).__name__,
            }
            if attempt < maximum_attempts:
                delay = backoffs[min(attempt - 1, len(backoffs) - 1)] if backoffs else 0
                if delay:
                    time.sleep(float(delay))
        finally:
            if response is not None:
                response.close()
    return None, last_meta


def base_checkpoint(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": "1.0.0",
        "announcement_id": str(row["announcement_id"]),
        "ts_code": str(row["ts_code"]),
        "sec_code": str(row.get("sec_code") or ""),
        "sec_name": str(row.get("sec_name") or ""),
        "announcement_title": str(row.get("announcement_title") or ""),
        "report_period": timestamp_iso(row["report_period"]),
        "period_type": str(row["period_type"]),
        "event_publication_date": timestamp_iso(row["event_publication_date"]),
        "official_timestamp_at": utc_or_local_iso(row.get("official_timestamp_at")),
        "canonical_official_pdf_url": str(row["official_pdf_url"]),
        "official_pdf_url": str(row["official_pdf_url"]),
        "official_pdf_announcement_id": str(row["announcement_id"]),
        "source_route": "PRIMARY_EVENT_ARCHIVE_PDF",
        "parser_version": PARSER_VERSION,
        "source_pdf_retained": False,
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "portfolio_return_calculated": False,
    }


def process_document(
    row: dict[str, Any],
    config: dict[str, Any],
    *,
    reparse_incomplete: bool,
) -> dict[str, Any]:
    path = checkpoint_path(config, row)
    existing = read_checkpoint(path, row)
    if existing is not None and existing.get("checkpoint_status") == "PARSED_COMPLETE":
        return {
            "announcement_id": str(row["announcement_id"]),
            "run_status": "REUSED_PARSED_COMPLETE",
        }
    if (
        existing is not None
        and existing.get("checkpoint_status") == "PARSED_INCOMPLETE"
        and not reparse_incomplete
    ):
        return {
            "announcement_id": str(row["announcement_id"]),
            "run_status": "REUSED_PARSED_INCOMPLETE",
        }

    now = datetime.now(TIMEZONE).isoformat()
    primary_candidate = {
        **row,
        "official_pdf_announcement_id": str(row["announcement_id"]),
        "official_pdf_title": str(row.get("announcement_title") or ""),
        "official_pdf_announcement_time_ms": None,
        "source_route": "PRIMARY_EVENT_ARCHIVE_PDF",
    }
    candidate_results: list[dict[str, Any]] = []

    def attempt_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        content, download_meta = download_official_pdf(candidate, config)
        result: dict[str, Any] = {
            "official_pdf_announcement_id": str(
                candidate["official_pdf_announcement_id"]
            ),
            "official_pdf_url": str(candidate["official_pdf_url"]),
            "official_pdf_title": str(candidate.get("official_pdf_title") or ""),
            "source_route": str(candidate["source_route"]),
            "download_meta": download_meta,
            "parsed": False,
            "extraction": None,
            "parse_error": None,
        }
        if content is None:
            return result
        try:
            result["extraction"] = extract_official_pdf_facts(
                content,
                period_type=str(row["period_type"]),
            )
            result["parsed"] = True
        except Exception as error:  # noqa: BLE001 - 解析异常形成终局NO_VIEW候选
            result["parsed"] = True
            result["parse_error"] = f"{type(error).__name__}: {error}"[:2000]
            result["extraction"] = {
                "parser_version": PARSER_VERSION,
                "document_complete": False,
                "document_status": "NO_VIEW_OFFICIAL_PDF_PARSE_FAILED",
                "metrics": [],
                "missing_metrics": list(REQUIRED_METRICS),
                "sections": {},
                "official_pdf_sha256": hashlib.sha256(content).hexdigest(),
                "official_pdf_size_bytes": len(content),
                "pdf_page_count": None,
                "text_character_count": 0,
                "page_errors": [],
            }
        finally:
            content = b""
        return result

    primary_result = attempt_candidate(primary_candidate)
    candidate_results.append(primary_result)
    sibling_query_error: str | None = None
    if not (
        primary_result["parsed"]
        and primary_result["extraction"]["document_complete"]
    ):
        sibling_candidates, sibling_query_error = query_same_event_official_pdf_candidates(
            row, config
        )
        for sibling in sibling_candidates:
            sibling_result = attempt_candidate(sibling)
            candidate_results.append(sibling_result)
            if (
                sibling_result["parsed"]
                and sibling_result["extraction"]["document_complete"]
            ):
                break

    complete_results = [
        result
        for result in candidate_results
        if result["parsed"] and result["extraction"]["document_complete"]
    ]
    parsed_results = [result for result in candidate_results if result["parsed"]]
    if complete_results:
        chosen = complete_results[0]
    elif parsed_results:
        chosen = max(
            parsed_results,
            key=lambda result: len(result["extraction"].get("metrics") or []),
        )
    else:
        chosen = None

    candidate_attempts = [
        {
            "official_pdf_announcement_id": result["official_pdf_announcement_id"],
            "official_pdf_url": result["official_pdf_url"],
            "source_route": result["source_route"],
            "parsed": result["parsed"],
            "document_complete": bool(
                result["extraction"] and result["extraction"]["document_complete"]
            ),
            "missing_metrics": (
                result["extraction"].get("missing_metrics")
                if result["extraction"]
                else list(REQUIRED_METRICS)
            ),
            "official_pdf_sha256": (
                result["extraction"].get("official_pdf_sha256")
                if result["extraction"]
                else None
            ),
            "attempts": result["download_meta"].get("attempts"),
            "download_error": result["download_meta"].get("download_error"),
            "parse_error": result["parse_error"],
        }
        for result in candidate_results
    ]
    if chosen is None:
        last_download = candidate_results[-1]["download_meta"]
        payload = {
            **base_checkpoint(row),
            "checkpoint_status": "DOWNLOAD_FAILED",
            "document_terminal": False,
            "document_complete": False,
            "document_status": "PARTIAL_OFFICIAL_PDF_DOWNLOAD_FAILED_RETRY_NEXT_RUN",
            "metrics": [],
            "missing_metrics": list(REQUIRED_METRICS),
            "official_pdf_sha256": None,
            "official_pdf_size_bytes": None,
            "pdf_page_count": None,
            "text_character_count": 0,
            "page_errors": [],
            "candidate_attempts": candidate_attempts,
            "sibling_query_error": sibling_query_error,
            "retrieved_at": now,
            **last_download,
        }
        atomic_write_gzip_json(path, payload)
        return {
            "announcement_id": str(row["announcement_id"]),
            "run_status": "DOWNLOAD_FAILED",
        }

    extraction = chosen["extraction"]
    checkpoint_status = (
        "PARSED_COMPLETE" if extraction["document_complete"] else "PARSED_INCOMPLETE"
    )
    payload = {
        **base_checkpoint(row),
        **extraction,
        **chosen["download_meta"],
        "official_pdf_url": chosen["official_pdf_url"],
        "official_pdf_announcement_id": chosen["official_pdf_announcement_id"],
        "official_pdf_title": chosen["official_pdf_title"],
        "source_route": chosen["source_route"],
        "checkpoint_status": checkpoint_status,
        "document_terminal": True,
        "parse_error": chosen["parse_error"],
        "candidate_attempts": candidate_attempts,
        "sibling_query_error": sibling_query_error,
        "retrieved_at": now,
    }
    atomic_write_gzip_json(path, payload)
    return {
        "announcement_id": str(row["announcement_id"]),
        "run_status": checkpoint_status,
    }


def queue_with_checkpoint_state(
    queue: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in queue.to_dict("records"):
        path = checkpoint_path(config, row)
        checkpoint = read_checkpoint(path, row)
        status = checkpoint.get("checkpoint_status") if checkpoint else "PENDING"
        rows.append(
            {
                **row,
                "checkpoint_status": status,
                "document_terminal": status in TERMINAL_CHECKPOINT_STATUSES,
                "document_complete": status == "PARSED_COMPLETE",
                "document_status": checkpoint.get("document_status") if checkpoint else None,
                "missing_metric_count": (
                    len(checkpoint.get("missing_metrics") or [])
                    if checkpoint
                    else len(REQUIRED_METRICS)
                ),
                "missing_metrics_json": json.dumps(
                    checkpoint.get("missing_metrics") if checkpoint else list(REQUIRED_METRICS),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "official_pdf_sha256": (
                    checkpoint.get("official_pdf_sha256") if checkpoint else None
                ),
                "official_pdf_announcement_id": (
                    checkpoint.get("official_pdf_announcement_id") if checkpoint else None
                ),
                "resolved_official_pdf_url": (
                    checkpoint.get("official_pdf_url") if checkpoint else None
                ),
                "source_route": checkpoint.get("source_route") if checkpoint else None,
                "official_pdf_size_bytes": (
                    checkpoint.get("official_pdf_size_bytes") if checkpoint else None
                ),
                "pdf_page_count": checkpoint.get("pdf_page_count") if checkpoint else None,
                "text_character_count": (
                    checkpoint.get("text_character_count") if checkpoint else None
                ),
                "attempts_last_run": checkpoint.get("attempts") if checkpoint else 0,
                "download_error": checkpoint.get("download_error") if checkpoint else None,
                "parse_error": checkpoint.get("parse_error") if checkpoint else None,
                "checkpoint_sha256": (
                    checkpoint.get("checkpoint_file_sha256") if checkpoint else None
                ),
            }
        )
    return pd.DataFrame(rows)


FACT_COLUMNS = [
    "announcement_id",
    "ts_code",
    "report_period",
    "period_type",
    "event_publication_date",
    "official_timestamp_at",
    "official_pdf_url",
    "official_pdf_announcement_id",
    "official_pdf_sha256",
    "official_pdf_size_bytes",
    "pdf_page_count",
    "parser_version",
    "metric_id",
    "metric_value_cny",
    "statement_scope",
    "value_period_scope",
    "source_page",
    "source_locator",
    "source_label",
    "source_raw_value",
    "source_unit",
    "source_unit_multiplier",
    "source_method",
    "verification_status",
    "market_price_read",
    "future_return_read",
]


def build_fact_archive(queue: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for document in queue.to_dict("records"):
        checkpoint = read_checkpoint(checkpoint_path(config, document), document)
        if checkpoint is None or checkpoint.get("checkpoint_status") not in TERMINAL_CHECKPOINT_STATUSES:
            continue
        for metric in checkpoint.get("metrics") or []:
            rows.append(
                {
                    "announcement_id": str(document["announcement_id"]),
                    "ts_code": str(document["ts_code"]),
                    "report_period": timestamp_iso(document["report_period"]),
                    "period_type": str(document["period_type"]),
                    "event_publication_date": timestamp_iso(
                        document["event_publication_date"]
                    ),
                    "official_timestamp_at": utc_or_local_iso(
                        document.get("official_timestamp_at")
                    ),
                    "official_pdf_url": str(checkpoint["official_pdf_url"]),
                    "official_pdf_announcement_id": str(
                        checkpoint["official_pdf_announcement_id"]
                    ),
                    "official_pdf_sha256": checkpoint["official_pdf_sha256"],
                    "official_pdf_size_bytes": checkpoint.get(
                        "official_pdf_size_bytes"
                    ),
                    "pdf_page_count": checkpoint.get("pdf_page_count"),
                    "parser_version": checkpoint["parser_version"],
                    **metric,
                    "market_price_read": False,
                    "future_return_read": False,
                }
            )
    facts = pd.DataFrame(rows, columns=FACT_COLUMNS)
    if not facts.empty:
        if facts.duplicated(["announcement_id", "metric_id"]).any():
            raise ValueError("官方财务事实公告-指标键不唯一")
        facts = facts.sort_values(
            ["report_period", "ts_code", "announcement_id", "metric_id"],
            kind="stable",
        ).reset_index(drop=True)
    return facts


def build_event_dependency_ledger(
    requirement_ledger: pd.DataFrame,
    queue_state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_columns = [
        "announcement_id",
        "checkpoint_status",
        "document_terminal",
        "document_complete",
        "missing_metric_count",
        "missing_metrics_json",
    ]
    state = queue_state[state_columns].rename(
        columns={"announcement_id": "dependency_announcement_id"}
    )
    runtime = requirement_ledger.merge(
        state,
        on="dependency_announcement_id",
        how="left",
        validate="many_to_one",
    )
    runtime["document_terminal"] = runtime["document_terminal"].fillna(False).astype(bool)
    runtime["document_complete"] = runtime["document_complete"].fillna(False).astype(bool)
    runtime["checkpoint_status"] = runtime["checkpoint_status"].fillna("PENDING")
    runtime["missing_metric_count"] = runtime["missing_metric_count"].fillna(
        len(REQUIRED_METRICS)
    ).astype(int)
    runtime["missing_metrics_json"] = runtime["missing_metrics_json"].fillna(
        json.dumps(list(REQUIRED_METRICS), ensure_ascii=False, separators=(",", ":"))
    )
    metadata_available = runtime["queued_for_official_pdf"].astype(bool)
    runtime["requirement_status"] = runtime["dependency_temporal_status"]
    runtime.loc[
        metadata_available & ~runtime["document_terminal"], "requirement_status"
    ] = "PARTIAL_OFFICIAL_PDF_NOT_TERMINALLY_ACQUIRED"
    runtime.loc[
        metadata_available
        & runtime["document_terminal"]
        & ~runtime["document_complete"],
        "requirement_status",
    ] = "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE"
    runtime.loc[
        metadata_available & runtime["document_complete"], "requirement_status"
    ] = "PASS_COMPLETE_OFFICIAL_FINANCIAL_FACT_DEPENDENCY"

    dependency_rows: list[dict[str, Any]] = []
    for announcement_id, group in runtime.groupby("target_announcement_id", sort=False):
        first = group.iloc[0]
        required_document_count = int(len(group))
        complete_document_count = int(group["document_complete"].sum())
        missing_document_count = int(
            ((~group["queued_for_official_pdf"]) | (~group["document_terminal"])).sum()
        )
        incomplete_document_count = int(
            (group["document_terminal"] & ~group["document_complete"]).sum()
        )
        missing_metric_count = int(group["missing_metric_count"].sum())
        ready = bool(
            complete_document_count == required_document_count
            and missing_document_count == 0
            and incomplete_document_count == 0
            and missing_metric_count == 0
        )
        dependency_rows.append(
            {
                "announcement_id": str(announcement_id),
                "ts_code": str(first["target_ts_code"]),
                "report_period": str(first["target_report_period"]),
                "period_type": str(first["target_period_type"]),
                "event_publication_date": str(first["target_publication_date"]),
                "publication_year": int(first["target_publication_year"]),
                "industry_l1": first["target_industry_l1"],
                "industry_l1_code": str(first["target_industry_l1_code"]),
                "target_event_ready": ready,
                "required_document_count": required_document_count,
                "complete_document_count": complete_document_count,
                "missing_metric_count": missing_metric_count,
                "missing_document_count": missing_document_count,
                "incomplete_document_count": incomplete_document_count,
                "dependency_status": (
                    "PASS_COMPLETE_OFFICIAL_FINANCIAL_FACT_DEPENDENCIES"
                    if ready
                    else "NO_VIEW_OFFICIAL_FINANCIAL_FACT_DEPENDENCIES_INCOMPLETE"
                ),
                "market_price_read": False,
                "future_return_read": False,
            }
        )
    dependency = pd.DataFrame(dependency_rows).sort_values(
        ["event_publication_date", "ts_code", "announcement_id"], kind="stable"
    ).reset_index(drop=True)
    return runtime, dependency


def coverage_metrics(
    dependency: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    if dependency.empty:
        return {
            "overall_complete_event_ratio": 0.0,
            "publication_year_coverage": [],
            "minimum_publication_year_complete_event_ratio": 0.0,
            "industry_coverage": [],
            "minimum_eligible_industry_complete_event_ratio": 0.0,
            "coverage_gate_passed": False,
        }
    by_year = (
        dependency.groupby("publication_year", dropna=False)["target_event_ready"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "size": "target_event_count",
                "sum": "complete_event_count",
                "mean": "complete_event_ratio",
            }
        )
    )
    by_industry = (
        dependency.groupby(["industry_l1_code", "industry_l1"], dropna=False)[
            "target_event_ready"
        ]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "size": "target_event_count",
                "sum": "complete_event_count",
                "mean": "complete_event_ratio",
            }
        )
    )
    minimum_industry_size = 50
    eligible_industries = by_industry.loc[
        by_industry["target_event_count"] >= minimum_industry_size
    ]
    overall = float(dependency["target_event_ready"].mean())
    year_minimum = float(by_year["complete_event_ratio"].min())
    industry_minimum = (
        float(eligible_industries["complete_event_ratio"].min())
        if len(eligible_industries)
        else 0.0
    )
    admission = config["admission"]
    passed = bool(
        overall >= float(admission["required_complete_event_ratio"])
        and year_minimum
        >= float(admission["required_complete_event_ratio_each_publication_year"])
        and industry_minimum
        >= float(
            admission[
                "required_complete_event_ratio_each_industry_with_at_least_50_targets"
            ]
        )
    )
    return {
        "overall_complete_event_ratio": overall,
        "publication_year_coverage": json.loads(
            by_year.to_json(orient="records", force_ascii=False)
        ),
        "minimum_publication_year_complete_event_ratio": year_minimum,
        "industry_coverage": json.loads(
            by_industry.to_json(orient="records", force_ascii=False)
        ),
        "minimum_eligible_industry_complete_event_ratio": industry_minimum,
        "coverage_gate_passed": passed,
    }


def render_report(receipt: dict[str, Any]) -> str:
    coverage = receipt["coverage"]
    counts = receipt["counts"]
    lines = [
        "# 沪深300基本面反应不足官方财务事实 V1 采集报告",
        "",
        f"- 状态：`{receipt['status']}`",
        f"- 目标事件：{counts['target_event_count']:,}",
        f"- 固定依赖行：{counts['requirement_count']:,}",
        f"- 去重官方PDF：{counts['queued_document_count']:,}",
        f"- 已终局解析PDF：{counts['terminal_document_count']:,}",
        f"- 九项完整PDF：{counts['complete_document_count']:,}",
        f"- 网络失败或尚未终局PDF：{counts['nonterminal_document_count']:,}",
        f"- 官方事实行：{counts['fact_row_count']:,}",
        f"- 目标事件完整率：{coverage['overall_complete_event_ratio']:.6f}",
        f"- 最低年度完整率：{coverage['minimum_publication_year_complete_event_ratio']:.6f}",
        f"- 最低合格行业完整率：{coverage['minimum_eligible_industry_complete_event_ratio']:.6f}",
        "- 市场价格读取：否",
        "- 未来收益读取：否",
        "- 完整PDF长期保留：否",
        "",
        "## 状态解释",
        "",
        (
            "只有去重PDF队列全部形成终局解析检查点，且总体、逐年、逐行业覆盖门同时通过，"
            "才允许主预检重新判断是否开放固定的60/120日相对机制评价。事实缺失、网络失败"
            "和解析不完整均保留原状态，不填零、不以后来报告回补。"
        ),
        "",
        f"收益评价：`{receipt['return_evaluation']}`",
        "",
    ]
    return "\n".join(lines)


def select_work_rows(
    queue: pd.DataFrame,
    config: dict[str, Any],
    *,
    reparse_incomplete: bool,
    limit: int | None,
) -> tuple[list[dict[str, Any]], int]:
    pending: list[dict[str, Any]] = []
    reusable = 0
    for row in queue.to_dict("records"):
        checkpoint = read_checkpoint(checkpoint_path(config, row), row)
        if checkpoint is None or checkpoint.get("checkpoint_status") == "DOWNLOAD_FAILED":
            pending.append(row)
        elif checkpoint.get("checkpoint_status") == "PARSED_INCOMPLETE" and reparse_incomplete:
            pending.append(row)
        else:
            reusable += 1
    if limit is not None and limit < len(pending):
        if limit <= 0:
            pending = []
        elif limit == 1:
            pending = [pending[len(pending) // 2]]
        else:
            positions = [
                round(index * (len(pending) - 1) / (limit - 1)) for index in range(limit)
            ]
            pending = [pending[position] for position in positions]
    return pending, reusable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集沪深300基本面反应不足研究所需的官方PDF财务事实"
    )
    parser.add_argument("--workers", type=int, default=None, help="并发下载解析数")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="本轮最多处理的待获取PDF数；用于冻结后的代表性小样本运行",
    )
    parser.add_argument(
        "--reparse-incomplete",
        action="store_true",
        help="重新下载并解析已有事实不完整检查点",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config, provenance = load_and_verify_config()
        inventory_path = project_path(config["inputs"]["target_inventory"]["path"])
        events_path = project_path(config["inputs"]["official_events"]["path"])
        inventory = pd.read_parquet(inventory_path)
        events = pd.read_parquet(events_path)
        requirement = build_requirement_ledger(inventory, events, config)
        queue = build_document_queue(requirement, config)
        artifacts = config["artifacts"]
        requirement_path = project_path(artifacts["requirement_ledger"])
        queue_path = project_path(artifacts["document_queue"])
        atomic_write_parquet(requirement_path, requirement)

        work_rows, reusable_count = select_work_rows(
            queue,
            config,
            reparse_incomplete=args.reparse_incomplete,
            limit=args.limit,
        )
        workers = args.workers or int(config["official_pdf"]["workers_default"])
        if workers < 1:
            raise ValueError("并发数必须至少为1")
        print(
            f"官方PDF去重队列：{len(queue):,}；复用终局断点：{reusable_count:,}；"
            f"本轮待处理：{len(work_rows):,}；并发：{workers}",
            flush=True,
        )
        run_counts: dict[str, int] = {}
        if work_rows:
            progress_interval = max(1, min(100, len(work_rows) // 20 or 1))
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_document,
                        row,
                        config,
                        reparse_incomplete=args.reparse_incomplete,
                    ): str(row["announcement_id"])
                    for row in work_rows
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    announcement_id = futures[future]
                    try:
                        result = future.result()
                        status = result["run_status"]
                    except Exception as error:  # noqa: BLE001 - 单文档失败不能终止整批
                        status = "UNEXPECTED_WORKER_FAILURE"
                        print(
                            f"公告{announcement_id}处理异常：{type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                    run_counts[status] = run_counts.get(status, 0) + 1
                    if completed % progress_interval == 0 or completed == len(work_rows):
                        print(
                            f"本轮进度：{completed:,}/{len(work_rows):,}；"
                            f"状态计数：{json.dumps(run_counts, ensure_ascii=False, sort_keys=True)}",
                            flush=True,
                        )

        queue_state = queue_with_checkpoint_state(queue, config)
        facts = build_fact_archive(queue, config)
        runtime_requirement, dependency = build_event_dependency_ledger(
            requirement, queue_state
        )
        atomic_write_parquet(requirement_path, runtime_requirement)
        atomic_write_parquet(queue_path, queue_state)
        facts_path = project_path(artifacts["facts"])
        dependency_path = project_path(artifacts["dependency_ledger"])
        atomic_write_parquet(facts_path, facts)
        atomic_write_parquet(dependency_path, dependency)

        coverage = coverage_metrics(dependency, config)
        terminal_count = int(queue_state["document_terminal"].sum()) if len(queue_state) else 0
        complete_count = int(queue_state["document_complete"].sum()) if len(queue_state) else 0
        queue_complete = bool(len(queue_state) and terminal_count == len(queue_state))
        if queue_complete and coverage["coverage_gate_passed"]:
            status = config["admission"]["pass_status"]
        elif queue_complete:
            status = config["admission"]["coverage_fail_status"]
        else:
            status = config["admission"]["partial_status"]

        receipt = {
            "protocol_id": config["protocol"]["protocol_id"],
            "protocol_version": config["protocol"]["version"],
            "status": status,
            "created_at": datetime.now(TIMEZONE).isoformat(),
            "parser_version": PARSER_VERSION,
            "provenance": provenance,
            "run": {
                "workers": workers,
                "limit": args.limit,
                "reparse_incomplete": bool(args.reparse_incomplete),
                "selected_document_count": len(work_rows),
                "reusable_document_count_before_run": reusable_count,
                "run_status_counts": run_counts,
            },
            "counts": {
                "target_event_count": int(len(dependency)),
                "requirement_count": int(len(runtime_requirement)),
                "temporally_available_requirement_count": int(
                    runtime_requirement["queued_for_official_pdf"].sum()
                ),
                "queued_document_count": int(len(queue_state)),
                "terminal_document_count": terminal_count,
                "complete_document_count": complete_count,
                "incomplete_terminal_document_count": int(
                    (
                        queue_state["document_terminal"]
                        & ~queue_state["document_complete"]
                    ).sum()
                )
                if len(queue_state)
                else 0,
                "nonterminal_document_count": int(len(queue_state) - terminal_count),
                "download_failed_document_count": int(
                    queue_state["checkpoint_status"].eq("DOWNLOAD_FAILED").sum()
                )
                if len(queue_state)
                else 0,
                "fact_row_count": int(len(facts)),
                "ready_target_event_count": int(dependency["target_event_ready"].sum()),
            },
            "checkpoint_status_counts": (
                queue_state["checkpoint_status"].value_counts(dropna=False).to_dict()
                if len(queue_state)
                else {}
            ),
            "dependency_status_counts": dependency["dependency_status"].value_counts(
                dropna=False
            ).to_dict(),
            "requirement_status_counts": runtime_requirement[
                "requirement_status"
            ].value_counts(dropna=False).to_dict(),
            "queue_acquisition_complete": queue_complete,
            "coverage": coverage,
            "artifacts": {
                "requirement_ledger_path": relative_path(requirement_path),
                "requirement_ledger_sha256": sha256_file(requirement_path),
                "document_queue_path": relative_path(queue_path),
                "document_queue_sha256": sha256_file(queue_path),
                "facts_path": relative_path(facts_path),
                "facts_sha256": sha256_file(facts_path),
                "dependency_ledger_path": relative_path(dependency_path),
                "dependency_ledger_sha256": sha256_file(dependency_path),
            },
            "full_pdf_retained": False,
            "market_price_read": False,
            "future_return_read": False,
            "future_label_created": False,
            "signal_score_calculated": False,
            "portfolio_return_calculated": False,
            "positions_generated": False,
            "orders_generated": False,
            "broker_connection_performed": False,
            "trading_authorization": False,
            "return_evaluation": "NOT_ALLOWED",
            "next_allowed_step": (
                "RERUN_FROZEN_MAIN_PREFLIGHT"
                if status == config["admission"]["pass_status"]
                else "CONTINUE_OR_CLOSE_OFFICIAL_FACT_ACQUISITION_WITH_EXPLICIT_NO_VIEW"
            ),
        }
        report_path = project_path(artifacts["report_markdown"])
        receipt_path = project_path(artifacts["receipt"])
        atomic_write_text(report_path, render_report(receipt))
        receipt["artifacts"]["report_markdown_path"] = relative_path(report_path)
        receipt["artifacts"]["report_markdown_sha256"] = sha256_file(report_path)
        atomic_write_json(receipt_path, receipt)
        print(f"官方财务事实采集状态：{status}", flush=True)
        print(
            f"终局PDF：{terminal_count:,}/{len(queue_state):,}；"
            f"目标事件完整率：{coverage['overall_complete_event_ratio']:.6f}",
            flush=True,
        )
        print("本程序未读取任何市场价格或未来收益。", flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 - 程序级失败必须明确
        print(
            f"官方财务事实采集程序失败：{type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
