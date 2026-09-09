"""采集中证官方定期调样公告并生成严格无收益事件账本。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_hs_csi300_official_addition_forced_demand_v1 import (  # noqa: E402
    STRATEGY_ID,
    build_event_ledger,
    extract_effective_date,
    extract_official_attachment_urls,
    load_open_trading_days,
    parse_official_csi300_changes,
    sha256_bytes,
    sha256_file,
    validate_no_result_columns,
)


DEFAULT_CONFIG = ROOT / "config" / "a_share_hs_csi300_official_addition_forced_demand_v1.yaml"
SUPPORTED_ATTACHMENT_SUFFIXES = {".xls", ".xlsx", ".pdf", ".doc", ".docx", ".zip"}


def now_shanghai() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.parquet")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.csv")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(temporary, path)


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
            "Origin": "https://www.csindex.com.cn",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def checked_response(response: requests.Response, context: str) -> bytes:
    if response.status_code != 200:
        raise RuntimeError(f"{context} HTTP 状态异常：{response.status_code}")
    if not response.content:
        raise RuntimeError(f"{context} 返回空内容")
    return response.content


def archive_content_addressed(directory: Path, prefix: str, payload: bytes, suffix: str) -> tuple[Path, str]:
    digest = sha256_bytes(payload)
    path = directory / f"{prefix}_{digest}{suffix.lower()}"
    if path.exists():
        if sha256_file(path) != digest:
            raise RuntimeError(f"已有同名原始对象哈希异常：{path}")
    else:
        atomic_write_bytes(path, payload)
    return path, digest


def load_single_cached_raw(directory: Path, prefix: str) -> tuple[bytes, Path, str] | None:
    """恢复同一策略本次已归档的唯一原始对象，避免失败后重复下载。"""

    files = sorted(directory.glob(f"{prefix}_*")) if directory.exists() else []
    if not files:
        return None
    if len(files) != 1:
        raise RuntimeError(f"原始对象存在多个版本，拒绝自动选择：{[str(path) for path in files]}")
    path = files[0]
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    if digest not in path.name:
        raise RuntimeError(f"原始对象文件名哈希与内容不一致：{path}")
    return payload, path, digest


def infer_attachment_suffix(url: str, response: requests.Response) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in SUPPORTED_ATTACHMENT_SUFFIXES:
        return suffix
    content_type = str(response.headers.get("Content-Type") or "").lower()
    if response.content.startswith(b"%PDF") or "application/pdf" in content_type:
        return ".pdf"
    if response.content.startswith(b"PK\x03\x04"):
        return ".xlsx"
    if response.content.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        return ".xls"
    raise ValueError(f"无法识别官方附件格式：{url}，Content-Type={content_type}")


def parse_json_bytes(payload: bytes, context: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} 不是有效 UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{context} JSON 顶层不是对象")
    return parsed


def make_source_object(
    *,
    kind: str,
    url: str,
    method: str,
    archive_path: Path,
    digest: str,
    size_bytes: int,
    request_body_sha256: str | None = None,
    retrieval_mode: str = "NETWORK_TLS_VERIFIED",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": kind,
        "url": url,
        "method": method,
        "archive_path": archive_path.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "size_bytes": size_bytes,
        "retrieval_mode": retrieval_mode,
    }
    if request_body_sha256:
        record["request_body_sha256"] = request_body_sha256
    return record


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("strategy_id") != STRATEGY_ID:
        raise ValueError("配置策略标识与代码不一致")
    universe = contract["event_universe"]
    cycles = universe["expected_cycles"]
    if len(cycles) != int(universe["expected_cycle_count"]):
        raise ValueError("冻结周期数量与 expected_cycle_count 不一致")
    if len({str(item["cycle_id"]) for item in cycles}) != len(cycles):
        raise ValueError("冻结周期标识重复")
    if len({int(item["announcement_id"]) for item in cycles}) != len(cycles):
        raise ValueError("冻结公告标识重复")
    if contract["data_gates"]["D8"]["return_output_allowed"]:
        raise ValueError("D8 必须禁止正式运行前的收益输出")
    if contract["governance"]["return_evaluation_attempts"] != 1:
        raise ValueError("历史收益评估次数必须冻结为一次")


def acquire(contract: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    source = contract["official_source_contract"]
    artifacts = contract["artifacts"]
    raw_root = resolve_path(artifacts["raw_root"])
    list_raw_root = raw_root / "list"
    detail_raw_root = raw_root / "details"
    attachment_raw_root = raw_root / "attachments"
    session = build_session()
    bootstrap = session.get("https://www.csindex.com.cn/", timeout=timeout_seconds)
    if bootstrap.status_code != 200:
        raise RuntimeError(f"中证官网会话初始化失败：HTTP {bootstrap.status_code}")
    retrieved_at = now_shanghai()

    list_query = source["list_query"]
    request_bytes = json.dumps(list_query, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cached_list = load_single_cached_raw(list_raw_root, "announcement_list")
    if cached_list is None:
        list_response = session.post(source["list_endpoint"], json=list_query, timeout=timeout_seconds)
        list_bytes = checked_response(list_response, "官方公告列表")
        list_path, list_digest = archive_content_addressed(list_raw_root, "announcement_list", list_bytes, ".json")
        list_retrieval_mode = "NETWORK_TLS_VERIFIED"
    else:
        list_bytes, list_path, list_digest = cached_list
        list_retrieval_mode = "REUSED_IMMUTABLE_RAW"
    list_json = parse_json_bytes(list_bytes, "官方公告列表")
    if str(list_json.get("code")) != "200" or not list_json.get("success"):
        raise RuntimeError(f"官方公告列表业务状态异常：{list_json.get('code')} {list_json.get('msg')}")
    list_source = make_source_object(
        kind="OFFICIAL_ANNOUNCEMENT_LIST",
        url=source["list_endpoint"],
        method="POST",
        archive_path=list_path,
        digest=list_digest,
        size_bytes=len(list_bytes),
        request_body_sha256=sha256_bytes(request_bytes),
        retrieval_mode=list_retrieval_mode,
    )
    listed_ids = {int(item["id"]) for item in list_json.get("data") or [] if isinstance(item, dict) and item.get("id") is not None}
    expected_cycles = contract["event_universe"]["expected_cycles"]
    missing_from_list = [int(item["announcement_id"]) for item in expected_cycles if int(item["announcement_id"]) not in listed_ids]
    if missing_from_list:
        raise RuntimeError(f"官方列表未包含冻结公告标识：{missing_from_list}")

    source_objects: list[dict[str, Any]] = [list_source]
    cycle_records: list[dict[str, Any]] = []
    cycle_manifest: list[dict[str, Any]] = []
    for expected in expected_cycles:
        cycle_id = str(expected["cycle_id"])
        announcement_id = int(expected["announcement_id"])
        detail_url = source["detail_endpoint_template"].format(announcement_id=announcement_id)
        cached_detail = load_single_cached_raw(detail_raw_root, f"announcement_{announcement_id}")
        if cached_detail is None:
            detail_response = session.get(detail_url, timeout=timeout_seconds)
            detail_bytes = checked_response(detail_response, f"{cycle_id} 官方公告详情")
            detail_path, detail_digest = archive_content_addressed(
                detail_raw_root,
                f"announcement_{announcement_id}",
                detail_bytes,
                ".json",
            )
            detail_retrieval_mode = "NETWORK_TLS_VERIFIED"
        else:
            detail_bytes, detail_path, detail_digest = cached_detail
            detail_retrieval_mode = "REUSED_IMMUTABLE_RAW"
        detail_json = parse_json_bytes(detail_bytes, f"{cycle_id} 官方公告详情")
        detail = detail_json.get("data")
        if str(detail_json.get("code")) != "200" or not detail_json.get("success") or not isinstance(detail, dict):
            raise RuntimeError(f"{cycle_id} 官方公告详情业务状态异常")
        if int(detail.get("id")) != announcement_id:
            raise RuntimeError(f"{cycle_id} 官方公告详情标识错配")
        title = str(detail.get("title") or "")
        if "沪深300" not in title:
            raise RuntimeError(f"{cycle_id} 官方公告标题不含沪深300：{title}")

        detail_source = make_source_object(
            kind="OFFICIAL_ANNOUNCEMENT_DETAIL",
            url=detail_url,
            method="GET",
            archive_path=detail_path,
            digest=detail_digest,
            size_bytes=len(detail_bytes),
            retrieval_mode=detail_retrieval_mode,
        )
        source_objects.append(detail_source)

        downloaded_attachments: list[dict[str, Any]] = []
        for attachment_index, attachment_meta in enumerate(extract_official_attachment_urls(detail), start=1):
            attachment_url = attachment_meta["url"]
            suffix_from_url = Path(urlparse(attachment_url).path).suffix.lower()
            if suffix_from_url and suffix_from_url not in SUPPORTED_ATTACHMENT_SUFFIXES:
                continue
            attachment_prefix = f"announcement_{announcement_id}_attachment_{attachment_index}"
            cached_attachment = load_single_cached_raw(attachment_raw_root, attachment_prefix)
            if cached_attachment is None:
                response = session.get(attachment_url, timeout=max(timeout_seconds, 60.0))
                payload = checked_response(response, f"{cycle_id} 官方附件")
                suffix = infer_attachment_suffix(attachment_url, response)
                archive_path, digest = archive_content_addressed(
                    attachment_raw_root,
                    attachment_prefix,
                    payload,
                    suffix,
                )
                attachment_retrieval_mode = "NETWORK_TLS_VERIFIED"
            else:
                payload, archive_path, digest = cached_attachment
                suffix = archive_path.suffix.lower()
                attachment_retrieval_mode = "REUSED_IMMUTABLE_RAW"
            source_record = make_source_object(
                kind="OFFICIAL_ANNOUNCEMENT_ATTACHMENT",
                url=attachment_url,
                method="GET",
                archive_path=archive_path,
                digest=digest,
                size_bytes=len(payload),
                retrieval_mode=attachment_retrieval_mode,
            )
            source_objects.append(source_record)
            downloaded_attachments.append(
                {
                    "url": attachment_url,
                    "file_name": attachment_meta["file_name"],
                    "payload": payload,
                    "suffix": suffix,
                    "sha256": digest,
                    "archive_path": archive_path.relative_to(ROOT).as_posix(),
                    "size_bytes": len(payload),
                }
            )
        if not downloaded_attachments:
            raise RuntimeError(f"{cycle_id} 未取得任何可归档官方名单附件")

        parsed = parse_official_csi300_changes(detail, downloaded_attachments)
        publish_date = str(detail.get("publishDate") or "")[:10]
        effective_date = extract_effective_date(str(detail.get("content") or ""), publish_date)
        parse_attachment = parsed.get("attachment") or {}
        cycle_record = {
            "cycle_id": cycle_id,
            "announcement_id": announcement_id,
            "announcement_title": title,
            "announcement_date": publish_date,
            "effective_date": effective_date,
            "source_detail_url": detail_url,
            "source_detail_sha256": detail_digest,
            "source_format": parsed["source_format"],
            "extraction_method": parsed["method"],
            "attachment": parse_attachment,
            "additions": parsed["additions"],
        }
        cycle_records.append(cycle_record)
        cycle_manifest.append(
            {
                "cycle_id": cycle_id,
                "announcement_id": announcement_id,
                "announcement_title": title,
                "announcement_date": publish_date,
                "effective_date": effective_date,
                "detail": detail_source,
                "attachments": [
                    {key: value for key, value in item.items() if key != "payload"}
                    for item in downloaded_attachments
                ],
                "extraction_method": parsed["method"],
                "source_format": parsed["source_format"],
                "addition_count": len(parsed["additions"]),
                "deletion_count": len(parsed["deletions"]),
                "additions": parsed["additions"],
            }
        )

    calendar_path = resolve_path(contract["calendar_contract"]["path"])
    open_days = load_open_trading_days(calendar_path, contract["calendar_contract"]["sha256"])
    ledger = build_event_ledger(cycle_records, open_days)
    expected_cycle_ids = [str(item["cycle_id"]) for item in expected_cycles]
    actual_cycle_ids = ledger["cycle_id"].drop_duplicates().tolist()
    if actual_cycle_ids != expected_cycle_ids:
        raise RuntimeError(f"事件账本周期顺序与冻结配置不一致：{actual_cycle_ids}")

    candidate_columns = [
        "event_key",
        "cycle_id",
        "announcement_id",
        "announcement_date",
        "entry_date",
        "effective_date",
        "security_code",
        "ts_code",
        "security_name",
        "exchange",
        "official_sequence",
        "candidate_state",
        "return_evaluation",
    ]
    candidates = ledger[candidate_columns].copy()
    validate_no_result_columns(candidates.columns)

    ledger_path = resolve_path(artifacts["event_ledger_without_returns"])
    candidate_path = resolve_path(artifacts["candidate_list_without_returns"])
    manifest_path = resolve_path(artifacts["official_announcement_manifest"])
    atomic_write_parquet(ledger_path, ledger)
    atomic_write_csv(candidate_path, candidates)

    manifest = {
        "schema_version": "1.0.0",
        "strategy_id": STRATEGY_ID,
        "state": "OFFICIAL_SOURCE_ACQUIRED_WITHOUT_RETURNS",
        "retrieved_at": retrieved_at,
        "evidence_cutoff": contract["evidence_cutoff"],
        "tls_verification": True,
        "source_hopping": False,
        "list_source": list_source,
        "source_objects": source_objects,
        "cycles": cycle_manifest,
        "cycle_count": len(cycle_manifest),
        "addition_event_count": int(len(ledger)),
        "unique_event_count": int(ledger["event_key"].nunique()),
        "event_ledger": {
            "path": ledger_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(ledger_path),
            "rows": int(len(ledger)),
            "columns": list(ledger.columns),
        },
        "candidate_list": {
            "path": candidate_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(candidate_path),
            "rows": int(len(candidates)),
            "columns": list(candidates.columns),
        },
        "calendar": {
            "path": calendar_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(calendar_path),
            "allowed_use": "EVENT_CLOCK_ONLY",
        },
        "return_evaluation": "NOT_ALLOWED",
    }
    atomic_write_json(manifest_path, manifest)

    complete_cycle_count = int(ledger["cycle_id"].nunique())
    unique_event_count = int(ledger["event_key"].nunique())
    gates = {
        "D1": {
            "status": "PASS" if complete_cycle_count == int(contract["event_universe"]["expected_cycle_count"]) else "FAIL",
            "observed_cycle_count": complete_cycle_count,
            "required_cycle_count": int(contract["event_universe"]["expected_cycle_count"]),
            "missing_official_list_ids": missing_from_list,
        },
        "D2": {
            "status": "PASS" if all(item["attachments"] and item["detail"]["sha256"] and item["effective_date"] for item in cycle_manifest) else "FAIL",
            "cycles_with_detail_hash_attachment_and_effective_date": sum(
                bool(item["attachments"] and item["detail"]["sha256"] and item["effective_date"])
                for item in cycle_manifest
            ),
            "required_cycles": len(expected_cycles),
        },
        "D3": {
            "status": "PASS"
            if complete_cycle_count >= int(contract["data_gates"]["D3"]["minimum_complete_cycles"])
            and unique_event_count >= int(contract["data_gates"]["D3"]["minimum_unique_additions"])
            else "FAIL",
            "complete_cycle_count": complete_cycle_count,
            "unique_addition_event_count": unique_event_count,
        },
        "D4": {"status": "NOT_ASSESSED", "reason": "价格、成交额、停牌、涨跌停和公司行为尚未执行覆盖准入"},
        "D5": {"status": "NOT_ASSESSED", "reason": "退出可卖性尚未执行完整性准入"},
        "D6": {"status": "NOT_ASSESSED", "reason": "由冻结入口运行核心机制测试"},
        "D7": {"status": "NOT_FROZEN", "reason": "由冻结入口生成协议清单"},
        "D8": {
            "status": "PASS",
            "price_or_return_outputs": [],
            "return_evaluation": "NOT_ALLOWED",
        },
    }
    report = {
        "schema_version": "1.0.0",
        "strategy_id": STRATEGY_ID,
        "generated_at": now_shanghai(),
        "status": "NO_VIEW_DATA_CONTRACT_FAILED",
        "return_evaluation": "NOT_ALLOWED",
        "gates": gates,
        "official_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "official_manifest_sha256": sha256_file(manifest_path),
        "event_ledger_path": ledger_path.relative_to(ROOT).as_posix(),
        "event_ledger_sha256": sha256_file(ledger_path),
        "candidate_list_path": candidate_path.relative_to(ROOT).as_posix(),
        "candidate_list_sha256": sha256_file(candidate_path),
        "next_allowed_step": "仅执行 D4/D5 数据覆盖准入和 D6/D7 冻结；不得计算收益",
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    gates = report["gates"]
    d1_detail = (
        f"，{gates['D1']['observed_cycle_count']}/{gates['D1']['required_cycle_count']}"
        if "observed_cycle_count" in gates["D1"]
        else ""
    )
    d2_detail = (
        f"，{gates['D2']['cycles_with_detail_hash_attachment_and_effective_date']}/{gates['D2']['required_cycles']}"
        if "cycles_with_detail_hash_attachment_and_effective_date" in gates["D2"]
        else (
            f"，原始详情 {gates['D2']['raw_detail_cycle_count']}/{gates['D2']['required_cycle_count']}，"
            f"附件 {gates['D2']['raw_attachment_cycle_count']}/{gates['D2']['required_cycle_count']}"
            if "raw_detail_cycle_count" in gates["D2"]
            else ""
        )
    )
    d3_detail = (
        f"，{gates['D3']['unique_addition_event_count']} 个事件"
        if "unique_addition_event_count" in gates["D3"]
        else ""
    )
    next_step = report.get("next_allowed_step", "修复官方源或解析错误后重试；不得计算收益")
    return "\n".join(
        [
            f"# {STRATEGY_ID} 官方源准入",
            "",
            f"- 状态：`{report['status']}`",
            f"- 收益评估：`{report['return_evaluation']}`",
            f"- D1 周期枚举：`{gates['D1']['status']}`{d1_detail}",
            f"- D2 官方原文与附件：`{gates['D2']['status']}`{d2_detail}",
            f"- D3 事件数量：`{gates['D3']['status']}`{d3_detail}",
            f"- D4：`{gates['D4']['status']}`",
            f"- D5：`{gates['D5']['status']}`",
            f"- D6：`{gates['D6']['status']}`",
            f"- D7：`{gates['D7']['status']}`",
            f"- D8 无收益边界：`{gates['D8']['status']}`",
            "",
            f"下一步：{next_step}。",
            "",
        ]
    )


def partial_source_progress(contract: dict[str, Any]) -> dict[str, Any]:
    """只按已归档原始对象报告进度，不把部分原文误记为完整 D2。"""

    raw_root = resolve_path(contract["artifacts"]["raw_root"])
    expected_cycles = contract["event_universe"]["expected_cycles"]
    expected_ids = {int(item["announcement_id"]): str(item["cycle_id"]) for item in expected_cycles}
    list_files = sorted((raw_root / "list").glob("announcement_list_*.json"))
    d1_pass = False
    if len(list_files) == 1:
        try:
            list_payload = parse_json_bytes(list_files[0].read_bytes(), "已归档官方公告列表")
            listed_ids = {
                int(item["id"])
                for item in list_payload.get("data") or []
                if isinstance(item, dict) and item.get("id") is not None
            }
            d1_pass = set(expected_ids).issubset(listed_ids)
        except Exception:  # noqa: BLE001 - 进度报告不得覆盖原始失败
            d1_pass = False
    detail_cycle_ids = [
        cycle_id
        for announcement_id, cycle_id in expected_ids.items()
        if len(list((raw_root / "details").glob(f"announcement_{announcement_id}_*.json"))) == 1
    ]
    attachment_cycle_ids = [
        cycle_id
        for announcement_id, cycle_id in expected_ids.items()
        if any((raw_root / "attachments").glob(f"announcement_{announcement_id}_attachment_*"))
    ]
    return {
        "d1_official_list_contains_all_frozen_ids": d1_pass,
        "raw_detail_cycle_count": len(detail_cycle_ids),
        "raw_attachment_cycle_count": len(attachment_cycle_ids),
        "raw_detail_cycle_ids": detail_cycle_ids,
        "raw_attachment_cycle_ids": attachment_cycle_ids,
        "required_cycle_count": len(expected_cycles),
    }


def failure_report(config_path: Path, contract: dict[str, Any], exc: Exception) -> dict[str, Any]:
    progress = partial_source_progress(contract)
    return {
        "schema_version": "1.0.0",
        "strategy_id": STRATEGY_ID,
        "generated_at": now_shanghai(),
        "status": "NO_VIEW_DATA_CONTRACT_FAILED",
        "return_evaluation": "NOT_ALLOWED",
        "config_path": config_path.relative_to(ROOT).as_posix() if config_path.is_relative_to(ROOT) else str(config_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "partial_source_progress": progress,
        "gates": {
            "D1": {
                "status": "PASS" if progress["d1_official_list_contains_all_frozen_ids"] else "FAIL",
                "observed_cycle_count": progress["required_cycle_count"]
                if progress["d1_official_list_contains_all_frozen_ids"]
                else 0,
                "required_cycle_count": progress["required_cycle_count"],
            },
            "D2": {
                "status": "FAIL",
                "raw_detail_cycle_count": progress["raw_detail_cycle_count"],
                "raw_attachment_cycle_count": progress["raw_attachment_cycle_count"],
                "required_cycle_count": progress["required_cycle_count"],
            },
            "D3": {"status": "NOT_ASSESSED"},
            "D4": {"status": "NOT_ASSESSED"},
            "D5": {"status": "NOT_ASSESSED"},
            "D6": {"status": "NOT_ASSESSED"},
            "D7": {"status": "NOT_FROZEN"},
            "D8": {"status": "PASS", "return_evaluation": "NOT_ALLOWED"},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="采集中证官方调样事件，不读取价格或收益")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    contract = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_contract(contract)
    report_path = resolve_path(contract["artifacts"]["acquisition_report_json"])
    markdown_path = resolve_path(contract["artifacts"]["acquisition_report_md"])
    try:
        report = acquire(contract, args.timeout_seconds)
        atomic_write_json(report_path, report)
        atomic_write_text(markdown_path, render_markdown(report))
    except Exception as exc:  # noqa: BLE001 - 失败状态必须落盘并禁止收益评估
        report = failure_report(config_path, contract, exc)
        atomic_write_json(report_path, report)
        atomic_write_text(markdown_path, render_markdown(report))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    print(
        json.dumps(
            {
                "状态": report["status"],
                "收益评估": report["return_evaluation"],
                "D1": report["gates"]["D1"],
                "D2": report["gates"]["D2"],
                "D3": report["gates"]["D3"],
                "D8": report["gates"]["D8"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
