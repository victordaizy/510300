"""采集510300 PCF与盘中IOPV快照，建立不回填的前向历史。"""

from __future__ import annotations

import argparse
import hashlib
import json
import msvcrt
import os
import sys
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_data.etf_primary_market import (
    EtfPrimaryMarketRequest,
    ProviderSnapshot,
    SseIopvSnapshotProvider,
    SsePcfProvider,
)


CONFIG_FILE = PROJECT_ROOT / "config" / "primary_market_forward.yaml"
LOCK_FILE = PROJECT_ROOT / "tmp" / "510300_primary_market_forward.lock"
FREE_SOURCE_RECEIPT_FIELDS = {
    "source_id",
    "source_url_or_endpoint",
    "acquired_at",
    "market_date",
    "raw_response_hash",
    "raw_response_path",
    "parser_version",
    "schema_version",
    "quality_status",
    "fallback_source",
    "failure_reason",
}


class ExternalSourceAcquisitionError(RuntimeError):
    """免费外部来源访问或载荷解析失败。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(relative_path: str) -> Path:
    return PROJECT_ROOT / Path(relative_path)


def _iso_calendar_date(value: date | datetime | str | pd.Timestamp) -> str:
    """将供应商日期统一成ISO日期字符串，避免date与字符串恒不相等。"""

    normalized = pd.Timestamp(value)
    if pd.isna(normalized):
        raise ValueError(f"无法识别日期：{value!r}")
    return normalized.date().isoformat()


def _relative(path: Path, root: Path = PROJECT_ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_raw(
    snapshot: ProviderSnapshot,
    raw_root: Path,
    kind: str,
    observed_at: datetime,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    target_directory = raw_root / kind / observed_at.strftime("%Y%m%d")
    target_directory.mkdir(parents=True, exist_ok=True)
    target = target_directory / f"{observed_at.strftime('%Y%m%dT%H%M%S_%f%z')}.json"
    envelope = {
        "kind": kind,
        "retrieved_at": observed_at.isoformat(),
        "normalized_record": snapshot.record,
        "raw_payload": snapshot.raw_payload,
    }
    encoded = (
        json.dumps(envelope, ensure_ascii=False, indent=2, default=str) + "\n"
    ).encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": _relative(target, root),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


def _write_source_receipt(
    *,
    receipt_root: Path,
    source_id: str,
    source_url_or_endpoint: str,
    acquired_at: datetime,
    market_date: str,
    raw_metadata: dict[str, Any] | None,
    parser_version: str,
    schema_version: str,
    quality_status: str,
    fallback_source: str,
    failure_reason: str | None,
    root: Path = PROJECT_ROOT,
) -> Path:
    if not source_id or any(character in source_id for character in "\\/:"):
        raise ValueError(f"免费来源ID无效：{source_id}")
    target_directory = receipt_root / source_id / market_date.replace("-", "")
    target_directory.mkdir(parents=True, exist_ok=True)
    receipt_id = (
        f"{acquired_at.astimezone(ZoneInfo('UTC')).strftime('%Y%m%dT%H%M%S%fZ')}"
        f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    target = target_directory / f"{receipt_id}.json"
    payload = {
        "source_id": source_id,
        "source_url_or_endpoint": source_url_or_endpoint,
        "acquired_at": acquired_at.isoformat(),
        "market_date": market_date,
        "raw_response_hash": raw_metadata["sha256"] if raw_metadata else None,
        "raw_response_path": raw_metadata["path"] if raw_metadata else None,
        "parser_version": parser_version,
        "schema_version": schema_version,
        "quality_status": quality_status,
        "fallback_source": fallback_source,
        "failure_reason": failure_reason,
        "raw_response_bytes": raw_metadata["bytes"] if raw_metadata else None,
        "access_cost_cny": 0,
        "paid_data_used": False,
        "immutable_receipt": True,
    }
    missing = FREE_SOURCE_RECEIPT_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"免费来源回执缺少字段：{sorted(missing)}")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _relative(target, root)
    return target


def _source_contract(config: dict[str, Any], kind: str) -> dict[str, Any]:
    contract = config["source_contract"]
    source = contract["sources"][kind]
    if int(source["access_cost_cny"]) != 0:
        raise ValueError(f"免费来源出现非零采购成本：{kind}")
    result = dict(source)
    result["schema_version"] = str(contract["schema_version"])
    return result


def _source_receipt(
    config: dict[str, Any],
    kind: str,
    acquired_at: datetime,
    market_date: str,
    raw_metadata: dict[str, Any] | None,
    quality_status: str,
    failure_reason: str | None,
) -> Path:
    source = _source_contract(config, kind)
    endpoint = str(source["source_url_or_endpoint"]).format(
        fund_code=config["collector"]["fund_code"]
    )
    return _write_source_receipt(
        receipt_root=_resolve(config["outputs"]["source_receipt_directory"]),
        source_id=str(source["source_id"]),
        source_url_or_endpoint=endpoint,
        acquired_at=acquired_at,
        market_date=market_date,
        raw_metadata=raw_metadata,
        parser_version=str(source["parser_version"]),
        schema_version=str(source["schema_version"]),
        quality_status=quality_status,
        fallback_source=str(source.get("fallback_source", "")),
        failure_reason=failure_reason,
    )


def _persist_record(record: dict[str, Any], output_file: Path, key: str) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fetched = pd.DataFrame([record])
    existing = pd.read_parquet(output_file) if output_file.exists() else pd.DataFrame()
    combined = pd.concat([existing, fetched], ignore_index=True) if not existing.empty else fetched
    combined = combined.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    temporary = output_file.with_suffix(".parquet.tmp")
    combined.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(output_file)
    return {
        "file": output_file.relative_to(PROJECT_ROOT).as_posix(),
        "row_count": int(len(combined)),
        "first_key": str(combined[key].iloc[0]),
        "last_key": str(combined[key].iloc[-1]),
        "sha256": _sha256(output_file),
    }


def _quality_payload(
    config: dict[str, Any],
    pcf_snapshot: ProviderSnapshot,
    iopv_snapshot: ProviderSnapshot,
    pcf_storage: dict[str, Any],
    iopv_storage: dict[str, Any],
    raw_files: list[dict[str, Any]],
    source_receipts: list[Path],
) -> dict[str, Any]:
    pcf = pcf_snapshot.record
    iopv = iopv_snapshot.record
    checked_at = datetime.now(ZoneInfo(config["collector"]["timezone"]))
    exchange_timestamp = pd.Timestamp(iopv["exchange_timestamp"])
    during_collection_session = (
        checked_at.date() == pcf["trading_day"]
        and config["collector"]["collection_start_time"]
        <= checked_at.strftime("%H:%M:%S")
        <= config["collector"]["collection_end_time"]
    )
    clock_delay_seconds = max(
        0.0,
        (checked_at.replace(tzinfo=None) - exchange_timestamp.to_pydatetime()).total_seconds(),
    )
    snapshot_fresh = (
        not during_collection_session
        or clock_delay_seconds
        <= float(config["quality"]["maximum_clock_delay_seconds_during_session"])
    )
    checks = {
        "same_fund_code": pcf["fund_code"] == iopv["fund_code"],
        "same_trade_date": pcf["trading_day"] == iopv["trade_date"],
        "iopv_publication_enabled": bool(pcf["publish_iopv"]),
        "positive_iopv": float(iopv["iopv"]) > 0,
        "positive_last_price": float(iopv["last_price"]) > 0,
        "positive_creation_unit": int(pcf["creation_redemption_unit"]) > 0,
        "snapshot_fresh_during_collection_session": snapshot_fresh,
    }
    required = [
        checks["same_fund_code"],
        checks["same_trade_date"],
        checks["positive_iopv"],
        checks["positive_last_price"],
        checks["positive_creation_unit"],
        checks["snapshot_fresh_during_collection_session"],
    ]
    if bool(config["quality"]["require_iopv_publication"]):
        required.append(checks["iopv_publication_enabled"])
    status = "PASS" if all(required) else "FAIL"
    creation_unit_notional = int(pcf["creation_redemption_unit"]) * float(iopv["last_price"])
    return {
        "status": status,
        "checked_at": checked_at.isoformat(),
        "collector": config["collector"],
        "checks": checks,
        "freshness": {
            "during_collection_session": during_collection_session,
            "clock_delay_seconds": clock_delay_seconds,
            "maximum_allowed_seconds_during_session": float(
                config["quality"]["maximum_clock_delay_seconds_during_session"]
            ),
        },
        "latest_pcf": pcf,
        "latest_iopv": iopv,
        "account_feasibility": {
            "reference_account_cny": 20_000.0,
            "creation_unit_notional_cny": creation_unit_notional,
            "can_directly_create_or_redeem": creation_unit_notional <= 20_000.0,
            "intended_use": "只用PCF和IOPV观察一级市场吸收状态，在二级市场交易510300",
        },
        "storage": {"pcf": pcf_storage, "iopv": iopv_storage},
        "raw_response_files": [item["path"] for item in raw_files],
        "raw_response_metadata": raw_files,
        "source_acquisition_receipts": [
            _relative(path) for path in source_receipts
        ],
        "governance": config["governance"],
    }


def _collect_once(
    config: dict[str, Any],
    pcf_snapshot: ProviderSnapshot | None = None,
    pcf_storage: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], ProviderSnapshot, dict[str, Any]]:
    request = EtfPrimaryMarketRequest(
        fund_code=str(config["collector"]["fund_code"]),
        timeout_seconds=float(config["collector"]["timeout_seconds"]),
        timezone=str(config["collector"]["timezone"]),
    )
    observed_at = datetime.now(ZoneInfo(request.timezone))
    raw_root = _resolve(config["outputs"]["raw_response_directory"])
    raw_files: list[dict[str, Any]] = []
    source_receipts: list[Path] = []
    if pcf_snapshot is None:
        try:
            pcf_snapshot = SsePcfProvider().fetch(request)
        except Exception as exc:
            failure_at = datetime.now(ZoneInfo(request.timezone))
            _source_receipt(
                config,
                "pcf",
                failure_at,
                failure_at.date().isoformat(),
                None,
                "FAILED_SOURCE_ACCESS_OR_PARSE",
                f"{type(exc).__name__}: {exc}",
            )
            raise ExternalSourceAcquisitionError(
                f"PCF免费来源访问或解析失败：{type(exc).__name__}: {exc}"
            ) from exc
        pcf_observed_at = datetime.fromisoformat(str(pcf_snapshot.record["retrieved_at"]))
        pcf_raw = _write_raw(pcf_snapshot, raw_root, "pcf", pcf_observed_at)
        raw_files.append(pcf_raw)
        source_receipts.append(
            _source_receipt(
                config,
                "pcf",
                pcf_observed_at,
                _iso_calendar_date(pcf_snapshot.record["trading_day"]),
                pcf_raw,
                "PASS_RAW_CAPTURED_AND_PARSED",
                None,
            )
        )
        pcf_storage = _persist_record(
            pcf_snapshot.record,
            _resolve(config["outputs"]["pcf_daily_file"]),
            "trading_day",
        )
    if pcf_storage is None:
        raise RuntimeError("PCF存储元数据未初始化")
    try:
        iopv_snapshot = SseIopvSnapshotProvider().fetch(request)
    except Exception as exc:
        failure_at = datetime.now(ZoneInfo(request.timezone))
        _source_receipt(
            config,
            "iopv",
            failure_at,
            failure_at.date().isoformat(),
            None,
            "FAILED_SOURCE_ACCESS_OR_PARSE",
            f"{type(exc).__name__}: {exc}",
        )
        raise ExternalSourceAcquisitionError(
            f"IOPV免费来源访问或解析失败：{type(exc).__name__}: {exc}"
        ) from exc
    iopv_observed_at = datetime.fromisoformat(str(iopv_snapshot.record["retrieved_at"]))
    iopv_raw = _write_raw(iopv_snapshot, raw_root, "iopv", iopv_observed_at)
    raw_files.append(iopv_raw)
    source_receipts.append(
        _source_receipt(
            config,
            "iopv",
            iopv_observed_at,
            _iso_calendar_date(iopv_snapshot.record["trade_date"]),
            iopv_raw,
            "PASS_RAW_CAPTURED_AND_PARSED",
            None,
        )
    )
    iopv_storage = _persist_record(
        iopv_snapshot.record,
        _resolve(config["outputs"]["iopv_snapshot_file"]),
        "exchange_timestamp",
    )
    report = _quality_payload(
        config,
        pcf_snapshot,
        iopv_snapshot,
        pcf_storage,
        iopv_storage,
        raw_files,
        source_receipts,
    )
    report_file = _resolve(config["outputs"]["quality_report_file"])
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report, pcf_snapshot, pcf_storage


def _collection_sessions(config: dict[str, Any]) -> list[tuple[str, str]]:
    configured = config["collector"].get("collection_sessions")
    if configured:
        return [(str(item[0]), str(item[1])) for item in configured]
    return [
        (
            str(config["collector"]["collection_start_time"]),
            str(config["collector"]["collection_end_time"]),
        )
    ]


def _inside_collection_session(config: dict[str, Any], now: datetime) -> bool:
    clock = now.strftime("%H:%M:%S")
    return any(start <= clock <= end for start, end in _collection_sessions(config))


def _acquire_watch_lock() -> Any:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_FILE.open("a+b")
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        raise RuntimeError("已有510300连续采集进程运行，本次不重复启动") from None
    return handle


def _release_watch_lock(handle: Any | None) -> None:
    if handle is None:
        return
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="采集510300 PCF与IOPV前向快照")
    parser.add_argument("--watch", action="store_true", help="按配置间隔连续采集")
    parser.add_argument("--max-samples", type=int, default=None, help="连续模式最多采集次数")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    interval = int(config["collector"]["observation_interval_seconds"])
    retry_delay = int(config["collector"].get("retry_delay_seconds", 15))
    maximum_consecutive_failures = int(
        config["collector"].get("maximum_consecutive_failures", 5)
    )
    if not 3 <= interval <= 60:
        print("采集间隔必须位于3至60秒", file=sys.stderr)
        return 2
    if not 1 <= retry_delay <= 300:
        print("重试间隔必须位于1至300秒", file=sys.stderr)
        return 2
    if not 1 <= maximum_consecutive_failures <= 20:
        print("最大连续失败次数必须位于1至20", file=sys.stderr)
        return 2
    sample_count = 0
    consecutive_failures = 0
    total_failures = 0
    pcf_snapshot: ProviderSnapshot | None = None
    pcf_storage: dict[str, Any] | None = None
    lock_handle = None
    try:
        if args.watch:
            lock_handle = _acquire_watch_lock()
        while True:
            now = datetime.now(ZoneInfo(config["collector"]["timezone"]))
            end_time = max(end for _, end in _collection_sessions(config))
            if args.watch and now.strftime("%H:%M:%S") > end_time:
                print("已超过当日采集结束时间，连续采集正常结束")
                break
            if args.watch and not _inside_collection_session(config, now):
                time.sleep(min(interval, 60))
                continue
            try:
                report, pcf_snapshot, pcf_storage = _collect_once(
                    config, pcf_snapshot=pcf_snapshot, pcf_storage=pcf_storage
                )
            except Exception as exc:
                if not args.watch:
                    raise
                consecutive_failures += 1
                total_failures += 1
                print(
                    "PCF/IOPV单次采集失败，"
                    f"连续失败{consecutive_failures}/{maximum_consecutive_failures}，"
                    f"累计失败{total_failures}：{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                if consecutive_failures >= maximum_consecutive_failures:
                    message = (
                        "PCF/IOPV连续采集失败达到停止门槛："
                        f"{maximum_consecutive_failures}次；最后错误为"
                        f"{type(exc).__name__}: {exc}"
                    )
                    if isinstance(exc, ExternalSourceAcquisitionError):
                        raise ExternalSourceAcquisitionError(message) from exc
                    raise RuntimeError(message) from exc
                time.sleep(retry_delay)
                continue
            consecutive_failures = 0
            sample_count += 1
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            if not args.watch:
                break
            pcf_trading_day = _iso_calendar_date(report["latest_pcf"]["trading_day"])
            if pcf_trading_day != now.date().isoformat():
                print("PCF交易日不是本地当天，按休市日处理并结束连续采集")
                break
            if args.max_samples is not None and sample_count >= args.max_samples:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("采集由用户中止", file=sys.stderr)
    except ExternalSourceAcquisitionError as exc:
        print(f"PCF/IOPV免费外部来源失败：{exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"PCF/IOPV采集失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        _release_watch_lock(lock_handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
