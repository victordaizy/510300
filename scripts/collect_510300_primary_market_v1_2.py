"""PCF/IOPV V1.2：内容寻址原始层与收盘官方日端点复核。"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from market_data.etf_daily_crosscheck_v1_2 import SseEtfDailyCrosscheckProvider
from market_data.etf_primary_market import EtfPrimaryMarketRequest
from scripts import collect_510300_primary_market as base
from scripts.free_source_storage_v1_2 import (
    persist_record_atomic,
    write_content_addressed_raw,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "primary_market_forward_v1_2.yaml"


def _resolve(value: str) -> Path:
    return PROJECT_ROOT / Path(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    )
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _trading_dates(config: dict[str, Any]) -> set[date]:
    calendar = _resolve(str(config["inputs"]["trading_calendar_file"]))
    if not calendar.is_file():
        raise RuntimeError(f"交易日历缺失：{calendar}")
    values = pd.read_csv(calendar, dtype=str)
    if "trade_date" not in values.columns:
        raise RuntimeError("交易日历缺少 trade_date")
    return {
        pd.Timestamp(value).date()
        for value in values["trade_date"].dropna().astype(str)
    }


def _production_close_crosscheck_required(
    *,
    watch: bool,
    max_samples: int | None,
    now: datetime,
    config: dict[str, Any],
) -> bool:
    return bool(
        watch
        and max_samples is None
        and now.strftime("%H:%M:%S") > config["collector"]["collection_end_time"]
        and now.date() in _trading_dates(config)
    )


def collect_daily_crosscheck(
    config: dict[str, Any],
    market_date: date,
    *,
    provider: SseEtfDailyCrosscheckProvider | None = None,
) -> dict[str, Any]:
    request = EtfPrimaryMarketRequest(
        fund_code=str(config["collector"]["fund_code"]),
        timeout_seconds=float(config["collector"]["timeout_seconds"]),
        timezone=str(config["collector"]["timezone"]),
    )
    timezone = ZoneInfo(request.timezone)
    retry_count = int(config["collector"]["daily_crosscheck_retry_count"])
    retry_delay = int(
        config["collector"]["daily_crosscheck_retry_delay_seconds"]
    )
    if not 1 <= retry_count <= 10:
        raise ValueError("官方日端点重试次数必须位于1至10")
    if not 1 <= retry_delay <= 120:
        raise ValueError("官方日端点重试间隔必须位于1至120秒")
    actual_provider = provider or SseEtfDailyCrosscheckProvider()
    last_error: Exception | None = None
    for attempt in range(1, retry_count + 1):
        try:
            snapshot = actual_provider.fetch(request, market_date)
            observed_at = datetime.fromisoformat(
                str(snapshot.record["retrieved_at"])
            )
            raw_metadata = write_content_addressed_raw(
                snapshot,
                _resolve(config["outputs"]["raw_response_directory"]),
                "daily_crosscheck",
                observed_at,
            )
            receipt = base._source_receipt(
                config,
                "daily_crosscheck",
                observed_at,
                market_date.isoformat(),
                raw_metadata,
                "PASS_RAW_CAPTURED_AND_PARSED",
                None,
            )
            record = {
                **snapshot.record,
                "raw_response_path": raw_metadata["path"],
                "raw_response_hash": raw_metadata["sha256"],
                "source_receipt_path": base._relative(receipt),
                "content_addressed": True,
            }
            storage = persist_record_atomic(
                record,
                _resolve(config["outputs"]["daily_crosscheck_file"]),
                "trade_date",
            )
            payload = {
                "schema_version": "1.0.0",
                "status": "PASS_OFFICIAL_DAILY_CROSSCHECK_CAPTURED",
                "market_date": market_date.isoformat(),
                "attempt": attempt,
                "source_id": "SSE_ETF_DAILY_TURNOVER_OFFICIAL",
                "raw_response": raw_metadata,
                "source_receipt_path": base._relative(receipt),
                "storage": storage,
                "generated_at": datetime.now(timezone).isoformat(),
            }
            _atomic_json(
                _resolve(config["outputs"]["daily_crosscheck_status_file"]),
                payload,
            )
            return payload
        except Exception as exc:
            last_error = exc
            failure_at = datetime.now(timezone)
            base._source_receipt(
                config,
                "daily_crosscheck",
                failure_at,
                market_date.isoformat(),
                None,
                f"FAILED_SOURCE_ACCESS_OR_PARSE_RETRY_{attempt}",
                f"{type(exc).__name__}: {exc}",
            )
            if attempt < retry_count:
                time.sleep(retry_delay)
    assert last_error is not None
    payload = {
        "schema_version": "1.0.0",
        "status": "FAILED_OFFICIAL_DAILY_CROSSCHECK_SOURCE",
        "market_date": market_date.isoformat(),
        "attempts": retry_count,
        "source_id": "SSE_ETF_DAILY_TURNOVER_OFFICIAL",
        "failure_reason": f"{type(last_error).__name__}: {last_error}",
        "generated_at": datetime.now(timezone).isoformat(),
    }
    _atomic_json(
        _resolve(config["outputs"]["daily_crosscheck_status_file"]), payload
    )
    raise base.ExternalSourceAcquisitionError(
        "上交所官方 ETF 成交概况在冻结重试次数内未形成当日证据："
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PCF/IOPV V1.2 配置顶层必须是对象")
    args = base._parser().parse_args()
    base.CONFIG_FILE = CONFIG_FILE
    base._write_raw = write_content_addressed_raw
    base._persist_record = persist_record_atomic
    exit_code = base.main()
    if exit_code != 0:
        return exit_code
    now = datetime.now(ZoneInfo(str(config["collector"]["timezone"])))
    if not _production_close_crosscheck_required(
        watch=args.watch,
        max_samples=args.max_samples,
        now=now,
        config=config,
    ):
        return 0
    try:
        payload = collect_daily_crosscheck(config, now.date())
    except base.ExternalSourceAcquisitionError as exc:
        print(f"PCF/IOPV第二官方端点失败：{exc}", file=sys.stderr)
        return 3
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
