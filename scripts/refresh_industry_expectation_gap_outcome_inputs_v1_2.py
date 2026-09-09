"""行业预期差零付费结果输入闸门；不调用商业数据源。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "industry_expectation_gap_outcome_refresh_v1_2.yaml"


class ZeroPaidOutcomeRefreshError(RuntimeError):
    """零付费结果输入闸门的内部错误。"""


def _path(relative: str) -> Path:
    root = ROOT.resolve()
    value = (root / relative).resolve()
    value.relative_to(root)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _write_status(config: dict[str, Any], payload: dict[str, Any]) -> Path:
    current = _path(config["outputs"]["current_status"])
    directory = _path(config["outputs"]["immutable_status_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(str(payload["retrieved_at"])).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    immutable = directory / f"outcome_refresh_{stamp}.json"
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    try:
        with immutable.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ZeroPaidOutcomeRefreshError(
            f"不可变刷新收据已存在：{immutable}"
        ) from exc
    _atomic_json(current, payload)
    return immutable


def _latest_date(path: Path) -> str:
    if not path.is_file():
        raise ZeroPaidOutcomeRefreshError(f"结果输入缺失：{path}")
    frame = pd.read_parquet(path, columns=["date"])
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        raise ZeroPaidOutcomeRefreshError(f"结果输入日期为空：{path}")
    return dates.max().date().isoformat()


def _target_date(
    config: dict[str, Any], now: datetime, explicit_date: str | None
) -> tuple[str | None, str | None]:
    calendar = pd.read_csv(_path(config["inputs"]["trading_calendar"]), dtype=str)
    column = "trade_date" if "trade_date" in calendar.columns else "date"
    dates = {
        pd.Timestamp(value).date().isoformat()
        for value in calendar[column].dropna().astype(str)
    }
    target = explicit_date or now.date().isoformat()
    if target not in dates:
        return None, "SKIPPED_NON_TRADING_DAY"
    cutoff = time.fromisoformat(config["quality"]["market_close_confirmation_time"])
    if explicit_date is None and now.timetz().replace(tzinfo=None) < cutoff:
        return None, "SKIPPED_BEFORE_MARKET_CLOSE"
    return target, None


def _registry_integrity(config: dict[str, Any]) -> dict[str, Any]:
    path = _path(config["inputs"]["free_source_registry"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    nonzero = [
        row["source_id"]
        for row in rows
        if float(row.get("access_cost_cny", "0") or 0) != 0
    ]
    if nonzero:
        raise ZeroPaidOutcomeRefreshError(
            f"免费来源登记出现非零成本：{','.join(nonzero)}"
        )
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "row_count": len(rows),
        "nonzero_cost_source_ids": nonzero,
    }


def _verified_acquisition_receipt(
    config: dict[str, Any], target_date: str, current_hashes: dict[str, str]
) -> bool:
    path = _path(config["inputs"]["qualified_acquisition_receipt"])
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        payload.get("status") == "PASS_ZERO_COST_300_CONSTITUENT_TOTAL_RETURN"
        and payload.get("target_date") == target_date
        and payload.get("access_cost_cny") == 0
        and payload.get("after_sha256") == current_hashes
        and payload.get("immutable_receipt") is True
    )


def refresh(
    config: dict[str, Any], now: datetime, explicit_date: str | None = None
) -> int:
    target_date, skip_status = _target_date(config, now, explicit_date)
    registry = _registry_integrity(config)
    base = {
        "schema_version": "1.2.0",
        "refresh_version": config["version"],
        "retrieved_at": now.isoformat(),
        "target_date": target_date or explicit_date or now.date().isoformat(),
        "data_purchase_budget_cny": 0,
        "paid_provider_call_enabled": False,
        "tushare_proxy_call_enabled": False,
        "external_request_attempted": False,
        "files_changed": False,
        "free_source_registry": registry,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }
    if skip_status:
        payload = {
            **base,
            "run_status": "SUCCESS",
            "collection_status": skip_status,
            "failure_class": None,
            "error": None,
        }
        _write_status(config, payload)
        return 0
    assert target_date is not None
    constituent = _path(config["inputs"]["constituent_total_return_daily"])
    etf = _path(config["inputs"]["etf_total_return_daily"])
    latest = {
        "constituent_total_return_daily": _latest_date(constituent),
        "etf_total_return_daily": _latest_date(etf),
    }
    current_hashes = {
        constituent.relative_to(ROOT).as_posix(): _sha256(constituent),
        etf.relative_to(ROOT).as_posix(): _sha256(etf),
    }
    receipt_verified = _verified_acquisition_receipt(
        config, target_date, current_hashes
    )
    if all(value >= target_date for value in latest.values()) and receipt_verified:
        payload = {
            **base,
            "run_status": "SUCCESS",
            "collection_status": "ALREADY_CURRENT_VERIFIED_ZERO_PAID",
            "failure_class": None,
            "latest_dates": latest,
            "current_sha256": current_hashes,
            "qualified_acquisition_receipt_verified": True,
            "error": None,
        }
        _write_status(config, payload)
        return 0
    payload = {
        **base,
        "run_status": "FAILED",
        "collection_status": "BLOCKED_ZERO_COST_SOURCE_UNAVAILABLE",
        "failure_class": "EXTERNAL_FREE_SOURCE_UNAVAILABLE",
        "latest_dates": latest,
        "current_sha256": current_hashes,
        "qualified_acquisition_receipt_verified": receipt_verified,
        "error": (
            "尚无满足300只成分股总回报字段、零付费、稳定归档和双源校验的合格来源；"
            "本次未调用 TuShare 代理或任何付费接口。"
        ),
    }
    _write_status(config, payload)
    return 3


def main() -> int:
    parser = argparse.ArgumentParser(description="行业预期差零付费结果输入闸门")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--date", help="目标交易日 YYYY-MM-DD")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        print("零付费结果输入配置顶层必须是对象", file=sys.stderr)
        return 1
    now = datetime.now(ZoneInfo(str(config["timezone"])))
    try:
        result = refresh(config, now, args.date)
    except Exception as exc:
        payload = {
            "schema_version": "1.2.0",
            "refresh_version": config.get("version"),
            "retrieved_at": now.isoformat(),
            "target_date": args.date or now.date().isoformat(),
            "run_status": "FAILED",
            "collection_status": "PROGRAM_FAILED",
            "failure_class": "INTERNAL_PROGRAM_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "data_purchase_budget_cny": 0,
            "paid_provider_call_enabled": False,
            "tushare_proxy_call_enabled": False,
            "external_request_attempted": False,
            "files_changed": False,
        }
        try:
            _write_status(config, payload)
        except Exception:
            pass
        print(f"行业结果输入程序失败：{payload['error']}", file=sys.stderr)
        return 1
    if result == 3:
        print("行业结果输入阻塞：BLOCKED_ZERO_COST_SOURCE_UNAVAILABLE", file=sys.stderr)
    else:
        print("行业结果输入零付费闸门完成")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
