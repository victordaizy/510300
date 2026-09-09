"""生成 PCF/IOPV V1.2 可重放成熟度报告。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.primary_market_forward_readiness_v1_2 import (
    evaluate_forward_readiness_v1_2,
)


CONFIG_FILE = PROJECT_ROOT / "config" / "primary_market_forward_v1_2.yaml"


def _resolve(value: str) -> Path:
    return PROJECT_ROOT / Path(value)


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def _merge_pcf(legacy: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not legacy.empty:
        value = legacy.copy()
        value["evidence_generation"] = "LEGACY_PRE_V1_2"
        frames.append(value)
    if not current.empty:
        value = current.copy()
        value["evidence_generation"] = "V1_2_CONTENT_ADDRESSED"
        frames.append(value)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["_date"] = pd.to_datetime(combined["trading_day"], errors="coerce")
    combined = combined.sort_values(["_date", "evidence_generation"])
    return combined.drop_duplicates("_date", keep="last").drop(columns="_date")


def _merge_iopv(legacy: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not legacy.empty:
        value = legacy.copy()
        value["evidence_generation"] = "LEGACY_PRE_V1_2"
        frames.append(value)
    if not current.empty:
        value = current.copy()
        value["evidence_generation"] = "V1_2_CONTENT_ADDRESSED"
        frames.append(value)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["_date"] = pd.to_datetime(combined["trade_date"], errors="coerce")
    combined["_timestamp"] = pd.to_datetime(
        combined["exchange_timestamp"], errors="coerce"
    )
    combined = combined.sort_values(
        ["_date", "_timestamp", "evidence_generation"]
    )
    return combined.drop_duplicates(
        ["_date", "_timestamp"], keep="last"
    ).drop(columns=["_date", "_timestamp"])


def _verify_crosscheck_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    raw_verified: list[bool] = []
    receipt_verified: list[bool] = []
    for record in result.to_dict(orient="records"):
        raw_path = _resolve(str(record.get("raw_response_path", "")))
        receipt_path = _resolve(str(record.get("source_receipt_path", "")))
        expected_hash = str(record.get("raw_response_hash", ""))
        raw_ok = bool(
            raw_path.is_file()
            and expected_hash
            and hashlib.sha256(raw_path.read_bytes()).hexdigest() == expected_hash
            and raw_path.stem == expected_hash
        )
        receipt_ok = False
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                receipt = {}
            receipt_ok = bool(
                receipt.get("source_id") == "SSE_ETF_DAILY_TURNOVER_OFFICIAL"
                and receipt.get("market_date")
                == pd.Timestamp(record["trade_date"]).date().isoformat()
                and receipt.get("raw_response_hash") == expected_hash
                and receipt.get("raw_response_path")
                == str(record.get("raw_response_path"))
                and receipt.get("quality_status")
                == "PASS_RAW_CAPTURED_AND_PARSED"
                and receipt.get("paid_data_used") is False
                and receipt.get("immutable_receipt") is True
            )
        raw_verified.append(raw_ok)
        receipt_verified.append(receipt_ok)
    result["raw_hash_verified"] = raw_verified
    result["receipt_verified"] = receipt_verified
    return result


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


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PCF/IOPV V1.2 配置顶层必须是对象")
    legacy_pcf = _read_optional(_resolve(config["inputs"]["legacy_pcf_daily_file"]))
    legacy_iopv = _read_optional(
        _resolve(config["inputs"]["legacy_iopv_snapshot_file"])
    )
    current_pcf = _read_optional(_resolve(config["outputs"]["pcf_daily_file"]))
    current_iopv = _read_optional(
        _resolve(config["outputs"]["iopv_snapshot_file"])
    )
    pcf = _merge_pcf(legacy_pcf, current_pcf)
    iopv = _merge_iopv(legacy_iopv, current_iopv)
    if pcf.empty or iopv.empty:
        print("缺少 PCF 或 IOPV 前向数据，暂时无法生成 V1.2 成熟度报告", file=sys.stderr)
        return 1
    daily_crosscheck = _verify_crosscheck_evidence(
        _read_optional(_resolve(config["outputs"]["daily_crosscheck_file"]))
    )
    quality = config["quality"]
    readiness_config = config["readiness"]
    readiness = evaluate_forward_readiness_v1_2(
        pcf,
        iopv,
        daily_crosscheck,
        minimum_full_coverage_days=int(
            readiness_config["minimum_full_coverage_days"]
        ),
        recommended_full_coverage_days=int(
            readiness_config["recommended_full_coverage_days"]
        ),
        minimum_snapshots_per_day=int(
            readiness_config["minimum_snapshots_per_day"]
        ),
        first_unseen_evaluation_full_coverage_days=int(
            readiness_config["first_unseen_evaluation_full_coverage_days"]
        ),
        replication_full_coverage_days=int(
            readiness_config["replication_full_coverage_days"]
        ),
        timezone=str(config["collector"]["timezone"]),
        maximum_iopv_staleness_seconds=float(
            quality["maximum_clock_delay_seconds_during_session"]
        ),
        maximum_negative_clock_skew_seconds=float(
            quality["maximum_negative_clock_skew_seconds"]
        ),
        crosscheck_required_from=date.fromisoformat(
            str(quality["crosscheck_required_from"])
        ),
        minimum_final_snapshot_time=str(quality["minimum_final_snapshot_time"]),
        maximum_ohl_price_difference_cny=float(
            quality["maximum_ohl_price_difference_cny"]
        ),
        maximum_close_price_difference_cny=float(
            quality["maximum_close_price_difference_cny"]
        ),
        fund_code=str(config["collector"]["fund_code"]),
    )
    timezone = ZoneInfo(str(config["collector"]["timezone"]))
    payload = {
        "schema_version": "1.2.0",
        "generated_at": datetime.now(timezone).isoformat(),
        "collector_id": config["collector"]["collector_id"],
        **readiness,
        "lineage": {
            "legacy_pcf_rows": int(len(legacy_pcf)),
            "legacy_iopv_rows": int(len(legacy_iopv)),
            "v1_2_pcf_rows": int(len(current_pcf)),
            "v1_2_iopv_rows": int(len(current_iopv)),
            "v1_2_daily_crosscheck_rows": int(len(daily_crosscheck)),
            "v1_2_raw_hash_verified_rows": int(
                daily_crosscheck.get("raw_hash_verified", pd.Series(dtype=bool)).sum()
            ),
            "v1_2_receipt_verified_rows": int(
                daily_crosscheck.get("receipt_verified", pd.Series(dtype=bool)).sum()
            ),
        },
        "governance": {
            "automatic_position_mapping_authorized": False,
            "automatic_ordering_authorized": False,
        },
    }
    output_file = _resolve(config["outputs"]["readiness_report_file"])
    _atomic_json(output_file, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
