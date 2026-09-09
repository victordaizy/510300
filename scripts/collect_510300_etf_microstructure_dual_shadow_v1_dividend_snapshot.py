"""采集510300前瞻分红发现快照，并控制其能否进入正式结算。

新浪累计分红序列只用于发现新事件和交叉核对。冻结历史分红表不得改写；新事件
只有在独立的上交所官方核验清单中具备登记日、除息日、发放日、每份金额、官方
原文文件及SHA-256后，才会被标记为可进入前瞻结算。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_V1"
TIMEZONE = ZoneInfo("Asia/Shanghai")
SOURCE_URL = "https://finance.sina.com.cn/realstock/company/sh510300/hfq.js"
CANDIDATE_MANIFEST = (
    PROJECT_ROOT / "config" / "510300_etf_microstructure_dual_shadow_v1_manifest.json"
)
IMPLEMENTATION_RECEIPT = (
    PROJECT_ROOT
    / "reports"
    / "frozen"
    / "510300_etf_microstructure_dual_shadow_v1_prospective_settlement_receipt.json"
)
FROZEN_DISTRIBUTIONS = PROJECT_ROOT / "data" / "reference" / "510300_dividends.csv"
SNAPSHOT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1"
    / "dividend_snapshots"
)
OFFICIAL_VERIFICATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1"
    / "dividend_official_verifications"
)
EXPECTED_FROZEN_KEY = "data\\reference\\510300_dividends.csv"
AMOUNT_TOLERANCE = Decimal("0.000000001")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _immutable_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _immutable_json(payload: dict[str, Any], path: Path) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    ).encode("utf-8")
    _immutable_bytes(encoded, path)


def _immutable_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"临时文件已存在，拒绝覆盖：{temporary}")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    try:
        temporary.rename(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _verify_frozen_distribution() -> tuple[str, str]:
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("project_id") != PROJECT_ID:
        raise ValueError("候选冻结清单项目编号不匹配")
    expected = manifest["historical_input_files_at_freeze"][EXPECTED_FROZEN_KEY]
    actual = _sha256(FROZEN_DISTRIBUTIONS)
    if actual != expected:
        raise ValueError(
            f"冻结分红表哈希漂移：expected={expected},actual={actual}"
        )
    return expected, _sha256(CANDIDATE_MANIFEST)


def _verify_implementation() -> dict[str, str]:
    receipt = json.loads(IMPLEMENTATION_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("project_id") != PROJECT_ID:
        raise ValueError("前瞻结算实现收据项目编号不匹配")
    key = "scripts\\collect_510300_etf_microstructure_dual_shadow_v1_dividend_snapshot.py"
    expected = receipt["implementations"][key]["sha256"]
    actual = _sha256(Path(__file__).resolve())
    if actual != expected:
        raise ValueError(
            f"分红快照采集器实现哈希漂移：expected={expected},actual={actual}"
        )
    return {
        "file": _relative(Path(__file__).resolve()),
        "sha256": actual,
        "receipt": _relative(IMPLEMENTATION_RECEIPT),
        "receipt_sha256": _sha256(IMPLEMENTATION_RECEIPT),
    }


def _parse_sina_payload(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8-sig")
    if "=" not in text:
        raise ValueError("新浪累计分红响应缺少赋值符号")
    payload_text = text.split("=", 1)[1].strip()
    payload, end = json.JSONDecoder().raw_decode(payload_text)
    remainder = payload_text[end:].strip().rstrip(";").strip()
    if remainder and not (remainder.startswith("/*") and remainder.endswith("*/")):
        raise ValueError("新浪累计分红响应在JSON对象后包含未知脚本内容")
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("新浪累计分红响应结构无效")
    rows: list[dict[str, Any]] = []
    for item in payload["data"]:
        ex_date = pd.Timestamp(item["d"]).normalize()
        if ex_date == pd.Timestamp("1900-01-01"):
            continue
        cumulative = Decimal(str(item["u"]))
        rows.append(
            {
                "ex_date": ex_date,
                "cumulative_dividend_per_share_decimal": cumulative,
            }
        )
    if not rows:
        raise ValueError("新浪累计分红响应没有有效事件")
    rows.sort(key=lambda item: item["ex_date"])
    if len({item["ex_date"] for item in rows}) != len(rows):
        raise ValueError("新浪累计分红响应存在重复除息日")
    previous = Decimal("0")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        cumulative = row["cumulative_dividend_per_share_decimal"]
        implied = cumulative - previous
        if implied <= 0:
            raise ValueError(f"累计分红差分非正：{row['ex_date'].date()}")
        normalized.append(
            {
                "ex_date": row["ex_date"],
                "cumulative_dividend_per_share": float(cumulative),
                "cash_dividend_implied": float(implied),
                "cumulative_dividend_decimal": format(cumulative, "f"),
                "cash_dividend_implied_decimal": format(implied, "f"),
            }
        )
        previous = cumulative
    return pd.DataFrame(normalized)


def _compare_with_frozen(
    cumulative: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    frozen = pd.read_csv(FROZEN_DISTRIBUTIONS, dtype=str)
    frozen["ex_date"] = pd.to_datetime(frozen["ex_date"], errors="raise").dt.normalize()
    frozen["cash_dividend_per_share"] = frozen["cash_dividend_per_share"].map(Decimal)
    if frozen["ex_date"].duplicated().any():
        raise ValueError("冻结分红表存在重复除息日")
    source_by_date = {
        pd.Timestamp(row.ex_date): Decimal(str(row.cash_dividend_implied_decimal))
        for row in cumulative.itertuples(index=False)
    }
    frozen_by_date = {
        pd.Timestamp(row.ex_date): row.cash_dividend_per_share
        for row in frozen.itertuples(index=False)
    }
    errors: list[str] = []
    for ex_date, amount in frozen_by_date.items():
        detected = source_by_date.get(ex_date)
        if detected is None:
            errors.append(f"新浪序列缺少冻结事件：{ex_date.date()}")
        elif abs(detected - amount) > AMOUNT_TOLERANCE:
            errors.append(
                f"冻结事件金额不一致：{ex_date.date()},frozen={amount},sina={detected}"
            )
    last_frozen_date = max(frozen_by_date)
    historical_extras = sorted(
        date for date in source_by_date if date not in frozen_by_date and date <= last_frozen_date
    )
    for date in historical_extras:
        errors.append(f"冻结区间内出现额外历史事件：{date.date()}")
    new_events = [
        {
            "ex_date": date.date().isoformat(),
            "cash_dividend_per_share": float(source_by_date[date]),
            "cash_dividend_per_share_decimal": format(source_by_date[date], "f"),
        }
        for date in sorted(source_by_date)
        if date > last_frozen_date
    ]
    tagged = cumulative.copy()
    tagged["is_frozen_event"] = tagged["ex_date"].isin(set(frozen_by_date))
    tagged["is_new_event"] = tagged["ex_date"].gt(last_frozen_date)
    return tagged, new_events, errors


def _resolve_verification_manifest(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else PROJECT_ROOT / path
    if not OFFICIAL_VERIFICATION_ROOT.exists():
        return None
    candidates = sorted(OFFICIAL_VERIFICATION_ROOT.glob("**/manifest.json"))
    return candidates[-1] if candidates else None


def _verify_official_events(
    manifest_path: Path | None, detected_events: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    if manifest_path is None:
        return [], None, ["发现新分红事件，但没有上交所官方核验清单"]
    resolved_manifest = manifest_path.resolve()
    if not resolved_manifest.is_relative_to(OFFICIAL_VERIFICATION_ROOT.resolve()):
        raise ValueError("官方核验清单必须位于前瞻官方核验目录内")
    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if payload.get("project_id") != PROJECT_ID or payload.get("status") != "PASS":
        raise ValueError("官方核验清单项目编号或状态无效")
    supplied = payload.get("verified_new_events")
    if not isinstance(supplied, list):
        raise ValueError("官方核验清单缺少verified_new_events数组")
    detected_map = {
        item["ex_date"]: Decimal(item["cash_dividend_per_share_decimal"])
        for item in detected_events
    }
    verified: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for item in supplied:
        required = {
            "record_date",
            "ex_date",
            "payment_date",
            "cash_dividend_per_share",
            "source_url",
            "source_snapshot_file",
            "source_snapshot_sha256",
        }
        missing = sorted(required - set(item))
        if missing:
            errors.append(f"官方核验事件缺少字段：{missing}")
            continue
        ex_date = pd.Timestamp(item["ex_date"]).normalize()
        record_date = pd.Timestamp(item["record_date"]).normalize()
        payment_date = pd.Timestamp(item["payment_date"]).normalize()
        ex_text = ex_date.date().isoformat()
        amount = Decimal(str(item["cash_dividend_per_share"]))
        if ex_text in seen:
            errors.append(f"官方核验事件重复：{ex_text}")
            continue
        seen.add(ex_text)
        if ex_text not in detected_map:
            errors.append(f"官方核验事件未被新浪新事件发现：{ex_text}")
        elif abs(amount - detected_map[ex_text]) > AMOUNT_TOLERANCE:
            errors.append(f"官方核验金额与发现序列不一致：{ex_text}")
        if not record_date < ex_date <= payment_date:
            errors.append(f"官方核验事件日期顺序无效：{ex_text}")
        host = (urlparse(item["source_url"]).hostname or "").lower()
        if host != "sse.com.cn" and not host.endswith(".sse.com.cn"):
            errors.append(f"官方来源域名不是上交所：{ex_text}")
        source_path = (PROJECT_ROOT / item["source_snapshot_file"]).resolve()
        if not source_path.is_relative_to(resolved_manifest.parent):
            errors.append(f"官方原文文件越出核验清单目录：{ex_text}")
        elif not source_path.exists():
            errors.append(f"官方原文文件缺失：{ex_text}")
        elif _sha256(source_path) != item["source_snapshot_sha256"]:
            errors.append(f"官方原文文件哈希漂移：{ex_text}")
        verified.append(
            {
                "symbol": "510300.SH",
                "record_date": record_date.date().isoformat(),
                "ex_date": ex_text,
                "payment_date": payment_date.date().isoformat(),
                "cash_dividend_per_share": float(amount),
                "source_url": item["source_url"],
                "source_snapshot_file": item["source_snapshot_file"],
                "source_snapshot_sha256": item["source_snapshot_sha256"],
            }
        )
    missing_dates = sorted(set(detected_map) - seen)
    if missing_dates:
        errors.append(f"新事件尚未全部完成官方核验：{missing_dates}")
    audit = {
        "manifest": _relative(resolved_manifest),
        "manifest_sha256": _sha256(resolved_manifest),
        "verified_event_count": len(verified),
    }
    return verified, audit, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="采集510300前瞻分红发现快照")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--official-verification-manifest", default=None)
    arguments = parser.parse_args()
    now = datetime.now(TIMEZONE)
    run_id = arguments.run_id or now.strftime("%Y%m%dT%H%M%S%z")
    output_dir = SNAPSHOT_ROOT / run_id
    if output_dir.exists():
        raise FileExistsError(f"分红快照目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    implementation_audit = _verify_implementation()
    expected_frozen_hash, candidate_manifest_hash = _verify_frozen_distribution()
    response = requests.get(
        SOURCE_URL,
        timeout=45,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    response.raise_for_status()
    raw_path = output_dir / "sina_sh510300_hfq.js"
    _immutable_bytes(response.content, raw_path)
    normalized = _parse_sina_payload(response.content)
    tagged, new_events, comparison_errors = _compare_with_frozen(normalized)
    normalized_path = output_dir / "sina_dividend_events.parquet"
    _immutable_parquet(tagged, normalized_path)

    verification_path = _resolve_verification_manifest(
        arguments.official_verification_manifest
    )
    verified_events: list[dict[str, Any]] = []
    official_audit: dict[str, Any] | None = None
    official_errors: list[str] = []
    if new_events and not comparison_errors:
        verified_events, official_audit, official_errors = _verify_official_events(
            verification_path, new_events
        )

    if comparison_errors:
        status = "FAIL_HISTORICAL_DIVIDEND_DRIFT"
        allowed = False
    elif not new_events:
        status = "PASS_NO_NEW_EVENT"
        allowed = True
    elif official_errors:
        status = "NEW_EVENT_DETECTED_REQUIRES_OFFICIAL_VERIFICATION"
        allowed = False
    else:
        status = "PASS_VERIFIED_NEW_EVENTS"
        allowed = True

    manifest = {
        "schema_version": 1,
        "project_id": PROJECT_ID,
        "status": status,
        "settlement_qualification_allowed": allowed,
        "run_id": run_id,
        "retrieved_at_asia_shanghai": now.isoformat(),
        "implementation": implementation_audit,
        "source": {
            "role": "secondary_detection_only",
            "url": SOURCE_URL,
            "http_status": response.status_code,
            "content_type": response.headers.get("Content-Type"),
            "raw_file": _relative(raw_path),
            "raw_sha256": _sha256(raw_path),
            "raw_bytes": raw_path.stat().st_size,
            "normalized_file": _relative(normalized_path),
            "normalized_sha256": _sha256(normalized_path),
            "normalized_event_count": int(len(tagged)),
        },
        "frozen_history": {
            "file": _relative(FROZEN_DISTRIBUTIONS),
            "expected_sha256": expected_frozen_hash,
            "actual_sha256": _sha256(FROZEN_DISTRIBUTIONS),
            "candidate_manifest": _relative(CANDIDATE_MANIFEST),
            "candidate_manifest_sha256": candidate_manifest_hash,
            "historical_match": not comparison_errors,
        },
        "detected_new_events": new_events,
        "verified_new_events": verified_events,
        "official_verification": official_audit,
        "errors": comparison_errors + official_errors,
        "accounting_contract": {
            "detection_source_may_not_enter_settlement_alone": True,
            "new_event_requires_sse_original_snapshot": True,
            "required_fields": [
                "record_date",
                "ex_date",
                "payment_date",
                "cash_dividend_per_share",
            ],
            "frozen_history_is_never_rewritten": True,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _immutable_json(manifest, manifest_path)
    result = {
        **manifest,
        "snapshot_manifest": _relative(manifest_path),
        "snapshot_manifest_sha256": _sha256(manifest_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if status == "FAIL_HISTORICAL_DIVIDEND_DRIFT":
        return 1
    if status == "NEW_EVENT_DETECTED_REQUIRES_OFFICIAL_VERIFICATION":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
