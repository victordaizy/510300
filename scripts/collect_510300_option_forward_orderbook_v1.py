"""在冻结收盘窗口内采集510300期权全活跃合约一档盘口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, time as clock_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import tushare as ts
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.option_research_data_acquisition import normalize_contract_master  # noqa: E402
from research.return_tail_supplemental_acquisition import (  # noqa: E402
    normalize_sina_option_quote,
    parse_sina_batch_quotes,
)
from scripts.download_csi300_all_etf_momentum_v1 import credentials  # noqa: E402
from scripts.freeze_510300_option_forward_orderbook_v1 import verify_protocol  # noqa: E402


TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"不可覆盖已存在的前向文件：{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"并发写入检测到已存在的前向文件：{path}")
    temporary.replace(path)


def parse_clock(value: str) -> clock_time:
    return datetime.strptime(value, "%H:%M:%S").time()


def capture_window_valid(now: datetime, expected_date: pd.Timestamp, contract: dict) -> bool:
    local = now.astimezone(TIMEZONE)
    expected = pd.Timestamp(expected_date).date()
    start = parse_clock(contract["capture"]["window_start"])
    end = parse_clock(contract["capture"]["window_end"])
    return local.date() == expected and start <= local.time().replace(tzinfo=None) <= end


def active_contracts(master: pd.DataFrame, expected_date: pd.Timestamp) -> pd.DataFrame:
    date = pd.Timestamp(expected_date).normalize()
    data = master.copy()
    for column in ("list_date", "expiry_date", "delist_date"):
        data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()
    active = data.loc[
        data["list_date"].le(date)
        & (data["delist_date"].isna() | data["delist_date"].ge(date))
    ].copy()
    if active.empty:
        raise RuntimeError("动态主表在预期交易日没有活跃510300期权合约")
    return active.sort_values("contract_code").reset_index(drop=True)


def fetch_dynamic_master(retrieved_at: datetime) -> pd.DataFrame:
    secret, endpoint = credentials()
    ts.set_token(secret)
    errors = []
    fields = (
        "ts_code,opt_code,call_put,exercise_price,maturity_date,list_date,delist_date,"
        "per_unit,opt_multiplier,name"
    )
    for candidate_endpoint in dict.fromkeys([endpoint, "https://tt.xiaodefa.cn", "https://fast.xiaodefa.cn"]):
        api = ts.pro_api()
        api._DataApi__http_url = candidate_endpoint
        for attempt in range(3):
            try:
                raw = api.opt_basic(exchange="SSE", fields=fields)
                return normalize_contract_master(raw, retrieved_at)
            except Exception as exc:
                errors.append(f"{candidate_endpoint}:{type(exc).__name__}")
                if attempt < 2:
                    time.sleep(1.0)
    raise RuntimeError(f"动态期权主表获取失败：{errors}")


def fetch_quotes(codes: list[str], contract: dict, retrieved_at: datetime) -> pd.DataFrame:
    batch_size = int(contract["capture"]["quote_batches_maximum_contracts"])
    timeout = int(contract["capture"]["request_timeout_seconds"])
    retries = int(contract["capture"]["retry_count"])
    session = requests.Session()
    frames = []
    for start in range(0, len(codes), batch_size):
        batch = codes[start : start + batch_size]
        symbols = ",".join(f"CON_OP_{code}" for code in batch)
        last_error: Exception | None = None
        records: dict[str, list[str]] = {}
        for attempt in range(retries):
            try:
                response = session.get(
                    f"https://hq.sinajs.cn/list={symbols}",
                    headers={
                        "Referer": "https://stock.finance.sina.com.cn/",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                records = parse_sina_batch_quotes(response.text)
                if set(records) != set(batch):
                    missing = sorted(set(batch).difference(records))
                    raise RuntimeError(f"批量盘口缺少合约：{missing[:10]}")
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(0.5)
        else:
            raise RuntimeError(f"批量盘口下载失败：{type(last_error).__name__}")
        frames.extend(normalize_sina_option_quote(code, records[code], retrieved_at) for code in batch)
    return pd.concat(frames, ignore_index=True).sort_values("contract_code").reset_index(drop=True)


def validate_snapshot(
    snapshot: pd.DataFrame,
    active: pd.DataFrame,
    expected_date: pd.Timestamp,
    started_at: datetime,
    finished_at: datetime,
    contract: dict,
) -> dict:
    expected_codes = set(active["contract_code"].astype(str))
    actual_codes = set(snapshot["contract_code"].astype(str))
    quote_dates = pd.to_datetime(snapshot["quote_timestamp"], errors="coerce").dt.normalize()
    mode_date = quote_dates.mode().iloc[0] if not quote_dates.mode().empty else pd.NaT
    coverage = len(expected_codes & actual_codes) / max(len(expected_codes), 1)
    positive_bid = float(pd.to_numeric(snapshot["bid1"], errors="coerce").gt(0).mean())
    positive_ask = float(pd.to_numeric(snapshot["ask1"], errors="coerce").gt(0).mean())
    inverted = int(
        (
            pd.to_numeric(snapshot["bid1"], errors="coerce")
            > pd.to_numeric(snapshot["ask1"], errors="coerce")
        ).sum()
    )
    duplicates = int(snapshot["contract_code"].duplicated().sum())
    gates = {
        "capture_started_in_window": capture_window_valid(started_at, expected_date, contract),
        "capture_finished_in_window": capture_window_valid(finished_at, expected_date, contract),
        "active_contract_coverage": coverage >= float(contract["quality"]["active_contract_coverage_minimum"]),
        "no_missing_or_extra_contracts": expected_codes == actual_codes,
        "positive_bid1_ratio": positive_bid >= float(contract["quality"]["positive_bid1_ratio_minimum"]),
        "positive_ask1_ratio": positive_ask >= float(contract["quality"]["positive_ask1_ratio_minimum"]),
        "no_inverted_quotes": inverted <= int(contract["quality"]["maximum_inverted_quote_count"]),
        "no_duplicate_contracts": duplicates <= int(contract["quality"]["maximum_duplicate_key_count"]),
        "quote_mode_date_matches_expected": pd.notna(mode_date) and pd.Timestamp(mode_date) == pd.Timestamp(expected_date).normalize(),
    }
    return {
        "status": "PASS" if all(gates.values()) else "NO_VIEW",
        "expected_contract_count": len(expected_codes),
        "actual_contract_count": len(actual_codes),
        "active_contract_coverage": coverage,
        "positive_bid1_ratio": positive_bid,
        "positive_ask1_ratio": positive_ask,
        "inverted_quote_count": inverted,
        "duplicate_contract_count": duplicates,
        "quote_mode_date": str(pd.Timestamp(mode_date).date()) if pd.notna(mode_date) else None,
        "gates": gates,
    }


def write_attempt(payload: dict, paths: dict, stamp: datetime) -> None:
    attempt_dir = ROOT / paths["attempt_directory"]
    attempt_path = attempt_dir / f"{stamp:%Y%m%d_%H%M%S_%f}.json"
    atomic_json(payload, attempt_path)
    atomic_json(payload, ROOT / paths["latest_collection_status"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集510300期权冻结前向收盘盘口")
    parser.add_argument("--expected-trade-date", required=True, help="预期交易日，YYYY-MM-DD")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract, manifest = verify_protocol()
    expected_date = pd.Timestamp(args.expected_trade_date).normalize()
    paths = contract["paths"]
    now = datetime.now(TIMEZONE)
    base_status = {
        "project_id": contract["protocol"]["project_id"],
        "expected_trade_date": str(expected_date.date()),
        "attempted_at": now.isoformat(),
        "protocol_manifest_sha256": sha256(ROOT / paths["protocol_manifest"]),
        "safety": contract["governance"],
    }
    earliest = pd.Timestamp(contract["protocol"]["earliest_permitted_capture_date"])
    if expected_date < earliest or not capture_window_valid(now, expected_date, contract):
        payload = {
            **base_status,
            "status": "NO_VIEW_CAPTURE_WINDOW",
            "reason": "预期交易日早于冻结边界，或当前墙钟不在14:58:30至15:05:30采集窗口",
        }
        write_attempt(payload, paths, now)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    snapshot_path = ROOT / paths["snapshot_directory"] / f"{expected_date:%Y%m%d}.parquet"
    master_path = ROOT / paths["master_snapshot_directory"] / f"{expected_date:%Y%m%d}.parquet"
    if snapshot_path.exists() and master_path.exists():
        payload = {
            **base_status,
            "status": "EXISTING_IMMUTABLE_SNAPSHOT_REUSED",
            "snapshot": snapshot_path.relative_to(ROOT).as_posix(),
            "snapshot_sha256": sha256(snapshot_path),
            "master_snapshot": master_path.relative_to(ROOT).as_posix(),
            "master_snapshot_sha256": sha256(master_path),
        }
        write_attempt(payload, paths, now)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if snapshot_path.exists() or master_path.exists():
        payload = {
            **base_status,
            "status": "NO_VIEW_ORPHAN_IMMUTABLE_FILE",
            "reason": "盘口与动态主表必须成对存在；孤儿文件禁止覆盖或自动修复",
            "snapshot_exists": snapshot_path.exists(),
            "master_snapshot_exists": master_path.exists(),
        }
        write_attempt(payload, paths, now)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 4
    started_at = datetime.now(TIMEZONE)
    try:
        master = fetch_dynamic_master(started_at)
        active = active_contracts(master, expected_date)
        codes = active["contract_code"].astype(str).str.replace(".SH", "", regex=False).tolist()
        quotes = fetch_quotes(codes, contract, started_at)
        finished_at = datetime.now(TIMEZONE)
        snapshot = quotes.merge(
            active[
                ["contract_code", "option_type", "expiry_date", "strike", "contract_unit", "is_adjusted", "list_date", "delist_date"]
            ],
            on="contract_code",
            how="left",
            validate="one_to_one",
            suffixes=("", "_master"),
        )
        snapshot["expected_trade_date"] = expected_date
        snapshot["capture_started_at"] = started_at.isoformat()
        snapshot["capture_finished_at"] = finished_at.isoformat()
        audit = validate_snapshot(snapshot, active, expected_date, started_at, finished_at, contract)
        payload = {
            **base_status,
            "status": "PASS" if audit["status"] == "PASS" else "NO_VIEW_QUALITY_GATE",
            "capture_started_at": started_at.isoformat(),
            "capture_finished_at": finished_at.isoformat(),
            "audit": audit,
        }
        if audit["status"] != "PASS":
            write_attempt(payload, paths, finished_at)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 3
        atomic_parquet(active, master_path)
        atomic_parquet(snapshot, snapshot_path)
        payload.update(
            {
                "master_snapshot": master_path.relative_to(ROOT).as_posix(),
                "master_snapshot_sha256": sha256(master_path),
                "snapshot": snapshot_path.relative_to(ROOT).as_posix(),
                "snapshot_sha256": sha256(snapshot_path),
            }
        )
        forward_start_path = ROOT / paths["forward_start_status"]
        if not forward_start_path.exists():
            atomic_json(
                {
                    "status": "TRUE_FORWARD_STARTED",
                    "true_forward_start": str(expected_date.date()),
                    "first_snapshot_sha256": payload["snapshot_sha256"],
                    "protocol_frozen_at": manifest["frozen_at"],
                    "protocol_manifest_sha256": payload["protocol_manifest_sha256"],
                    "created_at": finished_at.isoformat(),
                },
                forward_start_path,
            )
        write_attempt(payload, paths, finished_at)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        failed_at = datetime.now(TIMEZONE)
        payload = {
            **base_status,
            "status": "NO_VIEW_COLLECTION_FAILURE",
            "failed_at": failed_at.isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
        write_attempt(payload, paths, failed_at)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
