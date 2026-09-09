from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from akshare.stock.stock_zh_a_sina import (
    hk_js_decode,
    py_mini_racer,
    zh_sina_a_stock_hist_url,
)
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_a_share_hs_cash_option_floor_alpha_v1 import (  # noqa: E402
    load_config,
    verify_protocol,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
TERMS_PATH = (
    ROOT
    / "data/curated/a_share_hs_official_cash_option_rights_v1/cash_option_terms_v1.parquet"
)
RAW_ROOT = ROOT / "data/raw/a_share_hs_official_cash_option_floor_alpha_v1"
TUSHARE_DAILY_ROOT = RAW_ROOT / "tushare_daily"
TUSHARE_CALENDAR_ROOT = RAW_ROOT / "tushare_trade_cal"
TUSHARE_IDENTITY_ROOT = RAW_ROOT / "tushare_stock_basic"
SINA_HISTORY_ROOT = RAW_ROOT / "sina_history"
STAGING_ROOT = ROOT / "data/staging/a_share_hs_official_cash_option_floor_alpha_v1"
SUBJECT_DAILY_PATH = STAGING_ROOT / "subject_daily_v1.parquet"
CALENDAR_PATH = STAGING_ROOT / "trade_calendar_v1.parquet"
IDENTITY_PATH = STAGING_ROOT / "security_identity_v1.parquet"
RECEIPT_PATH = STAGING_ROOT / "market_acquisition_receipt.json"
ATTEMPT_ROOT = STAGING_ROOT / "acquisition_attempts"
MARKET_ARCHIVE_PATH = (
    ROOT / "config/a_share_hs_official_cash_option_floor_alpha_v1_market_archive_manifest.json"
)
TUSHARE_FIELDS = "ts_code,trade_date,pre_close,open,high,low,close,pct_chg,vol,amount"
STOCK_BASIC_FIELDS = (
    "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date"
)
ALLOWED_ENDPOINTS = {
    "https://api.tushare.pro",
    "https://fast.xiaodefa.cn",
    "https://tt.xiaodefa.cn",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_text(path: Path, content: str) -> None:
    atomic_bytes(path, content.encode("utf-8"))


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def sanitize_error(error: BaseException, secret: str) -> str:
    message = f"{type(error).__name__}: {error}"
    if secret:
        message = message.replace(secret, "[REDACTED_TOKEN]")
    return message[:1000]


def credentials() -> tuple[str, list[str], str]:
    load_dotenv(ROOT / ".env")
    standard = (os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN") or "").strip()
    proxy = (os.getenv("TUSHARE_PROXY_TOKEN") or "").strip()
    if standard:
        endpoint = os.getenv("TUSHARE_API_URL", "https://api.tushare.pro").strip().rstrip("/")
        endpoints = [endpoint]
        kind = "STANDARD_TUSHARE_TOKEN"
        secret = standard
    elif proxy:
        endpoint = os.getenv("TUSHARE_PROXY_URL", "https://fast.xiaodefa.cn").strip().rstrip("/")
        endpoints = [endpoint]
        for alternate in ("https://fast.xiaodefa.cn", "https://tt.xiaodefa.cn"):
            if alternate not in endpoints:
                endpoints.append(alternate)
        kind = "TUSHARE_PROXY_TOKEN"
        secret = proxy
    else:
        raise RuntimeError("未设置 TUSHARE_TOKEN、TS_TOKEN 或 TUSHARE_PROXY_TOKEN")
    invalid = [value for value in endpoints if value not in ALLOWED_ENDPOINTS]
    if invalid:
        raise RuntimeError(f"TuShare 接口不在冻结允许名单：{invalid}")
    return secret, endpoints, kind


def make_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/136.0 Safari/537.36"
            )
        }
    )
    return session


def parse_tushare_payload(content: bytes, expected_api_name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - 原始响应验证
        raise RuntimeError("TuShare 响应不是 UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("TuShare 响应顶层不是对象")
    try:
        code = int(payload.get("code"))
    except (TypeError, ValueError) as error:
        raise RuntimeError("TuShare 响应缺少数值 code") from error
    if code != 0:
        message = str(payload.get("msg") or "")[:500]
        raise RuntimeError(f"TuShare {expected_api_name} 返回 code={code}，msg={message}")
    data = payload.get("data") or {}
    fields = data.get("fields") or []
    items = data.get("items") or []
    if not isinstance(fields, list) or not isinstance(items, list):
        raise RuntimeError(f"TuShare {expected_api_name} data.fields/items 类型非法")
    if any(not isinstance(row, list) or len(row) != len(fields) for row in items):
        raise RuntimeError(f"TuShare {expected_api_name} 行宽与字段数不一致")
    return payload, pd.DataFrame(items, columns=[str(value) for value in fields])


def fetch_tushare_raw(
    *,
    api_name: str,
    params: dict[str, Any],
    fields: str,
    raw_path: Path,
    secret: str,
    endpoints: list[str],
) -> tuple[bytes, str, bool]:
    if raw_path.is_file() and raw_path.stat().st_size > 0:
        content = raw_path.read_bytes()
        parse_tushare_payload(content, api_name)
        return content, "IMMUTABLE_CACHE", True
    payload = {
        "api_name": api_name,
        "token": secret,
        "params": params,
        "fields": fields,
    }
    errors: list[str] = []
    for endpoint in endpoints:
        session = make_session()
        try:
            response = session.post(endpoint, json=payload, timeout=(12, 45))
            response.raise_for_status()
            content = response.content
            if not content:
                raise RuntimeError("服务器返回空响应")
            parse_tushare_payload(content, api_name)
            if secret.encode("utf-8") in content:
                raise RuntimeError("服务器响应意外包含访问令牌，拒绝落盘")
            atomic_bytes(raw_path, content)
            return content, endpoint, False
        except Exception as error:  # noqa: BLE001 - 跨节点重试并保留脱敏错误
            errors.append(f"{endpoint}: {sanitize_error(error, secret)}")
        finally:
            session.close()
        time.sleep(1.0)
    raise RuntimeError(f"TuShare {api_name} 所有审核节点失败：{errors}")


def normalize_tushare_daily(
    frame: pd.DataFrame,
    *,
    event_id: str,
    ts_code: str,
    query_start: str,
    query_end: str,
    raw_path: Path,
    endpoint: str,
) -> pd.DataFrame:
    required = {
        "ts_code",
        "trade_date",
        "pre_close",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    }
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "ts_code",
                "date",
                "pre_close",
                "raw_open",
                "raw_high",
                "raw_low",
                "raw_close",
                "vol",
                "volume",
                "amount",
                "price_source",
                "source_endpoint",
                "raw_path",
                "query_start",
                "query_end",
            ]
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"TuShare daily 缺字段：{missing}")
    data = frame.copy()
    if not data["ts_code"].astype(str).eq(ts_code).all():
        raise RuntimeError(f"TuShare daily 返回了错误证券：{event_id}")
    data = data.rename(
        columns={
            "trade_date": "date",
            "open": "raw_open",
            "high": "raw_high",
            "low": "raw_low",
            "close": "raw_close",
        }
    )
    data["date"] = pd.to_datetime(data["date"], format="%Y%m%d", errors="raise").dt.normalize()
    numeric = ["pre_close", "raw_open", "raw_high", "raw_low", "raw_close", "vol", "amount"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    if data[numeric].isna().any().any():
        raise RuntimeError(f"TuShare daily 存在无法解析数值：{event_id}")
    data["volume"] = data["vol"] * 100.0
    data["event_id"] = event_id
    data["price_source"] = "TUSHARE_PRO_DAILY_RAW_UNADJUSTED"
    data["source_endpoint"] = endpoint
    data["raw_path"] = relative(raw_path)
    data["query_start"] = query_start
    data["query_end"] = query_end
    keep = [
        "event_id",
        "ts_code",
        "date",
        "pre_close",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "vol",
        "volume",
        "amount",
        "price_source",
        "source_endpoint",
        "raw_path",
        "query_start",
        "query_end",
    ]
    return data[keep].sort_values("date", kind="stable").drop_duplicates("date", keep="last")


def ts_code_to_sina(ts_code: str) -> str:
    code, exchange = str(ts_code).split(".", 1)
    if exchange not in {"SH", "SZ"}:
        raise RuntimeError(f"不支持的交易所：{ts_code}")
    return ("sh" if exchange == "SH" else "sz") + code.zfill(6)


def fetch_sina_raw(ts_code: str, raw_path: Path) -> tuple[bytes, bool]:
    if raw_path.is_file() and raw_path.stat().st_size > 0:
        return raw_path.read_bytes(), True
    url = zh_sina_a_stock_hist_url.format(ts_code_to_sina(ts_code))
    session = make_session()
    session.headers.update({"Referer": "https://finance.sina.com.cn/"})
    error: Exception | None = None
    try:
        for attempt in range(4):
            try:
                response = session.get(url, timeout=(12, 45))
                response.raise_for_status()
                content = response.content
                if not content:
                    raise RuntimeError("新浪返回空响应")
                atomic_bytes(raw_path, content)
                return content, False
            except Exception as current:  # noqa: BLE001 - 有限重试
                error = current
                time.sleep(1.5 * (attempt + 1))
    finally:
        session.close()
    raise RuntimeError(f"新浪历史行情下载失败：{type(error).__name__}: {error}")


def parse_sina_history(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8", errors="strict")
    if "=" not in text:
        raise RuntimeError("新浪历史行情响应缺少赋值符")
    encoded = text.split("=", 1)[1].split(";", 1)[0].replace('"', "").strip()
    if not encoded or encoded in {"null", "NULL"}:
        return pd.DataFrame()
    runtime = py_mini_racer.MiniRacer()
    runtime.eval(hk_js_decode)
    rows = runtime.call("d", encoded)
    frame = pd.DataFrame(rows)
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"新浪解码历史缺字段：{missing}")
    frame["date"] = (
        pd.to_datetime(frame["date"], errors="raise", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if frame[numeric].isna().any().any():
        raise RuntimeError("新浪解码历史核心数值无法解析")
    previous = frame["close"].shift(1)
    if "prevclose" in frame.columns:
        previous = previous.fillna(pd.to_numeric(frame["prevclose"], errors="coerce"))
    frame["pre_close"] = previous
    return frame.sort_values("date", kind="stable").drop_duplicates("date", keep="last")


def normalize_sina_daily(
    full_history: pd.DataFrame,
    *,
    event_id: str,
    ts_code: str,
    query_start: str,
    query_end: str,
    raw_path: Path,
) -> pd.DataFrame:
    columns = [
        "event_id",
        "ts_code",
        "date",
        "pre_close",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "vol",
        "volume",
        "amount",
        "price_source",
        "source_endpoint",
        "raw_path",
        "query_start",
        "query_end",
    ]
    if full_history.empty:
        return pd.DataFrame(columns=columns)
    data = full_history.loc[
        full_history["date"].between(pd.Timestamp(query_start), pd.Timestamp(query_end))
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    data = data.rename(
        columns={
            "open": "raw_open",
            "high": "raw_high",
            "low": "raw_low",
            "close": "raw_close",
        }
    )
    data["event_id"] = event_id
    data["ts_code"] = ts_code
    data["vol"] = data["volume"] / 100.0
    data["price_source"] = "SINA_FINANCE_HISTDATA_KLC2_RAW_UNADJUSTED"
    data["source_endpoint"] = "SINA_HISTDATA_KLC2"
    data["raw_path"] = relative(raw_path)
    data["query_start"] = query_start
    data["query_end"] = query_end
    if data[["pre_close", "raw_open", "raw_high", "raw_low", "raw_close"]].isna().any().any():
        raise RuntimeError(f"新浪窗口存在缺失前收或 OHLC：{event_id}")
    return data[columns].sort_values("date", kind="stable").drop_duplicates("date", keep="last")


def normalize_trade_calendar(frame: pd.DataFrame, exchange: str, raw_path: Path) -> pd.DataFrame:
    required = {"exchange", "cal_date", "is_open", "pretrade_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"trade_cal 缺字段：{missing}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["cal_date"], format="%Y%m%d", errors="raise").dt.normalize()
    data["is_open"] = pd.to_numeric(data["is_open"], errors="raise").astype(int).astype(bool)
    data["exchange_query"] = exchange
    data["source"] = "TUSHARE_PRO_TRADE_CAL"
    data["raw_path"] = relative(raw_path)
    return data[["exchange_query", "date", "is_open", "pretrade_date", "source", "raw_path"]].sort_values("date")


def normalize_identity(frames: list[pd.DataFrame], raw_paths: list[Path]) -> pd.DataFrame:
    nonempty = []
    for frame, path in zip(frames, raw_paths, strict=True):
        if frame.empty:
            continue
        data = frame.copy()
        data["raw_path"] = relative(path)
        nonempty.append(data)
    if not nonempty:
        return pd.DataFrame(columns=[*STOCK_BASIC_FIELDS.split(","), "raw_path"])
    result = pd.concat(nonempty, ignore_index=True)
    result["list_date"] = pd.to_datetime(result["list_date"], format="%Y%m%d", errors="coerce")
    result["delist_date"] = pd.to_datetime(result["delist_date"], format="%Y%m%d", errors="coerce")
    return result.sort_values(["ts_code", "list_status"], kind="stable").drop_duplicates(
        ["ts_code", "list_status"], keep="first"
    )


def attempt_receipt(payload: dict[str, Any]) -> Path:
    ATTEMPT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(TIMEZONE).strftime("%Y%m%dT%H%M%S%f%z")
    path = ATTEMPT_ROOT / f"attempt_{stamp}.json"
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    return path


def acquire() -> tuple[dict[str, Any], int]:
    config = load_config()
    protocol = verify_protocol(config)
    for output in (SUBJECT_DAILY_PATH, CALENDAR_PATH, IDENTITY_PATH, RECEIPT_PATH, MARKET_ARCHIVE_PATH):
        if output.exists():
            raise RuntimeError(f"市场采集正式产物已存在，禁止覆盖：{relative(output)}")
    terms = pd.read_parquet(TERMS_PATH).sort_values(
        ["effective_announcement_date", "ts_code", "event_id"], kind="stable"
    )
    if len(terms) != 21 or terms["event_id"].astype(str).duplicated().any():
        raise RuntimeError("冻结条款事件集不是21个唯一事件")
    secret, endpoints, credential_kind = credentials()
    started_at = datetime.now(TIMEZONE)
    raw_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    primary_frames: list[pd.DataFrame] = []
    secondary_frames: list[pd.DataFrame] = []
    event_summaries: list[dict[str, Any]] = []

    for row in terms.itertuples(index=False):
        event_id = str(row.event_id)
        ts_code = str(row.ts_code)
        query_start_date = pd.Timestamp(row.effective_announcement_date).date() + timedelta(days=1)
        query_end_date = pd.Timestamp(row.rights_registration_date).date()
        query_start = query_start_date.isoformat()
        query_end = query_end_date.isoformat()
        raw_path = TUSHARE_DAILY_ROOT / f"{safe_name(event_id)}.json"
        try:
            content, endpoint, reused = fetch_tushare_raw(
                api_name="daily",
                params={
                    "ts_code": ts_code,
                    "start_date": query_start_date.strftime("%Y%m%d"),
                    "end_date": query_end_date.strftime("%Y%m%d"),
                },
                fields=TUSHARE_FIELDS,
                raw_path=raw_path,
                secret=secret,
                endpoints=endpoints,
            )
            _, frame = parse_tushare_payload(content, "daily")
            primary = normalize_tushare_daily(
                frame,
                event_id=event_id,
                ts_code=ts_code,
                query_start=query_start,
                query_end=query_end,
                raw_path=raw_path,
                endpoint=endpoint,
            )
            primary_frames.append(primary)
            raw_records.append(
                {
                    "source": "TUSHARE_PRO_DAILY_RAW_UNADJUSTED",
                    "event_id": event_id,
                    "path": relative(raw_path),
                    "sha256": sha256_file(raw_path),
                    "bytes": raw_path.stat().st_size,
                    "endpoint": endpoint,
                    "cache_reused": reused,
                    "query_start": query_start,
                    "query_end": query_end,
                    "row_count": len(primary),
                }
            )
        except Exception as error:  # noqa: BLE001 - 收集全事件失败后统一停机
            failures.append(
                {
                    "source": "TUSHARE_PRO_DAILY_RAW_UNADJUSTED",
                    "event_id": event_id,
                    "error": sanitize_error(error, secret),
                }
            )
            primary = pd.DataFrame()

        sina_path = SINA_HISTORY_ROOT / f"{safe_name(ts_code)}.js"
        try:
            sina_content, sina_reused = fetch_sina_raw(ts_code, sina_path)
            full_history = parse_sina_history(sina_content)
            secondary = normalize_sina_daily(
                full_history,
                event_id=event_id,
                ts_code=ts_code,
                query_start=query_start,
                query_end=query_end,
                raw_path=sina_path,
            )
            secondary_frames.append(secondary)
            raw_records.append(
                {
                    "source": "SINA_FINANCE_HISTDATA_KLC2_RAW_UNADJUSTED",
                    "event_id": event_id,
                    "path": relative(sina_path),
                    "sha256": sha256_file(sina_path),
                    "bytes": sina_path.stat().st_size,
                    "endpoint": "SINA_HISTDATA_KLC2",
                    "cache_reused": sina_reused,
                    "query_start": query_start,
                    "query_end": query_end,
                    "materialized_window_row_count": len(secondary),
                    "decoded_full_history_row_count": len(full_history),
                }
            )
        except Exception as error:  # noqa: BLE001 - 收集全事件失败后统一停机
            failures.append(
                {
                    "source": "SINA_FINANCE_HISTDATA_KLC2_RAW_UNADJUSTED",
                    "event_id": event_id,
                    "error": sanitize_error(error, secret),
                }
            )
            secondary = pd.DataFrame()
        event_summaries.append(
            {
                "event_id": event_id,
                "ts_code": ts_code,
                "query_start": query_start,
                "query_end": query_end,
                "primary_row_count": len(primary),
                "secondary_row_count": len(secondary),
            }
        )
        time.sleep(max(float(os.getenv("TUSHARE_REQUEST_INTERVAL_SECONDS", "0.8")), 0.5))

    earliest = (
        pd.to_datetime(terms["effective_announcement_date"]).min().date() + timedelta(days=1)
    ).strftime("%Y%m%d")
    latest = pd.to_datetime(terms["rights_registration_date"]).max().date().strftime("%Y%m%d")
    calendar_frames: list[pd.DataFrame] = []
    for exchange in ("SSE", "SZSE"):
        raw_path = TUSHARE_CALENDAR_ROOT / f"{exchange}_{earliest}_{latest}.json"
        try:
            content, endpoint, reused = fetch_tushare_raw(
                api_name="trade_cal",
                params={"exchange": exchange, "start_date": earliest, "end_date": latest},
                fields="exchange,cal_date,is_open,pretrade_date",
                raw_path=raw_path,
                secret=secret,
                endpoints=endpoints,
            )
            _, frame = parse_tushare_payload(content, "trade_cal")
            normalized = normalize_trade_calendar(frame, exchange, raw_path)
            calendar_frames.append(normalized)
            raw_records.append(
                {
                    "source": "TUSHARE_PRO_TRADE_CAL",
                    "exchange": exchange,
                    "path": relative(raw_path),
                    "sha256": sha256_file(raw_path),
                    "bytes": raw_path.stat().st_size,
                    "endpoint": endpoint,
                    "cache_reused": reused,
                    "row_count": len(normalized),
                }
            )
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "source": "TUSHARE_PRO_TRADE_CAL",
                    "exchange": exchange,
                    "error": sanitize_error(error, secret),
                }
            )

    identity_frames: list[pd.DataFrame] = []
    identity_raw_paths: list[Path] = []
    for list_status in ("L", "D", "P"):
        raw_path = TUSHARE_IDENTITY_ROOT / f"stock_basic_{list_status}.json"
        try:
            content, endpoint, reused = fetch_tushare_raw(
                api_name="stock_basic",
                params={"exchange": "", "list_status": list_status},
                fields=STOCK_BASIC_FIELDS,
                raw_path=raw_path,
                secret=secret,
                endpoints=endpoints,
            )
            _, frame = parse_tushare_payload(content, "stock_basic")
            identity_frames.append(frame)
            identity_raw_paths.append(raw_path)
            raw_records.append(
                {
                    "source": "TUSHARE_PRO_STOCK_BASIC",
                    "list_status": list_status,
                    "path": relative(raw_path),
                    "sha256": sha256_file(raw_path),
                    "bytes": raw_path.stat().st_size,
                    "endpoint": endpoint,
                    "cache_reused": reused,
                    "row_count": len(frame),
                }
            )
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "source": "TUSHARE_PRO_STOCK_BASIC",
                    "list_status": list_status,
                    "error": sanitize_error(error, secret),
                }
            )

    attempt = {
        "protocol_id": config["protocol_id"],
        "status": "ACQUISITION_ATTEMPT_COMPLETE" if not failures else "ACQUISITION_ATTEMPT_NETWORK_OR_PARSE_FAILURE",
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(TIMEZONE).isoformat(),
        "credential_kind": credential_kind,
        "credential_value_logged_or_persisted": False,
        "event_count": len(terms),
        "raw_file_count": len(raw_records),
        "failure_count": len(failures),
        "failures": failures,
        "market_price_value_read_after_protocol_freeze": True,
        "future_subject_values_used_for_screening": False,
    }
    attempt_path = attempt_receipt(attempt)
    if failures:
        result = {
            **attempt,
            "attempt_receipt": relative(attempt_path),
            "formal_market_receipt_created": False,
        }
        return result, 2

    subject = pd.concat([*primary_frames, *secondary_frames], ignore_index=True)
    subject = subject.sort_values(["event_id", "price_source", "date"], kind="stable").reset_index(drop=True)
    if subject.duplicated(["event_id", "price_source", "date"]).any():
        raise RuntimeError("物化标的日线存在事件-来源-日期重复")
    calendar = pd.concat(calendar_frames, ignore_index=True).sort_values(
        ["exchange_query", "date"], kind="stable"
    )
    identity_all = normalize_identity(identity_frames, identity_raw_paths)
    target_codes = set(terms["ts_code"].astype(str))
    identity = identity_all.loc[identity_all["ts_code"].astype(str).isin(target_codes)].copy()
    identity_codes = set(identity["ts_code"].astype(str))
    identity_missing = sorted(target_codes - identity_codes)

    atomic_parquet(SUBJECT_DAILY_PATH, subject)
    atomic_parquet(CALENDAR_PATH, calendar)
    atomic_parquet(IDENTITY_PATH, identity)
    raw_records = sorted(
        raw_records,
        key=lambda item: (
            str(item.get("source") or ""),
            str(item.get("event_id") or item.get("exchange") or item.get("list_status") or ""),
        ),
    )
    receipt = {
        "protocol_id": config["protocol_id"],
        "status": "PASS_CASH_OPTION_FLOOR_ALPHA_MARKET_INPUTS_ACQUIRED",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "evidence_cutoff": config["evidence_cutoff"],
        "protocol_verification": protocol,
        "credential_kind": credential_kind,
        "credential_value_logged_or_persisted": False,
        "event_count": len(terms),
        "event_source_summary": event_summaries,
        "raw_files": raw_records,
        "subject_daily": relative(SUBJECT_DAILY_PATH),
        "subject_daily_sha256": sha256_file(SUBJECT_DAILY_PATH),
        "subject_daily_rows": len(subject),
        "trade_calendar": relative(CALENDAR_PATH),
        "trade_calendar_sha256": sha256_file(CALENDAR_PATH),
        "trade_calendar_rows": len(calendar),
        "security_identity": relative(IDENTITY_PATH),
        "security_identity_sha256": sha256_file(IDENTITY_PATH),
        "security_identity_rows": len(identity),
        "security_identity_missing_codes": identity_missing,
        "attempt_receipt": relative(attempt_path),
        "attempt_receipt_sha256": sha256_file(attempt_path),
        "tushare_primary_queries_restricted_to_entry_windows": True,
        "sina_full_history_raw_response_required_by_endpoint": True,
        "sina_full_history_rows_decoded_to_isolate_entry_windows": True,
        "only_entry_window_rows_materialized": True,
        "future_subject_values_used_for_screening": False,
        "market_price_value_read_after_protocol_freeze": True,
        "price_screen_completed": False,
        "liquidity_filter_applied": False,
        "position_mapping": False,
        "order_generation": False,
        "shadow_authorized": False,
        "live_authorized": False,
    }
    atomic_text(RECEIPT_PATH, json.dumps(receipt, ensure_ascii=False, indent=2, default=str) + "\n")
    tracked = [
        ROOT / str(config["artifacts"]["protocol_manifest"]),
        TERMS_PATH,
        SUBJECT_DAILY_PATH,
        CALENDAR_PATH,
        IDENTITY_PATH,
        RECEIPT_PATH,
        attempt_path,
        *[ROOT / str(item["path"]) for item in raw_records],
    ]
    files = {relative(path): sha256_file(path) for path in tracked}
    archive = {
        "protocol_id": f"{config['protocol_id']}_MARKET_INPUT_ARCHIVE_FREEZE",
        "status": "FROZEN_CASH_OPTION_FLOOR_ALPHA_MARKET_INPUTS_BEFORE_PRICE_SCREEN",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "files": files,
        "content_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "event_count": len(terms),
        "raw_file_count": len(raw_records),
        "subject_daily_rows": len(subject),
        "market_price_value_read_after_protocol_freeze": True,
        "price_screen_completed": False,
        "future_subject_values_used_for_screening": False,
    }
    atomic_text(
        MARKET_ARCHIVE_PATH,
        json.dumps(archive, ensure_ascii=False, indent=2, default=str) + "\n",
    )
    receipt["market_archive_manifest"] = relative(MARKET_ARCHIVE_PATH)
    receipt["market_archive_manifest_sha256"] = sha256_file(MARKET_ARCHIVE_PATH)
    return receipt, 0


def verify_acquisition() -> dict[str, Any]:
    config = load_config()
    protocol = verify_protocol(config)
    if not RECEIPT_PATH.exists() or not MARKET_ARCHIVE_PATH.exists():
        raise RuntimeError("正式市场采集回执或冻结清单不存在")
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    archive = json.loads(MARKET_ARCHIVE_PATH.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    if receipt.get("status") != "PASS_CASH_OPTION_FLOOR_ALPHA_MARKET_INPUTS_ACQUIRED":
        failures.append({"failure": "RECEIPT_STATUS", "actual": receipt.get("status")})
    if archive.get("status") != "FROZEN_CASH_OPTION_FLOOR_ALPHA_MARKET_INPUTS_BEFORE_PRICE_SCREEN":
        failures.append({"failure": "ARCHIVE_STATUS", "actual": archive.get("status")})
    for item_path, expected_sha in (archive.get("files") or {}).items():
        path = ROOT / str(item_path)
        actual_sha = sha256_file(path) if path.exists() else None
        if actual_sha != expected_sha:
            failures.append(
                {
                    "failure": "FILE_HASH",
                    "path": item_path,
                    "expected": expected_sha,
                    "actual": actual_sha,
                }
            )
    subject = pd.read_parquet(SUBJECT_DAILY_PATH)
    calendar = pd.read_parquet(CALENDAR_PATH)
    identity = pd.read_parquet(IDENTITY_PATH)
    if sha256_file(SUBJECT_DAILY_PATH) != receipt.get("subject_daily_sha256"):
        failures.append({"failure": "SUBJECT_DAILY_RECEIPT_HASH"})
    if sha256_file(CALENDAR_PATH) != receipt.get("trade_calendar_sha256"):
        failures.append({"failure": "CALENDAR_RECEIPT_HASH"})
    if sha256_file(IDENTITY_PATH) != receipt.get("security_identity_sha256"):
        failures.append({"failure": "IDENTITY_RECEIPT_HASH"})
    result = {
        "status": (
            "PASS_CASH_OPTION_FLOOR_ALPHA_MARKET_INPUT_ARCHIVE_VERIFIED"
            if not failures
            else "FAILED_CASH_OPTION_FLOOR_ALPHA_MARKET_INPUT_ARCHIVE_VERIFICATION"
        ),
        "failure_count": len(failures),
        "failures": failures,
        "protocol_verification": protocol,
        "market_archive_manifest": relative(MARKET_ARCHIVE_PATH),
        "market_archive_manifest_sha256": sha256_file(MARKET_ARCHIVE_PATH),
        "event_count": int(receipt["event_count"]),
        "subject_daily_rows": len(subject),
        "trade_calendar_rows": len(calendar),
        "security_identity_rows": len(identity),
        "raw_file_count": len(receipt["raw_files"]),
        "market_price_value_read_after_protocol_freeze": True,
        "price_screen_completed": False,
        "future_subject_values_used_for_screening": False,
    }
    if failures:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, default=str))
    return result


def status() -> dict[str, Any]:
    config = load_config()
    protocol = verify_protocol(config)
    secret, endpoints, credential_kind = credentials()
    return {
        "status": "READY_TO_ACQUIRE_CASH_OPTION_FLOOR_ALPHA_MARKET_INPUTS",
        "protocol_verification": protocol,
        "terms_path": relative(TERMS_PATH),
        "terms_event_count_from_parquet_metadata": int(pd.read_parquet(TERMS_PATH, columns=["event_id"]).shape[0]),
        "credential_kind": credential_kind,
        "credential_present": bool(secret),
        "credential_value_logged_or_persisted": False,
        "allowed_endpoint_count": len(endpoints),
        "formal_receipt_exists": RECEIPT_PATH.exists(),
        "market_archive_exists": MARKET_ARCHIVE_PATH.exists(),
        "market_price_value_read_by_this_mode": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按冻结契约采集21个现金选择权事件的公告后入场窗口双源未复权日线"
    )
    parser.add_argument("--mode", choices=("status", "acquire", "verify"), default="status")
    args = parser.parse_args()
    if args.mode == "status":
        result = status()
        exit_code = 0
    elif args.mode == "acquire":
        result, exit_code = acquire()
    else:
        result = verify_acquisition()
        exit_code = 0
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
