from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import baostock as bs
import numpy as np
import pandas as pd
import requests
from akshare.stock.stock_zh_a_sina import zh_sina_a_stock_hist_url
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acquire_a_share_hs_cash_option_floor_alpha_v1_market_v1 import (  # noqa: E402
    parse_sina_history,
    safe_name,
)
from scripts.freeze_a_share_hs_cash_option_floor_alpha_v1 import (  # noqa: E402
    load_config as load_parent_config,
)
from scripts.freeze_a_share_hs_cash_option_floor_alpha_v1_0_1_source_remediation import (  # noqa: E402
    load_config as load_remediation_config,
    verify_protocol as verify_remediation_protocol,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
TERMS_PATH = (
    ROOT
    / "data/curated/a_share_hs_official_cash_option_rights_v1/cash_option_terms_v1.parquet"
)
PARENT_SINA_ROOT = ROOT / "data/raw/a_share_hs_official_cash_option_floor_alpha_v1/sina_history"
RAW_ROOT = ROOT / "data/raw/a_share_hs_official_cash_option_floor_alpha_v1_0_1"
EASTMONEY_SUBJECT_ROOT = RAW_ROOT / "eastmoney/subjects"
EASTMONEY_INDEX_ROOT = RAW_ROOT / "eastmoney/index"
SINA_INDEX_ROOT = RAW_ROOT / "sina_index"
BAOSTOCK_ROOT = RAW_ROOT / "baostock"
BAOSTOCK_CALENDAR_PATH = BAOSTOCK_ROOT / "trade_dates_20060923_20260206.csv"
BAOSTOCK_RECEIPT_PATH = BAOSTOCK_ROOT / "trade_dates_query_receipt.json"
FROZEN_2014_CALENDAR_PATH = (
    ROOT
    / "data/staging/a_share_hs_concentrated_low_risk_trend_v1_1/"
    "trading_calendar_observed_open_days.parquet"
)
STAGING_ROOT = ROOT / "data/staging/a_share_hs_official_cash_option_floor_alpha_v1_0_1"
SUBJECT_DAILY_PATH = STAGING_ROOT / "subject_daily_v1_0_1.parquet"
CALENDAR_PATH = STAGING_ROOT / "trade_calendar_v1_0_1.parquet"
INDEX_DAILY_PATH = STAGING_ROOT / "csi300_daily_v1_0_1.parquet"
RECEIPT_PATH = STAGING_ROOT / "market_acquisition_receipt_v1_0_1.json"
ATTEMPT_ROOT = STAGING_ROOT / "acquisition_attempts"
ARCHIVE_PATH = (
    ROOT / "config/a_share_hs_official_cash_option_floor_alpha_v1_0_1_market_archive_manifest.json"
)
EASTMONEY_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_FIELDS1 = "f1,f2,f3,f4,f5,f6"
EASTMONEY_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116"
EASTMONEY_PUBLIC_UT = "7eea3edcaed734bea9cbfc24409ed989"


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


def make_session() -> requests.Session:
    retry = Retry(
        total=0,
        connect=0,
        read=0,
        status=0,
        backoff_factor=0.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1),
    )
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/136.0 Safari/537.36"
            ),
            "Referer": "https://quote.eastmoney.com/",
            "Connection": "close",
        }
    )
    return session


def eastmoney_secid(ts_code: str) -> str:
    code, exchange = str(ts_code).split(".", 1)
    if exchange == "SH":
        market = "1"
    elif exchange == "SZ":
        market = "0"
    else:
        raise RuntimeError(f"不支持的交易所：{ts_code}")
    return f"{market}.{code.zfill(6)}"


def eastmoney_params(ts_code: str, begin: str, end: str) -> dict[str, str]:
    return {
        "fields1": EASTMONEY_FIELDS1,
        "fields2": EASTMONEY_FIELDS2,
        "ut": EASTMONEY_PUBLIC_UT,
        "klt": "101",
        "fqt": "0",
        "secid": eastmoney_secid(ts_code),
        "beg": begin,
        "end": end,
    }


def parse_eastmoney_payload(content: bytes, expected_ts_code: str) -> tuple[dict[str, Any], pd.DataFrame]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - 原始响应验证
        raise RuntimeError("东方财富响应不是 UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("东方财富响应顶层不是对象")
    try:
        rc = int(payload.get("rc"))
    except (TypeError, ValueError) as error:
        raise RuntimeError("东方财富响应缺少数值 rc") from error
    if rc != 0:
        raise RuntimeError(f"东方财富返回 rc={rc}，message={str(payload.get('message') or '')[:300]}")
    data = payload.get("data")
    if data is None:
        return payload, pd.DataFrame()
    if not isinstance(data, dict):
        raise RuntimeError("东方财富 data 不是对象或 null")
    expected_code = expected_ts_code.split(".", 1)[0]
    expected_exchange = expected_ts_code.split(".", 1)[1]
    expected_market = 1 if expected_exchange == "SH" else 0
    actual_code = str(data.get("code") or "")
    if actual_code and actual_code != expected_code:
        raise RuntimeError(f"东方财富响应证券代码冲突：预期={expected_code}，实际={actual_code}")
    actual_market = data.get("market")
    if actual_market is not None:
        try:
            normalized_market = int(actual_market)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"东方财富响应市场字段不可解析：{actual_market}") from error
        if normalized_market != expected_market:
            raise RuntimeError(
                f"东方财富响应市场冲突：预期={expected_market}，实际={normalized_market}"
            )
    klines = data.get("klines") or []
    if not isinstance(klines, list):
        raise RuntimeError("东方财富 klines 不是列表")
    if not klines:
        return payload, pd.DataFrame()
    rows = [str(item).split(",") for item in klines]
    widths = sorted({len(row) for row in rows})
    if len(widths) != 1 or widths[0] not in {11, 12}:
        raise RuntimeError(f"东方财富 K 线字段宽度不是统一的11或12：{widths}")
    columns = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude_pct",
        "pct_chg",
        "price_change",
        "turnover_pct",
    ]
    if widths[0] == 12:
        columns.append("extra_f116")
    frame = pd.DataFrame(rows, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    numeric = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude_pct",
        "pct_chg",
        "price_change",
        "turnover_pct",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if frame[["open", "close", "high", "low", "volume", "amount"]].isna().any().any():
        raise RuntimeError("东方财富 K 线核心数值无法解析")
    frame = frame.sort_values("date", kind="stable").drop_duplicates("date", keep="last")
    frame["pre_close"] = frame["close"].shift(1)
    return payload, frame


def fetch_eastmoney_raw(
    ts_code: str,
    begin: str,
    end: str,
    raw_path: Path,
) -> tuple[bytes, bool, dict[str, str]]:
    params = eastmoney_params(ts_code, begin, end)
    if raw_path.is_file() and raw_path.stat().st_size > 0:
        content = raw_path.read_bytes()
        parse_eastmoney_payload(content, ts_code)
        return content, True, params
    session = make_session()
    error: Exception | None = None
    try:
        for attempt in range(3):
            try:
                response = session.get(EASTMONEY_URL, params=params, timeout=(5, 10))
                response.raise_for_status()
                content = response.content
                if not content:
                    raise RuntimeError("东方财富返回空响应")
                parse_eastmoney_payload(content, ts_code)
                atomic_bytes(raw_path, content)
                return content, False, params
            except Exception as current:  # noqa: BLE001 - 有限重试
                error = current
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
    finally:
        session.close()
    raise RuntimeError(f"东方财富下载失败：{type(error).__name__}: {error}")


def eastmoney_index_chunk_specs(begin: str, end: str) -> list[tuple[str, str, str]]:
    start = pd.Timestamp(datetime.strptime(begin, "%Y%m%d")).normalize()
    finish = pd.Timestamp(datetime.strptime(end, "%Y%m%d")).normalize()
    if finish < start:
        raise RuntimeError(f"东方财富指数分段日期倒置：{begin}>{end}")
    chunks: list[tuple[str, str, str]] = []
    for year in range(start.year, finish.year + 1):
        chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(finish, pd.Timestamp(year=year, month=12, day=31))
        chunks.append(
            (
                chunk_start.strftime("%Y%m%d"),
                chunk_end.strftime("%Y%m%d"),
                f"000300_SH_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.json",
            )
        )
    return chunks


def fetch_sina_index_raw(raw_path: Path) -> tuple[bytes, bool]:
    if raw_path.is_file() and raw_path.stat().st_size > 0:
        return raw_path.read_bytes(), True
    url = zh_sina_a_stock_hist_url.format("sh000300")
    session = make_session()
    session.headers.update({"Referer": "https://finance.sina.com.cn/"})
    error: Exception | None = None
    try:
        for attempt in range(4):
            try:
                response = session.get(url, timeout=(12, 45))
                response.raise_for_status()
                if not response.content:
                    raise RuntimeError("新浪沪深300返回空响应")
                atomic_bytes(raw_path, response.content)
                return response.content, False
            except Exception as current:  # noqa: BLE001
                error = current
                time.sleep(1.5 * (attempt + 1))
    finally:
        session.close()
    raise RuntimeError(f"新浪沪深300下载失败：{type(error).__name__}: {error}")


def normalize_sina_subject(
    *,
    event_id: str,
    ts_code: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    raw_path: Path,
) -> pd.DataFrame:
    history = parse_sina_history(raw_path.read_bytes())
    data = history.loc[history["date"].between(window_start, window_end)].copy()
    columns = [
        "event_id",
        "ts_code",
        "date",
        "pre_close",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "volume",
        "amount",
        "price_source",
        "raw_path",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    data = data.rename(
        columns={"open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close"}
    )
    data["event_id"] = event_id
    data["ts_code"] = ts_code
    data["price_source"] = "SINA_FINANCE_HISTDATA_KLC2_RAW_UNADJUSTED"
    data["raw_path"] = relative(raw_path)
    if data[["pre_close", "raw_open", "raw_high", "raw_low", "raw_close", "volume"]].isna().any().any():
        raise RuntimeError(f"新浪入场窗口存在核心缺失：{event_id}")
    return data[columns].sort_values("date", kind="stable")


def normalize_eastmoney_subject(
    *,
    frame: pd.DataFrame,
    event_id: str,
    ts_code: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
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
        "volume",
        "amount",
        "price_source",
        "raw_path",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.loc[frame["date"].between(window_start, window_end)].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    data = data.rename(
        columns={"open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close"}
    )
    data["event_id"] = event_id
    data["ts_code"] = ts_code
    data["price_source"] = "EASTMONEY_PUSH2HIS_KLINE_RAW_UNADJUSTED"
    data["raw_path"] = relative(raw_path)
    return data[columns].sort_values("date", kind="stable")


def acquire_baostock_calendar(start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if BAOSTOCK_CALENDAR_PATH.is_file() and BAOSTOCK_RECEIPT_PATH.is_file():
        frame = pd.read_csv(BAOSTOCK_CALENDAR_PATH, dtype=str)
        receipt = json.loads(BAOSTOCK_RECEIPT_PATH.read_text(encoding="utf-8"))
        return frame, {**receipt, "cache_reused": True}
    login_code = ""
    login_message = ""
    query_code = ""
    query_message = ""
    frame = pd.DataFrame()
    final_error: Exception | None = None
    successful_attempt = 0
    for attempt in range(5):
        logged_in = False
        try:
            login = bs.login()
            login_code = str(login.error_code)
            login_message = str(login.error_msg)
            if login_code != "0":
                raise RuntimeError(
                    f"BaoStock 登录失败：code={login_code}，msg={login_message}"
                )
            logged_in = True
            result = bs.query_trade_dates(start_date=start, end_date=end)
            query_code = str(result.error_code)
            query_message = str(result.error_msg)
            fields = [str(value) for value in result.fields]
            if query_code != "0":
                raise RuntimeError(
                    f"BaoStock 交易日查询失败：code={query_code}，msg={query_message}"
                )
            rows: list[list[str]] = []
            while result.next():
                rows.append([str(value) for value in result.get_row_data()])
            query_code = str(result.error_code)
            query_message = str(result.error_msg)
            if query_code != "0":
                raise RuntimeError(
                    f"BaoStock 交易日迭代失败：code={query_code}，msg={query_message}"
                )
            frame = pd.DataFrame(rows, columns=fields)
            required = {"calendar_date", "is_trading_day"}
            if not required.issubset(frame.columns) or frame.empty:
                raise RuntimeError(
                    f"BaoStock 交易日结果为空或缺字段：{sorted(required - set(frame.columns))}"
                )
            successful_attempt = attempt + 1
            final_error = None
            break
        except Exception as current:  # noqa: BLE001 - 外部源有限重试
            final_error = current
            if attempt < 4:
                time.sleep(3.0 * (attempt + 1))
        finally:
            if logged_in:
                try:
                    bs.logout()
                except Exception:  # noqa: BLE001 - 不掩盖主查询结果
                    pass
    if final_error is not None:
        raise RuntimeError(
            f"BaoStock 五次有限重试仍失败：{type(final_error).__name__}: {final_error}"
        ) from final_error
    csv_content = frame.to_csv(index=False, lineterminator="\n")
    atomic_text(BAOSTOCK_CALENDAR_PATH, csv_content)
    receipt = {
        "source": "BAOSTOCK_QUERY_TRADE_DATES",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "start_date": start,
        "end_date": end,
        "login_error_code": login_code,
        "login_error_message": login_message,
        "query_error_code": query_code,
        "query_error_message": query_message,
        "successful_attempt": successful_attempt,
        "fields": fields,
        "row_count": len(frame),
        "calendar_csv": relative(BAOSTOCK_CALENDAR_PATH),
        "calendar_csv_sha256": sha256_file(BAOSTOCK_CALENDAR_PATH),
        "cache_reused": False,
    }
    atomic_text(
        BAOSTOCK_RECEIPT_PATH,
        json.dumps(receipt, ensure_ascii=False, indent=2, default=str) + "\n",
    )
    return frame, receipt


def attempt_receipt(payload: dict[str, Any]) -> Path:
    ATTEMPT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(TIMEZONE).strftime("%Y%m%dT%H%M%S%f%z")
    path = ATTEMPT_ROOT / f"attempt_{stamp}.json"
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    return path


def compare_event_sources(primary: pd.DataFrame, secondary: pd.DataFrame) -> dict[str, Any]:
    if primary.empty and secondary.empty:
        return {
            "coverage_status": "BOTH_COMPLETE_SOURCES_ZERO_WINDOW_ROWS",
            "primary_rows": 0,
            "secondary_rows": 0,
            "overlap_rows": 0,
            "ohlc_conflict_rows": 0,
            "max_ohlc_abs_diff_cny": None,
        }
    if primary.empty:
        return {
            "coverage_status": "PRIMARY_ZERO_SECONDARY_NONZERO_SOURCE_CONFLICT",
            "primary_rows": 0,
            "secondary_rows": len(secondary),
            "overlap_rows": 0,
            "ohlc_conflict_rows": len(secondary),
            "max_ohlc_abs_diff_cny": None,
        }
    if secondary.empty:
        return {
            "coverage_status": "PRIMARY_NONZERO_SECONDARY_ZERO_NO_VIEW",
            "primary_rows": len(primary),
            "secondary_rows": 0,
            "overlap_rows": 0,
            "ohlc_conflict_rows": len(primary),
            "max_ohlc_abs_diff_cny": None,
        }
    merged = primary.merge(
        secondary,
        on=["event_id", "ts_code", "date"],
        suffixes=("_sina", "_eastmoney"),
        how="outer",
        indicator=True,
    )
    overlap = merged.loc[merged["_merge"].eq("both")].copy()
    diffs = []
    for stem in ("open", "high", "low", "close"):
        diffs.append(
            (overlap[f"raw_{stem}_sina"] - overlap[f"raw_{stem}_eastmoney"]).abs()
        )
    max_by_row = pd.concat(diffs, axis=1).max(axis=1) if len(overlap) else pd.Series(dtype=float)
    tolerance = 0.005
    missing_date_rows = int(merged["_merge"].ne("both").sum())
    conflict_rows = int(max_by_row.gt(tolerance + 1e-12).sum()) + missing_date_rows
    return {
        "coverage_status": "PASS_DUAL_SOURCE_WINDOW" if conflict_rows == 0 else "NO_VIEW_DUAL_SOURCE_WINDOW_CONFLICT",
        "primary_rows": len(primary),
        "secondary_rows": len(secondary),
        "overlap_rows": len(overlap),
        "missing_date_rows": missing_date_rows,
        "ohlc_conflict_rows": conflict_rows,
        "max_ohlc_abs_diff_cny": float(max_by_row.max()) if len(max_by_row) else None,
    }


def acquire() -> tuple[dict[str, Any], int]:
    remediation = load_remediation_config()
    remediation_verification = verify_remediation_protocol(remediation)
    parent = load_parent_config()
    reference_specs = [
        item
        for item in parent["market_data_contract"]["preexisting_reference_inputs"]
        if item.get("role") == "2014_PLUS_CALENDAR_CROSSCHECK_ONLY"
    ]
    if len(reference_specs) != 1:
        raise RuntimeError("父协议2014年后交易日历交叉核对输入不是唯一一项")
    reference_spec = reference_specs[0]
    if ROOT / str(reference_spec["path"]) != FROZEN_2014_CALENDAR_PATH:
        raise RuntimeError("父协议2014年后交易日历路径与采集器不一致")
    if sha256_file(FROZEN_2014_CALENDAR_PATH) != str(reference_spec["sha256"]):
        raise RuntimeError("父协议2014年后交易日历哈希校验失败")
    for path in (SUBJECT_DAILY_PATH, CALENDAR_PATH, INDEX_DAILY_PATH, RECEIPT_PATH, ARCHIVE_PATH):
        if path.exists():
            raise RuntimeError(f"V1.0.1正式市场产物已存在，禁止覆盖：{relative(path)}")
    terms = pd.read_parquet(TERMS_PATH).sort_values(
        ["effective_announcement_date", "ts_code", "event_id"], kind="stable"
    )
    if len(terms) != 21:
        raise RuntimeError("冻结条款事件数不是21")
    started = datetime.now(TIMEZONE)
    failures: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    primary_frames: list[pd.DataFrame] = []
    secondary_frames: list[pd.DataFrame] = []
    source_summaries: list[dict[str, Any]] = []

    event_specs: list[dict[str, Any]] = []
    for row in terms.itertuples(index=False):
        announcement = pd.Timestamp(row.effective_announcement_date).normalize()
        window_end = pd.Timestamp(row.rights_registration_date).normalize()
        event_id = str(row.event_id)
        event_specs.append(
            {
                "event_id": event_id,
                "ts_code": str(row.ts_code),
                "announcement": announcement,
                "window_start": announcement + pd.Timedelta(days=1),
                "window_end": window_end,
                "query_begin": (announcement.date() - timedelta(days=30)).strftime("%Y%m%d"),
                "query_end": window_end.date().strftime("%Y%m%d"),
                "secondary_path": EASTMONEY_SUBJECT_ROOT / f"{safe_name(event_id)}.json",
            }
        )

    eastmoney_results: dict[str, tuple[bytes, bool, dict[str, str]]] = {}
    eastmoney_errors: dict[str, str] = {}
    for spec in event_specs:
        event_id = str(spec["event_id"])
        try:
            result = fetch_eastmoney_raw(
                str(spec["ts_code"]),
                str(spec["query_begin"]),
                str(spec["query_end"]),
                Path(spec["secondary_path"]),
            )
            eastmoney_results[event_id] = result
            if not result[1]:
                time.sleep(0.8)
        except Exception as error:  # noqa: BLE001 - 汇总后统一停机
            eastmoney_errors[event_id] = f"{type(error).__name__}: {error}"[:1000]

    for spec in event_specs:
        event_id = str(spec["event_id"])
        ts_code = str(spec["ts_code"])
        window_start = pd.Timestamp(spec["window_start"])
        window_end = pd.Timestamp(spec["window_end"])
        primary_path = PARENT_SINA_ROOT / f"{safe_name(ts_code)}.js"
        try:
            primary = normalize_sina_subject(
                event_id=event_id,
                ts_code=ts_code,
                window_start=window_start,
                window_end=window_end,
                raw_path=primary_path,
            )
            primary_frames.append(primary)
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "source": "SINA_FINANCE_HISTDATA_KLC2_RAW_UNADJUSTED",
                    "event_id": event_id,
                    "error": f"{type(error).__name__}: {error}"[:1000],
                }
            )
            primary = pd.DataFrame()

        secondary_path = Path(spec["secondary_path"])
        try:
            if event_id in eastmoney_errors:
                raise RuntimeError(eastmoney_errors[event_id])
            content, reused, params = eastmoney_results[event_id]
            _, history = parse_eastmoney_payload(content, ts_code)
            secondary = normalize_eastmoney_subject(
                frame=history,
                event_id=event_id,
                ts_code=ts_code,
                window_start=window_start,
                window_end=window_end,
                raw_path=secondary_path,
            )
            secondary_frames.append(secondary)
            raw_records.append(
                {
                    "source": "EASTMONEY_PUSH2HIS_KLINE_RAW_UNADJUSTED",
                    "event_id": event_id,
                    "ts_code": ts_code,
                    "path": relative(secondary_path),
                    "sha256": sha256_file(secondary_path),
                    "bytes": secondary_path.stat().st_size,
                    "cache_reused": reused,
                    "request_params_without_secret": params,
                    "decoded_query_rows": len(history),
                    "materialized_window_rows": len(secondary),
                }
            )
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "source": "EASTMONEY_PUSH2HIS_KLINE_RAW_UNADJUSTED",
                    "event_id": event_id,
                    "error": f"{type(error).__name__}: {error}"[:1000],
                }
            )
            secondary = pd.DataFrame()
        summary = compare_event_sources(primary, secondary)
        source_summaries.append(
            {
                "event_id": event_id,
                "ts_code": ts_code,
                "window_start": window_start.date().isoformat(),
                "window_end": window_end.date().isoformat(),
                **summary,
            }
        )
    calendar_frame = pd.DataFrame()
    baostock_receipt: dict[str, Any] = {}
    try:
        raw_calendar, baostock_receipt = acquire_baostock_calendar(
            remediation["remediated_market_data_contract"]["calendar_start"],
            remediation["remediated_market_data_contract"]["calendar_end"],
        )
        calendar_frame = raw_calendar.copy()
        calendar_frame["date"] = pd.to_datetime(
            calendar_frame["calendar_date"], errors="raise"
        ).dt.normalize()
        calendar_frame["is_open"] = calendar_frame["is_trading_day"].astype(str).eq("1")
        calendar_frame["source"] = "BAOSTOCK_QUERY_TRADE_DATES"
        calendar_frame = calendar_frame[["date", "is_open", "source"]].sort_values("date")
        raw_records.extend(
            [
                {
                    "source": "BAOSTOCK_QUERY_TRADE_DATES",
                    "path": relative(BAOSTOCK_CALENDAR_PATH),
                    "sha256": sha256_file(BAOSTOCK_CALENDAR_PATH),
                    "bytes": BAOSTOCK_CALENDAR_PATH.stat().st_size,
                    "row_count": len(calendar_frame),
                },
                {
                    "source": "BAOSTOCK_QUERY_RECEIPT",
                    "path": relative(BAOSTOCK_RECEIPT_PATH),
                    "sha256": sha256_file(BAOSTOCK_RECEIPT_PATH),
                    "bytes": BAOSTOCK_RECEIPT_PATH.stat().st_size,
                },
            ]
        )
    except Exception as error:  # noqa: BLE001
        failures.append(
            {
                "source": "BAOSTOCK_QUERY_TRADE_DATES",
                "error": f"{type(error).__name__}: {error}"[:1000],
            }
        )

    index_frames: list[pd.DataFrame] = []
    try:
        eastmoney_index_frames: list[pd.DataFrame] = []
        for begin, end, filename in eastmoney_index_chunk_specs(
            "20060101", parent["evidence_cutoff"].replace("-", "")
        ):
            eastmoney_index_path = EASTMONEY_INDEX_ROOT / filename
            content, reused, params = fetch_eastmoney_raw(
                "000300.SH", begin, end, eastmoney_index_path
            )
            _, chunk = parse_eastmoney_payload(content, "000300.SH")
            if chunk.empty:
                raise RuntimeError(f"东方财富沪深300分段返回零行：{begin}-{end}")
            chunk = chunk.copy()
            chunk["raw_path"] = relative(eastmoney_index_path)
            eastmoney_index_frames.append(chunk)
            raw_records.append(
                {
                    "source": "EASTMONEY_PUSH2HIS_INDEX_KLINE_RAW",
                    "path": relative(eastmoney_index_path),
                    "sha256": sha256_file(eastmoney_index_path),
                    "bytes": eastmoney_index_path.stat().st_size,
                    "cache_reused": reused,
                    "request_params_without_secret": params,
                    "row_count": len(chunk),
                }
            )
            if not reused:
                time.sleep(0.8)
        frame = pd.concat(eastmoney_index_frames, ignore_index=True).sort_values(
            "date", kind="stable"
        )
        if frame.duplicated("date").any():
            raise RuntimeError("东方财富沪深300分段存在重复交易日")
        if frame.empty:
            raise RuntimeError("东方财富沪深300完整区间返回零行")
        data = frame.rename(
            columns={"open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close"}
        )
        data["symbol"] = "000300.SH"
        data["price_source"] = "EASTMONEY_PUSH2HIS_INDEX_KLINE_RAW"
        index_frames.append(
            data[["symbol", "date", "raw_open", "raw_high", "raw_low", "raw_close", "volume", "amount", "price_source", "raw_path"]]
        )
    except Exception as error:  # noqa: BLE001
        failures.append(
            {
                "source": "EASTMONEY_PUSH2HIS_INDEX_KLINE_RAW",
                "error": f"{type(error).__name__}: {error}"[:1000],
            }
        )

    sina_index_path = SINA_INDEX_ROOT / "000300_SH.js"
    try:
        content, reused = fetch_sina_index_raw(sina_index_path)
        frame = parse_sina_history(content)
        frame = frame.loc[
            frame["date"].between(pd.Timestamp("2006-01-01"), pd.Timestamp(parent["evidence_cutoff"]))
        ].copy()
        if frame.empty:
            raise RuntimeError("新浪沪深300完整区间返回零行")
        data = frame.rename(
            columns={"open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close"}
        )
        data["symbol"] = "000300.SH"
        data["price_source"] = "SINA_FINANCE_HISTDATA_KLC2_INDEX_RAW"
        data["raw_path"] = relative(sina_index_path)
        index_frames.append(
            data[["symbol", "date", "raw_open", "raw_high", "raw_low", "raw_close", "volume", "amount", "price_source", "raw_path"]]
        )
        raw_records.append(
            {
                "source": "SINA_FINANCE_HISTDATA_KLC2_INDEX_RAW",
                "path": relative(sina_index_path),
                "sha256": sha256_file(sina_index_path),
                "bytes": sina_index_path.stat().st_size,
                "cache_reused": reused,
                "row_count": len(frame),
            }
        )
    except Exception as error:  # noqa: BLE001
        failures.append(
            {
                "source": "SINA_FINANCE_HISTDATA_KLC2_INDEX_RAW",
                "error": f"{type(error).__name__}: {error}"[:1000],
            }
        )

    attempt = {
        "remediation_id": remediation["remediation_id"],
        "status": "ACQUISITION_ATTEMPT_COMPLETE" if not failures else "ACQUISITION_ATTEMPT_NETWORK_OR_PARSE_FAILURE",
        "started_at": started.isoformat(),
        "completed_at": datetime.now(TIMEZONE).isoformat(),
        "event_count": len(terms),
        "failure_count": len(failures),
        "failures": failures,
        "raw_file_count": len(raw_records),
        "eastmoney_price_values_read_after_remediation_freeze": True,
        "baostock_calendar_values_read_after_remediation_freeze": not calendar_frame.empty,
        "price_screen_completed": False,
        "qualified_event_count_known": False,
    }
    attempt_path = attempt_receipt(attempt)
    if failures:
        return {
            **attempt,
            "attempt_receipt": relative(attempt_path),
            "formal_market_receipt_created": False,
        }, 2

    subject = pd.concat([*primary_frames, *secondary_frames], ignore_index=True)
    subject = subject.sort_values(["event_id", "price_source", "date"], kind="stable").reset_index(drop=True)
    if subject.duplicated(["event_id", "price_source", "date"]).any():
        raise RuntimeError("V1.0.1标的日线存在事件-来源-日期重复")
    index_daily = pd.concat(index_frames, ignore_index=True).sort_values(
        ["price_source", "date"], kind="stable"
    )
    if index_daily.duplicated(["price_source", "date"]).any():
        raise RuntimeError("沪深300日线存在来源-日期重复")

    calendar_start = pd.Timestamp(
        remediation["remediated_market_data_contract"]["calendar_start"]
    )
    calendar_end = pd.Timestamp(
        remediation["remediated_market_data_contract"]["calendar_end"]
    )
    open_calendar_dates = set(calendar_frame.loc[calendar_frame["is_open"], "date"])
    index_dates_by_source = {
        source: set(group.loc[group["date"].between(calendar_start, calendar_end), "date"])
        for source, group in index_daily.groupby("price_source", sort=True)
    }
    frozen_reference = pd.read_parquet(FROZEN_2014_CALENDAR_PATH)
    frozen_reference_dates = set(
        pd.to_datetime(
            frozen_reference.loc[
                frozen_reference["is_open"].astype(bool), "date"
            ],
            errors="raise",
        )
        .dt.normalize()
        .loc[lambda values: values.between(calendar_start, calendar_end)]
    )
    calendar_crosscheck_dates = set().union(
        *index_dates_by_source.values(), frozen_reference_dates
    )
    calendar_diagnostics = {
        "baostock_open_date_count": len(open_calendar_dates),
        "index_source_date_counts": {
            str(source): len(dates) for source, dates in index_dates_by_source.items()
        },
        "index_dates_not_baostock_open_counts": {
            str(source): len(dates - open_calendar_dates)
            for source, dates in index_dates_by_source.items()
        },
        "frozen_2014_plus_calendar_path": relative(FROZEN_2014_CALENDAR_PATH),
        "frozen_2014_plus_calendar_sha256": sha256_file(FROZEN_2014_CALENDAR_PATH),
        "frozen_2014_plus_open_date_count_in_contract_window": len(frozen_reference_dates),
        "baostock_open_dates_missing_all_crosschecks": len(
            open_calendar_dates - calendar_crosscheck_dates
        ),
        "crosscheck_dates_not_baostock_open": len(
            calendar_crosscheck_dates - open_calendar_dates
        ),
    }

    atomic_parquet(SUBJECT_DAILY_PATH, subject)
    atomic_parquet(CALENDAR_PATH, calendar_frame)
    atomic_parquet(INDEX_DAILY_PATH, index_daily)
    coverage_counts = Counter(item["coverage_status"] for item in source_summaries)
    receipt = {
        "remediation_id": remediation["remediation_id"],
        "status": "PASS_SOURCE_REMEDIATED_MARKET_INPUTS_ACQUIRED_WITH_EVENT_LEVEL_DISPOSITIONS_PENDING",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "evidence_cutoff": parent["evidence_cutoff"],
        "remediation_protocol_verification": remediation_verification,
        "event_count": len(terms),
        "event_source_summaries": source_summaries,
        "coverage_status_counts": dict(sorted(coverage_counts.items())),
        "raw_files": sorted(
            raw_records,
            key=lambda item: (str(item.get("source") or ""), str(item.get("event_id") or item.get("path") or "")),
        ),
        "subject_daily": relative(SUBJECT_DAILY_PATH),
        "subject_daily_sha256": sha256_file(SUBJECT_DAILY_PATH),
        "subject_daily_rows": len(subject),
        "trade_calendar": relative(CALENDAR_PATH),
        "trade_calendar_sha256": sha256_file(CALENDAR_PATH),
        "trade_calendar_rows": len(calendar_frame),
        "csi300_daily": relative(INDEX_DAILY_PATH),
        "csi300_daily_sha256": sha256_file(INDEX_DAILY_PATH),
        "csi300_daily_rows": len(index_daily),
        "baostock_query_receipt": baostock_receipt,
        "calendar_diagnostics": calendar_diagnostics,
        "attempt_receipt": relative(attempt_path),
        "attempt_receipt_sha256": sha256_file(attempt_path),
        "eastmoney_price_values_read_after_remediation_freeze": True,
        "price_screen_completed": False,
        "qualified_event_count_known": False,
        "liquidity_filter_applied": False,
        "position_mapping": False,
        "order_generation": False,
        "shadow_authorized": False,
        "live_authorized": False,
    }
    atomic_text(RECEIPT_PATH, json.dumps(receipt, ensure_ascii=False, indent=2, default=str) + "\n")
    remediation_manifest = ROOT / str(remediation["artifacts"]["protocol_manifest"])
    tracked = [
        remediation_manifest,
        TERMS_PATH,
        SUBJECT_DAILY_PATH,
        CALENDAR_PATH,
        INDEX_DAILY_PATH,
        RECEIPT_PATH,
        attempt_path,
        ROOT / "scripts/acquire_a_share_hs_cash_option_floor_alpha_v1_market_v1.py",
        ROOT / "scripts/acquire_a_share_hs_cash_option_floor_alpha_v1_0_1_market.py",
        FROZEN_2014_CALENDAR_PATH,
        *[PARENT_SINA_ROOT / f"{safe_name(str(code))}.js" for code in terms["ts_code"]],
        *[ROOT / str(item["path"]) for item in raw_records],
    ]
    unique_tracked = list(dict.fromkeys(path.resolve() for path in tracked))
    files = {relative(path): sha256_file(path) for path in unique_tracked}
    archive = {
        "remediation_id": f"{remediation['remediation_id']}_MARKET_ARCHIVE_FREEZE",
        "status": "FROZEN_SOURCE_REMEDIATED_MARKET_INPUTS_BEFORE_PRICE_SCREEN",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "files": files,
        "content_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "event_count": len(terms),
        "coverage_status_counts": dict(sorted(coverage_counts.items())),
        "eastmoney_price_values_read_after_remediation_freeze": True,
        "price_screen_completed": False,
        "qualified_event_count_known": False,
        "future_values_used_for_screening": False,
    }
    atomic_text(ARCHIVE_PATH, json.dumps(archive, ensure_ascii=False, indent=2, default=str) + "\n")
    receipt["market_archive_manifest"] = relative(ARCHIVE_PATH)
    receipt["market_archive_manifest_sha256"] = sha256_file(ARCHIVE_PATH)
    return receipt, 0


def verify_acquisition() -> dict[str, Any]:
    remediation = load_remediation_config()
    remediation_verification = verify_remediation_protocol(remediation)
    if not RECEIPT_PATH.exists() or not ARCHIVE_PATH.exists():
        raise RuntimeError("V1.0.1正式市场回执或冻结清单不存在")
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    archive = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    if receipt.get("status") != "PASS_SOURCE_REMEDIATED_MARKET_INPUTS_ACQUIRED_WITH_EVENT_LEVEL_DISPOSITIONS_PENDING":
        failures.append({"failure": "RECEIPT_STATUS", "actual": receipt.get("status")})
    if archive.get("status") != "FROZEN_SOURCE_REMEDIATED_MARKET_INPUTS_BEFORE_PRICE_SCREEN":
        failures.append({"failure": "ARCHIVE_STATUS", "actual": archive.get("status")})
    for item_path, expected_sha in (archive.get("files") or {}).items():
        path = ROOT / str(item_path)
        actual = sha256_file(path) if path.exists() else None
        if actual != expected_sha:
            failures.append(
                {
                    "failure": "FILE_HASH",
                    "path": item_path,
                    "expected": expected_sha,
                    "actual": actual,
                }
            )
    artifacts = [
        (SUBJECT_DAILY_PATH, "subject_daily_sha256"),
        (CALENDAR_PATH, "trade_calendar_sha256"),
        (INDEX_DAILY_PATH, "csi300_daily_sha256"),
    ]
    for path, field in artifacts:
        if sha256_file(path) != receipt.get(field):
            failures.append({"failure": "RECEIPT_ARTIFACT_HASH", "path": relative(path)})
    result = {
        "status": (
            "PASS_SOURCE_REMEDIATED_MARKET_ARCHIVE_VERIFIED"
            if not failures
            else "FAILED_SOURCE_REMEDIATED_MARKET_ARCHIVE_VERIFICATION"
        ),
        "failure_count": len(failures),
        "failures": failures,
        "remediation_protocol_verification": remediation_verification,
        "market_archive_manifest": relative(ARCHIVE_PATH),
        "market_archive_manifest_sha256": sha256_file(ARCHIVE_PATH),
        "event_count": int(receipt["event_count"]),
        "coverage_status_counts": receipt["coverage_status_counts"],
        "subject_daily_rows": int(receipt["subject_daily_rows"]),
        "trade_calendar_rows": int(receipt["trade_calendar_rows"]),
        "csi300_daily_rows": int(receipt["csi300_daily_rows"]),
        "price_screen_completed": False,
        "qualified_event_count_known": False,
    }
    if failures:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, default=str))
    return result


def status() -> dict[str, Any]:
    remediation = load_remediation_config()
    verification = verify_remediation_protocol(remediation)
    return {
        "status": "READY_TO_ACQUIRE_SOURCE_REMEDIATED_MARKET_INPUTS",
        "remediation_protocol_verification": verification,
        "terms_event_count_from_parquet_metadata": int(
            pd.read_parquet(TERMS_PATH, columns=["event_id"]).shape[0]
        ),
        "eastmoney_formal_output_exists": RECEIPT_PATH.exists(),
        "market_archive_exists": ARCHIVE_PATH.exists(),
        "eastmoney_price_values_read_by_this_mode": False,
        "baostock_calendar_values_read_by_this_mode": False,
        "price_screen_completed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按V1.0.1冻结源修复采集新浪/东方财富双源日线与BaoStock日历"
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
