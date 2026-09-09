"""采集并审计510300全市场换手率候选的冻结输入。

本脚本只读取既有510300行情的日期列作为中国交易日历，不读取价格、
未来收益、组合收益或夏普率。所有官方响应与月度因子在完整审计通过后
才原子写入；冻结清单存在后禁止覆盖。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import base64
import gzip
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable
import unicodedata
import zipfile
from zoneinfo import ZoneInfo

from lxml import html as lxml_html
import numpy as np
import pandas as pd
import pdfplumber
import requests


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from aggregate_turnover_visibility_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    ContractError,
    build_monthly_factor,
    build_monthly_factor_from_components,
    load_config,
    parse_sse_current_daily,
    parse_sse_current_monthly,
    parse_sse_legacy_daily,
    parse_sse_legacy_monthly,
    parse_szse_daily,
    parse_szse_monthly,
    sha256_bytes,
    sha256_file,
)


PRICE_PATH = ROOT / "data" / "raw" / "market" / "510300_daily_downside_risk_v1.parquet"
PRICE_AUDIT_PATH = ROOT / "reports" / "data_quality" / "510300_downside_risk_inputs_v1.json"


class InputAuditError(RuntimeError):
    """官方输入未通过冻结前审计。"""


@dataclass(frozen=True)
class ResponseSnapshot:
    source: str
    period: str
    requested_url: str
    resolved_url: str
    payload: bytes
    from_checkpoint: bool = False

    def raw_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "period": self.period,
            "requested_url": self.requested_url,
            "resolved_url": self.resolved_url,
            "response_sha256": sha256_bytes(self.payload),
            "response_bytes": len(self.payload),
            "response_text_utf8": self.payload.decode("utf-8"),
            "from_checkpoint": self.from_checkpoint,
        }


_THREAD_LOCAL = threading.local()
_RATE_LIMIT_LOCKS = {
    "SSE": threading.Lock(),
    "SZSE": threading.Lock(),
}
_LAST_REQUEST_MONOTONIC = {"SSE": 0.0, "SZSE": 0.0}


def _checkpoint_path(source: str, period: str) -> Path:
    safe_period = period.replace(":", "-").replace("/", "-")
    return (
        ROOT
        / "data"
        / "checkpoints"
        / "510300_aggregate_turnover_visibility_v1"
        / source
        / f"{safe_period}.json"
    )


def _read_checkpoint(source: str, period: str) -> ResponseSnapshot | None:
    path = _checkpoint_path(source, period)
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("source") != source or document.get("period") != period:
        raise InputAuditError(f"检查点身份不一致：{path}")
    payload = base64.b64decode(document["payload_base64"], validate=True)
    if sha256_bytes(payload) != document.get("response_sha256"):
        raise InputAuditError(f"检查点哈希不一致：{path}")
    return ResponseSnapshot(
        source=source,
        period=period,
        requested_url=str(document["requested_url"]),
        resolved_url=str(document["resolved_url"]),
        payload=payload,
        from_checkpoint=True,
    )


def _write_checkpoint(snapshot: ResponseSnapshot) -> None:
    path = _checkpoint_path(snapshot.source, snapshot.period)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "source": snapshot.source,
        "period": snapshot.period,
        "requested_url": snapshot.requested_url,
        "resolved_url": snapshot.resolved_url,
        "response_sha256": sha256_bytes(snapshot.payload),
        "response_bytes": len(snapshot.payload),
        "payload_base64": base64.b64encode(snapshot.payload).decode("ascii"),
        "tls_verification_disabled": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _respect_rate_limit(source: str, minimum_interval_seconds: float) -> None:
    family = "SZSE" if source.startswith("SZSE") else "SSE"
    with _RATE_LIMIT_LOCKS[family]:
        now = time.monotonic()
        remaining = (
            _LAST_REQUEST_MONOTONIC[family] + minimum_interval_seconds - now
        )
        if remaining > 0:
            time.sleep(remaining)
        _LAST_REQUEST_MONOTONIC[family] = time.monotonic()


def _session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/140.0 Safari/537.36"
                ),
                "Accept": "application/json,text/javascript,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9",
            }
        )
        _THREAD_LOCAL.session = session
    return session


def _request(
    *,
    source: str,
    period: str,
    endpoint: str,
    params: dict[str, Any],
    referer: str,
    timeout_seconds: int,
    retry_limit: int,
    minimum_interval_seconds: float,
) -> ResponseSnapshot:
    if not endpoint.lower().startswith("https://"):
        raise InputAuditError(f"正式官方数据源禁止非HTTPS地址：{endpoint}")
    cached = _read_checkpoint(source, period)
    if cached is not None:
        return cached
    last_error: Exception | None = None
    for attempt in range(1, retry_limit + 1):
        try:
            _respect_rate_limit(source, minimum_interval_seconds)
            response = _session().get(
                endpoint,
                params=params,
                headers={"Referer": referer},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            if not response.url.lower().startswith("https://"):
                raise InputAuditError(f"{source} {period}发生非HTTPS重定向")
            if not response.content:
                raise InputAuditError(f"{source} {period}响应为空")
            snapshot = ResponseSnapshot(
                source=source,
                period=period,
                requested_url=endpoint,
                resolved_url=response.url,
                payload=response.content,
            )
            _write_checkpoint(snapshot)
            return snapshot
        except (requests.RequestException, InputAuditError) as exc:
            last_error = exc
            if attempt < retry_limit:
                time.sleep(0.5 * (2 ** (attempt - 1)))
    raise InputAuditError(
        f"{source} {period}在{retry_limit}次尝试后失败：{last_error}"
    )


def _sse_daily_request(date: str, config: dict[str, Any]) -> tuple[dict[str, Any], ResponseSnapshot]:
    contract = config["data_contract"]
    specification = contract["sse"]
    legacy_last = pd.Timestamp(specification["daily_legacy_last_date"])
    endpoint = specification["query_endpoint"]
    if pd.Timestamp(date) <= legacy_last:
        source = "SSE_LEGACY_DAILY_A"
        params = {
            "jsonCallBack": "jsonpCallback",
            "sqlId": specification["daily_legacy_sql_id"],
            "isPagination": "false",
            "searchDate": date,
            "stockType": "90",
        }
        snapshot = _request(
            source=source,
            period=date,
            endpoint=endpoint,
            params=params,
            referer=specification.get(
                "daily_page_historical", specification["monthly_page_historical"]
            ),
            timeout_seconds=int(contract["request_timeout_seconds"]),
            retry_limit=int(contract["request_retry_limit"]),
            minimum_interval_seconds=float(
                contract["sse_minimum_request_interval_seconds"]
            ),
        )
        parsed = parse_sse_legacy_daily(
            snapshot.payload,
            date,
            specification["daily_legacy_a_product_types"],
        )
    else:
        source = "SSE_CURRENT_DAILY_A"
        params = {
            "jsonCallBack": "jsonpCallback",
            "sqlId": specification["daily_current_sql_id"],
            "PRODUCT_CODE": "01,02,03,11,17",
            "type": "inParams",
            "SEARCH_DATE": date,
        }
        snapshot = _request(
            source=source,
            period=date,
            endpoint=endpoint,
            params=params,
            referer=specification.get(
                "daily_page_current", specification["monthly_page_current"]
            ),
            timeout_seconds=int(contract["request_timeout_seconds"]),
            retry_limit=int(contract["request_retry_limit"]),
            minimum_interval_seconds=float(
                contract["sse_minimum_request_interval_seconds"]
            ),
        )
        parsed = parse_sse_current_daily(
            snapshot.payload,
            date,
            specification["daily_current_a_product_codes"],
        )
    return parsed, snapshot


def _szse_daily_request(date: str, config: dict[str, Any]) -> tuple[dict[str, Any], ResponseSnapshot]:
    contract = config["data_contract"]
    specification = contract["szse"]
    params = {
        "SHOWTYPE": "JSON",
        "CATALOGID": specification["daily_catalog_id"],
        "TABKEY": specification["daily_tab_key"],
        "txtQueryDate": date,
    }
    snapshot = _request(
        source="SZSE_DAILY_STOCK_TOTAL",
        period=date,
        endpoint=specification["query_endpoint"],
        params=params,
        referer=specification["daily_page"],
        timeout_seconds=int(contract["request_timeout_seconds"]),
        retry_limit=int(contract["request_retry_limit"]),
        minimum_interval_seconds=float(
            contract["szse_minimum_request_interval_seconds"]
        ),
    )
    return parse_szse_daily(snapshot.payload, date), snapshot


def _collect_daily_date(
    date: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[ResponseSnapshot]]:
    sse, sse_snapshot = _sse_daily_request(date, config)
    szse, szse_snapshot = _szse_daily_request(date, config)
    row = {
        "date": date,
        "sse_a_trade_amount_cny_100m": float(sse["trade_amount_cny_100m"]),
        "sse_a_market_cap_cny_100m": float(sse["market_cap_cny_100m"]),
        "szse_stock_trade_amount_cny_100m": float(szse["trade_amount_cny_100m"]),
        "szse_stock_market_cap_cny_100m": float(szse["market_cap_cny_100m"]),
        "sse_source": sse_snapshot.source,
        "sse_selected_categories": ",".join(sse["selected_categories_observed"]),
        "szse_scope": "STOCK_TOTAL_INCLUDING_B_AND_CDR",
        "sse_response_sha256": sha256_bytes(sse_snapshot.payload),
        "szse_response_sha256": sha256_bytes(szse_snapshot.payload),
    }
    row["aggregate_trade_amount_cny_100m"] = (
        row["sse_a_trade_amount_cny_100m"]
        + row["szse_stock_trade_amount_cny_100m"]
    )
    row["aggregate_market_cap_cny_100m"] = (
        row["sse_a_market_cap_cny_100m"]
        + row["szse_stock_market_cap_cny_100m"]
    )
    return row, [sse_snapshot, szse_snapshot]


def _sse_monthly_request(
    month: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], ResponseSnapshot]:
    contract = config["data_contract"]
    specification = contract["sse"]
    endpoint = specification["query_endpoint"]
    if month <= str(specification["monthly_legacy_last_month"]):
        source = "SSE_LEGACY_MONTHLY_A_CROSSCHECK"
        params = {
            "jsonCallBack": "jsonpCallback",
            "sqlId": specification["monthly_legacy_sql_id"],
            "inYear": month,
            "stockType": "90",
            "isPagination": "false",
        }
        snapshot = _request(
            source=source,
            period=month,
            endpoint=endpoint,
            params=params,
            referer=specification["monthly_page_historical"],
            timeout_seconds=int(contract["request_timeout_seconds"]),
            retry_limit=int(contract["request_retry_limit"]),
            minimum_interval_seconds=float(
                contract["sse_minimum_request_interval_seconds"]
            ),
        )
        parsed = parse_sse_legacy_monthly(
            snapshot.payload,
            month,
            specification["daily_legacy_a_product_types"],
        )
    else:
        source = "SSE_CURRENT_MONTHLY_A_CROSSCHECK"
        params = {
            "jsonCallBack": "jsonpCallback",
            "sqlId": specification["monthly_current_sql_id"],
            "PRODUCT_CODE": "01,02,03,11,17",
            "type": "inParams",
            "SEARCH_DATE": month,
        }
        snapshot = _request(
            source=source,
            period=month,
            endpoint=endpoint,
            params=params,
            referer=specification["monthly_page_current"],
            timeout_seconds=int(contract["request_timeout_seconds"]),
            retry_limit=int(contract["request_retry_limit"]),
            minimum_interval_seconds=float(
                contract["sse_minimum_request_interval_seconds"]
            ),
        )
        parsed = parse_sse_current_monthly(
            snapshot.payload,
            month,
            specification["daily_current_a_product_codes"],
        )
    return parsed, snapshot


def _szse_monthly_request(
    month: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], ResponseSnapshot]:
    contract = config["data_contract"]
    specification = contract["szse"]
    params = {
        "SHOWTYPE": "JSON",
        "CATALOGID": specification["monthly_catalog_id"],
        "txtQueryDate": month,
        "tjzqlb": "M",
    }
    snapshot = _request(
        source="SZSE_MONTHLY_STOCK_TOTAL_CROSSCHECK",
        period=month,
        endpoint=specification["query_endpoint"],
        params=params,
        referer=specification["monthly_page"],
        timeout_seconds=int(contract["request_timeout_seconds"]),
        retry_limit=int(contract["request_retry_limit"]),
        minimum_interval_seconds=float(
            contract["szse_minimum_request_interval_seconds"]
        ),
    )
    return parse_szse_monthly(snapshot.payload, month), snapshot


def _collect_monthly_crosscheck(
    month: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[ResponseSnapshot]]:
    sse, sse_snapshot = _sse_monthly_request(month, config)
    szse, szse_snapshot = _szse_monthly_request(month, config)
    return (
        {
            "factor_month": month,
            "sse_monthly_trade_amount_cny_100m": float(
                sse["trade_amount_cny_100m"]
            ),
            "sse_month_end_market_cap_cny_100m": float(
                sse["market_cap_cny_100m"]
            ),
            "sse_identical_duplicate_rows_collapsed": int(
                sse["identical_duplicate_rows_collapsed"]
            ),
            "szse_monthly_trade_amount_cny_100m": float(
                szse["trade_amount_cny_100m"]
            ),
            "sse_monthly_response_sha256": sha256_bytes(sse_snapshot.payload),
            "szse_monthly_response_sha256": sha256_bytes(szse_snapshot.payload),
        },
        [sse_snapshot, szse_snapshot],
    )


def _parallel_collect(
    values: list[str],
    function: Callable[[str, dict[str, Any]], tuple[dict[str, Any], list[ResponseSnapshot]]],
    config: dict[str, Any],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], list[ResponseSnapshot]]:
    workers = int(config["data_contract"]["request_workers"])
    rows: list[dict[str, Any]] = []
    snapshots: list[ResponseSnapshot] = []
    completed = 0
    print(f"开始{label}：{len(values)}个期间，{workers}个并发线程", flush=True)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="official-data") as executor:
        future_map = {
            executor.submit(function, value, config): value for value in values
        }
        try:
            for future in as_completed(future_map):
                value = future_map[future]
                try:
                    row, response_snapshots = future.result()
                except Exception as exc:
                    for pending in future_map:
                        pending.cancel()
                    raise InputAuditError(f"{label}{value}失败：{exc}") from exc
                rows.append(row)
                snapshots.extend(response_snapshots)
                completed += 1
                if completed % 250 == 0 or completed == len(values):
                    print(f"{label}进度：{completed}/{len(values)}", flush=True)
        finally:
            for pending in future_map:
                pending.cancel()
    return rows, snapshots


def _load_calendar(config: dict[str, Any]) -> pd.DataFrame:
    if not PRICE_PATH.exists() or not PRICE_AUDIT_PATH.exists():
        raise InputAuditError("缺少既有510300日期日历或其输入审计")
    prior_audit = json.loads(PRICE_AUDIT_PATH.read_text(encoding="utf-8"))
    if prior_audit.get("status") != "PASS":
        raise InputAuditError("既有510300输入审计不是PASS")
    if prior_audit.get("candidate_outcomes_computed") is not False:
        raise InputAuditError("既有510300输入审计边界异常")
    calendar = pd.read_parquet(PRICE_PATH, columns=["date"])
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    start = pd.Timestamp(config["dates"]["factor_warmup_start"])
    end = pd.Timestamp(config["dates"]["factor_last_complete_month_end"])
    calendar = calendar.loc[calendar["date"].between(start, end)].copy()
    calendar = calendar.sort_values("date").reset_index(drop=True)
    if calendar["date"].isna().any() or calendar["date"].duplicated().any():
        raise InputAuditError("交易日历存在空日期或重复日期")
    return calendar


def _crosscheck_table(
    daily: pd.DataFrame,
    monthly_crosscheck: pd.DataFrame,
) -> pd.DataFrame:
    frame = daily.copy()
    frame["factor_month"] = frame["date"].dt.to_period("M").astype(str)
    sums = frame.groupby("factor_month", sort=True).agg(
        sse_daily_sum_trade_amount_cny_100m=(
            "sse_a_trade_amount_cny_100m",
            "sum",
        ),
        szse_daily_sum_trade_amount_cny_100m=(
            "szse_stock_trade_amount_cny_100m",
            "sum",
        ),
    )
    month_end = (
        frame.groupby("factor_month", sort=True, as_index=False)
        .tail(1)
        .set_index("factor_month")
    )
    sums = sums.join(
        month_end[["sse_a_market_cap_cny_100m"]].rename(
            columns={
                "sse_a_market_cap_cny_100m": "sse_daily_month_end_market_cap_cny_100m"
            }
        )
    ).reset_index()
    output = sums.merge(monthly_crosscheck, on="factor_month", validate="one_to_one")
    pairs = [
        (
            "sse_trade",
            "sse_daily_sum_trade_amount_cny_100m",
            "sse_monthly_trade_amount_cny_100m",
        ),
        (
            "szse_trade",
            "szse_daily_sum_trade_amount_cny_100m",
            "szse_monthly_trade_amount_cny_100m",
        ),
        (
            "sse_market_cap",
            "sse_daily_month_end_market_cap_cny_100m",
            "sse_month_end_market_cap_cny_100m",
        ),
    ]
    for label, daily_column, monthly_column in pairs:
        output[f"{label}_absolute_difference"] = (
            output[daily_column] - output[monthly_column]
        ).abs()
        output[f"{label}_relative_difference"] = output[
            f"{label}_absolute_difference"
        ] / output[monthly_column].abs()
    return output


def _deterministic_gzip_jsonl(records: list[dict[str, Any]]) -> bytes:
    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as compressed:
        for record in sorted(records, key=lambda row: (row["source"], row["period"])):
            line = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            compressed.write((line + "\n").encode("utf-8"))
    return buffer.getvalue()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def _path(config: dict[str, Any], key: str) -> Path:
    return ROOT / config["inputs"][key]["path"]


_MONTH_LABEL_TO_NUMBER = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "SEPT": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
_MONTH_NUMBER_TO_NAME = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def _normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value)).replace("\u00a0", " ").strip()


def _month_header_number(value: Any) -> int | None:
    text = re.sub(r"[^A-Z]", "", _normalize_text(value).upper())
    return _MONTH_LABEL_TO_NUMBER.get(text)


def _group_words_by_top(
    words: list[dict[str, Any]], *, tolerance: float = 0.65
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    centers: list[float] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        top = float(word["top"])
        if not groups or abs(top - centers[-1]) > tolerance:
            groups.append([word])
            centers.append(top)
        else:
            groups[-1].append(word)
            centers[-1] = float(np.mean([float(item["top"]) for item in groups[-1]]))
    return groups


def _header_centers(
    words: list[dict[str, Any]],
    *,
    expected_months: list[int],
    anchor_top: float,
) -> list[float]:
    candidates: list[tuple[float, list[float]]] = []
    for group in _group_words_by_top(words):
        group_top = float(np.mean([float(word["top"]) for word in group]))
        if group_top > anchor_top + 1.0:
            continue
        month_words: dict[int, list[dict[str, Any]]] = {}
        for word in group:
            month_number = _month_header_number(word["text"])
            if month_number is not None:
                month_words.setdefault(month_number, []).append(word)
        if not all(month in month_words for month in expected_months):
            continue
        if any(len(month_words[month]) != 1 for month in expected_months):
            continue
        centers = [
            (
                float(month_words[month][0]["x0"])
                + float(month_words[month][0]["x1"])
            )
            / 2.0
            for month in expected_months
        ]
        if centers != sorted(centers):
            continue
        candidates.append((group_top, centers))
    if not candidates:
        raise InputAuditError(
            f"年鉴页面未找到完整月份表头：{expected_months}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def _parse_sequential_decimal_values(
    words: list[dict[str, Any]],
    *,
    lower_x: float,
    upper_x: float,
    expected_count: int,
) -> list[float] | None:
    fragments: list[str] = []
    for word in sorted(words, key=lambda item: float(item["x0"])):
        center = (float(word["x0"]) + float(word["x1"])) / 2.0
        if center < lower_x or center > upper_x:
            continue
        text = re.sub(r"\s+", "", _normalize_text(word["text"]))
        if not re.fullmatch(r"[0-9.,]+", text) or not any(
            character.isdigit() for character in text
        ):
            continue
        fragments.append(text)

    values: list[float] = []
    accumulator: list[str] = []
    for fragment in fragments:
        accumulator.append(fragment)
        combined = "".join(accumulator)
        if re.search(r"[.,][0-9]{2}$", combined) is None:
            continue
        decimal_position = max(combined.rfind("."), combined.rfind(","))
        if decimal_position <= 0 or len(combined) - decimal_position - 1 != 2:
            return None
        integer_part = re.sub(r"[.,]", "", combined[:decimal_position])
        decimal_part = combined[decimal_position + 1 :]
        if not integer_part.isdigit() or not decimal_part.isdigit():
            return None
        values.append(float(f"{integer_part}.{decimal_part}"))
        accumulator = []
    if accumulator or len(values) != expected_count:
        return None
    return values


def _extract_factbook_page_values(
    page: Any,
    *,
    expected_months: list[int],
    row_kind: str,
    anchor_top_hint: float | None = None,
) -> tuple[list[float], float]:
    words = page.extract_words(
        use_text_flow=False,
        keep_blank_chars=False,
        x_tolerance=1,
        y_tolerance=1,
    )
    if not words:
        raise InputAuditError("年鉴页面没有可提取文字")
    if row_kind == "market_cap":
        anchors = [
            word
            for word in words
            if "股票市价总值" in re.sub(r"\s+", "", _normalize_text(word["text"]))
        ]
    elif row_kind == "stock_turnover":
        anchors = [
            word for word in words if _normalize_text(word["text"]) == "Stocks"
        ]
    else:
        raise ValueError(f"未知年鉴行类型：{row_kind}")
    if not anchors and anchor_top_hint is None:
        raise InputAuditError(f"年鉴页面未找到目标行锚点：{row_kind}")

    ordered_anchors = sorted(
        anchors,
        key=lambda item: (float(item["top"]), float(item["x0"])),
    )
    anchor_tops = [float(ordered_anchors[0]["top"])] if ordered_anchors else []
    if not anchor_tops and anchor_top_hint is not None:
        anchor_tops = [float(anchor_top_hint)]
    successful: list[tuple[float, float, list[float]]] = []
    for anchor_top in sorted(anchor_tops):
        try:
            centers = _header_centers(
                words,
                expected_months=expected_months,
                anchor_top=anchor_top,
            )
        except InputAuditError:
            continue
        gaps = np.diff(np.asarray(centers, dtype=float))
        if len(gaps) == 0 or not np.isfinite(gaps).all() or (gaps <= 0).any():
            continue
        typical_gap = float(np.median(gaps))
        lower_x = float(centers[0] - 1.1 * typical_gap)
        upper_x = float(centers[-1] + 1.1 * typical_gap)
        candidate_words = [
            word
            for word in words
            if anchor_top - 2.5 <= float(word["top"]) <= anchor_top + 8.5
        ]
        for group in _group_words_by_top(candidate_words, tolerance=0.5):
            values = _parse_sequential_decimal_values(
                group,
                lower_x=lower_x,
                upper_x=upper_x,
                expected_count=len(expected_months),
            )
            if values is None:
                continue
            row_top = float(np.mean([float(word["top"]) for word in group]))
            successful.append((abs(row_top - anchor_top), anchor_top, values))
    if not successful:
        raise InputAuditError(
            f"年鉴页面无法从{row_kind}锚点解析{len(expected_months)}个数值"
        )
    successful.sort(key=lambda item: item[0])
    best_distance, best_anchor_top, best_values = successful[0]
    conflicting = [
        values
        for distance, _anchor_top, values in successful[1:]
        if math.isclose(distance, best_distance, rel_tol=0.0, abs_tol=0.05)
        and not np.allclose(values, best_values, rtol=0.0, atol=1e-6)
    ]
    if conflicting:
        raise InputAuditError(f"年鉴页面{row_kind}存在等距冲突数值行")
    return best_values, best_anchor_top


def parse_szse_factbook(
    payload: bytes,
    *,
    year: int,
    specification: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """用几何位置解析深交所年鉴中的月度市值与股票成交额。"""

    if not payload.startswith(b"%PDF"):
        raise InputAuditError(f"深交所{year}年鉴缺少PDF文件头")
    market_pages = [int(value) for value in specification["market_pages_zero_based"]]
    transaction_pages = [
        int(value) for value in specification["transaction_pages_zero_based"]
    ]
    if len(market_pages) != 2 or len(transaction_pages) != 2:
        raise InputAuditError(f"深交所{year}年鉴冻结页码不是两页结构")
    month_sets = [list(range(1, 5)), list(range(5, 13))]
    market_values: list[float] = []
    turnover_values: list[float] = []
    with pdfplumber.open(BytesIO(payload)) as document:
        all_pages = market_pages + transaction_pages
        if min(all_pages) < 0 or max(all_pages) >= len(document.pages):
            raise InputAuditError(f"深交所{year}年鉴冻结页码越界")
        first_market_values, market_anchor_top = _extract_factbook_page_values(
            document.pages[market_pages[0]],
            expected_months=month_sets[0],
            row_kind="market_cap",
        )
        continuation_market_values, _ = _extract_factbook_page_values(
            document.pages[market_pages[1]],
            expected_months=month_sets[1],
            row_kind="market_cap",
            anchor_top_hint=market_anchor_top,
        )
        market_values.extend(first_market_values)
        market_values.extend(continuation_market_values)
        first_turnover_values, turnover_anchor_top = _extract_factbook_page_values(
            document.pages[transaction_pages[0]],
            expected_months=month_sets[0],
            row_kind="stock_turnover",
        )
        continuation_turnover_values, _ = _extract_factbook_page_values(
            document.pages[transaction_pages[1]],
            expected_months=month_sets[1],
            row_kind="stock_turnover",
            anchor_top_hint=turnover_anchor_top,
        )
        turnover_values.extend(first_turnover_values)
        turnover_values.extend(continuation_turnover_values)
    if len(market_values) != 12 or len(turnover_values) != 12:
        raise InputAuditError(f"深交所{year}年鉴未解析出12个月")
    if min(market_values + turnover_values) <= 0:
        raise InputAuditError(f"深交所{year}年鉴存在非正数")
    observed_annual = float(sum(turnover_values))
    expected_annual = float(specification["expected_annual_stock_turnover_cny_million"])
    annual_difference = abs(observed_annual - expected_annual)
    rows = [
        {
            "factor_month": f"{year:04d}-{month:02d}",
            "szse_stock_monthly_trade_amount_cny_100m": turnover_values[month - 1]
            / 100.0,
            "szse_stock_month_end_market_cap_cny_100m": market_values[month - 1]
            / 100.0,
        }
        for month in range(1, 13)
    ]
    return rows, {
        "year": year,
        "parsed_months": 12,
        "market_pages_zero_based": market_pages,
        "transaction_pages_zero_based": transaction_pages,
        "observed_annual_stock_turnover_cny_million": observed_annual,
        "expected_annual_stock_turnover_cny_million": expected_annual,
        "annual_absolute_difference_cny_million": annual_difference,
    }


def _html_rows(payload: bytes) -> list[list[str]]:
    try:
        document = lxml_html.fromstring(payload)
    except (ValueError, TypeError) as exc:
        raise InputAuditError(f"深交所静态月报HTML无法解析：{exc}") from exc
    rows: list[list[str]] = []
    for table_row in document.xpath("//tr"):
        cells = [
            " ".join(cell.text_content().split())
            for cell in table_row.xpath("./th|./td")
        ]
        if cells:
            rows.append(cells)
    if not rows:
        raise InputAuditError("深交所静态月报没有表格行")
    return rows


def _cny_yuan(value: str, *, label: str) -> int:
    text = _normalize_text(value).replace(",", "")
    if not re.fullmatch(r"[0-9]+", text):
        raise InputAuditError(f"{label}不是整数人民币金额：{value!r}")
    return int(text)


def _first_english_row(rows: list[list[str]], english_label: str) -> list[str]:
    matches = [row for row in rows if len(row) > 1 and row[1] == english_label]
    if not matches:
        raise InputAuditError(f"深交所静态月报缺少{english_label}行")
    return matches[0]


def _assert_static_month(payload: bytes, month: str) -> None:
    document = lxml_html.fromstring(payload)
    text = " ".join(document.text_content().split())
    timestamp = pd.Period(month, freq="M")
    month_name = _MONTH_NUMBER_TO_NAME[timestamp.month]
    accepted = [
        rf"{month_name}\D{{0,12}}{timestamp.year}",
        rf"{month_name[:3]}\D{{0,12}}{timestamp.year}",
        rf"\({timestamp.year}\.{timestamp.month:02d}\)",
    ]
    if not any(re.search(pattern, text) is not None for pattern in accepted):
        raise InputAuditError(
            f"深交所静态月报未回显月份{month_name} {timestamp.year}"
        )


def parse_szse_static_month(
    listed_payload: bytes,
    transaction_payload: bytes,
    *,
    month: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """解析2026年深交所HTTPS静态月报，并核对Stocks组成项。"""

    _assert_static_month(listed_payload, month)
    _assert_static_month(transaction_payload, month)
    listed_rows = _html_rows(listed_payload)
    transaction_rows = _html_rows(transaction_payload)
    component_labels = [
        "MB A-shares",
        "MB B-shares",
        "MB CDR",
        "ChiNext A-shares",
        "ChiNext CDR",
    ]

    listed_total = _cny_yuan(
        _first_english_row(listed_rows, "Stocks")[3],
        label=f"SZSE_LISTED_STOCKS_{month}",
    )
    listed_components = {
        label: _cny_yuan(
            _first_english_row(listed_rows, label)[3],
            label=f"SZSE_LISTED_{label}_{month}",
        )
        for label in component_labels
    }
    transaction_total = _cny_yuan(
        _first_english_row(transaction_rows, "Stocks")[3],
        label=f"SZSE_TRANSACTION_STOCKS_{month}",
    )
    transaction_components = {
        label: _cny_yuan(
            _first_english_row(transaction_rows, label)[3],
            label=f"SZSE_TRANSACTION_{label}_{month}",
        )
        for label in component_labels
    }
    listed_difference = listed_total - sum(listed_components.values())
    transaction_difference = transaction_total - sum(transaction_components.values())
    if abs(listed_difference) > 1 or abs(transaction_difference) > 1:
        raise InputAuditError(
            f"深交所{month} Stocks合计与组成项不一致："
            f"市值差{listed_difference}元，成交差{transaction_difference}元"
        )
    return (
        {
            "factor_month": month,
            "szse_stock_monthly_trade_amount_cny_100m": transaction_total / 1e8,
            "szse_stock_month_end_market_cap_cny_100m": listed_total / 1e8,
        },
        {
            "factor_month": month,
            "listed_component_difference_cny_yuan": listed_difference,
            "transaction_component_difference_cny_yuan": transaction_difference,
        },
    )


def _https_get_snapshot(
    *,
    source: str,
    period: str,
    url: str,
    referer: str,
    config: dict[str, Any],
    checkpoint: bool,
) -> ResponseSnapshot:
    if not url.lower().startswith("https://"):
        raise InputAuditError(f"正式官方数据源禁止非HTTPS地址：{url}")
    if checkpoint:
        cached = _read_checkpoint(source, period)
        if cached is not None:
            if cached.requested_url != url:
                raise InputAuditError(f"{source} {period}检查点URL与冻结URL不一致")
            return cached
    contract = config["data_contract"]
    last_error: Exception | None = None
    for attempt in range(1, int(contract["request_retry_limit"]) + 1):
        try:
            _respect_rate_limit(source, float(contract["sse_minimum_request_interval_seconds"]))
            response = _session().get(
                url,
                headers={"Referer": referer},
                timeout=int(contract["request_timeout_seconds"]),
            )
            response.raise_for_status()
            if not response.url.lower().startswith("https://"):
                raise InputAuditError(f"{source} {period}发生非HTTPS重定向")
            if not response.content:
                raise InputAuditError(f"{source} {period}响应为空")
            snapshot = ResponseSnapshot(
                source=source,
                period=period,
                requested_url=url,
                resolved_url=response.url,
                payload=response.content,
            )
            if checkpoint:
                _write_checkpoint(snapshot)
            return snapshot
        except (requests.RequestException, InputAuditError) as exc:
            last_error = exc
            if attempt < int(contract["request_retry_limit"]):
                time.sleep(0.5 * (2 ** (attempt - 1)))
    raise InputAuditError(
        f"{source} {period}在{contract['request_retry_limit']}次尝试后失败：{last_error}"
    )


def _factbook_snapshot(
    *,
    year: str,
    specification: dict[str, Any],
    config: dict[str, Any],
) -> tuple[ResponseSnapshot, str]:
    url = str(specification["url"])
    if not url.lower().startswith("https://"):
        raise InputAuditError(f"深交所{year}年鉴不是HTTPS地址")
    directory = ROOT / config["data_contract"]["szse"]["prefetched_factbook_directory"]
    prefetched = directory / f"szse_factbook_{year}.pdf"
    if prefetched.exists():
        payload = prefetched.read_bytes()
        snapshot = ResponseSnapshot(
            source="SZSE_FACTBOOK_PDF",
            period=year,
            requested_url=url,
            resolved_url=url,
            payload=payload,
            from_checkpoint=True,
        )
        return snapshot, "LOCAL_PREFETCH_FROM_CURRENT_RUN"
    snapshot = _https_get_snapshot(
        source="SZSE_FACTBOOK_PDF",
        period=year,
        url=url,
        referer=config["data_contract"]["szse"]["annual_factbook_index"],
        config=config,
        checkpoint=False,
    )
    return snapshot, "HTTPS_DOWNLOAD"


def _source_index_entry(
    snapshot: ResponseSnapshot,
    *,
    archive_member: str,
    retrieval_mode: str,
) -> dict[str, Any]:
    return {
        "archive_member": archive_member,
        "source": snapshot.source,
        "period": snapshot.period,
        "requested_url": snapshot.requested_url,
        "resolved_url": snapshot.resolved_url,
        "response_sha256": sha256_bytes(snapshot.payload),
        "response_bytes": len(snapshot.payload),
        "retrieval_mode": retrieval_mode,
        "tls_verification_disabled": False,
    }


def _atomic_deterministic_zip(path: Path, entries: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for member in sorted(entries):
                information = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                information.compress_type = zipfile.ZIP_DEFLATED
                information.create_system = 3
                information.external_attr = 0o100644 << 16
                archive.writestr(information, entries[member], compresslevel=9)
        with zipfile.ZipFile(temporary, mode="r") as archive:
            broken = archive.testzip()
            if broken is not None:
                raise InputAuditError(f"官方源归档ZIP CRC失败：{broken}")
            if sorted(archive.namelist()) != sorted(entries):
                raise InputAuditError("官方源归档ZIP成员清单不一致")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _collect_sse_month(
    month: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[ResponseSnapshot]]:
    parsed, snapshot = _sse_monthly_request(month, config)
    return (
        {
            "factor_month": month,
            "sse_a_monthly_trade_amount_cny_100m": float(
                parsed["trade_amount_cny_100m"]
            ),
            "sse_a_month_end_market_cap_cny_100m": float(
                parsed["market_cap_cny_100m"]
            ),
            "sse_identical_duplicate_rows_collapsed": int(
                parsed["identical_duplicate_rows_collapsed"]
            ),
            "sse_source": snapshot.source,
            "sse_response_sha256": sha256_bytes(snapshot.payload),
        },
        [snapshot],
    )


def build() -> dict[str, Any]:
    config = load_config()
    if MANIFEST_PATH.exists():
        raise InputAuditError(f"协议已经冻结，禁止覆盖输入：{MANIFEST_PATH}")
    output_paths = [
        _path(config, "official_source_archive"),
        _path(config, "official_monthly_components"),
        _path(config, "monthly_factor"),
        _path(config, "input_audit"),
    ]
    preexisting = [path.relative_to(ROOT).as_posix() for path in output_paths if path.exists()]
    if preexisting:
        raise InputAuditError(f"输入产物已存在，禁止覆盖：{preexisting}")

    calendar = _load_calendar(config)
    months = calendar["date"].dt.to_period("M").astype(str).drop_duplicates().tolist()
    contract = config["data_contract"]
    expected_months = pd.period_range(
        start=str(contract["expected_first_factor_month"]),
        end=str(contract["expected_last_factor_month"]),
        freq="M",
    ).astype(str).tolist()
    if months != expected_months:
        raise InputAuditError("510300日期列形成的月序列与冻结因子月序列不一致")

    sse_rows, sse_snapshots = _parallel_collect(
        months,
        _collect_sse_month,
        config,
        label="上交所官方月度采集",
    )

    archive_entries: dict[str, bytes] = {}
    source_index_entries: list[dict[str, Any]] = []
    for snapshot in sse_snapshots:
        member = f"sse/monthly/{snapshot.period}.json"
        if member in archive_entries:
            raise InputAuditError(f"官方源归档成员重复：{member}")
        archive_entries[member] = snapshot.payload
        source_index_entries.append(
            _source_index_entry(
                snapshot,
                archive_member=member,
                retrieval_mode=(
                    "VERIFIED_JSON_CHECKPOINT"
                    if snapshot.from_checkpoint
                    else "HTTPS_DOWNLOAD"
                ),
            )
        )

    szse_contract = contract["szse"]
    factbook_details: list[dict[str, Any]] = []
    szse_rows: list[dict[str, Any]] = []
    factbook_snapshots: list[ResponseSnapshot] = []
    for year, specification in sorted(szse_contract["factbooks"].items()):
        print(f"解析深交所{year}年统计年鉴", flush=True)
        snapshot, retrieval_mode = _factbook_snapshot(
            year=year,
            specification=specification,
            config=config,
        )
        parsed_rows, detail = parse_szse_factbook(
            snapshot.payload,
            year=int(year),
            specification=specification,
        )
        detail["source_sha256"] = sha256_bytes(snapshot.payload)
        detail["source_bytes"] = len(snapshot.payload)
        detail["retrieval_mode"] = retrieval_mode
        factbook_details.append(detail)
        factbook_snapshots.append(snapshot)
        member = f"szse/factbooks/{year}.pdf"
        archive_entries[member] = snapshot.payload
        source_index_entries.append(
            _source_index_entry(
                snapshot,
                archive_member=member,
                retrieval_mode=retrieval_mode,
            )
        )
        source_hash = sha256_bytes(snapshot.payload)
        for row in parsed_rows:
            row["szse_source"] = "SZSE_OFFICIAL_ANNUAL_FACTBOOK"
            row["szse_market_source_sha256"] = source_hash
            row["szse_trade_source_sha256"] = source_hash
            szse_rows.append(row)

    static_details: list[dict[str, Any]] = []
    static_snapshots: list[ResponseSnapshot] = []
    for month, specification in sorted(szse_contract["static_months"].items()):
        print(f"采集并解析深交所{month}静态月报", flush=True)
        listed_snapshot = _https_get_snapshot(
            source="SZSE_STATIC_LISTED_MONTHLY",
            period=f"{month}-listed",
            url=str(specification["listed_url"]),
            referer=szse_contract["static_monthly_index"],
            config=config,
            checkpoint=True,
        )
        transaction_snapshot = _https_get_snapshot(
            source="SZSE_STATIC_TRANSACTION_MONTHLY",
            period=f"{month}-transaction",
            url=str(specification["transaction_url"]),
            referer=szse_contract["static_monthly_index"],
            config=config,
            checkpoint=True,
        )
        row, detail = parse_szse_static_month(
            listed_snapshot.payload,
            transaction_snapshot.payload,
            month=month,
        )
        listed_hash = sha256_bytes(listed_snapshot.payload)
        transaction_hash = sha256_bytes(transaction_snapshot.payload)
        row["szse_source"] = "SZSE_OFFICIAL_HTTPS_STATIC_MONTHLY"
        row["szse_market_source_sha256"] = listed_hash
        row["szse_trade_source_sha256"] = transaction_hash
        szse_rows.append(row)
        detail.update(
            {
                "listed_sha256": listed_hash,
                "transaction_sha256": transaction_hash,
            }
        )
        static_details.append(detail)
        static_snapshots.extend([listed_snapshot, transaction_snapshot])
        for snapshot, suffix in [
            (listed_snapshot, "listed.html"),
            (transaction_snapshot, "transaction.html"),
        ]:
            member = f"szse/monthly/{month}/{suffix}"
            archive_entries[member] = snapshot.payload
            source_index_entries.append(
                _source_index_entry(
                    snapshot,
                    archive_member=member,
                    retrieval_mode=(
                        "VERIFIED_BINARY_CHECKPOINT"
                        if snapshot.from_checkpoint
                        else "HTTPS_DOWNLOAD"
                    ),
                )
            )

    sse = pd.DataFrame(sse_rows).sort_values("factor_month").reset_index(drop=True)
    szse = pd.DataFrame(szse_rows)
    szse = szse.loc[szse["factor_month"].isin(expected_months)].copy()
    szse = szse.sort_values("factor_month").reset_index(drop=True)
    components = sse.merge(szse, on="factor_month", how="inner", validate="one_to_one")
    month_ends = (
        calendar.assign(
            factor_month=calendar["date"].dt.to_period("M").astype(str)
        )
        .groupby("factor_month", sort=True, as_index=False)
        .tail(1)[["factor_month", "date"]]
        .rename(columns={"date": "month_end_date"})
    )
    components = components.merge(
        month_ends,
        on="factor_month",
        how="inner",
        validate="one_to_one",
    )
    components["aggregate_monthly_trade_amount_cny_100m"] = (
        components["sse_a_monthly_trade_amount_cny_100m"]
        + components["szse_stock_monthly_trade_amount_cny_100m"]
    )
    components["aggregate_month_end_market_cap_cny_100m"] = (
        components["sse_a_month_end_market_cap_cny_100m"]
        + components["szse_stock_month_end_market_cap_cny_100m"]
    )
    components = components.sort_values("factor_month").reset_index(drop=True)
    monthly_factor = build_monthly_factor_from_components(components, config)

    source_index = {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "transport": "HTTPS_ONLY_TLS_VERIFICATION_REQUIRED",
        "historical_availability_status": contract["historical_availability_status"],
        "historical_archive_reconstruction_not_timestamp_proof": True,
        "candidate_outcomes_computed": False,
        "portfolio_returns_computed": False,
        "entries": sorted(source_index_entries, key=lambda item: item["archive_member"]),
    }
    archive_entries["source_index.json"] = (
        json.dumps(
            source_index,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    component_numeric_columns = [
        "sse_a_monthly_trade_amount_cny_100m",
        "sse_a_month_end_market_cap_cny_100m",
        "szse_stock_monthly_trade_amount_cny_100m",
        "szse_stock_month_end_market_cap_cny_100m",
        "aggregate_monthly_trade_amount_cny_100m",
        "aggregate_month_end_market_cap_cny_100m",
    ]
    annual_tolerance = float(szse_contract["annual_reconciliation_tolerance_cny_million"])
    all_snapshots = sse_snapshots + factbook_snapshots + static_snapshots
    checks = {
        "calendar_rows_exact": len(calendar) == int(contract["expected_trading_dates"]),
        "calendar_dates_unique": not calendar["date"].duplicated().any(),
        "calendar_bounds_exact": (
            calendar["date"].min().date().isoformat()
            == contract["expected_first_trading_date"]
            and calendar["date"].max().date().isoformat()
            == contract["expected_last_trading_date"]
        ),
        "factor_month_sequence_exact": months == expected_months,
        "sse_month_rows_exact": len(sse) == int(contract["expected_complete_months"]),
        "sse_months_unique": not sse["factor_month"].duplicated().any(),
        "szse_month_rows_exact": len(szse) == int(contract["expected_complete_months"]),
        "szse_months_unique": not szse["factor_month"].duplicated().any(),
        "component_rows_exact": len(components)
        == int(contract["expected_complete_months"]),
        "component_month_sequence_exact": components["factor_month"].tolist()
        == expected_months,
        "component_values_complete_positive": (
            not components[component_numeric_columns].isna().any().any()
            and bool((components[component_numeric_columns] > 0).all().all())
        ),
        "factbook_years_exact": len(factbook_details) == 12,
        "factbook_pdf_magic_all": all(
            snapshot.payload.startswith(b"%PDF") for snapshot in factbook_snapshots
        ),
        "factbook_twelve_months_each": all(
            detail["parsed_months"] == 12 for detail in factbook_details
        ),
        "factbook_annual_reconciliation": all(
            detail["annual_absolute_difference_cny_million"] <= annual_tolerance
            for detail in factbook_details
        ),
        "static_months_exact": len(static_details) == 7,
        "static_stocks_component_reconciliation": all(
            abs(detail["listed_component_difference_cny_yuan"]) <= 1
            and abs(detail["transaction_component_difference_cny_yuan"]) <= 1
            for detail in static_details
        ),
        "monthly_factor_rows_exact": len(monthly_factor)
        == int(contract["expected_complete_months"]),
        "monthly_factor_bounds_exact": (
            monthly_factor["factor_month"].iloc[0]
            == contract["expected_first_factor_month"]
            and monthly_factor["factor_month"].iloc[-1]
            == contract["expected_last_factor_month"]
        ),
        "monthly_factor_complete_positive": (
            not monthly_factor[
                [
                    "aggregate_monthly_trade_amount_cny_100m",
                    "aggregate_month_end_market_cap_cny_100m",
                    "aggregate_turnover_ratio",
                ]
            ]
            .isna()
            .any()
            .any()
            and bool(
                (
                    monthly_factor[
                        [
                            "aggregate_monthly_trade_amount_cny_100m",
                            "aggregate_month_end_market_cap_cny_100m",
                            "aggregate_turnover_ratio",
                        ]
                    ]
                    > 0
                )
                .all()
                .all()
            )
        ),
        "target_weights_in_bounds": bool(
            monthly_factor["target_weight_before_lot_rounding"].between(0.0, 1.0).all()
        ),
        "all_requested_urls_https": all(
            entry["requested_url"].lower().startswith("https://")
            for entry in source_index_entries
        ),
        "all_resolved_urls_https": all(
            entry["resolved_url"].lower().startswith("https://")
            for entry in source_index_entries
        ),
        "tls_verification_never_disabled": all(
            entry["tls_verification_disabled"] is False
            for entry in source_index_entries
        ),
        "source_archive_members_unique": len(archive_entries)
        == len(set(archive_entries)),
        "historical_archive_limitation_explicit": contract[
            "historical_availability_status"
        ]
        == "HISTORICAL_ARCHIVE_RECONSTRUCTION_NOT_TIMESTAMP_PROOF",
        "candidate_outcomes_not_computed": True,
        "portfolio_returns_not_computed": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "report_id": "510300_AGGREGATE_TURNOVER_VISIBILITY_INPUTS_V1",
        "status": status,
        "checked_at_asia_shanghai": datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "candidate_outcomes_computed": False,
        "portfolio_returns_computed": False,
        "calendar": {
            "source_path": PRICE_PATH.relative_to(ROOT).as_posix(),
            "only_date_column_read": True,
            "price_columns_read": False,
            "return_columns_read": False,
            "rows": int(len(calendar)),
            "first_date": calendar["date"].min().date().isoformat(),
            "last_date": calendar["date"].max().date().isoformat(),
            "source_sha256": sha256_file(PRICE_PATH),
            "source_audit_sha256": sha256_file(PRICE_AUDIT_PATH),
        },
        "official_monthly_components": {
            "rows": int(len(components)),
            "first_month": str(components["factor_month"].iloc[0]),
            "last_month": str(components["factor_month"].iloc[-1]),
            "sse_legacy_months": int(
                (sse["sse_source"] == "SSE_LEGACY_MONTHLY_A_CROSSCHECK").sum()
            ),
            "sse_current_months": int(
                (sse["sse_source"] == "SSE_CURRENT_MONTHLY_A_CROSSCHECK").sum()
            ),
            "sse_identical_duplicate_rows_collapsed": int(
                sse["sse_identical_duplicate_rows_collapsed"].sum()
            ),
            "sse_months_with_identical_duplicates": sse.loc[
                sse["sse_identical_duplicate_rows_collapsed"] > 0,
                "factor_month",
            ].astype(str).tolist(),
            "szse_frozen_scope": "STOCK_TOTAL_INCLUDING_B_AND_CDR",
        },
        "monthly_factor": {
            "rows": int(len(monthly_factor)),
            "first_month": str(monthly_factor["factor_month"].iloc[0]),
            "last_month": str(monthly_factor["factor_month"].iloc[-1]),
            "turnover_minimum": float(monthly_factor["aggregate_turnover_ratio"].min()),
            "turnover_maximum": float(monthly_factor["aggregate_turnover_ratio"].max()),
            "forecast_minimum": float(
                monthly_factor["external_forecast_monthly_return"].min()
            ),
            "forecast_maximum": float(
                monthly_factor["external_forecast_monthly_return"].max()
            ),
            "target_weight_minimum": float(
                monthly_factor["target_weight_before_lot_rounding"].min()
            ),
            "target_weight_maximum": float(
                monthly_factor["target_weight_before_lot_rounding"].max()
            ),
        },
        "szse_factbook_reconciliation": {
            "annual_tolerance_cny_million": annual_tolerance,
            "years": factbook_details,
            "maximum_annual_absolute_difference_cny_million": float(
                max(
                    detail["annual_absolute_difference_cny_million"]
                    for detail in factbook_details
                )
            ),
        },
        "szse_static_month_reconciliation": static_details,
        "source_contract": config["data_contract"],
        "official_source_archive": {
            "source_records": int(len(source_index_entries)),
            "archive_members_including_index": int(len(archive_entries)),
            "factbook_pdf_records": int(len(factbook_snapshots)),
            "sse_monthly_records": int(len(sse_snapshots)),
            "szse_static_html_records": int(len(static_snapshots)),
            "tls_verification_disabled": False,
            "verified_checkpoint_hits": int(
                sum(snapshot.from_checkpoint for snapshot in sse_snapshots + static_snapshots)
            ),
            "network_downloads": int(
                sum(
                    not snapshot.from_checkpoint
                    for snapshot in sse_snapshots + factbook_snapshots + static_snapshots
                )
            ),
            "local_prefetched_factbooks_from_current_run": int(
                sum(
                    detail["retrieval_mode"] == "LOCAL_PREFETCH_FROM_CURRENT_RUN"
                    for detail in factbook_details
                )
            ),
        },
        "checks": checks,
        "boundaries": {
            "factor_computed": True,
            "future_510300_returns_read": False,
            "candidate_outcomes_computed": False,
            "strategy_returns_read": False,
            "parameter_selection_performed": False,
            "historical_archive_reconstruction_not_timestamp_proof": True,
            "forward_month_end_capture_required": True,
            "live_trading_authorized": False,
        },
    }
    if status != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        report["failed_checks"] = failed
        raise InputAuditError(json.dumps(report, ensure_ascii=False, allow_nan=False))

    archive_path = _path(config, "official_source_archive")
    components_path = _path(config, "official_monthly_components")
    monthly_path = _path(config, "monthly_factor")
    audit_path = _path(config, "input_audit")
    print("写入并校验官方源ZIP归档", flush=True)
    _atomic_deterministic_zip(archive_path, archive_entries)
    _atomic_parquet(components_path, components)
    _atomic_parquet(monthly_path, monthly_factor)
    report["output_files"] = {
        archive_path.relative_to(ROOT).as_posix(): {
            "sha256": sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
        },
        components_path.relative_to(ROOT).as_posix(): {
            "sha256": sha256_file(components_path),
            "bytes": components_path.stat().st_size,
        },
        monthly_path.relative_to(ROOT).as_posix(): {
            "sha256": sha256_file(monthly_path),
            "bytes": monthly_path.stat().st_size,
        },
    }
    _atomic_json(audit_path, report)
    return report


def main() -> int:
    report = build()
    print(
        json.dumps(
            {
                "status": report["status"],
                "official_monthly_component_rows": report[
                    "official_monthly_components"
                ]["rows"],
                "monthly_factor_rows": report["monthly_factor"]["rows"],
                "official_source_records": report["official_source_archive"][
                    "source_records"
                ],
                "candidate_outcomes_computed": False,
                "portfolio_returns_computed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
