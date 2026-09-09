"""采集510300双影子规则2012至2021反向验证快照。

本程序只采集并核验输入，不计算候选特征、仓位或收益。所有网络响应正文按
请求粒度保存为只新增文件；同一run-id中断后可复用已经通过结构校验的正文，
但绝不覆盖。现有冻结候选、原始研究数据和前瞻数据均不修改。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_REVERSE_VALIDATION_V1"
CONFIG_FILE = (
    PROJECT_ROOT
    / "config"
    / "510300_etf_microstructure_dual_shadow_reverse_validation_v1.yaml"
)
FREEZE_MANIFEST = CONFIG_FILE.with_name(
    "510300_etf_microstructure_dual_shadow_reverse_validation_v1_manifest.json"
)
TIMEZONE = ZoneInfo("Asia/Shanghai")

SSE_COMMON_URL = "https://query.sse.com.cn/commonQuery.do"
SSE_SHARE_SQL_ID = "COMMON_SSE_ZQPZ_ETFZL_ETFJBXX_JJGM_SEARCH_L"
SSE_MARGIN_URL = "https://query.sse.com.cn/marketdata/tradedata/queryMargin.do"
EASTMONEY_NAV_URL = "https://api.fund.eastmoney.com/f10/lsjz"
SINA_NAV_URL = (
    "http://stock.finance.sina.com.cn/fundInfo/api/openapi.php/"
    "CaihuiFundInfoService.getNav"
)
SSE_HEADERS = {
    "Referer": "https://www.sse.com.cn/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/136.0 Safari/537.36"
    ),
}
EASTMONEY_HEADERS = {
    "Referer": "https://fundf10.eastmoney.com/jjjz_510300.html",
    "User-Agent": "Mozilla/5.0",
}
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}
_THREAD_LOCAL = threading.local()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def _atomic_status(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_new_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖只新增文件：{path}")
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new_json(payload: dict[str, Any] | list[Any], path: Path) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    ).encode("utf-8")
    _write_new_bytes(content, path)


def _write_new_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖快照文件：{path}")
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.rename(path)


def _copy_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"拒绝覆盖快照文件：{target}")
    with source.open("rb") as input_handle, target.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle)
        output_handle.flush()
        os.fsync(output_handle.fileno())


def _session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def _decode_json(content: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except Exception as exc:
        raise ValueError(f"{label}响应不是有效JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}响应顶层不是对象")
    return payload


def _raw_receipt(
    *,
    kind: str,
    identifier: str,
    path: Path,
    url: str,
    params: dict[str, str],
    reused: bool,
) -> dict[str, Any]:
    stored_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    return {
        "kind": kind,
        "identifier": identifier,
        "file": _relative(path),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
        "stored_at_utc_from_file_mtime": stored_at,
        "source_url": url,
        "request_params": params,
        "reused_from_interrupted_run": reused,
        "body_representation": "HTTP响应解压后的正文原始字节",
    }


def _fetch_json_body(
    *,
    kind: str,
    identifier: str,
    raw_path: Path,
    url: str,
    params: dict[str, str],
    headers: dict[str, str],
    validator: Callable[[dict[str, Any]], None],
    attempts: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if raw_path.exists():
        payload = _decode_json(raw_path.read_bytes(), f"{kind}/{identifier}")
        validator(payload)
        return payload, _raw_receipt(
            kind=kind,
            identifier=identifier,
            path=raw_path,
            url=url,
            params=params,
            reused=True,
        )

    final_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = _session().get(
                url,
                params=params,
                headers=headers,
                timeout=45,
            )
            response.raise_for_status()
            content = response.content
            payload = _decode_json(content, f"{kind}/{identifier}")
            validator(payload)
            _write_new_bytes(content, raw_path)
            return payload, _raw_receipt(
                kind=kind,
                identifier=identifier,
                path=raw_path,
                url=url,
                params=params,
                reused=False,
            )
        except Exception as exc:
            final_error = exc
            if attempt < attempts:
                time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))
    raise RuntimeError(
        f"{kind}/{identifier}请求在{attempts}次尝试后失败："
        f"{type(final_error).__name__}: {final_error}"
    )


def _load_and_verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    if not CONFIG_FILE.exists() or not FREEZE_MANIFEST.exists():
        raise FileNotFoundError("反向验证配置或冻结清单不存在")
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != PROJECT_ID:
        raise ValueError("反向验证项目编号不匹配")
    if manifest.get("project_id") != PROJECT_ID or not manifest.get(
        "implementation_frozen", False
    ):
        raise ValueError("反向验证冻结清单无效")
    mismatches: dict[str, str] = {}
    for group in ["tracked_files", "historical_input_files_at_freeze"]:
        for relative, expected in manifest[group].items():
            path = PROJECT_ROOT / relative
            actual = _sha256(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = f"expected={expected},actual={actual}"
    if mismatches:
        raise ValueError(f"反向验证冻结文件哈希漂移：{mismatches}")
    return config, manifest


def _normalize(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result.sort_values(column, inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result[column].duplicated().any():
        raise ValueError(f"{column}存在重复日期")
    return result


def _number(value: Any, label: str) -> float:
    try:
        number = float(str(value).replace(",", ""))
    except Exception as exc:
        raise ValueError(f"{label}不是数值：{value}") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label}不是有限数值")
    return number


def _collect_share_date(
    date: pd.Timestamp,
    raw_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    date_text = date.strftime("%Y-%m-%d")
    params = {
        "isPagination": "false",
        "sqlId": SSE_SHARE_SQL_ID,
        "SEC_CODE": "510300",
        "STAT_DATE": date_text,
    }

    def validate(payload: dict[str, Any]) -> None:
        rows = payload.get("result") or []
        target = [
            row
            for row in rows
            if str(row.get("SEC_CODE")) == "510300"
            and str(row.get("STAT_DATE")) == date_text
        ]
        if len(target) != 1:
            raise ValueError(f"份额记录数不是1：{len(target)}")
        if _number(target[0].get("TOT_VOL"), "TOT_VOL") <= 0:
            raise ValueError("TOT_VOL不是正数")

    raw_path = raw_dir / "fund_share" / f"{date.strftime('%Y%m%d')}.json"
    payload, receipt = _fetch_json_body(
        kind="fund_share",
        identifier=date_text,
        raw_path=raw_path,
        url=SSE_COMMON_URL,
        params=params,
        headers=SSE_HEADERS,
        validator=validate,
    )
    row = next(
        item
        for item in payload["result"]
        if str(item.get("SEC_CODE")) == "510300"
        and str(item.get("STAT_DATE")) == date_text
    )
    share_10k = _number(row["TOT_VOL"], "TOT_VOL")
    record = {
        "date": date,
        "ts_code": "510300.SH",
        "fund_share_10k": share_10k,
        "fund_shares": share_10k * 10_000.0,
        "fund_type": "ETF",
        "market": "SH",
        "source": f"sse.{SSE_SHARE_SQL_ID}",
    }
    return record, receipt


def _collect_shares(
    dates: list[pd.Timestamp],
    raw_dir: Path,
    workers: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: dict[pd.Timestamp, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_collect_share_date, date, raw_dir): date for date in dates
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            date = futures[future]
            record, receipt = future.result()
            rows[date] = record
            receipts.append(receipt)
            if completed % 100 == 0 or completed == len(dates):
                print(
                    f"份额采集进度：{completed}/{len(dates)}",
                    flush=True,
                )
    output = pd.DataFrame([rows[date] for date in dates])
    output.sort_values("date", inplace=True)
    output.reset_index(drop=True, inplace=True)
    if len(output) != len(dates) or output["date"].duplicated().any():
        raise ValueError("份额结果与交易日历不一致")
    return output, receipts


def _margin_params(start: pd.Timestamp, end: pd.Timestamp, page: int) -> dict[str, str]:
    return {
        "isPagination": "true",
        "tabType": "mxtype",
        "detailsDate": "",
        "stockCode": "510300",
        "beginDate": start.strftime("%Y%m%d"),
        "endDate": end.strftime("%Y%m%d"),
        "pageHelp.pageSize": "2000",
        "pageHelp.pageCount": "50",
        "pageHelp.pageNo": str(page),
        "pageHelp.beginPage": str(page),
        "pageHelp.cacheSize": "1",
        "pageHelp.endPage": str(page + 20),
    }


def _collect_margin(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dates: list[pd.Timestamp],
    close_by_date: dict[pd.Timestamp, float],
    raw_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    receipts: list[dict[str, Any]] = []

    def validate(payload: dict[str, Any]) -> None:
        if not isinstance(payload.get("result"), list):
            raise ValueError("融资融券响应缺少result数组")
        if not isinstance(payload.get("pageHelp"), dict):
            raise ValueError("融资融券响应缺少pageHelp对象")

    first_params = _margin_params(start, end, 1)
    first, receipt = _fetch_json_body(
        kind="margin",
        identifier="page_0001",
        raw_path=raw_dir / "margin" / "page_0001.json",
        url=SSE_MARGIN_URL,
        params=first_params,
        headers=SSE_HEADERS,
        validator=validate,
    )
    receipts.append(receipt)
    page_help = first["pageHelp"]
    page_count = int(page_help.get("pageCount") or 0)
    total = int(page_help.get("total") or 0)
    if page_count <= 0 or total <= 0:
        raise ValueError("融资融券分页元数据无效")
    payloads: dict[int, dict[str, Any]] = {1: first}
    for page in range(2, page_count + 1):
        params = _margin_params(start, end, page)
        payload, page_receipt = _fetch_json_body(
            kind="margin",
            identifier=f"page_{page:04d}",
            raw_path=raw_dir / "margin" / f"page_{page:04d}.json",
            url=SSE_MARGIN_URL,
            params=params,
            headers=SSE_HEADERS,
            validator=validate,
        )
        payloads[page] = payload
        receipts.append(page_receipt)
    source_rows = [
        row
        for page in range(1, page_count + 1)
        for row in (payloads[page].get("result") or [])
    ]
    if len(source_rows) != total:
        raise ValueError(f"融资融券分页记录数{len(source_rows)}不等于total={total}")

    records: list[dict[str, Any]] = []
    for source in source_rows:
        if str(source.get("stockCode")) != "510300":
            raise ValueError("融资融券响应混入非510300记录")
        date = pd.to_datetime(str(source.get("opDate")), format="%Y%m%d", errors="raise")
        if date not in close_by_date:
            raise ValueError(f"融资融券日期不在510300交易日历：{date.date()}")
        numeric = {
            name: _number(source.get(name), name)
            for name in ["rzye", "rzmre", "rzche", "rqyl", "rqchl", "rqmcl"]
        }
        if any(value < 0 for value in numeric.values()):
            raise ValueError(f"{date.date()}融资融券字段存在负数")
        rqye = numeric["rqyl"] * close_by_date[date]
        records.append(
            {
                "date": date,
                "ts_code": "510300.SH",
                "rzye": numeric["rzye"],
                "rqye": rqye,
                "rzmre": numeric["rzmre"],
                "rqyl": numeric["rqyl"],
                "rzche": numeric["rzche"],
                "rqchl": numeric["rqchl"],
                "rqmcl": numeric["rqmcl"],
                "rzrqye": numeric["rzye"] + rqye,
                "financing_net_buy_cny": numeric["rzmre"] - numeric["rzche"],
                "source": "sse.queryMargin.do;rqye=rqyl*510300_close",
            }
        )
    output = _normalize(pd.DataFrame(records))
    expected = pd.DatetimeIndex(dates)
    actual = pd.DatetimeIndex(output["date"])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            f"融资融券与交易日历不一致：missing={missing.strftime('%Y-%m-%d').tolist()[:20]},"
            f"extra={extra.strftime('%Y-%m-%d').tolist()[:20]}"
        )
    print(f"融资融券采集完成：{len(output)}行，{page_count}页", flush=True)
    return output, receipts


def _eastmoney_page_params(
    start: pd.Timestamp,
    end: pd.Timestamp,
    page: int,
) -> dict[str, str]:
    return {
        "fundCode": "510300",
        "pageIndex": str(page),
        "pageSize": "20",
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
    }


def _sina_page_params(
    start: pd.Timestamp,
    end: pd.Timestamp,
    page: int,
) -> dict[str, str]:
    return {
        "symbol": "510300",
        "datefrom": start.strftime("%Y-%m-%d"),
        "dateto": end.strftime("%Y-%m-%d"),
        "page": str(page),
        "num": "200",
    }


def _collect_paginated_nav(
    *,
    kind: str,
    url: str,
    raw_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if kind not in {"nav_eastmoney", "nav_sina"}:
        raise ValueError(f"未知净值来源：{kind}")
    params_builder = _eastmoney_page_params if kind == "nav_eastmoney" else _sina_page_params
    headers = EASTMONEY_HEADERS if kind == "nav_eastmoney" else DEFAULT_HEADERS

    def validate(payload: dict[str, Any]) -> None:
        if kind == "nav_eastmoney":
            if not isinstance((payload.get("Data") or {}).get("LSJZList"), list):
                raise ValueError("东方财富净值响应缺少LSJZList")
        else:
            result = payload.get("result") or {}
            status = result.get("status") or {}
            if int(status.get("code", -1)) != 0:
                raise ValueError("新浪净值响应状态不是0")
            if not isinstance((result.get("data") or {}).get("data"), list):
                raise ValueError("新浪净值响应缺少data数组")

    def fetch_page(page: int) -> tuple[int, dict[str, Any], dict[str, Any]]:
        params = params_builder(start, end, page)
        payload, receipt = _fetch_json_body(
            kind=kind,
            identifier=f"page_{page:04d}",
            raw_path=raw_dir / kind / f"page_{page:04d}.json",
            url=url,
            params=params,
            headers=headers,
            validator=validate,
        )
        return page, payload, receipt

    _, first, first_receipt = fetch_page(1)
    if kind == "nav_eastmoney":
        total = int(first.get("TotalCount") or 0)
        page_size = len((first.get("Data") or {}).get("LSJZList") or [])
    else:
        body = ((first.get("result") or {}).get("data") or {})
        total = int(body.get("total_num") or 0)
        page_size = len(body.get("data") or [])
    if total <= 0 or page_size <= 0:
        raise ValueError(f"{kind}分页元数据无效")
    page_count = math.ceil(total / page_size)
    payloads: dict[int, dict[str, Any]] = {1: first}
    receipts: list[dict[str, Any]] = [first_receipt]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_page, page): page for page in range(2, page_count + 1)}
        for completed, future in enumerate(as_completed(futures), start=2):
            page, payload, receipt = future.result()
            payloads[page] = payload
            receipts.append(receipt)
            if completed % 25 == 0 or completed == page_count:
                print(f"{kind}采集进度：{completed}/{page_count}页", flush=True)
    records: list[dict[str, Any]] = []
    for page in range(1, page_count + 1):
        payload = payloads[page]
        if kind == "nav_eastmoney":
            records.extend((payload.get("Data") or {}).get("LSJZList") or [])
        else:
            records.extend((((payload.get("result") or {}).get("data") or {}).get("data") or []))
    if len(records) != total:
        raise ValueError(f"{kind}记录数{len(records)}不等于total={total}")
    return records, receipts, page_count


def _collect_nav(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dates: list[pd.Timestamp],
    close_by_date: dict[pd.Timestamp, float],
    raw_dir: Path,
    workers: int,
    tolerance: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    east_rows, east_receipts, east_pages = _collect_paginated_nav(
        kind="nav_eastmoney",
        url=EASTMONEY_NAV_URL,
        raw_dir=raw_dir,
        start=start,
        end=end,
        workers=workers,
    )
    sina_rows, sina_receipts, sina_pages = _collect_paginated_nav(
        kind="nav_sina",
        url=SINA_NAV_URL,
        raw_dir=raw_dir,
        start=start,
        end=end,
        workers=workers,
    )
    east = pd.DataFrame(east_rows).rename(
        columns={"FSRQ": "date", "DWJZ": "nav_eastmoney", "LJJZ": "acc_nav_eastmoney"}
    )
    sina = pd.DataFrame(sina_rows).rename(
        columns={"fbrq": "date", "jjjz": "nav_sina", "ljjz": "acc_nav_sina"}
    )
    east = east[["date", "nav_eastmoney", "acc_nav_eastmoney"]].copy()
    sina = sina[["date", "nav_sina", "acc_nav_sina"]].copy()
    for frame, columns in [
        (east, ["nav_eastmoney", "acc_nav_eastmoney"]),
        (sina, ["nav_sina", "acc_nav_sina"]),
    ]:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame["date"].duplicated().any():
            raise ValueError("净值来源存在重复日期")
    merged = east.merge(sina, on="date", how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        bad = merged.loc[~merged["_merge"].eq("both"), ["date", "_merge"]]
        raise ValueError(f"两来源净值日期不一致：{bad.head(20).to_dict('records')}")
    unit_difference = (merged["nav_eastmoney"] - merged["nav_sina"]).abs()
    accumulated_difference = (
        merged["acc_nav_eastmoney"] - merged["acc_nav_sina"]
    ).abs()
    if (unit_difference > tolerance).any() or (accumulated_difference > tolerance).any():
        raise ValueError("两来源净值差异超过冻结容差")
    expected = pd.DatetimeIndex(dates)
    available = pd.DatetimeIndex(merged["date"])
    missing = expected.difference(available)
    if len(missing):
        raise ValueError(f"交易日净值缺失：{missing.strftime('%Y-%m-%d').tolist()[:20]}")
    output = merged.loc[merged["date"].isin(expected)].copy()
    output["unit_nav"] = output["nav_eastmoney"]
    output["accumulated_nav"] = output["acc_nav_eastmoney"]
    output["close"] = output["date"].map(close_by_date)
    output["close_premium_to_nav"] = output["close"] / output["unit_nav"] - 1.0
    output["close_premium_bps"] = output["close_premium_to_nav"] * 10_000.0
    output["symbol"] = "510300.SH"
    output["source_primary"] = "eastmoney.f10.lsjz"
    output["source_secondary"] = "sina.CaihuiFundInfoService.getNav"
    output.sort_values("date", inplace=True)
    output.reset_index(drop=True, inplace=True)
    audit = {
        "eastmoney_source_rows": int(len(east)),
        "sina_source_rows": int(len(sina)),
        "signal_calendar_rows": int(len(output)),
        "eastmoney_pages": east_pages,
        "sina_pages": sina_pages,
        "maximum_unit_nav_absolute_difference": float(unit_difference.max()),
        "maximum_accumulated_nav_absolute_difference": float(accumulated_difference.max()),
        "non_trading_or_extra_source_dates": int(len(available.difference(expected))),
    }
    print(
        f"净值采集完成：交易日{len(output)}行，东方财富{east_pages}页，新浪{sina_pages}页",
        flush=True,
    )
    return output, [*east_receipts, *sina_receipts], audit


def _one_row(frame: pd.DataFrame, date: pd.Timestamp, label: str) -> pd.Series:
    selected = frame.loc[frame["date"].eq(date)]
    if len(selected) != 1:
        raise ValueError(f"{label}在{date.date()}记录数不是1：{len(selected)}")
    return selected.iloc[0]


def _seam_audit(
    *,
    config: dict[str, Any],
    nav: pd.DataFrame,
    shares: pd.DataFrame,
    margin: pd.DataFrame,
) -> dict[str, Any]:
    seam = pd.Timestamp(config["seam_crosscheck"]["required_date"])
    paths = {
        name: PROJECT_ROOT / relative
        for name, relative in config["inputs_at_freeze"].items()
        if name.startswith("seam_")
    }
    frozen_nav = _normalize(pd.read_parquet(paths["seam_nav"]))
    frozen_shares = _normalize(pd.read_parquet(paths["seam_fund_share"]))
    frozen_margin = _normalize(pd.read_parquet(paths["seam_margin"]))
    current_nav = _one_row(nav, seam, "新净值")
    prior_nav = _one_row(frozen_nav, seam, "冻结净值")
    current_share = _one_row(shares, seam, "新份额")
    prior_share = _one_row(frozen_shares, seam, "冻结份额")
    current_margin = _one_row(margin, seam, "新融资融券")
    prior_margin = _one_row(frozen_margin, seam, "冻结融资融券")

    nav_difference = abs(
        float(current_nav["close_premium_to_nav"])
        - float(prior_nav["close_premium_to_nav"])
    )
    share_difference = abs(
        float(current_share["fund_shares"]) - float(prior_share["fund_shares"])
    )
    direct_columns = ["rzye", "rzmre", "rqyl", "rzche", "rqchl", "rqmcl"]
    direct_differences = {
        column: abs(float(current_margin[column]) - float(prior_margin[column]))
        for column in direct_columns
    }
    derived_columns = ["rqye", "rzrqye"]
    derived_differences = {
        column: abs(float(current_margin[column]) - float(prior_margin[column]))
        for column in derived_columns
    }
    specification = config["seam_crosscheck"]
    if nav_difference > float(specification["nav_absolute_tolerance"]):
        raise ValueError("净值折溢价接缝差异超过容差")
    if share_difference > float(specification["share_absolute_tolerance"]):
        raise ValueError("份额接缝差异超过容差")
    if any(
        value > float(specification["direct_margin_absolute_tolerance"])
        for value in direct_differences.values()
    ):
        raise ValueError("融资融券直接字段接缝差异超过容差")
    if any(
        value > float(specification["derived_margin_absolute_tolerance"])
        for value in derived_differences.values()
    ):
        raise ValueError("融资融券推导字段接缝差异超过容差")
    return {
        "status": "PASS",
        "date": seam.date().isoformat(),
        "nav_close_premium_absolute_difference": nav_difference,
        "fund_shares_absolute_difference": share_difference,
        "direct_margin_absolute_difference": direct_differences,
        "derived_margin_absolute_difference": derived_differences,
    }


def _dataset_receipt(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    return {
        "file": _relative(path),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
    }


def _count_raw_files(raw_dir: Path) -> int:
    return sum(1 for path in raw_dir.rglob("*.json") if path.is_file()) if raw_dir.exists() else 0


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    config, freeze = _load_and_verify_freeze()
    snapshot_root = PROJECT_ROOT / config["paths"]["snapshot_root"]
    target_dir = (snapshot_root / arguments.run_id).resolve()
    if not target_dir.is_relative_to(snapshot_root.resolve()):
        raise ValueError("run-id导致目标目录越界")
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "snapshot_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("project_id") == PROJECT_ID
            and manifest.get("status") == "PASS_COMPLETE_EXTERNAL_SIGNAL_SNAPSHOT"
            and manifest.get("freeze_manifest_sha256") == _sha256(FREEZE_MANIFEST)
        ):
            print("指定run-id已有完整且匹配的快照，直接复用。", flush=True)
            return manifest
        raise FileExistsError("指定run-id已有不匹配的快照清单")

    raw_dir = target_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_contract_path = target_dir / "run_contract.json"
    run_contract = {
        "project_id": PROJECT_ID,
        "run_id": arguments.run_id,
        "config_sha256": _sha256(CONFIG_FILE),
        "freeze_manifest_sha256": _sha256(FREEZE_MANIFEST),
        "collector_sha256": _sha256(Path(__file__).resolve()),
    }
    if run_contract_path.exists():
        existing_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
        if existing_contract != run_contract:
            raise ValueError("中断任务的运行契约与当前冻结契约不一致")
    else:
        _write_new_json(run_contract, run_contract_path)

    dates = config["dates"]
    signal_start = pd.Timestamp(dates["signal_collection_start"])
    signal_end = pd.Timestamp(dates["signal_collection_end"])
    market_end = pd.Timestamp(dates["execution_market_end"])
    etf_source = PROJECT_ROOT / config["inputs_at_freeze"]["etf_market"]
    benchmark_source = PROJECT_ROOT / config["inputs_at_freeze"]["benchmark_total_return"]
    dividend_source = PROJECT_ROOT / config["inputs_at_freeze"]["dividends"]
    etf_all = _normalize(pd.read_parquet(etf_source))
    benchmark_all = _normalize(pd.read_parquet(benchmark_source))
    signal_market = etf_all.loc[
        etf_all["date"].between(signal_start, signal_end)
    ].copy()
    execution_market = etf_all.loc[
        etf_all["date"].between(signal_start, market_end)
    ].copy()
    if signal_market.empty or execution_market.empty:
        raise ValueError("冻结510300行情不覆盖预声明日期")
    if signal_market["date"].iloc[0] != signal_start:
        raise ValueError("信号采集起点不是510300交易日")
    if signal_market["date"].iloc[-1] != signal_end:
        raise ValueError("信号采集终点不是510300交易日")
    if execution_market["date"].iloc[-1] != market_end:
        raise ValueError("执行行情终点不是510300交易日")
    benchmark = execution_market[["date"]].merge(
        benchmark_all,
        on="date",
        how="left",
        validate="one_to_one",
    )
    if benchmark["close"].isna().any():
        raise ValueError("H00300在510300执行交易日存在缺失")
    signal_dates = [pd.Timestamp(value) for value in signal_market["date"]]
    close_by_date = {
        pd.Timestamp(row.date): float(row.close)
        for row in signal_market[["date", "close"]].itertuples(index=False)
    }
    retrieved_at = datetime.now(TIMEZONE).isoformat()
    print(
        f"冻结日历：信号{len(signal_dates)}日，{signal_start.date()}至{signal_end.date()}。",
        flush=True,
    )

    shares, share_receipts = _collect_shares(
        signal_dates,
        raw_dir,
        workers=arguments.share_workers,
    )
    margin, margin_receipts = _collect_margin(
        start=signal_start,
        end=signal_end,
        dates=signal_dates,
        close_by_date=close_by_date,
        raw_dir=raw_dir,
    )
    nav, nav_receipts, nav_audit = _collect_nav(
        start=signal_start,
        end=signal_end,
        dates=signal_dates,
        close_by_date=close_by_date,
        raw_dir=raw_dir,
        workers=arguments.nav_workers,
        tolerance=float(
            config["acquisition"]["nav"]["maximum_unit_nav_absolute_difference"]
        ),
    )
    for frame in [shares, margin, nav]:
        frame["retrieved_at"] = retrieved_at
    seam = _seam_audit(config=config, nav=nav, shares=shares, margin=margin)

    paths = {
        "etf": target_dir / "510300_daily.parquet",
        "benchmark": target_dir / "H00300_total_return_daily.parquet",
        "nav": target_dir / "510300_nav_daily.parquet",
        "fund_share": target_dir / "510300_fund_share_daily.parquet",
        "margin": target_dir / "510300_margin_detail_daily.parquet",
        "dividends": target_dir / "510300_dividends.csv",
    }
    _write_new_parquet(execution_market, paths["etf"])
    _write_new_parquet(benchmark, paths["benchmark"])
    _write_new_parquet(nav, paths["nav"])
    _write_new_parquet(shares, paths["fund_share"])
    _write_new_parquet(margin, paths["margin"])
    _copy_new(dividend_source, paths["dividends"])

    raw_receipts = sorted(
        [*share_receipts, *margin_receipts, *nav_receipts],
        key=lambda item: (item["kind"], item["identifier"]),
    )
    raw_index_path = target_dir / "raw_response_index.json"
    _write_new_json(raw_receipts, raw_index_path)
    dividend_frame = pd.read_csv(paths["dividends"])
    datasets = {
        "etf": _dataset_receipt(execution_market, paths["etf"]),
        "benchmark": _dataset_receipt(benchmark, paths["benchmark"]),
        "nav": _dataset_receipt(nav, paths["nav"]),
        "fund_share": _dataset_receipt(shares, paths["fund_share"]),
        "margin": _dataset_receipt(margin, paths["margin"]),
        "dividends": {
            "file": _relative(paths["dividends"]),
            "sha256": _sha256(paths["dividends"]),
            "rows": int(len(dividend_frame)),
            "columns": list(dividend_frame.columns),
        },
    }
    manifest = {
        "project_id": PROJECT_ID,
        "status": "PASS_COMPLETE_EXTERNAL_SIGNAL_SNAPSHOT",
        "run_id": arguments.run_id,
        "retrieved_at_asia_shanghai": retrieved_at,
        "config_sha256": _sha256(CONFIG_FILE),
        "freeze_manifest_sha256": _sha256(FREEZE_MANIFEST),
        "candidate_rule_evaluated_during_collection": False,
        "candidate_return_or_metric_computed_during_collection": False,
        "signal_calendar": {
            "rows": int(len(signal_dates)),
            "first_date": signal_start.date().isoformat(),
            "last_date": signal_end.date().isoformat(),
            "complete_signal_dates": [date.date().isoformat() for date in signal_dates],
        },
        "execution_market": {
            "rows": int(len(execution_market)),
            "last_date": market_end.date().isoformat(),
            "h00300_missing_dates": [],
        },
        "collection_audit": {
            "fund_share_rows": int(len(shares)),
            "margin_rows": int(len(margin)),
            "nav": nav_audit,
            "seam_crosscheck": seam,
            "raw_response_count": int(len(raw_receipts)),
            "raw_response_index": {
                "file": _relative(raw_index_path),
                "sha256": _sha256(raw_index_path),
            },
            "raw_response_kind_counts": {
                kind: int(sum(item["kind"] == kind for item in raw_receipts))
                for kind in sorted({item["kind"] for item in raw_receipts})
            },
        },
        "datasets": datasets,
        "boundaries": {
            "time_reversed_external_historical_validation_input": True,
            "pristine_prospective_holdout": False,
            "research_only": True,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _write_new_json(manifest, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="采集510300双影子规则反向验证快照")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--share-workers", type=int, default=8)
    parser.add_argument("--nav-workers", type=int, default=8)
    arguments = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", arguments.run_id) is None:
        raise ValueError("run-id只能包含字母、数字、点、下划线和连字符，且长度不超过80")
    if ".." in arguments.run_id:
        raise ValueError("run-id不得包含连续点号")
    if not 1 <= arguments.share_workers <= 12:
        raise ValueError("share-workers必须在1至12之间")
    if not 1 <= arguments.nav_workers <= 12:
        raise ValueError("nav-workers必须在1至12之间")
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    target_dir = (
        PROJECT_ROOT / config["paths"]["snapshot_root"] / arguments.run_id
    ).resolve()
    status_path = target_dir / "collection_status.json"
    started_at = datetime.now(TIMEZONE).isoformat()
    try:
        manifest = _run(arguments)
    except Exception as exc:
        payload = {
            "project_id": PROJECT_ID,
            "status": "PARTIAL_SUCCESS_RESUMABLE",
            "run_id": arguments.run_id,
            "started_at_asia_shanghai": started_at,
            "failed_at_asia_shanghai": datetime.now(TIMEZONE).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "raw_response_files_preserved": _count_raw_files(target_dir / "raw_responses"),
            "resume_command": (
                ".venv\\Scripts\\python.exe scripts\\"
                "collect_510300_etf_microstructure_dual_shadow_reverse_validation_v1.py "
                f"--run-id {arguments.run_id}"
            ),
        }
        _atomic_status(payload, status_path)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 2
    payload = {
        "project_id": PROJECT_ID,
        "status": manifest["status"],
        "run_id": arguments.run_id,
        "started_at_asia_shanghai": started_at,
        "completed_at_asia_shanghai": datetime.now(TIMEZONE).isoformat(),
        "snapshot_manifest": _relative(target_dir / "snapshot_manifest.json"),
        "snapshot_manifest_sha256": _sha256(target_dir / "snapshot_manifest.json"),
        "signal_rows": manifest["signal_calendar"]["rows"],
        "raw_response_count": manifest["collection_audit"]["raw_response_count"],
        "seam_status": manifest["collection_audit"]["seam_crosscheck"]["status"],
    }
    _atomic_status(payload, status_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
