"""下载510300历史单位净值，并用东方财富与新浪两来源交叉核验。"""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
PRICE_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "fund" / "510300_nav_daily_raw.parquet"
METADATA_FILE = OUTPUT_FILE.with_suffix(".metadata.json")
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_nav_cross_source.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict,
    headers: dict | None = None,
    maximum_retries: int = 3,
) -> dict:
    final_error: Exception | None = None
    for attempt in range(1, maximum_retries + 1):
        try:
            response = session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            final_error = exc
            if attempt < maximum_retries:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"接口请求失败：{type(final_error).__name__}: {final_error}")


def fetch_eastmoney(start_date: str, end_date: str) -> pd.DataFrame:
    """获取东方财富历史净值。"""

    url = "https://api.fund.eastmoney.com/f10/lsjz"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://fundf10.eastmoney.com/jjjz_510300.html",
    }
    session = requests.Session()
    params = {
        "fundCode": "510300",
        "pageIndex": "1",
        "pageSize": "20",
        "startDate": start_date,
        "endDate": end_date,
    }
    first = _request_json(session, url, params=params, headers=headers)
    total = int(first.get("TotalCount") or 0)
    pages = math.ceil(total / 20)
    records = list((first.get("Data") or {}).get("LSJZList") or [])

    def fetch_page(page: int) -> list[dict]:
        page_params = {**params, "pageIndex": str(page)}
        with requests.Session() as page_session:
            payload = _request_json(page_session, url, params=page_params, headers=headers)
        return list((payload.get("Data") or {}).get("LSJZList") or [])

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_page, page): page for page in range(2, pages + 1)}
        page_records: dict[int, list[dict]] = {}
        for future in as_completed(futures):
            page = futures[future]
            page_records[page] = future.result()
    for page in range(2, pages + 1):
        records.extend(page_records[page])
    data = pd.DataFrame.from_records(records)
    required = {"FSRQ", "DWJZ", "LJJZ"}
    missing = required.difference(data.columns)
    if data.empty or missing:
        raise RuntimeError(f"东方财富净值为空或缺少字段：{sorted(missing)}")
    output = data.rename(
        columns={"FSRQ": "date", "DWJZ": "nav_eastmoney", "LJJZ": "acc_nav_eastmoney"}
    )[["date", "nav_eastmoney", "acc_nav_eastmoney"]].copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output[["nav_eastmoney", "acc_nav_eastmoney"]] = output[
        ["nav_eastmoney", "acc_nav_eastmoney"]
    ].apply(pd.to_numeric, errors="coerce")
    return output.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def fetch_sina(start_date: str, end_date: str) -> pd.DataFrame:
    """获取新浪财富汇历史净值。"""

    url = (
        "http://stock.finance.sina.com.cn/fundInfo/api/openapi.php/"
        "CaihuiFundInfoService.getNav"
    )
    session = requests.Session()
    params = {
        "symbol": "510300",
        "datefrom": start_date,
        "dateto": end_date,
        "page": "1",
        "num": "200",
    }
    first = _request_json(session, url, params=params)
    first_result = first.get("result") or {}
    if int((first_result.get("status") or {}).get("code", -1)) != 0:
        raise RuntimeError("新浪净值接口返回失败状态")
    body = first_result.get("data") or {}
    total = int(body.get("total_num") or 0)
    first_records = list(body.get("data") or [])
    page_size = len(first_records)
    if total <= 0 or page_size <= 0:
        raise RuntimeError("新浪净值接口返回空数据")
    records = first_records
    for page in range(2, math.ceil(total / page_size) + 1):
        params["page"] = str(page)
        payload = _request_json(session, url, params=params)
        result = payload.get("result") or {}
        if int((result.get("status") or {}).get("code", -1)) != 0:
            raise RuntimeError(f"新浪净值第{page}页返回失败状态")
        records.extend(((result.get("data") or {}).get("data") or []))
    data = pd.DataFrame.from_records(records)
    required = {"fbrq", "jjjz", "ljjz"}
    missing = required.difference(data.columns)
    if data.empty or missing:
        raise RuntimeError(f"新浪净值为空或缺少字段：{sorted(missing)}")
    output = data.rename(
        columns={"fbrq": "date", "jjjz": "nav_sina", "ljjz": "acc_nav_sina"}
    )[["date", "nav_sina", "acc_nav_sina"]].copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output[["nav_sina", "acc_nav_sina"]] = output[["nav_sina", "acc_nav_sina"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return output.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    start = config["project"]["start_date"]
    end = config["project"]["end_date"]
    eastmoney = fetch_eastmoney(start, end)
    sina = fetch_sina(start, end)
    merged = eastmoney.merge(sina, on="date", how="outer", validate="one_to_one", indicator=True)
    merged = merged.sort_values("date").reset_index(drop=True)
    missing_rows = merged.loc[
        merged[["date", "nav_eastmoney", "nav_sina"]].isna().any(axis=1),
        ["date", "nav_eastmoney", "nav_sina", "_merge"],
    ]
    if not missing_rows.empty:
        details = missing_rows.assign(date=missing_rows["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
        raise ValueError(f"两来源净值日期覆盖不一致或存在空值：{details[:20]}")
    nav_difference = (merged["nav_eastmoney"] - merged["nav_sina"]).abs()
    acc_difference = (merged["acc_nav_eastmoney"] - merged["acc_nav_sina"]).abs()
    if (nav_difference > 0.00005).any() or (acc_difference > 0.00005).any():
        raise ValueError("两来源净值差异超过四位小数精度容差")
    merged["unit_nav"] = merged["nav_eastmoney"]
    merged["accumulated_nav"] = merged["acc_nav_eastmoney"]
    merged["symbol"] = "510300.SH"
    merged["source_primary"] = "eastmoney.f10.lsjz"
    merged["source_secondary"] = "sina.CaihuiFundInfoService.getNav"
    merged["retrieved_at"] = datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat()

    prices = pd.read_parquet(PRICE_FILE)[["date", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    merged = merged.merge(prices, on="date", how="left", validate="one_to_one")
    merged["has_exchange_close"] = merged["close"].notna()
    merged["close_premium_to_nav"] = merged["close"] / merged["unit_nav"] - 1.0
    merged["close_premium_bps"] = merged["close_premium_to_nav"] * 10_000.0
    if (merged.loc[merged["has_exchange_close"], "unit_nav"] <= 0).any():
        raise ValueError("单位净值存在非正数")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix(".parquet.tmp")
    merged.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(OUTPUT_FILE)

    unmatched_nav_dates = merged.loc[~merged["has_exchange_close"], "date"].dt.strftime("%Y-%m-%d").tolist()
    report = {
        "status": "PASS",
        "checked_at": datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat(),
        "row_count": int(len(merged)),
        "first_date": merged["date"].min().date().isoformat(),
        "last_date": merged["date"].max().date().isoformat(),
        "eastmoney_sina_date_match": bool((merged["_merge"] == "both").all()),
        "maximum_unit_nav_absolute_difference": float(nav_difference.max()),
        "maximum_accumulated_nav_absolute_difference": float(acc_difference.max()),
        "dates_without_exchange_close": unmatched_nav_dates,
        "premium_observation_count": int(merged["close_premium_to_nav"].notna().sum()),
        "premium_bps": {
            "minimum": float(merged["close_premium_bps"].min()),
            "median": float(merged["close_premium_bps"].median()),
            "maximum": float(merged["close_premium_bps"].max()),
        },
        "interpretation": "收盘折溢价使用当日交易所收盘价除以当日单位净值减一；不是盘中IOPV偏离。",
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "symbol": "510300.SH",
        "file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "row_count": int(len(merged)),
        "actual_first_date": merged["date"].min().date().isoformat(),
        "actual_last_date": merged["date"].max().date().isoformat(),
        "sources": ["eastmoney.f10.lsjz", "sina.CaihuiFundInfoService.getNav"],
        "sha256": _sha256_file(OUTPUT_FILE),
        "retrieved_at": merged["retrieved_at"].iloc[0],
    }
    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": metadata, "quality": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
