"""独立下载旧披露口径下的北向资金日净流量历史。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import akshare as ak
import tushare as ts
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
START_DATE = pd.Timestamp("2016-12-05")
END_DATE = pd.Timestamp("2024-08-16")
DISCLOSURE_CHANGE_DATE = pd.Timestamp("2024-08-19")
OUTPUT_FILE = ROOT / "data" / "raw" / "flow" / "northbound_legacy_net_flow_v1.parquet"
REPORT_FILE = (
    ROOT / "reports" / "data_quality" / "northbound_legacy_net_flow_v1_status.json"
)


def sha256_file(path: Path) -> str:
    """流式计算文件哈希。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    """原子写入JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def get_token() -> str:
    """读取凭据，但绝不输出或写入报告。"""

    load_dotenv(ROOT / ".env")
    token = os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN")
    if not token:
        raise RuntimeError("未设置TUSHARE_TOKEN或TS_TOKEN")
    return token


def yearly_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, str]]:
    """按自然年拆分请求，避免接口行数上限。"""

    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        window_end = min(pd.Timestamp(cursor.year, 12, 31), end)
        windows.append((cursor.strftime("%Y%m%d"), window_end.strftime("%Y%m%d")))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def normalize(raw: pd.DataFrame, retrieved_at: datetime) -> pd.DataFrame:
    """规范字段并硬性排除披露变更后的不可比值。"""

    required = {
        "trade_date",
        "ggt_ss",
        "ggt_sz",
        "hgt",
        "sgt",
        "north_money",
        "south_money",
    }
    if missing := required - set(raw.columns):
        raise ValueError(f"moneyflow_hsgt缺少字段：{sorted(missing)}")
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    numeric = ["ggt_ss", "ggt_sz", "hgt", "sgt", "north_money", "south_money"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[
        frame["date"].between(START_DATE, END_DATE, inclusive="both")
    ].copy()
    frame.sort_values(["date", "trade_date"], kind="mergesort", inplace=True)
    frame.drop_duplicates("date", keep="last", inplace=True)
    if frame.empty:
        raise ValueError("接口没有返回旧口径北向资金记录")
    if frame["date"].max() >= DISCLOSURE_CHANGE_DATE:
        raise ValueError("原始结果混入披露口径变更后的日期")
    if frame["north_money"].isna().any():
        raise ValueError("旧口径north_money存在缺失值")
    frame["northbound_semantics"] = "LEGACY_DAILY_NET_FLOW"
    frame["usable_as_net_flow"] = True
    frame["source"] = "tushare.moneyflow_hsgt"
    frame["retrieved_at"] = retrieved_at
    return frame[
        [
            "date",
            *numeric,
            "northbound_semantics",
            "usable_as_net_flow",
            "source",
            "retrieved_at",
        ]
    ].reset_index(drop=True)


def normalize_akshare(raw: pd.DataFrame, retrieved_at: datetime) -> pd.DataFrame:
    """规范东方财富公开历史页返回的北向合计净买入。"""

    rename = {
        "日期": "date",
        "当日成交净买额": "north_net_buy_100m_cny",
        "买入成交额": "north_buy_100m_cny",
        "卖出成交额": "north_sell_100m_cny",
        "沪深300": "csi300_close",
        "沪深300-涨跌幅": "csi300_return_pct",
    }
    required = set(rename)
    if missing := required - set(raw.columns):
        raise ValueError(f"东方财富北向历史缺少字段：{sorted(missing)}")
    frame = raw[list(rename)].rename(columns=rename).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    numeric = [
        "north_net_buy_100m_cny",
        "north_buy_100m_cny",
        "north_sell_100m_cny",
        "csi300_close",
        "csi300_return_pct",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[
        frame["date"].between(START_DATE, END_DATE, inclusive="both")
    ].copy()
    frame.sort_values("date", kind="mergesort", inplace=True)
    frame.drop_duplicates("date", keep="last", inplace=True)
    if frame.empty:
        raise ValueError("东方财富公开历史没有返回目标区间")
    if frame["date"].max() >= DISCLOSURE_CHANGE_DATE:
        raise ValueError("东方财富结果混入披露口径变更后的日期")
    if frame["north_net_buy_100m_cny"].isna().any():
        raise ValueError("旧口径北向成交净买额存在缺失值")
    frame["northbound_semantics"] = "LEGACY_DAILY_NET_BUY"
    frame["usable_as_net_flow"] = True
    frame["source"] = "eastmoney_via_akshare.stock_hsgt_hist_em"
    frame["retrieved_at"] = retrieved_at
    return frame[
        [
            "date",
            *numeric,
            "northbound_semantics",
            "usable_as_net_flow",
            "source",
            "retrieved_at",
        ]
    ].reset_index(drop=True)


def main() -> int:
    """下载、核验并保存独立历史工件。"""

    token = ""
    try:
        if OUTPUT_FILE.exists() or REPORT_FILE.exists():
            raise FileExistsError("北向资金V1工件已存在，禁止覆盖")
        retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai"))
        raw = ak.stock_hsgt_hist_em(symbol="北向资金")
        frame = normalize_akshare(raw, retrieved_at)
        atomic_parquet(frame, OUTPUT_FILE)
        report = {
            "status": "PASS_LEGACY_NET_FLOW_HISTORY_ACQUIRED",
            "checked_at": retrieved_at.isoformat(),
            "scope": {
                "start_date": START_DATE.date().isoformat(),
                "end_date": END_DATE.date().isoformat(),
                "disclosure_change_date_excluded": DISCLOSURE_CHANGE_DATE.date().isoformat(),
            },
            "rows": int(len(frame)),
            "first_date": frame["date"].min().date().isoformat(),
            "last_date": frame["date"].max().date().isoformat(),
            "nonzero_rows": int(frame["north_net_buy_100m_cny"].ne(0.0).sum()),
            "provider": "东方财富公开历史页，经AKShare 1.18.84读取",
            "unit": "亿元人民币",
            "attempt_history": [
                {
                    "status": "PROGRAM_FAILED",
                    "reason": "首个Tushare下载入口缺少环境凭据，未产生或覆盖数据。",
                },
                {
                    "status": "SUCCESS",
                    "provider": "EASTMONEY_VIA_AKSHARE",
                },
            ],
            "artifact": {
                "file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(OUTPUT_FILE),
            },
            "boundary": "仅保存2024-08-19披露变更前的旧口径日净流量。",
        }
        atomic_json(report, REPORT_FILE)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        message = str(error).replace(token, "[REDACTED]") if token else str(error)
        print(f"北向资金旧口径历史下载失败：{type(error).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
