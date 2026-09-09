"""下载R6长历史日频面板，独立于R5/V3冻结输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_000300_daily import fetch_index
from scripts.download_510300_daily import fetch_from_sina
from scripts.download_h00300_total_return import normalize_columns


OUTPUT_DIR = ROOT / "data" / "raw" / "r6"
METADATA_FILE = OUTPUT_DIR / "daily_panel.metadata.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _validate_prices(frame: pd.DataFrame, name: str, price_columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in price_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if result.empty or result[["date", *price_columns]].isna().any().any():
        raise ValueError(f"{name}为空或存在必要字段空值")
    if (result[price_columns] <= 0).any().any():
        raise ValueError(f"{name}价格存在非正值")
    if not result["date"].is_monotonic_increasing:
        raise ValueError(f"{name}日期未升序")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="下载R6长历史日频面板")
    parser.add_argument("--start", default="2012-05-28", help="ETF成立后首个预期交易日")
    parser.add_argument("--end", default=None, help="结束日期，默认上海本地当天")
    args = parser.parse_args()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end or datetime.now(TIMEZONE).date()).normalize()
    retrieved_at = datetime.now(TIMEZONE).isoformat()
    outputs = {
        "510300": OUTPUT_DIR / "510300_daily.parquet",
        "000300": OUTPUT_DIR / "000300_daily.parquet",
        "H00300": OUTPUT_DIR / "H00300_total_return_daily.parquet",
    }
    try:
        etf = _validate_prices(
            fetch_from_sina("510300.SH", str(start.date()), str(end.date())),
            "510300",
            ["open", "high", "low", "close"],
        )
        index = _validate_prices(
            fetch_index("000300.SH", str(start.date()), str(end.date())),
            "000300",
            ["open", "high", "low", "close"],
        )
        raw_total = ak.stock_zh_index_hist_csindex(
            symbol="H00300",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        total = normalize_columns(raw_total)
        if "date" not in total or "close" not in total:
            raise ValueError("H00300缺少date或close字段")
        total = _validate_prices(total, "H00300", ["close"])
        latest = {
            "510300": pd.to_datetime(etf["date"]).max(),
            "000300": pd.to_datetime(index["date"]).max(),
            "H00300": pd.to_datetime(total["date"]).max(),
        }
        if max(latest.values()) - min(latest.values()) > pd.Timedelta(days=7):
            raise ValueError(f"三条序列末日相差超过7日：{latest}")
        for frame in (etf, index, total):
            frame["retrieved_at"] = retrieved_at
        _atomic_parquet(etf, outputs["510300"])
        _atomic_parquet(index, outputs["000300"])
        _atomic_parquet(total, outputs["H00300"])
        metadata = {
            "retrieved_at": retrieved_at,
            "requested_start": str(start.date()),
            "requested_end": str(end.date()),
            "series": {
                name: {
                    "file": path.relative_to(ROOT).as_posix(),
                    "rows": int(len(frame)),
                    "first_date": str(pd.to_datetime(frame["date"]).min().date()),
                    "last_date": str(pd.to_datetime(frame["date"]).max().date()),
                    "sha256": _sha256(path),
                }
                for name, path, frame in (
                    ("510300", outputs["510300"], etf),
                    ("000300", outputs["000300"], index),
                    ("H00300", outputs["H00300"], total),
                )
            },
            "governance": {
                "r5_v3_frozen_inputs_untouched": True,
                "paper_only": True,
            },
        }
        METADATA_FILE.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"R6长历史面板下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
