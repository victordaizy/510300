"""采集新浪 IF 一分钟短窗口，仅用于时间戳与合约字段验证。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "futures" / "IF_1m_sina_validation.parquet"
METADATA_FILE = OUTPUT_FILE.with_suffix(".metadata.json")
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "futures" / ".if_1m_sina_validation_chunks"
TIMEZONE = "Asia/Shanghai"
REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close", "volume", "hold"]


def parse_arguments() -> argparse.Namespace:
    now = pd.Timestamp.now(tz=TIMEZONE)
    current = f"IF{now:%y%m}"
    following = f"IF{(now + pd.offsets.MonthBegin(1)):%y%m}"
    parser = argparse.ArgumentParser(description="下载IF一分钟短窗口验证样本")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["IF0", current, following],
        help="新浪期货代码；默认主连、当月和次月实际合约",
    )
    return parser.parse_args()


def normalize_frame(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"{symbol}返回字段不完整：{missing}")
    output = data[REQUIRED_COLUMNS].copy().rename(
        columns={"datetime": "trade_time", "hold": "open_interest"}
    )
    output["trade_time"] = pd.to_datetime(output["trade_time"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume", "open_interest"]
    output[numeric] = output[numeric].apply(pd.to_numeric, errors="coerce")
    if output["trade_time"].isna().any() or output[numeric].isna().any().any():
        raise ValueError(f"{symbol}存在无法解析的时间或数值")
    if (output[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}存在非正价格")
    if (output[["volume", "open_interest"]] < 0).any().any():
        raise ValueError(f"{symbol}存在负成交量或持仓量")
    if output["trade_time"].duplicated().any():
        raise ValueError(f"{symbol}存在重复时间戳")
    output["contract"] = symbol
    output["is_continuous"] = symbol == "IF0"
    output["source"] = "akshare.futures_zh_minute_sina"
    return output.sort_values("trade_time").reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_with_retry(symbol: str, maximum_attempts: int = 6) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{symbol}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    for attempt in range(1, maximum_attempts + 1):
        try:
            frame = normalize_frame(
                ak.futures_zh_minute_sina(symbol=symbol, period="1"), symbol
            )
            if frame.empty:
                raise RuntimeError(f"{symbol}返回空数据")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(cache_path)
            return frame
        except Exception:
            if attempt == maximum_attempts:
                raise
            delay = min(2 ** (attempt - 1), 20)
            print(f"{symbol}第{attempt}次请求失败，{delay}秒后重试。", flush=True)
            time.sleep(delay)
    raise AssertionError("重试循环不应执行到此处")


def main() -> int:
    arguments = parse_arguments()
    frames: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    for symbol in dict.fromkeys(arguments.symbols):
        print(f"正在下载 {symbol} 1分钟验证样本……", flush=True)
        frame = _fetch_with_retry(symbol)
        frames.append(frame)
        coverage.append(
            {
                "symbol": symbol,
                "row_count": int(len(frame)),
                "first_trade_time": frame["trade_time"].min().isoformat(),
                "last_trade_time": frame["trade_time"].max().isoformat(),
                "trading_day_count": int(frame["trade_time"].dt.normalize().nunique()),
            }
        )
    combined = pd.concat(frames, ignore_index=True).sort_values(["contract", "trade_time"])
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix(".parquet.tmp")
    combined.to_parquet(temporary, index=False)
    temporary.replace(OUTPUT_FILE)
    metadata = {
        "status": "VALIDATION_ONLY_NOT_FORMAL_RESEARCH",
        "reason": "新浪每个代码仅返回约1023根，不能覆盖五年，也不能替代实际合约全量历史",
        "frequency": "1min",
        "timestamp_meaning": "供应商字段名为datetime；正式研究前仍需与510300共同跳变核对",
        "coverage": coverage,
        "row_count": int(len(combined)),
        "retrieved_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "file": {
            "path": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
            "bytes": OUTPUT_FILE.stat().st_size,
            "sha256": _sha256(OUTPUT_FILE),
        },
    }
    metadata_temporary = METADATA_FILE.with_suffix(".json.tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata_temporary.replace(METADATA_FILE)
    print(f"已保存 {len(combined)} 行验证样本：{OUTPUT_FILE}")
    print("状态：VALIDATION_ONLY_NOT_FORMAL_RESEARCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
