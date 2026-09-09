"""下载并标准化510300未复权15分钟行情。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from market_data.minute import MinuteDataRequest, SinaMinuteProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "minute_data.yaml"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_raw.parquet"
METADATA_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_raw.metadata.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))


def basic_validate(data: pd.DataFrame, expected_symbol: str, frequency_minutes: int) -> None:
    required = {
        "bar_start", "bar_end", "trade_date", "open", "high", "low", "close",
        "volume", "amount", "symbol", "frequency_minutes", "source", "retrieved_at",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"标准化分钟数据缺少字段：{sorted(missing)}")
    if data.empty:
        raise ValueError("分钟接口返回空数据")
    if data["bar_end"].duplicated().any():
        raise ValueError("分钟数据存在重复bar_end")
    if not data["bar_end"].is_monotonic_increasing:
        raise ValueError("分钟数据未按bar_end升序排列")
    if data["symbol"].unique().tolist() != [expected_symbol]:
        raise ValueError("分钟数据证券代码不一致")
    if data["frequency_minutes"].unique().tolist() != [frequency_minutes]:
        raise ValueError("分钟数据频率不一致")
    prices = data[["open", "high", "low", "close"]]
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("分钟价格存在空值或非正数")
    high_bad = data["high"] < data[["open", "low", "close"]].max(axis=1)
    low_bad = data["low"] > data[["open", "high", "close"]].min(axis=1)
    if (high_bad | low_bad).any():
        raise ValueError("分钟OHLC关系不一致")


def main() -> int:
    config = load_config()
    if config["provider"] != "sina":
        print(f"尚未实现的分钟数据源：{config['provider']}", file=sys.stderr)
        return 1
    request = MinuteDataRequest(
        symbol=config["symbol"],
        frequency_minutes=int(config["frequency_minutes"]),
        adjustment=config["adjustment"],
        maximum_bars=int(config["maximum_bars"]),
        timeout_seconds=float(config["timeout_seconds"]),
        timezone=config["timezone"],
    )
    provider = SinaMinuteProvider()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = OUTPUT_FILE.with_suffix(".parquet.tmp")
    try:
        fetched = provider.fetch(request)
        existing = pd.read_parquet(OUTPUT_FILE) if OUTPUT_FILE.exists() else pd.DataFrame()
        existing_rows = int(len(existing))
        data = pd.concat([existing, fetched], ignore_index=True) if not existing.empty else fetched.copy()
        data = data.drop_duplicates("bar_end", keep="last").sort_values("bar_end").reset_index(drop=True)
        basic_validate(data, request.symbol, request.frequency_minutes)
        data.to_parquet(temp_file, index=False, engine="pyarrow")
        temp_file.replace(OUTPUT_FILE)
    except Exception as exc:
        if temp_file.exists():
            temp_file.unlink()
        print(f"15分钟下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    counts = data.groupby("trade_date").size()
    expected_count = int(config["session"]["expected_bars_per_full_day"])
    metadata = {
        "symbol": request.symbol,
        "frequency_minutes": request.frequency_minutes,
        "adjustment": request.adjustment,
        "timestamp_meaning": "bar_end",
        "timezone": request.timezone,
        "provider": provider.provider_name,
        "requested_maximum_bars": request.maximum_bars,
        "provider_bar_limit": int(config["provider_constraints"]["bar_limit"]),
        "historical_completeness": config["provider_constraints"]["historical_completeness"],
        "persistence_policy": "每次合并最新窗口并按bar_end保留最新记录，避免滚动端点丢失旧数据",
        "existing_row_count_before_merge": existing_rows,
        "fetched_row_count": int(len(fetched)),
        "fetched_first_bar_end": pd.Timestamp(fetched["bar_end"].min()).isoformat(),
        "fetched_last_bar_end": pd.Timestamp(fetched["bar_end"].max()).isoformat(),
        "actual_first_bar_end": pd.Timestamp(data["bar_end"].min()).isoformat(),
        "actual_last_bar_end": pd.Timestamp(data["bar_end"].max()).isoformat(),
        "row_count": int(len(data)),
        "trading_day_count": int(data["trade_date"].nunique()),
        "full_trading_day_count": int((counts == expected_count).sum()),
        "partial_trading_days": {
            pd.Timestamp(date).date().isoformat(): int(count)
            for date, count in counts[counts != expected_count].items()
        },
        "retrieved_at": datetime.now(ZoneInfo(request.timezone)).isoformat(),
        "source_url": "https://quotes.sina.cn/cn/api/jsonp_v2.php/=/CN_MarketDataService.getKLineData",
        "file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
    }
    metadata["sha256"] = sha256_file(OUTPUT_FILE)
    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
