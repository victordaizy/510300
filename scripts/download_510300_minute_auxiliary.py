"""下载腾讯15分钟第二来源与新浪1分钟时间戳验证数据。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from market_data.minute import (
    MinuteDataRequest,
    SinaMinuteProvider,
    TencentIntradaySnapshotProvider,
    TencentMinuteProvider,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "minute_data.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "market"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persist_rolling_window(
    fetched: pd.DataFrame,
    output_file: Path,
    metadata_file: Path,
    metadata: dict,
) -> dict:
    existing = pd.read_parquet(output_file) if output_file.exists() else pd.DataFrame()
    existing_rows = int(len(existing))
    timestamp_column = "bar_end" if "bar_end" in fetched.columns else "timestamp"
    combined = pd.concat([existing, fetched], ignore_index=True) if not existing.empty else fetched.copy()
    combined = (
        combined.drop_duplicates(timestamp_column, keep="last")
        .sort_values(timestamp_column)
        .reset_index(drop=True)
    )
    temp_file = output_file.with_suffix(".parquet.tmp")
    combined.to_parquet(temp_file, index=False, engine="pyarrow")
    temp_file.replace(output_file)
    result = {
        **metadata,
        "persistence_policy": f"滚动窗口按{timestamp_column}合并，保留本地累积历史",
        "existing_row_count_before_merge": existing_rows,
        "fetched_row_count": int(len(fetched)),
        "stored_row_count": int(len(combined)),
        "fetched_first_timestamp": pd.Timestamp(fetched[timestamp_column].min()).isoformat(),
        "fetched_last_timestamp": pd.Timestamp(fetched[timestamp_column].max()).isoformat(),
        "stored_first_timestamp": pd.Timestamp(combined[timestamp_column].min()).isoformat(),
        "stored_last_timestamp": pd.Timestamp(combined[timestamp_column].max()).isoformat(),
        "file": output_file.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(output_file),
    }
    metadata_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo(config["timezone"])).isoformat()
    try:
        tencent_request = MinuteDataRequest(
            symbol=config["symbol"],
            frequency_minutes=15,
            maximum_bars=int(config["provider_constraints"]["secondary_bar_limit"]),
            timeout_seconds=float(config["timeout_seconds"]),
            timezone=config["timezone"],
        )
        tencent = TencentMinuteProvider().fetch(tencent_request)
        tencent_metadata = persist_rolling_window(
            tencent,
            RAW_DIR / "510300_15m_tencent_raw.parquet",
            RAW_DIR / "510300_15m_tencent_raw.metadata.json",
            {
                "symbol": config["symbol"],
                "provider": TencentMinuteProvider.provider_name,
                "frequency_minutes": 15,
                "adjustment": "raw",
                "provider_bar_limit": tencent_request.maximum_bars,
                "amount_available": False,
                "volume_normalization": "原始手数乘100转换为份",
                "retrieved_at": now,
            },
        )

        one_minute_request = MinuteDataRequest(
            symbol=config["symbol"],
            frequency_minutes=1,
            maximum_bars=int(config["provider_constraints"]["one_minute_validation_bar_limit"]),
            timeout_seconds=float(config["timeout_seconds"]),
            timezone=config["timezone"],
        )
        one_minute = SinaMinuteProvider().fetch(one_minute_request)
        one_minute_metadata = persist_rolling_window(
            one_minute,
            RAW_DIR / "510300_1m_validation_raw.parquet",
            RAW_DIR / "510300_1m_validation_raw.metadata.json",
            {
                "symbol": config["symbol"],
                "provider": SinaMinuteProvider.provider_name,
                "frequency_minutes": 1,
                "adjustment": "raw",
                "provider_bar_limit": one_minute_request.maximum_bars,
                "purpose": "验证15分钟bar时间戳与聚合边界，不是五年1分钟历史",
                "retrieved_at": now,
            },
        )
        intraday_request = MinuteDataRequest(
            symbol=config["symbol"],
            frequency_minutes=1,
            maximum_bars=1,
            timeout_seconds=float(config["timeout_seconds"]),
            timezone=config["timezone"],
        )
        intraday = TencentIntradaySnapshotProvider().fetch(intraday_request)
        intraday_metadata = persist_rolling_window(
            intraday,
            RAW_DIR / "510300_intraday_tencent_snapshots.parquet",
            RAW_DIR / "510300_intraday_tencent_snapshots.metadata.json",
            {
                "symbol": config["symbol"],
                "provider": TencentIntradaySnapshotProvider.provider_name,
                "frequency_minutes": 1,
                "snapshot_kind": "当日分钟累计成交量与成交额",
                "purpose": "从接入日起积累15:05—15:30盘后固定价格交易",
                "historical_backfill": False,
                "retrieved_at": now,
            },
        )
    except Exception as exc:
        print(f"分钟辅助数据下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "tencent_15m": tencent_metadata,
                "sina_1m": one_minute_metadata,
                "tencent_intraday_snapshots": intraday_metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
