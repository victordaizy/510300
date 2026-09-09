"""审计510300五年15分钟覆盖、日线边界和成交重建误差。"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.download_510300_15m_full import (
    CONFIG_FILE,
    DAILY_FILE,
    OUTPUT_FILE,
    _download_worker,
    _load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERLAP_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_tdx_overlap_audit.parquet"
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_15m_full_quality.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_dates(dates: list[pd.Timestamp], count: int) -> list[pd.Timestamp]:
    if len(dates) <= count:
        return dates
    positions = np.linspace(0, len(dates) - 1, num=count, dtype=int)
    return [dates[index] for index in sorted(set(positions.tolist()))]


def _download_overlap_sample(
    dates: list[pd.Timestamp],
    sample_count: int = 48,
) -> tuple[pd.DataFrame, list[dict]]:
    config, request, servers = _load_config()
    selected = _sample_dates(dates, sample_count)
    workers = int(config["tdx"]["worker_count"])
    chunks = [selected[index::workers] for index in range(workers)]
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            lambda item: _download_worker(
                item[0],
                item[1],
                request,
                servers,
                int(config["tdx"]["maximum_retries_per_day"]),
            ),
            [(index, chunk) for index, chunk in enumerate(chunks) if chunk],
        )
        for worker_frames, worker_failures in results:
            frames.extend(worker_frames)
            failures.extend(worker_failures)
    if not frames:
        return pd.DataFrame(), failures
    overlap = pd.concat(frames, ignore_index=True).sort_values("bar_end").reset_index(drop=True)
    OVERLAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OVERLAP_FILE.with_suffix(".parquet.tmp")
    overlap.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(OVERLAP_FILE)
    return overlap, failures


def _price_metrics(merged: pd.DataFrame, field: str, tick: float = 0.001) -> dict:
    difference = (merged[f"{field}_reconstructed"] - merged[f"{field}_direct"]).abs()
    return {
        "bars": int(len(difference)),
        "exact_count": int(np.isclose(difference, 0.0, atol=1e-12).sum()),
        "exact_rate": float(np.isclose(difference, 0.0, atol=1e-12).mean()),
        "within_one_tick_count": int((difference <= tick + 1e-12).sum()),
        "within_one_tick_rate": float((difference <= tick + 1e-12).mean()),
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
    }


def main() -> int:
    config, _, _ = _load_config()
    minute = pd.read_parquet(OUTPUT_FILE)
    daily = pd.read_parquet(DAILY_FILE)
    minute["bar_end"] = pd.to_datetime(minute["bar_end"])
    minute["trade_date"] = pd.to_datetime(minute["trade_date"]).dt.normalize()
    daily["trade_date"] = pd.to_datetime(daily["date"]).dt.normalize()
    counts = minute.groupby("trade_date").size()
    aggregate = minute.groupby("trade_date", as_index=False).agg(
        open_15m=("open", "first"),
        high_15m=("high", "max"),
        low_15m=("low", "min"),
        close_15m=("close", "last"),
        volume_15m=("volume", "sum"),
        amount_15m=("amount", "sum"),
        construction=("construction", "first"),
    )
    matched = aggregate.merge(
        daily[["trade_date", "open", "high", "low", "close", "volume", "amount"]],
        on="trade_date",
        how="inner",
        validate="one_to_one",
    )
    daily_metrics: dict[str, dict] = {}
    for construction, frame in matched.groupby("construction"):
        fields: dict[str, dict] = {}
        for field in ["open", "high", "low", "close"]:
            difference = (frame[f"{field}_15m"] - frame[field]).abs()
            fields[field] = {
                "exact_days": int(np.isclose(difference, 0.0, atol=1e-12).sum()),
                "within_one_tick_days": int((difference <= 0.001 + 1e-12).sum()),
                "mismatch_over_one_tick_days": int((difference > 0.001 + 1e-12).sum()),
                "maximum_absolute_error": float(difference.max()),
            }
        volume_relative = ((frame["volume_15m"] - frame["volume"]).abs() / frame["volume"])
        amount_relative = ((frame["amount_15m"] - frame["amount"]).abs() / frame["amount"])
        daily_metrics[str(construction)] = {
            "days": int(len(frame)),
            "price": fields,
            "maximum_relative_volume_error": float(volume_relative.max()),
            "maximum_relative_amount_error": float(amount_relative.max()),
        }

    direct_dates = sorted(
        set(minute.loc[minute["construction"] == "exchange_15m_kline", "trade_date"])
    )
    overlap, failures = _download_overlap_sample(direct_dates)
    overlap_metrics: dict = {"status": "FAIL", "failures": failures}
    if not overlap.empty:
        direct = minute.loc[
            minute["construction"] == "exchange_15m_kline",
            ["bar_end", "open", "high", "low", "close", "volume", "amount"],
        ]
        merged_overlap = overlap[
            ["bar_end", "trade_date", "open", "high", "low", "close", "volume", "amount"]
        ].merge(
            direct,
            on="bar_end",
            suffixes=("_reconstructed", "_direct"),
            how="inner",
            validate="one_to_one",
        )
        overlap_metrics = {
            "status": "PASS" if not failures else "WARN",
            "sample_days": int(merged_overlap["trade_date"].nunique()),
            "sample_bars": int(len(merged_overlap)),
            "price": {
                field: _price_metrics(merged_overlap, field)
                for field in ["open", "high", "low", "close"]
            },
            "volume": {
                "exact_bars": int(
                    np.isclose(
                        merged_overlap["volume_reconstructed"],
                        merged_overlap["volume_direct"],
                        atol=0.5,
                    ).sum()
                ),
                "maximum_relative_error": float(
                    (
                        (merged_overlap["volume_reconstructed"] - merged_overlap["volume_direct"]).abs()
                        / merged_overlap["volume_direct"]
                    ).max()
                ),
            },
            "amount": {
                "definition": "重建值为成交价乘成交量估算",
                "maximum_relative_error": float(
                    (
                        (merged_overlap["amount_reconstructed"] - merged_overlap["amount_direct"]).abs()
                        / merged_overlap["amount_direct"]
                    ).max()
                ),
            },
            "failures": failures,
        }

    reconstruction_daily = daily_metrics.get("transaction_reconstructed_15m", {})
    high_bad = reconstruction_daily.get("price", {}).get("high", {}).get(
        "mismatch_over_one_tick_days", 0
    )
    low_bad = reconstruction_daily.get("price", {}).get("low", {}).get(
        "mismatch_over_one_tick_days", 0
    )
    errors: list[dict] = []
    warnings: list[dict] = []
    if (counts != 16).any():
        errors.append({"code": "INCOMPLETE_TRADING_DAY"})
    if failures:
        errors.append({"code": "OVERLAP_AUDIT_DOWNLOAD_FAILED", "failures": failures})
    if high_bad or low_bad:
        warnings.append(
            {
                "code": "RECONSTRUCTED_EXTREME_PRICE_UNDERCOVERAGE",
                "high_mismatch_over_one_tick_days": int(high_bad),
                "low_mismatch_over_one_tick_days": int(low_bad),
                "impact": "重建区间不应直接用于依赖影线、FVG、ATR极值或盘中止损触发的正式回测",
            }
        )
    warnings.append(
        {
            "code": "POST_CLOSE_EXCLUDED",
            "impact": "2026-07-06以后15:05至15:30盘后固定价格成交不在主15分钟序列中",
        }
    )
    factor_readiness = {
        "all_five_years": {
            "status": "LIMITED",
            "allowed_for_prototype": [
                "以bar收盘价构造的收益、动量和实现波动代理",
                "成交量及相对成交量",
                "交易时段位置和日历特征",
            ],
            "not_allowed_for_formal_backtest": [
                "依赖精确high/low的FVG、影线、ATR极值",
                "使用bar内high/low判定成交、止损或止盈",
            ],
        },
        "direct_period_from_2024_07_29": {
            "status": "READY_WITH_EXECUTION_CAVEATS",
            "allowed": "可计算OHLCV技术因子；成交仍不能假设在bar最高或最低价完成",
        },
    }
    status = "FAIL" if errors else "WARN"
    report = {
        "status": status,
        "scope": "510300五年未复权15分钟覆盖与因子可用性",
        "checked_at": datetime.now(ZoneInfo(config["timezone"])).isoformat(),
        "file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": _sha256_file(OUTPUT_FILE),
        "coverage": {
            "status": "PASS" if not (counts != 16).any() else "FAIL",
            "row_count": int(len(minute)),
            "trading_day_count": int(minute["trade_date"].nunique()),
            "first_bar_end": minute["bar_end"].min().isoformat(),
            "last_bar_end": minute["bar_end"].max().isoformat(),
            "bars_per_day_distribution": {
                str(int(key)): int(value) for key, value in counts.value_counts().items()
            },
        },
        "daily_reconciliation_by_construction": daily_metrics,
        "reconstruction_overlap_audit": overlap_metrics,
        "factor_readiness": factor_readiness,
        "errors": errors,
        "warnings": warnings,
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

