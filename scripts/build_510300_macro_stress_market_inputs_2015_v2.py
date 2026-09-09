"""构建并审计2015起点敏感性研究的510300与H00300行情输入。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_START = pd.Timestamp("2015-01-01")
DATA_END = pd.Timestamp("2026-08-25")
EARLY_ETF_PATH = ROOT / "data" / "raw" / "r6" / "510300_daily.parquet"
CURRENT_ETF_PATH = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
EARLY_BENCHMARK_PATH = (
    ROOT / "data" / "raw" / "r6" / "H00300_total_return_daily.parquet"
)
CURRENT_BENCHMARK_PATH = (
    ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
)
OUTPUT_ETF_PATH = ROOT / "data" / "raw" / "market" / "510300_daily_2015_v2.parquet"
OUTPUT_BENCHMARK_PATH = (
    ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_2015_v2.parquet"
)
AUDIT_PATH = ROOT / "reports" / "data_quality" / "510300_market_inputs_2015_v2.json"
MANIFEST_PATH = ROOT / "config" / "510300_macro_stress_avoidance_2015_v2_manifest.json"

REQUIRED_SOURCE_HASHES = {
    "data/raw/r6/510300_daily.parquet": (
        "842528e66948899c28bca26b1c07e4ee03406ede7b25690a5204da8e61980979"
    ),
    "data/raw/market/510300_daily_raw.parquet": (
        "640cc9ec1155c00ad62ed728cce23ab2a39240122af4b7a122974a7c523a9aa4"
    ),
    "data/raw/r6/H00300_total_return_daily.parquet": (
        "2846746e1ae199cfae736655b29d90adeb7ca7ba339c04e87b46b98431f09961"
    ),
    "data/raw/market/H00300_total_return_daily_raw.parquet": (
        "49a2ed230fe26d8715ad0ec8fa64dc90fbf945d39bb370fbd2efda437bd86733"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f"{path.stem}.", suffix=".tmp.parquet", dir=path.parent
    )
    os.close(handle)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" not in frame or "close" not in frame:
        raise ValueError(f"行情输入缺少date或close：{path}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError(f"行情输入日期重复：{path}")
    return frame


def _overlap_audit(
    early: pd.DataFrame,
    current: pd.DataFrame,
    fields: list[str],
    label: str,
) -> dict[str, Any]:
    available = [field for field in fields if field in early and field in current]
    left = early.loc[:, ["date", *available]]
    right = current.loc[:, ["date", *available]]
    merged = left.merge(right, on="date", suffixes=("_early", "_current"))
    if merged.empty:
        raise ValueError(f"{label}没有可核验重叠区间")
    field_audit: dict[str, Any] = {}
    for field in available:
        early_values = pd.to_numeric(merged[f"{field}_early"], errors="coerce")
        current_values = pd.to_numeric(merged[f"{field}_current"], errors="coerce")
        both_missing = early_values.isna() & current_values.isna()
        comparable = ~(early_values.isna() | current_values.isna())
        absolute = (early_values - current_values).abs()
        mismatches = (~both_missing) & (~comparable | absolute.gt(1e-9))
        field_audit[field] = {
            "maximum_absolute_difference": (
                float(absolute.loc[comparable].max()) if comparable.any() else None
            ),
            "mismatches_above_1e_minus_9": int(mismatches.sum()),
        }
        if mismatches.any():
            raise ValueError(f"{label}重叠区间字段不一致：{field}")
    return {
        "rows": int(len(merged)),
        "first": str(merged["date"].min().date()),
        "last": str(merged["date"].max().date()),
        "fields": field_audit,
        "passed": True,
    }


def _combine(early: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    current_first = current["date"].min()
    early_only = early.loc[early["date"].lt(current_first)].copy()
    all_columns = list(current.columns)
    for column in early.columns:
        if column not in all_columns:
            all_columns.append(column)
    early_only = early_only.reindex(columns=all_columns)
    current = current.reindex(columns=all_columns)
    combined = pd.concat([early_only, current], ignore_index=True)
    combined = combined.loc[
        combined["date"].between(DATA_START, DATA_END)
    ].sort_values("date").reset_index(drop=True)
    if combined.empty or combined["date"].duplicated().any():
        raise ValueError("拼接行情为空或日期重复")
    if combined["date"].min() > pd.Timestamp("2015-01-05"):
        raise ValueError("拼接行情未覆盖2015年首个交易周")
    if combined["date"].max() < DATA_END:
        raise ValueError("拼接行情未覆盖冻结结束日")
    if pd.to_numeric(combined["close"], errors="coerce").isna().any():
        raise ValueError("拼接行情收盘价存在空值")
    return combined


def main() -> int:
    if MANIFEST_PATH.exists():
        raise FileExistsError("2015敏感性研究已经冻结，禁止重建行情输入")
    for path in (OUTPUT_ETF_PATH, OUTPUT_BENCHMARK_PATH, AUDIT_PATH):
        if path.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")

    source_paths = [
        EARLY_ETF_PATH,
        CURRENT_ETF_PATH,
        EARLY_BENCHMARK_PATH,
        CURRENT_BENCHMARK_PATH,
    ]
    actual_hashes = {_relative(path): sha256_file(path) for path in source_paths}
    if actual_hashes != REQUIRED_SOURCE_HASHES:
        raise ValueError("行情源文件哈希与冻结来源不匹配")

    early_etf = _read(EARLY_ETF_PATH)
    current_etf = _read(CURRENT_ETF_PATH)
    early_benchmark = _read(EARLY_BENCHMARK_PATH)
    current_benchmark = _read(CURRENT_BENCHMARK_PATH)
    etf_overlap = _overlap_audit(
        early_etf,
        current_etf,
        ["open", "high", "low", "close"],
        "510300",
    )
    benchmark_overlap = _overlap_audit(
        early_benchmark,
        current_benchmark,
        ["close"],
        "H00300",
    )
    etf = _combine(early_etf, current_etf)
    benchmark = _combine(early_benchmark, current_benchmark)
    missing_benchmark_dates = sorted(
        set(etf["date"]) - set(benchmark["date"])
    )
    if missing_benchmark_dates:
        raise ValueError(
            "H00300缺少510300交易日："
            + ",".join(str(value.date()) for value in missing_benchmark_dates[:10])
        )

    _atomic_parquet(OUTPUT_ETF_PATH, etf)
    _atomic_parquet(OUTPUT_BENCHMARK_PATH, benchmark)
    audit = {
        "status": "PASS",
        "study_id": "510300_MACRO_STRESS_AVOIDANCE_2015_START_V2",
        "scope": "2015-01-01至2026-08-25；冻结早期文件加当前文件精确重叠续接",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_hashes": actual_hashes,
        "overlap": {
            "510300": etf_overlap,
            "H00300": benchmark_overlap,
        },
        "outputs": {
            _relative(OUTPUT_ETF_PATH): {
                "sha256": sha256_file(OUTPUT_ETF_PATH),
                "rows": int(len(etf)),
                "first": str(etf["date"].min().date()),
                "last": str(etf["date"].max().date()),
                "duplicate_dates": int(etf["date"].duplicated().sum()),
            },
            _relative(OUTPUT_BENCHMARK_PATH): {
                "sha256": sha256_file(OUTPUT_BENCHMARK_PATH),
                "rows": int(len(benchmark)),
                "first": str(benchmark["date"].min().date()),
                "last": str(benchmark["date"].max().date()),
                "duplicate_dates": int(benchmark["date"].duplicated().sum()),
            },
        },
        "missing_h00300_dates_on_510300_calendar": 0,
        "governance": {
            "future_return_labels_computed": False,
            "performance_metrics_computed": False,
            "backtest_run": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }
    _atomic_json(AUDIT_PATH, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
