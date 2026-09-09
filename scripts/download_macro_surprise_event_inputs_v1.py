"""下载五类中国宏观经济日历实际值与预期值，仅形成事件输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONTRACT = (
    ROOT / "config" / "510300_macro_surprise_event_binary_screen_v1_candidates.yaml"
)
CANDIDATE_CONTRACT_SHA256 = (
    "60ffdf04bc4e7ec8bec83f8333de241f47d1d1b3fb53e6ad703eff71141130e7"
)
OUTPUT_PATH = ROOT / "data" / "raw" / "macro" / "macro_surprise_calendar_raw_v1.parquet"
AUDIT_PATH = ROOT / "reports" / "data_quality" / "macro_surprise_calendar_raw_v1.json"
START_DATE = pd.Timestamp("2015-01-01")
CUTOFF_DATE = pd.Timestamp("2025-09-30")
SERIES = [
    ("PMI", "macro_china_pmi_yearly"),
    ("M2", "macro_china_m2_yearly"),
    ("EXPORTS", "macro_china_exports_yoy"),
    ("IMPORTS", "macro_china_imports_yoy"),
    ("INDUSTRIAL_PRODUCTION", "macro_china_industrial_production_yoy"),
]


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fetch_with_retry(function: Callable[[], pd.DataFrame], retries: int) -> pd.DataFrame:
    """有限重试读取一个公开经济日历端点。"""

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return function()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise last_error


def normalize_series(
    series_id: str,
    function_name: str,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按端点固定的五列顺序标准化产品、日期、实际、预期与前值。"""

    if raw.shape[1] != 5:
        raise ValueError(f"{function_name}字段数不再是固定的5列")
    frame = raw.iloc[:, :5].copy()
    frame.columns = ["product", "release_date", "actual", "forecast", "previous"]
    frame["release_date"] = pd.to_datetime(frame["release_date"], errors="raise").dt.normalize()
    for column in ["actual", "forecast", "previous"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame["release_date"].between(START_DATE, CUTOFF_DATE, inclusive="both")
    ].copy()
    frame.sort_values("release_date", kind="mergesort", inplace=True)
    if frame["release_date"].duplicated().any():
        duplicates = frame.loc[
            frame["release_date"].duplicated(keep=False), "release_date"
        ].dt.date.astype(str).tolist()
        raise ValueError(f"{function_name}同一公布日存在重复记录：{duplicates[:10]}")
    frame["series_id"] = series_id
    frame["source_function"] = function_name
    frame["source"] = "AKShare/Eastmoney economic calendar"
    usable = frame["actual"].notna() & frame["forecast"].notna()
    if int(usable.sum()) < 100:
        raise ValueError(f"{function_name}实际与预期同时可用事件不足100")
    usable_dates = frame.loc[usable, "release_date"]
    if usable_dates.max() < pd.Timestamp("2025-08-01"):
        raise ValueError(f"{function_name}可用事件末日早于2025-08")
    audit = {
        "series_id": series_id,
        "source_function": function_name,
        "raw_rows_returned": int(len(raw)),
        "rows_within_fixed_window": int(len(frame)),
        "usable_actual_and_forecast_rows": int(usable.sum()),
        "excluded_missing_actual_rows": int(frame["actual"].isna().sum()),
        "excluded_missing_forecast_rows": int(frame["forecast"].isna().sum()),
        "first_usable_date": usable_dates.min().date().isoformat(),
        "last_usable_date": usable_dates.max().date().isoformat(),
        "duplicate_release_dates": 0,
    }
    return frame.reset_index(drop=True), audit


def acquire(retries: int) -> dict[str, Any]:
    """获取固定五序列并写入统一原始事件表。"""

    if sha256_file(CANDIDATE_CONTRACT) != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("候选规则合同在事件数据下载前发生漂移")
    retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    frames: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for series_id, function_name in SERIES:
        function = getattr(ak, function_name)
        raw = fetch_with_retry(function, retries)
        frame, audit = normalize_series(series_id, function_name, raw)
        frame["retrieved_at"] = retrieved_at
        frames.append(frame)
        audits[series_id] = audit
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["release_date", "series_id"], kind="mergesort", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    if set(combined["series_id"].unique()) != {item[0] for item in SERIES}:
        raise AssertionError("固定宏观序列集合不完整")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    combined.to_parquet(temporary, index=False)
    os.replace(temporary, OUTPUT_PATH)
    usable = combined["actual"].notna() & combined["forecast"].notna()
    report = {
        "status": "PASS_EVENT_INPUT_ACQUISITION_ONLY_NO_510300_PERFORMANCE_VIEW",
        "study_id": "510300_MACRO_SURPRISE_EVENT_BINARY_SCREEN_V1",
        "generated_at": retrieved_at,
        "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
        "fixed_start_date": START_DATE.date().isoformat(),
        "fixed_cutoff_date": CUTOFF_DATE.date().isoformat(),
        "series_count": len(SERIES),
        "combined_rows": int(len(combined)),
        "combined_usable_rows": int(usable.sum()),
        "series_audit": audits,
        "artifact": {
            "file": OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "bytes": int(OUTPUT_PATH.stat().st_size),
            "sha256": sha256_file(OUTPUT_PATH),
        },
        "target_510300_or_benchmark_read": False,
        "future_return_or_performance_computed": False,
    }
    atomic_json(report, AUDIT_PATH)
    return report


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    report = acquire(max(1, args.retries))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
