"""采集含已退市债券的可转债日线与逐日转股溢价；不计算任何策略收益。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "raw" / "cb_double_low_v1"
CHECKPOINT_DIR = OUTPUT_DIR / ".bond_checkpoints"
MASTER_PATH = OUTPUT_DIR / "bond_master.parquet"
PANEL_PATH = OUTPUT_DIR / "daily_panel.parquet"
VISIBLE_PANEL_PATH = OUTPUT_DIR / "visible_panel_through_2022.parquet"
SEALED_REPLICATION_PANEL_PATH = OUTPUT_DIR / "sealed_replication_panel_2023_2026.parquet"
STATUS_PATH = OUTPUT_DIR / "collection_status.json"
MINIMUM_LISTING_DATE = pd.Timestamp("2011-01-01")
DATA_END = pd.Timestamp("2026-08-14")
VISIBLE_END = pd.Timestamp("2022-12-30")
REPLICATION_START = pd.Timestamp("2023-01-03")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _clean_code(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def normalize_master(raw: pd.DataFrame) -> pd.DataFrame:
    """按东方财富可转债总表的稳定列位置标准化非绩效主表。"""

    if raw.shape[1] < 19:
        raise ValueError(f"可转债总表列数不足：{raw.shape[1]}")
    master = pd.DataFrame(
        {
            "bond_code": _clean_code(raw.iloc[:, 0]),
            "bond_name": raw.iloc[:, 1].astype(str),
            "issue_date": pd.to_datetime(raw.iloc[:, 2], errors="coerce"),
            "stock_code": _clean_code(raw.iloc[:, 5]),
            "issue_size_100m_cny": pd.to_numeric(raw.iloc[:, 14], errors="coerce"),
            "listing_date": pd.to_datetime(raw.iloc[:, 17], errors="coerce"),
        }
    )
    master = master.loc[
        master["bond_code"].str.fullmatch(r"\d{6}", na=False)
        & master["bond_code"].str.startswith(("11", "12"))
        & master["listing_date"].notna()
        & master["listing_date"].between(
            MINIMUM_LISTING_DATE, DATA_END, inclusive="both"
        )
    ].copy()
    master["exchange"] = np.where(
        master["bond_code"].str.startswith("11"), "SSE", "SZSE"
    )
    master["source"] = "akshare.bond_zh_cov_eastmoney"
    master.sort_values(["listing_date", "bond_code"], inplace=True)
    master.drop_duplicates("bond_code", keep="last", inplace=True)
    master.reset_index(drop=True, inplace=True)
    if master.empty or master["bond_code"].duplicated().any():
        raise ValueError("标准化可转债主表为空或代码重复")
    return master


def normalize_price_history(raw: pd.DataFrame, bond_code: str) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{bond_code}日线缺字段：{sorted(missing)}")
    frame = raw[list(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame["date"].notna()
        & frame["date"].le(DATA_END)
        & frame["open"].gt(0.0)
        & frame["high"].gt(0.0)
        & frame["low"].gt(0.0)
        & frame["close"].gt(0.0)
        & frame["volume"].ge(0.0)
    ].copy()
    frame.sort_values("date", inplace=True)
    frame.drop_duplicates("date", keep="last", inplace=True)
    return frame.reset_index(drop=True)


def normalize_value_history(raw: pd.DataFrame, bond_code: str) -> pd.DataFrame:
    """按东方财富历史价值表的稳定列位置标准化点时价值字段。"""

    if raw.shape[1] < 6:
        raise ValueError(f"{bond_code}价值历史列数不足：{raw.shape[1]}")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),
            "value_table_close": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
            "bond_value": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
            "conversion_value": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "bond_premium_rate_pct": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
            "conversion_premium_rate_pct": pd.to_numeric(
                raw.iloc[:, 5], errors="coerce"
            ),
        }
    )
    frame = frame.loc[frame["date"].notna() & frame["date"].le(DATA_END)].copy()
    frame.sort_values("date", inplace=True)
    frame.drop_duplicates("date", keep="last", inplace=True)
    return frame.reset_index(drop=True)


def _call_with_retry(
    call: Callable[[], pd.DataFrame],
    *,
    label: str,
    attempts: int,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = call()
            if result is None or result.empty:
                raise ValueError("返回空表")
            return result
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2.0**attempt, 12.0))
    raise RuntimeError(
        f"{label}连续{attempts}次失败：{type(last_error).__name__}: {last_error}"
    )


def collect_one_bond(
    bond_code: str,
    listing_date: pd.Timestamp,
    *,
    attempts: int,
) -> dict[str, Any]:
    checkpoint = CHECKPOINT_DIR / f"{bond_code}.parquet"
    if checkpoint.is_file():
        frame = pd.read_parquet(checkpoint)
        return {
            "bond_code": bond_code,
            "status": "CHECKPOINT_REUSED",
            "rows": int(len(frame)),
            "checkpoint": checkpoint,
        }
    exchange_prefix = "sh" if bond_code.startswith("11") else "sz"
    price_raw = _call_with_retry(
        lambda: ak.bond_zh_hs_cov_daily(symbol=f"{exchange_prefix}{bond_code}"),
        label=f"{bond_code}日线",
        attempts=attempts,
    )
    value_raw = _call_with_retry(
        lambda: ak.bond_zh_cov_value_analysis(symbol=bond_code),
        label=f"{bond_code}价值历史",
        attempts=attempts,
    )
    price = normalize_price_history(price_raw, bond_code)
    value = normalize_value_history(value_raw, bond_code)
    frame = price.merge(value, on="date", how="left", validate="one_to_one")
    frame = frame.loc[
        frame["date"].ge(pd.Timestamp(listing_date)) & frame["date"].le(DATA_END)
    ].copy()
    if frame.empty:
        raise ValueError(f"{bond_code}上市日至截止日没有交易行")
    frame["bond_code"] = bond_code
    frame["turnover_notional_proxy_cny"] = frame["close"] * frame["volume"]
    frame["close_cross_source_absolute_difference"] = (
        frame["close"] - frame["value_table_close"]
    ).abs()
    frame["price_source"] = "akshare.bond_zh_hs_cov_daily_sina"
    frame["value_source"] = "akshare.bond_zh_cov_value_analysis_eastmoney"
    columns = [
        "bond_code",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover_notional_proxy_cny",
        "bond_value",
        "conversion_value",
        "bond_premium_rate_pct",
        "conversion_premium_rate_pct",
        "value_table_close",
        "close_cross_source_absolute_difference",
        "price_source",
        "value_source",
    ]
    frame = frame[columns].sort_values("date").reset_index(drop=True)
    atomic_parquet(checkpoint, frame)
    return {
        "bond_code": bond_code,
        "status": "DOWNLOADED",
        "rows": int(len(frame)),
        "checkpoint": checkpoint,
    }


def assemble_panel(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pieces: list[pd.DataFrame] = []
    missing: list[str] = []
    for bond_code in master["bond_code"]:
        checkpoint = CHECKPOINT_DIR / f"{bond_code}.parquet"
        if not checkpoint.is_file():
            missing.append(bond_code)
            continue
        pieces.append(pd.read_parquet(checkpoint))
    if missing:
        raise RuntimeError(f"仍有{len(missing)}只债券缺少检查点")
    panel = pd.concat(pieces, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel.sort_values(["bond_code", "date"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    if panel.empty or panel["date"].isna().any():
        raise ValueError("合并日线为空或日期无效")
    if panel.duplicated(["bond_code", "date"]).any():
        raise ValueError("合并日线债券日期主键重复")
    finite_prices = np.isfinite(
        panel[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    ).all(axis=1)
    if not bool(finite_prices.all()) or (panel[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("合并日线价格或成交量含非有限值")
    value_coverage = float(panel["conversion_premium_rate_pct"].notna().mean())
    cross_source = panel["close_cross_source_absolute_difference"].dropna()
    audit = {
        "row_count": int(len(panel)),
        "bond_count": int(panel["bond_code"].nunique()),
        "first_date": panel["date"].min().date().isoformat(),
        "last_date": panel["date"].max().date().isoformat(),
        "conversion_premium_row_coverage": value_coverage,
        "cross_source_comparable_row_count": int(len(cross_source)),
        "cross_source_close_exact_within_0_001_fraction": (
            float(cross_source.le(0.001).mean()) if not cross_source.empty else None
        ),
        "strategy_return_or_rank_computed": False,
    }
    return panel, audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="采集含已退市债券的可转债双低原始历史"
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError("workers必须在1到12之间")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    raw_master = _call_with_retry(
        ak.bond_zh_cov,
        label="可转债总表",
        attempts=args.attempts,
    )
    master = normalize_master(raw_master)
    atomic_parquet(MASTER_PATH, master)
    tasks = [
        (row.bond_code, pd.Timestamp(row.listing_date))
        for row in master.itertuples(index=False)
    ]
    completed = 0
    failures: list[dict[str, str]] = []
    # Sina历史接口用V8解码；V8隔离器不能在线程间并发初始化，因此使用
    # Windows子进程隔离每个解码运行时。单债检查点使异常退出后仍可续跑。
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                collect_one_bond,
                code,
                listing_date,
                attempts=args.attempts,
            ): code
            for code, listing_date in tasks
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                future.result()
                completed += 1
            except Exception as exc:
                failures.append(
                    {
                        "bond_code": code,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if (completed + len(failures)) % 25 == 0:
                print(
                    f"进度 {completed + len(failures)}/{len(tasks)}；"
                    f"成功 {completed}；失败 {len(failures)}",
                    flush=True,
                )
    status: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "PARTIAL_FAILED" if failures else "COLLECTION_COMPLETE",
        "retrieved_at": retrieved_at.isoformat(),
        "data_end": DATA_END.date().isoformat(),
        "minimum_listing_date": MINIMUM_LISTING_DATE.date().isoformat(),
        "requested_bond_count": int(len(tasks)),
        "completed_bond_count": int(completed),
        "failed_bond_count": int(len(failures)),
        "failures": failures,
        "strategy_return_or_rank_computed": False,
        "source_endpoints": [
            "akshare.bond_zh_cov_eastmoney",
            "akshare.bond_zh_hs_cov_daily_sina",
            "akshare.bond_zh_cov_value_analysis_eastmoney",
        ],
    }
    if failures:
        atomic_json(STATUS_PATH, status)
        print(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False))
        return 2
    panel, audit = assemble_panel(master)
    atomic_parquet(PANEL_PATH, panel)
    visible_panel = panel.loc[panel["date"].le(VISIBLE_END)].copy()
    sealed_replication_panel = panel.loc[
        panel["date"].ge(REPLICATION_START)
    ].copy()
    if visible_panel.empty or sealed_replication_panel.empty:
        raise RuntimeError("可见期或封存复验物理分区为空")
    atomic_parquet(VISIBLE_PANEL_PATH, visible_panel)
    atomic_parquet(SEALED_REPLICATION_PANEL_PATH, sealed_replication_panel)
    status.update(
        {
            "status": "COLLECTION_COMPLETE",
            "panel_audit": audit,
            "master_path": MASTER_PATH.relative_to(ROOT).as_posix(),
            "master_sha256": sha256_file(MASTER_PATH),
            "panel_path": PANEL_PATH.relative_to(ROOT).as_posix(),
            "panel_sha256": sha256_file(PANEL_PATH),
            "visible_panel_path": VISIBLE_PANEL_PATH.relative_to(ROOT).as_posix(),
            "visible_panel_rows": int(len(visible_panel)),
            "visible_panel_sha256": sha256_file(VISIBLE_PANEL_PATH),
            "sealed_replication_panel_path": (
                SEALED_REPLICATION_PANEL_PATH.relative_to(ROOT).as_posix()
            ),
            "sealed_replication_panel_rows": int(len(sealed_replication_panel)),
            "sealed_replication_panel_sha256": sha256_file(
                SEALED_REPLICATION_PANEL_PATH
            ),
            "physical_partition_overlap_rows": 0,
            "physical_partition_date_order_valid": bool(
                visible_panel["date"].max()
                < sealed_replication_panel["date"].min()
            ),
        }
    )
    atomic_json(STATUS_PATH, status)
    print(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
