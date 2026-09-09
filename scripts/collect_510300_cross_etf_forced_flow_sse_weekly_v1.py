"""采集冻结大ETF池的上交所官方周度基金份额，不读取策略收益。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = (
    ROOT / "config" / "510300_cross_etf_forced_flow_binary_screen_v1_candidates.yaml"
)
PREFREEZE_RECEIPT_PATH = (
    ROOT
    / "reports"
    / "frozen"
    / "510300_cross_etf_forced_flow_binary_screen_v1_candidate_prefreeze_receipt.json"
)
CANDIDATE_SHA256 = "879800a413eac26c137a56d2cb4ea94c9a544a7a1538c7af09ed36550ff6a742"
PREFREEZE_RECEIPT_SHA256 = "632694d9415d3cb9e7e75e27e9bca23e3369f7d77c18d2515acb6c5fe5808e66"
SSE_QUERY_URL = "https://query.sse.com.cn/commonQuery.do"
SSE_SHARE_SQL_ID = "COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L"
HEADERS = {
    "Referer": "https://www.sse.com.cn/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    ),
}
RAW_DIRECTORY = (
    ROOT / "data" / "raw" / "cross_etf_forced_flow_v1" / "sse_weekly_snapshots"
)
OUTPUT_PATH = (
    ROOT / "data" / "raw" / "flow" / "510300_cross_etf_forced_flow_sse_weekly_v1.parquet"
)
REPORT_PATH = (
    ROOT / "reports" / "data_quality" / "510300_cross_etf_forced_flow_sse_weekly_v1.json"
)


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    """原子写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_frozen_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    """核对在份额历史采集前写入的候选合同和收据。"""

    if sha256_file(CANDIDATE_PATH) != CANDIDATE_SHA256:
        raise ValueError("份额历史采集前固定的候选合同发生漂移")
    if sha256_file(PREFREEZE_RECEIPT_PATH) != PREFREEZE_RECEIPT_SHA256:
        raise ValueError("候选预冻结收据发生漂移")
    contract = yaml.safe_load(CANDIDATE_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(PREFREEZE_RECEIPT_PATH.read_text(encoding="utf-8"))
    if (
        contract["protocol"]["state"]
        != "CANDIDATE_FAMILY_FIXED_BEFORE_SSE_WEEKLY_HISTORY_COLLECTION"
    ):
        raise ValueError("候选合同状态不允许采集")
    if receipt["status"] != "CANDIDATE_FAMILY_FROZEN_BEFORE_SSE_WEEKLY_HISTORY_COLLECTION":
        raise ValueError("候选预冻结收据状态无效")
    if receipt["candidate_contract"]["sha256"] != CANDIDATE_SHA256:
        raise ValueError("候选预冻结收据未绑定当前合同")
    return contract, receipt


def build_frozen_universe(
    contract: dict[str, Any], receipt: dict[str, Any]
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """仅用 2020 年及以前信息重建冻结的大型上交所 A 股 ETF 池。"""

    definition = contract["universe_freeze"]
    price_path = ROOT / definition["price_panel"]
    benchmark_path = ROOT / definition["benchmark_panel"]
    expected_hashes = receipt["universe_selection_inputs"]
    for path in [price_path, benchmark_path]:
        relative = path.relative_to(ROOT).as_posix()
        if sha256_file(path) != expected_hashes[relative]:
            raise ValueError(f"冻结宇宙输入漂移：{relative}")

    panel = pd.read_parquet(
        price_path,
        columns=["date", "con_code", "total_return_close", "raw_close", "amount"],
    )
    benchmark = pd.read_parquet(benchmark_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="raise").dt.normalize()
    benchmark["date"] = pd.to_datetime(
        benchmark["date"], errors="raise"
    ).dt.normalize()
    panel.sort_values(["con_code", "date"], kind="mergesort", inplace=True)
    benchmark.sort_values("date", kind="mergesort", inplace=True)

    correlation_start, correlation_end = map(
        pd.Timestamp, definition["return_correlation_period"]
    )
    liquidity_start, liquidity_end = map(
        pd.Timestamp, definition["liquidity_period"]
    )
    correlation_panel = panel.loc[
        panel["date"].between(correlation_start, correlation_end),
        ["date", "con_code", "total_return_close"],
    ]
    close = correlation_panel.pivot(
        index="date", columns="con_code", values="total_return_close"
    )
    returns = close.pct_change(fill_method=None)
    benchmark_returns = (
        benchmark.set_index("date")["close"]
        .astype(float)
        .pct_change(fill_method=None)
        .reindex(returns.index)
    )
    correlations = returns.corrwith(benchmark_returns)
    overlaps = returns.notna().mul(benchmark_returns.notna(), axis=0).sum()
    liquidity = (
        panel.loc[
            panel["date"].between(liquidity_start, liquidity_end),
            ["con_code", "amount"],
        ]
        .groupby("con_code")["amount"]
        .agg(average_daily_amount_cny="mean", liquidity_days="count")
    )
    last_trade = (
        panel.loc[
            panel["date"].le(pd.Timestamp(definition["selection_information_end"]))
        ]
        .groupby("con_code")["date"]
        .max()
        .rename("last_trade_date")
    )
    diagnostics = pd.concat(
        [
            correlations.rename("h00300_return_correlation"),
            overlaps.rename("return_overlap_days"),
            liquidity,
            last_trade,
        ],
        axis=1,
    ).reset_index(names="ts_code")
    excluded_prefixes = tuple(definition["excluded_cross_border_code_prefixes"])
    selected = diagnostics.loc[
        diagnostics["ts_code"].astype(str).str.endswith(
            definition["exchange_suffix"]
        )
        & ~diagnostics["ts_code"].astype(str).str.startswith(excluded_prefixes)
        & diagnostics["return_overlap_days"].ge(
            int(definition["minimum_return_overlap_days"])
        )
        & diagnostics["h00300_return_correlation"].ge(
            float(definition["minimum_h00300_return_correlation"])
        )
        & diagnostics["liquidity_days"].ge(
            int(definition["minimum_liquidity_days"])
        )
        & diagnostics["average_daily_amount_cny"].ge(
            float(definition["minimum_average_daily_amount_cny"])
        )
        & diagnostics["last_trade_date"].ge(
            pd.Timestamp(definition["minimum_last_trade_date"])
        )
    ].copy()
    symbols = sorted(selected["ts_code"].astype(str).tolist())
    expected = sorted(definition["broad_pool_expected_symbols"])
    if symbols != expected:
        raise ValueError(f"冻结大型ETF池重建不一致：actual={symbols}, expected={expected}")
    hs300_symbols = sorted(definition["hs300_pool_symbols"])
    if not set(hs300_symbols).issubset(symbols):
        raise ValueError("沪深300子池不是大型ETF池的子集")
    return symbols, selected.sort_values("ts_code"), panel


def weekly_snapshot_dates(contract: dict[str, Any], panel: pd.DataFrame) -> list[pd.Timestamp]:
    """从冻结价格面板生成每个 ISO 周最后一个交易日。"""

    collection = contract["official_share_collection"]
    start = pd.Timestamp(collection["first_possible_snapshot"])
    end = pd.Timestamp(collection["last_possible_snapshot"])
    dates = (
        panel.loc[
            panel["con_code"].eq("510300.SH")
            & panel["date"].between(start, end),
            "date",
        ]
        .drop_duplicates()
        .sort_values()
    )
    iso = dates.dt.isocalendar()
    frame = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "iso_year": iso["year"].to_numpy(),
            "iso_week": iso["week"].to_numpy(),
        }
    )
    snapshots = (
        frame.groupby(["iso_year", "iso_week"], sort=True)["date"].max().tolist()
    )
    if not snapshots or snapshots != sorted(set(snapshots)):
        raise ValueError("周度快照日期为空、重复或乱序")
    return [pd.Timestamp(value).normalize() for value in snapshots]


def parse_payload(
    payload: dict[str, Any], date: pd.Timestamp, symbols: list[str]
) -> list[dict[str, Any]]:
    """从单日上交所原始响应中严格抽取冻结ETF池。"""

    result = payload.get("result")
    if not isinstance(result, list) or not result:
        raise ValueError(f"{date.date()}上交所份额响应没有result记录")
    wanted = {symbol.split(".")[0]: symbol for symbol in symbols}
    rows = [row for row in result if str(row.get("SEC_CODE")) in wanted]
    seen = [str(row.get("SEC_CODE")) for row in rows]
    if sorted(seen) != sorted(wanted):
        raise ValueError(
            f"{date.date()}冻结ETF池不完整：actual={sorted(seen)}, expected={sorted(wanted)}"
        )
    normalized: list[dict[str, Any]] = []
    for row in rows:
        sec_code = str(row["SEC_CODE"])
        stat_date = pd.to_datetime(row.get("STAT_DATE"), errors="raise").normalize()
        if stat_date != date:
            raise ValueError(f"{date.date()}上交所记录日期错位：{sec_code}={stat_date.date()}")
        share_10k = float(row["TOT_VOL"])
        if not np.isfinite(share_10k) or share_10k <= 0.0:
            raise ValueError(f"{date.date()}上交所份额非法：{sec_code}")
        normalized.append(
            {
                "date": date,
                "ts_code": wanted[sec_code],
                "sec_code": sec_code,
                "fund_name": row.get("SEC_NAME"),
                "etf_type": row.get("ETF_TYPE"),
                "fund_share_10k": share_10k,
                "fund_shares": share_10k * 10_000.0,
                "source_result_row_count": len(result),
                "source": f"sse.{SSE_SHARE_SQL_ID}",
            }
        )
    return sorted(normalized, key=lambda item: item["ts_code"])


def fetch_or_load_snapshot(
    date: pd.Timestamp,
    symbols: list[str],
    *,
    maximum_retries: int,
) -> tuple[list[dict[str, Any]], Path, str, bool]:
    """读取既有不可变原始响应，或从上交所采集并原子落盘。"""

    raw_path = RAW_DIRECTORY / f"{date.date().isoformat()}.json"
    if raw_path.exists():
        wrapper = json.loads(raw_path.read_text(encoding="utf-8"))
        if wrapper.get("requested_stat_date") != date.date().isoformat():
            raise ValueError(f"原始响应请求日期错位：{raw_path}")
        records = parse_payload(wrapper["payload"], date, symbols)
        return records, raw_path, sha256_file(raw_path), True

    params = {
        "isPagination": "true",
        "pageHelp.pageSize": "10000",
        "pageHelp.pageNo": "1",
        "pageHelp.beginPage": "1",
        "pageHelp.cacheSize": "1",
        "pageHelp.endPage": "1",
        "sqlId": SSE_SHARE_SQL_ID,
        "STAT_DATE": date.strftime("%Y-%m-%d"),
    }
    final_error: Exception | None = None
    for attempt in range(1, maximum_retries + 1):
        try:
            time.sleep(0.10)
            response = requests.get(
                SSE_QUERY_URL,
                params=params,
                headers=HEADERS,
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            records = parse_payload(payload, date, symbols)
            wrapper = {
                "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "requested_stat_date": date.date().isoformat(),
                "request_url": SSE_QUERY_URL,
                "sql_id": SSE_SHARE_SQL_ID,
                "http_status": int(response.status_code),
                "payload": payload,
            }
            atomic_json(wrapper, raw_path)
            return records, raw_path, sha256_file(raw_path), False
        except Exception as exc:  # noqa: BLE001 - 需要保留网络和解析的最终错误
            final_error = exc
            if attempt < maximum_retries:
                time.sleep(float(attempt))
    raise RuntimeError(f"{date.date()}上交所ETF份额采集失败：{final_error}")


def main() -> int:
    """采集全部固定周度快照并输出可审计的规范化面板。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum-retries", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("workers必须在1到6之间")
    if not 1 <= args.maximum_retries <= 8:
        raise ValueError("maximum-retries必须在1到8之间")

    contract, receipt = load_frozen_contract()
    symbols, universe_diagnostics, panel = build_frozen_universe(contract, receipt)
    dates = weekly_snapshot_dates(contract, panel)
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records_by_date: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    raw_hashes: dict[str, str] = {}
    reused_count = 0
    failures: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_or_load_snapshot,
                date,
                symbols,
                maximum_retries=args.maximum_retries,
            ): date
            for date in dates
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            date = futures[future]
            try:
                records, raw_path, checksum, reused = future.result()
                records_by_date[date] = records
                raw_hashes[raw_path.relative_to(ROOT).as_posix()] = checksum
                reused_count += int(reused)
            except Exception as exc:  # noqa: BLE001 - 汇总所有失败日期后停止
                failures[date.date().isoformat()] = f"{type(exc).__name__}: {exc}"
            if completed % 20 == 0 or completed == len(futures):
                print(
                    f"周度份额进度 {completed}/{len(futures)}，"
                    f"成功 {len(records_by_date)}，失败 {len(failures)}",
                    flush=True,
                )
    if failures:
        failure_path = REPORT_PATH.with_name(REPORT_PATH.stem + "_failure.json")
        atomic_json(
            {
                "status": "FAILED_INCOMPLETE_SSE_WEEKLY_SHARE_COLLECTION",
                "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "expected_snapshot_dates": len(dates),
                "successful_snapshot_dates": len(records_by_date),
                "failures": failures,
            },
            failure_path,
        )
        raise RuntimeError(
            f"上交所周度份额采集不完整：{len(failures)}个日期失败；"
            f"详见{failure_path.relative_to(ROOT).as_posix()}"
        )

    output = pd.DataFrame(
        [record for date in dates for record in records_by_date[date]]
    ).sort_values(["date", "ts_code"], kind="mergesort")
    output.reset_index(drop=True, inplace=True)
    if output[["date", "ts_code"]].duplicated().any():
        raise ValueError("规范化周度份额存在重复证券日期")
    expected_rows = len(dates) * len(symbols)
    if len(output) != expected_rows:
        raise ValueError(f"规范化周度份额行数应为{expected_rows}，实际为{len(output)}")
    counts = output.groupby("date")["ts_code"].nunique()
    if not counts.eq(len(symbols)).all():
        raise ValueError("至少一个周度快照未覆盖完整冻结ETF池")
    output["retrieved_via_raw_snapshot"] = True
    output["historical_publication_timestamp_verified"] = False
    output["execution_earliest"] = "T_PLUS_2_OPEN"
    atomic_parquet(output, OUTPUT_PATH)

    output_hash = sha256_file(OUTPUT_PATH)
    report = {
        "status": "PASS_COMPLETE_SSE_WEEKLY_SHARE_COLLECTION_NO_FILL",
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_contract": {
            "path": CANDIDATE_PATH.relative_to(ROOT).as_posix(),
            "sha256": CANDIDATE_SHA256,
        },
        "candidate_prefreeze_receipt": {
            "path": PREFREEZE_RECEIPT_PATH.relative_to(ROOT).as_posix(),
            "sha256": PREFREEZE_RECEIPT_SHA256,
        },
        "universe": {
            "selection_information_end": contract["universe_freeze"][
                "selection_information_end"
            ],
            "symbols": symbols,
            "symbol_count": len(symbols),
            "hs300_pool_symbols": contract["universe_freeze"]["hs300_pool_symbols"],
            "diagnostics": json.loads(
                universe_diagnostics.to_json(
                    orient="records", date_format="iso", force_ascii=False
                )
            ),
        },
        "collection": {
            "source": f"sse.{SSE_SHARE_SQL_ID}",
            "snapshot_count": len(dates),
            "first_snapshot": dates[0].date().isoformat(),
            "last_snapshot": dates[-1].date().isoformat(),
            "rows": len(output),
            "expected_rows": expected_rows,
            "reused_raw_snapshot_count": reused_count,
            "new_raw_snapshot_count": len(dates) - reused_count,
            "missing_snapshot_count": 0,
            "missing_symbol_date_count": 0,
            "fill_used": False,
            "historical_publication_timestamp_verified": False,
            "conservative_execution_time": "T_PLUS_2_OPEN",
        },
        "output": {
            "path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "sha256": output_hash,
            "rows": len(output),
        },
        "raw_snapshot_hashes": dict(sorted(raw_hashes.items())),
    }
    atomic_json(report, REPORT_PATH)
    print(
        json.dumps(
            {
                "status": report["status"],
                "symbols": len(symbols),
                "snapshots": len(dates),
                "rows": len(output),
                "output_sha256": output_hash,
                "report": REPORT_PATH.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
