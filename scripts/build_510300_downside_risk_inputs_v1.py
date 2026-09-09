"""构建并审计510300下行风险研究的冻结行情输入。

本脚本只处理输入数据，不计算任何未来风险标签、因子收益或组合收益。
原始文件保持不变；三处收盘价仅在任务专用的双源共识层中修正。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_PRICE_PATH = ROOT / "data" / "raw" / "r6" / "510300_daily.parquet"
TUSHARE_PRICE_PATH = (
    ROOT
    / "data"
    / "raw"
    / "all_etf_momentum_v1r"
    / "fund_daily_checkpoints"
    / "510300_SH.parquet"
)
CORRECTION_PATH = (
    ROOT / "data" / "reference" / "510300_downside_risk_price_corrections_v1.csv"
)
CANONICAL_PRICE_PATH = (
    ROOT / "data" / "raw" / "market" / "510300_daily_downside_risk_v1.parquet"
)
BENCHMARK_PATH = ROOT / "data" / "raw" / "r6" / "H00300_total_return_daily.parquet"
EXTERNAL_BENCHMARK_PATH = (
    ROOT
    / "data"
    / "raw"
    / "external_validation"
    / "csi300_2014_2021"
    / "H00300_total_return_2014_2021.parquet"
)
DIVIDEND_PATH = ROOT / "data" / "reference" / "510300_dividends.csv"
DIVIDEND_CUMULATIVE_PATH = (
    ROOT / "data" / "raw" / "fund" / "510300_dividend_cumulative_sina.parquet"
)
REPORT_PATH = (
    ROOT / "reports" / "data_quality" / "510300_downside_risk_inputs_v1.json"
)

EVALUATION_START = pd.Timestamp("2015-01-05")
EVALUATION_END = pd.Timestamp("2026-08-12")
EXPECTED_EVALUATION_ROWS = 2821
EXPECTED_CORRECTION_ROWS = 3
ALLOWED_BENCHMARK_ONLY_DATES = {pd.Timestamp("2018-06-18")}


class InputAuditError(RuntimeError):
    """输入审计不满足冻结要求。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def _load_price(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise InputAuditError(f"行情缺少字段：{sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise InputAuditError(f"行情日期重复：{path}")
    numeric = frame[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or (numeric <= 0).any().any():
        raise InputAuditError(f"行情价格含空值、零值或负值：{path}")
    frame[["open", "high", "low", "close"]] = numeric
    invalid_ohlc = (
        (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame["low"])
    )
    if invalid_ohlc.any():
        raise InputAuditError(f"行情存在OHLC约束错误：{path}")
    return frame


def _build_canonical_price() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    primary = _load_price(PRIMARY_PRICE_PATH)
    tushare = pd.read_parquet(TUSHARE_PRICE_PATH).copy()
    tushare["date"] = pd.to_datetime(tushare["date"])
    corrections = pd.read_csv(CORRECTION_PATH)
    corrections["date"] = pd.to_datetime(corrections["date"])
    if len(corrections) != EXPECTED_CORRECTION_ROWS:
        raise InputAuditError(
            f"修正登记必须恰好为{EXPECTED_CORRECTION_ROWS}行，实际为{len(corrections)}行"
        )
    if corrections[["date", "field"]].duplicated().any():
        raise InputAuditError("修正登记存在重复日期字段")

    canonical = primary.copy()
    canonical["source_original"] = canonical.get("source", "")
    canonical["correction_applied"] = False
    canonical["correction_reason"] = ""
    applied: list[dict[str, Any]] = []
    tushare_by_date = tushare.set_index("date")
    for row in corrections.itertuples(index=False):
        date = pd.Timestamp(row.date)
        field = str(row.field)
        if field not in {"open", "high", "low", "close"}:
            raise InputAuditError(f"不允许修正的字段：{field}")
        mask = canonical["date"].eq(date)
        if int(mask.sum()) != 1:
            raise InputAuditError(f"修正日期在主源中不唯一：{date.date()}")
        original = float(canonical.loc[mask, field].iloc[0])
        if not np.isclose(original, float(row.original_sina_value), atol=1e-12):
            raise InputAuditError(f"主源原值与修正登记不一致：{date.date()} {field}")
        if date not in tushare_by_date.index:
            raise InputAuditError(f"TuShare缺少修正日期：{date.date()}")
        tushare_column = {
            "open": "ts_open",
            "high": "ts_high",
            "low": "ts_low",
            "close": "ts_close",
        }[field]
        observed_tushare = float(tushare_by_date.loc[date, tushare_column])
        adjudicated = float(row.adjudicated_value)
        if not np.isclose(observed_tushare, float(row.tushare_value), atol=1e-12):
            raise InputAuditError(f"TuShare值与修正登记不一致：{date.date()} {field}")
        if not np.isclose(float(row.tencent_value), adjudicated, atol=1e-12):
            raise InputAuditError(f"腾讯值未支持裁决值：{date.date()} {field}")
        if not np.isclose(observed_tushare, adjudicated, atol=1e-12):
            raise InputAuditError(f"TuShare值未支持裁决值：{date.date()} {field}")
        canonical.loc[mask, field] = adjudicated
        canonical.loc[mask, "source"] = "CONSENSUS_TUSHARE_TENCENT_OVER_SINA"
        canonical.loc[mask, "correction_applied"] = True
        canonical.loc[mask, "correction_reason"] = str(row.reason)
        applied.append(
            {
                "date": date.date().isoformat(),
                "field": field,
                "original_value": original,
                "tushare_value": observed_tushare,
                "tencent_value": float(row.tencent_value),
                "adjudicated_value": adjudicated,
                "absolute_change": abs(adjudicated - original),
                "reason": str(row.reason),
            }
        )
    return canonical, applied


def _cross_check_price(canonical: pd.DataFrame) -> dict[str, Any]:
    tushare = pd.read_parquet(TUSHARE_PRICE_PATH).copy()
    tushare["date"] = pd.to_datetime(tushare["date"])
    left = canonical.loc[
        canonical["date"].between(EVALUATION_START, EVALUATION_END),
        ["date", "open", "high", "low", "close"],
    ]
    right = tushare.loc[
        tushare["date"].between(EVALUATION_START, EVALUATION_END),
        ["date", "ts_open", "ts_high", "ts_low", "ts_close"],
    ]
    merged = left.merge(right, on="date", how="outer", indicator=True)
    unmatched = merged.loc[merged["_merge"] != "both", "date"].dt.strftime(
        "%Y-%m-%d"
    ).tolist()
    differences: dict[str, Any] = {}
    for field in ("open", "high", "low", "close"):
        difference = (merged[field] - merged[f"ts_{field}"]).abs()
        differences[field] = {
            "maximum_absolute": float(difference.max()),
            "nonzero_rows": int((difference > 1e-12).sum()),
        }
    return {
        "overlap_rows": int((merged["_merge"] == "both").sum()),
        "unmatched_dates": unmatched,
        "differences": differences,
    }


def _cross_check_benchmark() -> dict[str, Any]:
    benchmark = _load_price_like_benchmark(BENCHMARK_PATH)
    external = _load_price_like_benchmark(EXTERNAL_BENCHMARK_PATH)
    start = pd.Timestamp("2015-01-01")
    end = pd.Timestamp("2021-08-11")
    merged = benchmark.loc[
        benchmark["date"].between(start, end), ["date", "close"]
    ].merge(
        external.loc[external["date"].between(start, end), ["date", "close"]],
        on="date",
        how="outer",
        suffixes=("_canonical", "_external"),
        indicator=True,
    )
    overlap = merged["_merge"].eq("both")
    difference = (
        merged.loc[overlap, "close_canonical"]
        - merged.loc[overlap, "close_external"]
    ).abs()
    return {
        "overlap_rows": int(overlap.sum()),
        "unmatched_dates": merged.loc[~overlap, "date"].dt.strftime(
            "%Y-%m-%d"
        ).tolist(),
        "maximum_close_difference": float(difference.max()),
        "nonzero_close_difference_rows": int((difference > 1e-12).sum()),
    }


def _load_price_like_benchmark(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    if not {"date", "close"}.issubset(frame.columns):
        raise InputAuditError(f"基准缺少date或close：{path}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any() or frame["close"].isna().any():
        raise InputAuditError(f"基准存在重复或空值：{path}")
    if (frame["close"] <= 0).any():
        raise InputAuditError(f"基准存在非正价格：{path}")
    return frame


def _audit_calendar(canonical: pd.DataFrame) -> dict[str, Any]:
    benchmark = _load_price_like_benchmark(BENCHMARK_PATH)
    price_dates = set(
        canonical.loc[
            canonical["date"].between(EVALUATION_START, EVALUATION_END), "date"
        ]
    )
    benchmark_dates = set(
        benchmark.loc[
            benchmark["date"].between(EVALUATION_START, EVALUATION_END), "date"
        ]
    )
    missing_benchmark = sorted(price_dates - benchmark_dates)
    benchmark_only = sorted(benchmark_dates - price_dates)
    return {
        "missing_benchmark_dates": [date.date().isoformat() for date in missing_benchmark],
        "benchmark_only_dates": [date.date().isoformat() for date in benchmark_only],
        "benchmark_only_dates_exactly_allowed": set(benchmark_only)
        == ALLOWED_BENCHMARK_ONLY_DATES,
    }


def _audit_dividends() -> dict[str, Any]:
    dividends = pd.read_csv(DIVIDEND_PATH)
    cumulative = pd.read_parquet(DIVIDEND_CUMULATIVE_PATH).copy()
    required = {
        "symbol",
        "record_date",
        "ex_date",
        "payment_date",
        "cash_dividend_per_share",
        "source",
    }
    missing = required.difference(dividends.columns)
    if missing:
        raise InputAuditError(f"分红表缺少字段：{sorted(missing)}")
    for field in ("record_date", "ex_date", "payment_date"):
        dividends[field] = pd.to_datetime(dividends[field])
    cumulative["ex_date"] = pd.to_datetime(cumulative["ex_date"])
    merged = dividends.merge(cumulative, on="ex_date", how="outer", indicator=True)
    implied = merged["cash_dividend_implied"].copy()
    first = implied.isna()
    implied.loc[first] = merged.loc[first, "cumulative_dividend_per_share"]
    difference = (merged["cash_dividend_per_share"] - implied).abs()
    invalid_order = ~(
        (merged["record_date"] < merged["ex_date"])
        & (merged["ex_date"] <= merged["payment_date"])
    )
    return {
        "event_count": int(len(dividends)),
        "unmatched_rows": int((merged["_merge"] != "both").sum()),
        "maximum_cash_difference": float(difference.max()),
        "invalid_date_order_rows": int(invalid_order.sum()),
        "non_https_source_rows": int(
            (~dividends["source"].astype(str).str.startswith("https://")).sum()
        ),
        "evaluation_event_count": int(
            dividends["ex_date"].between(EVALUATION_START, EVALUATION_END).sum()
        ),
    }


def build() -> dict[str, Any]:
    required_paths = [
        PRIMARY_PRICE_PATH,
        TUSHARE_PRICE_PATH,
        CORRECTION_PATH,
        BENCHMARK_PATH,
        EXTERNAL_BENCHMARK_PATH,
        DIVIDEND_PATH,
        DIVIDEND_CUMULATIVE_PATH,
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required_paths if not path.exists()]
    if missing:
        raise InputAuditError(f"缺少输入文件：{missing}")

    canonical, corrections = _build_canonical_price()
    evaluation = canonical.loc[
        canonical["date"].between(EVALUATION_START, EVALUATION_END)
    ]
    price_cross_check = _cross_check_price(canonical)
    benchmark_cross_check = _cross_check_benchmark()
    calendar = _audit_calendar(canonical)
    dividends = _audit_dividends()

    checks = {
        "canonical_full_rows_3456": len(canonical) == 3456,
        "evaluation_rows_2821": len(evaluation) == EXPECTED_EVALUATION_ROWS,
        "evaluation_dates_exact": (
            evaluation["date"].min() == EVALUATION_START
            and evaluation["date"].max() == EVALUATION_END
        ),
        "correction_rows_exact": len(corrections) == EXPECTED_CORRECTION_ROWS,
        "post_correction_tushare_dates_exact": (
            price_cross_check["overlap_rows"] == EXPECTED_EVALUATION_ROWS
            and not price_cross_check["unmatched_dates"]
        ),
        "post_correction_tushare_ohlc_exact": all(
            details["maximum_absolute"] <= 1e-12
            for details in price_cross_check["differences"].values()
        ),
        "benchmark_overlap_exact": (
            benchmark_cross_check["overlap_rows"] == 1611
            and benchmark_cross_check["maximum_close_difference"] <= 1e-12
            and not benchmark_cross_check["unmatched_dates"]
        ),
        "calendar_matches_allowed_exception": (
            not calendar["missing_benchmark_dates"]
            and calendar["benchmark_only_dates_exactly_allowed"]
        ),
        "dividends_complete_and_reconciled": (
            dividends["event_count"] == 14
            and dividends["unmatched_rows"] == 0
            and dividends["maximum_cash_difference"] <= 1e-12
            and dividends["invalid_date_order_rows"] == 0
            and dividends["non_https_source_rows"] == 0
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema_version": "1.0.0",
        "report_id": "510300_DOWNSIDE_RISK_INPUTS_V1",
        "status": status,
        "checked_at_asia_shanghai": datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "candidate_outcomes_computed": False,
        "portfolio_returns_computed": False,
        "evaluation_window": {
            "start": EVALUATION_START.date().isoformat(),
            "end": EVALUATION_END.date().isoformat(),
            "trading_rows": int(len(evaluation)),
        },
        "canonical_price": {
            "source_path": PRIMARY_PRICE_PATH.relative_to(ROOT).as_posix(),
            "output_path": CANONICAL_PRICE_PATH.relative_to(ROOT).as_posix(),
            "source_rows": int(len(canonical)),
            "source_first_date": canonical["date"].min().date().isoformat(),
            "source_last_date": canonical["date"].max().date().isoformat(),
            "corrections": corrections,
        },
        "price_cross_check": price_cross_check,
        "benchmark_cross_check": benchmark_cross_check,
        "calendar": calendar,
        "dividends": dividends,
        "checks": checks,
        "input_hashes": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in required_paths
        },
        "boundaries": {
            "original_source_overwritten": False,
            "price_or_factor_parameters_selected": False,
            "future_risk_labels_read": False,
            "strategy_returns_read": False,
            "live_trading_authorized": False,
        },
    }
    if status != "PASS":
        raise InputAuditError(json.dumps(report, ensure_ascii=False, allow_nan=False))
    _atomic_parquet(CANONICAL_PRICE_PATH, canonical)
    report["canonical_price"]["output_sha256"] = sha256_file(CANONICAL_PRICE_PATH)
    report["canonical_price"]["output_bytes"] = CANONICAL_PRICE_PATH.stat().st_size
    _atomic_json(REPORT_PATH, report)
    return report


def main() -> int:
    report = build()
    print(
        json.dumps(
            {
                "status": report["status"],
                "evaluation_rows": report["evaluation_window"]["trading_rows"],
                "correction_rows": len(report["canonical_price"]["corrections"]),
                "candidate_outcomes_computed": report["candidate_outcomes_computed"],
                "portfolio_returns_computed": report["portfolio_returns_computed"],
                "canonical_sha256": report["canonical_price"]["output_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
