"""用批量历史加深交所官方缺口补齐构造2015-2025两融验证输入。"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import collect_market_margin_leverage_v0 as base


VALIDATION_START = pd.Timestamp("2021-01-01")
VALIDATION_END = pd.Timestamp("2025-12-31")
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "market_margin_validation_v2"
DEVELOPMENT_COMBINED_FILE = (
    PROJECT_ROOT / "data" / "raw" / "market_margin_leverage_v0" / "market_margin_sh_sz_daily.parquet"
)
GAP_CACHE_FILE = OUTPUT_DIR / "szse_official_gap_fill.parquet"
COMBINED_FILE = OUTPUT_DIR / "market_margin_sh_sz_daily_2015_2025.parquet"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

NUMERIC_COLUMNS = ["rzmre", "rzye", "rqmcl", "rqyl", "rqye", "rzrqye"]


def _calendar() -> pd.DatetimeIndex:
    frame = pd.read_parquet(
        base.ETF_CALENDAR_FILE,
        columns=["date"],
        filters=[
            ("date", ">=", VALIDATION_START.to_pydatetime()),
            ("date", "<=", VALIDATION_END.to_pydatetime()),
        ],
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    return pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())


def _load_gap_cache() -> pd.DataFrame:
    if not GAP_CACHE_FILE.exists():
        return pd.DataFrame(columns=["date", "exchange", *NUMERIC_COLUMNS, "source"])
    frame = pd.read_parquet(GAP_CACHE_FILE)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    return frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _save_gap_cache(frame: pd.DataFrame) -> None:
    ordered = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    base._atomic_parquet(ordered, GAP_CACHE_FILE)


def _fetch_missing_szse(missing: pd.DatetimeIndex) -> pd.DataFrame:
    cache = _load_gap_cache()
    cached_dates = pd.DatetimeIndex(cache["date"]) if not cache.empty else pd.DatetimeIndex([])
    remaining = missing.difference(cached_dates)
    collected = [cache] if not cache.empty else []
    for index, date in enumerate(remaining, start=1):
        row = base._fetch_szse_date(pd.Timestamp(date), maximum_attempts=8)
        collected.append(row)
        cache = pd.concat(collected, ignore_index=True)
        collected = [cache]
        if index % 10 == 0 or index == len(remaining):
            _save_gap_cache(cache)
            print(f"深交所官方缺口补齐 {index}/{len(remaining)}", flush=True)
        time.sleep(0.15)
    return _load_gap_cache()


def _coverage(expected: pd.DatetimeIndex, actual: pd.DatetimeIndex) -> dict[str, Any]:
    by_year: dict[str, Any] = {}
    all_missing: list[str] = []
    for year in range(VALIDATION_START.year, VALIDATION_END.year + 1):
        expected_year = expected[expected.year == year]
        observed = expected_year.intersection(actual)
        missing = expected_year.difference(actual)
        all_missing.extend(date.date().isoformat() for date in missing)
        by_year[str(year)] = {
            "expected_trading_days": int(len(expected_year)),
            "observed_common_days": int(len(observed)),
            "coverage": float(len(observed) / len(expected_year)) if len(expected_year) else None,
            "missing_dates": [date.date().isoformat() for date in missing],
        }
    return {"by_year": by_year, "missing_dates": all_missing}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.START_DATE = pd.Timestamp("2015-01-01")
    base.DEVELOPMENT_CUTOFF = VALIDATION_END
    expected = _calendar()
    print("下载批量两融历史与上交所官方全量历史", flush=True)
    sse_secondary = base._normalize_jin10(ak.macro_china_market_margin_sh(), "SSE")
    szse_secondary = base._normalize_jin10(ak.macro_china_market_margin_sz(), "SZSE")
    sse_official = base._normalize_sse_official(
        ak.stock_margin_sse(VALIDATION_START.strftime("%Y%m%d"), VALIDATION_END.strftime("%Y%m%d"))
    )
    sse_official = sse_official[sse_official["date"].between(VALIDATION_START, VALIDATION_END)].copy()
    sse_secondary_validation = sse_secondary[
        sse_secondary["date"].between(VALIDATION_START, VALIDATION_END)
    ].copy()
    szse_secondary_validation = szse_secondary[
        szse_secondary["date"].between(VALIDATION_START, VALIDATION_END)
    ].copy()

    sse_comparison, sse_summary = base._comparison(
        sse_official,
        sse_secondary_validation,
        ["rzye", "rzmre", "rqyl", "rqmcl", "rzrqye"],
        "SSE_OFFICIAL_VS_JIN10_VALIDATION_HISTORY",
        1e-6,
    )
    if sse_summary["status"] != "PASS":
        raise ValueError(f"上交所批量历史交叉核对失败：{sse_summary}")

    secondary_sz_dates = pd.DatetimeIndex(szse_secondary_validation["date"].drop_duplicates().sort_values())
    missing_sz = expected.difference(secondary_sz_dates)
    print(f"深交所批量源缺少{len(missing_sz)}个验证交易日，开始官方补齐", flush=True)
    gap_fill = _fetch_missing_szse(missing_sz)
    gap_fill = gap_fill[gap_fill["date"].isin(missing_sz)].copy()
    szse_complete = pd.concat(
        [szse_secondary_validation, gap_fill],
        ignore_index=True,
        sort=False,
    )
    szse_complete = szse_complete.sort_values("date").drop_duplicates("date", keep="last")

    year_end_dates = [
        expected[expected.year == year].max().strftime("%Y%m%d")
        for year in range(VALIDATION_START.year, VALIDATION_END.year + 1)
    ]
    official_samples = pd.concat(
        [base._fetch_szse_date(pd.Timestamp(date), maximum_attempts=8) for date in year_end_dates],
        ignore_index=True,
    )
    szse_comparison, szse_summary = base._comparison(
        official_samples,
        szse_secondary_validation,
        ["rzye", "rzmre", "rzrqye"],
        "SZSE_OFFICIAL_VS_JIN10_VALIDATION_YEAR_END_SAMPLES",
        2e-4,
    )
    if szse_summary["status"] != "PASS":
        raise ValueError(f"深交所批量历史交叉核对失败：{szse_summary}")

    validation_combined = base._build_combined(sse_official, szse_complete)
    validation_combined = validation_combined[
        validation_combined["date"].between(VALIDATION_START, VALIDATION_END)
    ].copy()
    validation_dates = pd.DatetimeIndex(validation_combined["date"].drop_duplicates().sort_values())
    validation_coverage = _coverage(expected, validation_dates)
    if validation_coverage["missing_dates"]:
        raise ValueError(
            f"官方补齐后仍缺少{len(validation_coverage['missing_dates'])}个验证交易日："
            f"{validation_coverage['missing_dates'][:20]}"
        )

    development = pd.read_parquet(DEVELOPMENT_COMBINED_FILE)
    development["date"] = pd.to_datetime(development["date"], errors="raise").dt.normalize()
    development = development[development["date"] < VALIDATION_START].copy()
    common_columns = sorted(set(development.columns).intersection(validation_combined.columns))
    full = pd.concat(
        [development[common_columns], validation_combined[common_columns]],
        ignore_index=True,
        sort=False,
    )
    full = full.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    full["market_rzye_change"] = pd.to_numeric(full["market_rzye"], errors="raise").diff()
    full["market_financing_balance_identity_residual"] = (
        full["market_rzye_change"] - pd.to_numeric(full["market_rzmre"], errors="raise")
    )

    artifacts = {
        "sse_secondary": (sse_secondary_validation, OUTPUT_DIR / "sse_margin_history_jin10.parquet"),
        "szse_secondary": (szse_secondary_validation, OUTPUT_DIR / "szse_margin_history_jin10.parquet"),
        "sse_official": (sse_official, OUTPUT_DIR / "sse_margin_history_official.parquet"),
        "szse_official_samples": (official_samples, OUTPUT_DIR / "szse_margin_samples_official.parquet"),
        "sse_comparison": (sse_comparison, OUTPUT_DIR / "sse_crosscheck.parquet"),
        "szse_comparison": (szse_comparison, OUTPUT_DIR / "szse_crosscheck.parquet"),
        "validation_combined": (
            validation_combined,
            OUTPUT_DIR / "market_margin_sh_sz_daily_2021_2025.parquet",
        ),
        "full_combined": (full, COMBINED_FILE),
    }
    for frame, path in artifacts.values():
        base._atomic_parquet(frame, path)

    payload = {
        "status": "PASS_COMPLETE_VALIDATION_COVERAGE",
        "dataset_id": "MARKET_MARGIN_VALIDATION_V2",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_period": [VALIDATION_START.date().isoformat(), VALIDATION_END.date().isoformat()],
        "sources": {
            "sse_primary": base.SSE_PAGE,
            "szse_secondary_batch": base.JIN10_SZ_PAGE,
            "szse_primary_gap_fill": base.SZSE_PAGE,
        },
        "crosschecks": {"sse": sse_summary, "szse": szse_summary},
        "szse_gap_fill": {
            "missing_in_batch_count": int(len(missing_sz)),
            "official_gap_fill_count": int(len(gap_fill)),
            "cache_file": str(GAP_CACHE_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "coverage": validation_coverage,
        "artifacts": {
            key: {
                "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "rows": int(len(frame)),
                "sha256": base._sha256(path),
            }
            for key, (frame, path) in artifacts.items()
        },
    }
    base._atomic_json(payload, METADATA_FILE)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
