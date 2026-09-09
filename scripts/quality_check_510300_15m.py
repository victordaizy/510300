"""审计510300未复权15分钟行情及其与日线的聚合一致性。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "minute_data.yaml"
MINUTE_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_raw.parquet"
METADATA_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_raw.metadata.json"
DAILY_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_15m_quality.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_minute_data(data: pd.DataFrame, config: dict) -> tuple[list[dict], list[dict], dict]:
    errors: list[dict] = []
    warnings: list[dict] = []
    evidence: dict = {}
    required = [
        "bar_start", "bar_end", "trade_date", "open", "high", "low", "close",
        "volume", "amount", "symbol", "frequency_minutes", "timestamp_meaning",
        "adjustment", "source", "timezone", "volume_unit", "amount_unit", "retrieved_at",
        "bar_position", "session_phase", "contains_opening_auction",
        "contains_closing_auction", "post_close_included",
        "post_close_separately_identifiable",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        return [{"code": "MISSING_COLUMNS", "columns": missing}], warnings, evidence
    if data.empty:
        return [{"code": "EMPTY_DATA"}], warnings, evidence

    bars = data.copy()
    bars["bar_end"] = pd.to_datetime(bars["bar_end"], errors="coerce")
    bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="coerce").dt.normalize()
    if bars[["bar_end", "trade_date"]].isna().any().any():
        errors.append({"code": "INVALID_TIMESTAMP"})
        return errors, warnings, evidence
    if bars["bar_end"].duplicated().any():
        errors.append({"code": "DUPLICATE_BAR_END", "count": int(bars["bar_end"].duplicated().sum())})
    if not bars["bar_end"].is_monotonic_increasing:
        errors.append({"code": "BAR_END_NOT_SORTED"})
    weekend_count = int((bars["trade_date"].dt.weekday >= 5).sum())
    if weekend_count:
        errors.append({"code": "WEEKEND_BARS", "count": weekend_count})

    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    numeric = bars[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        errors.append({"code": "INVALID_NUMERIC"})
        return errors, warnings, evidence
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        errors.append({"code": "NON_POSITIVE_PRICE"})
    if (numeric[["volume", "amount"]] <= 0).any().any():
        errors.append({"code": "NON_POSITIVE_FLOW"})
    high_bad = numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)
    low_bad = numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)
    if (high_bad | low_bad).any():
        errors.append({"code": "OHLC_INCONSISTENT", "count": int((high_bad | low_bad).sum())})

    tick = float(config["quality"]["price_tick"])
    tick_scaled = numeric[["open", "high", "low", "close"]] / tick
    invalid_tick = (~np.isclose(tick_scaled, np.round(tick_scaled), atol=1e-7)).any(axis=1)
    if invalid_tick.any():
        errors.append({"code": "INVALID_PRICE_TICK", "count": int(invalid_tick.sum())})
    implied_vwap = numeric["amount"] / numeric["volume"]
    vwap_bad = (implied_vwap < numeric["low"] - tick) | (implied_vwap > numeric["high"] + tick)
    if vwap_bad.any():
        errors.append({"code": "IMPLIED_VWAP_OUTSIDE_RANGE", "count": int(vwap_bad.sum())})

    expected_times = set(config["session"]["expected_bar_end_times"])
    actual_times = set(bars["bar_end"].dt.strftime("%H:%M:%S"))
    unexpected_times = sorted(actual_times.difference(expected_times))
    if unexpected_times:
        errors.append({"code": "UNEXPECTED_BAR_END_TIME", "times": unexpected_times})
    counts = bars.groupby("trade_date").size()
    expected_count = int(config["session"]["expected_bars_per_full_day"])
    partial = counts[counts != expected_count]
    edge_dates = {counts.index.min(), counts.index.max()}
    internal_partial = partial.loc[~partial.index.isin(edge_dates)]
    if not internal_partial.empty:
        errors.append(
            {
                "code": "INTERNAL_PARTIAL_TRADING_DAY",
                "days": {pd.Timestamp(k).date().isoformat(): int(v) for k, v in internal_partial.items()},
            }
        )
    for date, count in partial.items():
        position = "internal"
        if date == counts.index.min():
            position = "leading"
        elif date == counts.index.max():
            position = "trailing"
        warnings.append(
            {
                "code": f"{position.upper()}_PARTIAL_TRADING_DAY",
                "date": pd.Timestamp(date).date().isoformat(),
                "bars": int(count),
                "expected": expected_count,
            }
        )
    evidence["bar_count_by_day_distribution"] = {
        str(int(count)): int(frequency) for count, frequency in counts.value_counts().sort_index().items()
    }
    evidence["full_trading_days"] = int((counts == expected_count).sum())
    evidence["session_phase_distribution"] = {
        str(key): int(value) for key, value in bars["session_phase"].value_counts().items()
    }
    return errors, warnings, evidence


def _gap_statistics(frame: pd.DataFrame, prefix: str) -> dict:
    if frame.empty:
        return {"days": 0}
    volume_gap = frame["volume"] - frame["volume_15m"]
    amount_gap = frame["amount"] - frame["amount_15m"]
    volume_ratio = volume_gap / frame["volume"]
    amount_ratio = amount_gap / frame["amount"]
    return {
        "period": prefix,
        "days": int(len(frame)),
        "volume_gap_shares": {
            "mean": float(volume_gap.mean()),
            "median": float(volume_gap.median()),
            "total": float(volume_gap.sum()),
            "maximum_positive": float(volume_gap.max()),
            "maximum_absolute": float(volume_gap.abs().max()),
            "positive_days": int((volume_gap > 0.5).sum()),
            "zero_days": int(np.isclose(volume_gap, 0.0, atol=0.5).sum()),
            "negative_days": int((volume_gap < -0.5).sum()),
            "maximum_positive_ratio": float(volume_ratio.max()),
            "maximum_absolute_ratio": float(volume_ratio.abs().max()),
        },
        "amount_gap_cny": {
            "mean": float(amount_gap.mean()),
            "median": float(amount_gap.median()),
            "total": float(amount_gap.sum()),
            "maximum_positive": float(amount_gap.max()),
            "maximum_absolute": float(amount_gap.abs().max()),
            "positive_days": int((amount_gap > 0.01).sum()),
            "zero_days": int(np.isclose(amount_gap, 0.0, atol=0.01).sum()),
            "negative_days": int((amount_gap < -0.01).sum()),
            "maximum_positive_ratio": float(amount_ratio.max()),
            "maximum_absolute_ratio": float(amount_ratio.abs().max()),
        },
    }


def cross_check_daily(
    minute: pd.DataFrame,
    daily: pd.DataFrame,
    config: dict,
) -> tuple[list[dict], list[dict], dict]:
    errors: list[dict] = []
    warnings: list[dict] = []
    bars = minute.copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    expected_count = int(config["session"]["expected_bars_per_full_day"])
    counts = bars.groupby("trade_date").size()
    full_dates = counts[counts == expected_count].index
    bars = bars.loc[bars["trade_date"].isin(full_dates)]
    aggregate = bars.groupby("trade_date", as_index=False).agg(
        open_15m=("open", "first"),
        high_15m=("high", "max"),
        low_15m=("low", "min"),
        close_15m=("close", "last"),
        volume_15m=("volume", "sum"),
        amount_15m=("amount", "sum"),
    )
    reference = daily.copy()
    reference["trade_date"] = pd.to_datetime(reference["date"]).dt.normalize()
    merged = aggregate.merge(
        reference[["trade_date", "open", "high", "low", "close", "volume", "amount"]],
        on="trade_date",
        how="inner",
        validate="one_to_one",
    )
    unmatched_dates = aggregate.loc[~aggregate["trade_date"].isin(merged["trade_date"]), "trade_date"]
    if not unmatched_dates.empty:
        daily_max_date = reference["trade_date"].max()
        newer = unmatched_dates[unmatched_dates > daily_max_date]
        unexpected = unmatched_dates[unmatched_dates <= daily_max_date]
        if not unexpected.empty:
            errors.append(
                {
                    "code": "DAILY_REFERENCE_MISSING",
                    "dates": unexpected.dt.strftime("%Y-%m-%d").tolist(),
                }
            )
        if not newer.empty:
            warnings.append(
                {
                    "code": "MINUTE_DATA_NEWER_THAN_DAILY_REFERENCE",
                    "dates": newer.dt.strftime("%Y-%m-%d").tolist(),
                }
            )
    tick_tolerance = float(config["quality"]["price_tick"]) * float(
        config["quality"]["price_tolerance_ticks"]
    )
    epsilon = 1e-9
    price_differences: dict[str, float] = {}
    mismatch_days: set[str] = set()
    for column in ["open", "high", "low"]:
        difference = (merged[f"{column}_15m"] - merged[column]).abs()
        price_differences[column] = float(difference.max()) if not difference.empty else 0.0
        mismatch_days.update(
            merged.loc[difference > tick_tolerance + epsilon, "trade_date"].dt.strftime("%Y-%m-%d")
        )
    if mismatch_days:
        errors.append({"code": "DAILY_PRICE_CROSS_CHECK_FAILED", "days": sorted(mismatch_days)})

    close_difference = (merged["close_15m"] - merged["close"]).abs()
    price_differences["close"] = float(close_difference.max()) if not close_difference.empty else 0.0
    regime_start = pd.Timestamp(config["market_regime"]["close_auction_and_after_hours_start"])
    legacy_mismatch = (merged["trade_date"] < regime_start) & (close_difference > tick_tolerance + epsilon)
    current_mismatch = (merged["trade_date"] >= regime_start) & (close_difference > tick_tolerance + epsilon)
    if legacy_mismatch.any():
        warnings.append(
            {
                "code": "LEGACY_CLOSE_DEFINITION_DIFFERENCE",
                "days": merged.loc[legacy_mismatch, "trade_date"].dt.strftime("%Y-%m-%d").tolist(),
                "maximum_absolute_error": float(close_difference.loc[legacy_mismatch].max()),
                "explanation": "旧制日线close为最后一分钟VWAP，15分钟close为区间末笔价",
            }
        )
    if current_mismatch.any():
        errors.append(
            {
                "code": "CURRENT_REGIME_CLOSE_CROSS_CHECK_FAILED",
                "days": merged.loc[current_mismatch, "trade_date"].dt.strftime("%Y-%m-%d").tolist(),
            }
        )

    volume_relative = ((merged["volume_15m"] - merged["volume"]).abs() / merged["volume"]).replace(
        [np.inf, -np.inf], np.nan
    )
    amount_relative = ((merged["amount_15m"] - merged["amount"]).abs() / merged["amount"]).replace(
        [np.inf, -np.inf], np.nan
    )
    volume_max = float(volume_relative.max()) if not volume_relative.empty else 0.0
    amount_max = float(amount_relative.max()) if not amount_relative.empty else 0.0
    if volume_max > float(config["quality"]["volume_relative_tolerance"]):
        warnings.append({"code": "DAILY_VOLUME_CROSS_CHECK_WARNING", "maximum_relative_error": volume_max})
    if amount_max > float(config["quality"]["amount_relative_tolerance"]):
        warnings.append({"code": "DAILY_AMOUNT_CROSS_CHECK_WARNING", "maximum_relative_error": amount_max})
    pre_regime = merged.loc[merged["trade_date"] < regime_start]
    post_regime = merged.loc[merged["trade_date"] >= regime_start]
    gap_by_regime = {
        "before_2026_07_06": _gap_statistics(pre_regime, "before_2026_07_06"),
        "from_2026_07_06": _gap_statistics(post_regime, "from_2026_07_06"),
    }
    post_volume_gap = post_regime["volume"] - post_regime["volume_15m"]
    post_positive = post_regime.loc[post_volume_gap > 0.5].copy()
    if not post_regime.empty:
        warnings.append(
            {
                "code": "POST_CLOSE_DATA_MISSING",
                "period_start": regime_start.date().isoformat(),
                "explanation": "主15分钟K线止于15:00，不含15:05—15:30盘后固定价格交易",
            }
        )
    if not post_positive.empty:
        positive_gap = post_positive["volume"] - post_positive["volume_15m"]
        warnings.append(
            {
                "code": "POST_CLOSE_VOLUME_PROXY_POSITIVE",
                "days": post_positive["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
                "total_positive_gap_shares": float(positive_gap.sum()),
                "maximum_positive_gap_ratio": float((positive_gap / post_positive["volume"]).max()),
                "interpretation": (
                    "日线总量减15分钟总量与缺失盘后量方向一致，但旧制度也存在供应商差异，"
                    "因此这里只是盘后成交代理上界，不等同于逐日真实盘后量"
                ),
            }
        )
    if (
        not np.isclose(merged["volume_15m"], merged["volume"], atol=0.5).all()
        or not np.isclose(merged["amount_15m"], merged["amount"], atol=0.01).all()
    ):
        warnings.append(
            {
                "code": "VOLUME_AMOUNT_RECONCILIATION_NOT_EXACT",
                "explanation": "日线与15分钟聚合的成交量或成交额并非逐日完全相等",
            }
        )
    evidence = {
        "matched_full_days": int(len(merged)),
        "maximum_absolute_price_error": price_differences,
        "maximum_relative_volume_error": volume_max,
        "maximum_relative_amount_error": amount_max,
        "legacy_close_mismatch_days": int(legacy_mismatch.sum()),
        "current_regime_close_mismatch_days": int(current_mismatch.sum()),
        "daily_ohlc_boundary_reconciliation": (
            "PASS" if not mismatch_days and not current_mismatch.any() else "FAIL"
        ),
        "volume_amount_reconciliation": (
            "PASS"
            if np.isclose(merged["volume_15m"], merged["volume"], atol=0.5).all()
            and np.isclose(merged["amount_15m"], merged["amount"], atol=0.01).all()
            else "WARN"
        ),
        "daily_minus_15m_gap_by_market_regime": gap_by_regime,
    }
    return errors, warnings, evidence


def main() -> int:
    if not MINUTE_FILE.exists():
        print(f"找不到15分钟数据：{MINUTE_FILE}", file=sys.stderr)
        return 1
    if not DAILY_FILE.exists():
        print(f"找不到日线参考数据：{DAILY_FILE}", file=sys.stderr)
        return 1
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    minute = pd.read_parquet(MINUTE_FILE)
    daily = pd.read_parquet(DAILY_FILE)
    errors, warnings, evidence = check_minute_data(minute, config)
    cross_errors, cross_warnings, cross_evidence = cross_check_daily(minute, daily, config)
    errors.extend(cross_errors)
    warnings.extend(cross_warnings)
    evidence["daily_cross_check"] = cross_evidence
    data_hash = sha256_file(MINUTE_FILE)
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8")) if METADATA_FILE.exists() else {}
    if metadata.get("sha256") != data_hash:
        errors.append({"code": "METADATA_HASH_MISMATCH"})
    warnings.append(
        {
            "code": "LIMITED_HISTORY_NOT_FIVE_YEARS",
            "actual_first_bar_end": pd.Timestamp(minute["bar_end"].min()).isoformat(),
            "actual_last_bar_end": pd.Timestamp(minute["bar_end"].max()).isoformat(),
        }
    )
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    report = {
        "status": status,
        "scope": "510300未复权15分钟行情接口",
        "checked_at": datetime.now(ZoneInfo(config["timezone"])).isoformat(),
        "file": MINUTE_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": data_hash,
        "row_count": int(len(minute)),
        "actual_first_bar_end": pd.Timestamp(minute["bar_end"].min()).isoformat(),
        "actual_last_bar_end": pd.Timestamp(minute["bar_end"].max()).isoformat(),
        "errors": errors,
        "warnings": warnings,
        "evidence": evidence,
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
