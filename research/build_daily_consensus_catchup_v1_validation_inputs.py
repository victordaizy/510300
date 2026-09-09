"""按冻结定义构造2016-2025三域特征，并核对开发段无漂移。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import all_a_fragility_shape_5d_discovery_v0 as all_a_module
import global_liquidity_regime_5d_discovery_v0 as global_module
import market_leverage_cascade_5d_discovery_v0 as leverage_module


VALIDATION_END = pd.Timestamp("2025-12-31")
DEVELOPMENT_COMPARE_END = pd.Timestamp("2020-12-31")
DRIFT_ABSOLUTE_TOLERANCE = 1e-10

TRAINING_PANEL = PROJECT_ROOT / "data" / "raw" / "a_share_hash_holdout_v2" / "training_panel.parquet"
TIME_HOLDOUT_PANELS = [
    PROJECT_ROOT / "data" / "raw" / "a_share_bucket1_time_holdout_formula_v1" / "holdout_panel.parquet",
    PROJECT_ROOT / "data" / "raw" / "a_share_bucket2_time_holdout_lowvol_v1" / "holdout_panel.parquet",
    PROJECT_ROOT
    / "data"
    / "raw"
    / "a_share_bucket34_time_holdout_stratified_lowvol_v1"
    / "holdout_panel.parquet",
]
ETF_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
DIVIDEND_FILE = PROJECT_ROOT / "data" / "reference" / "510300_dividends.csv"
MARGIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "market_margin_validation_v2"
    / "market_margin_sh_sz_daily_2015_2025.parquet"
)

FROZEN_ALL_A_FILE = (
    PROJECT_ROOT / "data" / "features" / "510300_all_a_fragility_shape_5d_discovery_v0.parquet"
)
FROZEN_GLOBAL_FILE = (
    PROJECT_ROOT / "data" / "features" / "510300_global_liquidity_regime_5d_discovery_v0.parquet"
)
FROZEN_LEVERAGE_FILE = (
    PROJECT_ROOT / "data" / "features" / "510300_market_leverage_cascade_5d_discovery_v0.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "validation" / "510300_daily_consensus_catchup_v1"
OUTPUT_BREADTH = OUTPUT_DIR / "all_a_fragility_shape_daily_2014_2025.parquet"
OUTPUT_ALL_A = OUTPUT_DIR / "all_a_features_2016_2025.parquet"
OUTPUT_GLOBAL = OUTPUT_DIR / "global_features_2015_2025.parquet"
OUTPUT_LEVERAGE = OUTPUT_DIR / "leverage_features_2015_2025.parquet"
OUTPUT_AUDIT = OUTPUT_DIR / "input_build_audit.json"


def _sha256(path: Path) -> str:
    return global_module._sha256(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    global_module._atomic_parquet(frame, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    global_module._atomic_json(payload, path)


def _quoted(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _build_breadth() -> pd.DataFrame:
    training = _quoted(TRAINING_PANEL)
    holdouts = ", ".join(f"'{_quoted(path)}'" for path in TIME_HOLDOUT_PANELS)
    temporary = PROJECT_ROOT / "tmp" / "daily_consensus_catchup_v1_breadth"
    temporary.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    connection.execute(f"SET temp_directory='{_quoted(temporary)}'")
    query = f"""
        WITH source AS (
            SELECT * FROM read_parquet('{training}')
            WHERE CAST(date AS DATE) < DATE '2024-01-01'
            UNION ALL
            SELECT * FROM read_parquet([{holdouts}], union_by_name=true)
            WHERE CAST(date AS DATE) >= DATE '2024-01-01'
              AND CAST(date AS DATE) <= DATE '2025-12-31'
        ), valid AS (
            SELECT
                CAST(date AS DATE) AS date,
                CAST(pct_chg AS DOUBLE) / 100.0 AS ret,
                CAST(amount AS DOUBLE) AS amount,
                CAST(raw_open AS DOUBLE) AS open,
                CAST(raw_high AS DOUBLE) AS high,
                CAST(raw_low AS DOUBLE) AS low,
                CAST(raw_close AS DOUBLE) AS close,
                CAST(pre_close AS DOUBLE) AS pre_close
            FROM source
            WHERE CAST(date AS DATE) <= DATE '2025-12-31'
              AND NOT CAST(is_suspended AS BOOLEAN)
              AND CAST(raw_open AS DOUBLE) > 0
              AND CAST(raw_high AS DOUBLE) > 0
              AND CAST(raw_low AS DOUBLE) > 0
              AND CAST(raw_close AS DOUBLE) > 0
              AND CAST(pre_close AS DOUBLE) > 0
              AND CAST(amount AS DOUBLE) >= 0
        ), enriched AS (
            SELECT
                *,
                open / pre_close - 1.0 AS gap,
                close / open - 1.0 AS intraday_return,
                high / low - 1.0 AS intraday_range,
                CASE WHEN high > low THEN (close - low) / (high - low) ELSE 0.5 END AS close_location,
                ROW_NUMBER() OVER (PARTITION BY date ORDER BY amount DESC) AS amount_rank,
                COUNT(*) OVER (PARTITION BY date) AS daily_count
            FROM valid
        )
        SELECT
            date,
            COUNT(*) AS active_count,
            AVG(ret) AS cs_mean_ret,
            MEDIAN(ret) AS cs_median_ret,
            QUANTILE_CONT(ret, 0.01) AS cs_q01_ret,
            QUANTILE_CONT(ret, 0.05) AS cs_q05_ret,
            QUANTILE_CONT(ret, 0.10) AS cs_q10_ret,
            QUANTILE_CONT(ret, 0.90) AS cs_q90_ret,
            QUANTILE_CONT(ret, 0.95) AS cs_q95_ret,
            QUANTILE_CONT(ret, 0.99) AS cs_q99_ret,
            STDDEV_SAMP(ret) AS cs_dispersion,
            SKEWNESS(ret) AS cs_skewness,
            KURTOSIS(ret) AS cs_kurtosis,
            AVG(CASE WHEN ret < 0 THEN 1.0 ELSE 0.0 END) AS share_down,
            AVG(CASE WHEN ret <= -0.02 THEN 1.0 ELSE 0.0 END) AS share_down2,
            AVG(CASE WHEN ret <= -0.05 THEN 1.0 ELSE 0.0 END) AS share_down5,
            AVG(CASE WHEN ret >= 0.02 THEN 1.0 ELSE 0.0 END) AS share_up2,
            AVG(CASE WHEN ret >= 0.05 THEN 1.0 ELSE 0.0 END) AS share_up5,
            AVG(CASE WHEN ret <= -0.095 THEN 1.0 ELSE 0.0 END) AS share_limit_down,
            AVG(CASE WHEN ret >= 0.095 THEN 1.0 ELSE 0.0 END) AS share_limit_up,
            AVG(CASE WHEN close_location <= 0.10 THEN 1.0 ELSE 0.0 END) AS share_close_near_low,
            AVG(CASE WHEN close_location >= 0.90 THEN 1.0 ELSE 0.0 END) AS share_close_near_high,
            AVG(gap) AS mean_gap,
            AVG(intraday_return) AS mean_intraday_return,
            MEDIAN(intraday_range) AS median_intraday_range,
            SUM(CASE WHEN ret < 0 THEN amount ELSE 0 END) / NULLIF(SUM(amount), 0) AS down_amount_share,
            SUM(CASE WHEN ret > 0 THEN amount ELSE 0 END) / NULLIF(SUM(amount), 0) AS up_amount_share,
            SUM(CASE WHEN amount_rank <= CEIL(daily_count * 0.10) THEN amount ELSE 0 END)
                / NULLIF(SUM(amount), 0) AS top10pct_amount_share,
            LN(SUM(amount) + 1.0) AS total_amount_log
        FROM enriched
        GROUP BY date
        ORDER BY date
    """
    result = connection.execute(query).df()
    connection.close()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    if result["date"].duplicated().any():
        raise ValueError("全A聚合存在重复日期")
    return result.sort_values("date").reset_index(drop=True)


def _build_all_a_features(breadth: pd.DataFrame) -> pd.DataFrame:
    cross_section = all_a_module._add_cross_section_time_features(breadth)
    etf = pd.read_parquet(ETF_FILE)
    etf["date"] = pd.to_datetime(etf["date"], errors="raise").dt.normalize()
    etf = etf[etf["date"] <= VALIDATION_END].sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(etf["close"], errors="raise")
    open_price = pd.to_numeric(etf["open"], errors="raise")
    high = pd.to_numeric(etf["high"], errors="raise")
    low = pd.to_numeric(etf["low"], errors="raise")
    amount = pd.to_numeric(etf["amount"], errors="raise")
    returns = close.pct_change()
    target = pd.DataFrame({"date": etf["date"]})
    target["etf_return_1d"] = returns
    target["etf_return_5d"] = close.pct_change(5)
    target["etf_return_20d"] = close.pct_change(20)
    target["etf_rv_5d"] = returns.rolling(5, min_periods=5).std(ddof=0) * np.sqrt(242)
    target["etf_rv_20d"] = returns.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(242)
    downside_squared = returns.clip(upper=0.0).pow(2).rolling(20, min_periods=20).sum()
    total_squared = returns.pow(2).rolling(20, min_periods=20).sum()
    target["etf_downside_share_20d"] = downside_squared / total_squared.replace(0.0, np.nan)
    rolling_high = close.rolling(60, min_periods=60).max()
    rolling_low = close.rolling(60, min_periods=60).min()
    target["etf_range_position_60d"] = (close - rolling_low) / (rolling_high - rolling_low).replace(
        0.0, np.nan
    )
    target["etf_gap"] = open_price / close.shift(1) - 1.0
    target["etf_intraday_return"] = close / open_price - 1.0
    target["etf_close_location"] = (close - low) / (high - low).replace(0.0, np.nan)
    target["etf_amount_z20_prior"] = all_a_module._prior_z(amount)
    dividends = pd.read_csv(DIVIDEND_FILE, parse_dates=["record_date", "ex_date", "payment_date"])
    dividends = dividends[dividends["ex_date"] <= VALIDATION_END]
    dividend_by_date = dividends.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()
    next_open = open_price.shift(-1)
    following_open = open_price.shift(-2)
    following_date = etf["date"].shift(-2)
    following_dividend = following_date.map(
        lambda value: float(dividend_by_date.get(value, 0.0)) if pd.notna(value) else 0.0
    )
    target["one_day_open_total_return"] = (following_open + following_dividend) / next_open - 1.0
    frame = cross_section.merge(target, on="date", how="inner", validate="one_to_one")
    return frame.sort_values("date").reset_index(drop=True)


def _build_global_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    global_module.DEVELOPMENT_CUTOFF = VALIDATION_END
    dividends = global_module._load_dividends()
    market = global_module._load_market(dividends)
    frame, alignment = global_module._add_external_features(market)
    return frame.sort_values("date").reset_index(drop=True), alignment


def _build_leverage_features(breadth_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    global_module.DEVELOPMENT_CUTOFF = VALIDATION_END
    leverage_module.DEVELOPMENT_CUTOFF = VALIDATION_END
    leverage_module.MARGIN_FILE = MARGIN_FILE
    leverage_module.BREADTH_FILE = breadth_path
    frame, alignment = leverage_module._build_frame()
    return frame.sort_values("date").reset_index(drop=True), alignment


def _compare_frozen(
    built: pd.DataFrame,
    frozen_path: Path,
    columns: list[str],
    label: str,
) -> dict[str, Any]:
    frozen = pd.read_parquet(frozen_path, columns=["date", *columns])
    frozen["date"] = pd.to_datetime(frozen["date"], errors="raise").dt.normalize()
    frozen = frozen[frozen["date"] <= DEVELOPMENT_COMPARE_END].sort_values("date")
    candidate = built[["date", *columns]].copy()
    candidate["date"] = pd.to_datetime(candidate["date"], errors="raise").dt.normalize()
    candidate = candidate[candidate["date"].isin(frozen["date"])].sort_values("date")
    merged = frozen.merge(candidate, on="date", how="outer", suffixes=("_frozen", "_built"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{label}开发日期覆盖漂移：{merged['_merge'].value_counts().to_dict()}")
    maxima: dict[str, float] = {}
    nan_mismatch: dict[str, int] = {}
    for column in columns:
        left = pd.to_numeric(merged[f"{column}_frozen"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_built"], errors="coerce")
        mismatch = int(left.isna().ne(right.isna()).sum())
        nan_mismatch[column] = mismatch
        valid = left.notna() & right.notna()
        maxima[column] = float((left[valid] - right[valid]).abs().max()) if valid.any() else 0.0
    maximum = max(maxima.values(), default=0.0)
    mismatch_total = sum(nan_mismatch.values())
    status = "PASS" if maximum <= DRIFT_ABSOLUTE_TOLERANCE and mismatch_total == 0 else "FAIL"
    payload = {
        "label": label,
        "status": status,
        "matched_dates": int(len(merged)),
        "first_date": merged["date"].min().date().isoformat(),
        "last_date": merged["date"].max().date().isoformat(),
        "maximum_absolute_difference": maximum,
        "tolerance": DRIFT_ABSOLUTE_TOLERANCE,
        "nan_pattern_mismatch_count": mismatch_total,
        "maximum_by_column": maxima,
    }
    if status != "PASS":
        worst = sorted(maxima.items(), key=lambda item: item[1], reverse=True)[:10]
        raise ValueError(f"{label}特征漂移：maximum={maximum}, nan={mismatch_total}, worst={worst}")
    return payload


def _validation_coverage(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    selected = dates[(dates >= pd.Timestamp("2021-01-01")) & (dates <= VALIDATION_END)]
    return {
        "label": label,
        "rows": int(len(frame)),
        "validation_rows": int(selected.nunique()),
        "validation_first_date": selected.min().date().isoformat(),
        "validation_last_date": selected.max().date().isoformat(),
    }


def main() -> int:
    required = [
        TRAINING_PANEL,
        *TIME_HOLDOUT_PANELS,
        ETF_FILE,
        DIVIDEND_FILE,
        MARGIN_FILE,
        FROZEN_ALL_A_FILE,
        FROZEN_GLOBAL_FILE,
        FROZEN_LEVERAGE_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_AUDIT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("构造冻结TRAIN股票组的全A横截面特征", flush=True)
    breadth = _build_breadth()
    _atomic_parquet(breadth, OUTPUT_BREADTH)
    all_a = _build_all_a_features(breadth)
    print("构造全球流动性与跨市场特征", flush=True)
    global_features, global_alignment = _build_global_features()
    print("构造严格滞后一交易日的全市场融资杠杆特征", flush=True)
    leverage_features, leverage_alignment = _build_leverage_features(OUTPUT_BREADTH)

    drift = {
        "all_a": _compare_frozen(
            all_a,
            FROZEN_ALL_A_FILE,
            list(all_a_module.MODEL_FEATURES),
            "ALL_A_MODEL_FEATURES",
        ),
        "global": _compare_frozen(
            global_features,
            FROZEN_GLOBAL_FILE,
            list(global_module.FEATURE_COLUMNS),
            "GLOBAL_MODEL_FEATURES",
        ),
        "leverage": _compare_frozen(
            leverage_features,
            FROZEN_LEVERAGE_FILE,
            list(leverage_module.FEATURE_COLUMNS),
            "LEVERAGE_MODEL_FEATURES",
        ),
    }
    all_a_output = all_a[["date", "one_day_open_total_return", *all_a_module.MODEL_FEATURES]].copy()
    global_output = global_features[["date", *global_module.FEATURE_COLUMNS]].copy()
    leverage_output = leverage_features[
        ["date", "margin_observation_date", "margin_observation_age_calendar_days", *leverage_module.FEATURE_COLUMNS]
    ].copy()
    _atomic_parquet(all_a_output, OUTPUT_ALL_A)
    _atomic_parquet(global_output, OUTPUT_GLOBAL)
    _atomic_parquet(leverage_output, OUTPUT_LEVERAGE)

    payload = {
        "status": "PASS_VALIDATION_INPUTS_BUILT_DEVELOPMENT_FEATURES_REPRODUCED",
        "protocol_id": "510300_DAILY_CONSENSUS_CATCHUP_V1",
        "validation_period": ["2021-01-01", "2025-12-31"],
        "development_drift_checks": drift,
        "coverage": {
            "all_a": _validation_coverage(all_a_output, "ALL_A"),
            "global": _validation_coverage(global_output, "GLOBAL"),
            "leverage": _validation_coverage(leverage_output, "LEVERAGE"),
        },
        "alignment": {
            "global": global_alignment,
            "leverage": leverage_alignment,
        },
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in required
        },
        "artifacts": {
            "breadth": {
                "file": str(OUTPUT_BREADTH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "rows": int(len(breadth)),
                "sha256": _sha256(OUTPUT_BREADTH),
            },
            "all_a": {
                "file": str(OUTPUT_ALL_A.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "rows": int(len(all_a_output)),
                "sha256": _sha256(OUTPUT_ALL_A),
            },
            "global": {
                "file": str(OUTPUT_GLOBAL.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "rows": int(len(global_output)),
                "sha256": _sha256(OUTPUT_GLOBAL),
            },
            "leverage": {
                "file": str(OUTPUT_LEVERAGE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "rows": int(len(leverage_output)),
                "sha256": _sha256(OUTPUT_LEVERAGE),
            },
        },
        "contains_performance_or_validation_signal_results": False,
    }
    _atomic_json(payload, OUTPUT_AUDIT)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
