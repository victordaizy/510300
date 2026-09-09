"""510300 外部流动性与人民币压力的五日二元状态发现研究。

本脚本只允许在 2020-12-31 及以前形成、选择候选；2021 年及以后数据不参与
任何拟合、阈值选择或绩效计算。信号在 T 日收盘后形成，T+1 日开盘执行，
每个五交易日块只能处于 510300 满仓或现金空仓两种状态。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binary_state_feasibility_v1 import (
    CostModel,
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_CUTOFF = pd.Timestamp("2020-12-31")
FEATURE_HISTORY_START = pd.Timestamp("2015-01-01")
FIT_START = pd.Timestamp("2016-08-15")
FIT_END = pd.Timestamp("2018-12-31")
SELECTION_START = pd.Timestamp("2019-01-01")
HORIZON = 5
TRADING_DAYS_PER_YEAR = 242
CASH_ANNUAL_RATE = 0.015
INITIAL_CAPITAL_CNY = 500_000.0
RANDOM_STATE = 20260828

ETF_FILE = PROJECT_ROOT / "data" / "raw" / "r6" / "510300_daily.parquet"
BENCHMARK_FILE = PROJECT_ROOT / "data" / "raw" / "r6" / "H00300_total_return_daily.parquet"
DIVIDEND_FILE = PROJECT_ROOT / "data" / "reference" / "510300_dividends.csv"
GLOBAL_DIR = PROJECT_ROOT / "data" / "raw" / "cross_market_chart_ml_v1"
LIQUIDITY_DIR = PROJECT_ROOT / "data" / "raw" / "global_liquidity_shock_v0"
VIX_FILE = LIQUIDITY_DIR / "vix_daily.parquet"
CFETS_FILE = LIQUIDITY_DIR / "cfets_rmb_central_parity_daily.parquet"
CREDIT_FILE = PROJECT_ROOT / "data" / "raw" / "macro" / "china_credit_spread_3y_daily.parquet"

OUTPUT_FEATURES = PROJECT_ROOT / "data" / "features" / "510300_global_liquidity_regime_5d_discovery_v0.parquet"
OUTPUT_OFFSETS = PROJECT_ROOT / "data" / "research" / "510300_global_liquidity_regime_5d_discovery_v0" / "offset_metrics.parquet"
OUTPUT_BLOCKS = PROJECT_ROOT / "data" / "research" / "510300_global_liquidity_regime_5d_discovery_v0" / "block_predictions.parquet"
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "discovery" / "510300_global_liquidity_regime_5d_discovery_v0.json"

GLOBAL_MARKETS = {
    "GSPC": ("spx", False),
    "IXIC": ("nasdaq", False),
    "DJI": ("dow", False),
    "GDAXI": ("dax", False),
    "FTSE": ("ftse", False),
    "FCHI": ("cac", False),
    "HSI": ("hsi", False),
    "N225": ("nikkei", True),
    "KS11": ("kospi", True),
    "AXJO": ("asx", True),
}

FEATURE_COLUMNS = [
    "etf_ret1",
    "etf_ret5",
    "etf_ret20",
    "etf_vol20",
    "etf_drawdown60",
    "etf_intraday_ret",
    "etf_close_location",
    "etf_amount_z20",
    "vix_log_level",
    "vix_ret1",
    "vix_ret5",
    "vix_z63",
    "vix_z252",
    "us_mean_ret1",
    "us_mean_ret5",
    "eu_mean_ret1",
    "eu_mean_ret5",
    "hsi_ret1",
    "hsi_ret5",
    "asia_mean_ret1",
    "asia_mean_ret5",
    "global_min_ret1",
    "usd_cny_ret1",
    "usd_cny_ret5",
    "usd_cny_ret20",
    "fx_basket_weak1",
    "fx_basket_weak5",
    "fx_dispersion5",
    "credit_spread_level",
    "credit_spread_d1",
    "credit_spread_d5",
    "credit_spread_z252",
]

MECHANISM_COMPONENTS = {
    "vix_z63": 1.0,
    "vix_ret5": 1.0,
    "us_mean_ret5": -1.0,
    "asia_mean_ret5": -1.0,
    "fx_basket_weak5": 1.0,
    "credit_spread_d5": 1.0,
    "etf_ret20": -1.0,
    "etf_drawdown60": -1.0,
}

OBJECTIVE = {
    "annualization_trading_days": TRADING_DAYS_PER_YEAR,
    "rolling_window_trading_days": 242,
    "minimum_annualized_excess": 0.20,
    "minimum_rolling_excess_median": 0.20,
}

BASE_COSTS = CostModel(
    commission_rate=0.0001,
    minimum_commission=0.0,
    slippage_bps=5.0,
    cash_annual_rate=CASH_ANNUAL_RATE,
    trading_days_per_year=TRADING_DAYS_PER_YEAR,
    lot_size=100,
)
STRESS_COSTS = CostModel(
    commission_rate=0.0001,
    minimum_commission=0.0,
    slippage_bps=15.0,
    cash_annual_rate=CASH_ANNUAL_RATE,
    trading_days_per_year=TRADING_DAYS_PER_YEAR,
    lot_size=100,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _read_parquet_until(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[("date", ">=", FEATURE_HISTORY_START.to_pydatetime()), ("date", "<=", DEVELOPMENT_CUTOFF.to_pydatetime())],
    )
    frame["date"] = (
        pd.to_datetime(frame["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    )
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"开发区间输入为空：{path}")
    if frame["date"].max() > DEVELOPMENT_CUTOFF:
        raise ValueError(f"读取越过开发上限：{path}")
    return frame


def _load_dividends() -> pd.DataFrame:
    frame = pd.read_csv(DIVIDEND_FILE)
    for column in ["record_date", "ex_date", "payment_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
    frame["cash_dividend_per_share"] = pd.to_numeric(frame["cash_dividend_per_share"], errors="raise")
    frame = frame[frame["ex_date"] <= DEVELOPMENT_CUTOFF].copy()
    return frame.sort_values("ex_date").reset_index(drop=True)


def _dividend_per_share(dividends: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    eligible = dividends[(dividends["ex_date"] > start_date) & (dividends["ex_date"] <= end_date)]
    return float(eligible["cash_dividend_per_share"].sum())


def _load_market(dividends: pd.DataFrame) -> pd.DataFrame:
    etf = _read_parquet_until(
        ETF_FILE,
        columns=["date", "open", "high", "low", "close", "amount"],
    ).rename(
        columns={
            "open": "etf_open",
            "high": "etf_high",
            "low": "etf_low",
            "close": "etf_close",
            "amount": "etf_amount",
        }
    )
    benchmark = _read_parquet_until(BENCHMARK_FILE, columns=["date", "close"]).rename(
        columns={"close": "benchmark_close"}
    )
    market = etf.merge(benchmark, on="date", how="inner", validate="one_to_one").sort_values("date").reset_index(drop=True)
    if market["date"].duplicated().any():
        raise ValueError("510300 与 H00300 合并后存在重复日期")

    close = pd.to_numeric(market["etf_close"], errors="raise")
    amount = pd.to_numeric(market["etf_amount"], errors="raise")
    log_return = np.log(close / close.shift(1))
    market["etf_ret1"] = close.pct_change(1)
    market["etf_ret5"] = close.pct_change(5)
    market["etf_ret20"] = close.pct_change(20)
    market["etf_vol20"] = log_return.rolling(20, min_periods=15).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    market["etf_drawdown60"] = close / close.rolling(60, min_periods=40).max() - 1.0
    market["etf_intraday_ret"] = close / market["etf_open"] - 1.0
    intraday_range = market["etf_high"] - market["etf_low"]
    market["etf_close_location"] = np.where(
        intraday_range > 0,
        (close - market["etf_low"]) / intraday_range,
        0.5,
    )
    log_amount = np.log(amount.where(amount > 0))
    amount_mean = log_amount.rolling(20, min_periods=15).mean()
    amount_std = log_amount.rolling(20, min_periods=15).std(ddof=0).replace(0.0, np.nan)
    market["etf_amount_z20"] = (log_amount - amount_mean) / amount_std

    cash_factor = (1.0 + CASH_ANNUAL_RATE / TRADING_DAYS_PER_YEAR) ** HORIZON
    future_full_factor = np.full(len(market), np.nan, dtype=float)
    future_cash_factor = np.full(len(market), np.nan, dtype=float)
    for signal_index in range(len(market)):
        start_index = signal_index + 1
        end_index = start_index + HORIZON - 1
        if end_index >= len(market):
            continue
        start_date = pd.Timestamp(market.loc[start_index, "date"])
        end_date = pd.Timestamp(market.loc[end_index, "date"])
        dividend = _dividend_per_share(dividends, start_date, end_date)
        future_full_factor[signal_index] = (
            float(market.loc[end_index, "etf_close"]) + dividend
        ) / float(market.loc[start_index, "etf_open"])
        future_cash_factor[signal_index] = cash_factor
    market["future5_full_factor"] = future_full_factor
    market["future5_cash_factor"] = future_cash_factor
    market["future5_excess_factor"] = market["future5_full_factor"] - market["future5_cash_factor"]
    market["target_good5"] = np.where(
        market["future5_excess_factor"].notna(),
        (market["future5_excess_factor"] > 0.0).astype(float),
        np.nan,
    )
    return market


def _merge_asof(
    left: pd.DataFrame,
    source: pd.DataFrame,
    *,
    prefix: str,
    feature_columns: list[str],
    allow_exact_matches: bool,
) -> pd.DataFrame:
    observation_column = f"{prefix}_observation_date"
    prepared = source[["date", *feature_columns]].rename(columns={"date": observation_column}).copy()
    prepared[observation_column] = prepared[observation_column].astype("datetime64[ns]")
    left = left.copy()
    left["date"] = left["date"].astype("datetime64[ns]")
    return pd.merge_asof(
        left.sort_values("date"),
        prepared.sort_values(observation_column),
        left_on="date",
        right_on=observation_column,
        direction="backward",
        allow_exact_matches=allow_exact_matches,
    )


def _load_vix() -> pd.DataFrame:
    frame = _read_parquet_until(VIX_FILE, columns=["date", "close"])
    close = pd.to_numeric(frame["close"], errors="raise")
    log_close = np.log(close)
    frame["vix_log_level"] = log_close
    frame["vix_ret1"] = log_close.diff(1)
    frame["vix_ret5"] = log_close.diff(5)
    for window, name, minimum in [(63, "vix_z63", 40), (252, "vix_z252", 126)]:
        mean = log_close.rolling(window, min_periods=minimum).mean()
        std = log_close.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
        frame[name] = (log_close - mean) / std
    return frame[["date", "vix_log_level", "vix_ret1", "vix_ret5", "vix_z63", "vix_z252"]]


def _load_global_market(stem: str, output_stem: str) -> pd.DataFrame:
    frame = _read_parquet_until(GLOBAL_DIR / f"{stem}_daily.parquet")
    close_column = "adj_close" if "adj_close" in frame.columns else "close"
    close = pd.to_numeric(frame[close_column], errors="coerce")
    frame[f"{output_stem}_ret1"] = close.pct_change(1)
    frame[f"{output_stem}_ret5"] = close.pct_change(5)
    return frame[["date", f"{output_stem}_ret1", f"{output_stem}_ret5"]]


def _load_cfets() -> pd.DataFrame:
    columns = ["date", "usd_cny", "eur_cny", "jpy100_cny", "gbp_cny", "sgd_cny", "cny_krw"]
    frame = _read_parquet_until(CFETS_FILE, columns=columns)
    weakness_series: dict[str, pd.Series] = {}
    for column in ["usd_cny", "eur_cny", "jpy100_cny", "gbp_cny", "sgd_cny"]:
        log_value = np.log(pd.to_numeric(frame[column], errors="coerce"))
        weakness_series[f"{column}_weak1"] = log_value.diff(1)
        weakness_series[f"{column}_weak5"] = log_value.diff(5)
    krw_log = np.log(pd.to_numeric(frame["cny_krw"], errors="coerce"))
    weakness_series["cny_krw_weak1"] = -krw_log.diff(1)
    weakness_series["cny_krw_weak5"] = -krw_log.diff(5)
    weakness = pd.DataFrame(weakness_series, index=frame.index)

    usd_log = np.log(pd.to_numeric(frame["usd_cny"], errors="raise"))
    frame["usd_cny_ret1"] = usd_log.diff(1)
    frame["usd_cny_ret5"] = usd_log.diff(5)
    frame["usd_cny_ret20"] = usd_log.diff(20)
    frame["fx_basket_weak1"] = weakness.filter(like="weak1").mean(axis=1, skipna=True)
    frame["fx_basket_weak5"] = weakness.filter(like="weak5").mean(axis=1, skipna=True)
    frame["fx_dispersion5"] = weakness.filter(like="weak5").std(axis=1, ddof=0, skipna=True)
    return frame[
        [
            "date",
            "usd_cny_ret1",
            "usd_cny_ret5",
            "usd_cny_ret20",
            "fx_basket_weak1",
            "fx_basket_weak5",
            "fx_dispersion5",
        ]
    ]


def _load_credit() -> pd.DataFrame:
    frame = _read_parquet_until(CREDIT_FILE, columns=["date", "credit_spread_3y_bp"])
    spread = pd.to_numeric(frame["credit_spread_3y_bp"], errors="raise")
    frame["credit_spread_level"] = spread
    frame["credit_spread_d1"] = spread.diff(1)
    frame["credit_spread_d5"] = spread.diff(5)
    mean = spread.rolling(252, min_periods=126).mean()
    std = spread.rolling(252, min_periods=126).std(ddof=0).replace(0.0, np.nan)
    frame["credit_spread_z252"] = (spread - mean) / std
    return frame[["date", "credit_spread_level", "credit_spread_d1", "credit_spread_d5", "credit_spread_z252"]]


def _add_external_features(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = market.copy()
    frame = _merge_asof(
        frame,
        _load_vix(),
        prefix="vix",
        feature_columns=["vix_log_level", "vix_ret1", "vix_ret5", "vix_z63", "vix_z252"],
        allow_exact_matches=False,
    )

    prior_observation_prefixes = ["vix"]
    same_day_allowed_prefixes: list[str] = []
    for stem, (output_stem, same_day_known) in GLOBAL_MARKETS.items():
        columns = [f"{output_stem}_ret1", f"{output_stem}_ret5"]
        frame = _merge_asof(
            frame,
            _load_global_market(stem, output_stem),
            prefix=output_stem,
            feature_columns=columns,
            allow_exact_matches=same_day_known,
        )
        if same_day_known:
            same_day_allowed_prefixes.append(output_stem)
        else:
            prior_observation_prefixes.append(output_stem)

    frame["us_mean_ret1"] = frame[["spx_ret1", "nasdaq_ret1", "dow_ret1"]].mean(axis=1)
    frame["us_mean_ret5"] = frame[["spx_ret5", "nasdaq_ret5", "dow_ret5"]].mean(axis=1)
    frame["eu_mean_ret1"] = frame[["dax_ret1", "ftse_ret1", "cac_ret1"]].mean(axis=1)
    frame["eu_mean_ret5"] = frame[["dax_ret5", "ftse_ret5", "cac_ret5"]].mean(axis=1)
    frame["asia_mean_ret1"] = frame[["nikkei_ret1", "kospi_ret1", "asx_ret1"]].mean(axis=1)
    frame["asia_mean_ret5"] = frame[["nikkei_ret5", "kospi_ret5", "asx_ret5"]].mean(axis=1)
    frame["global_min_ret1"] = frame[
        [
            "spx_ret1",
            "nasdaq_ret1",
            "dow_ret1",
            "dax_ret1",
            "ftse_ret1",
            "cac_ret1",
            "hsi_ret1",
            "nikkei_ret1",
            "kospi_ret1",
            "asx_ret1",
        ]
    ].min(axis=1)

    frame = _merge_asof(
        frame,
        _load_cfets(),
        prefix="cfets",
        feature_columns=[
            "usd_cny_ret1",
            "usd_cny_ret5",
            "usd_cny_ret20",
            "fx_basket_weak1",
            "fx_basket_weak5",
            "fx_dispersion5",
        ],
        allow_exact_matches=True,
    )
    same_day_allowed_prefixes.append("cfets")
    frame = _merge_asof(
        frame,
        _load_credit(),
        prefix="credit",
        feature_columns=["credit_spread_level", "credit_spread_d1", "credit_spread_d5", "credit_spread_z252"],
        allow_exact_matches=False,
    )
    prior_observation_prefixes.append("credit")

    alignment: dict[str, Any] = {}
    for prefix in prior_observation_prefixes:
        column = f"{prefix}_observation_date"
        valid = frame[column].notna()
        violations = int((frame.loc[valid, column] >= frame.loc[valid, "date"]).sum())
        alignment[prefix] = {
            "rule": "observation_date_strictly_before_signal_date",
            "matched_rows": int(valid.sum()),
            "violations": violations,
        }
        if violations:
            raise ValueError(f"{prefix} 存在决策时点违规记录：{violations}")
    for prefix in same_day_allowed_prefixes:
        column = f"{prefix}_observation_date"
        valid = frame[column].notna()
        violations = int((frame.loc[valid, column] > frame.loc[valid, "date"]).sum())
        alignment[prefix] = {
            "rule": "observation_date_not_after_signal_date",
            "matched_rows": int(valid.sum()),
            "violations": violations,
        }
        if violations:
            raise ValueError(f"{prefix} 存在未来观测：{violations}")
    return frame, alignment


def _add_mechanism_score(frame: pd.DataFrame, fit_mask: pd.Series) -> tuple[pd.DataFrame, dict[str, Any], float]:
    result = frame.copy()
    transformed: list[pd.Series] = []
    parameters: dict[str, Any] = {}
    for column, sign in MECHANISM_COMPONENTS.items():
        fit_values = pd.to_numeric(result.loc[fit_mask, column], errors="coerce")
        median = float(fit_values.median())
        scale = float(fit_values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        values = pd.to_numeric(result[column], errors="coerce").fillna(median)
        transformed.append(sign * (values - median) / scale)
        parameters[column] = {"median": median, "scale": scale, "risk_sign": sign}
    result["score_mechanism_risk"] = pd.concat(transformed, axis=1).mean(axis=1)
    threshold = float(result.loc[fit_mask, "score_mechanism_risk"].quantile(0.80))
    return result, parameters, threshold


def _fit_models(frame: pd.DataFrame, fit_mask: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    valid_fit = fit_mask & result["target_good5"].notna()
    x_fit = result.loc[valid_fit, FEATURE_COLUMNS]
    y_good = result.loc[valid_fit, "target_good5"].astype(int)
    if y_good.nunique() != 2:
        raise ValueError("拟合区间五日状态标签不足两个类别")

    logistic_good = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    hgb_good = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.03,
                    max_iter=120,
                    max_leaf_nodes=7,
                    max_depth=2,
                    min_samples_leaf=25,
                    l2_regularization=10.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    logistic_good.fit(x_fit, y_good)
    hgb_good.fit(x_fit, y_good)

    bad_tail_threshold = float(result.loc[valid_fit, "future5_excess_factor"].quantile(0.20))
    y_bad = (result.loc[valid_fit, "future5_excess_factor"] <= bad_tail_threshold).astype(int)
    logistic_bad = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    logistic_bad.fit(x_fit, y_bad)

    valid_prediction = result[FEATURE_COLUMNS].notna().any(axis=1)
    x_all = result.loc[valid_prediction, FEATURE_COLUMNS]
    result["prob_logistic_good5"] = np.nan
    result["prob_hgb_good5"] = np.nan
    result["prob_logistic_bad_tail5"] = np.nan
    result.loc[valid_prediction, "prob_logistic_good5"] = logistic_good.predict_proba(x_all)[:, 1]
    result.loc[valid_prediction, "prob_hgb_good5"] = hgb_good.predict_proba(x_all)[:, 1]
    result.loc[valid_prediction, "prob_logistic_bad_tail5"] = logistic_bad.predict_proba(x_all)[:, 1]
    result["prob_ensemble_good5"] = result[["prob_logistic_good5", "prob_hgb_good5"]].mean(axis=1)

    good_coefficients = logistic_good.named_steps["model"].coef_[0]
    bad_coefficients = logistic_bad.named_steps["model"].coef_[0]
    summary = {
        "fit_rows": int(valid_fit.sum()),
        "fit_start": result.loc[valid_fit, "date"].min().date().isoformat(),
        "fit_end": result.loc[valid_fit, "date"].max().date().isoformat(),
        "fit_good_share": float(y_good.mean()),
        "bad_tail_threshold_excess_factor": bad_tail_threshold,
        "logistic_good_coefficients_standardized": {
            column: float(value) for column, value in zip(FEATURE_COLUMNS, good_coefficients, strict=True)
        },
        "logistic_bad_coefficients_standardized": {
            column: float(value) for column, value in zip(FEATURE_COLUMNS, bad_coefficients, strict=True)
        },
        "hgb_parameters": hgb_good.named_steps["model"].get_params(),
    }
    return result, summary


def _rule_functions(mechanism_threshold: float) -> dict[str, Callable[[pd.Series, int], int]]:
    return {
        "LOGISTIC_GOOD_050": lambda row, previous: int(float(row["prob_logistic_good5"]) >= 0.50),
        "LOGISTIC_GOOD_HYSTERESIS_045_055": lambda row, previous: (
            1
            if float(row["prob_logistic_good5"]) >= 0.55
            else 0
            if float(row["prob_logistic_good5"]) <= 0.45
            else int(previous)
        ),
        "HGB_GOOD_050": lambda row, previous: int(float(row["prob_hgb_good5"]) >= 0.50),
        "ENSEMBLE_GOOD_050": lambda row, previous: int(float(row["prob_ensemble_good5"]) >= 0.50),
        "LOGISTIC_BAD_TAIL_050": lambda row, previous: int(float(row["prob_logistic_bad_tail5"]) < 0.50),
        "MECHANISM_RISK_Q80": lambda row, previous: int(float(row["score_mechanism_risk"]) < mechanism_threshold),
    }


def _evaluate_rule_offset(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
    rule_name: str,
    rule: Callable[[pd.Series, int], int],
    offset: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selection_indices = np.flatnonzero(
        ((frame["date"] >= SELECTION_START) & (frame["date"] <= DEVELOPMENT_CUTOFF)).to_numpy()
    )
    if offset >= len(selection_indices):
        raise ValueError("错位超过选择区间长度")
    first_execution_index = int(selection_indices[offset])
    if first_execution_index <= 0:
        raise ValueError("选择区间缺少前一交易日信号锚点")
    available = len(selection_indices) - offset
    execution_days = (available // HORIZON) * HORIZON
    last_execution_index = first_execution_index + execution_days - 1
    anchor_index = first_execution_index - 1
    sample = frame.iloc[anchor_index : last_execution_index + 1][
        ["date", "etf_open", "etf_close", "benchmark_close"]
    ].reset_index(drop=True)
    states = np.zeros(len(sample), dtype=np.int8)
    block_rows: list[dict[str, Any]] = []
    previous_state = 1
    cash_factor = (1.0 + CASH_ANNUAL_RATE / TRADING_DAYS_PER_YEAR) ** HORIZON

    for relative_start in range(0, execution_days, HORIZON):
        absolute_start = first_execution_index + relative_start
        absolute_end = absolute_start + HORIZON - 1
        signal_index = absolute_start - 1
        signal_row = frame.iloc[signal_index]
        prediction = int(rule(signal_row, previous_state))
        if prediction not in {0, 1}:
            raise ValueError(f"规则 {rule_name} 产生非法状态：{prediction}")
        sample_start = 1 + relative_start
        sample_end = sample_start + HORIZON - 1
        states[sample_start : sample_end + 1] = prediction
        start_date = pd.Timestamp(frame.loc[absolute_start, "date"])
        end_date = pd.Timestamp(frame.loc[absolute_end, "date"])
        dividend = _dividend_per_share(dividends, start_date, end_date)
        full_factor = (
            float(frame.loc[absolute_end, "etf_close"]) + dividend
        ) / float(frame.loc[absolute_start, "etf_open"])
        oracle_state = int(full_factor > cash_factor)
        block_rows.append(
            {
                "rule": rule_name,
                "calendar_offset": offset,
                "block_index": int(relative_start // HORIZON),
                "signal_date": pd.Timestamp(signal_row["date"]),
                "start_date": start_date,
                "end_date": end_date,
                "predicted_state": prediction,
                "oracle_state": oracle_state,
                "full_factor": full_factor,
                "cash_factor": cash_factor,
                "avoidable_loss_factor": max(cash_factor - full_factor, 0.0),
                "prob_logistic_good5": float(signal_row["prob_logistic_good5"]),
                "prob_hgb_good5": float(signal_row["prob_hgb_good5"]),
                "prob_ensemble_good5": float(signal_row["prob_ensemble_good5"]),
                "prob_logistic_bad_tail5": float(signal_row["prob_logistic_bad_tail5"]),
                "score_mechanism_risk": float(signal_row["score_mechanism_risk"]),
            }
        )
        previous_state = prediction

    blocks = pd.DataFrame(block_rows)
    benchmark = build_benchmark_ledger(sample, INITIAL_CAPITAL_CNY)
    base_ledger, base_trades = simulate_binary_path(
        sample,
        dividends,
        states,
        costs=BASE_COSTS,
        initial_capital=INITIAL_CAPITAL_CNY,
        reinvest_paid_dividends=True,
    )
    stress_ledger, stress_trades = simulate_binary_path(
        sample,
        dividends,
        states,
        costs=STRESS_COSTS,
        initial_capital=INITIAL_CAPITAL_CNY,
        reinvest_paid_dividends=True,
    )
    base_summary = summarize_path(base_ledger, base_trades, benchmark, objective=OBJECTIVE)
    stress_summary = summarize_path(stress_ledger, stress_trades, benchmark, objective=OBJECTIVE)

    bad = blocks["oracle_state"].eq(0)
    good = ~bad
    predicted_cash = blocks["predicted_state"].eq(0)
    total_avoidable = float(blocks["avoidable_loss_factor"].sum())
    captured_avoidable = float(blocks.loc[predicted_cash, "avoidable_loss_factor"].sum())
    row: dict[str, Any] = {
        "rule": rule_name,
        "calendar_offset": offset,
        "block_count": int(len(blocks)),
        "bad_block_count": int(bad.sum()),
        "cash_block_count": int(predicted_cash.sum()),
        "bad_block_recall": _safe_float(predicted_cash[bad].mean()),
        "good_block_false_exit": _safe_float(predicted_cash[good].mean()),
        "bad_loss_capture": _safe_float(captured_avoidable / total_avoidable if total_avoidable > 0 else np.nan),
    }
    for prefix, summary in [("base", base_summary), ("stress", stress_summary)]:
        for key, value in summary.items():
            row[f"{prefix}_{key}"] = value
    return row, blocks


def _aggregate_offsets(offsets: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule, group in offsets.groupby("rule", sort=True):
        stress_annual = pd.to_numeric(group["stress_annualized_excess"], errors="coerce")
        stress_rolling = pd.to_numeric(group["stress_rolling_242d_excess_median"], errors="coerce")
        base_annual = pd.to_numeric(group["base_annualized_excess"], errors="coerce")
        base_rolling = pd.to_numeric(group["base_rolling_242d_excess_median"], errors="coerce")
        every_offset_passes = bool(group["stress_both_20pct_gates"].all() and len(group) == HORIZON)
        rows.append(
            {
                "rule": rule,
                "offset_count": int(len(group)),
                "base_annualized_excess_minimum": _safe_float(base_annual.min()),
                "base_annualized_excess_median": _safe_float(base_annual.median()),
                "base_rolling_242d_excess_median_across_offsets": _safe_float(base_rolling.median()),
                "stress_annualized_excess_minimum": _safe_float(stress_annual.min()),
                "stress_annualized_excess_median": _safe_float(stress_annual.median()),
                "stress_annualized_excess_maximum": _safe_float(stress_annual.max()),
                "stress_rolling_242d_excess_minimum_across_offsets": _safe_float(stress_rolling.min()),
                "stress_rolling_242d_excess_median_across_offsets": _safe_float(stress_rolling.median()),
                "stress_every_offset_both_20pct_gates": every_offset_passes,
                "stress_offset_pass_ratio": float(group["stress_both_20pct_gates"].mean()),
                "cash_block_share_median": float((group["cash_block_count"] / group["block_count"]).median()),
                "bad_block_recall_median": _safe_float(group["bad_block_recall"].median()),
                "good_block_false_exit_median": _safe_float(group["good_block_false_exit"].median()),
                "bad_loss_capture_median": _safe_float(group["bad_loss_capture"].median()),
                "stress_maximum_drawdown_median": _safe_float(group["stress_maximum_drawdown"].median()),
                "stress_trade_leg_count_median": _safe_float(group["stress_trade_leg_count"].median()),
                "development_gate": every_offset_passes,
            }
        )
    rows.sort(
        key=lambda item: (
            item["stress_annualized_excess_median"]
            if item["stress_annualized_excess_median"] is not None
            else -float("inf")
        ),
        reverse=True,
    )
    return rows


def main() -> int:
    required = [ETF_FILE, BENCHMARK_FILE, DIVIDEND_FILE, VIX_FILE, CFETS_FILE, CREDIT_FILE]
    required.extend(GLOBAL_DIR / f"{stem}_daily.parquet" for stem in GLOBAL_MARKETS)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    dividends = _load_dividends()
    market = _load_market(dividends)
    frame, alignment_audit = _add_external_features(market)
    if frame["date"].max() > DEVELOPMENT_CUTOFF:
        raise ValueError("特征表越过开发上限")
    fit_mask = (frame["date"] >= FIT_START) & (frame["date"] <= FIT_END)
    selection_mask = (frame["date"] >= SELECTION_START) & (frame["date"] <= DEVELOPMENT_CUTOFF)
    if int(fit_mask.sum()) < 400 or int(selection_mask.sum()) < 400:
        raise ValueError("拟合段或选择段样本不足")

    frame, mechanism_parameters, mechanism_threshold = _add_mechanism_score(frame, fit_mask)
    frame, model_summary = _fit_models(frame, fit_mask)
    rules = _rule_functions(mechanism_threshold)

    offset_rows: list[dict[str, Any]] = []
    block_frames: list[pd.DataFrame] = []
    for rule_name, rule in rules.items():
        for offset in range(HORIZON):
            offset_row, blocks = _evaluate_rule_offset(frame, dividends, rule_name, rule, offset)
            offset_rows.append(offset_row)
            block_frames.append(blocks)
    offsets = pd.DataFrame(offset_rows)
    blocks = pd.concat(block_frames, ignore_index=True)
    aggregates = _aggregate_offsets(offsets)
    best = aggregates[0]
    passed = bool(best["development_gate"])

    observation_columns = [column for column in frame.columns if column.endswith("_observation_date")]
    score_columns = [
        "score_mechanism_risk",
        "prob_logistic_good5",
        "prob_hgb_good5",
        "prob_ensemble_good5",
        "prob_logistic_bad_tail5",
    ]
    feature_output = frame[
        [
            "date",
            "etf_open",
            "etf_close",
            "benchmark_close",
            "future5_full_factor",
            "future5_cash_factor",
            "future5_excess_factor",
            "target_good5",
            *FEATURE_COLUMNS,
            *score_columns,
            *observation_columns,
        ]
    ].copy()
    _atomic_parquet(feature_output, OUTPUT_FEATURES)
    _atomic_parquet(offsets, OUTPUT_OFFSETS)
    _atomic_parquet(blocks, OUTPUT_BLOCKS)

    payload = {
        "status": (
            "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
            if passed
            else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
        ),
        "project_id": "510300_GLOBAL_LIQUIDITY_REGIME_5D_DISCOVERY_V0",
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "state_meanings": {"0": "CASH_CNY", "1": "FULL_510300"},
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
            "holding_block_trading_days": HORIZON,
            "benchmark": "H00300_TOTAL_RETURN",
            "initial_capital_cny": INITIAL_CAPITAL_CNY,
        },
        "development_contract": {
            "data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
            "fit_period": [FIT_START.date().isoformat(), FIT_END.date().isoformat()],
            "selection_period": [SELECTION_START.date().isoformat(), DEVELOPMENT_CUTOFF.date().isoformat()],
            "calendar_offsets": list(range(HORIZON)),
            "selection_rule": "按五种日历错位的压力成本年化超额中位数排序；只有每个错位的总体与242日滚动中位数均不低于20%才可冻结",
            "validation_2021_plus_loaded": False,
            "costs": {
                "commission_rate_per_leg": 0.0001,
                "base_slippage_bps_per_leg": 5.0,
                "stress_slippage_bps_per_leg": 15.0,
                "cash_annual_rate": CASH_ANNUAL_RATE,
                "lot_size": 100,
            },
        },
        "rows": {
            "feature_rows": int(len(frame)),
            "fit_rows_calendar": int(fit_mask.sum()),
            "selection_rows_calendar": int(selection_mask.sum()),
            "offset_metric_rows": int(len(offsets)),
            "block_prediction_rows": int(len(blocks)),
        },
        "point_in_time_alignment_audit": alignment_audit,
        "feature_columns": FEATURE_COLUMNS,
        "mechanism_score": {
            "components": MECHANISM_COMPONENTS,
            "fit_transform_parameters": mechanism_parameters,
            "fit_q80_threshold": mechanism_threshold,
        },
        "models": model_summary,
        "best_development_rule": best,
        "all_rule_aggregates": aggregates,
        "all_offset_metrics": offsets.to_dict("records"),
        "development_gate_passed": passed,
        "validation_2021_plus_loaded": False,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in required
        },
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "offset_metrics": str(OUTPUT_OFFSETS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "block_predictions": str(OUTPUT_BLOCKS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "interpretation": (
            "开发门槛通过；下一步只能先冻结公式与哈希，再读取2021年后的独立验证。"
            if passed
            else "外部流动性与人民币压力域未达到冻结门槛；保留负结果并继续搜索新证据域。"
        ),
        "is_trading_signal": False,
        "live_trading_authorized": False,
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
