"""全A股横截面脆弱性形状的5日二元择时发现扫描。

数据隔离：全A训练面板与510300目标都只读取到2020-12-31；2016-2018
拟合，2019-2020作为开发选择段。脚本比较两个预先固定的模型，风险最高
20%时下一交易日开盘转为空仓。2021及以后数据完全不读取。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.registered_factors import ExecutableTargetCosts, add_executable_targets


PANEL_FILE = PROJECT_ROOT / "data" / "raw" / "a_share_hash_holdout_v2" / "training_panel.parquet"
ETF_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
DIVIDEND_FILE = PROJECT_ROOT / "data" / "reference" / "510300_dividends.csv"
AGGREGATE_FILE = PROJECT_ROOT / "data" / "features" / "all_a_fragility_shape_daily_discovery_v0.parquet"
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "510300_all_a_fragility_shape_5d_discovery_v0.parquet"
REPORT_FILE = PROJECT_ROOT / "reports" / "discovery" / "510300_all_a_fragility_shape_5d_discovery_v0.json"

DATA_CEILING = pd.Timestamp("2020-12-31")
FIT_START = pd.Timestamp("2016-08-12")
FIT_END = pd.Timestamp("2018-12-28")
SELECTION_START = pd.Timestamp("2019-01-02")
RISK_QUANTILE = 0.80
BASE_ONE_WAY_COST = 0.000925
STRESS_ONE_WAY_COST = 0.00125
CASH_ANNUAL_RATE = 0.015
TRADING_DAYS_PER_YEAR = 242
RANDOM_STATE = 20260828

RAW_CROSS_SECTION_COLUMNS = [
    "active_count",
    "cs_mean_ret",
    "cs_median_ret",
    "cs_q01_ret",
    "cs_q05_ret",
    "cs_q10_ret",
    "cs_q90_ret",
    "cs_q95_ret",
    "cs_q99_ret",
    "cs_dispersion",
    "cs_skewness",
    "cs_kurtosis",
    "share_down",
    "share_down2",
    "share_down5",
    "share_up2",
    "share_up5",
    "share_limit_down",
    "share_limit_up",
    "share_close_near_low",
    "share_close_near_high",
    "mean_gap",
    "mean_intraday_return",
    "median_intraday_range",
    "down_amount_share",
    "up_amount_share",
    "top10pct_amount_share",
    "total_amount_log",
]

MODEL_FEATURES = [
    "cs_mean_ret",
    "cs_median_ret",
    "cs_q01_ret",
    "cs_q05_ret",
    "cs_q10_ret",
    "cs_q90_ret",
    "cs_q95_ret",
    "cs_dispersion",
    "cs_skewness",
    "cs_kurtosis",
    "share_down",
    "share_down2",
    "share_down5",
    "share_up2",
    "share_up5",
    "share_limit_down",
    "share_limit_up",
    "share_close_near_low",
    "share_close_near_high",
    "mean_gap",
    "mean_intraday_return",
    "median_intraday_range",
    "down_amount_share",
    "top10pct_amount_share",
    "total_amount_log",
    "share_down_change1",
    "share_down_change5",
    "share_down2_change1",
    "share_down2_change5",
    "cs_q05_change1",
    "cs_q05_change5",
    "cs_dispersion_change5",
    "down_amount_share_change5",
    "share_close_near_low_change5",
    "share_down_z20_prior",
    "share_down2_z20_prior",
    "cs_dispersion_z20_prior",
    "share_limit_down_z20_prior",
    "down_amount_share_z20_prior",
    "total_amount_log_z20_prior",
    "etf_return_1d",
    "etf_return_5d",
    "etf_return_20d",
    "etf_rv_5d",
    "etf_rv_20d",
    "etf_downside_share_20d",
    "etf_range_position_60d",
    "etf_gap",
    "etf_intraday_return",
    "etf_close_location",
    "etf_amount_z20_prior",
]


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _cagr(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0 or bool((values <= -1.0).any()):
        return float("nan")
    return float(np.prod(1.0 + values) ** (TRADING_DAYS_PER_YEAR / len(values)) - 1.0)


def _maximum_drawdown(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    equity = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks - 1.0)) if len(equity) else float("nan")


def _build_cross_section_aggregates() -> pd.DataFrame:
    parquet = PANEL_FILE.as_posix().replace("'", "''")
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    temporary = PROJECT_ROOT / "tmp" / "all_a_fragility_shape"
    temporary.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory='{temporary.as_posix().replace(chr(39), chr(39) * 2)}'")
    query = f"""
        WITH valid AS (
            SELECT
                CAST(date AS DATE) AS date,
                CAST(pct_chg AS DOUBLE) / 100.0 AS ret,
                CAST(amount AS DOUBLE) AS amount,
                CAST(raw_open AS DOUBLE) AS open,
                CAST(raw_high AS DOUBLE) AS high,
                CAST(raw_low AS DOUBLE) AS low,
                CAST(raw_close AS DOUBLE) AS close,
                CAST(pre_close AS DOUBLE) AS pre_close
            FROM read_parquet('{parquet}')
            WHERE CAST(date AS DATE) <= DATE '2020-12-31'
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
    result["date"] = pd.to_datetime(result["date"])
    return result


def _prior_z(series: pd.Series, window: int = 20, minimum: int = 10) -> pd.Series:
    prior = pd.to_numeric(series, errors="coerce").shift(1)
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (pd.to_numeric(series, errors="coerce") - mean) / std


def _add_cross_section_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values("date").copy()
    for column in ["share_down", "share_down2", "cs_q05_ret"]:
        output[f"{column.replace('_ret', '')}_change1"] = output[column].diff(1)
        output[f"{column.replace('_ret', '')}_change5"] = output[column].diff(5)
    output["cs_dispersion_change5"] = output["cs_dispersion"].diff(5)
    output["down_amount_share_change5"] = output["down_amount_share"].diff(5)
    output["share_close_near_low_change5"] = output["share_close_near_low"].diff(5)
    for column in [
        "share_down",
        "share_down2",
        "cs_dispersion",
        "share_limit_down",
        "down_amount_share",
        "total_amount_log",
    ]:
        output[f"{column}_z20_prior"] = _prior_z(output[column])
    return output


def _load_etf_and_targets() -> pd.DataFrame:
    etf = pd.read_parquet(ETF_FILE)
    etf["date"] = pd.to_datetime(etf["date"])
    etf = etf[etf["date"] <= DATA_CEILING].sort_values("date").reset_index(drop=True)
    dividends = pd.read_csv(DIVIDEND_FILE, parse_dates=["record_date", "ex_date", "payment_date"])
    dividends = dividends[dividends["ex_date"] <= DATA_CEILING].copy()
    costs = ExecutableTargetCosts(
        initial_cash=20_000.0,
        commission_rate=0.0003,
        minimum_commission_cny=5.0,
        stamp_duty_rate=0.0,
        slippage_bps=5.0,
        lot_size=100,
    )
    targets = add_executable_targets(etf, dividends, horizons=(5,), costs=costs)
    close = pd.to_numeric(etf["close"], errors="coerce")
    open_price = pd.to_numeric(etf["open"], errors="coerce")
    high = pd.to_numeric(etf["high"], errors="coerce")
    low = pd.to_numeric(etf["low"], errors="coerce")
    amount = pd.to_numeric(etf["amount"], errors="coerce")
    returns = close.pct_change()
    target = pd.DataFrame({"date": etf["date"]})
    target["etf_return_1d"] = returns
    target["etf_return_5d"] = close.pct_change(5)
    target["etf_return_20d"] = close.pct_change(20)
    target["etf_rv_5d"] = returns.rolling(5, min_periods=5).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    target["etf_rv_20d"] = returns.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    downside_squared = returns.clip(upper=0.0).pow(2).rolling(20, min_periods=20).sum()
    total_squared = returns.pow(2).rolling(20, min_periods=20).sum()
    target["etf_downside_share_20d"] = downside_squared / total_squared.replace(0.0, np.nan)
    rolling_high = close.rolling(60, min_periods=60).max()
    rolling_low = close.rolling(60, min_periods=60).min()
    target["etf_range_position_60d"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)
    target["etf_gap"] = open_price / close.shift(1) - 1.0
    target["etf_intraday_return"] = close / open_price - 1.0
    target["etf_close_location"] = (close - low) / (high - low).replace(0.0, np.nan)
    target["etf_amount_z20_prior"] = _prior_z(amount)
    target["future_5d_return"] = targets["exec_total_return_5d_gross"]
    dividend_by_date = dividends.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()
    next_open = open_price.shift(-1)
    following_open = open_price.shift(-2)
    following_date = etf["date"].shift(-2)
    following_dividend = following_date.map(lambda value: float(dividend_by_date.get(value, 0.0)) if pd.notna(value) else 0.0)
    target["one_day_open_total_return"] = (following_open + following_dividend) / next_open - 1.0
    return target


def _models() -> dict[str, Any]:
    return {
        "logistic_l2": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        loss="log_loss",
                        learning_rate=0.03,
                        max_iter=200,
                        max_leaf_nodes=7,
                        max_depth=3,
                        min_samples_leaf=20,
                        l2_regularization=10.0,
                        class_weight="balanced",
                        early_stopping=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _bootstrap_excess(strategy: np.ndarray, benchmark: np.ndarray, repetitions: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    length = len(strategy)
    block = 5
    starts = np.arange(length - block + 1)
    values: list[float] = []
    for _ in range(repetitions):
        indices: list[int] = []
        while len(indices) < length:
            start = int(rng.choice(starts))
            indices.extend(range(start, start + block))
        selected = np.asarray(indices[:length])
        values.append(_cagr(pd.Series(strategy[selected])) - _cagr(pd.Series(benchmark[selected])))
    return {
        "median": float(np.quantile(values, 0.50)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _evaluate(frame: pd.DataFrame, model_name: str, probabilities: pd.Series, threshold: float, bad_threshold: float) -> dict[str, Any]:
    evaluation = frame[frame["date"] >= SELECTION_START].copy()
    evaluation["probability"] = probabilities.loc[evaluation.index]
    evaluation = evaluation[evaluation[["probability", "one_day_open_total_return", "future_5d_return"]].notna().all(axis=1)]
    evaluation["cash"] = evaluation["probability"] >= threshold
    position = (~evaluation["cash"]).astype(int)
    transitions = position.diff().abs().fillna(0.0)
    cash_daily = (1.0 + CASH_ANNUAL_RATE) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    gross = pd.Series(
        np.where(position.eq(1), evaluation["one_day_open_total_return"], cash_daily),
        index=evaluation.index,
    )
    base = gross - transitions * BASE_ONE_WAY_COST
    stress = gross - transitions * STRESS_ONE_WAY_COST
    benchmark = evaluation["one_day_open_total_return"]
    base_cagr = _cagr(base)
    stress_cagr = _cagr(stress)
    benchmark_cagr = _cagr(benchmark)
    bad = evaluation["future_5d_return"] <= bad_threshold
    bad_losses = -evaluation.loc[bad, "future_5d_return"].clip(upper=0.0).sum()
    captured = -evaluation.loc[bad & evaluation["cash"], "future_5d_return"].clip(upper=0.0).sum()
    labels = bad.astype(int)
    yearly = []
    for year, group in evaluation.groupby(evaluation["date"].dt.year):
        group_position = (~group["cash"]).astype(int)
        group_transitions = group_position.diff().abs().fillna(0.0)
        group_gross = pd.Series(
            np.where(group_position.eq(1), group["one_day_open_total_return"], cash_daily), index=group.index
        )
        group_strategy = group_gross - group_transitions * BASE_ONE_WAY_COST
        yearly.append(
            {
                "year": int(year),
                "base_excess": _cagr(group_strategy) - _cagr(group["one_day_open_total_return"]),
                "cash_share": float(group["cash"].mean()),
            }
        )
    return {
        "model": model_name,
        "probability_threshold_from_fit": threshold,
        "rows": int(len(evaluation)),
        "cash_days": int(evaluation["cash"].sum()),
        "cash_share": float(evaluation["cash"].mean()),
        "trade_legs": int(transitions.sum()),
        "benchmark_cagr": benchmark_cagr,
        "base_strategy_cagr": base_cagr,
        "stress_strategy_cagr": stress_cagr,
        "base_annualized_excess": base_cagr - benchmark_cagr,
        "stress_annualized_excess": stress_cagr - benchmark_cagr,
        "base_max_drawdown": _maximum_drawdown(base),
        "bad5_auc": float(roc_auc_score(labels, evaluation["probability"])) if labels.nunique() == 2 else None,
        "bad5_average_precision": float(average_precision_score(labels, evaluation["probability"]))
        if labels.nunique() == 2
        else None,
        "bad5_recall": float(evaluation.loc[bad, "cash"].mean()) if bool(bad.any()) else None,
        "good5_false_exit": float(evaluation.loc[~bad, "cash"].mean()) if bool((~bad).any()) else None,
        "bad5_loss_capture": float(captured / bad_losses) if bad_losses > 0 else None,
        "cash_day_mean_one_day_return": float(evaluation.loc[evaluation["cash"], "one_day_open_total_return"].mean()),
        "cash_day_mean_future_5d_return": float(evaluation.loc[evaluation["cash"], "future_5d_return"].mean()),
        "yearly": yearly,
        "base_bootstrap_excess": _bootstrap_excess(base.to_numpy(dtype=float), benchmark.to_numpy(dtype=float)),
    }


def main() -> int:
    missing = [str(path) for path in [PANEL_FILE, ETF_FILE, DIVIDEND_FILE] if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, REPORT_FILE)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    aggregate = _build_cross_section_aggregates()
    if aggregate["date"].max() > DATA_CEILING:
        raise RuntimeError("全A聚合越过开发数据上限。")
    _atomic_parquet(aggregate, AGGREGATE_FILE)
    cross_section = _add_cross_section_time_features(aggregate)
    target = _load_etf_and_targets()
    frame = cross_section.merge(target, on="date", how="inner", validate="one_to_one")
    frame = frame[frame["date"].between(FIT_START, DATA_CEILING)].sort_values("date").reset_index(drop=True)
    absent = [column for column in MODEL_FEATURES if column not in frame.columns]
    if absent:
        raise RuntimeError(f"模型特征缺失：{absent}")
    fit_mask = frame["date"] <= FIT_END
    selection_mask = frame["date"] >= SELECTION_START
    fit = frame.loc[fit_mask].dropna(subset=["future_5d_return"])
    if len(fit) < 500 or int(selection_mask.sum()) < 450:
        raise RuntimeError("拟合段或开发选择段样本不足。")
    bad_threshold = float(fit["future_5d_return"].quantile(0.10))
    labels = (fit["future_5d_return"] <= bad_threshold).astype(int)
    model_results: list[dict[str, Any]] = []
    probability_columns: list[str] = []
    for model_name, model in _models().items():
        model.fit(fit[MODEL_FEATURES], labels)
        probabilities = pd.Series(model.predict_proba(frame[MODEL_FEATURES])[:, 1], index=frame.index)
        column = f"probability_{model_name}"
        frame[column] = probabilities
        probability_columns.append(column)
        threshold = float(probabilities.loc[fit.index].quantile(RISK_QUANTILE))
        model_results.append(_evaluate(frame, model_name, probabilities, threshold, bad_threshold))
    model_results.sort(key=lambda item: item["base_annualized_excess"], reverse=True)
    _atomic_parquet(
        frame[["date", "future_5d_return", "one_day_open_total_return", *MODEL_FEATURES, *probability_columns]],
        FEATURE_FILE,
    )
    payload = {
        "status": "DEVELOPMENT_DISCOVERY_COMPLETE_NO_2021_PLUS_READ",
        "project_id": "510300_ALL_A_FRAGILITY_SHAPE_5D_DISCOVERY_V0",
        "data_ceiling": DATA_CEILING.date().isoformat(),
        "fit_period": [FIT_START.date().isoformat(), FIT_END.date().isoformat()],
        "selection_period": [SELECTION_START.date().isoformat(), DATA_CEILING.date().isoformat()],
        "aggregate_rows": int(len(aggregate)),
        "model_rows": int(len(frame)),
        "fit_rows": int(len(fit)),
        "selection_rows": int(selection_mask.sum()),
        "feature_count": len(MODEL_FEATURES),
        "bad5_fit_threshold": bad_threshold,
        "risk_quantile": RISK_QUANTILE,
        "model_results": model_results,
        "best": model_results[0],
        "validation_2021_2022_loaded": False,
        "hash_replication_2023_plus_loaded": False,
        "outputs": {
            "aggregate": str(AGGREGATE_FILE.relative_to(PROJECT_ROOT)),
            "features": str(FEATURE_FILE.relative_to(PROJECT_ROOT)),
        },
        "interpretation": "开发选择扫描，不是正式候选。只有开发选择段接近20个百分点才值得冻结后读取2021-2022。",
    }
    _atomic_json(payload, REPORT_FILE)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
