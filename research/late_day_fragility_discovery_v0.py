"""只读开发样本的510300收盘前内部脆弱性发现扫描。

本脚本只允许读取截至2023-12-29的数据。它不生成交易指令；用途是从
2021-2022形成固定变换，在2023开发选择段比较少量、机制明确的风险分数，
为后续独立冻结的2024验证候选提供依据。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_CUTOFF = pd.Timestamp("2023-12-29")
FIT_END = pd.Timestamp("2022-12-30")
SELECTION_START = pd.Timestamp("2023-01-03")
SIGNAL_TIME = "14:50:00"
EXECUTION_TIME = "14:51:00"
TRADING_DAYS_PER_YEAR = 242
CASH_ANNUAL_RATE = 0.015
BASE_ONE_WAY_COST = 0.000925
STRESS_ONE_WAY_COST = 0.00125
RANDOM_STATE = 20260828

ETF_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_1m_tushare_raw.parquet"
INDEX_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "000300_1m_tushare_raw.parquet"
BREADTH_FILE = PROJECT_ROOT / "data" / "features" / "top50_breadth_no_if_1m.parquet"
GLOBAL_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "cross_market_chart_ml_v1"
OUTPUT_FEATURES = PROJECT_ROOT / "data" / "features" / "510300_late_day_fragility_discovery_v0.parquet"
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "discovery" / "510300_late_day_fragility_discovery_v0.json"

GLOBAL_MARKETS = {
    "GSPC": "us_sp500",
    "IXIC": "us_nasdaq",
    "DJI": "us_dow",
    "GDAXI": "eu_dax",
    "FTSE": "eu_ftse",
    "FCHI": "eu_cac",
    "HSI": "hk_hsi",
}

MODEL_FEATURES = [
    "etf_gap",
    "etf_ret_to_signal",
    "etf_ret_5m",
    "etf_ret_30m",
    "etf_ret_60m",
    "etf_realized_vol",
    "etf_downside_share",
    "etf_close_location",
    "etf_last30_amount_share",
    "index_relative_ret_to_signal",
    "breadth_last",
    "weighted_breadth_last",
    "weighted_minus_equal_last",
    "breadth_late_mean",
    "breadth_late_min",
    "breadth_late_change",
    "weighted_impulse_late_mean",
    "leader_pressure_late",
    "us_mean_ret1",
    "eu_mean_ret1",
    "global_min_ret1",
    "hk_hsi_ret1",
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


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def _cagr(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty or bool((clean <= -1.0).any()):
        return float("nan")
    growth = float(np.prod(1.0 + clean.to_numpy(dtype=float)))
    return growth ** (TRADING_DAYS_PER_YEAR / len(clean)) - 1.0


def _exact_close(group: pd.DataFrame, clock: str) -> float:
    rows = group.loc[group["clock"] == clock, "close"]
    return float(rows.iloc[-1]) if not rows.empty else float("nan")


def _minute_daily_features(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("trade_time").copy()
    signal = group[group["clock"] == SIGNAL_TIME]
    execution = group[group["clock"] == EXECUTION_TIME]
    if signal.empty or execution.empty:
        return pd.Series(dtype=float)
    signal_row = signal.iloc[-1]
    usable = group[group["trade_time"] <= signal_row["trade_time"]].copy()
    if usable.empty:
        return pd.Series(dtype=float)
    closes = pd.to_numeric(usable["close"], errors="coerce")
    opens = pd.to_numeric(usable["open"], errors="coerce")
    highs = pd.to_numeric(usable["high"], errors="coerce")
    lows = pd.to_numeric(usable["low"], errors="coerce")
    amounts = pd.to_numeric(usable["amount"], errors="coerce").fillna(0.0)
    log_returns = np.log(closes / closes.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    squared = log_returns.pow(2)
    downside = log_returns.clip(upper=0.0).pow(2)
    total_squared = float(squared.sum())
    signal_close = float(signal_row["close"])
    day_open = float(opens.iloc[0])
    day_high = float(highs.max())
    day_low = float(lows.min())
    total_amount = float(amounts.sum())
    last30_start = signal_row["trade_time"] - pd.Timedelta(minutes=29)
    last30_amount = float(amounts[usable["trade_time"] >= last30_start].sum())
    return pd.Series(
        {
            "session_date": pd.Timestamp(group["session_date"].iloc[0]),
            "execution_open": float(execution.iloc[-1]["open"]),
            "day_open": day_open,
            "etf_ret_to_signal": signal_close / day_open - 1.0,
            "etf_ret_5m": signal_close / _exact_close(usable, "14:45:00") - 1.0,
            "etf_ret_30m": signal_close / _exact_close(usable, "14:20:00") - 1.0,
            "etf_ret_60m": signal_close / _exact_close(usable, "13:50:00") - 1.0,
            "etf_realized_vol": math.sqrt(total_squared),
            "etf_downside_share": float(downside.sum()) / total_squared if total_squared > 0 else float("nan"),
            "etf_close_location": (signal_close - day_low) / (day_high - day_low) if day_high > day_low else 0.5,
            "etf_last30_amount_share": last30_amount / total_amount if total_amount > 0 else float("nan"),
        }
    )


def _load_etf_daily_features() -> pd.DataFrame:
    columns = ["trade_time", "open", "high", "low", "close", "amount"]
    minute = pd.read_parquet(ETF_FILE, columns=columns)
    minute["trade_time"] = pd.to_datetime(minute["trade_time"])
    minute = minute[minute["trade_time"].dt.normalize() <= DEVELOPMENT_CUTOFF].copy()
    minute["session_date"] = minute["trade_time"].dt.normalize()
    minute["clock"] = minute["trade_time"].dt.strftime("%H:%M:%S")
    rows: list[pd.Series] = []
    for session_date, group in minute.groupby("session_date", sort=True):
        features = _minute_daily_features(group)
        if features.empty:
            continue
        features["session_date"] = pd.Timestamp(session_date)
        rows.append(features)
    daily = pd.DataFrame(rows).reset_index(drop=True).sort_values("session_date")
    close_1500 = (
        minute[minute["clock"] == "15:00:00"]
        .set_index("session_date")["close"]
        .astype(float)
        .sort_index()
    )
    daily["previous_close"] = daily["session_date"].map(close_1500.shift(1))
    daily["etf_gap"] = daily["day_open"] / daily["previous_close"] - 1.0
    daily["next_execution_open"] = daily["execution_open"].shift(-1)
    daily["next_return"] = daily["next_execution_open"] / daily["execution_open"] - 1.0
    return daily


def _load_index_features() -> pd.DataFrame:
    minute = pd.read_parquet(INDEX_FILE, columns=["trade_time", "open", "close"])
    minute["trade_time"] = pd.to_datetime(minute["trade_time"])
    minute = minute[minute["trade_time"].dt.normalize() <= DEVELOPMENT_CUTOFF].copy()
    minute["session_date"] = minute["trade_time"].dt.normalize()
    minute["clock"] = minute["trade_time"].dt.strftime("%H:%M:%S")
    rows: list[dict[str, Any]] = []
    for session_date, group in minute.groupby("session_date", sort=True):
        group = group.sort_values("trade_time")
        signal = group[group["clock"] == SIGNAL_TIME]
        if signal.empty:
            continue
        signal_close = float(signal.iloc[-1]["close"])
        rows.append(
            {
                "session_date": pd.Timestamp(session_date),
                "index_ret_to_signal": signal_close / float(group.iloc[0]["open"]) - 1.0,
                "index_ret_30m": signal_close / _exact_close(group, "14:20:00") - 1.0,
            }
        )
    return pd.DataFrame(rows)


def _load_breadth_features() -> pd.DataFrame:
    breadth = pd.read_parquet(BREADTH_FILE)
    breadth["trade_time"] = pd.to_datetime(breadth["trade_time"])
    breadth = breadth[breadth["trade_time"].dt.normalize() <= DEVELOPMENT_CUTOFF].copy()
    breadth["session_date"] = breadth["trade_time"].dt.normalize()
    breadth["clock"] = breadth["trade_time"].dt.strftime("%H:%M:%S")
    rows: list[dict[str, Any]] = []
    for session_date, group in breadth.groupby("session_date", sort=True):
        group = group.sort_values("trade_time")
        last = group[group["clock"] == SIGNAL_TIME]
        start = group[group["clock"] == "14:20:00"]
        if last.empty or start.empty:
            continue
        last_row = last.iloc[-1]
        late_start = last_row["trade_time"] - pd.Timedelta(minutes=30)
        late = group[(group["trade_time"] >= late_start) & (group["trade_time"] <= last_row["trade_time"])]
        rows.append(
            {
                "session_date": pd.Timestamp(session_date),
                "breadth_last": float(last_row["top50_breadth"]),
                "weighted_breadth_last": float(last_row["top50_weighted_breadth"]),
                "weighted_minus_equal_last": float(last_row["top50_weighted_breadth"] - last_row["top50_breadth"]),
                "breadth_late_mean": float(late["top50_breadth"].mean()),
                "breadth_late_min": float(late["top50_breadth"].min()),
                "breadth_late_change": float(last_row["top50_breadth"] - start.iloc[-1]["top50_breadth"]),
                "weighted_impulse_late_mean": float(late["top50_weighted_breadth_impulse_5m"].mean()),
                "leader_pressure_late": float(late["leader_return"].mean()),
                "component_coverage": float(last_row["leader_weight_coverage"]),
            }
        )
    return pd.DataFrame(rows)


def _load_global_features(session_dates: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"session_date": pd.to_datetime(session_dates).sort_values().unique()})
    base["session_date"] = base["session_date"].astype("datetime64[ns]")
    one_day_columns: list[str] = []
    for file_stem, output_stem in GLOBAL_MARKETS.items():
        path = GLOBAL_DIRECTORY / f"{file_stem}_daily.parquet"
        market = pd.read_parquet(path)
        market["market_date"] = pd.to_datetime(market["date"]).dt.normalize().astype("datetime64[ns]")
        close_column = "adjusted_close" if "adjusted_close" in market.columns else "close"
        market = market.sort_values("market_date").drop_duplicates("market_date", keep="last")
        market[f"{output_stem}_ret1"] = pd.to_numeric(market[close_column], errors="coerce").pct_change()
        market[f"{output_stem}_ret5"] = pd.to_numeric(market[close_column], errors="coerce").pct_change(5)
        market = market[["market_date", f"{output_stem}_ret1", f"{output_stem}_ret5"]].dropna(
            subset=[f"{output_stem}_ret1"]
        )
        base = pd.merge_asof(
            base.sort_values("session_date"),
            market.sort_values("market_date"),
            left_on="session_date",
            right_on="market_date",
            direction="backward",
            allow_exact_matches=False,
        ).drop(columns=["market_date"])
        one_day_columns.append(f"{output_stem}_ret1")
    base["us_mean_ret1"] = base[["us_sp500_ret1", "us_nasdaq_ret1", "us_dow_ret1"]].mean(axis=1)
    base["eu_mean_ret1"] = base[["eu_dax_ret1", "eu_ftse_ret1", "eu_cac_ret1"]].mean(axis=1)
    base["global_min_ret1"] = base[one_day_columns].min(axis=1)
    return base


def _fit_transform_parameters(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    medians = frame[columns].median()
    scales = frame[columns].std(ddof=0).replace(0.0, 1.0)
    return medians, scales


def _z(frame: pd.DataFrame, column: str, medians: pd.Series, scales: pd.Series) -> pd.Series:
    return (pd.to_numeric(frame[column], errors="coerce").fillna(medians[column]) - medians[column]) / scales[column]


def _build_mechanism_scores(frame: pd.DataFrame, fit: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    required = sorted(
        {
            "etf_ret_to_signal",
            "etf_ret_30m",
            "etf_realized_vol",
            "etf_downside_share",
            "etf_close_location",
            "breadth_last",
            "weighted_minus_equal_last",
            "breadth_late_mean",
            "breadth_late_change",
            "weighted_impulse_late_mean",
            "us_mean_ret1",
            "eu_mean_ret1",
            "global_min_ret1",
        }
    )
    medians, scales = _fit_transform_parameters(fit, required)
    scored = frame.copy()
    scored["score_panic_reaction"] = (
        -_z(scored, "etf_ret_30m", medians, scales)
        + _z(scored, "etf_realized_vol", medians, scales)
        - _z(scored, "breadth_last", medians, scales)
        - _z(scored, "weighted_impulse_late_mean", medians, scales)
    ) / 4.0
    scored["score_masked_fragility"] = (
        _z(scored, "etf_ret_to_signal", medians, scales)
        - _z(scored, "breadth_late_mean", medians, scales)
        + _z(scored, "weighted_minus_equal_last", medians, scales)
        + _z(scored, "etf_realized_vol", medians, scales)
    ) / 4.0
    scored["score_late_failure"] = (
        -_z(scored, "etf_ret_30m", medians, scales)
        - _z(scored, "breadth_late_change", medians, scales)
        - _z(scored, "weighted_impulse_late_mean", medians, scales)
        + _z(scored, "etf_downside_share", medians, scales)
    ) / 4.0
    scored["score_global_fragility"] = (
        -_z(scored, "us_mean_ret1", medians, scales)
        - _z(scored, "eu_mean_ret1", medians, scales)
        - _z(scored, "global_min_ret1", medians, scales)
        + _z(scored, "etf_realized_vol", medians, scales)
        - _z(scored, "breadth_late_mean", medians, scales)
    ) / 5.0
    parameter_summary = {
        **{f"median_{key}": float(value) for key, value in medians.items()},
        **{f"scale_{key}": float(value) for key, value in scales.items()},
    }
    return scored, parameter_summary


def _fit_logistic_score(frame: pd.DataFrame, fit_mask: pd.Series, bad_threshold: float) -> tuple[pd.Series, dict[str, Any]]:
    fit = frame.loc[fit_mask].copy()
    valid_fit = fit[MODEL_FEATURES + ["next_return"]].dropna()
    labels = (valid_fit["next_return"] <= bad_threshold).astype(int)
    if labels.nunique() < 2:
        raise RuntimeError("开发拟合段严重亏损标签只有一个类别。")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    penalty="l2",
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(valid_fit[MODEL_FEATURES], labels)
    probabilities = pd.Series(np.nan, index=frame.index, dtype=float)
    valid_all = frame[MODEL_FEATURES].notna().all(axis=1)
    probabilities.loc[valid_all] = model.predict_proba(frame.loc[valid_all, MODEL_FEATURES])[:, 1]
    coefficients = model.named_steps["model"].coef_[0]
    summary = {
        "fit_rows": int(len(valid_fit)),
        "fit_bad_rows": int(labels.sum()),
        "bad_threshold": float(bad_threshold),
        "features": MODEL_FEATURES,
        "standardized_coefficients": {
            feature: float(coefficient) for feature, coefficient in zip(MODEL_FEATURES, coefficients, strict=True)
        },
    }
    return probabilities, summary


def _strategy_metrics(frame: pd.DataFrame, score_column: str, quantile: float, fit: pd.DataFrame) -> dict[str, Any]:
    threshold = float(fit[score_column].dropna().quantile(quantile))
    evaluation = frame[frame["session_date"] >= SELECTION_START].copy()
    evaluation = evaluation[evaluation[[score_column, "next_return"]].notna().all(axis=1)]
    evaluation["cash"] = evaluation[score_column] >= threshold
    position = (~evaluation["cash"]).astype(int)
    transitions = position.diff().abs().fillna(0.0)
    cash_daily = (1.0 + CASH_ANNUAL_RATE) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    gross = np.where(position.eq(1), evaluation["next_return"], cash_daily)
    base_returns = pd.Series(gross, index=evaluation.index) - transitions * BASE_ONE_WAY_COST
    stress_returns = pd.Series(gross, index=evaluation.index) - transitions * STRESS_ONE_WAY_COST
    benchmark_cagr = _cagr(evaluation["next_return"])
    base_cagr = _cagr(base_returns)
    stress_cagr = _cagr(stress_returns)
    bad_label = evaluation["next_return"] <= float(fit["next_return"].quantile(0.10))
    bad_losses = -evaluation.loc[bad_label, "next_return"].clip(upper=0.0).sum()
    captured_losses = -evaluation.loc[bad_label & evaluation["cash"], "next_return"].clip(upper=0.0).sum()
    score_values = evaluation[score_column]
    labels = bad_label.astype(int)
    auc = roc_auc_score(labels, score_values) if labels.nunique() == 2 else float("nan")
    average_precision = average_precision_score(labels, score_values) if labels.nunique() == 2 else float("nan")
    return {
        "score": score_column,
        "risk_quantile": quantile,
        "threshold_from_2021_2022": threshold,
        "evaluation_start": evaluation["session_date"].min().date().isoformat(),
        "evaluation_end": evaluation["session_date"].max().date().isoformat(),
        "evaluation_rows": int(len(evaluation)),
        "cash_days": int(evaluation["cash"].sum()),
        "cash_share": float(evaluation["cash"].mean()),
        "trade_legs": int(transitions.sum()),
        "benchmark_cagr": _safe_float(benchmark_cagr),
        "base_strategy_cagr": _safe_float(base_cagr),
        "stress_strategy_cagr": _safe_float(stress_cagr),
        "base_annualized_excess_vs_510300": _safe_float(base_cagr - benchmark_cagr),
        "stress_annualized_excess_vs_510300": _safe_float(stress_cagr - benchmark_cagr),
        "bad_day_recall": _safe_float(evaluation.loc[bad_label, "cash"].mean()),
        "good_day_false_exit": _safe_float(evaluation.loc[~bad_label, "cash"].mean()),
        "bad_loss_capture": _safe_float(captured_losses / bad_losses if bad_losses > 0 else float("nan")),
        "cash_day_mean_target_return": _safe_float(evaluation.loc[evaluation["cash"], "next_return"].mean()),
        "cash_day_precision_bad": _safe_float(labels[evaluation["cash"]].mean()),
        "bad_day_auc": _safe_float(auc),
        "bad_day_average_precision": _safe_float(average_precision),
    }


def main() -> int:
    required = [ETF_FILE, INDEX_FILE, BREADTH_FILE]
    required.extend(GLOBAL_DIRECTORY / f"{stem}_daily.parquet" for stem in GLOBAL_MARKETS)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    etf = _load_etf_daily_features()
    index = _load_index_features()
    breadth = _load_breadth_features()
    frame = etf.merge(index, on="session_date", how="inner").merge(breadth, on="session_date", how="inner")
    frame["index_relative_ret_to_signal"] = frame["index_ret_to_signal"] - frame["etf_ret_to_signal"]
    global_features = _load_global_features(frame["session_date"])
    frame = frame.merge(global_features, on="session_date", how="left")
    frame = frame.sort_values("session_date").reset_index(drop=True)
    if frame["session_date"].max() > DEVELOPMENT_CUTOFF:
        raise RuntimeError("发现扫描读取了2023-12-29之后的数据。")

    fit_mask = frame["session_date"] <= FIT_END
    selection_mask = frame["session_date"] >= SELECTION_START
    fit = frame.loc[fit_mask].copy()
    if len(fit) < 300 or int(selection_mask.sum()) < 150:
        raise RuntimeError("开发拟合段或2023选择段样本不足。")
    frame, transform_parameters = _build_mechanism_scores(frame, fit)
    fit = frame.loc[fit_mask].copy()
    bad_threshold = float(fit["next_return"].quantile(0.10))
    frame["score_logistic_tail"], model_summary = _fit_logistic_score(frame, fit_mask, bad_threshold)
    fit = frame.loc[fit_mask].copy()

    score_columns = [
        "score_panic_reaction",
        "score_masked_fragility",
        "score_late_failure",
        "score_global_fragility",
        "score_logistic_tail",
    ]
    metrics: list[dict[str, Any]] = []
    for score_column in score_columns:
        for quantile in (0.80, 0.90, 0.95):
            metrics.append(_strategy_metrics(frame, score_column, quantile, fit))
    metrics.sort(
        key=lambda item: (
            item["base_annualized_excess_vs_510300"]
            if item["base_annualized_excess_vs_510300"] is not None
            else -float("inf")
        ),
        reverse=True,
    )
    best = metrics[0]
    output_columns = [
        "session_date",
        "execution_open",
        "next_execution_open",
        "next_return",
        *MODEL_FEATURES,
        *score_columns,
    ]
    _atomic_parquet(frame[output_columns], OUTPUT_FEATURES)
    payload = {
        "status": "DEVELOPMENT_DISCOVERY_COMPLETE_NO_VALIDATION_READ",
        "project_id": "510300_LATE_DAY_FRAGILITY_DISCOVERY_V0",
        "data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
        "fit_period": [frame.loc[fit_mask, "session_date"].min().date().isoformat(), FIT_END.date().isoformat()],
        "selection_period": [SELECTION_START.date().isoformat(), frame.loc[selection_mask, "session_date"].max().date().isoformat()],
        "signal_time": SIGNAL_TIME,
        "execution_time": EXECUTION_TIME,
        "target": "本日14:51开盘至下一交易日14:51开盘的510300收益",
        "rows": int(len(frame)),
        "fit_rows": int(fit_mask.sum()),
        "selection_rows": int(selection_mask.sum()),
        "fit_bad_return_threshold": bad_threshold,
        "base_one_way_cost": BASE_ONE_WAY_COST,
        "stress_one_way_cost": STRESS_ONE_WAY_COST,
        "best_development_selection": best,
        "all_metrics": metrics,
        "model": model_summary,
        "fixed_transform_parameters": transform_parameters,
        "validation_2024_loaded": False,
        "replication_2025_plus_loaded": False,
        "output_features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)),
        "interpretation": "这是候选发现，不是通过结论。只有先冻结候选后才允许读取2024验证。",
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
