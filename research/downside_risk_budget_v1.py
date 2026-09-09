"""510300下行风险预算V1的一次性研究。

先检验技术因子是否能稳定预测未来20日下行风险。只有全部风险门通过后，
才允许构建510300/现金仓位并读取净费后夏普率。历史通过也不授权Paper、
Shadow、订单、券商连接或实盘。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_downside_risk_budget_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_downside_risk_budget_v1_manifest.json"


class ContractError(RuntimeError):
    """冻结协议、清单或输入数据不满足要求。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise ContractError(f"缺少协议文件：{path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ContractError("协议文件必须是YAML对象")
    validate_config(payload)
    return payload


def validate_config(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    scope = config["scope"]
    dates = config["dates"]
    features = config["features"]
    portfolio = config["portfolio"]
    governance = config["governance"]
    if protocol["project_id"] != "510300_DOWNSIDE_RISK_BUDGET_V1":
        raise ContractError("project_id不符合冻结候选")
    if protocol["one_shot"] is not True:
        raise ContractError("研究必须声明one_shot=true")
    if protocol["historical_data_already_contaminated"] is not True:
        raise ContractError("必须承认历史数据已经被观察")
    if scope["execution_asset"] != "510300.SH":
        raise ContractError("执行资产必须固定为510300.SH")
    if scope["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ContractError("允许持有范围必须仅为510300与现金")
    if any(
        [
            scope["leverage_allowed"],
            scope["short_selling_allowed"],
            scope["derivatives_execution_allowed"],
            scope["live_trading_authorized"],
        ]
    ):
        raise ContractError("禁止杠杆、卖空、衍生品执行和实盘")
    if dates["evaluation_start"] != "2015-01-05":
        raise ContractError("评价起点必须固定为2015-01-05")
    if dates["evaluation_end"] != "2026-08-12":
        raise ContractError("评价终点必须固定为2026-08-12")
    if int(features["causal_percentile"]["prior_observation_lookback"]) != 504:
        raise ContractError("因果分位参考窗必须固定为504个既往观测")
    weights = features["composite_risk_score"]["components"]
    if set(weights) != {
        "slow_trend_risk_percentile",
        "downside_variance_ratio_percentile",
        "gap_atr_percentile",
        "vol_of_vol_percentile",
    }:
        raise ContractError("复合风险分数的四个组件不得改变")
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0):
        raise ContractError("复合风险分数组件权重之和必须为1")
    if any(not math.isclose(float(value), 0.25) for value in weights.values()):
        raise ContractError("第一版四个风险组件必须严格等权")
    if float(portfolio["initial_capital_cny"]) != 20000.0:
        raise ContractError("初始资金必须固定为20000元")
    if int(portfolio["lot_size_shares"]) != 100:
        raise ContractError("交易单位必须固定为100份")
    if governance["order_generation"] or governance["broker_connection"]:
        raise ContractError("研究协议禁止订单生成和券商连接")
    if governance["position_change"] or governance["live_trading_authorized"]:
        raise ContractError("研究协议禁止真实仓位变化和实盘")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("缺少冻结清单；禁止在冻结前读取候选结果")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_FIRST_RESULT":
        raise ContractError("冻结清单状态无效")
    if manifest.get("result_preexisted_at_freeze") is not False:
        raise ContractError("冻结时不得已经存在结果")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("冻结后协议哈希发生变化")
    mismatches: dict[str, dict[str, str]] = {}
    for section in ("tracked_files", "input_files"):
        for relative, expected in manifest.get(section, {}).items():
            path = _project_path(relative)
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ContractError(f"冻结文件发生漂移：{mismatches}")
    selection = manifest.get("selection_bias_control", {})
    if int(selection.get("total_trial_count_including_current", 0)) < 1:
        raise ContractError("冻结清单缺少选择偏差试验计数")
    return manifest


def load_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    specifications = config["inputs"]
    for name, specification in specifications.items():
        path = _project_path(specification["path"])
        if not path.exists():
            raise ContractError(f"缺少输入{name}：{path}")

    audit = json.loads(
        _project_path(specifications["input_audit"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if audit.get("status") != specifications["input_audit"]["required_status"]:
        raise ContractError("输入质量报告未通过")
    if audit.get("candidate_outcomes_computed") is not False:
        raise ContractError("输入审计阶段不得读取候选未来结果")
    if audit.get("portfolio_returns_computed") is not False:
        raise ContractError("输入审计阶段不得读取组合收益")

    market = pd.read_parquet(specifications["etf_daily"]["path"])
    benchmark = pd.read_parquet(specifications["benchmark_total_return"]["path"])
    dividends = pd.read_csv(specifications["dividends"]["path"])
    required_market = set(specifications["etf_daily"]["required_columns"])
    required_benchmark = set(
        specifications["benchmark_total_return"]["required_columns"]
    )
    required_dividends = set(specifications["dividends"]["required_columns"])
    for name, frame, required in (
        ("etf_daily", market, required_market),
        ("benchmark_total_return", benchmark, required_benchmark),
        ("dividends", dividends, required_dividends),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ContractError(f"{name}缺少字段：{sorted(missing)}")

    market = market.copy()
    benchmark = benchmark.copy()
    dividends = dividends.copy()
    market["date"] = pd.to_datetime(market["date"])
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    for field in ("record_date", "ex_date", "payment_date"):
        dividends[field] = pd.to_datetime(dividends[field])
    market = market.sort_values("date").reset_index(drop=True)
    benchmark = benchmark.sort_values("date").reset_index(drop=True)
    dividends = dividends.sort_values("ex_date").reset_index(drop=True)
    if market["date"].duplicated().any() or benchmark["date"].duplicated().any():
        raise ContractError("行情或基准日期重复")
    if len(market) != int(config["data_contract"]["expected_raw_rows"]):
        raise ContractError("行情原始行数不符合冻结契约")
    if market["date"].min() != pd.Timestamp(
        config["data_contract"]["expected_first_date"]
    ) or market["date"].max() != pd.Timestamp(
        config["data_contract"]["expected_last_date"]
    ):
        raise ContractError("行情起止日期不符合冻结契约")
    numeric = market[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or (numeric <= 0).any().any():
        raise ContractError("行情价格存在空值、零值或负值")
    market[["open", "high", "low", "close"]] = numeric
    if (
        (market["high"] < market[["open", "close"]].max(axis=1))
        | (market["low"] > market[["open", "close"]].min(axis=1))
        | (market["high"] < market["low"])
    ).any():
        raise ContractError("行情违反OHLC约束")
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="coerce"
    )
    if dividends["cash_dividend_per_share"].isna().any() or (
        dividends["cash_dividend_per_share"] <= 0
    ).any():
        raise ContractError("分红金额必须为正数")
    return market, dividends, benchmark, audit


def causal_total_return_series(
    market: pd.DataFrame, dividends: pd.DataFrame
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回因果总收益率、总收益指数和除息日现金分红。"""

    dividend_by_ex_date = (
        dividends.groupby("ex_date", sort=False)["cash_dividend_per_share"].sum()
    )
    cash_dividend = market["date"].map(dividend_by_ex_date).fillna(0.0).astype(float)
    previous_close = market["close"].shift(1)
    total_return = (market["close"] + cash_dividend) / previous_close - 1.0
    total_return.iloc[0] = 0.0
    if (total_return <= -1).any() or not np.isfinite(total_return).all():
        raise ContractError("因果总收益率包含无效值")
    total_return_index = (1.0 + total_return).cumprod()
    return total_return.astype(float), total_return_index.astype(float), cash_dividend


def rolling_slope_t_stat(series: pd.Series, lookback: int) -> pd.Series:
    """计算滚动线性趋势斜率的t统计量。"""

    x = np.arange(lookback, dtype=float)
    x_centered = x - x.mean()
    sxx = float(np.dot(x_centered, x_centered))

    def calculate(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        y_centered = values - values.mean()
        beta = float(np.dot(x_centered, y_centered) / sxx)
        residual = y_centered - beta * x_centered
        residual_variance = float(np.dot(residual, residual) / (lookback - 2))
        if residual_variance <= 0:
            return 0.0
        standard_error = math.sqrt(residual_variance / sxx)
        return beta / standard_error if standard_error > 0 else 0.0

    return series.rolling(lookback, min_periods=lookback).apply(calculate, raw=True)


def causal_percentile(series: pd.Series, prior_lookback: int) -> pd.Series:
    """以严格位于当前观测之前的固定窗口计算经验分位。"""

    def calculate(values: np.ndarray) -> float:
        current = values[-1]
        history = values[:-1]
        if not np.isfinite(current) or not np.isfinite(history).all():
            return np.nan
        return float(np.mean(history <= current))

    return series.rolling(
        prior_lookback + 1, min_periods=prior_lookback + 1
    ).apply(calculate, raw=True)


def _structural_period(date: pd.Timestamp, config: dict[str, Any]) -> str:
    for label, bounds in config["dates"]["structural_periods"].items():
        if pd.Timestamp(bounds["start"]) <= date <= pd.Timestamp(bounds["end"]):
            return str(label)
    return "OUTSIDE"


def build_daily_features(
    market: pd.DataFrame, dividends: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    frame = market.copy()
    total_return, total_return_index, cash_dividend = causal_total_return_series(
        frame, dividends
    )
    frame["cash_dividend_on_ex_date"] = cash_dividend
    frame["causal_total_return"] = total_return
    frame["causal_total_return_index"] = total_return_index

    annualization = int(config["features"]["annualization_trading_days"])
    slow_lookback = int(config["features"]["slow_trend"]["lookback_trading_days"])
    frame["slow_trend_120_tstat"] = rolling_slope_t_stat(
        np.log(frame["causal_total_return_index"]), slow_lookback
    )

    negative_squared = np.square(np.minimum(frame["causal_total_return"], 0.0))
    downside_spec = config["features"]["downside_variance"]
    short_span = int(downside_spec["short_span"])
    long_span = int(downside_spec["long_span"])
    down_short = pd.Series(negative_squared).ewm(
        span=short_span,
        adjust=bool(downside_spec["adjust"]),
        min_periods=short_span,
    ).mean()
    down_long = pd.Series(negative_squared).ewm(
        span=long_span,
        adjust=bool(downside_spec["adjust"]),
        min_periods=long_span,
    ).mean()
    frame["downside_rms_5"] = np.sqrt(down_short)
    frame["downside_rms_20"] = np.sqrt(down_long)
    frame["downvar_ratio_5_20"] = np.divide(
        frame["downside_rms_5"],
        frame["downside_rms_20"],
        out=np.zeros(len(frame), dtype=float),
        where=frame["downside_rms_20"].to_numpy(dtype=float) > 0,
    )
    frame.loc[frame["downside_rms_20"].isna(), "downvar_ratio_5_20"] = np.nan

    previous_close = frame["close"].shift(1)
    adjusted_open = frame["open"] + frame["cash_dividend_on_ex_date"]
    adjusted_high = frame["high"] + frame["cash_dividend_on_ex_date"]
    adjusted_low = frame["low"] + frame["cash_dividend_on_ex_date"]
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (adjusted_high - previous_close).abs(),
            (adjusted_low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    normalized_true_range = true_range / previous_close
    atr_lookback = int(
        config["features"]["gap_risk"]["atr_lookback_trading_days"]
    )
    atr_lag = int(config["features"]["gap_risk"]["atr_lag_trading_days"])
    frame["atr20_normalized_lag1"] = normalized_true_range.rolling(
        atr_lookback, min_periods=atr_lookback
    ).mean().shift(atr_lag)
    frame["absolute_overnight_gap"] = (
        adjusted_open / previous_close - 1.0
    ).abs()
    frame["gap_atr_20"] = frame["absolute_overnight_gap"] / frame[
        "atr20_normalized_lag1"
    ]

    rv_lookback = int(
        config["features"]["realized_volatility"]["lookback_trading_days"]
    )
    frame["rv20"] = np.sqrt(
        frame["causal_total_return"].pow(2).rolling(
            rv_lookback, min_periods=rv_lookback
        ).mean()
        * annualization
    )
    vov_lookback = int(
        config["features"]["volatility_of_volatility"]["lookback_trading_days"]
    )
    rv_mean = frame["rv20"].rolling(vov_lookback, min_periods=vov_lookback).mean()
    rv_std = frame["rv20"].rolling(vov_lookback, min_periods=vov_lookback).std(
        ddof=1
    )
    frame["vol_of_vol_20"] = rv_std / rv_mean

    rank_lookback = int(
        config["features"]["causal_percentile"]["prior_observation_lookback"]
    )
    frame["slow_trend_risk_percentile"] = causal_percentile(
        -frame["slow_trend_120_tstat"], rank_lookback
    )
    frame["downside_variance_ratio_percentile"] = causal_percentile(
        frame["downvar_ratio_5_20"], rank_lookback
    )
    frame["gap_atr_percentile"] = causal_percentile(
        frame["gap_atr_20"], rank_lookback
    )
    frame["vol_of_vol_percentile"] = causal_percentile(
        frame["vol_of_vol_20"], rank_lookback
    )
    frame["rv20_risk_percentile"] = causal_percentile(frame["rv20"], rank_lookback)
    components = list(
        config["features"]["composite_risk_score"]["components"].keys()
    )
    frame["composite_risk_score"] = frame[components].mean(axis=1, skipna=False)

    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    evaluation = frame.loc[frame["date"].between(start, end)].copy().reset_index(
        drop=True
    )
    if len(evaluation) != int(config["data_contract"]["expected_evaluation_rows"]):
        raise ContractError("评价区间交易日行数不符合冻结契约")
    required_features = components + [
        "rv20_risk_percentile",
        "composite_risk_score",
        "rv20",
    ]
    if evaluation[required_features].isna().any().any():
        missing = evaluation[required_features].isna().sum()
        raise ContractError(f"评价期因子存在空值：{missing[missing > 0].to_dict()}")
    evaluation["structural_period"] = evaluation["date"].map(
        lambda value: _structural_period(pd.Timestamp(value), config)
    )
    if set(evaluation["structural_period"]) != {"EARLY", "MIDDLE", "LATE"}:
        raise ContractError("结构时期覆盖不完整")
    return evaluation


def add_forward_risk_targets(
    features: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """从下一交易日开盘构造固定20日未来风险标签。"""

    horizon = int(config["targets"]["primary_horizon_trading_days"])
    output = features.copy()
    market = market.sort_values("date").reset_index(drop=True)
    date_to_index = {pd.Timestamp(date): index for index, date in enumerate(market["date"])}
    dividend_by_record = (
        dividends.groupby("record_date", sort=False)["cash_dividend_per_share"].sum()
    )
    dividend_on_record = market["date"].map(dividend_by_record).fillna(0.0).to_numpy(
        dtype=float
    )
    cumulative_dividend = np.cumsum(dividend_on_record)
    opens = market["open"].to_numpy(dtype=float)
    closes = market["close"].to_numpy(dtype=float)
    market_dates = pd.to_datetime(market["date"]).to_numpy()

    mae_values: list[float] = []
    risk_values: list[float] = []
    downside_values: list[float] = []
    terminal_values: list[float] = []
    tail_values: list[float] = []
    entry_dates: list[pd.Timestamp | pd.NaT] = []
    end_dates: list[pd.Timestamp | pd.NaT] = []
    multiple = float(config["targets"]["tail_threshold"]["multiple"])
    annualization = float(config["features"]["annualization_trading_days"])
    for row in output.itertuples(index=False):
        signal_index = date_to_index[pd.Timestamp(row.date)]
        entry_index = signal_index + 1
        end_index = signal_index + horizon
        if end_index >= len(market):
            mae_values.append(np.nan)
            risk_values.append(np.nan)
            downside_values.append(np.nan)
            terminal_values.append(np.nan)
            tail_values.append(np.nan)
            entry_dates.append(pd.NaT)
            end_dates.append(pd.NaT)
            continue
        prior_cumulative = cumulative_dividend[entry_index - 1] if entry_index > 0 else 0.0
        entitled = cumulative_dividend[entry_index : end_index + 1] - prior_cumulative
        path_wealth = closes[entry_index : end_index + 1] + entitled
        entry_open = opens[entry_index]
        path_return = path_wealth / entry_open - 1.0
        mae = float(np.min(path_return))
        wealth_with_entry = np.concatenate(([entry_open], path_wealth))
        daily_returns = wealth_with_entry[1:] / wealth_with_entry[:-1] - 1.0
        downside_semivariance = float(np.mean(np.minimum(daily_returns, 0.0) ** 2))
        rv_daily = float(row.rv20) / math.sqrt(annualization)
        threshold = multiple * rv_daily * math.sqrt(horizon)
        mae_values.append(mae)
        risk_values.append(max(-mae, 0.0))
        downside_values.append(downside_semivariance)
        terminal_values.append(float(path_return[-1]))
        tail_values.append(float(mae <= -threshold))
        entry_dates.append(pd.Timestamp(market_dates[entry_index]))
        end_dates.append(pd.Timestamp(market_dates[end_index]))

    output["future_mae_20d"] = mae_values
    output["future_mae_magnitude_20d"] = risk_values
    output["future_downside_semivariance_20d"] = downside_values
    output["future_terminal_return_20d"] = terminal_values
    output["future_tail_event_20d"] = tail_values
    output["future_entry_date"] = pd.to_datetime(entry_dates)
    output["future_end_date"] = pd.to_datetime(end_dates)
    return output


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return None
    left_rank = pd.Series(left[mask]).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right[mask]).rank(method="average").to_numpy(dtype=float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _circular_block_indices(
    length: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    block_count = math.ceil(length / block_length)
    starts = rng.integers(0, length, size=block_count)
    indices = np.concatenate(
        [
            (start + np.arange(block_length, dtype=int)) % length
            for start in starts
        ]
    )
    return indices[:length]


def _confidence_interval(values: list[float], confidence: float) -> list[float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return [None, None]
    alpha = 1.0 - confidence
    return [
        float(np.quantile(array, alpha / 2.0)),
        float(np.quantile(array, 1.0 - alpha / 2.0)),
    ]


def bootstrap_risk_metrics(
    valid: pd.DataFrame, config: dict[str, Any]
) -> dict[str, list[float | None]]:
    bootstrap = config["bootstrap"]
    repetitions = int(bootstrap["repetitions"])
    block_length = int(bootstrap["block_length_trading_days"])
    confidence = float(bootstrap["confidence_level"])
    rng = np.random.default_rng(int(bootstrap["random_seed"]))
    risk_score = valid["composite_risk_score"].to_numpy(dtype=float)
    baseline_score = valid["rv20_risk_percentile"].to_numpy(dtype=float)
    target = valid["future_mae_magnitude_20d"].to_numpy(dtype=float)
    low_threshold = float(config["risk_groups"]["low_risk_maximum_score"])
    high_threshold = float(config["risk_groups"]["high_risk_minimum_score"])
    risk_rank = pd.Series(risk_score).rank(method="average").to_numpy(dtype=float)
    baseline_rank = pd.Series(baseline_score).rank(method="average").to_numpy(
        dtype=float
    )
    target_rank = pd.Series(target).rank(method="average").to_numpy(dtype=float)
    mean_differences: list[float] = []
    rank_ics: list[float] = []
    incremental_rank_ics: list[float] = []
    for _ in range(repetitions):
        indices = _circular_block_indices(len(valid), block_length, rng)
        sampled_score = risk_score[indices]
        sampled_target = target[indices]
        high = sampled_score >= high_threshold
        low = sampled_score <= low_threshold
        if high.any() and low.any():
            mean_differences.append(
                float(sampled_target[high].mean() - sampled_target[low].mean())
            )
        rank_ic = np.corrcoef(risk_rank[indices], target_rank[indices])[0, 1]
        baseline_ic = np.corrcoef(
            baseline_rank[indices], target_rank[indices]
        )[0, 1]
        rank_ics.append(float(rank_ic))
        incremental_rank_ics.append(float(rank_ic - baseline_ic))
    return {
        "high_minus_low_mean_mae_magnitude": _confidence_interval(
            mean_differences, confidence
        ),
        "composite_rank_ic": _confidence_interval(rank_ics, confidence),
        "incremental_rank_ic_vs_rv20": _confidence_interval(
            incremental_rank_ics, confidence
        ),
    }


def _group_comparison(
    frame: pd.DataFrame, score_column: str, config: dict[str, Any]
) -> dict[str, Any]:
    score = frame[score_column]
    target = frame["future_mae_magnitude_20d"]
    tail = frame["future_tail_event_20d"]
    low = score <= float(config["risk_groups"]["low_risk_maximum_score"])
    high = score >= float(config["risk_groups"]["high_risk_minimum_score"])
    low_values = target.loc[low].to_numpy(dtype=float)
    high_values = target.loc[high].to_numpy(dtype=float)
    low_tail = tail.loc[low].to_numpy(dtype=float)
    high_tail = tail.loc[high].to_numpy(dtype=float)
    return {
        "score_column": score_column,
        "low_count": int(len(low_values)),
        "high_count": int(len(high_values)),
        "low_mean_mae_magnitude": float(low_values.mean()) if len(low_values) else None,
        "high_mean_mae_magnitude": float(high_values.mean()) if len(high_values) else None,
        "high_minus_low_mean_mae_magnitude": (
            float(high_values.mean() - low_values.mean())
            if len(high_values) and len(low_values)
            else None
        ),
        "low_median_mae_magnitude": (
            float(np.median(low_values)) if len(low_values) else None
        ),
        "high_median_mae_magnitude": (
            float(np.median(high_values)) if len(high_values) else None
        ),
        "high_minus_low_median_mae_magnitude": (
            float(np.median(high_values) - np.median(low_values))
            if len(high_values) and len(low_values)
            else None
        ),
        "low_tail_rate": float(low_tail.mean()) if len(low_tail) else None,
        "high_tail_rate": float(high_tail.mean()) if len(high_tail) else None,
        "high_minus_low_tail_rate": (
            float(high_tail.mean() - low_tail.mean())
            if len(high_tail) and len(low_tail)
            else None
        ),
    }


def _positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def evaluate_risk_gate(
    features: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    required = [
        "composite_risk_score",
        "rv20_risk_percentile",
        "future_mae_magnitude_20d",
        "future_tail_event_20d",
    ]
    valid = features.dropna(subset=required).copy().reset_index(drop=True)
    composite = _group_comparison(valid, "composite_risk_score", config)
    baseline = _group_comparison(valid, "rv20_risk_percentile", config)
    composite_ic = _rank_correlation(
        valid["composite_risk_score"].to_numpy(dtype=float),
        valid["future_mae_magnitude_20d"].to_numpy(dtype=float),
    )
    baseline_ic = _rank_correlation(
        valid["rv20_risk_percentile"].to_numpy(dtype=float),
        valid["future_mae_magnitude_20d"].to_numpy(dtype=float),
    )
    incremental_ic = (
        composite_ic - baseline_ic
        if composite_ic is not None and baseline_ic is not None
        else None
    )
    bootstrap = bootstrap_risk_metrics(valid, config)

    structural: dict[str, Any] = {}
    for period in ("EARLY", "MIDDLE", "LATE"):
        subset = valid.loc[valid["structural_period"] == period]
        structural[period] = _group_comparison(
            subset, "composite_risk_score", config
        )

    leave_one_year_out: dict[str, Any] = {}
    for year in sorted(valid["date"].dt.year.unique()):
        subset = valid.loc[valid["date"].dt.year != int(year)]
        comparison = _group_comparison(subset, "composite_risk_score", config)
        leave_one_year_out[str(int(year))] = {
            "remaining_rows": int(len(subset)),
            "high_minus_low_mean_mae_magnitude": comparison[
                "high_minus_low_mean_mae_magnitude"
            ],
        }

    factor_columns = [
        "slow_trend_risk_percentile",
        "downside_variance_ratio_percentile",
        "gap_atr_percentile",
        "vol_of_vol_percentile",
    ]
    factor_rank_ics = {
        column: _rank_correlation(
            valid[column].to_numpy(dtype=float),
            valid["future_mae_magnitude_20d"].to_numpy(dtype=float),
        )
        for column in factor_columns
    }
    fixed_bins = pd.cut(
        valid["composite_risk_score"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
        right=True,
    )
    quintile_diagnostics: list[dict[str, Any]] = []
    for label, subset in valid.groupby(fixed_bins, observed=False):
        values = subset["future_mae_magnitude_20d"].dropna().to_numpy(dtype=float)
        quintile_diagnostics.append(
            {
                "fixed_score_interval": str(label),
                "count": int(len(values)),
                "mean_mae_magnitude": float(values.mean()) if len(values) else None,
                "median_mae_magnitude": (
                    float(np.median(values)) if len(values) else None
                ),
            }
        )

    minimum_count = int(config["risk_gates"]["minimum_group_observations_each"])
    minimum_tail_difference = float(
        config["risk_gates"]["high_minus_low_tail_rate_minimum_difference"]
    )
    gates = {
        "minimum_valid_target_rows": len(valid)
        >= int(config["risk_gates"]["minimum_valid_target_rows"]),
        "minimum_group_observations_each": (
            composite["low_count"] >= minimum_count
            and composite["high_count"] >= minimum_count
        ),
        "high_minus_low_mean_mae_magnitude_positive": _positive(
            composite["high_minus_low_mean_mae_magnitude"]
        ),
        "high_minus_low_median_mae_magnitude_positive": _positive(
            composite["high_minus_low_median_mae_magnitude"]
        ),
        "high_minus_low_bootstrap_90pct_lower_positive": _positive(
            bootstrap["high_minus_low_mean_mae_magnitude"][0]
        ),
        "high_minus_low_tail_rate_at_least_5pp": (
            composite["high_minus_low_tail_rate"] is not None
            and composite["high_minus_low_tail_rate"] >= minimum_tail_difference
        ),
        "composite_rank_ic_positive": _positive(composite_ic),
        "composite_rank_ic_bootstrap_90pct_lower_positive": _positive(
            bootstrap["composite_rank_ic"][0]
        ),
        "all_structural_period_mean_differences_positive": all(
            _positive(
                structural[period]["high_minus_low_mean_mae_magnitude"]
            )
            for period in ("EARLY", "MIDDLE", "LATE")
        ),
        "all_leave_one_calendar_year_out_mean_differences_positive": all(
            _positive(payload["high_minus_low_mean_mae_magnitude"])
            for payload in leave_one_year_out.values()
        ),
        "incremental_rank_ic_vs_rv20_positive": _positive(incremental_ic),
        "incremental_rank_ic_bootstrap_90pct_lower_positive": _positive(
            bootstrap["incremental_rank_ic_vs_rv20"][0]
        ),
    }
    passed = all(gates.values())
    return {
        "passed": passed,
        "status": (
            config["adjudication"]["risk_pass_status"]
            if passed
            else config["adjudication"]["risk_fail_status"]
        ),
        "valid_target_rows": int(len(valid)),
        "composite_group_comparison": composite,
        "rv20_baseline_group_comparison": baseline,
        "composite_rank_ic": composite_ic,
        "rv20_baseline_rank_ic": baseline_ic,
        "incremental_rank_ic_vs_rv20": incremental_ic,
        "bootstrap_90pct_intervals": bootstrap,
        "structural_periods": structural,
        "leave_one_calendar_year_out": leave_one_year_out,
        "factor_rank_ics": factor_rank_ics,
        "fixed_score_bin_diagnostics": quintile_diagnostics,
        "gates": gates,
    }


def build_portfolio_targets(
    features: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """按冻结的单调映射生成研究用目标仓位，不生成订单。"""

    maximum_step = float(config["portfolio"]["maximum_position_change_per_trading_day"])
    previous_target = float(config["portfolio"]["initial_position"])
    rows: list[dict[str, Any]] = []
    for row in features.itertuples(index=False):
        score = float(row.composite_risk_score)
        if not math.isfinite(score):
            desired = previous_target
            reason = "风险分数缺失，维持上一研究目标"
        elif score >= 0.80:
            desired = 0.0
            reason = "复合风险分数不低于0.80"
        elif score >= 0.60:
            desired = 0.5
            reason = "复合风险分数位于0.60至0.80之间"
        else:
            desired = 1.0
            reason = "复合风险分数低于0.60"
        change = float(np.clip(desired - previous_target, -maximum_step, maximum_step))
        target = float(np.clip(previous_target + change, 0.0, 1.0))
        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "composite_risk_score": score,
                "desired_position": desired,
                "target_position": target,
                "trade_allowed": True,
                "risk_off_override": desired == 0.0,
                "signal_reason": reason,
            }
        )
        previous_target = target
    return pd.DataFrame(rows)


def _benchmark_cagr(
    benchmark: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> float:
    subset = benchmark.loc[benchmark["date"].between(start_date, end_date)].copy()
    if len(subset) < 2:
        raise ContractError("基准区间不足两行")
    elapsed_days = max((subset["date"].iloc[-1] - subset["date"].iloc[0]).days, 1)
    total_return = float(subset["close"].iloc[-1] / subset["close"].iloc[0] - 1.0)
    return float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)


def probabilistic_sharpe_probability(
    returns: pd.Series, benchmark_annual_sharpe: float
) -> float | None:
    values = returns.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return None
    daily_sharpe = float(np.mean(values) / np.std(values, ddof=1))
    benchmark_daily = float(benchmark_annual_sharpe) / math.sqrt(242.0)
    skewness = float(pd.Series(values).skew())
    raw_kurtosis = float(pd.Series(values).kurt()) + 3.0
    denominator_squared = (
        1.0
        - skewness * daily_sharpe
        + ((raw_kurtosis - 1.0) / 4.0) * daily_sharpe**2
    )
    if denominator_squared <= 0:
        return None
    z_score = (
        (daily_sharpe - benchmark_daily)
        * math.sqrt(len(values) - 1)
        / math.sqrt(denominator_squared)
    )
    return float(NormalDist().cdf(z_score))


def deflated_sharpe_probability(
    returns: pd.Series, total_trial_count: int
) -> dict[str, float | int | None]:
    values = returns.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return {
            "total_trial_count": int(total_trial_count),
            "expected_maximum_null_sharpe_annualized": None,
            "probability": None,
        }
    trial_count = max(int(total_trial_count), 1)
    if trial_count == 1:
        expected_maximum_daily = 0.0
    else:
        gamma = 0.5772156649015329
        normal = NormalDist()
        trial_std_daily = 1.0 / math.sqrt(len(values) - 1)
        expected_maximum_daily = trial_std_daily * (
            (1.0 - gamma) * normal.inv_cdf(1.0 - 1.0 / trial_count)
            + gamma * normal.inv_cdf(1.0 - 1.0 / (trial_count * math.e))
        )
    probability = probabilistic_sharpe_probability(
        pd.Series(values), expected_maximum_daily * math.sqrt(242.0)
    )
    return {
        "total_trial_count": trial_count,
        "expected_maximum_null_sharpe_annualized": float(
            expected_maximum_daily * math.sqrt(242.0)
        ),
        "probability": probability,
    }


def _annual_contribution_analysis(
    strategy_ledger: pd.DataFrame, buy_hold_ledger: pd.DataFrame
) -> dict[str, Any]:
    merged = strategy_ledger[["date", "daily_return"]].merge(
        buy_hold_ledger[["date", "daily_return"]],
        on="date",
        suffixes=("_strategy", "_buy_hold"),
        validate="one_to_one",
    )
    merged["year"] = merged["date"].dt.year
    annual_rows: list[dict[str, Any]] = []
    for year, subset in merged.groupby("year", sort=True):
        strategy_return = float((1.0 + subset["daily_return_strategy"]).prod() - 1.0)
        buy_hold_return = float((1.0 + subset["daily_return_buy_hold"]).prod() - 1.0)
        annual_rows.append(
            {
                "year": int(year),
                "strategy_return": strategy_return,
                "buy_hold_return": buy_hold_return,
                "excess_return": strategy_return - buy_hold_return,
            }
        )
    positives = [max(float(row["excess_return"]), 0.0) for row in annual_rows]
    positive_sum = float(sum(positives))
    maximum_share = max(positives) / positive_sum if positive_sum > 0 else None
    delete_year: dict[str, float] = {}
    for year in sorted(merged["year"].unique()):
        subset = merged.loc[merged["year"] != int(year)]
        strategy_return = float((1.0 + subset["daily_return_strategy"]).prod() - 1.0)
        buy_hold_return = float((1.0 + subset["daily_return_buy_hold"]).prod() - 1.0)
        delete_year[str(int(year))] = strategy_return - buy_hold_return
    return {
        "annual_rows": annual_rows,
        "maximum_single_positive_year_share_of_positive_excess": maximum_share,
        "delete_calendar_year_total_return_excess_vs_buy_hold": delete_year,
    }


def evaluate_portfolio(
    features: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    benchmark: pd.DataFrame,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    from backtest.engine import BacktestCosts, run_long_cash_backtest, summarize_backtest

    targets = build_portfolio_targets(features, config)
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    initial_cash = float(config["portfolio"]["initial_capital_cny"])

    def costs_from(name: str) -> BacktestCosts:
        specification = config["portfolio"][name]
        return BacktestCosts(
            commission_rate=float(specification["commission_rate_per_leg"]),
            minimum_commission_cny=float(
                specification["minimum_commission_cny_per_leg"]
            ),
            stamp_duty_rate=float(specification["stamp_duty_rate"]),
            slippage_bps=float(specification["slippage_bps_per_leg"]),
            lot_size=int(config["portfolio"]["lot_size_shares"]),
            cash_annual_rate=float(config["portfolio"]["cash_annual_rate"]),
        )

    base_costs = costs_from("base_costs")
    double_costs = costs_from("double_costs")
    base_ledger, base_trades = run_long_cash_backtest(
        market, dividends, targets, initial_cash, base_costs, start, end
    )
    double_ledger, double_trades = run_long_cash_backtest(
        market, dividends, targets, initial_cash, double_costs, start, end
    )
    buy_hold_targets = targets[["date"]].copy()
    buy_hold_targets["target_position"] = 1.0
    buy_hold_ledger, buy_hold_trades = run_long_cash_backtest(
        market, dividends, buy_hold_targets, initial_cash, base_costs, start, end
    )
    base_summary = summarize_backtest(base_ledger, base_trades, initial_cash)
    double_summary = summarize_backtest(double_ledger, double_trades, initial_cash)
    buy_hold_summary = summarize_backtest(
        buy_hold_ledger, buy_hold_trades, initial_cash
    )
    h00300_cagr = _benchmark_cagr(benchmark, start, end)
    base_summary["annualized_excess_vs_510300_buy_hold"] = (
        base_summary["cagr"] - buy_hold_summary["cagr"]
    )
    base_summary["annualized_excess_vs_h00300_total_return"] = (
        base_summary["cagr"] - h00300_cagr
    )
    double_summary["annualized_excess_vs_510300_buy_hold"] = (
        double_summary["cagr"] - buy_hold_summary["cagr"]
    )
    double_summary["annualized_excess_vs_h00300_total_return"] = (
        double_summary["cagr"] - h00300_cagr
    )

    start_offsets: list[dict[str, Any]] = []
    evaluation_dates = market.loc[market["date"].between(start, end), "date"].reset_index(
        drop=True
    )
    for offset in config["portfolio"]["start_offset_trading_days"]:
        offset = int(offset)
        offset_start = pd.Timestamp(evaluation_dates.iloc[offset])
        strategy_ledger, strategy_trades = run_long_cash_backtest(
            market,
            dividends,
            targets,
            initial_cash,
            base_costs,
            offset_start,
            end,
        )
        hold_ledger, hold_trades = run_long_cash_backtest(
            market,
            dividends,
            buy_hold_targets,
            initial_cash,
            base_costs,
            offset_start,
            end,
        )
        strategy_summary = summarize_backtest(
            strategy_ledger, strategy_trades, initial_cash
        )
        hold_summary = summarize_backtest(hold_ledger, hold_trades, initial_cash)
        offset_h00300_cagr = _benchmark_cagr(benchmark, offset_start, end)
        start_offsets.append(
            {
                "offset_trading_days": offset,
                "start_date": offset_start.date().isoformat(),
                "net_sharpe": strategy_summary["sharpe_zero_cash_rate"],
                "strategy_cagr": strategy_summary["cagr"],
                "annualized_excess_vs_510300_buy_hold": (
                    strategy_summary["cagr"] - hold_summary["cagr"]
                ),
                "annualized_excess_vs_h00300_total_return": (
                    strategy_summary["cagr"] - offset_h00300_cagr
                ),
            }
        )

    if base_trades.empty:
        maximum_sell_legs = 0
        sell_legs_by_year: dict[str, int] = {}
    else:
        sells = base_trades.loc[base_trades["side"] == "卖出"].copy()
        if sells.empty:
            maximum_sell_legs = 0
            sell_legs_by_year = {}
        else:
            counts = sells.groupby(sells["date"].dt.year).size()
            sell_legs_by_year = {
                str(int(year)): int(value) for year, value in counts.items()
            }
            maximum_sell_legs = int(counts.max())
    annual = _annual_contribution_analysis(base_ledger, buy_hold_ledger)
    trial_count = int(
        manifest["selection_bias_control"]["total_trial_count_including_current"]
    )
    dsr = deflated_sharpe_probability(
        base_ledger["daily_return"].iloc[1:], trial_count
    )
    psr_above_one = probabilistic_sharpe_probability(
        base_ledger["daily_return"].iloc[1:], 1.0
    )
    drawdown_ratio = (
        abs(float(base_summary["max_drawdown"]))
        / abs(float(buy_hold_summary["max_drawdown"]))
        if float(buy_hold_summary["max_drawdown"]) < 0
        else None
    )

    rules = config["portfolio_gates"]
    gates = {
        "base_net_sharpe_at_least_1p20": (
            base_summary["sharpe_zero_cash_rate"] is not None
            and base_summary["sharpe_zero_cash_rate"]
            >= float(rules["base_net_sharpe_minimum"])
        ),
        "double_cost_net_sharpe_at_least_0p90": (
            double_summary["sharpe_zero_cash_rate"] is not None
            and double_summary["sharpe_zero_cash_rate"]
            >= float(rules["double_cost_net_sharpe_minimum"])
        ),
        "annualized_excess_vs_510300_buy_hold_positive": _positive(
            base_summary["annualized_excess_vs_510300_buy_hold"]
        ),
        "annualized_excess_vs_h00300_total_return_positive": _positive(
            base_summary["annualized_excess_vs_h00300_total_return"]
        ),
        "maximum_drawdown_ratio_vs_buy_hold_at_most_0p75": (
            drawdown_ratio is not None
            and drawdown_ratio
            <= float(rules["maximum_drawdown_ratio_vs_buy_hold_maximum"])
        ),
        "maximum_sell_legs_in_any_year_at_most_10": maximum_sell_legs
        <= int(rules["maximum_sell_legs_in_any_calendar_year"]),
        "every_start_offset_net_sharpe_at_least_1p00": all(
            payload["net_sharpe"] is not None
            and payload["net_sharpe"]
            >= float(rules["every_start_offset_net_sharpe_minimum"])
            for payload in start_offsets
        ),
        "every_start_offset_excess_vs_both_benchmarks_positive": all(
            _positive(payload["annualized_excess_vs_510300_buy_hold"])
            and _positive(payload["annualized_excess_vs_h00300_total_return"])
            for payload in start_offsets
        ),
        "delete_any_calendar_year_excess_vs_buy_hold_positive": all(
            _positive(value)
            for value in annual[
                "delete_calendar_year_total_return_excess_vs_buy_hold"
            ].values()
        ),
        "maximum_single_positive_year_share_at_most_0p50": (
            annual["maximum_single_positive_year_share_of_positive_excess"]
            is not None
            and annual["maximum_single_positive_year_share_of_positive_excess"]
            <= float(
                rules["maximum_single_positive_year_share_of_positive_excess"]
            )
        ),
        "deflated_sharpe_probability_at_least_0p95": (
            dsr["probability"] is not None
            and dsr["probability"]
            >= float(rules["deflated_sharpe_probability_minimum"])
        ),
        "probabilistic_sharpe_above_1p0_at_least_0p80": (
            psr_above_one is not None
            and psr_above_one
            >= float(rules["probabilistic_sharpe_above_1p0_minimum"])
        ),
    }
    passed = all(gates.values())
    evaluation = {
        "evaluated": True,
        "passed": passed,
        "status": (
            config["adjudication"]["portfolio_pass_status"]
            if passed
            else config["adjudication"]["portfolio_fail_status"]
        ),
        "base": base_summary,
        "double_cost": double_summary,
        "buy_hold_510300": buy_hold_summary,
        "h00300_total_return_cagr": h00300_cagr,
        "maximum_drawdown_ratio_vs_buy_hold": drawdown_ratio,
        "sell_legs_by_year": sell_legs_by_year,
        "maximum_sell_legs_in_any_calendar_year": maximum_sell_legs,
        "start_offsets": start_offsets,
        "annual_contribution_analysis": annual,
        "deflated_sharpe": dsr,
        "probabilistic_sharpe_above_1p0": psr_above_one,
        "pbo_status": config["selection_bias"]["pbo_status"],
        "gates": gates,
    }
    artifacts = {
        "portfolio_targets": targets,
        "base_ledger": base_ledger,
        "base_trades": base_trades,
        "double_cost_ledger": double_ledger,
        "double_cost_trades": double_trades,
        "buy_hold_ledger": buy_hold_ledger,
        "buy_hold_trades": buy_hold_trades,
    }
    return evaluation, artifacts


def _input_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, specification in config["inputs"].items():
        path = _project_path(specification["path"])
        payload[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return payload


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    audit: dict[str, Any],
    features: pd.DataFrame,
    risk_evaluation: dict[str, Any],
    portfolio_evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    risk_passed = bool(risk_evaluation["passed"])
    if portfolio_evaluation is None:
        status = risk_evaluation["status"]
        return_evaluation = "NOT_ALLOWED"
        net_sharpe: float | str = "NOT_COMPUTED"
        historical_target_achieved = False
        portfolio_payload: dict[str, Any] = {
            "evaluated": False,
            "status": "SKIPPED_RISK_GATE_FAILED",
            "reason": "风险预测门未全部通过，冻结协议禁止读取组合收益",
        }
    else:
        status = portfolio_evaluation["status"]
        return_evaluation = "HISTORICAL_RETROSPECTIVE_ONLY"
        net_sharpe = portfolio_evaluation["base"]["sharpe_zero_cash_rate"]
        historical_target_achieved = bool(portfolio_evaluation["passed"])
        portfolio_payload = portfolio_evaluation
    return {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "candidate_model_id": config["protocol"]["candidate_model_id"],
        "status": status,
        "risk_gate_passed": risk_passed,
        "evidence_class": config["protocol"]["evidence_class"],
        "evaluation_window": {
            "start": config["dates"]["evaluation_start"],
            "end": config["dates"]["evaluation_end"],
            "trading_rows": int(len(features)),
            "valid_future_risk_rows": int(risk_evaluation["valid_target_rows"]),
        },
        "scope": {
            "execution_asset": config["scope"]["execution_asset"],
            "allowed_holdings": config["scope"]["allowed_holdings"],
            "long_only": True,
            "leverage_allowed": False,
        },
        "input_audit": {
            "status": audit["status"],
            "report_id": audit["report_id"],
            "candidate_outcomes_computed_before_freeze": audit[
                "candidate_outcomes_computed"
            ],
            "portfolio_returns_computed_before_freeze": audit[
                "portfolio_returns_computed"
            ],
            "canonical_price_corrections": audit["canonical_price"]["corrections"],
            "snapshots": _input_snapshot(config),
        },
        "freeze": manifest,
        "risk_evaluation": risk_evaluation,
        "portfolio_evaluation": portfolio_payload,
        "adjudication": {
            "return_evaluation": return_evaluation,
            "net_sharpe": net_sharpe,
            "target_net_sharpe": float(
                config["adjudication"]["target_net_sharpe"]
            ),
            "historical_target_achieved": historical_target_achieved,
            "verified_forward_target_achieved": False,
            "goal_achieved": False,
            "reason": (
                "风险预测门失败，未运行投资组合"
                if not risk_passed
                else (
                    "历史回测虽通过全部门，但历史数据已污染，仍需冻结后的真正前向确认"
                    if historical_target_achieved
                    else "风险预测门通过，但净费后投资组合未通过全部冻结门"
                )
            ),
        },
        "boundaries": {
            "historical_data_already_contaminated": True,
            "parameter_rescue_after_result": "FORBIDDEN",
            "alternate_factor_rescue_after_result": "FORBIDDEN",
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "position_change": "DISABLED",
            "live_trading_authorized": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    risk = report["risk_evaluation"]
    comparison = risk["composite_group_comparison"]
    bootstrap = risk["bootstrap_90pct_intervals"]

    def percent(value: float | None) -> str:
        return "NA" if value is None else f"{value:.2%}"

    def number(value: float | None) -> str:
        return "NA" if value is None else f"{value:.4f}"

    lines = [
        "# 510300下行风险预算V1",
        "",
        f"- 状态：`{report['status']}`",
        f"- 风险预测门通过：`{str(report['risk_gate_passed']).lower()}`",
        f"- 评价区间：`{report['evaluation_window']['start']}—{report['evaluation_window']['end']}`",
        f"- 交易日：`{report['evaluation_window']['trading_rows']}`",
        f"- 完整未来风险标签：`{report['evaluation_window']['valid_future_risk_rows']}`",
        "- 交易资产边界：仅`510300.SH`与现金；无杠杆、无卖空。",
        "",
        "## 输入审计",
        "",
        f"- 数据质量：`{report['input_audit']['status']}`",
        f"- 双源共识价格修正：`{len(report['input_audit']['canonical_price_corrections'])}`处",
        "- 冻结前读取候选结果：`false`",
        "- 冻结前读取组合收益：`false`",
        "",
        "## 下行风险预测门",
        "",
        f"- 低风险/高风险样本：`{comparison['low_count']}` / `{comparison['high_count']}`",
        f"- 低风险未来20日平均最大不利幅度：`{percent(comparison['low_mean_mae_magnitude'])}`",
        f"- 高风险未来20日平均最大不利幅度：`{percent(comparison['high_mean_mae_magnitude'])}`",
        f"- 高减低平均差：`{percent(comparison['high_minus_low_mean_mae_magnitude'])}`",
        f"- 高减低中位数差：`{percent(comparison['high_minus_low_median_mae_magnitude'])}`",
        f"- 高减低尾部事件率差：`{percent(comparison['high_minus_low_tail_rate'])}`",
        f"- 平均差区块Bootstrap 90%区间：`[{percent(bootstrap['high_minus_low_mean_mae_magnitude'][0])}, {percent(bootstrap['high_minus_low_mean_mae_magnitude'][1])}]`",
        f"- 复合风险Rank IC：`{number(risk['composite_rank_ic'])}`",
        f"- RV20基准Rank IC：`{number(risk['rv20_baseline_rank_ic'])}`",
        f"- 相对RV20增量Rank IC：`{number(risk['incremental_rank_ic_vs_rv20'])}`",
        f"- 风险门通过：`{str(risk['passed']).lower()}`",
        "",
        "## 投资组合与夏普",
        "",
    ]
    portfolio = report["portfolio_evaluation"]
    if not portfolio["evaluated"]:
        lines.extend(
            [
                f"- 状态：`{portfolio['status']}`",
                "- 净费后夏普率：`NOT_COMPUTED`",
                "- 原因：风险预测门未全部通过，冻结协议禁止读取组合收益。",
            ]
        )
    else:
        base = portfolio["base"]
        stress = portfolio["double_cost"]
        lines.extend(
            [
                f"- 状态：`{portfolio['status']}`",
                f"- 基础成本净夏普：`{number(base['sharpe_zero_cash_rate'])}`",
                f"- 双倍成本净夏普：`{number(stress['sharpe_zero_cash_rate'])}`",
                f"- 基础成本CAGR：`{percent(base['cagr'])}`",
                f"- 相对510300买入持有年化超额：`{percent(base['annualized_excess_vs_510300_buy_hold'])}`",
                f"- 相对H00300全收益年化超额：`{percent(base['annualized_excess_vs_h00300_total_return'])}`",
                f"- 最大回撤：`{percent(base['max_drawdown'])}`",
                f"- 投资组合门通过：`{str(portfolio['passed']).lower()}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 最终裁决",
            "",
            f"- 收益评价：`{report['adjudication']['return_evaluation']}`",
            f"- 净夏普率：`{report['adjudication']['net_sharpe']}`",
            f"- 历史夏普1.2全部门通过：`{str(report['adjudication']['historical_target_achieved']).lower()}`",
            "- 真正前向目标实现：`false`",
            f"- 原因：{report['adjudication']['reason']}",
            "- Paper、Shadow、订单、券商连接、实盘：全部关闭。",
            "- 失败后禁止更换因子、参数、窗口、方向或仓位映射救援。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def run_study(*, write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    result_path = _project_path(config["paths"]["result_json"])
    if write and result_path.exists():
        raise FileExistsError(f"正式结果已经存在，禁止覆盖：{result_path}")
    market, dividends, benchmark, audit = load_inputs(config)
    features = build_daily_features(market, dividends, config)
    features = add_forward_risk_targets(features, market, dividends, config)
    risk_evaluation = evaluate_risk_gate(features, config)
    portfolio_evaluation: dict[str, Any] | None = None
    artifacts: dict[str, pd.DataFrame] = {}
    if risk_evaluation["passed"]:
        portfolio_evaluation, artifacts = evaluate_portfolio(
            features, market, dividends, benchmark, manifest, config
        )
    report = build_report(
        config,
        manifest,
        audit,
        features,
        risk_evaluation,
        portfolio_evaluation,
    )
    if write:
        paths = config["paths"]
        _atomic_parquet(_project_path(paths["feature_table"]), features)
        for name, frame in artifacts.items():
            _atomic_parquet(_project_path(paths[name]), frame)
        _atomic_json(result_path, report)
        _atomic_text(
            _project_path(paths["result_markdown"]), render_markdown(report)
        )
    return report


__all__ = [
    "CONFIG_PATH",
    "MANIFEST_PATH",
    "ContractError",
    "add_forward_risk_targets",
    "bootstrap_risk_metrics",
    "build_daily_features",
    "build_portfolio_targets",
    "causal_percentile",
    "causal_total_return_series",
    "deflated_sharpe_probability",
    "evaluate_risk_gate",
    "load_config",
    "load_inputs",
    "probabilistic_sharpe_probability",
    "rolling_slope_t_stat",
    "run_study",
    "sha256_file",
    "validate_config",
    "validate_manifest",
]
