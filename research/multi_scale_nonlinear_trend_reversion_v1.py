"""510300多尺度非线性趋势延续与反转V1的一次性冻结研究实现。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from statistics import NormalDist
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import (  # noqa: E402
    BacktestCosts,
    run_long_cash_backtest,
    summarize_backtest,
)


CONFIG_PATH = (
    ROOT / "config" / "510300_multi_scale_nonlinear_trend_reversion_v1.yaml"
)
MANIFEST_PATH = (
    ROOT
    / "config"
    / "510300_multi_scale_nonlinear_trend_reversion_v1_manifest.json"
)


class ContractError(RuntimeError):
    """冻结协议、输入数据或结果时钟不符合契约。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def _read_nested(config: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = config
    for key in keys:
        value = value[key]
    return value


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        ("protocol", "project_id"): "510300_MULTI_SCALE_NONLINEAR_TREND_REVERSION_V1",
        ("scope", "execution_asset"): "510300.SH",
        ("scope", "signal_asset"): "000300.SH_PRICE_INDEX",
        ("dates", "evaluation_start"): "2015-01-05",
        ("dates", "evaluation_end"): "2026-08-14",
        ("normalization", "training_log_return_count"): 2365,
        ("trend", "horizons_trading_days"): [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        ("trend", "lag_multiplier"): 2.25,
        ("trend", "maximum_horizon_lag_count"): 2304,
        ("trend", "fixed_linear_coefficient"): 0.0129,
        ("trend", "fixed_cubic_coefficient"): -0.0062,
        ("portfolio", "initial_capital_cny"): 20000.0,
        ("portfolio", "lot_size_shares"): 100,
        ("portfolio", "base_costs", "commission_rate_per_leg"): 0.0003,
        ("portfolio", "base_costs", "minimum_commission_cny_per_leg"): 5.0,
        ("portfolio", "base_costs", "slippage_bps_per_leg"): 5.0,
        ("portfolio", "stress_costs", "slippage_bps_per_leg"): 10.0,
        ("portfolio_gates", "base_net_sharpe_minimum"): 1.20,
        ("selection_bias", "non_manifest_prefreeze_failed_attempt_count"): 1,
        ("selection_bias", "expected_total_trial_count_including_current"): 345,
    }
    for keys, wanted in expected.items():
        observed = _read_nested(config, keys)
        if observed != wanted:
            raise ContractError(
                f"冻结字段{'.'.join(keys)}异常：期望{wanted!r}，得到{observed!r}"
            )
    if config["scope"]["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ContractError("持仓集合必须严格等于510300.SH与人民币现金")
    forbidden = [
        config["scope"]["leverage_allowed"],
        config["scope"]["short_selling_allowed"],
        config["scope"]["derivatives_execution_allowed"],
        config["scope"]["live_trading_authorized"],
        config["governance"]["paper_signal_allowed"],
        config["governance"]["shadow_signal_allowed"],
        config["governance"]["order_generation"],
        config["governance"]["broker_connection"],
        config["governance"]["position_change"],
        config["governance"]["live_trading_authorized"],
    ]
    if any(forbidden):
        raise ContractError("研究协议不得授权杠杆、卖空、信号映射、订单或实盘")
    if config["protocol"]["one_shot"] is not True:
        raise ContractError("候选必须保持一次性冻结检验")
    rescue_keys = [
        "parameter_rescue_after_result",
        "alternate_scale_rescue_after_result",
        "alternate_weight_rescue_after_result",
        "coefficient_refit_rescue_after_result",
        "direction_reversal_after_result",
        "combination_rescue_after_result",
    ]
    if any(config["protocol"][key] != "forbidden" for key in rescue_keys):
        raise ContractError("结果后救援边界没有全部冻结为forbidden")
    if config["trend"]["position_mapping"] if "position_mapping" in config["trend"] else False:
        raise ContractError("仓位映射只能在portfolio节定义")
    if config["portfolio"]["position_mapping"] != (
        "ONE_IF_FIXED_TREND_SCORE_POSITIVE_ELSE_ZERO"
    ):
        raise ContractError("仓位映射不符合冻结定义")
    if float(config["normalization"]["training_log_return_sample_std"]) <= 0:
        raise ContractError("训练期标准差必须为正数")
    horizons = config["trend"]["horizons_trading_days"]
    if horizons != sorted(set(horizons)) or len(horizons) != 10:
        raise ContractError("趋势尺度必须是十个互异升序尺度")
    for name in ("base_costs", "stress_costs"):
        costs = config["portfolio"][name]
        if float(costs["stamp_duty_rate"]) != 0.0:
            raise ContractError("ETF交易印花税必须冻结为零")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError(f"冻结清单不存在：{MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_FIRST_2015_PLUS_CANDIDATE_OUTCOME":
        raise ContractError("冻结清单状态错误")
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ContractError("冻结清单项目标识错误")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("冻结后配置文件发生漂移")
    mismatches: dict[str, dict[str, str]] = {}
    for section in ("tracked_files", "input_files"):
        for relative, expected in manifest.get(section, {}).items():
            path = _project_path(relative)
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ContractError(f"冻结文件发生漂移：{mismatches}")
    if manifest.get("candidate_2015_plus_outcomes_read_before_freeze") is not False:
        raise ContractError("冻结清单未证明2015年后候选结果不可见")
    if manifest.get("portfolio_returns_read_before_freeze") is not False:
        raise ContractError("冻结清单未证明组合收益不可见")
    total_trials = int(
        manifest.get("selection_bias_control", {}).get(
            "total_trial_count_including_current", 0
        )
    )
    if total_trials != int(
        config["selection_bias"]["expected_total_trial_count_including_current"]
    ):
        raise ContractError("冻结试验总数与协议不一致")
    return manifest


def trend_weights(
    horizon: int, lag_multiplier: float
) -> tuple[np.ndarray, float]:
    if horizon <= 0 or lag_multiplier <= 0:
        raise ValueError("趋势尺度与滞后倍数必须为正数")
    lag_count = int(math.ceil(horizon * lag_multiplier))
    lags = np.arange(1, lag_count + 1, dtype=float)
    raw = lags * np.exp(-2.0 * lags / float(horizon))
    norm = float(np.sqrt(np.square(raw).sum()))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("趋势权重无法归一化")
    weights = raw / norm
    reference_lags = np.arange(1, max(200000, 20 * horizon) + 1, dtype=float)
    reference = reference_lags * np.exp(-2.0 * reference_lags / float(horizon))
    capture = float(np.square(raw).sum() / np.square(reference).sum())
    return weights, capture


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(
    config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    specifications = config["inputs"]
    for name, specification in specifications.items():
        path = _project_path(specification["path"])
        if not path.exists():
            raise ContractError(f"缺少输入{name}：{path}")

    index_audit = _load_json(_project_path(specifications["index_input_audit"]["path"]))
    etf_audit = _load_json(_project_path(specifications["etf_input_audit"]["path"]))
    candidate_audit = _load_json(
        _project_path(specifications["candidate_input_audit"]["path"])
    )
    if index_audit.get("status") != specifications["index_input_audit"]["required_status"]:
        raise ContractError("000300输入审计未通过")
    if etf_audit.get("status") != specifications["etf_input_audit"]["required_status"]:
        raise ContractError("510300输入审计未通过")
    if candidate_audit.get("status") != specifications["candidate_input_audit"]["required_status"]:
        raise ContractError("候选预冻结输入审计未通过")
    if candidate_audit.get("candidate_2015_plus_outcomes_read_or_computed") is not False:
        raise ContractError("预冻结审计曾读取2015年后候选结果")
    if candidate_audit.get("candidate_portfolio_returns_read_or_computed") is not False:
        raise ContractError("预冻结审计曾读取组合收益")

    index_price = pd.read_parquet(_project_path(specifications["index_price"]["path"]))
    market = pd.read_parquet(_project_path(specifications["etf_daily"]["path"]))
    dividends = pd.read_csv(_project_path(specifications["dividends"]["path"]))
    benchmark = pd.read_parquet(
        _project_path(specifications["benchmark_total_return"]["path"])
    )
    frames = (
        ("index_price", index_price, specifications["index_price"]["required_columns"]),
        ("etf_daily", market, specifications["etf_daily"]["required_columns"]),
        ("dividends", dividends, specifications["dividends"]["required_columns"]),
        (
            "benchmark_total_return",
            benchmark,
            specifications["benchmark_total_return"]["required_columns"],
        ),
    )
    for name, frame, required in frames:
        missing = set(required).difference(frame.columns)
        if missing:
            raise ContractError(f"输入{name}缺少字段：{sorted(missing)}")

    index_price = index_price.copy()
    market = market.copy()
    dividends = dividends.copy()
    benchmark = benchmark.copy()
    for frame in (index_price, market, benchmark):
        frame["date"] = pd.to_datetime(frame["date"])
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame["date"].duplicated().any():
            raise ContractError("行情输入存在重复日期")
    for field in ("record_date", "ex_date", "payment_date"):
        dividends[field] = pd.to_datetime(dividends[field])
    dividends.sort_values("ex_date", inplace=True)
    dividends.reset_index(drop=True, inplace=True)

    contract = config["data_contract"]
    if len(index_price) != int(contract["index_expected_rows"]):
        raise ContractError("000300原始行数不符合冻结契约")
    if index_price["date"].min() != pd.Timestamp(contract["index_expected_first_date"]):
        raise ContractError("000300首日不符合冻结契约")
    if index_price["date"].max() != pd.Timestamp(contract["index_expected_last_date"]):
        raise ContractError("000300末日不符合冻结契约")
    if len(market) != int(contract["etf_expected_rows"]):
        raise ContractError("510300原始行数不符合冻结契约")
    if market["date"].min() != pd.Timestamp(contract["etf_expected_first_date"]):
        raise ContractError("510300首日不符合冻结契约")
    if market["date"].max() != pd.Timestamp(contract["etf_expected_last_date"]):
        raise ContractError("510300末日不符合冻结契约")

    index_price["close"] = pd.to_numeric(index_price["close"], errors="coerce")
    market[["open", "high", "low", "close"]] = market[
        ["open", "high", "low", "close"]
    ].apply(pd.to_numeric, errors="coerce")
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="coerce"
    )
    if index_price["close"].isna().any() or (index_price["close"] <= 0).any():
        raise ContractError("000300收盘价存在空值、零值或负值")
    if market[["open", "high", "low", "close"]].isna().any().any() or (
        market[["open", "high", "low", "close"]] <= 0
    ).any().any():
        raise ContractError("510300行情存在空值、零值或负值")
    if benchmark["close"].isna().any() or (benchmark["close"] <= 0).any():
        raise ContractError("H00300收盘值存在空值、零值或负值")
    if dividends["cash_dividend_per_share"].isna().any() or (
        dividends["cash_dividend_per_share"] <= 0
    ).any():
        raise ContractError("分红金额存在空值、零值或负值")
    if (
        (market["high"] < market[["open", "close"]].max(axis=1))
        | (market["low"] > market[["open", "close"]].min(axis=1))
        | (market["high"] < market["low"])
    ).any():
        raise ContractError("510300行情违反OHLC约束")

    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    index_dates = index_price.loc[index_price["date"].between(start, end), "date"]
    market_dates = market.loc[market["date"].between(start, end), "date"]
    if len(index_dates) != int(contract["index_expected_evaluation_rows"]):
        raise ContractError("000300评价期交易日数不符合冻结契约")
    if bool(contract["evaluation_market_dates_must_match_index"]) and not index_dates.reset_index(
        drop=True
    ).equals(market_dates.reset_index(drop=True)):
        raise ContractError("评价期000300与510300交易日不完全一致")
    audits = {
        "index_input_audit": index_audit,
        "etf_input_audit": etf_audit,
        "candidate_input_audit": candidate_audit,
    }
    return index_price, market, dividends, benchmark, audits


def build_trend_panel(index_price: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    frame = index_price[["date", "close"]].copy().sort_values("date").reset_index(drop=True)
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    training_end = pd.Timestamp(config["dates"]["training_normalization_end"])
    training = frame.loc[frame["date"] <= training_end, "log_return"].dropna()
    normalization = config["normalization"]
    if len(training) != int(normalization["training_log_return_count"]):
        raise ContractError("训练期收益行数与冻结常数不一致")
    observed_mean = float(training.mean())
    observed_std = float(training.std(ddof=1))
    expected_mean = float(normalization["training_log_return_mean"])
    expected_std = float(normalization["training_log_return_sample_std"])
    if not math.isclose(observed_mean, expected_mean, rel_tol=0.0, abs_tol=1e-15):
        raise ContractError("训练期收益均值与冻结常数不一致")
    if not math.isclose(observed_std, expected_std, rel_tol=0.0, abs_tol=1e-15):
        raise ContractError("训练期收益标准差与冻结常数不一致")
    clip = float(normalization["normalized_return_clip_absolute"])
    frame["normalized_log_return"] = (
        (frame["log_return"] - expected_mean) / expected_std
    ).clip(lower=-clip, upper=clip)

    values = frame["normalized_log_return"].to_numpy(dtype=float)
    trend_columns: list[str] = []
    for horizon in config["trend"]["horizons_trading_days"]:
        horizon = int(horizon)
        weights, capture = trend_weights(
            horizon, float(config["trend"]["lag_multiplier"])
        )
        if capture < float(config["trend"]["minimum_squared_weight_energy_capture"]):
            raise ContractError(f"趋势尺度{horizon}的固定权重能量覆盖不足")
        lag_count = len(weights)
        convolved = np.convolve(np.nan_to_num(values, nan=0.0), weights, mode="full")
        phi = np.full(len(frame), np.nan, dtype=float)
        first_valid = lag_count + 1
        if first_valid < len(frame):
            phi[first_valid:] = convolved[first_valid - 1 : len(frame) - 1]
        column = f"phi_{horizon}"
        frame[column] = np.clip(
            phi,
            float(config["trend"]["trend_clip_lower"]),
            float(config["trend"]["trend_clip_upper"]),
        )
        trend_columns.append(column)

    frame["phi_mean"] = frame[trend_columns].mean(axis=1, skipna=False)
    linear = float(config["trend"]["fixed_linear_coefficient"])
    cubic = float(config["trend"]["fixed_cubic_coefficient"])
    frame["fixed_paper_score"] = (
        linear * frame["phi_mean"] + cubic * np.power(frame["phi_mean"], 3)
    )
    frame["next_normalized_log_return"] = frame["normalized_log_return"].shift(-1)
    periods = config["dates"]["structural_periods"]
    frame["structural_period"] = "OUTSIDE"
    for label, bounds in periods.items():
        mask = frame["date"].between(
            pd.Timestamp(bounds["start"]), pd.Timestamp(bounds["end"])
        )
        frame.loc[mask, "structural_period"] = str(label)
    return frame


def _hac_regression(
    design: np.ndarray, outcome: np.ndarray, lags: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("回归设计矩阵与结果维度不一致")
    if len(y) <= x.shape[1] + lags:
        raise ValueError("回归有效行数不足")
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for index in range(len(y)):
        vector = x[index] * residual[index]
        meat += np.outer(vector, vector)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / float(lags + 1)
        gamma = np.zeros_like(meat)
        for index in range(lag, len(y)):
            current = x[index] * residual[index]
            previous = x[index - lag] * residual[index - lag]
            gamma += np.outer(current, previous)
        meat += weight * (gamma + gamma.T)
    covariance = xtx_inverse @ meat @ xtx_inverse
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_statistics = np.divide(
        beta,
        standard_errors,
        out=np.full_like(beta, np.nan),
        where=standard_errors > 0,
    )
    return beta, standard_errors, t_statistics


def _regression_metrics(frame: pd.DataFrame, lags: int) -> dict[str, float | int | None]:
    valid = frame.dropna(
        subset=["phi_mean", "fixed_paper_score", "next_normalized_log_return"]
    )
    phi = valid["phi_mean"].to_numpy(dtype=float)
    outcome = valid["next_normalized_log_return"].to_numpy(dtype=float)
    cubic_design = np.column_stack([np.ones(len(valid)), phi, np.power(phi, 3)])
    coefficient, standard_error, t_stat = _hac_regression(cubic_design, outcome, lags)
    score = valid["fixed_paper_score"].to_numpy(dtype=float)
    score_design = np.column_stack([np.ones(len(valid)), score])
    score_beta, score_se, score_t = _hac_regression(score_design, outcome, lags)
    mse_zero = float(np.mean(np.square(outcome)))
    mse_score = float(np.mean(np.square(outcome - score)))
    return {
        "valid_rows": int(len(valid)),
        "intercept": float(coefficient[0]),
        "linear_coefficient": float(coefficient[1]),
        "cubic_coefficient": float(coefficient[2]),
        "linear_newey_west_standard_error": float(standard_error[1]),
        "cubic_newey_west_standard_error": float(standard_error[2]),
        "linear_newey_west_t": float(t_stat[1]),
        "cubic_newey_west_t": float(t_stat[2]),
        "fixed_score_intercept": float(score_beta[0]),
        "fixed_score_slope": float(score_beta[1]),
        "fixed_score_newey_west_standard_error": float(score_se[1]),
        "fixed_score_newey_west_t": float(score_t[1]),
        "fixed_score_mse": mse_score,
        "zero_baseline_mse": mse_zero,
        "fixed_score_mse_improvement_vs_zero": (
            float(1.0 - mse_score / mse_zero) if mse_zero > 0 else None
        ),
    }


def _circular_block_indices(
    row_count: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    block_count = int(math.ceil(row_count / block_length))
    starts = rng.integers(0, row_count, size=block_count)
    offsets = np.arange(block_length, dtype=int)
    return np.concatenate([(start + offsets) % row_count for start in starts])[:row_count]


def _bootstrap_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[float]]:
    valid = frame.dropna(
        subset=["phi_mean", "fixed_paper_score", "next_normalized_log_return"]
    )
    phi = valid["phi_mean"].to_numpy(dtype=float)
    score = valid["fixed_paper_score"].to_numpy(dtype=float)
    outcome = valid["next_normalized_log_return"].to_numpy(dtype=float)
    specification = config["mechanism_evaluation"]["bootstrap"]
    repetitions = int(specification["repetitions"])
    block_length = int(specification["block_length_trading_days"])
    confidence = float(specification["confidence_level"])
    rng = np.random.default_rng(int(specification["random_seed"]))
    linear_values = np.empty(repetitions, dtype=float)
    cubic_values = np.empty(repetitions, dtype=float)
    score_values = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        indices = _circular_block_indices(len(valid), block_length, rng)
        sampled_phi = phi[indices]
        sampled_outcome = outcome[indices]
        design = np.column_stack(
            [np.ones(len(indices)), sampled_phi, np.power(sampled_phi, 3)]
        )
        beta = np.linalg.lstsq(design, sampled_outcome, rcond=None)[0]
        linear_values[repetition] = beta[1]
        cubic_values[repetition] = beta[2]
        score_design = np.column_stack([np.ones(len(indices)), score[indices]])
        score_values[repetition] = np.linalg.lstsq(
            score_design, sampled_outcome, rcond=None
        )[0][1]
    lower = (1.0 - confidence) / 2.0
    upper = 1.0 - lower

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.quantile(values, lower)), float(np.quantile(values, upper))]

    return {
        "linear_coefficient_90pct": interval(linear_values),
        "cubic_coefficient_90pct": interval(cubic_values),
        "fixed_score_slope_90pct": interval(score_values),
    }


def evaluate_mechanism(panel: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    evaluation = panel.loc[panel["date"].between(start, end)].copy()
    lags = int(config["mechanism_evaluation"]["newey_west_lags"])
    full = _regression_metrics(evaluation, lags)
    bootstrap = _bootstrap_metrics(evaluation, config)
    structural: dict[str, dict[str, float | int | None]] = {}
    for label in config["dates"]["structural_periods"]:
        structural[label] = _regression_metrics(
            evaluation.loc[evaluation["structural_period"] == label], lags
        )
    thresholds = config["mechanism_evaluation"]["gates"]
    gates = {
        "minimum_valid_rows": full["valid_rows"]
        >= int(config["mechanism_evaluation"]["minimum_valid_rows"]),
        "full_sample_linear_coefficient_positive": full["linear_coefficient"] > 0,
        "full_sample_cubic_coefficient_negative": full["cubic_coefficient"] < 0,
        "full_sample_linear_newey_west_t_minimum": full["linear_newey_west_t"]
        >= float(thresholds["full_sample_linear_newey_west_t_minimum"]),
        "full_sample_cubic_newey_west_t_maximum": full["cubic_newey_west_t"]
        <= float(thresholds["full_sample_cubic_newey_west_t_maximum"]),
        "linear_bootstrap_90pct_lower_positive": bootstrap[
            "linear_coefficient_90pct"
        ][0]
        > 0,
        "cubic_bootstrap_90pct_upper_negative": bootstrap[
            "cubic_coefficient_90pct"
        ][1]
        < 0,
        "fixed_paper_score_slope_positive": full["fixed_score_slope"] > 0,
        "fixed_paper_score_newey_west_t_minimum": full[
            "fixed_score_newey_west_t"
        ]
        >= float(thresholds["fixed_paper_score_newey_west_t_minimum"]),
        "fixed_paper_score_bootstrap_90pct_lower_positive": bootstrap[
            "fixed_score_slope_90pct"
        ][0]
        > 0,
        "fixed_paper_score_mse_improvement_vs_zero_positive": (
            full["fixed_score_mse_improvement_vs_zero"] is not None
            and full["fixed_score_mse_improvement_vs_zero"] > 0
        ),
        "both_structural_periods_preserve_linear_positive_cubic_negative": all(
            metrics["linear_coefficient"] > 0 and metrics["cubic_coefficient"] < 0
            for metrics in structural.values()
        ),
        "both_structural_periods_fixed_score_slope_positive": all(
            metrics["fixed_score_slope"] > 0 for metrics in structural.values()
        ),
    }
    passed = all(gates.values())
    return {
        "passed": passed,
        "status": (
            config["adjudication"]["mechanism_pass_status"]
            if passed
            else config["adjudication"]["mechanism_fail_status"]
        ),
        "full_sample": full,
        "bootstrap": bootstrap,
        "structural_periods": structural,
        "gates": gates,
    }


def build_portfolio_targets(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    previous = float(config["portfolio"]["initial_research_target"])
    rows: list[dict[str, Any]] = []
    for row in panel.itertuples(index=False):
        score = float(row.fixed_paper_score) if pd.notna(row.fixed_paper_score) else math.nan
        if not math.isfinite(score):
            target = previous
            reason = "固定论文分数缺失，维持上一研究目标"
        elif score > 0:
            target = 1.0
            reason = "固定非线性趋势分数为正"
        else:
            target = 0.0
            reason = "固定非线性趋势分数不为正"
        rows.append(
            {
                "date": pd.Timestamp(row.date),
                "fixed_paper_score": score,
                "target_position": target,
                "trade_allowed": True,
                "risk_off_override": target == 0.0,
                "signal_reason": reason,
            }
        )
        previous = target
    return pd.DataFrame(rows)


def _costs(config: dict[str, Any], name: str) -> BacktestCosts:
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


def _probabilistic_sharpe_probability(
    returns: pd.Series, benchmark_annual_sharpe: float
) -> float | None:
    values = returns.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return None
    daily_sharpe = float(np.mean(values) / np.std(values, ddof=1))
    benchmark_daily = benchmark_annual_sharpe / math.sqrt(242.0)
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


def _deflated_sharpe_probability(
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
    probability = _probabilistic_sharpe_probability(
        pd.Series(values), expected_maximum_daily * math.sqrt(242.0)
    )
    return {
        "total_trial_count": trial_count,
        "expected_maximum_null_sharpe_annualized": float(
            expected_maximum_daily * math.sqrt(242.0)
        ),
        "probability": probability,
    }


def _benchmark_cagr(
    benchmark: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> float:
    subset = benchmark.loc[benchmark["date"].between(start, end)]
    if len(subset) < 2:
        raise ContractError("H00300评价区间不足两行")
    elapsed_days = max((subset["date"].iloc[-1] - subset["date"].iloc[0]).days, 1)
    total_return = float(subset["close"].iloc[-1] / subset["close"].iloc[0] - 1.0)
    return float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)


def _annual_excess_analysis(
    strategy: pd.DataFrame, buy_hold: pd.DataFrame
) -> dict[str, Any]:
    merged = strategy[["date", "daily_return"]].merge(
        buy_hold[["date", "daily_return"]],
        on="date",
        suffixes=("_strategy", "_buy_hold"),
        validate="one_to_one",
    )
    merged["year"] = merged["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, subset in merged.groupby("year", sort=True):
        strategy_return = float((1.0 + subset["daily_return_strategy"]).prod() - 1.0)
        hold_return = float((1.0 + subset["daily_return_buy_hold"]).prod() - 1.0)
        rows.append(
            {
                "year": int(year),
                "strategy_return": strategy_return,
                "buy_hold_return": hold_return,
                "excess_return": strategy_return - hold_return,
            }
        )
    positives = [max(float(row["excess_return"]), 0.0) for row in rows]
    positive_sum = float(sum(positives))
    maximum_share = max(positives) / positive_sum if positive_sum > 0 else None
    delete_year: dict[str, float] = {}
    for year in sorted(merged["year"].unique()):
        subset = merged.loc[merged["year"] != year]
        strategy_return = float((1.0 + subset["daily_return_strategy"]).prod() - 1.0)
        hold_return = float((1.0 + subset["daily_return_buy_hold"]).prod() - 1.0)
        delete_year[str(int(year))] = strategy_return - hold_return
    return {
        "annual_rows": rows,
        "maximum_single_positive_year_share_of_positive_excess": maximum_share,
        "delete_calendar_year_total_return_excess_vs_buy_hold": delete_year,
    }


def _ledger_slice(
    ledger: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    result = ledger.loc[ledger["date"].between(start, end)].copy().reset_index(drop=True)
    if result.empty:
        raise ContractError("评价期净值为空")
    return result


def _trade_slice(
    trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    return trades.loc[trades["date"].between(start, end)].copy().reset_index(drop=True)


def _annualized_sharpe(returns: pd.Series) -> float | None:
    values = returns.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.std(values, ddof=1) <= 0:
        return None
    return float(np.mean(values) / np.std(values, ddof=1) * math.sqrt(242.0))


def evaluate_portfolio(
    panel: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    benchmark: pd.DataFrame,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    targets = build_portfolio_targets(panel, config)
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    previous_dates = market.loc[market["date"] < start, "date"]
    if previous_dates.empty:
        raise ContractError("评价首日前没有用于下一开盘执行的市场日期")
    engine_start = pd.Timestamp(previous_dates.iloc[-1])
    initial_cash = float(config["portfolio"]["initial_capital_cny"])
    base_ledger_all, base_trades_all = run_long_cash_backtest(
        market,
        dividends,
        targets,
        initial_cash,
        _costs(config, "base_costs"),
        engine_start,
        end,
    )
    stress_ledger_all, stress_trades_all = run_long_cash_backtest(
        market,
        dividends,
        targets,
        initial_cash,
        _costs(config, "stress_costs"),
        engine_start,
        end,
    )
    buy_hold_targets = targets[["date"]].copy()
    buy_hold_targets["target_position"] = 1.0
    buy_hold_ledger_all, buy_hold_trades_all = run_long_cash_backtest(
        market,
        dividends,
        buy_hold_targets,
        initial_cash,
        _costs(config, "base_costs"),
        engine_start,
        end,
    )
    base_ledger = _ledger_slice(base_ledger_all, start, end)
    stress_ledger = _ledger_slice(stress_ledger_all, start, end)
    buy_hold_ledger = _ledger_slice(buy_hold_ledger_all, start, end)
    base_trades = _trade_slice(base_trades_all, start, end)
    stress_trades = _trade_slice(stress_trades_all, start, end)
    buy_hold_trades = _trade_slice(buy_hold_trades_all, start, end)
    base_summary = summarize_backtest(base_ledger, base_trades, initial_cash)
    stress_summary = summarize_backtest(stress_ledger, stress_trades, initial_cash)
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
    stress_summary["annualized_excess_vs_510300_buy_hold"] = (
        stress_summary["cagr"] - buy_hold_summary["cagr"]
    )
    drawdown_ratio = (
        abs(float(base_summary["max_drawdown"]))
        / abs(float(buy_hold_summary["max_drawdown"]))
        if float(buy_hold_summary["max_drawdown"]) < 0
        else None
    )
    late_start = pd.Timestamp(
        config["dates"]["structural_periods"]["LATE_TEMPORAL_HOLDOUT"]["start"]
    )
    late_sharpe = _annualized_sharpe(
        base_ledger.loc[base_ledger["date"] >= late_start, "daily_return"]
    )
    annual = _annual_excess_analysis(base_ledger, buy_hold_ledger)
    total_trials = int(
        manifest["selection_bias_control"]["total_trial_count_including_current"]
    )
    dsr = _deflated_sharpe_probability(base_ledger["daily_return"], total_trials)
    rules = config["portfolio_gates"]
    gates = {
        "base_net_sharpe_at_least_1p20": (
            base_summary["sharpe_zero_cash_rate"] is not None
            and base_summary["sharpe_zero_cash_rate"]
            >= float(rules["base_net_sharpe_minimum"])
        ),
        "stress_net_sharpe_at_least_0p90": (
            stress_summary["sharpe_zero_cash_rate"] is not None
            and stress_summary["sharpe_zero_cash_rate"]
            >= float(rules["stress_net_sharpe_minimum"])
        ),
        "annualized_excess_vs_510300_buy_hold_positive": base_summary[
            "annualized_excess_vs_510300_buy_hold"
        ]
        > float(rules["annualized_excess_vs_510300_buy_hold_minimum"]),
        "annualized_excess_vs_h00300_total_return_positive": base_summary[
            "annualized_excess_vs_h00300_total_return"
        ]
        > float(rules["annualized_excess_vs_h00300_total_return_minimum"]),
        "maximum_drawdown_ratio_vs_buy_hold_at_most_0p75": (
            drawdown_ratio is not None
            and drawdown_ratio
            <= float(rules["maximum_drawdown_ratio_vs_buy_hold_maximum"])
        ),
        "late_temporal_holdout_net_sharpe_at_least_1p00": (
            late_sharpe is not None
            and late_sharpe >= float(rules["late_temporal_holdout_net_sharpe_minimum"])
        ),
        "maximum_single_positive_year_share_at_most_0p50": (
            annual["maximum_single_positive_year_share_of_positive_excess"]
            is not None
            and annual["maximum_single_positive_year_share_of_positive_excess"]
            <= float(
                rules["maximum_single_positive_year_share_of_positive_excess"]
            )
        ),
        "delete_any_calendar_year_excess_vs_buy_hold_positive": all(
            value > 0
            for value in annual[
                "delete_calendar_year_total_return_excess_vs_buy_hold"
            ].values()
        ),
        "deflated_sharpe_probability_at_least_0p95": (
            dsr["probability"] is not None
            and dsr["probability"]
            >= float(rules["deflated_sharpe_probability_minimum"])
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
        "stress": stress_summary,
        "buy_hold_510300": buy_hold_summary,
        "h00300_total_return_cagr": h00300_cagr,
        "maximum_drawdown_ratio_vs_buy_hold": drawdown_ratio,
        "late_temporal_holdout_net_sharpe": late_sharpe,
        "annual_contribution_analysis": annual,
        "deflated_sharpe": dsr,
        "gates": gates,
    }
    artifacts = {
        "target_table": targets,
        "base_ledger": base_ledger,
        "base_trades": base_trades,
        "stress_ledger": stress_ledger,
        "stress_trades": stress_trades,
        "buy_hold_ledger": buy_hold_ledger,
        "buy_hold_trades": buy_hold_trades,
    }
    return evaluation, artifacts


def _input_snapshot(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, specification in config["inputs"].items():
        path = _project_path(specification["path"])
        snapshot[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return snapshot


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    mechanism: dict[str, Any],
    portfolio: dict[str, Any] | None,
) -> dict[str, Any]:
    mechanism_passed = bool(mechanism["passed"])
    portfolio_evaluated = portfolio is not None
    if not mechanism_passed:
        status = mechanism["status"]
        return_evaluation = "NOT_ALLOWED"
        net_sharpe: float | str = "NOT_COMPUTED"
        historical_target = False
    else:
        if portfolio is None:
            raise ContractError("机制门通过后缺少组合评价")
        status = portfolio["status"]
        return_evaluation = "COMPLETED"
        net_sharpe = portfolio["base"]["sharpe_zero_cash_rate"]
        historical_target = bool(portfolio["passed"])
    return {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "candidate_model_id": config["protocol"]["candidate_model_id"],
        "status": status,
        "evidence_class": config["protocol"]["evidence_class"],
        "scope": {
            "execution_asset": config["scope"]["execution_asset"],
            "signal_asset": config["scope"]["signal_asset"],
            "allowed_holdings": config["scope"]["allowed_holdings"],
            "long_only": True,
            "leverage_allowed": False,
        },
        "literature_transfer": config["literature"],
        "evaluation_window": {
            "start": config["dates"]["evaluation_start"],
            "end": config["dates"]["evaluation_end"],
        },
        "freeze": {
            "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "frozen_at_asia_shanghai": manifest["frozen_at_asia_shanghai"],
            "candidate_2015_plus_outcomes_read_before_freeze": False,
            "portfolio_returns_read_before_freeze": False,
            "selection_bias_control": manifest["selection_bias_control"],
        },
        "input_snapshots": _input_snapshot(config),
        "mechanism_evaluation": mechanism,
        "mechanism_gate_passed": mechanism_passed,
        "portfolio_evaluated": portfolio_evaluated,
        "portfolio_evaluation": portfolio,
        "adjudication": {
            "return_evaluation": return_evaluation,
            "net_sharpe": net_sharpe,
            "target_net_sharpe": float(config["adjudication"]["target_net_sharpe"]),
            "historical_target_achieved": historical_target,
            "verified_forward_target_achieved": False,
            "goal_achieved": False,
            "reason": (
                "机制硬门未全部通过，冻结协议禁止组合收益评价"
                if not mechanism_passed
                else (
                    "历史组合全部硬门通过，仍需独立前向证据"
                    if historical_target
                    else "组合评价已完成但净费后与稳健性硬门未全部通过"
                )
            ),
        },
        "boundaries": {
            "parameter_rescue_after_result": "FORBIDDEN",
            "alternate_scale_or_weight_rescue_after_result": "FORBIDDEN",
            "coefficient_refit_or_direction_reversal_after_result": "FORBIDDEN",
            "combination_with_rejected_candidates": "FORBIDDEN",
            "paper_signal_allowed": False,
            "shadow_signal_allowed": False,
            "order_generation": False,
            "broker_connection": False,
            "position_change": False,
            "live_trading_authorized": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    mechanism = report["mechanism_evaluation"]
    full = mechanism["full_sample"]
    bootstrap = mechanism["bootstrap"]
    lines = [
        "# 510300 多尺度非线性趋势延续与反转 V1",
        "",
        f"- 最终状态：`{report['status']}`",
        f"- 机制门通过：`{str(report['mechanism_gate_passed']).lower()}`",
        f"- 组合评价执行：`{str(report['portfolio_evaluated']).lower()}`",
        f"- 收益评价：`{report['adjudication']['return_evaluation']}`",
        f"- 净夏普率：`{report['adjudication']['net_sharpe']}`",
        "- 实盘授权：`false`",
        "",
        "## 冻结机制结果",
        "",
        f"- 有效日：{full['valid_rows']}",
        f"- 线性系数：{full['linear_coefficient']:.8f}；Newey-West t={full['linear_newey_west_t']:.4f}",
        f"- 三次系数：{full['cubic_coefficient']:.8f}；Newey-West t={full['cubic_newey_west_t']:.4f}",
        f"- 线性系数区块Bootstrap 90%区间：{bootstrap['linear_coefficient_90pct']}",
        f"- 三次系数区块Bootstrap 90%区间：{bootstrap['cubic_coefficient_90pct']}",
        f"- 固定论文分数斜率：{full['fixed_score_slope']:.6f}；Newey-West t={full['fixed_score_newey_west_t']:.4f}",
        f"- 固定论文分数MSE相对零预测改善：{full['fixed_score_mse_improvement_vs_zero']:.8f}",
        "",
        "## 硬门",
        "",
    ]
    for name, passed in mechanism["gates"].items():
        lines.append(f"- `{name}`：`{str(passed).lower()}`")
    if report["portfolio_evaluation"] is not None:
        portfolio = report["portfolio_evaluation"]
        lines.extend(
            [
                "",
                "## 组合结果",
                "",
                f"- 基础成本净夏普：{portfolio['base']['sharpe_zero_cash_rate']}",
                f"- 压力成本净夏普：{portfolio['stress']['sharpe_zero_cash_rate']}",
                f"- 相对510300买入持有年化超额：{portfolio['base']['annualized_excess_vs_510300_buy_hold']}",
                f"- 相对H00300全收益年化超额：{portfolio['base']['annualized_excess_vs_h00300_total_return']}",
            ]
        )
    lines.extend(
        [
            "",
            "## 裁决",
            "",
            f"{report['adjudication']['reason']}。不得修改尺度、权重、系数、方向、仓位映射或与既有否决候选组合救援。",
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
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def run_study(*, write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    index_price, market, dividends, benchmark, _audits = load_inputs(config)
    panel = build_trend_panel(index_price, config)
    mechanism = evaluate_mechanism(panel, config)
    portfolio: dict[str, Any] | None = None
    portfolio_artifacts: dict[str, pd.DataFrame] = {}
    if mechanism["passed"]:
        portfolio, portfolio_artifacts = evaluate_portfolio(
            panel, market, dividends, benchmark, manifest, config
        )
    report = build_report(config, manifest, mechanism, portfolio)
    if write:
        paths = config["paths"]
        _atomic_parquet(
            _project_path(paths["mechanism_table"]),
            panel.loc[
                panel["date"].between(
                    pd.Timestamp(config["dates"]["evaluation_start"]),
                    pd.Timestamp(config["dates"]["evaluation_end"]),
                )
            ].copy(),
        )
        if mechanism["passed"]:
            for name, frame in portfolio_artifacts.items():
                _atomic_parquet(_project_path(paths[name]), frame)
        _atomic_json(_project_path(paths["result_json"]), report)
        _atomic_text(_project_path(paths["result_markdown"]), render_markdown(report))
    return report
