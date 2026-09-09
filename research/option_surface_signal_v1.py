"""冻结的510300期权曲面O1至O4原始因子实现。"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "510300_option_surface_signal_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def black_forward_price(
    forward: float,
    strike: float,
    maturity: float,
    volatility: float,
    option_type: str,
    discount: float = 1.0,
) -> float:
    if min(forward, strike, maturity, volatility, discount) <= 0:
        raise ValueError("Black远期定价输入必须为正")
    root_variance = volatility * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * root_variance**2) / root_variance
    d2 = d1 - root_variance
    if option_type == "C":
        return float(discount * (forward * ndtr(d1) - strike * ndtr(d2)))
    if option_type == "P":
        return float(discount * (strike * ndtr(-d2) - forward * ndtr(-d1)))
    raise ValueError(f"未知期权类型：{option_type}")


def implied_volatility(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    option_type: str,
    discount: float,
    bounds: list[float],
) -> float:
    intrinsic = discount * (
        max(forward - strike, 0.0) if option_type == "C" else max(strike - forward, 0.0)
    )
    upper = discount * (forward if option_type == "C" else strike)
    tolerance = max(forward, strike) * 1e-12
    if not intrinsic + tolerance < price < upper - tolerance:
        raise ValueError("期权均价违反Black无套利价格边界")

    def objective(volatility: float) -> float:
        return (
            black_forward_price(
                forward, strike, maturity, volatility, option_type, discount
            )
            - price
        )

    lower, upper_vol = map(float, bounds)
    if objective(lower) * objective(upper_vol) > 0:
        raise ValueError("隐含波动率不在冻结求根边界内")
    return float(brentq(objective, lower, upper_vol, xtol=1e-12, rtol=1e-12))


def estimate_forward_discount(expiry_chain: pd.DataFrame, config: dict) -> dict[str, float]:
    pairs = expiry_chain.pivot_table(
        index="strike", columns="option_type", values="mid", aggfunc="first"
    ).dropna(subset=["C", "P"])
    pairs = pairs.sort_index()
    minimum = int(config["quote_filter"]["minimum_paired_strikes"])
    if len(pairs) < minimum:
        raise ValueError(f"认购认沽配对执行价不足{minimum}个")
    strikes = pairs.index.to_numpy(float)
    differences = (pairs["C"] - pairs["P"]).to_numpy(float)
    minimum_step = int(config["parity"]["minimum_strike_step_separation"])
    slopes = [
        (differences[j] - differences[i]) / (strikes[j] - strikes[i])
        for i in range(len(strikes))
        for j in range(i + minimum_step, len(strikes))
        if strikes[j] > strikes[i]
    ]
    if not slopes:
        raise ValueError("执行价间隔不足以估计认购认沽平价")
    discount = -float(np.median(slopes))
    lower, upper = map(float, config["parity"]["discount_factor_bounds"])
    if not lower <= discount <= upper:
        raise ValueError(f"平价折现因子越界：{discount:.6f}")
    forward = float(np.median(strikes + differences / discount))
    if forward <= 0:
        raise ValueError("平价远期价格非正")
    fitted = discount * (forward - strikes)
    normalized_rmse = float(np.sqrt(np.mean(np.square(differences - fitted))) / forward)
    if normalized_rmse > float(config["parity"]["maximum_normalized_rmse"]):
        raise ValueError(f"平价归一化RMSE超限：{normalized_rmse:.6f}")
    return {
        "forward": forward,
        "discount": discount,
        "parity_normalized_rmse": normalized_rmse,
        "paired_strikes": int(len(pairs)),
    }


def svi_total_variance(log_moneyness: np.ndarray | float, parameters: np.ndarray) -> np.ndarray:
    a, b, rho, m, sigma = parameters
    k = np.asarray(log_moneyness, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt(np.square(k - m) + sigma**2))


def _prepare_expiry_chain(snapshot: pd.DataFrame, expiry: pd.Timestamp, config: dict) -> pd.DataFrame:
    data = snapshot.loc[pd.to_datetime(snapshot["expiry_date"]).dt.normalize().eq(expiry)].copy()
    data["bid1"] = pd.to_numeric(data["bid1"], errors="coerce")
    data["ask1"] = pd.to_numeric(data["ask1"], errors="coerce")
    data["strike"] = pd.to_numeric(data["strike"], errors="coerce")
    data = data.loc[
        data["option_type"].isin(["C", "P"])
        & data["bid1"].gt(0)
        & data["ask1"].gt(0)
        & data["bid1"].le(data["ask1"])
        & data["strike"].gt(0)
    ].copy()
    if config["quote_filter"]["standard_contracts_only"]:
        data = data.loc[~data["is_adjusted"].astype(bool)].copy()
    data["mid"] = (data["bid1"] + data["ask1"]) / 2.0
    if data.duplicated(["option_type", "strike"]).any():
        raise ValueError("同一到期日存在重复期权类型与执行价")
    return data.sort_values(["strike", "option_type"]).reset_index(drop=True)


def _otm_observations(
    expiry_chain: pd.DataFrame,
    forward: float,
    discount: float,
    maturity: float,
    config: dict,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for strike, group in expiry_chain.groupby("strike", sort=True):
        strike_value = float(strike)
        preferred = "P" if strike_value < forward else "C"
        selected = group.loc[group["option_type"].eq(preferred)]
        if selected.empty and math.isclose(strike_value, forward, rel_tol=0.0, abs_tol=1e-12):
            selected = group.iloc[[0]]
        if selected.empty:
            continue
        row = selected.iloc[0]
        try:
            volatility = implied_volatility(
                float(row["mid"]),
                forward,
                strike_value,
                maturity,
                str(row["option_type"]),
                discount,
                config["svi"]["implied_volatility_bounds"],
            )
        except ValueError:
            continue
        rows.append(
            {
                "strike": strike_value,
                "option_type": str(row["option_type"]),
                "log_moneyness": math.log(strike_value / forward),
                "implied_volatility": volatility,
                "total_variance": volatility**2 * maturity,
            }
        )
    result = pd.DataFrame(rows)
    minimum = int(config["quote_filter"]["minimum_otm_points_per_expiry"])
    if len(result) < minimum:
        raise ValueError(f"有效价外隐波点不足{minimum}个")
    return result.sort_values("strike").reset_index(drop=True)


def _check_svi_arbitrage(parameters: np.ndarray, maturity: float, config: dict) -> None:
    start, end = map(float, config["svi"]["arbitrage_grid_log_moneyness"])
    points = int(config["svi"]["arbitrage_grid_points"])
    grid = np.linspace(start, end, points)
    variance = svi_total_variance(grid, parameters)
    if not np.isfinite(variance).all() or np.min(variance) <= 0:
        raise ValueError("SVI网格存在非正或非有限总方差")
    strikes = np.exp(grid)
    calls = np.array(
        [
            black_forward_price(1.0, float(strike), maturity, math.sqrt(float(w) / maturity), "C")
            for strike, w in zip(strikes, variance)
        ]
    )
    slopes = np.diff(calls) / np.diff(strikes)
    minimum_convexity = float(np.min(np.diff(slopes)))
    tolerance = float(config["svi"]["maximum_negative_call_convexity"])
    if minimum_convexity < -tolerance:
        raise ValueError(f"SVI认购价格违反凸性：{minimum_convexity:.3e}")


def fit_svi_expiry(
    snapshot: pd.DataFrame,
    signal_date: pd.Timestamp,
    expiry: pd.Timestamp,
    config: dict,
) -> dict[str, Any]:
    days = int((expiry.normalize() - signal_date.normalize()).days)
    minimum_days = int(config["clock"]["minimum_calendar_days_to_expiry"])
    maximum_days = int(config["clock"]["maximum_calendar_days_to_expiry"])
    if not minimum_days <= days <= maximum_days:
        raise ValueError("到期日不在冻结期限范围")
    maturity = days / float(config["clock"]["day_count_basis"])
    chain = _prepare_expiry_chain(snapshot, expiry, config)
    parity = estimate_forward_discount(chain, config)
    observations = _otm_observations(
        chain, parity["forward"], parity["discount"], maturity, config
    )
    k = observations["log_moneyness"].to_numpy(float)
    observed_w = observations["total_variance"].to_numpy(float)
    lower = np.asarray(config["svi"]["parameter_lower_bounds"], dtype=float)
    upper = np.asarray(config["svi"]["parameter_upper_bounds"], dtype=float)
    best = None
    for initial in config["svi"]["fixed_initial_points"]:
        x0 = np.clip(np.asarray(initial, dtype=float), lower + 1e-10, upper - 1e-10)
        fit = least_squares(
            lambda parameters: svi_total_variance(k, parameters) - observed_w,
            x0=x0,
            bounds=(lower, upper),
            loss="linear",
            max_nfev=int(config["svi"]["maximum_function_evaluations"]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        squared_error = float(np.sum(np.square(fit.fun)))
        if fit.success and (best is None or squared_error < best[0]):
            best = (squared_error, fit.x)
    if best is None:
        raise ValueError("所有冻结SVI初值均拟合失败")
    parameters = np.asarray(best[1], dtype=float)
    fitted_w = svi_total_variance(k, parameters)
    fitted_iv = np.sqrt(fitted_w / maturity)
    iv_rmse = float(
        np.sqrt(np.mean(np.square(fitted_iv - observations["implied_volatility"].to_numpy(float))))
    )
    if iv_rmse > float(config["svi"]["maximum_iv_rmse"]):
        raise ValueError(f"SVI隐波RMSE超限：{iv_rmse:.6f}")
    _check_svi_arbitrage(parameters, maturity, config)
    model_free_variance = model_free_variance_expiry(
        chain, parity["forward"], parity["discount"], maturity, config
    )
    return {
        "expiry_date": expiry.normalize(),
        "calendar_days": days,
        "maturity": maturity,
        **parity,
        "parameters": parameters,
        "iv_rmse": iv_rmse,
        "otm_observations": int(len(observations)),
        "model_free_variance": model_free_variance,
    }


def model_free_variance_expiry(
    chain: pd.DataFrame,
    forward: float,
    discount: float,
    maturity: float,
    config: dict,
) -> float:
    pivot = chain.pivot_table(
        index="strike", columns="option_type", values="mid", aggfunc="first"
    ).sort_index()
    strikes = pivot.index.to_numpy(float)
    below = strikes[strikes <= forward]
    if not len(below):
        raise ValueError("没有不高于远期价的执行价K0")
    k0 = float(np.max(below))
    rows: list[tuple[float, float]] = []
    puts_below = 0
    calls_above = 0
    for strike, row in pivot.iterrows():
        strike_value = float(strike)
        if strike_value < k0 and pd.notna(row.get("P")):
            rows.append((strike_value, float(row["P"])))
            puts_below += 1
        elif strike_value > k0 and pd.notna(row.get("C")):
            rows.append((strike_value, float(row["C"])))
            calls_above += 1
        elif strike_value == k0 and pd.notna(row.get("C")) and pd.notna(row.get("P")):
            rows.append((strike_value, float((row["C"] + row["P"]) / 2.0)))
    if puts_below < int(config["quote_filter"]["minimum_put_strikes_below_forward"]):
        raise ValueError("远期价下方认沽执行价不足")
    if calls_above < int(config["quote_filter"]["minimum_call_strikes_above_forward"]):
        raise ValueError("远期价上方认购执行价不足")
    rows.sort()
    selected_strikes = np.asarray([item[0] for item in rows], dtype=float)
    prices = np.asarray([item[1] for item in rows], dtype=float)
    if len(selected_strikes) < 3 or np.any(np.diff(selected_strikes) <= 0):
        raise ValueError("模型无关方差执行价网格无效")
    delta_k = np.empty_like(selected_strikes)
    delta_k[0] = selected_strikes[1] - selected_strikes[0]
    delta_k[-1] = selected_strikes[-1] - selected_strikes[-2]
    delta_k[1:-1] = (selected_strikes[2:] - selected_strikes[:-2]) / 2.0
    variance = (
        2.0
        / (maturity * discount)
        * float(np.sum(delta_k / np.square(selected_strikes) * prices))
        - (1.0 / maturity) * (forward / k0 - 1.0) ** 2
    )
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("模型无关隐含方差非正或无效")
    return float(variance)


def build_slices(snapshot: pd.DataFrame, signal_date: pd.Timestamp, config: dict) -> list[dict[str, Any]]:
    required = {
        "expiry_date",
        "option_type",
        "strike",
        "bid1",
        "ask1",
        "is_adjusted",
    }
    missing = sorted(required.difference(snapshot.columns))
    if missing:
        raise ValueError(f"期权快照缺少曲面字段：{missing}")
    expiries = sorted(pd.to_datetime(snapshot["expiry_date"], errors="coerce").dropna().dt.normalize().unique())
    slices: list[dict[str, Any]] = []
    failures: list[str] = []
    for raw_expiry in expiries:
        expiry = pd.Timestamp(raw_expiry)
        days = int((expiry - signal_date.normalize()).days)
        if not int(config["clock"]["minimum_calendar_days_to_expiry"]) <= days <= int(
            config["clock"]["maximum_calendar_days_to_expiry"]
        ):
            continue
        try:
            slices.append(fit_svi_expiry(snapshot, signal_date, expiry, config))
        except ValueError as exc:
            failures.append(f"{expiry.date()}:{exc}")
    if len(slices) < 2:
        raise ValueError(f"有效期限切片不足2个；失败={failures}")
    slices.sort(key=lambda item: item["maturity"])
    grid_start, grid_end = map(float, config["svi"]["arbitrage_grid_log_moneyness"])
    grid = np.linspace(grid_start, grid_end, int(config["svi"]["arbitrage_grid_points"]))
    for left, right in zip(slices, slices[1:]):
        left_w = svi_total_variance(grid, left["parameters"])
        right_w = svi_total_variance(grid, right["parameters"])
        if np.any(right_w + 1e-10 < left_w):
            raise ValueError("相邻SVI期限存在日历价差套利")
    return slices


def _bracket_slices(
    slices: list[dict[str, Any]], target_days: int, config: dict
) -> tuple[dict[str, Any], dict[str, Any], float]:
    target = target_days / float(config["clock"]["day_count_basis"])
    exact = [item for item in slices if int(item["calendar_days"]) == int(target_days)]
    if exact:
        return exact[0], exact[0], 0.0
    lower = [item for item in slices if item["maturity"] < target]
    upper = [item for item in slices if item["maturity"] > target]
    if not lower or not upper:
        raise ValueError(f"固定{target_days}日期限缺少双侧真实到期日，禁止外推")
    left = max(lower, key=lambda item: item["maturity"])
    right = min(upper, key=lambda item: item["maturity"])
    weight = (target - left["maturity"]) / (right["maturity"] - left["maturity"])
    return left, right, float(weight)


def fixed_total_variance(
    slices: list[dict[str, Any]], target_days: int, log_moneyness: float, config: dict
) -> float:
    left, right, weight = _bracket_slices(slices, target_days, config)
    left_w = float(svi_total_variance(log_moneyness, left["parameters"]))
    if left is right:
        return left_w
    right_w = float(svi_total_variance(log_moneyness, right["parameters"]))
    if right_w + 1e-10 < left_w:
        raise ValueError("固定期限插值遇到日历方差倒挂")
    return float(left_w + weight * (right_w - left_w))


def fixed_implied_volatility(
    slices: list[dict[str, Any]], target_days: int, log_moneyness: float, config: dict
) -> float:
    maturity = target_days / float(config["clock"]["day_count_basis"])
    total_variance = fixed_total_variance(slices, target_days, log_moneyness, config)
    return float(math.sqrt(total_variance / maturity))


def fixed_model_free_variance(
    slices: list[dict[str, Any]], target_days: int, config: dict
) -> float:
    left, right, weight = _bracket_slices(slices, target_days, config)
    left_total = float(left["model_free_variance"] * left["maturity"])
    if left is right:
        return float(left["model_free_variance"])
    right_total = float(right["model_free_variance"] * right["maturity"])
    if right_total + 1e-10 < left_total:
        raise ValueError("模型无关总方差期限倒挂")
    target_maturity = target_days / float(config["clock"]["day_count_basis"])
    return float((left_total + weight * (right_total - left_total)) / target_maturity)


def delta_implied_volatility(
    slices: list[dict[str, Any]],
    target_days: int,
    option_type: str,
    absolute_delta: float,
    config: dict,
) -> float:
    maturity = target_days / float(config["clock"]["day_count_basis"])
    lower, upper = map(float, config["factors"]["O2"]["root_log_moneyness_bounds"])
    if option_type == "C":
        bracket = (0.0, upper)
        target_delta = absolute_delta
    elif option_type == "P":
        bracket = (lower, 0.0)
        target_delta = -absolute_delta
    else:
        raise ValueError(f"未知期权类型：{option_type}")

    def objective(k: float) -> float:
        total_variance = fixed_total_variance(slices, target_days, k, config)
        root_variance = math.sqrt(total_variance)
        d1 = (-k + 0.5 * total_variance) / root_variance
        delta = float(ndtr(d1)) if option_type == "C" else float(ndtr(d1) - 1.0)
        return delta - target_delta

    if objective(bracket[0]) * objective(bracket[1]) > 0:
        raise ValueError(f"{target_days}日{option_type}的Delta求根没有括号")
    k = float(brentq(objective, bracket[0], bracket[1], xtol=1e-12, rtol=1e-12))
    return fixed_implied_volatility(slices, target_days, k, config)


def left_tail_probability(slices: list[dict[str, Any]], config: dict) -> float:
    target_days = int(config["fixed_maturity"]["primary_calendar_days"])
    maturity = target_days / float(config["clock"]["day_count_basis"])
    ratio = float(config["factors"]["O1"]["strike_forward_ratio"])
    step = float(config["factors"]["O1"]["cdf_derivative_step_ratio"])

    def normalized_call(strike_ratio: float) -> float:
        k = math.log(strike_ratio)
        volatility = fixed_implied_volatility(slices, target_days, k, config)
        return black_forward_price(1.0, strike_ratio, maturity, volatility, "C")

    derivative = (normalized_call(ratio + step) - normalized_call(ratio - step)) / (2.0 * step)
    probability = 1.0 + derivative
    if not -1e-6 <= probability <= 1.0 + 1e-6:
        raise ValueError(f"风险中性左尾概率越界：{probability:.6f}")
    return float(np.clip(probability, 0.0, 1.0))


def total_return_variance(market: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "close"}
    missing = sorted(required.difference(market.columns))
    if missing:
        raise ValueError(f"510300日线缺少字段：{missing}")
    data = market[["date", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna().sort_values("date")
    if data["date"].duplicated().any():
        raise ValueError("510300日线存在重复日期")
    if data["close"].le(0).any():
        raise ValueError("510300日线存在非正收盘价")
    cash = dividends.loc[dividends["symbol"].astype(str).eq("510300.SH")].copy()
    cash["ex_date"] = pd.to_datetime(cash["ex_date"], errors="coerce").dt.normalize()
    cash["cash_dividend_per_share"] = pd.to_numeric(
        cash["cash_dividend_per_share"], errors="coerce"
    )
    if cash[["ex_date", "cash_dividend_per_share"]].isna().any(axis=None):
        raise ValueError("510300分红表存在无效值")
    if cash["cash_dividend_per_share"].lt(0).any():
        raise ValueError("510300分红表存在负现金分红")
    cash = cash.groupby("ex_date", as_index=False)["cash_dividend_per_share"].sum()
    data = data.merge(cash, left_on="date", right_on="ex_date", how="left")
    data["cash_dividend_per_share"] = data["cash_dividend_per_share"].fillna(0.0)
    data["total_return"] = (
        (data["close"] + data["cash_dividend_per_share"]) / data["close"].shift(1) - 1.0
    )
    return data[["date", "total_return"]].dropna().reset_index(drop=True)


def har_forecast(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    signal_date: pd.Timestamp,
    config: dict,
) -> dict[str, Any]:
    rules = config["factors"]["O4"]
    data = total_return_variance(market, dividends)
    signal_date = signal_date.normalize()
    data = data.loc[data["date"].le(signal_date)].copy().reset_index(drop=True)
    if data.empty or pd.Timestamp(data["date"].iloc[-1]) != signal_date:
        raise ValueError("HAR输入没有覆盖信号日收盘")
    annualization = int(rules["annualization_trading_days"])
    floor = float(rules["variance_floor"])
    data["rv"] = annualization * np.square(np.log1p(data["total_return"].clip(lower=-0.999999)))
    lags = list(map(int, rules["har_lags_trading_days"]))
    horizon = int(rules["realized_variance_horizon_trading_days"])
    for lag in lags:
        data[f"rv_{lag}"] = data["rv"].rolling(lag).mean()
    reverse = data["rv"].iloc[::-1].rolling(horizon).mean().iloc[::-1]
    data["future_rv"] = reverse.shift(-1)
    feature_columns = [f"rv_{lag}" for lag in lags]
    training = data.iloc[:-horizon].dropna(subset=feature_columns + ["future_rv"]).copy()
    minimum = int(rules["minimum_mature_training_rows"])
    if len(training) < minimum:
        raise ValueError(f"HAR已成熟训练行不足{minimum}条")
    x = np.column_stack(
        [
            np.ones(len(training)),
            *[np.log(training[column].clip(lower=floor).to_numpy(float)) for column in feature_columns],
        ]
    )
    y = np.log(training["future_rv"].clip(lower=floor).to_numpy(float))
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise ValueError("HAR扩展OLS设计矩阵不满秩")
    latest = data.iloc[-1]
    if latest[feature_columns].isna().any():
        raise ValueError("HAR信号日特征未成熟")
    score = np.array(
        [1.0, *[math.log(max(float(latest[column]), floor)) for column in feature_columns]]
    )
    forecast = float(math.exp(float(score @ coefficients)))
    if not np.isfinite(forecast) or forecast <= 0:
        raise ValueError("HAR预测方差无效")
    return {
        "forecast_variance20": forecast,
        "training_rows": int(len(training)),
        "coefficients": coefficients.tolist(),
        "latest_features": {column: float(latest[column]) for column in feature_columns},
    }


def compute_raw_factors(
    snapshot: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    signal_date: pd.Timestamp,
    config: dict,
) -> dict[str, Any]:
    slices = build_slices(snapshot, signal_date, config)
    delta = float(config["factors"]["O2"]["absolute_forward_delta"])
    put25 = delta_implied_volatility(slices, 25, "P", delta, config)
    call25 = delta_implied_volatility(slices, 25, "C", delta, config)
    put60 = delta_implied_volatility(slices, 60, "P", delta, config)
    implied25 = fixed_model_free_variance(slices, 25, config)
    har = har_forecast(market, dividends, signal_date, config)
    return {
        "signal_date": signal_date.normalize(),
        "O1_left_tail_probability_25d": left_tail_probability(slices, config),
        "O2_put_call_skew_25d": put25 - call25,
        "O3_downside_term_structure": put25 - put60,
        "O4_variance_risk_premium": implied25 - har["forecast_variance20"],
        "put25_iv": put25,
        "call25_iv": call25,
        "put60_iv": put60,
        "implied_variance25": implied25,
        "har_forecast_variance20": har["forecast_variance20"],
        "har_training_rows": har["training_rows"],
        "slice_count": len(slices),
        "slice_expiries": [str(item["expiry_date"].date()) for item in slices],
        "formula_version": config["protocol"]["project_id"],
        "research_output": "RAW_FACTOR_AUDIT_ONLY",
    }


def _write_status(payload: dict, config: dict) -> None:
    path = ROOT / config["paths"]["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.freeze_510300_option_surface_signal_v1 import verify_protocol

    config, manifest = verify_protocol()
    audit_path = ROOT / config["paths"]["orderbook_audit"]
    generated_at = datetime.now(TIMEZONE).isoformat()
    if not audit_path.exists():
        payload = {"status": "NO_VIEW_ORDERBOOK_AUDIT_MISSING", "generated_at": generated_at}
        _write_status(payload, config)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required_status = config["gates"]["required_orderbook_audit_status"]
    if audit.get("status") != required_status or not audit.get("return_test_authorized", False):
        payload = {
            "status": "NO_VIEW_ORDERBOOK_DATA_GATE",
            "generated_at": generated_at,
            "required_audit_status": required_status,
            "actual_audit_status": audit.get("status"),
            "eligible_orderbook_days": audit.get("eligible_orderbook_days", 0),
            "formula_manifest_frozen_at": manifest["frozen_at"],
        }
        _write_status(payload, config)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    market = pd.read_parquet(ROOT / "data/raw/market/510300_daily_raw.parquet")
    dividends = pd.read_csv(ROOT / "data/reference/510300_dividends.csv")
    rows: list[dict[str, Any]] = []
    for daily in audit["daily_audits"]:
        if daily["status"] != "PASS":
            continue
        date = pd.Timestamp(daily["date"])
        snapshot = pd.read_parquet(ROOT / daily["snapshot"])
        try:
            rows.append(compute_raw_factors(snapshot, market, dividends, date, config))
        except Exception as exc:
            rows.append(
                {
                    "signal_date": date,
                    "status": "NO_VIEW_FACTOR_CALCULATION_FAILURE",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "formula_version": config["protocol"]["project_id"],
                }
            )
    output = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    output_path = ROOT / config["paths"]["raw_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    valid = int(output.get("O1_left_tail_probability_25d", pd.Series(dtype=float)).notna().sum())
    payload = {
        "status": "RAW_FACTOR_AUDIT_COMPLETE",
        "generated_at": generated_at,
        "rows": int(len(output)),
        "valid_factor_rows": valid,
        "minimum_scored_days_before_factor_evaluation": int(
            config["gates"]["minimum_scored_days_before_factor_evaluation"]
        ),
        "factor_evaluation_authorized": valid
        >= int(config["gates"]["minimum_scored_days_before_factor_evaluation"]),
        "return_test_authorized": False,
        "position_mapping": "DISABLED",
        "order_generation": "DISABLED",
        "live_trading": "NOT_AUTHORIZED",
        "output": output_path.relative_to(ROOT).as_posix(),
    }
    _write_status(payload, config)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
