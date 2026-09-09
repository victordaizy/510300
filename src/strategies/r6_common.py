"""R6趋势、风险特征与非对称执行的公共实现。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr


TRADING_DAYS_PER_YEAR = 242


def _validate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "close"}
    if missing := required - set(daily.columns):
        raise ValueError(f"日线输入缺少字段：{sorted(missing)}")
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if data.empty or data["close"].isna().any() or (data["close"] <= 0).any():
        raise ValueError("日线收盘价为空或存在无效值")
    return data


def _past_z_score(series: pd.Series, window: int, minimum: int) -> pd.Series:
    """以严格早于当日的滚动分布标准化当日值。"""

    history = series.shift(1)
    mean = history.rolling(window, min_periods=minimum).mean()
    std = history.rolling(window, min_periods=minimum).std(ddof=1).replace(0.0, np.nan)
    return ((series - mean) / std).clip(-5.0, 5.0)


def _risk_percentile(series: pd.Series, window: int, minimum: int) -> pd.Series:
    return pd.Series(ndtr(_past_z_score(series, window, minimum)), index=series.index)


def build_regime_features(
    daily: pd.DataFrame,
    *,
    normalization_window: int = 756,
    minimum_history: int = 252,
) -> pd.DataFrame:
    """构造预注册TrendScore[-1,1]与RiskScore[0,1]，全程只用当日及过去数据。"""

    if normalization_window < minimum_history or minimum_history < 200:
        raise ValueError("状态标准化窗口必须不少于最小历史，且最小历史至少200日")
    data = _validate_daily(daily)
    data["return_1d"] = data["close"].pct_change(fill_method=None)
    data["log_return_1d"] = np.log(data["close"] / data["close"].shift(1))
    data["return_20d"] = data["close"].pct_change(20, fill_method=None)
    data["ma60"] = data["close"].rolling(60, min_periods=60).mean()
    data["ma200"] = data["close"].rolling(200, min_periods=200).mean()
    data["distance_ma200"] = np.log(data["close"] / data["ma200"])
    data["ma60_slope"] = np.log(data["ma60"] / data["ma60"].shift(20)) / 20.0
    data["ma200_slope"] = np.log(data["ma200"] / data["ma200"].shift(60)) / 60.0
    data["above_ma200_ratio"] = (
        data["close"].ge(data["ma200"]).astype(float).rolling(60, min_periods=60).mean()
    )
    data["trend_slope_change"] = data["ma60_slope"] - data["ma60_slope"].shift(20)

    trend_inputs = [
        "distance_ma200",
        "ma60_slope",
        "ma200_slope",
        "above_ma200_ratio",
        "trend_slope_change",
    ]
    trend_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    trend_components: list[pd.Series] = []
    for column in trend_inputs:
        z = _past_z_score(data[column], normalization_window, minimum_history)
        component = np.tanh(z / 2.0)
        data[f"{column}_score"] = component
        trend_components.append(component)
    data["trend_score"] = sum(
        weight * component for weight, component in zip(trend_weights, trend_components)
    ).clip(-1.0, 1.0)

    for label, days in (("rv5", 5), ("rv20", 20), ("rv60", 60)):
        data[label] = (
            data["log_return_1d"].rolling(days, min_periods=days).std(ddof=1)
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    downside_square = data["log_return_1d"].pow(2).where(
        data["log_return_1d"].lt(0.0), 0.0
    )
    data["downside_rv20"] = np.sqrt(
        downside_square.rolling(20, min_periods=20).mean() * TRADING_DAYS_PER_YEAR
    )
    data["rv5_rv20"] = data["rv5"] / data["rv20"].replace(0.0, np.nan)
    data["rv20_rv60"] = data["rv20"] / data["rv60"].replace(0.0, np.nan)
    data["downside_vol_ratio"] = data["downside_rv20"] / data["rv20"].replace(
        0.0, np.nan
    )
    data["drawdown"] = data["close"] / data["close"].cummax() - 1.0
    data["volatility_rising_positive_return"] = data["rv20"].gt(data["rv60"]) & data[
        "return_20d"
    ].gt(0.0)

    data["risk_short_shock"] = _risk_percentile(
        data["rv5_rv20"], normalization_window, minimum_history
    )
    data["risk_regime"] = _risk_percentile(
        data["rv20_rv60"], normalization_window, minimum_history
    )
    data["risk_level"] = _risk_percentile(
        data["rv20"], normalization_window, minimum_history
    )
    data["risk_downside"] = _risk_percentile(
        data["downside_vol_ratio"], normalization_window, minimum_history
    )
    data["risk_drawdown"] = _risk_percentile(
        -data["drawdown"], normalization_window, minimum_history
    )
    direction_penalty = data["risk_regime"].where(data["return_20d"].lt(0.0), 0.0)
    data["risk_direction"] = direction_penalty
    risk_columns = [
        "risk_short_shock",
        "risk_regime",
        "risk_level",
        "risk_downside",
        "risk_drawdown",
        "risk_direction",
    ]
    risk_weights = np.array([0.15, 0.15, 0.20, 0.20, 0.15, 0.15])
    data["risk_score"] = sum(
        weight * data[column] for weight, column in zip(risk_weights, risk_columns)
    ).clip(0.0, 1.0)
    data["crisis_event"] = data["return_1d"].le(-0.03) | (
        data["trend_score"].lt(0.0)
        & data["risk_score"].ge(0.75)
        & data["return_20d"].lt(0.0)
    )
    return data


def quantize_targets(position: pd.Series, step: float) -> pd.Series:
    if not 0.0 < step <= 1.0:
        raise ValueError("仓位档位必须位于(0,1]")
    return (np.floor(position.clip(0.0, 1.0) / step + 0.5) * step).clip(0.0, 1.0)


def schedule_asymmetric_targets(
    positions: pd.DataFrame,
    *,
    review_every: int = 5,
    maximum_increase: float = 0.10,
) -> pd.DataFrame:
    """普通目标每5日复核，风险恶化立即降仓，恢复每次最多10个百分点。"""

    required = {"date", "raw_target_position", "risk_score", "crisis_event"}
    if missing := required - set(positions.columns):
        raise ValueError(f"R6目标缺少字段：{sorted(missing)}")
    if review_every <= 0 or not 0.0 < maximum_increase <= 1.0:
        raise ValueError("复核间隔或恢复步长无效")
    result = positions.sort_values("date").reset_index(drop=True).copy()
    ready = result.index[result["raw_target_position"].notna()].tolist()
    target_values = np.full(len(result), np.nan)
    allowed = np.zeros(len(result), dtype=bool)
    risk_override = np.zeros(len(result), dtype=bool)
    review_day = np.zeros(len(result), dtype=bool)
    reasons = np.full(len(result), "数据不足", dtype=object)
    if not ready:
        result["target_position"] = target_values
        result["trade_allowed"] = allowed
        result["risk_off_override"] = risk_override
        result["is_review_day"] = review_day
        result["signal_reason"] = reasons
        return result

    first = ready[0]
    current = float(result.at[first, "raw_target_position"])
    previous_risk = float(result.at[first, "risk_score"])
    for index in range(first, len(result)):
        desired = result.at[index, "raw_target_position"]
        risk = result.at[index, "risk_score"]
        if pd.isna(desired) or pd.isna(risk):
            continue
        desired = float(desired)
        risk = float(risk)
        is_initial = index == first
        is_review = (index - first) % review_every == 0
        risk_worsened = risk >= previous_risk + 0.10
        immediate_reduction = desired < current - 1e-12 and (
            bool(result.at[index, "crisis_event"]) or risk_worsened
        )
        if is_initial:
            current = desired
            allowed[index] = True
            review_day[index] = True
            reasons[index] = "首次建仓"
        elif immediate_reduction:
            current = desired
            allowed[index] = True
            risk_override[index] = True
            reasons[index] = "风险恶化立即降仓"
        elif is_review:
            if desired > current:
                current = min(desired, current + maximum_increase)
                reasons[index] = "定期复核慢速恢复"
            elif desired < current:
                current = desired
                reasons[index] = "定期复核降仓"
            else:
                reasons[index] = "定期复核维持"
            allowed[index] = True
            review_day[index] = True
        else:
            reasons[index] = "非复核日维持"
        target_values[index] = current
        previous_risk = risk
    result["target_position"] = target_values
    result["trade_allowed"] = allowed
    result["risk_off_override"] = risk_override
    result["is_review_day"] = review_day
    result["signal_reason"] = reasons
    return result
