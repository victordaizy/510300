"""510300防守择时基线与估值增强仓位模型。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _scaled_stress(value: pd.Series, start: float, full: float) -> pd.Series:
    if full <= start:
        raise ValueError("风险压力满值阈值必须高于起始阈值")
    return ((value - start) / (full - start)).clip(0.0, 1.0)


def build_timing_features(daily: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """只使用当日收盘及此前510300日线构造趋势和非对称风险特征。"""

    required = {"date", "close"}
    if missing := required - set(daily.columns):
        raise ValueError(f"日线数据缺少字段：{sorted(missing)}")
    data = daily[["date", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if data["close"].isna().any() or (data["close"] <= 0).any():
        raise ValueError("收盘价存在空值、零值或负值")

    trend = config["trend"]
    risk = config["risk"]
    trading_days = int(config["backtest"]["trading_days_per_year"])
    fast_days = int(trend["fast_ma_days"])
    slow_days = int(trend["slow_ma_days"])
    rv_short_days = int(risk["rv_short_days"])
    rv_medium_days = int(risk["rv_medium_days"])
    rv_long_days = int(risk["rv_long_days"])
    if not 1 < rv_short_days < rv_medium_days < rv_long_days <= slow_days:
        raise ValueError("波动率窗口必须满足1 < 短期 < 中期 < 长期 <= 慢均线")
    if not 1 < fast_days < slow_days:
        raise ValueError("趋势窗口必须满足1 < 快均线 < 慢均线")

    data["log_return_1d"] = np.log(data["close"] / data["close"].shift(1))
    data["simple_return_1d"] = data["close"].pct_change(fill_method=None)
    data["ma_fast"] = data["close"].rolling(fast_days, min_periods=fast_days).mean()
    data["ma_slow"] = data["close"].rolling(slow_days, min_periods=slow_days).mean()
    for label, days in (
        ("rv_short", rv_short_days),
        ("rv_medium", rv_medium_days),
        ("rv_long", rv_long_days),
    ):
        data[label] = (
            data["log_return_1d"].rolling(days, min_periods=days).std(ddof=1)
            * np.sqrt(trading_days)
        )

    squared_down_returns = data["log_return_1d"].pow(2).where(
        data["log_return_1d"].lt(0.0), 0.0
    )
    data["downside_rv_20"] = np.sqrt(
        trading_days
        * squared_down_returns.rolling(rv_medium_days, min_periods=rv_medium_days).mean()
    )
    data["rv5_rv20_ratio"] = data["rv_short"] / data["rv_medium"].replace(0.0, np.nan)
    data["rv20_rv60_ratio"] = data["rv_medium"] / data["rv_long"].replace(0.0, np.nan)
    data["downside_vol_ratio"] = (
        data["downside_rv_20"] / data["rv_medium"].replace(0.0, np.nan)
    )

    fast_scale = data["rv_medium"] * np.sqrt(fast_days / trading_days)
    slow_scale = data["rv_long"] * np.sqrt((slow_days - fast_days) / trading_days)
    data["trend_fast_score"] = np.tanh(
        np.log(data["close"] / data["ma_fast"]) / fast_scale.replace(0.0, np.nan)
    )
    data["trend_slow_score"] = np.tanh(
        np.log(data["ma_fast"] / data["ma_slow"]) / slow_scale.replace(0.0, np.nan)
    )
    fast_weight = float(trend["fast_weight"])
    slow_weight = float(trend["slow_weight"])
    if not np.isclose(fast_weight + slow_weight, 1.0):
        raise ValueError("快慢趋势权重之和必须等于1")
    data["trend_score"] = (
        fast_weight * data["trend_fast_score"]
        + slow_weight * data["trend_slow_score"]
    ).clip(-1.0, 1.0)

    data["risk_shock_score"] = _scaled_stress(
        data["rv5_rv20_ratio"],
        float(risk["shock_ratio_start"]),
        float(risk["shock_ratio_full"]),
    )
    data["risk_regime_score"] = _scaled_stress(
        data["rv20_rv60_ratio"],
        float(risk["regime_ratio_start"]),
        float(risk["regime_ratio_full"]),
    )
    data["risk_downside_score"] = _scaled_stress(
        data["downside_vol_ratio"],
        float(risk["downside_ratio_start"]),
        float(risk["downside_ratio_full"]),
    )
    stress_weights = np.array(
        [
            float(risk["shock_weight"]),
            float(risk["regime_weight"]),
            float(risk["downside_weight"]),
        ]
    )
    if not np.isclose(stress_weights.sum(), 1.0):
        raise ValueError("三项风险压力权重之和必须等于1")
    data["risk_stress_score"] = (
        stress_weights[0] * data["risk_shock_score"]
        + stress_weights[1] * data["risk_regime_score"]
        + stress_weights[2] * data["risk_downside_score"]
    ).clip(0.0, 1.0)
    data["risk_penalty"] = float(risk["maximum_penalty"]) * data["risk_stress_score"]
    data["crisis_event"] = data["simple_return_1d"].le(
        float(risk["crisis_daily_return"])
    ) | (
        data["rv5_rv20_ratio"].ge(float(risk["crisis_rv5_rv20_ratio"]))
        & data["downside_vol_ratio"].ge(float(risk["crisis_downside_ratio"]))
        & data["trend_score"].lt(0.0)
    )
    return data


def build_model_positions(
    timing: pd.DataFrame,
    config: dict[str, Any],
    valuation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """生成防守基线，或加入估值战略仓位后的增强模型目标。"""

    required = {"date", "trend_score", "risk_penalty", "crisis_event"}
    if missing := required - set(timing.columns):
        raise ValueError(f"择时特征缺少字段：{sorted(missing)}")
    result = timing.copy()
    if valuation is None:
        trend = config["trend"]
        result["strategic_position"] = np.nan
        result["position_before_risk"] = (
            float(trend["baseline_mid_position"])
            + float(trend["baseline_trend_amplitude"]) * result["trend_score"]
        ).clip(0.0, 1.0)
        result["model_name"] = "防守型趋势择时基线"
    else:
        if missing := {"date", "raw_continuous_position"} - set(valuation.columns):
            raise ValueError(f"估值特征缺少字段：{sorted(missing)}")
        value = valuation[["date", "raw_continuous_position"]].copy()
        value["date"] = pd.to_datetime(value["date"])
        result = result.merge(value, on="date", how="left", validate="one_to_one")
        valuation_config = config["valuation"]
        minimum = float(valuation_config["minimum_strategic_position"])
        result["strategic_position"] = minimum + (1.0 - minimum) * result[
            "raw_continuous_position"
        ]
        result["position_before_risk"] = (
            result["strategic_position"]
            + float(valuation_config["trend_confirmation_amplitude"])
            * result["trend_score"]
        ).clip(0.0, 1.0)
        result["model_name"] = "估值战略仓位增强"

    result["continuous_model_position"] = (
        result["position_before_risk"] - result["risk_penalty"]
    ).clip(0.0, 1.0)
    grid = float(config["execution"]["position_grid_step"])
    if not 0.0 < grid <= 1.0:
        raise ValueError("仓位离散步长必须位于0到1之间")
    result["discrete_model_position"] = (
        np.floor(result["continuous_model_position"] / grid + 0.5) * grid
    ).clip(0.0, 1.0)
    return result


def schedule_asymmetric_execution(
    model: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """普通信号按周调整，风险事件立即降仓，恢复仓位按固定步长慢速进行。"""

    required = {"date", "discrete_model_position", "crisis_event"}
    if missing := required - set(model.columns):
        raise ValueError(f"模型仓位缺少字段：{sorted(missing)}")
    result = model.sort_values("date").reset_index(drop=True).copy()
    review_every = int(config["execution"]["normal_review_every_trading_days"])
    risk_on_step = float(config["execution"]["risk_on_maximum_step"])
    if review_every <= 0 or not 0.0 < risk_on_step <= 1.0:
        raise ValueError("普通复核间隔或风险恢复步长无效")

    ready_indices = result.index[result["discrete_model_position"].notna()].to_list()
    scheduled = np.full(len(result), np.nan)
    trade_allowed = np.zeros(len(result), dtype=bool)
    risk_override = np.zeros(len(result), dtype=bool)
    review_day = np.zeros(len(result), dtype=bool)
    allow_increase = np.zeros(len(result), dtype=bool)
    allow_decrease = np.zeros(len(result), dtype=bool)
    reasons = np.full(len(result), "数据不足", dtype=object)
    if not ready_indices:
        result["target_position"] = scheduled
        result["trade_allowed"] = trade_allowed
        result["risk_off_override"] = risk_override
        result["is_review_day"] = review_day
        result["allow_position_increase"] = allow_increase
        result["allow_position_decrease"] = allow_decrease
        result["signal_reason"] = reasons
        return result

    first_ready = ready_indices[0]
    current = float(result.at[first_ready, "discrete_model_position"])
    for index in range(first_ready, len(result)):
        desired = result.at[index, "discrete_model_position"]
        if pd.isna(desired):
            continue
        desired = float(desired)
        is_initial = index == first_ready
        is_review = (index - first_ready) % review_every == 0
        is_crisis_reduction = bool(result.at[index, "crisis_event"]) and desired < current - 1e-12
        if is_initial:
            current = desired
            trade_allowed[index] = True
            review_day[index] = True
            allow_increase[index] = True
            allow_decrease[index] = True
            reasons[index] = "首次建仓"
        elif is_crisis_reduction:
            current = desired
            trade_allowed[index] = True
            risk_override[index] = True
            allow_decrease[index] = True
            reasons[index] = "风险事件立即降仓"
        elif is_review:
            if desired > current:
                current = min(desired, current + risk_on_step)
                reasons[index] = "周度复核慢速恢复"
            elif desired < current:
                current = desired
                reasons[index] = "周度复核降仓"
            else:
                reasons[index] = "周度复核维持"
            trade_allowed[index] = True
            review_day[index] = True
            allow_increase[index] = True
            allow_decrease[index] = True
        else:
            reasons[index] = "非复核日维持"
        scheduled[index] = current

    result["target_position"] = scheduled
    result["trade_allowed"] = trade_allowed
    result["risk_off_override"] = risk_override
    result["is_review_day"] = review_day
    result["allow_position_increase"] = allow_increase
    result["allow_position_decrease"] = allow_decrease
    result["signal_reason"] = reasons
    return result
