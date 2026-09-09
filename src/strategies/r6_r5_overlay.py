"""实验C：R5估值底仓与趋势—风险状态乘数的消融。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.r6_common import build_regime_features, schedule_asymmetric_targets


VALUATION_MODES = {"NONE", "BOUNDS_ONLY", "LOW_WEIGHT", "FULL_R5"}
MULTIPLIER_VARIANTS = {"BASE", "NO_STRONG_BOOST"}


def _state_multiplier(features: pd.DataFrame, variant: str) -> pd.Series:
    trend = features["trend_score"]
    risk = features["risk_score"]
    multiplier = pd.Series(1.0, index=features.index, dtype=float)
    multiplier.loc[trend.lt(-0.35) & risk.ge(0.65)] = 0.40
    multiplier.loc[
        multiplier.eq(1.0) & (trend.lt(-0.10) | risk.ge(0.65))
    ] = 0.70
    if variant == "BASE":
        multiplier.loc[trend.ge(0.35) & risk.lt(0.45)] = 1.15
    return multiplier


def build_r5_overlay_positions(
    daily: pd.DataFrame,
    r5_positions: pd.DataFrame,
    *,
    valuation_mode: str,
    multiplier_variant: str,
    normalization_window: int = 756,
    review_every: int = 5,
    maximum_increase: float = 0.10,
) -> pd.DataFrame:
    if valuation_mode not in VALUATION_MODES:
        raise ValueError("估值消融模式不在预注册集合")
    if multiplier_variant not in MULTIPLIER_VARIANTS:
        raise ValueError("状态乘数版本不在预注册集合")
    required = {"date", "r5_target_position"}
    if missing := required - set(r5_positions.columns):
        raise ValueError(f"R5底仓输入缺少字段：{sorted(missing)}")
    features = build_regime_features(
        daily,
        normalization_window=normalization_window,
        minimum_history=252,
    )
    base = r5_positions[["date", "r5_target_position"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    features = features.merge(base, on="date", how="left", validate="one_to_one")
    multiplier = _state_multiplier(features, multiplier_variant)
    regime_only = multiplier.clip(0.0, 1.0)
    r5 = features["r5_target_position"].clip(0.0, 1.0)
    if valuation_mode == "NONE":
        raw = regime_only
    elif valuation_mode == "BOUNDS_ONLY":
        lower = np.where(r5.ge(0.75), 0.25, 0.0)
        upper = np.where(r5.lt(0.50), 0.75, 1.0)
        raw = regime_only.clip(lower=pd.Series(lower, index=features.index), upper=pd.Series(upper, index=features.index))
    elif valuation_mode == "LOW_WEIGHT":
        raw = ((0.80 + 0.20 * r5) * multiplier).clip(0.0, 1.0)
    else:
        raw = (r5 * multiplier).clip(0.0, 1.0)
    features["regime_multiplier"] = multiplier
    features["raw_target_position"] = raw
    features.loc[r5.isna(), "raw_target_position"] = np.nan
    return schedule_asymmetric_targets(
        features,
        review_every=review_every,
        maximum_increase=maximum_increase,
    )
