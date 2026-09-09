"""实验B：趋势—波动双状态仓位模型。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.r6_common import build_regime_features, schedule_asymmetric_targets


CONTINUOUS_MAPPINGS = {
    "CONTINUOUS_BASE": (0.50, 0.40, 0.50),
    "CONTINUOUS_TREND_LIGHT": (0.50, 0.30, 0.50),
    "CONTINUOUS_RISK_LIGHT": (0.50, 0.40, 0.40),
}


def _matrix_position(features: pd.DataFrame) -> pd.Series:
    trend = features["trend_score"]
    risk = features["risk_score"]
    positive_volatility = features["volatility_rising_positive_return"]
    target = pd.Series(0.50, index=features.index, dtype=float)
    target.loc[trend.ge(0.20) & risk.lt(0.65)] = 1.00
    target.loc[trend.ge(0.20) & risk.ge(0.65) & positive_volatility] = 0.75
    target.loc[trend.lt(-0.20) & risk.lt(0.70)] = 0.25
    target.loc[
        trend.lt(-0.20)
        & risk.ge(0.70)
        & features["risk_downside"].ge(0.65)
    ] = 0.00
    return target


def build_trend_vol_positions(
    daily: pd.DataFrame,
    *,
    mapping: str,
    normalization_window: int,
    review_every: int = 5,
    maximum_increase: float = 0.10,
) -> pd.DataFrame:
    if mapping not in {"MATRIX_BASE", *CONTINUOUS_MAPPINGS}:
        raise ValueError("趋势—波动映射不在预注册集合")
    features = build_regime_features(
        daily,
        normalization_window=normalization_window,
        minimum_history=252,
    )
    if mapping == "MATRIX_BASE":
        features["raw_target_position"] = _matrix_position(features)
        features.loc[features["trend_score"].isna() | features["risk_score"].isna(), "raw_target_position"] = np.nan
    else:
        intercept, trend_weight, risk_weight = CONTINUOUS_MAPPINGS[mapping]
        features["raw_target_position"] = (
            intercept
            + trend_weight * features["trend_score"]
            - risk_weight * features["risk_score"]
        ).clip(0.0, 1.0)
    return schedule_asymmetric_targets(
        features,
        review_every=review_every,
        maximum_increase=maximum_increase,
    )
