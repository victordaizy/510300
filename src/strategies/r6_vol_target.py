"""实验A：不加杠杆的纯风险目标模型。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.r6_common import TRADING_DAYS_PER_YEAR, schedule_asymmetric_targets


def build_vol_target_positions(
    daily: pd.DataFrame,
    *,
    target_volatility: float,
    estimator: str,
    review_every: int = 5,
    maximum_increase: float = 0.10,
) -> pd.DataFrame:
    if target_volatility not in {0.08, 0.10, 0.12}:
        raise ValueError("目标波动率不在预注册集合{8%,10%,12%}")
    if estimator not in {"RV20", "RV60", "RV20_RV60_BLEND"}:
        raise ValueError("波动估计器不在预注册集合")
    data = daily[["date", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    log_return = np.log(data["close"] / data["close"].shift(1))
    data["rv20"] = log_return.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    data["rv60"] = log_return.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    if estimator == "RV20":
        data["volatility_estimate"] = data["rv20"]
    elif estimator == "RV60":
        data["volatility_estimate"] = data["rv60"]
    else:
        data["volatility_estimate"] = 0.70 * data["rv20"] + 0.30 * data["rv60"]
    data["raw_target_position"] = (
        target_volatility / data["volatility_estimate"].replace(0.0, np.nan)
    ).clip(0.0, 1.0)
    data["risk_score"] = (data["rv20"] / data["rv60"].replace(0.0, np.nan)).clip(
        0.0, 2.0
    ) / 2.0
    data["crisis_event"] = data["close"].pct_change(fill_method=None).le(-0.03)
    return schedule_asymmetric_targets(
        data,
        review_every=review_every,
        maximum_increase=maximum_increase,
    )
