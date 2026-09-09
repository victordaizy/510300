"""偏度V2有效观测窗口修正测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.option_smirk_gamma1_v2 import apply_valid_observation_threshold


def test_threshold_uses_prior_valid_observations_across_missing_days() -> None:
    dates = pd.bdate_range("2020-01-01", periods=520)
    values = np.linspace(-1.0, 1.0, len(dates))
    values[::3] = np.nan
    daily = pd.DataFrame({"trade_date": dates, "gamma1_30d": values})
    config = {
        "signal": {
            "trailing_valid_observations": 252,
            "threshold_lag_valid_observations": 1,
            "lower_quantile": 0.20,
        }
    }
    result = apply_valid_observation_threshold(daily, config)
    assert result["gamma1_lower_threshold"].notna().sum() > 0
    first = result["gamma1_lower_threshold"].first_valid_index()
    assert first is not None
    prior_valid = result.loc[: first - 1, "gamma1_30d"].dropna()
    assert len(prior_valid) == 252
    assert result.loc[first, "gamma1_lower_threshold"] == prior_valid.quantile(0.20)


def test_missing_surface_day_never_generates_signal() -> None:
    daily = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2020-01-01", periods=5),
            "gamma1_30d": [1.0, np.nan, 0.0, -1.0, -2.0],
        }
    )
    config = {
        "signal": {
            "trailing_valid_observations": 2,
            "threshold_lag_valid_observations": 1,
            "lower_quantile": 0.20,
        }
    }
    result = apply_valid_observation_threshold(daily, config)
    assert not bool(result.loc[1, "buy_call_signal"])
    assert pd.isna(result.loc[1, "gamma1_lower_threshold"])
