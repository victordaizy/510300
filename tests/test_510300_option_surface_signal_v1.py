"""冻结期权曲面O1至O4公式的纯合成数据测试。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.special import ndtr

from research.option_surface_signal_v1 import (
    black_forward_price,
    build_slices,
    delta_implied_volatility,
    estimate_forward_discount,
    fixed_model_free_variance,
    har_forecast,
    left_tail_probability,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "510300_option_surface_signal_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def synthetic_chain(
    signal_date: pd.Timestamp,
    calendar_days: list[int],
    volatility: float = 0.20,
) -> pd.DataFrame:
    forward = 5.0
    rows = []
    for days in calendar_days:
        maturity = days / 365.0
        discount = math.exp(-0.02 * maturity)
        expiry = signal_date + pd.Timedelta(days=days)
        for strike in np.arange(3.5, 6.51, 0.25):
            for option_type in ("C", "P"):
                mid = black_forward_price(
                    forward, float(strike), maturity, volatility, option_type, discount
                )
                half_spread = min(0.00005, mid * 0.1)
                rows.append(
                    {
                        "contract_code": f"{days}_{strike:.2f}_{option_type}",
                        "option_type": option_type,
                        "expiry_date": expiry,
                        "strike": float(strike),
                        "contract_unit": 10000,
                        "is_adjusted": False,
                        "bid1": mid - half_spread,
                        "ask1": mid + half_spread,
                    }
                )
    return pd.DataFrame(rows)


def test_pairwise_parity_recovers_forward_and_discount(config: dict) -> None:
    signal_date = pd.Timestamp("2026-08-19")
    snapshot = synthetic_chain(signal_date, [35])
    snapshot["mid"] = (snapshot["bid1"] + snapshot["ask1"]) / 2.0
    result = estimate_forward_discount(snapshot, config)
    expected_discount = math.exp(-0.02 * 35 / 365.0)
    assert result["forward"] == pytest.approx(5.0, abs=1e-10)
    assert result["discount"] == pytest.approx(expected_discount, abs=1e-10)
    assert result["parity_normalized_rmse"] < 1e-10


def test_svi_fixed_maturity_delta_tail_and_model_free_variance(config: dict) -> None:
    signal_date = pd.Timestamp("2026-08-19")
    snapshot = synthetic_chain(signal_date, [15, 35, 75])
    slices = build_slices(snapshot, signal_date, config)
    put25 = delta_implied_volatility(slices, 25, "P", 0.25, config)
    call25 = delta_implied_volatility(slices, 25, "C", 0.25, config)
    put60 = delta_implied_volatility(slices, 60, "P", 0.25, config)
    assert put25 == pytest.approx(0.20, abs=0.01)
    assert call25 == pytest.approx(0.20, abs=0.01)
    assert put60 == pytest.approx(0.20, abs=0.01)
    probability = left_tail_probability(slices, config)
    maturity = 25 / 365.0
    strike_ratio = 0.90
    d2 = (math.log(1.0 / strike_ratio) - 0.5 * 0.20**2 * maturity) / (
        0.20 * math.sqrt(maturity)
    )
    expected_probability = float(ndtr(-d2))
    assert probability == pytest.approx(expected_probability, abs=0.015)
    implied_variance = fixed_model_free_variance(slices, 25, config)
    assert implied_variance > 0
    assert implied_variance == pytest.approx(0.20**2, abs=0.015)


def test_fixed_maturity_extrapolation_is_forbidden(config: dict) -> None:
    signal_date = pd.Timestamp("2026-08-19")
    snapshot = synthetic_chain(signal_date, [35, 75])
    slices = build_slices(snapshot, signal_date, config)
    with pytest.raises(ValueError, match="禁止外推"):
        delta_implied_volatility(slices, 25, "P", 0.25, config)


def test_har_uses_only_mature_training_rows(config: dict) -> None:
    dates = pd.bdate_range("2025-01-02", periods=360)
    returns = 0.0002 + 0.006 * np.sin(np.arange(len(dates)) / 7.0)
    close = 4.0 * np.cumprod(1.0 + returns)
    market = pd.DataFrame({"date": dates, "close": close})
    dividends = pd.DataFrame(
        columns=["symbol", "ex_date", "cash_dividend_per_share"]
    )
    result = har_forecast(market, dividends, dates[-1], config)
    assert result["training_rows"] >= 252
    assert result["training_rows"] <= len(dates) - 20
    assert result["forecast_variance20"] > 0
    assert len(result["coefficients"]) == 4


def test_har_rejects_insufficient_mature_history(config: dict) -> None:
    dates = pd.bdate_range("2026-01-02", periods=100)
    market = pd.DataFrame(
        {"date": dates, "close": 4.0 * np.cumprod(np.repeat(1.001, len(dates)))}
    )
    dividends = pd.DataFrame(
        columns=["symbol", "ex_date", "cash_dividend_per_share"]
    )
    with pytest.raises(ValueError, match="训练行不足"):
        har_forecast(market, dividends, dates[-1], config)


def test_har_prediction_cannot_see_prices_after_signal_date(config: dict) -> None:
    dates = pd.bdate_range("2025-01-02", periods=380)
    returns = 0.0001 + 0.007 * np.sin(np.arange(len(dates)) / 9.0)
    close = 4.0 * np.cumprod(1.0 + returns)
    market = pd.DataFrame({"date": dates, "close": close})
    dividends = pd.DataFrame(
        columns=["symbol", "ex_date", "cash_dividend_per_share"]
    )
    signal_date = dates[359]
    original = har_forecast(market, dividends, signal_date, config)
    contaminated = market.copy()
    contaminated.loc[contaminated["date"].gt(signal_date), "close"] *= 10.0
    repeated = har_forecast(contaminated, dividends, signal_date, config)
    assert repeated["forecast_variance20"] == pytest.approx(
        original["forecast_variance20"], rel=0, abs=1e-15
    )
    assert repeated["coefficients"] == pytest.approx(original["coefficients"], abs=1e-15)
