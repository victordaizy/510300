from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.daily_macro_01_m2_accel_trend_v1 import (
    add_exact_m2_acceleration,
    build_signal_table,
    build_total_return_series,
    circular_block_bootstrap_conditional_spread,
    conditional_spread,
    newey_west_regression,
    profit_factor,
)
from scripts.download_china_money_supply_monthly import normalize_money_supply


def test_normalize_money_supply_ignores_garbled_labels() -> None:
    raw = pd.DataFrame(
        [
            ["2018��02�", 1_800_000, 8.8, 0.1, 500_000, 6.0, 0.1, 70_000, 4.0, 0.1],
            ["2018��01�", 1_790_000, 8.6, 0.1, 490_000, 5.8, 0.1, 69_000, 3.8, 0.1],
        ]
    )
    result = normalize_money_supply(raw, datetime.now(timezone.utc))
    assert result["month"].dt.strftime("%Y-%m").tolist() == ["2018-01", "2018-02"]
    assert result["m2_yoy_pct"].tolist() == [8.6, 8.8]


def test_m2_acceleration_uses_exact_calendar_lag() -> None:
    money = pd.DataFrame(
        {
            "month": pd.to_datetime(["2018-01-31", "2018-02-28", "2018-07-31", "2018-08-31"]),
            "m2_yoy_pct": [8.0, 8.5, 9.0, 8.0],
        }
    )
    result = add_exact_m2_acceleration(money, 6).set_index("period")
    assert result.loc[pd.Period("2018-07", freq="M"), "m2_accel_pct_point"] == 1.0
    assert result.loc[pd.Period("2018-08", freq="M"), "m2_accel_pct_point"] == -0.5


def test_total_return_series_adds_cash_dividend_on_ex_date() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "close": [10.0, 9.0],
        }
    )
    dividends = pd.DataFrame(
        {"ex_date": pd.to_datetime(["2020-01-03"]), "cash_dividend_per_share": [1.0]}
    )
    result = build_total_return_series(market, dividends)
    assert np.isclose(result["total_return"].iloc[1], 0.0)
    assert np.isclose(result["total_return_index"].iloc[-1], 1.0)


def test_signal_uses_twentieth_close_and_next_open() -> None:
    dates = pd.bdate_range("2017-01-02", "2018-09-28")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(3.0, 5.0, len(dates)),
            "high": np.linspace(3.01, 5.01, len(dates)),
            "low": np.linspace(2.99, 4.99, len(dates)),
            "close": np.linspace(3.0, 5.0, len(dates)),
        }
    )
    dividends = pd.DataFrame(columns=["ex_date", "cash_dividend_per_share"])
    money = pd.DataFrame(
        {
            "month": pd.date_range("2017-01-31", "2018-08-31", freq="ME"),
            "m2_yoy_pct": np.linspace(8.0, 10.0, 20),
        }
    )
    config = {
        "protocol": {"historical_evaluation_start": "2018-08-20", "historical_evaluation_end": "2018-09-28"},
        "factor": {"trend_window_trading_days": 200, "m2_lag_months": 6, "m2_risk_on_threshold_pct_point": 0.0},
        "account_and_execution": {"risk_on_target_exposure": 0.985, "risk_off_target_exposure": 0.25},
    }
    signals = build_signal_table(market, dividends, money, config)
    july = signals.loc[signals["m2_period"].eq("2018-07")].iloc[0]
    assert july["decision_date"] == pd.Timestamp("2018-08-20")
    assert july["entry_date"] == pd.Timestamp("2018-08-21")


def _synthetic_intervals() -> pd.DataFrame:
    rows = []
    for index in range(48):
        m2_on = index % 2 == 0
        rows.append(
            {
                "trend_on": True,
                "m2_on": m2_on,
                "m2_accel_pct_point": 1.0 if m2_on else -1.0,
                "future_excess_cash_return": 0.03 if m2_on else -0.01,
            }
        )
    return pd.DataFrame(rows)


def test_conditional_spread_and_hac_detect_positive_synthetic_effect() -> None:
    intervals = _synthetic_intervals()
    spread = conditional_spread(intervals)
    hac = newey_west_regression(intervals, lag=3)
    assert np.isclose(spread["spread"], 0.04)
    assert hac["coefficient_per_m2_pct_point"] > 0
    assert hac["one_sided_p_value"] < 0.05


def test_block_bootstrap_is_deterministic() -> None:
    intervals = _synthetic_intervals()
    first = circular_block_bootstrap_conditional_spread(intervals, 6, 200, 7)
    second = circular_block_bootstrap_conditional_spread(intervals, 6, 200, 7)
    assert first == second
    assert first["lower_95"] > 0


def test_profit_factor_uses_positive_over_absolute_negative() -> None:
    assert np.isclose(profit_factor(pd.Series([0.03, -0.01, 0.02, -0.01])), 2.5)

