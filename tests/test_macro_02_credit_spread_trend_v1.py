from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.daily_macro_02_credit_spread_trend_v1 import (
    add_credit_compression,
    build_signal_table,
    circular_block_bootstrap_conditional_spread,
    conditional_spread,
    newey_west_regression,
)
from scripts.download_china_credit_spread_daily import (
    build_download_windows,
    normalize_curves,
)


def test_download_windows_are_strictly_less_than_one_year() -> None:
    windows = build_download_windows(pd.Timestamp("2016-08-12"), pd.Timestamp("2026-08-18"))
    assert len(windows) == 11
    assert all((end - start).days <= 364 for start, end in windows)
    assert windows[0][0] == pd.Timestamp("2016-08-12")
    assert windows[-1][1] == pd.Timestamp("2026-08-18")


def test_normalize_curves_builds_same_tenor_spread_in_basis_points() -> None:
    raw = pd.DataFrame(
        {
            "曲线名称": ["中债国债收益率曲线", "中债中短期票据收益率曲线(AAA)"],
            "日期": ["2026-08-18", "2026-08-18"],
            "1年": [1.2008, 1.5253],
            "3年": [1.2493, 1.6556],
            "10年": [1.6864, 2.0493],
        }
    )
    result = normalize_curves(raw, datetime.now(timezone.utc), True)
    assert len(result) == 1
    assert np.isclose(result.loc[0, "credit_spread_3y_bp"], 40.63)
    assert bool(result.loc[0, "transport_tls_verified"])


def test_credit_compression_uses_exact_curve_observation_lag() -> None:
    credit = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=65),
            "credit_spread_3y_bp": np.arange(100.0, 35.0, -1.0),
        }
    )
    result = add_credit_compression(credit, 63)
    assert np.isnan(result.loc[62, "compression_impulse_bp"])
    assert np.isclose(result.loc[63, "compression_impulse_bp"], 63.0)


def test_signal_uses_calendar_month_last_common_date_and_next_open() -> None:
    dates = pd.bdate_range("2019-01-01", "2021-03-05")
    levels = np.linspace(3.0, 5.0, len(dates))
    market = pd.DataFrame(
        {
            "date": dates,
            "open": levels,
            "high": levels + 0.01,
            "low": levels - 0.01,
            "close": levels,
        }
    )
    dividends = pd.DataFrame(columns=["ex_date", "cash_dividend_per_share"])
    credit = pd.DataFrame(
        {
            "date": dates,
            "cgb_3y": 2.5,
            "cpnote_aaa_3y": 3.5,
            "credit_spread_3y_bp": np.linspace(120.0, 60.0, len(dates)),
        }
    )
    config = {
        "protocol": {
            "historical_evaluation_start": "2020-12-01",
            "historical_evaluation_end": "2021-02-28",
        },
        "factor": {
            "trend_window_trading_days": 200,
            "compression_lookback_curve_observations": 63,
            "compression_risk_on_threshold_bp": 0.0,
        },
        "account_and_execution": {
            "risk_on_target_exposure": 0.985,
            "risk_off_target_exposure": 0.25,
        },
    }
    signals = build_signal_table(market, dividends, credit, config)
    february = signals.loc[signals["calendar_month"].eq(pd.Period("2021-02"))].iloc[0]
    assert february["decision_date"] == pd.Timestamp("2021-02-26")
    assert february["entry_date"] == pd.Timestamp("2021-03-01")
    assert bool(february["compression_on"])


def synthetic_intervals() -> pd.DataFrame:
    rows = []
    for index in range(48):
        compression_on = index % 2 == 0
        rows.append(
            {
                "trend_on": True,
                "compression_on": compression_on,
                "compression_impulse_bp": 10.0 if compression_on else -10.0,
                "future_excess_cash_return": 0.03 if compression_on else -0.01,
            }
        )
    return pd.DataFrame(rows)


def test_conditional_spread_and_hac_detect_positive_compression_effect() -> None:
    intervals = synthetic_intervals()
    spread = conditional_spread(intervals)
    hac = newey_west_regression(intervals, lag=3)
    assert np.isclose(spread["spread"], 0.04)
    assert hac["coefficient_per_compression_bp"] > 0
    assert hac["one_sided_p_value"] < 0.05


def test_block_bootstrap_is_deterministic() -> None:
    intervals = synthetic_intervals()
    first = circular_block_bootstrap_conditional_spread(intervals, 6, 200, 7)
    second = circular_block_bootstrap_conditional_spread(intervals, 6, 200, 7)
    assert first == second
    assert first["lower_95"] > 0
