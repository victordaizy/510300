"""行业预期差可变结果输入刷新测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from research.industry_expectation_gap_outcome_refresh import (
    OutcomeRefreshError,
    build_etf_total_return_extension,
    merge_constituent_extension,
    trading_dates_between,
    validate_close_crosscheck,
)


def _existing_etf() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-18"]),
            "open": [4.80],
            "high": [4.82],
            "low": [4.75],
            "close": [4.787],
            "volume": [100.0],
            "amount": [480.0],
            "symbol": ["510300.SH"],
            "source": ["source"],
            "volume_unit": ["share"],
            "amount_unit": ["CNY"],
            "retrieved_at": ["2026-08-18T16:00:00+08:00"],
            "cash_dividend_per_share": [0.0],
            "prev_close": [4.79],
            "price_return": [-0.0006263048],
            "dividend_yield": [0.0],
            "total_return": [-0.0006263048],
            "overnight_price_return": [0.0020876827],
            "overnight_total_contribution": [0.0020876827],
            "intraday_return": [-0.0027083333],
            "intraday_contribution": [-0.0027139875],
            "decomposition_error": [0.0],
            "range_return": [0.0147368421],
            "year": [2026],
        }
    )


def _raw_etf() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-19"]),
            "open": [4.748],
            "high": [4.749],
            "low": [4.627],
            "close": [4.654],
            "volume": [1_532_722_915.0],
            "amount": [7_147_497_737.0],
            "symbol": ["510300.SH"],
            "source": ["tushare_proxy.fund_daily"],
            "volume_unit": ["share"],
            "amount_unit": ["CNY"],
            "retrieved_at": ["2026-08-19T18:00:00+08:00"],
        }
    )


def test_trading_dates_are_strictly_after_existing_cutoff() -> None:
    calendar = pd.DataFrame(
        {"trade_date": ["2026-08-14", "2026-08-17", "2026-08-18", "2026-08-19"]}
    )
    result = trading_dates_between(calendar, "2026-08-14", "2026-08-19")
    assert [value.date().isoformat() for value in result] == [
        "2026-08-17",
        "2026-08-18",
        "2026-08-19",
    ]


def test_etf_extension_uses_prior_close_and_preserves_return_identity() -> None:
    dividends = pd.DataFrame(
        {
            "ex_date": ["2026-08-19"],
            "cash_dividend_per_share": [0.01],
        }
    )
    result = build_etf_total_return_extension(
        _existing_etf(),
        _raw_etf(),
        dividends,
    )
    row = result.iloc[0]
    assert row["prev_close"] == pytest.approx(4.787)
    assert row["total_return"] == pytest.approx((4.654 + 0.01) / 4.787 - 1.0)
    assert row["decomposition_error"] == pytest.approx(0.0, abs=1e-12)
    assert list(result.columns) == list(_existing_etf().columns)


def _constituents(date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime([date, date]),
            "con_code": ["A.SH", "B.SH"],
            "raw_open": [10.0, 20.0],
            "raw_high": [11.0, 21.0],
            "raw_low": [9.0, 19.0],
            "raw_close": [10.5, 20.5],
            "total_return_open": [12.0, 24.0],
            "total_return_high": [13.2, 25.2],
            "total_return_low": [10.8, 22.8],
            "total_return_close": [12.6, 24.6],
        }
    )


def test_constituent_extension_is_append_only_and_complete() -> None:
    result = merge_constituent_extension(
        _constituents("2026-08-18"),
        _constituents("2026-08-19"),
        required_constituent_count=2,
    )
    assert len(result) == 4
    assert result[["date", "con_code"]].duplicated().sum() == 0
    assert result["date"].max() == pd.Timestamp("2026-08-19")


def test_constituent_extension_cannot_overwrite_existing_date() -> None:
    with pytest.raises(OutcomeRefreshError, match="不得覆盖"):
        merge_constituent_extension(
            _constituents("2026-08-18"),
            _constituents("2026-08-18"),
            required_constituent_count=2,
        )


def test_independent_close_crosscheck_detects_supplier_difference() -> None:
    primary = _constituents("2026-08-19")
    independent = primary[["date", "con_code", "raw_close"]].copy()
    independent.loc[independent["con_code"].eq("B.SH"), "raw_close"] += 0.01
    with pytest.raises(OutcomeRefreshError, match="超过"):
        validate_close_crosscheck(
            primary,
            independent,
            target_date="2026-08-19",
            maximum_close_difference=0.001,
            required_constituent_count=2,
        )


def test_independent_close_crosscheck_passes_exact_match() -> None:
    primary = _constituents("2026-08-19")
    independent = primary[["date", "con_code", "raw_close"]].copy()
    result = validate_close_crosscheck(
        primary,
        independent,
        target_date="2026-08-19",
        maximum_close_difference=0.001,
        required_constituent_count=2,
    )
    assert result["status"] == "PASS"
    assert result["comparable_count"] == 2
    assert result["exact_match_ratio"] == 1.0
