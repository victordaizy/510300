from __future__ import annotations

import numpy as np
import pandas as pd

from research.total_return_component_ledger_v1 import (
    build_component_ledger,
    build_etf_total_return_series,
    build_forward_schedule,
)


def test_etf_total_return_series_reinvests_cash_on_ex_date() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]),
            "close": [10.0, 9.5, 10.0],
        }
    )
    dividends = pd.DataFrame(
        {
            "ex_date": ["2021-01-05"],
            "cash_dividend_per_share": [0.5],
        }
    )
    result = build_etf_total_return_series(prices, dividends)

    assert np.isclose(result.loc[1, "daily_total_return_factor"], 1.0)
    assert np.isclose(
        result.loc[2, "etf_total_return_index"], 10.0 / 9.5
    )


def test_forward_schedule_uses_exact_market_day_offsets_and_censors_tail() -> None:
    calendar = pd.bdate_range("2021-01-01", periods=130)
    schedule = build_forward_schedule(
        pd.Series([calendar[0], calendar[80]]),
        pd.Series(calendar),
        horizons_market_days=[60, 120],
        origin_start="2021-01-01",
        observation_cutoff=calendar[-1].date().isoformat(),
    )

    first_60 = schedule.loc[
        schedule["origin"].eq(calendar[0])
        & schedule["horizon_market_days"].eq(60)
    ].iloc[0]
    assert first_60["target_date"] == calendar[60]
    assert first_60["schedule_status"] == "PASS_EXACT_MARKET_DAY_TARGET"
    late = schedule.loc[
        schedule["origin"].eq(calendar[80])
        & schedule["horizon_market_days"].eq(60)
    ].iloc[0]
    assert late["schedule_status"] == "CENSORED_TARGET_BEYOND_AVAILABLE_CALENDAR"


def test_three_scope_ledger_preserves_exact_multiplicative_identities() -> None:
    origin = pd.Timestamp("2021-01-04")
    target = pd.Timestamp("2021-03-30")
    schedule = pd.DataFrame(
        {
            "origin": [origin],
            "horizon_market_days": [60],
            "target_date": [target],
            "schedule_status": ["PASS_EXACT_MARKET_DAY_TARGET"],
        }
    )
    price_index = pd.DataFrame(
        {"date": [origin, target], "close": [100.0, 110.0]}
    )
    total_index = pd.DataFrame(
        {
            "date": [origin, target],
            "close": [100.0, 112.0],
            "pe_ttm": [10.0, 11.0],
        }
    )
    etf_total = pd.DataFrame(
        {
            "date": [origin, target],
            "etf_total_return_index": [1.0, 1.11],
        }
    )
    component_state = pd.DataFrame(
        {
            "origin": [origin, origin, pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-01")],
            "stock_code": ["A", "B", "A", "B"],
            "state_weight": [0.6, 0.4, 0.55, 0.45],
            "parent_net_profit_ttm": [100.0, 200.0, 120.0, 220.0],
            "income_available_at": [
                pd.Timestamp("2020-12-31"),
                pd.Timestamp("2020-12-31"),
                pd.Timestamp("2021-02-28"),
                pd.Timestamp("2021-02-28"),
            ],
            "total_shares": [100.0, 100.0, 100.0, 100.0],
        }
    )
    raw = pd.DataFrame(
        {
            "date": [origin, origin, target, target],
            "stock_code": ["A", "B", "A", "B"],
            "raw_close": [10.0, 20.0, 11.0, 18.0],
        }
    )
    component_tr = pd.DataFrame(
        {
            "date": [origin, origin, target, target],
            "con_code": ["A", "B", "A", "B"],
            "total_return_close": [10.0, 20.0, 11.2, 19.0],
        }
    )
    ledger, coverage, identity = build_component_ledger(
        schedule=schedule,
        price_index=price_index,
        total_return_index=total_index,
        etf_total_return=etf_total,
        component_state=component_state,
        component_raw_price=raw,
        component_total_return=component_tr,
        effective_share_columns=["total_shares"],
        market_minimum_weight_coverage=0.95,
        financial_pass_minimum_weight_coverage=0.95,
        financial_partial_minimum_weight_coverage=0.80,
        identity_tolerance=1.0e-12,
    )

    assert set(ledger["scope"]) == {
        "ACTUAL_INDEX_CHAINED_SCOPE",
        "ORIGIN_FIXED_COMPONENTS_AND_WEIGHTS_SCOPE",
        "ORIGIN_FIXED_FINANCIAL_COVERAGE_COHORT_SCOPE",
    }
    assert identity["identity_pass"].all()
    fixed = ledger.loc[
        ledger["scope"].eq("ORIGIN_FIXED_COMPONENTS_AND_WEIGHTS_SCOPE")
    ].iloc[0]
    assert np.isnan(fixed["dividend_component_factor"])
    assert fixed["dividend_component_status"] == (
        "NO_VIEW_NO_COMPONENT_DISTRIBUTION_ARCHIVE"
    )
    financial = ledger.loc[
        ledger["scope"].eq("ORIGIN_FIXED_FINANCIAL_COVERAGE_COHORT_SCOPE")
    ].iloc[0]
    assert np.isclose(financial["earnings_growth_component_factor"], 1.16)
    assert financial["ledger_status"] == "PASS_FIXED_FINANCIAL_COHORT_COVERAGE"
    assert np.isclose(coverage["financial_weight_coverage"].iloc[0], 1.0)
