"""季度会计、先后时序和最新报告前沿的反例测试。"""
import numpy as np
import pandas as pd

from research.factor96_earnings_cashflow_measurement_v1 import (
    FinancialDependencies, amount, daily_frontier, industry_at, next_open,
    quarter_profit, seasonal_surprise, trailing_flow,
)


def test_quarter_split_does_not_subtract_previous_year_for_q1():
    assert quarter_profit(30., np.nan, 1) == 30.
    assert quarter_profit(80., 30., 2) == 50.
    assert quarter_profit(170., 110., 4) == 60.
    assert np.isnan(quarter_profit(80., np.nan, 2))


def test_ttm_is_not_annualized_ytd():
    assert trailing_flow(70., 180., 40., 2) == 210.
    assert trailing_flow(180., np.nan, np.nan, 4) == 180.
    assert np.isnan(trailing_flow(70., np.nan, 40., 2))


def test_seasonal_zero_dispersion_stays_unknown():
    score, mean, sd = seasonal_surprise(.06, .02, .04)
    assert np.isclose(score, .03 / np.sqrt(.0002))
    assert np.isclose(mean, .03) and np.isclose(sd, np.sqrt(.0002))
    assert np.isnan(seasonal_surprise(.06, .02, .02)[0])
    assert np.isnan(seasonal_surprise(.06, np.nan, .04)[0])


def test_publication_date_requires_next_session_even_on_session_day():
    sessions = pd.DatetimeIndex(["2024-04-26", "2024-04-29", "2024-04-30"])
    values = pd.Series(pd.to_datetime(["2024-04-26", "2024-04-28", "2024-04-30"]))
    main = next_open(values, sessions)
    delayed = next_open(values, sessions, extra=1)
    assert main.iloc[0] == main.iloc[1] == pd.Timestamp("2024-04-29")
    assert delayed.iloc[0] == pd.Timestamp("2024-04-30") and pd.isna(main.iloc[2])


def test_late_historical_report_cannot_supply_a_value():
    events = pd.DataFrame([{"ts_code": "A", "quarter_ordinal": 215, "announcement_id": "OLD", "available_date": pd.Timestamp("2024-05-06")}])
    facts = pd.DataFrame([{"announcement_id": "OLD", "metric_id": "TOTAL_ASSETS_END", "fact_status": "VERIFIED_SAVED_ORIGINAL_FACT", "verified_value": 100.}])
    dependency = FinancialDependencies(events, facts)
    dependency.target = {"ts_code": "A", "announcement_id": "NEW", "report_period": pd.Timestamp("2024-03-31"), "available_date": pd.Timestamp("2024-04-30")}
    assert np.isnan(dependency.get(215, "TOTAL_ASSETS_END", "TEST"))
    assert dependency.ledger[-1]["status"] == "DISCLOSED_AFTER_TARGET"
    assert np.isnan(dependency.ledger[-1]["value"])


def test_latest_missing_report_is_not_replaced_by_old_valid_report():
    frame = pd.DataFrame([
        {"ts_code": "A", "announcement_id": "Q3", "report_period": "2023-09-30", "available_date": "2023-10-30", "known": True},
        {"ts_code": "A", "announcement_id": "Q1", "report_period": "2024-03-31", "available_date": "2024-04-26", "known": False},
        {"ts_code": "A", "announcement_id": "FY_LATE", "report_period": "2023-12-31", "available_date": "2024-04-29", "known": True},
    ])
    for key in ["report_period", "available_date"]:
        frame[key] = pd.to_datetime(frame[key])
    result = daily_frontier(frame)
    assert list(result.announcement_id) == ["Q3", "Q1"] and not result.known.iloc[-1]


def test_industry_available_time_and_ambiguous_intervals():
    points = pd.DataFrame({"ts_code": ["A", "B", "C"], "date": pd.to_datetime(["2024-04-30"] * 3)})
    intervals = pd.DataFrame([
        {"ts_code": "A", "industry_l1_code": "801780.SI", "valid_from": "2020-01-01", "valid_to": None, "available_at": "2020-01-01"},
        {"ts_code": "B", "industry_l1_code": "801010.SI", "valid_from": "2024-04-30", "valid_to": None, "available_at": "2024-04-30 15:00"},
        {"ts_code": "C", "industry_l1_code": "801010.SI", "valid_from": "2020-01-01", "valid_to": None, "available_at": "2020-01-01"},
        {"ts_code": "C", "industry_l1_code": "801020.SI", "valid_from": "2020-01-01", "valid_to": None, "available_at": "2020-01-01"},
    ])
    for key in ["valid_from", "valid_to", "available_at"]:
        intervals[key] = pd.to_datetime(intervals[key], format="mixed")
    result = industry_at(points, intervals)
    assert result.is_financial.iloc[0]
    assert result.industry_match_count.tolist() == [1, 0, 2]
    assert pd.isna(result.industry_l1_code.iloc[1]) and pd.isna(result.industry_l1_code.iloc[2])


def test_source_units_and_unknowns():
    assert amount("（1,234.50）", 10000) == -12345000.
    assert np.isnan(amount("[空白]", 10000))
