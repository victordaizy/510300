"""检查两年历史边界、观察与成交隔离、缺失及分红处理。"""
import numpy as np
import pandas as pd

from research.factor96_t11_o02_date_proxy_v1 import measure_first_responses


def sample(dates=None, closes=None):
    dates = pd.to_datetime(dates or ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"])
    prices = pd.DataFrame({"date": dates, "raw_close": closes or [100, 102, 106.08, 105.0192, 107, 108],
                           "cash_dividend": 0., "price_source_admitted": True, "cash_dividend_known": True})
    financial = pd.DataFrame({"date": dates[1:4], "cohort_known": True, "positive_financial_measurement": True,
                              "L02_cohort": 2., "L04_industry_cohort": .5, "L04_raw_change_cohort": .01})
    return financial, prices


def test_current_response_excluded_and_negative_response_not_filtered():
    financial, prices = sample()
    result, dependencies = measure_first_responses(financial, prices)
    current = result.iloc[-1]
    assert current.reference_count == 2
    assert np.isclose(current.reference_median, .03)
    assert np.isclose(current.first_response, -.01)
    assert current.underreaction_condition and current.negative_first_response
    assert (dependencies.source_date < dependencies.target_date).all()
    assert current.earliest_entry_date == pd.Timestamp("2020-01-08")
    assert current.lag1_entry_date == pd.Timestamp("2020-01-09")


def test_future_prices_and_financial_rows_cannot_change_past_feature():
    financial, prices = sample()
    original, _ = measure_first_responses(financial, prices)
    later = financial.iloc[[-1]].copy()
    later["date"], later["L02_cohort"] = pd.Timestamp("2020-01-09"), 5000.
    augmented = pd.concat([financial, later], ignore_index=True)
    changed = prices.copy()
    changed.loc[changed.date.gt(pd.Timestamp("2020-01-07")), "raw_close"] = 10000.
    rebuilt, _ = measure_first_responses(augmented, changed)
    pd.testing.assert_frame_equal(original, rebuilt.iloc[:len(original)].reset_index(drop=True))


def test_missing_dividend_or_price_identity_is_unknown_not_zero_return():
    financial, prices = sample()
    prices.loc[prices.date.eq(pd.Timestamp("2020-01-07")), "cash_dividend_known"] = False
    result, _ = measure_first_responses(financial, prices)
    row = result.iloc[-1]
    assert row.status == "NO_VIEW_FIRST_RESPONSE_OR_DIVIDEND"
    assert pd.isna(row.first_response) and pd.isna(row.underreaction_condition)
    assert not row.condition_known


def test_ex_dividend_cash_is_counted_in_observation_not_future_holding():
    financial, prices = sample()
    prices.loc[prices.date.eq(pd.Timestamp("2020-01-07")), ["raw_close", "cash_dividend"]] = [105.08, 1.]
    result, _ = measure_first_responses(financial, prices)
    assert np.isclose(result.iloc[-1].first_response, 0.)
    assert result.iloc[-1].earliest_entry_date > result.iloc[-1].date
    assert "future_return" not in result.columns


def test_two_calendar_year_cutoff_retains_boundary_and_excludes_older_day():
    dates = ["2017-01-02", "2017-01-03", "2017-01-04", "2018-05-02", "2019-01-04", "2019-01-07", "2019-01-08"]
    financial, prices = sample(dates=dates, closes=[100, 200, 202, 206.04, 210, 215, 216])
    added = financial.iloc[[-1]].copy()
    added["date"] = pd.Timestamp("2019-01-04")
    financial = pd.concat([financial, added], ignore_index=True)
    result, dependency = measure_first_responses(financial, prices)
    current = result.iloc[-1]
    assert current.reference_count == 2
    refs = dependency.loc[dependency.target_date.eq(current.date)]
    assert set(refs.source_date) == {pd.Timestamp("2017-01-04"), pd.Timestamp("2018-05-02")}


def test_current_unknown_financial_basket_has_no_cash_or_trade_interpretation():
    financial, prices = sample()
    financial.loc[2, "cohort_known"] = False
    result, _ = measure_first_responses(financial, prices)
    assert result.iloc[-1].status == "NO_VIEW_CURRENT_FINANCIAL_COHORT"
    assert pd.isna(result.iloc[-1].underreaction_condition)
    assert not result.iloc[-1].condition_known


def test_empty_disclosure_input_preserves_explicit_empty_measurement_schema():
    financial, prices = sample()
    result, dependencies = measure_first_responses(financial.iloc[:0], prices)
    assert result.empty and dependencies.empty
    assert {"condition_known", "underreaction_condition", "status"}.issubset(result.columns)
    assert str(result.underreaction_condition.dtype) == "boolean"
