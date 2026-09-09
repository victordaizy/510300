"""验证日历边界、利率可用日及真实账户的无观点和退出语义。"""
import numpy as np
import pandas as pd
import pytest

from research.cash_distribution_funding_inputs_v1 import distribution_frame, known_funding
from research.event_clock_account_v1 import simulate_event_account


def distributions(dates, amounts):
    return pd.DataFrame({"ex_date": pd.to_datetime(dates), "cash_dividend_per_share": amounts})


def test_known_rate_uses_next_stock_open_and_latest_prior_source_day():
    dates = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
    rates = pd.DataFrame({"date": pd.to_datetime(["2020-01-03", "2020-01-04", "2020-01-06"]), "dr007": [2., 3., 4.]})
    result = known_funding(dates, rates)
    np.testing.assert_allclose(result.dr_annual_rate, [np.nan, .03, .04], equal_nan=True)
    assert result.dr_available_at.iloc[1] == pd.Timestamp("2020-01-06 09:30")
    assert result.dr_date.iloc[1] == pd.Timestamp("2020-01-04")


def test_seven_natural_days_is_valid_but_eight_is_no_view():
    rates = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "dr007": [2.]})
    result = known_funding(pd.to_datetime(["2020-01-08", "2020-01-09"]), rates)
    assert result.dr_valid.tolist() == [True, False]


def test_cash_distribution_uses_calendar_year_exclusive_left_and_ex_date_inclusive():
    data = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), "close": 10.})
    div = distributions(["2019-01-01", "2019-01-02", "2020-01-02", "2020-01-04"], [.1, .2, .3, 20.])
    rates = pd.DataFrame({"date": pd.to_datetime(["2019-12-31"]), "dr007": [1.]})
    result = distribution_frame(data, div, rates, "2018-01-01", "2021-01-01")
    np.testing.assert_allclose(result.past_year_distribution_per_share, [.2, .3, .3])
    assert result.past_year_distribution_count.tolist() == [1, 1, 1]


def test_leap_day_is_inside_next_february_end_but_outside_march():
    data = pd.DataFrame({"date": pd.to_datetime(["2021-02-28", "2021-03-01"]), "close": 10.})
    rates = pd.DataFrame({"date": pd.to_datetime(["2021-02-27"]), "dr007": [1.]})
    result = distribution_frame(data, distributions(["2020-02-29"], [.2]), rates, "2018-01-01", "2022-01-01")
    np.testing.assert_allclose(result.past_year_distribution_per_share, [.2, 0.])


def test_equality_exits_and_unknown_is_distinct_from_known_zero_distribution():
    data = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), "close": 10.})
    rates = pd.DataFrame({"date": pd.to_datetime(["2019-12-31"]), "dr007": [2.5]})
    result = distribution_frame(data, distributions(["2020-01-02"], [.25]), rates, "2019-01-02", "2021-01-01")
    assert np.isnan(result.target.iloc[0]) and result.target.iloc[1] == 0
    known_zero = distribution_frame(data, distributions([], []), rates, "2018-01-01", "2021-01-01")
    assert known_zero.target.eq(0).all() and known_zero.past_year_distribution_per_share.eq(0).all()


def test_changing_future_rates_and_distributions_cannot_change_past_targets():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=5), "close": 10.})
    rates = pd.DataFrame({"date": pd.bdate_range("2019-12-31", periods=6), "dr007": 2.})
    div = distributions(["2019-12-01", "2020-01-06"], [.3, .1])
    expected = distribution_frame(data, div, rates, "2018-01-01", "2021-01-01")
    rates.loc[rates.date >= "2020-01-03", "dr007"] = 99.
    div.loc[1, "cash_dividend_per_share"] = 99.
    data.loc[data.date > "2020-01-03", "close"] = 100.
    actual = distribution_frame(data, div, rates, "2018-01-01", "2021-01-01")
    pd.testing.assert_frame_equal(expected.iloc[:3], actual.iloc[:3])


def test_account_keeps_shares_on_missing_rate_then_exits_next_open_on_known_nonpositive_spread():
    dates = pd.bdate_range("2020-01-01", periods=8)
    data = pd.DataFrame({"date": dates, "open": 10., "high": 10., "low": 10., "close": 10.,
        "previous_close": 10., "dividend": 0., "wealth": 1., "variance60": .0001})
    rates = pd.DataFrame({"date": pd.bdate_range("2019-12-31", periods=9),
        "dr007": [2., 2., np.nan, 4., 2., 2., 2., 2., 2.]})
    div = distributions(["2019-12-01"], [.3])
    target = distribution_frame(data, div, rates, "2018-01-01", "2021-01-01").target.to_numpy()
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    empty = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, decisions = simulate_event_account(data, empty, cfg, cost, str(dates[1].date()), "分红测试", targets=target, event_mask=np.ones(len(data), bool))
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [dates[1], dates[5]]
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [dates[4], dates[7]]
    missing = ledger[ledger.date.eq(dates[3])].iloc[0]
    assert missing.shares > 0 and missing.requested_quantity == 0
    assert decisions.loc[decisions.origin.eq(dates[2]), "reference_weight"].isna().all()
    assert ledger.accounting_error.abs().max() < 1e-7 and ledger.shares.iloc[-1] == 0
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0


@pytest.mark.parametrize("column,value", [("close", -1.), ("close", np.inf)])
def test_invalid_prices_do_not_get_repaired(column, value):
    data = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "close": [value]})
    rates = pd.DataFrame({"date": pd.to_datetime(["2019-12-31"]), "dr007": [2.]})
    with pytest.raises(ValueError):
        distribution_frame(data, distributions([], []), rates, "2018-01-01", "2021-01-01")
