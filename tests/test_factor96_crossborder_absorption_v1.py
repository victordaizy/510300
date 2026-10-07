"""T10重点检查时区、分配处理、信息成熟与等待后的止损边界。"""
import numpy as np
import pandas as pd

from research.factor96_crossborder_absorption_v1 import (
    align_external, lag_features, parse_ashr, rolling_response_model, signal_for,
)


def make_us(dates, closes, dividends=None):
    dates = pd.DatetimeIndex(dates)
    timestamps = ((dates.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC").as_unit("s").asi8).tolist()
    distributions = {} if dividends is None else {str(timestamps[i]): {"date": timestamps[i], "amount": amount} for i, amount in dividends.items()}
    payload = {"chart": {"result": [{"meta": {"symbol": "ASHR", "currency": "USD", "exchangeTimezoneName": "America/New_York"},
        "timestamp": timestamps, "events": {"dividends": distributions}, "indicators": {"quote": [{"open": closes,
            "high": closes, "low": closes, "close": closes, "volume": [100]*len(closes)}]}}]}}
    return parse_ashr(payload)


def make_market(dates):
    n = len(dates)
    return pd.DataFrame({"date": pd.to_datetime(dates), "open": np.arange(n)+100., "close": np.arange(n)+100., "dividend": np.zeros(n)})


def make_fx(dates, delay=None):
    dates = pd.DatetimeIndex(dates)
    times = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=15)
    frame = pd.DataFrame({"date": dates, "available_at": times, "first_release_value": np.arange(len(dates))*.01+7.})
    if delay is not None:
        frame.loc[delay, "available_at"] += pd.Timedelta(minutes=10)
    return frame


def test_cash_distribution_not_misread_as_overseas_crash():
    us = make_us(["2020-12-17", "2020-12-18", "2020-12-21"], [100., 91., 92.], {1: 10.})
    assert np.isclose(us.log_return.iloc[1], np.log(1.01))
    assert np.isclose(us.log_return.iloc[2], np.log(92./91.))


def test_dst_and_no_same_calendar_day_us_final_bar():
    dates = ["2024-03-08", "2024-03-11", "2024-03-12"]
    us = make_us(["2024-03-06", "2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12"], [99., 100., 102., 105., 900.])
    assert us.available_at.iloc[2] == pd.Timestamp("2024-03-09 06:00:00", tz="Asia/Shanghai")
    assert us.available_at.iloc[3] == pd.Timestamp("2024-03-12 05:00:00", tz="Asia/Shanghai")
    frame = align_external(make_market(dates), us, make_fx(dates))
    assert frame.latest_us_date.iloc[-1] == pd.Timestamp("2024-03-11")
    assert np.isclose(frame.ashr_log_return.iloc[-1], np.log(105./102.))


def test_china_holiday_aggregates_only_completed_sessions():
    dates = ["2024-02-08", "2024-02-19"]
    us = make_us(["2024-02-07", "2024-02-08", "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-19"],
                 [100., 101., 102., 103., 104., 105., 106., 107., 900.])
    frame = align_external(make_market(dates), us, make_fx(dates))
    assert frame.us_session_count.iloc[-1] == 7
    assert np.isclose(frame.ashr_log_return.iloc[-1], np.log(1.07))


def test_no_new_us_session_is_missing_not_zero():
    dates = ["2024-05-27", "2024-05-28"]
    us = make_us(["2024-05-23", "2024-05-24", "2024-05-28"], [100., 101., 104.])
    frame = align_external(make_market(dates), us, make_fx(dates))
    assert frame.us_session_count.iloc[-1] == 0
    assert np.isnan(frame.ashr_log_return.iloc[-1]) and not frame.external_known.iloc[-1]


def test_late_fx_fix_not_backfilled_into_0920_forecast():
    dates = ["2024-03-08", "2024-03-11", "2024-03-12"]
    us = make_us(["2024-03-06", "2024-03-07", "2024-03-08", "2024-03-11"], [99., 100., 102., 105.])
    frame = align_external(make_market(dates), us, make_fx(dates, delay=2))
    assert frame.fx_date.iloc[-1] == pd.Timestamp("2024-03-11")
    assert np.isnan(frame.fx_log_return.iloc[-1])


def model_data(n=900):
    rng = np.random.default_rng(141)
    dates = pd.bdate_range("2016-01-04", periods=n)
    x = rng.normal(0, .01, (n, 3))
    y = .001+x @ np.array([.3, -.2, .1])+rng.normal(0, .006, n)
    return pd.DataFrame({"date": dates, "forecast_at": dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9, minutes=20),
        "ashr_log_return": x[:, 0], "fx_log_return": x[:, 1], "prior_a_log_return": x[:, 2],
        "a_log_return": y, "a_gap_log_return": y*.4, "external_known": True})


def test_model_cannot_see_current_response_or_future_rows():
    data = model_data(600)
    full, records = rolling_response_model(data)
    t = 420
    changed = data.copy()
    changed.loc[t:, ["a_log_return", "a_gap_log_return"]] += 50
    other, _ = rolling_response_model(changed)
    np.testing.assert_allclose(full.predicted_response.iloc[:t+1], other.predicted_response.iloc[:t+1], equal_nan=True)
    prefix, _ = rolling_response_model(data.iloc[:t+1])
    pd.testing.assert_frame_equal(full.iloc[:t+1], prefix)
    assert max(records[0]["training_indices"]) < records[0]["idx"]


def test_training_is_two_calendar_years_and_252_mature_rows():
    data = model_data()
    result, records = rolling_response_model(data)
    assert not result.model_known.iloc[:252].any()
    assert result.model_known.iloc[252]
    for record in records:
        dates = data.date.iloc[record["training_indices"]]
        assert len(dates) >= 252
        assert dates.min() >= pd.Timestamp(record["date"])-pd.DateOffset(years=2)
        assert dates.max() < record["date"]


def test_extra_day_wait_keeps_current_risk_and_cancels_broken_first_low():
    model, _ = rolling_response_model(model_data(270))
    model.loc[268, ["price_signal", "external_positive", "underreaction"]] = True
    price = pd.DataFrame({"date": model.date, "wealth": 1., "low_w": .99, "es95": np.arange(270)/10000,
                          "pressure5": 0., "trend20": 0., "log_rv5_rv60": 0.})
    price.loc[269, "wealth"] = .98
    shifted = lag_features(model, price, 1)
    assert shifted.source_idx.iloc[-1] == 268
    assert shifted.es95.iloc[-1] == price.es95.iloc[-1]
    assert not signal_for(shifted.iloc[-1], "FULL")


def test_underreaction_does_not_enable_wrong_direction_or_no_model():
    row = pd.Series({"price_signal": True, "external_positive": False, "underreaction": True, "overreaction": False})
    assert signal_for(row, "PRICE_COMMON")
    assert not signal_for(row, "FULL")
    row.external_positive = True
    assert signal_for(row, "FULL")
    row.price_signal = False
    assert not signal_for(row, "FULL")
