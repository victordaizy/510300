"""D03原卡两列的新条件用途：可知窗口、经济单位、完整成员和固定入场版本。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_d03_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELDS, KIND, D03EntryController, d03_information_field,
    validate_wealth_clock, fit_d03_within, d03_prediction, training_rows, training_identity)
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction
CFG = {"ridge_alpha": 1., "feature_clip": 5., "recent_cycles": 20, "minimum_cycles": 10, "minimum_rows": 100}


def market_rows(prices=None, length=90):
    if prices is None:
        prices = 100+np.cumsum(np.random.default_rng(821).normal(.05, .7, length))
    prices = np.asarray(prices, float)
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(prices)), "close": prices,
        "open": prices, "high": prices+2., "low": prices-2.,
        "previous_close": np.r_[np.nan, prices[:-1]], "dividend": 0., "symbol": "510300.SH"})
    return recalculate_wealth(data)


def recalculate_wealth(data):
    out = data.copy()
    out["total_simple"] = (out.close+out.dividend)/out.previous_close-1
    out["total_log"] = np.log1p(out.total_simple)
    out["wealth"] = np.r_[1., np.cumprod(1+out.total_simple.to_numpy(float)[1:])]
    return out

def sample_rows():
    rng = np.random.default_rng(781)
    x = rng.normal(size=(144, 8))
    x[:, 3] = 1.
    rows = pd.DataFrame(x, columns=BASE_FEATURES)
    rows[FIELDS] = rng.normal(size=(len(rows), 2))
    rows["cycle_id"] = np.repeat(np.arange(12), 12)
    rows["target"] = .02*x[:, 1] - .01*rows[FIELDS[0]] + rng.normal(0, .01, len(rows))
    rows["sample_weight"] = 1/12
    rows["origin_index"] = np.tile(np.arange(12), 12) + np.repeat(np.arange(12)*20, 12)
    rows["exit_index"] = 50 + rows.cycle_id*20
    return rows

def test_constant_two_feature_block_matches_original_eight_predictions():
    rows = sample_rows()
    constants = [.4, .8]
    rows[FIELDS] = constants
    old, new = fit_within_cycle_exit(rows, CFG), fit_d03_within(rows, CFG)
    assert len(new["features"]) == 10
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(new["coefficients"][-2:], 0., atol=1e-12, rtol=0)
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert d03_prediction(new, np.r_[values, constants]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)

def test_unmatured_labels_excluded_and_missing_any_training_column_rejects_whole_input():
    rows = sample_rows()
    past, _ = training_rows(rows, 400, CFG)
    future = rows.copy()
    future["cycle_id"] += 100
    future["origin_index"] += 1000
    future["exit_index"] += 1000
    future["target"] += 1000
    future[FIELDS] += 1000
    selected, _ = training_rows(pd.concat([rows, future], ignore_index=True), 400, CFG)
    left, right = fit_d03_within(past, CFG), fit_d03_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    for field in FIELDS:
        missing = rows.copy()
        missing.loc[0, field] = np.nan
        with pytest.raises(ValueError, match="完整训练输入缺失"):
            fit_d03_within(missing, CFG)

def test_entry_unknown_stays_fixed_and_exit_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2,
                         FIELDS[0]: .3, FIELDS[1]: .5})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*10, "scale": [1.]*10,
             "feature_clip": 5., "coefficients": [0.]*10, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = D03EntryController(data, records, 2)
    first = {"cycle_id": 1, "entry_index": 2, "entry_cost_cny": 1000., "mode": 1}
    assert controller(2, first, 1000., 1000.)["continuation_prediction"] is None
    assert controller(4, first, 1000., 1000.)["continuation_prediction"] is None
    second = {"cycle_id": 2, "entry_index": 4, "entry_cost_cny": 1000., "mode": 1}
    assert not controller(4, second, 1000., 1000.)["learned_exit_requested"]
    assert controller(5, second, 1000., 1000.)["learned_exit_requested"]
    altered = copy.deepcopy(second)
    altered["entry_index"] = 5
    with pytest.raises(ValueError):
        controller(6, altered, 1000., 1000.)


def test_inclusive_five_days_mean_and_strict_positive_count():
    data = market_rows(np.repeat(100., 12))
    daily = np.array([-1., -.5, 0., .5, 1., 0., -.5, .5, 1., -1., 0., 0.])
    data["high"] = 102.
    data["low"] = 98.
    data["close"] = 100+2*daily
    data["open"] = data.close
    field = d03_information_field(data)
    assert field.loc[:3, FIELDS].isna().all().all()
    for t in range(4, len(data)):
        assert field.loc[t, FIELDS[0]] == pytest.approx(daily[t-4:t+1].mean(), abs=1e-14)
        assert field.loc[t, FIELDS[1]] == np.sum(daily[t-4:t+1] > 0)
        assert field.loc[t, "five_day_start_index"] == t-4
        assert field.loc[t, "latest_source_index"] == t
    assert field.loc[4, FIELDS[0]] == 0.
    assert field.loc[4, FIELDS[1]] == 2.
    changed = data.copy()
    changed.loc[4, "close"] = changed.loc[4, "open"] = 98.
    assert d03_information_field(changed).loc[4, FIELDS[1]] == 1.


def test_zero_range_invalid_ohlc_or_missing_poison_exactly_five_actual_days():
    for mode in ["zero_range", "missing_high", "close_outside", "negative_low", "infinite_open"]:
        data = market_rows(np.repeat(100., 18))
        if mode == "zero_range":
            data.loc[7, ["high", "low"]] = 100.
        elif mode == "missing_high":
            data.loc[7, "high"] = np.nan
        elif mode == "close_outside":
            data.loc[7, "close"] = 103.
        elif mode == "negative_low":
            data.loc[7, "low"] = -1.
        else:
            data.loc[7, "open"] = np.inf
        field = d03_information_field(data)
        assert field.loc[7:11, FIELDS].isna().all().all()
        assert np.isfinite(field.loc[12, FIELDS].to_numpy(float)).all()
        assert np.isfinite(field.loc[6, FIELDS].to_numpy(float)).all()
    neutral = d03_information_field(market_rows(np.repeat(100., 18)))
    assert (neutral.loc[4:, FIELDS] == 0.).all().all()
    assert neutral.loc[4:, "status"].eq("D03_FIVE_COMPLETE_OHLC_DAYS_AVAILABLE").all()


def test_future_ohlc_and_cash_cannot_repaint_common_cash_shift_and_scale_cancel():
    data = market_rows(length=70)
    original = d03_information_field(data.iloc[:35])
    future = data.copy()
    future.loc[35:, ["open", "high", "low", "close"]] *= 3.
    future.loc[40:, "dividend"] = 10.
    pd.testing.assert_frame_equal(original, d03_information_field(future).iloc[:35].reset_index(drop=True), check_exact=True)
    shifted = data.copy()
    for column in ["open", "high", "low", "close"]:
        shifted[column] += np.linspace(0., 4., len(data))
    scaled = data.copy()
    scaled[["open", "high", "low", "close"]] *= 7.
    for changed in [shifted, scaled]:
        np.testing.assert_allclose(d03_information_field(data)[FIELDS], d03_information_field(changed)[FIELDS],
                                   atol=1e-12, rtol=0, equal_nan=True)
    wrong_symbol = data.copy()
    wrong_symbol.loc[0, "symbol"] = "000300.SH"
    with pytest.raises(ValueError): d03_information_field(wrong_symbol)
    disorder = data.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError): d03_information_field(disorder)


def test_same_close_path_and_economic_returns_can_have_distinct_intraday_location():
    first = market_rows(length=35)
    second = first.copy()
    first["high"], first["low"] = first.close+1., first.close-3.
    second["high"], second["low"] = second.close+3., second.close-1.
    for name in ["close", "wealth", "total_log", "total_simple"]:
        pd.testing.assert_series_equal(first[name], second[name], check_exact=True)
    a, b = d03_information_field(first), d03_information_field(second)
    assert a.loc[34, FIELDS[0]] == .5 and b.loc[34, FIELDS[0]] == -.5
    assert a.loc[34, FIELDS[1]] == 5. and b.loc[34, FIELDS[1]] == 0.
    validate_wealth_clock(first)
    validate_wealth_clock(second)
