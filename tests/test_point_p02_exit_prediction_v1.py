"""P02事前方差惊喜与尾部超越：原时钟、未知和固定模型。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_p02_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELDS, KIND, P02EntryController, p02_information_field,
    validate_wealth_clock, fit_p02_within, p02_prediction, training_rows, training_identity)
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction
CFG = {"ridge_alpha": 1., "feature_clip": 5., "recent_cycles": 20, "minimum_cycles": 10, "minimum_rows": 100}


def market_rows(prices=None,length=100):
    if prices is None:
        units = 100000+np.rint(np.cumsum(np.random.default_rng(821).normal(50.,700.,length))).astype(int)
        prices = units/1000.
    prices = np.asarray(prices,float)
    previous = np.r_[prices[0],prices[:-1]]
    data = pd.DataFrame({"date":pd.bdate_range("2020-01-01",periods=len(prices)),
        "close":prices,"high":np.maximum(previous,prices)+.003,"dividend":0.,"symbol":"510300.SH"})
    return recalculate_wealth(data)


def recalculate_wealth(data):
    out = data.copy()
    out["previous_close"] = out.close.shift()
    out["open"] = out.previous_close
    out.loc[0,"open"] = out.close.iloc[0]
    out["total_simple"] = (out.close+out.dividend)/out.previous_close-1
    out["total_log"] = np.log1p(out.total_simple)
    out["overnight_log"] = np.log((out.open+out.dividend)/out.previous_close)
    out["intraday_log"] = np.log((out.close+out.dividend)/(out.open+out.dividend))
    out["wealth"] = np.r_[1.,np.cumprod(1+out.total_simple.to_numpy(float)[1:])]
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
    old, new = fit_within_cycle_exit(rows, CFG), fit_p02_within(rows, CFG)
    assert len(new["features"]) == 10
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(new["coefficients"][-2:], 0., atol=1e-12, rtol=0)
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert p02_prediction(new, np.r_[values, constants]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)

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
    left, right = fit_p02_within(past, CFG), fit_p02_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    for field in FIELDS:
        missing = rows.copy()
        missing.loc[0, field] = np.nan
        with pytest.raises(ValueError, match="完整训练输入缺失"):
            fit_p02_within(missing, CFG)

def test_entry_unknown_stays_fixed_and_exit_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2,
                         FIELDS[0]: .3, FIELDS[1]: .5})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*10, "scale": [1.]*10,
             "feature_clip": 5., "coefficients": [0.]*10, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = P02EntryController(data, records, 2)
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


def test_causal_sixty_day_seed_current_return_excluded_and_tail_uses_previous_ratios():
    data = market_rows(length=380)
    validate_wealth_clock(data)
    field = p02_information_field(data)
    r = data.total_simple.to_numpy()
    assert field.loc[:60, "forecast_variance_before_current_return"].isna().all()
    assert field.loc[61, "forecast_variance_before_current_return"] == pytest.approx(np.mean(r[1:61]**2))
    expected = .94*field.loc[61, "forecast_variance_before_current_return"]+(1-.94)*r[61]**2
    assert field.loc[62, "forecast_variance_before_current_return"] == expected
    assert field.loc[:180, FIELDS].isna().all().all()
    assert np.isfinite(field.loc[181, FIELDS]).all()
    assert field.loc[181, "previous_valid_ratio_count252"] == 120
    t = 250
    history = field.raw_variance_surprise_ratio.iloc[max(0, t-252):t].dropna()
    q90 = history.quantile(.9)
    assert field.loc[t, "previous_ratio_q90_252_min120"] == q90
    assert field.loc[t, FIELDS[1]] == float(field.loc[t, FIELDS[0]] > q90)
    assert field.loc[t, "variance_latest_source_index"] == t-1
    assert field.loc[t, "tail_threshold_latest_source_index"] == t-1
    current = data.copy()
    current.loc[t, "close"] *= 1.1
    current = recalculate_wealth(current)
    changed = p02_information_field(current)
    assert changed.loc[t, "forecast_variance_before_current_return"] == field.loc[t, "forecast_variance_before_current_return"]
    assert changed.loc[t, "previous_ratio_q90_252_min120"] == q90
    assert changed.loc[t, FIELDS[0]] != field.loc[t, FIELDS[0]]


def test_source_gap_rebuilds_sixty_actual_intervals_without_filling_tail_unknown():
    for column, invalid in [("total_simple", np.nan), ("total_simple", np.inf),
                            ("overnight_log", np.nan), ("intraday_log", np.inf)]:
        data = market_rows(length=380)
        data.loc[210, column] = invalid
        field = p02_information_field(data)
        assert len(field) == len(data)
        assert field.loc[210:270, FIELDS].isna().all().all()
        assert field.loc[211:270, "forecast_variance_before_current_return"].isna().all()
        assert np.isfinite(field.loc[209, FIELDS]).all() and np.isfinite(field.loc[271, FIELDS]).all()
        assert field.loc[271, "variance_seed_initialized"]
        assert field.loc[271, "previous_valid_ratio_count252"] >= 120


def test_true_zero_shock_and_equal_tail_valid_but_zero_variance_unknown():
    zero = market_rows(np.repeat(100., 380))
    field = p02_information_field(zero)
    assert field.loc[61:, "forecast_variance_before_current_return"].eq(0.).all()
    assert field[FIELDS].isna().all().all()
    assert field.loc[61:, "status"].eq("NO_VIEW_NONPOSITIVE_FORECAST_VARIANCE").all()
    r = np.tile([.125, -.125], 190)[:379]
    prices = 100*np.cumprod(np.r_[1., 1+r])
    constant = market_rows(prices)
    constant["total_simple"] = np.r_[np.nan, r]
    equal = p02_information_field(constant)
    assert equal.loc[181:, FIELDS[0]].eq(1.).all()
    assert equal.loc[181:, FIELDS[1]].eq(0.).all()
    data = market_rows(length=380)
    data.loc[220, "total_simple"] = 0.
    event = p02_information_field(data)
    assert event.loc[220, FIELDS[0]] == 0. and event.loc[220, FIELDS[1]] == 0.


def test_future_price_cash_cannot_repaint_and_wealth_clock_not_repaired():
    data = market_rows(length=380)
    left = p02_information_field(data.iloc[:270])
    future = data.copy()
    future.loc[270:, "close"] *= 3.
    future.loc[280, "dividend"] = .1
    future = recalculate_wealth(future)
    pd.testing.assert_frame_equal(left, p02_information_field(future).iloc[:270].reset_index(drop=True), check_exact=True)
    cash = data.copy()
    cash.loc[200, "close"] -= .1
    cash.loc[200, "dividend"] = .1
    cash = recalculate_wealth(cash)
    validate_wealth_clock(cash)
    assert cash.total_simple.iloc[200] == pytest.approx(data.total_simple.iloc[200], abs=1e-14)
    bad = cash.copy()
    bad.loc[200, "wealth"] *= 2.
    with pytest.raises(AssertionError): validate_wealth_clock(bad)
    bad = data.copy()
    bad.loc[0, "total_log"] = bad.loc[0, "total_simple"] = 0.
    with pytest.raises(ValueError): validate_wealth_clock(bad)
    wrong = data.copy()
    wrong["symbol"] = "510500.SH"
    with pytest.raises(ValueError): p02_information_field(wrong)
