"""B01原卡两列的新条件用途：可知窗口、经济单位、完整成员和固定入场版本。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_b01_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELDS, KIND, B01EntryController, b01_information_block,
    validate_market_clock, fit_b01_within, b01_prediction, training_rows, training_identity)
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction
CFG = {"ridge_alpha": 1., "feature_clip": 5., "recent_cycles": 20, "minimum_cycles": 10, "minimum_rows": 100}


def market_rows(prices=None, length=90):
    if prices is None:
        prices = 100+np.cumsum(np.random.default_rng(821).normal(.05, .7, length))
    prices = np.asarray(prices, float)
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(prices)), "close": prices,
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
    old, new = fit_within_cycle_exit(rows, CFG), fit_b01_within(rows, CFG)
    assert len(new["features"]) == 10
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(new["coefficients"][-2:], 0., atol=1e-12, rtol=0)
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert b01_prediction(new, np.r_[values, constants]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)

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
    left, right = fit_b01_within(past, CFG), fit_b01_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    for field in FIELDS:
        missing = rows.copy()
        missing.loc[0, field] = np.nan
        with pytest.raises(ValueError, match="完整训练输入缺失"):
            fit_b01_within(missing, CFG)

def test_entry_unknown_stays_fixed_and_exit_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2,
                         FIELDS[0]: .3, FIELDS[1]: .5})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*10, "scale": [1.]*10,
             "feature_clip": 5., "coefficients": [0.]*10, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = B01EntryController(data, records, 2)
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


def test_five_points_include_today_and_rv_has_twenty_daily_log_intervals():
    data = market_rows(); field = b01_information_block(data)
    assert field.loc[:3, FIELDS[0]].isna().all()
    assert field.loc[:19, FIELDS[1]].isna().all()
    for t in [4, 20, 39, 70]:
        raw = data.wealth.iloc[t]/np.mean(data.wealth.iloc[t-4:t+1])-1
        assert field.loc[t, FIELDS[0]] == pytest.approx(raw, abs=1e-14)
        assert field.loc[t, "ma5_start_index"] == t-4
        if t >= 20:
            rv = np.std(data.total_log.iloc[t-19:t+1].to_numpy(float), ddof=1)
            assert field.loc[t, FIELDS[1]] == pytest.approx(raw/(rv*np.sqrt(5)), abs=1e-12)
            assert field.loc[t, "rv20_wealth_start_index"] == t-20
    assert field.loc[20, FIELDS[0]] != pytest.approx(data.wealth.iloc[20]/data.wealth.iloc[15:20].mean()-1, abs=1e-8)


def test_known_cash_wealth_and_economic_log_identity_are_required():
    data = market_rows(); ex = data.copy()
    ex.loc[30, "close"] -= 1.; ex.loc[30, "dividend"] = 1.
    ex.loc[31, "previous_close"] = ex.loc[30, "close"]
    ex = recalculate_wealth(ex)
    validate_market_clock(ex)
    np.testing.assert_allclose(b01_information_block(ex).loc[30, FIELDS].to_numpy(float),
        b01_information_block(data).loc[30, FIELDS].to_numpy(float), atol=1e-12, rtol=0)
    bad = data.copy(); bad.loc[30, "wealth"] *= 1.1
    with pytest.raises(AssertionError): validate_market_clock(bad)
    bad = data.copy(); bad.loc[30, "total_log"] += .01
    with pytest.raises(AssertionError): validate_market_clock(bad)
    bad = data.copy(); bad.loc[30, "symbol"] = "000300.SH"
    with pytest.raises(ValueError): validate_market_clock(bad)


def test_future_price_and_cash_cannot_repaint_and_wealth_scale_cancels():
    data = market_rows(); left = b01_information_block(data.iloc[:45])
    changed = data.copy(); changed.loc[45:, "close"] *= 3.; changed.loc[60, "dividend"] = 4.
    changed["previous_close"] = changed.close.shift()
    changed = recalculate_wealth(changed)
    pd.testing.assert_frame_equal(left, b01_information_block(changed).iloc[:45].reset_index(drop=True), check_exact=True)
    scaled = data.copy(); scaled["wealth"] *= 7.
    np.testing.assert_allclose(b01_information_block(data)[FIELDS], b01_information_block(scaled)[FIELDS],
        atol=1e-12, rtol=0, equal_nan=True)


def test_missing_windows_and_zero_volatility_remain_unknown_but_true_raw_zero_is_valid():
    data = market_rows(); data.loc[35, "wealth"] = np.nan
    field = b01_information_block(data)
    assert field.loc[35:39, FIELDS[0]].isna().all()
    assert field.loc[35:55, FIELDS[1]].isna().all()
    assert np.isfinite(field.loc[56, FIELDS[1]])
    data = market_rows(); data.loc[35, "total_log"] = np.nan
    assert b01_information_block(data).loc[35:54, FIELDS[1]].isna().all()
    flat = market_rows(np.repeat(100., 40)); f = b01_information_block(flat)
    assert (f.loc[4:, FIELDS[0]] == 0.).all()
    assert f.loc[20:, FIELDS[1]].isna().all()
    assert (f.loc[20:, "field_status"] == "NO_VIEW_ZERO_RETURN_VOLATILITY").all()
    zero = market_rows(np.r_[np.repeat(100., 20), [99., 101., 99., 101., 100.]])
    f = b01_information_block(zero)
    assert f.loc[24, FIELDS[0]] == 0. and f.loc[24, FIELDS[1]] == 0.


def test_same_five_day_endpoint_and_return_volatility_can_have_different_ma_position():
    first = np.r_[np.repeat(.001, 15), [-.03, .01, .02, -.01, .015]]
    second = np.r_[np.repeat(.001, 15), [.015, -.01, .02, .01, -.03]]
    a = market_rows(100*np.exp(np.r_[0., np.cumsum(first)]))
    b = market_rows(100*np.exp(np.r_[0., np.cumsum(second)]))
    left, right = b01_information_block(a), b01_information_block(b)
    assert a.wealth.iloc[-1] == pytest.approx(b.wealth.iloc[-1], abs=1e-14)
    assert a.total_log.iloc[-5:].sum() == pytest.approx(b.total_log.iloc[-5:].sum(), abs=1e-14)
    assert left.daily_log_std20.iloc[-1] == pytest.approx(right.daily_log_std20.iloc[-1], abs=1e-14)
    assert abs(left[FIELDS[0]].iloc[-1]-right[FIELDS[0]].iloc[-1]) > .01


def test_ordered_cache_includes_both_fields_targets_weights_and_members():
    rows = sample_rows(); reference = training_identity(rows, CFG)
    for column in FIELDS+["target", "sample_weight", "origin_index"]:
        changed = rows.copy(); changed.loc[0, column] += 1 if column == "origin_index" else .1
        assert training_identity(changed, CFG) != reference
    assert training_identity(rows.iloc[::-1], CFG) != reference
