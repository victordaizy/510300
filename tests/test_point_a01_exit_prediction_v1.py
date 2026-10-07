"""A01既有路径字段的新用途：财富时钟、窗口、原模型退化和固定版本。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_a01_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELD, KIND, A01EntryController, directional_path_efficiency,
    validate_wealth_clock, fit_a01_within, a01_prediction, training_rows, training_identity)
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction

CFG = {"ridge_alpha": 1., "feature_clip": 5., "recent_cycles": 20,
       "minimum_cycles": 10, "minimum_rows": 100}


def sample_rows():
    rng = np.random.default_rng(702)
    x = rng.normal(size=(144, 8))
    x[:, 3] = 1.
    rows = pd.DataFrame(x, columns=BASE_FEATURES)
    rows[FIELD] = rng.uniform(0, 1, len(rows))
    rows["cycle_id"] = np.repeat(np.arange(12), 12)
    rows["target"] = .02*x[:, 1]-.01*rows[FIELD]+rng.normal(0, .01, len(rows))
    rows["sample_weight"] = 1/12
    rows["origin_index"] = np.tile(np.arange(12), 12)+np.repeat(np.arange(12)*20, 12)
    rows["exit_index"] = 50+rows.cycle_id*20
    return rows

def test_constant_ninth_feature_degenerates_to_the_original_eight_feature_model():
    rows = sample_rows(); rows[FIELD] = .4
    old, new = fit_within_cycle_exit(rows, CFG), fit_a01_within(rows, CFG)
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    assert abs(new["coefficients"][-1]) < 1e-12
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert a01_prediction(new, np.r_[values, .4]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)

def test_unmatured_future_rows_cannot_change_the_fit_and_missing_rows_are_not_deleted():
    rows = sample_rows()
    past, _ = training_rows(rows, 400, CFG)
    future = rows.copy(); future["cycle_id"] += 100
    future["origin_index"] += 1000; future["exit_index"] += 1000
    future["target"] += 1000; future[FIELD] += 1000
    selected, _ = training_rows(pd.concat([rows, future], ignore_index=True), 400, CFG)
    left, right = fit_a01_within(past, CFG), fit_a01_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    missing = rows.copy(); missing.loc[0, FIELD] = np.nan
    with pytest.raises(ValueError):
        fit_a01_within(missing, CFG)

def test_unknown_entry_version_stays_unknown_and_new_cycle_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2, FIELD: .3})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*9,
             "scale": [1.]*9, "feature_clip": 5., "coefficients": [0.]*9, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = A01EntryController(data, records, 2)
    first = {"cycle_id": 1, "entry_index": 2, "entry_cost_cny": 1000., "mode": 1}
    assert controller(2, first, 1000., 1000.)["continuation_prediction"] is None
    assert controller(4, first, 1000., 1000.)["continuation_prediction"] is None
    second = {"cycle_id": 2, "entry_index": 4, "entry_cost_cny": 1000., "mode": 1}
    assert not controller(4, second, 1000., 1000.)["learned_exit_requested"]
    assert controller(5, second, 1000., 1000.)["learned_exit_requested"]
    third = copy.deepcopy(second); third["entry_index"] = 5
    with pytest.raises(ValueError):
        controller(6, third, 1000., 1000.)


def market_rows(prices=None, length=90):
    if prices is None:
        prices = 100+np.cumsum(np.random.default_rng(703).normal(.05, .7, length))
    prices = np.asarray(prices, float)
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(prices)), "close": prices,
                         "previous_close": np.r_[np.nan, prices[:-1]], "dividend": 0., "symbol": "510300.SH"})
    return recalculate_wealth(data)


def recalculate_wealth(data):
    out = data.copy()
    out["total_simple"] = (out.close+out.dividend)/out.previous_close-1
    out["wealth"] = np.r_[1., np.cumprod(1+out.total_simple.to_numpy(float)[1:])]
    return out


def test_twenty_intervals_have_direction_and_true_net_zero_is_valid():
    up = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=21), "wealth": np.arange(1., 22.)})
    down = up.copy(); down["wealth"] = up.wealth.iloc[::-1].to_numpy()
    loop = up.copy(); loop["wealth"] = np.array([1., 2.]*10+[1.])
    assert directional_path_efficiency(up)[FIELD].iloc[:20].isna().all()
    assert directional_path_efficiency(up)[FIELD].iloc[20] == 1.
    assert directional_path_efficiency(down)[FIELD].iloc[20] == -1.
    result = directional_path_efficiency(loop).iloc[20]
    assert result[FIELD] == 0. and result.status == "A01_AVAILABLE"
    assert result.window_start_index == 0 and result.latest_source_index == 20


def test_ex_dividend_wealth_and_source_identity_are_checked():
    original = market_rows(np.full(90, 100.))
    ex = original.copy(); ex.loc[45:, "close"] -= 2.
    ex.loc[45, "dividend"] = 2.
    ex["previous_close"] = ex.close.shift()
    ex = recalculate_wealth(ex)
    validate_wealth_clock(ex)
    np.testing.assert_allclose(ex.wealth, original.wealth, atol=1e-12, rtol=0)
    assert directional_path_efficiency(ex)[FIELD].isna().all()
    bad = ex.copy(); bad["wealth"] = bad.close/bad.close.iloc[0]
    with pytest.raises(AssertionError): validate_wealth_clock(bad)
    wrong = ex.copy(); wrong.loc[30, "previous_close"] *= 1.01
    with pytest.raises(AssertionError): validate_wealth_clock(wrong)
    wrong_symbol = ex.copy(); wrong_symbol.loc[0, "symbol"] = "510300.SZ"
    with pytest.raises(ValueError): validate_wealth_clock(wrong_symbol)


def test_future_prices_and_cash_do_not_repaint_and_scale_cancels():
    data = market_rows()
    longer = market_rows(np.r_[data.close.to_numpy(), np.linspace(data.close.iloc[-1]+500, 1000., 35)])
    longer.loc[90:, "dividend"] = 9.
    longer = recalculate_wealth(longer)
    validate_wealth_clock(longer)
    original = directional_path_efficiency(data)
    appended = directional_path_efficiency(longer).iloc[:len(data)].reset_index(drop=True)
    pd.testing.assert_frame_equal(original, appended)
    scaled = data.copy(); scaled["wealth"] *= 7.
    np.testing.assert_allclose(original[FIELD], directional_path_efficiency(scaled)[FIELD], equal_nan=True, atol=1e-14, rtol=0)


def test_missing_complete_window_and_zero_path_remain_unknown():
    data = market_rows(np.full(60, 100.))
    result = directional_path_efficiency(data)
    assert result[FIELD].isna().all()
    assert result.status.iloc[20] == "NO_VIEW_ZERO_OR_INVALID_PATH"
    missing = market_rows(); missing.loc[30, "wealth"] = np.nan
    result = directional_path_efficiency(missing)
    assert result[FIELD].iloc[30:51].isna().all()
    assert result.status.iloc[30:51].eq("NO_VIEW_INCOMPLETE_21_WEALTH_POINTS").all()
    assert np.isfinite(result[FIELD].iloc[51])


def test_complete_ordered_cache_identity_includes_ninth_target_and_weights():
    rows = sample_rows()
    reference = training_identity(rows, CFG)
    for column in (FIELD, "target", "sample_weight"):
        changed = rows.copy(); changed.loc[0, column] += .1
        assert training_identity(changed, CFG) != reference
    assert training_identity(rows.iloc[::-1], CFG) != reference
