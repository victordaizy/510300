"""C06代理的除息、窗口和事前时钟，以及原模型退化和锁定版本检验。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_c06_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELD, KIND, C06EntryController,
    close_volume_distribution, fit_c06_within, c06_prediction, training_rows)
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction


CFG = {"ridge_alpha": 1., "feature_clip": 5., "recent_cycles": 20,
       "minimum_cycles": 10, "minimum_rows": 100}


def market_rows(length=90):
    rng = np.random.default_rng(701)
    price = 100+np.cumsum(rng.normal(0, .7, length))
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=length),
                         "close": price, "high": price+1., "low": price-1.,
                         "dividend": 0., "volume": rng.integers(100, 2000, length).astype(float)})


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


def test_future_prices_volumes_and_dividends_do_not_change_prefix():
    first = market_rows(90)
    future = market_rows(35)
    future["date"] = pd.date_range(first.date.iloc[-1]+pd.Timedelta(days=1), periods=35)
    future[["close", "high", "low"]] += 500
    future["dividend"] = 9.
    future["volume"] *= 10000
    before = close_volume_distribution(first)
    after = close_volume_distribution(pd.concat([first, future], ignore_index=True)).iloc[:90].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_ex_dividend_cash_shift_preserves_the_same_price_coordinate():
    original = market_rows()
    ex = original.copy()
    ex.loc[45:, ["close", "high", "low"]] -= 2.5
    ex.loc[45, "dividend"] = 2.5
    before, after = close_volume_distribution(original), close_volume_distribution(ex)
    np.testing.assert_allclose(before[FIELD], after[FIELD], equal_nan=True, atol=1e-14, rtol=0)
    np.testing.assert_allclose(before.known_atr20, after.known_atr20, equal_nan=True, atol=1e-14, rtol=0)


def test_bin_edges_are_left_closed_right_open_and_volume_unit_cancels():
    data = market_rows(60)
    data["close"] = 100.; data["high"] = 101.; data["low"] = 99.; data["volume"] = 1.
    for t, price, volume in [(5, 102., 2.), (6, 104., 3.), (7, 106., 5.), (8, 101.9, 7.)]:
        data.loc[t, ["close", "high", "low", "volume"]] = [price, price+1, price-1, volume]
    result = close_volume_distribution(data).iloc[-1]
    assert result.known_atr20 == 2.
    assert result[FIELD] == pytest.approx(5/73)
    assert result.volume_bin_1_to_2_atr_fraction == pytest.approx(2/73)
    assert result.volume_bin_2_to_3_atr_fraction == pytest.approx(3/73)
    scaled = data.copy(); scaled.volume *= 10000
    assert close_volume_distribution(scaled)[FIELD].iloc[-1] == pytest.approx(result[FIELD])


def test_missing_or_zero_atr_is_unknown_and_window_cannot_drop_a_day():
    data = market_rows()
    data.loc[35, "volume"] = np.nan
    result = close_volume_distribution(data)
    assert result.loc[59:89, FIELD].isna().all()
    flat = market_rows(60)
    flat[["close", "high", "low"]] = 100.
    result = close_volume_distribution(flat).iloc[-1]
    assert pd.isna(result[FIELD]) and result.status == "NO_VIEW_NONPOSITIVE_KNOWN_ATR20"


def test_constant_ninth_feature_degenerates_to_the_original_eight_feature_model():
    rows = sample_rows(); rows[FIELD] = .4
    old, new = fit_within_cycle_exit(rows, CFG), fit_c06_within(rows, CFG)
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    assert abs(new["coefficients"][-1]) < 1e-12
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert c06_prediction(new, np.r_[values, .4]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)


def test_unmatured_future_rows_cannot_change_the_fit_and_missing_rows_are_not_deleted():
    rows = sample_rows()
    past, _ = training_rows(rows, 400, CFG)
    future = rows.copy(); future["cycle_id"] += 100
    future["origin_index"] += 1000; future["exit_index"] += 1000
    future["target"] += 1000; future[FIELD] += 1000
    selected, _ = training_rows(pd.concat([rows, future], ignore_index=True), 400, CFG)
    left, right = fit_c06_within(past, CFG), fit_c06_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    missing = rows.copy(); missing.loc[0, FIELD] = np.nan
    with pytest.raises(ValueError):
        fit_c06_within(missing, CFG)


def test_unknown_entry_version_stays_unknown_and_new_cycle_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2, FIELD: .3})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*9,
             "scale": [1.]*9, "feature_clip": 5., "coefficients": [0.]*9, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = C06EntryController(data, records, 2)
    first = {"cycle_id": 1, "entry_index": 2, "entry_cost_cny": 1000., "mode": 1}
    assert controller(2, first, 1000., 1000.)["continuation_prediction"] is None
    assert controller(4, first, 1000., 1000.)["continuation_prediction"] is None
    second = {"cycle_id": 2, "entry_index": 4, "entry_cost_cny": 1000., "mode": 1}
    assert not controller(4, second, 1000., 1000.)["learned_exit_requested"]
    assert controller(5, second, 1000., 1000.)["learned_exit_requested"]
    third = copy.deepcopy(second); third["entry_index"] = 5
    with pytest.raises(ValueError):
        controller(6, third, 1000., 1000.)
