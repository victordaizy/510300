"""A04拐点延迟、除息和原退出模型时钟的必要检验。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_a04_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELD, KIND, A04EntryController,
    confirmed_swing_retracement, fit_a04_within, a04_prediction, training_rows)
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction


CFG = {"ridge_alpha": 1., "feature_clip": 5., "recent_cycles": 20,
       "minimum_cycles": 10, "minimum_rows": 100}


def market(prices):
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(prices)),
                         "close": np.asarray(prices, float), "dividend": 0.})


def sample_rows():
    rng = np.random.default_rng(751)
    x = rng.normal(size=(144, 8)); x[:, 3] = 1.
    rows = pd.DataFrame(x, columns=BASE_FEATURES)
    rows[FIELD] = rng.normal(.4, .7, len(rows))
    rows["cycle_id"] = np.repeat(np.arange(12), 12)
    rows["target"] = .02*x[:, 1]-.01*rows[FIELD]+rng.normal(0, .01, len(rows))
    rows["sample_weight"] = 1/12
    rows["origin_index"] = np.tile(np.arange(12), 12)+np.repeat(np.arange(12)*20, 12)
    rows["exit_index"] = 50+rows.cycle_id*20
    return rows


def test_high_is_available_only_after_two_complete_right_bars():
    data = market([10, 9, 8, 9, 10, 12, 11, 10])
    values = confirmed_swing_retracement(data)
    assert values[FIELD].iloc[:7].isna().all()
    last = values.iloc[7]
    assert last.low_index == 2 and last.high_index == 5
    assert last.low_confirmation_index == 4 and last.high_confirmation_index == 7
    assert last[FIELD] == pytest.approx(.5)
    assert last.pair_changed_today


def test_future_bars_and_dividends_cannot_repaint_past_pivots():
    prefix = market([10, 9, 8, 9, 10, 12, 11, 10])
    future = market([25, 20, 29, 15, 40, 12])
    future["date"] = pd.date_range(prefix.date.iloc[-1]+pd.Timedelta(days=1), periods=len(future))
    future["dividend"] = 3.
    before = confirmed_swing_retracement(prefix)
    after = confirmed_swing_retracement(pd.concat([prefix, future], ignore_index=True)).iloc[:len(prefix)].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_ex_dividend_shift_preserves_confirmations_and_fraction():
    data = market([10, 9, 8, 9, 10, 12, 11, 10, 11, 13])
    ex = data.copy(); ex.loc[4:, "close"] -= 2.5; ex.loc[4, "dividend"] = 2.5
    before, after = confirmed_swing_retracement(data), confirmed_swing_retracement(ex)
    pd.testing.assert_frame_equal(before, after, check_exact=False, atol=1e-14, rtol=0)


def test_equal_pivots_remain_unknown_and_fraction_can_cross_zero_and_one():
    tied = confirmed_swing_retracement(market([10, 9, 8, 9, 10, 12, 12, 10]))
    assert tied[FIELD].isna().all()
    monotone = confirmed_swing_retracement(market([8, 9, 10, 11, 12, 13, 14, 15]))
    assert monotone[FIELD].isna().all()
    above = confirmed_swing_retracement(market([10, 9, 8, 9, 10, 12, 11, 10, 11, 13])).iloc[-1]
    below = confirmed_swing_retracement(market([10, 9, 8, 9, 10, 12, 11, 10, 7, 6])).iloc[-1]
    assert above[FIELD] == pytest.approx(-.25)
    assert below[FIELD] == pytest.approx(1.5)
    assert above.high_index == below.high_index == 5
    missing = market([10, 9, 8, 9, 10, 12, 11, 10]); missing.loc[1, "close"] = np.nan
    with pytest.raises(ValueError):
        confirmed_swing_retracement(missing)


def test_constant_ninth_feature_matches_original_eight_feature_predictions():
    rows = sample_rows(); rows[FIELD] = .4
    old, new = fit_within_cycle_exit(rows, CFG), fit_a04_within(rows, CFG)
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    assert abs(new["coefficients"][-1]) < 1e-12
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert a04_prediction(new, np.r_[values, .4]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)


def test_unmatured_future_labels_are_excluded_and_missing_training_is_not_deleted():
    rows = sample_rows(); past, _ = training_rows(rows, 400, CFG)
    future = rows.copy(); future["cycle_id"] += 100; future["origin_index"] += 1000
    future["exit_index"] += 1000; future["target"] += 1000; future[FIELD] += 1000
    selected, _ = training_rows(pd.concat([rows, future], ignore_index=True), 400, CFG)
    left, right = fit_a04_within(past, CFG), fit_a04_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    missing = rows.copy(); missing.loc[0, FIELD] = np.nan
    with pytest.raises(ValueError):
        fit_a04_within(missing, CFG)


def test_entry_version_stays_fixed_and_exit_requires_two_negative_forecasts():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2, FIELD: .3})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*9, "scale": [1.]*9,
             "feature_clip": 5., "coefficients": [0.]*9, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = A04EntryController(data, records, 2)
    first = {"cycle_id": 1, "entry_index": 2, "entry_cost_cny": 1000., "mode": 1}
    assert controller(2, first, 1000., 1000.)["continuation_prediction"] is None
    assert controller(4, first, 1000., 1000.)["continuation_prediction"] is None
    second = {"cycle_id": 2, "entry_index": 4, "entry_cost_cny": 1000., "mode": 1}
    assert not controller(4, second, 1000., 1000.)["learned_exit_requested"]
    assert controller(5, second, 1000., 1000.)["learned_exit_requested"]
    altered = copy.deepcopy(second); altered["entry_index"] = 5
    with pytest.raises(ValueError):
        controller(6, altered, 1000., 1000.)
