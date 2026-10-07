"""C02三列时钟、金额和原模型退化、成熟及入场版本的必要检验。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_c02_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELDS, KIND, C02EntryController,
    c02_information_block, fit_c02_within, c02_prediction, training_rows)
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction


CFG = {"ridge_alpha": 1., "feature_clip": 5., "recent_cycles": 20,
       "minimum_cycles": 10, "minimum_rows": 100}


def market(n=45):
    r = np.resize(np.array([.004, .012, -.007, .001, -.002, .009, -.010]), n)
    close = 10 * np.exp(np.cumsum(r))
    amount = np.arange(1, n + 1, dtype=float) * 100
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n),
                         "close": close, "previous_close": np.r_[np.nan, close[:-1]],
                         "dividend": 0., "amount": amount, "volume": amount / close,
                         "symbol": "510300.SH", "amount_unit": "CNY", "volume_unit": "share"})


def sample_rows():
    rng = np.random.default_rng(781)
    x = rng.normal(size=(144, 8))
    x[:, 3] = 1.
    rows = pd.DataFrame(x, columns=BASE_FEATURES)
    rows[FIELDS] = rng.normal(size=(len(rows), 3))
    rows["cycle_id"] = np.repeat(np.arange(12), 12)
    rows["target"] = .02*x[:, 1] - .01*rows[FIELDS[0]] + rng.normal(0, .01, len(rows))
    rows["sample_weight"] = 1/12
    rows["origin_index"] = np.tile(np.arange(12), 12) + np.repeat(np.arange(12)*20, 12)
    rows["exit_index"] = 50 + rows.cycle_id*20
    return rows


def test_prior20_amount_excludes_today_and_progress_has_exact_daily_volatility_units():
    data = market()
    f = c02_information_block(data)
    assert f.loc[:19, FIELDS[0]].isna().all()
    assert f.loc[20, FIELDS[0]] == pytest.approx(2100 / 1050)
    r = np.log(data.close / data.previous_close)
    assert f.loc[20, FIELDS[1]] == pytest.approx(abs(r.iloc[20]) / r.iloc[1:21].std(ddof=1))
    assert f.loc[20, FIELDS[2]] == np.sign(r.iloc[20])
    assert f.loc[20, "field_status"] == "FIELD_BLOCK_AVAILABLE"
    assert f.loc[20, "latest_source_index"] == 20


def test_future_amount_and_dividend_cannot_repaint_prefix():
    data = market()
    before = c02_information_block(data.iloc[:35])
    data.loc[35:, "amount"] *= 1000
    data.loc[40, "dividend"] = 3.
    data["volume"] = data.amount / data.close
    after = c02_information_block(data).iloc[:35].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_known_ex_dividend_is_in_current_return_and_wrong_units_or_previous_close_fail():
    data = market(23)
    ex = data.copy()
    ex.loc[22, "close"] -= .4
    ex.loc[22, "dividend"] = .4
    ex["volume"] = ex.amount / ex.close
    left, right = c02_information_block(data), c02_information_block(ex)
    np.testing.assert_allclose(left.loc[22, FIELDS].to_numpy(float), right.loc[22, FIELDS].to_numpy(float), atol=1e-13, rtol=0)
    bad = data.copy()
    bad.loc[22, "previous_close"] += 1.
    with pytest.raises(ValueError, match="前收盘"):
        c02_information_block(bad)
    bad = data.copy()
    bad.loc[22, "amount_unit"] = "万元"
    with pytest.raises(ValueError, match="单位"):
        c02_information_block(bad)


def test_missing_window_is_not_skipped_and_true_zero_differs_from_zero_volatility():
    data = market()
    data.loc[20, "amount"] = np.nan
    f = c02_information_block(data)
    assert f.loc[20:40, FIELDS[0]].isna().all()
    assert np.isfinite(f.loc[41, FIELDS[0]])
    data = market()
    data.loc[20, ["amount", "volume"]] = 0.
    assert c02_information_block(data).loc[20, FIELDS[0]] == 0.
    flat = market()
    flat["close"] = 10.
    flat["previous_close"] = 10.
    flat.loc[0, "previous_close"] = np.nan
    flat["volume"] = flat.amount / 10.
    f = c02_information_block(flat)
    assert f.loc[20:, FIELDS[1]].isna().all()
    assert (f.loc[20:, FIELDS[2]] == 0.).all()
    assert (f.loc[20:, "field_status"] == "NO_VIEW_INCOMPLETE_MARKET_BLOCK").all()


def test_constant_three_feature_block_matches_original_eight_predictions():
    rows = sample_rows()
    constants = [.4, .8, -1.]
    rows[FIELDS] = constants
    old, new = fit_within_cycle_exit(rows, CFG), fit_c02_within(rows, CFG)
    assert len(new["features"]) == 11
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(new["coefficients"][-3:], 0., atol=1e-12, rtol=0)
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert c02_prediction(new, np.r_[values, constants]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)


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
    left, right = fit_c02_within(past, CFG), fit_c02_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    for field in FIELDS:
        missing = rows.copy()
        missing.loc[0, field] = np.nan
        with pytest.raises(ValueError, match="完整训练输入缺失"):
            fit_c02_within(missing, CFG)


def test_entry_unknown_stays_fixed_and_exit_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2,
                         FIELDS[0]: .3, FIELDS[1]: .5, FIELDS[2]: -1.})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*11, "scale": [1.]*11,
             "feature_clip": 5., "coefficients": [0.]*11, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = C02EntryController(data, records, 2)
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
