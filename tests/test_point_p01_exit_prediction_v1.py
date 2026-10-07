"""P01全天下行占比/总波动：真实窗口、零风险、经济坐标及固定模型。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_p01_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELDS, KIND, P01EntryController, p01_information_field,
    validate_wealth_clock, fit_p01_within, p01_prediction, training_rows, training_identity)
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
    old, new = fit_within_cycle_exit(rows, CFG), fit_p01_within(rows, CFG)
    assert len(new["features"]) == 10
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(new["coefficients"][-2:], 0., atol=1e-12, rtol=0)
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert p01_prediction(new, np.r_[values, constants]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)

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
    left, right = fit_p01_within(past, CFG), fit_p01_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    for field in FIELDS:
        missing = rows.copy()
        missing.loc[0, field] = np.nan
        with pytest.raises(ValueError, match="完整训练输入缺失"):
            fit_p01_within(missing, CFG)

def test_entry_unknown_stays_fixed_and_exit_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2,
                         FIELDS[0]: .3, FIELDS[1]: .5})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*10, "scale": [1.]*10,
             "feature_clip": 5., "coefficients": [0.]*10, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = P01EntryController(data, records, 2)
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


def test_exact_twenty_real_log_intervals_downside_ratio_and_sample_volatility():
    data = market_rows(length=100)
    field = p01_information_field(data)
    assert field.loc[:19, FIELDS].isna().all().all()
    assert field.loc[20:, FIELDS].notna().all().all()
    for t in [20, 21, 55, 99]:
        logs = data.total_log.iloc[t-19:t+1].to_numpy(float)
        expected_down,expected_all = np.minimum(logs,0.).dot(np.minimum(logs,0.)),logs.dot(logs)
        assert field.loc[t,"twenty_downside_square_sum"] == pytest.approx(expected_down,abs=1e-15)
        assert field.loc[t,"twenty_total_square_sum"] == pytest.approx(expected_all,abs=1e-15)
        assert field.loc[t,FIELDS[0]] == pytest.approx(expected_down/expected_all,abs=1e-12)
        assert field.loc[t,FIELDS[1]] == pytest.approx(np.std(logs,ddof=1),abs=1e-14)
        assert field.loc[t,"full_field_earliest_source_index"] == t-20
        assert field.loc[t,"return_window_start_index"] == t-19
        assert field.loc[t,"latest_source_index"] == t
    changed = data.copy()
    changed.loc[55,"close"] *= 1.015
    changed["previous_close"] = changed.close.shift()
    changed = recalculate_wealth(changed)
    new = p01_information_field(changed)
    pd.testing.assert_frame_equal(field.iloc[:55],new.iloc[:55],check_exact=True)
    assert new.loc[55,FIELDS[0]] != field.loc[55,FIELDS[0]]


def test_unknown_windows_not_compressed_true_zero_one_and_zero_volatility_preserved():
    data = market_rows(length=150)
    for invalid in [np.nan,0.,-1.,np.inf]:
        broken = data.copy()
        broken.loc[75,"wealth"] = invalid
        field = p01_information_field(broken)
        assert field.loc[75:95,FIELDS].isna().all().all()
        assert np.isfinite(field.loc[74,FIELDS].to_numpy(float)).all()
        assert np.isfinite(field.loc[96,FIELDS].to_numpy(float)).all()
    for invalid in [np.nan,np.inf]:
        broken = data.copy()
        broken.loc[90,"total_log"] = invalid
        field = p01_information_field(broken)
        assert field.loc[90:109,FIELDS].isna().all().all()
        assert np.isfinite(field.loc[110,FIELDS].to_numpy(float)).all()
    flat = p01_information_field(market_rows(np.repeat(100.,80)))
    assert flat.loc[20:,FIELDS[0]].isna().all()
    assert flat.loc[20:,FIELDS[1]].eq(0.).all()
    assert flat.loc[20:,"status"].eq("NO_VIEW_ZERO_TOTAL_SQUARED_RETURN").all()
    for sign,expected in [(1.,0.),(-1.,1.)]:
        data = market_rows(100*np.exp(sign*.01*np.arange(80)))
        data.loc[1:,"total_log"] = sign*.01
        field = p01_information_field(data)
        assert field.loc[20:,FIELDS[0]].eq(expected).all()
        assert field.loc[20:,FIELDS[1]].eq(0.).all()
        assert field.loc[20:,"status"].eq("P01_COMPLETE_DOWNSIDE_AND_VOLATILITY_BLOCK_AVAILABLE").all()


def test_future_cash_prices_and_scale_cannot_repaint_and_full_day_clock_is_bound():
    data = market_rows(length=140)
    left = p01_information_field(data.iloc[:80])
    future = data.copy()
    future.loc[90:,"close"] *= 2.
    future.loc[90,"dividend"] = 2.
    future["previous_close"] = future.close.shift()
    future = recalculate_wealth(future)
    validate_wealth_clock(future)
    pd.testing.assert_frame_equal(left,p01_information_field(future).iloc[:80].reset_index(drop=True),check_exact=True)
    scaled = data.copy()
    scaled["wealth"] *= 7.
    np.testing.assert_allclose(p01_information_field(data)[FIELDS],p01_information_field(scaled)[FIELDS],
                               atol=1e-12,rtol=0,equal_nan=True)
    cash = data.copy()
    cash.loc[80,"close"] -= 1.
    cash.loc[80,"dividend"] = 1.
    cash["previous_close"] = cash.close.shift()
    cash = recalculate_wealth(cash)
    validate_wealth_clock(cash)
    np.testing.assert_allclose(p01_information_field(data).loc[80,FIELDS].to_numpy(float),
                               p01_information_field(cash).loc[80,FIELDS].to_numpy(float),atol=1e-12,rtol=0)
    wrong = data.copy()
    wrong.loc[80,"total_log"] += .01
    with pytest.raises(AssertionError): p01_information_field(wrong)
    bad = data.copy()
    bad.loc[80,"wealth"] *= 1.1
    with pytest.raises(AssertionError): validate_wealth_clock(bad)
    symbol = data.copy()
    symbol.loc[0,"symbol"] = "000300.SH"
    with pytest.raises(ValueError): p01_information_field(symbol)


def test_full_day_share_distinct_from_overnight_share_and_not_determined_by_log_std():
    from research.overnight_downside_exit_inputs_v1 import downside_frame,ADDED
    a = np.r_[np.tile([-.02,.01,.01],5),np.zeros(5)]
    b = np.r_[np.tile([-.01,-.01,.02],5),np.zeros(5)]
    first = market_rows(100*np.exp(np.r_[0.,np.cumsum(a)]))
    second = market_rows(100*np.exp(np.r_[0.,np.cumsum(b)]))
    for data in [first,second]:
        data["overnight_log"] = 0.
        data.loc[0,"overnight_log"] = np.nan
        data["intraday_log"] = data.total_log
    left,right = p01_information_field(first),p01_information_field(second)
    assert first.total_log.iloc[-20:].sum() == pytest.approx(second.total_log.iloc[-20:].sum(),abs=1e-14)
    assert first.total_log.iloc[-5:].sum() == pytest.approx(second.total_log.iloc[-5:].sum(),abs=1e-14)
    assert left[FIELDS[1]].iloc[-1] == pytest.approx(right[FIELDS[1]].iloc[-1],abs=1e-14)
    assert downside_frame(first)[ADDED].iloc[-1] == 0.
    assert downside_frame(second)[ADDED].iloc[-1] == 0.
    assert left[FIELDS[0]].iloc[-1] == pytest.approx(2/3,abs=1e-12)
    assert right[FIELDS[0]].iloc[-1] == pytest.approx(1/3,abs=1e-12)
    shifted = first.copy()
    shifted.loc[1:,"overnight_log"] = -.005
    shifted.loc[1:,"intraday_log"] = shifted.total_log.iloc[1:].to_numpy()+.005
    pd.testing.assert_frame_equal(left,p01_information_field(shifted),check_exact=True)
    assert downside_frame(shifted)[ADDED].iloc[-1] > 0.
