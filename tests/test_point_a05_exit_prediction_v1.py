"""A05事前事件锚定量价：原时钟、首次事件、缺口与固定模型。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_a05_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELDS, KIND, A05EntryController, a05_information_field,
    validate_wealth_clock, fit_a05_within, a05_prediction, training_rows, training_identity,
    a05_price_events, event_anchored_field)
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction
CFG = {"ridge_alpha": 1., "feature_clip": 5., "recent_cycles": 20, "minimum_cycles": 10, "minimum_rows": 100}


def market_rows(prices=None, length=90):
    if prices is None:
        prices = 100+np.cumsum(np.random.default_rng(821).normal(.05, .7, length))
    prices = np.asarray(prices, float)
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(prices)), "close": prices,
        "dividend": 0., "symbol": "510300.SH", "volume": 1000.+np.arange(len(prices))*17.,
        "volume_unit": "share", "feature_valid": True})
    return recalculate_wealth(data)


def recalculate_wealth(data):
    out = data.copy()
    out["previous_close"] = out.close.shift()
    out["open"] = out.previous_close*np.exp(-.003)
    out.loc[0, "open"] = out.close.iloc[0]
    out["total_simple"] = (out.close+out.dividend)/out.previous_close-1
    out["total_log"] = np.log1p(out.total_simple)
    out["overnight_log"] = np.log((out.open+out.dividend)/out.previous_close)
    out["intraday_log"] = np.log((out.close+out.dividend)/(out.open+out.dividend))
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
    old, new = fit_within_cycle_exit(rows, CFG), fit_a05_within(rows, CFG)
    assert len(new["features"]) == 10
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(new["coefficients"][-2:], 0., atol=1e-12, rtol=0)
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert a05_prediction(new, np.r_[values, constants]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)

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
    left, right = fit_a05_within(past, CFG), fit_a05_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    for field in FIELDS:
        missing = rows.copy()
        missing.loc[0, field] = np.nan
        with pytest.raises(ValueError, match="完整训练输入缺失"):
            fit_a05_within(missing, CFG)

def test_entry_unknown_stays_fixed_and_exit_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2,
                         FIELDS[0]: .3, FIELDS[1]: .5})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*10, "scale": [1.]*10,
             "feature_clip": 5., "coefficients": [0.]*10, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = A05EntryController(data, records, 2)
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


def test_exact_cumulative_anchor_average_reset_and_first_positive_pulse():
    data = market_rows([10.,9.,8.,8.,9.,7.,9.,11.,9.,9.,10.,12.])
    data["volume"] = np.arange(len(data))+1.
    events = np.zeros(len(data), dtype=bool)
    events[[2,8]] = True
    field = event_anchored_field(data, events)
    assert field.loc[:1,FIELDS].isna().all().all()
    for t in range(2,len(data)):
        e = 2 if t<8 else 8
        p,q = data.wealth.iloc[e:t+1].to_numpy(),data.volume.iloc[e:t+1].to_numpy()
        assert field.loc[t,FIELDS[0]] == pytest.approx(p[-1]/np.average(p,weights=q)-1,abs=1e-12)
        assert field.loc[t,"anchor_index"] == e
        assert field.loc[t,"full_field_earliest_source_index"] == e
        assert field.loc[t,"latest_source_index"] == t
    assert field.loc[[2,3,8,9],FIELDS[0]].eq(0).all()
    assert field.loc[[4,10],FIELDS[1]].eq(1).all()
    assert field.loc[[2,3,5,6,7,8,9,11],FIELDS[1]].eq(0).all()
    assert field.loc[3,"first_positive_index_known_so_far"] == -1
    assert field.loc[7,"first_positive_index_known_so_far"] == 4
    assert field.loc[8,"first_positive_index_known_so_far"] == -1


def test_missing_any_anchored_price_volume_poison_until_new_known_anchor():
    base = market_rows(length=40)
    events = np.zeros(len(base),dtype=bool)
    events[[2,25]] = True
    for name in ["wealth","volume"]:
        for invalid in [np.nan, np.inf, 0., -1.]:
            data = base.copy()
            data.loc[8,name] = invalid
            field = event_anchored_field(data,events)
            assert field.loc[8:24,FIELDS].isna().all().all()
            assert field.loc[7,FIELDS].notna().all()
            assert field.loc[25:,FIELDS].notna().all().all()
    flat = event_anchored_field(market_rows(np.repeat(100.,40)),events)
    assert flat.loc[2:,FIELDS].eq(0).all().all()
    unknown = event_anchored_field(base,np.zeros(len(base),dtype=bool))
    assert unknown[FIELDS].isna().all().all()
    bad = base.copy()
    bad["volume_unit"] = "lot"
    with pytest.raises(ValueError, match="份额单位"):
        event_anchored_field(bad,events)


def test_future_cash_price_volume_do_not_repaint_and_volume_is_distinct_information():
    data = market_rows(length=100)
    events = np.zeros(len(data),dtype=bool)
    events[[10,70]] = True
    original = event_anchored_field(data,events)
    future = data.copy()
    future.loc[60:,"close"] *= 2.
    future.loc[60,"dividend"] = 2.
    future.loc[60:,"volume"] *= 3.
    future = recalculate_wealth(future)
    validate_wealth_clock(future)
    pd.testing.assert_frame_equal(original.iloc[:60],event_anchored_field(future,events).iloc[:60],check_exact=True)
    scaled = data.copy()
    scaled["wealth"] *= 7.
    np.testing.assert_allclose(original[FIELDS],event_anchored_field(scaled,events)[FIELDS],atol=1e-12,rtol=0,equal_nan=True)
    scaled = data.copy()
    scaled["volume"] *= 100.
    np.testing.assert_allclose(original[FIELDS],event_anchored_field(scaled,events)[FIELDS],atol=1e-12,rtol=0,equal_nan=True)
    changed = data.copy()
    changed.loc[15,"volume"] *= 30.
    pd.testing.assert_series_equal(data.total_log,changed.total_log)
    assert original.loc[20,FIELDS[0]] != event_anchored_field(changed,events).loc[20,FIELDS[0]]
    wrong = data.copy()
    wrong.loc[30,"wealth"] *= 1.1
    with pytest.raises(AssertionError): validate_wealth_clock(wrong)
    wrong = data.copy()
    wrong.loc[30,"overnight_log"] += .01
    with pytest.raises(AssertionError): validate_wealth_clock(wrong)


def test_production_anchor_is_original_confirmed_d60_activation_with_no_future_fill():
    data = market_rows(length=120)
    events, entry, factor = a05_price_events(data)
    assert events.sum() >= 1
    assert not events[:61].any()
    assert np.all(events == (entry & ~np.r_[False,entry[:-1]]))
    field = a05_information_field(data)
    first = int(np.flatnonzero(events)[0])
    assert field.loc[:first-1,FIELDS].isna().all().all()
    assert field.loc[first,FIELDS].eq(0).all()
    future = data.copy()
    future.loc[90:,"open"] *= 1.1
    future["overnight_log"] = np.log((future.open+future.dividend)/future.previous_close)
    future["intraday_log"] = np.log((future.close+future.dividend)/(future.open+future.dividend))
    pd.testing.assert_frame_equal(field.iloc[:90],a05_information_field(future).iloc[:90],check_exact=True)
    flat = market_rows(np.repeat(100.,120))
    flat["open"] = flat.close
    flat["overnight_log"] = np.log(flat.open/flat.previous_close)
    flat["intraday_log"] = 0.
    assert a05_information_field(flat)[FIELDS].isna().all().all()
    dirty = data.copy()
    dirty.loc[0,"total_log"] = 0.
    with pytest.raises(ValueError, match="首个市场点"):
        validate_wealth_clock(dirty)
