"""B06三日速度差/前高首次确认：原时钟、严格零边界与固定模型。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_b06_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELDS, KIND, B06EntryController, b06_information_field,
    validate_wealth_clock, fit_b06_within, b06_prediction, training_rows, training_identity)
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
    old, new = fit_within_cycle_exit(rows, CFG), fit_b06_within(rows, CFG)
    assert len(new["features"]) == 10
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(new["coefficients"][-2:], 0., atol=1e-12, rtol=0)
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert b06_prediction(new, np.r_[values, constants]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)

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
    left, right = fit_b06_within(past, CFG), fit_b06_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    for field in FIELDS:
        missing = rows.copy()
        missing.loc[0, field] = np.nan
        with pytest.raises(ValueError, match="完整训练输入缺失"):
            fit_b06_within(missing, CFG)

def test_entry_unknown_stays_fixed_and_exit_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2,
                         FIELDS[0]: .3, FIELDS[1]: .5})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*10, "scale": [1.]*10,
             "feature_clip": 5., "coefficients": [0.]*10, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = B06EntryController(data, records, 2)
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


def test_exact_six_log_intervals_and_first_confirmation_requires_previous_source():
    data=market_rows()
    validate_wealth_clock(data)
    field=b06_information_field(data)
    assert field.loc[:6,FIELDS].isna().all().all()
    for t in [7,20,55,99]:
        r=data.total_log
        speed=r.iloc[t-2:t+1].mean()-r.iloc[t-5:t-2].mean()
        previous_speed=r.iloc[t-3:t].mean()-r.iloc[t-6:t-3].mean()
        h=data.wealth*(data.high+data.dividend)/(data.close+data.dividend)
        current=bool(speed>0 and data.wealth.iloc[t]>h.iloc[t-1])
        previous=bool(previous_speed>0 and data.wealth.iloc[t-1]>h.iloc[t-2])
        assert field.loc[t,FIELDS[0]]==pytest.approx(speed,abs=1e-12)
        assert field.loc[t,FIELDS[1]]==float(current and not previous)
        assert field.loc[t,"full_field_earliest_source_index"]==t-7
        assert field.loc[t,"current_six_return_start_index"]==t-5
        assert field.loc[t,"latest_source_index"]==t


def test_missing_real_intervals_or_high_not_compressed_and_true_zero_retained():
    for column,span in [("wealth",8),("total_log",7)]:
        for invalid in ([np.nan,np.inf,0.,-1.] if column=="wealth" else [np.nan,np.inf]):
            data=market_rows(length=120)
            data.loc[60,column]=invalid
            field=b06_information_field(data)
            assert field.loc[60:60+span-1,FIELDS].isna().all().all()
            assert field.loc[59,FIELDS].notna().all()
            assert field.loc[60+span,FIELDS].notna().all()
    data=market_rows(length=120)
    data.loc[60,"high"]=np.nan
    field=b06_information_field(data)
    assert field.loc[60,FIELDS].notna().all()
    assert field.loc[61:62,FIELDS].isna().all().all()
    assert field.loc[63,FIELDS].notna().all()
    flat=b06_information_field(market_rows(np.repeat(10.,40)))
    assert flat.loc[7:,FIELDS].eq(0).all().all()
    exponential=b06_information_field(market_rows(2.**np.arange(20)))
    assert exponential.loc[7:,FIELDS].eq(0).all().all()


def test_future_cash_quotes_and_wealth_scale_do_not_repaint():
    data=market_rows(length=120)
    field=b06_information_field(data)
    future=data.copy()
    future.loc[90:,"close"]*=2.
    future.loc[90:,"high"]*=2.
    future.loc[90,"dividend"]=2.
    future=recalculate_wealth(future)
    future.loc[90:,"high"]=np.maximum(future.loc[90:,"high"],future.loc[90:,"open"])
    validate_wealth_clock(future)
    pd.testing.assert_frame_equal(field.iloc[:90],b06_information_field(future).iloc[:90],check_exact=True)
    scaled=data.copy()
    scaled["wealth"]*=7.
    np.testing.assert_allclose(field[FIELDS],b06_information_field(scaled)[FIELDS],atol=1e-12,rtol=0,equal_nan=True)
    dirty=data.copy()
    dirty.loc[60,"wealth"]*=1.1
    with pytest.raises(AssertionError):validate_wealth_clock(dirty)
    dirty=data.copy()
    dirty.loc[0,"total_log"]=0.
    with pytest.raises(ValueError,match="首个市场点"):validate_wealth_clock(dirty)


def test_cash_high_mapping_exact_equality_and_distinct_high_information():
    data=market_rows([10.,9.9,9.8,9.7,9.6,9.5,9.6,9.7,9.8,9.9,10.,10.1])
    data.loc[7,"high"]=9.85
    field=b06_information_field(data)
    assert field.loc[9,FIELDS[1]]==1.
    changed=data.copy()
    changed.loc[8,"high"]=10.9
    assert b06_information_field(changed).loc[9,FIELDS[1]]==0.
    pd.testing.assert_series_equal(data.total_log,changed.total_log)
    assert field.loc[9,FIELDS[0]]==b06_information_field(changed).loc[9,FIELDS[0]]
    cash=market_rows([10.,9.9,9.8,9.7,9.6,9.5,9.6,9.7,10.,11.,11.1,11.2])
    cash.loc[8,"dividend"]=.1
    cash.loc[8,"high"]=11.01
    cash=recalculate_wealth(cash)
    validate_wealth_clock(cash)
    equal=b06_information_field(cash)
    assert equal.loc[9,"strict_above_cash_mapped_previous_high"]==0.
    cash.loc[9,"dividend"]=.001
    larger=b06_information_field(recalculate_wealth(cash))
    assert larger.loc[9,"strict_above_cash_mapped_previous_high"]==1.
    off_grid=data.copy()
    off_grid.loc[8,"high"]+=.0001
    with pytest.raises(ValueError,match="0.001"):b06_information_field(off_grid)
    wrong=data.copy()
    wrong.loc[0,"symbol"]="OTHER"
    with pytest.raises(ValueError,match="标的"):b06_information_field(wrong)
