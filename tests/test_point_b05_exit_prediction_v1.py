"""B05六负事件、严格金额基准与原条件退出合同必要测试。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_b05_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELD, KIND, B05EntryController, b05_information_field,
    validate_wealth_clock, fit_b05_within, b05_prediction, training_rows)
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction

CFG = {"ridge_alpha": 1., "feature_clip": 5., "recent_cycles": 20,
       "minimum_cycles": 10, "minimum_rows": 100}


def recalculate_cash_returns(data):
    data = data.copy()
    data["previous_close"] = data.close.shift(1)
    data["total_simple"] = (data.close+data.dividend)/data.previous_close-1
    data["total_log"] = np.log1p(data.total_simple)
    data["wealth"] = np.r_[1., np.cumprod(1+data.total_simple.to_numpy()[1:])]
    return data


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
    old, new = fit_within_cycle_exit(rows, CFG), fit_b05_within(rows, CFG)
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    assert abs(new["coefficients"][-1]) < 1e-12
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert b05_prediction(new, np.r_[values, .4]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)

def test_unmatured_future_rows_cannot_change_the_fit_and_missing_rows_are_not_deleted():
    rows = sample_rows()
    past, _ = training_rows(rows, 400, CFG)
    future = rows.copy(); future["cycle_id"] += 100
    future["origin_index"] += 1000; future["exit_index"] += 1000
    future["target"] += 1000; future[FIELD] += 1000
    selected, _ = training_rows(pd.concat([rows, future], ignore_index=True), 400, CFG)
    left, right = fit_b05_within(past, CFG), fit_b05_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    missing = rows.copy(); missing.loc[0, FIELD] = np.nan
    with pytest.raises(ValueError):
        fit_b05_within(missing, CFG)

def test_unknown_entry_version_stays_unknown_and_new_cycle_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2, FIELD: .3})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*9,
             "scale": [1.]*9, "feature_clip": 5., "coefficients": [0.]*9, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = B05EntryController(data, records, 2)
    first = {"cycle_id": 1, "entry_index": 2, "entry_cost_cny": 1000., "mode": 1}
    assert controller(2, first, 1000., 1000.)["continuation_prediction"] is None
    assert controller(4, first, 1000., 1000.)["continuation_prediction"] is None
    second = {"cycle_id": 2, "entry_index": 4, "entry_cost_cny": 1000., "mode": 1}
    assert not controller(4, second, 1000., 1000.)["learned_exit_requested"]
    assert controller(5, second, 1000., 1000.)["learned_exit_requested"]
    third = copy.deepcopy(second); third["entry_index"] = 5
    with pytest.raises(ValueError):
        controller(6, third, 1000., 1000.)
def event_market(length=150, negative_indices=None):
    if negative_indices is None:
        negative_indices = [20, 22, 24, 30, 55, 70, 80, 81, 82, 83, 84, 85, 100, 101, 102, 103, 104, 105]
    returns = np.repeat(.001, length)
    returns[0] = 0.
    for k, t in enumerate(negative_indices):
        returns[t] = -.003-.001*k
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=length),
        "close": 100*np.exp(np.cumsum(returns)), "dividend": 0., "symbol": "510300.SH",
        "amount_unit": "CNY", "amount": 1000.+np.arange(length)*13.})
    return recalculate_cash_returns(data)


def test_six_real_negative_events_have_no_twenty_day_limit_and_current_event_is_included():
    import json
    data = event_market()
    field = b05_information_field(data)
    indexes = [20, 22, 24, 30, 55, 70]
    values = []
    for t in indexes:
        median = data.amount.iloc[t-20:t].median()
        impact = abs(data.total_log.iloc[t])/(data.amount.iloc[t]/median)
        assert field.loc[t, "strict_prior20_amount_median"] == median
        values.append(impact)
    assert json.loads(field.loc[70, "selected_negative_indices"]) == indexes
    assert field.loc[70, "six_event_calendar_span"] == 51
    assert field.loc[70, FIELD] == pytest.approx(np.mean(values[3:])/np.mean(values[:3]), abs=1e-12)
    assert field.loc[71, FIELD] == field.loc[70, FIELD]
    changed = data.copy()
    changed.loc[70, "amount"] *= 2.
    updated = b05_information_field(changed)
    assert updated.loc[70, "strict_prior20_amount_median"] == field.loc[70, "strict_prior20_amount_median"]
    assert updated.loc[70, FIELD] < field.loc[70, FIELD]


def test_unknown_negative_impact_is_retained_not_replaced_by_older_qualified_events():
    import json
    data = event_market()
    data.loc[55, "amount"] = 0.
    field = b05_information_field(data)
    assert json.loads(field.loc[70, "selected_negative_indices"]) == [20, 22, 24, 30, 55, 70]
    assert field.loc[70:84, FIELD].isna().all()
    assert np.isfinite(field.loc[85, FIELD])
    assert json.loads(field.loc[85, "selected_negative_indices"]) == [80, 81, 82, 83, 84, 85]
    gap = event_market()
    gap.loc[90, "total_log"] = np.nan
    missing = b05_information_field(gap)
    assert missing.loc[90:104, FIELD].isna().all()
    assert np.isfinite(missing.loc[105, FIELD])
    assert json.loads(missing.loc[105, "selected_negative_indices"]) == [100, 101, 102, 103, 104, 105]


def test_future_cash_price_amount_cannot_repaint_and_amount_scale_cancels():
    data = event_market()
    prefix = b05_information_field(data.iloc[:100])
    future = data.copy()
    future.loc[110:, "close"] *= 2.
    future.loc[112, "dividend"] = 2.
    future.loc[110:, "amount"] *= 10.
    future = recalculate_cash_returns(future)
    validate_wealth_clock(future)
    pd.testing.assert_frame_equal(prefix, b05_information_field(future).iloc[:100].reset_index(drop=True), check_exact=True)
    scaled = data.copy()
    scaled["amount"] *= 1000.
    np.testing.assert_allclose(b05_information_field(data)[FIELD], b05_information_field(scaled)[FIELD],
                               atol=1e-12, rtol=0, equal_nan=True)
    wrong = data.copy()
    wrong["amount_unit"] = "万元"
    with pytest.raises(ValueError): b05_information_field(wrong)
    bad = data.copy()
    bad.loc[70, "wealth"] *= 1.1
    with pytest.raises(AssertionError): validate_wealth_clock(bad)


def test_no_negative_events_or_unknown_first_six_impacts_remain_unknown():
    import json
    for change in [0., .001]:
        data = event_market(length=80, negative_indices=[])
        data["close"] = 100*np.exp(np.arange(80)*change)
        data = recalculate_cash_returns(data)
        field = b05_information_field(data)
        assert field[FIELD].isna().all()
        assert field.loc[1:, "selected_negative_event_count"].eq(0).all()
    warmup = event_market(length=80, negative_indices=[1, 2, 3, 4, 5, 6, 40, 41, 42, 43, 44, 45])
    field = b05_information_field(warmup)
    assert field.loc[:44, FIELD].isna().all()
    assert np.isfinite(field.loc[45, FIELD])
    assert json.loads(field.loc[45, "selected_negative_indices"]) == [40, 41, 42, 43, 44, 45]
    unused = warmup.copy()
    unused.loc[50, "amount"] = np.nan
    assert b05_information_field(unused).loc[50, FIELD] == field.loc[50, FIELD]
