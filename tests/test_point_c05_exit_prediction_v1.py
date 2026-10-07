"""C05固定事前OLS、完整窗口与原条件退出合同回归测试。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_c05_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELD, KIND, C05EntryController, c05_information_field,
    validate_wealth_clock, fit_c05_within, c05_prediction, training_rows)
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


def market(length=600):
    rng = np.random.default_rng(51030086)
    changes = rng.normal(0.0002, 0.01, length)
    close = 100*np.exp(np.cumsum(changes))
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=length),
        "close": close, "dividend": 0., "symbol": "510300.SH", "amount_unit": "CNY"})
    data = recalculate_cash_returns(data)
    r = data.total_log.to_numpy()
    rv = data.total_log.rolling(20, min_periods=20).std(ddof=1).to_numpy()
    weekday = data.date.dt.dayofweek.to_numpy()
    explained = (19+5*np.nan_to_num(np.abs(r),nan=0.)+40*np.nan_to_num(rv,nan=0.)
                 +0.15*(weekday==1)-0.05*(weekday==2)+0.1*(weekday==3)-0.08*(weekday==4))
    data["amount"] = np.exp(explained)
    return data


def test_current_log_amount_is_not_used_in_prior252_OLS_and_warmup_is_complete():
    data = market(320)
    data.loc[300,"amount"] *= np.exp(3.)
    field = c05_information_field(data)
    assert field[FIELD].iloc[:272].isna().all()
    assert field[FIELD].iloc[272:300].notna().all()
    assert field.loc[300,"training_start_index"] == 48
    assert field.loc[300,"latest_OLS_training_index"] == 299
    assert field.loc[300,"prior_OLS_rank"] == 7
    assert field.loc[300,FIELD] == pytest.approx(3.,abs=1e-11)
    assert field.loc[300,"OLS_abs_log_return_coefficient"] == pytest.approx(5.,abs=1e-9)
    assert field.loc[300,"OLS_std20_coefficient"] == pytest.approx(40.,abs=1e-9)


def test_zero_amount_invalidates_all_252_actual_prior_days_without_compressing_rows():
    data = market()
    data.loc[280,"amount"] = 0.
    field = c05_information_field(data)
    assert len(field) == len(data)
    assert np.isnan(field.loc[280,FIELD])
    assert field.loc[280,"status"] == "NO_VIEW_CURRENT_LOG_AMOUNT_OR_COVARIATES"
    assert field[FIELD].iloc[281:533].isna().all()
    assert field.loc[532,"training_start_index"] == 280
    assert np.isfinite(field.loc[533,FIELD])
    assert field.loc[533,"training_start_index"] == 281


def test_future_prices_cash_and_amount_cannot_repaint_prior_fields():
    data = market(380)
    prefix = c05_information_field(data.iloc[:330].copy())
    data.loc[350:,"close"] *= 2
    data.loc[350,"dividend"] = 12.
    data.loc[350:,"amount"] *= 1000.
    data = recalculate_cash_returns(data)
    longer = c05_information_field(data)
    pd.testing.assert_frame_equal(prefix,longer.iloc[:330].reset_index(drop=True),check_exact=True)


def test_log_amount_unit_shift_cancels_true_residual_zero_and_rank_failure_stays_unknown():
    data = market(320)
    field = c05_information_field(data)
    assert np.isfinite(field.loc[300,FIELD]) and abs(field.loc[300,FIELD]) < 1e-11
    assert field.loc[300,"status"] == "C05_COMPLETE_PRIOR252_OLS_AVAILABLE"
    scaled = data.copy(); scaled["amount"] *= 1000.
    np.testing.assert_allclose(field[FIELD],c05_information_field(scaled)[FIELD],atol=2e-11,rtol=0,equal_nan=True)
    wrong = data.copy(); wrong["amount_unit"] = "万元"
    with pytest.raises(ValueError):
        c05_information_field(wrong)
    flat = data.copy(); flat["close"] = 100.
    flat = recalculate_cash_returns(flat)
    unknown = c05_information_field(flat)
    assert unknown[FIELD].isna().all()
    assert unknown.status.iloc[272:].eq("NO_VIEW_RANK_DEFICIENT_PRIOR_OLS").all()


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
    old, new = fit_within_cycle_exit(rows, CFG), fit_c05_within(rows, CFG)
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    assert abs(new["coefficients"][-1]) < 1e-12
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert c05_prediction(new, np.r_[values, .4]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)

def test_unmatured_future_rows_cannot_change_the_fit_and_missing_rows_are_not_deleted():
    rows = sample_rows()
    past, _ = training_rows(rows, 400, CFG)
    future = rows.copy(); future["cycle_id"] += 100
    future["origin_index"] += 1000; future["exit_index"] += 1000
    future["target"] += 1000; future[FIELD] += 1000
    selected, _ = training_rows(pd.concat([rows, future], ignore_index=True), 400, CFG)
    left, right = fit_c05_within(past, CFG), fit_c05_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    missing = rows.copy(); missing.loc[0, FIELD] = np.nan
    with pytest.raises(ValueError):
        fit_c05_within(missing, CFG)

def test_unknown_entry_version_stays_unknown_and_new_cycle_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2, FIELD: .3})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*9,
             "scale": [1.]*9, "feature_clip": 5., "coefficients": [0.]*9, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = C05EntryController(data, records, 2)
    first = {"cycle_id": 1, "entry_index": 2, "entry_cost_cny": 1000., "mode": 1}
    assert controller(2, first, 1000., 1000.)["continuation_prediction"] is None
    assert controller(4, first, 1000., 1000.)["continuation_prediction"] is None
    second = {"cycle_id": 2, "entry_index": 4, "entry_cost_cny": 1000., "mode": 1}
    assert not controller(4, second, 1000., 1000.)["learned_exit_requested"]
    assert controller(5, second, 1000., 1000.)["learned_exit_requested"]
    third = copy.deepcopy(second); third["entry_index"] = 5
    with pytest.raises(ValueError):
        controller(6, third, 1000., 1000.)
