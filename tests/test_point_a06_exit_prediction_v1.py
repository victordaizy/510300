"""A06固定两维：真实完整窗口、经济单位、未知和固定继续价值模型。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_a06_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELDS, KIND, A06EntryController, a06_information_field,
    validate_wealth_clock, fit_a06_within, a06_prediction, training_rows, training_identity)
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
    old, new = fit_within_cycle_exit(rows, CFG), fit_a06_within(rows, CFG)
    assert len(new["features"]) == 10
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(new["coefficients"][-2:], 0., atol=1e-12, rtol=0)
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert a06_prediction(new, np.r_[values, constants]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)

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
    left, right = fit_a06_within(past, CFG), fit_a06_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    for field in FIELDS:
        missing = rows.copy()
        missing.loc[0, field] = np.nan
        with pytest.raises(ValueError, match="完整训练输入缺失"):
            fit_a06_within(missing, CFG)

def test_entry_unknown_stays_fixed_and_exit_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2,
                         FIELDS[0]: .3, FIELDS[1]: .5})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*10, "scale": [1.]*10,
             "feature_clip": 5., "coefficients": [0.]*10, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = A06EntryController(data, records, 2)
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


def test_exact_inclusive_sixty_mean_twenty_log_std_and_five_interval_speed():
    data = market_rows(length=125)
    field = a06_information_field(data)
    assert field.loc[:58, FIELDS].isna().all().all()
    assert field.loc[59:, FIELDS].notna().all().all()
    for t in [59, 60, 90, 124]:
        prices = data.wealth.iloc[t-59:t+1].to_numpy(float)
        logs = data.total_log.iloc[t-19:t+1].to_numpy(float)
        rv = np.std(logs, ddof=1)
        total = data.total_log.iloc[t-4:t+1].sum()
        assert field.loc[t, "inclusive_sixty_point_wealth_mean"] == pytest.approx(prices.mean(), abs=1e-14)
        assert field.loc[t, "inclusive_log_std20"] == pytest.approx(rv, abs=1e-14)
        assert field.loc[t, FIELDS[0]] == pytest.approx(total/(rv*np.sqrt(5)), abs=1e-12)
        assert field.loc[t, FIELDS[1]] == pytest.approx(np.log(prices[-1]/prices.mean())/(rv*np.sqrt(60)), abs=1e-12)
        assert field.loc[t, "full_field_earliest_source_index"] == t-59
        assert field.loc[t, "volatility_wealth_start_index"] == t-20
        assert field.loc[t, "speed_return_start_index"] == t-4
        assert field.loc[t, "latest_source_index"] == t
    changed = data.copy()
    changed.loc[80, "close"] *= 1.015
    changed["previous_close"] = changed.close.shift()
    changed = recalculate_wealth(changed)
    new = a06_information_field(changed)
    pd.testing.assert_frame_equal(field.iloc[:80], new.iloc[:80], check_exact=True)
    assert new.loc[80, FIELDS[0]] != field.loc[80, FIELDS[0]]
    assert new.loc[80, "inclusive_log_std20"] != field.loc[80, "inclusive_log_std20"]


def test_unknown_real_windows_not_compressed_and_nonpositive_volatility_is_unknown():
    data = market_rows(length=150)
    for invalid in [np.nan, 0., -1., np.inf]:
        broken = data.copy()
        broken.loc[75, "wealth"] = invalid
        field = a06_information_field(broken)
        assert field.loc[75:134, FIELDS].isna().all().all()
        assert np.isfinite(field.loc[74, FIELDS].to_numpy(float)).all()
        assert np.isfinite(field.loc[135, FIELDS].to_numpy(float)).all()
        assert len(field)==len(data)
    for invalid in [np.nan, np.inf]:
        broken = data.copy()
        broken.loc[90, "total_log"] = invalid
        field = a06_information_field(broken)
        assert field.loc[90:109, FIELDS].isna().all().all()
        assert np.isfinite(field.loc[89, FIELDS].to_numpy(float)).all()
        assert np.isfinite(field.loc[110, FIELDS].to_numpy(float)).all()
    flat = a06_information_field(market_rows(np.repeat(100.,80)))
    assert flat.loc[59:, "inclusive_log_std20"].eq(0.).all()
    assert flat.loc[59:, FIELDS].isna().all().all()
    assert flat.loc[59:, "status"].eq("NO_VIEW_NONPOSITIVE_OR_NONFINITE_LOG_VOL20").all()
    prices = np.r_[np.tile([99.,101.],37),np.repeat(100.,6)]
    zero = a06_information_field(market_rows(prices))
    assert zero.loc[79, "inclusive_log_std20"] > 0
    assert zero.loc[79, FIELDS[0]] == pytest.approx(0.,abs=1e-12)
    assert zero.loc[79, FIELDS[1]] == pytest.approx(0.,abs=1e-12)
    assert zero.loc[79, "status"] == "A06_COMPLETE_SPEED_DISTANCE_BLOCK_AVAILABLE"


def test_future_cash_prices_and_scale_cannot_repaint_and_economic_clock_is_bound():
    data = market_rows(length=140)
    left = a06_information_field(data.iloc[:80])
    future = data.copy()
    future.loc[90:, "close"] *= 2.
    future.loc[90, "dividend"] = 2.
    future["previous_close"] = future.close.shift()
    future = recalculate_wealth(future)
    validate_wealth_clock(future)
    pd.testing.assert_frame_equal(left, a06_information_field(future).iloc[:80].reset_index(drop=True), check_exact=True)
    scaled = data.copy()
    scaled["wealth"] *= 7.
    np.testing.assert_allclose(a06_information_field(data)[FIELDS], a06_information_field(scaled)[FIELDS],
                               atol=1e-12, rtol=0, equal_nan=True)
    cash = data.copy()
    cash.loc[80, "close"] -= 1.
    cash.loc[80, "dividend"] = 1.
    cash["previous_close"] = cash.close.shift()
    cash = recalculate_wealth(cash)
    validate_wealth_clock(cash)
    np.testing.assert_allclose(a06_information_field(data).loc[80,FIELDS].to_numpy(float),
                               a06_information_field(cash).loc[80,FIELDS].to_numpy(float),atol=1e-12,rtol=0)
    bad = data.copy()
    bad.loc[80,"wealth"] *= 1.1
    with pytest.raises(AssertionError): validate_wealth_clock(bad)
    wrong_return = data.copy()
    wrong_return.loc[80,"total_log"] += .01
    with pytest.raises(AssertionError): a06_information_field(wrong_return)
    wrong_symbol = data.copy()
    wrong_symbol.loc[0,"symbol"] = "000300.SH"
    with pytest.raises(ValueError): a06_information_field(wrong_symbol)


def test_distance60_can_differ_with_same_recent_momentum_volatility_and_sma120():
    base = 100+np.cumsum(np.random.default_rng(931).normal(.15,.35,140))
    a,b = base.copy(),base.copy()
    a[85] += 4.
    b[50] += 4.
    first,second = market_rows(a),market_rows(b)
    for window in [5,20]:
        assert first.total_log.iloc[-window:].sum() == pytest.approx(second.total_log.iloc[-window:].sum(),abs=1e-14)
    assert first.total_simple.iloc[-20:].std(ddof=1) == pytest.approx(second.total_simple.iloc[-20:].std(ddof=1),abs=1e-14)
    assert first.wealth.iloc[-1]/first.wealth.iloc[-120:].mean()-1 == pytest.approx(
        second.wealth.iloc[-1]/second.wealth.iloc[-120:].mean()-1,abs=1e-14)
    left,right = a06_information_field(first),a06_information_field(second)
    assert left[FIELDS[0]].iloc[-1] == pytest.approx(right[FIELDS[0]].iloc[-1],abs=1e-12)
    assert left[FIELDS[1]].iloc[-1] != pytest.approx(right[FIELDS[1]].iloc[-1],abs=1e-12)
