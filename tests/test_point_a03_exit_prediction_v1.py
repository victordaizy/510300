"""A03新高频率两列的新条件用途：可知窗口、经济单位、完整成员和固定入场版本。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_a03_exit_prediction_v1 import (
    BASE_FEATURES, FEATURES, FIELDS, KIND, A03EntryController, a03_information_field,
    validate_wealth_clock, fit_a03_within, a03_prediction, training_rows, training_identity)
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
    old, new = fit_within_cycle_exit(rows, CFG), fit_a03_within(rows, CFG)
    assert len(new["features"]) == 10
    np.testing.assert_allclose(new["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(new["coefficients"][-2:], 0., atol=1e-12, rtol=0)
    for values in rows[BASE_FEATURES].to_numpy(float)[::17]:
        assert a03_prediction(new, np.r_[values, constants]) == pytest.approx(within_cycle_prediction(old, values), abs=1e-12)

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
    left, right = fit_a03_within(past, CFG), fit_a03_within(selected, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(left[key], right[key], atol=1e-12, rtol=0)
    for field in FIELDS:
        missing = rows.copy()
        missing.loc[0, field] = np.nan
        with pytest.raises(ValueError, match="完整训练输入缺失"):
            fit_a03_within(missing, CFG)

def test_entry_unknown_stays_fixed_and_exit_requires_two_negative_predictions():
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=7), "mom5": 0.,
                         "mom20": 0., "sma120": 0., "vol20": .2,
                         FIELDS[0]: .3, FIELDS[1]: .5})
    model = {"kind": KIND, "features": FEATURES.copy(), "mean": [0.]*10, "scale": [1.]*10,
             "feature_clip": 5., "coefficients": [0.]*10, "intercept": -.1}
    records = [{"fit_index": 1, "status": "NO_VIEW_NO_MATURE_MODEL", "model": None},
               {"fit_index": 4, "status": "FIT_COMPLETE", "latest_exit_index": 3, "model": model}]
    controller = A03EntryController(data, records, 2)
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


def test_strict_prior20_and_inclusive_twenty_events_need_forty_complete_points():
    data = market_rows(np.arange(100., 165.))
    field = a03_information_field(data)
    assert field.loc[:19, "known_new_high_event"].isna().all()
    assert field.loc[20:, "known_new_high_event"].eq(1.).all()
    assert field.loc[:38, FIELDS].isna().all().all()
    assert field.loc[39:, FIELDS[0]].eq(1.).all()
    assert field.loc[39:, FIELDS[1]].eq(0.).all()
    for t in [39, 44, 60]:
        assert field.loc[t, "strict_prior20_wealth_high"] == data.wealth.iloc[t-20:t].max()
        assert field.loc[t, "full_field_earliest_source_index"] == t-39
        assert field.loc[t, "latest_source_index"] == t
    plateau = market_rows(np.repeat(100., 65))
    same = a03_information_field(plateau)
    assert same.loc[20:, "known_new_high_event"].eq(0.).all()
    assert same.loc[39:, FIELDS].eq(0.).all().all()
    assert same.loc[39:, "status"].eq("A03_FORTY_COMPLETE_WEALTH_POINTS_AVAILABLE").all()
    changed = plateau.copy()
    changed.loc[39, "wealth"] = 1.1
    current = a03_information_field(changed)
    assert current.loc[39, FIELDS[0]] == .05
    assert current.loc[39, FIELDS[1]] == .2


def test_five_and_previous_fifteen_event_windows_are_disjoint_and_exact():
    data = market_rows(length=95)
    field = a03_information_field(data)
    w = data.wealth.to_numpy(float)
    events = np.full(len(data), np.nan)
    for t in range(20, len(data)):
        events[t] = float(w[t] > w[t-20:t].max())
    for t in [39, 55, 80, 94]:
        assert field.loc[t, FIELDS[0]] == pytest.approx(events[t-19:t+1].mean(), abs=1e-14)
        assert field.loc[t, "recent5_new_high_frequency"] == events[t-4:t+1].mean()
        assert field.loc[t, "earlier15_new_high_frequency"] == pytest.approx(events[t-19:t-4].mean(), abs=1e-14)
        assert field.loc[t, FIELDS[1]] == pytest.approx(events[t-4:t+1].mean()-events[t-19:t-4].mean(), abs=1e-14)
    for invalid in [np.nan, 0., np.inf]:
        broken = data.copy()
        broken.loc[50, "wealth"] = invalid
        unknown = a03_information_field(broken)
        assert unknown.loc[50:89, FIELDS].isna().all().all()
        assert np.isfinite(unknown.loc[49, FIELDS].to_numpy(float)).all()
        assert np.isfinite(unknown.loc[90, FIELDS].to_numpy(float)).all()
        assert len(unknown) == len(data)


def test_future_cash_and_prices_cannot_repaint_and_wealth_coordinate_is_bound():
    data = market_rows(length=90)
    left = a03_information_field(data.iloc[:50])
    future = data.copy()
    future.loc[60:, "close"] *= 2.
    future.loc[60, "dividend"] = 2.
    future["previous_close"] = future.close.shift()
    future = recalculate_wealth(future)
    validate_wealth_clock(future)
    pd.testing.assert_frame_equal(left, a03_information_field(future).iloc[:50].reset_index(drop=True), check_exact=True)
    scaled = data.copy()
    scaled["wealth"] *= 7.
    np.testing.assert_allclose(a03_information_field(data)[FIELDS], a03_information_field(scaled)[FIELDS],
                               atol=1e-14, rtol=0, equal_nan=True)
    cash = data.copy()
    cash.loc[45, "close"] -= 1.
    cash.loc[45, "dividend"] = 1.
    cash["previous_close"] = cash.close.shift()
    cash = recalculate_wealth(cash)
    validate_wealth_clock(cash)
    assert cash.wealth.iloc[45] == pytest.approx(data.wealth.iloc[45], abs=1e-14)
    bad = data.copy()
    bad.loc[45, "wealth"] *= 1.1
    with pytest.raises(AssertionError): validate_wealth_clock(bad)
    symbol = data.copy()
    symbol.loc[0, "symbol"] = "000300.SH"
    with pytest.raises(ValueError): a03_information_field(symbol)


def test_same_five_twenty_momentum_and_twenty_volatility_can_have_different_new_high_counts():
    common = np.repeat(.001, 15)
    a = np.r_[common, [-.03, .01, .02, -.01, .015]]
    b = np.r_[common, [.015, -.01, .02, .01, -.03]]
    first = market_rows(100*np.exp(np.r_[np.zeros(40), np.cumsum(a)]))
    second = market_rows(100*np.exp(np.r_[np.zeros(40), np.cumsum(b)]))
    for window in [5, 20]:
        assert first.total_log.iloc[-window:].sum() == pytest.approx(second.total_log.iloc[-window:].sum(), abs=1e-14)
    assert first.total_log.iloc[-20:].std(ddof=1) == pytest.approx(second.total_log.iloc[-20:].std(ddof=1), abs=1e-14)
    assert first.wealth.iloc[-1] == pytest.approx(second.wealth.iloc[-1], abs=1e-14)
    left, right = a03_information_field(first), a03_information_field(second)
    assert left[FIELDS[0]].iloc[-1] != right[FIELDS[0]].iloc[-1]
    assert left[FIELDS[1]].iloc[-1] != right[FIELDS[1]].iloc[-1]
