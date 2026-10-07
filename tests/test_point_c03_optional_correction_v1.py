"""C03十日源时钟、原分母、现金收益及固定单系数函数必要反例。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_c03_information_intake_v1 import FIELD, build_field, validate_source_clock
from research.point_c03_optional_correction_v1 import align, fit, predict
from research.point_optional_residual_model_v1 import design, system
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction
from research.learned_cycle_exit_v1 import FEATURES


def market(returns=None, length=70):
    if returns is None:
        returns = np.tile([.012, -.008, .003, -.005, 0.], length//5+1)[:length-1]
    returns = np.asarray(returns, float)
    close = 100*np.exp(np.r_[0., np.cumsum(returns)])
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(close)),
        "close": close, "previous_close": np.r_[np.nan, close[:-1]], "dividend": 0.,
        "symbol": "510300.SH", "amount_unit": "CNY", "amount": 1000.+np.arange(len(close))*13.})
    return recalculate(data)


def recalculate(data):
    out = data.copy()
    out["previous_close"] = out.close.shift()
    out["total_simple"] = (out.close+out.dividend)/out.previous_close-1
    out["total_log"] = np.log1p(out.total_simple)
    out["wealth"] = np.r_[1., np.cumprod(1+out.total_simple.to_numpy()[1:])]
    return out


def test_exact_ten_days_include_current_and_zero_day_amount_in_mean():
    data = market()
    field = build_field(data)
    assert field.loc[:9, FIELD].isna().all()
    for t in [10, 24, 45, 69]:
        r = data.total_log.iloc[t-9:t+1].to_numpy()
        a = data.amount.iloc[t-9:t+1].to_numpy()
        pos, neg = r > 0, r < 0
        expected = a.mean()*(r[pos].sum()/a[pos].sum()-np.abs(r[neg]).sum()/a[neg].sum())
        assert field.loc[t, FIELD] == pytest.approx(expected, abs=1e-14)
        assert field.loc[t, "ten_day_start_index"] == t-9
        assert field.loc[t, "latest_source_index"] == t
    changed = data.copy()
    changed.loc[24, "amount"] *= 10.
    assert build_field(changed).loc[24, FIELD] != pytest.approx(field.loc[24, FIELD], abs=1e-9)


def test_missing_or_nonpositive_amount_poison_exact_actual_window_without_compressing():
    for invalid in [0., -1., np.nan, np.inf]:
        data = market()
        data.loc[30, "amount"] = invalid
        field = build_field(data)
        assert len(field) == len(data)
        assert field.loc[30:39, FIELD].isna().all()
        assert np.isfinite(field.loc[29, FIELD]) and np.isfinite(field.loc[40, FIELD])
    data = market()
    data.loc[30, "total_log"] = np.nan
    assert build_field(data).loc[30:39, FIELD].isna().all()


def test_absent_positive_or_negative_group_is_unknown_but_balanced_true_zero_valid():
    for returns, status in [
        (np.repeat(.01, 30), "NO_VIEW_ZERO_NEGATIVE_GROUP_AMOUNT"),
        (np.repeat(-.01, 30), "NO_VIEW_ZERO_POSITIVE_GROUP_AMOUNT"),
        (np.zeros(30), "NO_VIEW_BOTH_GROUP_AMOUNTS_ZERO")]:
        data = market(returns)
        field = build_field(data)
        assert field.loc[10:, FIELD].isna().all()
        assert field.loc[10:, "field_status"].eq(status).all()
    balanced = market(np.tile([.01, -.01], 20))
    balanced["total_log"] = np.r_[np.nan, np.tile([.01, -.01], 20)]
    balanced["amount"] = 1000.
    field = build_field(balanced)
    assert field.loc[10:, FIELD].eq(0.).all()
    assert field.loc[10:, "field_status"].eq("C03_TEN_COMPLETE_DAYS_AND_BOTH_GROUPS_AVAILABLE").all()


def test_future_price_cash_and_amount_cannot_repaint_and_money_units_cancel():
    data = market()
    left = build_field(data.iloc[:40])
    future = data.copy()
    future.loc[40:, "close"] *= 3.
    future.loc[45, "dividend"] = 10.
    future.loc[40:, "amount"] *= 1000.
    future = recalculate(future)
    pd.testing.assert_frame_equal(left, build_field(future).iloc[:40].reset_index(drop=True), check_exact=True)
    scaled = data.copy()
    scaled["amount"] *= 1000.
    np.testing.assert_allclose(build_field(data)[FIELD], build_field(scaled)[FIELD],
                               atol=1e-14, rtol=0, equal_nan=True)
    wrong = data.copy()
    wrong["amount_unit"] = "万元"
    with pytest.raises(ValueError): build_field(wrong)


def test_cash_economic_log_identity_and_unobserved_initial_interval_are_bound():
    data = market()
    validate_source_clock(data)
    cash = data.copy()
    cash.loc[20, "close"] -= 1.
    cash.loc[20, "dividend"] = 1.
    cash = recalculate(cash)
    validate_source_clock(cash)
    assert cash.total_log.iloc[20] == pytest.approx(data.total_log.iloc[20], abs=1e-14)
    bad = cash.copy()
    bad.loc[20, "total_log"] += .1
    with pytest.raises(AssertionError): validate_source_clock(bad)
    bad = data.copy()
    bad.loc[0, "total_log"] = bad.loc[0, "total_simple"] = 0.
    assert build_field(bad).loc[:9, FIELD].isna().all()
    with pytest.raises(ValueError): validate_source_clock(bad)


def test_same_return_path_can_have_different_group_amount_efficiency():
    first = market()
    second = first.copy()
    second.loc[second.total_log.gt(0), "amount"] *= 3.
    pd.testing.assert_series_equal(first.total_log, second.total_log, check_exact=True)
    a, b = build_field(first), build_field(second)
    assert a.loc[39, FIELD] != pytest.approx(b.loc[39, FIELD], abs=1e-8)
    assert a.loc[39, "positive_return_sum10"] == b.loc[39, "positive_return_sum10"]
    assert a.loc[39, "negative_abs_return_sum10"] == b.loc[39, "negative_abs_return_sum10"]



def training():
    records = []
    for cycle in range(1, 4):
        for i in range(6):
            row = {"cycle_id": cycle, "origin_index": 100*cycle+i,
                "target": .02*i+.1*cycle, "sample_weight": 1./6,
                FIELD: (i+cycle-3)/6. if i else np.nan, "auxiliary_available": i != 0}
            row.update({name: .001*i for name in FEATURES})
            records.append(row)
    core = {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES.copy(),
        "mean": [0.]*8, "scale": [1.]*8, "coefficients": [.01]*8,
        "intercept": .02, "feature_clip": 5.}
    return pd.DataFrame(records), core


def samples(daily, indices):
    return pd.DataFrame({"cycle_id": [1]*len(indices), "origin_index": indices,
        "origin": daily.date.iloc[indices].to_numpy()})


def test_align_retains_every_identity_with_exact_ten_day_and_close_clock():
    daily = build_field(market())
    field = align(samples(daily, [9, 10, 24]), daily)
    assert field.origin_index.tolist() == [9, 10, 24]
    assert field.auxiliary_available.tolist() == [False, True, True]
    assert np.isnan(field.loc[0, FIELD])
    assert field.loc[1, "ten_day_start_index"] == 1
    assert field.loc[1, "latest_source_index"] == 10
    assert field.loc[1, "origin_at"].hour == 15 and field.loc[1, "origin_at"].minute == 5
    assert field.loc[2, FIELD] == daily.loc[24, FIELD]


def test_align_rejects_identity_future_source_or_compressed_ten_day_window():
    daily = build_field(market())
    wrong = samples(daily, [24])
    wrong.origin = daily.date.iloc[[25]].to_numpy()
    with pytest.raises(ValueError, match="日期或索引"):
        align(wrong, daily)
    changed = daily.copy()
    changed.loc[24, "latest_source_index"] = 25
    with pytest.raises(ValueError, match="原点之外"):
        align(samples(daily, [24]), changed)
    changed = daily.copy()
    changed.loc[24, "ten_day_start_index"] = 16
    with pytest.raises(ValueError, match="十个完整"):
        align(samples(daily, [24]), changed)
    changed = daily.copy()
    changed.loc[24, "negative_amount_sum10"] = 0.
    with pytest.raises(ValueError, match="分母缺失"):
        align(samples(daily, [24]), changed)


def test_single_coefficient_matches_registered_closed_form_and_core_is_unchanged():
    rows, core = training()
    old_core, old_rows = copy.deepcopy(core), rows.copy(deep=True)
    model = fit(rows, core)
    dx, dy, w = system(rows, core, model)
    expected = float(np.sum(w*dx[:, 0]*dy)/(np.sum(w*dx[:, 0]**2)+1.))
    assert model["coefficients"] == [expected] and model["global_intercept"] == 0.
    assert core == old_core
    pd.testing.assert_frame_equal(rows, old_rows, check_exact=True)
    assert model["training_rows"] == 18 and model["unknown_training_rows_retained"] == 3
    np.testing.assert_allclose(np.average(design(rows, model), axis=0, weights=rows.sample_weight), 0., atol=1e-12)


def test_missing_origin_exactly_returns_original_float_and_unknown_raw_stays_null():
    rows, core = training()
    model = fit(rows, core)
    a, b, status = predict(core, model, [.01]*8, [np.nan])
    assert a == b == within_cycle_prediction(core, [.01]*8) and status == "EXACT_CORE_FALLBACK"
    assert rows.loc[~rows.auxiliary_available, FIELD].isna().all()
    with pytest.raises(ValueError, match="一个可选原值"):
        predict(core, model, [.01]*8, [1., 2.])


def test_all_unknown_training_keeps_zero_coefficient_and_exact_core():
    rows, core = training()
    rows[FIELD] = np.nan
    rows.auxiliary_available = False
    model = fit(rows, core)
    assert model["coefficients"] == [0.] and not model["identified"]
    assert model["unknown_training_rows_retained"] == len(rows)
    a, b, status = predict(core, model, [.01]*8, [0.])
    assert a == b and status == "EXACT_CORE_FALLBACK"


def test_cycle_common_target_shift_does_not_enter_auxiliary_coefficient():
    rows, core = training()
    model = fit(rows, core)
    changed = rows.copy()
    changed.target += changed.cycle_id.map({1: 2., 2: -1., 3: 3.})
    shifted = fit(changed, core)
    np.testing.assert_allclose(model["coefficients"], shifted["coefficients"], atol=1e-12, rtol=0)


def test_unknown_training_rows_are_retained_in_cycle_contrasts():
    rows, core = training()
    before = fit(rows, core)
    altered = rows.copy()
    altered.loc[(altered.cycle_id == 1) & ~altered.auxiliary_available, "target"] += 1.
    after = fit(altered, core)
    assert after["training_rows"] == before["training_rows"]
    assert abs(before["coefficients"][0]-after["coefficients"][0]) > 1e-7



def test_known_zero_remains_active_while_unknown_source_is_exact_fallback():
    rows, core = training()
    model = fit(rows, core)
    a, b, status = predict(core, model, [.01]*8, [0.])
    assert status == "OPTIONAL_CORRECTION_AVAILABLE" and b != a
    a, b, status = predict(core, model, [.01]*8, [np.nan])
    assert status == "EXACT_CORE_FALLBACK" and a == b


def test_negative_efficiency_and_unbounded_finite_value_are_legal_inputs():
    rows, core = training()
    assert rows[FIELD].lt(0).any()
    rows.loc[1, FIELD] = -1.2
    model = fit(rows, core)
    for value in [-1.2, 1.2]:
        a, b, status = predict(core, model, [.01]*8, [value])
        assert np.isfinite(a) and np.isfinite(b) and status == "OPTIONAL_CORRECTION_AVAILABLE"
