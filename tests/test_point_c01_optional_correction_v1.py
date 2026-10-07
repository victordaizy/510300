"""C01二十日源时钟、组数边界、真实比1与固定单系数必要反例。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_c01_information_intake_v1 import FIELD, build_field, validate_source_clock
from research.point_c01_optional_correction_v1 import align, fit, predict
from research.point_optional_residual_model_v1 import design, system
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction
from research.learned_cycle_exit_v1 import FEATURES


def market(returns=None, length=90):
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


def test_twenty_actual_intervals_and_signed_medians_not_group_means():
    data = market()
    field = build_field(data)
    assert field.loc[:19, FIELD].isna().all()
    for t in [20, 39, 65, 89]:
        r = data.total_log.iloc[t-19:t+1].to_numpy()
        a = data.amount.iloc[t-19:t+1].to_numpy()
        pos, neg = r > 0, r < 0
        expected = np.median(np.abs(r[neg])/a[neg])/np.median(r[pos]/a[pos])
        assert field.loc[t, FIELD] == pytest.approx(expected, abs=1e-13)
        assert field.loc[t, "twenty_day_start_index"] == t-19
        assert field.loc[t, "latest_source_index"] == t
    changed = market(np.tile([.01, -.01], 30))
    changed["total_log"] = np.r_[np.nan, np.tile([.01, -.01], 30)]
    changed["amount"] = 1000.
    changed.loc[39, "amount"] = 1.
    assert build_field(changed).loc[40, FIELD] == 1.


def test_exact_five_per_sign_is_available_and_four_is_unknown_zero_days_excluded():
    for n_positive, n_negative, status in [
        (5, 5, "C01_TWENTY_COMPLETE_DAYS_AND_FIVE_PER_SIGN_AVAILABLE"),
        (4, 6, "NO_VIEW_POSITIVE_GROUP_FEWER_THAN_FIVE"),
        (6, 4, "NO_VIEW_NEGATIVE_GROUP_FEWER_THAN_FIVE"),
        (4, 4, "NO_VIEW_BOTH_SIGN_GROUPS_FEWER_THAN_FIVE")]:
        r = np.r_[np.repeat(.02, n_positive), np.repeat(-.01, n_negative),
                  np.zeros(20-n_positive-n_negative)]
        data = market(r)
        data["total_log"] = np.r_[np.nan, r]
        data["amount"] = 1000.
        field = build_field(data)
        assert field.loc[20, "field_status"] == status
        assert field.loc[20, "positive_day_count20"] == n_positive
        assert field.loc[20, "negative_day_count20"] == n_negative
        if n_positive >= 5 and n_negative >= 5:
            assert field.loc[20, FIELD] == .5
        else:
            assert pd.isna(field.loc[20, FIELD])


def test_source_hole_poison_full_twenty_days_even_on_neutral_return_day():
    for invalid in [0., -1., np.nan, np.inf]:
        data = market()
        assert data.total_log.iloc[30] == 0.
        data.loc[30, "amount"] = invalid
        field = build_field(data)
        assert len(field) == len(data)
        assert field.loc[30:49, FIELD].isna().all()
        assert np.isfinite(field.loc[29, FIELD]) and np.isfinite(field.loc[50, FIELD])
    for invalid in [np.nan, np.inf, -np.inf]:
        data = market()
        data.loc[30, "total_log"] = invalid
        assert build_field(data).loc[30:49, FIELD].isna().all()


def test_signless_windows_unknown_and_balanced_ratio_one_is_valid():
    for r, status in [
        (np.repeat(.01, 35), "NO_VIEW_NEGATIVE_GROUP_FEWER_THAN_FIVE"),
        (np.repeat(-.01, 35), "NO_VIEW_POSITIVE_GROUP_FEWER_THAN_FIVE"),
        (np.zeros(35), "NO_VIEW_BOTH_SIGN_GROUPS_FEWER_THAN_FIVE")]:
        field = build_field(market(r))
        assert field.loc[20:, FIELD].isna().all()
        assert field.loc[20:, "field_status"].eq(status).all()
    data = market(np.tile([.01, -.01], 25))
    data["total_log"] = np.r_[np.nan, np.tile([.01, -.01], 25)]
    data["amount"] = 1000.
    assert build_field(data).loc[20:, FIELD].eq(1.).all()


def test_future_data_cannot_repaint_money_scale_cancels_and_group_amount_matters():
    data = market()
    left = build_field(data.iloc[:50])
    future = data.copy()
    future.loc[50:, "close"] *= 3.
    future.loc[55, "dividend"] = 10.
    future.loc[50:, "amount"] *= 1000.
    pd.testing.assert_frame_equal(left, build_field(recalculate(future)).iloc[:50].reset_index(drop=True),
                                  check_exact=True)
    scaled = data.copy()
    scaled["amount"] *= 1000.
    np.testing.assert_allclose(build_field(data)[FIELD], build_field(scaled)[FIELD],
                               atol=1e-13, rtol=0, equal_nan=True)
    asymmetric = data.copy()
    asymmetric.loc[asymmetric.total_log.gt(0), "amount"] *= 3.
    pd.testing.assert_series_equal(data.total_log, asymmetric.total_log, check_exact=True)
    assert build_field(asymmetric).loc[49, FIELD] == pytest.approx(3*build_field(data).loc[49, FIELD])
    wrong = data.copy()
    wrong["amount_unit"] = "万元"
    with pytest.raises(ValueError): build_field(wrong)


def test_cash_economic_identity_initial_interval_and_invalid_wealth_not_repaired():
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
    assert build_field(bad).loc[:19, FIELD].isna().all()
    with pytest.raises(ValueError): validate_source_clock(bad)
    for invalid in [0., -1., np.nan, np.inf]:
        bad = data.copy()
        bad.loc[20, "wealth"] = invalid
        with pytest.raises(ValueError): validate_source_clock(bad)



def training():
    records = []
    for cycle in range(1, 4):
        for i in range(6):
            row = {"cycle_id": cycle, "origin_index": 100*cycle+i,
                "target": .02*i+.1*cycle, "sample_weight": 1./6,
                FIELD: (i+cycle)/6. if i else np.nan, "auxiliary_available": i != 0}
            row.update({name: .001*i for name in FEATURES})
            records.append(row)
    core = {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES.copy(),
        "mean": [0.]*8, "scale": [1.]*8, "coefficients": [.01]*8,
        "intercept": .02, "feature_clip": 5.}
    return pd.DataFrame(records), core


def samples(daily, indices):
    return pd.DataFrame({"cycle_id": [1]*len(indices), "origin_index": indices,
        "origin": daily.date.iloc[indices].to_numpy()})


def test_align_keeps_original_identity_complete_twenty_day_and_close_clock():
    daily = build_field(market())
    field = align(samples(daily, [19, 20, 39]), daily)
    assert field.origin_index.tolist() == [19, 20, 39]
    assert field.auxiliary_available.tolist() == [False, True, True]
    assert np.isnan(field.loc[0, FIELD])
    assert field.loc[1, "twenty_day_start_index"] == 1
    assert field.loc[1, "latest_source_index"] == 20
    assert field.loc[1, "origin_at"].hour == 15 and field.loc[1, "origin_at"].minute == 5
    assert field.loc[2, FIELD] == daily.loc[39, FIELD]


def test_align_rejects_identity_future_source_or_incomplete_group_contract():
    daily = build_field(market())
    wrong = samples(daily, [39])
    wrong.origin = daily.date.iloc[[40]].to_numpy()
    with pytest.raises(ValueError, match="日期或索引"):
        align(wrong, daily)
    changed = daily.copy()
    changed.loc[39, "latest_source_index"] = 40
    with pytest.raises(ValueError, match="原点之外"):
        align(samples(daily, [39]), changed)
    changed = daily.copy()
    changed.loc[39, "twenty_day_start_index"] = 21
    with pytest.raises(ValueError, match="二十个完整"):
        align(samples(daily, [39]), changed)
    changed = daily.copy()
    changed.loc[39, "negative_day_count20"] = 4.
    with pytest.raises(ValueError, match="不足五"):
        align(samples(daily, [39]), changed)


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
    a, b, status = predict(core, model, [.01]*8, [1.])
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




def test_true_balanced_ratio_one_is_active_while_unknown_is_exact_fallback():
    rows, core = training()
    model = fit(rows, core)
    a, b, status = predict(core, model, [.01]*8, [1.])
    assert status == "OPTIONAL_CORRECTION_AVAILABLE" and b != a
    a, b, status = predict(core, model, [.01]*8, [np.nan])
    assert status == "EXACT_CORE_FALLBACK" and a == b


def test_nonpositive_ratio_rejected_but_positive_below_or_above_one_legal():
    rows, core = training()
    for value in [0., -.1]:
        changed = rows.copy()
        changed.loc[1, FIELD] = value
        with pytest.raises(ValueError, match="必须为正"):
            fit(changed, core)
    model = fit(rows, core)
    for value in [0., -.1]:
        with pytest.raises(ValueError, match="必须为正"):
            predict(core, model, [.01]*8, [value])
    for value in [.1, 2.]:
        a, b, status = predict(core, model, [.01]*8, [value])
        assert np.isfinite(a) and np.isfinite(b) and status == "OPTIONAL_CORRECTION_AVAILABLE"
