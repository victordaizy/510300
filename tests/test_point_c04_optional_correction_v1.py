"""C04阶段内单系数的必要时钟、成员、原值和残差反例。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_c04_optional_correction_v1 import align, fit, predict, FIELDS, FIELD
from research.point_c04_information_intake_v1 import build_field
from research.point_optional_residual_model_v1 import design, system
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction
from research.learned_cycle_exit_v1 import FEATURES


def market():
    close = np.asarray([10, 9, 8, 9, 10, 12, 11, 10, 9.5, 10, 12, 11], float)
    amount = np.arange(1, len(close)+1)*100.
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(close)), "close": close,
        "high": close+.2, "low": close-.2, "amount": amount, "volume": amount/close, "dividend": 0.,
        "symbol": "510300.SH", "amount_unit": "CNY", "volume_unit": "share"})


def training():
    records = []
    for cycle in range(1, 4):
        for i in range(6):
            row = {"cycle_id": cycle, "origin_index": 100*cycle+i, "target": .02*i+.1*cycle,
                "sample_weight": 1./6, FIELD: float(i+cycle) if i else np.nan, "auxiliary_available": i != 0}
            row.update({name: .001*i for name in FEATURES})
            records.append(row)
    core = {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES.copy(),
        "mean": [0.]*8, "scale": [1.]*8, "coefficients": [.01]*8, "intercept": .02, "feature_clip": 5.}
    return pd.DataFrame(records), core


def samples(daily, indices):
    return pd.DataFrame({"cycle_id": [1]*len(indices), "origin_index": indices,
        "origin": daily.date.iloc[indices].to_numpy()})


def test_confirmed_stage_opens_on_current_day_without_backfill_and_baseline_is_fixed():
    data = market()
    daily = build_field(data)
    field = align(samples(daily, [5, 6, 7, 8]), daily)
    assert field.auxiliary_available.tolist() == [False, False, True, True]
    assert field.loc[2, "high_index"] == 5 and field.loc[2, "high_confirmation_index"] == 7
    assert field.loc[2, "phase_start_index"] == 7
    assert field.loc[2, FIELD] == pytest.approx(800./500.)
    assert field.loc[3, "baseline_amount_cny"] == 500.
    assert field.loc[3, FIELD] == pytest.approx(850./500.)
    assert field.loc[2, "origin_at"].hour == 15 and field.loc[2, "origin_at"].minute == 5


def test_phase_ending_and_reopening_preserve_unknown_and_new_baseline():
    daily = build_field(market())
    field = align(samples(daily, [9, 10, 11]), daily)
    assert field.auxiliary_available.tolist() == [True, False, True]
    assert np.isnan(field.loc[1, FIELD])
    assert field.loc[2, "phase_start_index"] == 11 and field.loc[2, "baseline_amount_cny"] == 900.


def test_bad_amount_stays_unknown_and_true_zero_ratio_stays_observed():
    data = market()
    data.loc[7, "amount"] = np.nan
    daily = build_field(data)
    field = align(samples(daily, [7, 8]), daily)
    assert field[FIELD].isna().all() and not field.auxiliary_available.any()
    data = market()
    data.loc[7, "amount"] = 0.
    data.loc[7, "volume"] = 0.
    daily = build_field(data)
    field = align(samples(daily, [7]), daily)
    assert field.auxiliary_available.iloc[0] and field[FIELD].iloc[0] == 0.


def test_future_data_dividend_and_cash_price_adjustment_cannot_change_prefix():
    data = market()
    whole = build_field(data)
    pd.testing.assert_frame_equal(whole.iloc[:9].reset_index(drop=True), build_field(data.iloc[:9]), check_exact=True)
    altered = data.copy()
    altered.loc[9:, "amount"] *= 5.
    altered.loc[9:, "dividend"] = .1
    for name in ["close", "high", "low"]:
        altered.loc[9:, name] += .3
    pd.testing.assert_frame_equal(whole.iloc[:9].reset_index(drop=True), build_field(altered).iloc[:9].reset_index(drop=True), check_exact=True)


def test_original_identity_and_future_confirmation_are_rejected():
    daily = build_field(market())
    wrong = samples(daily, [7])
    wrong.origin = daily.date.iloc[[8]].to_numpy()
    with pytest.raises(ValueError, match="日期或索引"):
        align(wrong, daily)
    altered = daily.copy()
    altered.loc[7, "high_confirmation_index"] = 8.
    with pytest.raises(ValueError, match="未来确认"):
        align(samples(daily, [7]), altered)


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
