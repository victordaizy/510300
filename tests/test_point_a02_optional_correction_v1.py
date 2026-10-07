"""验证A02完整阶段时钟、真实零、原成员和固定单系数函数。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.point_a02_optional_correction_v1 import align, fit, predict, FIELD
from research.point_a02_information_intake_v1 import build_field
from research.point_optional_residual_model_v1 import design, system
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction
from research.learned_cycle_exit_v1 import FEATURES


def market():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=40),
        "symbol": "510300.SH", "close": 10., "high": 11., "low": 9., "dividend": 0.})
    data.loc[21, ["close", "high", "low"]] = [12., 13., 9.]
    data.loc[22, ["close", "high", "low"]] = [11., 12., 10.]
    data.loc[23, ["close", "high", "low"]] = [11., 12., 9.9]
    data.loc[24, ["close", "high", "low"]] = [10.9, 12., 10.]
    return data


def training():
    records = []
    for cycle in range(1, 4):
        for i in range(6):
            row = {"cycle_id": cycle, "origin_index": 100*cycle+i,
                "target": .02*i+.1*cycle, "sample_weight": 1./6,
                FIELD: (i+cycle-2)/6. if i else np.nan, "auxiliary_available": i != 0}
            row.update({name: .001*i for name in FEATURES})
            records.append(row)
    core = {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES.copy(),
        "mean": [0.]*8, "scale": [1.]*8, "coefficients": [.01]*8,
        "intercept": .02, "feature_clip": 5.}
    return pd.DataFrame(records), core


def samples(daily, indices):
    return pd.DataFrame({"cycle_id": [1]*len(indices), "origin_index": indices,
        "origin": daily.date.iloc[indices].to_numpy()})


def test_prior_upper_and_atr_freeze_before_breakout_with_inclusive_boundaries():
    daily = build_field(market())
    field = align(samples(daily, [21, 22, 23, 24]), daily)
    assert field.auxiliary_available.tolist() == [False, True, True, True]
    assert field.loc[1, "frozen_breakout_upper"] == 11.
    assert field.loc[1, "frozen_previous_atr20"] == 2.
    assert field.loc[1:3, FIELD].tolist() == [1., .5, 1/3]
    assert field.loc[1, "origin_at"].hour == 15 and field.loc[1, "origin_at"].minute == 5


def test_expired_phase_preserves_unknown_and_true_zero_remains_observed():
    data = market()
    data.loc[22, ["close", "high", "low"]] = [10., 11., 9.]
    daily = build_field(data)
    field = align(samples(daily, [21, 22, 24, 25]), daily)
    assert field.auxiliary_available.tolist() == [False, True, True, False]
    assert field.loc[1, FIELD] == 0.
    assert field.loc[2, "observed_day_count"] == 3
    assert np.isnan(field.loc[3, FIELD])
    assert field.loc[3, "field_status"] == "NO_VIEW_THREE_DAY_PHASE_EXPIRED"


def test_new_breakout_becomes_anchor_only_from_next_complete_day():
    data = market()
    data.loc[22, ["close", "high", "low"]] = [14., 15., 10.]
    data.loc[23, ["close", "high", "low"]] = [13., 14., 12.]
    daily = build_field(data)
    field = align(samples(daily, [22, 23, 25, 26]), daily)
    assert field.loc[0, "latest_prior_breakout_index"] == 21
    assert field.loc[1, "latest_prior_breakout_index"] == 22
    assert field.loc[1, "frozen_breakout_upper"] == 13.
    assert field.loc[1, "observed_day_count"] == 1
    assert field.loc[2, "observed_day_count"] == 3
    assert not field.auxiliary_available.iloc[3]


def test_future_price_and_dividend_suffix_cannot_rewrite_prefix():
    data = market()
    whole = build_field(data)
    altered = data.copy()
    altered.loc[28, "dividend"] = .8
    altered.loc[29:, ["close", "high", "low"]] = [30., 35., 29.]
    pd.testing.assert_frame_equal(build_field(data.iloc[:24]),
        whole.iloc[:24].reset_index(drop=True), check_exact=True)
    pd.testing.assert_frame_equal(whole.iloc[:24].reset_index(drop=True),
        build_field(altered).iloc[:24].reset_index(drop=True), check_exact=True)


def test_original_identity_and_current_or_future_anchor_are_rejected():
    daily = build_field(market())
    bad = samples(daily, [22])
    bad.origin = daily.date.iloc[[23]].to_numpy()
    with pytest.raises(ValueError, match="日期或索引"):
        align(bad, daily)
    changed = daily.copy()
    changed.loc[22, "latest_prior_breakout_index"] = 22.
    with pytest.raises(ValueError, match="当前或未来"):
        align(samples(daily, [22]), changed)


def test_bad_phase_or_known_raw_range_cannot_be_silently_used():
    daily = build_field(market())
    changed = daily.copy()
    changed.loc[22, "phase_age"] = 4.
    with pytest.raises(ValueError, match="第1至3"):
        align(samples(daily, [22]), changed)
    changed = daily.copy()
    changed.loc[22, FIELD] = 1.1
    with pytest.raises(ValueError, match="超出0至1"):
        align(samples(daily, [22]), changed)
    rows, core = training()
    rows.loc[1, FIELD] = -.1
    with pytest.raises(ValueError, match="超出0至1"):
        fit(rows, core)


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


def test_known_zero_uses_correction_and_is_not_missing_fallback():
    rows, core = training()
    model = fit(rows, core)
    a, b, status = predict(core, model, [.01]*8, [0.])
    assert status == "OPTIONAL_CORRECTION_AVAILABLE" and b != a
    with pytest.raises(ValueError, match="超出0至1"):
        predict(core, model, [.01]*8, [1.1])
