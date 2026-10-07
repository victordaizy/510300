"""B04原事件字段时钟与隔离两系数可选函数的必要测试。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_b04_information_intake_v1 import (
    FIELD, FIELDS, LOW, METADATA, build_field, member_support, sample_std, validate_source_clock)


def recalculate(data):
    out = data.copy()
    out["previous_close"] = out.close.shift()
    out["total_simple"] = (out.close + out.dividend) / out.previous_close - 1
    out["total_log"] = np.log1p(out.total_simple)
    out["wealth"] = np.r_[1., np.cumprod(1 + out.total_simple.to_numpy()[1:])]
    out["open"] = out.previous_close.fillna(out.close)
    out["high"] = out[["open", "close"]].max(axis=1) + .003
    out["low"] = out[["open", "close"]].min(axis=1) - .003
    return out


def market(length=100, shocks=(35,)):
    returns = np.resize(np.array([.001, -.001, .002, -.002, 0.]), length)
    prices = [100000]
    for i in range(1, length):
        growth = .6 if i in shocks else 1 + returns[i]
        prices.append(round(prices[-1] * growth))
    return recalculate(pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=length),
        "symbol": "510300.SH", "close": np.array(prices)/1000., "dividend": 0.}))


def test_pre_event_reference_current_three_returns_and_fixed_stage_clock():
    data = market()
    validate_source_clock(data)
    fields = build_field(data)
    assert fields.loc[35, "accepted_shock"]
    assert fields.loc[35, FIELDS].isna().all()
    reference = sample_std(data.total_log.iloc[15:35].to_numpy())
    for i in range(36, 46):
        assert fields.loc[i, FIELD] == sample_std(data.total_log.iloc[i-2:i+1].to_numpy()) / reference
        assert fields.loc[i, "frozen_pre_shock_rv20"] == reference
        assert fields.loc[i, "post_shock_age"] == i-35
        assert fields.loc[i, "earliest_source_index"] == 15
        assert fields.loc[i, "latest_source_index"] == i
    assert fields.loc[46, FIELDS].isna().all()
    boundary = data.copy()
    boundary.loc[35, "total_log"] = -2 * reference
    assert not build_field(boundary).loc[35, "raw_shock"]


def test_ten_actual_session_dedup_and_new_shock_reset_overrides_prior_age_ten():
    data = market(shocks=(35, 44, 45))
    fields = build_field(data)
    assert fields.loc[35, "accepted_shock"]
    assert fields.loc[44, "raw_shock"] and fields.loc[44, "deduplicated_shock"]
    assert fields.loc[44, "shock_origin_index"] == 35
    assert fields.loc[45, "accepted_shock"]
    assert fields.loc[45, "shock_origin_index"] == 45
    assert fields.loc[45, FIELDS].isna().all()
    assert fields.loc[46, "post_shock_age"] == 1


def test_true_zero_short_volatility_and_ratios_above_one_remain_distinct_from_unknown():
    data = market()
    data.loc[37:39, "total_log"] = .01
    fields = build_field(data)
    assert fields.loc[39, FIELD] == 0
    assert fields.loc[36, FIELD] > 1
    assert fields.loc[34, FIELDS].isna().all()
    constant = market(shocks=())
    constant["total_log"] = .001
    constant.loc[0, "total_log"] = np.nan
    assert build_field(constant).loc[21, "field_status"] == "NO_VIEW_NONPOSITIVE_PRE_SHOCK_VARIANCE"


def test_cash_mapped_low_strict_integer_equality_and_off_grid_rejection():
    data = market()
    data.loc[35, ["close", "dividend"]] = [10., .1]
    data.loc[36, "close"] = 9.5
    data = recalculate(data)
    data.loc[35, "low"] = 9.091
    data.loc[36, "low"] = 9.1
    validate_source_clock(data)
    assert build_field(data).loc[36, LOW] == 0
    data.loc[36, "low"] = 9.099
    assert build_field(data).loc[36, LOW] == 1
    data.loc[36, ["low", "dividend"]] = [9.1, .001]
    data = recalculate(data)
    data.loc[35, "low"] = 9.091
    data.loc[36, "low"] = 9.1
    assert build_field(data).loc[36, LOW] == 0
    data.loc[36, "low"] += .0001
    with pytest.raises(ValueError, match="0.001"):
        build_field(data)


def test_missing_source_clears_event_and_never_compresses_or_fills_windows():
    data = market()
    data.loc[37, "total_log"] = np.nan
    fields = build_field(data)
    assert fields.loc[37:57, FIELDS].isna().all().all()
    assert fields.loc[58, FIELDS].isna().all()
    assert fields.loc[38, "field_status"] == "NO_VIEW_MISSING_RAW_RETURN_OR_CASH_SOURCE"
    anchor_missing = market()
    anchor_missing.loc[35, "low"] = np.nan
    assert build_field(anchor_missing).loc[36:45, FIELDS].isna().all().all()
    current_missing = market()
    current_missing.loc[37, "low"] = np.nan
    partial = build_field(current_missing)
    assert partial.loc[37, FIELDS].isna().all()
    assert np.isfinite(partial.loc[38, FIELDS].to_numpy(float)).all()


def test_future_changes_do_not_repaint_and_economic_source_identity_is_checked():
    original = market()
    changed = original.copy()
    changed.loc[80:, "close"] *= 2
    changed.loc[80:, "dividend"] += .1
    changed = recalculate(changed)
    pd.testing.assert_frame_equal(build_field(original).iloc[:80], build_field(changed).iloc[:80],
                                  check_exact=True)
    tampered = original.copy()
    tampered.loc[35, "wealth"] += .01
    with pytest.raises(AssertionError):
        validate_source_clock(tampered)
    tampered = original.copy()
    tampered.loc[0, ["previous_close", "total_simple", "total_log"]] = [100., 0., 0.]
    with pytest.raises(ValueError, match="首个市场点"):
        validate_source_clock(tampered)
    wrong = original.copy()
    wrong.loc[36, "symbol"] = "OTHER"
    with pytest.raises(ValueError, match="标的"):
        build_field(wrong)


def test_original_members_and_maturity_are_preserved_when_one_phase_is_unknown():
    data = market()
    fields = build_field(data)
    indexes = [34, 36, 37]
    samples = pd.DataFrame({"cycle_id": [1, 1, 1], "origin_index": indexes,
        "origin": data.date.iloc[indexes].to_numpy(), "exit_index": [50, 50, 50],
        "mature_date": data.date.iloc[[50, 50, 50]].to_numpy()})[METADATA]
    record = {"training_cycles": [1], "training_cycle_count": 1, "training_rows": 3,
        "fit_index": 60, "fit_origin": data.date.iloc[60].isoformat(), "model": {"saved": True}}
    support = member_support(samples, [record], fields)
    assert support.loc[0, "original_training_rows"] == 3
    assert support.loc[0, "b04_available_rows"] == 2
    assert support.loc[0, "b04_missing_rows"] == 1
    assert not support.loc[0, "all_original_members_supported"]
    with pytest.raises(ValueError, match="数量变化"):
        member_support(samples.iloc[[1, 2]], [record], fields)
    premature = copy.deepcopy(record)
    premature["fit_index"] = 49
    with pytest.raises(ValueError, match="尚未成熟"):
        member_support(samples, [premature], fields)


from research.point_b04_optional_correction_v1 import align, fit, predict, design, system, KIND
from research.learned_cycle_exit_v1 import FEATURES
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction


def training():
    records = []
    for cycle in range(1, 4):
        for i in range(6):
            row = {"cycle_id": cycle, "origin_index": cycle*100+i,
                "target": .02*i+.1*cycle, "sample_weight": 1./6,
                FIELD: float(i+cycle)/2. if i else np.nan,
                LOW: float(i % 2) if i else np.nan, "auxiliary_available": i != 0}
            row.update({name: .001*i for name in FEATURES})
            records.append(row)
    core = {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES.copy(),
        "mean": [0.]*8, "scale": [1.]*8, "coefficients": [.01]*8,
        "intercept": .02, "feature_clip": 5.}
    return pd.DataFrame(records), core


def optional_states(daily, indices):
    return pd.DataFrame({"cycle_id": [1]*len(indices), "origin_index": indices,
        "origin": daily.date.iloc[indices].to_numpy()})


def test_optional_align_keeps_shock_day_and_expired_stage_unknown_with_original_records():
    data = market()
    daily = build_field(data)
    states = optional_states(daily, [34, 35, 36, 45, 46])
    out = align(states, daily)
    assert out.origin_index.tolist() == [34, 35, 36, 45, 46]
    assert out.auxiliary_available.tolist() == [False, False, True, True, False]
    assert out.loc[[0, 1, 4], FIELDS].isna().all().all()
    assert out.loc[2, "post_shock_age"] == 1 and out.loc[3, "post_shock_age"] == 10
    assert out.loc[2, FIELD] > 1 and out.loc[2, LOW] in (0., 1.)
    assert out.loc[2, "origin_at"].hour == 15 and out.loc[2, "origin_at"].minute == 5
    pd.testing.assert_frame_equal(states, optional_states(daily, [34, 35, 36, 45, 46]), check_exact=True)


def test_optional_align_rejects_bad_identity_future_clock_changed_stage_or_anchor():
    daily = build_field(market())
    wrong = optional_states(daily, [39])
    wrong.origin = daily.date.iloc[[40]].to_numpy()
    with pytest.raises(ValueError, match="日期或索引"):
        align(wrong, daily)
    for key, value in (("latest_source_index", 40), ("post_shock_age", 0),
        ("post_shock_age", 11), ("earliest_source_index", 16),
        ("frozen_pre_shock_rv20", 0.), (LOW, .5)):
        changed = daily.copy()
        changed.loc[39, key] = value
        with pytest.raises(ValueError):
            align(optional_states(daily, [39]), changed)
    changed = daily.copy()
    changed.loc[35, "accepted_shock"] = False
    with pytest.raises(ValueError, match="未经接受"):
        align(optional_states(daily, [39]), changed)


def test_two_coefficients_match_registered_equation_and_all_core_members_stay():
    rows, core = training()
    original_rows, original_core = rows.copy(deep=True), copy.deepcopy(core)
    model = fit(rows, core)
    dx, dy, w = system(rows, core, model)
    expected = np.linalg.solve(dx.T@(w[:, None]*dx)+np.eye(2), dx.T@(w*dy))
    np.testing.assert_allclose(model["coefficients"], expected, atol=1e-15, rtol=0)
    assert model["kind"] == KIND and model["features"] == FIELDS
    assert model["global_intercept"] == 0. and model["ridge_alpha"] == 1.
    assert model["training_rows"] == 18 and model["unknown_training_rows_retained"] == 3
    assert original_core == core
    pd.testing.assert_frame_equal(rows, original_rows, check_exact=True)
    np.testing.assert_allclose(np.average(design(rows, model), axis=0, weights=rows.sample_weight), 0., atol=1e-12)


def test_stage_outside_two_raw_nulls_return_exact_original_float_and_no_partial_splicing():
    rows, core = training()
    model = fit(rows, core)
    a, b, status = predict(core, model, [.01]*8, [np.nan, np.nan])
    assert a == b == within_cycle_prediction(core, [.01]*8)
    assert status == "EXACT_CORE_FALLBACK"
    assert rows.loc[~rows.auxiliary_available, FIELDS].isna().all().all()
    for wrong in ([np.nan], [np.nan, 0.], [np.nan, np.inf], [0., 1., 2.]):
        with pytest.raises(ValueError):
            predict(core, model, [.01]*8, wrong)


def test_all_unknown_original_training_does_not_create_coefficients_or_intercept():
    rows, core = training()
    rows[FIELDS] = np.nan
    rows.auxiliary_available = False
    model = fit(rows, core)
    assert model["coefficients"] == [0., 0.] and not model["identified"]
    assert model["unknown_training_rows_retained"] == len(rows)
    a, b, status = predict(core, model, [.01]*8, [0., 0.])
    assert a == b and status == "EXACT_CORE_FALLBACK" and model["global_intercept"] == 0.


def test_common_cycle_target_shift_cannot_enter_two_auxiliary_coefficients():
    rows, core = training()
    model = fit(rows, core)
    changed = rows.copy()
    changed.target += changed.cycle_id.map({1: 2., 2: -1., 3: 3.})
    shifted = fit(changed, core)
    np.testing.assert_allclose(model["coefficients"], shifted["coefficients"], atol=1e-12, rtol=0)


def test_original_stage_outside_training_rows_remain_in_cycle_contrasts():
    rows, core = training()
    model = fit(rows, core)
    changed = rows.copy()
    changed.loc[(changed.cycle_id == 1) & ~changed.auxiliary_available, "target"] += 1.
    after = fit(changed, core)
    assert after["training_rows"] == model["training_rows"] == 18
    assert np.max(np.abs(np.asarray(after["coefficients"])-model["coefficients"])) > 1e-7


def test_true_zero_and_fractional_or_above_one_ratio_active_and_invalid_low_flag_rejected():
    rows, core = training()
    model = fit(rows, core)
    for raw in ([0., 0.], [.5, 0.], [2.5, 1.]):
        a, b, status = predict(core, model, [.01]*8, raw)
        assert np.isfinite(a) and np.isfinite(b) and status == "OPTIONAL_CORRECTION_AVAILABLE"
    for raw in ([-.1, 0.], [.5, .5], [0., -1.], [0., 2.], [np.inf, np.inf]):
        with pytest.raises(ValueError):
            predict(core, model, [.01]*8, raw)
    for key, value in ((FIELD, -.5), (LOW, .5)):
        changed = rows.copy()
        changed.loc[1, key] = value
        with pytest.raises(ValueError):
            fit(changed, core)
