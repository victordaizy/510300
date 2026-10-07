"""B03原字段时钟与隔离两系数可选函数的必要测试。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_b03_information_intake_v1 import (
    COUNT, FIELD, FIELDS, METADATA, build_field, member_support, validate_source_clock)


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


def market(length=100, event=True):
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=length),
        "symbol": "510300.SH", "close": 100. + .01*np.arange(length), "dividend": 0.})
    if event:
        data.loc[35:42, "close"] = [60., 90., 101., 101., 90., 89., 101., 91.]
    return recalculate(data)


def test_frozen_previous_twenty_low_current_clock_and_first_delay_are_causal():
    data = market()
    validate_source_clock(data)
    fields = build_field(data)
    assert fields.loc[35, "raw_break20"]
    assert fields.loc[35, "break_origin_index"] == 35
    assert fields.loc[35, "frozen_low_origin_index"] == 15
    assert fields.loc[35:36, FIELDS].isna().all().all()
    assert fields.loc[37, FIELD] == 2
    assert fields.loc[37, "first_reclaim_index"] == 37
    for i in range(37, 43):
        assert fields.loc[i, FIELD] == 2
        assert fields.loc[i, "earliest_source_index"] == 15
        assert fields.loc[i, "latest_source_index"] == i


def test_rebreak_counts_transitions_only_and_new_strict_break_resets_anchor():
    data = market()
    fields = build_field(data)
    assert fields.loc[37, COUNT] == 0
    assert fields.loc[39, COUNT] == 1
    assert fields.loc[40, COUNT] == 1
    assert fields.loc[42, COUNT] == 2
    data.loc[44, "close"] = 55.
    data = recalculate(data)
    changed = build_field(data)
    assert changed.loc[44, "raw_break20"]
    assert changed.loc[44, "break_origin_index"] == 44
    assert changed.loc[44, FIELDS].isna().all()
    assert changed.loc[45, FIELD] == 1 and changed.loc[45, COUNT] == 0
    assert changed.loc[42, COUNT] == 2


def test_same_day_reclaim_true_zero_strict_equal_unknown_and_superseded_pending_kept():
    data = market(event=False)
    data.loc[35, "close"] = 100.2
    data = recalculate(data)
    data.loc[35, "low"] = 100.
    fields = build_field(data)
    assert fields.loc[35, "raw_break20"]
    assert fields.loc[35, FIELD] == fields.loc[35, COUNT] == 0
    pending = market()
    threshold = pending.loc[15, "low"]
    pending.loc[37, "close"] = threshold
    pending.loc[38, "close"] = threshold + .001
    pending = recalculate(pending)
    equal = build_field(pending)
    assert equal.loc[37, FIELDS].isna().all()
    assert equal.loc[38, FIELD] == 3
    superseded = market()
    superseded.loc[36, "close"] = 59.
    superseded = recalculate(superseded)
    superseded_fields = build_field(superseded)
    assert superseded_fields.loc[35, "raw_break20"] and superseded_fields.loc[36, "raw_break20"]
    assert superseded_fields.loc[35, FIELDS].isna().all()
    assert superseded_fields.loc[37, FIELD] == 1
    assert superseded_fields.loc[37, "break_origin_index"] == 36
    assert build_field(market(event=False))[FIELDS].isna().all().all()


def test_exact_cash_low_equality_is_not_new_break_and_off_grid_is_rejected():
    data = market(event=False)
    data["close"] = 10.
    data.loc[20, "dividend"] = .1
    data.loc[21, "close"] = 9.5
    data = recalculate(data)
    data.loc[20, "low"] = 9.091
    data.loc[21, "low"] = 9.1
    validate_source_clock(data)
    fields = build_field(data)
    assert fields.loc[20, "raw_break20"]
    assert not fields.loc[21, "raw_break20"]
    data.loc[21, "low"] = 9.099
    assert build_field(data).loc[21, "raw_break20"]
    data.loc[21, "low"] += .0001
    with pytest.raises(ValueError, match="0.001"):
        build_field(data)


def test_missing_low_or_cash_clears_event_and_requires_uncompressed_prior_window():
    for column in ["low", "dividend", "close"]:
        data = market()
        data.loc[40, column] = np.nan
        fields = build_field(data)
        assert fields.loc[40:60, FIELDS].isna().all().all()
        assert fields.loc[61, FIELDS].isna().all()
        assert fields.loc[41, "field_status"] == "NO_VIEW_TWENTY_PRIOR_LOW_SOURCE_NOT_READY"
    data = market()
    data.loc[40, "dividend"] = -1.
    assert build_field(data).loc[40, FIELDS].isna().all()


def test_future_source_does_not_repaint_and_source_identity_is_validated():
    data = market()
    changed = data.copy()
    changed.loc[80:, "close"] *= 2
    changed.loc[80:, "dividend"] += .1
    changed = recalculate(changed)
    pd.testing.assert_frame_equal(build_field(data).iloc[:80], build_field(changed).iloc[:80],
                                  check_exact=True)
    tampered = data.copy()
    tampered.loc[35, "wealth"] += .01
    with pytest.raises(AssertionError):
        validate_source_clock(tampered)
    initial = data.copy()
    initial.loc[0, ["previous_close", "total_simple", "total_log"]] = [100., 0., 0.]
    with pytest.raises(ValueError, match="首个市场点"):
        validate_source_clock(initial)
    wrong = data.copy()
    wrong.loc[36, "symbol"] = "OTHER"
    with pytest.raises(ValueError, match="标的"):
        build_field(wrong)


def test_original_members_unknown_and_maturity_are_preserved_without_dropping_pending():
    data = market()
    fields = build_field(data)
    indexes = [35, 37, 39]
    samples = pd.DataFrame({"cycle_id": [1, 1, 1], "origin_index": indexes,
        "origin": data.date.iloc[indexes].to_numpy(), "exit_index": [50, 50, 50],
        "mature_date": data.date.iloc[[50, 50, 50]].to_numpy()})[METADATA]
    record = {"training_cycles": [1], "training_cycle_count": 1, "training_rows": 3,
        "fit_index": 60, "fit_origin": data.date.iloc[60].isoformat(), "model": {"saved": True}}
    support = member_support(samples, [record], fields)
    assert support.loc[0, "original_training_rows"] == 3
    assert support.loc[0, "b03_available_rows"] == 2
    assert support.loc[0, "b03_missing_rows"] == 1
    assert not support.loc[0, "all_original_members_supported"]
    with pytest.raises(ValueError, match="数量变化"):
        member_support(samples.iloc[[1, 2]], [record], fields)
    premature = copy.deepcopy(record)
    premature["fit_index"] = 49
    with pytest.raises(ValueError, match="尚未成熟"):
        member_support(samples, [premature], fields)


from research.point_b03_optional_correction_v1 import align, fit, predict, design, system, KIND
from research.learned_cycle_exit_v1 import FEATURES
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction


def training():
    records = []
    for cycle in range(1, 4):
        for i in range(6):
            row = {"cycle_id": cycle, "origin_index": cycle*100+i,
                "target": .02*i+.1*cycle, "sample_weight": 1./6,
                FIELD: float(i+cycle) if i else np.nan,
                COUNT: float(i % 3) if i else np.nan, "auxiliary_available": i != 0}
            row.update({name: .001*i for name in FEATURES})
            records.append(row)
    core = {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES.copy(),
        "mean": [0.]*8, "scale": [1.]*8, "coefficients": [.01]*8,
        "intercept": .02, "feature_clip": 5.}
    return pd.DataFrame(records), core


def optional_states(daily, indices):
    return pd.DataFrame({"cycle_id": [1]*len(indices), "origin_index": indices,
        "origin": daily.date.iloc[indices].to_numpy()})


def test_optional_align_keeps_pending_raw_unknown_and_confirmed_causal_records():
    daily = build_field(market())
    states = optional_states(daily, [35, 37, 39, 42])
    out = align(states, daily)
    assert out.origin_index.tolist() == [35, 37, 39, 42]
    assert out.auxiliary_available.tolist() == [False, True, True, True]
    assert out.loc[0, FIELDS].isna().all()
    assert out.loc[1, FIELD] == 2 and out.loc[1, COUNT] == 0
    assert out.loc[2, COUNT] == 1 and out.loc[3, COUNT] == 2
    assert out.loc[1, "origin_at"].hour == 15 and out.loc[1, "origin_at"].minute == 5
    pd.testing.assert_frame_equal(states, optional_states(daily, [35, 37, 39, 42]), check_exact=True)


def test_optional_align_rejects_bad_identity_future_clock_and_changed_event_records():
    daily = build_field(market())
    wrong = optional_states(daily, [39])
    wrong.origin = daily.date.iloc[[40]].to_numpy()
    with pytest.raises(ValueError, match="日期或索引"):
        align(wrong, daily)
    for key, value in (("latest_source_index", 40), ("first_reclaim_index", 40),
        ("earliest_source_index", 16), (FIELD, 3.), (COUNT, 3.)):
        changed = daily.copy()
        changed.loc[39, key] = value
        with pytest.raises(ValueError):
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


def test_pending_two_raw_nulls_return_exact_original_float_and_no_partial_splicing():
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


def test_original_pending_training_rows_remain_in_cycle_contrasts():
    rows, core = training()
    model = fit(rows, core)
    changed = rows.copy()
    changed.loc[(changed.cycle_id == 1) & ~changed.auxiliary_available, "target"] += 1.
    after = fit(changed, core)
    assert after["training_rows"] == model["training_rows"] == 18
    assert np.max(np.abs(np.asarray(after["coefficients"])-model["coefficients"])) > 1e-7


def test_true_same_day_zero_is_active_and_invalid_fractional_or_negative_records_rejected():
    rows, core = training()
    model = fit(rows, core)
    a, b, status = predict(core, model, [.01]*8, [0., 0.])
    assert np.isfinite(a) and np.isfinite(b) and status == "OPTIONAL_CORRECTION_AVAILABLE"
    for raw in ([-1., 0.], [.5, 0.], [0., -.1], [0., .5], [np.inf, np.inf]):
        with pytest.raises(ValueError):
            predict(core, model, [.01]*8, raw)
    for key in FIELDS:
        changed = rows.copy()
        changed.loc[1, key] = .5
        with pytest.raises(ValueError):
            fit(changed, core)
