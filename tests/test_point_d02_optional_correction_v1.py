"""D02原现金缺口及保留部分未知的隔离两系数函数必要测试。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_d02_information_intake_v1 import (
    FIELD, FIELDS, GAP, METADATA, build_field, member_support, validate_source_clock)


def recalculate(data):
    out = data.copy()
    out["previous_close"] = out.close.shift()
    out["total_simple"] = (out.close+out.dividend)/out.previous_close-1
    out["total_log"] = np.log1p(out.total_simple)
    out["wealth"] = np.r_[1., np.cumprod(1+out.total_simple.to_numpy()[1:])]
    return out


def market(length=70):
    closes = 5000+2*np.arange(length)
    previous = np.r_[5000, closes[:-1]]
    shifts = np.resize(np.array([-9, 7, 0]), length)
    opens = previous+shifts
    opens[0] = closes[0]
    return recalculate(pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=length), "symbol": "510300.SH",
        "open": opens/1000., "close": closes/1000., "dividend": 0.}))


def test_exact_current_day_log_absorption_keeps_values_above_one():
    data = market()
    validate_source_clock(data)
    fields = build_field(data)
    for t in [3, 24, 45, 69]:
        previous, opens, closes = data.loc[t, ["previous_close", "open", "close"]]
        gap = np.log(opens/previous)
        intra = np.log(closes/opens)
        assert fields.loc[t, GAP] == pytest.approx(gap, abs=1e-14)
        assert fields.loc[t, FIELD] == pytest.approx(intra/abs(gap), abs=1e-12)
        assert fields.loc[t, FIELD] > 1
        assert fields.loc[t, "earliest_source_index"] == t-1
        assert fields.loc[t, "latest_source_index"] == t
    assert fields.loc[0, FIELDS].isna().all()


def test_strict_negative_gap_true_zero_and_exact_decimal_zero_are_distinct():
    data = market()
    data.loc[3, "close"] = data.loc[3, "open"]
    data.loc[6, "close"] = data.loc[6, "open"]-.001
    fields = build_field(data)
    assert fields.loc[3, FIELD] == 0
    assert fields.loc[6, FIELD] == 0
    assert fields.loc[1, GAP] > 0 and pd.isna(fields.loc[1, FIELD])
    assert fields.loc[2, GAP] == 0 and pd.isna(fields.loc[2, FIELD])
    data.loc[9, "open"] = data.loc[9, "previous_close"]-.001
    assert np.isfinite(build_field(data).loc[9, FIELD])
    data.loc[12, ["previous_close", "open", "dividend", "close"]] = [.7, .5, .2, .6]
    exact = build_field(data)
    assert exact.loc[12, GAP] == 0
    assert exact.loc[12, "overnight_delta_units"] == 0
    assert pd.isna(exact.loc[12, FIELD])


def test_raw_missing_is_unknown_without_fill_or_future_carry():
    for column in ["open", "close", "previous_close", "dividend"]:
        for invalid in [np.nan, np.inf, -1.]:
            data = market()
            data.loc[30, column] = invalid
            fields = build_field(data)
            assert fields.loc[30, FIELDS].isna().all()
            assert fields.loc[30, "field_status"] == "NO_VIEW_MISSING_RAW_PRICE_OR_CASH_SOURCE"
            assert np.isfinite(fields.loc[33, FIELD])
    data = market()
    data.loc[30, "open"] = 0.
    assert pd.isna(build_field(data).loc[30, FIELD])
    data = market()
    data.loc[0, ["previous_close", "total_simple", "total_log"]] = [5., 0., 0.]
    with pytest.raises(ValueError, match="首个市场点"):
        validate_source_clock(data)


def test_cash_rights_log_identity_and_source_tamper_detection():
    data = market()
    data.loc[3, "dividend"] = .2
    fields = build_field(recalculate(data))
    assert fields.loc[3, GAP] > 0
    assert pd.isna(fields.loc[3, FIELD])
    data = recalculate(data)
    validate_source_clock(data)
    fields = build_field(data)
    np.testing.assert_allclose(fields[GAP].iloc[1:]+fields.cash_inclusive_intraday_log_return.iloc[1:],
                               data.total_log.iloc[1:], atol=1e-12, rtol=0)
    data.loc[3, "wealth"] += .01
    with pytest.raises(AssertionError):
        validate_source_clock(data)


def test_future_source_does_not_repaint_and_off_grid_is_not_rounded():
    original = market()
    changed = original.copy()
    changed.loc[31:, "open"] += 1.
    changed.loc[31:, "close"] += 2.
    changed.loc[31:, "dividend"] += .1
    pd.testing.assert_frame_equal(build_field(original).iloc[:31], build_field(changed).iloc[:31],
                                  check_exact=True)
    off_grid = original.copy()
    off_grid.loc[30, "open"] += .0001
    with pytest.raises(ValueError, match="0.001"):
        build_field(off_grid)
    wrong = original.copy()
    wrong.loc[30, "symbol"] = "OTHER"
    with pytest.raises(ValueError, match="标的"):
        build_field(wrong)


def test_same_economic_close_path_different_open_has_distinct_coordinate():
    data = market()
    changed = data.copy()
    changed.loc[30, "open"] -= .02
    pd.testing.assert_series_equal(data.total_log, changed.total_log)
    pd.testing.assert_series_equal(data.wealth, changed.wealth)
    before, after = build_field(data), build_field(changed)
    assert before.loc[30, GAP] != after.loc[30, GAP]
    assert before.loc[30, FIELD] != after.loc[30, FIELD]


def test_original_member_support_preserves_phase_unknown_and_maturity():
    data = market()
    fields = build_field(data)
    indexes = [3, 4, 6]
    samples = pd.DataFrame({"cycle_id": [1, 1, 1], "origin_index": indexes,
        "origin": data.date.iloc[indexes].to_numpy(), "exit_index": [10, 10, 10],
        "mature_date": data.date.iloc[[10, 10, 10]].to_numpy()})[METADATA]
    record = {"training_cycles": [1], "training_cycle_count": 1, "training_rows": 3,
        "fit_index": 20, "fit_origin": data.date.iloc[20].isoformat(), "model": {"saved": True}}
    support = member_support(samples, [record], fields)
    assert len(samples) == 3
    assert support.loc[0, "original_training_rows"] == 3
    assert support.loc[0, "d02_available_rows"] == 2
    assert support.loc[0, "d02_missing_rows"] == 1
    assert not support.loc[0, "all_original_members_supported"]
    with pytest.raises(ValueError, match="数量变化"):
        member_support(samples.iloc[[0, 2]], [record], fields)
    premature = copy.deepcopy(record)
    premature["fit_index"] = 9
    with pytest.raises(ValueError, match="尚未成熟"):
        member_support(samples, [premature], fields)
    wrong = samples.copy()
    wrong.loc[0, "origin"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="日期"):
        member_support(wrong, [record], fields)


from research.point_d02_optional_correction_v1 import align, fit, predict, design, system, KIND
from research.learned_cycle_exit_v1 import FEATURES
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction


def training():
    records = []
    for cycle in range(1, 4):
        for i in range(6):
            row = {"cycle_id": cycle, "origin_index": cycle*100+i,
                "target": .02*i+.1*cycle, "sample_weight": 1./6,
                FIELD: float(i+cycle)/2. if i else np.nan,
                GAP: -.001*(1+(i*i+cycle) % 5) if i else .001*cycle, "auxiliary_available": i != 0}
            row.update({name: .001*i for name in FEATURES})
            records.append(row)
    core = {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES.copy(),
        "mean": [0.]*8, "scale": [1.]*8, "coefficients": [.01]*8,
        "intercept": .02, "feature_clip": 5.}
    return pd.DataFrame(records), core


def optional_states(daily, indices):
    return pd.DataFrame({"cycle_id": [1]*len(indices), "origin_index": indices,
        "origin": daily.date.iloc[indices].to_numpy()})


def test_optional_align_preserves_initial_unknown_and_known_positive_or_zero_gap_without_absorption():
    daily = build_field(market())
    states = optional_states(daily, [0, 1, 2, 3, 6])
    out = align(states, daily)
    assert out.origin_index.tolist() == [0, 1, 2, 3, 6]
    assert out.auxiliary_available.tolist() == [False, False, False, True, True]
    assert out.loc[0, FIELDS].isna().all()
    assert pd.isna(out.loc[1, FIELD]) and out.loc[1, GAP] > 0
    assert pd.isna(out.loc[2, FIELD]) and out.loc[2, GAP] == 0
    assert out.loc[3, FIELD] > 1 and out.loc[3, GAP] < 0
    assert out.loc[3, "origin_at"].hour == 15 and out.loc[3, "origin_at"].minute == 5
    pd.testing.assert_frame_equal(states, optional_states(daily, [0, 1, 2, 3, 6]), check_exact=True)


def test_optional_align_rejects_bad_identity_clock_cash_decomposition_or_removed_partial_gap():
    daily = build_field(market())
    wrong = optional_states(daily, [3])
    wrong.origin = daily.date.iloc[[4]].to_numpy()
    with pytest.raises(ValueError, match="日期或索引"):
        align(wrong, daily)
    for key, value in (("latest_source_index", 4), ("earliest_source_index", 1),
        ("overnight_delta_units", 0), (FIELD, .5), (GAP, .01)):
        changed = daily.copy()
        changed.loc[3, key] = value
        with pytest.raises(ValueError):
            align(optional_states(daily, [3]), changed)
    changed = daily.copy()
    changed.loc[1, GAP] = np.nan
    with pytest.raises(ValueError, match="阶段外原gap"):
        align(optional_states(daily, [1]), changed)


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


def test_partial_or_full_unknown_input_returns_exact_core_without_rewriting_original_gap():
    rows, core = training()
    model = fit(rows, core)
    partial = rows.loc[~rows.auxiliary_available, FIELDS].copy()
    assert partial[FIELD].isna().all() and np.isfinite(partial[GAP]).all()
    for raw in ([np.nan, np.nan], [np.nan, .01], [np.nan, 0.], [np.nan, -.01]):
        a, b, status = predict(core, model, [.01]*8, raw)
        assert a == b == within_cycle_prediction(core, [.01]*8)
        assert status == "EXACT_CORE_FALLBACK"
    pd.testing.assert_frame_equal(partial, rows.loc[~rows.auxiliary_available, FIELDS], check_exact=True)
    changed = rows.copy()
    changed.loc[~changed.auxiliary_available, GAP] += 1.
    after = fit(changed, core)
    np.testing.assert_allclose(model["coefficients"], after["coefficients"], atol=0., rtol=0)
    for wrong in ([np.nan], [np.nan, np.inf], [0., -.01, 2.]):
        with pytest.raises(ValueError):
            predict(core, model, [.01]*8, wrong)


def test_all_unknown_original_training_does_not_create_coefficients_or_intercept():
    rows, core = training()
    rows[FIELD] = np.nan
    rows[GAP] = .001*rows.cycle_id
    rows.auxiliary_available = False
    model = fit(rows, core)
    assert model["coefficients"] == [0., 0.] and not model["identified"]
    assert model["unknown_training_rows_retained"] == len(rows)
    a, b, status = predict(core, model, [.01]*8, [0., -.01])
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


def test_true_zero_fractional_or_above_one_absorption_active_only_with_negative_signed_gap():
    rows, core = training()
    model = fit(rows, core)
    for raw in ([0., -.01], [.5, -.005], [2.5, -.03]):
        a, b, status = predict(core, model, [.01]*8, raw)
        assert np.isfinite(a) and np.isfinite(b) and status == "OPTIONAL_CORRECTION_AVAILABLE"
    for raw in ([-.1, -.01], [.5, 0.], [0., .01], [np.inf, np.inf]):
        with pytest.raises(ValueError):
            predict(core, model, [.01]*8, raw)
    for key, value in ((FIELD, -.5), (GAP, 0.)):
        changed = rows.copy()
        changed.loc[1, key] = value
        with pytest.raises(ValueError):
            fit(changed, core)
