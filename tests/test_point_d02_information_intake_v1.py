"""D02现金分解、严格负缺口、阶段外未知及完整原成员的必要测试。"""
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
