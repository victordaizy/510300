"""B03严格现金低点、首次收复、再破计数及完整原成员的必要测试。"""
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
