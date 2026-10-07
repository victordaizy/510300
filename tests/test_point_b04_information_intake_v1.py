"""B04事前冲击、十日事件时钟、现金低点和完整成员的必要测试。"""
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
