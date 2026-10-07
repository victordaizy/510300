"""B02两日确认、十日二测、前序ATR/量窗及完整成员的必要测试。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_b02_information_intake_v1 import (
    FIELDS, METADATA, build_field, member_support, validate_source_clock)


def recalculate(data):
    out = data.copy()
    out["previous_close"] = out.close.shift()
    out["total_simple"] = (out.close + out.dividend) / out.previous_close - 1
    out["total_log"] = np.log1p(out.total_simple)
    out["wealth"] = np.r_[1., np.cumprod(1 + out.total_simple.to_numpy()[1:])]
    return out


def market(length=90, second=34):
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=length),
        "symbol": "510300.SH", "open": 100., "high": 100.003, "low": 99.997,
        "close": 100., "dividend": 0., "amount": 1000. + np.arange(length)})
    data.loc[30, ["close", "low"]] = [99.999, 99.994]
    data.loc[second, ["close", "low"]] = [99.999, 99.993]
    return recalculate(data)


def test_exact_old_five_records_completed_amount_window_and_source_clock():
    data = market()
    validate_source_clock(data)
    field = build_field(data)
    assert field.loc[32, "first_low_confirmed_today"]
    assert field.loc[34, "selected_second_test_today"]
    expected = [data.amount.iloc[34]/data.amount.iloc[28:33].mean(),
                data.total_log.iloc[30], data.total_log.iloc[34],
                (data.close.iloc[30]-data.low.iloc[30])/(data.high.iloc[30]-data.low.iloc[30]),
                (data.close.iloc[34]-data.low.iloc[34])/(data.high.iloc[34]-data.low.iloc[34])]
    np.testing.assert_allclose(field.loc[34, FIELDS].to_numpy(float), expected, atol=1e-12, rtol=0)
    assert field.loc[34, "first_test_origin_index"] == 30
    assert field.loc[34, "first_test_confirm_index"] == 32
    assert field.loc[34, "earliest_source_index"] == 9
    assert field.loc[34, "latest_source_index"] == 34
    assert field.loc[33, FIELDS].isna().all() and field.loc[35, FIELDS].isna().all()


def test_two_closed_higher_lows_ten_day_boundary_and_frozen_pre_first_atr():
    unconfirmed = market()
    unconfirmed.loc[32, "low"] = unconfirmed.loc[30, "low"]
    assert not build_field(unconfirmed).loc[34, "selected_second_test_today"]
    assert build_field(market(second=40)).loc[40, "selected_second_test_today"]
    assert not build_field(market(second=41)).loc[41, "selected_second_test_today"]
    distance_equal = market()
    distance_equal.loc[34, "low"] = 99.991
    assert not build_field(distance_equal).loc[34, "selected_second_test_today"]
    future_atr = distance_equal.copy()
    future_atr.loc[30, "high"] = 105.
    assert not build_field(future_atr).loc[34, "selected_second_test_today"]


def test_first_low_claimed_once_and_true_zero_records_remain_known():
    data = market()
    data.loc[35, ["close", "low"]] = [99.999, 99.992]
    field = build_field(recalculate(data))
    assert field.loc[34, "selected_second_test_today"]
    assert not field.loc[35, "selected_second_test_today"]
    zero = market()
    zero["amount"] = 1000.
    zero.loc[30, "close"] = 100.
    known = build_field(recalculate(zero))
    assert known.loc[34, "second_test_amount_ratio"] == 1
    assert known.loc[34, "first_test_log_return"] == 0
    assert np.isfinite(known.loc[34, FIELDS].to_numpy(float)).all()


def test_exact_cash_low_equality_not_a_break_and_off_grid_not_repaired():
    data = market()
    data[["open", "close"]] = 10.
    data["high"], data["low"] = 10.003, 9.997
    data.loc[30, ["low", "dividend"]] = [9.091, .1]
    data.loc[34, ["close", "low"]] = [9.5, 9.1]
    data = recalculate(data)
    field = build_field(data)
    assert field.loc[30, "raw_break20"]
    assert not field.loc[34, "raw_break20"]
    data.loc[34, "low"] = 9.099
    assert build_field(data).loc[34, "selected_second_test_today"]
    data.loc[34, "low"] += .0001
    with pytest.raises(ValueError, match="0.001"):
        build_field(data)


def test_unknown_amount_return_flat_range_and_missing_price_never_fill_records():
    for index in [28, 30, 32, 34]:
        for bad in [np.nan, 0., -1., np.inf]:
            data = market()
            data.loc[index, "amount"] = bad
            fields = build_field(data)
            assert fields.loc[34, "selected_second_test_today"]
            assert fields.loc[34, FIELDS].isna().all()
    missing_return = market()
    missing_return.loc[30, "total_log"] = np.nan
    assert build_field(missing_return).loc[34, FIELDS].isna().all()
    flat = market()
    flat.loc[30, ["open", "high", "low", "close"]] = 99.994
    flat = recalculate(flat)
    validate_source_clock(flat)
    assert build_field(flat).loc[34, "field_status"] == "NO_VIEW_ZERO_PRICE_RANGE_AT_TEST"
    missing_price = market()
    missing_price.loc[31, "low"] = np.nan
    assert not build_field(missing_price).loc[34, "selected_second_test_today"]


def test_future_data_does_not_repaint_and_source_identity_is_checked():
    original = market()
    changed = original.copy()
    changed.loc[70:, ["open", "high", "low", "close"]] *= 2
    changed.loc[70:, "dividend"] += .1
    changed.loc[70:, "amount"] *= 10
    changed = recalculate(changed)
    pd.testing.assert_frame_equal(build_field(original).iloc[:70], build_field(changed).iloc[:70],
                                  check_exact=True)
    tampered = original.copy()
    tampered.loc[30, "wealth"] += .01
    with pytest.raises(AssertionError):
        validate_source_clock(tampered)
    initial = original.copy()
    initial.loc[0, ["previous_close", "total_simple", "total_log"]] = [100., 0., 0.]
    with pytest.raises(ValueError, match="首个市场点"):
        validate_source_clock(initial)
    wrong = original.copy()
    wrong.loc[34, "symbol"] = "OTHER"
    with pytest.raises(ValueError, match="标的"):
        build_field(wrong)


def test_original_members_phase_unknown_and_maturity_are_not_changed():
    data = market()
    field = build_field(data)
    indexes = [33, 34, 35]
    samples = pd.DataFrame({"cycle_id": [1, 1, 1], "origin_index": indexes,
        "origin": data.date.iloc[indexes].to_numpy(), "exit_index": [50, 50, 50],
        "mature_date": data.date.iloc[[50, 50, 50]].to_numpy()})[METADATA]
    record = {"training_cycles": [1], "training_cycle_count": 1, "training_rows": 3,
        "fit_index": 60, "fit_origin": data.date.iloc[60].isoformat(), "model": {"saved": True}}
    support = member_support(samples, [record], field)
    assert support.loc[0, "original_training_rows"] == 3
    assert support.loc[0, "b02_available_rows"] == 1
    assert support.loc[0, "b02_missing_rows"] == 2
    assert not support.loc[0, "all_original_members_supported"]
    with pytest.raises(ValueError, match="数量变化"):
        member_support(samples.iloc[[1]], [record], field)
    premature = copy.deepcopy(record)
    premature["fit_index"] = 49
    with pytest.raises(ValueError, match="尚未成熟"):
        member_support(samples, [premature], field)
