"""C01二十日中位数比、组数边界、未知、现金时钟及完整原成员必要测试。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_c01_information_intake_v1 import (
    FIELD, METADATA, build_field, member_support, validate_source_clock)


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


def test_original_members_not_deleted_and_unmatured_records_rejected():
    data = market()
    field = build_field(data)
    field.loc[24, FIELD] = np.nan
    samples = pd.DataFrame({"cycle_id": [1, 1, 2], "origin_index": [20, 24, 26],
        "origin": data.date.iloc[[20, 24, 26]].to_numpy(), "exit_index": [35, 35, 37],
        "mature_date": data.date.iloc[[35, 35, 37]].to_numpy()})
    assert list(samples.columns) == METADATA
    record = {"training_cycles": [1, 2], "training_cycle_count": 2, "training_rows": 3,
              "fit_index": 40, "fit_origin": str(data.date.iloc[40].date()), "model": {"saved": True}}
    result = member_support(samples, [record], field).iloc[0]
    assert result.c01_available_rows == 2 and result.c01_missing_rows == 1
    assert result.original_training_rows == 3 and not result.all_original_members_supported
    future = copy.deepcopy(record)
    future["fit_index"] = 36
    with pytest.raises(ValueError): member_support(samples, [future], field)
    with pytest.raises(ValueError): member_support(samples.iloc[:2].copy(), [record], field)
