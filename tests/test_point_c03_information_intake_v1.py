"""C03十日完整源、正负分母、现金时钟与原成员支持的必要测试。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_c03_information_intake_v1 import (
    FIELD, METADATA, build_field, member_support, validate_source_clock)


def market(returns=None, length=70):
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


def test_exact_ten_days_include_current_and_zero_day_amount_in_mean():
    data = market()
    field = build_field(data)
    assert field.loc[:9, FIELD].isna().all()
    for t in [10, 24, 45, 69]:
        r = data.total_log.iloc[t-9:t+1].to_numpy()
        a = data.amount.iloc[t-9:t+1].to_numpy()
        pos, neg = r > 0, r < 0
        expected = a.mean()*(r[pos].sum()/a[pos].sum()-np.abs(r[neg]).sum()/a[neg].sum())
        assert field.loc[t, FIELD] == pytest.approx(expected, abs=1e-14)
        assert field.loc[t, "ten_day_start_index"] == t-9
        assert field.loc[t, "latest_source_index"] == t
    changed = data.copy()
    changed.loc[24, "amount"] *= 10.
    assert build_field(changed).loc[24, FIELD] != pytest.approx(field.loc[24, FIELD], abs=1e-9)


def test_missing_or_nonpositive_amount_poison_exact_actual_window_without_compressing():
    for invalid in [0., -1., np.nan, np.inf]:
        data = market()
        data.loc[30, "amount"] = invalid
        field = build_field(data)
        assert len(field) == len(data)
        assert field.loc[30:39, FIELD].isna().all()
        assert np.isfinite(field.loc[29, FIELD]) and np.isfinite(field.loc[40, FIELD])
    data = market()
    data.loc[30, "total_log"] = np.nan
    assert build_field(data).loc[30:39, FIELD].isna().all()


def test_absent_positive_or_negative_group_is_unknown_but_balanced_true_zero_valid():
    for returns, status in [
        (np.repeat(.01, 30), "NO_VIEW_ZERO_NEGATIVE_GROUP_AMOUNT"),
        (np.repeat(-.01, 30), "NO_VIEW_ZERO_POSITIVE_GROUP_AMOUNT"),
        (np.zeros(30), "NO_VIEW_BOTH_GROUP_AMOUNTS_ZERO")]:
        data = market(returns)
        field = build_field(data)
        assert field.loc[10:, FIELD].isna().all()
        assert field.loc[10:, "field_status"].eq(status).all()
    balanced = market(np.tile([.01, -.01], 20))
    balanced["total_log"] = np.r_[np.nan, np.tile([.01, -.01], 20)]
    balanced["amount"] = 1000.
    field = build_field(balanced)
    assert field.loc[10:, FIELD].eq(0.).all()
    assert field.loc[10:, "field_status"].eq("C03_TEN_COMPLETE_DAYS_AND_BOTH_GROUPS_AVAILABLE").all()


def test_future_price_cash_and_amount_cannot_repaint_and_money_units_cancel():
    data = market()
    left = build_field(data.iloc[:40])
    future = data.copy()
    future.loc[40:, "close"] *= 3.
    future.loc[45, "dividend"] = 10.
    future.loc[40:, "amount"] *= 1000.
    future = recalculate(future)
    pd.testing.assert_frame_equal(left, build_field(future).iloc[:40].reset_index(drop=True), check_exact=True)
    scaled = data.copy()
    scaled["amount"] *= 1000.
    np.testing.assert_allclose(build_field(data)[FIELD], build_field(scaled)[FIELD],
                               atol=1e-14, rtol=0, equal_nan=True)
    wrong = data.copy()
    wrong["amount_unit"] = "万元"
    with pytest.raises(ValueError): build_field(wrong)


def test_cash_economic_log_identity_and_unobserved_initial_interval_are_bound():
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
    assert build_field(bad).loc[:9, FIELD].isna().all()
    with pytest.raises(ValueError): validate_source_clock(bad)


def test_same_return_path_can_have_different_group_amount_efficiency():
    first = market()
    second = first.copy()
    second.loc[second.total_log.gt(0), "amount"] *= 3.
    pd.testing.assert_series_equal(first.total_log, second.total_log, check_exact=True)
    a, b = build_field(first), build_field(second)
    assert a.loc[39, FIELD] != pytest.approx(b.loc[39, FIELD], abs=1e-8)
    assert a.loc[39, "positive_return_sum10"] == b.loc[39, "positive_return_sum10"]
    assert a.loc[39, "negative_abs_return_sum10"] == b.loc[39, "negative_abs_return_sum10"]


def test_original_members_not_deleted_and_unmatured_records_rejected():
    data = market()
    field = build_field(data)
    field.loc[12, FIELD] = np.nan
    samples = pd.DataFrame({"cycle_id": [1, 1, 2], "origin_index": [10, 12, 14],
        "origin": data.date.iloc[[10, 12, 14]].to_numpy(), "exit_index": [20, 20, 22],
        "mature_date": data.date.iloc[[20, 20, 22]].to_numpy()})
    assert list(samples.columns) == METADATA
    record = {"training_cycles": [1, 2], "training_cycle_count": 2, "training_rows": 3,
              "fit_index": 25, "fit_origin": str(data.date.iloc[25].date()), "model": {"saved": True}}
    result = member_support(samples, [record], field).iloc[0]
    assert result.c03_available_rows == 2 and result.c03_missing_rows == 1
    assert result.original_training_rows == 3 and not result.all_original_members_supported
    future = copy.deepcopy(record)
    future["fit_index"] = 21
    with pytest.raises(ValueError): member_support(samples, [future], field)
    missing_member = samples.iloc[:2].copy()
    with pytest.raises(ValueError): member_support(missing_member, [record], field)
