"""相对价格来源、过去窗口、成员和缺失回退的必要检查。"""
import numpy as np
import pandas as pd
import pytest
from research.point_relative_price_optional_correction_v1_0_1 import FIELDS, align, build_field, validate_raw, compare_saved_source


def inputs():
    dates = pd.bdate_range("2020-01-01", periods=35)
    stock = pd.DataFrame({"date": dates, "close": 2. + np.arange(35) / 100., "symbol": "510300.SH"})
    comparison = pd.DataFrame({"date": dates, "ts_close": 3. + np.arange(35) / 200., "ts_code": "510500.SH"})
    return stock, comparison


def test_exact_fixed_raw_price_differences():
    stock, other = inputs()
    daily = build_field(stock, other)
    for field, window in zip(FIELDS, [5, 20], strict=True):
        expected = (stock.close.iloc[25] / stock.close.iloc[25-window] - 1.) - (
            other.ts_close.iloc[25] / other.ts_close.iloc[25-window] - 1.)
        assert daily.loc[25, field] == expected


def test_all_original_rows_and_preparation_missing_preserved():
    stock, other = inputs()
    daily = build_field(stock, other)
    assert len(daily) == len(stock)
    assert daily.loc[:4, FIELDS[0]].isna().all()
    assert daily.loc[:19, FIELDS[1]].isna().all()
    assert daily.loc[5:19, FIELDS[0]].notna().all()
    assert not daily.loc[:19, "auxiliary_available"].any()


def test_internal_missing_is_not_endpoint_only_or_filled():
    stock, other = inputs()
    other.loc[12, "ts_close"] = np.nan
    daily = build_field(stock, other)
    assert pd.isna(daily.loc[15, FIELDS[0]])
    assert pd.isna(daily.loc[30, FIELDS[1]])
    assert pd.notna(daily.loc[18, FIELDS[0]])
    assert pd.notna(daily.loc[33, FIELDS[1]])


def test_absent_calendar_day_stays_missing_without_stale_price():
    stock, other = inputs()
    daily = build_field(stock, other.drop(index=12))
    assert pd.isna(daily.loc[12, "original_510500_close"])
    assert pd.isna(daily.loc[15, FIELDS[0]])


def test_true_zero_and_both_signs_are_legal():
    assert validate_raw(np.asarray([[0., 0.], [-.1, .2], [np.nan, .1]])).tolist() == [True, True, False]


def test_infinity_and_nonpositive_quotes_are_rejected():
    stock, other = inputs()
    for value in [0., -1., np.inf]:
        invalid = other.copy()
        invalid.loc[12, "ts_close"] = value
        with pytest.raises(ValueError):
            build_field(stock, invalid)
    with pytest.raises(ValueError):
        validate_raw(np.asarray([[np.inf, .1]]))


def test_asset_identity_cannot_be_substituted():
    stock, other = inputs()
    for code in ["000905.SZ", "510300.SH"]:
        invalid = other.assign(ts_code=code)
        with pytest.raises(ValueError):
            build_field(stock, invalid)


def test_duplicate_or_reordered_original_dates_are_rejected():
    stock, other = inputs()
    for invalid in [pd.concat([other, other.iloc[[-1]]], ignore_index=True), other.iloc[::-1]]:
        with pytest.raises(ValueError):
            build_field(stock, invalid)


def test_future_extension_cannot_change_past_features():
    stock, other = inputs()
    full = build_field(stock, other)
    prefix = build_field(stock.iloc[:27], other)
    pd.testing.assert_frame_equal(prefix, full.iloc[:27].reset_index(drop=True), check_exact=True)
    changed = other.copy()
    changed.loc[28:, "ts_close"] *= 2.
    pd.testing.assert_frame_equal(build_field(stock, changed).iloc[:28], full.iloc[:28], check_exact=True)


def test_alignment_keeps_original_identity_and_available_clock():
    stock, other = inputs()
    daily = build_field(stock, other)
    states = pd.DataFrame({"cycle_id": [2, 4], "origin_index": [8, 25], "origin": stock.date.iloc[[8, 25]].to_numpy()})
    field = align(states, daily)
    assert field.cycle_id.tolist() == [2, 4]
    assert field.origin_index.tolist() == [8, 25]
    assert field.auxiliary_available.tolist() == [False, True]
    assert (field.origin_at - field.known_at).eq(pd.Timedelta(minutes=5)).all()


def test_alignment_rejects_duplicate_identity_wrong_date_or_bounds():
    stock, other = inputs()
    daily = build_field(stock, other)
    state = pd.DataFrame({"cycle_id": [2], "origin_index": [25], "origin": [stock.date.iloc[25]]})
    for invalid in [pd.concat([state, state], ignore_index=True), state.assign(origin=stock.date.iloc[24]),
                    state.assign(origin_index=-1), state.assign(origin_index=35)]:
        with pytest.raises(ValueError):
            align(invalid, daily)


def test_end_of_saved_comparison_does_not_forward_fill():
    stock, other = inputs()
    daily = build_field(stock, other.iloc[:27])
    assert daily.loc[27:, FIELDS].isna().all().all()
    assert not daily.loc[27:, "auxiliary_available"].any()



def scoped_saved_source():
    stock, other = inputs()
    daily = build_field(stock, other)
    saved = daily[["date", "original_510300_close", "original_510500_close"] + FIELDS].copy()
    saved = saved.rename(columns=dict(zip(FIELDS, ["510300相对510500原价格五日变化差", "510300相对510500原价格二十日变化差"], strict=True)))
    columns = ["510300相对510500原价格五日变化差", "510300相对510500原价格二十日变化差"]
    cutoff = stock.date.iloc[24]
    saved.loc[saved.date > cutoff, columns] = np.nan
    return daily, saved, cutoff, columns


def test_other_experiment_scope_mask_stays_separate_from_raw_price_availability():
    daily, saved, cutoff, columns = scoped_saved_source()
    before = saved.copy(deep=True)
    assert compare_saved_source(daily, saved, cutoff) == 25
    assert daily.loc[25:, FIELDS].notna().all().all()
    assert saved.loc[25:, columns].isna().all().all()
    pd.testing.assert_frame_equal(saved, before, check_exact=True)


def test_common_period_mismatch_is_not_allowed_by_scope_repair():
    daily, saved, cutoff, columns = scoped_saved_source()
    saved.loc[23, columns[0]] += .001
    with pytest.raises(AssertionError):
        compare_saved_source(daily, saved, cutoff)


def test_old_outside_scope_mask_cannot_be_relabelled_known():
    daily, saved, cutoff, columns = scoped_saved_source()
    saved.loc[26, columns[0]] = .1
    with pytest.raises(ValueError):
        compare_saved_source(daily, saved, cutoff)
