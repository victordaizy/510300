"""C04数量准入的确认时钟、金额窗口、缺失及原成员测试。"""
import numpy as np
import pandas as pd
import pytest
from research.point_c04_information_intake_v1 import FIELD, build_field, member_support


def market(prices, amounts=None):
    prices = np.asarray(prices, dtype=float)
    amounts = np.asarray(amounts if amounts is not None else np.arange(1, len(prices) + 1) * 100., dtype=float)
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(prices)),
                         "close": prices, "high": prices + .2, "low": prices - .2,
                         "dividend": np.zeros(len(prices)), "amount": amounts, "volume": amounts / prices,
                         "symbol": "510300.SH", "amount_unit": "CNY", "volume_unit": "share"})


def test_confirmation_day_starts_without_backfilling_and_freezes_previous_five_days():
    data = market([10, 9, 8, 9, 10, 12, 11, 10, 9.5])
    f = build_field(data)
    assert f.loc[:6, FIELD].isna().all()
    assert f.loc[7, "high_index"] == 5 and f.loc[7, "high_confirmation_index"] == 7
    assert f.loc[7, "phase_start_index"] == 7
    assert f.loc[7, FIELD] == pytest.approx(800. / 500.)
    assert f.loc[8, "baseline_amount_cny"] == 500.
    assert f.loc[8, FIELD] == pytest.approx(850. / 500.)


def test_reclaim_ends_phase_and_later_pullback_has_new_baseline():
    data = market([10, 9, 8, 9, 10, 12, 11, 10, 9.5, 10, 12, 11])
    f = build_field(data)
    assert f.loc[10, "field_status"] == "NO_VIEW_NOT_IN_PULLBACK"
    assert np.isnan(f.loc[10, FIELD])
    assert f.loc[11, "phase_start_index"] == 11
    assert f.loc[11, "baseline_amount_cny"] == 900.
    assert f.loc[11, FIELD] == pytest.approx(1200. / 900.)


def test_broken_low_cannot_resume_old_phase_and_initial_or_equal_prices_stay_unknown():
    f = build_field(market([10, 9, 8, 9, 10, 12, 11, 10, 8, 9]))
    assert f.loc[8, "field_status"] == f.loc[9, "field_status"] == "NO_VIEW_UP_SWING_INVALIDATED"
    assert f.loc[8:9, FIELD].isna().all()
    assert build_field(market([10] * 10))[FIELD].isna().all()
    assert build_field(market(range(10, 20)))[FIELD].isna().all()


def test_future_data_cannot_change_prefix_and_known_dividend_preserves_price_clock():
    original = market([10, 9, 8, 9, 10, 12, 11, 10, 9.5, 10, 12, 11])
    pd.testing.assert_frame_equal(build_field(original.iloc[:9]), build_field(original).iloc[:9].reset_index(drop=True))
    corrected = original.copy()
    for name in ["close", "high", "low"]:
        corrected.loc[6:, name] -= .4
    corrected.loc[6, "dividend"] = .4
    corrected["volume"] = corrected.amount / corrected.close
    left, right = build_field(original), build_field(corrected)
    assert np.allclose(left[FIELD], right[FIELD], equal_nan=True)
    assert left.field_status.tolist() == right.field_status.tolist()
    assert left.high_confirmation_index.fillna(-1).tolist() == right.high_confirmation_index.fillna(-1).tolist()


def test_bad_amount_is_not_skipped_and_zero_baseline_or_wrong_units_are_rejected():
    prices = [10, 9, 8, 9, 10, 12, 11, 10, 9.5]
    data = market(prices)
    data.loc[7, "amount"] = np.nan
    f = build_field(data)
    assert f.loc[7:8, FIELD].isna().all()
    assert f.loc[8, "field_status"] == "NO_VIEW_INVALID_PULLBACK_AMOUNT"
    data = market(prices, [100, 200, 0, 0, 0, 0, 0, 800, 900])
    assert build_field(data).loc[7, "field_status"] == "NO_VIEW_NONPOSITIVE_BASELINE_AMOUNT"
    data = market(prices)
    data.loc[3, "amount_unit"] = "万元"
    with pytest.raises(ValueError, match="人民币元"):
        build_field(data)


def test_support_preserves_all_members_and_rejects_future_maturity_without_labels():
    f = build_field(market([10, 9, 8, 9, 10, 12, 11, 10, 9.5]))
    samples = pd.DataFrame({"cycle_id": [1, 1], "origin_index": [5, 7], "origin": [f.loc[5, "date"], f.loc[7, "date"]],
                            "exit_index": [8, 8], "mature_date": [f.loc[8, "date"]] * 2})
    record = {"fit_index": 8, "fit_origin": str(f.loc[8, "date"].date()), "training_cycles": [1],
              "training_cycle_count": 1, "training_rows": 2, "model": {}}
    s = member_support(samples, [record], f).iloc[0]
    assert s.original_training_rows == 2 and s.c04_available_rows == s.c04_missing_rows == 1
    assert not s.all_original_members_supported
    with pytest.raises(ValueError, match="成熟"):
        member_support(samples, [{**record, "fit_index": 7}], f)
    with pytest.raises(ValueError, match="禁止删行"):
        member_support(samples.iloc[1:], [record], f)
