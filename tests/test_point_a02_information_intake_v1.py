"""验证A02事前窗口、完整阶段、真零和原成员保留。"""
import numpy as np
import pandas as pd

from research.point_a02_information_intake_v1 import FIELD, build_field, member_support


def prices():
    data = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=40), "symbol": "510300.SH",
        "close": 10.0, "high": 11.0, "low": 9.0, "dividend": 0.0,
    })
    data.loc[21, ["close", "high", "low"]] = [12.0, 13.0, 9.0]
    data.loc[22, ["close", "high", "low"]] = [11.0, 12.0, 10.0]
    data.loc[23, ["close", "high", "low"]] = [11.0, 12.0, 9.9]
    data.loc[24, ["close", "high", "low"]] = [10.9, 12.0, 10.0]
    return data


def test_prior_high_and_atr_exclude_breakout_day_and_boundaries_are_inclusive():
    field = build_field(prices())
    assert field.loc[21, "breakout_confirmed_today"]
    assert field.loc[22, "latest_prior_breakout_index"] == 21
    assert field.loc[22, "frozen_breakout_upper"] == 11.0
    assert field.loc[22, "frozen_previous_atr20"] == 2.0
    assert field.loc[22, FIELD] == 1.0
    assert field.loc[23, FIELD] == 0.5
    assert field.loc[24, FIELD] == 1 / 3


def test_only_completed_first_to_third_days_are_defined_and_true_zero_is_available():
    data = prices()
    data.loc[22, ["close", "high", "low"]] = [10.0, 11.0, 9.0]
    field = build_field(data)
    assert np.isnan(field.loc[21, FIELD])
    assert field.loc[22, FIELD] == 0.0
    assert field.loc[22, "observed_day_count"] == 1
    assert field.loc[24, "observed_day_count"] == 3
    assert np.isnan(field.loc[25, FIELD])
    assert field.loc[25, "field_status"] == "NO_VIEW_THREE_DAY_PHASE_EXPIRED"


def test_repeated_breakout_becomes_latest_anchor_only_from_next_completed_day():
    data = prices()
    data.loc[22, ["close", "high", "low"]] = [14.0, 15.0, 10.0]
    data.loc[23, ["close", "high", "low"]] = [13.0, 14.0, 12.0]
    field = build_field(data)
    assert field.loc[22, "breakout_confirmed_today"]
    assert field.loc[22, "latest_prior_breakout_index"] == 21
    assert field.loc[23, "latest_prior_breakout_index"] == 22
    assert field.loc[23, "frozen_breakout_upper"] == 13.0
    assert field.loc[23, "observed_day_count"] == 1
    assert field.loc[25, "observed_day_count"] == 3
    assert np.isnan(field.loc[26, FIELD])


def test_future_prices_and_future_cash_dividends_do_not_rewrite_the_prefix():
    data = prices()
    data.loc[28, "dividend"] = 0.8
    data.loc[29:, ["close", "high", "low"]] = [30.0, 35.0, 29.0]
    complete = build_field(data)
    prefix = build_field(data.iloc[:24].copy())
    pd.testing.assert_frame_equal(prefix, complete.iloc[:24].reset_index(drop=True), check_exact=True)


def test_member_support_keeps_missing_rows_and_counts_genuine_zero_as_finite():
    data = prices()
    data.loc[22, ["close", "high", "low"]] = [10.0, 11.0, 9.0]
    field = build_field(data)
    samples = pd.DataFrame({
        "cycle_id": [1, 1, 2], "origin_index": [22, 25, 23],
        "origin": data.date.iloc[[22, 25, 23]].to_numpy(dtype="datetime64[ms]"),
        "exit_index": [26, 26, 30],
        "mature_date": data.date.iloc[[26, 26, 30]].to_numpy(),
    })
    records = [{"training_cycles": [1], "training_cycle_count": 1, "training_rows": 2,
                "fit_index": 28, "fit_origin": str(data.date.iloc[28].date()), "model": {}}]
    support = member_support(samples, records, field)
    assert len(samples) == 3
    assert support.loc[0, "original_training_rows"] == 2
    assert support.loc[0, "a02_available_rows"] == 1
    assert support.loc[0, "a02_missing_rows"] == 1
    assert not support.loc[0, "all_original_members_supported"]
