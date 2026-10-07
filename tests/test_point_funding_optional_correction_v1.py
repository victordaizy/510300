"""资金源可用钟、实施日、窗口缺失以及原模型可选分支的必要验证。"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from research.point_funding_optional_correction_v1 import FIELDS, funding_fields
from research.point_optional_residual_model_v1 import design, fit, predict
from research.point_macro_optional_correction_v1 import FIELDS as OLD_FIELDS, fit_optional
from research.learned_cycle_exit_v1 import FEATURES


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=28)
    market = pd.DataFrame({"date": dates, "symbol": "510300.SH"})
    dr = pd.DataFrame({"date": dates, "dr007": 2. + np.arange(len(dates)) * .01})
    policy = pd.DataFrame({"effective_date": [pd.Timestamp("2024-12-01")],
                           "known_at": [pd.Timestamp("2024-12-01T09:20:30+08:00")], "rate": [1.5]})
    return market, dr, policy


def core() -> dict:
    return {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES,
            "mean": [0.] * 8, "scale": [1.] * 8, "coefficients": [.01] + [0.] * 7,
            "intercept": .02, "feature_clip": 5.}


def training() -> pd.DataFrame:
    rows = []
    for cycle in (1, 2, 3):
        for k in range(5):
            row = {name: k * .1 for name in FEATURES}
            row.update(cycle_id=cycle, origin_index=cycle * 10 + k, target=k * .02 - cycle * .01,
                       sample_weight=.2, auxiliary_available=k != 0)
            row[FIELDS[0]] = k * .1 + cycle * .2 if k else np.nan
            row[FIELDS[1]] = (k-cycle) * .05 if k else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def test_exact_prior_five_sessions_and_next_open_clock():
    market, dr, policy = fixture()
    daily, source = funding_fields(market, dr, policy)
    assert not daily.auxiliary_available.iloc[:10].any()
    assert daily.auxiliary_available.iloc[10:].all()
    assert daily.iloc[10][FIELDS[0]] == pytest.approx((dr.dr007.iloc[5:10]-1.5).mean())
    assert daily.iloc[10][FIELDS[1]] == pytest.approx(.05)
    assert source.available_at.iloc[9] == pd.Timestamp(market.date.iloc[10]).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)


def test_source_cutoff_is_not_forward_filled():
    market, dr, policy = fixture()
    daily, source = funding_fields(market, dr.iloc[:16].copy(), policy)
    assert daily.auxiliary_available.iloc[16]
    assert not daily.auxiliary_available.iloc[17:].any()
    assert daily[FIELDS[0]].iloc[17:].isna().all()
    assert daily.dr007.iloc[16:].isna().all()
    assert len(source) == 16


def test_missing_required_date_invalidates_exact_ten_session_window():
    market, dr, policy = fixture()
    daily, _ = funding_fields(market, dr.drop(index=11).copy(), policy)
    assert daily.auxiliary_available.iloc[11]
    assert not daily.auxiliary_available.iloc[12:22].any()
    assert daily.auxiliary_available.iloc[22]
    assert np.isnan(daily.gap_pp_on_stat_day.iloc[11])


def test_current_and_future_rate_changes_do_not_change_current_origin():
    market, dr, policy = fixture()
    baseline, _ = funding_fields(market, dr, policy)
    altered = dr.copy(deep=True)
    altered.loc[15:, "dr007"] += 5.
    future_policy = pd.concat([policy, pd.DataFrame({"effective_date": [market.date.iloc[15]],
                       "known_at": [pd.Timestamp(market.date.iloc[15]).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)], "rate": [1.]})], ignore_index=True)
    after, _ = funding_fields(market, altered, future_policy)
    pd.testing.assert_frame_equal(baseline.loc[:15, FIELDS + ["auxiliary_available"]], after.loc[:15, FIELDS + ["auxiliary_available"]])


def test_future_effective_announced_target_is_not_actual_rate():
    market, dr, policy = fixture()
    extra = pd.DataFrame({"effective_date": [market.date.iloc[17]],
              "known_at": [pd.Timestamp(market.date.iloc[12]).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=10)], "rate": [1.]})
    daily, source = funding_fields(market, dr, pd.concat([policy, extra], ignore_index=True))
    assert source.policy_rate_percent.iloc[12:17].eq(1.5).all()
    assert source.policy_rate_percent.iloc[17:].eq(1.).all()
    assert daily.policy_rate_percent.iloc[16] == 1.5


def test_extra_interbank_date_is_retained_without_changing_stock_day_unit():
    market, dr, policy = fixture()
    baseline, _ = funding_fields(market, dr, policy)
    extras = pd.concat([dr, pd.DataFrame({"date": [pd.Timestamp("2025-01-04")], "dr007": [8.]})], ignore_index=True)
    after, source = funding_fields(market, extras, policy)
    assert len(source) == len(dr) + 1
    assert (~source.is_original_stock_session).sum() == 1
    pd.testing.assert_frame_equal(baseline, after)


def test_funding_optional_model_matches_registered_residual_math():
    rows, original = training(), core()
    model = fit(rows, original, FIELDS)
    old = fit_optional(rows.rename(columns=dict(zip(FIELDS, OLD_FIELDS))), original)
    for name in ["mean", "scale", "clip_center", "coefficients"]:
        np.testing.assert_allclose(model[name], old[name], atol=1e-14, rtol=0)


def test_unknown_raw_values_fallback_exactly_and_core_is_preserved():
    rows, original = training(), core()
    snapshot = copy.deepcopy(original)
    before = rows.copy(deep=True)
    model = fit(rows, original, FIELDS)
    a, b, status = predict(original, model, [.1] * 8, [np.nan, .2], FIELDS)
    assert a == b and status == "EXACT_CORE_FALLBACK"
    assert original == snapshot
    pd.testing.assert_frame_equal(rows, before)
    with pytest.raises(ValueError, match="字段不同"):
        predict(original, model, [.1] * 8, [.1, .2], list(reversed(FIELDS)))


def test_all_unknown_training_keeps_exact_core_and_zero_coefficients():
    rows = training()
    rows[FIELDS] = np.nan
    rows["auxiliary_available"] = False
    model = fit(rows, core(), FIELDS)
    a, b, status = predict(core(), model, [.1] * 8, [.1, .2], FIELDS)
    assert a == b and status == "EXACT_CORE_FALLBACK"
    assert model["coefficients"] == [0., 0.]
    assert model["unknown_training_rows_retained"] == len(rows)


def test_cycle_common_target_shift_has_no_auxiliary_effect():
    rows = training()
    first = fit(rows, core(), FIELDS)
    other = rows.copy(deep=True)
    other["target"] += other.cycle_id * .4
    second = fit(other, core(), FIELDS)
    np.testing.assert_allclose(first["coefficients"], second["coefficients"], atol=1e-14, rtol=0)


def test_unknown_rows_are_retained_in_cycle_fit_and_design_has_no_global_intercept():
    rows = training()
    first = fit(rows, core(), FIELDS)
    other = rows.copy(deep=True)
    other.loc[0, "target"] += .5
    second = fit(other, core(), FIELDS)
    assert not np.allclose(first["coefficients"], second["coefficients"], atol=1e-8, rtol=0)
    assert first["global_intercept"] == 0.
    np.testing.assert_allclose(np.average(design(rows, first), axis=0, weights=rows.sample_weight), [0., 0.], atol=1e-14, rtol=0)
