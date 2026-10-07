"""验证宏观可选分支的时钟、缺失、原样本和周期固定截距。"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from research.learned_cycle_exit_v1 import FEATURES
from research.point_macro_optional_correction_v1 import (
    FIELDS, align_fields, design, fit_optional, monthly_values,
    original_training, predict_optional, training_identity,
)


def core() -> dict:
    return {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES,
            "mean": [0.] * 8, "scale": [1.] * 8, "coefficients": [.02] + [0.] * 7,
            "intercept": .01, "feature_clip": 5.}


def training() -> pd.DataFrame:
    rows = []
    for cycle in (1, 2, 3):
        for k in range(5):
            known = k != 0
            row = {name: float(k) / 10 for name in FEATURES}
            row.update(cycle_id=cycle, origin_index=cycle * 10 + k, exit_index=cycle * 10 + 7,
                       mature_date=pd.Timestamp("2025-01-01") + pd.Timedelta(days=cycle),
                       target=.03 * k - .02 * cycle, sample_weight=.2,
                       auxiliary_available=known)
            row[FIELDS[0]] = float(cycle + k) if known else np.nan
            row[FIELDS[1]] = float(cycle - k) if known else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def reports() -> list[dict]:
    return [{"stat_month": f"2025-{m:02d}", "published_at": f"2025-{m+1:02d}-01T09:30:00+08:00",
             "known_at": f"2025-{m+1:02d}-01T23:59:59+08:00", "method_group": "A",
             FIELDS[0]: float(m)} for m in range(1, 7)]


def states(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"cycle_id": [1] * len(dates), "origin_index": np.arange(len(dates)),
                         "origin": pd.to_datetime(dates)})


def test_publication_day_and_future_prefix_invariance():
    monthly = monthly_values(reports())
    origin = states(["2025-05-01", "2025-05-02"])
    values = align_fields(origin, monthly)
    assert values.stat_month.tolist() == ["2025-03", "2025-04"]
    assert values.auxiliary_available.tolist() == [False, True]
    future = reports()
    future[-1][FIELDS[0]] = 99.
    pd.testing.assert_frame_equal(values, align_fields(origin, monthly_values(future)))
    pd.testing.assert_frame_equal(values, align_fields(origin, monthly.iloc[:4].copy()))


def test_latest_incomparable_release_never_uses_old_valid_report():
    records = reports()
    records[4]["method_group"] = "B"
    monthly = monthly_values(records)
    values = align_fields(states(["2025-05-02", "2025-06-02"]), monthly)
    assert values.auxiliary_available.tolist() == [True, False]
    assert values.stat_month.tolist() == ["2025-04", "2025-05"]
    assert values.iloc[1][FIELDS[0]] == 5.
    assert pd.isna(values.iloc[1][FIELDS[1]])


def test_unknown_auxiliary_is_exact_core_and_raw_missing_stays_missing():
    rows = training()
    rows[FIELDS] = np.nan
    rows["auxiliary_available"] = False
    saved = rows.copy(deep=True)
    original = core()
    model = fit_optional(rows, original)
    a, b, status = predict_optional(original, model, [.1] * 8, [np.nan, np.nan])
    assert a == b and status == "EXACT_CORE_FALLBACK"
    assert model["coefficients"] == [0., 0.]
    assert model["unknown_training_rows_retained"] == len(rows)
    pd.testing.assert_frame_equal(rows, saved)


def test_fit_preserves_original_core_and_every_training_row():
    rows, original = training(), core()
    before, snapshot = rows.copy(deep=True), copy.deepcopy(original)
    model = fit_optional(rows, original)
    assert original == snapshot and model["training_rows"] == len(rows)
    pd.testing.assert_frame_equal(rows, before)
    record = {"fit_index": 100, "fit_origin": "2025-02-01", "training_cycles": [1, 2, 3], "training_rows": len(rows)}
    cfg = {"recent_cycles": 20}
    assert len(original_training(rows, record, cfg)) == len(rows)
    with pytest.raises(ValueError, match="原全部成熟成员"):
        original_training(rows.iloc[1:], record, cfg)


def test_cycle_common_target_shift_does_not_change_auxiliary_coefficients():
    rows = training()
    model = fit_optional(rows, core())
    altered = rows.copy(deep=True)
    altered["target"] += altered.cycle_id.map({1: .5, 2: -.3, 3: .8})
    other = fit_optional(altered, core())
    np.testing.assert_allclose(model["coefficients"], other["coefficients"], atol=1e-14, rtol=0)


def test_unknown_row_target_still_contributes_in_partially_known_cycle():
    rows = training()
    model = fit_optional(rows, core())
    altered = rows.copy(deep=True)
    altered.loc[0, "target"] += .4
    other = fit_optional(altered, core())
    assert not np.allclose(model["coefficients"], other["coefficients"], atol=1e-8, rtol=0)


def test_clipped_design_has_zero_global_mean_and_no_intercept():
    rows = training()
    rows.loc[1, FIELDS[0]] = 1000.
    rows.loc[1, "sample_weight"] = .2
    model = fit_optional(rows, core())
    phi = design(rows, model)
    np.testing.assert_allclose(np.average(phi, axis=0, weights=rows.sample_weight), [0., 0.], atol=1e-14, rtol=0)
    assert model["global_intercept"] == 0.
    assert (phi[~rows.auxiliary_available.to_numpy()] == 0.).all()


def test_training_identity_and_maturity_reject_future_information():
    rows, original = training(), core()
    key = training_identity(rows, original)
    altered = rows.copy(deep=True)
    altered.loc[0, "target"] += .01
    assert key != training_identity(altered, original)
    record = {"fit_index": 100, "fit_origin": "2024-12-31", "training_cycles": [1, 2, 3], "training_rows": len(rows)}
    with pytest.raises(ValueError, match="未成熟"):
        original_training(rows, record, {"recent_cycles": 20})
