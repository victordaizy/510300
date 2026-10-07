"""检验入场前来源钟、价格单位、完整性及周期截距的必要反例。"""
import numpy as np
import pandas as pd
import pytest

from research.point_p03_entry_atr_exit_prediction_v1 import FIELD, FEATURES, field_values, fit_increment


def example():
    dates = pd.bdate_range("2020-01-02", periods=40)
    close = np.arange(40, dtype=float)*.1+10
    data = pd.DataFrame({"date": dates, "symbol": "510300.SH", "open": close,
                         "high": close+.2, "low": close-.2, "close": close})
    cycles = pd.DataFrame({"cycle_id": [1], "entry_index": [20]})
    states = pd.DataFrame({"cycle_id": [1, 1], "origin_index": [20, 25],
                           "origin": dates[[20, 25]], "cycle_drawdown": [-.01, -.02],
                           "log_holding_days": np.log1p([1, 6])})
    return data, cycles, states


def test_price_unit_is_invariant():
    data, cycles, states = example()
    _, first = field_values(data, cycles, states)
    scaled = data.copy()
    scaled[["open", "high", "low", "close"]] *= 100
    _, second = field_values(scaled, cycles, states)
    np.testing.assert_allclose(first[FIELD], second[FIELD], atol=1e-12, rtol=0)


def test_entry_day_high_low_and_future_do_not_change_frozen_atr():
    data, cycles, states = example()
    _, first = field_values(data, cycles, states)
    changed = data.copy()
    changed.loc[20:, "high"] += 50
    changed.loc[20:, "low"] = 1.
    _, second = field_values(changed, cycles, states)
    np.testing.assert_array_equal(first.frozen_entry_atr14, second.frozen_entry_atr14)
    np.testing.assert_array_equal(first[FIELD], second[FIELD])
    assert second.atr_latest_source_index.eq(19).all()


def test_no_pre_entry_seed_preserves_unknown():
    data, cycles, states = example()
    cycles.loc[0, "entry_index"] = 5
    states.log_holding_days = np.log1p(states.origin_index-5+1)
    _, values = field_values(data, cycles, states)
    assert values[FIELD].isna().all()
    assert values.field_available.eq(False).all()


def test_zero_atr_is_unknown_without_epsilon():
    data, cycles, states = example()
    data[["open", "high", "low", "close"]] = 10.
    _, values = field_values(data, cycles, states)
    assert values.frozen_entry_atr14.eq(0).all()
    assert values[FIELD].isna().all()


def test_future_entry_is_rejected():
    data, cycles, states = example()
    cycles.loc[0, "entry_index"] = 30
    with pytest.raises(ValueError):
        field_values(data, cycles, states)


def training():
    rng = np.random.default_rng(302014)
    rows = pd.DataFrame(rng.normal(size=(12, 9)), columns=FEATURES)
    rows["cycle_id"] = np.repeat([1, 2, 3], 4)
    rows["sample_weight"] = .25
    rows["target"] = rows.cycle_drawdown*.03+rows[FIELD]*.01+rows.cycle_id*.1
    return rows


def test_cycle_target_level_shift_does_not_change_slopes():
    rows = training()
    cfg = {"ridge_alpha": 1., "feature_clip": 5.}
    first = fit_increment(rows, cfg)
    changed = rows.copy()
    changed.target += changed.cycle_id*.2
    second = fit_increment(changed, cfg)
    np.testing.assert_allclose(first["coefficients"], second["coefficients"], atol=1e-12, rtol=0)
    assert np.isclose(second["intercept"]-first["intercept"], .4, atol=1e-12, rtol=0)


def test_missing_single_training_member_is_not_dropped():
    rows = training()
    rows.loc[0, FIELD] = np.nan
    with pytest.raises(ValueError):
        fit_increment(rows, {"ridge_alpha": 1., "feature_clip": 5.})
