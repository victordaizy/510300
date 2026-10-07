"""验证退出贡献比较中的时钟、模型锁定、股息方向和再入场状态。"""
import numpy as np
import pandas as pd
import pytest

from research.point_entry_exit_contribution_v1 import (
    MODEL_FEATURES, evaluate_point, make_features, return_values, selected_model, sequential)


def frame(n=12):
    d = pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=n), "open": np.full(n, 4.), "close": np.full(n, 4.),
                      "high": np.full(n, 4.1), "low": np.full(n, 3.9), "volume": np.full(n, 100000), "dividend": np.zeros(n),
                      "mom5": np.zeros(n), "mom20": np.zeros(n), "sma120": np.zeros(n), "vol20": np.full(n, .2),
                      "entry_long": np.ones(n, bool), "entry_short": np.zeros(n, bool),
                      "exit_long": np.zeros(n, bool), "exit_short": np.zeros(n, bool), "feature_valid": np.ones(n, bool)})
    div = pd.DataFrame({"record_date": pd.to_datetime([]), "ex_date": pd.to_datetime([]), "cash_dividend_per_share": []})
    return d, div


def model(index, intercept):
    return {"fit_index": index, "fit_origin": "2020-01-02", "latest_exit_index": 0, "status": "FIT_COMPLETE",
            "model": {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": MODEL_FEATURES,
                      "mean": [0.]*8, "scale": [1.]*8, "coefficients": [0.]*8,
                      "intercept": intercept, "feature_clip": 5.}}


def test_training_exit_requires_two_closes_then_next_open():
    d, div = frame()
    result, _ = evaluate_point(d, div, [model(0, -.01)], 1, 1, True)
    assert result["exit_idx"] == 3
    assert result["holding_sessions"] == 2
    assert result["exit_reasons"] == "MODEL_NEGATIVE_TWO_CLOSES"


def test_entry_model_is_locked_and_missing_entry_model_is_not_backfilled():
    d, div = frame()
    result, _ = evaluate_point(d, div, [model(0, .01), model(3, -.01)], 1, 1, True)
    assert result["status"] == "RIGHT_CENSORED"
    result, _ = evaluate_point(d, div, [model(3, -.01)], 1, 1, True)
    assert result["status"] == "RIGHT_CENSORED"
    assert not result["model_available_at_entry"]


def test_pending_protection_cannot_be_cancelled_by_later_recovery():
    d, div = frame()
    d.loc[1, "close"] = 3.72
    d.loc[2, "open"] = round(3.72*.9, 3)
    d.loc[2, "close"] = 4.
    result, _ = evaluate_point(d, div, [], 1, 1, False)
    assert result["exit_idx"] == 3
    assert result["blocked_exit_sessions"] == 1
    assert "LOSS_6_PERCENT" in result["exit_reasons"]


def test_short_reference_debits_dividend_and_pays_two_cost_sides():
    without = return_values(4., 3.8, 0., -1, 25000)
    with_dividend = return_values(4., 3.8, .08, -1, 25000)
    assert np.isclose(without["point_net_return"]-with_dividend["point_net_return"], .02)
    assert with_dividend["point_net_return"] < with_dividend["point_gross_return"]


def test_reentry_requires_condition_reset_while_flat_and_two_day_wait():
    d, div = frame(16)
    models = [model(0, -.01)]
    d.loc[4, "entry_long"] = False
    points, _, _ = sequential(d, div, models, "LONG_MODEL", d.date.iloc[1])
    assert list(points.entry_idx) == [1, 6]
    assert list(points.exit_idx) == [3, 8]


def test_do_not_apply_long_model_to_short_or_train_on_future_exit():
    d, div = frame()
    with pytest.raises(ValueError):
        evaluate_point(d, div, [model(0, -.01)], 1, -1, True)
    bad = model(0, -.01)
    bad["latest_exit_index"] = 10
    with pytest.raises(ValueError):
        selected_model([bad], 1)


def test_daily_signal_prefix_unchanged_by_future_prices():
    d, div = frame(310)
    rng = np.random.default_rng(12)
    d["close"] = 4 * np.cumprod(1+rng.normal(.0002, .009, len(d)))
    d["open"] = d.close * np.exp(rng.normal(0, .004, len(d)))
    full = make_features(d, div)
    prefix = make_features(d.iloc[:260], div)
    pd.testing.assert_frame_equal(full.iloc[:260], prefix)
