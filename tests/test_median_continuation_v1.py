"""核对中位数目标、整周期权重、成熟边界和实际退出。"""
import numpy as np
import pandas as pd
import pytest

from research.median_continuation_inputs_v1 import fit_median, predict_median, MedianExitController
from research.learned_cycle_exit_v1 import FEATURES, training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


def model(value, t=0, latest=None):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE", "model": {
        "kind": "LINEAR_CONDITIONAL_MEDIAN", "mean": [0.] * 8, "scale": [1.] * 8, "coefficients": [0.] * 8, "feature_clip": 5., "intercept": value}}


def fixture():
    dates = pd.bdate_range("2020-01-01", periods=10)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.,
        "mom5": .01, "mom20": .02, "sma120": .03, "vol20": .15})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    rule = {"entry": np.ones(10, dtype=int), "exit": {1: np.zeros(10, dtype=bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    return [data, div, cfg, cost, str(dates[1].date()), rule, spec]


def test_weighted_median_is_not_the_outlier_influenced_mean():
    rows = pd.DataFrame(np.zeros((5, 8)), columns=FEATURES)
    rows["target"], rows["sample_weight"] = [-.02, .01, .03, .04, 100.], 1.
    a = fit_median(rows, {"feature_clip": 5.})
    assert predict_median(a, np.zeros(8)) == pytest.approx(.03)
    rows.loc[4, "target"] = 10000.
    b = fit_median(rows, {"feature_clip": 5.})
    assert predict_median(b, np.zeros(8)) == pytest.approx(.03)


def test_whole_cycle_equal_weight_and_future_cycle_exclusion_change_the_median_correctly():
    rows = pd.DataFrame(np.zeros((8, 8)), columns=FEATURES)
    rows["cycle_id"], rows["origin_index"] = [1, 1, 1, 1, 1, 2, 3, 4], np.arange(8)
    rows["exit_index"], rows["target"] = [10, 10, 10, 10, 10, 12, 14, 100], [-.5] * 5 + [.03, .04, 999.]
    selected, ids = training_rows(rows, 14, {"recent_cycles": 20})
    assert ids == [1, 2, 3] and set(selected.cycle_id) == {1, 2, 3}
    np.testing.assert_allclose(selected.groupby("cycle_id").sample_weight.sum(), 1.)
    a = fit_median(selected, {"feature_clip": 5.})
    assert predict_median(a, np.zeros(8)) == pytest.approx(.03)


def test_saved_coefficients_reproduce_a_known_linear_median():
    rows = pd.DataFrame(np.zeros((21, 8)), columns=FEATURES)
    rows[FEATURES[0]] = np.linspace(-2, 2, 21)
    rows["target"], rows["sample_weight"] = .02 + .03 * rows[FEATURES[0]], 1.
    stored = fit_median(rows, {"feature_clip": 5.})
    actual = np.array([predict_median(stored, v) for v in rows[FEATURES].to_numpy()])
    np.testing.assert_allclose(actual, rows.target, atol=1e-10)
    assert stored["training_weighted_absolute_error"] < 1e-10


def test_negative_confirmation_resets_at_zero_missing_and_new_cycle():
    data = fixture()[0]
    models = [model(-.02), model(0., 2), {"fit_index": 4, "latest_exit_index": None, "status": "NO_VIEW_MODEL_FIT_FAILED", "model": None}, model(-.03, 5)]
    controller = MedianExitController(data, models)
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    counts = [controller(t, cycle, 100000., 100000.)["negative_confirmation_count"] for t in range(1, 7)]
    assert counts == [1, 0, 0, 0, 1, 2]
    cycle.update(cycle_id=2, entry_index=7)
    assert controller(7, cycle, 100000., 100000.)["negative_confirmation_count"] == 1


def test_future_model_is_not_used_and_future_training_exit_is_rejected():
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    controller = MedianExitController(data, [model(-.5, 5)])
    assert controller(3, cycle, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        MedianExitController(data, [model(-.5, 0, latest=5)])(3, cycle, 100000., 100000.)


def test_blocked_exit_persists_and_new_entry_needs_signal_reset():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    controller = MedianExitController(args[0], [model(-.02), model(.03, 3)])
    ledger, decisions, cycles = simulate_rearmed_exit(*args, controller)
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    positive = decisions[decisions.origin_index.eq(3)].iloc[0]
    assert positive.continuation_prediction > 0 and positive.requested_quantity < 0
    assert ledger.shares.iloc[-1] == 0 and ledger.accounting_error.abs().max() < 1e-7
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0


def test_no_model_follows_original_price_exit_without_zero_prediction():
    args = fixture()
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, MedianExitController(args[0], []))
    assert decisions.continuation_prediction.dropna().empty
    assert "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].iloc[0] == args[0].date.iloc[3]


@pytest.mark.parametrize("bad_weight", [0., -1., np.nan])
def test_invalid_weights_are_rejected(bad_weight):
    rows = pd.DataFrame(np.zeros((5, 8)), columns=FEATURES)
    rows["target"], rows["sample_weight"] = .03, 1.
    rows.loc[0, "sample_weight"] = bad_weight
    with pytest.raises(ValueError, match="权重"):
        fit_median(rows, {"feature_clip": 5.})
