"""概率目标、成熟时间、边界和受阻退出的必要测试。"""
import numpy as np
import pandas as pd
import pytest
from research.directional_continuation_inputs_v1 import FEATURES, fit_directional_continuation, directional_probability, DirectionalExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def stored(value=-.5, t=0, latest=None):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE", "model": {
        "kind": "DIRECTIONAL_CONTINUATION_LOGISTIC", "features": FEATURES.copy(), "mean": [0.]*8,
        "scale": [1.]*8, "coefficients": [0.]*8, "feature_clip": 5., "intercept": value}}


def test_weighted_frequency_and_zero_target_class():
    rows = pd.DataFrame(np.zeros((4, 8)), columns=FEATURES)
    rows["target"], rows["sample_weight"] = [1., 1., 0., -1.], [.4, .3, .1, .2]
    model = fit_directional_continuation(rows, {"feature_clip": 5., "logistic_C": 1.})
    assert directional_probability(model, np.zeros(8))[0] == pytest.approx(.7, abs=1e-7)
    assert model["zero_target_rows"] == 1 and model["positive_rows"] == 2


def test_target_magnitude_does_not_change_direction_model():
    rng = np.random.default_rng(79)
    rows = pd.DataFrame(rng.normal(size=(40, 8)), columns=FEATURES)
    rows["target"], rows["sample_weight"] = np.where(rows.cycle_return > 0, .01, -.02), .1
    first = fit_directional_continuation(rows, {"feature_clip": 5., "logistic_C": 1.})
    rows["target"] *= np.where(rows.target > 0, 1000, 100)
    assert first == fit_directional_continuation(rows, {"feature_clip": 5., "logistic_C": 1.})


def test_single_class_has_no_probability_model():
    rows = pd.DataFrame(np.zeros((10, 8)), columns=FEATURES)
    rows["target"], rows["sample_weight"] = 0., .1
    assert fit_directional_continuation(rows, {"feature_clip": 5., "logistic_C": 1.}) is None


def test_large_log_odds_are_numerically_finite():
    assert directional_probability(stored(1000)["model"], np.zeros(8)) == (1., 1000.)
    assert directional_probability(stored(-1000)["model"], np.zeros(8)) == (0., -1000.)


def test_half_probability_missing_and_new_cycle_reset_confirmation():
    data = fixture()[0]
    models = [stored(), stored(0, t=2), {"fit_index": 4, "status": "NO_VIEW_SINGLE_CLASS", "model": None}, stored(t=5)]
    c = DirectionalExitController(data, models)
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    records = [c(t, cycle, 100000., 100000.) for t in range(1, 7)]
    assert [x["negative_confirmation_count"] for x in records] == [1, 0, 0, 0, 1, 2]
    assert records[3]["continuation_probability"] is None
    assert all(x["continuation_prediction"] is None for x in records)
    cycle.update(cycle_id=2, entry_index=7)
    assert c(7, cycle, 100000., 100000.)["negative_confirmation_count"] == 1


def test_future_models_and_unfinished_cycles_cannot_be_used():
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    assert DirectionalExitController(data, [stored(t=5)])(2, cycle, 100000., 100000.)["continuation_probability"] is None
    with pytest.raises(ValueError, match="未来周期"):
        DirectionalExitController(data, [stored(latest=5)])(2, cycle, 100000., 100000.)


def test_next_open_blocked_request_survives_probability_recovery():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    c = DirectionalExitController(args[0], [stored(), stored(.8, t=3)])
    ledger, decisions, cycles = simulate_rearmed_exit(*args, c)
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    assert ledger.accounting_error.abs().max() < 1e-7
