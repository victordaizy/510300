"""只核对经济权重、评分含义、成熟时间及真实退出。"""
import numpy as np
import pandas as pd
import pytest
from research.economic_regret_inputs_v1 import FEATURES, economic_weights, economic_support, fit_economic_regret, economic_score, EconomicExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def stored(value=-.5, t=0, latest=None):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE", "model": {
        "kind": "ECONOMIC_REGRET_LOGISTIC", "features": FEATURES.copy(), "mean": [0.]*8,
        "scale": [1.]*8, "coefficients": [0.]*8, "feature_clip": 5., "intercept": value}}


def test_economic_weights_and_global_units_are_exactly_defined():
    target, base = np.array([.1, -.2, .2, 0.]), np.array([.5, .5, 1., 1.])
    result = economic_weights(target, base)
    np.testing.assert_allclose(result, [3/7, 6/7, 12/7, 0.], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result, economic_weights(target*1000, base), rtol=0, atol=1e-14)
    assert result.sum() == pytest.approx(base.sum())


def test_economic_decision_can_differ_from_most_frequent_class():
    rows = pd.DataFrame(np.zeros((4, 8)), columns=FEATURES)
    rows["target"], rows["sample_weight"] = [.01, .01, .01, -.1], 1.
    model = fit_economic_regret(rows, {"feature_clip": 5., "logistic_C": 1.})
    assert economic_score(model, np.zeros(8))[0] == pytest.approx(.03/.13, abs=1e-7)
    assert not model["score_is_calibrated_probability"]


def test_zero_economic_loss_and_weightless_second_class_do_not_make_a_model():
    rows = pd.DataFrame(np.zeros((4, 8)), columns=FEATURES)
    rows["target"], rows["sample_weight"] = 0., 1.
    assert economic_support(rows) == "NO_VIEW_NO_ECONOMIC_DIFFERENCE"
    assert fit_economic_regret(rows, {"feature_clip": 5., "logistic_C": 1.}) is None
    rows.loc[0, "target"] = .1
    assert economic_support(rows) == "NO_VIEW_SINGLE_ECONOMIC_CLASS"
    assert fit_economic_regret(rows, {"feature_clip": 5., "logistic_C": 1.}) is None


def test_standardization_keeps_original_cycle_basis():
    rows = pd.DataFrame(np.zeros((4, 8)), columns=FEATURES)
    rows[FEATURES[0]], rows["target"], rows["sample_weight"] = [1., 2., 3., 9.], [.1, -.2, .5, -.1], [.5, .5, 1., 1.]
    model = fit_economic_regret(rows, {"feature_clip": 5., "logistic_C": 1.})
    assert model["mean"][0] == pytest.approx(np.average(rows[FEATURES[0]], weights=rows.sample_weight))
    assert model["scale"][1] == 1.


def test_confirmation_resets_and_future_models_are_rejected():
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    models = [stored(), stored(0., t=2), {"fit_index": 4, "status": "NO_VIEW_SINGLE_ECONOMIC_CLASS", "model": None}, stored(t=5)]
    controller = EconomicExitController(data, models)
    rows = [controller(t, cycle, 100000., 100000.) for t in range(1, 7)]
    assert [r["negative_confirmation_count"] for r in rows] == [1, 0, 0, 0, 1, 2]
    assert all(r["continuation_probability"] is None and r["continuation_prediction"] is None for r in rows)
    cycle.update(cycle_id=2, entry_index=7)
    assert controller(7, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    assert EconomicExitController(data, [stored(t=9)])(7, cycle, 100000., 100000.)["economic_continuation_score"] is None
    with pytest.raises(ValueError, match="未来周期"):
        EconomicExitController(data, [stored(latest=9)])(7, cycle, 100000., 100000.)


def test_next_open_request_survives_block_and_score_recovery():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, EconomicExitController(args[0], [stored(), stored(.8, t=3)]))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    assert ledger.accounting_error.abs().max() < 1e-7
