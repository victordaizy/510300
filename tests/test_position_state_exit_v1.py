"""检验三项持仓输入、真正重拟合、成熟时钟及实际成交。"""
import numpy as np
import pandas as pd
import pytest

from research.position_state_exit_inputs_v1 import FEATURES, position_values, fit_position_state, position_prediction, PositionStateExitController
from research.learned_cycle_exit_v1 import training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def stored(value=-.01, t=0, latest=None):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE",
        "model": {"kind": "POSITION_STATE_RIDGE", "features": FEATURES.copy(), "mean": [0.] * 3, "scale": [1.] * 3,
            "coefficients": [0.] * 3, "feature_clip": 5., "intercept": value}}


def test_actual_holding_state_includes_cost_and_confirmed_distribution():
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100050., "mode": 1}
    values = position_values(4, cycle, 110000. + 1200., 115000.)
    np.testing.assert_allclose(values, [np.log(5), 111200. / 100050. - 1, 111200. / 115000. - 1], atol=0, rtol=0)


def test_ignored_market_columns_do_not_change_fitting_or_prediction():
    rng = np.random.default_rng(73)
    rows = pd.DataFrame(rng.normal(size=(40, 3)), columns=FEATURES)
    rows["target"] = .02 * rows.cycle_return - .01 * rows.cycle_drawdown + .003
    rows["sample_weight"] = np.repeat([1. / 10, 1. / 30], [10, 30])
    cfg = {"ridge_alpha": 1., "feature_clip": 5.}
    base = fit_position_state(rows, cfg)
    for name in ["mom5", "mom20", "sma120", "vol20", "entry_mode"]:
        rows[name] = np.nan
    again = fit_position_state(rows, cfg)
    assert base == again and len(base["coefficients"]) == 3
    assert abs(base["coefficients"][1]) > 0
    assert position_prediction(base, np.zeros(3)) == position_prediction(again, np.zeros(3))


def test_constant_column_and_weighted_response_are_finite():
    rows = pd.DataFrame({FEATURES[0]: [1.] * 12, FEATURES[1]: np.arange(12) / 100., FEATURES[2]: [0.] * 12,
        "target": [.03] * 12, "sample_weight": [1. / 12] * 12})
    model = fit_position_state(rows, {"ridge_alpha": 1., "feature_clip": 5.})
    assert model["scale"][0] == model["scale"][2] == 1.
    assert position_prediction(model, np.array([1., .5, 0.])) == pytest.approx(.03)


def test_missing_unused_prices_do_not_prevent_mature_holding_prediction():
    data = fixture()[0].drop(columns=["mom5", "mom20", "sma120", "vol20"])
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000.}
    controller = PositionStateExitController(data, [stored()])
    assert controller(1, cycle, 100000., 100000.)["learning_status"] == "PREDICTION_AVAILABLE"
    assert controller(2, cycle, 100000., 100000.)["learned_exit_requested"]


def test_missing_used_input_and_zero_prediction_reset_confirmation():
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000.}
    controller = PositionStateExitController(data, [stored(), stored(0., t=3), stored(t=4)])
    assert controller(1, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    missing = controller(2, cycle, np.nan, 100000.)
    assert missing["continuation_prediction"] is None and missing["negative_confirmation_count"] == 0
    assert controller(3, cycle, 100000., 100000.)["negative_confirmation_count"] == 0
    assert controller(4, cycle, 100000., 100000.)["negative_confirmation_count"] == 1


def test_whole_cycle_maturity_precedes_original_training_window():
    rows = pd.DataFrame({"cycle_id": [1, 1, 2, 2, 3], "origin_index": [0, 1, 3, 4, 5], "exit_index": [2, 2, 5, 5, 8]})
    chosen, ids = training_rows(rows, 4, {"recent_cycles": 20})
    assert ids == [1] and chosen.origin_index.tolist() == [0, 1]
    np.testing.assert_allclose(chosen.sample_weight, [.5, .5])


def test_future_models_and_future_maturity_cannot_be_used():
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000.}
    assert PositionStateExitController(data, [stored(t=5)])(2, cycle, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期或模型"):
        PositionStateExitController(data, [stored(latest=5)])(2, cycle, 100000., 100000.)


def test_blocked_exit_stays_pending_after_prediction_turns_positive_and_reentry_needs_reset():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    controller = PositionStateExitController(args[0], [stored(), stored(.03, t=3)])
    ledger, decisions, cycles = simulate_rearmed_exit(*args, controller)
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    assert ledger.accounting_error.abs().max() < 1e-7


def test_no_model_preserves_original_exit_and_missing_prediction():
    args = fixture()
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, PositionStateExitController(args[0], []))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].iloc[0] == args[0].date.iloc[3]


def test_new_actual_cycle_resets_learning_confirmation():
    data = fixture()[0]
    controller = PositionStateExitController(data, [stored()])
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000.}
    assert controller(1, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    assert controller(2, cycle, 100000., 100000.)["negative_confirmation_count"] == 2
    cycle.update(cycle_id=2, entry_index=4)
    assert controller(4, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
