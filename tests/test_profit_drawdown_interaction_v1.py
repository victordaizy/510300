"""检查交互定义、原周期成熟性、九项拟合及真实退出。"""
import numpy as np
import pandas as pd
import pytest
from research.profit_drawdown_interaction_inputs_v1 import FEATURES, INTERACTION, interaction_value, attach_interaction, interaction_values, fit_interaction_exit, interaction_prediction, InteractionExitController
from research.learned_cycle_exit_v1 import training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def stored(value=-.01, t=0, latest=None):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE",
        "model": {"kind": "PROFIT_DRAWDOWN_INTERACTION_RIDGE", "features": FEATURES.copy(), "mean": [0.]*9, "scale": [1.]*9,
            "coefficients": [0.]*9, "intercept": value, "feature_clip": 5.}}


def test_profit_loss_zero_and_missing_are_distinct_and_training_matches_actual_state():
    np.testing.assert_allclose(interaction_value([.1, -.1, 0., np.nan, -.1], [-.2, -.2, -.2, -.2, np.nan]), [-.02, 0., 0., np.nan, np.nan], equal_nan=True)
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100., "mode": 1}
    values = interaction_values(data, 3, cycle, 110., 125.)
    row = pd.DataFrame([dict(zip(FEATURES[:-1], values[:-1]))])
    assert attach_interaction(row)[INTERACTION].iloc[0] == pytest.approx(values[-1])
    assert values[-1] == pytest.approx(.1*(110/125-1))


def test_mature_membership_all_rows_and_future_inputs_cannot_change_past():
    s = pd.DataFrame({"cycle_id": [1, 1, 2], "origin_index": [10, 20, 25], "exit_index": [21, 21, 50], "cycle_return": [.1, -.1, .5], "cycle_drawdown": [-.02, -.1, -.05]})
    extended = attach_interaction(s)
    pd.testing.assert_frame_equal(extended.drop(columns=INTERACTION), s)
    chosen, ids = training_rows(extended, 30, {"recent_cycles": 20})
    assert ids == [1] and len(chosen) == 2
    np.testing.assert_allclose(chosen.sample_weight, [.5, .5])
    changed = s.copy()
    changed.loc[2, "cycle_return"] = 100.
    other, _ = training_rows(attach_interaction(changed), 30, {"recent_cycles": 20})
    pd.testing.assert_frame_equal(chosen, other)


def test_weighted_ninth_feature_solution_matches_penalized_normal_equation():
    rng = np.random.default_rng(102)
    rows = pd.DataFrame(rng.normal(size=(80, 8)), columns=FEATURES[:-1])
    rows["entry_mode"] = 1.
    rows["cycle_drawdown"] = -rows.cycle_drawdown.abs()
    rows = attach_interaction(rows)
    rows["target"] = .04*rows[INTERACTION]+.02*rows.cycle_return
    rows["sample_weight"] = np.repeat([1/20, 1/60], [20, 60])
    model = fit_interaction_exit(rows, {"feature_clip": 5., "ridge_alpha": 1.})
    x = np.clip((rows[FEATURES].to_numpy()-model["mean"])/model["scale"], -5, 5)
    a = np.c_[np.ones(len(x)), x]
    w = rows.sample_weight.to_numpy()
    solution = np.linalg.solve(a.T@(w[:, None]*a)+np.diag([0.]+[1.]*9), a.T@(w*rows.target.to_numpy()))
    np.testing.assert_allclose([model["intercept"]]+model["coefficients"], solution, atol=1e-12, rtol=0)
    assert model["scale"][3] == 1. and abs(model["coefficients"][-1]) > .001
    v = np.zeros(9)
    original = interaction_prediction(model, v)
    v[-1] = -.1
    assert interaction_prediction(model, v) != original


def test_missing_current_state_future_models_and_maturity():
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    controller = InteractionExitController(data, [stored()])
    assert controller(1, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    missing = controller(2, cycle, np.nan, 100000.)
    assert missing["continuation_prediction"] is None and missing["negative_confirmation_count"] == 0 and np.isnan(missing[INTERACTION])
    assert InteractionExitController(data, [stored(t=5)])(3, cycle, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        InteractionExitController(data, [stored(latest=5)])(3, cycle, 100000., 100000.)


def test_actual_exit_remains_locked_after_blocked_fill_and_positive_model():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, InteractionExitController(args[0], [stored(), stored(.03, t=3)]))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    assert ledger.accounting_error.abs().max() < 1e-7 and ledger.shares.iloc[-1] == 0


def test_no_model_retains_price_exit_and_new_cycle_resets_confirmation():
    args = fixture()
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, InteractionExitController(args[0], []))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].iloc[0] == args[0].date.iloc[3]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    controller = InteractionExitController(args[0], [stored()])
    assert controller(1, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    assert controller(2, cycle, 100000., 100000.)["negative_confirmation_count"] == 2
    cycle.update(cycle_id=2, entry_index=4)
    assert controller(4, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
