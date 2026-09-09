"""验证退出学习的成熟时钟、周期权重、分红权益和账户动作。"""
import numpy as np
import pandas as pd
import pytest

from research.learned_cycle_exit_v1 import FEATURES, ExitController, continuation_label, fit_one, predict, training_rows
from research.learned_cycle_exit_account_v1 import simulate_learned_exit
from research.simple_price_entry_exit_v1 import simulate_policy


def fixture_data():
    dates = pd.bdate_range("2020-01-01", periods=7)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.,
                         "mom5": .01, "mom20": .02, "sma120": .03, "vol20": .15})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    config = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.array([1, 0, 0, 0, 1, 0, 0]), "exit": {1: np.zeros(7, dtype=bool)}}
    spec = {"cooldown": 1, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    return data, div, config, cost, str(dates[1].date()), rule, spec


def constant_model(value, fit_index=0):
    return {"fit_index": fit_index, "latest_exit_index": fit_index, "status": "FIT_COMPLETE", "model": {
        "kind": "RIDGE", "mean": [0.] * 8, "scale": [1.] * 8, "coefficients": [0.] * 8,
        "feature_clip": 5., "intercept": value}}


def test_callback_disabled_preserves_complete_original_path():
    args = fixture_data()
    actual = simulate_learned_exit(*args)
    original = simulate_policy(*args)
    for a, b in zip(actual, original):
        pd.testing.assert_frame_equal(a, b)


def test_two_negative_closes_exit_next_open_and_new_cycle_resets():
    args = fixture_data()
    controller = ExitController(args[0], [constant_model(-.02)])
    ledger, decisions, cycles = simulate_learned_exit(*args, controller)
    assert ledger.iloc[0].filled_quantity == 10000
    assert ledger.iloc[1].filled_quantity == 0
    assert ledger.iloc[2].filled_quantity == -10000
    assert "学习条件" in cycles.iloc[0].exit_reasons
    second_entry = decisions[decisions.origin == args[0].date.iloc[5]].iloc[0]
    assert second_entry.negative_confirmation_count == 1
    assert cycles.iloc[0].holding_intervals == 2


def test_missing_model_keeps_original_rules_and_prediction_missing():
    args = fixture_data()
    model = {"fit_index": 0, "latest_exit_index": None, "status": "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", "model": None}
    ledger, decisions, cycles = simulate_learned_exit(*args, ExitController(args[0], [model]))
    assert (ledger.iloc[1:-1].filled_quantity == 0).all()
    assert decisions.continuation_prediction.dropna().empty
    assert not decisions.learned_exit_requested.fillna(False).any()
    assert ledger.iloc[-1].filled_quantity == -10000


def test_blocked_learned_exit_persists_when_prediction_turns_positive():
    args = fixture_data()
    args[0].loc[3, "open"] = 9.
    args[0].loc[4, "open"] = 10.2
    controller = ExitController(args[0], [constant_model(-.02), constant_model(.03, 3)])
    ledger, decisions, cycles = simulate_learned_exit(*args, controller)
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.iloc[3].filled_quantity == -10000
    assert ledger.iloc[3].open_price == pytest.approx(10.2)
    positive = decisions[decisions.origin == args[0].date.iloc[3]].iloc[0]
    assert positive.continuation_prediction > 0
    assert positive.requested_quantity == -10000


def test_future_fit_cannot_control_earlier_decisions():
    args = fixture_data()
    model = constant_model(-.5, 4)
    ledger, decisions, cycles = simulate_learned_exit(*args, ExitController(args[0], [model]))
    early = decisions[decisions.origin_index < 4]
    assert early.continuation_prediction.dropna().empty
    assert (ledger.iloc[1:4].filled_quantity == 0).all()


def test_training_excludes_whole_unfinished_cycles_before_selecting_recent():
    samples = pd.DataFrame([
        {"cycle_id": 1, "origin_index": 1, "exit_index": 4, "target": .1},
        {"cycle_id": 2, "origin_index": 3, "exit_index": 8, "target": .2},
        {"cycle_id": 2, "origin_index": 4, "exit_index": 8, "target": .3},
        {"cycle_id": 3, "origin_index": 7, "exit_index": 15, "target": 99.},
        {"cycle_id": 3, "origin_index": 12, "exit_index": 15, "target": 99.},
    ])
    rows, ids = training_rows(samples, 8, {"recent_cycles": 2})
    assert ids == [1, 2]
    assert set(rows.cycle_id) == {1, 2}
    np.testing.assert_allclose(rows.groupby("cycle_id").sample_weight.sum(), 1.)
    recent, ids = training_rows(samples, 16, {"recent_cycles": 2})
    assert ids == [2, 3]


def test_dividend_uses_record_date_not_late_payment_or_ex_date():
    data = fixture_data()[0]
    data.loc[5, "open"] = 11.
    div = pd.DataFrame([
        {"record_date": data.date.iloc[1], "ex_date": data.date.iloc[3], "payment_date": data.date.iloc[6], "cash_dividend_per_share": .2},
        {"record_date": data.date.iloc[3], "ex_date": data.date.iloc[4], "payment_date": data.date.iloc[6], "cash_dividend_per_share": .4},
        {"record_date": data.date.iloc[5], "ex_date": data.date.iloc[6], "payment_date": data.date.iloc[6], "cash_dividend_per_share": .6},
    ])
    y, extra = continuation_label(data, div, 1000, 2, 5, {"commission": 0., "minimum": 0., "slippage": 0.}, .001)
    assert extra == pytest.approx(400)
    assert y == pytest.approx(.14)


def test_both_selling_costs_are_in_continuation_advantage():
    data, div, _, _, _, _, _ = fixture_data()
    data.loc[5, "open"] = 11.
    cost = {"commission": .001, "minimum": 5., "slippage": .001}
    y, extra = continuation_label(data, div, 1000, 2, 5, cost, .001)
    early_proceeds = 1000 * 9.99 - 9.99
    late_proceeds = 1000 * 10.989 - 10.989
    assert y == pytest.approx((late_proceeds - early_proceeds) / 10000)


@pytest.mark.parametrize("kind", ["RIDGE", "TREE"])
def test_saved_model_predicts_constant_outcome_for_constant_target(kind):
    rng = np.random.default_rng(12)
    rows = pd.DataFrame(rng.normal(size=(60, 8)), columns=FEATURES)
    rows["target"], rows["sample_weight"] = .03, .1
    cfg = {"feature_clip": 5., "ridge_alpha": 1., "tree_max_depth": 2,
           "tree_min_samples_leaf": 20, "tree_min_weight_fraction_leaf": .1, "random_seed": 20260907}
    model = fit_one(rows, kind, cfg)
    assert predict(model, np.zeros(8)) == pytest.approx(.03)
