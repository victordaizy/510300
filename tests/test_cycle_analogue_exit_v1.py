"""检验不同周期邻居限额、平局、成熟时点和实际退出。"""
import json
import numpy as np
import pandas as pd
import pytest
from research.cycle_analogue_exit_inputs_v1 import FEATURES, fit_cycle_analogue, analogue_prediction, CycleAnalogueExitController
from research.learned_cycle_exit_v1 import training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture

CFG = {"feature_clip": 5., "distinct_cycle_neighbors": 5}


def samples():
    rows = []
    for cycle in range(1, 7):
        for t in range(10 if cycle == 1 else 2):
            rows.append({**dict.fromkeys(FEATURES, 0.), "cycle_id": cycle, "origin_index": 20*cycle+t,
                         "exit_index": 20*cycle+15, "target": cycle/100, "sample_weight": 1/(10 if cycle == 1 else 2)})
    return pd.DataFrame(rows)


def constant(value=-.01, t=0, latest=None):
    rows = samples()
    rows.target = value
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE", "model": fit_cycle_analogue(rows, CFG)}


def cycle(quantity=10000, cost=100000., number=1, entry=1):
    return {"cycle_id": number, "entry_index": entry, "entry_quantity": quantity, "entry_cost_cny": cost, "mode": 1}


def test_long_cycle_cannot_fill_five_neighbors_and_exact_ties_ignore_target_or_row_order():
    rows = samples()
    value, detail = analogue_prediction(fit_cycle_analogue(rows, CFG), np.zeros(8))
    assert value == pytest.approx(.03)
    assert json.loads(detail["analogue_cycle_ids"]) == [1, 2, 3, 4, 5]
    assert json.loads(detail["analogue_origin_indices"]) == [20, 40, 60, 80, 100]
    shuffled = rows.sample(frac=1, random_state=4)
    assert fit_cycle_analogue(shuffled, CFG) == fit_cycle_analogue(rows, CFG)
    rows.loc[rows.cycle_id.eq(6), "target"] = -1000.
    again, after = analogue_prediction(fit_cycle_analogue(rows, CFG), np.zeros(8))
    assert again == value and after["analogue_cycle_ids"] == detail["analogue_cycle_ids"]


def test_standardization_gives_each_cycle_equal_weight_and_nearest_representative_is_selected():
    rows = samples()
    rows[FEATURES[0]] = rows.cycle_id.astype(float)
    rows.loc[rows.origin_index.eq(21), FEATURES[0]] = 0.
    model = fit_cycle_analogue(rows, CFG)
    x, w = rows[FEATURES].to_numpy(), rows.sample_weight.to_numpy()
    expected_mean = sum(rows[rows.cycle_id.eq(c)][FEATURES].mean().to_numpy() for c in range(1, 7))/6
    np.testing.assert_allclose(model["mean"], expected_mean, atol=1e-12)
    assert model["scale"][1:] == [1.]*7
    query = np.zeros(8)
    value, detail = analogue_prediction(model, query)
    assert json.loads(detail["analogue_origin_indices"])[0] == 21
    assert value == pytest.approx(.03)
    far, checked = analogue_prediction(model, np.full(8, 100000.))
    assert all(d >= 0 and np.isfinite(d) for d in json.loads(checked["analogue_squared_distances"]))


def test_unfinished_cycle_and_future_perturbation_never_change_current_training():
    rows = samples()
    current, ids = training_rows(rows, 115, {"recent_cycles": 20})
    before = fit_cycle_analogue(current, CFG)
    rows.loc[rows.cycle_id.eq(6), [FEATURES[0], "target"]] = 10000.
    after, repeated = training_rows(rows, 115, {"recent_cycles": 20})
    assert ids == repeated == [1, 2, 3, 4, 5]
    assert fit_cycle_analogue(after, CFG) == before
    current.loc[current.index[0], FEATURES[1]] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        fit_cycle_analogue(current, CFG)


def test_current_market_prediction_is_invariant_to_sunk_fees_and_future_price_rows():
    data = fixture()[0]
    changed = data.copy()
    changed.loc[5:, ["open", "close", "mom5"]] = 1000.
    a, b = CycleAnalogueExitController(data, [constant()]), CycleAnalogueExitController(changed, [constant()])
    ca, cb = cycle(), cycle(quantity=9000, cost=93000.)
    for t in range(1, 4):
        left, right = a(t, ca, 101000., 102000.), b(t, cb, 90900., 93000.)
        for key in FEATURES+["continuation_prediction"]:
            assert left[key] == pytest.approx(right[key], abs=1e-12)
        assert left["analogue_cycle_ids"] == right["analogue_cycle_ids"]


def test_missing_path_future_model_and_new_cycle_reset_negative_confirmation():
    data = fixture()[0]
    controller = CycleAnalogueExitController(data, [constant()])
    c = cycle()
    assert controller(1, c, 100000., 100000.)["negative_confirmation_count"] == 1
    unknown = controller(2, c, np.nan, 100000.)
    assert unknown["continuation_prediction"] is None and unknown["negative_confirmation_count"] == 0
    assert controller(3, c, 100000., 100000.)["continuation_prediction"] is None
    assert controller(4, cycle(number=2, entry=4), 100000., 100000.)["negative_confirmation_count"] == 1
    assert CycleAnalogueExitController(data, [constant(t=5)])(1, c, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        CycleAnalogueExitController(data, [constant(latest=5)])(1, c, 100000., 100000.)


def test_real_next_open_exit_remains_locked_and_reentry_preserves_original_waiting():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    controller = CycleAnalogueExitController(args[0], [constant(), constant(.03, t=3)])
    ledger, decisions, cycles = simulate_rearmed_exit(*args, controller)
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    assert cycles.holding_intervals.ge(1).all() and ledger.accounting_error.abs().max() < 1e-6


def test_true_fee_risk_exit_is_independent_of_positive_market_analogue_prediction():
    args = fixture()
    args[3]["commission"] = .1
    args[6]["modes"][1]["loss"] = .06
    ledger, decisions, cycles = simulate_rearmed_exit(*args, CycleAnalogueExitController(args[0], [constant(.1)]))
    assert "固定止损" in cycles.iloc[0].exit_reasons and not decisions.iloc[1].learned_exit_requested
    assert decisions.iloc[1].market_cycle_return == pytest.approx(0.) and decisions.iloc[1].account_cycle_return < -.06
    assert ledger.commission.sum() > 0 and not ledger.terminal_unliquidated.iloc[-1]
