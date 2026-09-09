"""检验提升模型保存数值、原完整周期成熟边界与真实退出。"""
import json
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits
from research.adaptive_allocation_v1 import model_for
from research.boosted_continuation_exit_inputs_v1 import export_boosted_model, fit_boosted_exit, boosted_prediction, BoostedExitController
from research.learned_cycle_exit_v1 import FEATURES, training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def constant_record(value=-.01, t=0, latest=None):
    model = {"kind": "HGB_CONDITIONAL_CONTINUATION", "features": FEATURES.copy(), "baseline": value, "trees": []}
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE", "model": model}


def training_fixture():
    rng = np.random.default_rng(106)
    rows = pd.DataFrame(rng.normal(size=(400, 8)), columns=FEATURES)
    rows["entry_mode"] = 1.
    rows["target"] = .1*(rows.cycle_return > 0)-.15*(rows.mom5 > .5)+.01*rows.vol20
    rows["sample_weight"] = np.r_[np.full(100, 1/100), np.full(300, 1/300)]
    return rows


def test_full_tree_json_matches_library_at_new_inputs_and_split_equalities():
    rows = training_fixture()
    x, y, w = rows[FEATURES].to_numpy(), rows.target.to_numpy(), rows.sample_weight.to_numpy()
    fitted = model_for("HGB", 20260907)
    with threadpool_limits(limits=1):
        fitted.fit(x, y, sample_weight=w)
    saved = json.loads(json.dumps(export_boosted_model(fitted)))
    rng = np.random.default_rng(107)
    queries = rng.normal(size=(200, 8)).tolist()
    for tree in saved["trees"]:
        for node, leaf in enumerate(tree["is_leaf"]):
            if not leaf:
                row = np.zeros(8)
                row[tree["feature"][node]] = tree["threshold"][node]
                queries.append(row.tolist())
    queries = np.asarray(queries)
    with threadpool_limits(limits=1):
        expected = fitted.predict(queries)
    np.testing.assert_allclose([boosted_prediction(saved, q) for q in queries], expected, atol=1e-12, rtol=0)
    assert saved["iterations"] == 80 and np.std(expected) > .001


def test_complete_cycle_weights_maturity_and_future_sample_invariance():
    samples = pd.DataFrame({"cycle_id": [1, 1, 2, 3], "origin_index": [1, 2, 5, 11], "exit_index": [4, 4, 10, 20], "target": [1., 2., 3., 4.]})
    original, ids = training_rows(samples, 10, {"recent_cycles": 20})
    assert ids == [1, 2]
    np.testing.assert_allclose(original.groupby("cycle_id").sample_weight.sum(), 1.)
    samples.loc[3, "target"] = -999.
    altered, _ = training_rows(samples, 10, {"recent_cycles": 20})
    pd.testing.assert_frame_equal(original, altered)


def test_frozen_parameters_weighted_baseline_and_no_missing_row_deletion():
    rows = training_fixture()
    cfg = {"random_seed": 20260907, "model_parameters": model_for("HGB", 20260907).get_params()}
    saved = fit_boosted_exit(rows, cfg)
    assert saved["baseline"] == pytest.approx(np.average(rows.target, weights=rows.sample_weight))
    assert saved["training_check"]["rows"] == len(rows)
    assert saved["training_check"]["cycle_weighted_training_mse"] < saved["training_check"]["baseline_mse"]
    rows.loc[0, "mom5"] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        fit_boosted_exit(rows, cfg)


def test_unknown_current_state_and_future_model_reset_confirmation():
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    controller = BoostedExitController(data, [constant_record()])
    assert controller(1, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    missing = controller(2, cycle, np.nan, 100000.)
    assert missing["continuation_prediction"] is None and missing["negative_confirmation_count"] == 0
    assert BoostedExitController(data, [constant_record(t=5)])(3, cycle, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        BoostedExitController(data, [constant_record(latest=5)])(3, cycle, 100000., 100000.)


def test_actual_exit_is_next_open_and_remains_locked_when_new_model_turns_positive():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, BoostedExitController(args[0], [constant_record(), constant_record(.03, 3)]))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    assert ledger.accounting_error.abs().max() < 1e-7


def test_no_model_preserves_original_price_exit_and_new_cycle_resets_counter():
    args = fixture()
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, BoostedExitController(args[0], []))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].iloc[0] == args[0].date.iloc[3]
    controller = BoostedExitController(args[0], [constant_record()])
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    controller(1, cycle, 100000., 100000.)
    assert controller(2, cycle, 100000., 100000.)["negative_confirmation_count"] == 2
    cycle.update(cycle_id=2, entry_index=4)
    assert controller(4, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
