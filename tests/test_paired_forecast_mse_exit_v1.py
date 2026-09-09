"""检验混合数学、整周期成熟边界和实际成交，不扩展搜索设置。"""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from research.paired_forecast_mse_exit_inputs_v1 import reconstruct_pairs, complete_mature_rows, minimum_mse_weights, calibrate_months, mix_predictions, PairedForecastExitController
from research.learned_cycle_exit_v1 import FEATURES, ExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture
from tests.test_profit_drawdown_interaction_v1 import stored as new_stored


def old_stored(value=-.01, t=0):
    record = new_stored(value, t)
    record.update(fit_origin=f"2020-01-{t+1:02d}", training_cycles=[])
    record["model"].update(kind="RIDGE", mean=[0.]*8, scale=[1.]*8, coefficients=[0.]*8)
    record["model"].pop("features")
    return record


def new_record(value=-.01, t=0):
    record = new_stored(value, t)
    record.update(fit_origin=f"2020-01-{t+1:02d}", training_cycles=[])
    return record


def pairs():
    return pd.DataFrame({"cycle_id": [1, 1, 2, 2, 3], "origin_index": [1, 2, 5, 6, 10], "exit_index": [4, 4, 8, 8, 20],
                         "both_available": [True, True, True, False, True], "old_error": [1., -1., 3., 3., 4.], "new_error": [-1., 1., 2., 2., 5.]})


def test_uncentered_mse_interior_endpoints_and_identical_errors():
    rows = pd.DataFrame({"old_error": [1., -1.], "new_error": [-1., 1.], "sample_weight": [.5, .5]})
    fitted = minimum_mse_weights(rows, (1., 0.))
    assert fitted["old_weight"] == pytest.approx(.5) and fitted["mixed_mse"] == pytest.approx(0.)
    rows["old_error"], rows["new_error"] = 1., 2.
    assert minimum_mse_weights(rows, (.5, .5))["old_weight"] == 1.
    rows["old_error"], rows["new_error"] = 2., 1.
    result = minimum_mse_weights(rows, (.5, .5))
    assert result["old_weight"] == 0. and result["new_mse"] == 1.
    rows["old_error"] = 1.
    result = minimum_mse_weights(rows, (.3, .7))
    assert result["old_weight"] == .3 and result["calibration_status"].startswith("NO_VIEW_IDENTICAL")


def test_complete_cycles_maturity_equal_total_weight_and_unknown_errors_preserved():
    chosen, ids = complete_mature_rows(pairs(), 10, {"recent_cycles": 20})
    assert ids == [1] and len(chosen) == 2
    np.testing.assert_allclose(chosen.sample_weight, [.5, .5])
    chosen.loc[chosen.index[0], "old_error"] = np.nan
    result = minimum_mse_weights(chosen, (.25, .75))
    assert result["calibration_status"] == "NO_VIEW_INCOMPLETE_PAIRED_ERRORS" and result["old_weight"] == .25
    full = pairs()
    full["both_available"] = True
    selected, _ = complete_mature_rows(full, 25, {"recent_cycles": 2})
    assert set(selected.cycle_id) == {2, 3}
    np.testing.assert_allclose(selected.groupby("cycle_id").sample_weight.sum(), 1.)


def test_reconstructed_predictions_exclude_future_model_and_current_cycle():
    sample = pd.DataFrame([{**dict(zip(FEATURES, [1., .1, -.1, 1., .01, .02, .03, .15])),
                            "cycle_id": 1, "origin_index": 2, "early_exit_index": 3, "exit_index": 8,
                            "origin": pd.Timestamp("2020-01-03"), "mature_date": pd.Timestamp("2020-01-09"), "target": .04}])
    first = reconstruct_pairs(sample, [old_stored()], [new_record()])
    altered = reconstruct_pairs(sample, [old_stored(), old_stored(50., 5)], [new_record(), new_record(-50., 5)])
    pd.testing.assert_frame_equal(first, altered)
    assert first.old_prediction.iloc[0] == -.01 and first.old_error.iloc[0] == -.05
    bad = old_stored()
    bad["training_cycles"] = [1]
    with pytest.raises(ValueError, match="当前未成熟"):
        reconstruct_pairs(sample, [bad], [new_record()])
    bad = old_stored()
    bad["latest_exit_index"] = 3
    with pytest.raises(ValueError, match="未来周期"):
        reconstruct_pairs(sample, [bad], [new_record()])


def test_monthly_calibration_does_not_use_future_labels_or_reduce_minimum():
    source = pairs()
    schedule = [{"fit_index": t, "fit_origin": str(t)} for t in [2, 4, 10, 25]]
    cfg = {"recent_cycles": 20, "minimum_cycles": 1, "minimum_rows": 2}
    records, _ = calibrate_months(source, schedule, cfg)
    assert records[0]["old_weight"] == 1. and records[1]["old_weight"] == .5
    changed = source.copy()
    changed.loc[changed.exit_index.gt(10), ["old_error", "new_error"]] = [1000., -100.]
    altered, _ = calibrate_months(changed, schedule, cfg)
    assert altered[:3] == records[:3]
    initial, _ = calibrate_months(source, schedule, {**cfg, "minimum_cycles": 10, "minimum_rows": 100})
    assert all(r["old_weight"] == 1. and not r["supports_minimum"] for r in initial)
    unknown = source.copy()
    unknown.loc[1, "old_error"] = np.nan
    uncertain, _ = calibrate_months(unknown, schedule, cfg)
    assert uncertain[1]["old_weight"] == 1. and uncertain[1]["calibration_status"].startswith("NO_VIEW_INCOMPLETE")


def test_zero_weight_unknown_is_ignored_but_required_unknown_resets_confirmation():
    assert mix_predictions(-.01, None, 1., 0.) == -.01
    assert mix_predictions(None, -.01, 0., 1.) == -.01
    assert mix_predictions(-.01, None, .5, .5) is None
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    calibrations = [{"fit_index": 2, "latest_exit_index": 0, "old_weight": .5, "new_weight": .5,
                     "calibration_status": "CALIBRATION_COMPLETE", "weight_update_origin": "2020-01-03"}]
    controller = PairedForecastExitController(data, [old_stored()], [], calibrations)
    assert controller(1, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    result = controller(2, cycle, 100000., 100000.)
    assert result["continuation_prediction"] is None and result["negative_confirmation_count"] == 0


def test_actual_exit_next_open_lock_and_new_cycle_reset():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    controller = PairedForecastExitController(args[0], [old_stored(), old_stored(.03, 3)], [new_record(), new_record(.03, 3)], [])
    ledger, decisions, cycles = simulate_rearmed_exit(*args, controller)
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    negative = PairedForecastExitController(args[0], [old_stored()], [], [])
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    assert negative(1, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    assert negative(2, cycle, 100000., 100000.)["negative_confirmation_count"] == 2
    cycle.update(cycle_id=2, entry_index=4)
    assert negative(4, cycle, 100000., 100000.)["negative_confirmation_count"] == 1


def test_initial_baseline_reproduces_original_account_and_retains_price_exit():
    args = fixture()
    args[5]["exit"][1][2] = True
    models = [old_stored()]
    expected = simulate_rearmed_exit(*deepcopy(args), ExitController(args[0], models))
    actual = simulate_rearmed_exit(*deepcopy(args), PairedForecastExitController(args[0], models, [], []))
    pd.testing.assert_frame_equal(actual[0], expected[0])
    unknown = simulate_rearmed_exit(*deepcopy(args), PairedForecastExitController(args[0], [], [], []))
    assert unknown[1].continuation_prediction.dropna().empty and "价格退出条件" in unknown[2].iloc[0].exit_reasons
    assert unknown[0].loc[unknown[0].filled_quantity.lt(0), "date"].iloc[0] == args[0].date.iloc[3]
