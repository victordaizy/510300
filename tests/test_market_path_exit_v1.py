"""验证市场状态费用不变性、完整路径峰值、成熟模型与真实费用风控。"""
import numpy as np
import pandas as pd
import pytest
from research.market_path_state_v1 import MarketPathTracker, reference_market_states, attach_reference_market_states
from research.market_path_exit_inputs_v1 import FEATURES, fit_market_path_exit, market_path_prediction, MarketPathExitController
from research.learned_cycle_exit_v1 import FEATURES as OLD_FEATURES, state_values, training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_within_cycle_exit_v1 import samples as old_samples
from tests.test_median_continuation_v1 import fixture

CFG = {"feature_clip": 5., "ridge_alpha": 1.}


def constant(value=-.01, t=0, latest=None, return_coefficient=0.):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE",
            "model": {"kind": "MARKET_PATH_WITHIN_CYCLE_RIDGE", "features": FEATURES.copy(), "mean": [0.]*8, "scale": [1.]*8,
                      "coefficients": [0., return_coefficient]+[0.]*6, "intercept": value, "feature_clip": 5.,
                      "cycle_intercepts": [{"cycle_id": 999, "cycle_intercept": 100.}]}}


def cycle(quantity=10000, cost=100000., number=1, entry=1):
    return {"cycle_id": number, "entry_index": entry, "entry_quantity": quantity, "entry_cost_cny": cost, "mode": 1}


def test_same_market_path_has_same_prediction_for_different_sunk_fees_and_quantities():
    data = fixture()[0]
    left, right = MarketPathExitController(data, [constant(0., return_coefficient=-1.)]), MarketPathExitController(data, [constant(0., return_coefficient=-1.)])
    a, b = cycle(10000, 100500.), cycle(9000, 91800.)
    for t, unit in enumerate([10.02, 10.2, 10.1], start=1):
        x, y = left(t, a, unit*10000, 102000.), right(t, b, unit*9000, 91800.)
        for key in FEATURES+["market_entry_open", "market_unit_value", "market_unit_peak", "continuation_prediction"]:
            assert x[key] == pytest.approx(y[key], abs=1e-12)
        assert x["account_cycle_return"] != pytest.approx(y["account_cycle_return"])
    assert x["market_unit_peak"] == pytest.approx(10.2)
    assert x["market_cycle_drawdown"] == pytest.approx(10.1/10.2-1)


def test_missing_or_skipped_close_keeps_peak_unknown_until_new_actual_cycle():
    data = fixture()[0]; c = cycle(); tracker = MarketPathTracker(data)
    assert tracker.observe(1, c, 101000.)["market_unit_peak"] == 10.1
    skipped = tracker.observe(3, c, 102000.)
    following = tracker.observe(4, c, 103000.)
    assert np.isnan(skipped["market_cycle_drawdown"]) and np.isnan(following["market_cycle_drawdown"])
    c = cycle(number=2, entry=5)
    assert tracker.observe(5, c, 99500.)["market_unit_peak"] == 10.
    assert np.isnan(tracker.observe(6, c, np.nan)["market_cycle_drawdown"])
    assert np.isnan(tracker.observe(7, c, 110000.)["market_cycle_drawdown"])
    with pytest.raises(ValueError, match="严格递增"):
        tracker.observe(7, c, 110000.)


def test_reference_peak_uses_full_holding_ledger_before_filtered_training_rows():
    args = fixture(); args[0].loc[3, "close"] = 13.
    def observer(t, c, value, peak):
        return {"learning_cycle_id": c["cycle_id"], "learned_exit_requested": False, **dict(zip(OLD_FEATURES, state_values(args[0], t, c, value, peak)))}
    ledger, decisions, cycles = simulate_rearmed_exit(*args, observer)
    states = reference_market_states(args[0], ledger, decisions, cycles)
    rows = pd.DataFrame({"cycle_id": [1, 1], "origin_index": [1, 4], "target": [.1, .2]})
    selected = attach_reference_market_states(rows, states)
    assert selected.loc[1, "market_cycle_drawdown"] == pytest.approx(10./13.-1.)
    assert selected.target.tolist() == [.1, .2] and len(states) == 8


def test_market_eight_factor_equation_and_future_maturity_preserve_original_members():
    rows = old_samples().rename(columns={"cycle_return": "market_cycle_return", "cycle_drawdown": "market_cycle_drawdown"})
    m = fit_market_path_exit(rows, CFG)
    z = np.clip((rows[FEATURES].to_numpy()-m["mean"])/m["scale"], -5, 5)
    design = np.c_[z, pd.get_dummies(rows.cycle_id, dtype=float).to_numpy()]; w = rows.sample_weight.to_numpy()
    b = np.linalg.solve(design.T@(w[:, None]*design)+np.diag([1.]*8+[0.]*3), design.T@(w*rows.target.to_numpy()))
    np.testing.assert_allclose(m["coefficients"], b[:8], atol=1e-12, rtol=0)
    assert m["intercept"] == pytest.approx(b[8:].mean())
    assert market_path_prediction(m, rows[FEATURES].iloc[0]) == pytest.approx(b[8:].mean()+z[0]@b[:8])
    rows["origin_index"] = np.arange(len(rows)); rows["exit_index"] = rows.cycle_id.map({1: 25, 2: 55, 3: 95})
    past, ids = training_rows(rows, 55, {"recent_cycles": 20}); old = fit_market_path_exit(past, CFG)
    rows.loc[rows.cycle_id.eq(3), ["market_cycle_return", "target"]] = 1000.
    assert ids == [1, 2] and fit_market_path_exit(training_rows(rows, 55, {"recent_cycles": 20})[0], CFG) == old
    past.loc[past.index[0], "market_cycle_drawdown"] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        fit_market_path_exit(past, CFG)


def test_missing_path_resets_confirmation_without_future_model_or_cycle_intercept():
    data = fixture()[0]; c = cycle(number=999)
    controller = MarketPathExitController(data, [constant()])
    assert controller(1, c, 100000., 100000.)["negative_confirmation_count"] == 1
    unknown = controller(2, c, np.nan, 100000.)
    assert unknown["continuation_prediction"] is None and unknown["negative_confirmation_count"] == 0
    assert controller(3, c, 100000., 100000.)["continuation_prediction"] is None
    assert MarketPathExitController(data, [constant(t=5)])(1, c, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        MarketPathExitController(data, [constant(latest=5)])(1, c, 100000., 100000.)
    c = cycle(number=1000, entry=4)
    assert controller(4, c, 100000., 100000.)["negative_confirmation_count"] == 1


def test_actual_next_open_exit_lock_and_true_fee_risk_protection_still_apply():
    args = fixture(); args[0].loc[3, "open"] = 9.; args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, MarketPathExitController(args[0], [constant(), constant(.03, 3)]))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    args = fixture(); args[3]["commission"] = .1; args[6]["modes"][1]["loss"] = .06
    ledger, decisions, cycles = simulate_rearmed_exit(*args, MarketPathExitController(args[0], [constant(.1)]))
    assert "固定止损" in cycles.iloc[0].exit_reasons and not decisions.iloc[1].learned_exit_requested
    assert decisions.iloc[1].market_cycle_return == pytest.approx(0.) and decisions.iloc[1].account_cycle_return < -.06
    assert ledger.commission.sum() > 0 and ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]
