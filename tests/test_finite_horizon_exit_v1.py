"""逐年龄倒推的现金流、支持边界、共同回归和真实退出规则。"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from research.finite_horizon_exit_inputs_v1 import (FEATURES, METHODS, PRIMARY, CONTROL,
    liquidation_curve, fit_pair, predict_pair, fit_backward_month, FiniteHorizonExitController, simulate_finite_exit)
from research.learned_cycle_exit_v1 import training_rows
from tests.test_median_continuation_v1 import fixture

CFG = {"feature_clip": 5., "ridge_alpha": 1., "minimum_age_cycles": 3, "maximum_age": 3}


def constant_model(value):
    return {"kind": "AGE_PAIRED_RIDGE", "features": FEATURES, "methods": METHODS,
            "mean": [0.]*8, "scale": [1.]*8, "feature_clip": 5.,
            "coefficients": [[0., 0.]]*8, "intercepts": [value, value]}


def month(value, t=0, latest=None, ages=range(1, 10)):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "MONTH_COMPLETE",
            "age_models": {str(age): constant_model(value) for age in ages},
            "age_statuses": {str(age): "FIT_COMPLETE" for age in ages}}


def toy():
    states, curves = [], {}
    for cid, terminal_value in enumerate([60., 60., 150.], 1):
        entry = cid*10
        curves[cid] = {entry+1: 100., entry+2: 120., entry+3: 80., entry+4: terminal_value}
        for age in [1, 2, 3]:
            first = entry+age
            states.append({**dict.fromkeys(FEATURES, 0.), "cycle_id": cid, "holding_age": age,
                           "origin_index": first-1, "exit_index": entry+4, "immediate_exit_index": first,
                           "immediate_exit_value": curves[cid][first], "natural_exit_value": terminal_value,
                           "target_denominator": 100.})
    return pd.DataFrame(states), curves


def test_cashflow_waits_for_first_executable_open_and_counts_dividend_right_once():
    data, div, cfg, cost, *_ = fixture()
    data.loc[2, "open"] = 9.
    div = pd.DataFrame({"record_date": [data.date.iloc[3]], "ex_date": [data.date.iloc[4]],
                        "payment_date": [data.date.iloc[8]], "cash_dividend_per_share": [.1]})
    values, checks = liquidation_curve(data, div, 1, 7, 100, cfg, cost)
    assert 1 not in values and 2 not in values and min(values) == 3
    assert checks[0]["status"] == "UNFILLED_DIRECTIONAL_LIMIT"
    assert values[3] == pytest.approx(100*9.995-5)
    assert values[4] == pytest.approx(values[3]+10.)
    assert values[7] == pytest.approx(values[4])
    assert checks[2]["earned_dividend_cny"] == 10.
    assert checks[2]["commission"] == 5. and checks[2]["slippage_cost"] > 0


def test_unconfirmed_dividend_and_unavailable_terminal_fail_the_original_maturity():
    data, div, cfg, cost, *_ = fixture()
    div = pd.DataFrame({"record_date": [data.date.iloc[3]], "ex_date": [data.date.iloc[8]],
                        "payment_date": [data.date.iloc[9]], "cash_dividend_per_share": [.1]})
    with pytest.raises(ValueError, match="仍未确认"):
        liquidation_curve(data, div, 1, 7, 100, cfg, cost)
    data.loc[7, "open"] = 9.
    with pytest.raises(ValueError, match="自然退出开盘"):
        liquidation_curve(data, div.iloc[:0], 1, 7, 100, cfg, cost)


def test_joint_ridge_matches_two_independent_targets_and_constant_features_have_no_effect():
    rng = np.random.default_rng(118)
    x, y = rng.normal(size=(20, 8)), rng.normal(size=(20, 2))
    x[:, 0], x[:, 3] = 2., 1.
    fitted = fit_pair(x, y, CFG)
    z = np.clip((x-fitted["mean"])/fitted["scale"], -5, 5)
    for column in [0, 1]:
        separate = Ridge(alpha=1., fit_intercept=True, solver="svd").fit(z, y[:, column])
        np.testing.assert_allclose(predict_pair(fitted, x)[:, column], separate.predict(z), atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.asarray(fitted["coefficients"])[[0, 3]], 0., atol=1e-12)
    altered = y.copy(); altered[:, 0] += 1000.
    other = fit_pair(x, altered, CFG)
    np.testing.assert_allclose(predict_pair(other, x)[:, 1], predict_pair(fitted, x)[:, 1], atol=1e-12, rtol=0)


def test_backward_target_follows_predicted_later_exit_instead_of_realized_maximum():
    rows, curves = toy()
    models, receipts, records = fit_backward_month(rows, curves, CFG)
    assert len(models) == 3 and all(r["status"] == "FIT_COMPLETE" for r in receipts)
    members = pd.DataFrame(records)
    last = members[members.holding_age.eq(3)]
    assert last.backward_prediction.gt(0).all()
    middle = members[members.holding_age.eq(2)]
    assert middle.backward_requests_exit.all()
    third = middle[middle.cycle_id.eq(3)].iloc[0]
    assert third.backward_target > 0 and third.backward_prediction < 0
    assert third.future_exit_after_decision == 32 and curves[3][32] < curves[3][34]
    first = members[members.holding_age.eq(1)]
    np.testing.assert_allclose(first.backward_target, .2)
    assert first.backward_prediction.gt(0).all() and first.natural_prediction.lt(0).all()


def test_mature_whole_cycle_and_exact_age_support_do_not_borrow_future_or_nearby_models():
    rows, curves = toy()
    selected, ids = training_rows(rows, 24, {"recent_cycles": 20})
    assert ids == [1, 2] and selected.exit_index.le(24).all()
    models, receipts, records = fit_backward_month(selected, curves, CFG)
    assert not models and not records and all(r["status"] == "NO_VIEW_MINIMUM_AGE_CYCLES" for r in receipts)
    sparse = rows[~(rows.cycle_id.eq(3) & rows.holding_age.eq(2))]
    models, receipts, records = fit_backward_month(sparse, curves, CFG)
    assert set(models) == {"1", "3"} and next(r for r in receipts if r["holding_age"] == 2)["cycles"] == 2
    one = pd.DataFrame(records).query("holding_age == 1")
    assert (one.future_exit_before_decision == one.exit_index).all()


def test_controller_uses_exact_age_mature_month_and_resets_missing_or_new_cycle():
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_quantity": 100, "entry_cost_cny": 1005., "mode": 1}
    controller = FiniteHorizonExitController(data, [month(-.1, ages=[1, 3])], PRIMARY)
    decisions = [controller(t, cycle, 1000., 1005.) for t in [1, 2, 3]]
    assert [d["learned_exit_requested"] for d in decisions] == [True, False, True]
    assert decisions[1]["continuation_prediction"] is None
    assert decisions[2]["negative_confirmation_count"] == 1
    cycle.update(cycle_id=2, entry_index=4, entry_cost_cny=1010.)
    assert controller(4, cycle, 1000., 1010.)["negative_confirmation_count"] == 1
    assert FiniteHorizonExitController(data, [month(-.1, t=5)], PRIMARY)(4, cycle, 1000., 1010.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        FiniteHorizonExitController(data, [month(-.1, latest=5)], PRIMARY)(4, cycle, 1000., 1010.)


def test_single_confirmation_next_open_locked_sale_and_reentry_reset_are_actual():
    args = fixture()
    args[0].loc[2, "open"] = 9.
    args[5]["entry"][5] = 0
    controller = FiniteHorizonExitController(args[0], [month(-.02), month(.03, t=2)], CONTROL)
    ledger, decisions, cycles = simulate_finite_exit(*args, controller)
    assert ledger.iloc[1].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[3], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    pending = decisions[decisions.origin_index.eq(2)].iloc[0]
    assert pending.continuation_prediction > 0 and pending.requested_quantity < 0
    assert "当日收盘" in cycles.iloc[0].exit_reasons and "连续两个收盘" not in cycles.iloc[0].exit_reasons
    assert ledger.accounting_error.abs().max() < 1e-7 and ledger.shares.iloc[-1] == 0
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0


def test_missing_model_keeps_original_risk_exit_without_zero_prediction():
    args = fixture()
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_finite_exit(*args, FiniteHorizonExitController(args[0], [], PRIMARY))
    assert decisions.continuation_prediction.dropna().empty
    assert "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].iloc[0] == args[0].date.iloc[3]
