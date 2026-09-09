"""验证两层分离、周期成熟、买入前时钟和实际退出。"""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from research.entry_context_intercept_inputs_v1 import CONTEXT_FEATURES, cycle_entry_context, select_context, fit_entry_context_intercept, entry_context_prediction, EntryContextInterceptController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_within_cycle_exit_v1 import constant as within_constant
from tests.test_median_continuation_v1 import fixture

CFG = {"feature_clip": 5., "context_ridge_alpha": 1.}


def training():
    rng = np.random.default_rng(115)
    table = pd.DataFrame(rng.normal(size=(12, 3)), columns=CONTEXT_FEATURES)
    table["cycle_id"], table["entry_origin_index"] = np.arange(1, 13), np.arange(12)
    table["exit_date"] = pd.Timestamp("2019-01-01")
    table = table.set_index("cycle_id", drop=False)
    record = within_constant(t=30)
    record.update(fit_origin="2020-01-01", training_cycles=list(range(1, 11)))
    record["model"]["cycle_intercepts"] = [{"cycle_id": i, "cycle_intercept": .1+.03*table.loc[i, "entry_mom20"], "rows": i*10} for i in range(1, 11)]
    return record, table


def constant(value=-.01, t=0, latest=None, slope=0.):
    original = within_constant(value=100., t=t, latest=latest)
    return {**original, "model": {"kind": "ENTRY_CONTEXT_PLUS_SAVED_WITHIN_CYCLE", "within_model": original["model"],
            "context_model": {"features": CONTEXT_FEATURES.copy(), "mean": [0.]*3, "scale": [1.]*3,
                              "coefficients": [slope, 0., 0.], "intercept": value, "feature_clip": 5.}}}


def test_two_stage_fit_matches_equal_cycle_ridge_without_refitting_within_slopes():
    record, context = training(); original = deepcopy(record)
    model = fit_entry_context_intercept(record, context, CFG)
    m = model["context_model"]; selected = select_context(record, context)
    z = np.clip((selected[CONTEXT_FEATURES].to_numpy()-m["mean"])/m["scale"], -5, 5)
    design = np.c_[np.ones(len(z)), z]
    expected = np.linalg.solve(design.T@design+np.diag([0., 1., 1., 1.]), design.T@selected.target_intercept.to_numpy())
    np.testing.assert_allclose([m["intercept"], *m["coefficients"]], expected, atol=1e-12, rtol=0)
    assert record == original and model["within_model"] == original["model"]
    for group in record["model"]["cycle_intercepts"]:
        group["rows"] *= 100
    assert fit_entry_context_intercept(record, context, CFG)["context_model"] == m


def test_unselected_future_cycles_cannot_change_model_and_unfinished_members_fail():
    record, context = training(); model = fit_entry_context_intercept(record, context, CFG)
    context.loc[[11, 12], CONTEXT_FEATURES] = 9999.
    assert fit_entry_context_intercept(record, context, CFG) == model
    context.loc[1, "exit_date"] = pd.Timestamp("2020-01-02")
    with pytest.raises(ValueError, match="尚未完成"):
        fit_entry_context_intercept(record, context, CFG)
    context.loc[1, "exit_date"] = pd.Timestamp("2019-01-01")
    context.loc[2, "entry_vol20"] = np.nan
    with pytest.raises(ValueError, match="禁止删周期"):
        fit_entry_context_intercept(record, context, CFG)


def test_entry_context_uses_request_close_instead_of_buy_day_or_later_state():
    data = fixture()[0]; data.loc[1:, ["mom20", "sma120", "vol20"]] = 100.
    cycle = {"cycle_id": 999, "entry_index": 1, "entry_origin": data.date.iloc[0], "entry_cost_cny": 100000., "mode": 1}
    c = EntryContextInterceptController(data, [constant(value=0., slope=1.)])
    first, second = c(1, cycle, 100000., 100000.), c(2, cycle, 100000., 100000.)
    assert first["context_intercept_prediction"] == second["context_intercept_prediction"] == .02
    assert first["entry_context_origin"] == data.date.iloc[0]
    assert entry_context_prediction(constant()["model"], np.zeros(8), np.zeros(3))[0] == -.01
    reference = pd.DataFrame([{**cycle, "exit_date": data.date.iloc[4]}])
    assert cycle_entry_context(data, reference).loc[999, "entry_mom20"] == .02


def test_unknown_entry_state_resets_confirmation_and_has_no_average_fallback():
    data = fixture()[0]; cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    c = EntryContextInterceptController(data, [constant()])
    assert c(1, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    data.loc[0, "vol20"] = np.nan
    row = c(2, cycle, 100000., 100000.)
    assert row["learning_status"] == "NO_VIEW_INCOMPLETE_ENTRY_CONTEXT" and row["continuation_prediction"] is None
    assert row["negative_confirmation_count"] == 0 and not row["learned_exit_requested"]
    assert EntryContextInterceptController(data, [constant(t=5)])(1, cycle, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        EntryContextInterceptController(data, [constant(latest=5)])(1, cycle, 100000., 100000.)
    cycle.update(cycle_id=2, entry_index=3)
    assert c(3, cycle, 100000., 100000.)["negative_confirmation_count"] == 1


def test_new_month_updates_context_coefficients_but_retains_original_entry_inputs():
    data = fixture()[0]; data.loc[2, "mom20"] = 3.
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    c = EntryContextInterceptController(data, [constant(value=0., slope=1.), constant(value=0., t=3, slope=-1.)])
    assert c(2, cycle, 100000., 100000.)["context_intercept_prediction"] == .02
    assert c(3, cycle, 100000., 100000.)["context_intercept_prediction"] == -.02
    assert c(4, cycle, 100000., 100000.)["learned_exit_requested"]


def test_actual_next_open_exit_lock_and_original_price_exit_without_model():
    args = fixture(); args[0].loc[3, "open"] = 9.; args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, EntryContextInterceptController(args[0], [constant(), constant(.03, 3)]))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    args = fixture(); args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, EntryContextInterceptController(args[0], []))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]
