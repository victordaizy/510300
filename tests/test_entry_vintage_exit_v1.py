"""合成输入验证首次持仓收盘模型固定、换笔更新及真实下一开盘。"""
import numpy as np
import pandas as pd
import pytest
from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController, prediction_identity
from research.within_cycle_exit_inputs_v1 import WithinCycleExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_within_cycle_exit_v1 import constant
from tests.test_median_continuation_v1 import fixture


def cycle(number=1, entry=1):
    return {"cycle_id": number, "entry_index": entry, "entry_cost_cny": 100000., "mode": 1}


def test_opposite_monthly_model_is_ignored_until_a_new_actual_cycle():
    data = fixture()[0]
    models = [constant(.01, 0), constant(-.02, 3)]
    controller = EntryVintageExitController(data, models)
    for t in range(1, 6):
        row = controller(t, cycle(), 100000., 100000.)
        assert row["continuation_prediction"] == .01 and row["learning_fit_origin"] == data.date.iloc[0]
        assert row["model_selection_origin"] == data.date.iloc[1] and not row["learned_exit_requested"]
    first = controller(6, cycle(2, 6), 100000., 100000.)
    second = controller(7, cycle(2, 6), 100000., 100000.)
    assert first["continuation_prediction"] == -.02 and first["negative_confirmation_count"] == 1
    assert second["negative_confirmation_count"] == 2 and second["learned_exit_requested"]
    assert first["learning_fit_origin"] == data.date.iloc[3] and first["model_selection_index"] == 6


def test_initial_absent_or_immature_model_is_not_backfilled_during_the_cycle():
    data = fixture()[0]
    immature = {"fit_index": 0, "latest_exit_index": None, "status": "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", "model": None}
    for models in [[constant(-.02, 3)], [immature, constant(-.02, 3)]]:
        controller = EntryVintageExitController(data, models)
        for t in range(1, 5):
            row = controller(t, cycle(), 100000., 100000.)
            assert row["continuation_prediction"] is None and row["negative_confirmation_count"] == 0
            assert row["fixed_prediction_identity"] == "NO_MODEL" and row["learning_status"] == "NO_VIEW_NO_MATURE_MODEL"
        assert controller(5, cycle(2, 5), 100000., 100000.)["continuation_prediction"] == -.02


def test_missing_current_input_resets_confirmation_but_not_the_selected_version():
    data = fixture()[0]
    data.loc[2, "mom5"] = np.nan
    controller = EntryVintageExitController(data, [constant(-.01, 0), constant(.1, 2)])
    rows = [controller(t, cycle(), 100000., 100000.) for t in range(1, 5)]
    assert [r["negative_confirmation_count"] for r in rows] == [1, 0, 1, 2]
    assert rows[1]["continuation_prediction"] is None and rows[2]["continuation_prediction"] == -.01
    assert len({r["fixed_prediction_identity"] for r in rows}) == 1
    data.loc[1, "mom5"] = np.nan
    data.loc[2, "mom5"] = .01
    controller = EntryVintageExitController(data, [constant(-.01, 0), constant(.1, 2)])
    assert controller(1, cycle(), 100000., 100000.)["continuation_prediction"] is None
    assert controller(2, cycle(), 100000., 100000.)["continuation_prediction"] == -.01


def test_selection_requires_actual_entry_close_and_excludes_future_model_information():
    data = fixture()[0]
    with pytest.raises(ValueError, match="首次持仓收盘"):
        EntryVintageExitController(data, [constant()])(2, cycle(), 100000., 100000.)
    with pytest.raises(ValueError, match="未来周期"):
        EntryVintageExitController(data, [constant(t=1, latest=2)])(1, cycle(), 100000., 100000.)
    same_day = EntryVintageExitController(data, [constant(.02, 1), constant(-.03, 2)])
    assert same_day(1, cycle(), 100000., 100000.)["continuation_prediction"] == .02
    assert same_day(2, cycle(), 100000., 100000.)["continuation_prediction"] == .02


def test_actual_next_open_blocked_exit_reentry_and_original_protection_without_model():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    controller = EntryVintageExitController(args[0], [constant(-.01, 0), constant(.03, 3)])
    ledger, decisions, cycles = simulate_rearmed_exit(*args, controller)
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    assert decisions[decisions.origin_index.eq(7)].iloc[0].continuation_prediction == .03
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    args = fixture()
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, EntryVintageExitController(args[0], [constant(-.5, 2)]))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]


def test_identical_prediction_functions_with_new_dates_preserve_the_real_account():
    args = fixture()
    models = [constant(.01, 0), constant(.01, 3)]
    assert prediction_identity(models[0]) == prediction_identity(models[1])
    fixed, _, _ = simulate_rearmed_exit(*args, EntryVintageExitController(args[0], models))
    original, _, _ = simulate_rearmed_exit(*args, WithinCycleExitController(args[0], models))
    pd.testing.assert_frame_equal(fixed, original)
