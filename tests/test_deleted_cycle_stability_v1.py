"""验证完整周期删除、全部模型一致、成熟时钟和真实退出。"""
import copy
import numpy as np
import pandas as pd
import pytest

from research.deleted_cycle_stability_inputs_v1 import deletion_subsets, fit_deletions, committee_values, DeletedCycleExitController
from research.learned_cycle_exit_v1 import FEATURES
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def linear(value):
    return {"kind": "RIDGE", "mean": [0.] * 8, "scale": [1.] * 8, "coefficients": [0.] * 8, "feature_clip": 5., "intercept": value}


def stored(base=-.01, deleted=(-.02, -.03), t=0, latest=None):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE", "training_cycle_count": len(deleted),
        "model": {"kind": "DELETED_CYCLE_COMMITTEE", "base_model": linear(base),
            "deletions": [{"deleted_cycle_id": k, "model": linear(v)} for k, v in enumerate(deleted)]}}


def test_whole_cycles_are_deleted_without_replacement_and_remaining_cycles_are_equal():
    rows = pd.DataFrame({"cycle_id": [1, 1, 2, 3, 3, 3], "origin_index": [0, 1, 3, 5, 6, 7], "exit_index": [2, 2, 4, 8, 8, 8]})
    original = rows.copy(deep=True)
    subsets = list(deletion_subsets(rows, 8))
    assert [r for r, selected in subsets] == [1, 2, 3]
    for removed, selected in subsets:
        assert set(selected.cycle_id) == {1, 2, 3} - {removed}
        np.testing.assert_allclose(selected.groupby("cycle_id").sample_weight.sum(), 1.)
    pd.testing.assert_frame_equal(rows, original)
    with pytest.raises(ValueError, match="已经完整结束"):
        list(deletion_subsets(rows, 7))


def test_every_deleted_model_preserves_constant_response_without_changing_base_model():
    rng = np.random.default_rng(73)
    rows = pd.DataFrame(rng.normal(size=(36, 8)), columns=FEATURES)
    rows["cycle_id"] = np.repeat([1, 2, 3], 12)
    rows["origin_index"], rows["exit_index"], rows["target"] = np.arange(36), 40, .03
    fitted = fit_deletions(rows, 40, {"feature_clip": 5., "ridge_alpha": 1.})
    base = linear(.03)
    preserved = copy.deepcopy(base)
    combined = {"kind": "DELETED_CYCLE_COMMITTEE", "base_model": base, "deletions": fitted}
    result = committee_values(combined, np.zeros(8))
    assert result["continuation_prediction"] == pytest.approx(.03) and len(fitted) == 3
    assert base == preserved and all(len(m["training_cycles"]) == 2 for m in fitted)


@pytest.mark.parametrize("base,deleted,maximum,nonnegative", [(-.01, [-.02, -.03], -.01, 0), (-.01, [-.02, 0.], 0., 1), (-.01, [-.02, .03], .03, 1), (.01, [-.02, -.03], .01, 1)])
def test_one_nonnegative_model_prevents_unanimous_negative(base, deleted, maximum, nonnegative):
    result = committee_values(stored(base, deleted)["model"], np.zeros(8))
    assert result["continuation_prediction"] == pytest.approx(maximum)
    assert result["nonnegative_predictions"] == nonnegative


def test_zero_missing_future_and_new_cycle_reset_confirmation():
    data = fixture()[0]
    models = [stored(), stored(deleted=[-.02, 0.], t=2), {"fit_index": 3, "latest_exit_index": None, "status": "NO_VIEW_MODEL_FIT_FAILED", "model": None}, stored(t=5)]
    controller = DeletedCycleExitController(data, models)
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    assert [controller(t, cycle, 100000., 100000.)["negative_confirmation_count"] for t in range(1, 7)] == [1, 0, 0, 0, 1, 2]
    cycle.update(cycle_id=2, entry_index=7)
    assert controller(7, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    cycle.update(cycle_id=1, entry_index=1)
    assert DeletedCycleExitController(data, [stored(t=5)])(1, cycle, 100000., 100000.)["continuation_prediction"] is None


def test_future_training_cycle_is_rejected():
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    with pytest.raises(ValueError, match="未来模型或周期"):
        DeletedCycleExitController(data, [stored(latest=5)])(2, cycle, 100000., 100000.)


def test_actual_blocked_exit_persists_after_one_member_changes_sign_and_reentry_needs_reset():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    controller = DeletedCycleExitController(args[0], [stored(), stored(deleted=[-.02, .03], t=3)])
    ledger, decisions, cycles = simulate_rearmed_exit(*args, controller)
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    changed = decisions[decisions.origin_index.eq(3)].iloc[0]
    assert changed.base_continuation_prediction < 0 < changed.continuation_prediction and changed.requested_quantity < 0
    assert ledger.shares.iloc[-1] == 0 and ledger.accounting_error.abs().max() < 1e-7


def test_no_models_keep_original_price_exit_without_filling_predictions():
    args = fixture()
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, DeletedCycleExitController(args[0], []))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].iloc[0] == args[0].date.iloc[3]
