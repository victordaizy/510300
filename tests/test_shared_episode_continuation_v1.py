"""检查共同持仓时段的因果边界、层级等权、共享模型和实际退出。"""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from research.shared_episode_continuation_inputs_v1 import SIGNALS, FEATURES, TASK_FEATURES, calendar_episodes, grouped_samples, shared_training_rows, support_status, fit_shared_exit, shared_prediction, SharedEpisodeExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def reference_fixture():
    dates = pd.bdate_range("2020-01-01", periods=16)
    ledgers = {s: pd.DataFrame({"date": dates, "shares": np.zeros(16)}) for s in SIGNALS}
    for s, held in [(SIGNALS[0], [1, 2, 3, 7, 8, 13, 14]), (SIGNALS[1], [2, 3, 4, 8, 9, 10]), (SIGNALS[2], [1, 2, 7, 8])]:
        ledgers[s].loc[held, "shares"] = 100.
    records = []
    for s, cycle, origins, end in [(SIGNALS[0], 1, [1, 2], 4), (SIGNALS[1], 1, [2, 3], 5), (SIGNALS[2], 1, [1], 3),
                                  (SIGNALS[0], 2, [7], 9), (SIGNALS[1], 2, [8, 9], 11), (SIGNALS[2], 2, [7], 9)]:
        records += [{"signal": s, "cycle_id": cycle, "origin_index": t, "origin": dates[t], "exit_index": end,
                     "mature_date": dates[end], "target": .01*t} for t in origins]
    return dates, ledgers, pd.DataFrame(records)


def constant_record(value=-.01, t=0, latest_group=None):
    return {"fit_index": t, "latest_exit_index": 0, "latest_group_mature_index": t if latest_group is None else latest_group,
            "status": "FIT_COMPLETE", "model": {"kind": "SHARED_EPISODE_CONTINUATION_RIDGE", "features": FEATURES.copy(),
            "mean": [0.]*10, "scale": [1.]*10, "coefficients": [0.]*10, "intercept": value, "feature_clip": 5.}}


def test_overlapping_positions_form_one_episode_and_terminal_is_not_natural_maturity():
    dates, ledgers, samples = reference_fixture()
    groups, assignments = calendar_episodes(ledgers, dates[-1])
    assert groups.group_mature_date.iloc[:2].tolist() == [dates[5], dates[11]]
    assert pd.isna(groups.group_mature_date.iloc[2])
    assert assignments.loc[assignments.origin.eq(dates[3]), "group_id"].iloc[0] == 1
    corrupt = deepcopy(ledgers); corrupt[SIGNALS[0]].loc[3, "shares"] = np.nan
    with pytest.raises(ValueError, match="不能当作空仓"):
        calendar_episodes(corrupt, dates[-1])
    corrupt = deepcopy(ledgers); corrupt[SIGNALS[1]] = corrupt[SIGNALS[1]].iloc[:-1]
    with pytest.raises(ValueError, match="完整日历"):
        calendar_episodes(corrupt, dates[-1])


def test_future_extension_and_future_labels_cannot_change_past_available_groups():
    dates, ledgers, samples = reference_fixture()
    groups, assignments = calendar_episodes(ledgers, dates[-1])
    full = grouped_samples(samples, assignments, groups)
    cfg = {"recent_groups": 20}
    past, ids = shared_training_rows(full, dates[9], cfg)
    assert ids == [1] and len(past) == 5
    changed = deepcopy(ledgers)
    changed[SIGNALS[1]].loc[10:14, "shares"] = 100.
    new_groups, new_assignment = calendar_episodes(changed, dates[-1])
    altered_samples = samples.copy(); altered_samples.loc[altered_samples.cycle_id.eq(2), "target"] = 999.
    altered, _ = shared_training_rows(grouped_samples(altered_samples, new_assignment, new_groups), dates[9], cfg)
    pd.testing.assert_frame_equal(past, altered)
    prefix = {s: l.iloc[:10].copy() for s, l in ledgers.items()}
    prefix_groups, _ = calendar_episodes(prefix, dates[-1])
    pd.testing.assert_frame_equal(groups[groups.group_mature_date.le(dates[9])], prefix_groups[prefix_groups.group_mature_date.le(dates[9])])


def test_original_targets_task_markers_and_hierarchical_equal_weights_preserved():
    dates, ledgers, samples = reference_fixture()
    groups, assignments = calendar_episodes(ledgers, dates[-1])
    all_rows = grouped_samples(samples, assignments, groups)
    pd.testing.assert_frame_equal(all_rows[samples.columns], samples)
    chosen, ids = shared_training_rows(all_rows, dates[11], {"recent_groups": 20})
    assert ids == [1, 2] and len(chosen) == 9
    np.testing.assert_allclose(chosen.groupby("group_id").sample_weight.sum(), 1.)
    np.testing.assert_allclose(chosen.groupby(["group_id", "signal", "cycle_id"]).sample_weight.sum(), 1/3)
    assert chosen.loc[chosen.signal.eq(SIGNALS[0]), TASK_FEATURES].eq(0).all().all()
    assert chosen.loc[chosen.signal.eq(SIGNALS[1]), TASK_FEATURES[0]].eq(1).all()
    assert chosen.loc[chosen.signal.eq(SIGNALS[2]), TASK_FEATURES[1]].eq(1).all()
    assert support_status(chosen, ids, {"minimum_groups": 2, "minimum_rows": 9})
    assert not support_status(chosen, ids, {"minimum_groups": 10, "minimum_rows": 100})
    assert not support_status(chosen[chosen.signal.ne(SIGNALS[2])], ids, {"minimum_groups": 1, "minimum_rows": 1})


def test_ten_factor_solution_and_d60_task_prediction_match_weighted_ridge_equation():
    rng = np.random.default_rng(108)
    rows = pd.DataFrame(rng.normal(size=(120, 8)), columns=FEATURES[:8])
    rows["entry_mode"] = 1.
    rows[TASK_FEATURES[0]] = np.repeat([0., 1., 0.], 40)
    rows[TASK_FEATURES[1]] = np.repeat([0., 0., 1.], 40)
    rows["target"] = -.03*rows.cycle_drawdown+.02*rows[TASK_FEATURES[0]]-.05*rows[TASK_FEATURES[1]]
    rows["sample_weight"] = np.repeat([1/40, 1/80, 1/80], 40)
    model = fit_shared_exit(rows, {"feature_clip": 5., "ridge_alpha": 1.})
    z = np.clip((rows[FEATURES].to_numpy()-model["mean"])/model["scale"], -5, 5)
    a = np.c_[np.ones(len(rows)), z]; w = rows.sample_weight.to_numpy()
    expected = np.linalg.solve(a.T@(w[:, None]*a)+np.diag([0.]+[1.]*10), a.T@(w*rows.target.to_numpy()))
    np.testing.assert_allclose([model["intercept"]]+model["coefficients"], expected, atol=1e-12, rtol=0)
    x = np.zeros(10)
    d60 = shared_prediction(model, x)
    x[-1] = 1.
    assert abs(shared_prediction(model, x)-d60) > .001
    rows.loc[0, "target"] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        fit_shared_exit(rows, {"feature_clip": 5., "ridge_alpha": 1.})


def test_unknown_actual_state_and_pending_group_maturity_block_prediction():
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    controller = SharedEpisodeExitController(data, [constant_record()])
    first = controller(1, cycle, 100000., 100000.)
    assert first["negative_confirmation_count"] == 1 and first[TASK_FEATURES[0]] == first[TASK_FEATURES[1]] == 0.
    missing = controller(2, cycle, np.nan, 100000.)
    assert missing["continuation_prediction"] is None and missing["negative_confirmation_count"] == 0
    assert SharedEpisodeExitController(data, [constant_record(t=5)])(3, cycle, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未成熟共同时段"):
        SharedEpisodeExitController(data, [constant_record(latest_group=5)])(3, cycle, 100000., 100000.)
    controller(3, cycle, 100000., 100000.)
    assert controller(4, cycle, 100000., 100000.)["negative_confirmation_count"] == 2
    cycle.update(cycle_id=2, entry_index=5)
    assert controller(5, cycle, 100000., 100000.)["negative_confirmation_count"] == 1


def test_actual_next_open_pending_exit_lock_and_original_exit_without_shared_model():
    args = fixture(); args[0].loc[3, "open"] = 9.; args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, SharedEpisodeExitController(args[0], [constant_record(), constant_record(.03, 3)]))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    args = fixture(); args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, SharedEpisodeExitController(args[0], []))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].iloc[0] == args[0].date.iloc[3]
