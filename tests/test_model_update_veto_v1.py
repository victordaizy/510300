"""检验同状态模型比较、否决锁定、未知状态、时钟和完整账户。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.model_update_veto_inputs_v1 import model_update_veto_frames, PRIMARY, MODELS
from research.entry_vintage_exit_inputs_v1 import prediction_identity
from research.within_cycle_exit_inputs_v1 import FEATURES
from research.event_clock_account_v1 import simulate_event_account
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, "combination": "FIXED_NONNEGATIVE_LATEST_NEGATIVE_VETO_UNTIL_PARENT_ZERO"}


def fixture():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=80), "open": 10., "close": 10., "previous_close": 10.,
        "dividend": 0., "variance60": .0004})
    model = {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES.copy(), "mean": [0.]*8, "scale": [1.]*8,
        "coefficients": [0.]*8, "intercept": .02, "feature_clip": 6.}
    changed = copy.deepcopy(model)
    changed["intercept"] = -.02
    changed["coefficients"][FEATURES.index("mom5")] = 1.
    records = [{"fit_index": index, "fit_origin": str(data.date.iloc[index].date()), "fit_time": str(data.date.iloc[index]+pd.Timedelta(hours=15, minutes=5)),
        "latest_exit_index": index-1, "status": "FIT_COMPLETE", "model": saved} for index, saved in [(20, model), (30, changed)]]
    indices = np.arange(24, 79)
    parents = {cost: {MODELS[0]: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "origin_index": indices,
        "execution_date": data.date.iloc[indices+1].to_numpy(), "reference_weight": .8 if cost == "BASE" else .6,
        "source_cost": cost, "source_model": MODELS[0]})} for cost in CFG["costs"]}
    signals = {}
    for cost, group in parents.items():
        parent = group[MODELS[0]]
        parent.loc[parent.origin_index.eq(35), "reference_weight"] = 0
        signal = parent[["origin", "origin_index", "execution_date", "source_cost"]].copy()
        signal["reference_current_shares"] = 0
        signal["learning_status"] = pd.Series([None]*len(signal), dtype=object)
        signal["continuation_prediction"] = np.nan
        signal["learning_fit_origin"] = pd.NaT
        signal["model_selection_index"] = np.nan
        signal["model_selection_origin"] = pd.NaT
        signal["fixed_prediction_identity"] = pd.Series([None]*len(signal), dtype=object)
        for feature in FEATURES:
            signal[feature] = np.nan
        held = signal.origin_index.between(25, 34)
        signal.loc[held, "reference_current_shares"] = 100
        signal.loc[held, "learning_status"] = "PREDICTION_AVAILABLE"
        signal.loc[held, "continuation_prediction"] = .02
        signal.loc[held, "learning_fit_origin"] = data.date.iloc[20]
        signal.loc[held, "model_selection_index"] = 25
        signal.loc[held, "model_selection_origin"] = data.date.iloc[25]
        signal.loc[held, "fixed_prediction_identity"] = prediction_identity(records[0])
        signal.loc[held, FEATURES] = 0.
        signal.loc[held, "log_holding_days"] = np.log1p(signal.loc[held, "origin_index"]-25+1)
        signal.loc[held, "entry_mode"] = 1.
        signal.loc[signal.origin_index.between(31, 34), "mom5"] = .2
        signals[cost] = signal
    return data, parents, signals, records, data.date.iloc[25]


def run_fixture(values):
    data, parents, signals, records, start = values
    return model_update_veto_frames(data, parents, signals, records, CFG, start)


def test_same_parameters_no_change_negative_update_locked_until_zero():
    frames, summaries = run_fixture(fixture())
    f = frames["BASE"]
    assert f.target.iloc[24:30].eq(.8).all() and f.target.iloc[30:36].eq(0).all()
    assert f.target.iloc[36] == .8 and f.veto_started.iloc[30] and f.veto_reset.iloc[35]
    assert f.latest_continuation_prediction.iloc[25:30].eq(.02).all() and f.latest_continuation_prediction.iloc[30] == -.02
    assert f.latest_continuation_prediction.iloc[31:35].isna().all()
    assert f.source_mom5.iloc[31] == .2 and f.veto_active.iloc[31]
    assert summaries[0]["new_latest_predictions"] == 6 and summaries[0]["veto_episodes_started"] == 1
    assert frames["STRESS"].target.iloc[24] == .6


def test_flat_and_no_fixed_model_keep_missing_predictions_without_veto():
    values = fixture()
    data, parents, signals, records, start = values
    records[0].update(status="NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", model=None)
    for signal in signals.values():
        held = signal.reference_current_shares.gt(0)
        signal.loc[held, "learning_status"] = "NO_VIEW_NO_MATURE_MODEL"
        signal.loc[held, "continuation_prediction"] = np.nan
        signal.loc[held, "fixed_prediction_identity"] = "NO_MODEL"
    frames, summaries = run_fixture(values)
    f = frames["BASE"]
    assert f.target.iloc[24:35].eq(.8).all() and not f.veto_started.any()
    assert f.fixed_continuation_prediction.isna().all() and f.latest_continuation_prediction.isna().all()
    assert summaries[0]["new_latest_predictions"] == 0


def test_unknown_is_not_no_disagreement_and_zero_or_existing_veto_has_priority():
    values = fixture()
    data, parents, signals, records, start = values
    signal = signals["BASE"]
    signal.loc[signal.origin_index.eq(25), "learning_status"] = "NO_VIEW_MODEL_FIT_FAILED"
    signal.loc[signal.origin_index.eq(26), "mom5"] = np.nan
    signal.loc[signal.origin_index.eq(32), "learning_status"] = "NO_VIEW_MODEL_FIT_FAILED"
    signal.loc[signal.origin_index.eq(35), "learning_status"] = "NO_VIEW_MODEL_FIT_FAILED"
    parents["BASE"][MODELS[0]].loc[parents["BASE"][MODELS[0]].origin_index.eq(31), "reference_weight"] = np.nan
    frames, _ = run_fixture(values)
    f = frames["BASE"]
    assert f.target.iloc[25:27].isna().all() and f.target.iloc[27] == .8
    assert pd.isna(f.target.iloc[31]) and f.veto_active.iloc[31]
    assert f.target.iloc[32] == 0 and f.target.iloc[35] == 0 and f.veto_reset.iloc[35]


def test_wrong_identity_future_fit_wrong_cost_and_invalid_scale_rejected():
    for field, value in [("fixed_prediction_identity", "错误身份"), ("source_cost", "STRESS"), ("model_selection_index", 40)]:
        values = fixture()
        signal = values[2]["BASE"]
        signal.loc[signal.origin_index.eq(25), field] = value
        with pytest.raises(ValueError):
            run_fixture(values)
    values = fixture()
    values[3][1]["fit_time"] = str(values[0].date.iloc[31]+pd.Timedelta(hours=15, minutes=5))
    with pytest.raises(ValueError):
        run_fixture(values)
    values = fixture()
    values[3][1]["model"]["scale"][0] = 0
    with pytest.raises(ValueError):
        run_fixture(values)


def test_future_model_state_and_parent_changes_and_prefix_isolation():
    values = fixture()
    original, _ = run_fixture(values)
    altered = copy.deepcopy(values)
    data = altered[0]
    record = copy.deepcopy(altered[3][-1])
    record.update(fit_index=60, fit_origin=str(data.date.iloc[60].date()), fit_time=str(data.date.iloc[60]+pd.Timedelta(hours=15, minutes=5)), latest_exit_index=59)
    record["model"]["intercept"] = -1
    altered[3].append(record)
    for group in altered[1].values():
        for parent in group.values():
            parent.loc[parent.origin_index.ge(60), "reference_weight"] = 0
    future, _ = run_fixture(altered)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:60], future["BASE"].iloc[:60])
    data, parents, signals, records, start = values
    pp = {cost: {model: parent[parent.origin_index.lt(59)].copy() for model, parent in group.items()} for cost, group in parents.items()}
    ss = {cost: signal[signal.origin_index.lt(59)].copy() for cost, signal in signals.items()}
    prefix, _ = model_update_veto_frames(data.iloc[:60], pp, ss, records, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:59], prefix["BASE"].iloc[:59])


def test_actual_veto_exit_reentry_and_dividend():
    values = fixture()
    data, parents, signals, records, start = values
    data.loc[28, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[27]], "ex_date": [data.date.iloc[28]],
        "payment_date": [data.date.iloc[29]], "cash_dividend_per_share": [.1]})
    frames, _ = run_fixture(values)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[25], "filled_quantity"] > 0
    assert by_date.loc[data.date.iloc[31], "shares"] == 0 and by_date.loc[data.date.iloc[37], "filled_quantity"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[27], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum()) and ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
