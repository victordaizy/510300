"""检验当日份额连接、无观点、预测幅度边界和实际进入退出。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.continuation_strength_blend_inputs_v1 import continuation_strength_frames, current_close_signals, PRIMARY, MODELS
from research.entry_vintage_exit_inputs_v1 import prediction_identity
from research.event_clock_account_v1 import simulate_event_account
from tests.test_trend_noise_reference_blend_v1 import fixture as market_fixture, CFG as OLD_CFG

CFG = {**OLD_CFG, "maximum_holding_days": 60, "combination": "POSITIVE_CONTINUATION_OVER_CONTINUATION_PLUS_REMAINING_NOISE"}


def fixture():
    data, old, start = market_fixture()
    parents = {cost: {new: group[old_model].copy().assign(source_model=new) for new, old_model in zip(MODELS, group)} for cost, group in old.items()}
    model = {"features": ["log_holding_days"], "mean": [0.], "scale": [1.], "coefficients": [1.], "intercept": 0., "feature_clip": 10.}
    records = [{"fit_index": 110, "fit_origin": str(data.date.iloc[110].date()), "fit_time": str(data.date.iloc[110]+pd.Timedelta(hours=15, minutes=5)),
        "status": "FIT_COMPLETE", "latest_exit_index": 100, "model": model}]
    signals = {}
    for cost in CFG["costs"]:
        signal = parents[cost][MODELS[0]][["origin", "origin_index", "execution_date", "source_cost"]].copy()
        signal["reference_current_shares"] = 100
        signal["learning_status"] = "PREDICTION_AVAILABLE"
        signal["continuation_prediction"] = .02 if cost == "BASE" else .01
        signal["learning_fit_origin"] = data.date.iloc[110]
        signal["model_selection_index"] = 119
        signal["model_selection_origin"] = data.date.iloc[119]
        signal["fixed_prediction_identity"] = prediction_identity(records[0])
        signal["log_holding_days"] = np.log1p(signal.origin_index-119+1)
        signal["vol20"] = data.vol20.iloc[signal.origin_index].to_numpy()
        signals[cost] = signal
    return data, parents, signals, records, start


def run_fixture(values):
    data, parents, signals, records, start = values
    return continuation_strength_frames(data, parents, signals, records, CFG, start)[0]


def test_prediction_positive_nonpositive_zero_noise_and_horizon():
    values = fixture()
    data, parents, signals, records, start = values
    signals["BASE"].loc[signals["BASE"].origin_index.isin([119, 122]), "continuation_prediction"] = 0
    signals["BASE"].loc[signals["BASE"].origin_index.eq(123), "continuation_prediction"] = -.02
    frames = run_fixture(values)
    f = frames["BASE"]
    assert f.reference_budget131.iloc[119] == 0 and f.target.iloc[119] == .2
    assert 0 < f.reference_budget131.iloc[120] < 1
    assert f.reference_budget131.iloc[122] == 0 and f.reference_budget131.iloc[123] == 0
    assert f.reference_budget131.iloc[145] == 1 and f.target.iloc[145] == .8
    assert f.remaining_comparison_days.iloc[119] == 59 and f.remaining_comparison_days.iloc[178] == 1
    assert f.reference_budget131.iloc[120] != frames["STRESS"].reference_budget131.iloc[120]
    assert f.target.iloc[:119].isna().all() and pd.isna(f.target.iloc[-1])


def test_flat_and_explicit_no_model_do_not_fabricate_predictions():
    values = fixture()
    data, parents, signals, records, start = values
    records.append({**records[0], "fit_index": 109, "fit_origin": str(data.date.iloc[109].date()),
        "fit_time": str(data.date.iloc[109]+pd.Timedelta(hours=15, minutes=5)), "status": "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", "model": None})
    signal = signals["BASE"]
    signal.loc[0, ["reference_current_shares", "learning_status", "continuation_prediction"]] = [0, None, np.nan]
    signal.loc[1, ["learning_status", "continuation_prediction", "learning_fit_origin", "fixed_prediction_identity"]] = ["NO_VIEW_NO_MATURE_MODEL", np.nan, data.date.iloc[109], "NO_MODEL"]
    frame = run_fixture(values)["BASE"]
    assert frame.target.iloc[119] == .2 and frame.target.iloc[120] == .2
    assert frame.continuation_prediction.iloc[119:121].isna().all()
    assert frame.positive_continuation_advantage.iloc[119:121].isna().all()
    assert frame.allocation_status.iloc[120] == "NO_MATURE_MODEL_USE143_WITHOUT_LEARNING"


def test_unknowns_and_zero_budget_parent_missing_keep_no_view():
    values = fixture()
    data, parents, signals, records, start = values
    signal = signals["BASE"]
    signal.loc[0, "learning_status"] = "NO_VIEW_MODEL_FIT_FAILED"
    signal.loc[1, "continuation_prediction"] = np.nan
    signal.loc[2, "model_selection_index"] = np.nan
    signal.loc[3, "continuation_prediction"] = -.1
    parents["BASE"][MODELS[0]].loc[3, "reference_weight"] = np.nan
    frame = run_fixture(values)["BASE"]
    assert frame.target.iloc[119:123].isna().all()
    assert frame.reference_budget131.iloc[122] == 0


def test_future_model_wrong_identity_cost_and_execution_rejected():
    for field, value in [("model_selection_index", 160), ("fixed_prediction_identity", "错误身份"), ("source_cost", "STRESS")]:
        values = fixture()
        values[2]["BASE"].loc[0, field] = value
        with pytest.raises(ValueError):
            run_fixture(values)
    values = fixture()
    values[3][0]["fit_time"] = str(values[0].date.iloc[130]+pd.Timedelta(hours=15, minutes=5))
    with pytest.raises(ValueError):
        run_fixture(values)
    values = fixture()
    values[2]["BASE"]["execution_date"] = values[2]["BASE"].origin
    with pytest.raises(ValueError):
        run_fixture(values)


def test_future_prediction_parent_changes_and_prefix_isolation():
    values = fixture()
    original = run_fixture(values)
    changed = copy.deepcopy(values)
    for signal in changed[2].values():
        signal.loc[signal.origin_index.ge(150), "continuation_prediction"] = -.5
    for group in changed[1].values():
        for parent in group.values():
            parent.loc[parent.origin_index.ge(150), "reference_weight"] = 0
    future = run_fixture(changed)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:150], future["BASE"].iloc[:150])
    data, parents, signals, records, start = values
    prefix_parents = {cost: {model: parent[parent.origin_index.lt(149)].copy() for model, parent in group.items()} for cost, group in parents.items()}
    prefix_signals = {cost: signal[signal.origin_index.lt(149)].copy() for cost, signal in signals.items()}
    prefix, _ = continuation_strength_frames(data.iloc[:150], prefix_parents, prefix_signals, records, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:149], prefix["BASE"].iloc[:149])


def test_current_close_join_actual_exit_reentry_and_dividend():
    values = fixture()
    data, parents, signals, records, start = values
    for model in MODELS:
        parent = parents["BASE"][model]
        parent["reference_weight"] = .8
        parent.loc[parent.origin_index.eq(124), "reference_weight"] = 0
        parent.loc[parent.origin_index.eq(128), "reference_weight"] = .3
    data.loc[123, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[122]], "ex_date": [data.date.iloc[123]],
        "payment_date": [data.date.iloc[124]], "cash_dividend_per_share": [.1]})
    frames = run_fixture(values)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[125], "shares"] == 0 and by_date.loc[data.date.iloc[126], "filled_quantity"] > 0
    assert by_date.loc[data.date.iloc[129], "filled_quantity"] < 0 and by_date.loc[data.date.iloc[129], "shares"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[122], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum()) and ledger.accounting_error.abs().max() < 1e-6
    joined = current_close_signals(decisions, ledger, data, start, "BASE")
    assert joined.reference_current_shares.iloc[0] == 0
    assert joined.loc[joined.origin.eq(data.date.iloc[124]), "reference_current_shares"].iloc[0] > 0
    assert joined.loc[joined.origin.eq(data.date.iloc[125]), "reference_current_shares"].iloc[0] == 0
    broken = ledger.drop(ledger.index[3])
    with pytest.raises(ValueError):
        current_close_signals(decisions, broken, data, start, "BASE")
