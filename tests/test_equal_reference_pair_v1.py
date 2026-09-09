"""检查各半目标、来源时钟、未知预算和真实合并账户。"""
import numpy as np
import pandas as pd
import pytest
from research.equal_reference_pair_inputs_v1 import equal_reference_targets, MODELS, PRIMARY
from research.event_clock_account_v1 import simulate_event_account

CFG = {"reference_weights": [.5, .5], "weight_band": .10, "initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}


def fixture(n=12):
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": 10., "close": 10.,
        "previous_close": 10., "dividend": 0., "variance60": .0004})
    indices = np.arange(1, n-1)
    parents = {model: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "execution_date": data.date.iloc[indices+1].to_numpy(),
        "origin_index": indices, "reference_weight": .8 if i == 0 else .4, "source_cost": "BASE", "source_model": model}) for i, model in enumerate(MODELS)}
    return data, parents, data.date.iloc[2]


def test_equal_weight_and_single_reference_exit_keeps_other_budget():
    data, parents, start = fixture()
    parents[MODELS[0]].loc[1, "reference_weight"] = 0.
    parents[MODELS[0]].loc[2, "reference_weight"] = 0.
    parents[MODELS[1]].loc[2, "reference_weight"] = 0.
    f, summary = equal_reference_targets(data, parents, CFG, start, "BASE")
    assert np.isclose(f.target.iloc[1], .6) and f.target.iloc[2] == .2 and f.target.iloc[3] == 0.
    assert f.target.iloc[[0, -1]].isna().all() and summary["zero_target_origins"] == 1


def test_unknown_reference_is_not_zero_or_full_allocation_to_other():
    data, parents, start = fixture()
    parents[MODELS[0]].loc[2, "reference_weight"] = np.nan
    parents[MODELS[1]].loc[2, "reference_weight"] = 0.
    parents[MODELS[1]].loc[3, "reference_weight"] = np.nan
    f, summary = equal_reference_targets(data, parents, CFG, start, "BASE")
    assert f.target.iloc[3:5].isna().all() and summary["unknown_target_origins"] == 2


def test_source_identity_cost_and_next_open_clock_are_required():
    data, parents, start = fixture()
    with pytest.raises(ValueError):
        equal_reference_targets(data, parents, CFG, start, "STRESS")
    changed = {k: v.copy() for k, v in parents.items()}; changed[MODELS[1]].execution_date += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        equal_reference_targets(data, changed, CFG, start, "BASE")
    changed = {k: v.copy() for k, v in parents.items()}; changed[MODELS[1]].source_model = MODELS[0]
    with pytest.raises(ValueError):
        equal_reference_targets(data, changed, CFG, start, "BASE")
    with pytest.raises(ValueError):
        equal_reference_targets(data, {MODELS[0]: parents[MODELS[0]]}, CFG, start, "BASE")


def test_invalid_sizes_and_changed_weights_are_rejected():
    data, parents, start = fixture()
    for value in [-.1, 1.1, np.inf]:
        changed = {k: v.copy() for k, v in parents.items()}; changed[MODELS[0]].loc[0, "reference_weight"] = value
        with pytest.raises(ValueError):
            equal_reference_targets(data, changed, CFG, start, "BASE")
    with pytest.raises(ValueError):
        equal_reference_targets(data, parents, {**CFG, "reference_weights": [.4, .6]}, start, "BASE")


def test_future_and_short_prefix_preserve_prior_targets():
    data, parents, start = fixture()
    original, _ = equal_reference_targets(data, parents, CFG, start, "BASE")
    changed = {k: v.copy() for k, v in parents.items()}
    for parent in changed.values():
        parent.loc[parent.origin_index.ge(7), "reference_weight"] = 0.
    modified, _ = equal_reference_targets(data, changed, CFG, start, "BASE")
    pd.testing.assert_frame_equal(original.iloc[:7], modified.iloc[:7])
    prefix = {k: v[v.origin_index.lt(7)].copy() for k, v in parents.items()}
    short, _ = equal_reference_targets(data.iloc[:8], prefix, CFG, start, "BASE")
    pd.testing.assert_frame_equal(original.iloc[:7], short.iloc[:7])
    assert pd.isna(short.target.iloc[-1])


def test_real_account_partial_exit_and_reentry_use_own_cash_and_dividends():
    data, parents, start = fixture()
    parents[MODELS[0]].loc[parents[MODELS[0]].origin_index.eq(3), "reference_weight"] = 0.
    for parent in parents.values():
        parent.loc[parent.origin_index.eq(5), "reference_weight"] = 0.
    data.loc[3, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[2]], "ex_date": [data.date.iloc[3]],
        "payment_date": [data.date.iloc[4]], "cash_dividend_per_share": [.1]})
    factors, _ = equal_reference_targets(data, parents, CFG, start, "BASE")
    ledger, decisions = simulate_event_account(data, dividends, CFG, {"commission": .0002, "minimum": 5., "slippage": .0005}, start, PRIMARY,
        targets=factors.target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[4], "filled_quantity"] < 0 and by_date.loc[data.date.iloc[4], "shares"] > 0
    assert by_date.loc[data.date.iloc[6], "shares"] == 0 and by_date.loc[data.date.iloc[7], "filled_quantity"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[2], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum())
    assert ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
